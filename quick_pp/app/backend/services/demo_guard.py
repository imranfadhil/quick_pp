"""Live Demo Session Tracker, Concurrency Limiter, SSRF Guard, and 3-Tier Artifact Cleanup.

Active when `QPP_DEMO_MODE=true`:
1. Tracks active browser sessions in memory via `X-Demo-Session-Id` heartbeats (`--workers 1`),
   exposing online user counts, lag warnings, and soft admission capacity control.
2. Caps simultaneous heavy compute/plotting requests so bursts on a 1-vCPU / 2 GB droplet
   queue gracefully instead of triggering OOM kills.
3. Enforces 3-tier ephemeral disk & RAM hygiene:
   - Tier 1: Immediate post-commit deletion of staged upload files & disabled pickle disk sync
   - Tier 2: Per-session cleanup on tab leave / heartbeat expiry + full state reset after global idle
   - Tier 3: Periodic background TTL sweep + hard ring-buffer caps on reports and staged uploads
4. Validates external LLM base URLs to prevent SSRF and server API-key exfiltration.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from quick_pp.app.backend.utils.auth import is_demo_mode

logger = logging.getLogger(__name__)

DEMO_SESSION_HEADER = "x-demo-session-id"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")

_current_demo_session_id: ContextVar[Optional[str]] = ContextVar(
    "current_demo_session_id", default=None
)

_lock = threading.RLock()
_demo_sessions: Dict[str, Dict[str, Any]] = {}
_had_activity_since_reset: bool = False
_last_zero_users_at: Optional[float] = None

_compute_cond = threading.Condition()
_active_compute_jobs: int = 0


# ---------------------------------------------------------------------------
# Dynamic Configuration Readers (env-driven for runtime & test overrides)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def get_demo_max_users() -> int:
    return _env_int("QPP_DEMO_MAX_USERS", 6, minimum=1)


def get_demo_session_ttl() -> int:
    return _env_int("QPP_DEMO_SESSION_TTL_SECONDS", 75, minimum=10)


def get_demo_idle_timeout() -> int:
    return _env_int("QPP_DEMO_IDLE_TIMEOUT_SECONDS", 600, minimum=30)


def get_demo_hidden_idle_timeout() -> int:
    return _env_int("QPP_DEMO_HIDDEN_IDLE_TIMEOUT_SECONDS", 300, minimum=20)


def get_demo_lag_threshold() -> int:
    return _env_int("QPP_DEMO_LAG_WARNING_USERS", 3, minimum=1)


def get_demo_max_concurrent_compute() -> int:
    return _env_int("QPP_DEMO_MAX_CONCURRENT_COMPUTE", 2, minimum=1)


def get_demo_compute_timeout() -> float:
    return _env_float("QPP_DEMO_COMPUTE_TIMEOUT_SECONDS", 15.0, minimum=0.05)


def get_demo_report_ttl() -> int:
    return _env_int("QPP_DEMO_REPORT_TTL_SECONDS", 1800, minimum=60)


def get_demo_max_reports() -> int:
    return _env_int("QPP_DEMO_MAX_REPORTS", 15, minimum=1)


def get_demo_staging_ttl() -> int:
    return _env_int("QPP_DEMO_STAGING_TTL_SECONDS", 900, minimum=30)


def get_demo_max_staged_files() -> int:
    return _env_int("QPP_DEMO_MAX_STAGED_FILES", 20, minimum=1)


def get_demo_zero_users_reset_seconds() -> int:
    return _env_int("QPP_DEMO_ZERO_USERS_RESET_SECONDS", 600, minimum=10)


def get_demo_sweep_interval_seconds() -> int:
    return _env_int("QPP_DEMO_SWEEP_INTERVAL_SECONDS", 300, minimum=15)


# ---------------------------------------------------------------------------
# Context & Session Helpers
# ---------------------------------------------------------------------------


def normalize_session_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = str(raw).strip()
    if _SESSION_ID_RE.match(cleaned):
        return cleaned
    return None


def get_current_demo_session_id() -> Optional[str]:
    return _current_demo_session_id.get()


def set_current_demo_session_id(session_id: Optional[str]):
    return _current_demo_session_id.set(normalize_session_id(session_id))


def reset_current_demo_session_id(token) -> None:
    _current_demo_session_id.reset(token)


def mark_demo_state_dirty() -> None:
    global _had_activity_since_reset
    with _lock:
        _had_activity_since_reset = True


def reset_demo_sessions_for_testing() -> None:
    """Clear in-memory demo session tracking (used by tests)."""
    global _had_activity_since_reset, _last_zero_users_at
    with _lock:
        _demo_sessions.clear()
        _had_activity_since_reset = False
        _last_zero_users_at = None


def evict_expired_sessions(now: Optional[float] = None) -> List[str]:
    global _last_zero_users_at
    ts = time.time() if now is None else now
    ttl = get_demo_session_ttl()
    evicted: List[str] = []

    with _lock:
        had_any = len(_demo_sessions) > 0
        for sid, meta in list(_demo_sessions.items()):
            if ts - float(meta.get("last_seen", 0.0)) > ttl:
                _demo_sessions.pop(sid, None)
                evicted.append(sid)
        if had_any and len(_demo_sessions) == 0:
            _last_zero_users_at = ts

    for sid in evicted:
        try:
            purge_session_artifacts(sid)
        except Exception as exc:
            logger.debug(
                "Failed to purge artifacts for expired session %s: %s", sid, exc
            )
    return evicted


def _build_snapshot_unlocked(
    session_id: Optional[str] = None,
    admitted: Optional[bool] = None,
) -> Dict[str, Any]:
    online = len(_demo_sessions)
    max_users = get_demo_max_users()
    lag_thresh = get_demo_lag_threshold()
    with _compute_cond:
        active_compute = _active_compute_jobs
    is_member = bool(session_id and session_id in _demo_sessions)
    at_cap = online >= max_users and not is_member
    return {
        "demo_mode": is_demo_mode(),
        "online_users": online,
        "max_users": max_users,
        "at_capacity": at_cap,
        "admitted": is_member if admitted is None else admitted,
        "session_active": is_member,
        "lag_warning": online >= lag_thresh or active_compute >= 1,
        "lag_warning_threshold": lag_thresh,
        "active_compute_jobs": active_compute,
        "max_compute_jobs": get_demo_max_concurrent_compute(),
        "idle_timeout_seconds": get_demo_idle_timeout(),
        "hidden_idle_timeout_seconds": get_demo_hidden_idle_timeout(),
        "session_ttl_seconds": get_demo_session_ttl(),
    }


def get_demo_status_snapshot(
    session_id: Optional[str] = None, now: Optional[float] = None
) -> Dict[str, Any]:
    evict_expired_sessions(now=now)
    sid = normalize_session_id(session_id)
    with _lock:
        return _build_snapshot_unlocked(session_id=sid)


def register_or_touch_session(
    session_id: Optional[str],
    client_ip: str = "",
    now: Optional[float] = None,
) -> Dict[str, Any]:
    global _had_activity_since_reset, _last_zero_users_at
    ts = time.time() if now is None else now
    evict_expired_sessions(now=ts)
    sid = normalize_session_id(session_id)
    if not sid:
        with _lock:
            return _build_snapshot_unlocked(session_id=None, admitted=True)

    with _lock:
        if sid in _demo_sessions:
            _demo_sessions[sid]["last_seen"] = ts
            if client_ip:
                _demo_sessions[sid]["ip"] = client_ip
            return _build_snapshot_unlocked(session_id=sid, admitted=True)

        if len(_demo_sessions) >= get_demo_max_users():
            return _build_snapshot_unlocked(session_id=sid, admitted=False)

        _demo_sessions[sid] = {
            "created_at": ts,
            "last_seen": ts,
            "ip": client_ip,
        }
        _had_activity_since_reset = True
        _last_zero_users_at = None
        return _build_snapshot_unlocked(session_id=sid, admitted=True)


def leave_session(
    session_id: Optional[str], now: Optional[float] = None
) -> Dict[str, Any]:
    global _last_zero_users_at
    ts = time.time() if now is None else now
    sid = normalize_session_id(session_id)
    removed = False
    with _lock:
        if sid and sid in _demo_sessions:
            _demo_sessions.pop(sid, None)
            removed = True
            if len(_demo_sessions) == 0:
                _last_zero_users_at = ts

    purged = {"reports_deleted": 0, "staged_files_deleted": 0}
    if sid:
        purged = purge_session_artifacts(sid)

    with _lock:
        snap = _build_snapshot_unlocked(session_id=sid, admitted=False)
    snap["removed"] = removed
    snap["purged"] = purged
    return snap


# ---------------------------------------------------------------------------
# Heavy Compute Concurrency Guard
# ---------------------------------------------------------------------------

_LIGHTWEIGHT_WORKSPACE_SUFFIXES = (
    "/demo/reset",
    "/config",
    "/module_params",
)


def is_heavy_demo_compute_request(method: str, path: str) -> bool:
    """Return True if this HTTP request runs CPU/RAM-intensive petrophysics, LLM, or PDF work."""
    m = method.upper()
    p = path.rstrip("/")

    # Report PDF export is a GET that compiles high-DPI PDF
    if m == "GET":
        return p.endswith("/pdf") and "/reports/" in p

    if m not in ("POST", "PUT"):
        return False

    if "/reports/" in p and (p.endswith("/generate") or p.endswith("/enhance")):
        return True

    if p.endswith(("/assistant/chat", "/mcp/call", "/mcp")):
        return True

    if "/quick_pp/workspace/projects/" in p:
        if any(p.endswith(sfx) for sfx in _LIGHTWEIGHT_WORKSPACE_SUFFIXES):
            return False
        return True

    heavy_prefixes = (
        "/quick_pp/lithology/",
        "/quick_pp/porosity/",
        "/quick_pp/saturation/",
        "/quick_pp/permeability/",
        "/quick_pp/reservoir_summary/",
        "/quick_pp/uncertainty/",
        "/quick_pp/upscaling/",
        "/quick_pp/scal/",
        "/quick_pp/rock_typing/",
        "/quick_pp/perm_transform/",
        "/quick_pp/shf/",
    )
    return any(p.startswith(pfx) for pfx in heavy_prefixes)


def acquire_compute_slot_sync(timeout: Optional[float] = None) -> bool:
    global _active_compute_jobs
    wait_s = get_demo_compute_timeout() if timeout is None else max(0.0, float(timeout))
    deadline = time.monotonic() + wait_s
    with _compute_cond:
        while _active_compute_jobs >= get_demo_max_concurrent_compute():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _compute_cond.wait(timeout=min(remaining, 0.25))
        _active_compute_jobs += 1
        return True


def release_compute_slot_sync() -> None:
    global _active_compute_jobs
    with _compute_cond:
        if _active_compute_jobs > 0:
            _active_compute_jobs -= 1
        _compute_cond.notify(1)


# ---------------------------------------------------------------------------
# 3-Tier Ephemeral Artifact Cleanup (Reports, Staged Uploads, Temp Pickles, RAM)
# ---------------------------------------------------------------------------


def _safe_rmtree(path: Path) -> bool:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            return True
    except Exception:
        pass
    return False


def _safe_unlink(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        pass
    return False


def purge_session_artifacts(session_id: str) -> Dict[str, int]:
    """Tier 2: Immediately delete reports and uncommitted staged files owned by `session_id`."""
    sid = normalize_session_id(session_id)
    if not sid:
        return {"reports_deleted": 0, "staged_files_deleted": 0}

    reports_deleted = 0
    staged_deleted = 0

    try:
        from quick_pp.app.backend.services import report as report_svc

        for rpt_id, rpt_data in list(report_svc._REPORTS_STORE.items()):
            if (
                rpt_data.get("_demo_session_id") == sid
                or rpt_data.get("demo_session_id") == sid
            ):
                report_svc._REPORTS_STORE.pop(rpt_id, None)
                if _safe_rmtree(report_svc.REPORTS_STATIC_ROOT / rpt_id):
                    reports_deleted += 1

        for job_id, job_data in list(report_svc._REPORT_JOBS.items()):
            if job_data.get("demo_session_id") == sid:
                report_svc._REPORT_JOBS.pop(job_id, None)
                _safe_unlink(report_svc._job_path(job_id))
    except Exception as exc:
        logger.debug("Session report cleanup skipped for %s: %s", sid, exc)

    try:
        from quick_pp.app.backend.services import las_handler as las_svc

        for fid, meta in list(las_svc._STAGED_FILES.items()):
            if meta.get("demo_session_id") == sid:
                las_svc._STAGED_FILES.pop(fid, None)
                fpath = meta.get("path")
                if fpath and _safe_unlink(Path(fpath)):
                    staged_deleted += 1
    except Exception as exc:
        logger.debug("Session staging cleanup skipped for %s: %s", sid, exc)

    return {
        "reports_deleted": reports_deleted,
        "staged_files_deleted": staged_deleted,
    }


def enforce_report_storage_limits(
    now: Optional[float] = None, force_all: bool = False
) -> int:
    """Evict expired reports (> TTL) and cap total report folders on disk (ring buffer)."""
    ts = time.time() if now is None else now
    ttl = get_demo_report_ttl()
    max_reports = get_demo_max_reports()
    deleted = 0

    try:
        from quick_pp.app.backend.services import report as report_svc

        root = report_svc.REPORTS_STATIC_ROOT
        if root.exists():
            rpt_dirs: List[Tuple[float, Path]] = []
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                if child.name == "_jobs":
                    for jf in child.glob("*.json"):
                        try:
                            age = ts - jf.stat().st_mtime
                            if force_all or age > ttl:
                                report_svc._REPORT_JOBS.pop(jf.stem, None)
                                if _safe_unlink(jf):
                                    deleted += 1
                        except OSError:
                            pass
                    continue
                if not child.name.startswith("rpt_"):
                    continue
                try:
                    mtime = child.stat().st_mtime
                except OSError:
                    mtime = 0.0
                mem_entry = report_svc._REPORTS_STORE.get(child.name)
                if isinstance(mem_entry, dict) and mem_entry.get("_created_ts"):
                    mtime = max(mtime, float(mem_entry["_created_ts"]))
                if force_all or (ts - mtime > ttl):
                    report_svc._REPORTS_STORE.pop(child.name, None)
                    if _safe_rmtree(child):
                        deleted += 1
                else:
                    rpt_dirs.append((mtime, child))

            # Ring-buffer cap: keep only newest `max_reports`
            if len(rpt_dirs) > max_reports:
                rpt_dirs.sort(key=lambda item: item[0], reverse=True)
                for _, old_dir in rpt_dirs[max_reports:]:
                    report_svc._REPORTS_STORE.pop(old_dir.name, None)
                    if _safe_rmtree(old_dir):
                        deleted += 1

        # Also cap in-memory _REPORTS_STORE if directories were already removed
        if force_all:
            report_svc._REPORTS_STORE.clear()
            report_svc._REPORT_JOBS.clear()
        elif len(report_svc._REPORTS_STORE) > max_reports:
            ordered = sorted(
                report_svc._REPORTS_STORE.items(),
                key=lambda kv: float(kv[1].get("_created_ts") or 0.0),
                reverse=True,
            )
            for old_id, _ in ordered[max_reports:]:
                report_svc._REPORTS_STORE.pop(old_id, None)
    except Exception as exc:
        logger.debug("Report storage cleanup error: %s", exc)

    # Clean orphaned headless browser PDF temp profiles in /tmp
    try:
        tmp_root = Path(tempfile.gettempdir())
        for pdf_tmp in tmp_root.glob("qpp_pdf_*"):
            try:
                if force_all or (ts - pdf_tmp.stat().st_mtime > 900):
                    _safe_rmtree(pdf_tmp)
            except OSError:
                pass
    except Exception:
        pass

    return deleted


def enforce_upload_storage_limits(
    now: Optional[float] = None, force_all: bool = False
) -> int:
    """Evict expired staged/uploaded log files (> TTL) and cap staged file count."""
    ts = time.time() if now is None else now
    ttl = get_demo_staging_ttl()
    max_files = get_demo_max_staged_files()
    deleted = 0

    try:
        from quick_pp.app.backend.services import las_handler as las_svc

        for fid, meta in list(las_svc._STAGED_FILES.items()):
            created = float(meta.get("created_at") or 0.0)
            fpath = Path(meta["path"]) if meta.get("path") else None
            if (
                force_all
                or (ts - created > ttl)
                or (fpath is not None and not fpath.exists())
            ):
                las_svc._STAGED_FILES.pop(fid, None)
                if fpath and _safe_unlink(fpath):
                    deleted += 1

        upload_dirs = [
            las_svc.STAGING_DIR,
            Path("uploads/las_db"),
            Path("uploads/las"),
            Path("uploads/parquet"),
        ]
        surviving_staged: List[Tuple[float, Path]] = []
        for udir in upload_dirs:
            if not udir.exists():
                continue
            for item in udir.iterdir():
                if not item.is_file():
                    continue
                try:
                    mtime = item.stat().st_mtime
                except OSError:
                    continue
                if force_all or (ts - mtime > ttl):
                    if _safe_unlink(item):
                        deleted += 1
                elif udir == las_svc.STAGING_DIR:
                    surviving_staged.append((mtime, item))

        if len(surviving_staged) > max_files:
            surviving_staged.sort(key=lambda pair: pair[0], reverse=True)
            keep_paths = {str(p.resolve()) for _, p in surviving_staged[:max_files]}
            for _, old_file in surviving_staged[max_files:]:
                if _safe_unlink(old_file):
                    deleted += 1
            for fid, meta in list(las_svc._STAGED_FILES.items()):
                mp = meta.get("path")
                if mp and str(Path(mp).resolve()) not in keep_paths:
                    las_svc._STAGED_FILES.pop(fid, None)
    except Exception as exc:
        logger.debug("Upload storage cleanup error: %s", exc)

    return deleted


def reset_all_ephemeral_demo_memory() -> None:
    """Reset global in-memory demo DataFrames, caches, and parameter overrides back to seed DB."""
    global _had_activity_since_reset, _last_zero_users_at
    try:
        from quick_pp.app.backend.services import database as db_svc
        from quick_pp.app.backend.services import workspace as ws_svc

        db_svc._ephemeral_wells.clear()
        db_svc._merged_data_cache.clear()
        db_svc.reset_demo_project_state(None)
        ws_svc._ephemeral_well_configs.clear()
        if hasattr(ws_svc, "_ephemeral_project_module_params"):
            ws_svc._ephemeral_project_module_params.clear()
        _safe_rmtree(db_svc._SHARED_CACHE_DIR)
    except Exception as exc:
        logger.debug("Ephemeral memory reset error: %s", exc)

    with _lock:
        _had_activity_since_reset = False
        _last_zero_users_at = None


def run_periodic_demo_cleanup(
    now: Optional[float] = None, force_all: bool = False
) -> Dict[str, Any]:
    """Run Tier 2 + Tier 3 cleanup across sessions, reports, uploads, temp pickles, and idle memory."""
    ts = time.time() if now is None else now
    evicted_sessions = evict_expired_sessions(now=ts)
    reports_cleaned = enforce_report_storage_limits(now=ts, force_all=force_all)
    uploads_cleaned = enforce_upload_storage_limits(now=ts, force_all=force_all)

    # Remove any stray /tmp/qpp_well_cache_sync pickles in single-worker demo mode
    if is_demo_mode():
        try:
            from quick_pp.app.backend.services import database as db_svc

            _safe_rmtree(db_svc._SHARED_CACHE_DIR)
        except Exception:
            pass

    memory_reset = False
    with _lock:
        should_reset_mem = force_all or (
            len(_demo_sessions) == 0
            and _had_activity_since_reset
            and _last_zero_users_at is not None
            and (ts - _last_zero_users_at) >= get_demo_zero_users_reset_seconds()
        )
    if should_reset_mem:
        reset_all_ephemeral_demo_memory()
        memory_reset = True

    return {
        "evicted_sessions": len(evicted_sessions),
        "reports_cleaned": reports_cleaned,
        "uploads_cleaned": uploads_cleaned,
        "memory_reset": memory_reset,
    }


async def demo_periodic_cleanup_loop() -> None:
    """Background asyncio task started in FastAPI lifespan when `QPP_DEMO_MODE=true`."""
    while True:
        interval = get_demo_sweep_interval_seconds()
        await asyncio.sleep(interval)
        if is_demo_mode():
            try:
                await asyncio.to_thread(run_periodic_demo_cleanup)
            except Exception as exc:
                logger.warning("Periodic demo cleanup failed: %s", exc)


# ---------------------------------------------------------------------------
# External LLM SSRF & Server API-Key Exfiltration Protection
# ---------------------------------------------------------------------------

_CANONICAL_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _is_private_or_internal_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().strip("[]")
    if not host:
        return True
    if host in ("localhost", "0.0.0.0") or host.endswith((".local", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_external_llm_target(
    provider: str,
    caller_base_url: Optional[str],
    caller_api_key: Optional[str],
    env_openrouter_url: str = _CANONICAL_OPENROUTER_URL,
    env_openrouter_key: str = "",
) -> Tuple[str, str]:
    """Validate `(base_url, api_key)` for external LLM requests, blocking SSRF and env-key leaks.

    - In `QPP_DEMO_MODE=true`:
      - Blocks `ollama` and any custom `base_url` (including private/metadata IPs like `169.254.169.254`).
      - Allows only `https://openrouter.ai/api/v1` with a caller-supplied `api_key` (or explicit demo env key).
    - In all modes:
      - Never attaches `env_openrouter_key` when the caller overrides `base_url` to a different host.
    """
    prov = (provider or "native_mcp").strip().lower()
    raw_caller_url = (caller_base_url or "").strip().rstrip("/")
    clean_caller_key = (caller_api_key or "").strip()
    default_or_url = (
        (env_openrouter_url or _CANONICAL_OPENROUTER_URL).strip().rstrip("/")
    )

    if is_demo_mode():
        if prov == "ollama":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Local Ollama connections are disabled in the public live demo.",
            )
        if raw_caller_url and raw_caller_url.lower() != _CANONICAL_OPENROUTER_URL:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Custom LLM base URLs are disabled in the public live demo.",
            )
        effective_url = _CANONICAL_OPENROUTER_URL
        effective_key = clean_caller_key or env_openrouter_key
        return effective_url, effective_key

    # Non-demo mode
    if prov == "ollama":
        effective_url = raw_caller_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        ).strip().rstrip("/")
        parsed = urllib.parse.urlsplit(effective_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Invalid Ollama base_url; must be an http(s) URL.")
        return effective_url, clean_caller_key or "ollama"

    effective_url = raw_caller_url or default_or_url
    parsed = urllib.parse.urlsplit(effective_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid LLM base_url; must be an http(s) URL.")

    # Block cloud link-local metadata endpoints (169.254.169.254) unconditionally
    try:
        ip = ipaddress.ip_address(parsed.hostname.strip("[]"))
        if ip.is_link_local:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Link-local metadata addresses are not permitted as LLM base_url.",
            )
    except ValueError:
        pass

    # Never send server's OPENROUTER_API_KEY to a caller-overridden host!
    default_parsed = urllib.parse.urlsplit(default_or_url)
    same_trusted_origin = (
        parsed.scheme.lower() == default_parsed.scheme.lower()
        and (parsed.hostname or "").lower() == (default_parsed.hostname or "").lower()
        and parsed.port == default_parsed.port
    )
    if raw_caller_url and not same_trusted_origin:
        effective_key = clean_caller_key
    else:
        effective_key = clean_caller_key or env_openrouter_key

    return effective_url, effective_key


# ---------------------------------------------------------------------------
# FastAPI Router (`/quick_pp/demo/*`)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/demo", tags=["Live Demo Telemetry & Session"])


class DemoHeartbeatRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Client tab session identifier (falls back to X-Demo-Session-Id header)",
    )


def _extract_client_ip(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else "")
        or ""
    ).strip()


@router.get("/status", summary="Get live demo online user count and capacity status")
def api_demo_status(request: Request):
    sid = request.headers.get(DEMO_SESSION_HEADER)
    return get_demo_status_snapshot(session_id=sid)


@router.post("/heartbeat", summary="Register or refresh active demo browser session")
def api_demo_heartbeat(
    request: Request,
    payload: Optional[DemoHeartbeatRequest] = None,
):
    sid = (
        payload.session_id if payload and payload.session_id else None
    ) or request.headers.get(DEMO_SESSION_HEADER)
    if not is_demo_mode():
        return get_demo_status_snapshot(session_id=sid)
    return register_or_touch_session(
        session_id=sid,
        client_ip=_extract_client_ip(request),
    )


@router.post(
    "/leave",
    summary="Release active demo browser session and clean up session artifacts",
)
async def api_demo_leave(request: Request):
    sid = request.headers.get(DEMO_SESSION_HEADER)
    if not sid:
        try:
            raw = await request.body()
            if raw:
                import json

                body = json.loads(raw.decode("utf-8"))
                if isinstance(body, dict):
                    sid = body.get("session_id")
        except Exception:
            pass
    return leave_session(session_id=sid)
