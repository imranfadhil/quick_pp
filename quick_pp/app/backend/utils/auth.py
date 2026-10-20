"""Self-hosted AWS Cognito-style Authentication, TOTP MFA, and RBAC utilities.

Provides:
- PBKDF2-HMAC-SHA256 password hashing & Cognito-grade password policy validation
- RFC 6238 Time-Based One-Time Password (TOTP) generation & verification
- HS256 JWT token creation & verification for session and multi-step auth challenges
- FastAPI dependencies for user authentication, admin authorization, and project RBAC
- Automatic initial admin bootstrapping & optional SMTP invitation delivery
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import smtplib
import struct
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from quick_pp.app.backend.utils.db import get_db
from quick_pp.database import models as db_models

log = logging.getLogger(__name__)

# Cookie name used for browser sessions alongside Bearer header support
AUTH_COOKIE_NAME = "qpp_access_token"

# Role hierarchy for project-level access control
PROJECT_ROLE_RANK = {
    "viewer": 1,
    "editor": 2,
    "admin": 3,
}


def _get_secret_key() -> bytes:
    """Return the signing secret key for JWT tokens."""
    secret = (
        os.environ.get("QPP_AUTH_SECRET_KEY")
        or os.environ.get("LANGFLOW_SECRET_KEY")
        or os.environ.get("POSTGRES_PASSWORD")
        or "quick-pp-default-cognito-uam-secret-key-change-in-prod"
    )
    return hashlib.sha256(secret.encode("utf-8")).digest()


# ---------------------------------------------------------------------------
# Password Hashing (PBKDF2-HMAC-SHA256) & Cognito Policy Validation
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a 16-byte cryptographic salt."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt_b64}${dk_b64}"


def _b64url_decode(val: str) -> bytes:
    pad = "=" * ((4 - len(val) % 4) % 4)
    return base64.urlsafe_b64decode(val + pad)


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash in constant time."""
    if not stored_hash or not password:
        return False
    try:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = _b64url_decode(parts[2])
        expected_dk = _b64url_decode(parts[3])
        actual_dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual_dk, expected_dk)
    except Exception:
        return False


def validate_password_policy(password: str) -> Tuple[bool, List[str]]:
    """Validate password against AWS Cognito standard security policy.

    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 numeric digit
    - At least 1 symbol / special character
    """
    errors: List[str] = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter (A-Z).")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter (a-z).")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number (0-9).")
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append("Password must contain at least one special character (e.g. !@#$%^&*).")
    return (len(errors) == 0, errors)


def generate_temp_password(length: int = 12) -> str:
    """Generate a cryptographically random temporary password satisfying the Cognito policy."""
    upper = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    lower = "abcdefghijkmnopqrstuvwxyz"
    digits = "23456789"
    symbols = "!@#$%&*?"
    all_chars = upper + lower + digits + symbols

    chars = [
        secrets.choice(upper),
        secrets.choice(upper),
        secrets.choice(lower),
        secrets.choice(lower),
        secrets.choice(digits),
        secrets.choice(digits),
        secrets.choice(symbols),
        secrets.choice(symbols),
    ]
    while len(chars) < max(8, length):
        chars.append(secrets.choice(all_chars))

    # Shuffle using cryptographic randomness
    for i in range(len(chars) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


# ---------------------------------------------------------------------------
# RFC 6238 TOTP (Time-Based One-Time Password)
# ---------------------------------------------------------------------------


def generate_totp_secret() -> str:
    """Generate a 160-bit Base32-encoded TOTP secret (32 characters, no padding)."""
    raw = secrets.token_bytes(20)
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _normalize_base32_secret(secret: str) -> bytes:
    cleaned = re.sub(r"\s+", "", secret).upper().rstrip("=")
    pad = "=" * ((8 - len(cleaned) % 8) % 8)
    return base64.b32decode(cleaned + pad, casefold=True)


def get_totp_code(
    secret: str, timestamp: Optional[float] = None, interval: int = 30, digits: int = 6
) -> str:
    """Compute the RFC 6238 6-digit TOTP code for `secret` at `timestamp`."""
    ts = time.time() if timestamp is None else timestamp
    counter = int(ts // interval)
    key = _normalize_base32_secret(secret)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code_int % (10**digits)).zfill(digits)


def verify_totp_code(
    secret: str,
    code: str,
    timestamp: Optional[float] = None,
    valid_window: int = 1,
    interval: int = 30,
) -> bool:
    """Verify a 6-digit TOTP code allowing `±valid_window` 30-second steps for clock drift."""
    if not secret or not code:
        return False
    cleaned_code = re.sub(r"\s+", "", str(code))
    if not re.fullmatch(r"\d{6}", cleaned_code):
        return False
    ts = time.time() if timestamp is None else timestamp
    try:
        for step in range(-valid_window, valid_window + 1):
            candidate = get_totp_code(secret, timestamp=ts + step * interval, interval=interval)
            if hmac.compare_digest(candidate, cleaned_code):
                return True
    except Exception:
        return False
    return False


def build_otpauth_uri(username: str, secret: str, issuer: str = "quick_pp") -> str:
    """Build standard `otpauth://totp/...` URI for Authenticator QR codes."""
    label = f"{quote(issuer)}:{quote(username)}"
    return f"otpauth://totp/{label}?secret={quote(secret)}&issuer={quote(issuer)}"


# ---------------------------------------------------------------------------
# JWT Creation & Verification (HS256)
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def create_jwt_token(
    subject: int | str,
    token_type: str = "access",
    expires_in_seconds: int = 43200,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed HS256 JWT token."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload: Dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": now,
        "exp": now + int(expires_in_seconds),
        "jti": secrets.token_hex(8),
    }
    if extra_claims:
        payload.update(extra_claims)

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(_get_secret_key(), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_jwt_token(token: str, expected_type: Optional[str] = None) -> Dict[str, Any]:
    """Decode and verify a signed HS256 JWT token."""
    if not token:
        raise ValueError("Missing authentication token")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed authentication token")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(_get_secret_key(), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    now = int(time.time())
    if int(payload.get("exp", 0)) < now:
        raise ValueError("Authentication token has expired")

    if expected_type and payload.get("type") != expected_type:
        raise ValueError(f"Invalid token type (expected '{expected_type}')")

    return payload


# ---------------------------------------------------------------------------
# User Serialization & Audit Logging
# ---------------------------------------------------------------------------


def serialize_user(user: db_models.User, session: Optional[Session] = None) -> Dict[str, Any]:
    """Serialize a `User` ORM instance into a safe dictionary (never exposing password/MFA secret)."""
    memberships: List[Dict[str, Any]] = []
    if session is not None:
        rows = (
            session.execute(
                select(db_models.ProjectMember, db_models.Project.name)
                .join(
                    db_models.Project,
                    db_models.ProjectMember.project_id == db_models.Project.project_id,
                )
                .where(db_models.ProjectMember.user_id == user.user_id)
            )
            .all()
        )
        memberships = [
            {
                "project_id": pm.project_id,
                "project_name": proj_name,
                "role": pm.role,
            }
            for pm, proj_name in rows
        ]

    created_str = (
        user.created_at.isoformat()
        if hasattr(user.created_at, "isoformat")
        else str(user.created_at or "")
    )
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name or "",
        "role": user.role or "user",
        "status": user.status or "FORCE_CHANGE_PASSWORD",
        "is_active": bool(user.is_active),
        "mfa_enabled": bool(user.mfa_enabled),
        "has_invite_token": bool(user.invite_token),
        "invited_by": user.invited_by,
        "last_login_at": user.last_login_at,
        "created_at": created_str,
        "project_memberships": memberships,
    }


def record_audit_event(
    session: Session,
    user_id: Optional[int],
    action: str,
    table_name: str,
    record_id: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a UAM or data action to the `audit_log` table."""
    if is_demo_mode():
        return
    try:
        entry = db_models.AuditLog(
            user_id=user_id,
            action=action[:10],
            table_name=table_name[:100],
            record_id=str(record_id) if record_id is not None else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            changes=changes or {},
        )
        session.add(entry)
        session.flush()
    except Exception as exc:
        log.debug("Failed to record audit log entry: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI Request Extractors & RBAC Dependencies
# ---------------------------------------------------------------------------


def extract_token_from_request(request: Request) -> Optional[str]:
    """Extract JWT token from `Authorization: Bearer <token>` header or `qpp_access_token` cookie."""
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token.strip()
    return None


def is_demo_mode() -> bool:
    """Return True when running in the public interactive demo (`QPP_DEMO_MODE=true`)."""
    return os.environ.get("QPP_DEMO_MODE", "false").lower() in ("1", "true", "yes")


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Return the authenticated user dict if a valid access token is present, else None."""
    token = extract_token_from_request(request)
    if not token:
        if is_demo_mode():
            try:
                with get_db() as session:
                    user = session.scalar(
                        select(db_models.User).where(db_models.User.role == "admin").limit(1)
                    )
                    if user:
                        u_dict = serialize_user(user, session=session)
                        u_dict["status"] = "CONFIRMED"
                        u_dict["mfa_enabled"] = True
                        u_dict["demo_mode"] = True
                        return u_dict
            except Exception:
                pass
            return {
                "user_id": 1,
                "username": "demo_visitor",
                "email": "demo@quickpp.local",
                "full_name": "Interactive Demo Visitor",
                "role": "admin",
                "status": "CONFIRMED",
                "is_active": True,
                "mfa_enabled": True,
                "has_invite_token": False,
                "invited_by": None,
                "last_login_at": None,
                "created_at": "",
                "project_memberships": [],
                "demo_mode": True,
            }
        return None
    try:
        payload = decode_jwt_token(token, expected_type="access")
        user_id = int(payload["sub"])
        with get_db() as session:
            user = session.get(db_models.User, user_id)
            if not user or not user.is_active or user.status == "DISABLED":
                return None
            return serialize_user(user, session=session)
    except Exception:
        return None


def get_current_user(request: Request) -> Dict[str, Any]:
    """Require a valid, active authenticated user."""
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require the current user to have global `admin` role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges are required for this action.",
        )
    return user


def check_project_permission(
    session: Session,
    user: Optional[Dict[str, Any]],
    project_id: int,
    min_role: str = "viewer",
) -> str:
    """Verify whether `user` has at least `min_role` ('viewer', 'editor', 'admin') on `project_id`.

    Note: If `user` is None (e.g., internal CLI/test call without session header), allows access
    unless `QPP_STRICT_API_AUTH=true` is set. When `user` is authenticated, enforces full RBAC.
    """
    if user is None:
        if os.environ.get("QPP_STRICT_API_AUTH", "").lower() in ("1", "true", "yes"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to access this project.",
            )
        return "admin"

    if user.get("role") == "admin":
        return "admin"

    uid = int(user["user_id"])
    proj = session.get(db_models.Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    if proj.created_by == uid:
        return "admin"

    membership = session.get(db_models.ProjectMember, (project_id, uid))
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project. Ask an administrator for an invitation.",
        )

    user_rank = PROJECT_ROLE_RANK.get(membership.role, 0)
    required_rank = PROJECT_ROLE_RANK.get(min_role, 1)
    if user_rank < required_rank:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient project permissions: '{min_role}' role required (your role is '{membership.role}').",
        )
    return membership.role


# ---------------------------------------------------------------------------
# Bootstrap Initial Admin & Optional Email Sending
# ---------------------------------------------------------------------------


def bootstrap_default_admin() -> Optional[Dict[str, str]]:
    """Ensure at least one global administrator exists in the `users` table on startup.

    If no user exists in the database, creates the bootstrap administrator using:
    - `QPP_ADMIN_USERNAME` (default: `admin`)
    - `QPP_ADMIN_EMAIL` (default: `admin@quickpp.local`)
    - `QPP_ADMIN_PASSWORD` (default: `QuickPP@Admin2026!`)
    """
    admin_username = os.environ.get("QPP_ADMIN_USERNAME", "admin").strip() or "admin"
    admin_email = (
        os.environ.get("QPP_ADMIN_EMAIL", "admin@quickpp.local").strip()
        or "admin@quickpp.local"
    )
    admin_password = (
        os.environ.get("QPP_ADMIN_PASSWORD", "QuickPP@Admin2026!").strip()
        or "QuickPP@Admin2026!"
    )

    try:
        with get_db() as session:
            existing_admin = session.scalar(
                select(db_models.User).where(db_models.User.role == "admin")
            )
            if existing_admin is not None:
                return None

            invite_token = secrets.token_urlsafe(24)
            new_admin = db_models.User(
                username=admin_username,
                email=admin_email,
                full_name="System Administrator",
                hashed_password=hash_password(admin_password),
                role="admin",
                status="FORCE_CHANGE_PASSWORD",
                is_active=1,
                mfa_enabled=0,
                invite_token=invite_token,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            session.add(new_admin)
            session.flush()
            record_audit_event(
                session,
                user_id=new_admin.user_id,
                action="BOOTSTRAP",
                table_name="users",
                record_id=str(new_admin.user_id),
                changes={
                    "username": admin_username,
                    "role": "admin",
                    "status": "FORCE_CHANGE_PASSWORD",
                },
            )
            log.warning(
                "=================================================================\n"
                "  quick_pp UAM Bootstrap Administrator Created!\n"
                "  Username          : %s\n"
                "  Temporary Password: %s\n"
                "  Status            : FORCE_CHANGE_PASSWORD + MFA_SETUP_REQUIRED\n"
                "=================================================================",
                admin_username,
                admin_password,
            )
            return {
                "username": admin_username,
                "email": admin_email,
                "temp_password": admin_password,
                "invite_token": invite_token,
            }
    except Exception as exc:
        log.warning("Could not bootstrap default admin user: %s", exc)
        return None


def send_invitation_email(
    to_email: str,
    username: str,
    temp_password: str,
    invite_url: str,
    invited_by_name: str = "Administrator",
) -> Tuple[bool, str]:
    """Send an invitation email via SMTP if `SMTP_HOST` is configured in environment variables."""
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    if not smtp_host:
        return False, "SMTP_HOST not configured; use the generated invite link and temporary password."

    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASSWORD", "").strip()
    smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@quickpp.local").strip()
    use_tls = os.environ.get("SMTP_TLS", "true").lower() in ("1", "true", "yes")

    msg = EmailMessage()
    msg["Subject"] = "You have been invited to quick_pp Subsurface Workbench"
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(
        f"Hello {username},\n\n"
        f"You have been invited by {invited_by_name} to access the quick_pp Subsurface Petrophysical Workbench.\n\n"
        f"Sign-In URL       : {invite_url}\n"
        f"Username          : {username}\n"
        f"Temporary Password: {temp_password}\n\n"
        f"On your first sign-in, you will be prompted to set a permanent password and configure Multi-Factor Authentication (MFA) using an authenticator app (Google Authenticator, Microsoft Authenticator, Authy, or 1Password).\n"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            if use_tls:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, f"Invitation email sent to {to_email}"
    except Exception as exc:
        log.warning("Failed to send SMTP invitation email to %s: %s", to_email, exc)
        return False, f"SMTP delivery failed ({exc}); share the invite link manually."

