#!/usr/bin/env python3
"""Pre-publish verification, version synchronization, Docker Hub push, and PyPI release script for quick_pp.

Usage examples:
    # 1. Verify current version (run tests, frontend check/build, build wheel+sdist, twine check, wheel audit)
    python scripts/release.py

    # 2. Bump version across all files and run full verification
    python scripts/release.py --bump 0.3.2

    # 3. Fast build & check (skip full pytest suite or frontend build if already verified)
    python scripts/release.py --skip-tests

    # 4. Upload verified dist/* to PyPI and build+push Docker image to Docker Hub
    python scripts/release.py --publish

    # 5. Upload to PyPI without pushing Docker image (or push Docker image only)
    python scripts/release.py --publish --skip-docker
    python scripts/release.py --docker-only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "quick_pp" / "app" / "frontend"
DOCKERFILE_PATH = ROOT / "quick_pp" / "app" / "docker" / "Dockerfile"
DOCKER_IMAGE_REPO = "imranfadhil86/quick_pp-backend"

VERSION_FILES = {
    "pyproject.toml": (
        ROOT / "pyproject.toml",
        r'(^version\s*=\s*")([^"]+)(")',
    ),
    "quick_pp/__init__.py": (
        ROOT / "quick_pp" / "__init__.py",
        r'(^__version__\s*=\s*")([^"]+)(")',
    ),
    "CITATION.cff": (
        ROOT / "CITATION.cff",
        r"(^version:\s*)(\S+)()",
    ),
    "quick_pp/app/frontend/package.json": (
        FRONTEND_DIR / "package.json",
        r'("version":\s*")([^"]+)(")',
    ),
    "quick_pp/app/frontend/package-lock.json": (
        FRONTEND_DIR / "package-lock.json",
        r'("name":\s*"frontend",\r?\n\s*"version":\s*")([^"]+)(")',
    ),
    "quick_pp/app/frontend/src/lib/state/auth.svelte.ts": (
        FRONTEND_DIR / "src" / "lib" / "state" / "auth.svelte.ts",
        r"(appVersion\s*=\s*\$state<string>\(')([^']+)('\))",
    ),
}

FORBIDDEN_WHEEL_PATTERNS = (
    "node_modules/",
    ".svelte-kit/",
    ".env",
    ".db",
    "__pycache__/",
)


def step(title: str) -> None:
    print(f"\n{'=' * 72}\n>>> {title}\n{'=' * 72}", flush=True)


def run_cmd(cmd: list[str], cwd: Path = ROOT, shell: bool = False) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(cwd), shell=shell)
    if res.returncode != 0:
        sys.exit(res.returncode)


def read_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for label, (path, pattern) in VERSION_FILES.items():
        text = path.read_text(encoding="utf-8")
        m = re.search(pattern, text, flags=re.MULTILINE)
        if not m:
            raise RuntimeError(f"Could not find version pattern in {label}")
        versions[label] = m.group(2)
    return versions


def bump_version(new_version: str) -> None:
    step(f"Bumping version to {new_version}")
    if not re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9.\-]+)?$", new_version):
        raise ValueError(f"Invalid semantic version: {new_version}")

    for label, (path, pattern) in VERSION_FILES.items():
        text = path.read_text(encoding="utf-8")
        max_replacements = 2 if label.endswith("package-lock.json") else 1
        updated, count = re.subn(
            pattern,
            rf"\g<1>{new_version}\g<3>",
            text,
            count=max_replacements,
            flags=re.MULTILINE,
        )
        if count == 0:
            raise RuntimeError(f"Failed to update version in {label}")
        path.write_text(updated, encoding="utf-8")
        print(f"  Updated {label} -> {new_version}")

    uv_bin = shutil.which("uv")
    if uv_bin and (ROOT / "uv.lock").exists():
        run_cmd([uv_bin, "lock"])


def verify_version_sync() -> str:
    step("Checking version consistency across repository")
    versions = read_versions()
    for label, ver in versions.items():
        print(f"  {label:<52} : {ver}")
    unique = set(versions.values())
    if len(unique) != 1:
        raise RuntimeError(
            f"Version mismatch detected across files: {versions}. "
            "Run `python scripts/release.py --bump <version>` to synchronize."
        )
    version = unique.pop()
    print(f"All files synchronized at v{version}")
    return version


def audit_wheel(wheel_path: Path, expected_version: str) -> None:
    step(f"Auditing wheel contents: {wheel_path.name}")
    with zipfile.ZipFile(wheel_path, "r") as zf:
        names = zf.namelist()
        leaked = [
            name
            for name in names
            for pat in FORBIDDEN_WHEEL_PATTERNS
            if pat in name and not name.endswith(".env.example")
        ]
        if leaked:
            raise RuntimeError(
                f"Forbidden files found inside wheel {wheel_path.name}:\n  "
                + "\n  ".join(leaked[:20])
            )

        init_candidates = [n for n in names if n.endswith("quick_pp/__init__.py")]
        if not init_candidates:
            raise RuntimeError("quick_pp/__init__.py missing from wheel!")

        init_content = zf.read(init_candidates[0]).decode("utf-8")
        if f'__version__ = "{expected_version}"' not in init_content:
            raise RuntimeError(
                f'Wheel quick_pp/__init__.py does not contain __version__ = "{expected_version}"'
            )

        frontend_entry = [
            n for n in names if n.endswith("quick_pp/app/frontend/build/index.js")
        ]
        if not frontend_entry:
            raise RuntimeError(
                "quick_pp/app/frontend/build/index.js missing from wheel! "
                "Ensure `npm run build` succeeded before building the wheel."
            )

    size_mb = wheel_path.stat().st_size / (1024 * 1024)
    print(f"  Total files in wheel : {len(names)}")
    print(f"  Wheel archive size   : {size_mb:.2f} MB")
    print("  Wheel audit passed (no node_modules, .svelte-kit, .env, or .db files leaked).")


def build_and_push_docker(version: str) -> None:
    """Build and push `imranfadhil86/quick_pp-backend:<version>` and `:latest` to Docker Hub."""
    step(f"Building & pushing Docker image ({DOCKER_IMAGE_REPO}:{version} & :latest)")
    if not shutil.which("docker"):
        raise RuntimeError("`docker` CLI not found in PATH. Install/start Docker or pass `--skip-docker`.")

    # Load .env if present for DOCKER_USERNAME / DOCKER_PASSWORD
    try:
        from dotenv import load_dotenv

        if (ROOT / ".env").exists():
            load_dotenv(dotenv_path=str(ROOT / ".env"))
    except Exception:
        pass

    tag_latest = f"{DOCKER_IMAGE_REPO}:latest"
    tag_version = f"{DOCKER_IMAGE_REPO}:{version}"

    run_cmd(
        [
            "docker",
            "build",
            "-f",
            str(DOCKERFILE_PATH),
            "-t",
            tag_latest,
            "-t",
            tag_version,
            str(ROOT),
        ]
    )

    docker_user = os.environ.get("DOCKER_USERNAME", "").strip()
    docker_pass = os.environ.get("DOCKER_PASSWORD", "").strip()
    if docker_user and docker_pass:
        print(f"$ docker login --username {docker_user} --password-stdin", flush=True)
        proc = subprocess.Popen(
            ["docker", "login", "--username", docker_user, "--password-stdin"],
            stdin=subprocess.PIPE,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        proc.communicate(input=docker_pass.encode("utf-8"))
        if proc.returncode != 0:
            sys.exit(proc.returncode)

    run_cmd(["docker", "push", tag_version])
    run_cmd(["docker", "push", tag_latest])
    print(f"Successfully pushed {tag_version} and {tag_latest} to Docker Hub.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize version, test, build, and publish quick_pp to PyPI and Docker Hub."
    )
    parser.add_argument(
        "--bump",
        metavar="VERSION",
        help="Bump version across pyproject.toml, __init__.py, CITATION.cff, and frontend",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest suite",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip running npm run check & npm run build",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip building and pushing Docker image during --publish",
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help="Only build and push the Docker image (skip Python build and PyPI upload)",
    )
    parser.add_argument(
        "--testpypi",
        action="store_true",
        help="Upload built distributions to TestPyPI after verification",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload built distributions to production PyPI and push Docker image after verification",
    )
    args = parser.parse_args()

    if args.bump:
        bump_version(args.bump.strip())

    version = verify_version_sync()

    if args.docker_only:
        build_and_push_docker(version)
        return

    # 1. Run Python tests
    if not args.skip_tests:
        step("Running Python test suite (pytest)")
        run_cmd([sys.executable, "-m", "pytest", "-q"])

    # 2. Frontend type-check & production build
    if not args.skip_frontend and FRONTEND_DIR.exists():
        step("Running Frontend svelte-check & production build")
        frontend_build_dir = FRONTEND_DIR / "build"
        if frontend_build_dir.exists():
            shutil.rmtree(frontend_build_dir)
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        run_cmd([npm_cmd, "run", "check"], cwd=FRONTEND_DIR)
        run_cmd([npm_cmd, "run", "build"], cwd=FRONTEND_DIR)

    # 3. Clean & build wheel + sdist
    step("Building Python sdist and wheel into dist/")
    for folder in ("dist", "build"):
        target = ROOT / folder
        if target.exists():
            shutil.rmtree(target)

    uv_bin = shutil.which("uv")
    if uv_bin:
        run_cmd([uv_bin, "build"])
    else:
        run_cmd([sys.executable, "-m", "build"])

    dist_files = [
        f
        for f in sorted((ROOT / "dist").iterdir())
        if f.name.endswith((".whl", ".tar.gz"))
    ]
    if not dist_files:
        raise RuntimeError("No distribution files produced in dist/")

    # 4. Twine metadata check
    step("Validating package metadata and README with twine check")
    twine_cmd = (
        [uv_bin, "tool", "run", "twine"]
        if uv_bin
        and subprocess.run(
            [sys.executable, "-c", "import twine"], capture_output=True
        ).returncode
        != 0
        else [sys.executable, "-m", "twine"]
    )
    run_cmd([*twine_cmd, "check", *[str(f) for f in dist_files]])

    # 5. Wheel content audit
    wheels = list((ROOT / "dist").glob("*.whl"))
    if not wheels:
        raise RuntimeError("No .whl file found in dist/")
    audit_wheel(wheels[0], version)

    # 6. Smoke-test import
    step("Smoke-testing quick_pp version and CLI entrypoint")
    run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import quick_pp; "
                f"assert quick_pp.__version__ == '{version}', f'Got {{quick_pp.__version__}}'; "
                "from quick_pp.cli import cli; "
                "print('Smoke test OK -> quick_pp v' + quick_pp.__version__)"
            ),
        ]
    )

    # 7. Optional upload to TestPyPI / Production PyPI + Docker Hub
    if args.testpypi:
        step("Uploading to TestPyPI")
        run_cmd(
            [
                *twine_cmd,
                "upload",
                "--repository",
                "testpypi",
                *[str(f) for f in dist_files],
            ]
        )
    elif args.publish:
        step("Uploading to Production PyPI")
        run_cmd([*twine_cmd, "upload", *[str(f) for f in dist_files]])
        if not args.skip_docker:
            build_and_push_docker(version)
    else:
        step(f"Release v{version} verified and ready in dist/")
        print("To publish to PyPI + Docker Hub when ready, run:")
        print("  python scripts/release.py --skip-tests --skip-frontend --publish")


if __name__ == "__main__":
    main()
