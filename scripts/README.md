# `quick_pp` Release & PyPI Publishing Guide

The [`release.py`](./release.py) script automates version synchronization, full-stack testing, wheel/sdist building, metadata validation (`twine check`), wheel content auditing, and PyPI uploading.

---

## 1. Files Kept in Sync by `--bump`

Running `python scripts/release.py --bump <VERSION>` automatically updates:
1. [`pyproject.toml`](../pyproject.toml) (`version = "<VERSION>"`)
2. [`quick_pp/__init__.py`](../quick_pp/__init__.py) (`__version__ = "<VERSION>"`)
3. [`CITATION.cff`](../CITATION.cff) (`version: <VERSION>`)
4. [`quick_pp/app/frontend/package.json`](../quick_pp/app/frontend/package.json) (`"version": "<VERSION>"`)
5. [`quick_pp/app/frontend/src/lib/state/auth.svelte.ts`](../quick_pp/app/frontend/src/lib/state/auth.svelte.ts) (`appVersion` fallback)
6. [`uv.lock`](../uv.lock) (via `uv lock`)

> **Note:** Remember to also add release notes for the new version in [`docs/community/changelog.md`](../docs/community/changelog.md).

---

## 2. Standard Release Workflow (Future Releases)

### Step A — Bump Version & Run Full Verification Pipeline
From the repository root:
```powershell
.\.venv\Scripts\python.exe scripts/release.py --bump 0.3.1
```
Or if the version is already bumped and you want to run the full verification pipeline:
```powershell
.\.venv\Scripts\python.exe scripts/release.py
```

This executes 6 automated checks before any upload:
1. **Version Sync Check**: Confirms all 5 version files match.
2. **Backend & Library Tests**: Runs `pytest -q` across all unit, API, UAM, and live demo tests.
3. **Frontend Check & Build**: Runs `npm run check` (`svelte-check`) and `npm run build` in `quick_pp/app/frontend` so the compiled production UI is bundled into the wheel.
4. **Clean Package Build**: Cleans `dist/` and `build/`, then builds `dist/quick_pp-<ver>.tar.gz` and `dist/quick_pp-<ver>-py3-none-any.whl` via `uv build`.
5. **Twine & Wheel Audit**: Runs `twine check` on the archives and inspects the `.whl` zip contents to guarantee no `node_modules/`, `.svelte-kit/`, `.env`, or `.db` files leaked.
6. **Smoke Test**: Imports `quick_pp` and verifies `quick_pp.__version__` and the `quick_pp` CLI entrypoint.

### Step B — Commit & Tag the Release in Git
```powershell
git add pyproject.toml quick_pp/__init__.py CITATION.cff uv.lock docs/community/changelog.md quick_pp/app/frontend/package.json quick_pp/app/frontend/src/lib/state/auth.svelte.ts
git commit -m "Bump version to 0.3.1"
git tag v0.3.1
git push origin main --tags
```

### Step C — Publish to PyPI
Once Step A has succeeded and built `dist/`, publish directly without re-running tests:
```powershell
# Optional: Upload to TestPyPI first
.\.venv\Scripts\python.exe scripts/release.py --skip-tests --skip-frontend --testpypi

# Upload to Production PyPI
.\.venv\Scripts\python.exe scripts/release.py --skip-tests --skip-frontend --publish
```

> **Authentication Tip:** `twine` will prompt for your PyPI API token (`__token__`), or you can set `$env:TWINE_USERNAME="__token__"` and `$env:TWINE_PASSWORD="pypi-..."` (or configure `~/.pypirc`).

---

## 3. CLI Reference

| Command | Description |
| :--- | :--- |
| `python scripts/release.py` | Verify version sync, run `pytest`, build frontend, build wheel/sdist, audit & smoke-test. |
| `python scripts/release.py --bump X.Y.Z` | Update version across all files + `uv.lock`, then run the full verification pipeline. |
| `python scripts/release.py --skip-tests` | Skip `pytest` during build verification. |
| `python scripts/release.py --skip-frontend` | Skip `npm run check` and `npm run build` if frontend is already built. |
| `python scripts/release.py --testpypi` | Upload verified `dist/*` artifacts to TestPyPI. |
| `python scripts/release.py --publish` | Upload verified `dist/*` artifacts to production PyPI. |

