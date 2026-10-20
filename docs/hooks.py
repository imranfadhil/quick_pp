"""MkDocs build hook that syncs tutorial notebooks from `notebooks/` into `docs/tutorials/`.

This avoids duplicating `.ipynb` files in the repository while allowing `mkdocs-jupyter`
to render `notebooks/03_*.ipynb` and `notebooks/04_*.ipynb` directly as visual tutorials.
"""

from pathlib import Path
import shutil


TUTORIAL_PATTERNS = ("03_*.ipynb", "04_*.ipynb")


def on_pre_build(config, **kwargs) -> None:
    repo_root = Path(config["config_file_path"]).resolve().parent
    notebooks_dir = repo_root / "notebooks"
    target_dir = Path(config["docs_dir"]) / "tutorials"
    target_dir.mkdir(parents=True, exist_ok=True)

    for pattern in TUTORIAL_PATTERNS:
        for nb_path in sorted(notebooks_dir.glob(pattern)):
            dest = target_dir / nb_path.name
            if not dest.exists() or nb_path.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(nb_path, dest)

