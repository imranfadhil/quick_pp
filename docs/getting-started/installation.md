# Installation

## Modular Extras (`pyproject.toml`)

`quick-pp` organizes its dependencies into modular install targets so you can install only what your workflow requires:

| Install Target | Command | Description |
| :--- | :--- | :--- |
| **`core`** *(default)* | `uv pip install "quick-pp[core]"` | Core petrophysics, SCAL, uncertainty, upscaling, LAS handling, and plotting (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `plotly`, `lasio`, `welly`, `wellpathpy`). |
| **`ml`** | `uv pip install "quick-pp[ml]"` | Physics-Informed Machine Learning, MLflow tracking, SMOTE, and MAPIE conformal prediction (`mlflow`, `mapie`, `imbalanced-learn`, `seaborn`, `pyarrow`, `fastparquet`). |
| **`app`** | `uv pip install "quick-pp[app]"` | FastAPI web backend, MCP server, and SQLAlchemy/PostgreSQL (`fastapi`, `uvicorn`, `orjson`, `sqlalchemy`, `psycopg2-binary`). |
| **`all`** | `uv pip install "quick-pp[all]"` | Installs `core`, `ml`, `app`, plus Jupyter notebook dependencies (`jupyter`, `ipykernel`, `ipywidgets`, `bokeh`, `openpyxl`). |

---

## Local Development Setup with `uv`

1. Clone the repository and create a Python 3.11+ virtual environment:

```bash
git clone https://github.com/imranfadhil/quick_pp.git
cd quick_pp
uv venv --python 3.11

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

2. Install in editable mode with your desired extras:

```bash
uv pip install -e ".[all]"
```

3. Build or serve documentation locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

