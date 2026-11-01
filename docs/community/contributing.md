# Contributing to `quick_pp`

Thank you for your interest in contributing to `quick_pp`! We welcome contributions from petrophysicists, geoscientists, data scientists, software engineers, and students.

Whether you are reporting a bug, writing documentation, adding a tutorial notebook, or proposing a new petrophysical method, your help is appreciated.

---

## 💡 Core Philosophy & Guiding Principles

Before writing code, please keep `quick_pp`'s architectural principles in mind:

1. **Lightweight & Modular**:
   - `quick_pp` is built to be fast, scriptable, and transparent.
   - **Keep dependencies minimal.** Always consider whether a routine can be solved using standard Python or existing core packages (`numpy`, `scipy`, `pandas`) before proposing a new external library.
   - Specialized capabilities (FastAPI backend, MLflow machine learning pipelines, visualization servers) belong in modular optional extras (`[app]`, `[ml]`, `[all]`), keeping the `[core]` petrophysics package featherlight.

2. **Discuss Before Implementing Large Features**:
   - **Please open an [Issue](https://github.com/imranfadhil/quick_pp/issues) or [GitHub Discussion](https://github.com/imranfadhil/quick_pp/discussions) first** before embarking on significant new features, architectural modifications, or API redesigns.
   - This ensures your proposal fits within `quick_pp`'s scope and design goals, avoiding unnecessary or conflicting work.
   - For bug fixes, typos, doc updates, or small optimizations, you are welcome to submit a Pull Request directly.

3. **Geoscientific & Physical Rigor**:
   - Petrophysical algorithms, empirical equations, and cutoffs should be physically grounded and cite literature/benchmarks where applicable (e.g., Archie, Simandoux, Thomas-Stieber, Leverett, Brooks-Corey).

---

## 🛠️ Development Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment management.

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/quick_pp.git
cd quick_pp
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv --python 3.11
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Or standard Python venv
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Development Dependencies

Install the editable package with full development and testing extras:

```bash
# Using uv
uv pip install -e ".[dev]"

# Or pip
pip install -e ".[dev]"
```

### 4. Create a Working Branch

Use descriptive branch names:
```bash
git checkout -b feat/rock-typing-enhancement
# or
git checkout -b fix/las-splicing-null-gap
```

---

## 📐 Standards & Quality Checks

All contributions must meet our formatting, testing, and typing standards before they can be merged.

### 1. Testing (`pytest`)
All unit and regression tests must pass. When adding new functions or fixing bugs, include corresponding tests in `tests/`.

```bash
# Run test suite
pytest -q

# Run with coverage report
pytest --cov=quick_pp tests/
```

- Tests must run fast and run offline.
- Use synthetic or mock log data rather than committing proprietary commercial well logs.

### 2. Code Formatting & Linting (`ruff` & `black`)
We enforce consistent style across the entire codebase:
- **Line length**: 88 characters.
- **Ruff**: Used for high-speed linting, code style, and import sorting.
- **Black**: Standard code formatting.

Run the linter and formatters:
```bash
# Check for linting violations and import order
ruff check quick_pp tests

# Auto-fix common issues
ruff check --fix quick_pp tests

# Check formatting
black --check quick_pp tests

# Format in-place
black quick_pp tests
```

### 3. Type Annotations (`mypy`)
Public functions, methods, and classes should include type annotations:

```bash
mypy quick_pp
```

### 4. Documentation (`mkdocs`)
Documentation lives in `docs/` and is built using Material for MkDocs.
- Function docstrings should follow **Google style**.
- Preview documentation changes locally:

```bash
mkdocs serve
# Open http://127.0.0.1:8000 in your browser
```

---

## 📥 Submitting a Pull Request

1. **Commit your changes**: Write clear, imperative commit messages (e.g., `feat(lithology): add modified kuttan silt solver` or `fix(las): preserve missing null values`).
2. **Sync with upstream**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
3. **Verify all checks pass**:
   ```bash
   ruff check quick_pp tests
   black --check quick_pp tests
   pytest -q
   ```
4. **Push and open PR**: Push your branch to your fork and submit a PR against `main`. Provide a clear description of what changed, reference any linked issues (`Closes #123`), and include screenshots if modifying visual plots or the SvelteKit UI.

---

## 👥 Recognition

All contributors are valued! Contributors will be featured in the repository's `README.md` and `SUPPORTERS.md`.
