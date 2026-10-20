# Contributing

Contributions are welcome and greatly appreciated!

## Ways to Contribute

### Report Bugs
Report bugs at [https://github.com/imranfadhil/quick_pp/issues](https://github.com/imranfadhil/quick_pp/issues). Please include:
- Your operating system name and version.
- Relevant details about your Python environment.
- Detailed steps to reproduce the issue.

### Fix Bugs & Implement Features
Look through the GitHub issues tagged with `bug`, `enhancement`, or `help wanted`.

### Write Documentation
Documentation improvements—whether in docstrings, tutorial notebooks (`notebooks/`), or Markdown guides (`docs/`)—are always welcome.

---

## Development Setup with `uv`

1. Fork the `quick_pp` repository on GitHub and clone your fork locally:
   ```bash
   git clone git@github.com:your_username/quick_pp.git
   cd quick_pp
   ```
2. Create a Python 3.11+ virtual environment and install development dependencies:
   ```bash
   uv venv --python 3.11
   # Activate venv (.venv\Scripts\activate on Windows, source .venv/bin/activate on Linux/macOS)
   uv pip install -e ".[dev]"
   ```
3. Create a branch for your feature or fix:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. Run linting and unit tests:
   ```bash
   ruff check quick_pp tests
   pytest -q
   ```
5. Preview documentation locally:
   ```bash
   mkdocs serve
   ```
6. Commit your changes, push your branch, and open a Pull Request.

