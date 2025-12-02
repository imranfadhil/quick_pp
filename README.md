# quick_pp

A Python package for quick-look preliminary petrophysical estimations.

![quick_pp demo](docs/static/quick_pp_demo.gif)

## Installation

You can install `quick_pp` directly from PyPI:

```bash
pip install quick_pp
```

For development or to use the `qpp_assistant`, you'll need to clone the repository and install dependencies:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/imranfadhil/quick_pp.git
    cd quick_pp
    ```

2.  **Create and activate a virtual environment** (tested with Python 3.11):
    ```bash
    uv venv --python 3.11
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Quick Start

### Jupyter Notebook Examples

> More structured analysis/ examples are done in https://github.com/imranfadhil/pp_portfolio

The included notebooks demonstrate the core functionalities:

-   `01_data_handler`: Create a MOCK `qppp` project file.
-   `02_EDA`: Perform a quick exploratory data analysis.
-   `03_*`: Carry out petrophysical interpretation of the MOCK wells.

**Note:** For the API notebook, you need to run `python main.py app` before executing the cells.

### `qpp_assistant` Setup

To use the `qpp_assistant`, follow these steps after the development installation:

1.  Specify the required credentials in a `.env` file (you can use `.env copy` as a template).
2.  Run Docker Compose: `docker-compose up -d`.
3.  Build your flow in Langflow at `http://localhost:7860`.
4.  Run the main application: `python main.py app`.
5.  Test your flow in the qpp Assistant at `http://localhost:6312/qpp_assistant`.

## CLI

### Train a Machine Learning Model

**Requirements:**
-   The input data must be a Parquet file located at `/data/input/<data_hash>___.parquet`.
-   The Parquet file must contain the input and target features as specified in `MODELLING_CONFIG` in `config.py`.

**Command:**

> quick_pp train <model_config> <data_hash>

    quick_pp train mock mock

### Run the MLflow Server

**Command:**
```bash
quick_pp mlflow-server
```
You can access the MLflow UI at `http://localhost:5015`.

### Run Predictions

> **Note:** Trained models must be registered in MLflow before running predictions.

> quick_pp predict <model_config> <data_hash>

**Example:**
```bash
    quick_pp predict mock mock
```

### Deploy Trained Models as an API

```bash
quick_pp model-deployment
```
You can access the deployed model's Swagger UI at `http://localhost:5555/docs`.

### Start the Main Application

```bash
quick_pp app
```
-   **API Docs:** `http://localhost:6312/docs`
-   **qpp_assistant:** `http://localhost:6312/qpp_assistant` (you can log in with any username and password).

To use the mcp tools, you would need to first add the following SSE URLS through the interface;
http://localhost:6312/mcp - quick_pp tools.

http://localhost:5555/mcp - quick_pp ML model prediction tools (need to run `quick_pp model-deployment` first).

## Documentation

Documentation is available at:
<https://quick-pp.readthedocs.io/en/latest/index.html>
