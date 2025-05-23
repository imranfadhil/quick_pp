# quick_pp

Python package to assist in providing quick-look/ preliminary petrophysical estimation.
![quick_pp demo](docs/static/quick_pp_demo.gif)

## Quick Start (Jupyter Notebook Examples)
1. Create virtual environment (tested working with Python3.10.9)
    > python -m venv venv
2. Activate virtual environment
    > venv\Scripts\activate (Windows)

    > source venv/bin/activate (Linux)
3. Install requirements
    > pip install -r requirements.txt

4. Launch the notebook and run the cells
    - 01_data_handler: create the MOCK qppp project file.
    - 02_EDA: quick look on the data
    - 03_*: quick petropohysical interpretation of the MOCK wells.
    - For API notebook, need to run the following before running the cells
        > uvicorn quick_pp.api.main:app

## Install
To install, use the following command:  
  
  `pip install quick_pp`

To use qpp_assistant, you would need to;
1. Install Ollama
2. Run `ollama pull qwen3` in the terminal

## CLI
To start the API server 
> quick_pp api-server

You can then access the Swagger UI at http://localhost:8888/docs and qpp_assistant at http://localhost:8888/qpp_assistant.

To use the mcp tools, you would need to first add the following SSE URLS through the interface;
http://localhost:8888/mcp - quick_pp tools.

http://localhost:5555/mcp - quick_pp ML model prediction tools (will only be available after ML models are trained and registered).


To train an ML model 
> quick_pp train <model_key> <data_hash>

To run prediction
> quick_pp predict <model_key> <data_hash>

To run the MLFlow server 
> quick_pp mlflow-server

You can access the mlflow server at http://localhost:5015

To deploy the trained ML models 
> quick_pp model-deployment

You can access the deployed model Swagger UI at http://localhost:5555/docs

## Documentation
Documentation is available at:
<https://quick-pp.readthedocs.io/en/latest/index.html>
