# quick_pp

Python package to assist in providing quick-look/ preliminary petrophysical estimation.
![quick_pp demo](docs/static/quick_pp_demo.gif)

## Quick Start (Jupyter Notebook Examples)
1. Create virtual environment (tested working with Python3.11)
    
        python -m venv venv

2. Activate virtual environment
        
        > venv\Scripts\activate (Windows)

        > source venv/bin/activate (Linux)

3. Install requirements

        pip install -r requirements.txt

4. Launch the notebook and run the cells
    - 01_data_handler: create the MOCK qppp project file.
    - 02_EDA: quick look on the data
    - 03_*: quick petropohysical interpretation of the MOCK wells.
    - For API notebook, need to run the following before running the cells

            python main.py app

## Install
To install, use the following command:  
  
  `pip install quick_pp`

To use qpp_assistant, you would need to;
1. Run `git clone https://github.com/imranfadhil/quick_pp.git`
2. Run `pip install -r requirements.txt`
3. Specify the required credentials in .env (based on `.env copy` file)
4. Run `docker-compose up -d`
5. Go to Langflow at http://localhost:7860 and build your flow.
6. Run `python main.py app` and go to the qpp Assistant at http://localhost:8888/qpp_assistant to test your flow.

## CLI

To train an ML model, these are the requirements;

1. The input file in parquet format need to be available; /data/input/<data_hash>___.parquet

2. The parquet file need to have the input and target features as specified in MODELLING_CONFIG in config.py.

> quick_pp train <model_config> <data_hash>

    quick_pp train mock mock

To run the MLflow server 

    quick_pp mlflow-server

You can access the mlflow server at http://localhost:5015

To run prediction, the trained models need to be registered in MLflow first.

> quick_pp predict <model_config> <data_hash>

    quick_pp predict mock mock

To deploy the trained ML models

    quick_pp model-deployment

You can access the deployed model Swagger UI at http://localhost:5555/docs

To start the App

    quick_pp app

You can then access the Swagger UI at http://localhost:8888/docs and qpp_assistant at http://localhost:8888/qpp_assistant.
You can enter any user name and password to login the qpp_assistant.

To use the mcp tools, you would need to first add the following SSE URLS through the interface;
http://localhost:8888/mcp - quick_pp tools.

http://localhost:5555/mcp - quick_pp ML model prediction tools (need to run `quick_pp model-deployment` first).

## Documentation
Documentation is available at:
<https://quick-pp.readthedocs.io/en/latest/index.html>
