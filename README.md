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

## Documentation
Documentation is available at:
<https://quick-pp.readthedocs.io/en/latest/index.html>
