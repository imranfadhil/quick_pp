from fastapi import FastAPI, Request
from fastapi_mcp import FastApiMCP
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.types import Message
from datetime import datetime
import mlflow
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking
from importlib import resources

# # Uncomment below 3 lines to run >> if __name__ == "__main__"
# import os
# import sys
# sys.path.append(os.getcwd())

from quick_pp.api.fastapi_mlflow.applications import build_app
from quick_pp.modelling.utils import get_model_info, run_mlflow_server
from quick_pp.logger import logger


app = FastAPI(
    title='quick_pp - ML Models',
    description="API for quick_pp library.",
    contact={"name": "Imran Fadhil",
             "url": "https://github.com/imranfadhil/quick_pp", "email": "imranfadhil@gmail.com"},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    debug=True)
with resources.files('quick_pp.api') as api_folder:
    static_folder = api_folder / "public"
app.mount("/static", StaticFiles(directory=static_folder), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


# set_body and get_body required to extract request body in middleware
async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}
    request._receive = receive


async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body

# Run MLflow server
run_mlflow_server('local')

client = mlflow_tracking.MlflowClient()

start_time = datetime.now()
model_count = 0
logger.info(f"Tracking uri {mlflow.get_tracking_uri()}")

try:
    # Get latest registered models
    for rm in client.search_registered_models():
        model_info = get_model_info(rm.latest_versions)
        if 'reg_model_name' in model_info.keys():
            model_uri = f"models:/{model_info['reg_model_name']}/{model_info['version']}"
            logger.info(f"Loading model: {model_uri}")
            model = load_model(model_uri)

            model_count += 1
            # Build API for the loaded model
            route = fr"/{model_info['reg_model_name']}"
            app = build_app(app, model, route)
            logger.info(f"Mounted model #{model_count} at route: {route}")

    # Set up CORS middleware
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=origins,
        allow_headers=origins,
    )

    # Mount the model using FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount()
    logger.info("FastApiMCP mounted successfully.")

except Exception as e:
    logger.error(f"Mounting #{model_count} | {model_info.get('reg_model_name', 'unknown')}/"
                 f"{model_info.get('version', 'unknown')} - Error: {e}")

duration = round((datetime.now() - start_time).total_seconds() / 60, 3)
logger.info(f"Completed mounting {model_count} models in {duration} minutes")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("quick_pp.api.mlflow_model_deployment:app", host='0.0.0.0', port=5555, reload=False)
