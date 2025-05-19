from fastapi import FastAPI, Request
from fastapi_mcp import FastApiMCP
from starlette.types import Message
from datetime import datetime
import mlflow
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking

# # Uncomment below 3 lines to run >> if __name__ == "__main__"
# import os
# import sys
# sys.path.append(os.getcwd())

from quick_pp.api.fastapi_mlflow.applications import build_app
from quick_pp.modelling.utils import get_model_info, run_mlflow_server


app = FastAPI(title='quick_pp - ML Models', debug=True)


# set_body and get_body required to extract request body in middleware
async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}
    request._receive = receive


async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body


client = mlflow_tracking.MlflowClient()

start_time = datetime.now()
model_count = 0
print(f"Tracking uri {mlflow.get_tracking_uri()}")

try:
    # Get latest registered models
    for rm in client.search_registered_models():
        model_info = get_model_info(rm.latest_versions)
        if 'reg_model_name' in model_info.keys():
            model_uri = f"models:/{model_info['reg_model_name']}/{model_info['version']}"
            model = load_model(model_uri)

            model_count += 1
            # Build API for the loaded model
            route = fr"/{model_info['reg_model_name']}"
            app = build_app(app, model, route)
            print(f"Mounting #{model_count} | {route}")

    # Mount the model using FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount()

except Exception as e:
    print(f"Mounting #{model_count} | {model_info['reg_model_name']}/{model_info['version']} - Error: {e}")

duration = round((datetime.now() - start_time).total_seconds() / 60, 3)
print(f"Completed mounting {model_count} models in {duration} minutes")


if __name__ == '__main__':
    import uvicorn

    # Make sure mlflow server is running first
    env = 'local'
    run_mlflow_server(env)

    uvicorn.run("quick_pp.api.mlflow_model_deployment:app", host='0.0.0.0', port=5555, reload=False)
