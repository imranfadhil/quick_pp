from fastapi import FastAPI

from .router import api_router

app = FastAPI(
    title="QuickPP API",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    debug=True
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
