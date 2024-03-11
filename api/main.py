from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .exceptions import return_exception_message

from .router import api_router

tags_metadata = [
    {
        "name": "Lithology",
        "description": "Lithology related endpoints."
    },
    {
        "name": "Porosity",
        "description": "Porosity related endpoints."
    },
    {
        "name": "Saturation",
        "description": "Saturation related endpoints."
    },
    {
        "name": "Permeability",
        "description": "Permeability related endpoints."
    },
    {
        "name": "Reservoir Summary",
        "description": "Reservoir summary related endpoints."
    }
]

app = FastAPI(
    title="quick_pp API",
    description="API for quick_pp library.",
    contact={"name": "Imran Fadhil",
             "url": "https://github.com/imranfadhil/quick_pp", "email": "imranfadhil@gmail.com"},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_tags=tags_metadata,
    debug=False
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Welcome to quick_pp API. Please refer to the documentation at /docs or /redoc."}


@app.exception_handler(Exception)
async def internal_server_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": return_exception_message(exc),
            "detail": f"{repr(exc)}"
        }
    )
