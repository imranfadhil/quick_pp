from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .exceptions import return_exception_message

from .router import api_router

app = FastAPI(
    title="quick_pp API",
    license_info={"name": "imranfadhil", "email": "imranfadhil@gmail.com",
                  "url": "https://github.com/imranfadhil/quick_pp"},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    debug=False
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.exception_handler(Exception)
async def internal_server_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": return_exception_message(exc),
            "detail": f"{repr(exc)}"
        }
    )
