from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .exceptions import return_exception_message
from fastapi_mcp import FastApiMCP
from chainlit.utils import mount_chainlit

from .router import api_router

tags_metadata = [
    {
        "name": "LAS File Handler",
        "description": "LAS file related endpoints."
    },
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
    debug=True
)
app.include_router(api_router)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins,
)


@app.get("/")
async def root():
    return {"message": "Welcome to quick_pp API. Please refer to the documentation at /docs or /redoc. "
            "You can also use the chat assistant at /qpp_assistant."}


@app.exception_handler(Exception)
async def internal_server_exception_handler(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": return_exception_message(exc),
            "detail": f"{repr(exc)}"
        }
    )

mcp = FastApiMCP(
    app,
    name="quick_pp API MCP",
    describe_all_responses=True,  # Include all possible response schemas
    describe_full_response_schema=True  # Include full JSON schema in descriptions
)
mcp.mount()

mount_chainlit(app=app, target=r"quick_pp\api\qpp_chainlit.py", path="/qpp_assistant")
