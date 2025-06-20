from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from .exceptions import return_exception_message
try:
    from fastapi_mcp import FastApiMCP
except ImportError:
    FastApiMCP = None
from importlib import resources
import os

from .router import api_router

# Configuration
LANGFLOW_HOST = os.getenv("LANGFLOW_HOST", "http://localhost:7860")

tags_metadata = [
    {
        "name": "File Handler",
        "description": "File handler related endpoints."
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
    },
    {
        "name": "Langflow",
        "description": "Langflow related endpoints."
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

# Setup static files and templates
with resources.files('quick_pp.api') as api_folder:
    static_folder = api_folder / "static"
    template_folder = api_folder / "template"

app.mount("/static", StaticFiles(directory=static_folder), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=str(template_folder))

app.include_router(api_router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(static_folder / "favicon.ico"))

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins,
)


@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/qpp_assistant", include_in_schema=False)
async def qpp_assistant(request: Request):
    """
    Serve the Langflow chat interface with dynamic project/flow selection.
    """
    # Get the base URL for API calls
    base_url = str(request.base_url).rstrip('/')

    # Configuration for the template
    context = {
        "request": request,
        "api_base_url": base_url + "/quick_pp",
        "langflow_host": LANGFLOW_HOST
    }

    return templates.TemplateResponse("chat.html", context)


@app.exception_handler(Exception)
async def internal_server_exception_handler(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": return_exception_message(exc),
            "detail": f"{repr(exc)}"
        }
    )


if FastApiMCP is not None:
    mcp = FastApiMCP(
        app,
        name="quick_pp API MCP",
        describe_all_responses=True,  # Include all possible response schemas
        describe_full_response_schema=True  # Include full JSON schema in descriptions
    )
    mcp.mount()
