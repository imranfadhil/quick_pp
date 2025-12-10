import logging
import os
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_mcp import FastApiMCP

from quick_pp.database.db_connector import DBConnector
from .exceptions import return_exception_message
from .router import api_router

# Configuration
LANGFLOW_HOST = os.getenv("LANGFLOW_HOST", "http://localhost:7860")

tags_metadata = [
    {"name": "File Handler", "description": "File handler related endpoints."},
    {"name": "Lithology", "description": "Lithology related endpoints."},
    {"name": "Porosity", "description": "Porosity related endpoints."},
    {"name": "Saturation", "description": "Saturation related endpoints."},
    {"name": "Permeability", "description": "Permeability related endpoints."},
    {
        "name": "Reservoir Summary",
        "description": "Reservoir summary related endpoints.",
    },
    {"name": "Database", "description": "Database related endpoints."},
    {"name": "Ancillary", "description": "Ancillary related endpoints."},
    {"name": "Langflow", "description": "Langflow related endpoints."},
    {"name": "Plotter", "description": "Plotter related endpoints."},
]


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Lifespan context manager replacing deprecated on_event handlers.

    Initializes per-process DB engine on startup and disposes it on shutdown.
    """
    try:
        db_url = os.environ.get("QPP_DATABASE_URL", "sqlite:///./data/quick_pp.db")
        DBConnector(db_url=db_url)
        logging.getLogger(__name__).info("DBConnector initialized on startup")
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to initialize DBConnector: %s", e)

    try:
        yield
    finally:
        try:
            DBConnector().dispose()
            logging.getLogger(__name__).info("DBConnector disposed on shutdown")
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to dispose DBConnector during shutdown"
            )


app = FastAPI(
    title="quick_pp API",
    description="API for quick_pp library.",
    contact={
        "name": "Imran Fadhil",
        "url": "https://github.com/imranfadhil/quick_pp",
        "email": "imranfadhil@gmail.com",
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_tags=tags_metadata,
    debug=True,
    lifespan=lifespan,
)

# Setup static files and templates
with resources.files("quick_pp.app.backend") as api_folder:
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
    allow_methods=["*"],
    allow_headers=["*"],
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
    base_url = str(request.base_url).rstrip("/")

    # Configuration for the template
    context = {
        "request": request,
        "api_base_url": base_url + "/quick_pp",
        "langflow_host": LANGFLOW_HOST,
    }

    return templates.TemplateResponse("chat.html", context)


@app.exception_handler(Exception)
async def internal_server_exception_handler(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": return_exception_message(exc), "detail": f"{repr(exc)}"},
    )


if FastApiMCP is not None:
    mcp = FastApiMCP(
        app,
        name="quick_pp API MCP",
        describe_all_responses=True,  # Include all possible response schemas
        describe_full_response_schema=True,  # Include full JSON schema in descriptions
    )
    mcp.mount()
