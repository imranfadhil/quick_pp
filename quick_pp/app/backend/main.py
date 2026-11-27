import asyncio
import logging
import os
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from fastapi_mcp import FastApiMCP
except Exception:
    FastApiMCP = None

from quick_pp import __version__

from .exceptions import return_exception_message
from .router import api_router
from .services.ancillary import import_ancillary_router
from .services.las_handler import import_logs_router
from .services.report import router as report_v1_router

# Robust per-process DBConnector singleton
_db_connector_instance = None


def get_db_connector():
    global _db_connector_instance
    if _db_connector_instance is None:
        from quick_pp.app.backend.utils.db import _connector

        _db_connector_instance = _connector
        _db_connector_instance.setup_db()
        logging.getLogger(__name__).info("DBConnector initialized (singleton)")
    return _db_connector_instance


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
    {"name": "Ancillary Data", "description": "Ancillary related endpoints."},
    {
        "name": "Agentic Assistant & MCP",
        "description": "Petrophysical MCP Server and Agentic Assistant endpoints.",
    },
    {"name": "Plotter", "description": "Plotter related endpoints."},
]


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Lifespan context manager replacing deprecated on_event handlers.

    Initializes per-process DB engine on startup and disposes it on shutdown.
    """
    try:
        get_db_connector()
        from quick_pp.app.backend.utils.auth import bootstrap_default_admin

        bootstrap_default_admin()
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to initialize DBConnector: %s", e)

    cleanup_task: asyncio.Task | None = None
    try:
        from quick_pp.app.backend.utils.auth import is_demo_mode

        if is_demo_mode():
            from quick_pp.app.backend.services.demo_guard import (
                demo_periodic_cleanup_loop,
                run_periodic_demo_cleanup,
            )

            run_periodic_demo_cleanup()
            cleanup_task = asyncio.create_task(demo_periodic_cleanup_loop())
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        global _db_connector_instance
        try:
            if _db_connector_instance is not None:
                _db_connector_instance.dispose()
                logging.getLogger(__name__).info("DBConnector disposed on shutdown")
                _db_connector_instance = None
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to dispose DBConnector during shutdown"
            )


_initial_demo_mode = os.environ.get("QPP_DEMO_MODE", "false").lower() in (
    "1",
    "true",
    "yes",
)

app = FastAPI(
    title="quick_pp API",
    version=__version__,
    description="API for quick_pp library.",
    contact={
        "name": "Imran Fadhil",
        "url": "https://github.com/imranfadhil/quick_pp",
        "email": "imranfadhil@gmail.com",
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    openapi_tags=tags_metadata,
    debug=not _initial_demo_mode,
    lifespan=lifespan,
)

# Setup static files and templates
api_folder = resources.files("quick_pp.app.backend")
static_folder = api_folder / "static"
template_folder = api_folder / "template"

app.mount("/static", StaticFiles(directory=static_folder), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=str(template_folder))

# Add CORS middleware BEFORE including routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(report_v1_router, prefix="/api/v1", include_in_schema=False)
app.include_router(import_logs_router, prefix="/api/v1", include_in_schema=False)
app.include_router(import_ancillary_router, prefix="/api/v1", include_in_schema=False)


@app.middleware("http")
async def demo_session_and_compute_guard_middleware(request: Request, call_next):
    from quick_pp.app.backend.utils.auth import is_demo_mode

    if not is_demo_mode():
        return await call_next(request)

    from quick_pp.app.backend.services.demo_guard import (
        DEMO_SESSION_HEADER,
        acquire_compute_slot_sync,
        is_heavy_demo_compute_request,
        normalize_session_id,
        register_or_touch_session,
        release_compute_slot_sync,
        reset_current_demo_session_id,
        set_current_demo_session_id,
    )

    path = request.url.path
    sid = normalize_session_id(request.headers.get(DEMO_SESSION_HEADER))
    ctx_token = set_current_demo_session_id(sid)
    acquired_compute = False

    try:
        # Enforce soft capacity gate when a browser tab presents X-Demo-Session-Id
        if sid and not path.startswith(
            ("/quick_pp/demo/", "/quick_pp/auth/", "/static/")
        ):
            client_ip = (
                request.headers.get("cf-connecting-ip")
                or request.headers.get("x-real-ip")
                or (request.client.host if request.client else "")
            )
            snap = register_or_touch_session(sid, client_ip=client_ip)
            if not snap.get("admitted", True):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Live demo is currently at capacity "
                            f"({snap['online_users']}/{snap['max_users']} active users). "
                            "Waiting for an open slot..."
                        ),
                        "code": "DEMO_AT_CAPACITY",
                        "online_users": snap["online_users"],
                        "max_users": snap["max_users"],
                    },
                )

        # Enforce hard concurrency semaphore on CPU/RAM-heavy compute & report routes
        if is_heavy_demo_compute_request(request.method, path):
            acquired_compute = await asyncio.to_thread(acquire_compute_slot_sync)
            if not acquired_compute:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            "Demo compute server is busy processing other visitors' "
                            "calculations. Please retry in a few seconds."
                        ),
                        "code": "DEMO_COMPUTE_BUSY",
                    },
                )

        return await call_next(request)
    finally:
        if acquired_compute:
            release_compute_slot_sync()
        reset_current_demo_session_id(ctx_token)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(static_folder / "favicon.ico"))


@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/qpp_assistant", include_in_schema=False)
async def qpp_assistant(request: Request):
    """
    Serve the Petrophysical AI Assistant & MCP chat interface.
    """
    # Get the base URL for API calls
    base_url = str(request.base_url).rstrip("/")

    # Configuration for the template
    context = {
        "api_base_url": base_url + "/quick_pp",
    }

    return templates.TemplateResponse(
        request=request, name="chat.html", context=context
    )


@app.exception_handler(Exception)
async def internal_server_exception_handler(request: Request, exc: Exception):
    from quick_pp.app.backend.utils.auth import is_demo_mode

    msg = return_exception_message(exc)
    if is_demo_mode():
        return JSONResponse(
            status_code=500,
            content={"message": msg, "detail": msg},
        )
    return JSONResponse(
        status_code=500,
        content={"message": msg, "detail": f"{repr(exc)}"},
    )


if FastApiMCP is not None and not _initial_demo_mode:
    try:
        mcp = FastApiMCP(
            app,
            name="quick_pp API MCP",
            describe_all_responses=True,
            describe_full_response_schema=True,
        )
        mcp.mount()
    except Exception as _mcp_err:
        logging.getLogger(__name__).debug("FastApiMCP setup skipped: %s", _mcp_err)
