from fastapi import APIRouter

from .services.las_handler import router as las_router
from .services.lithology import router as lithology_router
from .services.porosity import router as porosity_router
from .services.saturation import router as saturation_router
from .services.permeability import router as permeability_router
from .services.reservoir_summary import router as ressum_router
from .services.chat import router as chat_router
from .services.database import router as database_router
from .services.plotter import router as plotter_router
from .services.ancillary_well import router as ancillary_well_router
from .services.ancillary_project import project_router as ancillary_project_router

api_router = APIRouter(prefix="/quick_pp")

api_router.include_router(las_router)
api_router.include_router(lithology_router)
api_router.include_router(porosity_router)
api_router.include_router(saturation_router)
api_router.include_router(permeability_router)
api_router.include_router(ressum_router)
api_router.include_router(chat_router)
api_router.include_router(database_router)
api_router.include_router(plotter_router)
api_router.include_router(ancillary_well_router)
api_router.include_router(ancillary_project_router)
