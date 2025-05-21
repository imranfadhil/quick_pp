from fastapi import APIRouter

from .services.las_handler import router as las_router
from .services.lithology import router as lithology_router
from .services.porosity import router as porosity_router
from .services.saturation import router as saturation_router
from .services.permeability import router as permeability_router
from .services.reservoir_summary import router as ressum_router


api_router = APIRouter(prefix="/quick_pp")

api_router.include_router(las_router)
api_router.include_router(lithology_router)
api_router.include_router(porosity_router)
api_router.include_router(saturation_router)
api_router.include_router(permeability_router)
api_router.include_router(ressum_router)
