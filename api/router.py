from fastapi import APIRouter

from .services.lithology import router as lithology_router
from .services.porosity import router as porosity_router
from .services.saturation import router as saturation_router


api_router = APIRouter(prefix="/api")

api_router.include_router(lithology_router)
api_router.include_router(porosity_router)
api_router.include_router(saturation_router)
