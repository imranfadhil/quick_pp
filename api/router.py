from fastapi import APIRouter

from .services.sand_silt_clay import router as ssc_router


api_router = APIRouter(prefix="/api")

api_router.include_router(ssc_router)
