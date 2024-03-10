from fastapi import APIRouter

from .services.sand_silt_clay import router as ssc_router
from .services.shale import router as vsh_gr_router
from .services.porosity import router as porosity_router
from .services.hc_correction import router as hc_correction_router


api_router = APIRouter(prefix="/api")

api_router.include_router(ssc_router)
api_router.include_router(vsh_gr_router)
api_router.include_router(porosity_router)
api_router.include_router(hc_correction_router)
