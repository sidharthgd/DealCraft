from fastapi import APIRouter

from .endpoints import upload, search, memo, deals

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(memo.router, prefix="/memo", tags=["memo"])
api_router.include_router(deals.router, prefix="/deals", tags=["deals"]) 