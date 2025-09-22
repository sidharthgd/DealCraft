import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db
from app.api.v1.api import api_router
import logging
import traceback
import sys
from sqlalchemy import select

# Enhanced logging configuration for Docker containers
def setup_logging():
    """Configure logging to be visible in Docker containers"""
    # Create a custom formatter that includes timestamp and level
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create console handler (for Docker logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Also create a file handler for persistent logs
    try:
        # Use local path instead of Docker path
        import os
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'dealcraft.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If we can't create file handler, just log to console
        print(f"Could not create file handler: {e}")
    
    # Print redirection removed to avoid recursion issues
    # sys.stdout redirection was causing infinite recursion
    
    return logging.getLogger(__name__)

# Set up logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Continuing startup without database initialization")
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="DealCraft AI API",
    description="AI-powered deal document analysis platform",
    version="1.0.0",
    lifespan=lifespan,
    # Increase request size limits for large file uploads
    max_request_size=100 * 1024 * 1024,  # 100MB
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Custom exception handler for HTTP exceptions (including auth errors)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"=== HTTP EXCEPTION ===")
    logger.error(f"Status: {exc.status_code}")
    logger.error(f"Detail: {exc.detail}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    
    # Rely on CORSMiddleware to attach the correct CORS headers
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Custom exception handler for 422 validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"=== 422 VALIDATION ERROR ===")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Validation errors: {exc.errors()}")
    logger.error(f"Request body type: {request.headers.get('content-type', 'unknown')}")
    
    # Try to log form data if it's a multipart request
    try:
        if request.headers.get('content-type', '').startswith('multipart/form-data'):
            logger.error("This is a multipart form request (file upload)")
    except Exception as e:
        logger.error(f"Could not inspect request body: {str(e)}")
    
    logger.error("=== END 422 ERROR ===")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "code": "422"
        }
    )

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed with exception: {str(e)}")
        logger.exception("Full traceback:")
        raise

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "DealCraft AI API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=1800,  # 30 minutes for large file uploads
        timeout_graceful_shutdown=300  # 5 minutes graceful shutdown
    ) 