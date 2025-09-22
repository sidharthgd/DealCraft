#!/usr/bin/env python3
"""
Startup script to ensure proper initialization before starting the FastAPI app.
This helps prevent multiprocessing issues with ChromaDB and other services.
"""

import os
import sys
import logging

# Add the app directory to Python path
sys.path.insert(0, '/app')

# Set ChromaDB environment variables to disable telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['CHROMA_CLIENT_AUTH_PROVIDER'] = ''
os.environ['CHROMA_SERVER_AUTH_PROVIDER'] = ''

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def initialize_services():
    """Initialize services before starting the main app"""
    try:
        logger.info("Starting service initialization...")
        
        # Import and initialize core services early
        from app.core.config import settings
        logger.info(f"Settings loaded: DEBUG={settings.debug}")
        
        # Test database connection
        logger.info("Testing database connection...")
        from app.core.database import engine
        logger.info("Database engine initialized")
        
        # Pre-initialize ChromaDB to avoid multiprocessing issues
        logger.info("Pre-initializing ChromaDB clients...")
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Create a test client to ensure ChromaDB is working
            test_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            logger.info("ChromaDB client pre-initialized successfully")
        except Exception as e:
            logger.warning(f"ChromaDB pre-initialization failed: {e}")
            logger.info("Application will continue but vector search may not work")
        
        # Initialize vertex AI service if available
        logger.info("Initializing Vertex AI service...")
        try:
            from app.services.vertex_ai import vertex_ai_service
            vertex_ai_ready = vertex_ai_service.validate_configuration()
            logger.info(f"Vertex AI service ready: {vertex_ai_ready}")
        except Exception as e:
            logger.warning(f"Vertex AI initialization failed: {e}")
            logger.info("Application will continue with fallback search methods")
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        logger.error("Application will continue but some features may not work properly")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting DealCraft backend initialization...")
    initialize_services()
    
    # Import and run the main app
    import uvicorn
    from main import app
    
    logger.info("🌟 Starting uvicorn server...")
    # Check if we're in development based on environment or mounted volumes
    is_dev = os.path.exists("/app/main.py") and os.path.getmtime("/app/main.py") > 0
    
    if is_dev:
        # Development with selective file watching
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["/app/app"],  # Only watch app code, not uploads/logs
            reload_excludes=["*.tmp", "*.log", "*.pdf", "*.docx", "*.xlsx", "*.txt", "__pycache__/*"],
            log_level="info",
            timeout_keep_alive=1800,  # 30 minutes for large file uploads
            timeout_graceful_shutdown=300  # 5 minutes graceful shutdown
        )
    else:
        # Production-like mode without file watching
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            timeout_keep_alive=1800,  # 30 minutes for large file uploads
            timeout_graceful_shutdown=300  # 5 minutes graceful shutdown
        )