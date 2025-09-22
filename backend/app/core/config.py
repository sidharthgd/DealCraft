import os
from typing import List
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Look for .env file in the backend directory relative to this file
    _backend_dir = Path(__file__).parent.parent.parent
    # Allow extra env vars so unexpected keys don't crash local runs
    model_config = ConfigDict(env_file=_backend_dir / ".env", extra='ignore')
    
    # Application settings
    debug: bool = False
    log_level: str = "info"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Google Cloud Platform
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
    GCP_CREDENTIALS_PATH: str = os.getenv("GCP_CREDENTIALS_PATH", "")
    # Some Google libraries read GOOGLE_CLOUD_PROJECT
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    
    # Docker-specific (used in docker-compose.yml)
    CREDENTIALS_HOST_PATH: str = os.getenv("CREDENTIALS_HOST_PATH", "")
    
    # Vertex AI
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    VERTEX_AI_MODEL: str = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash")
    VERTEX_AI_EMBEDDING_MODEL: str = os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-004")
    
    # Vector Database - Vertex AI Vector Search
    VERTEX_VECTOR_INDEX_ENDPOINT: str = os.getenv("VERTEX_VECTOR_INDEX_ENDPOINT", "")
    VERTEX_VECTOR_INDEX_ID: str = os.getenv("VERTEX_VECTOR_INDEX_ID", "")
    
    # Vector Database (keeping for gradual migration)
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # OpenAI (keeping for gradual migration)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "http://frontend:3000",  # Docker internal networking
        "https://dealcraft.info",  # Production domain
        "https://www.dealcraft.info",  # Production domain with www
        os.getenv("FRONTEND_URL", "")  # Dynamic frontend URL for Cloud Run
    ]
    
    # Google Cloud Storage
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
    GCS_CREDENTIALS_PATH: str = os.getenv("GCS_CREDENTIALS_PATH", "")
    
    # File Upload (Legacy - for local development)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Authentication
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_WEB_API_KEY: str = os.getenv("FIREBASE_WEB_API_KEY", "")
    # In production this must be False to prevent anonymous access
    ALLOW_AUTH_FALLBACK: bool = os.getenv("ALLOW_AUTH_FALLBACK", "false").lower() == "true"
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf", 
        "application/msword", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
        "text/plain",
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]
    
    # Document Processing
    CHUNK_SIZE: int = 2000  # Increased from 1000 for better context
    CHUNK_OVERLAP: int = 200  # Increased from 100 for better context preservation
    
    # Search
    SEARCH_TOP_K: int = 6
    
    # Model (keeping for backward compatibility)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Google Custom Search API settings
    GOOGLE_SEARCH_API_KEY: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_SEARCH_ENGINE_ID: str = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
    
    # Vector search settings
    VECTOR_SEARCH_TOP_K: int = 10


settings = Settings() 