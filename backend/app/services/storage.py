"""
Google Cloud Storage service for file uploads and management.
"""
import os
import uuid
import logging
from typing import Optional, BinaryIO, Union
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound
from app.core.config import settings

logger = logging.getLogger(__name__)

class GCSStorageService:
    """Service for managing file uploads to Google Cloud Storage."""
    
    def __init__(self):
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.client = None
        self.bucket = None
        
        if self.bucket_name:
            try:
                # Initialize Google Cloud Storage client
                if settings.GCS_CREDENTIALS_PATH and os.path.exists(settings.GCS_CREDENTIALS_PATH):
                    self.client = storage.Client.from_service_account_json(settings.GCS_CREDENTIALS_PATH)
                else:
                    # Use default credentials (for Cloud Run, this will work automatically)
                    self.client = storage.Client(project=settings.GCP_PROJECT_ID)
                
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"GCS Storage service initialized for bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                self.client = None
                self.bucket = None
        else:
            logger.warning("GCS_BUCKET_NAME not configured, file uploads will use local storage")
    
    def is_available(self) -> bool:
        """Check if GCS is properly configured and available."""
        return self.client is not None and self.bucket is not None
    
    async def upload_file(
        self, 
        file_content: Union[bytes, BinaryIO], 
        filename: str,
        content_type: str = "application/octet-stream",
        folder: str = "uploads"
    ) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            file_content: File content as bytes or file-like object
            filename: Original filename
            content_type: MIME type of the file
            folder: Folder path within the bucket
            
        Returns:
            The GCS file path/key
        """
        if not self.is_available():
            raise RuntimeError("Google Cloud Storage is not properly configured")
        
        # Generate unique filename to avoid conflicts
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        gcs_path = f"{folder}/{unique_filename}"
        
        try:
            # Create blob and upload
            blob = self.bucket.blob(gcs_path)
            
            # Handle both bytes and file-like objects
            if isinstance(file_content, bytes):
                blob.upload_from_string(file_content, content_type=content_type)
            else:
                blob.upload_from_file(file_content, content_type=content_type)
            
            logger.info(f"Successfully uploaded file to GCS: {gcs_path}")
            return gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
            raise RuntimeError(f"File upload failed: {e}")
    
    async def download_file(self, gcs_path: str) -> bytes:
        """
        Download a file from Google Cloud Storage.
        
        Args:
            gcs_path: The GCS file path/key
            
        Returns:
            File content as bytes
        """
        if not self.is_available():
            raise RuntimeError("Google Cloud Storage is not properly configured")
        
        try:
            blob = self.bucket.blob(gcs_path)
            content = blob.download_as_bytes()
            logger.info(f"Successfully downloaded file from GCS: {gcs_path}")
            return content
            
        except NotFound:
            logger.error(f"File not found in GCS: {gcs_path}")
            raise FileNotFoundError(f"File not found: {gcs_path}")
        except Exception as e:
            logger.error(f"Failed to download file from GCS: {e}")
            raise RuntimeError(f"File download failed: {e}")
    
    async def delete_file(self, gcs_path: str) -> bool:
        """
        Delete a file from Google Cloud Storage.
        
        Args:
            gcs_path: The GCS file path/key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"Successfully deleted file from GCS: {gcs_path}")
            return True
            
        except NotFound:
            logger.warning(f"File not found for deletion in GCS: {gcs_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file from GCS: {e}")
            return False
    
    async def get_signed_url(self, gcs_path: str, expiration_minutes: int = 60) -> str:
        """
        Generate a signed URL for temporary file access.
        
        Args:
            gcs_path: The GCS file path/key
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Signed URL string
        """
        if not self.is_available():
            raise RuntimeError("Google Cloud Storage is not properly configured")
        
        try:
            blob = self.bucket.blob(gcs_path)
            from datetime import timedelta
            
            signed_url = blob.generate_signed_url(
                expiration=timedelta(minutes=expiration_minutes),
                method="GET"
            )
            
            logger.info(f"Generated signed URL for GCS file: {gcs_path}")
            return signed_url
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for GCS file: {e}")
            raise RuntimeError(f"Signed URL generation failed: {e}")
    
    async def list_files(self, folder: str = "uploads", limit: int = 100) -> list:
        """
        List files in a specific folder.
        
        Args:
            folder: Folder path within the bucket
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        if not self.is_available():
            return []
        
        try:
            blobs = self.client.list_blobs(
                self.bucket_name, 
                prefix=f"{folder}/",
                max_results=limit
            )
            
            files = []
            for blob in blobs:
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "created": blob.time_created,
                    "content_type": blob.content_type
                })
            
            logger.info(f"Listed {len(files)} files from GCS folder: {folder}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files from GCS: {e}")
            return []


class LocalStorageService:
    """Fallback local storage service for development."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage service initialized: {self.upload_dir}")
    
    def is_available(self) -> bool:
        """Local storage is always available."""
        return True
    
    async def upload_file(
        self, 
        file_content: Union[bytes, BinaryIO], 
        filename: str,
        content_type: str = "application/octet-stream",
        folder: str = "uploads"
    ) -> str:
        """Save file to local filesystem."""
        # Generate unique filename
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        local_path = self.upload_dir / unique_filename
        
        try:
            with open(local_path, "wb") as f:
                if isinstance(file_content, bytes):
                    f.write(file_content)
                else:
                    f.write(file_content.read())
            
            logger.info(f"Successfully saved file locally: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to save file locally: {e}")
            raise RuntimeError(f"Local file save failed: {e}")
    
    async def download_file(self, file_path: str) -> bytes:
        """Read file from local filesystem."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            logger.info(f"Successfully read local file: {file_path}")
            return content
            
        except FileNotFoundError:
            logger.error(f"Local file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read local file: {e}")
            raise RuntimeError(f"File read failed: {e}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            Path(file_path).unlink()
            logger.info(f"Successfully deleted local file: {file_path}")
            return True
        except FileNotFoundError:
            logger.warning(f"Local file not found for deletion: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete local file: {e}")
            return False


# Global storage service instances
_gcs_service = None
_local_service = None

def get_storage_service():
    """Get the appropriate storage service (GCS or local fallback)."""
    global _gcs_service, _local_service
    
    # Try GCS first
    if _gcs_service is None:
        _gcs_service = GCSStorageService()
    
    if _gcs_service.is_available():
        return _gcs_service
    
    # Fallback to local storage
    if _local_service is None:
        _local_service = LocalStorageService()
    
    logger.warning("Using local storage fallback - not suitable for production")
    return _local_service