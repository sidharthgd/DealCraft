import os
import uuid
import tempfile
import logging
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)
from app.services.ingest import get_document_ingest_service
from app.services.storage import get_storage_service
from app.services.auth import get_current_user
from app.schemas.document import Document, DocumentCreate
from app.core.database import get_async_session, AsyncSessionLocal
from app.models.deal import Deal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

router = APIRouter()


async def get_or_create_deal(deal_name: str, user_id: str, session: AsyncSession) -> str:
    """Get existing deal or create a new one."""
    # Try to find existing deal
    result = await session.execute(
        select(Deal).where(Deal.name == deal_name, Deal.user_id == user_id)
    )
    existing_deal = result.scalar_one_or_none()
    
    if existing_deal:
        return existing_deal.id
    
    # Create new deal
    deal_id = str(uuid.uuid4())
    new_deal = Deal(
        id=deal_id,
        name=deal_name,
        user_id=user_id,
        description=f"Deal for {deal_name}"
    )
    session.add(new_deal)
    await session.commit()
    return deal_id


@router.post("/", response_model=List[Document])
@router.post("", response_model=List[Document])  # Allow without trailing slash
async def upload_files(
    files: List[UploadFile] = File(...),
    deal_name: Optional[str] = Form(None),
    deal_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process multiple files for a deal."""
    
    # Instantiate services
    storage_service = get_storage_service()
    document_ingest_service = get_document_ingest_service()

    # Validate input: at least one of deal_id or deal_name must be provided
    if not deal_id and not deal_name:
        raise HTTPException(status_code=422, detail="Either 'deal_id' or 'deal_name' must be provided")

    # Validate files
    for file in files:
        if file.content_type not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
    
    # Get user ID from authenticated user
    user_id = current_user.get('uid', 'default-user')
    
    processed_documents = []
    
    # Use a single session for the entire upload operation to avoid race conditions
    async with AsyncSessionLocal() as session:
        # Resolve deal_id
        if deal_id:
            # Optional safety: ensure the deal belongs to the user
            result = await session.execute(
                select(Deal).where(Deal.id == deal_id, Deal.user_id == user_id)
            )
            existing = result.scalar_one_or_none()
            if not existing:
                raise HTTPException(status_code=404, detail="Deal not found for current user")
        else:
            # Create or fetch by name
            deal_id = await get_or_create_deal(deal_name, user_id, session)
        
        for file in files:
            file_path = None
            temp_file = None
            
            try:
                # Read file content
                content = await file.read()
                
                # Upload to storage service (GCS or local)
                storage_path = await storage_service.upload_file(
                    file_content=content,
                    filename=file.filename,
                    content_type=file.content_type,
                    folder=f"deals/{deal_id}"
                )
                
                # For processing, we need a local file path
                # Create a temporary file for the ingest service
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                    temp_file.write(content)
                    file_path = temp_file.name
                
                # Process and ingest the document using the same session
                document = await document_ingest_service.ingest_document(
                    file_path=file_path,
                    filename=file.filename,
                    deal_id=deal_id,
                    user_id=user_id,
                    file_size=len(content),
                    content_type=file.content_type,
                    storage_path=storage_path,  # Store the cloud storage path
                    session=session  # Pass the session to avoid race condition
                )
                
                processed_documents.append(document)
                
            except Exception as e:            
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process file {file.filename}: {str(e)}"
                )
            finally:
                # Clean up temporary file
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temp file {file_path}: {cleanup_error}")
    
    return processed_documents


@router.post("/directory")
async def upload_directory(
    directory_path: str = Form(...),
    deal_name: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Process all files in a directory for a deal (mainly for CLI usage)."""
    
    if not os.path.exists(directory_path):
        raise HTTPException(
            status_code=400,
            detail=f"Directory {directory_path} does not exist"
        )
    
    # Get user ID from authenticated user
    user_id = current_user.get('uid', 'default-user')
    
    try:
        documents = await document_ingest_service.ingest_directory(
            directory_path=directory_path,
            deal_name=deal_name,
            user_id=user_id
        )
        
        return {
            "message": f"Successfully processed {len(documents)} documents",
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process directory: {str(e)}"
        )


@router.get("/status/{document_id}")
async def get_upload_status(document_id: str):
    """Get the processing status of an uploaded document."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document.id,
            "filename": document.name,
            "status": document.status,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        } 