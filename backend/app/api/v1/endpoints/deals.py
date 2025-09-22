import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
import os
from pathlib import Path
import logging

from app.core.database import get_async_session, AsyncSessionLocal
from app.models.deal import Deal
from app.models.document import Document
from app.schemas.deal import DealCreate, DealUpdate, Deal as DealSchema
from app.core.config import settings
from app.services.auth import get_current_user
from app.services.ingest import get_document_ingest_service
from app.services.search import get_search_service
from app.services.llm import llm_service
from app.schemas.search import SearchQuery, SearchResponse
from app.models.memo import Memo
from app.schemas.memo import Memo as MemoSchema
from pydantic import BaseModel
from app.services.deal_manager import create_default_folder_tree
from app.models.user import User

router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/")
@router.post("")  # Allow without trailing slash
async def create_deal(
    deal_data: DealCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new deal."""
    async with AsyncSessionLocal() as session:
        try:
            # Ensure the authenticated user exists (first-login provisioning)
            uid = current_user["uid"]
            existing_user = (await session.execute(select(User).where(User.id == uid))).scalar_one_or_none()
            if existing_user is None:
                provisioned_user = User(
                    id=uid,
                    email=current_user.get("email") or f"{uid}@users.noreply",
                    name=current_user.get("name") or (current_user.get("email") or uid)
                )
                session.add(provisioned_user)
                # Commit user first to satisfy FK constraints reliably
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()

            deal_id = str(uuid.uuid4())
            deal = Deal(
                id=deal_id,
                name=deal_data.name,
                description=deal_data.description,
                user_id=current_user["uid"]
            )
            
            session.add(deal)
            await session.commit()
            await session.refresh(deal)
            
            # Create default folder structure on disk
            create_default_folder_tree(deal_id)
            
            # Return deal data as dict to avoid serialization issues
            return {
                "id": deal.id,
                "name": deal.name,
                "description": deal.description,
                "user_id": deal.user_id,
                "document_count": 0,
                "created_at": deal.created_at.isoformat(),
                "updated_at": deal.updated_at.isoformat()
            }
            
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create deal: {str(e)}")


@router.get("/{deal_id}", response_model=DealSchema)
async def get_deal(
    deal_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a specific deal by ID."""
    # Get deal with document count
    result = await session.execute(
        select(Deal, func.count(Document.id).label('document_count'))
        .outerjoin(Document, Deal.id == Document.deal_id)
        .where(Deal.id == deal_id, Deal.user_id == current_user["uid"])
        .group_by(Deal.id)
    )
    row = result.first()
    
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    deal = row[0]
    document_count = row[1] or 0
    
    # Convert to schema with document count
    deal_dict = {
        "id": deal.id,
        "name": deal.name,
        "description": deal.description,
        "user_id": deal.user_id,
        "document_count": document_count,
        "created_at": deal.created_at,
        "updated_at": deal.updated_at
    }
    
    return DealSchema(**deal_dict)


@router.put("/{deal_id}", response_model=DealSchema)
async def update_deal(
    deal_id: str,
    deal_update: DealUpdate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Update an existing deal."""
    result = await session.execute(
        select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user["uid"])
    )
    deal = result.scalar_one_or_none()
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    try:
        # Update fields if provided
        if deal_update.name is not None:
            deal.name = deal_update.name
        if deal_update.description is not None:
            deal.description = deal_update.description
        
        await session.commit()
        await session.refresh(deal)
        
        # Get updated deal with document count
        result = await session.execute(
            select(Deal, func.count(Document.id).label('document_count'))
            .outerjoin(Document, Deal.id == Document.deal_id)
            .where(Deal.id == deal_id)
            .group_by(Deal.id)
        )
        row = result.first()
        document_count = row[1] if row else 0
        
        deal_dict = {
            "id": deal.id,
            "name": deal.name,
            "description": deal.description,
            "user_id": deal.user_id,
            "document_count": document_count,
            "created_at": deal.created_at,
            "updated_at": deal.updated_at
        }
        
        return DealSchema(**deal_dict)
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update deal: {str(e)}"
        )


@router.delete("/{deal_id}")
async def delete_deal(
    deal_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a deal and all associated documents."""
    result = await session.execute(
        select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user["uid"])
    )
    deal = result.scalar_one_or_none()
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    try:
        # Get all documents associated with this deal
        documents_result = await session.execute(
            select(Document).where(Document.deal_id == deal_id)
        )
        documents = documents_result.scalars().all()
        
        # Clean up vector embeddings and files for each document
        from app.services.ingest import document_ingest_service
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from pathlib import Path
        import shutil
        
        # Initialize ChromaDB client for cleanup
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        try:
            collection = chroma_client.get_collection(name="documents")
            
            # Delete vector embeddings for all documents in this deal
            logger.info(f"Cleaning up vector embeddings for deal {deal_id}")
            collection.delete(where={"deal_id": deal_id})
            logger.info(f"Vector embeddings cleaned up for deal {deal_id}")
            
        except Exception as e:
            logger.warning(f"Failed to clean up vector embeddings for deal {deal_id}: {e}")
        
        # Delete physical files
        for document in documents:
            if document.file_path and Path(document.file_path).exists():
                try:
                    Path(document.file_path).unlink()
                    logger.info(f"Deleted file: {document.file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {document.file_path}: {e}")
        
        # Delete deal folder structure
        deal_folder = Path(settings.UPLOAD_DIR) / deal_id
        if deal_folder.exists():
            try:
                shutil.rmtree(deal_folder)
                logger.info(f"Deleted deal folder: {deal_folder}")
            except Exception as e:
                logger.warning(f"Failed to delete deal folder {deal_folder}: {e}")
        
        # Delete the deal (cascading deletes will handle documents, chunks, memos)
        await session.delete(deal)
        await session.commit()
        
        logger.info(f"Deal {deal_id} deleted successfully")
        return {"message": "Deal deleted successfully"}
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to delete deal {deal_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete deal: {str(e)}"
        )


@router.get("/", response_model=List[DealSchema])
@router.get("", response_model=List[DealSchema])  # Allow path without trailing slash
async def list_deals(
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """List deals for the authenticated user."""
    query = (
        select(Deal, func.count(Document.id).label('document_count'))
        .outerjoin(Document, Deal.id == Document.deal_id)
        .where(Deal.user_id == current_user["uid"])
        .group_by(Deal.id)
        .order_by(Deal.updated_at.desc())
    )
    
    result = await session.execute(query)
    rows = result.all()
    
    deals = []
    for row in rows:
        deal = row[0]
        document_count = row[1] or 0
        
        deal_dict = {
            "id": deal.id,
            "name": deal.name,
            "description": deal.description,
            "user_id": deal.user_id,
            "document_count": document_count,
            "created_at": deal.created_at,
            "updated_at": deal.updated_at
        }
        deals.append(DealSchema(**deal_dict))
    
    return deals


@router.get("/{deal_id}/documents")
async def list_deal_documents(
    deal_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """List all documents for a specific deal."""
    # First check if deal exists
    deal_result = await session.execute(
        select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user["uid"])
    )
    deal = deal_result.scalar_one_or_none()
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    # Get documents for this user's deal
    result = await session.execute(
        select(Document)
        .where(Document.deal_id == deal_id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    
    return documents


@router.post("/{deal_id}/documents/upload")
async def upload_documents_to_deal(
    deal_id: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Upload and process multiple documents for a specific deal."""
    
    logger.info(f"=== UPLOAD REQUEST START ===")
    logger.info(f"Deal ID: {deal_id}")
    logger.info(f"Number of files: {len(files) if files else 'None'}")
    
    try:
        # Helper to determine file size without reading the whole file twice
        def _get_file_size(upload_file: UploadFile) -> int:
            """Return the size (in bytes) of an UploadFile without consuming its contents."""
            # SpooledTemporaryFile supports tell after seek
            position = upload_file.file.tell()
            upload_file.file.seek(0, os.SEEK_END)
            size = upload_file.file.tell()
            upload_file.file.seek(position)
            return size

        for i, file in enumerate(files):
            file_size = _get_file_size(file)
            logger.info(f"File {i}: name={file.filename}, size={file_size}, content_type={file.content_type}")
        
        # First check if deal exists and user owns it
        logger.info(f"Checking if deal {deal_id} exists and user {current_user['uid']} owns it...")
        deal_result = await session.execute(
            select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user["uid"])
        )
        deal = deal_result.scalar_one_or_none()
        
        if not deal:
            logger.error(f"Deal {deal_id} not found or access denied for user {current_user['uid']}")
            raise HTTPException(status_code=404, detail="Deal not found")
        
        logger.info(f"Deal found: {deal.name} (user: {deal.user_id})")
        
        # Validate files
        logger.info("Starting file validation...")
        for i, file in enumerate(files):
            logger.info(f"Validating file {i}: {file.filename}")
            file_size = _get_file_size(file)
            logger.info(f"  Content type: {file.content_type}")
            logger.info(f"  File size: {file_size}")
            logger.info(f"  Allowed types: {settings.ALLOWED_FILE_TYPES}")
            
            if file.content_type not in settings.ALLOWED_FILE_TYPES:
                logger.error(f"File type {file.content_type} not allowed for file {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file.content_type} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
                )
            
            if file_size and file_size > settings.MAX_FILE_SIZE:
                logger.error(f"File {file.filename} too large: {file_size} bytes")
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
                )
            
            logger.info(f"  File {file.filename} validation passed")
        
        logger.info("All files validated successfully")
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        logger.info(f"Upload directory: {upload_dir}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        processed_documents = []
        
        for i, file in enumerate(files):
            logger.info(f"Processing file {i}: {file.filename}")
            try:
                # Recalculate size since content will be loaded next
                file_size = _get_file_size(file)

                # Generate unique filename
                file_extension = Path(file.filename).suffix if file.filename else ""
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = upload_dir / unique_filename
                
                logger.info(f"  Saving to: {file_path}")
                
                # Save file to disk
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                logger.info(f"  File saved, size: {len(content)} bytes")
                
                # Process and ingest the document
                logger.info(f"  Starting document ingestion...")
                ingest_service = get_document_ingest_service()
                document = await ingest_service.ingest_document(
                    file_path=str(file_path),
                    filename=file.filename,
                    deal_id=deal_id,
                    user_id=deal.user_id,  # Use the deal's user_id
                    file_size=len(content),
                    content_type=file.content_type,
                    session=session  # Pass the current session
                )
                
                logger.info(f"  Document ingested successfully: {document['id']}")
                processed_documents.append(document)
                
            except Exception as e:
                logger.error(f"  Error processing file {file.filename}: {str(e)}")
                logger.exception("Full traceback:")
                
                # Clean up file if processing failed
                if 'file_path' in locals() and file_path.exists():
                    file_path.unlink()
                    logger.info(f"  Cleaned up file: {file_path}")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process file {file.filename}: {str(e)}"
                )
        
        logger.info(f"=== UPLOAD REQUEST SUCCESS: {len(processed_documents)} documents processed ===")
        
        # Since ingest service now returns dictionaries, we can return them directly
        return processed_documents
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )


class _MemoCreateInline(BaseModel):
    """Schema for creating a memo from the deal-scoped endpoint."""
    title: str
    content: str

@router.post("/{deal_id}/search", response_model=SearchResponse)
async def deal_scoped_search(
    deal_id: str,
    search_query: SearchQuery,
):
    """Proxy endpoint so the frontend can call /deals/{deal_id}/search
    This simply forwards the request to the vector search / LLM pipeline
    while injecting the deal_id filter.
    """
    try:
        # Vector similarity search filtered by deal
        search_service = get_search_service()
        search_results = await search_service.search_documents(
            query=search_query.query,
            deal_id=deal_id,
            top_k=None,
        )

        # Generate RAG answer
        answer = await llm_service.generate_answer(search_query.query, search_results)

        return SearchResponse(
            results=search_results,
            query=search_query.query,
            total_results=len(search_results),
            answer=answer,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/{deal_id}/memos", response_model=List[MemoSchema])
async def list_deal_memos(
    deal_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Return all memos attached to a deal (ordered by updated_at desc)."""
    # Ensure deal belongs to current user
    deal_check = await session.execute(select(Deal.id).where(Deal.id == deal_id, Deal.user_id == current_user["uid"]))
    if deal_check.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Deal not found")

    result = await session.execute(
        select(Memo).where(Memo.deal_id == deal_id).order_by(Memo.updated_at.desc())
    )
    memos = result.scalars().all()
    return memos


@router.post("/{deal_id}/memos", response_model=MemoSchema)
async def create_deal_memo(
    deal_id: str,
    memo_data: _MemoCreateInline,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new memo under a given deal. The user_id is set to a default
    placeholder until user management is implemented."""
    try:
        # Ensure deal belongs to current user
        deal_check = await session.execute(select(Deal.id).where(Deal.id == deal_id, Deal.user_id == current_user["uid"]))
        if deal_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Deal not found")

        # Ensure the authenticated user exists (first-login provisioning)
        uid = current_user["uid"]
        existing_user = (await session.execute(select(User).where(User.id == uid))).scalar_one_or_none()
        if existing_user is None:
            provisioned_user = User(
                id=uid,
                email=current_user.get("email") or f"{uid}@users.noreply",
                name=current_user.get("name") or (current_user.get("email") or uid)
            )
            session.add(provisioned_user)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()

        memo_id = str(uuid.uuid4())
        memo = Memo(
            id=memo_id,
            title=memo_data.title,
            content=memo_data.content,
            deal_id=deal_id,
            user_id=current_user["uid"],
        )
        session.add(memo)
        await session.commit()
        await session.refresh(memo)
        return memo
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create memo: {str(e)}") 