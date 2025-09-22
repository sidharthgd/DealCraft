import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.core.database import get_async_session
from app.models.document import Document
from app.models.memo import Memo
from app.schemas.memo import MemoCreate, MemoUpdate, Memo as MemoSchema
from app.services.memo_generator import auto_memo_generator
from app.services.auth import get_current_user

router = APIRouter()


class MemoGenerateRequest(BaseModel):
    # Allow either explicit document IDs or a deal_id to infer documents
    document_ids: Optional[List[str]] = None
    deal_id: Optional[str] = None
    discussion_date: Optional[str] = None
    ftf_equity_size: Optional[str] = None
    expected_closing: Optional[str] = None
    # Optional data URL (base64) for org chart image, e.g., "data:image/png;base64,...."
    org_chart_image_base64: Optional[str] = None


@router.post("/", response_model=MemoSchema)
async def create_memo(
    memo_data: MemoCreate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new memo."""
    try:
        memo_id = str(uuid.uuid4())
        # Ensure the memo is created for the current authenticated user
        memo = Memo(
            id=memo_id,
            title=memo_data.title,
            content=memo_data.content,
            deal_id=memo_data.deal_id,
            user_id=current_user["uid"]
        )
        
        session.add(memo)
        await session.commit()
        await session.refresh(memo)
        
        return memo
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create memo: {str(e)}"
        )


@router.get("/{memo_id}", response_model=MemoSchema)
async def get_memo(
    memo_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a specific memo by ID."""
    result = await session.execute(
        select(Memo).where(Memo.id == memo_id, Memo.user_id == current_user["uid"])
    )
    memo = result.scalar_one_or_none()
    
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    
    return memo


@router.put("/{memo_id}", response_model=MemoSchema)
async def update_memo(
    memo_id: str,
    memo_update: MemoUpdate,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Update an existing memo."""
    result = await session.execute(
        select(Memo).where(Memo.id == memo_id, Memo.user_id == current_user["uid"])
    )
    memo = result.scalar_one_or_none()
    
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    
    try:
        # Update fields if provided
        if memo_update.title is not None:
            memo.title = memo_update.title
        if memo_update.content is not None:
            memo.content = memo_update.content
        
        await session.commit()
        await session.refresh(memo)
        
        return memo
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update memo: {str(e)}"
        )


@router.delete("/{memo_id}")
async def delete_memo(
    memo_id: str,
    current_user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a memo."""
    result = await session.execute(
        select(Memo).where(Memo.id == memo_id, Memo.user_id == current_user["uid"])
    )
    memo = result.scalar_one_or_none()
    
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    
    try:
        await session.delete(memo)
        await session.commit()
        
        return {"message": "Memo deleted successfully"}
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete memo: {str(e)}"
        )


@router.get("/", response_model=List[MemoSchema])
async def list_memos(
    deal_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """List memos with optional filtering by deal_id or user_id."""
    query = select(Memo)
    
    if deal_id:
        query = query.where(Memo.deal_id == deal_id)
    if user_id:
        query = query.where(Memo.user_id == user_id)
    
    result = await session.execute(query.order_by(Memo.updated_at.desc()))
    memos = result.scalars().all()
    
    return memos


@router.post("/generate")
async def generate_memo(
    request: MemoGenerateRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user)
):
    """Generate a memo using the JSON template and selected documents, save to PDF file."""
    try:
        import os
        from datetime import datetime
        
        # Prepare custom fields dictionary
        custom_fields = {}
        if request.discussion_date:
            custom_fields['discussion_date'] = request.discussion_date
        if request.ftf_equity_size:
            custom_fields['ftf_equity_size'] = request.ftf_equity_size
        if request.expected_closing:
            custom_fields['expected_closing'] = request.expected_closing
        # Note: org chart image is handled separately (not a text field)
        
        # Determine document IDs: prefer explicit list, otherwise infer from deal_id
        document_ids: Optional[List[str]] = request.document_ids
        if not document_ids:
            if not request.deal_id:
                raise HTTPException(status_code=422, detail="Either 'document_ids' or 'deal_id' must be provided")
            result = await session.execute(select(Document.id).where(Document.deal_id == request.deal_id))
            rows = result.all()
            document_ids = [row[0] for row in rows]
            if not document_ids:
                raise HTTPException(status_code=400, detail="No documents found for the specified deal")

        memo_data = await auto_memo_generator.generate_complete_memo(
            document_ids=document_ids,
            session=session,
            custom_fields=custom_fields if custom_fields else None,
            user_id=current_user.get("uid")  # SECURITY: Pass user_id for isolation
        )

        # If an Organization Chart image was provided, inject it into the corresponding section
        if request.org_chart_image_base64:
            # Minimal validation: ensure it is a data URL for an image
            data_url = request.org_chart_image_base64.strip()
            if not data_url.startswith("data:image/"):
                raise HTTPException(status_code=400, detail="org_chart_image_base64 must be a data URL for an image")
            # Construct HTML to center and fit the image nicely on the page
            image_html = (
                f'<img src="{data_url}" alt="Organization Chart" '
                f'style="display:block;margin:0 auto;max-width:100%;height:auto;max-height:9in;" />'
            )
            # Overwrite the auto-generated placeholder for this section
            memo_data['organization_chart'] = image_html

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investment_memo_{timestamp}.pdf"
        
        # Ensure uploads directory exists
        os.makedirs("uploads/memos", exist_ok=True)
        file_path = f"uploads/memos/{filename}"
        
        # Generate PDF (returns actual file path, may be .html if PDF generation fails)
        actual_file_path = auto_memo_generator.generate_pdf(memo_data, file_path)
        
        # Get file size from the actual file created
        file_size = os.path.getsize(actual_file_path)
        
        # Update filename to match actual file created
        actual_filename = os.path.basename(actual_file_path)

        return {
            "title": "AI Generated Investment Memo",
            "filename": actual_filename,
            "file_path": actual_file_path,
            "generated_at": datetime.now().isoformat(),
            "document_count": len(document_ids),
            "file_size": file_size,
            "message": f"Memo generated successfully and saved to {actual_filename.split('.')[-1].upper()} file"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memo: {str(e)}")


@router.get("/download/{filename}")
async def download_memo_file(filename: str):
    """Download a previously generated memo file."""
    try:
        import os
        from fastapi.responses import FileResponse
        
        # Security: only allow files from memos directory with .pdf extension
        if not filename.endswith('.pdf') or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        file_path = f"uploads/memos/{filename}"
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@router.post("/generate/download")
async def generate_and_download_memo(
    request: MemoGenerateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Generate a memo and return it as a downloadable PDF file."""
    try:
        import tempfile
        import os
        from fastapi.responses import FileResponse
        
        # Determine document IDs as in the other endpoint
        document_ids: Optional[List[str]] = request.document_ids
        if not document_ids:
            if not request.deal_id:
                raise HTTPException(status_code=422, detail="Either 'document_ids' or 'deal_id' must be provided")
            result = await session.execute(select(Document.id).where(Document.deal_id == request.deal_id))
            rows = result.all()
            document_ids = [row[0] for row in rows]
            if not document_ids:
                raise HTTPException(status_code=400, detail="No documents found for the specified deal")

        memo_data = await auto_memo_generator.generate_complete_memo(
            document_ids=document_ids,
            session=session,
        )

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            
        # Generate PDF
        auto_memo_generator.generate_pdf(memo_data, temp_path)
        
        # Return as downloadable PDF file
        return FileResponse(
            path=temp_path,
            filename="investment_memo.pdf",
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=investment_memo.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memo: {str(e)}") 