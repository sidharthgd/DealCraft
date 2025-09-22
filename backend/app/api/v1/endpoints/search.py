from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from app.services.search import get_search_service
from app.services.llm import llm_service
from app.schemas.search import SearchQuery, SearchResponse
from app.services.auth import get_current_user

router = APIRouter()


@router.get("/", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., description="Search query"),
    deal_id: Optional[str] = Query(None, description="Filter by deal ID"),
    top_k: int = Query(6, description="Number of results to return", ge=1, le=20),
    include_answer: bool = Query(True, description="Generate AI answer using RAG"),
    current_user: dict = Depends(get_current_user)
):
    """Search for relevant document chunks and optionally generate an AI answer."""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    try:
        # Perform vector search
        search_service = get_search_service()
        search_results = await search_service.search_documents(
            query=query,
            deal_id=deal_id,
            top_k=top_k,
            user_id=current_user.get("uid")  # SECURITY: Filter by user_id
        )
        
        # Generate AI answer if requested
        answer = None
        if include_answer:
            answer = await llm_service.generate_answer(query, search_results)
        
        return SearchResponse(
            results=search_results,
            query=query,
            total_results=len(search_results),
            answer=answer
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/", response_model=SearchResponse)
async def search_documents_post(search_query: SearchQuery):
    """Search for relevant document chunks using POST method (for complex queries)."""
    
    if not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    try:
        # Perform vector search
        search_service = get_search_service()
        search_results = await search_service.search_documents(
            query=search_query.query,
            deal_id=None,  # Could be extended to include deal_id in schema
            top_k=6
        )
        
        # Generate AI answer
        answer = await llm_service.generate_answer(search_query.query, search_results)
        
        return SearchResponse(
            results=search_results,
            query=search_query.query,
            total_results=len(search_results),
            answer=answer
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/chunk/{chunk_id}")
async def get_chunk_details(chunk_id: str):
    """Get detailed information about a specific document chunk."""
    
    try:
        chunk = await search_service.get_chunk_details(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "created_at": chunk.created_at
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get chunk details: {str(e)}"
        ) 