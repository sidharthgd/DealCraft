from typing import List, Optional
from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str


class SearchResult(BaseModel):
    id: str
    document_id: str
    document_name: str
    content: str
    similarity_score: float
    chunk_index: int
    start_char: int
    end_char: int
    # Optional fields for source tracking and RAG explainability
    source_query: Optional[str] = None  # Which query found this result
    query_similarity: Optional[float] = None  # Similarity score for the specific query


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    answer: Optional[str] = None
    total_results: int 