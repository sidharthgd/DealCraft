from typing import List, Optional
import logging

from app.core.config import settings
from app.core.database import get_async_session
from app.models.document import DocumentChunk
from app.schemas.search import SearchResult, SearchResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .vertex_ai import vertex_ai_service
from .vector_search import get_vector_search_service

# Set up logging
logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        # Initialize Vertex AI service for embeddings
        logger.info("Initializing search service with Vertex AI embeddings...")
        self.use_vertex_ai = vertex_ai_service.validate_configuration()
        logger.info(f"Using Vertex AI for search embeddings: {self.use_vertex_ai}")
        
        # Initialize vector search service (Vertex AI Vector Search or ChromaDB fallback)
        self.vector_service = get_vector_search_service()
        logger.info(f"Vector search service initialized: {type(self.vector_service).__name__}")
        
        # Legacy ChromaDB support (will be deprecated)
        self.collection = None
        try:
            if hasattr(self.vector_service, 'collection'):
                self.collection = self.vector_service.collection
                logger.info("Using ChromaDB collection for backward compatibility")
        except Exception as e:
            logger.warning(f"ChromaDB collection not available: {e}")

    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for search query using Vertex AI or raise exception for fallback"""
        if self.use_vertex_ai:
            try:
                logger.info(f"Using Vertex AI embeddings for query: '{query[:50]}...'")
                embedding = await vertex_ai_service.get_single_embedding(query)
                if embedding and len(embedding) > 0:
                    logger.info(f"Successfully generated Vertex AI embedding with {len(embedding)} dimensions")
                    return embedding
                else:
                    raise RuntimeError("Vertex AI returned empty embedding")
            except Exception as e:
                logger.warning(f"Vertex AI embedding failed for query: {e}")
                # Don't return fallback embedding - let the search service handle it
                raise RuntimeError(f"Embedding generation failed: {e}") from e
        else:
            raise RuntimeError("Vertex AI not configured, no embedding available")

    async def search_documents(
        self,
        query: str,
        deal_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for relevant document chunks using vector similarity."""
        if not self.collection:
            logger.warning("No ChromaDB collection available for search, falling back to database search")
            return await self._fallback_database_search(query, deal_id, document_ids, top_k, user_id)
        
        # Debug: Check ChromaDB collection status
        try:
            total_count = self.collection.count()
            logger.info(f"ChromaDB collection contains {total_count} items")
            if total_count == 0:
                logger.warning("ChromaDB collection is empty - this may indicate GCS sync issues or first startup")
        except Exception as count_error:
            logger.warning(f"Could not check ChromaDB collection count: {count_error}")
        
        top_k = top_k or settings.SEARCH_TOP_K
        
        # Try to generate query embedding using Vertex AI
        try:
            logger.info(f"Generating embedding for search query: '{query[:50]}...'")
            query_embedding = await self.get_query_embedding(query)
            
            # Log embedding details for debugging
            logger.info(f"Query embedding generated: {len(query_embedding)} dimensions, "
                       f"first 5 values: {query_embedding[:5]}")
            
            # Prepare where filter (ChromaDB requires a single operator at the top level)
            def _build_where(user_id_val, deal_id_val, document_ids_val):
                conditions = []
                if user_id_val:
                    conditions.append({"user_id": user_id_val})
                if deal_id_val:
                    conditions.append({"deal_id": deal_id_val})
                if document_ids_val:
                    conditions.append({"document_id": {"$in": document_ids_val}})
                if not conditions:
                    return None
                if len(conditions) == 1:
                    return conditions[0]
                return {"$and": conditions}

            where_filter = _build_where(user_id, deal_id, document_ids)

            # Logging for transparency
            if user_id:
                logger.info(f"Filtering search results by user_id: {user_id}")
            else:
                logger.warning("No user_id provided for search - this is a security risk!")
            if deal_id:
                logger.info(f"Filtering search results by deal_id: {deal_id}")
            if document_ids:
                logger.info(f"Filtering search results by document_ids: {document_ids}")
            
            # Search in vector database
            try:
                logger.info(f"Searching ChromaDB for top {top_k} results...")
                logger.info(f"Where filter: {where_filter}")
                
                # Debug: Check total collection size first
                try:
                    total_count = self.collection.count()
                    logger.info(f"Total items in ChromaDB collection: {total_count}")
                except Exception as count_error:
                    logger.warning(f"Could not get collection count: {count_error}")
                
                # Debug: Try search without filters first
                if where_filter is not None:
                    try:
                        unfiltered_results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k,
                            where=None,
                            include=['documents', 'metadatas', 'distances']
                        )
                        unfiltered_count = len(unfiltered_results['ids'][0]) if unfiltered_results['ids'] else 0
                        logger.info(f"Unfiltered search returned {unfiltered_count} results")
                        if unfiltered_count > 0:
                            sample_metadata = unfiltered_results['metadatas'][0][0] if unfiltered_results['metadatas'] else {}
                            logger.info(f"Sample metadata: {sample_metadata}")
                    except Exception as debug_error:
                        logger.warning(f"Debug unfiltered search failed: {debug_error}")
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
                results_count = len(results['ids'][0]) if results['ids'] else 0
                logger.info(f"ChromaDB search completed, found {results_count} results")
                
                # Fallback to database search if ChromaDB returns 0 results
                if results_count == 0:
                    logger.info("ChromaDB returned 0 results, falling back to database search")
                    return await self._fallback_database_search(query, deal_id, document_ids, top_k, user_id)
            except Exception as e:
                logger.error(f"Error searching in ChromaDB: {str(e)}")
                logger.info("Falling back to database search due to ChromaDB error")
                return await self._fallback_database_search(query, deal_id, document_ids, top_k, user_id)
                
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            logger.info("Falling back to database search due to embedding failure")
            return await self._fallback_database_search(query, deal_id, document_ids, top_k, user_id)

        # Convert results to SearchResult objects
        search_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (higher is better)
                similarity_score = 1.0 - distance
                
                search_result = SearchResult(
                    id=chunk_id,
                    document_id=metadata['document_id'],
                    document_name=metadata['document_name'],
                    content=document,
                    similarity_score=similarity_score,
                    chunk_index=metadata['chunk_index'],
                    start_char=metadata['start_char'],
                    end_char=metadata['end_char'],
                    source_query=f"Query: {query[:50]}...",  # Add source tracking
                    query_similarity=similarity_score  # Add query-specific similarity
                )
                search_results.append(search_result)
            
            logger.info(f"Converted {len(search_results)} search results with similarity scores")
            
            # Log top results for debugging
            if search_results:
                top_result = search_results[0]
                logger.info(f"Top result: {top_result.document_name} (Chunk {top_result.chunk_index}) - "
                           f"Similarity: {top_result.similarity_score:.3f}")

        return search_results

    async def _fallback_database_search(
        self,
        query: str,
        deal_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Fallback search using direct database queries when ChromaDB is unavailable."""
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import select, and_, or_, func
        from app.models.document import Document, DocumentChunk
        
        top_k = top_k or settings.SEARCH_TOP_K
        
        async with AsyncSessionLocal() as session:
            # Extract search keywords from query with generic business terms
            generic_business_keywords = ['ceo', 'founded', 'headquarters', 'leadership', 'executive', 'president', 
                                       'founder', 'location', 'address', 'established', 'incorporated', 'management']
            
            query_words = [word.lower().strip() for word in query.split() if len(word.strip()) > 2]
            query_words = [kw for kw in query_words if kw not in ['the', 'and', 'for', 'are', 'with']]
            
            # Combine query words with generic business keywords
            all_keywords = list(set(query_words + generic_business_keywords))[:15]  # Use more keywords
            
            logger.info(f"Database fallback search for keywords: {all_keywords}")
            
            # Build filter conditions
            conditions = []
            
            # SECURITY: Always filter by user_id for isolation
            if user_id:
                conditions.append(Document.user_id == user_id)
            else:
                logger.warning("No user_id provided for database fallback search - security risk!")
            
            # Add document ID filter if provided
            if document_ids:
                conditions.append(Document.id.in_(document_ids))
            
            if deal_id:
                conditions.append(Document.deal_id == deal_id)
            
            # Build content search condition with enhanced matching
            if all_keywords:
                content_conditions = []
                for keyword in all_keywords:
                    content_conditions.append(DocumentChunk.content.ilike(f'%{keyword}%'))
                # Use OR for keyword matching to find chunks containing any keyword
                conditions.append(or_(*content_conditions))
            
            # Execute search query
            query_obj = (
                select(
                    DocumentChunk.id,
                    DocumentChunk.content,
                    DocumentChunk.document_id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.start_char,
                    DocumentChunk.end_char,
                    Document.name.label('document_name')
                )
                .join(Document)
                .where(and_(*conditions) if conditions else True)
                .limit(top_k)
            )
            
            result = await session.execute(query_obj)
            chunks = result.fetchall()
            
            logger.info(f"Database fallback search found {len(chunks)} chunks")
            
            # Convert to SearchResult objects
            search_results = []
            for i, chunk in enumerate(chunks):
                # Calculate enhanced relevance score based on keyword matches
                relevance_score = 0.5  # Base score
                content_lower = chunk.content.lower()
                
                # Count matches for different keyword types
                business_matches = sum(1 for kw in generic_business_keywords if kw in content_lower)
                query_matches = sum(1 for kw in query_words if kw in content_lower)
                
                # Weight business-related terms and direct query matches
                relevance_score += (business_matches * 0.12) + (query_matches * 0.10)
                relevance_score = min(relevance_score, 0.95)  # Cap at 0.95
                
                search_result = SearchResult(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    content=chunk.content,
                    similarity_score=relevance_score,
                    chunk_index=chunk.chunk_index or 0,
                    start_char=chunk.start_char or 0,
                    end_char=chunk.end_char or len(chunk.content),
                    source_query=f"Database search: {query[:50]}...",
                    query_similarity=relevance_score
                )
                search_results.append(search_result)
            
            logger.info(f"Converted {len(search_results)} results from database search")
            return search_results

    async def search_similar_to_text(
        self,
        text: str,
        deal_id: Optional[str] = None,
        top_k: int = None,
        exclude_document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Find documents similar to a given text (useful for finding related documents)"""
        if not self.collection:
            return []
        
        top_k = top_k or settings.SEARCH_TOP_K
        
        # Generate embedding for the input text
        text_embedding = await self.get_query_embedding(text)
        
        # Prepare where filter
        where_filter = {}
        if deal_id:
            where_filter["deal_id"] = deal_id
        if exclude_document_id:
            # Note: ChromaDB doesn't support "not equal" filters directly
            # This would need to be filtered after the query
            pass
        
        # Search in vector database
        try:
            results = self.collection.query(
                query_embeddings=[text_embedding],
                n_results=top_k * 2 if exclude_document_id else top_k,  # Get more if filtering
                where=where_filter if where_filter else None,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.error(f"Error searching similar text in ChromaDB: {str(e)}")
            return []

        # Convert and filter results
        search_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Skip if excluding this document
                if exclude_document_id and metadata['document_id'] == exclude_document_id:
                    continue
                
                # Convert distance to similarity score (higher is better)
                similarity_score = 1.0 - distance
                
                search_result = SearchResult(
                    id=chunk_id,
                    document_id=metadata['document_id'],
                    document_name=metadata['document_name'],
                    content=document,
                    similarity_score=similarity_score,
                    chunk_index=metadata['chunk_index'],
                    start_char=metadata['start_char'],
                    end_char=metadata['end_char'],
                    source_query=f"Similar to text: {text[:50]}...",  # Add source tracking
                    query_similarity=similarity_score  # Add query-specific similarity
                )
                search_results.append(search_result)
                
                # Stop if we have enough results
                if len(search_results) >= top_k:
                    break

        return search_results

    async def get_chunk_details(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get full chunk details from the database."""
        async with get_async_session() as session:
            result = await session.execute(
                select(DocumentChunk).where(DocumentChunk.id == chunk_id)
            )
            return result.scalar_one_or_none()

    async def search_with_context(
        self,
        query: str,
        deal_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = None
    ) -> SearchResponse:
        """Search and return formatted response with context."""
        logger.info(f"Performing contextual search for query: '{query[:50]}...'")
        search_results = await self.search_documents(query, deal_id, document_ids, top_k)
        
        logger.info(f"Search completed with {len(search_results)} results")
        return SearchResponse(
            results=search_results,
            query=query,
            total_results=len(search_results),
            answer=None  # Will be filled by LLM service
        )

    async def get_document_summary(self, document_id: str) -> Optional[str]:
        """Get a summary of a document by analyzing its chunks"""
        if not self.collection:
            return None
        
        try:
            # Get all chunks for this document
            results = self.collection.query(
                query_embeddings=None,
                n_results=1000,  # Get many chunks
                where={"document_id": document_id},
                include=['documents', 'metadatas']
            )
            
            if not results['ids'] or not results['ids'][0]:
                return None
            
            # Combine chunks to create a summary using Vertex AI
            chunks = results['documents'][0]
            combined_text = "\n\n".join(chunks[:5])  # Use first 5 chunks for summary
            
            if self.use_vertex_ai:
                summary_prompt = f"""Please provide a concise summary of this document:

{combined_text[:2000]}...

Summary:"""
                try:
                    summary = await vertex_ai_service.generate_text(summary_prompt, max_tokens=200)
                    return summary
                except Exception as e:
                    logger.warning(f"Failed to generate AI summary: {e}")
            
            # Fallback: return first chunk as summary
            return chunks[0][:500] + "..." if len(chunks[0]) > 500 else chunks[0]
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return None


# Global instance - will be initialized lazily
search_service = None

def get_search_service():
    global search_service
    if search_service is None:
        search_service = SearchService()
    return search_service 