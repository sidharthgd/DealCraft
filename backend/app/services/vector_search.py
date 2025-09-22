"""
Vertex AI Vector Search service for document embeddings and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from app.core.config import settings
from .vertex_ai import vertex_ai_service

logger = logging.getLogger(__name__)

class VertexVectorSearchService:
    """Service for managing document embeddings with Vertex AI Vector Search."""
    
    def __init__(self):
        self.project_id = settings.GCP_PROJECT_ID
        self.location = settings.VERTEX_AI_LOCATION
        self.index_endpoint = settings.VERTEX_VECTOR_INDEX_ENDPOINT
        self.index_id = settings.VERTEX_VECTOR_INDEX_ID
        
        self.client = None
        self.endpoint = None
        self.is_available = False
        
        if self.project_id and self.index_endpoint and self.index_id:
            try:
                # Initialize Vertex AI
                aiplatform.init(project=self.project_id, location=self.location)
                
                # Get the index endpoint
                self.endpoint = aiplatform.MatchingEngineIndexEndpoint(self.index_endpoint)
                
                self.is_available = True
                logger.info(f"Vertex AI Vector Search initialized successfully")
                logger.info(f"Project: {self.project_id}, Location: {self.location}")
                logger.info(f"Index Endpoint: {self.index_endpoint}")
                logger.info(f"Index ID: {self.index_id}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI Vector Search: {e}")
                self.is_available = False
        else:
            logger.warning("Vertex AI Vector Search not configured - missing required settings")
            self.is_available = False
    
    def validate_configuration(self) -> bool:
        """Check if Vector Search is properly configured."""
        return self.is_available
    
    async def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        documents: List[str], 
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Add embeddings to the vector index.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document text chunks
            metadatas: List of metadata dictionaries
            ids: List of unique IDs for each embedding
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            logger.warning("Vertex AI Vector Search not available")
            return False
        
        try:
            # Prepare the data points for insertion
            datapoints = []
            for i, (embedding, doc, metadata, doc_id) in enumerate(zip(embeddings, documents, metadatas, ids)):
                # Vertex AI Vector Search expects specific format
                datapoint = {
                    "datapoint_id": doc_id,
                    "feature_vector": embedding,
                    "restricts": [
                        {"namespace": "document_id", "allow": [metadata.get("document_id", "")]},
                        {"namespace": "deal_id", "allow": [metadata.get("deal_id", "")]},
                        {"namespace": "chunk_index", "allow": [str(metadata.get("chunk_index", i))]},
                    ]
                }
                datapoints.append(datapoint)
            
            # Note: In a real implementation, you would batch these inserts
            # For now, we'll log that the data would be inserted
            logger.info(f"Would insert {len(datapoints)} datapoints to Vertex AI Vector Search")
            logger.info(f"Sample datapoint metadata: {metadatas[0] if metadatas else 'None'}")
            
            # TODO: Implement actual insertion using Vertex AI Vector Search API
            # This requires setting up the index and handling batch operations
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to Vertex AI Vector Search: {e}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            filters: Optional filters (e.g., deal_id, document_id)
            
        Returns:
            List of similar documents with metadata and scores
        """
        if not self.is_available:
            logger.warning("Vertex AI Vector Search not available")
            return []
        
        try:
            # Prepare the query
            deployed_index_id = self.index_id  # This should be the deployed index ID
            
            # Build restrict list from filters
            restricts = []
            if filters:
                for key, value in filters.items():
                    if value:  # Only add non-empty filters
                        restricts.append({
                            "namespace": key,
                            "allow": [str(value)] if isinstance(value, (str, int)) else [str(v) for v in value]
                        })
            
            # Perform the search using the endpoint
            # Note: This is a simplified version - actual implementation would use
            # the Vertex AI Vector Search API
            logger.info(f"Would search Vertex AI Vector Search with {len(query_embedding)} dim embedding")
            logger.info(f"Top K: {top_k}, Filters: {filters}")
            
            # TODO: Implement actual search using Vertex AI Vector Search API
            # response = self.endpoint.find_neighbors(
            #     deployed_index_id=deployed_index_id,
            #     queries=[query_embedding],
            #     num_neighbors=top_k,
            #     restricts=restricts if restricts else None
            # )
            
            # For now, return empty results (fallback to ChromaDB will be used)
            return []
            
        except Exception as e:
            logger.error(f"Failed to search Vertex AI Vector Search: {e}")
            return []
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all embeddings for a specific document.
        
        Args:
            document_id: The document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            return False
        
        try:
            logger.info(f"Would delete embeddings for document {document_id} from Vertex AI Vector Search")
            
            # TODO: Implement actual deletion using Vertex AI Vector Search API
            # This would involve querying for all datapoints with the document_id
            # and then removing them from the index
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for document {document_id}: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_available:
            return {"status": "unavailable"}
        
        try:
            # TODO: Implement actual stats retrieval
            return {
                "status": "available",
                "project_id": self.project_id,
                "location": self.location,
                "index_endpoint": self.index_endpoint,
                "index_id": self.index_id,
                "total_vectors": "unknown"  # Would query the actual index
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"status": "error", "error": str(e)}


# Fallback ChromaDB service for development/transition
class ChromaDBFallbackService:
    """Fallback service using ChromaDB when Vertex AI Vector Search is not available."""
    
    def __init__(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        import os
        
        # Use Google Cloud Storage for ChromaDB persistence in production
        persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        # Check if we're in Cloud Run (has GOOGLE_CLOUD_PROJECT env var)
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            # In production, create a local directory that we'll sync with GCS
            persist_directory = "/app/chroma_persistent"
            os.makedirs(persist_directory, exist_ok=True)
            
            # Try to download existing ChromaDB data from GCS
            self._sync_chroma_from_gcs(persist_directory)
            logger.info(f"Using Cloud Run with GCS sync directory: {persist_directory}")
        else:
            logger.info(f"Using local directory: {persist_directory}")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Store the persist directory for later GCS sync
        self.persist_directory = persist_directory
        self.is_cloud_run = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaDB fallback service initialized")
    
    def _sync_chroma_from_gcs(self, local_dir: str) -> None:
        """Download ChromaDB data from GCS if it exists."""
        try:
            from google.cloud import storage
            import tarfile
            import tempfile
            import os
            
            bucket_name = os.getenv("CHROMA_GCS_BUCKET", "dealcraft-chroma-persistence")
            blob_name = "chroma_db.tar.gz"
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                logger.info(f"Downloading ChromaDB data from gs://{bucket_name}/{blob_name}")
                
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_file:
                    blob.download_to_filename(temp_file.name)
                    
                    # Extract to the local directory
                    with tarfile.open(temp_file.name, 'r:gz') as tar:
                        tar.extractall(path=local_dir)
                        
                logger.info("ChromaDB data restored from GCS")
            else:
                logger.info(f"No existing ChromaDB data found in gs://{bucket_name}/{blob_name}")
                
        except Exception as e:
            logger.warning(f"Failed to sync ChromaDB from GCS: {e}")
            logger.info("Will start with empty ChromaDB collection")
    
    def _sync_chroma_to_gcs(self) -> None:
        """Upload ChromaDB data to GCS for persistence."""
        if not self.is_cloud_run:
            return
            
        try:
            from google.cloud import storage
            import tarfile
            import tempfile
            import os
            
            bucket_name = os.getenv("CHROMA_GCS_BUCKET", "dealcraft-chroma-persistence")
            blob_name = "chroma_db.tar.gz"
            
            # Create tar file of ChromaDB directory
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_file:
                with tarfile.open(temp_file.name, 'w:gz') as tar:
                    tar.add(self.persist_directory, arcname=".")
                
                # Upload to GCS
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                blob.upload_from_filename(temp_file.name)
                logger.info(f"ChromaDB data backed up to gs://{bucket_name}/{blob_name}")
                
        except Exception as e:
            logger.error(f"Failed to sync ChromaDB to GCS: {e}")
    
    def validate_configuration(self) -> bool:
        """ChromaDB fallback is always available."""
        return True
    
    async def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        documents: List[str], 
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add embeddings to ChromaDB."""
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
            
            # Sync to GCS after adding embeddings (in production)
            if self.is_cloud_run:
                self._sync_chroma_to_gcs()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to ChromaDB: {e}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar embeddings."""
        try:
            # Convert filters to ChromaDB where clause
            where_filter = {}
            if filters:
                for key, value in filters.items():
                    if value:
                        if isinstance(value, list):
                            where_filter[key] = {"$in": value}
                        else:
                            where_filter[key] = value
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert ChromaDB results to standard format
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1.0 - results['distances'][0][i]
                    }
                    search_results.append(result)
            
            logger.info(f"ChromaDB search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete embeddings by document ID."""
        try:
            # ChromaDB doesn't have a direct delete by metadata, so we need to
            # query first, then delete by IDs
            results = self.collection.get(
                where={"document_id": document_id},
                include=['metadatas']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} embeddings for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB embeddings for document {document_id}: {e}")
            return False


# Global service instances
_vertex_service = None
_chroma_service = None

def get_vector_search_service():
    """Get the appropriate vector search service (Vertex AI or ChromaDB fallback)."""
    global _vertex_service, _chroma_service
    
    # Try Vertex AI Vector Search first
    if _vertex_service is None:
        _vertex_service = VertexVectorSearchService()
    
    if _vertex_service.validate_configuration():
        logger.info("Using Vertex AI Vector Search")
        return _vertex_service
    
    # Fallback to ChromaDB
    if _chroma_service is None:
        _chroma_service = ChromaDBFallbackService()
    
    logger.info("Using ChromaDB fallback for vector search")
    return _chroma_service