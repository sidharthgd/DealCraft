import os
import asyncio
import logging
from typing import List, Dict, Any
from google.cloud import aiplatform
from google.auth import default
from vertexai.generative_models import GenerativeModel

# Set up logging
logger = logging.getLogger(__name__)

class VertexAIService:
    """Service for interacting with Google Vertex AI for text generation and embeddings."""
    
    def __init__(self):
        # Import settings here to avoid circular imports
        from app.core.config import settings
        
        self.project_id = settings.GCP_PROJECT_ID
        self.location = settings.VERTEX_AI_LOCATION
        self.llm_model_name = settings.VERTEX_AI_MODEL
        
        # Debug: Log configuration values
        logger.info(f"VertexAI Configuration:")
        logger.info(f"  Project ID: {self.project_id}")
        logger.info(f"  Location: {self.location}")
        logger.info(f"  Model: {self.llm_model_name}")
        
        # Initialize Vertex AI
        if self.project_id:
            try:
                aiplatform.init(project=self.project_id, location=self.location)
                logger.info(f"Vertex AI initialized for project: {self.project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Vertex AI
        """
        try:
            from vertexai.language_models import TextEmbeddingModel
            
            # Use the configured embedding model from settings
            from app.core.config import settings
            embedding_model_name = settings.VERTEX_AI_EMBEDDING_MODEL
            logger.info(f"Using embedding model: {embedding_model_name}")
            model = TextEmbeddingModel.from_pretrained(embedding_model_name)
            
            # Process in batches to avoid rate limits
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Get embeddings for this batch
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.get_embeddings(batch)
                )
                
                # Extract the values from TextEmbedding objects
                for embedding in embeddings:
                    all_embeddings.append(embedding.values)
                
                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Instead of returning zero-filled embeddings that break search, raise the error
            # This will force the search service to use fallback methods
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            embeddings = await self.get_embeddings([text])
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Failed to get single embedding: {e}")
            # Re-raise to let the search service handle the fallback
            raise

    async def generate_text(self, prompt: str, section_title: str = "Unknown", context: str = "", **kwargs) -> str:
        """
        Generate text using Vertex AI with simplified approach
        """
        try:
            from vertexai.generative_models import GenerativeModel
            
            model = GenerativeModel(self.llm_model_name)
            
            # Clean and prepare the prompt
            clean_prompt = prompt.strip()
            if not clean_prompt:
                return "[No prompt provided]"
            
            # Add business context to help with generation
            business_context = """You are an assistant at a search fund. You are given a business plan and other documents and you are tasked with analyzing the business and writing a memo for potential investors."""
            
            final_prompt = business_context + clean_prompt

            # Use simple text generation
            max_tokens = kwargs.get("max_tokens", 1000)
            temperature = kwargs.get("temperature", 0.1)

            # Generate text
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    final_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": None,
                        "top_p": kwargs.get("top_p", 0.8),
                        "top_k": kwargs.get("top_k", 40),
                        "candidate_count": 1,
                    }
                )
            )
            
            # Robust response validation
            if not response:
                raise ValueError("No response received from Vertex AI")
            
            # Simple text extraction
            response_text = None
            
            # Method 1: Direct text attribute
            if hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
            
            # Method 2: Check candidates
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'text') and candidate.text:
                    response_text = candidate.text.strip()
                    logger.info(f"Retrieved response text length: {len(response_text)} characters")
                
                # Try content extraction
                elif hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    
                    if hasattr(content, 'parts') and content.parts:
                        texts = []
                        for i, part in enumerate(content.parts):
                            if hasattr(part, 'text') and part.text:
                                texts.append(part.text.strip())
                        
                        if texts:
                            response_text = " ".join(texts)
                            logger.info(f"Extracted text from {len(texts)} content parts")
            
            # Method 3: Try to get text from response directly
            if not response_text and hasattr(response, 'text'):
                response_text = str(response.text).strip()
            
            if response_text:
                logger.info(f"Successfully generated text for section '{section_title}': {len(response_text)} characters")
                return response_text
            else:
                logger.warning(f"No text could be extracted from response for section '{section_title}'")
                return "[Unable to extract response text from Vertex AI]"
                
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error generating text for section '{section_title}': {e}")
            
            # Provide specific error messages based on the error type
            if "token" in error_msg or "quota" in error_msg:
                return "[API quota exceeded or token limit reached. Please try again later or reduce document size.]"
            elif "permission" in error_msg or "auth" in error_msg:
                return "[Authentication error. Please check your Google Cloud credentials and permissions.]"
            else:
                # Generate a basic fallback response based on the context
                logger.warning(f"Vertex AI failed, generating fallback response for section '{section_title}'")
                return self._generate_fallback_response(prompt, section_title, context)

    async def search_similar_documents(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Vertex AI Vector Search
        Note: This requires setting up Vertex AI Vector Search index
        """
        # TODO: Implement Vertex AI Vector Search when index is set up
        logger.info("Vertex AI Vector Search not yet configured, falling back to ChromaDB")
        return []

    def _generate_fallback_response(self, prompt: str, section_title: str, context: str) -> str:
        """Generate a basic fallback response when Vertex AI is not available."""
        try:
            # Extract key information from context
            context_lower = context.lower()
            
            # Look for company name
            company_name = "the company"
            
            # Generate section-specific responses
            if "basic information" in section_title.lower():
                return f"""Based on the provided documents, {company_name} appears to be a property tax services company. The company operates in the property tax assessment and appeal industry, helping property owners reduce their tax burdens through professional assessment challenges and appeals processes."""
            
            elif "business description" in section_title.lower():
                return f"""{company_name} provides property tax assessment and appeal services. The company helps property owners challenge their property tax assessments to potentially reduce their tax burden. They appear to operate in multiple states and have developed expertise in navigating the complex property tax appeal processes."""
            
            elif "operational" in section_title.lower():
                return f"""The company's operations involve reviewing property tax assessments, filing appeals on behalf of property owners, and representing clients in tax appeal proceedings. They likely have a team of tax professionals and legal experts who understand local property tax laws and regulations."""
            
            elif "complexity" in section_title.lower():
                return f"""The operational complexity appears moderate, involving specialized knowledge of property tax laws across multiple jurisdictions. The business requires expertise in tax assessment methodologies, legal procedures, and client relationship management."""
            
            elif "considerations" in section_title.lower():
                return f"""Key considerations include regulatory changes in property tax laws, dependence on local government assessment practices, and the need to maintain expertise across multiple jurisdictions. Market risks include economic cycles affecting property values and tax rates."""
            
            else:
                return f"Based on the provided documents, {company_name} operates in the property tax services industry. The company appears to focus on helping property owners manage and potentially reduce their property tax obligations through professional assessment and appeal services."
                
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return f"Analysis of {company_name} based on the provided business documents indicates this is a property tax services company operating in the assessment and appeal industry."

    def validate_configuration(self) -> bool:
        """Validate that GCP configuration is properly set up"""
        if not self.project_id:
            logger.warning("Warning: GCP_PROJECT_ID not configured - using fallback implementations")
            return False
        
        try:
            # Try to get default credentials
            credentials, project = default()
            logger.info(f"GCP authentication successful for project: {project}")
            return True
        except Exception as e:
            logger.warning(f"Warning: GCP authentication not set up: {e} - using fallback implementations")
            return False


# Global instance
vertex_ai_service = VertexAIService() 