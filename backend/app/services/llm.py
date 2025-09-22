from typing import List, Optional
import logging

from app.core.config import settings
from app.schemas.search import SearchResult
from .vertex_ai import vertex_ai_service

# Set up logging
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        # Initialize Vertex AI service
        logger.info("Initializing LLM service with Vertex AI...")
        self.use_vertex_ai = vertex_ai_service.validate_configuration()
        logger.info(f"Using Vertex AI for LLM: {self.use_vertex_ai}")

        # Fallback model info
        self.model = settings.LLM_MODEL

    def build_rag_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """Build a RAG prompt with context from search results."""
        if not search_results:
            return f"""You are a helpful AI assistant for deal analysis in search funds. Answer briefly:

        Question: {query}

        Provide a concise answer based on your knowledge."""

        # Build context from search results (limit context size)
        context_parts = []
        total_chars = 0
        max_context_chars = 2000  # Limit context to avoid token overflow

        for i, result in enumerate(search_results, 1):
            content_preview = (
                result.content[:300] + "..."
                if len(result.content) > 300
                else result.content
            )
            context_part = (
                f"Doc {i}: {result.document_name}\nContent: {content_preview}\n"
            )

            if total_chars + len(context_part) > max_context_chars:
                break

            context_parts.append(context_part)
            total_chars += len(context_part)

        context = "\n---\n".join(context_parts)

        prompt = f"""You are an AI assistant for deal analysis. Use the document excerpts to answer the question concisely.

                    CONTEXT:
                    {context}

                    QUESTION: {query}

                    Provide a brief, focused answer based on the documents above."""

        return prompt

    async def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> str:
        """Generate an AI answer using Vertex AI with RAG context."""
        if not self.use_vertex_ai:
            logger.warning("Vertex AI not configured, using fallback response")
            # Fallback response when Vertex AI is not available
            if search_results:
                context_info = (
                    f"\n\nBased on {len(search_results)} relevant documents found:"
                )
                for i, result in enumerate(search_results[:3], 1):
                    context_info += (
                        f"\n{i}. {result.document_name}: {result.content[:100]}..."
                    )
                return f"Vertex AI LLM response for: {query}{context_info}"
            else:
                return (
                    f"Vertex AI LLM response for: {query} (no relevant documents found)"
                )

        try:
            # Build the RAG prompt
            prompt = self.build_rag_prompt(query, search_results)

            logger.info(
                f"Generating answer for query: '{query[:50]}...' with {len(search_results)} search results"
            )

            # Call Vertex AI API
            answer = await vertex_ai_service.generate_text(
                prompt, max_tokens=max_tokens, temperature=temperature
            )

            logger.info("Answer generated successfully by Vertex AI")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer with Vertex AI: {str(e)}")
            return (
                f"Sorry, I encountered an error while generating the answer: {str(e)}"
            )

    async def generate_document_summary(
        self, document_content: str, document_name: str, max_tokens: int = 500
    ) -> str:
        """Generate a summary of a document using Vertex AI."""
        if not self.use_vertex_ai:
            logger.warning("Vertex AI not configured, using fallback summary")
            # Simple fallback summary
            content_preview = document_content[:500]
            return (
                f"Summary of {document_name}:\n\n{content_preview}..."
                if len(document_content) > 500
                else content_preview
            )

        try:
            prompt = f"""Please provide a concise summary of the following document:

Document Name: {document_name}

Content:
{document_content[:4000]}  

Provide a summary that includes:
1. Document type and purpose
2. Key parties involved
3. Main terms and conditions
4. Important dates and milestones
5. Financial terms (if applicable)

Summary:"""

            logger.info(f"Generating summary for document: {document_name}")

            summary = await vertex_ai_service.generate_text(
                prompt, max_tokens=max_tokens, temperature=0.1
            )

            logger.info("Document summary generated successfully by Vertex AI")
            return summary

        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def generate_investment_memo_section(
        self,
        section_title: str,
        context_documents: str,
        specific_prompt: str = "",
        max_tokens: int = 800,
    ) -> str:
        """Generate a specific section of an investment memo using Vertex AI."""
        if not self.use_vertex_ai:
            return f"Vertex AI LLM-generated content for {section_title}:\n\n[Investment memo section content would be generated here using the provided context documents.]"

        try:
            prompt = f"""You are an expert investment analyst writing a professional investment memorandum section.

SECTION: {section_title}

CONTEXT DOCUMENTS:
{context_documents[:3000]}

{specific_prompt if specific_prompt else f"Write a comprehensive {section_title} section for an investment memorandum based on the provided context."}

INSTRUCTIONS:
1. Write in a professional, analytical tone
2. Synthesize and analyze the information from the context documents - do not quote extensively from the source material
3. Present key insights and conclusions based on the data rather than repeating verbatim quotes
4. Structure the content with clear headings and bullet points where appropriate
5. Be concise but thorough
6. When referencing specific data points, paraphrase rather than quote directly

{section_title.upper()}:"""

            logger.info(f"Generating investment memo section: {section_title}")

            section_content = await vertex_ai_service.generate_text(
                prompt, max_tokens=max_tokens, temperature=0.2
            )

            logger.info(
                f"Investment memo section '{section_title}' generated successfully"
            )
            return section_content

        except Exception as e:
            logger.error(f"Error generating memo section '{section_title}': {str(e)}")
            return f"Error generating {section_title} section: {str(e)}"

    async def extract_key_metrics(
        self, financial_documents: str, max_tokens: int = 600
    ) -> str:
        """Extract key financial metrics from documents using Vertex AI."""
        if not self.use_vertex_ai:
            return "Key financial metrics extraction requires Vertex AI configuration."

        try:
            prompt = f"""Extract and summarize key financial metrics from the following documents:

FINANCIAL DOCUMENTS:
{financial_documents[:3000]}

Please extract and organize the following information if available:
1. Revenue figures (historical and projected)
2. Profit margins and EBITDA
3. Growth rates
4. Key ratios (P/E, debt-to-equity, etc.)
5. Market size and share
6. Customer metrics (acquisition cost, lifetime value, etc.)
7. Operational metrics

Format the response as a structured list with clear categories.

KEY FINANCIAL METRICS:"""

            logger.info("Extracting key financial metrics from documents")

            metrics = await vertex_ai_service.generate_text(
                prompt, max_tokens=max_tokens, temperature=0.1
            )

            logger.info("Key financial metrics extracted successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error extracting financial metrics: {str(e)}")
            return f"Error extracting financial metrics: {str(e)}"


# Global instance
llm_service = LLMService()
