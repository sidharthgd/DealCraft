import uuid
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.models.document import DocumentChunk
from app.services.vertex_ai import vertex_ai_service

# Set up logging
logger = logging.getLogger(__name__)


class MemoGeneratorService:
    """Generate investment memos using JSON template and selected documents."""

    def __init__(self):
        self.template_path = Path(__file__).parent.parent / "core" / "memo_template.json"
        self.template_sections = self._load_template()
        
        # Register Georgia font for PDF generation
        try:
            # Try to register Georgia font from correct macOS location
            pdfmetrics.registerFont(TTFont('Georgia', '/System/Library/Fonts/Supplemental/Georgia.ttf'))
            self.font_name = 'Georgia'
            logger.info("Successfully registered Georgia font")
        except Exception as e:
            try:
                # Fallback to other common locations
                pdfmetrics.registerFont(TTFont('Georgia', '/System/Library/Fonts/Georgia.ttf'))
                self.font_name = 'Georgia'
                logger.info("Successfully registered Georgia font from alternate location")
            except Exception as e2:
                try:
                    # Another fallback location
                    pdfmetrics.registerFont(TTFont('Georgia', '/Library/Fonts/Georgia.ttf'))
                    self.font_name = 'Georgia'
                    logger.info("Successfully registered Georgia font from Library folder")
                except Exception as e3:
                    # If Georgia is not available, use default font
                    logger.warning(f"Georgia font not found, using default font. Errors: {e}, {e2}, {e3}")
                    self.font_name = 'Helvetica'

    def _load_template(self) -> List[Dict[str, Any]]:
        """Load the JSON template from file."""
        try:
            with open(self.template_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memo template: {e}")
            return []

    def _extract_company_info(self, context: str) -> Dict[str, str]:
        """Extract specific company information from context using pattern matching."""
        import re
        
        company_info = {
            "name": "Not specified",
            "industry": "Not specified", 
            "location": "Not specified",
            "size": "Not specified"
        }
        
        # Try multiple strategies to extract company name
        company_name = None
        
        # Strategy 1: Look for explicit company name patterns
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for "Company Name:" or "Company:" patterns
            if re.search(r'company\s*name\s*:', line, re.IGNORECASE):
                match = re.search(r'company\s*name\s*:\s*(.+)', line, re.IGNORECASE)
                if match:
                    company_name = match.group(1).strip()
                    break
            elif re.search(r'company\s*:', line, re.IGNORECASE):
                match = re.search(r'company\s*:\s*(.+)', line, re.IGNORECASE)
                if match:
                    company_name = match.group(1).strip()
                    break
        
        # Strategy 2: Look for corporate entity suffixes
        if not company_name:
            # Find text with corporate suffixes (Inc., LLC, Corp., Ltd., etc.)
            entity_pattern = r'\b([A-Z][A-Za-z\s&]+(?:Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?|Limited|LP|LLP))\b'
            matches = re.findall(entity_pattern, context)
            if matches:
                # Take the first match that looks like a company name
                for match in matches:
                    if len(match.strip()) > 3 and not any(word in match.lower() for word in ['the', 'and', 'or', 'of']):
                        company_name = match.strip()
                        break
        
        # Strategy 3: Look for title case names in header sections
        if not company_name:
            # Look for title case text near the beginning of the document
            first_500_chars = context[:500]
            title_case_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
            matches = re.findall(title_case_pattern, first_500_chars)
            if matches:
                # Filter for potential company names (2-4 words, no common words)
                for match in matches:
                    words = match.split()
                    if 2 <= len(words) <= 4 and not any(word.lower() in ['the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to'] for word in words):
                        company_name = match.strip()
                        break
        
        if company_name and company_name != "Not specified":
            # Clean up the company name
            company_name = re.sub(r'\s+', ' ', company_name).strip()
            company_info["name"] = company_name
        
        return company_info

    def _replace_placeholders(self, prompt_template: str, company_info: Dict[str, str]) -> str:
        """Replace placeholders in prompt template with actual company information."""
        # Replace [Company] with the actual company name
        company_name = company_info.get("name", "the company")
        if company_name == "Not specified":
            company_name = "the company"
        
        # Replace [Company] placeholder
        prompt_template = prompt_template.replace("[Company]", company_name)
        
        return prompt_template

    def _create_source_traceability_report(
        self, 
        section_title: str, 
        source_log: List[Dict[str, Any]]
    ) -> str:
        """Create a source traceability report for a section."""
        if not source_log:
            return "No source documents referenced."
        
        report_lines = []
        report_lines.append(f"=== SOURCE TRACEABILITY REPORT FOR {section_title.upper()} ===")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Group by document
        doc_sources = {}
        for entry in source_log:
            doc_id = entry.get('document_id', 'Unknown')
            if doc_id not in doc_sources:
                doc_sources[doc_id] = []
            doc_sources[doc_id].append(entry)
        
        for doc_id, entries in doc_sources.items():
            report_lines.append(f"Document ID: {doc_id}")
            report_lines.append(f"Number of chunks used: {len(entries)}")
            
            total_similarity = sum(entry.get('similarity', 0) for entry in entries)
            avg_similarity = total_similarity / len(entries) if entries else 0
            report_lines.append(f"Average similarity score: {avg_similarity:.3f}")
            
            # Show a few sample chunks
            report_lines.append("Sample chunks:")
            for i, entry in enumerate(entries[:3]):  # Show first 3 chunks
                chunk_preview = entry.get('chunk_text', '')[:100] + "..." if len(entry.get('chunk_text', '')) > 100 else entry.get('chunk_text', '')
                report_lines.append(f"  {i+1}. Similarity: {entry.get('similarity', 0):.3f} - {chunk_preview}")
            
            if len(entries) > 3:
                report_lines.append(f"  ... and {len(entries) - 3} more chunks")
            report_lines.append("")
        
        return "\n".join(report_lines)

    async def _collect_context_for_section(
        self, 
        document_ids: List[str], 
        section_config: Dict[str, Any],
        session: AsyncSession
    ) -> str:
        """Collect relevant context for a specific section using RAG."""
        try:
            section_title = section_config["section_title"]
            logger.info(f"Collecting context for section: {section_title}")
            
            # Get the query for this section
            query = section_config.get("query", "")
            if not query:
                logger.warning(f"No query found for section: {section_title}")
                return ""
            
            # Get embeddings for the query
            query_embedding = await vertex_ai_service.get_single_embedding(query)
            
            # Search for similar document chunks
            similar_chunks = await self._search_similar_chunks(
                query_embedding, document_ids, session, top_k=10
            )
            
            if not similar_chunks:
                logger.warning(f"No similar chunks found for section: {section_title}")
                return ""
            
            # Build context from chunks
            context_parts = []
            source_log = []
            
            for chunk in similar_chunks:
                chunk_text = chunk.get('chunk_text', '')
                similarity = chunk.get('similarity', 0)
                document_id = chunk.get('document_id', 'Unknown')
                
                if chunk_text and similarity > 0.3:  # Only include relevant chunks
                    context_parts.append(f"RELEVANT CONTENT (Source: {chunk.get('source_name', 'Unknown document')} (similarity: {similarity:.3f})): {chunk_text}")
                    
                    source_log.append({
                        'document_id': document_id,
                        'chunk_text': chunk_text,
                        'similarity': similarity,
                        'source_name': chunk.get('source_name', 'Unknown')
                    })
            
            context = "\n\n".join(context_parts)
            
            # Create and save source traceability report
            traceability_report = self._create_source_traceability_report(section_title, source_log)
            
            # Save traceability report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_title = "".join(c for c in section_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_').replace('-', '_')
            report_filename = f"source_traceability_{safe_title}_{timestamp}.txt"
            report_path = Path("logs") / report_filename
            
            # Ensure directory exists
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(traceability_report)
            
            logger.info(f"Source traceability report saved: {report_path}")
            logger.info(f"Collected {len(context_parts)} context chunks for section: {section_title}")
            
            return context
            
        except Exception as e:
            logger.error(f"Error collecting context for section {section_config.get('section_title', 'Unknown')}: {e}")
            return ""

    async def _search_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_ids: List[str], 
        session: AsyncSession, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks using embeddings."""
        try:
            # Get all chunks for the specified documents
            stmt = select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
            result = await session.execute(stmt)
            chunks = result.scalars().all()
            
            if not chunks:
                logger.warning(f"No chunks found for documents: {document_ids}")
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                if chunk.embedding:
                    # Convert embedding to list if it's not already
                    chunk_embedding = chunk.embedding if isinstance(chunk.embedding, list) else chunk.embedding.tolist()
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append({
                        'chunk_text': chunk.content,
                        'similarity': similarity,
                        'document_id': chunk.document_id,
                        'source_name': f"Document {chunk.document_id}"
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                logger.warning(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5
            
            # Avoid division by zero
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            return dot_product / (mag1 * mag2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _post_process_response(self, response: str, section_title: str) -> str:
        """Post-process the AI response to clean it up."""
        if not response:
            return "[No response generated]"
        
        # Clean up any remaining brackets
        if response.startswith("[") and response.endswith("]"):
            response = response[1:-1].strip()
        
        # Convert markdown to HTML if needed
        if "**" in response or "*" in response or "`" in response:
            response = self._convert_markdown_to_html(response)
        
        return response

    def _convert_markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML."""
        # Bold
        text = text.replace("**", "<strong>").replace("**", "</strong>")
        text = text.replace("*", "<em>").replace("*", "</em>")
        
        # Code
        text = text.replace("`", "<code>").replace("`", "</code>")
        
        # Lists
        lines = text.split('\n')
        html_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('• '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        
        if in_list:
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)

    async def generate_section(
        self,
        section_config: Dict[str, Any],
        document_ids: List[str],
        session: AsyncSession
    ) -> str:
        """Generate content for a single section."""
        try:
            section_title = section_config["section_title"]
            logger.info(f"Generating section: {section_title}")
            
            # Collect context for this section
            context = await self._collect_context_for_section(document_ids, section_config, session)
            
            if not context:
                logger.warning(f"No context found for section: {section_title}")
                return f"[No relevant information found for {section_title}]"
            
            # Build the prompt
            query = section_config.get("query", "")
            company_info = self._extract_company_info(context)
            
            # Replace placeholders in the query with actual company information
            query = self._replace_placeholders(query, company_info)
            
            business_prompt = f"""Based on the following business documents, please answer this question: {query}

{context[:100000]}  # Increased context size for comprehensive information coverage

            INSTRUCTIONS:
            - Answer the question directly using information from the documents
            - Synthesize and analyze the information rather than quoting extensively
            - Present key facts, numbers, and names in your own words when possible
            - If information is not found, say "Not available"
            - Be concise and factual
            - Focus on the specific question asked"""

            logger.info(f"Generated prompt for {section_config['section_title']}: {len(business_prompt)} characters")
            logger.info(f"Token limit for section: {section_config.get('max_tokens', 1500)} tokens")

            # Generate using Vertex AI with section-specific parameters
            try:
                response = await vertex_ai_service.generate_text(
                    prompt=business_prompt,
                    section_title=section_config['section_title'],
                    context=context[:100000],  # Increased context size for comprehensive information coverage
                    max_tokens=section_config.get("max_tokens", 1500),
                    temperature=section_config.get("temperature", 0.2)
                )

                # Debug: Write uncleaned response to file for Basic Information
                if "Basic Information" in section_config['section_title']:
                    with open("basic_info_debug_output.txt", "w") as f:
                        f.write("=== UNCLEANED RESPONSE ===\n")
                        f.write(response)
                        f.write("\n\n=== CLEANED RESPONSE ===\n")
                        cleaned_response = self._post_process_response(response, section_config['section_title'])
                        f.write(cleaned_response)
                        f.write("\n\n=== END DEBUG OUTPUT ===\n")
                    logger.info("Debug output written to basic_info_debug_output.txt")
                
                # Debug: Write response to file for Business Description
                if "Business Description" in section_config['section_title']:
                    with open("business_description_debug_output.txt", "w") as f:
                        f.write("=== BUSINESS DESCRIPTION RESPONSE ===\n")
                        f.write(response)
                        f.write("\n\n=== END BUSINESS DESCRIPTION ===\n")
                    logger.info("Business description debug output written to business_description_debug_output.txt")
                
                # Debug: Write response to file for Operational Complexity
                if "Operational Complexity" in section_config['section_title']:
                    with open("operational_complexity_output.txt", "w") as f:
                        f.write("=== OPERATIONAL COMPLEXITY RESPONSE ===\n")
                        f.write(response)
                        f.write("\n\n=== END OPERATIONAL COMPLEXITY ===\n")
                    logger.info("Operational complexity debug output written to operational_complexity_output.txt")
                
                # Post-process the response
                cleaned_response = self._post_process_response(response, section_config['section_title'])
                
                logger.info(f"Successfully generated section '{section_title}' with {len(cleaned_response)} characters")
                return cleaned_response
                
            except Exception as e:
                logger.error(f"Error generating section '{section_title}': {e}")
                return f"[Error generating content for {section_title}: {str(e)}]"
                
        except Exception as e:
            logger.error(f"Error in generate_section for '{section_config.get('section_title', 'Unknown')}': {e}")
            return f"[Error processing section: {str(e)}]"

    def generate_pdf(self, memo_data: Dict[str, Any], output_path: str) -> None:
        """Generate a PDF file from the memo data."""
        try:
            logger.info(f"Generating PDF: {output_path}")
            
            # Create the PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=self.font_name,
                fontSize=16,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=self.font_name,
                fontSize=14,
                spaceAfter=20,
                textColor=colors.darkblue
            )
            content_style = ParagraphStyle(
                'CustomContent',
                parent=styles['Normal'],
                fontName=self.font_name,
                fontSize=11,
                spaceAfter=12,
                leading=14
            )
            
            # Add title
            story.append(Paragraph("Investment Memo", title_style))
            story.append(Spacer(1, 20))
            
            # Add timestamp
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"Generated on {timestamp}", content_style))
            story.append(Spacer(1, 30))
            
            try:
                # Add content sections
                section_mappings = {
                    'basic_info': 'Basic Information',
                    'simple_description': 'Business Description', 
                    'operational_complexity': 'Operational Complexity Assessment',
                    'what_happens_operationally': 'Operational Overview'
                }
                
                for key, title in section_mappings.items():
                    content = memo_data.get(key, '')
                    if content and content != '[No content available]':
                        story.append(Paragraph(title, heading_style))
                        
                        # Process content - split into paragraphs
                        if isinstance(content, str):
                            paragraphs = content.split('\n\n')
                            for para in paragraphs:
                                para = para.strip()
                                if para:
                                    try:
                                        if para.startswith('- ') or para.startswith('• '):
                                            para = f"• {para[2:]}"
                                        story.append(Paragraph(para, content_style))
                                    except Exception as e:
                                        logger.warning(f"Error adding paragraph: {e}")
                                        # Add as plain text if paragraph fails
                                        story.append(Paragraph(para.replace('<', '&lt;').replace('>', '&gt;'), content_style))
                        else:
                            story.append(Paragraph("(No content available)", content_style))

                        story.append(Spacer(1, 12))

            except Exception as e:
                logger.error(f"Error processing content sections: {e}")
                story.append(Paragraph("Error processing memo content", content_style))

            logger.info("Added all content sections to PDF")

            try:
                # Build the PDF
                logger.info(f"Building PDF with {len(story)} elements")
                doc.build(story)
                logger.info(f"PDF generated successfully: {output_path}")

            except Exception as e:
                logger.error(f"Error building PDF: {e}")
                raise

        except Exception as e:
            logger.error(f"Critical error in PDF generation: {e}")
            logger.error(f"Memo data: {memo_data}")
            raise Exception(f"Failed to generate PDF: {str(e)}")

    async def generate_complete_memo(
        self,
        document_ids: List[str],
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """Generate a complete memo using all sections from the JSON template."""
        logger.info(f"Generating complete memo using {len(document_ids)} documents")

        if not self.template_sections:
            raise ValueError("No template sections loaded")

        memo_data = {}

        # Generate each section
        for i, section_config in enumerate(self.template_sections):
            section_title = section_config["section_title"]

            # Add a small delay between API calls to avoid rate limiting
            if i > 0:
                import asyncio
                await asyncio.sleep(2)  # 2 second delay between sections

            # Generate the section content
            section_content = await self.generate_section(section_config, document_ids, session)

            # Map sections to our expected data structure
            logger.info(f"Processing section: '{section_title}' with content length: {len(str(section_content))}")
            logger.info(f"Section content preview: {str(section_content)[:200]}...")

            if section_title == "Basic Information":
                memo_data['basic_info'] = section_content
                logger.info(f"=== BASIC INFO DEBUG ===")
                logger.info(f"Stored basic_info in memo_data: {section_content}")
                logger.info(f"basic_info type: {type(section_content)}")
                logger.info(f"basic_info length: {len(str(section_content))}")
            elif section_title == "Simple Business Description":
                memo_data['simple_description'] = section_content
            elif section_title == "Operational Complexity Assessment":
                memo_data['operational_complexity'] = section_content
            elif section_title == "What happens operationally":
                memo_data['what_happens_operationally'] = section_content
            else:
                # For any additional sections
                key = section_title.lower().replace(' ', '_')
                memo_data[key] = section_content
                logger.info(f"Mapped section '{section_title}' to key '{key}'")

        logger.info("Complete memo generated successfully")
        logger.info(f"=== FINAL MEMO DATA DEBUG ===")
        logger.info(f"Final memo_data keys: {list(memo_data.keys())}")
        logger.info(f"basic_info in final memo_data: {memo_data.get('basic_info', 'NOT FOUND')}")
        logger.info(f"basic_info type in final memo_data: {type(memo_data.get('basic_info', 'NOT FOUND'))}")
        return memo_data


# Global instance
auto_memo_generator = MemoGeneratorService() 
