import uuid
import logging
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime

# WeasyPrint imports for PDF generation - temporarily commented out for macOS dependency issues
# from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader
import tempfile

from app.models.document import DocumentChunk, Document
from app.services.vertex_ai import vertex_ai_service
from app.services.search import SearchService
from app.services.web_search import web_search_service
from app.services.financials_extractor import (
	extract_three_year_pnl_from_documents,
	extract_three_year_pnl_with_llm_from_documents,
	build_three_year_pnl_table_html,
	extract_net_working_capital_from_excel_async,
	extract_net_working_capital_from_excel_legacy,
)

# Set up logging
logger = logging.getLogger(__name__)


class MemoGeneratorService:
    """Generate investment memos using JSON template and selected documents."""

    def __init__(self):
        self.template_path = Path(__file__).parent.parent / "core" / "memo_template.json"
        self.template_sections = self._load_template()

        # WeasyPrint handles fonts through CSS in the HTML template
        logger.info("MemoGeneratorService initialized with WeasyPrint")

        # Keep a copy of the raw JSON emitted by the Company Header section
        # so later sections (e.g., SFE Bio) can reliably parse leadership data
        # even if the formatted markdown makes JSON extraction fail.
        self._last_company_header_json: Optional[str] = None
        self._last_company_header_dict: Optional[Dict[str, Any]] = None
        
        # Store financial insights extracted from Operational Complexity section
        self._financial_insights: Optional[str] = None
        # Track topics already covered to avoid repetition in later sections
        self._covered_topics: set[str] = set()

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
        company_info = {
            "name": "Not specified",
            "industry": "Not specified", 
            "location": "Not specified",
            "size": "Not specified"
        }

        # Simple pattern matching for company name
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for company name patterns
            if any(keyword in line.lower() for keyword in ['company:', 'corporation:', 'inc.', 'llc', 'ltd']):
                if ':' in line:
                    name = line.split(':', 1)[1].strip()
                    if name and name != "Not specified":
                        company_info["name"] = name
                        break

        return company_info

    def _extract_financial_insights(self, operational_complexity_response: str) -> Optional[str]:
        """Extract financial insights and balance sheet trends from Operational Complexity response.
        - FINANCIAL_INSIGHTS: paragraph level only
        - BALANCE_SHEET_TRENDS: single paragraph describing YOY trends on balance sheet
        """
        try:
            import re
            
            # Capture only the first paragraph after the FINANCIAL_INSIGHTS heading
            fin_pattern = r'^\s*###\s*FINANCIAL_INSIGHTS\s*\n+([\s\S]*?)(?:\n{2,}|^\s*###|\Z)'
            fin_match = re.search(fin_pattern, operational_complexity_response, re.IGNORECASE | re.MULTILINE)
            insights: Optional[str] = None
            if fin_match:
                insights = fin_match.group(1).strip()
                logger.info(f"Extracted financial insights: {insights[:100]}...")

            # Capture balance sheet trends (single paragraph)
            bs_pattern = r'^\s*###\s*BALANCE_SHEET_TRENDS\s*\n+([\s\S]*?)(?:\n{2,}|^\s*###|\Z)'
            bs_match = re.search(bs_pattern, operational_complexity_response, re.IGNORECASE | re.MULTILINE)
            if bs_match:
                self._balance_sheet_trends = bs_match.group(1).strip()
                logger.info("Extracted balance sheet trends paragraph from Operational Complexity response")
            else:
                self._balance_sheet_trends = None
                logger.info("No balance sheet trends paragraph found in Operational Complexity response")

            if insights:
                return insights
            logger.warning("No financial insights section found in Operational Complexity response")
            return None
                
        except Exception as e:
            logger.error(f"Error extracting financial insights: {e}")
            return None

    async def _generate_financial_insights_for_pnl(self, doc_records: List[Dict[str, str]]) -> Optional[str]:
        """Generate financial insights directly for the P&L section."""
        try:
            from pathlib import Path as _Path
            
            # Find an Excel file to analyze
            selected_path = None
            for rec in doc_records:
                for cand in [rec.get("file_path") or "", rec.get("storage_path") or ""]:
                    if cand and str(cand).lower().endswith((".xlsx", ".xls")) and _Path(cand).exists():
                        selected_path = str(_Path(cand))
                        break
                if selected_path:
                    break
            
            if not selected_path:
                logger.warning("No Excel file found for financial insights generation")
                return None
            
            # Extract monthly/quarterly data and general financial context
            import pandas as pd
            excel = pd.ExcelFile(selected_path)
            
            # Look for sheets with detailed monthly/quarterly data
            monthly_sheets = []
            for sheet_name in excel.sheet_names:
                name_lower = sheet_name.lower()
                if any(term in name_lower for term in ['monthly', 'quarterly', 'revenue', 'sales', 'pnl', 'p&l']):
                    monthly_sheets.append(sheet_name)
            
            # Also get the income statement data we already extracted
            from app.services.financials_extractor import find_income_statement_tab, sheet_to_csv_string
            
            context_data = ""
            
            # Add main income statement
            income_statement_tab = find_income_statement_tab(excel)
            if income_statement_tab:
                is_csv = sheet_to_csv_string(excel, income_statement_tab)
                context_data += f"=== MAIN INCOME STATEMENT ({income_statement_tab}) ===\n{is_csv}\n\n"
            
            # Add monthly/quarterly sheets if found
            for sheet in monthly_sheets[:3]:  # Limit to first 3 to avoid token limits
                try:
                    sheet_csv = sheet_to_csv_string(excel, sheet)
                    context_data += f"=== {sheet} ===\n{sheet_csv}\n\n"
                except Exception:
                    continue
            
            if not context_data.strip():
                logger.warning("No financial data found for insights generation")
                return None
            
            # Generate insights using LLM
            prompt = f"""Analyze the financial data below and generate a 3-4 sentence financial insights paragraph about revenue patterns and trends.

Financial Data:
{context_data}

Generate insights that cover:
- Revenue growth trends and patterns
- Seasonality (if monthly data shows patterns)
- Business reasons behind any fluctuations or trends
- Key financial performance drivers

Requirements:
- 3-4 sentences maximum
- Focus on trends, growth patterns, and what drives revenue
- If you see monthly patterns, explain seasonality and business reasons
- Use specific numbers when relevant
- Be concise and investment-focused

Return only the paragraph, no headers or formatting."""

            from app.services.vertex_ai import vertex_ai_service
            response = await vertex_ai_service.generate_text(
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            # Clean up the response
            insights = response.strip()
            if insights:
                logger.info(f"Generated financial insights: {insights[:100]}...")
                return insights
            else:
                logger.warning("Empty response from financial insights generation")
                return None
                
        except Exception as e:
            logger.error(f"Error generating financial insights for P&L: {e}")
            return None

    async def _generate_sfe_bio_with_web_search(self, company_header_data: Optional[Dict[str, Any]]) -> str:
        """Generate SFE Bio section using web search for executive biographical information."""
        try:
            if not company_header_data:
                logger.warning("No company header data provided for SFE Bio generation")
                return "Biographical information not available - no leadership data found."
            
            # Extract leadership information from company header
            leadership_info = company_header_data.get("Leadership/Executive Team", "")
            company_name = company_header_data.get("Company Name", "")
            company_location = company_header_data.get("Company Location", "")
            
            if not leadership_info:
                logger.warning("No leadership information found in company header")
                return "Biographical information not available - no leadership team identified."
            
            # Parse leadership names (handle both string and list formats)
            leadership_names = []
            if isinstance(leadership_info, list):
                leadership_names = leadership_info
            elif isinstance(leadership_info, str):
                # Split by common delimiters and clean up
                # Remove common titles and parenthetical info for cleaner search
                clean_leadership = re.sub(r'\([^)]*\)', '', leadership_info)
                leadership_names = [name.strip() for name in re.split(r'[,;]|\band\b', clean_leadership) if name.strip()]
            
            logger.info(f"Extracted leadership names: {leadership_names}")
            logger.info(f"Company context: {company_name} in {company_location}")
            
            # Focus on the first executive (typically CEO/founder)
            if not leadership_names:
                return "Biographical information not available - no executive names identified."
            
            primary_executive = leadership_names[0].strip()
            
            # Remove titles like "CEO", "President", etc. for cleaner search
            executive_name = re.sub(r'\b(CEO|President|Founder|Co-Founder|CFO|COO|CTO|VP|Vice President|Director|Manager)\b', '', primary_executive).strip()
            executive_name = re.sub(r'[,\-\(\)]', ' ', executive_name).strip()
            
            logger.info(f"Searching biographical info for: {executive_name}")
            
            # Perform web search for biographical information
            bio_data = await web_search_service.search_executive_bio(
                executive_name=executive_name,
                company_name=company_name,
                company_location=company_location
            )
            
            # Format the biographical information
            formatted_bio = web_search_service.format_bio_section(bio_data)
            
            logger.info(f"Generated SFE Bio for {executive_name}: {len(formatted_bio)} characters")
            return formatted_bio
            
        except Exception as e:
            logger.error(f"Error generating SFE Bio with web search: {e}")
            return f"Error generating biographical information: {str(e)}"

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

    def _get_search_query_for_strategy(self, section_config: Dict[str, Any]) -> str:
        """Generate appropriate search queries based on context strategy."""
        context_strategy = section_config.get("context_strategy", "general")
        section_title = section_config.get("section_title", "")

        if context_strategy == "company_info" or section_title == "Company Header":
            # Use specific search terms for company information
            return "company name leadership team executives founders CEO president location headquarters address founding date established incorporation search fund sponsor investor"
        elif context_strategy == "business_overview":
            return "business model products services what company does revenue customers market competitive advantage technology platform"
        elif context_strategy == "operational":
            return "employees headcount workforce team size total number staff contractors seasonal workers full-time part-time FTE customer service representatives offshore outsourced programming team organizational structure geographic distribution remote work operations workflow process steps locations offices cities facilities technology platform automation efficiency agents properties residential commercial hybrid model international operations"
        elif context_strategy == "financial_metrics":
            return "revenue EBITDA margin growth retention rate LTV CAC metrics performance financial data properties revenue per property commission rate savings"
        elif context_strategy == "competitive_analysis":
            return "competitors competition market share competitive advantage differentiation technology platform performance comparison industry landscape"
        else:
            # For other strategies, use the original prompt
            return section_config.get("prompt", "")

    async def _collect_context_for_section(
        self, 
        document_ids: List[str], 
        section_config: Dict[str, Any],
        session: AsyncSession,
        user_id: Optional[str] = None
    ) -> str:
        """Collect relevant context for a specific section using SearchService."""
        try:
            section_title = section_config["section_title"]
            logger.info(f"Collecting context for section: {section_title}")
            logger.info(f"Document IDs: {document_ids}")

            # Get the search query for this section based on context strategy
            query = self._get_search_query_for_strategy(section_config)
            if not query:
                logger.warning(f"No search query generated for section: {section_title}")
                return ""

            logger.info(f"Search query: {query}")
            logger.info(f"Context strategy: {section_config.get('context_strategy', 'general')}")

            # Use SearchService instead of broken similarity search
            search_service = SearchService()

            # Search for similar document chunks using SearchService
            search_results = await search_service.search_documents(
                query=query,
                document_ids=document_ids,
                top_k=50,  # Further increased to capture employee details
                user_id=user_id  # SECURITY: Filter by user_id
            )

            logger.info(f"Search returned {len(search_results)} results")

            if not search_results:
                logger.warning(f"No similar chunks found for section: {section_title}")
                return ""

            # Build context from chunks
            context_parts = []
            source_log = []

            # Debug: Log all search results
            logger.info(f"=== SEARCH RESULTS FOR {section_title} ===")
            for i, result in enumerate(search_results):
                logger.info(f"Result {i+1}: Similarity={result.similarity_score:.3f}, Doc={result.document_name}, Content={result.content[:100]}...")

            for result in search_results:
                chunk_text = result.content
                similarity = result.similarity_score
                document_id = result.document_id

                # For fallback embeddings, accept all results since similarity scores are not meaningful
                if chunk_text:
                    context_parts.append(f"RELEVANT CONTENT (Source: {result.document_name} (similarity: {similarity:.3f})): {chunk_text}")

                    source_log.append({
                        'document_id': document_id,
                        'chunk_text': chunk_text,
                        'similarity': similarity,
                        'source_name': result.document_name
                    })

            logger.info(f"After filtering, using {len(context_parts)} chunks")

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
            logger.info(f"Context length: {len(context)} characters")

            return context

        except Exception as e:
            logger.error(f"Error collecting context for section {section_config.get('section_title', 'Unknown')}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def _post_process_response(self, response: str, section_title: str, custom_fields: Optional[Dict[str, str]] = None) -> str:
        """Post-process the AI response to clean it up."""
        if not response:
            return "[No response generated]"

        # Clean up any remaining brackets
        if response.startswith("[") and response.endswith("]"):
            response = response[1:-1].strip()

        # Special handling for Company Header (JSON format)
        if section_title == "Company Header":
            return self._format_basic_info_json(response, custom_fields)

        # Special handling for "What happens operationally" - remove asterisk marks
        if section_title == "What happens operationally":
            return self._remove_asterisk_formatting(response)

        # Preserve markdown formatting for other sections
        if "**" in response or "*" in response or "`" in response or response.startswith("- ") or "- **" in response:
            response = self._preserve_markdown_formatting(response)

        return response

    def _convert_markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML and clean for PDF generation."""
        import re

        # Remove any existing HTML tags first to avoid conflicts
        text = re.sub(r'<[^>]+>', '', text)

        # Convert markdown to simple text formatting
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'<strong>(.*?)</strong>', r'\1', text)

        # Italic
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'<em>(.*?)</em>', r'\1', text)

        # Code
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'<code>(.*?)</code>', r'\1', text)

        # Clean up bullet points and lists
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                clean_lines.append(f"• {line[2:]}")
            elif line.startswith('<li>') and line.endswith('</li>'):
                clean_lines.append(f"• {line[4:-5]}")
            elif line and not line.startswith('<') and not line.endswith('>'):
                clean_lines.append(line)

        return '\n'.join(clean_lines)

    def _format_balance_sheet_for_ai(self, df) -> str:
        """Format balance sheet data to emphasize year-over-year changes for AI analysis."""
        try:
            import pandas as pd
            import re
            
            # Look for columns that contain years (2023, 2024, etc.)
            year_columns = []
            other_columns = []
            
            for col in df.columns:
                col_str = str(col)
                # Look for 4-digit years or year-like patterns
                if re.search(r'20\d{2}', col_str) or any(year in col_str.lower() for year in ['2023', '2024', 'dec', 'oct', 'sep', 'jul', 'ttm']):
                    year_columns.append(col)
                else:
                    other_columns.append(col)
            
            if len(year_columns) < 2:
                # Fallback to original format if can't identify year columns
                return df.to_string(index=False, max_rows=50)
            
            # Try to identify account names column
            account_col = None
            for col in other_columns:
                col_name = str(col).lower()
                if any(keyword in col_name for keyword in ['account', 'item', 'description', 'line']):
                    account_col = col
                    break
            
            if account_col is None and len(other_columns) > 0:
                account_col = other_columns[0]  # Use first non-year column
            
            # Format for AI interpretation
            formatted_text = "BALANCE SHEET YEAR-OVER-YEAR COMPARISON:\n"
            formatted_text += "=" * 50 + "\n\n"
            
            # Sort year columns to get consistent order
            sorted_years = sorted(year_columns, key=lambda x: str(x))
            
            # Header
            if account_col:
                formatted_text += f"{'Account':<30} "
                for year_col in sorted_years[:2]:  # Limit to 2 most recent years
                    formatted_text += f"{str(year_col):<15} "
                formatted_text += "Change\n"
                formatted_text += "-" * 80 + "\n"
                
                # Process each row
                for idx, row in df.iterrows():
                    if account_col and pd.notna(row[account_col]):
                        account_name = str(row[account_col])[:28]  # Truncate long names
                        
                        # Skip header rows or empty accounts
                        if account_name.lower() in ['account', 'item', 'description'] or not account_name.strip():
                            continue
                            
                        formatted_text += f"{account_name:<30} "
                        
                        values = []
                        for year_col in sorted_years[:2]:
                            val = row[year_col]
                            if pd.notna(val) and val != '':
                                try:
                                    # Try to convert to number
                                    if isinstance(val, (int, float)):
                                        formatted_text += f"${val:>12,.0f} "
                                        values.append(float(val))
                                    else:
                                        val_str = str(val).replace(',', '').replace('$', '')
                                        num_val = float(val_str)
                                        formatted_text += f"${num_val:>12,.0f} "
                                        values.append(num_val)
                                except (ValueError, TypeError):
                                    formatted_text += f"{str(val):>14} "
                                    values.append(None)
                            else:
                                formatted_text += f"{'N/A':>14} "
                                values.append(None)
                        
                        # Calculate change
                        if len(values) == 2 and values[0] is not None and values[1] is not None:
                            if values[0] != 0:
                                change_pct = ((values[1] - values[0]) / abs(values[0])) * 100
                                change_direction = "↑" if values[1] > values[0] else "↓" if values[1] < values[0] else "→"
                                formatted_text += f" {change_direction}{abs(change_pct):>5.1f}%"
                            else:
                                change_direction = "↑" if values[1] > 0 else "↓" if values[1] < 0 else "→"
                                formatted_text += f" {change_direction}New"
                        
                        formatted_text += "\n"
                        
                        # Limit to 20 rows to avoid overwhelming the AI
                        if idx > 20:
                            formatted_text += "... (additional rows truncated)\n"
                            break
            else:
                # Fallback format without account column
                formatted_text += df.to_string(index=False, max_rows=30)
            
            formatted_text += "\n\nKEY OBSERVATIONS FOR ANALYSIS:\n"
            formatted_text += "- Focus on significant balance sheet changes between periods\n"
            formatted_text += "- Look for trends in cash position, receivables, payables, and debt\n"
            formatted_text += "- Consider business implications of major increases/decreases\n"
            
            return formatted_text
            
        except Exception as e:
            # Fallback to original format if formatting fails
            logger.warning(f"Failed to format balance sheet for AI: {e}")
            return df.to_string(index=False, max_rows=50)

    def _format_basic_info_json(self, response: str, custom_fields: Optional[Dict[str, str]] = None) -> str:
        """Parse JSON response for Company Header and format it nicely."""
        try:
            import json
            import re

            # Clean up the response to extract JSON
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = re.sub(r'`json\s*', '', response)
            response = re.sub(r'`\s*', '', response)
            response = response.strip()

            # Try to parse as JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the original response
                logger.warning("Failed to parse Company Header as JSON, returning original response")
                return response

            # Format the data nicely
            formatted_lines = []

            # Company Name and Search Fund Name - combine on one line
            company_name = data.get("Company Name", "")
            search_fund_name = data.get("Search Fund Name", "")

            if company_name or search_fund_name:
                if company_name and search_fund_name and company_name != search_fund_name:
                    formatted_lines.append(f"**Company/Search:** {company_name}/{search_fund_name}")
                elif company_name:
                    formatted_lines.append(f"**Company/Search:** {company_name}")
                elif search_fund_name:
                    formatted_lines.append(f"**Company/Search:** {search_fund_name}")

            # Company Location
            if "Company Location" in data:
                formatted_lines.append(f"**Location:** {data['Company Location']}")

            # Leadership/Executive Team
            if "Leadership/Executive Team" in data:
                leadership = data["Leadership/Executive Team"]
                if isinstance(leadership, list) and leadership:
                    formatted_lines.append("**Leadership Team:**")
                    for member in leadership:
                        if isinstance(member, dict):
                            name = member.get("Name", "Unknown")
                            role = member.get("Role", "Unknown")
                            formatted_lines.append(f"• {name} - {role}")
                        else:
                            formatted_lines.append(f"• {member}")
                elif isinstance(leadership, str):
                    formatted_lines.append(f"**Leadership Team:** {leadership}")

            # Sponsor
            if "Sponsor" in data:
                formatted_lines.append(f"**Sponsor:** {data['Sponsor']}")

            # Discussion Date - use custom field if provided
            discussion_date = custom_fields.get('discussion_date') if custom_fields else None
            if discussion_date:
                formatted_lines.append(f"**Discussion Date:** {discussion_date}")
            elif "Discussion Date" in data:
                formatted_lines.append(f"**Discussion Date:** {data['Discussion Date']}")

            # Company Founding Date - always show even if "Not available" or "Not located in CIM"
            if "Company Founding Date" in data:
                founding_date = data["Company Founding Date"]
                if founding_date:  # Show as long as it has any value
                    formatted_lines.append(f"**Company Founded:** {founding_date}")

            # FTF Equity Size - use custom field if provided
            ftf_equity_size = custom_fields.get('ftf_equity_size') if custom_fields else None
            if ftf_equity_size:
                formatted_lines.append(f"**FTF Equity Size:** {ftf_equity_size}")
            elif "FTF Equity Size" in data:
                formatted_lines.append(f"**FTF Equity Size:** {data['FTF Equity Size']}")

            # Expected Closing - use custom field if provided
            expected_closing = custom_fields.get('expected_closing') if custom_fields else None
            if expected_closing:
                formatted_lines.append(f"**Expected Closing:** {expected_closing}")
            elif "Expected Closing" in data:
                formatted_lines.append(f"**Expected Closing:** {data['Expected Closing']}")

            # Add any other fields that might be present
            other_fields = {k: v for k, v in data.items() if k not in [
                "Company Name", "Search Fund Name", "Company Location", "Leadership/Executive Team", 
                "Sponsor", "Discussion Date", "Company Founding Date", "FTF Equity Size", "Expected Closing"
            ]}

            for key, value in other_fields.items():
                if value:  # Show all fields that have any value, including "Not available" variants
                    formatted_lines.append(f"**{key}:** {value}")

            # Add extra spacing between each item to ensure separate paragraphs in PDF
            return '\n\n'.join(formatted_lines) + '\n'

        except Exception as e:
            logger.error(f"Error formatting Company Header JSON: {e}")
            return response

    def _preserve_markdown_formatting(self, text: str) -> str:
        """Preserve markdown formatting including bullet points for PDF generation.
        Critical: do NOT strip leading spaces so nested bullets remain indented.
        """
        import re

        lines = text.split('\n')
        formatted_lines = []

        for raw_line in lines:
            if raw_line.strip() == '':
                formatted_lines.append('')
                continue

            # Keep original indentation for detection
            stripped = raw_line.lstrip()
            indent_len = len(raw_line) - len(stripped)

            # Preserve bullet points with various formats
            if stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('* '):
                formatted_lines.append(raw_line)
            elif indent_len >= 2 and (stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('◦ ') or stripped.startswith('* ')):
                # Indented sub-bullet
                formatted_lines.append(raw_line)
            elif re.match(r'^[-*]\s+\*\*.*?\*\*:', stripped):
                formatted_lines.append(raw_line)
            else:
                # Preserve other formatting as-is
                formatted_lines.append(raw_line)

        result = '\n'.join(formatted_lines)

        # Clean up any problematic characters for PDF generation
        result = result.replace('<', '&lt;').replace('>', '&gt;')

        return result

    def _remove_asterisk_formatting(self, text: str) -> str:
        """Remove asterisk formatting marks from text for the operationally section."""
        import re

        # Remove double asterisks used for bold formatting (e.g., **1. Customer** -> 1. Customer)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

        # Remove any remaining single asterisks that might be random marks
        # But preserve bullet points if they start with asterisks
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Don't remove asterisks that are at the start of a line (bullet points)
            if line.strip().startswith('* '):
                cleaned_lines.append(line)
            else:
                # Remove any remaining asterisks that aren't at the start of lines
                # This handles cases like "some text * random mark" -> "some text random mark"
                cleaned_line = re.sub(r'(?<!^)\s*\*\s*(?![a-zA-Z])', '', line)
                cleaned_lines.append(cleaned_line)

        result = '\n'.join(cleaned_lines)

        # Clean up any extra spaces that might have been created, but preserve line structure
        # Only clean up multiple spaces within lines, not line breaks
        lines = result.split('\n')
        cleaned_lines = []
        for line in lines:
            # Clean up multiple spaces within each line, but preserve the line
            cleaned_line = re.sub(r' +', ' ', line.strip())
            cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)

    def _add_formatted_content_to_pdf(self, content: str, story: list, content_style, heading_style) -> None:
        """Add formatted content to PDF story, handling markdown and bullet points."""
        try:
            import re
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib import colors

            # Create a bullet point style - NO PARENT to avoid font override
            bullet_style = ParagraphStyle(
                'BulletStyle',
                fontName=self.font_name,
                leftIndent=20,
                bulletIndent=0,
                fontSize=11,
                spaceAfter=6
            )

            # Create a sub-bullet style (for indented bullets) - NO PARENT to avoid font override
            sub_bullet_style = ParagraphStyle(
                'SubBulletStyle',
                fontName=self.font_name,
                leftIndent=40,
                bulletIndent=20,
                fontSize=11,
                spaceAfter=6
            )

            lines = content.split('\n')
            current_paragraph = []

            for raw_line in lines:
                # Preserve indentation for nested bullet detection
                stripped = raw_line.lstrip()
                indent_len = len(raw_line) - len(stripped)

                if stripped == '':
                    # Empty line - process any accumulated paragraph
                    if current_paragraph:
                        # Each line becomes its own paragraph instead of joining with spaces
                        for para_line in current_paragraph:
                            if para_line.strip():
                                try:
                                    story.append(Paragraph(self._format_text_for_pdf(para_line), content_style))
                                except Exception as e:
                                    logger.warning(f"Error adding paragraph: {e}")
                                    story.append(Paragraph(para_line.replace('<', '&lt;').replace('>', '&gt;'), content_style))
                        current_paragraph = []
                    story.append(Spacer(1, 6))  # Add some space for empty lines
                    continue

                # Handle bullet points
                if stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('* '):
                    # Process any accumulated paragraph first
                    if current_paragraph:
                        for para_line in current_paragraph:
                            if para_line.strip():
                                try:
                                    story.append(Paragraph(self._format_text_for_pdf(para_line), content_style))
                                except Exception as e:
                                    story.append(Paragraph(para_line.replace('<', '&lt;').replace('>', '&gt;'), content_style))
                        current_paragraph = []

                    # Add bullet point
                    bullet_text = stripped[2:].strip()
                    try:
                        story.append(Paragraph(f"• {self._format_text_for_pdf(bullet_text)}", bullet_style))
                    except Exception as e:
                        logger.warning(f"Error adding bullet: {e}")
                        story.append(Paragraph(f"• {bullet_text.replace('<', '&lt;').replace('>', '&gt;')}", bullet_style))

                elif indent_len >= 2 and (stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('◦ ') or stripped.startswith('* ')):
                    # Handle indented bullet points (sub-bullets)
                    if current_paragraph:
                        for para_line in current_paragraph:
                            if para_line.strip():
                                try:
                                    story.append(Paragraph(self._format_text_for_pdf(para_line), content_style))
                                except Exception as e:
                                    story.append(Paragraph(para_line.replace('<', '&lt;').replace('>', '&gt;'), content_style))
                        current_paragraph = []

                    # Extract bullet text based on the bullet character used
                    if stripped.startswith('◦ '):
                        bullet_text = stripped[2:].strip()
                    elif stripped.startswith('• ') or stripped.startswith('- ') or stripped.startswith('* '):
                        bullet_text = stripped[2:].strip()
                    else:
                        bullet_text = stripped

                    try:
                        story.append(Paragraph(f"◦ {self._format_text_for_pdf(bullet_text)}", sub_bullet_style))
                    except Exception as e:
                        logger.warning(f"Error adding sub-bullet: {e}")
                        story.append(Paragraph(f"◦ {bullet_text.replace('<', '&lt;').replace('>', '&gt;')}", sub_bullet_style))

                else:
                    # Regular text - preserve line structure for Company Header formatting
                    current_paragraph.append(raw_line)

            # Process any remaining paragraph
            if current_paragraph:
                for para_line in current_paragraph:
                    if para_line.strip():
                        try:
                            story.append(Paragraph(self._format_text_for_pdf(para_line), content_style))
                        except Exception as e:
                            logger.warning(f"Error adding final paragraph: {e}")
                            story.append(Paragraph(para_line.replace('<', '&lt;').replace('>', '&gt;'), content_style))

        except Exception as e:
            logger.error(f"Error formatting content for PDF: {e}")
            # Fallback: add as simple text
            story.append(Paragraph(content.replace('<', '&lt;').replace('>', '&gt;'), content_style))

    def _convert_content_to_html(self, content: str) -> str:
        """Convert markdown-style content to HTML with proper nested bullet support."""
        if not content:
            return ""

        lines = content.split('\n')
        html_lines = []
        list_stack = []  # Track nested list levels
        
        for raw_line in lines:
            stripped = raw_line.lstrip()
            indent_len = len(raw_line) - len(stripped)

            if stripped == '':
                # Close all open lists on empty line
                while list_stack:
                    level_info = list_stack.pop()
                    if level_info['is_nested']:
                        html_lines.append('</ul></li>')
                    else:
                        html_lines.append('</li></ul>')
                html_lines.append('<br>')
                continue

            # Handle bullet points
            if stripped.startswith('• ') or stripped.startswith('- ') or stripped.startswith('* ') or stripped.startswith('◦ '):
                bullet_char = '◦' if stripped.startswith('◦ ') else '•'
                bullet_text = stripped[2:].strip()
                
                # Determine if this is a main bullet (indent=0) or sub-bullet (indent>=2)
                is_sub_bullet = indent_len >= 2 or stripped.startswith('◦ ')
                
                if is_sub_bullet:
                    # This is a sub-bullet
                    # If no parent list exists, create one
                    if not list_stack:
                        html_lines.append('<ul><li>Parent Item<ul>')
                        list_stack.append({'indent': 0, 'is_nested': False})
                        list_stack.append({'indent': indent_len, 'is_nested': True})
                    elif not any(level['is_nested'] for level in list_stack):
                        # We have a main list but no nested list yet
                        html_lines.append('<ul>')
                        list_stack.append({'indent': indent_len, 'is_nested': True})
                    
                    html_lines.append(f'<li>{self._escape_html(bullet_text)}</li>')
                    
                else:
                    # This is a main bullet
                    # Close any existing nested lists
                    while list_stack and list_stack[-1]['is_nested']:
                        list_stack.pop()
                        html_lines.append('</ul>')
                    
                    # Close previous main bullet if exists
                    if list_stack and not list_stack[-1]['is_nested']:
                        html_lines.append('</li>')
                    
                    # Start main list if needed
                    if not list_stack:
                        html_lines.append('<ul>')
                        list_stack.append({'indent': 0, 'is_nested': False})
                    
                    html_lines.append(f'<li>{self._escape_html(bullet_text)}')
                    # Note: leave <li> open for potential sub-bullets

            else:
                # Regular text - close all open lists
                while list_stack:
                    level_info = list_stack.pop()
                    if level_info['is_nested']:
                        html_lines.append('</ul></li>')
                    else:
                        html_lines.append('</li></ul>')
                
                html_lines.append(f'<p>{self._escape_html(stripped)}</p>')

        # Close any remaining open lists
        while list_stack:
            level_info = list_stack.pop()
            if level_info['is_nested']:
                html_lines.append('</ul></li>')
            else:
                html_lines.append('</li></ul>')

        return '\n'.join(html_lines)

    async def _generate_balance_sheet_trends_with_llm(self, balance_sheet_data: Dict[str, Any], context: str) -> Optional[str]:
        """Generate balance sheet trends using LLM with balance sheet data and CIM context."""
        try:
            from app.services.vertex_ai import vertex_ai_service
            
            # Check if balance_sheet_data is valid
            if not balance_sheet_data or not isinstance(balance_sheet_data, dict):
                logger.warning(f"Invalid balance sheet data for trends generation: {type(balance_sheet_data)}")
                return None
            
            # Extract key metrics for LLM analysis
            assets = balance_sheet_data.get("current_assets", {}) or {}
            liabs = balance_sheet_data.get("current_liabilities", {}) or {}
            nwc = balance_sheet_data.get("net_working_capital", {}) or {}
            year_labels = balance_sheet_data.get("year_headers", {}) or {"2023": "2023", "2024": "2024"}
            
            def _coerce_number(value: Any) -> Optional[float]:
                """Coerce various LLM/CSV value shapes into a float if possible.
                Accepts numbers, numeric strings (with $, commas), and single-item lists.
                Returns None if not parseable.
                """
                try:
                    if value is None:
                        return None
                    if isinstance(value, (int, float)):
                        return float(value)
                    if isinstance(value, list) and value:
                        return _coerce_number(value[0])
                    s = str(value).strip()
                    if not s or s.lower() in {"nan", "none", "null", "-"}:
                        return None
                    # Remove currency symbols and commas
                    s = s.replace("$", "").replace(",", "")
                    return float(s)
                except Exception:
                    return None

            # Build structured data summary for LLM
            data_summary = []
            logger.info(f"Balance sheet sections for LLM: assets={assets}, liabs={liabs}, nwc={nwc}")
            
            # Handle Current Assets and Current Liabilities where items are dicts with year keys
            for section_name, section_data in [("Current Assets", assets), ("Current Liabilities", liabs)]:
                logger.info(f"Processing section '{section_name}' with {len(section_data) if section_data else 0} items")
                if not section_data:
                    continue
                for item_name, item_values in section_data.items():
                    logger.info(f"  Item '{item_name}': {item_values}")
                    if not isinstance(item_values, dict):
                        logger.warning(f"  Item '{item_name}' is not a dict: {type(item_values)}")
                        continue
                    v23 = _coerce_number(item_values.get("2023"))
                    v24 = _coerce_number(item_values.get("2024"))
                    logger.info(f"  Values for {item_name}: 2023={v23}, 2024={v24}")
                    if isinstance(v23, (int, float)) and isinstance(v24, (int, float)):
                        pct_change = ((v24 - v23) / v23 * 100) if v23 != 0 else 0
                        data_summary.append(f"{item_name}: {year_labels.get('2023', '2023')} ${v23:,.0f} → {year_labels.get('2024', '2024')} ${v24:,.0f} ({pct_change:+.0f}%)")
                    elif isinstance(v23, (int, float)) or isinstance(v24, (int, float)):
                        # If only one year available, still include directional/value-only note
                        val_str_23 = f"${v23:,.0f}" if isinstance(v23, (int, float)) else "N/A"
                        val_str_24 = f"${v24:,.0f}" if isinstance(v24, (int, float)) else "N/A"
                        data_summary.append(f"{item_name}: {year_labels.get('2023', '2023')} {val_str_23} → {year_labels.get('2024', '2024')} {val_str_24}")

            # Special handling for Net Working Capital which may be a flat dict {"2023": x, "2024": y}
            if isinstance(nwc, dict):
                nwc_23 = _coerce_number(nwc.get("2023"))
                nwc_24 = _coerce_number(nwc.get("2024"))
                if isinstance(nwc_23, (int, float)) or isinstance(nwc_24, (int, float)):
                    if isinstance(nwc_23, (int, float)) and isinstance(nwc_24, (int, float)):
                        pct_change = ((nwc_24 - nwc_23) / nwc_23 * 100) if nwc_23 != 0 else 0
                        data_summary.append(f"Net Working Capital: {year_labels.get('2023', '2023')} ${nwc_23:,.0f} → {year_labels.get('2024', '2024')} ${nwc_24:,.0f} ({pct_change:+.0f}%)")
                    else:
                        val_str_23 = f"${nwc_23:,.0f}" if isinstance(nwc_23, (int, float)) else "N/A"
                        val_str_24 = f"${nwc_24:,.0f}" if isinstance(nwc_24, (int, float)) else "N/A"
                        data_summary.append(f"Net Working Capital: {year_labels.get('2023', '2023')} {val_str_23} → {year_labels.get('2024', '2024')} {val_str_24}")
            
            logger.info(f"Balance sheet data summary for LLM: {data_summary}")

            if not data_summary:
                return None
                
            prompt = f"""Analyze the balance sheet trends and provide a single concise paragraph (2-3 sentences) focusing on key directional changes and business implications. If the CIM context is empty or not relevant, base your analysis solely on the Balance Sheet Data below.

Balance Sheet Data:
{chr(10).join(data_summary)}

CIM Context (use if relevant to balance sheet changes):
{context[:2000]}

Generate a natural, analytical paragraph about the balance sheet trends. Focus on:
- Key directional changes in working capital components
- Business implications of these changes
- Reference CIM explanations if they relate to balance sheet movements
- Keep it concise and investment-focused

Do not use template language like "Balance sheet trends (Year vs Year):". Write as natural commentary."""

            response = await vertex_ai_service.generate_text(
                prompt=prompt,
                section_title="Balance Sheet Trends Analysis",
                max_tokens=300,
                temperature=0.3
            )
            
            return response.strip() if response and response.strip() else None
            
        except Exception as e:
            import traceback
            logger.warning(f"Failed to generate LLM balance sheet trends: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            logger.warning(f"Balance sheet data type: {type(balance_sheet_data)}, keys: {list(balance_sheet_data.keys()) if isinstance(balance_sheet_data, dict) else 'N/A'}")
            return None

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text."""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

    def _format_text_for_pdf(self, text: str) -> str:
        """Format text with basic markdown for PDF generation."""
        import re

        # Convert **bold** to <b>bold</b> (double asterisks are clear markdown intent)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        # Handle single asterisks at the beginning of lines (bullet-style formatting)
        # Convert "* Text:" to "Text:" and make it bold
        text = re.sub(r'^\*\s*([^:]+):', r'<b>\1:</b>', text, flags=re.MULTILINE)

        # Handle nested asterisks (like "* **Item:** content")
        text = re.sub(r'^\*\s*\*\*([^:]+)\*\*:', r'<b>\1:</b>', text, flags=re.MULTILINE)

        # Clean up any remaining single asterisks that are clearly bullet points
        text = re.sub(r'^\*\s+', '', text, flags=re.MULTILINE)

        # Escape any remaining problematic characters
        text = text.replace('<', '&lt;').replace('>', '&gt;')

        # Restore the formatting tags we just added
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')

        return text

    async def generate_section(
        self,
        section_config: Dict[str, Any],
        document_ids: List[str],
        session: AsyncSession,
        custom_fields: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        company_header_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate content for a single section."""
        try:
            section_title = section_config["section_title"]
            logger.info(f"Generating section: {section_title}")

            # Special handling: Three year historical P&L section → insert HTML table only
            section_lower = section_title.strip().lower()
            logger.info(f"Checking section title for P&L match: '{section_title}' -> '{section_lower}'")
            
            # Multiple ways to match the financial section
            is_pnl_section = (
                "three year historical p&l" in section_lower or 
                "three year historical p" in section_lower or
                "historical p&l" in section_lower or
                section_title == "Three year historical P&L and Balance Sheet, updated to within the last 45 days"
            )
            
            logger.info(f"P&L section match result: {is_pnl_section}")
            
            if is_pnl_section:
                try:
                    # Fetch file paths for provided document IDs
                    stmt = select(Document).where(Document.id.in_(document_ids))
                    result = await session.execute(stmt)
                    docs = result.scalars().all()
                    doc_records = [{
                        "id": d.id,
                        "name": d.name,
                        "file_path": d.file_path or "",
                        "storage_path": getattr(d, "storage_path", "") or "",
                        "content_type": d.content_type or "",
                    } for d in docs]

                    logger.info(f"Financial extraction: Found {len(doc_records)} documents")
                    for i, doc in enumerate(doc_records):
                        logger.info(f"  Doc {i}: name='{doc['name']}', file_path='{doc['file_path']}', content_type='{doc['content_type']}'")

                    # Use new LLM-based P&L extraction
                    pnl_html = await extract_three_year_pnl_with_llm_from_documents(doc_records)
                    logger.info(f"LLM P&L extraction completed")
                    
                    # Do NOT generate insights here. We will inject the paragraph after
                    # the Operational Complexity section runs (post-processing step).
                    # This ensures insights come from CIM context, not just Excel.

                    # Extract current assets from balance sheet using Gemini
                    current_assets_html = ""
                    auto_bs_blurb = None  # Initialize to ensure it's available later
                    try:
                        # Find Excel file for balance sheet analysis
                        logger.info(f"Looking for Excel files in {len(doc_records)} documents")
                        # Consider BOTH file_path and storage_path when identifying Excel docs
                        excel_docs = []
                        for d in doc_records:
                            candidates = [(d.get("file_path") or ""), (d.get("storage_path") or "")]
                            if any(p.lower().endswith((".xlsx", ".xls")) for p in candidates if p):
                                excel_docs.append(d)
                        logger.info(f"Found {len(excel_docs)} Excel documents (considering file_path and storage_path)")

                        from pathlib import Path
                        selected_path = None
                        selected_doc = None
                        for ex in excel_docs:
                            file_path = ex.get("file_path") or ""
                            storage_path = ex.get("storage_path") or ""
                            logger.info(f"Examining Excel doc - name: {ex.get('name')}, file_path: {file_path}, storage_path: {storage_path}")

                            candidate_paths = []
                            if file_path:
                                candidate_paths.append(file_path)
                            if storage_path and storage_path not in candidate_paths:
                                candidate_paths.append(storage_path)
                            logger.info(f"Checking candidate paths: {candidate_paths}")

                            for p in candidate_paths:
                                try_path = Path(p)
                                exists = try_path.exists()
                                logger.info(f"Checking path: {try_path} - exists: {exists}")
                                if exists:
                                    selected_path = str(try_path)
                                    selected_doc = ex
                                    break
                            if selected_path:
                                break

                        if not selected_path:
                            # Fallback: attempt to download from GCS if storage_path exists (prod case)
                            try:
                                from app.services.storage import get_storage_service
                                storage_selected = None
                                for ex in excel_docs:
                                    storage_path = ex.get("storage_path") or ""
                                    if not storage_path:
                                        continue
                                    try:
                                        storage_service = get_storage_service()
                                        if hasattr(storage_service, "is_available") and storage_service.is_available():
                                            logger.info(f"Attempting GCS download for storage_path: {storage_path}")
                                            file_bytes = await storage_service.download_file(storage_path)
                                            import tempfile
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                                                tmp.write(file_bytes)
                                                storage_selected = tmp.name
                                                selected_doc = ex
                                                logger.info(f"Downloaded Excel from GCS to temp path: {storage_selected}")
                                                break
                                    except Exception as gcs_err:
                                        logger.warning(f"Failed GCS download for '{storage_path}': {gcs_err}")
                                if storage_selected:
                                    selected_path = storage_selected
                            except Exception as storage_init_err:
                                logger.warning(f"Storage fallback unavailable: {storage_init_err}")

                        if selected_path:
                            logger.info(f"Using selected Excel path: {selected_path} (doc: {selected_doc.get('name') if selected_doc else 'unknown'})")
                            # Try extracting structured balance sheet data to build both table and trends blurb
                            auto_bs_blurb = None
                            try:
                                import pandas as pd
                                from app.services.financials_extractor import (
                                    find_balance_sheet_tab,
                                    sheet_to_csv_string,
                                    extract_current_assets_with_gemini,
                                    build_current_assets_table_html,
                                )
                                excel = pd.ExcelFile(selected_path)
                                bs_tab = find_balance_sheet_tab(excel)
                                if bs_tab:
                                    csv_text = sheet_to_csv_string(excel, bs_tab)
                                    if csv_text.strip():
                                        bs_data = await extract_current_assets_with_gemini(csv_text)
                                        logger.info(f"Balance sheet data extraction result: {type(bs_data)}, keys: {list(bs_data.keys()) if isinstance(bs_data, dict) else 'N/A'}")
                                        
                                        # If NWC missing from LLM extraction, add it from legacy extraction for completeness
                                        if isinstance(bs_data, dict):
                                            nwc_from_llm = bs_data.get("net_working_capital", {})
                                            if not nwc_from_llm or (nwc_from_llm.get("2023") is None and nwc_from_llm.get("2024") is None):
                                                logger.info("NWC missing from LLM extraction, adding from legacy extraction")
                                                legacy_nwc = extract_net_working_capital_from_excel_legacy(selected_path)
                                                if isinstance(legacy_nwc, dict):
                                                    nwc_adjusted = legacy_nwc.get("nwc_adjusted", {})
                                                    if nwc_adjusted and (nwc_adjusted.get("2023") is not None or nwc_adjusted.get("2024") is not None):
                                                        bs_data["net_working_capital"] = nwc_adjusted
                                                        logger.info(f"Added NWC from legacy extraction: {nwc_adjusted}")
                                        
                                        current_assets_html = build_current_assets_table_html(bs_data, sheet_name=bs_tab)
                                        # Get collected context for this section for balance sheet analysis
                                        context_for_bs = await self._collect_context_for_section(document_ids, {"section_title": "Balance Sheet Analysis", "context_strategy": "general"}, session)
                                        logger.info(f"Balance sheet context length: {len(context_for_bs) if context_for_bs else 0}")
                                        auto_bs_blurb = await self._generate_balance_sheet_trends_with_llm(bs_data, context_for_bs or "")
                                        logger.info(f"Auto-generated balance sheet blurb: {auto_bs_blurb}")
                                        # Store as instance variable for post-processing
                                        if auto_bs_blurb and auto_bs_blurb.strip():
                                            self._auto_balance_sheet_blurb = auto_bs_blurb
                                            logger.info("Stored auto-generated balance sheet blurb as instance variable")
                                    else:
                                        current_assets_html = '<p><strong>Failed to convert balance sheet to CSV</strong></p>'
                                else:
                                    current_assets_html = '<p><strong>Balance sheet not found</strong></p>'
                            except Exception as _auto_bs_err:
                                logger.warning(f"Auto balance sheet trends failed, falling back to legacy HTML: {_auto_bs_err}")
                                current_assets_html = await extract_net_working_capital_from_excel_async(selected_path)
                        else:
                            logger.warning("No valid Excel file path found across all candidate documents (local or GCS)")
                            current_assets_html = '<p><strong>Excel file not found for balance sheet analysis</strong></p>'
                    except Exception as e:
                        logger.error(f"Failed to extract current assets: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        current_assets_html = '<p><strong>Error extracting current assets from balance sheet</strong></p>'

                    # Combine financial insights (if available) with tables
                    final_content = ""
                    
                    # Add financial insights above the table if available
                    logger.info(f"Financial insights check: self._financial_insights={bool(self._financial_insights)}, content='{getattr(self, '_financial_insights', None)}'")
                    if self._financial_insights and "Not located in CIM" not in self._financial_insights:
                        final_content += f'<div style="margin-bottom: 16px;"><p>{self._financial_insights}</p></div>\n'
                        logger.info("Added financial insights above P&L table")
                    else:
                        # Explicitly show not-found message; do not auto-generate
                        final_content += '<div style="margin-bottom: 8px;"><p class="meta">Financial insights not found</p></div>\n'
                        logger.info(f"Financial insights not found - using fallback message")

                    # Add P&L table first
                    final_content += pnl_html

                    # Balance sheet trends will be injected during post-processing
                    # This ensures single injection and consistent handling with operational complexity trends

                    # Add balance sheet table
                    final_content += '\n<div style="height:16px"></div>\n' + current_assets_html
                    
                    return final_content
                except Exception as fin_err:
                    logger.error(f"Failed to build Three year historical P&L table: {fin_err}")
                    logger.exception("Full traceback:")
                    # Return empty but explicit placeholder table
                    empty_html = build_three_year_pnl_table_html(
                        {"2022": None, "2023": None, "2024": None},
                        {"2022": None, "2023": None, "2024": None},
                        {}
                    )
                    return empty_html

            # Handle SFE Bio section - show placeholder message
            if section_title == "SFE Bio":
                logger.info("SFE Bio section - showing placeholder message")
                return '<p style="color: red; font-weight: bold;">User should fill in here</p>'
            
            # Simple OFCF section implementation
            if section_title.strip().lower().startswith("ofcf"):
                try:
                    # Collect financials from already ingested docs
                    stmt = select(Document).where(Document.id.in_(document_ids))
                    result = await session.execute(stmt)
                    docs = result.scalars().all()
                    doc_records = [{
                        "id": d.id,
                        "name": d.name,
                        "file_path": d.file_path or "",
                        "storage_path": getattr(d, "storage_path", "") or "",
                        "content_type": d.content_type or "",
                    } for d in docs]

                    # Use new LLM-based P&L extraction to get structured data for OFCF calculation
                    ebitda_2024 = None
                    try:
                        from pathlib import Path as _Path
                        selected_path = None
                        for rec in doc_records:
                            for cand in [rec.get("file_path") or "", rec.get("storage_path") or ""]:
                                if cand and str(cand).lower().endswith((".xlsx", ".xls")) and _Path(cand).exists():
                                    selected_path = str(_Path(cand))
                                    break
                            if selected_path:
                                break

                        logger.info(f"OFCF DEBUG: Excel file path found: {selected_path}")
                        if selected_path:
                            # Extract P&L data using LLM for EBITDA
                            import pandas as pd
                            excel = pd.ExcelFile(selected_path)
                            from app.services.financials_extractor import find_income_statement_tab, sheet_to_csv_string, extract_pnl_with_gemini
                            
                            income_statement_tab = find_income_statement_tab(excel)
                            logger.info(f"OFCF DEBUG: Income statement tab found: {income_statement_tab}")
                            if income_statement_tab:
                                csv_content = sheet_to_csv_string(excel, income_statement_tab)
                                logger.info(f"OFCF DEBUG: CSV content length: {len(csv_content) if csv_content else 0}")
                                if csv_content.strip():
                                    pnl_data = await extract_pnl_with_gemini(csv_content)
                                    logger.info(f"OFCF DEBUG: PnL data extracted: {pnl_data}")
                                    # Try multiple possible EBITDA keys
                                    ebitda_2024 = (pnl_data.get("adjusted_ebitda", {}).get("2024") or 
                                                 pnl_data.get("ebitda", {}).get("2024") or
                                                 pnl_data.get("EBITDA", {}).get("2024"))
                                    logger.info(f"OFCF DEBUG: EBITDA 2024 value: {ebitda_2024}")
                                    logger.info(f"OFCF DEBUG: Available EBITDA keys: {list(pnl_data.keys())}")
                        else:
                            logger.warning("OFCF DEBUG: No Excel file found")
                    except Exception as pnl_err:
                        logger.error(f"Failed to extract EBITDA for OFCF: {pnl_err}")
                        import traceback
                        logger.error(f"OFCF EBITDA stack trace: {traceback.format_exc()}")

                    # Attempt to get NWC from balance sheet Excel if present
                    nwc_2023 = None
                    nwc_2024 = None
                    try:
                        from pathlib import Path as _Path
                        selected_path = None
                        for rec in doc_records:
                            for cand in [rec.get("file_path") or "", rec.get("storage_path") or ""]:
                                if cand and str(cand).lower().endswith((".xlsx", ".xls")) and _Path(cand).exists():
                                    selected_path = str(_Path(cand))
                                    break
                            if selected_path:
                                break
                        logger.info(f"OFCF DEBUG: NWC Excel file path found: {selected_path}")
                        if selected_path:
                            nwc_dict = extract_net_working_capital_from_excel_legacy(selected_path)
                            logger.info(f"OFCF DEBUG: NWC dict extracted: {nwc_dict}")
                            adj = nwc_dict.get("nwc_adjusted", {}) if isinstance(nwc_dict, dict) else {}
                            logger.info(f"OFCF DEBUG: NWC adjusted dict: {adj}")
                            nwc_2023 = adj.get("2023")
                            nwc_2024 = adj.get("2024")
                            logger.info(f"OFCF DEBUG: NWC 2023 value: {nwc_2023}")
                            logger.info(f"OFCF DEBUG: NWC 2024 value: {nwc_2024}")
                        else:
                            logger.warning("OFCF DEBUG: No Excel file found for NWC")
                    except Exception as _nwc_err:
                        logger.error(f"NWC extraction for OFCF failed: {_nwc_err}")
                        import traceback
                        logger.error(f"OFCF NWC stack trace: {traceback.format_exc()}")
                    logger.info(f"OFCF DEBUG: Final values - EBITDA 2024: {ebitda_2024}, NWC 2023: {nwc_2023}, NWC 2024: {nwc_2024}")
                    if ebitda_2024 is not None and nwc_2023 is not None and nwc_2024 is not None:
                        ofcf = ebitda_2024 - (nwc_2024 - nwc_2023)
                        logger.info(f"OFCF DEBUG: Calculated OFCF: {ofcf} = {ebitda_2024} - ({nwc_2024} - {nwc_2023})")
                        return f"The OFCF is ${ofcf:,.0f}."
                    else:
                        logger.error(f"OFCF DEBUG: Missing values - EBITDA 2024: {ebitda_2024}, NWC 2023: {nwc_2023}, NWC 2024: {nwc_2024}")
                        return "OFCF could not be determined."
                except Exception as ofcf_err:
                    logger.error(f"Failed to generate OFCF section: {ofcf_err}")
                    return "OFCF could not be determined."

            # Check if this is a blank section (empty prompt)
            query = section_config.get("prompt", "")
            if not query or query.strip() == "":
                logger.info(f"Blank section detected: {section_title} - returning placeholder")
                return f"[{section_title} - To be completed]"

            # Collect context for this section
            context = await self._collect_context_for_section(document_ids, section_config, session, user_id)

            if not context:
                logger.warning(f"No context found for section: {section_title}")
                return f"[No relevant information found for {section_title}]"

            # Special handling: Add balance sheet data to Operational Complexity context
            if "Operational Complexity" in section_config['section_title']:
                try:
                    # Get document records for balance sheet extraction
                    stmt = select(Document).where(Document.id.in_(document_ids))
                    result = await session.execute(stmt)
                    docs = result.scalars().all()
                    doc_records = [{
                        "id": d.id,
                        "name": d.name,
                        "file_path": d.file_path or "",
                        "storage_path": getattr(d, "storage_path", "") or "",
                        "content_type": d.content_type or "",
                    } for d in docs]

                    # Find Excel file and extract balance sheet data
                    from pathlib import Path
                    selected_path = None
                    for doc in doc_records:
                        candidates = [(doc.get("file_path") or ""), (doc.get("storage_path") or "")]
                        for path in candidates:
                            if path and path.lower().endswith((".xlsx", ".xls")) and Path(path).exists():
                                selected_path = path
                                break
                        if selected_path:
                            break

                    if selected_path:
                        # Extract raw balance sheet data for AI analysis
                        import pandas as pd
                        excel_file = pd.ExcelFile(selected_path)
                        
                        # Look for balance sheet tab
                        balance_sheet_tab = None
                        for sheet_name in excel_file.sheet_names:
                            name_lower = sheet_name.lower()
                            if "balance" in name_lower or "bs" in name_lower or "sheet" in name_lower:
                                balance_sheet_tab = sheet_name
                                break
                        
                        if balance_sheet_tab:
                            # Read balance sheet data
                            df = pd.read_excel(selected_path, sheet_name=balance_sheet_tab)
                            
                            # Convert to AI-friendly format emphasizing YoY changes
                            balance_sheet_text = f"\n\n=== BALANCE SHEET DATA FROM {balance_sheet_tab} ===\n"
                            balance_sheet_text += self._format_balance_sheet_for_ai(df)
                            balance_sheet_text += "\n=== END BALANCE SHEET DATA ===\n"
                            
                            context += balance_sheet_text
                            logger.info(f"Added balance sheet data from {balance_sheet_tab} to Operational Complexity context")
                        else:
                            logger.info("No balance sheet tab found in Excel file")
                    else:
                        logger.info("No Excel file found for balance sheet data extraction")
                except Exception as e:
                    logger.warning(f"Failed to add balance sheet data to Operational Complexity context: {e}")
                    # Continue without balance sheet data

            # Build the prompt
            company_info = self._extract_company_info(context)

            business_prompt = f"""Based on the following business documents, please answer this question: {query}

{context[:100000]}  # Increased context size for comprehensive information coverage

            INSTRUCTIONS:
            - Answer the question directly using information from the documents
            - Be concise, punchy, and direct while maintaining thoroughness
            - Focus on key insights and critical information - avoid unnecessary elaboration
            - Focus ONLY on the target company - DO NOT include information about competitors, other companies, or industry comparisons in factual sections
            - CRITICAL: When extracting employee information, prioritize specific headcounts, organizational structure, and geographic distribution of workforce
            - For factual sections (employees, locations, product map, operational steps), include EXACT numbers, specific city names, headcounts, percentages, and concrete details as they appear in the documents
            - For analytical sections, synthesize and analyze the information rather than quoting extensively
            - Preserve specific data points: employee counts, office locations, geographic areas, automation percentages, etc.
            - Each paragraph should be focused and impactful, not exhaustive
            - If information is not found, say "Not located in CIM"
            - IMPORTANT: Do not restate the request or include any introductory phrases or contextual statements such as 'based on the documents provided ...'. Omit them from your response. Simply answer the question."""

            logger.info(f"Generated prompt for {section_config['section_title']}: {len(business_prompt)} characters")
            logger.info(f"Token limit for section: {section_config.get('max_tokens', 1500)} tokens")

            # Generate using Vertex AI with section-specific parameters
            try:
                effective_prompt = business_prompt
                if section_title == "Other Considerations" and getattr(self, "_covered_topics", None):
                    try:
                        topics = sorted(list(self._covered_topics))[:12]
                        if topics:
                            topics_line = ", ".join(topics)
                            prefix = (
                                "Covered topics (do not repeat; if overlap, provide a distinct new angle or skip): "
                                + topics_line + "\n\n"
                            )
                            effective_prompt = prefix + business_prompt
                    except Exception:
                        pass

                response = await vertex_ai_service.generate_text(
                    prompt=effective_prompt,
                    section_title=section_config['section_title'],
                    context=context,  # keep full context
                    max_tokens=section_config.get("max_tokens", 1500),
                    temperature=section_config.get("temperature", 0.2)
                )

                # Debug: Write uncleaned response to file for Company Header and capture raw JSON
                if "Company Header" in section_config['section_title']:
                    # Attempt to capture raw JSON block from the unprocessed response
                    try:
                        import re as _re
                        json_block_match = _re.search(r"```json[\s\S]*?\{[\s\S]*?\}[\s\S]*?```|\{[\s\S]*?\}", response)
                        raw_json_str = None
                        if json_block_match:
                            block = json_block_match.group(0)
                            # Strip markdown fences if present
                            if block.strip().startswith("```json"):
                                # remove first line and trailing ```
                                lines = block.splitlines()
                                raw_json_str = "\n".join(lines[1:-1])
                            else:
                                raw_json_str = block
                        if raw_json_str:
                            self._last_company_header_json = raw_json_str
                            try:
                                self._last_company_header_dict = json.loads(raw_json_str)
                                logger.info(f"Captured raw Company Header JSON with keys: {list(self._last_company_header_dict.keys())}")
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse captured Company Header JSON: {parse_err}")
                    except Exception as capture_err:
                        logger.warning(f"Failed to capture raw Company Header JSON: {capture_err}")

                    with open("basic_info_debug_output.txt", "w") as f:
                        f.write("=== UNCLEANED RESPONSE ===\n")
                        f.write(response)
                        f.write("\n\n=== CLEANED RESPONSE ===\n")
                        cleaned_response = self._post_process_response(response, section_config['section_title'], custom_fields)
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

                # Debug: Write response to file for Operational Complexity and extract financial insights
                if "Operational Complexity" in section_config['section_title']:
                    with open("operational_complexity_output.txt", "w") as f:
                        f.write("=== OPERATIONAL COMPLEXITY RESPONSE ===\n")
                        f.write(response)
                        f.write("\n\n=== END OPERATIONAL COMPLEXITY ===\n")
                    logger.info("Operational complexity debug output written to operational_complexity_output.txt")
                    
                    # Extract financial insights from the response
                    self._financial_insights = self._extract_financial_insights(response)
                    if self._financial_insights:
                        logger.info("Financial insights extracted and stored for financial section")

                # Post-process the response
                cleaned_response = self._post_process_response(response, section_config['section_title'], custom_fields)

                # Record covered topics (headings/lead bullets) for deduping "Other Considerations"
                try:
                    title_l = section_title.lower()
                    if any(k in title_l for k in [
                        'operational complexity', 'what happens operationally', 'industry', 'stickiness', 'pricing', 'financial']):
                        for line in cleaned_response.split('\n'):
                            s = line.strip()
                            if s.startswith('- ') or s.startswith('• '):
                                head = s[2:]
                                # take heading before ':' if present and strip bold
                                import re
                                head = re.sub(r'^\*\*(.*?)\*\*:?$', r'\1', head).split(':')[0].strip().lower()
                                if head:
                                    self._covered_topics.add(head)
                except Exception:
                    pass
                
                # Remove financial insights and balance sheet trends from Operational Complexity response since they'll appear in financial section
                if "Operational Complexity" in section_config['section_title']:
                    import re
                    if self._financial_insights:
                        cleaned_response = re.sub(r'### FINANCIAL_INSIGHTS.*?(?=\n###|\n\n###|$)', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE).strip()
                        logger.info("Removed financial insights from Operational Complexity response")
                    if getattr(self, '_balance_sheet_trends', None):
                        cleaned_response = re.sub(r'### BALANCE_SHEET_TRENDS.*?(?=\n###|\n\n###|$)', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE).strip()
                        logger.info("Removed balance sheet trends from Operational Complexity response")

                logger.info(f"Successfully generated section '{section_title}' with {len(cleaned_response)} characters")
                return cleaned_response

            except Exception as e:
                logger.error(f"Error generating section '{section_title}': {e}")
                return f"[Error generating content for {section_title}: {str(e)}]"

        except Exception as e:
            logger.error(f"Error in generate_section for '{section_config.get('section_title', 'Unknown')}': {e}")
            return f"[Error processing section: {str(e)}]"

    def generate_pdf(self, memo_data: Dict[str, Any], output_path: str) -> str:
        """Generate a PDF file from the memo data using WeasyPrint. Returns the actual file path created."""
        try:
            logger.info(f"Generating PDF with WeasyPrint: {output_path}")

            # Load template sections to maintain correct order
            template_sections = self._load_template()

            # Section mappings based on template order
            section_order = []
            for section_config in template_sections:
                section_title = section_config["section_title"]
                if section_title == "Company Header":
                    section_order.append(('basic_info', section_title))
                elif section_title == "Description of the business as told to a 12-year old":
                    section_order.append(('simple_description', section_title))
                elif section_title == "What happens operationally":
                    section_order.append(('what_happens_operationally', section_title))
                elif section_title == "Operational Complexity Assessment":
                    section_order.append(('operational_complexity', section_title))
                elif section_title == "Other Considerations":
                    section_order.append(('other_considerations', section_title))
                else:
                    # Handle new sections using auto-generated keys (same logic as in generate_complete_memo)
                    key = section_title.lower().replace(' ', '_')
                    section_order.append((key, section_title))

            # Prepare sections for template in correct order
            sections = []
            for key, title in section_order:
                content = memo_data.get(key, '')
                if content and content != '[No content available]':
                    # Convert content to HTML
                    html_content = self._convert_markdown_to_html(content)
                    sections.append({
                        'title': title,
                        'content': html_content
                    })

            # Set up Jinja2 environment
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template('memo_template.html')

            # Render HTML
            html_content = template.render(sections=sections)

            # Generate PDF with WeasyPrint
            try:
                from weasyprint import HTML as WeasyprintHTML
                html_doc = WeasyprintHTML(string=html_content)
                html_doc.write_pdf(output_path)
                logger.info(f"PDF generated successfully: {output_path}")
            except ImportError:
                logger.warning("WeasyPrint not available, falling back to HTML")
                # Save HTML as fallback
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML content saved as fallback: {html_path}")
                # Update output_path to the actual file created
                output_path = html_path
            except Exception as pdf_error:
                logger.error(f"PDF generation failed: {pdf_error}, falling back to HTML")
                # Save HTML as fallback
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML content saved as fallback: {html_path}")
                # Update output_path to the actual file created
                output_path = html_path

            return output_path

        except Exception as e:
            logger.error(f"Critical error in PDF generation: {e}")
            logger.error(f"Memo data: {memo_data}")
            logger.exception("Full traceback:")
            raise Exception(f"Failed to generate PDF: {str(e)}")

    def _convert_markdown_to_html(self, content: str) -> str:
        """Convert markdown-like content to HTML for PDF generation."""
        if not content:
            return ""

        content = str(content)

        # If content already contains an HTML table, pass it through unchanged
        stripped = content.lstrip()
        if stripped.startswith("<table"):
            return content

        # Convert markdown-style formatting to HTML
        import re

        # Convert **bold** to <strong>bold</strong>
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)

        # Highlight meta phrases
        content = re.sub(r'(Not available in CIM|Not located in CIM)', r'<span class="meta">\1</span>', content, flags=re.IGNORECASE)

        # Split into lines and process
        lines = content.split('\n')
        html_lines = []
        in_list = False
        in_sublist = False
        li_open = False

        # Treat these parent bullets as section headings that own sub-bullets
        parent_heads = {h.lower(): h for h in [
            'Employees', 'Locations', 'Product Map', 'Operational Steps'
        ]}
        current_parent_active = False

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                if in_sublist:
                    html_lines.append('</ul>')
                    in_sublist = False
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('')
                continue

            # Handle main bullet points (• or -)
            if stripped_line.startswith('• ') or stripped_line.startswith('- '):
                # Close any open sublist first
                if in_sublist:
                    html_lines.append('</ul>')
                    in_sublist = False
                if li_open:
                    html_lines.append('</li>')
                    li_open = False

                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                bullet_text = stripped_line[2:].strip()  # Remove bullet and space

                # Detect parent heading bullets (case-insensitive match, ignore trailing colon)
                normalized = bullet_text.rstrip(':').strip().lower()
                is_parent = normalized in parent_heads

                html_lines.append(f'<li>{bullet_text}')
                li_open = True
                current_parent_active = is_parent

            # Handle sub-bullet points (◦ or indented bullets with 2 or 4 spaces)
            elif stripped_line.startswith('◦ ') or line.startswith('  ') or line.startswith('    ') or line.startswith('  - ') or line.startswith('    ◦ '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True

                # Ensure we have an open parent li
                if not li_open:
                    html_lines.append('<li>')
                    li_open = True

                if not in_sublist:
                    html_lines.append('<ul>')
                    in_sublist = True

                # Extract bullet text
                if stripped_line.startswith('◦ '):
                    bullet_text = stripped_line[2:].strip()
                elif stripped_line.startswith('• ') or stripped_line.startswith('- '):
                    bullet_text = stripped_line[2:].strip()
                else:
                    bullet_text = stripped_line

                html_lines.append(f'<li>{bullet_text}</li>')
            # Heuristic: if we are inside a parent heading (e.g., "Employees") and the model
            # emitted the next bullets at the same indentation level (no leading spaces),
            # treat those as sub-bullets until a blank line or next parent heading appears.
            elif (current_parent_active and (stripped_line.startswith('• ') or stripped_line.startswith('- '))):
                # Ensure list and parent <li> are open
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                if not li_open:
                    html_lines.append('<li>')
                    li_open = True
                if not in_sublist:
                    html_lines.append('<ul>')
                    in_sublist = True
                bullet_text = stripped_line[2:].strip()
                # If this new bullet is itself a known parent, close current parent context first
                normalized_child = bullet_text.rstrip(':').strip().lower()
                if normalized_child in parent_heads:
                    # Close current sublist and parent li, then open a new parent li
                    if in_sublist:
                        html_lines.append('</ul>')
                        in_sublist = False
                    if li_open:
                        html_lines.append('</li>')
                        li_open = False
                    html_lines.append(f'<li>{bullet_text}')
                    li_open = True
                    current_parent_active = True
                else:
                    html_lines.append(f'<li>{bullet_text}</li>')
            else:
                # Close any open lists
                if in_sublist:
                    html_lines.append('</ul>')
                    in_sublist = False
                if li_open:
                    html_lines.append('</li>')
                    li_open = False
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<p>{stripped_line}</p>')
                # Reset parent context at paragraph boundaries
                current_parent_active = False

        # Close any remaining open lists
        if in_sublist:
            html_lines.append('</ul>')
        if li_open:
            html_lines.append('</li>')
        if in_list:
            html_lines.append('</ul>')

        return '\n'.join(html_lines)

    async def generate_complete_memo(
        self,
        document_ids: List[str],
        session: AsyncSession,
        custom_fields: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete memo using all sections from the JSON template."""
        logger.info(f"Generating complete memo using {len(document_ids)} documents")

        if not self.template_sections:
            raise ValueError("No template sections loaded")

        memo_data = {}
        company_header_data = None

        # Generate each section
        for i, section_config in enumerate(self.template_sections):
            section_title = section_config["section_title"]

            # Add a small delay between API calls to avoid rate limiting
            if i > 0:
                import asyncio
                await asyncio.sleep(2)  # 2 second delay between sections

            # Generate the section content
            section_content = await self.generate_section(section_config, document_ids, session, custom_fields, user_id, company_header_data)

            # Map sections to our expected data structure
            logger.info(f"Processing section: '{section_title}' with content length: {len(str(section_content))}")
            logger.info(f"Section content preview: {str(section_content)[:200]}...")

            if section_title == "Company Header":
                memo_data['basic_info'] = section_content

                # Prefer raw JSON captured directly from the unprocessed response
                if self._last_company_header_dict:
                    company_header_data = self._last_company_header_dict
                    logger.info("Using captured raw Company Header JSON for downstream sections")
                else:
                    # Fallback: parse from the formatted content if possible
                    try:
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', section_content)
                        if json_match:
                            json_str = json_match.group()
                            company_header_data = json.loads(json_str)
                            logger.info(f"Parsed company header data from formatted content: {company_header_data}")
                        else:
                            logger.warning("Could not extract JSON from company header response")
                    except Exception as e:
                        logger.error(f"Error parsing company header data: {e}")

                logger.info(f"=== COMPANY HEADER DEBUG ===")
                logger.info(f"Stored basic_info in memo_data: {section_content}")
                logger.info(f"basic_info type: {type(section_content)}")
                logger.info(f"basic_info length: {len(str(section_content))}")
            elif section_title == "Description of the business as told to a 12-year old":
                memo_data['simple_description'] = section_content
            elif section_title == "What happens operationally":
                memo_data['what_happens_operationally'] = section_content
            elif section_title == "Operational Complexity Assessment":
                memo_data['operational_complexity'] = section_content
            elif section_title == "Other Considerations":
                # Deduplicate against covered topics
                try:
                    dedup_lines = []
                    for line in section_content.split('\n'):
                        s = line.strip()
                        if s.startswith('• ') or s.startswith('- '):
                            head = s[2:]
                            import re
                            norm = re.sub(r'^\*\*(.*?)\*\*:?$', r'\1', head).split(':')[0].strip().lower()
                            if norm and any(norm == t or (norm in t) or (t in norm) for t in self._covered_topics):
                                continue
                        dedup_lines.append(line)
                    section_content = '\n'.join(dedup_lines)
                except Exception:
                    pass
                memo_data['other_considerations'] = section_content
            else:
                # For any additional sections not in template
                key = section_title.lower().replace(' ', '_')
                memo_data[key] = section_content
                logger.info(f"Mapped section '{section_title}' to key '{key}'")

        # Post-process: inject financial blurbs into the P&L section in the correct structure
        try:
            pnl_key_detected = None
            for key, value in list(memo_data.items()):
                if isinstance(value, str) and "<table" in value and "Three year historical P&L" in value:
                    pnl_key_detected = key
                    break

            if pnl_key_detected:
                current_content = memo_data[pnl_key_detected]
                
                # Structure should be: financial_insights -> P&L table -> balance_sheet_trends -> balance sheet table
                # Split on the existing "Financial insights not found" and "Balance sheet trends not found" markers
                import re
                
                # Remove existing "not found" messages if they exist
                current_content = re.sub(r'<div[^>]*><p class="meta">Financial insights not found</p></div>\s*', '', current_content)
                current_content = re.sub(r'<p class="meta">Balance sheet trends not found</p>\s*', '', current_content)
                
                # Build the correct structure
                new_content = ""
                
                # 1. Add financial insights or "not found" message
                if self._financial_insights and "Not located in CIM" not in self._financial_insights:
                    new_content += f'<div style="margin-bottom: 16px;"><p>{self._financial_insights}</p></div>\n'
                    logger.info("Post-injecting financial insights blurb above P&L table")
                else:
                    new_content += '<div style="margin-bottom: 8px;"><p class="meta">Financial insights not found</p></div>\n'
                    logger.info("Post-injecting 'financial insights not found' message")
                
                # 2. Find P&L table and its sourcing
                pnl_table_match = re.search(r'(<table[^>]*>.*?</table>)', current_content, re.DOTALL)
                if pnl_table_match:
                    pnl_table_html = pnl_table_match.group(1)
                    
                    # Find P&L sourcing that comes immediately after the P&L table
                    pnl_end_pos = pnl_table_match.end()
                    pnl_source_match = re.search(r'(<div[^>]*>\s*<strong><span class="meta">Source:</span></strong>.*?</div>)', current_content[pnl_end_pos:], re.DOTALL)
                    
                    # Add P&L table + sourcing
                    new_content += pnl_table_html + '\n'
                    if pnl_source_match:
                        new_content += pnl_source_match.group(1) + '\n'
                        content_after_pnl_source = pnl_end_pos + pnl_source_match.end()
                    else:
                        content_after_pnl_source = pnl_end_pos
                    
                    # 3. Add balance sheet trends or "not found" message
                    balance_sheet_trends = getattr(self, '_balance_sheet_trends', None)
                    auto_bs_blurb = getattr(self, '_auto_balance_sheet_blurb', None)
                    
                    # Use operational complexity trends if available and valid, otherwise use auto-generated blurb
                    if balance_sheet_trends and "Not located in CIM" not in balance_sheet_trends:
                        new_content += f'\n<div style="height:16px"></div>\n<div style="margin-bottom: 8px;"><p>{balance_sheet_trends}</p></div>\n'
                        logger.info("Post-injecting balance sheet trends blurb from operational complexity")
                    elif auto_bs_blurb and auto_bs_blurb.strip():
                        new_content += f'\n<div style="height:16px"></div>\n<div style="margin-bottom: 8px;"><p>{auto_bs_blurb}</p></div>\n'
                        logger.info("Post-injecting auto-generated balance sheet trends blurb")
                    else:
                        new_content += '\n<div style="height:16px"></div>\n<p class="meta">Balance sheet trends not found</p>\n'
                        logger.info("Post-injecting 'balance sheet trends not found' message")
                    
                    # 4. Add balance sheet table and its sourcing (everything after P&L sourcing)
                    remaining_content = current_content[content_after_pnl_source:]
                    new_content += '\n<div style="height:16px"></div>\n' + remaining_content
                    
                    memo_data[pnl_key_detected] = new_content
                    logger.info(f"Post-processed P&L section with correct blurb structure (key='{pnl_key_detected}')")
                else:
                    logger.warning("Could not find P&L table in financial section for post-processing")
                    
        except Exception as inject_err:
            logger.error(f"Failed post-processing of financial blurbs: {inject_err}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info("Complete memo generated successfully")
        logger.info(f"=== FINAL MEMO DATA DEBUG ===")
        logger.info(f"Final memo_data keys: {list(memo_data.keys())}")
        logger.info(f"basic_info in final memo_data: {memo_data.get('basic_info', 'NOT FOUND')}")
        logger.info(f"basic_info type in final memo_data: {type(memo_data.get('basic_info', 'NOT FOUND'))}")
        return memo_data


# Global instance
auto_memo_generator = MemoGeneratorService()
