# Standard library imports
import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Third-party imports
import pdfplumber
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings

# Logging
import logging

# Set up a module-level logger
logger = logging.getLogger(__name__)

from app.core.config import settings
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.deal import Deal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .vertex_ai import vertex_ai_service
from .ocr_service import ocr_service
from .vector_search import get_vector_search_service


@dataclass
class CategorizationResult:
    """Structured result from document categorization"""
    category: str
    confidence: float
    source: str  # "filename", "semantic", "keyword"
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestService:
    def __init__(self):
        # Initialize Vertex AI service
        logger.info("Initializing Vertex AI service for embeddings...")
        self.use_vertex_ai = vertex_ai_service.validate_configuration()
        logger.info(f"Vertex AI service ready: {self.use_vertex_ai}")

        # Initialize vector search service (Vertex AI Vector Search or ChromaDB fallback)
        logger.info("Initializing vector search service...")
        self.vector_service = get_vector_search_service()
        logger.info(f"Vector search service ready: {type(self.vector_service).__name__}")
        
        # Legacy ChromaDB support (will be deprecated)
        self.collection = None
        if hasattr(self.vector_service, 'collection'):
            self.collection = self.vector_service.collection
            logger.info("ChromaDB collection available for backward compatibility")

        # Cache category embeddings for performance optimization
        self._category_embeddings = None
        self._category_embeddings_initialized = False

    async def _ensure_category_embeddings_cached(self) -> List[List[float]]:
        """Initialize and cache category embeddings for performance optimization"""
        if not self._category_embeddings_initialized:
            if self.use_vertex_ai:
                try:
                    logger.info("Initializing category embeddings cache...")
                    category_prompts = [self.CATEGORY_DESCRIPTIONS[label] for label in self.CATEGORY_LABELS]
                    self._category_embeddings = await vertex_ai_service.get_embeddings(category_prompts)
                    logger.info(f"Category embeddings cached successfully ({len(self._category_embeddings)} embeddings)")
                except Exception as e:
                    logger.warning(f"Failed to cache category embeddings: {e}")
                    self._category_embeddings = None
            else:
                self._category_embeddings = None
            
            self._category_embeddings_initialized = True
        
        return self._category_embeddings

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Vertex AI or fallback method"""
        if self.use_vertex_ai:
            try:
                logger.info(f"Generating embeddings for {len(texts)} texts using Vertex AI...")
                return await vertex_ai_service.get_embeddings(texts)
            except Exception as e:
                logger.warning(f"Vertex AI embeddings failed, using fallback: {e}")
        
        # Fallback: simple hash-based embeddings for development/testing
        logger.info(f"Using fallback embeddings for {len(texts)} texts...")
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for testing
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert hex to float array (simplified)
            embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 32), 2)]
            # Pad to 768 dimensions (standard embedding size)
            embedding.extend([0.0] * (768 - len(embedding)))
            embeddings.append(embedding[:768])
        
        return embeddings

    async def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content and tables from PDF file using pdfplumber with OCR fallback."""
        text_content = ""

        # Primary extraction using pdfplumber (works well for most PDFs)
        try:
            logger.info(f"Starting text extraction from PDF: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # STEP 1: Extract formal tables first and build content set
                    formal_tables = page.extract_tables()
                    formal_table_content = set()
                    
                    # Build a set of content already captured by formal tables
                    for table in formal_tables:
                        for row in table:
                            if row:
                                for cell in row:
                                    if cell and str(cell).strip():
                                        # Normalize for comparison
                                        normalized = str(cell).strip().lower()
                                        formal_table_content.add(normalized)
                    
                    # STEP 2: Extract all text from the page
                    page_text = page.extract_text()
                    if page_text:
                        # STEP 3: Process text, excluding content already captured by formal tables
                        enhanced_text = self._enhance_text_based_tables_with_deduplication(
                            page_text, page_num, formal_table_content
                        )
                        text_content += f"\n=== PAGE {page_num} CONTENT ===\n"
                        text_content += enhanced_text + "\n"
                    
                    # STEP 4: Add formal tables with enhanced formatting
                    if formal_tables:
                        text_content += f"\n=== PAGE {page_num} FORMAL TABLES ===\n"
                        for table_num, table in enumerate(formal_tables, 1):
                            formatted_table = self._format_table_for_text(table, page_num, table_num)
                            text_content += formatted_table + "\n"
            
            # Check if we got meaningful text - if not, try OCR
            if ocr_service.should_use_ocr(file_path, text_content):
                logger.warning(f"Very little text extracted from {file_path} using pdfplumber, trying OCR...")
                return await self._extract_text_with_ocr_fallback(file_path)
            
            logger.info(f"Text extraction completed successfully. Total characters: {len(text_content)}")
            return text_content
                    
        except Exception as e:
            # If pdfplumber fails, try OCR
            logger.warning(f"pdfplumber failed to extract text from {file_path}: {str(e)} – trying OCR")
            return await self._extract_text_with_ocr_fallback(file_path)

    async def _extract_text_with_ocr_fallback(self, file_path: str) -> str:
        """Extract text using OCR when traditional methods fail."""
        try:
            # Try OCR first if available
            if ocr_service.ocr_available:
                logger.info(f"Using OCR for {file_path}")
                return await ocr_service.extract_text_with_ocr(file_path)
            else:
                # Try PyPDF2 as a fallback before giving up
                logger.info(f"OCR not available, trying PyPDF2 fallback for {file_path}")
                return await self._extract_text_with_pypdf2(file_path)
                
        except Exception as e:
            logger.error(f"OCR extraction also failed for {file_path}: {str(e)}")
            # Try PyPDF2 as last resort
            try:
                return await self._extract_text_with_pypdf2(file_path)
            except Exception as final_e:
                logger.error(f"All text extraction methods failed for {file_path}: {str(final_e)}")
                return f"[ERROR: Could not extract text from PDF - {str(final_e)}]"

    async def _extract_text_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 as a fallback."""
        try:
            import PyPDF2
            
            text_content = ""
            with open(file_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n=== PAGE {page_num} (PyPDF2) ===\n"
                        text_content += page_text + "\n"
            
            logger.info(f"PyPDF2 extraction completed. Total characters: {len(text_content)}")
            return text_content
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise

    def _enhance_text_based_tables_with_deduplication(self, text: str, page_num: int, formal_content: set) -> str:
        """Enhanced version that excludes content already captured by formal tables."""
        lines = text.split('\n')
        enhanced_lines = []
        current_table = []
        in_potential_table = False
        table_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line contains content already captured by formal tables
            line_words = set(word.lower() for word in line.split() if word.strip())
            overlap_with_formal = line_words.intersection(formal_content)
            
            # If significant overlap, skip this line to avoid duplication
            if len(overlap_with_formal) > len(line_words) * 0.5:  # 50% overlap threshold
                i += 1
                continue
            
            # Check if this line looks like it could be part of a table
            is_table_line = self._is_table_line(line, lines, i)
            
            if is_table_line and not in_potential_table:
                # Start of a potential table
                in_potential_table = True
                current_table = [line]
            elif is_table_line and in_potential_table:
                # Continue building the table
                current_table.append(line)
            elif in_potential_table and not is_table_line:
                # End of table - process what we collected
                if len(current_table) >= 2:  # At least header + 1 data row
                    table_count += 1
                    formatted_table = self._format_text_based_table(current_table, page_num, table_count)
                    enhanced_lines.append(formatted_table)
                else:
                    # Not a real table, add lines back as regular text
                    enhanced_lines.extend(current_table)
                
                current_table = []
                in_potential_table = False
                enhanced_lines.append(line)
            else:
                # Regular text line
                enhanced_lines.append(line)
            
            i += 1
        
        # Handle any remaining table at end of page
        if in_potential_table and len(current_table) >= 2:
            table_count += 1
            formatted_table = self._format_text_based_table(current_table, page_num, table_count)
            enhanced_lines.append(formatted_table)
        elif current_table:
            enhanced_lines.extend(current_table)
        
        return '\n'.join(enhanced_lines)

    def _enhance_text_based_tables(self, text: str, page_num: int) -> str:
        """Detect and enhance text-based tables within PDF content (original version for backward compatibility)."""
        lines = text.split('\n')
        enhanced_lines = []
        current_table = []
        in_potential_table = False
        table_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line looks like it could be part of a table
            is_table_line = self._is_table_line(line, lines, i)
            
            if is_table_line and not in_potential_table:
                # Start of a potential table
                in_potential_table = True
                current_table = [line]
            elif is_table_line and in_potential_table:
                # Continue building the table
                current_table.append(line)
            elif in_potential_table and not is_table_line:
                # End of table - process what we collected
                if len(current_table) >= 2:  # At least header + 1 data row
                    table_count += 1
                    formatted_table = self._format_text_based_table(current_table, page_num, table_count)
                    enhanced_lines.append(formatted_table)
                else:
                    # Not a real table, add lines back as regular text
                    enhanced_lines.extend(current_table)
                
                current_table = []
                in_potential_table = False
                enhanced_lines.append(line)
            else:
                # Regular text line
                enhanced_lines.append(line)
            
            i += 1
        
        # Handle any remaining table at end of page
        if in_potential_table and len(current_table) >= 2:
            table_count += 1
            formatted_table = self._format_text_based_table(current_table, page_num, table_count)
            enhanced_lines.append(formatted_table)
        elif current_table:
            enhanced_lines.extend(current_table)
        
        return '\n'.join(enhanced_lines)

    def _is_table_line(self, line: str, all_lines: list, line_index: int) -> bool:
        """Determine if a line is likely part of a table structure."""
        if not line.strip():
            return False
        
        # Look for lines with multiple columns separated by spaces
        parts = [part.strip() for part in line.split() if part.strip()]
        
        # Must have at least 2 parts to be a table
        if len(parts) < 2:
            return False
        
        # Check for financial indicators
        has_numbers = any(self._contains_number(part) for part in parts)
        has_percentages = '%' in line
        has_currency = '$' in line or any(keyword in line.lower() for keyword in ['million', 'billion', 'thousand'])
        
        # Check for year patterns (common in financial tables)
        has_years = any(part.isdigit() and len(part) == 4 and part.startswith('20') for part in parts)
        
        # Look for financial keywords
        financial_keywords = ['revenue', 'profit', 'margin', 'ebitda', 'cash', 'debt', 'assets', 'growth']
        has_financial_terms = any(keyword in line.lower() for keyword in financial_keywords)
        
        # Check if nearby lines have similar structure (table consistency)
        similar_structure_nearby = False
        for offset in [-1, 1]:
            nearby_index = line_index + offset
            if 0 <= nearby_index < len(all_lines):
                nearby_line = all_lines[nearby_index].strip()
                if nearby_line:
                    nearby_parts = [part.strip() for part in nearby_line.split() if part.strip()]
                    if abs(len(nearby_parts) - len(parts)) <= 1:  # Similar number of columns
                        similar_structure_nearby = True
                        break
        
        # Consider it a table line if it has financial characteristics
        return (has_numbers and (has_percentages or has_currency or has_years or has_financial_terms)) or \
               (len(parts) >= 3 and similar_structure_nearby and (has_numbers or has_financial_terms))

    def _contains_number(self, text: str) -> bool:
        """Check if text contains a number (including formatted numbers)."""
        import re
        # Match numbers with optional commas and decimals, percentages, currency
        number_pattern = r'[\d,]+\.?\d*[%]?|\$[\d,]+\.?\d*'
        return bool(re.search(number_pattern, text))

    def _format_text_based_table(self, table_lines: list, page_num: int, table_num: int) -> str:
        """Format a detected text-based table for better searchability."""
        if not table_lines:
            return ""
        
        # Determine if this is a financial table
        table_text = ' '.join(table_lines).lower()
        is_financial = any(keyword in table_text for keyword in [
            'revenue', 'profit', 'margin', 'ebitda', 'cash', 'debt', 'assets', 
            'income', 'expenses', 'growth', '$', '%', 'million', 'billion'
        ])
        
        formatted_text = f"\n{'='*60}\n"
        formatted_text += f"TEXT-BASED TABLE {table_num} (Page {page_num})\n"
        if is_financial:
            formatted_text += "[FINANCIAL DATA TABLE]\n"
        formatted_text += f"{'='*60}\n"
        
        # Try to identify headers and data
        header_line = table_lines[0]
        data_lines = table_lines[1:]
        
        # Format the table with clear structure
        formatted_text += f"HEADER: {header_line}\n"
        formatted_text += f"{'-'*40}\n"
        
        for i, data_line in enumerate(data_lines, 1):
            formatted_text += f"Row {i}: {data_line}\n"
        
        # Add analysis for financial tables
        if is_financial:
            analysis = self._analyze_financial_table(table_lines)
            if analysis:
                formatted_text += f"\nTABLE ANALYSIS: {analysis}\n"
        
        formatted_text += f"{'='*60}\n"
        return formatted_text

    def _analyze_financial_table(self, table_lines: list) -> str:
        """Provide analysis of financial table content."""
        analysis_parts = []
        
        # Count numeric values
        import re
        all_numbers = []
        for line in table_lines:
            numbers = re.findall(r'[\d,]+\.?\d*', line)
            all_numbers.extend([num.replace(',', '') for num in numbers if num.replace(',', '').replace('.', '').isdigit()])
        
        if all_numbers:
            analysis_parts.append(f"{len(all_numbers)} numeric values found")
        
        # Look for time periods
        years = re.findall(r'20\d{2}', ' '.join(table_lines))
        if years:
            analysis_parts.append(f"Time periods: {', '.join(sorted(set(years)))}")
        
        # Look for percentages
        percentages = re.findall(r'\d+\.?\d*%', ' '.join(table_lines))
        if percentages:
            analysis_parts.append(f"{len(percentages)} percentage values")
        
        # Look for currency
        currency_values = re.findall(r'\$[\d,]+\.?\d*', ' '.join(table_lines))
        if currency_values:
            analysis_parts.append(f"{len(currency_values)} currency values")
        
        return "; ".join(analysis_parts) if analysis_parts else "Financial data structure detected"

    def _format_table_for_text(self, table: list, page_num: int, table_num: int) -> str:
        """Format extracted table data into readable text format for embeddings and search."""
        if not table or not any(table):
            return ""
        
        formatted_text = f"\nTABLE {table_num} (Page {page_num}):\n"
        formatted_text += "=" * 50 + "\n"
        
        # Clean and process table data
        cleaned_table = []
        for row in table:
            if row and any(cell and str(cell).strip() for cell in row if cell is not None):
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        # Try to identify if this looks like a financial table
        is_financial = self._is_financial_table(cleaned_table)
        
        if is_financial:
            formatted_text += "[FINANCIAL DATA TABLE]\n"
        
        # Format as structured text that preserves relationships
        headers = cleaned_table[0] if cleaned_table else []
        data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
        
        # Add headers
        if headers:
            formatted_text += "COLUMNS: " + " | ".join(headers) + "\n"
            formatted_text += "-" * 50 + "\n"
        
        # Add data rows with clear structure
        for row_idx, row in enumerate(data_rows):
            if len(row) == len(headers):
                # Format as key-value pairs for better semantic understanding
                row_text = f"Row {row_idx + 1}: "
                for header, value in zip(headers, row):
                    if value.strip():  # Only include non-empty values
                        row_text += f"{header}={value}, "
                formatted_text += row_text.rstrip(", ") + "\n"
            else:
                # Fallback for misaligned rows
                formatted_text += f"Row {row_idx + 1}: " + " | ".join(row) + "\n"
        
        # Add summary for financial tables
        if is_financial:
            summary = self._generate_table_summary(headers, data_rows)
            if summary:
                formatted_text += "\nTABLE SUMMARY: " + summary + "\n"
        
        formatted_text += "=" * 50 + "\n"
        return formatted_text

    def _is_financial_table(self, table: list) -> bool:
        """Detect if a table contains financial data based on headers and content."""
        if not table:
            return False
        
        # Financial keywords to look for in headers and content
        financial_keywords = [
            'revenue', 'sales', 'income', 'profit', 'loss', 'ebitda', 'margin',
            'cash', 'flow', 'assets', 'liabilities', 'equity', 'debt',
            'gross', 'net', 'operating', 'expenses', 'cost', 'costs',
            'balance', 'statement', 'year', 'quarter', 'fy', 'q1', 'q2', 'q3', 'q4',
            '$', '%', 'million', 'billion', 'thousand', 'growth', 'roi', 'return'
        ]
        
        # Check headers (first row)
        headers = table[0] if table else []
        header_text = " ".join(str(cell).lower() for cell in headers if cell)
        
        # Check first few data rows for patterns
        sample_text = ""
        for row in table[1:4]:  # Check first 3 data rows
            sample_text += " ".join(str(cell).lower() for cell in row if cell)
        
        combined_text = header_text + " " + sample_text
        
        # Count financial keyword matches
        matches = sum(1 for keyword in financial_keywords if keyword in combined_text)
        
        # Also check for currency symbols and percentage signs
        has_currency = '$' in combined_text or '€' in combined_text or '£' in combined_text
        has_percentage = '%' in combined_text
        has_numbers = any(char.isdigit() for char in combined_text)
        
        # Consider it financial if multiple indicators are present
        return matches >= 2 or (matches >= 1 and (has_currency or has_percentage) and has_numbers)

    def _generate_table_summary(self, headers: list, data_rows: list) -> str:
        """Generate a summary of financial table content for better searchability."""
        if not headers or not data_rows:
            return ""
        
        summary_parts = []
        
        # Identify key financial metrics in headers
        key_metrics = []
        for header in headers:
            header_lower = str(header).lower()
            if any(keyword in header_lower for keyword in ['revenue', 'sales', 'income', 'profit', 'ebitda']):
                key_metrics.append(header)
        
        if key_metrics:
            summary_parts.append(f"Key metrics tracked: {', '.join(key_metrics)}")
        
        # Count time periods (years, quarters)
        time_indicators = []
        for row in data_rows:
            for cell in row:
                cell_str = str(cell).lower()
                if any(indicator in cell_str for indicator in ['2020', '2021', '2022', '2023', '2024', 'fy', 'q1', 'q2', 'q3', 'q4']):
                    time_indicators.append(cell)
                    break
        
        if time_indicators:
            summary_parts.append(f"Time periods: {len(set(time_indicators))} periods")
        
        # Note data dimensions
        summary_parts.append(f"Data points: {len(data_rows)} rows x {len(headers)} columns")
        
        return "; ".join(summary_parts)

    async def extract_text_from_csv(self, file_path: str) -> str:
        """Extract and format text content from CSV file."""
        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path)
            
            # Get basic information about the dataset
            filename = Path(file_path).name
            text_content = f"CSV File: {filename}\n"
            text_content += f"Dataset Overview: {len(df)} rows, {len(df.columns)} columns\n\n"
            
            # Add column information
            text_content += "Columns and Data Types:\n"
            for col in df.columns:
                dtype = str(df[col].dtype)
                text_content += f"- {col}: {dtype}\n"
            text_content += "\n"
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text_content += "Summary Statistics for Numeric Columns:\n"
                for col in numeric_cols:
                    text_content += f"{col}:\n"
                    text_content += f"  - Min: {df[col].min()}\n"
                    text_content += f"  - Max: {df[col].max()}\n"
                    text_content += f"  - Mean: {df[col].mean():.2f}\n"
                    text_content += f"  - Count: {df[col].count()}\n"
                text_content += "\n"
            
            # Add sample data (first 10 rows)
            text_content += "Sample Data (First 10 Rows):\n"
            for idx, row in df.head(10).iterrows():
                row_text = f"Row {idx + 1}: "
                for col in df.columns:
                    row_text += f"{col}={row[col]}, "
                text_content += row_text.rstrip(", ") + "\n"
            
            # If there are many rows, add info about the full dataset
            if len(df) > 10:
                text_content += f"\n[Dataset continues with {len(df) - 10} more rows...]\n"
            
            # Add patterns and insights for specific data types
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    unique_values = df[col].nunique()
                    if unique_values <= 10:  # If few unique values, list them
                        text_content += f"\nUnique values in {col}: {list(df[col].unique())}\n"
                    else:
                        text_content += f"\n{col} has {unique_values} unique values\n"
            
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to extract text from CSV {file_path}: {str(e)}")
            # Fallback to reading as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f"CSV File (parsed as text): {filename}\n\n" + f.read()

    async def extract_text_from_excel(self, file_path: str) -> str:
        """Extract and format text content from Excel file."""
        try:
            filename = Path(file_path).name
            text_content = f"Excel File: {filename}\n\n"
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            text_content += f"Total Sheets: {len(sheet_names)}\n"
            text_content += f"Sheet Names: {', '.join(sheet_names)}\n\n"
            
            # Process each sheet
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                text_content += f"=== SHEET: {sheet_name} ===\n"
                text_content += f"Dimensions: {len(df)} rows, {len(df.columns)} columns\n\n"
                
                # Add column information
                text_content += "Columns:\n"
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    text_content += f"- {col}: {dtype}\n"
                text_content += "\n"
                
                # Add summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    text_content += "Summary Statistics:\n"
                    for col in numeric_cols:
                        text_content += f"{col}: Min={df[col].min()}, Max={df[col].max()}, Mean={df[col].mean():.2f}\n"
                    text_content += "\n"
                
                # Add sample data (first 5 rows for each sheet)
                text_content += "Sample Data:\n"
                for idx, row in df.head(5).iterrows():
                    row_text = f"Row {idx + 1}: "
                    for col in df.columns:
                        row_text += f"{col}={row[col]}, "
                    text_content += row_text.rstrip(", ") + "\n"
                
                if len(df) > 5:
                    text_content += f"[{len(df) - 5} more rows in this sheet...]\n"
                
                text_content += "\n"
            
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to extract text from Excel {file_path}: {str(e)}")
            return f"Excel File (could not parse): {filename}\nError: {str(e)}"

    def create_text_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[dict]:
        """Split text into overlapping chunks for better context preservation."""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries first
            if end < len(text):
                # Look for sentence endings (., !, ?) in the last 200 characters
                search_start = max(start + chunk_size - 200, start)
                search_text = text[search_start:end]
                
                # Find the last sentence ending
                last_period = search_text.rfind('.')
                last_exclamation = search_text.rfind('!')
                last_question = search_text.rfind('?')
                last_newline = search_text.rfind('\n')
                
                # Find the latest sentence boundary
                sentence_boundaries = [last_period, last_exclamation, last_question, last_newline]
                sentence_boundaries = [pos for pos in sentence_boundaries if pos != -1]
                
                if sentence_boundaries:
                    latest_boundary = max(sentence_boundaries)
                    if latest_boundary > 0:  # Only use if it's not at the very beginning
                        end = search_start + latest_boundary + 1
                        chunk_text = text[start:end]
                else:
                    # If no sentence boundary found, try to break at word boundaries
                    if not text[end].isspace():
                        last_space = chunk_text.rfind(' ')
                        if last_space > start + chunk_size * 0.7:  # Only break if we're not too far back
                            end = start + last_space
                            chunk_text = text[start:end]
            
            chunks.append({
                "content": chunk_text.strip(),
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end
            })
            
            start = end - overlap
            chunk_index += 1
        
        return chunks

    # Add category labels and rich descriptions as class variables
    CATEGORY_LABELS = [
        "Income Statement",
        "Balance Sheet",
        "Cash Flow",
        "LOI",
        "CIM",
        "Diligence Tracker",
        "Customer List"
    ]

    # Rich category descriptions for better semantic matching
    CATEGORY_DESCRIPTIONS = {
        "Income Statement": "This document contains a summary of a company's revenues, expenses, and profits over a specific period. It includes line items such as revenue, cost of goods sold, gross profit, operating expenses, operating income, net income, and earnings per share. The document is also known as a profit and loss statement (P&L).",
        "Balance Sheet": "This document provides a snapshot of a company's financial position at a specific point in time. It lists assets, liabilities, and shareholders' equity. Key sections include current assets, non-current assets, current liabilities, long-term liabilities, and equity. The balance sheet shows the accounting equation: Assets = Liabilities + Equity.",
        "Cash Flow": "This document details the inflows and outflows of cash within a company over a period. It includes sections for operating activities, investing activities, and financing activities. The cash flow statement helps assess a company's liquidity, solvency, and financial flexibility.",
        "LOI": "This document is a Letter of Intent, outlining the preliminary terms and intentions of parties involved in a potential transaction. It includes key deal terms, exclusivity, confidentiality, and the framework for further negotiation. The LOI is typically non-binding except for certain provisions.",
        "CIM": "This is a Confidential Information Memorandum, providing detailed information about a company for potential buyers or investors. It includes business overview, financials, management, market analysis, and investment highlights. The CIM is used in M&A and fundraising processes.",
        "Diligence Tracker": "This document tracks the status and progress of due diligence activities in a transaction. It lists required documents, responsible parties, deadlines, and completion status for each diligence item. The tracker helps coordinate and monitor the diligence process.",
        "Customer List": "This document contains a list of a company's customers, including names, contact information, contract values, and other relevant details. It may be used for analysis of customer concentration, revenue sources, and relationship management."
    }

    async def categorize_from_filename(self, filename: str) -> CategorizationResult:
        """
        Smart filename-based categorization using AI to handle variations.
        Returns structured CategorizationResult with confidence and reasoning
        """
        try:
            # Remove file extension and clean up filename
            clean_filename = Path(filename).stem.lower()
            
            # Quick exit for obviously generic filenames
            generic_patterns = [
                'document', 'file', 'untitled', 'new', 'copy', 'scan', 'image',
                'doc1', 'doc2', 'temp', 'draft', 'version'
            ]
            # Check if filename is predominantly generic (starts with or is mostly generic terms)
            is_generic = False
            for pattern in generic_patterns:
                if clean_filename.startswith(pattern) or clean_filename == pattern:
                    is_generic = True
                    break
            
            if is_generic and len(clean_filename) < 15:
                logger.info(f"Generic filename detected: {filename}")
                return CategorizationResult(
                    category="General Document",
                    confidence=0.0,
                    source="filename",
                    reasoning=f"Generic filename pattern detected: '{clean_filename}'"
                )
            
            # Use AI to categorize filename if we have Vertex AI
            if self.use_vertex_ai:
                return await self._ai_categorize_filename(clean_filename)
            else:
                # Fallback to rule-based categorization
                return await self._rule_based_categorize_filename(clean_filename)
                
        except Exception as e:
            logger.warning(f"Error categorizing filename {filename}: {e}")
            return CategorizationResult(
                category="General Document",
                confidence=0.0,
                source="filename",
                reasoning=f"Error during filename categorization: {str(e)}"
            )

    async def _ai_categorize_filename(self, clean_filename: str) -> CategorizationResult:
        """Use AI to categorize filename with confidence scoring"""
        try:
            # Create a prompt that asks AI to categorize and provide confidence
            category_list = ", ".join(self.CATEGORY_LABELS)
            
            prompt = f"""Classify this filename into one of the provided categories.

FILENAME: "{clean_filename}"

CATEGORIES:
{category_list}

Analyze the filename for keywords and variations. Handle abbreviations like "cash_flow_stmt" → "Cash Flow", "inc_stmt" → "Income Statement", "bal_sheet" → "Balance Sheet".

Respond in this exact format:
Category: [exact category name from list]
Confidence: [number between 0.0 and 1.0]

Be conservative with confidence. Use 0.9+ only for very clear matches."""

            # Get AI response
            response = await vertex_ai_service.generate_text(
                prompt, 
                max_tokens=150, 
                temperature=0.1
            )
            
            # Parse response
            category, confidence = self._parse_ai_categorization_response(response)
            
            if category in self.CATEGORY_LABELS:
                logger.info(f"AI filename categorization: {clean_filename} → {category} (confidence: {confidence:.2f})")
                return CategorizationResult(
                    category=category,
                    confidence=confidence,
                    source="filename",
                    reasoning=f"AI analysis of filename '{clean_filename}' matched pattern for {category}",
                    metadata={"ai_response": response.strip()}
                )
            else:
                logger.warning(f"AI returned invalid category: {category}")
                # Fall back to rule-based
                return await self._rule_based_categorize_filename(clean_filename)
                
        except Exception as e:
            logger.warning(f"AI filename categorization failed: {e}")
            # Fall back to rule-based
            return await self._rule_based_categorize_filename(clean_filename)

    def _parse_ai_categorization_response(self, response: str) -> tuple[str, float]:
        """Parse AI response to extract category and confidence"""
        import re
        
        try:
            # Extract category
            category_match = re.search(r'Category:\s*(.+)', response, re.IGNORECASE)
            category = category_match.group(1).strip() if category_match else "General Document"
            
            # Clean up category name (remove extra text after the category)
            category = category.split('\n')[0].strip()
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*([0-9.]+)', response, re.IGNORECASE)
            
            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                # If no confidence provided, infer based on category match quality
                if category in self.CATEGORY_LABELS:
                    confidence = 0.8  # Good match but no explicit confidence
                else:
                    confidence = 0.0
            
            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return category, confidence
            
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return "General Document", 0.0

    async def _rule_based_categorize_filename(self, clean_filename: str) -> CategorizationResult:
        """Fallback rule-based filename categorization with fuzzy matching"""
        # Enhanced patterns with variations and common abbreviations
        filename_patterns = {
            "Income Statement": {
                "high": ["income_statement", "profit_loss", "p_l", "profit_and_loss"],
                "medium": ["income", "profit", "loss", "statement"],
                "low": ["statement"]
            },
            "Balance Sheet": {
                "high": ["balance_sheet", "financial_position", "balance", "equity"],
                "medium": ["balance", "equity", "assets", "liabilities"],
                "low": ["sheet"]
            },
            "Cash Flow": {
                "high": ["cash_flow", "cash_statement", "cash", "flow"],
                "medium": ["cash", "flow", "operating", "investing", "financing"],
                "low": ["statement"]
            },
            "LOI": {
                "high": ["letter_of_intent", "nda", "non_disclosure", "confidentiality"],
                "medium": ["intent", "disclosure", "confidential"],
                "low": ["letter"]
            },
            "CIM": {
                "high": ["confidential_information_memorandum", "investment_memorandum", "info_memo"],
                "medium": ["memorandum", "investment_memo", "conf_memo"],
                "low": ["memo"]
            },
            "Diligence Tracker": {
                "high": ["due_diligence", "dd_checklist", "diligence_checklist", "dd_items"],
                "medium": ["checklist", "diligence", "dd", "due_dilig"],
                "low": ["items"]
            },
            "Customer List": {
                "high": ["customer_contracts", "contract_list", "customer_list", "client_contracts"],
                "medium": ["contracts", "customers", "clients"],
                "low": ["list"]
            }
        }
        
        best_category = "General Document"
        best_confidence = 0.0
        matched_patterns = []
        confidence_level = "none"
        
        for category, patterns in filename_patterns.items():
            # Check high confidence patterns first
            for pattern in patterns.get("high", []):
                if pattern in clean_filename:
                    # More flexible length check: pattern must be meaningful relative to filename
                    # Either cover a good portion of short filenames OR be a substantial match for longer ones
                    pattern_coverage = len(pattern) / len(clean_filename)
                    is_meaningful_match = (
                        pattern_coverage >= 0.3 or  # Pattern covers 30%+ of filename
                        (len(pattern) >= 8 and pattern_coverage >= 0.2) or  # Long pattern with 20%+ coverage
                        len(clean_filename) <= 20  # For short filenames, be more lenient
                    )
                    
                    if is_meaningful_match:
                        best_category = category
                        best_confidence = 0.9
                        matched_patterns.append(f"high:{pattern}")
                        confidence_level = "high"
                        break
            
            if best_confidence >= 0.9:
                break
                
            # Check medium confidence patterns
            for pattern in patterns.get("medium", []):
                if pattern in clean_filename:
                    pattern_coverage = len(pattern) / len(clean_filename)
                    confidence = 0.7 if pattern_coverage >= 0.25 else 0.5
                    if confidence > best_confidence:
                        best_category = category
                        best_confidence = confidence
                        matched_patterns = [f"medium:{pattern}"]
                        confidence_level = "medium"
            
            # Check low confidence patterns only if nothing better found
            if best_confidence < 0.5:
                for pattern in patterns.get("low", []):
                    if pattern in clean_filename:
                        pattern_coverage = len(pattern) / len(clean_filename)
                        confidence = 0.3 if pattern_coverage >= 0.15 else 0.1
                        if confidence > best_confidence:
                            best_category = category
                            best_confidence = confidence
                            matched_patterns = [f"low:{pattern}"]
                            confidence_level = "low"
        
        reasoning = f"Rule-based analysis found {confidence_level} confidence match"
        if matched_patterns:
            reasoning += f" with patterns: {', '.join(matched_patterns)}"
        
        logger.info(f"Rule-based filename categorization: {clean_filename} → {best_category} (confidence: {best_confidence:.2f})")
        return CategorizationResult(
            category=best_category,
            confidence=best_confidence,
            source="filename",
            reasoning=reasoning,
            metadata={"matched_patterns": matched_patterns, "confidence_level": confidence_level}
        )

    async def categorize_document(self, text: str, filename: str = None) -> CategorizationResult:
        """
        Smart hybrid categorization: filename-first with AI, then content-based fallback.
        Returns structured CategorizationResult with confidence, source, and reasoning.
        """
        try:
            logger.info("Starting smart hybrid document categorization...")
            
            # Step 1: Try filename-based categorization if filename provided
            if filename:
                filename_result = await self.categorize_from_filename(filename)
                
                # Use filename result if confidence is high enough
                if filename_result.confidence >= 0.8:
                    logger.info(f"High confidence filename match: {filename_result.category} (confidence: {filename_result.confidence:.2f})")
                    return filename_result
                elif filename_result.confidence >= 0.6:
                    logger.info(f"Medium confidence filename match: {filename_result.category} (confidence: {filename_result.confidence:.2f})")
                    # For medium confidence, we'll use it but also validate with keyword check
                    keyword_result = await self._enhanced_fallback_categorization(text)
                    if keyword_result == filename_result.category:
                        logger.info(f"Filename and keyword results agree: {filename_result.category}")
                        # Update the result to reflect validation
                        filename_result.source = "filename+keyword"
                        filename_result.reasoning += f" (validated by keyword analysis)"
                        filename_result.confidence = min(0.95, filename_result.confidence + 0.1)  # Boost confidence slightly
                        return filename_result
                    else:
                        logger.info(f"Filename ({filename_result.category}) and keyword ({keyword_result}) disagree, using content analysis...")
                else:
                    logger.info(f"Low filename confidence ({filename_result.confidence:.2f}), proceeding to content analysis...")
            else:
                logger.info("No filename provided, using content analysis...")
            
            # Step 2: Fall back to content-based categorization (existing logic with optimizations)
            logger.info("Running content-based categorization...")
            
            # ALWAYS start with keyword-based categorization as it's more reliable
            logger.info("Running keyword-based categorization first...")
            keyword_result = await self._enhanced_fallback_categorization(text)
            
            # If we don't have Vertex AI, return keyword result
            if not self.use_vertex_ai:
                logger.info(f"No Vertex AI available, using keyword result: {keyword_result}")
                return CategorizationResult(
                    category=keyword_result,
                    confidence=0.7,  # Moderate confidence for keyword-only
                    source="keyword",
                    reasoning=f"Keyword-based analysis identified {keyword_result}"
                )
            
            # Use more document content for better categorization
            doc_text = text[:4000]
            logger.info(f"Using {len(doc_text)} characters from document for semantic categorization")
            
            # If document is very short, use multiple chunks to get better representation
            if len(text) < 4000 and len(text) > 1000:
                # For medium-length docs, use beginning and end
                mid_point = len(text) // 2
                doc_text = text[:2000] + " " + text[mid_point:mid_point+2000]
            elif len(text) >= 4000:
                # For longer docs, use beginning, middle, and sample from later sections
                quarter = len(text) // 4
                doc_text = text[:1500] + " " + text[quarter:quarter+1500] + " " + text[quarter*2:quarter*2+1000]
            
            # Try semantic categorization with Vertex AI (with cached embeddings!)
            try:
                # Use cached category embeddings (MAJOR PERFORMANCE IMPROVEMENT!)
                category_embeddings = await self._ensure_category_embeddings_cached()
                
                if category_embeddings is None:
                    logger.warning("Category embeddings not available, falling back to keyword result")
                    return CategorizationResult(
                        category=keyword_result,
                        confidence=0.7,
                        source="keyword",
                        reasoning="Semantic analysis unavailable, using keyword fallback"
                    )
                
                # Use Vertex AI for document embedding only (categories are cached!)
                logger.info("Generating document embedding using Vertex AI...")
                doc_embedding = await vertex_ai_service.get_single_embedding(doc_text)
                
                # Compute cosine similarity using cached category embeddings
                import numpy as np
                similarities = []
                for i, cat_emb in enumerate(category_embeddings):
                    similarity = np.dot(doc_embedding, cat_emb) / (np.linalg.norm(doc_embedding) * np.linalg.norm(cat_emb))
                    similarities.append(similarity)
                
                # Find best match
                best_idx = int(np.argmax(similarities))
                best_similarity = similarities[best_idx]
                semantic_result = self.CATEGORY_LABELS[best_idx]
                
                # Log similarity scores for debugging
                logger.info(f"Semantic categorization results:")
                logger.info(f"Best match: {semantic_result} (similarity: {best_similarity:.4f})")
                
                # Check if semantic and keyword results agree
                if semantic_result == keyword_result:
                    logger.info(f"Semantic and keyword results agree: {semantic_result}")
                    return CategorizationResult(
                        category=semantic_result,
                        confidence=min(0.95, best_similarity + 0.1),
                        source="semantic+keyword",
                        reasoning=f"Semantic analysis (similarity: {best_similarity:.3f}) and keyword analysis both identified {semantic_result}",
                        metadata={
                            "semantic_similarity": best_similarity,
                            "top_similarities": sorted(similarities, reverse=True)[:3],
                            "content_length": len(doc_text)
                        }
                    )
                
                # If they disagree, prefer keyword result if semantic confidence is low
                if best_similarity < 0.7:  # Higher threshold for semantic confidence
                    logger.info(f"Low semantic confidence ({best_similarity:.4f}), preferring keyword result: {keyword_result}")
                    return CategorizationResult(
                        category=keyword_result,
                        confidence=0.8,
                        source="keyword",
                        reasoning=f"Semantic analysis had low confidence ({best_similarity:.3f}), keyword analysis preferred",
                        metadata={"semantic_similarity": best_similarity, "semantic_result": semantic_result}
                    )
                
                # Check if top similarities are very close (indicating uncertainty)
                sorted_similarities = sorted(similarities, reverse=True)
                if len(sorted_similarities) > 1 and (sorted_similarities[0] - sorted_similarities[1]) < 0.1:
                    logger.warning(f"Semantic similarities are close (diff: {sorted_similarities[0] - sorted_similarities[1]:.4f}), preferring keyword result: {keyword_result}")
                    return CategorizationResult(
                        category=keyword_result,
                        confidence=0.75,
                        source="keyword",
                        reasoning=f"Semantic analysis showed ambiguous results (top similarities close), keyword analysis preferred",
                        metadata={
                            "semantic_similarity": best_similarity,
                            "similarity_difference": sorted_similarities[0] - sorted_similarities[1],
                            "semantic_result": semantic_result
                        }
                    )
                
                # High confidence semantic result that disagrees with keywords
                logger.info(f"High confidence semantic result: {semantic_result} (vs keyword: {keyword_result})")
                return CategorizationResult(
                    category=semantic_result,
                    confidence=min(0.95, best_similarity),
                    source="semantic",
                    reasoning=f"High confidence semantic analysis (similarity: {best_similarity:.3f}) overrode keyword analysis",
                    metadata={
                        "semantic_similarity": best_similarity,
                        "keyword_result": keyword_result,
                        "content_length": len(doc_text)
                    }
                )
                
            except Exception as e:
                logger.warning(f"Vertex AI embeddings failed, using keyword result: {e}")
                return CategorizationResult(
                    category=keyword_result,
                    confidence=0.7,
                    source="keyword",
                    reasoning=f"Semantic analysis failed ({str(e)}), using keyword fallback"
                )
            
        except Exception as e:
            logger.error(f"Failed to categorize document: {e}")
            logger.exception("Full traceback:")
            # Final fallback to enhanced keyword-based categorization
            fallback_result = await self._enhanced_fallback_categorization(text)
            return CategorizationResult(
                category=fallback_result,
                confidence=0.5,
                source="keyword",
                reasoning=f"Error in categorization process, using fallback keyword analysis"
            )

    async def _get_fallback_embedding(self, text: str) -> List[float]:
        """Get fallback embedding for a single text"""
        embeddings = await self._get_fallback_embeddings([text])
        return embeddings[0]
    
    async def _get_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get fallback embeddings for multiple texts"""
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for testing
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert hex to float array (simplified)
            embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(len(hash_hex), 32), 2)]
            # Pad to 768 dimensions (standard embedding size)
            embedding.extend([0.0] * (768 - len(embedding)))
            embeddings.append(embedding[:768])
        
        return embeddings

    async def _enhanced_fallback_categorization(self, text: str) -> str:
        """Enhanced fallback categorization using multiple strategies"""
        text_lower = text.lower()
        
        # Define enhanced keyword patterns for each category
        keyword_patterns = {
            "Income Statement": [
                ["income", "statement"], ["profit", "loss"], ["p_l"], ["profit", "and", "loss"]
            ],
            "Balance Sheet": [
                ["balance", "sheet"], ["financial", "position"], ["balance"], ["equity"], ["assets"], ["liabilities"]
            ],
            "Cash Flow": [
                ["cash", "flow"], ["cash", "statement"], ["cash"], ["operating", "activities"], ["investing", "activities"], ["financing", "activities"]
            ],
            "LOI": [
                ["letter", "of", "intent"], ["nda"], ["non", "disclosure"], ["confidentiality", "agreement"]
            ],
            "CIM": [
                ["confidential", "information", "memorandum"], ["investment", "memorandum"],
                ["business", "overview"], ["management", "team"], ["financial", "performance"],
                ["growth", "opportunities"], ["competitive", "positioning"], ["investment", "highlights"]
            ],
            "Diligence Tracker": [
                ["due", "diligence"], ["dd_checklist"], ["diligence", "checklist"], ["dd_items"], ["review", "items"], ["verification"], ["required", "documents"]
            ],
            "Customer List": [
                ["customer", "contracts"], ["contract", "list"], ["customer", "list"], ["client", "contracts"]
            ]
        }
        
        # Score each category based on keyword matches
        category_scores = {}
        for category, patterns in keyword_patterns.items():
            score = 0
            for pattern in patterns:
                if isinstance(pattern, list):
                    # Multi-word pattern - all words must be present
                    if all(word in text_lower for word in pattern):
                        score += 2  # Higher weight for multi-word matches
                else:
                    # Single word pattern
                    if pattern in text_lower:
                        score += 1
            category_scores[category] = score
        
        # Find best match
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                logger.info(f"Enhanced fallback categorization: {best_category[0]} (score: {best_category[1]})")
                return best_category[0]
        
        # If no good matches, try simple substring matching
        for label in self.CATEGORY_LABELS:
            if any(word in text_lower for word in label.lower().split()):
                logger.info(f"Simple fallback categorization: {label}")
                return label
        
        logger.info("No category matches found, defaulting to General Document")
        return "General Document"

    async def ingest_document(
        self,
        file_path: str,
        filename: str,
        deal_id: str,
        user_id: str,
        file_size: int,
        content_type: str,
        storage_path: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> dict:  # Changed return type to dict
        """Process and ingest a single document."""
        from app.core.database import AsyncSessionLocal
        
        logger.info("=== INGEST DOCUMENT START :: %s (deal_id=%s) ===", filename, deal_id)

        # Use provided session or create a new one
        if session is not None:
            # Use the provided session (don't close it)
            return await self._ingest_document_with_session(
                session, file_path, filename, deal_id, user_id, file_size, content_type, storage_path
            )
        else:
            # Create our own session
            async with AsyncSessionLocal() as session:
                return await self._ingest_document_with_session(
                    session, file_path, filename, deal_id, user_id, file_size, content_type, storage_path
                )

    async def _ingest_document_with_session(
        self,
        session: AsyncSession,
        file_path: str,
        filename: str,
        deal_id: str,
        user_id: str,
        file_size: int,
        content_type: str,
        storage_path: Optional[str] = None
    ) -> dict:
        """Internal method to ingest document with a specific session."""
        try:
            # -----------------------
            # 1. Extract text first to categorize document
            # -----------------------
            logger.info("Extracting text (content-type=%s)…", content_type)

            if content_type == "application/pdf":
                text_content = await self.extract_text_from_pdf(file_path)
            elif content_type in ["text/csv", "application/csv"]:
                text_content = await self.extract_text_from_csv(file_path)
            elif content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                text_content = await self.extract_text_from_excel(file_path)
            else:
                # TODO: Add support for other file types
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()

            logger.info("Text extracted (length=%d characters)", len(text_content))

            # Categorize document using smart hybrid approach (filename + content)
            category = await self.categorize_document(text_content, filename)
            logger.info(f"Document categorized as: {category.category}")

            # -----------------------
            # 2. Create DB document row with category
            # -----------------------
            logger.info("Creating document DB record…")
            document_id = str(uuid.uuid4())
            document = Document(
                id=document_id,
                name=filename,
                file_path=file_path,
                storage_path=storage_path,
                file_size=file_size,
                content_type=content_type,
                deal_id=deal_id,
                user_id=user_id,
                status=DocumentStatus.processing,
                category=category.category
            )
            
            session.add(document)
            await session.commit()
            logger.info("Document %s created with status 'processing'", document_id)

            # -----------------------
            # 3. Skip chunking and embedding for Excel files (use structured extraction instead)
            # -----------------------
            is_excel = content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
            
            if is_excel:
                logger.info("Excel file detected - skipping chunking and embedding (using structured extraction instead)")
                document.status = DocumentStatus.completed
                await session.commit()
                logger.info("Excel document %s completed without chunking", document_id)
                return self._document_to_dict(document)

            # -----------------------
            # 3. Chunk text (for non-Excel files)
            # -----------------------
            chunks_data = self.create_text_chunks(text_content)

            if not chunks_data:
                logger.warning("No text chunks created for document %s", document_id)
                document.status = DocumentStatus.completed
                await session.commit()
                return self._document_to_dict(document)

            # -----------------------
            # 4. Generate embeddings for all chunks using Vertex AI
            # -----------------------
            logger.info("Generating embeddings for %d chunks using Vertex AI…", len(chunks_data))
            chunk_contents = [chunk["content"] for chunk in chunks_data]
            embeddings = await self.get_embeddings(chunk_contents)
            logger.info("Embeddings generated successfully")

            # -----------------------
            # 5. Store chunks in database and vector store
            # -----------------------
            logger.info("Storing chunks in database and ChromaDB…")
            
            # Prepare chunk records for database
            document_chunks = []
            chunk_ids = []
            metadatas = []
            
            for i, chunk_data in enumerate(chunks_data):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # Create database record
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=chunk_data["content"],
                    chunk_index=chunk_data["chunk_index"],
                    start_char=chunk_data["start_char"],
                    end_char=chunk_data["end_char"]
                )
                document_chunks.append(chunk)
                
                # Prepare metadata for ChromaDB
                metadatas.append({
                    "document_id": document_id,
                    "document_name": filename,
                    "deal_id": deal_id,
                    "user_id": user_id,  # SECURITY: Include user_id for isolation
                    "chunk_index": chunk_data["chunk_index"],
                    "start_char": chunk_data["start_char"],
                    "end_char": chunk_data["end_char"],
                    "category": category.category,
                    "embedding_model": "vertex_ai" if self.use_vertex_ai else "fallback"
                })

            # Add chunks to database
            session.add_all(document_chunks)
            
            # Add to vector database
            logger.info("Storing embeddings in vector search service (count=%d)…", len(chunk_ids))
            await self.vector_service.add_embeddings(
                embeddings=embeddings,
                documents=chunk_contents,
                metadatas=metadatas,
                ids=chunk_ids
            )

            # Update document status
            document.status = DocumentStatus.completed
            await session.commit()
            logger.info("Document %s processed successfully – status set to 'completed'", document_id)
            
            return self._document_to_dict(document)

        except Exception as e:
            logger.error("Failed to ingest document %s: %s", filename, str(e))
            logger.exception("Full traceback:")
            
            # Update document status to error if it exists
            try:
                document.status = DocumentStatus.error
                await session.commit()
            except:
                pass
            
            raise

        logger.info("=== INGEST DOCUMENT END :: %s ===", filename)

    def _document_to_dict(self, document: Document) -> dict:
        """Convert Document model to dictionary"""
        return {
            "id": document.id,
            "name": document.name,
            "file_path": document.file_path,
            "file_size": document.file_size,
            "content_type": document.content_type,
            "deal_id": document.deal_id,
            "user_id": document.user_id,
            "status": document.status.value if hasattr(document.status, 'value') else str(document.status),
            "category": document.category,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }

    async def ingest_directory(self, directory_path: str, deal_name: str, user_id: str) -> List[Document]:
        """Ingest all supported files from a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        # TODO: Create deal if it doesn't exist
        deal_id = str(uuid.uuid4())  # This should be handled properly

        documents = []
        supported_extensions = ['.pdf', '.txt', '.doc', '.docx', '.csv', '.xlsx', '.xls']
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    # Determine content type
                    content_type_map = {
                        '.pdf': 'application/pdf',
                        '.txt': 'text/plain',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.csv': 'text/csv',
                        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        '.xls': 'application/vnd.ms-excel'
                    }
                    
                    content_type = content_type_map.get(file_path.suffix.lower(), 'application/octet-stream')
                    file_size = file_path.stat().st_size
                    
                    document = await self.ingest_document(
                        str(file_path),
                        file_path.name,
                        deal_id,
                        user_id,
                        file_size,
                        content_type
                    )
                    documents.append(document)
                    
                except Exception as e:
                    print(f"Failed to ingest {file_path}: {str(e)}")
                    continue

        return documents

    def get_category_string(self, categorization_result: CategorizationResult) -> str:
        """Helper method to extract category string for backward compatibility"""
        return categorization_result.category
    
    async def categorize_document_simple(self, text: str, filename: str = None) -> str:
        """
        Simple categorization method that returns just the category string.
        For backward compatibility with existing code.
        """
        result = await self.categorize_document(text, filename)
        return result.category
    
    async def get_categorization_details(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Get detailed categorization information for debugging/review purposes.
        Returns a dictionary with all categorization details.
        """
        result = await self.categorize_document(text, filename)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "source": result.source,
            "reasoning": result.reasoning,
            "metadata": result.metadata or {},
            "requires_review": result.confidence < 0.6,  # Flag for human review
            "confidence_level": (
                "high" if result.confidence >= 0.8 else
                "medium" if result.confidence >= 0.6 else
                "low"
            )
        }


# Global instance - will be initialized lazily
document_ingest_service = None

def get_document_ingest_service():
    global document_ingest_service
    if document_ingest_service is None:
        document_ingest_service = DocumentIngestService()
    return document_ingest_service 