import os
import logging
from pathlib import Path
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Standard PDF processing
import pdfplumber

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from image-based PDFs using OCR."""
    
    def __init__(self):
        self.ocr_available = self._check_ocr_dependencies()
        if self.ocr_available:
            # Configure tesseract for better business document recognition
            self._configure_tesseract()
        else:
            logger.warning("OCR service initialized but libraries not available")
        # Reuse a persistent thread pool so we don't block on per-page executor shutdowns
        max_workers = min(4, (os.cpu_count() or 4))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Default timeout for a single page OCR in seconds
        self.page_timeout_seconds = 60
    
    def _check_ocr_dependencies(self) -> bool:
        """Check if OCR dependencies are available."""
        try:
            import pytesseract
            from PIL import Image
            import pdf2image
            return True
        except ImportError as e:
            logger.warning(f"OCR libraries not available: {e}. Install pytesseract, Pillow, and pdf2image for OCR support.")
            return False
    
    def _configure_tesseract(self):
        """Configure tesseract for optimal business document recognition."""
        try:
            import pytesseract
            
            # Set OCR configuration for business documents
            self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$%()[]{}:;-_/\\ '
            
            # Test tesseract availability
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR configured successfully")
        except Exception as e:
            logger.error(f"Tesseract OCR not available: {e}")
            self.ocr_available = False
    
    async def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR when traditional methods fail."""
        if not self.ocr_available:
            raise RuntimeError("OCR libraries not available. Please install pytesseract, Pillow, and pdf2image.")
        
        try:
            logger.info(f"Starting OCR extraction for: {pdf_path}")

            # Stream pages one-by-one to reduce memory and isolate slow pages
            total_pages = await self._get_pdf_page_count(pdf_path)
            if total_pages == 0:
                logger.warning("No pages detected in PDF for OCR")
                return ""

            extracted_text = ""
            for page_index in range(total_pages):
                page_num = page_index + 1
                logger.info(f"Processing page {page_num}/{total_pages} with OCR…")
                started_at = time.perf_counter()
                image = await self._convert_single_page_to_image(pdf_path, page_num)
                if image is None:
                    logger.warning(f"Skipping page {page_num}: image conversion failed or timed out")
                    continue
                page_text = await self._extract_text_from_image(image, page_num)
                if page_text:
                    extracted_text += f"\n=== PAGE {page_num} (OCR) ===\n"
                    extracted_text += page_text + "\n"
                elapsed = time.perf_counter() - started_at
                logger.info(f"Page {page_num} done with OCR in {elapsed:.2f}s")
                # Release image memory promptly
                try:
                    del image
                except Exception:
                    pass

            logger.info(f"OCR extraction completed. Total text length: {len(extracted_text)}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    async def _get_pdf_page_count(self, pdf_path: str) -> int:
        """Return the number of pages in the PDF."""
        try:
            loop = asyncio.get_event_loop()
            def count_pages() -> int:
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages)
            return await loop.run_in_executor(self.executor, count_pages)
        except Exception as e:
            logger.error(f"Failed to get PDF page count: {e}")
            return 0

    async def _convert_single_page_to_image(self, pdf_path: str, page_num: int):
        """Convert a single page to an image with a timeout to prevent hangs."""
        try:
            import pdf2image
            loop = asyncio.get_event_loop()
            def convert_one():
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    fmt='PNG',
                    first_page=page_num,
                    last_page=page_num,
                    thread_count=1
                )
                return images[0] if images else None
            # Give conversion a generous timeout per page
            return await asyncio.wait_for(loop.run_in_executor(self.executor, convert_one), timeout=90)
        except asyncio.TimeoutError:
            logger.warning(f"PDF->image conversion timed out for page {page_num} (90s)")
            return None
        except Exception as e:
            logger.error(f"Failed to convert page {page_num} to image: {e}")
            return None
    
    async def _extract_text_from_image(self, image, page_num: int) -> str:
        """Extract text from a single image using OCR."""
        try:
            import pytesseract
            
            # Preprocess image for better OCR
            processed_image = await self._preprocess_image(image)
            
            # Run tesseract with its native timeout to prevent runaway subprocesses
            loop = asyncio.get_event_loop()
            def run_ocr() -> str:
                return pytesseract.image_to_string(
                    processed_image,
                    config=self.tesseract_config,
                    lang='eng',
                    timeout=self.page_timeout_seconds
                )
            try:
                text = await loop.run_in_executor(self.executor, run_ocr)
            except pytesseract.TesseractError as te:
                # pytesseract raises on timeout or processing errors
                if 'timed out' in str(te).lower() or 'timeout' in str(te).lower():
                    logger.warning(f"Tesseract timed out for page {page_num} ({self.page_timeout_seconds}s)")
                    return ""
                logger.error(f"Tesseract error on page {page_num}: {te}")
                return ""
            except Exception as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                return ""

            # Clean up the extracted text
            cleaned_text = self._clean_ocr_text(text)
            
            logger.info(f"Page {page_num}: Extracted {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return ""
    
    async def _preprocess_image(self, image) -> object:
        """Preprocess image to improve OCR accuracy."""
        try:
            from PIL import ImageEnhance
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
            
            # Optional: Resize if image is too large (helps with memory)
            max_size = 4000
            if max(image.size) > max_size:
                from PIL import Image as PILImage
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and format OCR-extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive spaces
            cleaned_line = ' '.join(line.split())
            
            # Only keep lines with meaningful content
            if cleaned_line and len(cleaned_line) > 2:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def should_use_ocr(self, pdf_path: str, extracted_text: str = None) -> bool:
        """Determine if OCR should be used based on PDF content analysis."""
        try:
            # If we already have extracted text, analyze it
            if extracted_text is not None:
                # Count meaningful characters (excluding whitespace and basic formatting)
                meaningful_chars = len([c for c in extracted_text if c.isalnum() or c in '.,;:'])
                
                # Check for indicators of image-based PDF (lots of bullets, minimal actual content)
                bullet_chars = extracted_text.count('▪') + extracted_text.count('•')
                actual_words = len([word for word in extracted_text.split() if len(word) > 2 and word.isalpha()])
                
                # Use OCR if:
                # 1. Very few meaningful characters (< 1000), OR
                # 2. Lots of bullets with very few actual words (indicates slide deck), OR  
                # 3. Bullet-to-word ratio is too high (> 0.3)
                should_use = (
                    meaningful_chars < 1000 or 
                    (bullet_chars > 50 and actual_words < 100) or
                    (actual_words > 0 and bullet_chars / actual_words > 0.3)
                )
                
                logger.info(f"Text analysis: {len(extracted_text)} total chars, "
                          f"{meaningful_chars} meaningful chars, {actual_words} words, "
                          f"{bullet_chars} bullets, OCR recommended: {should_use}")
                return should_use
            
            # If no extracted text provided, analyze PDF directly
            with pdfplumber.open(pdf_path) as pdf:
                total_text_length = 0
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        total_text_length += len(page_text)
                
                # Calculate average text per page
                avg_text_per_page = total_text_length / page_count if page_count > 0 else 0
                
                # Use OCR if average text per page is very low (< 100 characters)
                # This indicates the PDF contains mostly images
                should_use = avg_text_per_page < 100
                
                logger.info(f"PDF analysis: {page_count} pages, {total_text_length} total chars, "
                          f"{avg_text_per_page:.1f} avg chars/page, OCR recommended: {should_use}")
                
                return should_use
                
        except Exception as e:
            logger.warning(f"Could not analyze PDF for OCR decision: {e}")
            # Default to OCR if analysis fails
            return True


# Global OCR service instance
ocr_service = OCRService()