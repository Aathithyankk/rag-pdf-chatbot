import pypdf as PyPDF2
import io
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """PDF loader for extracting text from PDF files"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def load_pdf(self, pdf_file) -> List[Dict[str, str]]:
        """
        Load and extract text from PDF file
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            List of dictionaries containing page content and metadata
        """
        try:
            # Get filename safely - handle different UploadFile object types
            filename = pdf_file.filename
            # Convert UploadFile to BytesIO for PyPDF2 compatibility
            file_content = pdf_file.file.read()
            pdf_bytes = io.BytesIO(file_content)
            
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            
            pages = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages.append({
                            'page_number': page_num + 1,
                            'content': text.strip(),
                            'metadata': {
                                'source': filename,
                                'page': page_num + 1,
                                'total_pages': len(pdf_reader.pages)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(pages)} pages from PDF: {filename}")
            return pages
            
        except Exception as e:
            logger.error(f"Error loading PDF file: {e}")
            raise Exception(f"Failed to load PDF: {str(e)}")
    
    def chunk_text(self, pages: List[Dict[str, str]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, str]]:
        """
        Split text into chunks for better vector search
        
        Args:
            pages: List of page dictionaries
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        for page in pages:
            text = page['content']
            page_metadata = page['metadata']
            
            # Split text into chunks
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                # Try to break at sentence boundary
                if end < len(text):
                    last_period = chunk_text.rfind('.')
                    last_newline = chunk_text.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > start + chunk_size // 2:
                        end = start + break_point + 1
                        chunk_text = text[start:end]
                
                chunks.append({
                    'chunk_id': f"{page_metadata['source']}_page_{page_metadata['page']}_chunk_{chunk_id}",
                    'content': chunk_text.strip(),
                    'metadata': {
                        **page_metadata,
                        'chunk_id': chunk_id,
                        'chunk_start': start,
                        'chunk_end': end
                    }
                })
                
                start = end - overlap
                chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks