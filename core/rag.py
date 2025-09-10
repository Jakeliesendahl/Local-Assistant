"""
Document ingestion module for RAG system.
Supports PDF, DOCX, and Markdown file processing.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

# Set up logging
logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str) -> Optional[str]:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text content or None if extraction fails
    """
    if PdfReader is None:
        logger.error("pypdf is not installed. Please install it with: pip install pypdf")
        return None
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        reader = PdfReader(file_path)
        text_content = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text_content:
            logger.warning(f"No text content extracted from PDF: {file_path}")
            return None
        
        return "\n\n".join(text_content)
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        return None


def ingest_docx(file_path: str) -> Optional[str]:
    """
    Extract text from a DOCX file using python-docx.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        Optional[str]: Extracted text content or None if extraction fails
    """
    if Document is None:
        logger.error("python-docx is not installed. Please install it with: pip install python-docx")
        return None
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        doc = Document(file_path)
        text_content = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    table_text.append(" | ".join(row_text))
            if table_text:
                text_content.append("--- Table ---\n" + "\n".join(table_text))
        
        if not text_content:
            logger.warning(f"No text content extracted from DOCX: {file_path}")
            return None
        
        return "\n\n".join(text_content)
        
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {e}")
        return None


def ingest_markdown(file_path: str) -> Optional[str]:
    """
    Read text from a Markdown file.
    
    Args:
        file_path (str): Path to the Markdown file
        
    Returns:
        Optional[str]: File content or None if reading fails
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if not content.strip():
            logger.warning(f"Markdown file is empty: {file_path}")
            return None
        
        return content
        
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            logger.info(f"Successfully read file with latin-1 encoding: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading Markdown file with alternative encoding {file_path}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error reading Markdown file {file_path}: {e}")
        return None


def ingest_document(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Unified document ingestion function that handles PDF, DOCX, and Markdown files.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing file info and extracted content,
                                 or None if ingestion fails
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    file_path = Path(file_path)
    virign_file_extension = file_path.suffix
    file_extension = file_path.suffix.lower()
    
    # Determine file type and call appropriate function
    content = None
    if file_extension == '.pdf':
        content = ingest_pdf(str(file_path))
    elif file_extension == '.docx':
        content = ingest_docx(str(file_path))
    elif file_extension in ['.md', '.markdown']:
        content = ingest_markdown(str(file_path))
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return None
    
    if content is None:
        return None
    
    # Return structured data
    return {
        'file_path': str(file_path),
        'file_name': file_path.name,
        'file_type': virign_file_extension,
        'content': content,
        'content_length': len(content),
        'word_count': len(content.split()) if content else 0
    }


def ingest_documents_from_directory(directory_path: str, 
                                  file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Ingest all supported documents from a directory.
    
    Args:
        directory_path (str): Path to the directory containing documents
        file_types (Optional[List[str]]): List of file extensions to include 
                                        (default: ['.pdf', '.docx', '.md', '.markdown'])
        
    Returns:
        List[Dict[str, Any]]: List of successfully ingested documents
    """
    if file_types is None:
        file_types = ['.pdf', '.docx', '.md', '.markdown']
    
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    ingested_documents = []
    directory = Path(directory_path)
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in file_types:
            logger.info(f"Processing file: {file_path}")
            document_data = ingest_document(str(file_path))
            if document_data:
                ingested_documents.append(document_data)
                logger.info(f"Successfully ingested: {file_path.name}")
            else:
                logger.warning(f"Failed to ingest: {file_path.name}")
    
    logger.info(f"Ingested {len(ingested_documents)} documents from {directory_path}")
    return ingested_documents
