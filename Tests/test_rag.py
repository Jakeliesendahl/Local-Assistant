"""
Comprehensive tests for the RAG document ingestion module.
Tests PDF, DOCX, and Markdown file processing functions.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.rag import (
    ingest_pdf,
    ingest_docx,
    ingest_markdown,
    ingest_document,
    ingest_documents_from_directory
)


class TestRAGIngestion(unittest.TestCase):
    """Test cases for document ingestion functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_markdown_content = """# Test Document
        
This is a test markdown document.

## Section 1
Some content here.

- List item 1
- List item 2

## Section 2
More content with **bold** and *italic* text.
"""
        
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_sample_markdown(self, filename="test.md", content=None):
        """Create a sample markdown file for testing."""
        if content is None:
            content = self.sample_markdown_content
        
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def create_empty_file(self, filename):
        """Create an empty file for testing."""
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("")
        return file_path


class TestMarkdownIngestion(TestRAGIngestion):
    """Test cases specifically for Markdown file ingestion."""
    
    def test_ingest_markdown_success(self):
        """Test successful markdown file ingestion."""
        file_path = self.create_sample_markdown()
        result = ingest_markdown(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn("# Test Document", result)
        self.assertIn("## Section 1", result)
        self.assertIn("List item 1", result)
    
    def test_ingest_markdown_nonexistent_file(self):
        """Test markdown ingestion with non-existent file."""
        result = ingest_markdown("nonexistent_file.md")
        self.assertIsNone(result)
    
    def test_ingest_markdown_empty_file(self):
        """Test markdown ingestion with empty file."""
        file_path = self.create_empty_file("empty.md")
        result = ingest_markdown(file_path)
        self.assertIsNone(result)
    
    def test_ingest_markdown_with_special_characters(self):
        """Test markdown ingestion with special characters."""
        content = "# Test with émojis 🚀\n\nSpecial chars: àáâãäå"
        file_path = self.create_sample_markdown("special.md", content)
        result = ingest_markdown(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn("émojis 🚀", result)
        self.assertIn("àáâãäå", result)
    
    @patch('builtins.open')
    def test_ingest_markdown_encoding_fallback(self, mock_open):
        """Test markdown ingestion with encoding fallback."""
        # First call raises UnicodeDecodeError, second succeeds
        mock_open.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'),
            unittest.mock.mock_open(read_data="Test content").return_value
        ]
        
        with patch('os.path.exists', return_value=True):
            result = ingest_markdown("test.md")
            self.assertEqual(result, "Test content")
            self.assertEqual(mock_open.call_count, 2)


class TestPDFIngestion(TestRAGIngestion):
    """Test cases specifically for PDF file ingestion."""
    
    @patch('core.rag.PdfReader')
    def test_ingest_pdf_success(self, mock_pdf_reader):
        """Test successful PDF file ingestion."""
        # Mock PDF reader and pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        file_path = os.path.join(self.test_dir, "test.pdf")
        with open(file_path, 'w') as f:  # Create dummy file
            f.write("dummy")
        
        result = ingest_pdf(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn("--- Page 1 ---", result)
        self.assertIn("Page 1 content", result)
        self.assertIn("--- Page 2 ---", result)
        self.assertIn("Page 2 content", result)
    
    def test_ingest_pdf_no_pypdf(self):
        """Test PDF ingestion when pypdf is not available."""
        with patch('core.rag.PdfReader', None):
            result = ingest_pdf("test.pdf")
            self.assertIsNone(result)
    
    def test_ingest_pdf_nonexistent_file(self):
        """Test PDF ingestion with non-existent file."""
        result = ingest_pdf("nonexistent.pdf")
        self.assertIsNone(result)
    
    @patch('core.rag.PdfReader')
    def test_ingest_pdf_empty_pages(self, mock_pdf_reader):
        """Test PDF ingestion with empty pages."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "   "  # Only whitespace
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        file_path = os.path.join(self.test_dir, "empty.pdf")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_pdf(file_path)
        self.assertIsNone(result)
    
    @patch('core.rag.PdfReader')
    def test_ingest_pdf_page_extraction_error(self, mock_pdf_reader):
        """Test PDF ingestion with page extraction errors."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Good page"
        mock_page2 = MagicMock()
        mock_page2.extract_text.side_effect = Exception("Extraction error")
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        file_path = os.path.join(self.test_dir, "test.pdf")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_pdf(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn("Good page", result)
        self.assertNotIn("Extraction error", result)


class TestDOCXIngestion(TestRAGIngestion):
    """Test cases specifically for DOCX file ingestion."""
    
    @patch('core.rag.Document')
    def test_ingest_docx_success(self, mock_document):
        """Test successful DOCX file ingestion."""
        # Mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"
        
        # Mock table
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = [mock_para1, mock_para2]
        mock_doc_instance.tables = [mock_table]
        mock_document.return_value = mock_doc_instance
        
        file_path = os.path.join(self.test_dir, "test.docx")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_docx(file_path)
        
        self.assertIsNotNone(result)
        self.assertIn("First paragraph", result)
        self.assertIn("Second paragraph", result)
        self.assertIn("--- Table ---", result)
        self.assertIn("Cell 1 | Cell 2", result)
    
    def test_ingest_docx_no_python_docx(self):
        """Test DOCX ingestion when python-docx is not available."""
        with patch('core.rag.Document', None):
            result = ingest_docx("test.docx")
            self.assertIsNone(result)
    
    def test_ingest_docx_nonexistent_file(self):
        """Test DOCX ingestion with non-existent file."""
        result = ingest_docx("nonexistent.docx")
        self.assertIsNone(result)
    
    @patch('core.rag.Document')
    def test_ingest_docx_empty_document(self, mock_document):
        """Test DOCX ingestion with empty document."""
        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = []
        mock_doc_instance.tables = []
        mock_document.return_value = mock_doc_instance
        
        file_path = os.path.join(self.test_dir, "empty.docx")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_docx(file_path)
        self.assertIsNone(result)


class TestUnifiedIngestion(TestRAGIngestion):
    """Test cases for the unified document ingestion function."""
    
    def test_ingest_document_markdown(self):
        """Test unified ingestion with markdown file."""
        file_path = self.create_sample_markdown()
        result = ingest_document(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['file_type'], '.md')
        self.assertEqual(result['file_name'], 'test.md')
        self.assertIn('content', result)
        self.assertIn('content_length', result)
        self.assertIn('word_count', result)
        self.assertGreater(result['word_count'], 0)
    
    def test_ingest_document_nonexistent_file(self):
        """Test unified ingestion with non-existent file."""
        result = ingest_document("nonexistent.txt")
        self.assertIsNone(result)
    
    def test_ingest_document_unsupported_type(self):
        """Test unified ingestion with unsupported file type."""
        file_path = os.path.join(self.test_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("Some content")
        
        result = ingest_document(file_path)
        self.assertIsNone(result)
    
    @patch('core.rag.ingest_pdf')
    def test_ingest_document_pdf(self, mock_ingest_pdf):
        """Test unified ingestion with PDF file."""
        mock_ingest_pdf.return_value = "PDF content"
        
        file_path = os.path.join(self.test_dir, "test.pdf")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_document(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['file_type'], '.pdf')
        self.assertEqual(result['content'], "PDF content")
        mock_ingest_pdf.assert_called_once()
    
    @patch('core.rag.ingest_docx')
    def test_ingest_document_docx(self, mock_ingest_docx):
        """Test unified ingestion with DOCX file."""
        mock_ingest_docx.return_value = "DOCX content"
        
        file_path = os.path.join(self.test_dir, "test.docx")
        with open(file_path, 'w') as f:
            f.write("dummy")
        
        result = ingest_document(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['file_type'], '.docx')
        self.assertEqual(result['content'], "DOCX content")
        mock_ingest_docx.assert_called_once()


class TestBatchProcessing(TestRAGIngestion):
    """Test cases for batch document processing."""
    
    def test_ingest_documents_from_directory_success(self):
        """Test successful batch ingestion from directory."""
        # Create test files
        self.create_sample_markdown("doc1.md")
        self.create_sample_markdown("doc2.markdown", "# Document 2\nContent here.")
        
        # Create a subdirectory with another file
        subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "doc3.md"), 'w') as f:
            f.write("# Document 3\nSubdirectory content.")
        
        results = ingest_documents_from_directory(self.test_dir)
        
        self.assertEqual(len(results), 3)
        filenames = [result['file_name'] for result in results]
        self.assertIn('doc1.md', filenames)
        self.assertIn('doc2.markdown', filenames)
        self.assertIn('doc3.md', filenames)
    
    def test_ingest_documents_from_directory_filtered(self):
        """Test batch ingestion with file type filtering."""
        # Create mixed file types
        self.create_sample_markdown("doc1.md")
        with open(os.path.join(self.test_dir, "doc2.pdf"), 'w') as f:
            f.write("dummy pdf")
        with open(os.path.join(self.test_dir, "doc3.txt"), 'w') as f:
            f.write("text file")
        
        # Only process markdown files
        results = ingest_documents_from_directory(self.test_dir, ['.md'])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_name'], 'doc1.md')
    
    def test_ingest_documents_from_directory_nonexistent(self):
        """Test batch ingestion from non-existent directory."""
        results = ingest_documents_from_directory("nonexistent_directory")
        self.assertEqual(results, [])
    
    def test_ingest_documents_from_directory_empty(self):
        """Test batch ingestion from empty directory."""
        results = ingest_documents_from_directory(self.test_dir)
        self.assertEqual(results, [])
    
    @patch('core.rag.ingest_document')
    def test_ingest_documents_partial_failures(self, mock_ingest_document):
        """Test batch ingestion with some files failing."""
        # Create test files
        self.create_sample_markdown("good.md")
        self.create_sample_markdown("bad.md")
        
        # Mock one success, one failure
        mock_ingest_document.side_effect = [
            {'file_name': 'good.md', 'content': 'Good content'},
            None  # Simulates failure
        ]
        
        results = ingest_documents_from_directory(self.test_dir)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_name'], 'good.md')


class TestErrorHandling(TestRAGIngestion):
    """Test cases for error handling and edge cases."""
    
    @patch('core.rag.ingest_markdown')
    def test_ingest_document_function_failure(self, mock_ingest_markdown):
        """Test unified ingestion when individual function fails."""
        mock_ingest_markdown.return_value = None
        
        file_path = self.create_sample_markdown()
        result = ingest_document(file_path)
        
        self.assertIsNone(result)
    
    def test_file_extension_case_insensitive(self):
        """Test that file extensions are handled case-insensitively."""
        file_path = self.create_sample_markdown("test.MD")  # Uppercase extension
        result = ingest_document(file_path)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['file_type'], '.MD')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
