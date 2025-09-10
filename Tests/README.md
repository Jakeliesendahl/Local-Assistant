# Tests for Local Assistant

This directory contains comprehensive tests for the Local Assistant project.

## Test Structure

- `test_rag.py` - Comprehensive tests for the RAG document ingestion module
- `test_llm.py` - Tests for the LLM client module
- `run_tests.py` - Test runner script
- `conftest.py` - Pytest configuration and fixtures

## Running Tests

### Using the built-in test runner:
```bash
python Tests/run_tests.py
```

### Using pytest (recommended):
```bash
# Run all tests
pytest Tests/

# Run with verbose output
pytest Tests/ -v

# Run specific test file
pytest Tests/test_rag.py -v

# Run with coverage report
pytest Tests/ --cov=core --cov-report=html
```

### Using unittest:
```bash
# Run all tests
python -m unittest discover Tests/ -v

# Run specific test file
python -m unittest Tests.test_rag -v
```

## Test Coverage

The RAG tests (`test_rag.py`) provide comprehensive coverage including:

### Individual Function Tests
- **Markdown ingestion**: Success cases, empty files, encoding issues, special characters
- **PDF ingestion**: Success cases, missing dependencies, empty pages, extraction errors
- **DOCX ingestion**: Success cases, missing dependencies, tables, empty documents

### Unified Function Tests
- File type detection and routing
- Metadata extraction
- Error handling for unsupported types

### Batch Processing Tests
- Directory traversal and recursive processing
- File type filtering
- Partial failure handling

### Error Handling Tests
- Non-existent files
- Permission errors
- Malformed files
- Missing dependencies

## Test Dependencies

The tests use mocking to avoid requiring actual PDF/DOCX files and external dependencies:
- `unittest.mock` for mocking external libraries
- `tempfile` for creating temporary test files
- `pytest` for enhanced testing features (optional)

## Adding New Tests

When adding new functionality to the RAG module:

1. Add corresponding test methods to the appropriate test class
2. Use the existing fixtures and helper methods
3. Mock external dependencies to ensure tests run reliably
4. Test both success and failure cases
5. Update this README if new test patterns are introduced

## Test Data

Tests create temporary files and directories automatically. No external test data files are required.
