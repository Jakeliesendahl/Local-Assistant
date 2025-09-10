"""
Pytest configuration and fixtures for the Local Assistant tests.
"""

import pytest
import tempfile
import shutil
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing and clean up after."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_markdown_content():
    """Provide sample markdown content for testing."""
    return """# Test Document

This is a test markdown document for testing purposes.

## Features
- Feature 1
- Feature 2
- Feature 3

## Code Example
```python
def hello():
    print("Hello, World!")
```

## Conclusion
This concludes our test document.
"""


@pytest.fixture
def create_test_file():
    """Factory fixture to create test files."""
    created_files = []
    
    def _create_file(directory, filename, content):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
        return file_path
    
    yield _create_file
    
    # Cleanup
    for file_path in created_files:
        if os.path.exists(file_path):
            os.remove(file_path)
