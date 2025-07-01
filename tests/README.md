# RAG Application API Test Suite

This directory contains comprehensive unit and integration tests for the RAG Application API using FastAPI's testing framework.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_health_check.py     # Health check endpoint tests
├── test_login.py           # Authentication endpoint tests  
├── test_upload.py          # Document upload endpoint tests
├── test_ask.py             # Question answering endpoint tests
├── test_ask_stream.py      # Streaming question answering tests
├── test_integration.py     # End-to-end integration tests
└── README.md              # This file
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Optional: Install coverage reporting
pip install pytest-cov
```

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type health

# Run with coverage
python run_tests.py --coverage

# Verbose output with fail-fast
python run_tests.py --verbose --fail-fast
```

### Manual pytest Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_login.py

# Run specific test class
pytest tests/test_ask.py::TestAsk

# Run specific test method
pytest tests/test_ask.py::TestAsk::test_ask_success

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run with verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

## Test Categories

### Unit Tests

**Health Check Tests** (`test_health_check.py`)
- ✅ Successful health check response
- ✅ Content type validation
- ✅ No authentication required
- ✅ Response schema validation
- ✅ Multiple request consistency

**Login Tests** (`test_login.py`)
- ✅ Successful login with valid credentials
- ✅ Login failure with invalid credentials
- ✅ Input validation (missing fields, invalid email format)
- ✅ JWT token validation
- ✅ Response schema validation
- ✅ Content type handling

**Upload Tests** (`test_upload.py`)
- ✅ Successful PDF upload and processing
- ✅ Authentication requirement
- ✅ File type validation (PDF only)
- ✅ Error handling (invalid PDF, processing failures)
- ✅ Response schema validation
- ✅ Large file handling
- ✅ Temporary file cleanup

**Ask Tests** (`test_ask.py`)
- ✅ Successful question answering
- ✅ Authentication requirement
- ✅ Input validation (question length, required fields)
- ✅ No documents error handling
- ✅ RAG service error handling
- ✅ Confidence level calculation
- ✅ Response schema validation
- ✅ Special character handling

**Streaming Ask Tests** (`test_ask_stream.py`)
- ✅ Successful streaming response
- ✅ Server-Sent Events format validation
- ✅ Progress tracking
- ✅ Multiple content chunks
- ✅ Error handling in streams
- ✅ Content type validation

### Integration Tests

**Full Workflow** (`test_integration.py`)
- ✅ Complete login → upload → ask flow
- ✅ Authentication enforcement across endpoints
- ✅ Error handling consistency
- ✅ Content type consistency
- ✅ Data consistency across endpoints

## Test Fixtures

The test suite uses several pytest fixtures defined in `conftest.py`:

### Core Fixtures
- `client`: FastAPI TestClient instance
- `auth_token`: Valid JWT token for authentication
- `auth_headers`: Authorization headers with valid token

### Mock Fixtures
- `mock_rag_service`: Mocked RAG service with predefined responses
- `mock_empty_rag_service`: RAG service with no documents
- `mock_pdf_loader`: Mocked PDF document loader
- `mock_text_splitter`: Mocked text splitting functionality

### Data Fixtures
- `sample_pdf_content`: Sample PDF content for upload tests
- `valid_credentials`: Valid login credentials
- `invalid_credentials`: Invalid login credentials for error testing

## Mocking Strategy

The tests use comprehensive mocking to isolate API logic from external dependencies:

1. **RAG Service**: Mocked to return predictable responses without requiring actual document processing
2. **PDF Processing**: Mocked to avoid file system dependencies
3. **Text Splitting**: Mocked to control document chunking behavior
4. **External APIs**: All external service calls are mocked

## Test Coverage

The test suite aims for high coverage of:

- **Happy path scenarios**: Normal operation with valid inputs
- **Error scenarios**: Invalid inputs, missing authentication, service failures
- **Edge cases**: Empty files, special characters, boundary conditions
- **Validation**: Input validation, response schema validation
- **Security**: Authentication and authorization enforcement

## Best Practices

### Test Organization
- Each endpoint has its own test file
- Tests are grouped by functionality in test classes
- Descriptive test names explain what is being tested

### Test Data
- Use fixtures for reusable test data
- Mock external dependencies consistently
- Create realistic but minimal test data

### Assertions
- Test both success and failure scenarios
- Validate response schemas thoroughly
- Check status codes, headers, and content

### Error Testing
- Test all validation scenarios
- Test authentication and authorization
- Test service failure scenarios

## Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    python run_tests.py --coverage
    
- name: Upload coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/rag-app

# Install in development mode
pip install -e .
```

**Async Test Issues**
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Add to pytest.ini
addopts = --asyncio-mode=auto
```

**Mock Issues**
- Ensure mocks are applied to the correct module path
- Use `patch.object()` for instance methods
- Reset mocks between tests using fixtures

### Debug Mode

```bash
# Run tests with Python debugger
pytest tests/ --pdb

# Run specific test with maximum verbosity
pytest tests/test_ask.py::TestAsk::test_ask_success -vvv
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate fixtures for new functionality
3. Test both success and failure scenarios
4. Update this README if adding new test categories
5. Ensure tests are isolated and don't depend on each other 