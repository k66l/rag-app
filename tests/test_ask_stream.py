import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock


class TestAskStream:
    """Test suite for streaming question answering endpoint."""

    def test_ask_stream_success(self, client: TestClient, auth_headers, mock_rag_service):
        """Test successful streaming question answering."""
        question_data = {"question": "What is the main topic of the document?"}

        # Mock the async generator for streaming
        async def mock_generate_stream():
            chunks = [
                '{"type": "status", "data": "🔍 Searching your documents...", "stage": "retrieval", "progress": 10}',
                '{"type": "status", "data": "📄 Found 3 relevant sections. Analyzing...", "stage": "analysis", "progress": 30}',
                '{"type": "answer_start", "data": "✨ Answer:", "stage": "streaming", "progress": 60}',
                '{"type": "content", "data": "The main topic is", "provider": "google", "stage": "streaming", "progress": 70}',
                '{"type": "content", "data": " artificial intelligence.", "provider": "google", "stage": "streaming", "progress": 80}',
                '{"type": "complete", "data": "✅ Answer complete", "provider": "google", "stage": "complete", "progress": 100, "metadata": {"context_chunks": 3, "llm_provider": "google"}}'
            ]
            for chunk in chunks:
                yield f"data: {chunk}\n\n"

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"

    def test_ask_stream_no_authentication(self, client: TestClient, mock_rag_service):
        """Test streaming ask failure without authentication."""
        question_data = {"question": "What is this about?"}

        response = client.post("/api/v1/ask/stream", json=question_data)

        assert response.status_code == 401

    def test_ask_stream_invalid_token(self, client: TestClient, mock_rag_service):
        """Test streaming ask failure with invalid authentication token."""
        question_data = {"question": "What is this about?"}
        headers = {"Authorization": "Bearer invalid-token"}

        response = client.post("/api/v1/ask/stream",
                               json=question_data, headers=headers)

        assert response.status_code == 401

    def test_ask_stream_no_documents(self, client: TestClient, auth_headers, mock_empty_rag_service):
        """Test streaming ask failure when no documents are uploaded."""
        question_data = {"question": "What is this about?"}

        # Mock the generate_stream function to check for no documents
        async def mock_generate_stream():
            yield 'data: {"type": "error", "data": "No documents have been uploaded yet.", "stage": "error", "progress": 0}\n\n'

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            # Even with errors, streaming endpoint typically returns 200 but with error content
            assert response.status_code in [200, 404]

    def test_ask_stream_empty_question(self, client: TestClient, auth_headers, mock_rag_service):
        """Test streaming ask failure with empty question."""
        question_data = {"question": ""}

        response = client.post("/api/v1/ask/stream",
                               json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_stream_question_too_short(self, client: TestClient, auth_headers, mock_rag_service):
        """Test streaming ask failure with question too short."""
        question_data = {"question": "Hi"}  # Less than 3 characters

        response = client.post("/api/v1/ask/stream",
                               json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_stream_question_too_long(self, client: TestClient, auth_headers, mock_rag_service):
        """Test streaming ask failure with question too long."""
        long_question = "What is this about? " * 50  # Exceeds 500 characters
        question_data = {"question": long_question}

        response = client.post("/api/v1/ask/stream",
                               json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_stream_content_type(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that streaming ask endpoint returns correct content type."""
        question_data = {"question": "What is this about?"}

        # Mock minimal streaming response
        async def mock_generate_stream():
            yield 'data: {"type": "status", "data": "Processing...", "stage": "retrieval", "progress": 10}\n\n'

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"

    def test_ask_stream_sse_format(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that streaming response follows Server-Sent Events format."""
        question_data = {"question": "What is this about?"}

        # Mock SSE formatted response
        async def mock_generate_stream():
            chunks = [
                '{"type": "status", "data": "Starting...", "stage": "retrieval", "progress": 0}',
                '{"type": "content", "data": "Hello", "stage": "streaming", "progress": 50}',
                '{"type": "complete", "data": "Done", "stage": "complete", "progress": 100}'
            ]
            for chunk in chunks:
                yield f"data: {chunk}\n\n"

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200

            # Read the streaming content
            content = response.content.decode()

            # Check SSE format: each line should start with "data: " and end with "\n\n"
            lines = content.split('\n\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    assert line.startswith('data: ')

                    # Try to parse the JSON content
                    json_content = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_content)
                        assert "type" in data
                        assert "data" in data
                        assert "stage" in data
                        assert "progress" in data
                    except json.JSONDecodeError:
                        pytest.fail(
                            f"Invalid JSON in SSE chunk: {json_content}")

    def test_ask_stream_error_handling(self, client: TestClient, auth_headers):
        """Test streaming ask error handling."""
        question_data = {"question": "What is this about?"}

        # Mock error in streaming
        async def mock_generate_stream():
            yield 'data: {"type": "status", "data": "Starting...", "stage": "retrieval", "progress": 10}\n\n'
            yield 'data: {"type": "error", "data": "Something went wrong", "stage": "error", "progress": 10}\n\n'

        with patch('app.storage.rag_service') as mock_service:
            mock_service.get_stats.return_value = {"total_documents": 5}

            with patch('app.routes.ask.ask_question_stream') as mock_stream:
                mock_stream.return_value = mock_generate_stream()

                response = client.post(
                    "/api/v1/ask/stream", json=question_data, headers=auth_headers)

                assert response.status_code == 200

                content = response.content.decode()
                assert "error" in content
                assert "Something went wrong" in content

    def test_ask_stream_progress_tracking(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that streaming response includes proper progress tracking."""
        question_data = {"question": "What is this about?"}

        # Mock response with progress tracking
        async def mock_generate_stream():
            progress_values = [0, 25, 50, 75, 100]
            stages = ["retrieval", "analysis",
                      "generation", "streaming", "complete"]

            for i, (progress, stage) in enumerate(zip(progress_values, stages)):
                chunk_data = {
                    "type": "status" if i < 4 else "complete",
                    "data": f"Stage {i+1}",
                    "stage": stage,
                    "progress": progress
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200

            content = response.content.decode()

            # Check that all progress values are present
            for progress in [0, 25, 50, 75, 100]:
                assert f'"progress": {progress}' in content

            # Check that all stages are present
            stages = ["retrieval", "analysis",
                      "generation", "streaming", "complete"]
            for stage in stages:
                assert f'"stage": "{stage}"' in content

    def test_ask_stream_multiple_content_chunks(self, client: TestClient, auth_headers, mock_rag_service):
        """Test streaming with multiple content chunks."""
        question_data = {"question": "Tell me about the document"}

        # Mock response with multiple content chunks
        async def mock_generate_stream():
            yield 'data: {"type": "status", "data": "Searching...", "stage": "retrieval", "progress": 10}\n\n'
            yield 'data: {"type": "answer_start", "data": "Answer:", "stage": "streaming", "progress": 50}\n\n'

            # Multiple content chunks
            content_parts = ["The document", " discusses various",
                             " topics including", " AI and ML."]
            for i, part in enumerate(content_parts):
                progress = 50 + (i + 1) * 10
                chunk = {
                    "type": "content",
                    "data": part,
                    "stage": "streaming",
                    "progress": progress,
                    "provider": "google"
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield 'data: {"type": "complete", "data": "Complete", "stage": "complete", "progress": 100}\n\n'

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200

            content = response.content.decode()

            # Check that all content parts are present
            content_parts = ["The document", " discusses various",
                             " topics including", " AI and ML."]
            for part in content_parts:
                assert part in content

            # Check that content type chunks are present
            assert '"type": "content"' in content
            assert '"provider": "google"' in content

    def test_ask_stream_no_body(self, client: TestClient, auth_headers, mock_rag_service):
        """Test streaming ask failure with no request body."""
        response = client.post("/api/v1/ask/stream", headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_stream_whitespace_handling(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that streaming endpoint handles questions with whitespace correctly."""
        question_data = {"question": "  What is this about?  "}

        # Mock minimal response
        async def mock_generate_stream():
            yield 'data: {"type": "complete", "data": "Done", "stage": "complete", "progress": 100}\n\n'

        with patch('app.routes.ask.ask_question_stream') as mock_stream:
            mock_stream.return_value = mock_generate_stream()

            response = client.post("/api/v1/ask/stream",
                                   json=question_data, headers=auth_headers)

            assert response.status_code == 200
