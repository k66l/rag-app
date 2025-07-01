"""
Examples demonstrating how to use the logging library in different scenarios.

This file shows various ways to integrate logging into your modules
and functions throughout the RAG application.
"""

# Example 1: Using the function decorator
from app.utils.logger import log_function_call, LoggerMixin, inject_logger, get_logger


@log_function_call(log_args=True, log_performance=True)
def process_document(doc_id: str, content: str) -> dict:
    """
    Example function using the logging decorator.

    This will automatically log:
    - Function entry with arguments
    - Execution time
    - Return value (if enabled)
    - Any exceptions
    """
    # Simulate some processing
    import time
    time.sleep(0.1)

    return {
        "doc_id": doc_id,
        "status": "processed",
        "content_length": len(content)
    }


# Example 2: Using the LoggerMixin class
class DocumentProcessor(LoggerMixin):
    """
    Example service class using LoggerMixin for built-in logging methods.
    """

    def __init__(self):
        super().__init__()
        self.log_info("DocumentProcessor initialized")

    def validate_document(self, document: dict) -> bool:
        """Validate a document with automatic logging."""
        self.log_info("Starting document validation", extra_data={
            "doc_id": document.get("id"),
            "doc_type": document.get("type")
        })

        try:
            # Validation logic here
            if not document.get("content"):
                self.log_warning("Document has no content", extra_data={
                    "doc_id": document.get("id")
                })
                return False

            self.log_success("Document validation passed", extra_data={
                "doc_id": document.get("id"),
                "content_length": len(document.get("content", ""))
            })
            return True

        except Exception as e:
            self.log_error("Document validation failed", extra_data={
                "doc_id": document.get("id"),
                "error": str(e)
            }, exc_info=True)
            return False


# Example 3: Using the class decorator
@inject_logger
class SimpleProcessor:
    """
    Example class using the inject_logger decorator.
    """

    def process(self, data: str) -> str:
        """Process data with injected logger."""
        self.logger.info(f"Processing data of length {len(data)}")

        try:
            # Some processing logic
            result = data.upper()
            self.logger.info("Processing completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            raise


# Example 4: Manual logger usage
def manual_logging_example():
    """
    Example of manually getting and using a logger.
    """
    # Get a logger for this module
    logger = get_logger(__name__)

    logger.info("Starting manual logging example")

    try:
        # Some operation that might fail
        result = 10 / 2

        logger.info("Operation successful", extra={'extra_fields': {
            "result": result,
            "operation": "division",
            "event_type": "calculation_success"
        }})

        return result

    except Exception as e:
        logger.error("Operation failed", extra={'extra_fields': {
            "operation": "division",
            "error": str(e),
            "event_type": "calculation_error"
        }}, exc_info=True)
        raise


# Example 5: Async function logging
@log_function_call(log_args=True, log_performance=True)
async def async_process_data(data: list) -> dict:
    """
    Example async function with automatic logging.
    """
    import asyncio

    # Simulate async processing
    await asyncio.sleep(0.1)

    return {
        "processed_items": len(data),
        "status": "completed"
    }


# Example 6: Complex service with detailed logging
class ComplexDocumentService(LoggerMixin):
    """
    Example of a complex service with comprehensive logging.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.log_info("Service initialized", extra_data={
            "service": "ComplexDocumentService",
            "config_keys": list(config.keys())
        })

    @log_function_call(log_performance=True, log_result=False)
    def process_batch(self, documents: list) -> dict:
        """Process a batch of documents with detailed logging."""
        batch_id = f"batch_{hash(str(documents))}"

        self.log_info("Starting batch processing", extra_data={
            "batch_id": batch_id,
            "document_count": len(documents),
            "event_type": "batch_start"
        })

        results = {
            "processed": 0,
            "failed": 0,
            "errors": []
        }

        for i, doc in enumerate(documents):
            try:
                self.log_debug(f"Processing document {i+1}/{len(documents)}", extra_data={
                    "batch_id": batch_id,
                    "doc_index": i,
                    "doc_id": doc.get("id")
                })

                # Process individual document
                self._process_single_document(doc)
                results["processed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))

                self.log_error(f"Failed to process document {i+1}", extra_data={
                    "batch_id": batch_id,
                    "doc_index": i,
                    "doc_id": doc.get("id"),
                    "error": str(e)
                }, exc_info=True)

        self.log_info("Batch processing completed", extra_data={
            "batch_id": batch_id,
            "results": results,
            "event_type": "batch_complete"
        })

        return results

    def _process_single_document(self, doc: dict):
        """Process a single document (private method)."""
        # Simulate processing that might fail
        if not doc.get("content"):
            raise ValueError("Document content is empty")

        # Simulate some processing time
        import time
        time.sleep(0.01)


if __name__ == "__main__":
    # Demo the logging examples
    print("Running logging examples...")

    # Example 1: Function decorator
    result = process_document("doc123", "Sample content here")
    print(f"Function result: {result}")

    # Example 2: LoggerMixin
    processor = DocumentProcessor()
    is_valid = processor.validate_document({
        "id": "doc456",
        "type": "pdf",
        "content": "Document content here"
    })
    print(f"Validation result: {is_valid}")

    # Example 3: Class decorator
    simple = SimpleProcessor()
    processed = simple.process("hello world")
    print(f"Simple processing result: {processed}")

    # Example 4: Manual logging
    manual_result = manual_logging_example()
    print(f"Manual logging result: {manual_result}")

    # Example 6: Complex service
    service = ComplexDocumentService({"max_size": 1000, "format": "pdf"})
    batch_result = service.process_batch([
        {"id": "doc1", "content": "Content 1"},
        {"id": "doc2", "content": "Content 2"},
        {"id": "doc3", "content": ""}  # This will fail
    ])
    print(f"Batch processing result: {batch_result}")

    print("Logging examples completed. Check the logs for detailed output!")
