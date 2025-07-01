# Global storage for the vector store
# This module prevents circular imports between main.py and routes

from app.utils.vector_store import RAGService

# Global RAG service instance
rag_service = RAGService()

# Backward compatibility
vector_store = None  # Keep for now but will be replaced by rag_service
