from typing import List, Optional
from abc import ABC, abstractmethod
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService(ABC):
    """Abstract base class for embedding services"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass


class LocalEmbeddingService(EmbeddingService):
    """Local embedding service using sentence-transformers"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL or "all-MiniLM-L6-v2"
        self._model = None
        logger.info(
            f"Initializing local embedding service with model: {self.model_name}")

    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(
                    f"Loading sentence transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {e}")
                raise
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using local model"""
        try:
            logger.info(f"Embedding {len(texts)} documents locally")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents locally: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using local model"""
        try:
            logger.debug("Embedding query locally")
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query locally: {e}")
            raise


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service"""

    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required for OpenAI embedding service")

        logger.info(
            f"Initializing OpenAI embedding service with model: {self.model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using OpenAI API"""
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.info(f"Embedding {len(texts)} documents with OpenAI")
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model=self.model
            )
            result = embeddings.embed_documents(texts)
            logger.info("OpenAI embedding completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error embedding documents with OpenAI: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using OpenAI API"""
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.debug("Embedding query with OpenAI")
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model=self.model
            )
            result = embeddings.embed_query(text)
            return result
        except Exception as e:
            logger.error(f"Error embedding query with OpenAI: {e}")
            raise


class EmbeddingServiceFactory:
    """Factory to create the appropriate embedding service"""

    @staticmethod
    def create_embedding_service() -> EmbeddingService:
        """Create embedding service based on configuration"""
        # Check if we should use local embeddings (default)
        use_local = getattr(settings, 'USE_LOCAL_EMBEDDINGS', True)

        if use_local:
            logger.info("Creating local embedding service")
            return LocalEmbeddingService()

        # Check for OpenAI if local is disabled
        if settings.OPENAI_API_KEY:
            logger.info("Creating OpenAI embedding service")
            try:
                return OpenAIEmbeddingService()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize OpenAI embeddings, falling back to local: {e}")
                return LocalEmbeddingService()

        # Default fallback to local
        logger.info(
            "No specific embedding service configured, defaulting to local")
        return LocalEmbeddingService()


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingServiceFactory.create_embedding_service()
    return _embedding_service


def reset_embedding_service():
    """Reset the global embedding service (useful for testing)"""
    global _embedding_service
    _embedding_service = None
