"""
Configuration Management Module

This module handles all configuration settings for the RAG system,
including environment variables, LangSmith setup, and model parameters.
"""

import logging
import os
from dotenv import load_dotenv

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_env_bool(key: str, default: bool) -> bool:
    """
    Helper to parse boolean environment variables.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


class Config:
    """
    Configuration class that loads and manages all settings.

    This centralizes configuration management and provides easy access
    to settings throughout the application.
    """

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Load environment variables from .env file
        load_dotenv()

        # LangSmith Configuration
        # These settings enable observability and tracing for debugging
        self.langchain_tracing = _get_env_bool("LANGCHAIN_TRACING_V2", False)
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "langchain-rag-learning")
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

        # Groq Configuration
        # Groq runs LLMs on their cloud servers - fast, free tier, no local GPU needed
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        # Embedding Model Configuration
        # Runs locally via sentence-transformers - lightweight, no API needed
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # ChromaDB Configuration
        # Vector database for storing document embeddings
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")

        # Document Processing Configuration
        # These parameters control how documents are split into chunks
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        # Retrieval Configuration
        # k = number of similar chunks to retrieve for each query
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))

        # Apply LangSmith configuration
        self._configure_langsmith()

        # Log configuration status
        self._log_config()

    def _configure_langsmith(self) -> None:
        """
        Configure LangSmith for tracing and observability.

        LangSmith provides powerful debugging capabilities by tracing
        every step of your LangChain execution.
        """
        if self.langchain_tracing:
            if not self.langchain_api_key:
                logger.warning(
                    "LangSmith tracing is enabled but no API key found. "
                    "Set LANGCHAIN_API_KEY in your .env file. "
                    "Get your key at https://smith.langchain.com"
                )
            else:
                # Set environment variables for LangSmith
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
                os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
                os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
                logger.info(f"✅ LangSmith tracing enabled for project: {self.langchain_project}")
        else:
            logger.info("ℹ️  LangSmith tracing disabled")

    def _log_config(self) -> None:
        """Log current configuration for debugging."""
        logger.info("=" * 60)
        logger.info("Configuration Loaded")
        logger.info("=" * 60)
        logger.info(f"Groq Model: {self.groq_model}")
        logger.info(f"Embedding Model: {self.embedding_model}")
        logger.info(f"ChromaDB Directory: {self.chroma_persist_directory}")
        logger.info(f"Chunk Size: {self.chunk_size}")
        logger.info(f"Chunk Overlap: {self.chunk_overlap}")
        logger.info(f"Retrieval K: {self.retrieval_k}")
        logger.info("=" * 60)

    def validate_groq_connection(self) -> bool:
        """
        Validate that a Groq API key is configured.

        Returns:
            True if a Groq API key is present, False otherwise
        """
        if self.groq_api_key:
            logger.info("✅ Groq API key found")
            return True
        else:
            logger.error("❌ No Groq API key found. Set GROQ_API_KEY in your .env file.")
            logger.error("Get a free key at https://console.groq.com")
            return False


# Global configuration instance
# Import this in other modules: from src.config import config
config = Config()