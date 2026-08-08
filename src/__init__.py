"""
LangChain RAG System Package

A beginner-friendly RAG system demonstrating:
- Document processing and chunking
- Vector embeddings with Ollama
- Semantic search with ChromaDB
- LLM generation with streaming
- LangSmith tracing for observability
"""

__version__ = "1.0.0"
__author__ = "LangChain Learning Project"

from src.config import config
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain, create_rag_chain
from src.streaming import (
    StreamingCallbackHandler,
    stream_chain_response,
    format_streaming_response
)

__all__ = [
    'config',
    'DocumentLoader',
    'VectorStore',
    'RAGChain',
    'create_rag_chain',
    'StreamingCallbackHandler',
    'stream_chain_response',
    'format_streaming_response',
]