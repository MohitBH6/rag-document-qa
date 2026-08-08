"""
Vector Store Module

This module manages ChromaDB vector store operations including
storing embeddings and retrieving similar documents.
"""

import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.

    What is a Vector Store?
    - Stores document embeddings (numerical representations of text)
    - Enables fast similarity search
    - Essential for RAG systems

    Why ChromaDB?
    - Lightweight and easy to use
    - Persistent storage (survives restarts)
    - Fast similarity search
    - Great for local development
    """

    def __init__(self):
        """Initialize vector store with embedding model."""
      # Initialize local HuggingFace embeddings
        # This model converts text into high-dimensional vectors
        # that capture semantic meaning - runs locally, lightweight, no API needed
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )

        self.persist_directory = config.chroma_persist_directory
        self._vectorstore: Optional[Chroma] = None

        logger.info(
            f"VectorStore initialized with model: {config.embedding_model}"
        )

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.

        This method:
        1. Generates embeddings for each document chunk
        2. Stores embeddings in ChromaDB
        3. Persists to disk for future use

        Args:
            documents: List of chunked documents to embed

        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")

        logger.info(f"Creating vector store with {len(documents)} documents...")

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        try:
            # Create vector store with embeddings
            # This is where the "magic" happens:
            # 1. Each document is converted to an embedding vector
            # 2. Vectors are stored with metadata (source, page, etc.)
            # 3. Database is persisted to disk
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
                collection_name="research_papers"  # Name for this collection
            )

            logger.info(f"✅ Vector store created with {len(documents)} documents")
            logger.info(f"📁 Persisted to: {self.persist_directory}")

            return self._vectorstore

        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            raise

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk.

        This allows us to reuse previously created embeddings
        without reprocessing documents.

        Returns:
            Chroma vector store instance or None if not found
        """
        persist_path = Path(self.persist_directory)

        if not persist_path.exists():
            logger.warning(f"Vector store not found at: {self.persist_directory}")
            return None

        try:
            # Load existing vector store
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name="research_papers"
            )

            logger.info(f"✅ Vector store loaded from: {self.persist_directory}")
            return self._vectorstore

        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return None

    def similarity_search(
            self,
            query: str,
            k: int = None
    ) -> List[Document]:
        """
        Search for documents similar to the query.

        How it works:
        1. Query is converted to an embedding vector
        2. ChromaDB finds k most similar document vectors
        3. Returns the original text chunks

        This is the "retrieval" part of RAG!

        Args:
            query: Search query text
            k: Number of results to return (default from config)

        Returns:
            List of most similar documents with metadata
        """
        if self._vectorstore is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call create_vectorstore() or load_vectorstore() first."
            )

        if k is None:
            k = config.retrieval_k

        try:
            # Perform similarity search
            # This uses cosine similarity between embeddings
            results = self._vectorstore.similarity_search(
                query=query,
                k=k
            )

            logger.info(f"Found {len(results)} similar documents for query")
            logger.debug(f"Query: {query[:100]}...")

            return results

        except Exception as e:
            logger.error(f"❌ Error during similarity search: {e}")
            raise

    def similarity_search_with_score(
            self,
            query: str,
            k: int = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.

        Useful for understanding retrieval quality.
        Higher scores = more similar documents.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized")

        if k is None:
            k = config.retrieval_k

        try:
            results = self._vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            logger.info(f"Found {len(results)} documents with scores")
            for i, (doc, score) in enumerate(results, 1):
                logger.debug(f"  {i}. Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

            return results

        except Exception as e:
            logger.error(f"❌ Error during scored search: {e}")
            raise

    def get_retriever(self, k: int = None):
        """
        Get a retriever object for use in chains.

        Retrievers are LangChain components that implement
        a standard interface for document retrieval.

        Args:
            k: Number of documents to retrieve

        Returns:
            VectorStoreRetriever instance
        """
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized")

        if k is None:
            k = config.retrieval_k

        retriever = self._vectorstore.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={"k": k}  # Return k results
        )

        logger.info(f"Retriever created with k={k}")
        return retriever

    def delete_collection(self) -> None:
        """
        Delete the vector store collection.

        Useful for starting fresh with new documents.
        """
        try:
            if self._vectorstore is not None:
                self._vectorstore.delete_collection()
                self._vectorstore = None
                logger.info("✅ Vector store collection deleted")
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            raise