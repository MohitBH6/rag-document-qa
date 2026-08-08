"""
Document Loader Module

This module handles loading PDF documents and splitting them into
chunks suitable for embedding and retrieval.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading and processing of PDF documents.

    This class is responsible for:
    1. Loading PDFs from a directory
    2. Splitting documents into manageable chunks
    3. Preparing documents for embedding
    """

    def __init__(self):
        """Initialize the document loader with configuration."""
        # Text splitter configuration
        # RecursiveCharacterTextSplitter tries to keep semantically related
        # text together by splitting on paragraph breaks, then sentences
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # Maximum chunk size in characters
            chunk_overlap=config.chunk_overlap,  # Overlap between chunks
            length_function=len,  # Function to measure chunk size
            separators=["\n\n", "\n", " ", ""]  # Split hierarchy
        )

        logger.info(
            f"DocumentLoader initialized with chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}"
        )

    def load_pdfs(self, directory_path: str) -> List[Document]:
        """
        Load all PDF files from a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of Document objects with content and metadata

        Raises:
            ValueError: If directory doesn't exist or contains no PDFs
        """
        path = Path(directory_path)

        # Validate directory exists
        if not path.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Check for PDF files
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {directory_path}")

        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

        # Load all PDFs using DirectoryLoader
        # This is more efficient than loading files one by one
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",  # Pattern to match PDF files
                loader_cls=PyPDFLoader,  # Use PyPDFLoader for each file
                show_progress=True,  # Show loading progress
                use_multithreading=True  # Speed up loading
            )

            documents = loader.load()
            logger.info(f"✅ Successfully loaded {len(documents)} pages from PDFs")

            return documents

        except Exception as e:
            logger.error(f"❌ Error loading PDFs: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.

        Why chunking is important:
        1. LLMs have context limits (can't process entire books)
        2. Smaller chunks improve retrieval precision
        3. Embeddings work better on focused content

        The overlap ensures important context isn't lost at chunk boundaries.
        For example, if a sentence is split, the overlap might capture it.

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects with preserved metadata
        """
        if not documents:
            logger.warning("No documents to split")
            return []

        try:
            # Split documents while preserving metadata
            # Metadata includes: source file, page number, etc.
            chunks = self.text_splitter.split_documents(documents)

            logger.info(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")

            # Log sample chunk for verification
            if chunks:
                sample = chunks[0]
                logger.debug(f"Sample chunk length: {len(sample.page_content)} characters")
                logger.debug(f"Sample metadata: {sample.metadata}")

            return chunks

        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            raise

    def load_and_split(self, directory_path: str) -> List[Document]:
        """
        Convenience method to load and split in one call.

        This is the main entry point for document processing.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of chunked Document objects ready for embedding
        """
        logger.info(f"Loading and splitting documents from: {directory_path}")

        # Step 1: Load PDFs
        documents = self.load_pdfs(directory_path)

        # Step 2: Split into chunks
        chunks = self.split_documents(documents)

        logger.info(
            f"✅ Processing complete: {len(documents)} documents → "
            f"{len(chunks)} chunks"
        )

        return chunks

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks for analysis.

        Useful for tuning chunk_size and overlap parameters.

        Args:
            chunks: List of chunked documents

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {"count": 0}

        lengths = [len(chunk.page_content) for chunk in chunks]

        stats = {
            "count": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths),
        }

        logger.info(f"Chunk statistics: {stats}")
        return stats