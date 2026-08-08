"""
RAG Chain Module

This module implements the core RAG (Retrieval-Augmented Generation) chain
using LangChain Expression Language (LCEL) with streaming support.
"""

import logging

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.config import config
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Retrieval-Augmented Generation chain using LCEL.

    This class implements the RAG pattern:
    1. Retrieve: Find relevant documents from vector store
    2. Augment: Add retrieved context to the prompt
    3. Generate: LLM generates answer using context

    Key Features:
    - Built with LCEL for clean, declarative syntax
    - Streaming support out of the box
    - Automatic LangSmith tracing
    - Source citation in responses
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG chain with vector store.

        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store

      # Initialize ChatGroq
        self.llm = ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=0.7,  # Some creativity, but mostly factual
        )

        # Get retriever from vector store
        # The retriever will find relevant chunks for each query
        self.retriever = vector_store.get_retriever(k=config.retrieval_k)

        # Create the prompt template
        # This defines how we format the context and question for the LLM
        self.prompt = self._create_prompt()

        # Build the LCEL chain
        self.chain = self._build_chain()

        logger.info("✅ RAG chain initialized with streaming support")

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create the RAG prompt template.

        This is crucial for RAG quality. The prompt:
        1. Provides clear instructions
        2. Includes retrieved context
        3. Asks for source citations
        4. Sets expectations for handling unknown info

        Returns:
            ChatPromptTemplate instance
        """
        # System message sets the assistant's behavior
        system_template = """You are a helpful research assistant with expertise in analyzing academic papers.

Your task is to answer questions based ONLY on the provided context from research papers. 

Guidelines:
- Answer questions accurately using the context provided
- If the context doesn't contain enough information, say "I don't have enough information in the provided papers to answer that question."
- Be concise but thorough
- Cite specific papers when making claims
- Don't make up information not present in the context

Context from research papers:
{context}
"""

        # Human message contains the actual question
        human_template = """Based on the research papers above, please answer the following question:

{question}

Answer:"""

        # Combine into chat prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        logger.debug("Prompt template created")
        return prompt

    def _build_chain(self):
        """
        Build the RAG chain using LCEL.

        LCEL uses the pipe operator (|) to chain components:
        - First, we format inputs for the retriever and prompt
        - Then, we pass through the prompt template
        - Then through the LLM for generation
        - Finally through output parser to get clean text

        The beauty of LCEL:
        - Declarative syntax (describe what, not how)
        - Automatic streaming support
        - Built-in async support
        - Automatic LangSmith tracing

        Returns:
            Runnable chain
        """

        # Format the retrieved documents into a single context string
        def format_docs(docs):
            """Helper to format documents with source info."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content
                formatted.append(f"[Document {i} - {source}, page {page}]\n{content}")
            return "\n\n".join(formatted)

        # Build the LCEL chain
        # This is the modern LangChain way of building chains!

        # Wrapped in RunnableParallel to fix type errors
        inputs = RunnableParallel(
            context=(self.retriever | format_docs),
            question=RunnablePassthrough()
        )

        chain = (
                inputs
                | self.prompt  # Format with prompt template
                | self.llm  # Generate with LLM (streams automatically!)
                | StrOutputParser()  # Parse LLM output to string
        )

        logger.debug("LCEL chain built successfully")
        return chain

    def invoke(self, question: str) -> str:
        """
        Invoke the chain with a question (non-streaming).

        This waits for the complete response before returning.
        Use this when you don't need streaming.

        Args:
            question: User question

        Returns:
            Complete response string
        """
        try:
            logger.info(f"Invoking chain with question: {question[:100]}...")
            response = self.chain.invoke(question)
            logger.info("✅ Response generated successfully")
            return response
        except Exception as e:
            logger.error(f"❌ Error invoking chain: {e}")
            raise

    def stream(self, question: str):
        """
        Stream the chain response token-by-token.

        This yields chunks as they're generated, allowing for
        real-time display of the response.

        Args:
            question: User question

        Yields:
            Response chunks as they're generated
        """
        try:
            logger.info(f"Streaming response for question: {question[:100]}...")

            # Simply call .stream() on the LCEL chain
            # This works automatically because we built with LCEL!
            for chunk in self.chain.stream(question):
                yield chunk

            logger.info("✅ Streaming completed")

        except Exception as e:
            logger.error(f"❌ Error streaming response: {e}")
            raise

    def get_sources(self, question: str) -> list:
        """
        Get source documents for a question without generating answer.

        Useful for debugging retrieval quality.

        Args:
            question: User question

        Returns:
            List of retrieved documents
        """
        try:
            sources = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(sources)} source documents")
            return sources
        except Exception as e:
            logger.error(f"❌ Error retrieving sources: {e}")
            raise

    def invoke_with_sources(self, question: str) -> dict:
        """
        Invoke chain and return both answer and sources.

        This provides transparency by showing which documents
        contributed to the answer.

        Args:
            question: User question

        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        try:
            # Get sources first
            sources = self.get_sources(question)

            # Generate answer
            answer = self.invoke(question)

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"❌ Error in invoke_with_sources: {e}")
            raise

    def stream_with_sources(self, question: str):
        """
        Stream response and yield sources at the end.

        This combines streaming with source transparency.

        Args:
            question: User question

        Yields:
            Response chunks, followed by a final dict with sources
        """
        try:
            # Get sources first (doesn't stream)
            sources = self.get_sources(question)

            # Stream the answer
            for chunk in self.stream(question):
                yield chunk

            # Yield sources at the end
            yield {"sources": sources}

        except Exception as e:
            logger.error(f"❌ Error in stream_with_sources: {e}")
            raise

    def update_retrieval_k(self, k: int) -> None:
        """
        Update number of documents retrieved.

        Higher k = more context but slower and potentially noisier.
        Lower k = faster but might miss relevant info.

        Args:
            k: Number of documents to retrieve
        """
        self.retriever = self.vector_store.get_retriever(k=k)
        # Rebuild chain with new retriever
        self.chain = self._build_chain()
        logger.info(f"Updated retrieval k to {k}")


def create_rag_chain(vector_store: VectorStore) -> RAGChain:
    """
    Factory function to create a RAG chain.

    Args:
        vector_store: Initialized VectorStore instance

    Returns:
        RAGChain instance
    """
    return RAGChain(vector_store)