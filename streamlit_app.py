"""
Streamlit Web UI for the RAG System

Upload PDFs and ask questions about them, all in the browser.
Reuses the existing DocumentLoader, VectorStore, and RAGChain classes.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

st.title("📚 RAG Document Q&A")
st.caption("Upload PDFs, then ask questions about them — answers are grounded in your documents with source citations.")

# Session state to keep things loaded across interactions
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# --- Sidebar: Upload and process PDFs ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing PDFs... this may take a minute."):
            # Save uploaded files to a temp directory so DocumentLoader can read them
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            try:
                # Step 1: Load and chunk documents
                loader = DocumentLoader()
                documents = loader.load_pdfs(temp_dir)
                chunks = loader.text_splitter.split_documents(documents)

                st.info(f"Loaded {len(documents)} pages → split into {len(chunks)} chunks")

                # Step 2: Create embeddings and store in vector DB
                vector_store = VectorStore()
                vector_store.create_vectorstore(chunks)
                st.session_state.vector_store = vector_store

                # Step 3: Build the RAG chain on top of the vector store
                st.session_state.rag_chain = RAGChain(vector_store)
                st.session_state.documents_loaded = True

                st.success(f"✅ Ready! Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

            except Exception as e:
                st.error(f"Error processing documents: {e}")

    if st.session_state.documents_loaded:
        st.success("Documents are loaded and ready for questions.")

# --- Main area: Ask questions ---
st.header("2. Ask a Question")

if not st.session_state.documents_loaded:
    st.warning("👈 Upload and process PDFs first using the sidebar.")
else:
    question = st.text_input("Your question:", placeholder="e.g. Summarize this document")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain.invoke_with_sources(question)

                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Sources")
                for i, doc in enumerate(result["sources"], 1):
                    source_name = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    with st.expander(f"Source {i}: {Path(source_name).name} (page {page})"):
                        st.text(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"Error generating answer: {e}")
