Local RAG Chatbot (Document Q&A)

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF, CSV, or DOCX) and ask natural-language questions about their contents. The system retrieves the most relevant sections of the document and generates accurate, context-aware answers in real time.

How it works:

Uploaded documents are parsed and split into semantic text chunks

Chunks are converted into vector embeddings and stored in a local vector database

User questions are matched against the most relevant chunks

A local large language model generates responses grounded in the retrieved content

Tech stack:

Streamlit for the interactive web UI

LangChain for RAG orchestration and conversational retrieval

ChromaDB as the local vector database

Ollama running a local LLM (e.g., Llama 3) for inference

HuggingFace sentence-transformers for local embeddings

Why it’s useful:

Enables private, offline document analysis (no API calls or data leakage)

Eliminates usage costs and quota limits by running fully locally

Makes large or complex documents easy to explore via conversational search

Demonstrates real-world GenAI patterns used in enterprise search, analytics, and knowledge systems

This project showcases practical experience with LLMs, vector search, and RAG architectures, making it directly relevant for roles in Data Science, Applied ML, Analytics Engineering, and GenAI development.
