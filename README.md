# DealCraft AI

A full-stack RAG (Retrieval-Augmented Generation) web application for intelligent deal document analysis and search.

## Architecture

- **Frontend**: Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + PostgreSQL + ChromaDB + Gemini 2.5-flash
- **Features**: Document upload, semantic search, AI-powered Q&A, memo editing

## RAG Workflow

1. **Document Ingestion**: PDFs are processed with pdfplumber, chunked into ~1000 token segments
2. **Embedding**: Text chunks are embedded using sentence-transformers (all-MiniLM-L6-v2)
3. **Storage**: Embeddings stored in ChromaDB, metadata in PostgreSQL
4. **Search**: User queries are embedded and matched via vector similarity (top-k=6)
5. **Generation**: Retrieved context is sent to OpenAI GPT for intelligent responses

Head to dealcraft.info to try out the system with your own CIM and financial documents
