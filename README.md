# DealCraft AI

A full-stack RAG (Retrieval-Augmented Generation) web application for intelligent deal document analysis and search.

## Architecture

- **Frontend**: Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + PostgreSQL + ChromaDB + OpenAI
- **Features**: Document upload, semantic search, AI-powered Q&A, memo editing

## Quick Start (Docker)

```bash
# Clone and navigate to the project
git clone https://github.com/your-username/dealcraft.git
cd dealcraft

# Copy environment files and configure
cp .env.example .env
# Edit .env with your required API keys and credentials (see Configuration section)

# Start all services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.11+
- PostgreSQL
- OpenAI API key

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export DATABASE_URL="postgresql+asyncpg://username:password@localhost/dealcraft"
export CHROMA_PATH="./chroma"

# Run FastAPI server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Opens on http://localhost:3000
```

### Sample API Usage

```bash
# Upload documents
curl -X POST "http://localhost:8000/upload/" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "deal_id=1"

# Search documents
curl -X GET "http://localhost:8000/search/?query=What%20is%20the%20purchase%20price&deal_id=1"

# Save memo
curl -X POST "http://localhost:8000/memo/" \
  -H "Content-Type: application/json" \
  -d '{"deal_id": 1, "content": "Important notes about the deal", "title": "Key Points"}'
```

## RAG Workflow

1. **Document Ingestion**: PDFs are processed with pdfplumber, chunked into ~1000 token segments
2. **Embedding**: Text chunks are embedded using sentence-transformers (all-MiniLM-L6-v2)
3. **Storage**: Embeddings stored in ChromaDB, metadata in PostgreSQL
4. **Search**: User queries are embedded and matched via vector similarity (top-k=6)
5. **Generation**: Retrieved context is sent to OpenAI GPT for intelligent responses

## Project Structure

```
dealcraft/
├── frontend/          # Next.js 14 app
├── backend/           # FastAPI server
├── scripts/           # CLI tools
├── infra/            # Docker configuration
└── .env.example      # Environment template
```

## Configuration

### Required Environment Variables

Create a `.env` file in the root directory with the following variables:

**Essential:**
- `DATABASE_URL` - PostgreSQL connection string
- `GCP_PROJECT_ID` - Your Google Cloud Project ID
- `FIREBASE_PROJECT_ID` - Firebase project for authentication
- `SECRET_KEY` - Generate with `openssl rand -hex 32`

**Optional:**
- `OPENAI_API_KEY` - For legacy OpenAI integration
- `GOOGLE_SEARCH_API_KEY` - For web search functionality
- `GCS_BUCKET_NAME` - Cloud Storage bucket name

⚠️ **Security Note:** In production, always set `ALLOW_AUTH_FALLBACK=false`

See `.env.example` for complete configuration options.

## Development Notes

- Frontend uses Zustand for state management and React Query for API calls
- Backend implements async patterns throughout
- TipTap editor for rich memo editing
- shadcn/ui components for consistent design system
- ChromaDB for local vector storage, Vertex AI for production 