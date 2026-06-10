# AI-Semantic RAG Workspace Assistant

A production-ready, highly optimized FastAPI + LangGraph application integrating agentic workflows with isolated RAG workspaces. The system routes document analysis and retrieval tasks through high-precision semantic chunking and pgvector search.

### 🌐 Live Deployment URL
The application is configured to deploy to Render:
👉 **[https://production-langgraph-api.onrender.com/](https://production-langgraph-api.onrender.com/)**

---

## 🏗️ Architecture & Tech Stack

```
[ Frontend: HTML/CSS/JS Client ]
              │
              ▼
    [ API: FastAPI Server ] <────> [ Response Cache ]
              │
              ▼
    [ Agent: LangGraph ]
        ├── retrieve (pgvector / Cosine Distance)
        ├── process (Gemini 2.5 Flash / Primary Model)
        └── fallback (Secondary Model on failure)
              │
              ▼
   [ Vector Store & Database ]
        ├── PostgreSQL + pgvector (Production)
        └── Local JSON + InMemoryVectorStore (Local Fallback)
```

### Core Technologies
*   **Backend Framework:** FastAPI (Asynchronous request handling, Pydantic validation)
*   **Orchestration Engine:** LangGraph (Stateful agent loops, model fallback path, reducer-based memory)
*   **Vector Embeddings:** Hugging Face Serverless Inference API (`sentence-transformers/all-mpnet-base-v2` - 768 dimensions)
*   **LLM Reasoner:** Gemini 2.5 Flash (`gemini-2.5-flash`)
*   **Vector Database:** Supabase PostgreSQL with `pgvector`
*   **Observability:** LangSmith (Complete tracing and debugger logs)
*   **Security:** PII masking and regular expression inputs filter

### 📁 Project Structure & Modular Design
The codebase is refactored into modular, single-responsibility layers:
*   **[`app/db.py`](file:///c:/projects/myRAG/app/db.py):** Database Access Layer. Manages PostgreSQL connection context pools (`psycopg2`), transaction commit/rollback safety, default schema migration checks, UUID conversion utilities, and local file storage fallbacks (under `data/chats`).
*   **[`app/embeddings.py`](file:///c:/projects/myRAG/app/embeddings.py):** Embeddings Service Layer. Wraps Hugging Face Serverless Inference embedding generation with batching (sizes of 16), connection retries, server error exponential backoffs, and router-to-bypass-DNS overrides.
*   **[`app/document_processor.py`](file:///c:/projects/myRAG/app/document_processor.py):** Document Processor. Handles raw PDF parsing (`pdfplumber`), text scrubbing rules, and chunk segment splitters configured as lazy property getters (essential for unit testing mock scopes).
*   **[`app/rag.py`](file:///c:/projects/myRAG/app/rag.py):** RAG Coordinating Facade. Re-exposes public classes and methods (e.g. `RecursiveCharacterTextSplitter`, `ChatGoogleGenerativeAI`) to maintain complete backwards-compatibility with agent workflows and mock testing patch strings.

---

## 🌟 Key Features

### 1. Isolated RAG Workspaces
*   Supports creation, renaming, and deletion of custom chats.
*   Document indexing is strictly isolated at the workspace level.
*   Deleting a workspace cascades and automatically deletes all index chunks and vector embeddings.

### 2. Hierarchical Parent-Child Ingestion
*   **Semantic Cleansing:** Normalizes text (rebuilding word splittings and collapsing whitespace noise).
*   **Child Chunks (~300 chars):** Indexed in the vector store for granular, high-precision similarity searches.
*   **Parent Chunks (~1200 chars):** Returned as the actual context injected into the agent's prompts.
*   **Deduplication:** Merges redundant matches pointing to the same parent text block.

### 3. High-Performance API Client
*   **Embeddings Batching:** Processes chunks in sizes of 16 to minimize network roundtrips.
*   **Bulk Ingestion:** Inserts elements using database transactional statements (`execute_values`), completing 300+ chunk uploads under 1 minute.
*   **Retry and Fallback:** Incorporates exponential backoff and connection error routing to bypass DNS resolution issues.

---

## 🛠️ Local Development & Setup

### Prerequisites
*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (Fast package manager)

### 1. Installation
Clone the repository and sync dependencies:
```bash
uv sync
```

### 2. Environment Configuration
Create a `.env` file in the root directory (based on `.env.example`):
```ini
# Core LLM API Key
GEMINI_API_KEY=your-gemini-api-key

# Database Connection (Leave blank to use local JSON storage fallback)
DATABASE_URL=postgresql://postgres:password@localhost:5432/myrag

# Hugging Face Access Token for Embeddings
HF_TOKEN=your-huggingface-token

# LangSmith Observability Tracing
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=myRAG
```

### 3. Run Server
Start the local server:
```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Visit **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to access the web application.

---

## 🧪 Running Unit Tests

Run the complete suite of tests to verify api routes, cache keys, security levels, and database integrations:
```bash
uv run pytest
```

---

## 📋 API Endpoint Reference

### Workspace Operations
*   `GET /chats` - List all chat workspaces.
*   `POST /chats` - Create a new chat workspace.
*   `PUT /chats/{chat_id}` - Rename an existing workspace.
*   `DELETE /chats/{chat_id}` - Delete a workspace and its files.
*   `GET /chats/{chat_id}/history` - Fetch message logs for a workspace.

### Document Operations
*   `GET /documents?chat_id={chat_id}` - List all documents indexed in a workspace.
*   `POST /documents/upload?chat_id={chat_id}` - Upload a `.pdf`, `.txt`, or `.md` file for background chunking.
*   `DELETE /documents/{chat_id}/{doc_id}` - Remove a document and delete its vectors from index.

### Conversational Chat
*   `POST /chat` - Send a message to the agentic RAG loop.
    ```json
    {
      "message": "What is the project budget?",
      "chat_id": "workspace-uuid",
      "thread_id": "thread-uuid"
    }
    ```

### Monitoring & Operations
*   `GET /health` - Check health status for services.
*   `GET /metrics` - Performance dashboard metrics (latencies, token counts).
