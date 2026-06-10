import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


# Clean data dir for tests
TEST_DATA_DIR = "data"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    yield
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


@pytest.fixture
def mock_dependencies():
    # Mock settings & API keys
    mock_settings = MagicMock()
    mock_settings.gemini_api_key = "test_api_key"
    mock_settings.primary_llm = "gemini-2.5-flash"
    mock_settings.fallback_llm = "gemini-2.5-flash"
    mock_settings.langchain_tracing_v2 = False
    mock_settings.langchain_api_key = ""
    mock_settings.langchain_project = "test-project"
    mock_settings.app_env = "development"
    mock_settings.log_level = "INFO"
    mock_settings.rate_limit = "100/minute"
    mock_settings.cache_ttl_seconds = 300
    mock_settings.max_retries = 3
    mock_settings.is_production = False
    
    # Mock embeddings to return predictable vectors
    mock_emb = MagicMock()
    mock_emb.embed_documents.side_effect = lambda texts: [[0.1] * 768] * len(texts)
    mock_emb.embed_query.return_value = [0.1] * 768
    
    # Mock LLM for LangGraph and Semantic Chunking
    from app.rag import ChunkingResponse, SemanticChunk
    mock_llm_response = ChunkingResponse(
        chunks=[SemanticChunk(chunk_text="Fake content snippet.", context="Fake context.")]
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_llm_response
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # Mock the LLM calls in Agent
    mock_agent_response = AIMessage(content="This is a grounded answer from the bot.")
    mock_llm.invoke.return_value = mock_agent_response
    
    with patch("app.main.get_settings", return_value=mock_settings), \
         patch("app.config.Settings", return_value=mock_settings), \
         patch("app.rag.get_settings", return_value=mock_settings), \
         patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_emb), \
         patch("app.rag.ChatGoogleGenerativeAI", return_value=mock_llm), \
         patch("app.agent.ChatGoogleGenerativeAI", return_value=mock_llm), \
         patch("app.rag.asyncio.sleep", new_callable=AsyncMock):
        yield


def test_workspaces_lifecycle_endpoint(mock_dependencies):
    from app.main import app
    
    # Using 'with TestClient' is required to properly trigger the lifespan startup/shutdown events
    with TestClient(app) as client:
        # 1. GET /chats - List chats (includes default Project Assistant)
        response = client.get("/chats")
        assert response.status_code == 200
        chats = response.json()
        assert len(chats) == 1
        assert chats[0]["chat_id"] != ""
        assert chats[0]["title"] == "New Chat"
        
        # 2. POST /chats - Create a new chat workspace
        response = client.post("/chats", json={"title": "Marketing Plan"})
        assert response.status_code == 200
        new_chat = response.json()
        assert new_chat["title"] == "Marketing Plan"
        chat_id = new_chat["chat_id"]
        
        # 3. GET /chats again - Confirm list has 2 entries
        response = client.get("/chats")
        chats = response.json()
        assert len(chats) == 2
        assert any(c["chat_id"] == chat_id for c in chats)
        
        # 4. DELETE /chats/{chat_id} - Delete the workspace
        response = client.delete(f"/chats/{chat_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
        
        # Verify it is deleted from list
        response = client.get("/chats")
        assert len(response.json()) == 1


def test_documents_upload_and_listing_endpoints(mock_dependencies):
    from app.main import app
    
    with TestClient(app) as client:
        # Create workspace
        chat = client.post("/chats", json={"title": "Analytics Workspace"}).json()
        chat_id = chat["chat_id"]
        
        # Upload text file
        response = client.post(
            f"/documents/upload?chat_id={chat_id}",
            files={"file": ("report.txt", b"This is clean data analysis text content.", "text/plain")}
        )
        assert response.status_code == 200
        assert "doc_id" in response.json()
        doc_id = response.json()["doc_id"]
        
        # GET /documents - Verify listing includes file
        response = client.get(f"/documents?chat_id={chat_id}")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "report.txt"
        
        # DELETE /documents/{chat_id}/{doc_id} - Verify deletion
        response = client.delete(f"/documents/{chat_id}/{doc_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
        
        # Verify list is empty
        response = client.get(f"/documents?chat_id={chat_id}")
        assert len(response.json()) == 0


def test_chat_interaction_endpoint(mock_dependencies):
    from app.main import app
    
    with TestClient(app) as client:
        # Send a chat message using the default chat_id
        response = client.post(
            "/chat",
            json={"message": "What is this project?", "thread_id": "thread_123", "chat_id": "default"}
        )
        assert response.status_code == 200
        res = response.json()
        assert "response" in res
        assert res["response"] == "This is a grounded answer from the bot."
        assert res["thread_id"] == "thread_123"
        assert "sources" in res
