import os
import pytest
import shutil
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from app.rag import RAGManager, ChunkingResponse, SemanticChunk

TEST_DATA_DIR = "data"

@pytest.fixture(autouse=True)
def setup_and_teardown(tmp_path, monkeypatch):
    # Wrap RAGManager.__init__ to use a unique temp directory for base_dir
    original_init = RAGManager.__init__
    def patched_init(self, *args, **kwargs):
        if 'base_dir' not in kwargs and len(args) == 0:
            kwargs['base_dir'] = str(tmp_path)
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(RAGManager, "__init__", patched_init)
    
    yield


@pytest.fixture
def mock_embeddings():
    # Mock embeddings to return text-aware vectors for predictable retrieval sorting
    embeddings = MagicMock()
    embeddings.embed_documents.side_effect = lambda texts: [
        ([1.0] + [0.0] * 767) if "FastAPI" in text else ([0.0, 1.0] + [0.0] * 766)
        for text in texts
    ]
    embeddings.embed_query.side_effect = lambda query: (
        ([1.0] + [0.0] * 767) if "FastAPI" in query else ([0.0, 1.0] + [0.0] * 766)
    )
    return embeddings


def test_rag_manager_registry_init(mock_embeddings):
    """Verify that directories and default chats.json are created on initialization."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            assert os.path.exists(manager.chats_dir)
            assert os.path.exists(os.path.join(manager.chats_dir, "chats.json"))
            
            chats = manager.load_chats()
            assert len(chats) == 1
            assert chats[0]["chat_id"] != ""
            assert chats[0]["title"] == "New Chat"


def test_workspace_management(mock_embeddings):
    """Verify creating, listing, and deleting workspaces."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            
            # Create two new chats
            manager.create_chat("chat_1", "Chat One")
            manager.create_chat("chat_2", "Chat Two")
            
            chats = manager.load_chats()
            assert len(chats) == 3  # default, chat_1, chat_2
            assert any(c["chat_id"] == "chat_1" for c in chats)
            assert any(c["chat_id"] == "chat_2" for c in chats)
            
            # Delete one chat
            manager.delete_chat("chat_1")
            chats_after = manager.load_chats()
            assert len(chats_after) == 2
            assert not any(c["chat_id"] == "chat_1" for c in chats_after)
            assert not os.path.exists("data/chats/chat_1")


def test_metadata_persistence(mock_embeddings):
    """Verify document metadata loading and saving per workspace."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            manager.create_chat("chat_a", "Workspace A")
            
            # Save document metadata
            doc_data = {
                "doc_1": {
                    "id": "doc_1",
                    "filename": "file.pdf",
                    "status": "Indexed",
                    "char_count": 500,
                    "chunk_count": 2,
                    "uploaded_at": "2026-06-09T11:00:00Z"
                }
            }
            manager.save_documents("chat_a", doc_data)
            
            # Load and verify
            loaded = manager.load_documents("chat_a")
            assert "doc_1" in loaded
            assert loaded["doc_1"]["filename"] == "file.pdf"
            assert loaded["doc_1"]["status"] == "Indexed"


@pytest.mark.anyio
async def test_document_processing_lifecycle(mock_embeddings):
    """Verify async batching, parent-child chunking, embedding generation, and status updates."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_batch"
            manager.create_chat(chat_id, "Batch Chat")
            
            # Setup RecursiveCharacterTextSplitter mocks
            with patch("app.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls:
                mock_parent_splitter = MagicMock()
                mock_parent_splitter.split_text.return_value = ["Parent Chunk 1", "Parent Chunk 2"]
                
                mock_child_splitter = MagicMock()
                mock_child_splitter.split_text.side_effect = lambda t: [f"Child 1 of {t}", f"Child 2 of {t}"]
                
                mock_splitter_cls.side_effect = [mock_parent_splitter, mock_child_splitter]
                
                with patch("asyncio.sleep", return_value=None):  # bypass throttling delay in test
                    file_content = b"Some file contents to process."
                    await manager.process_document_background(
                        chat_id=chat_id,
                        file_content=file_content,
                        filename="test.txt",
                        doc_id="doc_123"
                    )
            
            # Verify status and chunk count
            docs = manager.load_documents(chat_id)
            assert "doc_123" in docs
            assert docs["doc_123"]["status"] == "Indexed"
            # 2 parents * 2 children = 4 chunks
            assert docs["doc_123"]["chunk_count"] == 4
            
            # Check vectorstore entries
            vs = manager.load_vectorstore(chat_id)
            assert len(vs.store) == 4


def test_vector_search_and_isolation(mock_embeddings):
    """Verify document retrieval matching and strict workspace data isolation."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            
            # Create two isolated workspaces
            manager.create_chat("workspace_1", "Workspace 1")
            manager.create_chat("workspace_2", "Workspace 2")
            
            # Manually inject documents into workspace_1's vector store
            store1 = manager.load_vectorstore("workspace_1")
            docs_to_add = [
                Document(page_content="[Context: FastAPI info]\n\nFastAPI is a modern web framework.", metadata={"source": "doc.txt"}),
                Document(page_content="[Context: LangGraph info]\n\nLangGraph is stateful.", metadata={"source": "doc.txt"})
            ]
            store1.add_documents(docs_to_add)
            manager.save_vectorstore("workspace_1", store1)
            
            # 1. Verify similarity search works in workspace_1
            results = manager.retrieve("workspace_1", "Tell me about FastAPI", k=1)
            assert len(results) == 1
            assert "FastAPI is a modern web framework" in results[0]["content"]
            assert results[0]["source"] == "doc.txt"
            
            # 2. Verify workspace isolation: search in workspace_2 must be empty
            results_empty = manager.retrieve("workspace_2", "Tell me about FastAPI", k=1)
            assert len(results_empty) == 0
            
            # 3. Verify search in empty vectorstore returns empty list
            manager.create_chat("workspace_empty", "Empty")
            results_empty_store = manager.retrieve("workspace_empty", "FastAPI", k=1)
            assert results_empty_store == []


def test_document_deletion(mock_embeddings):
    """Verify deleting a single document and its associated chunks from a workspace vector store."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_delete"
            manager.create_chat(chat_id, "Delete Test Chat")
            
            # Manually inject two documents with different doc_ids
            store = manager.load_vectorstore(chat_id)
            docs_to_add = [
                Document(page_content="Content of Doc A", metadata={"doc_id": "doc_a", "source": "a.txt"}),
                Document(page_content="Content of Doc B", metadata={"doc_id": "doc_b", "source": "b.txt"})
            ]
            store.add_documents(docs_to_add)
            manager.save_vectorstore(chat_id, store)
            
            # Save metadata
            manager.save_documents(chat_id, {
                "doc_a": {"id": "doc_a", "filename": "a.txt", "status": "Indexed"},
                "doc_b": {"id": "doc_b", "filename": "b.txt", "status": "Indexed"}
            })
            
            # Verify they are present
            assert len(manager.load_documents(chat_id)) == 2
            assert len(manager.load_vectorstore(chat_id).store) == 2
            
            # Delete doc_a
            manager.delete_document(chat_id, "doc_a")
            
            # Verify doc_a is deleted from documents registry and vector store
            docs_metadata = manager.load_documents(chat_id)
            assert "doc_a" not in docs_metadata
            assert "doc_b" in docs_metadata
            assert len(docs_metadata) == 1
            
            # Verify doc_a's vectors are deleted, but doc_b's vectors remain
            vs = manager.load_vectorstore(chat_id)
            assert len(vs.store) == 1
            entry = list(vs.store.values())[0]
            assert entry["metadata"]["doc_id"] == "doc_b"


@pytest.mark.anyio
async def test_document_processing_text_file(mock_embeddings):
    """Verify that plain text (.txt) files are chunked and indexed correctly."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_txt"
            manager.create_chat(chat_id, "Text File Chat")
            
            mock_llm_response = ChunkingResponse(
                chunks=[
                    SemanticChunk(chunk_text="This is a text file chunk.", context="Text file context.")
                ]
            )
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_llm_response
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            
            with patch("app.rag.ChatGoogleGenerativeAI", return_value=mock_llm):
                with patch("asyncio.sleep", return_value=None):
                    # Process plain text file
                    await manager.process_document_background(
                        chat_id=chat_id,
                        file_content=b"Sample plain text document contents.",
                        filename="document.txt",
                        doc_id="doc_txt_001"
                    )
            
            docs = manager.load_documents(chat_id)
            assert docs["doc_txt_001"]["status"] == "Indexed"
            assert docs["doc_txt_001"]["char_count"] == len("Sample plain text document contents.")
            
            vs = manager.load_vectorstore(chat_id)
            assert len(vs.store) == 1


@pytest.mark.anyio
async def test_document_processing_empty_text_failure(mock_embeddings):
    """Verify that uploading an empty file fails gracefully with a status update."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_empty_fail"
            manager.create_chat(chat_id, "Empty Fail Chat")
            
            await manager.process_document_background(
                chat_id=chat_id,
                file_content=b"",  # Empty content
                filename="empty.txt",
                doc_id="doc_empty"
            )
            
            docs = manager.load_documents(chat_id)
            assert "doc_empty" in docs
            assert "Failed:" in docs["doc_empty"]["status"]
            assert "No extractable text" in docs["doc_empty"]["status"]


@pytest.mark.anyio
async def test_document_processing_llm_exception_failure(mock_embeddings):
    """Verify that if the text splitter fails, the background job handles the error and registers status as Failed."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_llm_fail"
            manager.create_chat(chat_id, "LLM Fail Chat")
            
            # Mock RecursiveCharacterTextSplitter to throw an Exception
            with patch("app.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls:
                mock_splitter = MagicMock()
                mock_splitter.split_text.side_effect = Exception("Splitter error.")
                mock_splitter_cls.return_value = mock_splitter
                
                await manager.process_document_background(
                    chat_id=chat_id,
                    file_content=b"Some document content.",
                    filename="doc.txt",
                    doc_id="doc_fail"
                )
            
            docs = manager.load_documents(chat_id)
            assert "doc_fail" in docs
            assert "Failed:" in docs["doc_fail"]["status"]
            assert "Splitter error" in docs["doc_fail"]["status"]


@pytest.mark.anyio
async def test_document_processing_pdf_file(mock_embeddings):
    """Verify that PDF files are parsed using pdfplumber, chunked, and indexed correctly."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_pdf"
            manager.create_chat(chat_id, "PDF File Chat")
            
            # Mock pdfplumber pages list
            mock_page1 = MagicMock()
            mock_page1.extract_text.return_value = "Page 1 text. " + ("A" * 1000) + " Page 1 end."
            
            mock_page2 = MagicMock()
            mock_page2.extract_text.return_value = "Page 2 text. " + ("B" * 1000) + " Page 2 end."
            
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page1, mock_page2]
            
            # Mock pdfplumber.open context manager
            mock_open = MagicMock()
            mock_open.__enter__.return_value = mock_pdf
            
            with patch("pdfplumber.open", return_value=mock_open):
                with patch("asyncio.sleep", return_value=None):
                    # Process a PDF file
                    await manager.process_document_background(
                        chat_id=chat_id,
                        file_content=b"fake pdf content",
                        filename="document.pdf",
                        doc_id="doc_pdf_001"
                    )
            
            docs = manager.load_documents(chat_id)
            assert docs["doc_pdf_001"]["status"] == "Indexed"
            assert docs["doc_pdf_001"]["chunk_count"] == 13
            
            vs = manager.load_vectorstore(chat_id)
            assert len(vs.store) == 13


def test_db_retrieval(mock_embeddings):
    """Verify that when database is enabled, retrieve queries the DB using the cosine distance operator."""
    import json
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "postgresql://mock_user:mock_pass@mock_host:5432/mock_db"
        mock_settings.return_value.hf_token = "mock_hf_token"
        
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            
            # Setup mock database response
            # Columns: content, metadata, distance
            mock_cursor.fetchall.return_value = [
                ("FastAPI is a modern web framework.", json.dumps({"source": "fastapi.txt", "pages": [1]}), 0.2),
                ("LangGraph is for stateful agents.", json.dumps({"source": "langgraph.txt", "pages": [2, 3]}), 0.5)
            ]
            
            manager = RAGManager()
            assert manager.use_db is True
            
            with patch.object(manager, "get_conn") as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn
                
                results = manager.retrieve("default", "What is FastAPI?", k=2)
                
                # Verify cursor executed query
                mock_cursor.execute.assert_called_once()
                call_args = mock_cursor.execute.call_args[0]
                assert "embedding <=> %s::vector" in call_args[0]
                assert "document_embeddings" in call_args[0]
                
                # Check results
                assert len(results) == 2
                assert results[0]["content"] == "FastAPI is a modern web framework."
                assert results[0]["source"] == "fastapi.txt"
                assert results[0]["pages"] == [1]
                assert results[0]["score"] == 90
                
                assert results[1]["content"] == "LangGraph is for stateful agents."
                assert results[1]["source"] == "langgraph.txt"
                assert results[1]["pages"] == [2, 3]
                assert results[1]["score"] == 75


def test_history_persistence_file(mock_embeddings):
    """Verify that in file-based mode, messages are successfully added and loaded from history.json."""
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = "test_api_key"
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            manager = RAGManager()
            chat_id = "test_chat_hist_file"
            manager.create_chat(chat_id, "History Test File")
            
            # Save messages
            msg1 = manager.add_message_to_history(chat_id, "user", "Hello assistant", sources=[])
            msg2 = manager.add_message_to_history(chat_id, "bot", "Hello human", sources=[{"source": "doc1.txt", "score": 95}])
            
            # Load and assert
            history = manager.load_history(chat_id)
            assert len(history) == 2
            assert history[0]["sender"] == "user"
            assert history[0]["text"] == "Hello assistant"
            assert history[1]["sender"] == "bot"
            assert history[1]["text"] == "Hello human"
            assert len(history[1]["sources"]) == 1
            assert history[1]["sources"][0]["source"] == "doc1.txt"
            
            # Verify file exists
            path = manager.get_history_path(chat_id)
            assert os.path.exists(path)
            
            # Verify cascading deletion under file-based mode
            manager.delete_chat(chat_id)
            assert not os.path.exists(path)


def test_history_persistence_db(mock_embeddings):
    """Verify that in database mode, messages are saved/loaded through database SQL execute queries."""
    import json
    with patch("app.rag.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "postgresql://mock_user:mock_pass@mock_host:5432/mock_db"
        mock_settings.return_value.hf_token = "mock_hf_token"
        
        with patch("app.rag.HuggingFaceInferenceAPIEmbeddings", return_value=mock_embeddings):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            
            # Setup mock database response for load_history
            from datetime import datetime, timezone
            mock_time = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
            mock_cursor.fetchall.return_value = [
                ("user", "Hello assistant", None, mock_time),
                ("bot", "Hello human", json.dumps([{"source": "doc1.txt"}]), mock_time)
            ]
            
            manager = RAGManager()
            assert manager.use_db is True
            
            with patch.object(manager, "get_conn") as mock_get_conn:
                mock_get_conn.return_value.__enter__.return_value = mock_conn
                
                # Test load_history
                history = manager.load_history("test_chat_hist_db")
                assert len(history) == 2
                assert history[0]["sender"] == "user"
                assert history[0]["text"] == "Hello assistant"
                assert history[1]["sender"] == "bot"
                assert history[1]["sources"] == [{"source": "doc1.txt"}]
                assert history[1]["timestamp"] == mock_time.isoformat()
                
                # Test add_message_to_history
                mock_cursor.fetchone.return_value = (mock_time,)
                manager.add_message_to_history("test_chat_hist_db", "user", "New test message")
                mock_cursor.execute.assert_called_with(
                    "INSERT INTO chat_messages (chat_id, sender, text, sources) VALUES (%s, %s, %s, %s) RETURNING timestamp;",
                    ("test_chat_hist_db", "user", "New test message", "[]")
                )



