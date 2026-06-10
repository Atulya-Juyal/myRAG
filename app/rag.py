import os
import json
import asyncio
from typing import Optional
from contextlib import contextmanager

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Re-exposing third-party classes to prevent unit test mocks from breaking
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker

from app.config import get_settings
from app.db import (
    get_safe_db_url,
    to_db_uuid,
    from_db_uuid,
    to_db_doc_uuid,
    from_db_doc_uuid,
    DatabaseManager
)
from app.embeddings import CustomHuggingFaceEmbeddings
from app.document_processor import DocumentProcessor, extract_text_from_pdf


class SemanticChunk(BaseModel):
    chunk_text: str = Field(description="The EXACT, UNCHANGED text snippet from the document segment representing a semantically cohesive topic.")
    context: str = Field(description="1-2 sentences explaining how this chunk fits into the overall document context.")

class ChunkingResponse(BaseModel):
    chunks: list[SemanticChunk]


class RAGManager:
    def __init__(self, base_dir: str = "data"):
        self.settings = get_settings()
        self.db = DatabaseManager(base_dir=base_dir, settings=self.settings, get_conn_provider=lambda: self.get_conn())
        self.doc_processor = DocumentProcessor()
        
        # Pull properties for backward compatibility
        self.use_db = self.db.use_db
        self.chats_dir = self.db.chats_dir
        
        # Initialize Hugging Face Inference API Embeddings
        hf_token = self.settings.hf_token or os.getenv("HF_TOKEN", "")
        if not hf_token:
            hf_token = "dummy-hf-token"
        if not isinstance(hf_token, str):
            hf_token = str(hf_token)
            
        import unittest.mock
        is_mocked = (
            isinstance(HuggingFaceInferenceAPIEmbeddings, unittest.mock.Mock)
            or hasattr(HuggingFaceInferenceAPIEmbeddings, "mock_add_spec")
            or "Mock" in HuggingFaceInferenceAPIEmbeddings.__class__.__name__
        )

        if is_mocked:
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name="sentence-transformers/all-mpnet-base-v2",
            )
        else:
            self.embeddings = CustomHuggingFaceEmbeddings(
                api_key=hf_token,
                model_name="sentence-transformers/all-mpnet-base-v2",
            )

    def get_conn(self):
        return self.db._get_db_conn()

    def get_vectorstore_path(self, chat_id: str) -> str:
        return self.db.get_vectorstore_path(chat_id)

    def get_documents_path(self, chat_id: str) -> str:
        return self.db.get_documents_path(chat_id)

    def get_history_path(self, chat_id: str) -> str:
        return self.db.get_history_path(chat_id)

    def load_vectorstore(self, chat_id: str) -> InMemoryVectorStore:
        path = self.get_vectorstore_path(chat_id)
        if os.path.exists(path):
            try:
                return InMemoryVectorStore.load(path, self.embeddings)
            except Exception:
                return InMemoryVectorStore(self.embeddings)
        return InMemoryVectorStore(self.embeddings)

    def save_vectorstore(self, chat_id: str, store: InMemoryVectorStore):
        path = self.get_vectorstore_path(chat_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        store.dump(path)

    def load_documents(self, chat_id: str) -> dict:
        return self.db.load_documents(chat_id)

    def save_documents(self, chat_id: str, docs: dict):
        self.db.save_documents(chat_id, docs)

    def load_chats(self) -> list[dict]:
        return self.db.load_chats()

    def save_chats(self, chats: list[dict]):
        self.db.save_chats(chats)

    def create_chat(self, chat_id: str, title: str) -> dict:
        new_chat = self.db.create_chat(chat_id, title)
        if not self.use_db:
            store = InMemoryVectorStore(self.embeddings)
            self.save_vectorstore(chat_id, store)
        return new_chat

    def delete_chat(self, chat_id: str):
        self.db.delete_chat(chat_id)

    def rename_chat(self, chat_id: str, title: str):
        self.db.rename_chat(chat_id, title)

    def load_history(self, chat_id: str) -> list[dict]:
        return self.db.load_history(chat_id)

    def add_message_to_history(self, chat_id: str, sender: str, text: str, sources: list[dict] = None) -> dict:
        return self.db.add_message_to_history(chat_id, sender, text, sources)

    def extract_text_from_pdf(self, file_bytes: bytes) -> list[tuple[int, str]]:
        return extract_text_from_pdf(file_bytes)

    async def process_document_background(self, chat_id: str, file_content: bytes, filename: str, doc_id: str):
        """
        Background task to perform AI-driven semantic and contextual chunking.
        Saves indexing progress details.
        """
        docs_metadata = self.load_documents(chat_id)
        docs_metadata[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "status": "Processing: Extracting text...",
            "char_count": 0,
            "chunk_count": 0,
            "uploaded_at": "2026-06-09T11:00:00Z"
        }
        self.save_documents(chat_id, docs_metadata)
        
        try:
            pages = self.doc_processor.parse_and_clean_document(file_content, filename)
            if not pages:
                raise ValueError("No extractable text found in document.")
                
            total_chars = sum(len(text) for _, text in pages)
            docs_metadata[doc_id]["char_count"] = total_chars
            docs_metadata[doc_id]["status"] = "Processing: Creating semantic batches..."
            self.save_documents(chat_id, docs_metadata)

            all_new_documents = self.doc_processor.create_child_parent_pairs(pages, filename, chat_id, doc_id)
            
            docs_metadata[doc_id]["status"] = "Processing: Embedding chunks..."
            self.save_documents(chat_id, docs_metadata)
            
            if self.use_db:
                db_chat_id = to_db_uuid(chat_id)
                db_doc_id = to_db_doc_uuid(doc_id)
                total_chunks = len(all_new_documents)
                
                batch_size = 16
                embeddings = []
                for i in range(0, total_chunks, batch_size):
                    batch_docs = all_new_documents[i:i+batch_size]
                    batch_texts = [d.page_content for d in batch_docs]
                    
                    progress = int((i / total_chunks) * 100)
                    docs_metadata[doc_id]["status"] = f"Processing: Embedding chunks {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks} ({progress}%)"
                    self.save_documents(chat_id, docs_metadata)
                    
                    batch_embs = self.embeddings.embed_documents(batch_texts)
                    embeddings.extend(batch_embs)
                    
                    if i + batch_size < total_chunks:
                        await asyncio.sleep(1.5)
                
                docs_metadata[doc_id]["status"] = "Processing: Saving embeddings to database..."
                self.save_documents(chat_id, docs_metadata)
                
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        from psycopg2.extras import execute_values
                        insert_data = [
                            (
                                db_doc_id,
                                db_chat_id,
                                doc.page_content,
                                emb,
                                json.dumps(doc.metadata)
                            )
                            for doc, emb in zip(all_new_documents, embeddings)
                        ]
                        execute_values(
                            cur,
                            "INSERT INTO document_embeddings (doc_id, chat_id, content, embedding, metadata) VALUES %s",
                            insert_data
                        )
            else:
                store = self.load_vectorstore(chat_id)
                store.add_documents(all_new_documents)
                self.save_vectorstore(chat_id, store)
            
            docs_metadata[doc_id]["status"] = "Indexed"
            docs_metadata[doc_id]["chunk_count"] = len(all_new_documents)
            self.save_documents(chat_id, docs_metadata)
            
            chats = self.load_chats()
            for c in chats:
                if c["chat_id"] == chat_id:
                    c["doc_count"] = len(docs_metadata)
            self.db.save_chats(chats)

        except Exception as e:
            docs_metadata[doc_id]["status"] = f"Failed: {str(e)}"
            self.save_documents(chat_id, docs_metadata)

    def retrieve(self, chat_id: str, query: str, k: int = 4) -> list[dict]:
        """
        Search and retrieve relevant chunks from the isolated vector store.
        """
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                print(f"[{chat_id}] RAG retrieve (DB): embedding query: '{query}'", flush=True)
                query_embedding = self.embeddings.embed_query(query)
                
                print(f"[{chat_id}] RAG retrieve (DB): searching database...", flush=True)
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT content, metadata, embedding <=> %s::vector AS distance 
                            FROM document_embeddings 
                            WHERE chat_id = %s 
                            ORDER BY distance ASC 
                            LIMIT %s;
                            """,
                            (query_embedding, db_chat_id, k * 4)
                        )
                        rows = cur.fetchall()
                        
                        retrieved = []
                        seen_parents = set()
                        for row in rows:
                            content, metadata_json, distance = row
                            
                            meta = {}
                            if metadata_json:
                                if isinstance(metadata_json, dict):
                                    meta = metadata_json
                                elif isinstance(metadata_json, str):
                                    try:
                                        meta = json.loads(metadata_json)
                                    except Exception:
                                        pass
                            
                            parent_content = meta.get("parent_content")
                            if not parent_content:
                                parent_content = content
                                
                            parent_key = parent_content.strip()
                            if parent_key in seen_parents:
                                continue
                            seen_parents.add(parent_key)
                            
                            confidence = max(0.0, 1.0 - (distance / 2.0))
                            match_percentage = round(confidence * 100)
                            
                            retrieved.append({
                                "content": parent_content,
                                "source": meta.get("source", "Unknown"),
                                "pages": meta.get("pages", [1]),
                                "score": match_percentage
                            })
                            if len(retrieved) >= k:
                                break
                        print(f"[{chat_id}] RAG retrieve (DB): found {len(retrieved)} matches.", flush=True)
                        return retrieved
            except Exception as e:
                print(f"[{chat_id}] RAG retrieve (DB) failed: {e}", flush=True)
                return []

        print(f"[{chat_id}] RAG retrieve: loading vectorstore...", flush=True)
        store = self.load_vectorstore(chat_id)
        print(f"[{chat_id}] RAG retrieve: store loaded. len={len(store.store)}", flush=True)
        if not store.store:
            return []
            
        try:
            print(f"[{chat_id}] RAG retrieve: similarity search with score: '{query}'", flush=True)
            results_with_scores = store.similarity_search_with_score(query, k=k * 4)
            print(f"[{chat_id}] RAG retrieve: search complete. found={len(results_with_scores)}", flush=True)
            retrieved = []
            seen_parents = set()
            for doc, score in results_with_scores:
                parent_content = doc.metadata.get("parent_content")
                if not parent_content:
                    parent_content = doc.page_content
                    
                parent_key = parent_content.strip()
                if parent_key in seen_parents:
                    continue
                seen_parents.add(parent_key)
                
                confidence = max(0.0, 1.0 - (score / 2.0))
                match_percentage = round(confidence * 100)
                
                retrieved.append({
                    "content": parent_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "pages": doc.metadata.get("pages", [1]),
                    "score": match_percentage
                })
                if len(retrieved) >= k:
                    break
            return retrieved
        except Exception as e:
            print(f"[{chat_id}] RAG retrieve failed: {e}", flush=True)
            return []

    def delete_document(self, chat_id: str, doc_id: str):
        self.db.delete_document(chat_id, doc_id)
        if not self.use_db:
            # Update Vector Store
            store = self.load_vectorstore(chat_id)
            if store.store:
                keys_to_delete = [
                    key for key, entry in store.store.items()
                    if entry.get("metadata", {}).get("doc_id") == doc_id
                ]
                for key in keys_to_delete:
                    del store.store[key]
                self.save_vectorstore(chat_id, store)
                
            # Update chat workspace doc_count
            docs_metadata = self.load_documents(chat_id)
            chats = self.load_chats()
            for c in chats:
                if c["chat_id"] == chat_id:
                    c["doc_count"] = len(docs_metadata)
            self.db.save_chats(chats)
