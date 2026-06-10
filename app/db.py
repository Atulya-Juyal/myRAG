import os
import json
import uuid
import psycopg2
from typing import Optional
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import quote_plus
from app.config import get_settings

def get_safe_db_url(url: str) -> str:
    if not url:
        return url
    if "@" in url:
        try:
            prefix, rest = url.split("://", 1)
            credentials, host_db = rest.split("@", 1)
            if ":" in credentials:
                username, password = credentials.split(":", 1)
                encoded_password = quote_plus(password)
                return f"{prefix}://{username}:{encoded_password}@{host_db}"
        except Exception:
            return url
    return url

def to_db_uuid(chat_id: str) -> str:
    if chat_id == "default":
        return "00000000-0000-0000-0000-000000000000"
    return chat_id

def from_db_uuid(uuid_val) -> str:
    if not uuid_val:
        return "default"
    uuid_str = str(uuid_val)
    if uuid_str == "00000000-0000-0000-0000-000000000000":
        return "default"
    return uuid_str

def to_db_doc_uuid(doc_id: str) -> str:
    if doc_id == "doc_sample":
        return "ffffffff-ffff-ffff-ffff-ffffffffffff"
    return doc_id

def from_db_doc_uuid(uuid_val) -> str:
    if not uuid_val:
        return "doc_sample"
    uuid_str = str(uuid_val)
    if uuid_str == "ffffffff-ffff-ffff-ffff-ffffffffffff":
        return "doc_sample"
    return uuid_str


class DatabaseManager:
    def __init__(self, base_dir: str = "data", settings=None, get_conn_provider=None):
        self.settings = settings or get_settings()
        self.base_dir = base_dir
        self.chats_dir = os.path.join(self.base_dir, "chats")
        os.makedirs(self.chats_dir, exist_ok=True)
        self.get_conn_provider = get_conn_provider
        
        self.use_db = False
        try:
            if hasattr(self.settings, "database_url") and self.settings.database_url:
                if isinstance(self.settings.database_url, str) and "postgresql" in self.settings.database_url:
                    self.use_db = True
        except Exception:
            self.use_db = False
            
        self._init_chats_registry()

    def _init_chats_registry(self):
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS chat_messages (
                                id SERIAL PRIMARY KEY,
                                chat_id UUID NOT NULL REFERENCES workspaces(chat_id) ON DELETE CASCADE,
                                sender VARCHAR(50) NOT NULL,
                                text TEXT NOT NULL,
                                sources JSONB DEFAULT '[]'::jsonb,
                                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                            );
                        """)
                        cur.execute("SELECT COUNT(*) FROM workspaces;")
                        if cur.fetchone()[0] == 0:
                            chat_id = str(uuid.uuid4())
                            cur.execute(
                                "INSERT INTO workspaces (chat_id, title) VALUES (%s, %s);",
                                (chat_id, "New Chat")
                            )
            except Exception as e:
                print(f"Failed to initialize default chats registry in DB: {e}", flush=True)
        else:
            chats_json_path = os.path.join(self.chats_dir, "chats.json")
            if not os.path.exists(chats_json_path):
                chat_id = str(uuid.uuid4())
                default_chats = [
                    {
                        "chat_id": chat_id,
                        "title": "New Chat",
                        "created_at": "2026-06-09T11:00:00Z",
                        "doc_count": 0
                    }
                ]
                with open(chats_json_path, "w", encoding="utf-8") as f:
                    json.dump(default_chats, f, indent=2)
                os.makedirs(os.path.join(self.chats_dir, chat_id), exist_ok=True)

    @contextmanager
    def _get_db_conn(self):
        safe_url = get_safe_db_url(self.settings.database_url)
        conn = psycopg2.connect(safe_url)
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def get_conn(self):
        if self.get_conn_provider is not None:
            with self.get_conn_provider() as conn:
                yield conn
            return
        with self._get_db_conn() as conn:
            yield conn

    def get_vectorstore_path(self, chat_id: str) -> str:
        return os.path.join(self.chats_dir, chat_id, "vectorstore.json")

    def get_documents_path(self, chat_id: str) -> str:
        return os.path.join(self.chats_dir, chat_id, "documents.json")

    def get_history_path(self, chat_id: str) -> str:
        return os.path.join(self.chats_dir, chat_id, "history.json")

    def load_documents(self, chat_id: str) -> dict:
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT doc_id, filename, status, char_count, chunk_count, uploaded_at FROM documents WHERE chat_id = %s;",
                            (db_chat_id,)
                        )
                        rows = cur.fetchall()
                        docs = {}
                        for row in rows:
                            pydoc_id = from_db_doc_uuid(row[0])
                            docs[pydoc_id] = {
                                "id": pydoc_id,
                                "filename": row[1],
                                "status": row[2],
                                "char_count": row[3],
                                "chunk_count": row[4],
                                "uploaded_at": row[5].isoformat() if row[5] else None
                            }
                        return docs
            except Exception as e:
                print(f"Failed to load documents from DB: {e}", flush=True)
                return {}
        else:
            path = self.get_documents_path(chat_id)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

    def save_documents(self, chat_id: str, docs: dict):
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        for pydoc_id, doc in docs.items():
                            db_doc_id = to_db_doc_uuid(pydoc_id)
                            filename = str(doc.get("filename", ""))[:255]
                            status = str(doc.get("status", ""))[:255]
                            cur.execute("""
                                INSERT INTO documents (doc_id, chat_id, filename, status, char_count, chunk_count)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (doc_id) 
                                DO UPDATE SET 
                                    status = EXCLUDED.status,
                                    char_count = EXCLUDED.char_count,
                                    chunk_count = EXCLUDED.chunk_count;
                            """, (db_doc_id, db_chat_id, filename, status, doc["char_count"], doc["chunk_count"]))
            except Exception as e:
                print(f"Failed to save documents to DB: {e}", flush=True)
                raise
        else:
            path = self.get_documents_path(chat_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=2)

    def load_chats(self) -> list[dict]:
        if self.use_db:
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT chat_id, title, created_at,
                                   (SELECT COUNT(*) FROM documents WHERE chat_id = workspaces.chat_id) as doc_count
                            FROM workspaces
                            ORDER BY created_at ASC;
                        """)
                        rows = cur.fetchall()
                        chats = []
                        for row in rows:
                            chats.append({
                                "chat_id": from_db_uuid(row[0]),
                                "title": row[1],
                                "created_at": row[2].isoformat() if row[2] else None,
                                "doc_count": row[3]
                            })
                        return chats
            except Exception as e:
                print(f"Failed to load chats from DB: {e}", flush=True)
                return []
        else:
            chats_json_path = os.path.join(self.chats_dir, "chats.json")
            if os.path.exists(chats_json_path):
                with open(chats_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []

    def save_chats(self, chats: list[dict]):
        if self.use_db:
            pass
        else:
            chats_json_path = os.path.join(self.chats_dir, "chats.json")
            with open(chats_json_path, "w", encoding="utf-8") as f:
                json.dump(chats, f, indent=2)

    def create_chat(self, chat_id: str, title: str) -> dict:
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT chat_id, title, created_at FROM workspaces WHERE chat_id = %s;",
                            (db_chat_id,)
                        )
                        row = cur.fetchone()
                        if row:
                            return {
                                "chat_id": from_db_uuid(row[0]),
                                "title": row[1],
                                "created_at": row[2].isoformat() if row[2] else None,
                                "doc_count": 0
                            }
                        
                        title = str(title)[:255]
                        cur.execute(
                            "INSERT INTO workspaces (chat_id, title) VALUES (%s, %s) RETURNING created_at;",
                            (db_chat_id, title)
                        )
                        created_at = cur.fetchone()[0]
                        return {
                            "chat_id": chat_id,
                            "title": title,
                            "created_at": created_at.isoformat() if created_at else None,
                            "doc_count": 0
                        }
            except Exception as e:
                print(f"Failed to create chat in DB: {e}", flush=True)
                raise
        else:
            os.makedirs(os.path.join(self.chats_dir, chat_id), exist_ok=True)
            chats = self.load_chats()
            for c in chats:
                if c["chat_id"] == chat_id:
                    return c
            new_chat = {
                "chat_id": chat_id,
                "title": title,
                "created_at": "2026-06-09T11:00:00Z",
                "doc_count": 0
            }
            chats.append(new_chat)
            self.save_chats(chats)
            self.save_documents(chat_id, {})
            return new_chat

    def delete_chat(self, chat_id: str):
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM workspaces WHERE chat_id = %s;",
                            (db_chat_id,)
                        )
            except Exception as e:
                print(f"Failed to delete chat {chat_id} from DB: {e}", flush=True)
                raise
        else:
            import shutil
            chat_path = os.path.join(self.chats_dir, chat_id)
            if os.path.exists(chat_path):
                shutil.rmtree(chat_path)
            chats = self.load_chats()
            chats = [c for c in chats if c["chat_id"] != chat_id]
            self.save_chats(chats)

    def rename_chat(self, chat_id: str, title: str):
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        title = str(title)[:255]
                        cur.execute(
                            "UPDATE workspaces SET title = %s WHERE chat_id = %s;",
                            (title, db_chat_id)
                        )
            except Exception as e:
                print(f"Failed to rename chat {chat_id} in DB: {e}", flush=True)
                raise
        else:
            chats = self.load_chats()
            for c in chats:
                if c["chat_id"] == chat_id:
                    c["title"] = title
                    break
            self.save_chats(chats)

    def load_history(self, chat_id: str) -> list[dict]:
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT sender, text, sources, timestamp FROM chat_messages WHERE chat_id = %s ORDER BY timestamp ASC;",
                            (db_chat_id,)
                        )
                        rows = cur.fetchall()
                        messages = []
                        for row in rows:
                            meta_sources = []
                            if row[2]:
                                if isinstance(row[2], list):
                                    meta_sources = row[2]
                                elif isinstance(row[2], str):
                                    try:
                                        meta_sources = json.loads(row[2])
                                    except Exception:
                                        pass
                            messages.append({
                                "sender": row[0],
                                "text": row[1],
                                "sources": meta_sources,
                                "timestamp": row[3].isoformat() if row[3] else None
                            })
                        return messages
            except Exception as e:
                print(f"Failed to load history from DB: {e}", flush=True)
                return []
        else:
            path = self.get_history_path(chat_id)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return []
            return []

    def add_message_to_history(self, chat_id: str, sender: str, text: str, sources: list[dict] = None) -> dict:
        if sources is None:
            sources = []
        
        timestamp_str = datetime.now(timezone.utc).isoformat()
        
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO chat_messages (chat_id, sender, text, sources) VALUES (%s, %s, %s, %s) RETURNING timestamp;",
                            (db_chat_id, sender, text, json.dumps(sources))
                        )
                        ret = cur.fetchone()
                        if ret and ret[0]:
                            timestamp_str = ret[0].isoformat()
            except Exception as e:
                print(f"Failed to save message to DB: {e}", flush=True)
                raise
        else:
            path = self.get_history_path(chat_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            history = self.load_history(chat_id)
            new_msg = {
                "sender": sender,
                "text": text,
                "sources": sources,
                "timestamp": timestamp_str
            }
            history.append(new_msg)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
                
        return {
            "sender": sender,
            "text": text,
            "sources": sources,
            "timestamp": timestamp_str
        }

    def delete_document(self, chat_id: str, doc_id: str):
        if self.use_db:
            db_chat_id = to_db_uuid(chat_id)
            db_doc_id = to_db_doc_uuid(doc_id)
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM documents WHERE chat_id = %s AND doc_id = %s;",
                            (db_chat_id, db_doc_id)
                        )
            except Exception as e:
                print(f"Failed to delete document {doc_id} in DB: {e}", flush=True)
                raise
        else:
            docs_metadata = self.load_documents(chat_id)
            if doc_id in docs_metadata:
                del docs_metadata[doc_id]
                self.save_documents(chat_id, docs_metadata)
