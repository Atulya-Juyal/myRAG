import re
import io
import pdfplumber
from langchain_core.documents import Document


def deep_clean_text(text: str) -> str:
    if not text:
        return ""
    # 1. Fix words split across lines by a hyphen and newline
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    # 2. Replace common PDF ligatures back to normal characters
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff"}
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)
    # 3. Collapse horizontal whitespace noise (tabs, multiple spaces)
    text = re.sub(r'[ \t\r\x0b\x0c]+', ' ', text)
    # 4. Collapse three or more newlines down to a double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages

class DocumentProcessor:
    def __init__(self):
        self._parent_splitter = None
        self._child_splitter = None

    @property
    def parent_splitter(self):
        if self._parent_splitter is None:
            from app.rag import RecursiveCharacterTextSplitter
            self._parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800, 
                chunk_overlap=300,
                separators=["\n\n", ".\n", ". ", " ", ""]
            )
        return self._parent_splitter

    @property
    def child_splitter(self):
        if self._child_splitter is None:
            from app.rag import RecursiveCharacterTextSplitter
            self._child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
        return self._child_splitter

    def parse_and_clean_document(self, file_content: bytes, filename: str) -> list[tuple[int, str]]:
        pages = []
        if filename.lower().endswith(".pdf"):
            raw_pages = extract_text_from_pdf(file_content)
            for item in raw_pages:
                if isinstance(item, dict):
                    p_num = item.get("page") or item.get("page_num") or item.get("page_label") or 1
                    p_text = item.get("text") or item.get("content") or item.get("page_content") or ""
                    pages.append((p_num, p_text))
                elif isinstance(item, tuple):
                    pages.append(item)
        else:
            text_content = file_content.decode("utf-8", errors="ignore")
            pages = [(1, text_content)]
            
        pages = [(p, txt.strip()) for p, txt in pages if txt and txt.strip()]
        return pages

    def create_child_parent_pairs(self, pages: list[tuple[int, str]], filename: str, chat_id: str, doc_id: str) -> list[Document]:
        full_document_text = ""
        for page_num, text in pages:
            cleaned_page = deep_clean_text(text)
            full_document_text += f"\n[PAGE_MARKER:{page_num}]\n{cleaned_page}"

        parent_texts = self.parent_splitter.split_text(full_document_text)
        all_new_documents = []
        global_context_prefix = f"Document Source: {filename}\nContext: "

        for p_text in parent_texts:
            found_pages = [int(num) for num in re.findall(r'\[PAGE_MARKER:(\d+)\]', p_text)]
            if not found_pages:
                found_pages = [1]
            
            clean_p_text = re.sub(r'\[PAGE_MARKER:\d+\]\n?', '', p_text).strip()
            if not clean_p_text:
                continue

            import uuid
            parent_node_id = str(uuid.uuid4())
            child_texts = self.child_splitter.split_text(clean_p_text)
            
            for c_text in child_texts:
                enriched_child_text = global_context_prefix + c_text
                doc_node = Document(
                    page_content=enriched_child_text,
                    metadata={
                        "doc_id": doc_id,
                        "chat_id": chat_id,
                        "parent_id": parent_node_id,
                        "parent_text": clean_p_text,
                        "parent_content": clean_p_text,
                        "pages": list(set(found_pages)),
                        "source": filename
                    }
                )
                all_new_documents.append(doc_node)
                
        return all_new_documents
