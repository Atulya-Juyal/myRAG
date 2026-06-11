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
    import io
    import pdfplumber
    from collections import Counter
    import re
    
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # 1. First pass: find the body text font size mode across the entire document
        all_sizes = []
        for page in pdf.pages:
            chars = getattr(page, "chars", None)
            if isinstance(chars, list):
                h = page.height
                top_margin = 0.06 * h
                bottom_margin = 0.94 * h
                all_sizes.extend([
                    c["size"] for c in chars 
                    if top_margin < c["top"] < bottom_margin and not c["text"].isspace()
                ])
                
        if all_sizes:
            body_font_size = Counter(all_sizes).most_common(1)[0][0]
        else:
            body_font_size = 10.0
            
        # 2. Second pass: extract structured lines and convert to Markdown per page
        for i, page in enumerate(pdf.pages):
            chars = getattr(page, "chars", None)
            if not isinstance(chars, list) or not chars:
                text = page.extract_text()
                if text:
                    pages.append((i + 1, text))
                continue
                
            h = page.height
            top_margin = 0.06 * h
            bottom_margin = 0.94 * h
            
            filtered_chars = [
                c for c in chars 
                if top_margin < c["top"] < bottom_margin and not c["text"].isspace()
            ]
            
            if not filtered_chars:
                text = page.extract_text()
                if text:
                    pages.append((i + 1, text))
                continue
                
            # Group chars into lines by top coordinate with a tolerance of 3 points
            filtered_chars.sort(key=lambda c: c["top"])
            lines_data = []
            current_line_chars = []
            current_top = None
            
            for c in filtered_chars:
                if current_top is None:
                    current_top = c["top"]
                    current_line_chars.append(c)
                elif abs(c["top"] - current_top) <= 3:
                    current_line_chars.append(c)
                else:
                    lines_data.append(current_line_chars)
                    current_line_chars = [c]
                    current_top = c["top"]
            if current_line_chars:
                lines_data.append(current_line_chars)
                
            page_markdown_lines = []
            for line_chars in lines_data:
                # Sort left to right
                line_chars.sort(key=lambda c: c["x0"])
                
                # Reconstruct text with spacing tolerance
                line_text = ""
                for idx, c in enumerate(line_chars):
                    if idx > 0:
                        prev_c = line_chars[idx - 1]
                        avg_width = (c["x1"] - c["x0"] + prev_c["x1"] - prev_c["x0"]) / 2
                        if c["x0"] - prev_c["x1"] > 0.25 * avg_width:
                            line_text += " "
                    line_text += c["text"]
                    
                line_text = line_text.strip()
                if not line_text:
                    continue
                    
                # Compute styling properties
                avg_size = sum(c["size"] for c in line_chars) / len(line_chars)
                bold_chars_count = sum(
                    1 for c in line_chars 
                    if any(x in c.get("fontname", "").lower() for x in ["bold", "heavy", "black", "medium"])
                )
                is_bold = bold_chars_count / len(line_chars) > 0.5
                
                # Classify line into Markdown structure
                if avg_size >= 1.4 * body_font_size:
                    page_markdown_lines.append(f"# {line_text}")
                elif avg_size >= 1.2 * body_font_size:
                    page_markdown_lines.append(f"## {line_text}")
                elif is_bold and (avg_size > 1.05 * body_font_size or re.match(r'^\d+(\.\d+)*\b', line_text)):
                    page_markdown_lines.append(f"### {line_text}")
                else:
                    page_markdown_lines.append(line_text)
                    
            page_md = "\n\n".join(page_markdown_lines)
            pages.append((i + 1, page_md))
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

        # Check if the document has Markdown headers to guide structural splitting
        has_headers = any(line.strip().startswith("#") for line in full_document_text.splitlines())
        
        all_new_documents = []
        global_context_prefix = f"Document Source: {filename}\n"

        if has_headers:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            parent_docs = markdown_splitter.split_text(full_document_text)

            for p_doc in parent_docs:
                p_text = p_doc.page_content
                found_pages = [int(num) for num in re.findall(r'\[PAGE_MARKER:(\d+)\]', p_text)]
                if not found_pages:
                    found_pages = [1]
                
                clean_p_text = re.sub(r'\[PAGE_MARKER:\d+\]\n?', '', p_text).strip()
                if not clean_p_text:
                    continue

                # Build context path breadcrumbs from Markdown headers
                headers = [p_doc.metadata[k] for k in ["Header 1", "Header 2", "Header 3"] if k in p_doc.metadata]
                context_path = " > ".join(headers) if headers else ""
                
                # Sub-split oversized parent sections to keep returned context blocks clean and properly sized
                if len(clean_p_text) > 2000:
                    sub_parents = self.parent_splitter.split_text(clean_p_text)
                else:
                    sub_parents = [clean_p_text]

                for sub_p in sub_parents:
                    import uuid
                    parent_node_id = str(uuid.uuid4())
                    child_texts = self.child_splitter.split_text(sub_p)
                    
                    for c_text in child_texts:
                        if context_path:
                            enriched_child_text = f"{global_context_prefix}[Context: {context_path}]\n{c_text}"
                        else:
                            enriched_child_text = f"{global_context_prefix}Context: {c_text}"
                            
                        meta = {
                            "doc_id": doc_id,
                            "chat_id": chat_id,
                            "parent_id": parent_node_id,
                            "parent_text": sub_p,
                            "parent_content": sub_p,
                            "pages": list(set(found_pages)),
                            "source": filename
                        }
                        # Merge header metadata fields
                        meta.update(p_doc.metadata)
                        
                        doc_node = Document(
                            page_content=enriched_child_text,
                            metadata=meta
                        )
                        all_new_documents.append(doc_node)
        else:
            # Fallback to standard recursive text splitting when no headings are found
            parent_texts = self.parent_splitter.split_text(full_document_text)
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
                    enriched_child_text = f"{global_context_prefix}Context: {c_text}"
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
