# SSE Streaming Responses + Markdown Rendering

Add real-time token streaming via Server-Sent Events and full Markdown rendering for bot responses. The existing `/chat` endpoint stays untouched for backward compatibility; streaming is an additive feature.

## Proposed Changes

### Backend — Streaming Infrastructure

#### [MODIFY] [agent.py](file:///c:/projects/myRAG/app/agent.py)

Add a new `stream()` method to `ProductionAgent` that:
1. Loads history and builds the message list (same as `invoke`)
2. Runs the `retrieve_context` node synchronously to get RAG chunks
3. Builds the system prompt with context (same logic as `process_message`)
4. Calls `self.primary_llm.stream(messages)` to get a token-by-token generator
5. On failure, falls back to `self.fallback_llm.stream(messages)`
6. `yield`s each token chunk as a string
7. Returns sources/model_used via metadata

**Why bypass the LangGraph graph for streaming?** LangGraph's `graph.invoke()` runs nodes sequentially and returns full state — it doesn't support mid-node token-level streaming. We reuse the same retrieve + prompt logic but call the LLM's `.stream()` directly.

---

#### [MODIFY] [main.py](file:///c:/projects/myRAG/app/main.py)

Add a new `POST /chat/stream` endpoint that:
1. Runs the same security check + cache lookup as `/chat`
2. If cache hit → emit a single SSE `data:` event with the full cached response + `[DONE]`
3. If cache miss → call `agent.stream()` and emit each token as an SSE `data:` event
4. After streaming completes, emit a final `[DONE]` event containing `{ sources, model_used }`
5. Save the assembled full response to cache + history in a background task

SSE format per line:
```
data: {"type":"token","content":"Hello"}
data: {"type":"token","content":" world"}
data: {"type":"done","sources":[...],"model_used":"primary"}
```

Import `StreamingResponse` from `fastapi.responses`.

---

### Frontend — Streaming Consumer + Markdown Renderer

#### [MODIFY] [index.js](file:///c:/projects/myRAG/static/index.js)

**Streaming changes (`handleSendMessage`):**
- Switch the `fetch` call from `/chat` to `/chat/stream`
- Read the response body using `response.body.getReader()` with a `TextDecoder`
- Parse incoming SSE `data:` lines
- For each `type:"token"` event: append the text to the live bot bubble (replace typing indicator with a growing message)
- For the `type:"done"` event: finalize the message with sources, save to local history
- Keep error/fallback handling for non-200 responses

**Markdown renderer (`parseMessageText`):**
Replace the minimal regex parser with a full block-aware markdown renderer supporting:
- Fenced code blocks (` ```lang ... ``` `) with syntax label and copy button
- Headings (`# H1` through `#### H4`)
- Bullet lists (`- item` and `* item`), including nested
- Numbered lists (`1. item`)
- Bold (`**text**`), italic (`*text*`), inline code (`` `code` ``)
- Horizontal rules (`---`)
- Block-level processing first, then inline processing per block
- All content is HTML-escaped before transformation (XSS safe)

---

#### [MODIFY] [index.css](file:///c:/projects/myRAG/static/index.css)

Add Markdown typography styles scoped to `.message-content`:
- `h1`–`h4` sizing and spacing
- `ul`, `ol` list styles with proper indentation
- `pre > code` for fenced code blocks (dark background, monospace, horizontal scroll, copy button positioning)
- `code` inline styles
- `hr` separator
- `strong`, `em` text styles
- `@keyframes` for a streaming cursor blink effect

---

## Verification Plan

### Manual Verification
1. Send a chat message and verify tokens appear live in the bubble as they stream
2. Verify that the typing indicator transitions smoothly into streaming text
3. Verify that sources panel renders correctly after streaming completes
4. Verify that cache hits still work (instant full response)
5. Verify that Markdown renders correctly: code blocks, headers, lists, bold/italic
6. Verify the existing `/chat` endpoint still works unchanged (backward compatibility)
7. Test error scenarios: network disconnect mid-stream, LLM failure during streaming
