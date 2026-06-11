"""
Production-Ready FastAPI + LangGraph Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""


import time
import os
import asyncio
import uuid
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent
from app.rag import RAGManager

load_dotenv()
logger = get_logger()


async def render_keep_alive_pinger():
    """
    Background task to periodically ping the Render URL using standard urllib to keep the instance active.
    """
    import urllib.request
    # Wait 60 seconds before first ping to allow server to fully start
    await asyncio.sleep(60)
    
    settings = get_settings()
    target_url = settings.render_ping_url or os.getenv("RENDER_PING_URL", "https://myrag-atulyajuyal.onrender.com")
    
    if not target_url:
        logger.info("Keep-alive pinger disabled: RENDER_PING_URL not specified.")
        return
        
    if not target_url.endswith("/health") and not target_url.endswith("/"):
        ping_url = f"{target_url}/health"
    elif target_url.endswith("/"):
        ping_url = f"{target_url}health"
    else:
        ping_url = target_url

    logger.info(f"Starting Render keep-alive pinger for: {ping_url}")
    
    while True:
        try:
            # Execute the synchronous urllib call in a worker thread to keep the event loop non-blocking
            def do_ping():
                req = urllib.request.Request(
                    ping_url, 
                    headers={'User-Agent': 'myRAG-KeepAlivePinger/1.0'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.getcode()
                    
            status_code = await asyncio.to_thread(do_ping)
            logger.info(f"Keep-alive ping to {ping_url}: status={status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive ping failed to reach {ping_url}: {e}")
        
        # Ping every 10 minutes (600 seconds) to prevent Render free-tier spin down
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    global security, cache, metrics, agent, rag_manager

    settings = get_settings()

    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_llm,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()
    rag_manager = RAGManager()

    # Seed default workspace if empty
    try:
        # Check if workspaces table/registry is empty
        chats = rag_manager.load_chats()
        if not chats:
            logger.info("No workspaces found. Creating default 'New Chat' workspace...")
            chat_id = str(uuid.uuid4())
            rag_manager.create_chat(chat_id, "New Chat")
    except Exception as e:
        logger.error(f"Failed to seed default workspace: {e}")

    logger.info("All components initialized. Ready to serve requests.")

    # Spawn the keep-alive pinger task
    pinger_future = asyncio.create_task(render_keep_alive_pinger())

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})
    pinger_future.cancel()
    try:
        await pinger_future
    except asyncio.CancelledError:
        pass



limiter = Limiter(key_func=get_remote_address)



app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.1.5",
    lifespan=lifespan,
)
app.state.limiter = limiter



@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded.",
            "detail": "Too many requests. Please slow down."
        },
    )



@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        print("API: Starting security check...", flush=True)
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        print(f"API: Security check complete. allowed={is_allowed}", flush=True)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )

        # ---- Step 2: Cache Lookup ----
        print("API: Starting cache lookup...", flush=True)
        cached_response = cache.get(cleaned_message)
        print(f"API: Cache lookup complete. hit={cached_response is not None}", flush=True)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            # Save to history
            try:
                rag_manager.add_message_to_history(body.chat_id, "user", cleaned_message)
                rag_manager.add_message_to_history(body.chat_id, "bot", cached_response, sources=[])
            except Exception as e:
                logger.error(f"Failed to save cache hit to history: {e}")

            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
                sources=[],
            )

        # ---- Step 3: Invoke LangGraph Agent ----
        print("API: Starting agent invoke...", flush=True)
        try:
            result = agent.invoke(cleaned_message, chat_id=body.chat_id)
            print("API: Agent invoke complete.", flush=True)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

        # ---- Step 6: Log & Record Metrics ----
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)

        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        if security_notes:
            logger.info("Security notes", extra={"extra_data": {
                "notes": security_notes,
                "thread_id": body.thread_id,
            }})

        logger.info("Request completed", extra={"extra_data": {
            "thread_id": body.thread_id,
            "model_used": model_used,
            "latency_ms": round(timer.elapsed_ms, 2),
        }})

        # ---- Step 7: Save to History ----
        try:
            rag_manager.add_message_to_history(body.chat_id, "user", cleaned_message)
            rag_manager.add_message_to_history(
                body.chat_id,
                "bot",
                validated_response,
                sources=result.get("sources", [])
            )
        except Exception as e:
            logger.error(f"Failed to save message to history: {e}")

        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
            sources=result.get("sources", []),
        )


@app.post("/chat/stream")
@limiter.limit(get_settings().rate_limit)
async def chat_stream(request: Request, body: ChatRequest):
    """
    Additive real-time token streaming endpoint via SSE (Server-Sent Events).
    """
    # ---- Step 1: Security Check ----
    is_allowed, cleaned_message, notes = security.check_input(body.message)
    if not is_allowed:
        logger.warning("Request blocked by security (stream)", extra={"extra_data": {
            "reason": notes,
            "thread_id": body.thread_id,
        }})
        metrics.record_request(latency_ms=0, error=True)
        raise HTTPException(
            status_code=400,
            detail="Your message was blocked by our security filters."
        )

    # ---- Step 2: Cache Lookup ----
    cached_response = cache.get(cleaned_message)
    if cached_response is not None:
        metrics.record_request(latency_ms=0, cache_hit=True)
        logger.info("Cache hit (stream)", extra={"extra_data": {
            "thread_id": body.thread_id,
        }})
        # Save to history
        try:
            rag_manager.add_message_to_history(body.chat_id, "user", cleaned_message)
            rag_manager.add_message_to_history(body.chat_id, "bot", cached_response, sources=[])
        except Exception as e:
            logger.error(f"Failed to save cache hit to history: {e}")

        async def cache_generator():
            yield f"data: {json.dumps({'type': 'token', 'content': cached_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'model_used': 'cache'})}\n\n"

        return StreamingResponse(cache_generator(), media_type="text/event-stream")

    # ---- Step 3: Stream from Agent ----
    async def event_generator():
        full_response = ""
        model_used = "primary"
        sources = []
        has_error = False
        
        try:
            async for token, model, src in agent.stream(cleaned_message, chat_id=body.chat_id):
                if token:
                    full_response += token
                    model_used = model
                    sources = src
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    # Small sleep to smooth stream chunks
                    await asyncio.sleep(0.005)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'token', 'content': 'Error occurred during streaming.'})}\n\n"
            has_error = True
            model_used = "error_handler"

        if not has_error:
            # ---- Step 4: Output Validation ----
            validated_response, output_warnings = security.check_output(full_response)
            
            # ---- Step 5: Cache Store ----
            cache.set(cleaned_message, validated_response)
            
            # ---- Step 6: Save to History ----
            try:
                rag_manager.add_message_to_history(body.chat_id, "user", cleaned_message)
                rag_manager.add_message_to_history(
                    body.chat_id,
                    "bot",
                    validated_response,
                    sources=sources
                )
            except Exception as e:
                logger.error(f"Failed to save streamed message to history: {e}")

        # ---- Step 7: Emit done metadata event ----
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'model_used': model_used})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check for Docker.
    """
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )



@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Metrics for monitoring dashboards.
    """
    summary = metrics.summary
    return MetricsResponse(**summary)



@app.get("/cache/stats")
async def cache_stats():
    """
    Cache performance statistics.
    """
    return cache.stats


# ==========================================
# RAG Workspaces & Documents Endpoints
# ==========================================

from pydantic import BaseModel

class CreateChatBody(BaseModel):
    title: str

@app.get("/chats")
async def list_chats():
    return rag_manager.load_chats()

@app.post("/chats")
async def create_chat(body: CreateChatBody):
    chat_id = str(uuid.uuid4())
    return rag_manager.create_chat(chat_id, body.title)

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    rag_manager.delete_chat(chat_id)
    return {"status": "deleted"}

class RenameChatBody(BaseModel):
    title: str

@app.put("/chats/{chat_id}")
async def rename_chat(chat_id: str, body: RenameChatBody):
    rag_manager.rename_chat(chat_id, body.title)
    return {"status": "renamed", "title": body.title}

@app.get("/chats/{chat_id}/history")
async def get_chat_history(chat_id: str):
    return rag_manager.load_history(chat_id)

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    chat_id: str,
    file: UploadFile = File(...)
):
    if not (file.filename.lower().endswith(".pdf") or 
            file.filename.lower().endswith(".txt") or 
            file.filename.lower().endswith(".md")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, or MD documents are supported."
        )
        
    doc_id = str(uuid.uuid4())
    file_bytes = await file.read()
    
    background_tasks.add_task(
        rag_manager.process_document_background,
        chat_id=chat_id,
        file_content=file_bytes,
        filename=file.filename,
        doc_id=doc_id
    )
    return {"status": "Processing: Uploaded", "doc_id": doc_id}

@app.get("/documents")
async def list_documents(chat_id: str):
    docs = rag_manager.load_documents(chat_id)
    return list(docs.values())

@app.delete("/documents/{chat_id}/{doc_id}")
async def delete_document(chat_id: str, doc_id: str):
    rag_manager.delete_document(chat_id, doc_id)
    return {"status": "deleted"}

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")



