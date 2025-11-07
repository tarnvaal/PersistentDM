from contextlib import asynccontextmanager
import os
import asyncio
import uuid
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .routers.chat import router as chat_router
from .routers.ingest import router as ingest_router
from .routers.search_router import router as search_router
from .sessions.router import router as sessions_router
from .dependencies import get_chatter, get_search_service
from .world.search_service import SearchService
from .utility.embeddings import EmbeddingModelSingleton
from .utility.llama import Chatter
from .logging_config import get_logger, set_request_context, clear_request_context

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure embeddings are initialized synchronously to avoid false-green health
    # This will raise on failure and prevent the app from starting.
    await asyncio.to_thread(EmbeddingModelSingleton.initialize)
    # Llama preload can remain backgrounded if desired
    asyncio.create_task(asyncio.to_thread(get_chatter))
    yield
    # Shutdown: (if needed in the future)


app = FastAPI(title="PersistentDM API", lifespan=lifespan)

# Bind default provider so tests can override `SearchService` directly
app.dependency_overrides[SearchService] = get_search_service


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Middleware to track requests with structured logging."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Set request context for logging
    set_request_context(request_id)

    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        user_agent=request.headers.get("user-agent", ""),
        remote_addr=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log request completion
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(process_time * 1000, 2),
        )

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            latency_ms=round(process_time * 1000, 2),
            error=str(e),
        )
        raise
    finally:
        clear_request_context()


# allow React dev server to talk to API in development
# Support dynamic frontend port (Vite may use 5173-5180 range)
FRONTEND_PORT = os.getenv("FRONTEND_PORT", "5173")
allowed_origins = []
# Allow all common Vite ports (5173-5180) for both localhost and 127.0.0.1
for port in range(5173, 5181):  # 5181 to include 5180
    allowed_origins.extend(
        [
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
            f"http://[::1]:{port}",  # IPv6 localhost
        ]
    )

# Allow explicitly provided origins (comma-separated)
extra_origins = os.getenv("ADDITIONAL_CORS_ORIGINS", "").strip()
if extra_origins:
    for origin in extra_origins.split(","):
        origin = origin.strip()
        if origin:
            allowed_origins.append(origin)

allow_all = os.getenv("CORS_ALLOW_ALL", "0").lower() in ("1", "true", "yes")

if allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # required when using wildcard origins
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    """Liveness probe - just check if the service is running."""
    logger.debug("Health check called")
    return {"status": "ok", "service": "PersistentDM API"}


@app.get("/ready")
def ready():
    """Readiness probe - check if all dependencies are available."""
    try:
        # Check if embeddings are loaded
        from .utility.embeddings import EmbeddingModelSingleton

        embeddings_ready = EmbeddingModelSingleton.is_initialized()

        # Check if chatter can be instantiated (will use cache if available)
        chatter_ready = True
        try:
            get_chatter()
        except Exception:
            chatter_ready = False

        ready_status = embeddings_ready and chatter_ready

        status_info = {
            "ready": ready_status,
            "embeddings": embeddings_ready,
            "chatter": chatter_ready,
        }

        if ready_status:
            logger.debug("Readiness check passed", **status_info)
            return status_info
        else:
            logger.warning("Readiness check failed", **status_info)
            return status_info

    except Exception as e:
        logger.error("Readiness check error", error=str(e))
        return {
            "ready": False,
            "error": "Readiness check failed",
            "details": str(e) if os.getenv("PDM_DEBUG_ERRORS", "1") == "1" else None,
        }


@app.get("/status")
def status():
    """Model/load status for UI.

    Returns: { state: unloaded|loading|ready|failed, free_vram_mib, min_required_vram_mib }
    """
    return Chatter.get_status()


app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(sessions_router)
