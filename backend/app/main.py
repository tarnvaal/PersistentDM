from contextlib import asynccontextmanager
import os
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.chat import router as chat_router
from .dependencies import get_chatter
from .utility.llama import Chatter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload the model in the background so health is immediate
    asyncio.create_task(asyncio.to_thread(get_chatter))
    yield
    # Shutdown: (if needed in the future)


app = FastAPI(title="PersistentDM API", lifespan=lifespan)

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
    return {"status": "ok"}


@app.get("/status")
def status():
    """Model/load status for UI.

    Returns: { state: unloaded|loading|ready|failed, free_vram_mib, min_required_vram_mib }
    """
    return Chatter.get_status()


app.include_router(chat_router)
