import os
from functools import lru_cache
from fastapi import Depends

from .utility.llama import Chatter
from .utility.embeddings import get_embedding_model, EmbeddingModel
from .world.memory import WorldMemory
from .world.conversation_service import ConversationService
from .services.state_service import StateService
from .settings import DEFAULT_MODEL_PATH


# NOTE: @lru_cache singletons are per-process, not shared across Uvicorn workers.
# With multiple workers, each process maintains its own cached instances.
# This is usually desired for model isolation, but means state is not shared
# between workers. Use external storage (Redis, database) for cross-worker state.
@lru_cache(maxsize=1)
def get_chatter() -> Chatter:
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    return Chatter(model_path)


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingModel:
    return get_embedding_model()


@lru_cache(maxsize=1)
def get_world_memory() -> WorldMemory:
    embedder = get_embeddings()
    wm = WorldMemory(embedder.embed)
    return wm


@lru_cache(maxsize=1)
def get_state_service() -> StateService:
    chatter = get_chatter()
    world_memory = get_world_memory()
    return StateService(
        chatter=chatter,
        world_memory=world_memory,
        chatter_reset_callback=lambda: get_chatter.cache_clear(),
    )


def get_conversation_service(
    chatter: Chatter = Depends(get_chatter),
    world_memory: WorldMemory = Depends(get_world_memory),
) -> ConversationService:
    return ConversationService(chatter, world_memory)
