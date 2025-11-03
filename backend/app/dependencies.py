import os
from functools import lru_cache
from fastapi import Depends

from .utility.llama import Chatter
from .utility.embeddings import get_embedding_model, EmbeddingModel
from .world.memory import WorldMemory
from .world.conversation_service import ConversationService


DEFAULT_MODEL_PATH = "~/dev/llm/Harbinger-24B-Q5_K_M.gguf"


@lru_cache(maxsize=1)
def get_chatter() -> Chatter:
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    return Chatter(model_path)


def reset_chatter() -> Chatter:
    get_chatter.cache_clear()
    return get_chatter()


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingModel:
    return get_embedding_model()


@lru_cache(maxsize=1)
def get_world_memory() -> WorldMemory:
    embedder = get_embeddings()
    wm = WorldMemory(embedder.embed)
    return wm


def get_conversation_service(
    chatter: Chatter = Depends(get_chatter),
    world_memory: WorldMemory = Depends(get_world_memory),
) -> ConversationService:
    return ConversationService(chatter, world_memory)
