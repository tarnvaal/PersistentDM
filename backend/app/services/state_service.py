# app/services/state_service.py

import threading
from typing import Protocol, Callable, Dict, Any

from backend.app.utility.llama import Chatter


class Resettable(Protocol):
    def reset(self) -> None: ...

    def state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state."""
        ...


class StateService:
    def __init__(
        self,
        chatter: Chatter,
        world_memory: Resettable,
        chatter_reset_callback: Callable[[], None],
    ):
        self.chatter = chatter
        self.world_memory = world_memory
        self._chatter_reset_callback = chatter_reset_callback
        self._reset_lock = threading.RLock()

    def reset(self) -> None:
        """Reset both the chatter and world memory state.

        This clears:
        - Chatter model cache (forces reload on next use)
        - All session memories, NPC index, and location graph
        - In-memory ingest layers (persistent ingest shards on disk are untouched)

        This does NOT clear:
        - Persistent ingest data stored on disk
        - Any cached embeddings or model state beyond the chatter instance

        Idempotent and thread-safe - safe to call multiple times concurrently.
        """
        with self._reset_lock:
            # Reset chatter via callback (clears lru_cache)
            self._chatter_reset_callback()
            # Reset world memory state
            self.world_memory.reset()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for monitoring/debugging."""
        return self.world_memory.state_summary()
