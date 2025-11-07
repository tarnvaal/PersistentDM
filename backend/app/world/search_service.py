import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable

from .memory import WorldMemory
from .scoring import score_memory_item
from .search_schemas import SourceInfo
from ..config.search_config import SearchConfig
from ..utility.embeddings import get_embedding_model


class SearchService:
    """Service for searching memories with hybrid ranking (semantic + literal + recency + type)."""

    def __init__(
        self,
        memory_store: WorldMemory,
        embedder: Optional[Callable[[str], List[float]]] = None,
        config: Optional[SearchConfig] = None,
    ):
        self.memory_store = memory_store
        self.embedder = embedder or get_embedding_model().embed
        self.config = config or SearchConfig()

    def _gather_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories (session + ingest) with consistent format."""
        from .context_builder import _gather_all_memories

        return _gather_all_memories(self.memory_store)

    def _apply_filters(
        self,
        memories: List[Dict[str, Any]],
        types: Optional[List[str]] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Apply type and time filters to memories."""
        filtered = memories

        # Filter by types
        if types:
            filtered = [m for m in filtered if m.get("type") in types]

        # Filter by timestamp
        if since:
            since_timestamp = since.timestamp()
            filtered = [m for m in filtered if m.get("timestamp", 0) >= since_timestamp]

        return filtered

    def _get_memory_text(self, memory: Dict[str, Any]) -> str:
        """Extract searchable text from memory item."""
        # Prefer explicit text when present; fallback to embedding text builder
        text = memory.get("text")
        if isinstance(text, str) and text.strip():
            return text
        from .memory import _build_memory_text_for_embedding

        return _build_memory_text_for_embedding(memory)

    def _prepare_memory_for_search(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare memory item for search by ensuring required fields."""
        prepared = dict(memory)  # Copy

        # Ensure text field
        if "text" not in prepared:
            prepared["text"] = self._get_memory_text(prepared)

        # Normalize timestamp/updated_at
        ts_val = prepared.get("timestamp")
        upd_val = prepared.get("updated_at")
        # If updated_at is a datetime, derive timestamp
        if isinstance(upd_val, datetime):
            try:
                ts_val = float(upd_val.timestamp())
            except Exception:
                ts_val = ts_val or time.time()
        # If timestamp exists but is a datetime, convert to float
        if isinstance(ts_val, datetime):
            try:
                ts_val = float(ts_val.timestamp())
            except Exception:
                ts_val = time.time()
        # If still missing, fall back
        if not isinstance(ts_val, (int, float)):
            try:
                ts_val = float(prepared.get("updated_at", 0))
            except Exception:
                ts_val = time.time()
        prepared["timestamp"] = ts_val
        # Ensure updated_at is a timezone-aware datetime
        if not isinstance(upd_val, datetime):
            prepared["updated_at"] = datetime.fromtimestamp(ts_val, tz=timezone.utc)

        return prepared

    def _search_literal(
        self, query: str, candidates: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        """Literal substring search only."""
        query_lower = query.lower().strip()
        matches = []

        for memory in candidates:
            text = self._get_memory_text(memory).lower()
            if query_lower in text:
                prepared = self._prepare_memory_for_search(memory)
                score_breakdown = {
                    "total": 1.0,  # Simple binary score for literal matches
                    "similarity": 0.0,
                    "literal_boost": 1.0,
                    "recency_bonus": 0.0,
                    "type_bonus": 0.0,
                }
                result = {
                    **prepared,
                    "score": score_breakdown["total"],
                    "explanation": score_breakdown,
                }
                matches.append(result)

        # Sort by recency (most recent first) for literal matches
        matches.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return matches[:k]

    def _search_semantic(
        self,
        query: str,
        query_vector: List[float],
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings only."""
        scored_results = []

        for memory in candidates:
            prepared = self._prepare_memory_for_search(memory)
            score_breakdown = score_memory_item(
                query,
                query_vector,
                prepared,
                {
                    "weights": {
                        "w_sim": 1.0,
                        "w_literal": 0.0,
                        "w_rec": 0.0,
                        "w_type": 0.0,
                    },
                    "half_life_hours": self.config.half_life_hours,
                    "type_bonus_map": self.config.type_bonus_map,
                    "literal_boost_value": self.config.literal_boost_value,
                },
            )

            result = {
                **prepared,
                "score": score_breakdown["total"],
                "explanation": score_breakdown,
            }
            scored_results.append(result)

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:k]

    def _search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining all scoring components."""
        scored_results = []

        for memory in candidates:
            prepared = self._prepare_memory_for_search(memory)
            score_breakdown = score_memory_item(
                query,
                query_vector,
                prepared,
                {
                    "weights": self.config.weights,
                    "half_life_hours": self.config.half_life_hours,
                    "type_bonus_map": self.config.type_bonus_map,
                    "literal_boost_value": self.config.literal_boost_value,
                },
            )

            result = {
                **prepared,
                "score": score_breakdown["total"],
                "explanation": score_breakdown,
            }
            scored_results.append(result)

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:k]

    def search(
        self,
        q: str,
        mode: str = "hybrid",
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform search across all memories.

        Args:
            q: Search query string
            mode: Search mode ("literal", "semantic", "hybrid")
            k: Maximum number of results
            filters: Optional filters dict with "types" and "since" keys

        Returns:
            Search response dict
        """
        # Cap k to 100
        k_capped = min(int(k), 100)

        # Parse filters
        types_filter = None
        since_filter = None
        if filters:
            if "types" in filters and filters["types"]:
                types_filter = (
                    filters["types"]
                    if isinstance(filters["types"], list)
                    else [filters["types"]]
                )
            since_filter = filters.get("since")
        # Merge explicit types param
        if types is not None:
            addl = [t.strip() for t in types if isinstance(t, str) and t.strip()]
            types_filter = addl if not types_filter else list({*types_filter, *addl})

        # Get all candidate memories (read-only snapshot)
        all_memories = self._gather_all_memories()
        candidates = self._apply_filters(all_memories, types_filter, since_filter)

        # Perform search based on mode
        if mode == "literal":
            results = self._search_literal(q, candidates, k_capped)
        elif mode == "semantic":
            query_vector = self.embedder(q)
            results = self._search_semantic(q, query_vector, candidates, k_capped)
        else:  # hybrid
            query_vector = self.embedder(q)
            results = self._search_hybrid(q, query_vector, candidates, k_capped)

        # Format results for API response
        formatted_results = []
        for result in results:
            # Generate item_id if not present
            item_id = result.get("id") or result.get("item_id") or str(uuid.uuid4())

            # Determine source information
            if result.get("ingest_id"):
                source = SourceInfo(shard=result["ingest_id"], origin="ingest")
            else:
                source = SourceInfo(shard="session", origin="memory")

            formatted_result = {
                "item_id": item_id,
                "type": result.get("type", ""),
                "text": result.get("text", ""),
                "score": result["score"],
                "explanation": result["explanation"],
                "updated_at": result.get("updated_at", datetime.now(timezone.utc)),
                "source": source,
            }
            formatted_results.append(formatted_result)

        return {"query": q, "mode": mode, "k": k_capped, "results": formatted_results}
