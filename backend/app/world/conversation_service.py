from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from ..utility.llama import Chatter
from .memory import WorldMemory
from .context_builder import (
    weighted_retrieve_with_scores,
    format_world_facts,
    format_npc_cards,
)
from .memory_utils import sanitize_entities


class ConversationService:
    """Server-side orchestration for building context and handling chat turns."""

    def __init__(self, chatter: Chatter, world_memory: WorldMemory):
        self.chatter = chatter
        self.world_memory = world_memory

    def _chatter_accepts_world_facts(self) -> bool:
        try:
            sig = inspect.signature(self.chatter.chat)
            return any(
                p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name == "world_facts"
                for p in sig.parameters.values()
            )
        except Exception:
            return False

    def _maybe_analyze_and_store_memory(
        self, user_message: str, dm_response: str
    ) -> None:
        analyze = getattr(self.chatter, "analyze_conversation_for_memories", None)
        if not callable(analyze):
            return

        conversation_context: Dict[str, Any] = {
            "user_message": user_message,
            "dm_response": dm_response,
            "context": f"Player said: {user_message}\n\nDM responded: {dm_response}",
        }

        try:
            result: object = analyze(conversation_context)  # runtime-typed
        except Exception:
            return

        summary: Optional[Dict[str, Any]] = result if isinstance(result, dict) else None
        if summary is None:
            return

        try:
            conf = float(summary.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf <= 0.6:
            return

        entities = sanitize_entities(summary.get("entities"))
        npc_payload = summary.get("npc")
        try:
            self.world_memory.add_memory(
                summary.get("summary", ""),
                entities,
                summary.get("type", "other"),
                npc=npc_payload,
            )
        except Exception:
            # Fail-closed; memory storage must not break chats
            return

    def handle_user_message(
        self, user_message: str
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        # If chatter doesn't support world_facts, skip building context entirely
        supports_context = self._chatter_accepts_world_facts()

        merged_context: Optional[str] = None
        relevance_payload: Optional[Dict[str, Any]] = None
        if supports_context:
            try:
                # Retrieve memories with scores and thresholds
                mem_scored = weighted_retrieve_with_scores(
                    self.world_memory, user_message, k=4, min_total_score=0.25
                )
                facts_str = format_world_facts([m["_raw"] for m in mem_scored])

                # Retrieve NPC snapshots with scores and thresholds
                npc_scored = self.world_memory.get_relevant_npc_snapshots_scored(
                    user_message, k=2, min_score=0.35
                )
                npc_cards = format_npc_cards([n["_raw"] for n in npc_scored])

                if npc_cards and facts_str:
                    merged_context = npc_cards + "\n\n" + facts_str
                elif npc_cards:
                    merged_context = npc_cards
                else:
                    merged_context = facts_str or None

                # Build lightweight relevance payload for UI/debug
                relevance_payload = {
                    "memories": [
                        {
                            "summary": m.get("summary", ""),
                            "type": m.get("type", "unknown"),
                            "entities": m.get("entities", []),
                            "score": round(float(m.get("total", 0.0)), 2),
                        }
                        for m in mem_scored
                    ],
                    "npcs": [
                        {
                            "name": n.get("name", "Unknown"),
                            "intent": n.get("intent"),
                            "last_seen_location": n.get("last_seen_location"),
                            "relationship_to_player": n.get(
                                "relationship_to_player", "unknown"
                            ),
                            "score": round(float(n.get("score", 0.0)), 2),
                        }
                        for n in npc_scored
                    ],
                }
            except Exception:
                merged_context = None
                relevance_payload = None

        # Call chatter with or without world_facts depending on signature support
        if supports_context and merged_context is not None:
            try:
                dm_response = self.chatter.chat(
                    user_message, world_facts=merged_context
                )
            except TypeError as e:
                # Only fallback on actual signature mismatch for world_facts
                msg = str(e)
                if "world_facts" in msg and "unexpected keyword" in msg:
                    dm_response = self.chatter.chat(user_message)
                else:
                    # Re-raise to avoid masking real TypeError inside chat()
                    raise
        else:
            dm_response = self.chatter.chat(user_message)

        # Only analyze/store memory if chatter provides analyzer and we could build context
        if supports_context:
            self._maybe_analyze_and_store_memory(user_message, dm_response)

        return dm_response, merged_context, relevance_payload
