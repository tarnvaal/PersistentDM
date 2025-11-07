from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

from ..utility.llama import Chatter
from ..settings import (
    MEMORY_K_GENERAL,
    MEMORY_K_PER_ENTITY,
    MEMORY_K_PER_TYPE,
    MEMORY_MIN_TOTAL_SCORE,
    NPC_K_DEFAULT,
    NPC_MIN_SCORE,
    CONFIDENCE_THRESHOLD_MEMORY,
    CONFIDENCE_THRESHOLD_LOCATION,
)
from .memory import WorldMemory
from .context_builder import (
    format_world_facts,
    format_npc_cards,
    format_location_context,
    summarize_memory_context,
)
from .memory_utils import sanitize_entities

logger = logging.getLogger(__name__)


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
        except Exception as e:
            logger.debug(f"Failed to check chatter signature for world_facts: {e}")
            return False

    def _maybe_analyze_and_store_memory(
        self, user_message: str, dm_response: str
    ) -> Optional[Dict[str, Any]]:
        analyze = getattr(self.chatter, "analyze_conversation_for_memories", None)
        if not callable(analyze):
            return None

        conversation_context: Dict[str, Any] = {
            "user_message": user_message,
            "dm_response": dm_response,
            "context": f"Player said: {user_message}\n\nDM responded: {dm_response}",
        }

        try:
            result: object = analyze(conversation_context)  # runtime-typed
        except Exception as e:
            logger.debug(f"Failed to analyze conversation for memories: {e}")
            return None

        summary: Optional[Dict[str, Any]] = result if isinstance(result, dict) else None
        if summary is None:
            return

        try:
            conf = float(summary.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf <= CONFIDENCE_THRESHOLD_MEMORY:
            return None

        entities = sanitize_entities(summary.get("entities"))
        npc_payload = summary.get("npc")
        try:
            self.world_memory.add_memory(
                summary.get("summary", ""),
                entities,
                summary.get("type", "other"),
                npc=npc_payload,
                source_context=conversation_context.get("context"),
            )
            return {
                "summary": summary.get("summary", ""),
                "type": summary.get("type", "other"),
                "entities": entities,
                "npc": npc_payload if isinstance(npc_payload, dict) else None,
                "confidence": conf,
            }
        except Exception as e:
            # Fail-closed; memory storage must not break chats
            logger.debug(f"Failed to store analyzed memory: {e}")
            return None

    def handle_user_message(
        self, user_message: str
    ) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        # If chatter doesn't support world_facts, skip building context entirely
        supports_context = self._chatter_accepts_world_facts()

        merged_context: Optional[str] = None
        relevance_payload: Optional[Dict[str, Any]] = None
        saved_this_turn: Optional[Dict[str, Any]] = None
        if supports_context:
            try:
                # Retrieve memories using multi-index approach for diverse context
                from .context_builder import multi_index_retrieve_with_scores

                mem_scored = multi_index_retrieve_with_scores(
                    self.world_memory,
                    user_message,
                    k_general=MEMORY_K_GENERAL,
                    k_per_entity=MEMORY_K_PER_ENTITY,
                    k_per_type=MEMORY_K_PER_TYPE,
                    min_total_score=MEMORY_MIN_TOTAL_SCORE,
                )
                facts_str = format_world_facts([m["_raw"] for m in mem_scored])

                # Retrieve NPC snapshots with scores and thresholds
                npc_scored = self.world_memory.get_relevant_npc_snapshots_scored(
                    user_message, k=NPC_K_DEFAULT, min_score=NPC_MIN_SCORE
                )
                npc_cards = format_npc_cards([n["_raw"] for n in npc_scored])

                # Retrieve player's current location context
                location_str = format_location_context(self.world_memory)

                parts = []
                if npc_cards:
                    parts.append(npc_cards)
                if facts_str:
                    parts.append(facts_str)
                if location_str:
                    parts.append(location_str)
                merged_context = "\n\n".join(parts) if parts else None

                # Add total word count to context
                if merged_context:
                    word_count = len(merged_context.split())
                    merged_context = f"{merged_context}\n\n[Total: {word_count} words]"

                # Build lightweight relevance payload for UI/debug
                relevance_payload = {
                    "memories": [
                        {
                            "summary": m.get("summary", ""),
                            "type": m.get("type", "unknown"),
                            "entities": m.get("entities", []),
                            "score": round(float(m.get("total", 0.0)), 2),
                            "explanation": summarize_memory_context(m.get("_raw", {})),
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
            except Exception as e:
                logger.debug(f"Failed to build context for message: {e}")
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
            saved_this_turn = self._maybe_analyze_and_store_memory(
                user_message, dm_response
            )
            if relevance_payload is not None:
                relevance_payload["saved"] = saved_this_turn

            # Prefer LLM-guided movement/graph updates; fallback to heuristic
            if not self._maybe_llm_update_location_and_graph(user_message, dm_response):
                self._maybe_update_player_location(user_message, dm_response)

        return dm_response, merged_context, relevance_payload

    def _maybe_update_player_location(
        self, user_message: str, dm_response: str
    ) -> None:
        """Heuristic: move the player if message/reply implies traveling to a connected location.

        Keeps it conservative to avoid wrong moves; checks current location's exits by
        name or description and looks for travel phrases in the DM response.
        """
        loc = self.world_memory.location_graph.get_current_location()
        if not loc or not loc.connections:
            return

        msg = (user_message or "").lower()
        reply = (dm_response or "").lower()

        travel_ok = any(
            phrase in reply
            for phrase in (
                "you go to",
                "you walk to",
                "you head to",
                "you enter",
                "you move to",
            )
        )

        for edge in loc.connections:
            target = (edge.to_location or "").lower()
            edesc = (edge.description or "").lower()
            if not target:
                continue

            if target in msg or edesc and edesc in msg:
                if travel_ok or target in reply:
                    self.world_memory.location_graph.move_player(edge.to_location)
                    break

    def _maybe_llm_update_location_and_graph(
        self, user_message: str, dm_response: str
    ) -> bool:
        """Use the LLM to infer movement and extract graph changes. Returns True if any update applied.

        NOTE: Ingested locations/memories are persisted and loaded into an ingest layer.
        They are not injected into the live session graph. After restart, lore is available
        for retrieval, but navigation into ingest-only locations will not work until merged
        view + traversal logic is implemented.
        """
        updated = False
        loc = self.world_memory.location_graph.get_current_location()
        try:
            exits = []
            if loc:
                for e in loc.connections:
                    exits.append(
                        {
                            "to_location": e.to_location,
                            "description": e.description,
                            "travel_verb": e.travel_verb,
                        }
                    )
            # 1) Movement inference
            mv = getattr(self.chatter, "infer_player_movement", None)
            if callable(mv):
                res = mv(
                    loc.__dict__ if loc else None, exits, user_message, dm_response
                )
                if isinstance(res, dict) and res.get("move") and res.get("target"):
                    try:
                        conf = float(res.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    if conf >= CONFIDENCE_THRESHOLD_LOCATION:
                        target = str(res.get("target"))
                        if self.world_memory.location_graph.move_player(target):
                            updated = True

            # 2) Graph extraction (new nodes/edges)
            gx = getattr(self.chatter, "extract_graph_changes", None)
            if callable(gx):
                res2 = gx(user_message, dm_response, loc.name if loc else None)
                if isinstance(res2, dict):
                    try:
                        conf2 = float(res2.get("confidence", 0.0))
                    except Exception:
                        conf2 = 0.0
                    if conf2 >= CONFIDENCE_THRESHOLD_LOCATION:
                        # Add locations
                        for node in res2.get("new_locations", []) or []:
                            name = str(node.get("name", "")).strip()
                            if (
                                name
                                and name
                                not in self.world_memory.location_graph.locations
                            ):
                                desc = str(node.get("description", "")).strip()
                                from .memory import (
                                    LocationNode,
                                )  # local import to avoid cycle

                                self.world_memory.location_graph.add_location(
                                    LocationNode(name, desc)
                                )
                                updated = True
                        # Add connections
                        for edge in res2.get("new_connections", []) or []:
                            src = str(edge.get("from", "")).strip()
                            dst = str(edge.get("to", "")).strip()
                            edesc = str(edge.get("description", "")).strip()
                            if src and dst and edesc:
                                self.world_memory.location_graph.add_connection(
                                    src, dst, edesc
                                )
                                updated = True
        except Exception as e:
            # Never let LLM graph/movement updates break the chat
            logger.debug(f"Failed to update location/graph via LLM: {e}")
            return updated
        return updated
