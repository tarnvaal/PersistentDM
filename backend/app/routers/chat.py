import os
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel

from ..dependencies import get_conversation_service, reset_chatter, get_world_memory


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    context: str | None = None
    relevance: dict | None = None


class ClearRequest(BaseModel):
    clear: bool


class ClearResponse(BaseModel):
    success: bool


@router.post("/clear", response_model=ClearResponse)
def clear_chat(req: ClearRequest):
    if req.clear:
        reset_chatter()
        # Also clear world memory and NPC index so the game state resets
        wm = get_world_memory()
        wm.memories.clear()
        wm.npc_index.clear()
        return ClearResponse(success=True)
    else:
        return ClearResponse(success=False)


@router.post("", response_model=ChatResponse)
def post_chat(req: ChatRequest, conversation=Depends(get_conversation_service)):
    try:
        reply, context, relevance = conversation.handle_user_message(req.message)
        return ChatResponse(reply=reply, context=context, relevance=relevance)
    except Exception as e:
        expose = os.getenv("PDM_DEBUG_ERRORS", "1") == "1"
        detail = {"error": "Internal server error"}
        if expose:
            detail["message"] = str(e)
        raise HTTPException(status_code=500, detail=detail)


@router.get("/stream")
def stream_chat(
    message: str = Query(..., min_length=0),
    conversation=Depends(get_conversation_service),
):
    """Server-Sent Events: reply first, then context/relevance, then done.

    Note: This endpoint streams staged results for better UX. It mirrors /chat logic but
    emits events in order: reply -> meta -> done. Use GET with a query param to support
    standard EventSource.
    """

    def sse_event(event: str, payload: dict) -> str:
        return f"event: {event}\n" f"data: {json.dumps(payload)}\n\n"

    def generate():
        try:
            # Build context similarly to ConversationService but inline to stage results
            conv = conversation
            supports_context = True
            try:
                supports_context = conv._chatter_accepts_world_facts()
            except Exception:
                supports_context = True

            merged_context = None
            relevance_payload = None
            if supports_context:
                try:
                    mem_scored = []
                    try:
                        mem_scored = weighted_retrieve_with_scores(
                            conv.world_memory, message, k=4, min_total_score=0.25
                        )
                    except Exception:
                        mem_scored = []
                    facts_str = format_world_facts(
                        [m.get("_raw", m) for m in mem_scored]
                    )
                    npc_scored = conv.world_memory.get_relevant_npc_snapshots_scored(
                        message, k=2, min_score=0.35
                    )
                    npc_cards = format_npc_cards([n.get("_raw", n) for n in npc_scored])
                    location_str = format_location_context(conv.world_memory)
                    parts = []
                    if npc_cards:
                        parts.append(npc_cards)
                    if facts_str:
                        parts.append(facts_str)
                    if location_str:
                        parts.append(location_str)
                    merged_context = "\n\n".join(parts) if parts else None
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

            # Generate DM reply
            try:
                if supports_context and merged_context is not None:
                    dm_response = conv.chatter.chat(message, world_facts=merged_context)
                else:
                    dm_response = conv.chatter.chat(message)
            except TypeError as e:
                msg = str(e)
                if "world_facts" in msg and "unexpected keyword" in msg:
                    dm_response = conv.chatter.chat(message)
                else:
                    raise

            yield sse_event("reply", {"reply": dm_response})

            # Analyze/store memory and emit meta
            saved = None
            try:
                if supports_context:
                    saved = conv._maybe_analyze_and_store_memory(message, dm_response)
            except Exception:
                saved = None

            meta = {"context": merged_context, "relevance": relevance_payload or {}}
            if meta["relevance"] is not None:
                meta["relevance"]["saved"] = saved
            yield sse_event("meta", meta)

            yield sse_event("done", {"ok": True})

        except Exception:
            yield sse_event("error", {"error": "Internal server error"})

    # Import functions used above
    from ..world.context_builder import (
        weighted_retrieve_with_scores,
        format_world_facts,
        format_npc_cards,
        format_location_context,
    )

    return StreamingResponse(generate(), media_type="text/event-stream")
