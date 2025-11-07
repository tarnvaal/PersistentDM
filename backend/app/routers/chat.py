from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel

from ..dependencies import get_conversation_service, get_state_service
from ..logging_config import get_logger

logger = get_logger(__name__)


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
def clear_chat(req: ClearRequest, state_service=Depends(get_state_service)):
    if req.clear:
        # Get state summary before reset for audit logging
        pre_reset_summary = state_service.get_state_summary()

        # Perform the reset
        state_service.reset()

        # Audit log the reset
        logger.info(
            "State reset performed",
            action="reset",
            pre_reset_memories=pre_reset_summary.get("memories", 0),
            pre_reset_npcs=pre_reset_summary.get("npc_index", 0),
            pre_reset_locations=pre_reset_summary.get("location_graph", 0),
            reason="user_requested",
        )

        return ClearResponse(success=True)
    else:
        return ClearResponse(success=False)


class StateSummaryResponse(BaseModel):
    memories: int
    npc_index: int
    location_graph: int
    ingest_subgraphs: int
    ingest_memories: int
    ingest_names: int


@router.get("/state/summary", response_model=StateSummaryResponse)
def get_state_summary(state_service=Depends(get_state_service)):
    """Get a summary of current world state for monitoring/debugging."""
    summary = state_service.get_state_summary()
    if "error" in summary:
        raise HTTPException(status_code=500, detail=summary["error"])
    return StateSummaryResponse(**summary)


@router.post("", response_model=ChatResponse)
def post_chat(req: ChatRequest, conversation=Depends(get_conversation_service)):
    try:
        reply, context, relevance = conversation.handle_user_message(req.message)
        return ChatResponse(reply=reply, context=context, relevance=relevance)
    except Exception as e:
        from ..settings import PDM_DEBUG_ERRORS

        detail = {"error": "Internal server error"}
        if PDM_DEBUG_ERRORS:
            detail["message"] = str(e)
        raise HTTPException(status_code=500, detail=detail)


@router.get("/stream")
def stream_chat(
    message: str = Query(..., min_length=0),
    conversation=Depends(get_conversation_service),
):
    """Server-Sent Events: reply first, then context/relevance, then done.

    Note: This endpoint streams staged results for better UX. It uses ConversationService
    to handle context building and emits events in order: reply -> meta -> done.
    Use GET with a query param to support standard EventSource.
    """

    def sse_event(event: str, payload: dict) -> str:
        return f"event: {event}\n" f"data: {json.dumps(payload)}\n\n"

    def generate():
        try:
            # Use ConversationService to handle the complete message processing
            dm_response, merged_context, relevance_payload = (
                conversation.handle_user_message(message)
            )

            yield sse_event("reply", {"reply": dm_response})

            # Emit meta with context and relevance information
            meta = {"context": merged_context, "relevance": relevance_payload or {}}
            yield sse_event("meta", meta)

            yield sse_event("done", {"ok": True})

        except Exception as e:
            # Provide more specific error information in debug mode
            from ..settings import PDM_DEBUG_ERRORS

            error_detail = {"error": "Internal server error"}
            if PDM_DEBUG_ERRORS:
                error_detail["message"] = str(e)
            yield sse_event("error", error_detail)

    return StreamingResponse(generate(), media_type="text/event-stream")
