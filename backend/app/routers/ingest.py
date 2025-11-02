from __future__ import annotations

import json
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..dependencies import get_conversation_service
from ..world.memory_utils import sanitize_entities
from ..world.memory import LocationNode


router = APIRouter(prefix="/ingest", tags=["ingest"])


# Simple in-process upload store (development only)
_uploads: Dict[str, str] = {}


class UploadRequest(BaseModel):
    text: str


class UploadResponse(BaseModel):
    id: str
    totalWords: int
    totalLines: int


@router.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest) -> UploadResponse:
    text = req.text or ""
    uid = str(uuid.uuid4())
    _uploads[uid] = text
    total_words = len(text.strip().split()) if text.strip() else 0
    total_lines = (text.count("\n") + 1) if text else 0
    return UploadResponse(id=uid, totalWords=total_words, totalLines=total_lines)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(payload)}\n\n"


@router.get("/stream")
def stream(
    request: Request,
    id: str = Query(...),
    stride_words_override: Optional[int] = Query(None, alias="strideWords"),
    conversation=Depends(get_conversation_service),
):
    text = _uploads.pop(id, None)
    if text is None:
        raise HTTPException(status_code=404, detail={"error": "not found"})

    # Chunk by words using token approximation: ~1.3 tokens/word
    words = text.split() if text.strip() else []
    total_words = len(words)
    total_lines = (text.count("\n") + 1) if text else 0
    tokens_per_word = (
        1.3
        if total_words == 0
        else max(0.5, min(2.0, (len(text) / 4) / max(1, total_words)))
    )
    # Tokenization granularity (approx): process 200 tokens with stride 100
    window_tokens = 200
    stride_tokens = 100
    window_words = max(1, int(window_tokens / tokens_per_word))
    if stride_words_override is not None:
        try:
            stride_words = max(1, min(5000, int(stride_words_override)))
        except Exception:
            stride_words = max(1, int(stride_tokens / tokens_per_word))
    else:
        stride_words = max(1, int(stride_tokens / tokens_per_word))

    total_steps = max(
        1, ((max(0, total_words - window_words) + stride_words - 1) // stride_words) + 1
    )
    approx_tokens = int(round(total_words * tokens_per_word))
    # Every ~1000 tokens advanced, run summarization + hygiene
    checkpoint_tokens = 1000
    checkpoint_step_interval = max(
        1, (checkpoint_tokens + stride_tokens - 1) // stride_tokens
    )

    chatter = conversation.chatter
    world_memory = conversation.world_memory

    def _norm(s: Optional[str]) -> str:
        return " ".join((s or "").strip().lower().split())

    def _node_descriptor(node) -> str:
        parts = [
            node.get("name") if isinstance(node, dict) else getattr(node, "name", None)
        ]
        aliases = []
        if isinstance(node, dict):
            aliases = node.get("aliases") or []
            desc = node.get("description") or ""
        else:
            aliases = getattr(node, "aliases", []) or []
            desc = getattr(node, "description", "")
        if aliases:
            parts.extend(aliases)
        if desc:
            parts.append(desc)
        return " | ".join([p for p in parts if p])

    def generate():
        try:
            # Initial info for UI panel
            try:
                yield _sse(
                    "info",
                    {
                        "words": total_words,
                        "lines": total_lines,
                        "approxTokens": approx_tokens,
                        "windowTokens": window_tokens,
                        "strideTokens": stride_tokens,
                        "windowWords": window_words,
                        "strideWords": stride_words,
                        "totalSteps": total_steps,
                        "checkpointTokenInterval": checkpoint_tokens,
                        "checkpointStepInterval": checkpoint_step_interval,
                    },
                )
            except GeneratorExit:
                return
            consumed_words = 0
            for step in range(total_steps):
                # Check for client disconnection by attempting a small probe yield
                # If client disconnected, GeneratorExit will be raised and caught by outer handler
                start_idx = step * stride_words
                end_idx = min(total_words, start_idx + window_words)
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                # Emit memory (conservative)
                try:
                    # Prefer multi-extractor
                    multi = getattr(chatter, "extract_memories_from_text", None)
                    single = getattr(chatter, "extract_memory_from_text", None)
                    extracted = []
                    if callable(multi):
                        items = multi(chunk_text, max_items=5) or []
                        if isinstance(items, list):
                            extracted = [m for m in items if isinstance(m, dict)]
                    if not extracted and callable(single):
                        m = single(chunk_text)
                        if isinstance(m, dict):
                            extracted = [m]

                    for mem in extracted:
                        try:
                            conf = float(mem.get("confidence", 0.0))
                        except Exception:
                            conf = 0.0
                        if conf < 0.7:
                            continue
                        entities = sanitize_entities(mem.get("entities"))
                        npc_payload = mem.get("npc")
                        try:
                            world_memory.add_memory(
                                mem.get("summary", ""),
                                entities,
                                mem.get("type", "other"),
                                npc=npc_payload,
                            )
                            # If this appears to be a location, upsert a node for hygiene to work
                            if str(mem.get("type", "")).lower() == "location":
                                loc_name = None
                                if entities:
                                    loc_name = str(entities[0]).strip()
                                if not loc_name:
                                    # crude fallback: take first phrase from summary
                                    loc_name = (
                                        mem.get("summary", "").split(" is ", 1)[0] or ""
                                    ).strip() or None
                                if (
                                    loc_name
                                    and loc_name
                                    not in world_memory.location_graph.locations
                                ):
                                    world_memory.location_graph.add_location(
                                        LocationNode(loc_name, mem.get("summary", ""))
                                    )

                            try:
                                yield _sse(
                                    "saved",
                                    {
                                        "summary": mem.get("summary", ""),
                                        "type": mem.get("type", "other"),
                                        "entities": entities,
                                        "npc": npc_payload
                                        if isinstance(npc_payload, dict)
                                        else None,
                                        "confidence": conf,
                                    },
                                )
                            except GeneratorExit:
                                return
                        except Exception:
                            pass
                except Exception:
                    pass

                # Checkpoint summary emission removed

                # Graph hygiene removed

                # Progress update
                consumed_words = min(total_words, (step + 1) * stride_words)
                ratio = (
                    0.0 if total_words == 0 else min(1.0, consumed_words / total_words)
                )
                approx_lines = int(total_lines * ratio)
                try:
                    yield _sse(
                        "progress",
                        {
                            "step": step,
                            "totalSteps": total_steps,
                            "consumedWords": consumed_words,
                            "consumedLines": approx_lines,
                            "progress": ratio,
                        },
                    )
                except GeneratorExit:
                    return

            try:
                yield _sse(
                    "done",
                    {
                        "words": total_words,
                        "lines": total_lines,
                        "steps": total_steps,
                    },
                )
            except GeneratorExit:
                return
        except GeneratorExit:
            # Client disconnected - exit cleanly
            return

    return StreamingResponse(generate(), media_type="text/event-stream")
