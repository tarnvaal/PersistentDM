from __future__ import annotations

import json
import uuid
import os
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..dependencies import get_conversation_service
from ..world.memory_utils import sanitize_entities
from ..world.memory import LocationNode
from ..world.context_builder import summarize_memory_context


router = APIRouter(prefix="/ingest", tags=["ingest"])


# Simple in-process upload store (development only)
_uploads: Dict[str, str] = {}


class UploadRequest(BaseModel):
    text: str


class UploadResponse(BaseModel):
    id: str
    totalWords: int
    totalLines: int


class RenameRequest(BaseModel):
    name: str


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


def _ingests_dir() -> str:
    """Resolve ingests directory to an absolute path.

    Preference order: env INGESTS_DIR -> project-root/data/ingests
    """
    env_dir = os.getenv("INGESTS_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(project_root, "data", "ingests")


@router.get("/list")
def list_ingests():
    base_dir = _ingests_dir()
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass
    results = []
    try:
        for fname in os.listdir(base_dir):
            if not fname.endswith(".json"):
                continue
            ingest_id = fname[:-5]
            name = None
            fp = os.path.join(base_dir, fname)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("name"), str) and data["name"].strip():
                    name = data["name"].strip()
                subgraph_raw = data.get("subgraph", {}) or {}
                memories_raw = data.get("memories", []) or []
                loc_count = len(subgraph_raw) if isinstance(subgraph_raw, dict) else 0
                mem_count = len(memories_raw) if isinstance(memories_raw, list) else 0
                try:
                    file_bytes = os.path.getsize(fp)
                except Exception:
                    file_bytes = 0
                results.append(
                    {
                        "id": ingest_id,
                        "name": name or ingest_id,
                        "locations": loc_count,
                        "memories": mem_count,
                        "bytes": file_bytes,
                    }
                )
            except Exception:
                results.append({"id": ingest_id, "name": name or ingest_id})
    except Exception:
        results = []
    # sort by name for now
    results.sort(key=lambda x: (x.get("name") or "").lower())
    return {"ingests": results}


@router.put("/shard/{ingest_id}/name")
def rename_ingest(
    ingest_id: str, body: RenameRequest, conversation=Depends(get_conversation_service)
):
    base_dir = _ingests_dir()
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{ingest_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": "not found"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = " ".join((body.name or "").strip().split())[:120]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        # update in-memory
        try:
            conversation.world_memory.set_ingest_name(ingest_id, data["name"])
        except Exception:
            pass
        return {"ok": True, "id": ingest_id, "name": data["name"]}
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "rename failed"})


@router.delete("/shard/{ingest_id}")
def delete_ingest(ingest_id: str, conversation=Depends(get_conversation_service)):
    base_dir = _ingests_dir()
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{ingest_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": "not found"})
    try:
        os.remove(path)
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "delete failed"})
    # clear in-memory copy if present
    try:
        wm = conversation.world_memory
        wm.ingest_subgraphs.pop(ingest_id, None)
        wm.ingest_memories.pop(ingest_id, None)
        wm.ingest_names.pop(ingest_id, None)
    except Exception:
        pass
    return {"ok": True, "id": ingest_id}


@router.post("/shard/{ingest_id}/load")
def load_ingest(ingest_id: str, conversation=Depends(get_conversation_service)):
    base_dir = _ingests_dir()
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{ingest_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": "not found"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        subgraph_raw = data.get("subgraph", {}) or {}
        memories_raw = data.get("memories", []) or []
        name_raw = data.get("name")
        wm = conversation.world_memory
        shard_graph = {}
        for name, nd in subgraph_raw.items():
            try:
                shard_graph[name] = LocationNode.from_dict(nd)
            except Exception:
                continue
        wm.ingest_subgraphs[ingest_id] = shard_graph
        wm.ingest_memories[ingest_id] = [m for m in memories_raw if isinstance(m, dict)]
        if isinstance(name_raw, str) and name_raw.strip():
            wm.ingest_names[ingest_id] = name_raw.strip()[:120]
        # stats
        loc_count = len(shard_graph)
        mem_count = len(wm.ingest_memories.get(ingest_id, []))
        try:
            file_bytes = os.path.getsize(path)
        except Exception:
            file_bytes = 0
        return {
            "ok": True,
            "id": ingest_id,
            "name": wm.ingest_names.get(ingest_id),
            "locations": loc_count,
            "memories": mem_count,
            "bytes": file_bytes,
        }
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "load failed"})


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
    ingest_id = id
    # Ensure shard containers exist for this ingest run
    try:
        world_memory.ensure_ingest_shard(ingest_id)
    except Exception:
        pass

    # Generate a concise ingest name once per shard using the raw text
    try:
        if not world_memory.ingest_names.get(ingest_id):
            sample = (text or "").strip()
            sample = sample[:1200]
            prompt = (
                "Summarize the following text into a concise 4-8 word title that captures its setting or theme. "
                "Output only the title without quotes.\n\n" + sample
            )
            try:
                title = chatter.chat(prompt)
            except Exception:
                title = None
            if isinstance(title, str):
                # take first line, strip markdowny artifacts
                name = title.splitlines()[0].strip().strip("# ")
                world_memory.set_ingest_name(ingest_id, name)
    except Exception:
        pass

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
                            # Provide a brief provenance for explanations
                            # Use a short snippet of the current chunk as source context
                            snippet = (chunk_text or "").strip()
                            if len(snippet) > 300:
                                snippet = snippet[:299] + "…"
                            source_context = f"Ingested: {snippet}" if snippet else None

                            # Store in ingest layer for this shard (no session mutation)
                            try:
                                entry = {
                                    "summary": mem.get("summary", ""),
                                    "type": mem.get("type", "other"),
                                    "entities": entities,
                                    "npc": npc_payload
                                    if isinstance(npc_payload, dict)
                                    else None,
                                    "confidence": conf,
                                    "timestamp": time.time(),
                                    "source_context": source_context,
                                }
                                world_memory.add_ingest_memory(ingest_id, entry)
                            except Exception:
                                pass
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
                                if loc_name:
                                    try:
                                        shard = world_memory.ingest_subgraphs.get(
                                            ingest_id, {}
                                        )
                                        if loc_name not in shard:
                                            world_memory.upsert_ingest_location(
                                                ingest_id,
                                                LocationNode(
                                                    loc_name, mem.get("summary", "")
                                                ),
                                            )
                                    except Exception:
                                        pass

                            try:
                                explanation = (
                                    summarize_memory_context(
                                        {"source_context": source_context}
                                    )
                                    if source_context
                                    else None
                                )
                                yield _sse(
                                    "saved",
                                    {
                                        "ingestId": ingest_id,
                                        "summary": mem.get("summary", ""),
                                        "type": mem.get("type", "other"),
                                        "entities": entities,
                                        "npc": npc_payload
                                        if isinstance(npc_payload, dict)
                                        else None,
                                        "confidence": conf,
                                        "explanation": explanation,
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
                # Persist shard before signaling done so it survives restarts
                try:
                    base_dir = _ingests_dir()
                    world_memory.persist_ingest_shard(ingest_id, base_dir)
                except Exception:
                    pass
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
