from __future__ import annotations

import json
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..dependencies import get_conversation_service
from ..world.memory_utils import sanitize_entities
from ..utility.embeddings import dot_sim
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
def stream(id: str = Query(...), conversation=Depends(get_conversation_service)):
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

    def _graph_hygiene(world_memory) -> dict:
        g = world_memory.location_graph
        merged = 0
        pruned_nodes = 0
        pruned_edges = 0

        # Build vectors for nodes
        names = list(g.locations.keys())
        vecs = {}
        for n in names:
            node = g.locations[n]
            desc = _node_descriptor(
                {
                    "name": node.name,
                    "aliases": getattr(node, "aliases", []),
                    "description": getattr(node, "description", ""),
                }
            )
            vecs[n] = world_memory.embed_fn(desc)

        # Merge highly similar nodes
        threshold = 0.88
        removed = set()
        for i in range(len(names)):
            a = names[i]
            if a in removed or a not in g.locations:
                continue
            for j in range(i + 1, len(names)):
                b = names[j]
                if b in removed or b not in g.locations:
                    continue
                if _norm(a) == _norm(b):
                    # treat identical after normalization as duplicates
                    # keep 'a', remove 'b'
                    node_a = g.locations[a]
                    node_b = g.locations[b]
                    # merge simple fields
                    if getattr(node_b, "description", "") and not getattr(
                        node_a, "description", ""
                    ):
                        node_a.description = node_b.description
                    # merge aliases if present
                    if hasattr(node_a, "aliases") and hasattr(node_b, "aliases"):
                        node_a.aliases = list(
                            {*(node_a.aliases or []), *(node_b.aliases or [])}
                        )
                    # move edges
                    for e in list(node_b.connections):
                        node_a.connections.append(e)
                    # redirect edges from others pointing to b
                    for k, nnode in g.locations.items():
                        for e in nnode.connections:
                            if e.to_location == b:
                                e.to_location = a
                    del g.locations[b]
                    removed.add(b)
                    merged += 1
                    continue
                # cosine similarity
                s = dot_sim(vecs[a], vecs[b]) if (a in vecs and b in vecs) else 0.0
                if s >= threshold:
                    # merge b into a (simple strategy)
                    node_a = g.locations[a]
                    node_b = g.locations[b]
                    if getattr(node_b, "description", "") and not getattr(
                        node_a, "description", ""
                    ):
                        node_a.description = node_b.description
                    if hasattr(node_a, "aliases") and hasattr(node_b, "aliases"):
                        node_a.aliases = list(
                            {*(node_a.aliases or []), *(node_b.aliases or [])}
                        )
                    for e in list(node_b.connections):
                        node_a.connections.append(e)
                    for k, nnode in g.locations.items():
                        for e in nnode.connections:
                            if e.to_location == b:
                                e.to_location = a
                    del g.locations[b]
                    removed.add(b)
                    merged += 1

        # Dedupe edges by to_location
        for k, node in g.locations.items():
            seen = set()
            new_conns = []
            for e in node.connections:
                key = (e.to_location, _norm(e.description))
                if key in seen:
                    pruned_edges += 1
                    continue
                seen.add(key)
                new_conns.append(e)
            node.connections = new_conns

        # Compute references to node names in memories
        refs = {}
        for k in g.locations.keys():
            refs[k] = 0
        for mem in world_memory.memories:
            for ent in mem.get("entities", []) or []:
                entn = _norm(str(ent))
                for name in list(g.locations.keys()):
                    if entn == _norm(name):
                        refs[name] = refs.get(name, 0) + 1

        # Prune isolated, unreferenced nodes
        to_delete = []
        for name, node in g.locations.items():
            out_deg = len(node.connections)
            in_deg = 0
            for other in g.locations.values():
                for e in other.connections:
                    if e.to_location == name:
                        in_deg += 1
            if out_deg == 0 and in_deg == 0 and refs.get(name, 0) == 0:
                to_delete.append(name)
        for name in to_delete:
            del g.locations[name]
            pruned_nodes += 1

        return {
            "merged": merged,
            "pruned_nodes": pruned_nodes,
            "pruned_edges": pruned_edges,
        }

    def generate():
        # Initial info for UI panel
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
        consumed_words = 0
        for step in range(total_steps):
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
                    except Exception:
                        pass
            except Exception:
                pass

            # Every checkpointStepInterval steps provide a small checkpoint summary (not persisted)
            if (step + 1) % checkpoint_step_interval == 0:
                try:
                    summarizer = getattr(chatter, "summarize_snippet", None)
                    if callable(summarizer):
                        cp = summarizer(chunk_text)
                        if isinstance(cp, dict) and cp.get("summary"):
                            yield _sse(
                                "checkpoint",
                                {"step": step, "summary": cp.get("summary")},
                            )
                except Exception:
                    pass

                # Graph hygiene: merge duplicates and prune isolated nodes/edges
                try:
                    stats = _graph_hygiene(world_memory)
                    yield _sse("hygiene", stats)
                except Exception:
                    pass

            # Progress update
            consumed_words = min(total_words, (step + 1) * stride_words)
            ratio = 0.0 if total_words == 0 else min(1.0, consumed_words / total_words)
            approx_lines = int(total_lines * ratio)
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

        yield _sse(
            "done",
            {
                "words": total_words,
                "lines": total_lines,
                "steps": total_steps,
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
