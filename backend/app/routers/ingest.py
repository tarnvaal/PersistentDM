from __future__ import annotations

import json
import uuid
import os
import time
import re
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..dependencies import get_conversation_service
from ..world.memory_utils import sanitize_entities
from ..world.memory import LocationNode, _build_memory_text_for_embedding
from ..world.context_builder import summarize_memory_context
from ..utility.embeddings import dot_sim
from ..settings import MAX_CHUNK_SIZE


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
        npc_index_raw = data.get("npc_index", {}) or {}
        wm = conversation.world_memory
        shard_graph = {}
        for name, nd in subgraph_raw.items():
            try:
                shard_graph[name] = LocationNode.from_dict(nd)
            except Exception:
                continue
        wm.ingest_subgraphs[ingest_id] = shard_graph
        wm.ingest_memories[ingest_id] = [m for m in memories_raw if isinstance(m, dict)]
        # Load shard-local NPC index
        try:
            idx = {}
            if isinstance(npc_index_raw, dict):
                for k, v in npc_index_raw.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        idx[k] = v
            wm.ingest_npc_index[ingest_id] = idx
        except Exception:
            wm.ingest_npc_index[ingest_id] = {}
        # Eagerly compute embeddings for loaded memories so retrieval has no first-hit cost
        embedding_ms = None
        try:
            start = time.perf_counter()
            embed = wm.embed_fn
            for m in wm.ingest_memories.get(ingest_id, []):
                try:
                    if isinstance(m, dict):
                        # Primary vector: use explicit explanation if present, else combined
                        expl = m.get("explanation")
                        if isinstance(expl, str) and expl.strip():
                            m["vector"] = embed(expl)
                        else:
                            combined_text = _build_memory_text_for_embedding(m)
                            if combined_text:
                                m["vector"] = embed(combined_text)
                        # Window vector: use explicit window_text if present
                        wtxt = m.get("window_text")
                        if isinstance(wtxt, str) and wtxt.strip():
                            m["window_vector"] = embed(wtxt)
                        # Set recency timestamp to the time of vector computation
                        m["timestamp"] = time.time()
                except Exception:
                    # best-effort: skip problematic entries and continue
                    continue
            embedding_ms = int(round((time.perf_counter() - start) * 1000))
        except Exception:
            # if embedding model is unavailable, loading should still succeed
            embedding_ms = None
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
            "embeddingMs": embedding_ms,
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
    # Fixed internal window size in words
    window_tokens = 200
    stride_tokens = 100
    window_words = 134
    if stride_words_override is not None:
        try:
            stride_words = max(1, min(MAX_CHUNK_SIZE, int(stride_words_override)))
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
            # Rolling ingest context carried across windows
            ingest_ctx = {"npcs": []}

            def _make_header(ctx: dict) -> str:
                parts = []
                protagonist = ctx.get("protagonist")
                if isinstance(protagonist, str) and protagonist.strip():
                    parts.append(f"- Protagonist: {protagonist.strip()}")
                goal = ctx.get("goal")
                if isinstance(goal, str) and goal.strip():
                    parts.append(f"- Goal: {goal.strip()}")
                area = ctx.get("current_area")
                if isinstance(area, str) and area.strip():
                    parts.append(f"- Current Area: {area.strip()}")
                npcs = ctx.get("npcs") or []
                if isinstance(npcs, list) and npcs:
                    parts.append(
                        f"- NPCs Mentioned: {', '.join([str(n) for n in npcs[:5]])}"
                    )
                header = "Context so far:\n" + ("\n".join(parts) if parts else "- None")
                # Cap to ~300 chars to avoid prompt bloat
                return header[:300]

            def _update_ctx_from_mem(ctx: dict, mem: dict) -> None:
                try:
                    conf = float(mem.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                # Use slightly stricter threshold for context updates
                if conf < 0.75:
                    return
                mtype = str(mem.get("type", "")).lower()
                entities_local = sanitize_entities(mem.get("entities"))
                if mtype == "location" and entities_local:
                    ctx["current_area"] = entities_local[0]
                if mtype == "goal":
                    summary = mem.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        ctx["goal"] = summary.strip()[:200]
                npc_payload_local = mem.get("npc")
                if isinstance(npc_payload_local, dict):
                    name = str(npc_payload_local.get("name", "")).strip()
                    if name:
                        npcs_list = ctx.get("npcs") or []
                        try:
                            if name in npcs_list:
                                npcs_list.remove(name)
                        except Exception:
                            pass
                        npcs_list.insert(0, name)
                        ctx["npcs"] = npcs_list[:5]

            def _select_relevant_snippet(
                text_block: str, summary: str, max_chars: int = 300
            ) -> str:
                """Find the sentence most relevant to summary and return it with one before/after.

                Finds the sentence with highest similarity to summary using embeddings,
                then returns that sentence plus one sentence before (if available) and
                one sentence after (if available).
                Falls back to leading slice on failure.
                """
                try:
                    text = (text_block or "").strip()
                    if not text:
                        return ""
                    summ = (summary or "").strip()
                    # If no usable summary, fallback to naive slice
                    if not summ or len(summ) < 8:
                        return text[: max_chars - 1] + (
                            "…" if len(text) > max_chars else ""
                        )

                    # Split into sentences, retaining punctuation
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    sentences = [s for s in sentences if s and s.strip()]
                    if not sentences:
                        return text[: max_chars - 1] + (
                            "…" if len(text) > max_chars else ""
                        )

                    embed = world_memory.embed_fn
                    sum_vec = embed(summ)

                    # Find the sentence with highest similarity to summary
                    best_idx = 0
                    best_score = float("-inf")
                    for i, s in enumerate(sentences):
                        try:
                            sent_vec = embed(s)
                            score = dot_sim(sum_vec, sent_vec)
                            if score > best_score:
                                best_score = score
                                best_idx = i
                        except Exception:
                            continue

                    # Ensure the primary target sentence is fully captured if possible
                    target_sentence = sentences[best_idx]
                    if len(target_sentence) >= max_chars:
                        # If the target sentence alone exceeds window, hard cap with ellipsis
                        return target_sentence[: max_chars - 1] + "…"

                    # Compute remaining budget and expand evenly to left and right
                    remaining = max_chars - len(target_sentence)
                    left_budget = remaining // 2
                    right_budget = remaining - left_budget

                    # Join surrounding context into plain text buffers
                    left_context = " ".join(sentences[:best_idx])
                    right_context = " ".join(sentences[best_idx + 1 :])

                    left_piece = left_context[-left_budget:] if left_budget > 0 else ""
                    right_piece = (
                        right_context[:right_budget] if right_budget > 0 else ""
                    )

                    # Normalize spacing at the boundaries without double-counting into budget
                    left_join = (
                        ""
                        if not left_piece
                        else ("" if left_piece.endswith(" ") else " ")
                    )
                    right_join = (
                        ""
                        if not right_piece
                        else ("" if right_piece.startswith(" ") else " ")
                    )

                    snippet = f"{left_piece}{left_join}{target_sentence}{right_join}{right_piece}"

                    # If we exceeded max_chars due to added joins, trim from the outsides but never from the target sentence
                    if len(snippet) > max_chars:
                        # Find target span
                        base_start = len(left_piece) + len(left_join)
                        base_end = base_start + len(target_sentence)

                        # Trim alternately from right and left until within budget
                        excess = len(snippet) - max_chars
                        left_prefix = snippet[:base_start]
                        right_suffix = snippet[base_end:]
                        while excess > 0 and (left_prefix or right_suffix):
                            if len(right_suffix) >= len(left_prefix) and right_suffix:
                                right_suffix = right_suffix[:-1]
                            elif left_prefix:
                                left_prefix = left_prefix[1:]
                            excess -= 1
                        snippet = (
                            f"{left_prefix}{snippet[base_start:base_end]}{right_suffix}"
                        )

                    return snippet
                except Exception:
                    # Robust fallback
                    t = (text_block or "").strip()
                    return t[: max_chars - 1] + ("…" if len(t) > max_chars else "")

            for step in range(total_steps):
                # Check for client disconnection by attempting a small probe yield
                # If client disconnected, GeneratorExit will be raised and caught by outer handler
                start_idx = step * stride_words
                end_idx = min(total_words, start_idx + window_words)
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)
                header_text = _make_header(ingest_ctx)
                chunk_for_model = f"{header_text}\n\nAnalyze this excerpt for new durable facts:\n{chunk_text}"

                # Emit memory (conservative)
                try:
                    # Prefer multi-extractor
                    multi = getattr(chatter, "extract_memories_from_text", None)
                    single = getattr(chatter, "extract_memory_from_text", None)
                    extracted = []
                    if callable(multi):
                        items = multi(chunk_for_model, max_items=5) or []
                        if isinstance(items, list):
                            extracted = [m for m in items if isinstance(m, dict)]
                    if not extracted and callable(single):
                        m = single(chunk_for_model)
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
                            mem_summary = mem.get("summary", "")
                            snippet = _select_relevant_snippet(
                                chunk_text, mem_summary, 300
                            )
                            source_context = f"Ingested: {snippet}" if snippet else None
                            # Full window text for this memory (entire chunk)
                            window_text = chunk_text

                            # Store in ingest layer for this shard (no session mutation)
                            try:
                                # Build explanation text and embed both explanation+window
                                explanation = (
                                    summarize_memory_context(
                                        {"source_context": source_context}
                                    )
                                    if source_context
                                    else None
                                )
                                entry = {
                                    "id": str(uuid.uuid4()),
                                    "summary": mem.get("summary", ""),
                                    "type": mem.get("type", "other"),
                                    "entities": entities,
                                    "npc": npc_payload
                                    if isinstance(npc_payload, dict)
                                    else None,
                                    "confidence": conf,
                                    "timestamp": time.time(),
                                    "source_context": source_context,
                                    "explanation": explanation,
                                    "window_text": window_text,
                                }
                                # Compute dual embeddings: explanation vector (primary) and window vector
                                try:
                                    embed = world_memory.embed_fn
                                    if (
                                        isinstance(explanation, str)
                                        and explanation.strip()
                                    ):
                                        entry["vector"] = embed(explanation)
                                    else:
                                        combined_text = (
                                            _build_memory_text_for_embedding(entry)
                                        )
                                        entry["vector"] = embed(combined_text)
                                    if (
                                        isinstance(window_text, str)
                                        and window_text.strip()
                                    ):
                                        entry["window_vector"] = embed(window_text)
                                except Exception:
                                    pass
                                world_memory.add_ingest_memory(ingest_id, entry)
                                # Also upsert shard-local NPC index for persistence
                                if isinstance(npc_payload, dict):
                                    try:
                                        world_memory.add_ingest_npc_update(
                                            ingest_id, npc_payload, entry
                                        )
                                    except Exception:
                                        pass
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
                                            node = LocationNode(
                                                loc_name, mem.get("summary", "")
                                            )
                                            # Add canonical aliases for better recall
                                            try:
                                                canon = " ".join(
                                                    loc_name.strip().lower().split()
                                                )
                                                art_stripped = canon
                                                if art_stripped.startswith("the "):
                                                    art_stripped = art_stripped[4:]
                                                aliases = []
                                                for a in (canon, art_stripped):
                                                    if (
                                                        a
                                                        and a != loc_name
                                                        and a not in aliases
                                                    ):
                                                        aliases.append(a)
                                                node.aliases = aliases
                                            except Exception:
                                                node.aliases = []
                                            world_memory.upsert_ingest_location(
                                                ingest_id, node
                                            )
                                    except Exception:
                                        pass

                            try:
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
                                        "explanation": entry.get("explanation"),
                                        "windowText": window_text,
                                    },
                                )
                                # Update rolling context after successful save event
                                try:
                                    _update_ctx_from_mem(ingest_ctx, mem)
                                except Exception:
                                    pass
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
                # Consolidate duplicates before persisting
                try:
                    shard_list = world_memory.ingest_memories.get(ingest_id, [])
                    if isinstance(shard_list, list) and len(shard_list) >= 6:
                        seen: Dict[str, dict] = {}
                        for m in shard_list:
                            if not isinstance(m, dict):
                                continue
                            summary_key = " ".join(
                                str(m.get("summary", "")).strip().lower().split()
                            )
                            ents_norm = sanitize_entities(m.get("entities"))
                            ents_key = "|".join(sorted([e.lower() for e in ents_norm]))
                            key = f"{summary_key}##{ents_key}"
                            prev = seen.get(key)
                            try:
                                cur_conf = float(m.get("confidence", 0.0))
                            except Exception:
                                cur_conf = 0.0
                            prev_conf = 0.0
                            if isinstance(prev, dict):
                                try:
                                    prev_conf = float(prev.get("confidence", 0.0))
                                except Exception:
                                    prev_conf = 0.0
                            if prev is None or cur_conf >= prev_conf:
                                seen[key] = m
                        world_memory.ingest_memories[ingest_id] = list(seen.values())
                except Exception:
                    pass

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
