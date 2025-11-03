from __future__ import annotations

import os
import json
import shutil
import time
import uuid
from typing import Any, Dict, List, Tuple

from ..dependencies import get_chatter, get_world_memory
from ..utility.message import Message
from ..utility.history import History
from ..world.memory import (
    LocationNode,
    WorldGraph,
    WorldMemory,
    _build_memory_text_for_embedding,
)


def _sessions_dir() -> str:
    base = os.getenv("SESSIONS_DIR", "/home/tarnv/dev/PersistentDM/data/sessions")
    os.makedirs(base, exist_ok=True)
    return base


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), session_id)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_size_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _sizeof_dir(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    return total


def _export_location_graph(graph: WorldGraph) -> Dict[str, Any]:
    try:
        locs = {}
        for name, node in graph.locations.items():
            locs[name] = node.to_dict()
        return {
            "locations": locs,
            "player_location": graph.player_location,
        }
    except Exception:
        return {"locations": {}, "player_location": None}


def _import_location_graph(payload: Dict[str, Any]) -> WorldGraph:
    graph = WorldGraph()
    try:
        raw = payload or {}
        locs = raw.get("locations", {}) or {}
        for _, nd in locs.items():
            try:
                node = LocationNode.from_dict(nd)
                graph.add_location(node)
            except Exception:
                continue
        pl = raw.get("player_location")
        if isinstance(pl, str) and pl.strip():
            graph.player_location = pl
    except Exception:
        pass
    return graph


def _strip_vectors(memory_item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {
            k: v for k, v in memory_item.items() if k not in ("vector", "window_vector")
        }
    except Exception:
        return memory_item


def export_current_state() -> (
    Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]
):
    """Return (world_state, chat_messages, runtime_state).

    - world_state: { memories, npc_index, location_graph }
    - chat_messages: list of {role, content, timestamp, active}
    - runtime_state: reserved for future use
    """
    chatter = get_chatter()
    wm = get_world_memory()

    # World
    memories = []
    try:
        for m in wm.memories:
            if isinstance(m, dict):
                memories.append(_strip_vectors(m))
    except Exception:
        memories = []

    npc_index = {}
    try:
        for k, v in wm.npc_index.items():
            if isinstance(k, str) and isinstance(v, dict):
                npc_index[k] = v
    except Exception:
        npc_index = {}

    location_graph = _export_location_graph(wm.location_graph)

    world_state = {
        "memories": memories,
        "npc_index": npc_index,
        "location_graph": location_graph,
    }

    # Chat history (exclude the boot system message for portability)
    chat_messages: List[Dict[str, Any]] = []
    try:
        for i, msg in enumerate(chatter.history.history):
            if not isinstance(msg, Message):
                continue
            if i == 0 and msg.role == "system":
                continue
            chat_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "active": msg.active,
                    "timestamp": getattr(msg.timestamp, "isoformat", lambda: None)()
                    or None,
                }
            )
    except Exception:
        chat_messages = []

    runtime_state: Dict[str, Any] = {}
    return world_state, chat_messages, runtime_state


def _rebuild_session_embeddings(wm: WorldMemory) -> None:
    try:
        embed = wm.embed_fn
        for m in wm.memories:
            try:
                combined = _build_memory_text_for_embedding(m)
                if combined:
                    m["vector"] = embed(combined)
                wtxt = m.get("window_text")
                if isinstance(wtxt, str) and wtxt.strip():
                    m["window_vector"] = embed(wtxt)
                # Refresh timestamp to keep recency meaningful
                if "timestamp" not in m:
                    m["timestamp"] = time.time()
            except Exception:
                continue
    except Exception:
        return


def import_state(
    world_state: Dict[str, Any],
    chat_messages: List[Dict[str, Any]],
    mode: str = "replace",
) -> Dict[str, Any]:
    """Load the provided state into the live singletons.

    Returns brief summary { worldMemories, npcs, locations, chatMessages }.
    """
    chatter = get_chatter()
    wm = get_world_memory()

    mode = (mode or "replace").strip().lower()

    if mode == "replace":
        # World: replace in-place
        try:
            wm.memories = [
                m for m in (world_state.get("memories") or []) if isinstance(m, dict)
            ]
        except Exception:
            wm.memories = []

        try:
            npc = world_state.get("npc_index") or {}
            wm.npc_index = {
                str(k): v
                for k, v in npc.items()
                if isinstance(k, str) and isinstance(v, dict)
            }
        except Exception:
            wm.npc_index = {}

        try:
            lg = _import_location_graph(world_state.get("location_graph") or {})
            wm.location_graph = lg
        except Exception:
            wm.location_graph = WorldGraph()

        # Rebuild embeddings for session memories
        _rebuild_session_embeddings(wm)

        # Chat: reset to a fresh History (keep system prompt) and add messages
        try:
            chatter.history = History(
                chatter.max_history_tokens,
                chatter.sysprompt_content,
                chatter.sysprompt_role,
                len(chatter.sysprompt_tokens),
            )
        except Exception:
            try:
                chatter.history.history = chatter.history.history[:1]
            except Exception:
                pass

        try:
            for msg in chat_messages:
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", ""))
                if not role or not content:
                    continue
                chatter.history.add_message(
                    role, content, chatter._get_token_count(content)
                )
        except Exception:
            pass

    else:
        # Merge mode: non-destructive merge of world state; append chat, no reset
        # Incoming data
        incoming_memories = []
        try:
            incoming_memories = [
                m for m in (world_state.get("memories") or []) if isinstance(m, dict)
            ]
        except Exception:
            incoming_memories = []

        incoming_npc = {}
        try:
            tmp = world_state.get("npc_index") or {}
            incoming_npc = {
                str(k): v
                for k, v in tmp.items()
                if isinstance(k, str) and isinstance(v, dict)
            }
        except Exception:
            incoming_npc = {}

        try:
            incoming_graph = _import_location_graph(
                world_state.get("location_graph") or {}
            )
        except Exception:
            incoming_graph = WorldGraph()

        # Merge memories with dedupe
        def _mem_key(m: Dict[str, Any]) -> str:
            try:
                mid = m.get("id")
                if isinstance(mid, str) and mid.strip():
                    return f"id:{mid.strip()}"
            except Exception:
                pass
            try:
                combined = _build_memory_text_for_embedding(m)
                return f"txt:{combined.strip().lower()}"
            except Exception:
                return json.dumps(m, sort_keys=True)

        existing_keys = set()
        for em in wm.memories:
            try:
                existing_keys.add(_mem_key(em))
            except Exception:
                continue
        new_added: List[Dict[str, Any]] = []
        for mem in incoming_memories:
            try:
                key = _mem_key(mem)
                if key in existing_keys:
                    continue
                # Ensure timestamp and id
                if not isinstance(mem.get("timestamp"), (int, float)):
                    mem["timestamp"] = time.time()
                if not isinstance(mem.get("id"), str) or not mem.get("id"):
                    mem["id"] = str(uuid.uuid4())
                wm.memories.append(mem)
                new_added.append(mem)
                existing_keys.add(key)
            except Exception:
                continue

        # Embed only newly added memories
        try:
            embed = wm.embed_fn
            for m in new_added:
                try:
                    combined = _build_memory_text_for_embedding(m)
                    if combined:
                        m["vector"] = embed(combined)
                    wtxt = m.get("window_text")
                    if isinstance(wtxt, str) and wtxt.strip():
                        m["window_vector"] = embed(wtxt)
                except Exception:
                    continue
        except Exception:
            pass

        # Merge NPC index (incoming takes precedence for provided fields)
        try:
            for k, v in incoming_npc.items():
                if k not in wm.npc_index:
                    wm.npc_index[k] = v
                else:
                    try:
                        cur = wm.npc_index.get(k) or {}
                        if isinstance(cur, dict) and isinstance(v, dict):
                            cur.update({kk: vv for kk, vv in v.items()})
                            wm.npc_index[k] = cur
                        else:
                            wm.npc_index[k] = v
                    except Exception:
                        wm.npc_index[k] = v
        except Exception:
            pass

        # Merge location graph
        try:
            for name, node in incoming_graph.locations.items():
                if name not in wm.location_graph.locations:
                    wm.location_graph.add_location(node)
                    continue
                cur = wm.location_graph.locations.get(name)
                if not cur:
                    wm.location_graph.add_location(node)
                    continue
                # description: prefer non-empty and longer
                try:
                    desc_cur = str(cur.description or "")
                    desc_new = str(node.description or "")
                    if desc_new and (not desc_cur or len(desc_new) > len(desc_cur)):
                        cur.description = desc_new
                except Exception:
                    pass
                # aliases: union
                try:
                    have = {a for a in (cur.aliases or [])}
                    for a in node.aliases or []:
                        if a not in have:
                            cur.aliases.append(a)
                            have.add(a)
                except Exception:
                    pass
                # connections: union by (to, description, verb)
                try:
                    seen = {
                        (e.to_location, e.description, e.travel_verb)
                        for e in (cur.connections or [])
                    }
                    for e in node.connections or []:
                        sig = (e.to_location, e.description, e.travel_verb)
                        if sig not in seen:
                            cur.connections.append(e)
                            seen.add(sig)
                except Exception:
                    pass
                # npcs_present: union
                try:
                    have = {n for n in (cur.npcs_present or [])}
                    for n in node.npcs_present or []:
                        if n not in have:
                            cur.npcs_present.append(n)
                            have.add(n)
                except Exception:
                    pass
            # player location: set only if not already set
            try:
                if (
                    not wm.location_graph.player_location
                    and incoming_graph.player_location
                ):
                    wm.location_graph.player_location = incoming_graph.player_location
            except Exception:
                pass
        except Exception:
            pass

        # Chat: append without clearing
        try:
            for msg in chat_messages:
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", ""))
                if not role or not content:
                    continue
                chatter.history.add_message(
                    role, content, chatter._get_token_count(content)
                )

        except Exception:
            pass

    return {
        "worldMemories": len(wm.memories),
        "npcs": len(wm.npc_index),
        "locations": len(getattr(wm.location_graph, "locations", {}) or {}),
        "chatMessages": len(chat_messages),
    }


def list_sessions() -> Dict[str, Any]:
    base = _sessions_dir()
    items: List[Dict[str, Any]] = []
    try:
        for entry in os.listdir(base):
            sp = os.path.join(base, entry)
            if not os.path.isdir(sp):
                continue
            meta_path = os.path.join(sp, "metadata.json")
            name = entry
            created = None
            updated = None
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                name = meta.get("name", name)
                created = meta.get("createdAt")
                updated = meta.get("updatedAt")
            except Exception:
                pass
            items.append(
                {
                    "id": entry,
                    "name": name,
                    "createdAt": created,
                    "updatedAt": updated,
                    "bytes": _sizeof_dir(sp),
                }
            )
    except Exception:
        items = []
    return {
        "sessions": sorted(
            items,
            key=lambda x: (
                x.get("updatedAt") or x.get("createdAt") or "",
                x.get("name"),
            ),
        ),
    }


def save_session(
    name: str, notes: str | None = None, overwrite_session_id: str | None = None
) -> Dict[str, Any]:
    sid = overwrite_session_id or str(uuid.uuid4())
    sp = _session_path(sid)
    os.makedirs(sp, exist_ok=True)

    world_state, chat_messages, runtime_state = export_current_state()

    # Write files
    with open(os.path.join(sp, "world.json"), "w", encoding="utf-8") as f:
        json.dump(world_state, f, ensure_ascii=False)
    with open(os.path.join(sp, "chat.jsonl"), "w", encoding="utf-8") as f:
        for msg in chat_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    with open(os.path.join(sp, "runtime.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_state, f, ensure_ascii=False)

    meta_path = os.path.join(sp, "metadata.json")
    meta = {
        "id": sid,
        "name": name or sid,
        "notes": (notes or None),
        "createdAt": _now_iso(),
        "updatedAt": _now_iso(),
        "schema": 1,
    }
    # Preserve createdAt if overwriting
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old.get("createdAt"), str):
                meta["createdAt"] = old["createdAt"]
            if not name and isinstance(old.get("name"), str):
                meta["name"] = old["name"]
            if notes is None and old.get("notes") is not None:
                meta["notes"] = old.get("notes")
    except Exception:
        pass
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    return {
        "id": sid,
        "name": meta["name"],
        "createdAt": meta["createdAt"],
        "updatedAt": meta["updatedAt"],
        "bytes": _sizeof_dir(sp),
    }


def rename_session(
    session_id: str, name: str | None = None, notes: str | None = None
) -> Dict[str, Any]:
    sp = _session_path(session_id)
    meta_path = os.path.join(sp, "metadata.json")
    meta = {"id": session_id}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    if name:
        meta["name"] = name
    if notes is not None:
        meta["notes"] = notes
    meta["updatedAt"] = _now_iso()
    os.makedirs(sp, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return {
        "id": session_id,
        "name": meta.get("name", session_id),
        "updatedAt": meta.get("updatedAt"),
    }


def load_session(session_id: str, mode: str = "replace") -> Dict[str, Any]:
    sp = _session_path(session_id)
    world_state: Dict[str, Any] = {}
    chat_messages: List[Dict[str, Any]] = []
    try:
        with open(os.path.join(sp, "world.json"), "r", encoding="utf-8") as f:
            world_state = json.load(f)
    except Exception:
        world_state = {}
    try:
        chat_fp = os.path.join(sp, "chat.jsonl")
        if os.path.exists(chat_fp):
            with open(chat_fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            chat_messages.append(obj)
                    except Exception:
                        continue
        else:
            # legacy: allow chat.json as array
            with open(os.path.join(sp, "chat.json"), "r", encoding="utf-8") as f:
                arr = json.load(f)
            if isinstance(arr, list):
                chat_messages = [x for x in arr if isinstance(x, dict)]
    except Exception:
        chat_messages = []

    summary = import_state(world_state, chat_messages, mode=mode)

    # Touch updatedAt
    try:
        rename_session(session_id, name=None)
    except Exception:
        pass
    return {"id": session_id, **summary}


def delete_session(session_id: str) -> Dict[str, Any]:
    sp = _session_path(session_id)
    try:
        if os.path.isdir(sp):
            shutil.rmtree(sp)
            return {"ok": True}
    except Exception:
        pass
    return {"ok": False}
