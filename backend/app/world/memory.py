import time
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from glob import glob
import uuid
from threading import RLock

from ..utility.embeddings import dot_sim
from ..settings import RELATIONSHIP_ORDER, MEMORY_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def _build_memory_text_for_embedding(memory_dict: Dict[str, Any]) -> str:
    """Combine memory fields (except timestamp) into a single text string for embedding.

    Includes: type, summary, entities, and source_context.
    NPC data is handled separately via NPC index, so not included here.
    """
    parts: List[str] = []

    # Type as prefix for semantic context
    mem_type = memory_dict.get("type", "")
    if mem_type:
        parts.append(f"[{mem_type}]")

    # Summary (main content)
    summary = memory_dict.get("summary", "")
    if summary:
        parts.append(summary)

    # Entities as comma-separated list
    entities = memory_dict.get("entities", [])
    if entities and isinstance(entities, list):
        entities_str = ", ".join(str(e) for e in entities if e)
        if entities_str:
            parts.append(f"Entities: {entities_str}")

    # Source context (window snippet from ingestion or conversation context)
    source_context = memory_dict.get("source_context")
    if source_context and isinstance(source_context, str) and source_context.strip():
        parts.append(f"Context: {source_context.strip()}")

    return " ".join(parts)


class WorldMemory:
    def __init__(self, embed_fn):
        self._lock = RLock()
        self._embed = embed_fn
        self._init_state()
        self.memories: List[Dict[str, Any]] = []
        self.embed_fn = embed_fn
        # Lightweight NPC index mapping canonical_name -> snapshot dict
        self.npc_index: Dict[str, Dict[str, Any]] = {}
        # Simple in-process location graph
        self.location_graph = WorldGraph()
        # Ingest layer (persistent shards)
        self.ingest_subgraphs: Dict[str, Dict[str, "LocationNode"]] = {}
        self.ingest_memories: Dict[str, List[Dict[str, Any]]] = {}
        self.ingest_names: Dict[str, str] = {}
        self.ingest_npc_index: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ---------------- State initialization ----------------
    def _init_state(self):
        self.memories = []
        self.npc_index = {}
        self.location_graph = WorldGraph()
        self.ingest_subgraphs = {}
        self.ingest_memories = {}

    def reset(self):
        with self._lock:
            self._init_state()

    def state_summary(self) -> Dict[str, Any]:
        return {
            "memories": len(self.memories),
            "npc_index": len(self.npc_index),
            "location_graph": len(self.location_graph.locations),
            "ingest_subgraphs": len(self.ingest_subgraphs),
            "ingest_memories": len(self.ingest_memories),
            "ingest_names": len(self.ingest_names),
        }

    # ---------------- Ingest Layer helpers ----------------
    def ensure_ingest_shard(self, ingest_id: str) -> None:
        if ingest_id not in self.ingest_subgraphs:
            self.ingest_subgraphs[ingest_id] = {}
        if ingest_id not in self.ingest_memories:
            self.ingest_memories[ingest_id] = []
        if ingest_id not in self.ingest_npc_index:
            self.ingest_npc_index[ingest_id] = {}

    def add_ingest_memory(self, ingest_id: str, memory_dict: Dict[str, Any]) -> None:
        self.ensure_ingest_shard(ingest_id)
        self.ingest_memories[ingest_id].append(memory_dict)

    def upsert_ingest_location(self, ingest_id: str, node: "LocationNode") -> None:
        self.ensure_ingest_shard(ingest_id)
        self.ingest_subgraphs[ingest_id][node.name] = node

    def set_ingest_name(self, ingest_id: str, name: str) -> None:
        if not isinstance(name, str):
            return
        cleaned = " ".join(name.strip().split())
        if cleaned:
            self.ingest_names[ingest_id] = cleaned[:120]

    def persist_ingest_shard(
        self, ingest_id: str, base_dir: Optional[str] = None
    ) -> None:
        """Write a single ingest shard to disk as JSON. Best-effort; silent on error."""
        try:
            base = base_dir or os.getenv("INGESTS_DIR", "./data/ingests")
            os.makedirs(base, exist_ok=True)
            subgraph = self.ingest_subgraphs.get(ingest_id, {})
            # Strip vectors before persisting; embeddings are recomputed on load
            raw_memories = self.ingest_memories.get(ingest_id, [])
            memories = []
            for m in raw_memories:
                if isinstance(m, dict):
                    try:
                        cleaned = {
                            k: v
                            for k, v in m.items()
                            if k not in ("vector", "window_vector")
                        }
                        memories.append(cleaned)
                    except Exception:
                        memories.append(m)
                else:
                    memories.append(m)
            payload = {
                "name": self.ingest_names.get(ingest_id),
                "subgraph": {name: node.to_dict() for name, node in subgraph.items()},
                "memories": memories,
                "npc_index": self.ingest_npc_index.get(ingest_id, {}),
            }
            path = os.path.join(base, f"{ingest_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            # Fail-closed: persistence must not crash the server
            logger.warning(f"Failed to persist ingest shard {ingest_id}: {e}")
            return

    def load_ingest_shards(self, base_dir: Optional[str] = None) -> None:
        """Load all shards from disk into ingest layer. Best-effort; silent on error."""
        try:
            base = base_dir or os.getenv("INGESTS_DIR", "./data/ingests")
            os.makedirs(base, exist_ok=True)
            for fp in glob(os.path.join(base, "*.json")):
                try:
                    ingest_id = os.path.splitext(os.path.basename(fp))[0]
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    subgraph_raw = data.get("subgraph", {}) or {}
                    memories_raw = data.get("memories", []) or []
                    name_raw = data.get("name")
                    npc_index_raw = data.get("npc_index", {}) or {}
                    shard_graph: Dict[str, LocationNode] = {}
                    for name, nd in subgraph_raw.items():
                        try:
                            shard_graph[name] = LocationNode.from_dict(nd)
                        except Exception:
                            continue
                    self.ingest_subgraphs[ingest_id] = shard_graph
                    self.ingest_memories[ingest_id] = [
                        m for m in memories_raw if isinstance(m, dict)
                    ]
                    # Load per-shard NPC index
                    try:
                        idx: Dict[str, Dict[str, Any]] = {}
                        if isinstance(npc_index_raw, dict):
                            for k, v in npc_index_raw.items():
                                if isinstance(k, str) and isinstance(v, dict):
                                    idx[k] = v
                        self.ingest_npc_index[ingest_id] = idx
                    except Exception:
                        self.ingest_npc_index[ingest_id] = {}
                    # Recompute embeddings for all memories upon load (do not trust persisted vectors)
                    try:
                        embed = self.embed_fn
                        for m in self.ingest_memories.get(ingest_id, []):
                            try:
                                if isinstance(m, dict):
                                    # Ensure each ingest memory has a stable id
                                    try:
                                        mid = m.get("id")
                                        if not isinstance(mid, str) or not mid.strip():
                                            m["id"] = str(uuid.uuid4())
                                    except Exception:
                                        m["id"] = str(uuid.uuid4())
                                    # Recompute primary explanation vector from explicit explanation when available
                                    expl = m.get("explanation")
                                    if isinstance(expl, str) and expl.strip():
                                        m["vector"] = embed(expl)
                                    else:
                                        combined_text = (
                                            _build_memory_text_for_embedding(m)
                                        )
                                        if combined_text:
                                            m["vector"] = embed(combined_text)
                                    # Recompute window vector if window_text present
                                    wtxt = m.get("window_text")
                                    if isinstance(wtxt, str) and wtxt.strip():
                                        m["window_vector"] = embed(wtxt)
                                        # Align recency with vector compute time on load
                                        m["timestamp"] = time.time()
                            except Exception:
                                continue
                    except Exception:
                        pass
                    if isinstance(name_raw, str) and name_raw.strip():
                        self.ingest_names[ingest_id] = name_raw.strip()[:120]
                except Exception:
                    # keep loading others
                    continue
        except Exception as e:
            logger.warning(f"Failed to load ingest shards: {e}")
            return

    # ---------------- Ingest NPC index helpers ----------------
    def add_ingest_npc_update(
        self, ingest_id: str, npc: Dict[str, Any], source_entry: Dict[str, Any]
    ) -> None:
        """Upsert an NPC snapshot in the per-shard npc index for persistence during ingest."""
        try:
            self.ensure_ingest_shard(ingest_id)
            name = str(npc.get("name", "")).strip()
            if not name:
                return
            cid = self._canonicalize_name(name)
            now = time.time()
            idx = self.ingest_npc_index.get(ingest_id, {})
            snapshot = idx.get(
                cid,
                {
                    "name": name,
                    "aliases": [],
                    "last_seen_location": None,
                    "last_seen_time": 0.0,
                    "intent": None,
                    "relationship_to_player": "unknown",
                    "history": [],
                    "confidence": 0.0,
                },
            )
            aliases = npc.get("aliases", []) or []
            if isinstance(aliases, list):
                existing = {
                    self._canonicalize_name(a): a for a in snapshot.get("aliases", [])
                }
                for a in aliases:
                    if isinstance(a, str):
                        key = self._canonicalize_name(a)
                        if key not in existing and key != cid:
                            snapshot["aliases"].append(a)
            loc = npc.get("last_seen_location")
            if isinstance(loc, str) and loc.strip():
                snapshot["last_seen_location"] = loc.strip()
                snapshot["last_seen_time"] = now
            intent = npc.get("intent")
            if isinstance(intent, str) and intent.strip():
                snapshot["intent"] = intent.strip()
            rel = str(npc.get("relationship_to_player", "")).lower().strip()
            if rel in RELATIONSHIP_ORDER:
                current = str(snapshot.get("relationship_to_player", "unknown")).lower()
                if RELATIONSHIP_ORDER.get(rel, 0) >= RELATIONSHIP_ORDER.get(current, 0):
                    snapshot["relationship_to_player"] = rel
            conf = npc.get("confidence")
            try:
                cval = float(conf) if conf is not None else 0.0
            except Exception:
                cval = 0.0
            try:
                prev = snapshot.get("confidence", 0.0)
                prev_f = float(prev) if prev is not None else 0.0
                snapshot["confidence"] = max(prev_f, cval)
            except Exception:
                snapshot["confidence"] = cval
            history_line = source_entry.get("summary")
            if isinstance(history_line, str) and history_line:
                hist = snapshot.get("history", [])
                hist.append(history_line[:160])
                snapshot["history"] = hist[-10:]
            idx[cid] = snapshot
            self.ingest_npc_index[ingest_id] = idx
        except Exception:
            return

    def add_memory(
        self,
        summary: str,
        entities: List[str],
        mem_type: str,
        npc: Dict[str, Any] | None = None,
        dedupe_check: bool = False,
        similarity_threshold: float = MEMORY_SIMILARITY_THRESHOLD,
        source_context: Optional[str] = None,
    ) -> str:
        """Store a durable world fact."""
        with self._lock:
            # Build entry dict for embedding helper
            entry_dict = {
                "summary": summary,
                "entities": entities,
                "type": mem_type,
                "source_context": source_context,
            }

            if dedupe_check and self.memories:
                combined_text = _build_memory_text_for_embedding(entry_dict)
                vec = self.embed_fn(combined_text)
                recent_memories = self.memories[-10:]
                for memory in recent_memories:
                    similarity = dot_sim(vec, memory["vector"])
                    if similarity >= similarity_threshold:
                        return memory["id"]

            memory_id = str(uuid.uuid4())
            combined_text = _build_memory_text_for_embedding(entry_dict)
            vec = self.embed_fn(combined_text)

            entry = {
                "id": memory_id,
                "summary": summary,
                "entities": entities,
                "type": mem_type,
                "timestamp": time.time(),
                "vector": vec,
                # Optional short provenance of what happened when this was saved
                "source_context": source_context,
            }

            self.memories.append(entry)
            # If this is an NPC memory with structured data, upsert the NPC snapshot
            npc_payload = npc
            if mem_type == "npc" and isinstance(npc_payload, dict):
                self._upsert_npc_from_payload(npc_payload, entry)
            return memory_id

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top-k relevant memories by cosine similarity.
        """
        qvec = self.embed_fn(query)

        # qvec and m["vector"] are both normalized  dot product
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for m in self.memories:
            score = dot_sim(qvec, m["vector"])
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for (score, m) in scored[:k]]

    # ---------- NPC support ----------
    def _canonicalize_name(self, name: str) -> str:
        return " ".join(name.strip().lower().split())

    def _upsert_npc_from_payload(
        self, npc: Dict[str, Any], source_entry: Dict[str, Any]
    ):
        with self._lock:
            name = str(npc.get("name", "")).strip()
            if not name:
                return
            cid = self._canonicalize_name(name)

            now = time.time()
            snapshot = self.npc_index.get(
                cid,
                {
                    "name": name,
                    "aliases": [],
                    "last_seen_location": None,
                    "last_seen_time": 0.0,
                    "intent": None,
                    "relationship_to_player": "unknown",
                    "history": [],
                    "confidence": 0.0,
                },
            )

            # merge aliases
            aliases = npc.get("aliases", []) or []
            if isinstance(aliases, list):
                existing = {
                    self._canonicalize_name(a): a for a in snapshot.get("aliases", [])
                }
                for a in aliases:
                    if isinstance(a, str):
                        key = self._canonicalize_name(a)
                        if key not in existing and key != cid:
                            snapshot["aliases"].append(a)

            # update last seen
            loc = npc.get("last_seen_location")
            if isinstance(loc, str) and loc.strip():
                snapshot["last_seen_location"] = loc.strip()
                snapshot["last_seen_time"] = now
                # Opportunistically reflect presence in live location graph for fast UI
                try:
                    loc_name = loc.strip()
                    node = self.location_graph.locations.get(loc_name)
                    if node is not None:
                        cname = cid  # canonicalized NPC name
                        if cname not in node.npcs_present:
                            node.npcs_present.append(cname)
                except Exception as e:
                    # Never let graph hygiene break memory upserts
                    logger.debug(f"Failed to update location graph for NPC {cid}: {e}")

            # update intent (replace if provided and non-empty)
            intent = npc.get("intent")
            if isinstance(intent, str) and intent.strip():
                snapshot["intent"] = intent.strip()

            # relationship precedence: hostile > friendly > neutral > unknown
            rel = str(npc.get("relationship_to_player", "")).lower().strip()
            if rel in RELATIONSHIP_ORDER:
                current = str(snapshot.get("relationship_to_player", "unknown")).lower()
                if RELATIONSHIP_ORDER.get(rel, 0) >= RELATIONSHIP_ORDER.get(current, 0):
                    snapshot["relationship_to_player"] = rel

            # confidence (max)
            conf = npc.get("confidence")
            try:
                cval = float(conf) if conf is not None else 0.0
            except Exception:
                cval = 0.0
            try:
                prev = snapshot.get("confidence", 0.0)
                prev_f = float(prev) if prev is not None else 0.0
                snapshot["confidence"] = max(prev_f, cval)
            except Exception:
                snapshot["confidence"] = cval

            # append concise history line
            history_line = source_entry.get("summary")
            if isinstance(history_line, str) and history_line:
                hist = snapshot.get("history", [])
                hist.append(history_line[:160])
                snapshot["history"] = hist[-10:]  # cap length

            self.npc_index[cid] = snapshot

    def get_relevant_npc_snapshots(
        self, query: str, k: int = 2
    ) -> List[Dict[str, Any]]:
        """Return up to k NPC snapshots relevant to the query by name/alias similarity."""
        if not self.npc_index:
            return []
        qvec = self.embed_fn(query)

        # Combine session NPCs with ingest NPCs (session entries take precedence)
        combined: Dict[str, Dict[str, Any]] = {}
        for key, v in self.npc_index.items():
            combined[key] = v
        for idx in self.ingest_npc_index.values():
            for key, v in idx.items():
                if key not in combined:
                    combined[key] = v

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for snap in combined.values():
            # build a small text rep for similarity: name + aliases + intent + location
            parts = [snap.get("name", "")]
            parts.extend(snap.get("aliases", []) or [])
            parts.append(snap.get("intent", "") or "")
            # include both raw and canonicalized last_seen_location for better recall
            raw_loc = snap.get("last_seen_location", "") or ""
            if raw_loc:
                parts.append(raw_loc)
                canon_loc = " ".join(str(raw_loc).strip().lower().split())
                if canon_loc and canon_loc != raw_loc:
                    parts.append(canon_loc)
            text = " | ".join([p for p in parts if p])
            svec = self.embed_fn(text) if text else qvec
            score = dot_sim(qvec, svec)
            # slight boost for recency
            age_sec = max(0.0, time.time() - float(snap.get("last_seen_time", 0.0)))
            recency = pow(0.5, age_sec / 600.0) * 0.05
            scored.append((score + recency, snap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [snap for (_, snap) in scored[:k]]

    def get_relevant_npc_snapshots_scored(
        self, query: str, k: int = 2, min_score: float | None = None
    ) -> List[Dict[str, Any]]:
        """Like get_relevant_npc_snapshots but returns score alongside snapshot and supports thresholding.

        Returns list of dicts: {"name", "relationship_to_player", "last_seen_location", "intent", "score", "_raw"}
        Ensures at least one item if any NPCs exist and threshold filters everything out.
        """
        if not self.npc_index:
            return []
        qvec = self.embed_fn(query)

        # Combine session NPCs with ingest NPCs (session entries take precedence)
        combined: Dict[str, Dict[str, Any]] = {}
        for key, v in self.npc_index.items():
            combined[key] = v
        for idx in self.ingest_npc_index.values():
            for key, v in idx.items():
                if key not in combined:
                    combined[key] = v

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for snap in combined.values():
            parts = [snap.get("name", "")]
            parts.extend(snap.get("aliases", []) or [])
            parts.append(snap.get("intent", "") or "")
            raw_loc = snap.get("last_seen_location", "") or ""
            if raw_loc:
                parts.append(raw_loc)
                canon_loc = " ".join(str(raw_loc).strip().lower().split())
                if canon_loc and canon_loc != raw_loc:
                    parts.append(canon_loc)
            text = " | ".join([p for p in parts if p])
            svec = self.embed_fn(text) if text else qvec
            score = dot_sim(qvec, svec)
            age_sec = max(0.0, time.time() - float(snap.get("last_seen_time", 0.0)))
            recency = pow(0.5, age_sec / 600.0) * 0.05
            scored.append((score + recency, snap))

        scored.sort(key=lambda x: x[0], reverse=True)
        if min_score is not None:
            filtered = [row for row in scored if row[0] >= min_score]
            scored = filtered if filtered else scored[:1]

        top = scored[:k]
        results: List[Dict[str, Any]] = []
        for s, snap in top:
            results.append(
                {
                    "name": snap.get("name", "Unknown"),
                    "relationship_to_player": snap.get(
                        "relationship_to_player", "unknown"
                    ),
                    "last_seen_location": snap.get("last_seen_location"),
                    "intent": snap.get("intent"),
                    "score": float(s),
                    "_raw": snap,
                }
            )
        return results


# ---------------- Location Graph ----------------


class LocationEdge:
    def __init__(self, to_location: str, description: str, travel_verb: str = "go"):
        self.to_location = to_location
        self.description = description
        self.travel_verb = travel_verb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "to": self.to_location,
            "description": self.description,
            "travel_verb": self.travel_verb,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LocationEdge":
        return LocationEdge(
            to_location=str(d.get("to", "")),
            description=str(d.get("description", "")),
            travel_verb=str(d.get("travel_verb", "go")),
        )


class LocationNode:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.aliases: List[str] = []
        self.connections: List[LocationEdge] = []
        self.npcs_present: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "connections": [e.to_dict() for e in self.connections],
            "npcs_present": list(self.npcs_present),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LocationNode":
        node = LocationNode(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
        )
        conns = d.get("connections", []) or []
        for e in conns:
            try:
                node.connections.append(LocationEdge.from_dict(e))
            except Exception:
                continue
        npcs = d.get("npcs_present", []) or []
        if isinstance(npcs, list):
            node.npcs_present = [str(x) for x in npcs if isinstance(x, str)]
        aliases_raw = d.get("aliases", []) or []
        if isinstance(aliases_raw, list):
            node.aliases = [str(x) for x in aliases_raw if isinstance(x, str)]
        return node


class WorldGraph:
    def __init__(self):
        self.locations: Dict[str, LocationNode] = {}
        self.player_location: Optional[str] = None
        self._lock = RLock()

    def add_location(self, node: LocationNode) -> None:
        with self._lock:
            self.locations[node.name] = node

    def add_connection(
        self,
        from_name: str,
        to_name: str,
        edge_description: str,
        travel_verb: str = "go",
    ) -> None:
        with self._lock:
            if from_name in self.locations and to_name in self.locations:
                self.locations[from_name].connections.append(
                    LocationEdge(
                        to_location=to_name,
                        description=edge_description,
                        travel_verb=travel_verb,
                    )
                )

    def get_current_location(self) -> Optional[LocationNode]:
        with self._lock:
            if not self.player_location:
                return None
            return self.locations.get(self.player_location)

    def move_player(self, new_location_name: str) -> bool:
        with self._lock:
            if new_location_name in self.locations:
                self.player_location = new_location_name
                return True
            return False
