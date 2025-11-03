import time
from typing import List, Dict, Any, Tuple

from ..utility.embeddings import dot_sim
from .memory import WorldMemory, _build_memory_text_for_embedding


def _type_bonus(mem_type: str) -> float:
    t = (mem_type or "").lower()
    if t == "threat":
        return 0.06
    if t in ("npc", "relationship"):
        return 0.05
    if t == "goal":
        return 0.04
    if t == "item":
        return 0.02
    return 0.0


def _gather_all_memories(world_memory: WorldMemory) -> List[Dict[str, Any]]:
    """Return a combined list of session memories and all ingest memories.

    Ensures each memory has an embedding vector (computes on demand for ingest entries).
    """
    combined: List[Dict[str, Any]] = []
    # Session memories already carry vectors
    try:
        combined.extend(world_memory.memories)
    except Exception:
        pass
    # Ingest memories: flatten across shards
    try:
        for lst in world_memory.ingest_memories.values():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if not isinstance(m, dict):
                    continue
                # Ensure primary and window vectors present (compute and cache in-memory only)
                try:
                    if "vector" not in m:
                        expl = m.get("explanation")
                        if isinstance(expl, str) and expl.strip():
                            m["vector"] = world_memory.embed_fn(expl)
                        else:
                            combined_text = _build_memory_text_for_embedding(m)
                            if combined_text:
                                m["vector"] = world_memory.embed_fn(combined_text)
                    if "window_vector" not in m:
                        wtxt = m.get("window_text")
                        if isinstance(wtxt, str) and wtxt.strip():
                            m["window_vector"] = world_memory.embed_fn(wtxt)
                except Exception:
                    continue
                combined.append(m)
    except Exception:
        pass
    return combined


def weighted_retrieve(
    world_memory: WorldMemory, query: str, k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve memories with simple weighting (similarity + recency + type bonus).

    Returns: top-k memory dicts sorted by weighted score.
    """
    # Retrieve across session + all ingest memories
    base = _gather_all_memories(world_memory)
    if not base:
        return []
    if not base:
        return []

    now = time.time()
    qvec = world_memory.embed_fn(query)

    weighted: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
    for m in base:
        base_score = dot_sim(qvec, m["vector"]) if "vector" in m else 0.0
        win_score = (
            dot_sim(qvec, m["window_vector"]) if "window_vector" in m else float("-inf")
        )
        score = max(base_score, win_score)  # similarity across explanation/window
        age_sec = max(0.0, now - float(m.get("timestamp", now)))
        recency = pow(0.5, age_sec / 600.0) * 0.05  # half-life ~10 min, max +0.05
        bonus = _type_bonus(str(m.get("type", "")))
        total = score + recency + bonus
        weighted.append((total, score, recency, bonus, m))

    weighted.sort(key=lambda x: x[0], reverse=True)
    top = weighted[:k]
    return [m for (_, _, _, _, m) in top]


def weighted_retrieve_with_scores(
    world_memory: WorldMemory,
    query: str,
    k: int = 5,
    min_total_score: float | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve memories with scores and optional thresholding.

    Returns list of dicts: {"summary", "type", "entities", "_raw", "total", "similarity", "recency", "bonus"}
    Only the top-k by total are returned, filtered by min_total_score if provided.
    Ensures at least one item is returned if any base memories exist.
    """
    base = _gather_all_memories(world_memory)
    if not base:
        return []

    now = time.time()
    qvec = world_memory.embed_fn(query)

    weighted: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
    for m in base:
        base_score = dot_sim(qvec, m["vector"]) if "vector" in m else 0.0
        win_score = (
            dot_sim(qvec, m["window_vector"]) if "window_vector" in m else float("-inf")
        )
        score = max(base_score, win_score)  # similarity across explanation/window
        age_sec = max(0.0, now - float(m.get("timestamp", now)))
        recency = pow(0.5, age_sec / 600.0) * 0.05  # half-life ~10 min, max +0.05
        bonus = _type_bonus(str(m.get("type", "")))
        total = score + recency + bonus
        weighted.append((total, score, recency, bonus, m))

    weighted.sort(key=lambda x: x[0], reverse=True)
    # Apply threshold if provided, but keep the single best if everything filters out
    if min_total_score is not None:
        filtered = [row for row in weighted if row[0] >= min_total_score]
        weighted = filtered if filtered else weighted[:1]

    top = weighted[:k]
    results: List[Dict[str, Any]] = []
    for total, score, recency, bonus, m in top:
        results.append(
            {
                "summary": m.get("summary", ""),
                "type": m.get("type", "unknown"),
                "entities": m.get("entities", []),
                "_raw": m,
                "total": float(total),
                "similarity": float(score),
                "recency": float(recency),
                "bonus": float(bonus),
            }
        )
    return results


def multi_index_retrieve_with_scores(
    world_memory: WorldMemory,
    query: str,
    k_general: int = 3,
    k_per_entity: int = 2,
    k_per_type: int = 1,
    min_total_score: float | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve memories using multiple indices for diverse context.

    Retrieves memories from:
    1. General similarity search (top k_general)
    2. Entity-based: for each entity in top results, get k_per_entity memories
    3. Type-based: for important types (threat, npc, goal), get k_per_type memories each

    Returns deduplicated list with scores.
    """
    base = _gather_all_memories(world_memory)
    if not base:
        return []

    now = time.time()
    qvec = world_memory.embed_fn(query)

    # Compute all scores once
    all_scored: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
    for m in base:
        base_score = dot_sim(qvec, m["vector"]) if "vector" in m else 0.0
        win_score = (
            dot_sim(qvec, m["window_vector"]) if "window_vector" in m else float("-inf")
        )
        score = max(base_score, win_score)
        age_sec = max(0.0, now - float(m.get("timestamp", now)))
        recency = pow(0.5, age_sec / 600.0) * 0.05
        bonus = _type_bonus(str(m.get("type", "")))
        total = score + recency + bonus
        all_scored.append((total, score, recency, bonus, m))

    all_scored.sort(key=lambda x: x[0], reverse=True)

    if not all_scored:
        return []

    # Apply threshold, but ensure we always have enough results
    # Similar to old weighted_retrieve_with_scores: if threshold filters everything, keep top results anyway
    thresholded = all_scored
    if min_total_score is not None:
        filtered = [row for row in all_scored if row[0] >= min_total_score]
        if filtered:
            thresholded = filtered
        # If threshold filters everything, keep top k_general anyway (same as old behavior)
        elif all_scored:
            thresholded = all_scored[
                : max(k_general, 10)
            ]  # Keep enough for entity/type extraction

    # 1. General similarity (top k_general from thresholded, or all if fewer)
    selected_ids: set = set()
    results: List[Dict[str, Any]] = []

    for total, score, recency, bonus, m in thresholded[:k_general]:
        # Build a resilient dedupe key: prefer id; fallback to summary+timestamp
        mid = m.get("id")
        fallback_key = f"{m.get('summary','')}|{m.get('timestamp','')}"
        key = mid if isinstance(mid, str) and mid else fallback_key
        if key not in selected_ids:
            selected_ids.add(key)
            results.append(
                {
                    "summary": m.get("summary", ""),
                    "type": m.get("type", "unknown"),
                    "entities": m.get("entities", []),
                    "_raw": m,
                    "total": float(total),
                    "similarity": float(score),
                    "recency": float(recency),
                    "bonus": float(bonus),
                }
            )

    # 2. Extract entities from top results and retrieve more by entity
    # Search in larger pool (top 100) for entity matches, with relaxed threshold
    entity_search_pool = all_scored[:100] if len(all_scored) > 100 else all_scored
    # Use a more relaxed threshold for entity/type matches (50% of original threshold or 0.1, whichever is lower)
    entity_min_score = (
        min(min_total_score * 0.5, 0.1) if min_total_score is not None else 0.0
    )

    entity_counts: Dict[str, int] = {}
    # Look at more results for entity extraction to get better entity diversity
    for _, _, _, _, m in thresholded[: min(k_general * 3, len(thresholded))]:
        entities = m.get("entities", [])
        if isinstance(entities, list):
            for e in entities:
                if isinstance(e, str) and e.strip():
                    entity_counts[e.strip()] = entity_counts.get(e.strip(), 0) + 1

    # For top entities, retrieve memories containing them from larger pool
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[
        :3
    ]  # Top 3 entities
    for entity, _ in top_entities:
        entity_lower = entity.lower()
        count = 0
        for total, score, recency, bonus, m in entity_search_pool:
            if count >= k_per_entity:
                break
            # Still respect minimum score for entity matches
            if total < entity_min_score:
                continue
            mid = m.get("id")
            fallback_key = f"{m.get('summary','')}|{m.get('timestamp','')}"
            key = mid if isinstance(mid, str) and mid else fallback_key
            if key not in selected_ids:
                entities = m.get("entities", [])
                if isinstance(entities, list):
                    mem_entities_lower = [str(e).lower() for e in entities if e]
                    if entity_lower in mem_entities_lower:
                        selected_ids.add(key)
                        results.append(
                            {
                                "summary": m.get("summary", ""),
                                "type": m.get("type", "unknown"),
                                "entities": m.get("entities", []),
                                "_raw": m,
                                "total": float(total),
                                "similarity": float(score),
                                "recency": float(recency),
                                "bonus": float(bonus),
                            }
                        )
                        count += 1

    # 3. Type-based: get k_per_type from important types (also search in larger pool)
    important_types = ["threat", "npc", "goal", "location"]
    for mem_type in important_types:
        count = 0
        for total, score, recency, bonus, m in entity_search_pool:
            if count >= k_per_type:
                break
            # Respect minimum score for type matches
            if total < entity_min_score:
                continue
            mid = m.get("id")
            fallback_key = f"{m.get('summary','')}|{m.get('timestamp','')}"
            key = mid if isinstance(mid, str) and mid else fallback_key
            if key not in selected_ids:
                if str(m.get("type", "")).lower() == mem_type:
                    selected_ids.add(key)
                    results.append(
                        {
                            "summary": m.get("summary", ""),
                            "type": m.get("type", "unknown"),
                            "entities": m.get("entities", []),
                            "_raw": m,
                            "total": float(total),
                            "similarity": float(score),
                            "recency": float(recency),
                            "bonus": float(bonus),
                        }
                    )
                    count += 1

    # Sort final results by total score
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def format_world_facts(
    memories: List[Dict[str, Any]] | None, char_cap: int = 800
) -> str:
    """Format a compact world facts string for prompt injection."""
    if not memories:
        return ""
    lines: List[str] = ["World Facts (use to stay consistent; do not contradict):"]
    for m in memories:
        ents = ", ".join(map(str, m.get("entities", [])))
        line = f"- [{m.get('type','unknown')}] {m.get('summary','').strip()}"
        if ents:
            line += f" (entities: {ents})"
        lines.append(line)

    out = "\n".join(lines)
    if len(out) > char_cap:
        keep = [lines[0]]
        for line in lines[1:]:
            if len("\n".join(keep + [line])) > char_cap:
                break
            keep.append(line)
        out = "\n".join(keep)
    return out


def format_npc_cards(
    npc_snaps: List[Dict[str, Any]] | None, max_cards: int = 2, char_cap: int = 350
) -> str:
    if not npc_snaps:
        return ""
    cards: List[str] = ["NPC Cards:"]
    for snap in npc_snaps[:max_cards]:
        name = snap.get("name", "Unknown")
        rel = snap.get("relationship_to_player", "unknown")
        loc = snap.get("last_seen_location") or "unknown"
        intent = snap.get("intent") or "unknown"
        line = f"- {name}: rel={rel}; last_seen={loc}; intent={intent}"
        cards.append(line[:char_cap])
    return "\n".join(cards)


def format_location_context(world_memory: WorldMemory, char_cap: int = 600) -> str:
    """Format player's current location and available exits.

    Keeps output compact and capped for prompt budget.
    """
    loc = world_memory.location_graph.get_current_location()
    if not loc:
        return ""

    lines: List[str] = ["Location Context:", f"You are at: {loc.name}"]
    desc = (loc.description or "").strip()
    if desc:
        lines.append(f"Description: {desc}")

    if loc.connections:
        lines.append("Exits:")
        for edge in loc.connections:
            lines.append(f"- {edge.description} (leads to {edge.to_location})")

    if loc.npcs_present:
        lines.append("People here: " + ", ".join(loc.npcs_present))

    out = "\n".join(lines)
    if len(out) > char_cap:
        return out[:char_cap] + "..."
    return out


def summarize_memory_context(memory: Dict[str, Any], max_len: int = 160) -> str | None:
    """Return a concise one-line explanation of what happened when the memory was saved.

    Uses the optional "source_context" captured at creation time in the form:
    "Player said: ...\n\nDM responded: ...". Falls back to None if unavailable.
    """
    ctx = (memory or {}).get("source_context")
    if not isinstance(ctx, str) or not ctx.strip():
        return None
    # Extract the player and DM lines if present
    player_part = None
    dm_part = None
    for line in ctx.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("player said:") and player_part is None:
            player_part = line[len("player said:") :].strip()
        elif line.lower().startswith("dm responded:") and dm_part is None:
            dm_part = line[len("dm responded:") :].strip()
    if player_part or dm_part:
        parts: list[str] = []
        if player_part:
            parts.append(f"Player: {player_part}")
        if dm_part:
            parts.append(f"DM: {dm_part}")
        text = "; ".join(parts)
    else:
        text = ctx.strip()
    # Compact to a single line and trim
    one_line = " ".join(text.split())
    return (one_line[: max_len - 1] + "…") if len(one_line) > max_len else one_line
