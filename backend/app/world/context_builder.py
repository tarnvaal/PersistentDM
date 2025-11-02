import time
from typing import List, Dict, Any, Tuple

from ..utility.embeddings import dot_sim
from .memory import WorldMemory


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


def weighted_retrieve(
    world_memory: WorldMemory, query: str, k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve memories with simple weighting (similarity + recency + type bonus).

    Returns: top-k memory dicts sorted by weighted score.
    """
    base = world_memory.retrieve(query, k=max(k * 2, 5))
    if not base:
        return []

    now = time.time()
    qvec = world_memory.embed_fn(query)

    weighted: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
    for m in base:
        score = dot_sim(qvec, m["vector"])  # similarity
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
    base = world_memory.retrieve(query, k=max(k * 2, 5))
    if not base:
        return []

    now = time.time()
    qvec = world_memory.embed_fn(query)

    weighted: List[Tuple[float, float, float, float, Dict[str, Any]]] = []
    for m in base:
        score = dot_sim(qvec, m["vector"])  # similarity
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
