import math
import time
from datetime import datetime
from typing import Dict, Any, Iterable


def _l2_norm(vec: Iterable[float]) -> float:
    total = 0.0
    for v in vec:
        total += v * v
    return math.sqrt(total)


def similarity(query_vector: list[float], item_vector: list[float]) -> float:
    """Calculate cosine similarity between query and item vectors, clipped to [0,1].

    Args:
        query_vector: Normalized query embedding vector
        item_vector: Normalized item embedding vector

    Returns:
        Similarity score in [0,1] range
    """
    if not query_vector or not item_vector:
        return 0.0

    # Cosine similarity with normalization for robustness (tests provide raw vectors)
    qn = _l2_norm(query_vector)
    inorm = _l2_norm(item_vector)
    if qn == 0.0 or inorm == 0.0:
        return 0.0
    sim = sum((x / qn) * (y / inorm) for x, y in zip(query_vector, item_vector))

    # Clip to [0,1] range (cosine can be negative)
    return max(0.0, min(1.0, sim))


def literal_boost(query: str, item_text: str, boost_value: float = 0.2) -> float:
    """Return boost_value if query substring found in item_text (case-insensitive), else 0.

    Args:
        query: Search query string
        item_text: Item text to search in
        boost_value: Boost value to return if match found

    Returns:
        boost_value if substring match, 0 otherwise
    """
    if not query or not item_text:
        return 0.0

    query_lower = query.lower().strip()
    text_lower = item_text.lower()

    return boost_value if query_lower in text_lower else 0.0


def recency_bonus(updated_at: float | datetime, half_life_hours: float = 72.0) -> float:
    """Calculate recency bonus using exponential decay.

    Args:
        updated_at: Timestamp when item was last updated
        half_life_hours: Half-life in hours for the decay

    Returns:
        Recency bonus in [0,1] range
    """
    now = time.time()
    if isinstance(updated_at, datetime):
        updated_ts = updated_at.timestamp()
    else:
        try:
            updated_ts = float(updated_at)
        except Exception:
            updated_ts = 0.0
    age_hours = (now - updated_ts) / 3600.0

    if age_hours <= 0:
        return 1.0  # Future items get max bonus

    # Exponential decay: exp(-age_hours / half_life_hours)
    return math.exp(-age_hours / half_life_hours)


def type_bonus(item_type: str, type_bonus_map: Dict[str, float]) -> float:
    """Lookup type bonus from map, return 0.0 if not found.

    Args:
        item_type: Memory type (e.g., "npc", "location")
        type_bonus_map: Dict mapping types to bonus values

    Returns:
        Type bonus value or 0.0 if not found
    """
    return type_bonus_map.get(item_type, 0.0)


def combine_weights(
    similarity_score: float,
    literal_boost: float,
    recency_bonus: float,
    type_bonus: float,
    weights: Dict[str, float],
) -> float:
    """Combine all scoring components using linear weights.

    Args:
        similarity_score: Semantic similarity [0,1]
        literal_boost: Literal match boost [0, boost_value]
        recency_bonus: Recency bonus [0,1]
        type_bonus: Type bonus [0, max_bonus]
        weights: Dict with keys: w_sim, w_literal, w_rec, w_type

    Returns:
        Combined weighted score
    """
    return (
        weights.get("w_sim", 1.0) * similarity_score
        + weights.get("w_literal", 0.2) * literal_boost
        + weights.get("w_rec", 0.15) * recency_bonus
        + weights.get("w_type", 0.05) * type_bonus
    )


def score_memory_item(
    query: str,
    query_vector: list[float],
    memory_item: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single memory item and return score breakdown.

    Args:
        query: Search query string
        query_vector: Query embedding vector
        memory_item: Memory dict with vector, text, type, updated_at
        config: Scoring configuration

    Returns:
        Dict with total score and component breakdown
    """
    # Extract item data
    item_vector = memory_item.get("vector", [])
    item_text = memory_item.get("text", "")
    item_type = memory_item.get("type", "")
    updated_at = memory_item.get("updated_at", time.time())

    # Calculate components
    sim_score = similarity(query_vector, item_vector)
    lit_boost = literal_boost(query, item_text, config.get("literal_boost_value", 0.2))
    rec_bonus = recency_bonus(updated_at, config.get("half_life_hours", 72.0))
    typ_bonus = type_bonus(item_type, config.get("type_bonus_map", {}))

    # Combine weights
    total_score = combine_weights(
        sim_score, lit_boost, rec_bonus, typ_bonus, config.get("weights", {})
    )

    return {
        "total": total_score,
        "similarity": sim_score,
        "literal_boost": lit_boost,
        "recency_bonus": rec_bonus,
        "type_bonus": typ_bonus,
    }
