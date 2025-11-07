import json
import os
from typing import Dict, Any


class SearchConfig:
    """Configuration for search service loaded from environment variables."""

    def __init__(self):
        # Load environment variables with defaults
        self.mode_default = os.getenv("SEARCH_MODE_DEFAULT", "hybrid")

        # Weights for scoring components
        self.w_sim = float(os.getenv("SEARCH_W_SIM", "1.0"))
        self.w_literal = float(os.getenv("SEARCH_W_LITERAL", "0.2"))
        self.w_rec = float(os.getenv("SEARCH_W_REC", "0.15"))
        self.w_type = float(os.getenv("SEARCH_W_TYPE", "0.05"))

        # Scoring parameters
        self.half_life_hours = float(os.getenv("SEARCH_HALF_LIFE_HOURS", "72"))

        # Index backend (for future ANN implementation)
        self.index_backend = os.getenv("SEARCH_INDEX_BACKEND", "naive")

        # Type bonus map - load from JSON env var or use defaults
        type_bonus_json = os.getenv(
            "SEARCH_TYPE_BONUS", '{"npc": 0.02, "location": 0.01}'
        )
        try:
            self.type_bonus_map = json.loads(type_bonus_json)
        except (json.JSONDecodeError, TypeError):
            # Fallback to empty map if JSON parsing fails
            self.type_bonus_map = {}

        # Literal boost value
        self.literal_boost_value = self.w_literal  # Use the weight as the boost value

    @property
    def weights(self) -> Dict[str, float]:
        """Return scoring weights as a dict."""
        return {
            "w_sim": self.w_sim,
            "w_literal": self.w_literal,
            "w_rec": self.w_rec,
            "w_type": self.w_type,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return all config as a dict for easy access."""
        return {
            "mode_default": self.mode_default,
            "weights": self.weights,
            "half_life_hours": self.half_life_hours,
            "index_backend": self.index_backend,
            "type_bonus_map": self.type_bonus_map,
            "literal_boost_value": self.literal_boost_value,
        }


# Global config instance
_search_config: SearchConfig | None = None


def get_search_config() -> SearchConfig:
    """Get the global search config instance (singleton)."""
    global _search_config
    if _search_config is None:
        _search_config = SearchConfig()
    return _search_config
