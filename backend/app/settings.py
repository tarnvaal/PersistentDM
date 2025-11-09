"""
Application settings and configuration constants.
All magic numbers and configurable values should be defined here.
"""

import os
from typing import Dict, Any


# Memory retrieval settings
MEMORY_K_GENERAL = int(os.getenv("MEMORY_K_GENERAL", "25"))
MEMORY_K_PER_ENTITY = int(os.getenv("MEMORY_K_PER_ENTITY", "5"))
MEMORY_K_PER_TYPE = int(os.getenv("MEMORY_K_PER_TYPE", "3"))
MEMORY_MIN_TOTAL_SCORE = float(os.getenv("MEMORY_MIN_TOTAL_SCORE", "0.75"))
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.75"))

# NPC retrieval settings
NPC_K_DEFAULT = int(os.getenv("NPC_K_DEFAULT", "3"))
NPC_MIN_SCORE = float(os.getenv("NPC_MIN_SCORE", "0.55"))

# Confidence thresholds for LLM operations
CONFIDENCE_THRESHOLD_MEMORY = float(os.getenv("CONFIDENCE_THRESHOLD_MEMORY", "0.6"))
CONFIDENCE_THRESHOLD_LOCATION = float(os.getenv("CONFIDENCE_THRESHOLD_LOCATION", "0.7"))

# Context building limits
MAX_WORLD_FACTS_WORDS = int(os.getenv("MAX_WORLD_FACTS_WORDS", "2000"))
MAX_NPC_CARDS = int(os.getenv("MAX_NPC_CARDS", "8"))

# Ingest settings
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "12000"))

# Debug settings
PDM_DEBUG_ERRORS = os.getenv("PDM_DEBUG_ERRORS", "1") == "1"

# Model settings
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "~/dev/llm/Harbinger-24B-Q5_K_M.gguf")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16384"))
TOKEN_BUFFER_SIZE = int(os.getenv("TOKEN_BUFFER_SIZE", "2048"))

# Relationship precedence order (higher number = higher precedence)
RELATIONSHIP_ORDER: Dict[str, int] = {
    "hostile": 3,
    "friendly": 2,
    "neutral": 1,
    "unknown": 0,
}

# Data directories
INGESTS_DIR = os.getenv("INGESTS_DIR", "./data/ingests")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "text"


def get_settings_summary() -> Dict[str, Any]:
    """Get a summary of current settings for debugging."""
    return {
        "memory": {
            "k_general": MEMORY_K_GENERAL,
            "k_per_entity": MEMORY_K_PER_ENTITY,
            "k_per_type": MEMORY_K_PER_TYPE,
            "min_total_score": MEMORY_MIN_TOTAL_SCORE,
            "similarity_threshold": MEMORY_SIMILARITY_THRESHOLD,
        },
        "npc": {
            "k_default": NPC_K_DEFAULT,
            "min_score": NPC_MIN_SCORE,
        },
        "confidence": {
            "memory": CONFIDENCE_THRESHOLD_MEMORY,
            "location": CONFIDENCE_THRESHOLD_LOCATION,
        },
        "limits": {
            "max_world_facts_words": MAX_WORLD_FACTS_WORDS,
            "max_npc_cards": MAX_NPC_CARDS,
            "max_chunk_size": MAX_CHUNK_SIZE,
        },
        "model": {
            "default_path": DEFAULT_MODEL_PATH,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        },
        "debug": {
            "errors": PDM_DEBUG_ERRORS,
        },
    }
