from __future__ import annotations

import json
import re
import os
import threading
from typing import List, cast
import logging
from llama_cpp import (
    Llama,
    llama_log_set,
    CreateChatCompletionResponse,
    ChatCompletionRequestMessage,
)
from ctypes import CFUNCTYPE, c_int, c_char_p, c_void_p
from os.path import expanduser
from .history import History
from .gpu import get_free_vram_mib

MODEL_PATH = "~/dev/llm/Harbinger-24B-Q5_K_M.gguf"
# Allow context size override via env; default to 16k for tighter history window
MAX_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16384"))
TOKEN_BUFFER_SIZE = 2048
MIN_FREE_VRAM_MIB = int(os.getenv("MIN_FREE_VRAM_MIB", "23400"))
INIT_WAIT_SECS = float(os.getenv("LLAMA_INIT_WAIT_SECS", "300"))

LOG_CB_TYPE = CFUNCTYPE(None, c_int, c_char_p, c_void_p)


def _noop_log(level, text, user_data):
    return None


_NOOP_LOG_CB = LOG_CB_TYPE(_noop_log)
llama_log_set(_NOOP_LOG_CB, None)  # type: ignore


class Chatter:
    """
    Chatter instances share one global Llama model in VRAM.
    The first Chatter() call loads it. Later Chatter() calls reuse it.
    Each instance still has its own History.
    """

    # class-level shared state
    _llm: Llama | None = None
    _init_error: Exception | None = None
    _initialization_attempted = False  # at least one attempt has been made
    _initialized_successfully = False  # last attempt succeeded
    _init_lock = threading.Lock()
    _ready_event = threading.Event()

    def __init__(self, model_path: str):
        # step 1: ensure model is initialized at class level
        if not Chatter._initialized_successfully:
            # Only one thread initializes; others wait until ready or timeout
            with Chatter._init_lock:
                # Start an attempt if none has occurred yet, or if the last attempt
                # finished and failed. If an attempt is already in progress, we just wait.
                if not Chatter._initialization_attempted or (
                    Chatter._initialization_attempted
                    and not Chatter._initialized_successfully
                    and Chatter._ready_event.is_set()
                ):
                    Chatter._initialize_model(model_path)

        if not Chatter._ready_event.is_set():
            # Wait for initialization to complete (success or failure)
            Chatter._ready_event.wait(timeout=INIT_WAIT_SECS)

        # If still not ready after waiting, surface a clean loading error
        if not Chatter._ready_event.is_set():
            raise RuntimeError(
                "Llama model is still loading. Please try again shortly."
            )

        # if model init failed earlier raise now
        if Chatter._llm is None:
            err = Chatter._init_error
            raise RuntimeError(
                f"Failed to initialize shared Llama model{f': {err}' if err else ''}"
            )

        # bind the shared model handle to this instance
        self.llm = cast(Llama, Chatter._llm)

        # step 2: per-instance setup (your old stuff)
        self.sysprompt_role = "system"
        self.display_name = "DM"
        self.sysprompt_content = (
            "You are the dungeon master. "
            "You describe the world to the player in second person present tense. "
            "You end each response with a question to the player. "
            "If additional system messages include World Facts or NPC Cards, treat them as optional context: "
            "use them only when directly relevant to the player's latest action and the current scene; otherwise ignore them. "
            "Do not force unrelated facts into the narrative. Prioritize the immediate scene, the player's intent, and recent dialogue."
        )

        self.sysprompt_tokens = self.llm.tokenize(
            self.sysprompt_content.encode("utf-8")
        )

        self.token_buffer_size = TOKEN_BUFFER_SIZE
        self.max_history_tokens = MAX_TOKENS - self.token_buffer_size

        self.history = History(
            self.max_history_tokens,
            self.sysprompt_content,
            self.sysprompt_role,
            len(self.sysprompt_tokens),
        )

    @classmethod
    def _initialize_model(cls, model_path: str) -> None:
        """
        Attempt to load the model into VRAM.
        Safe to call multiple times; guarded by _init_lock by callers.
        """
        if cls._initialized_successfully:
            return  # already loaded

        # mark attempt starting
        cls._initialization_attempted = True
        cls._initialized_successfully = False
        cls._init_error = None
        cls._ready_event.clear()

        # check GPU first
        free_vram = get_free_vram_mib()
        if free_vram is None:
            cls._init_error = RuntimeError("No GPU detected. GPU is required.")
            cls._llm = None
            cls._ready_event.set()
            return
        if free_vram < MIN_FREE_VRAM_MIB:
            cls._init_error = RuntimeError(
                f"Not enough VRAM free. Free VRAM: {free_vram} MiB. "
                f"Required: {MIN_FREE_VRAM_MIB} MiB."
            )
            cls._llm = None
            cls._ready_event.set()
            return

        # try to actually build llama
        try:
            cls._llm = Llama(
                model_path=expanduser(model_path),
                n_ctx=MAX_TOKENS,
                n_gpu_layers=-1,  # put all layers on GPU (no CPU fallback)
                n_batch=512,
                verbose=False,
            )
            cls._initialized_successfully = True
        except Exception as e:
            cls._init_error = e
            cls._llm = None
            cls._initialized_successfully = False
        finally:
            cls._ready_event.set()

    @classmethod
    def force_reload(cls, model_path: str | None = None) -> None:
        """Force a re-initialization attempt. Safe if model is already loaded."""
        with cls._init_lock:
            # If already loaded successfully, no-op
            if cls._initialized_successfully and cls._llm is not None:
                return
            # Reset state and attempt again
            cls._llm = None
            cls._init_error = None
            cls._initialization_attempted = False
            cls._initialized_successfully = False
            cls._ready_event.clear()
            cls._initialize_model(model_path or MODEL_PATH)

    def _get_token_count(self, content: str) -> int:
        try:
            tokens = self.llm.tokenize(content.encode("utf-8"))
            return len(tokens) + 5
        except Exception:
            return 10

    @classmethod
    def get_status(cls) -> dict:
        """Return model load status for UI/health.

        States:
        - unloaded: no initialization attempted
        - loading: initialization in progress
        - ready: model loaded and ready
        - failed: initialization completed with error
        """
        state = "unloaded"
        if not cls._initialization_attempted:
            state = "unloaded"
        elif not cls._ready_event.is_set():
            state = "loading"
        elif cls._initialized_successfully and cls._llm is not None:
            state = "ready"
        else:
            state = "failed"

        info: dict[str, object] = {
            "state": state,
            "has_error": cls._init_error is not None,
            "attempted": cls._initialization_attempted,
            "success": cls._initialized_successfully,
        }
        try:
            free_vram = get_free_vram_mib()  # may be None
        except Exception:
            free_vram = None
        info["free_vram_mib"] = free_vram
        info["min_required_vram_mib"] = MIN_FREE_VRAM_MIB
        if cls._init_error is not None:
            info["error"] = str(cls._init_error)
        # Provide a simple UI hint for a traffic-light indicator
        traffic_light = "red"
        if state == "ready":
            traffic_light = "green"
        elif state == "loading":
            traffic_light = "yellow"
        info["traffic_light"] = traffic_light
        return info

    def chat(self, user_input: str, world_facts: str | None = None) -> str:
        # record player message
        self.history.add_message(
            "user",
            user_input,
            self._get_token_count(user_input),
        )

        context = cast(
            List[ChatCompletionRequestMessage],
            self.history.build_context(),
        )

        # If world facts are provided, prepend a transient system message
        # with the facts to guide the model. This message is not recorded
        # in history and applies only to this completion call.
        messages: List[ChatCompletionRequestMessage] = context
        if world_facts:
            facts_msg: ChatCompletionRequestMessage = {
                "role": "system",
                "content": world_facts,
            }
            if messages and messages[0].get("role") == "system":
                messages = [messages[0], facts_msg] + messages[1:]
            else:
                messages = [facts_msg] + messages

        raw_response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False,
        )

        response = cast(CreateChatCompletionResponse, raw_response)
        model_text = (
            response["choices"][0]["message"]["content"] or "[No response generated]"
        )

        # record assistant message
        self.history.add_message(
            "assistant",
            model_text,
            self._get_token_count(model_text),
        )

        return model_text

    def analyze_conversation_for_memories(
        self, conversation_context: dict
    ) -> dict | None:
        """Analyze a conversation turn to extract memorable facts."""
        system_prompt = (
            "You are analyzing a conversation between a player and DM to extract ONE important persistent fact.\n"
            "Look for the MOST important new information:\n"
            "- New NPCs introduced (names, relationships, hostility/friendship) - HIGHEST PRIORITY\n"
            "- Threats or dangers - HIGH PRIORITY\n"
            "- Character goals or objectives stated by the player\n"
            "- New locations discovered (names, descriptions)\n"
            "- Important items mentioned\n"
            "- World state changes\n"
            "\n"
            "CRITICAL: Return EXACTLY ONE JSON object. If there are multiple facts, pick the MOST IMPORTANT one.\n"
            "Do NOT wrap in markdown code blocks. Do NOT return multiple JSON objects.\n"
            "\n"
            "Required JSON structure:\n"
            '{"summary": "concise fact", "entities": ["entity1", "entity2"], '
            '"type": "npc|location|item|goal|threat|world_state|other", '
            '"confidence": 0.85, '
            '"npc": {"name": "Name", "aliases": ["..."], "last_seen_location": "...", "intent": "...", "relationship_to_player": "hostile|friendly|neutral|unknown", "confidence": 0.0}}\n'
            "\n"
            'If NO new persistent information, return: {"summary": "NO_CHANGES", "entities": [], "type": "none", "confidence": 0.0}\n'
            "\n"
            "Good examples:\n"
            '{"summary": "MadHatter Finnigan is hostile to player and attacks on sight", "entities": ["MadHatter Finnigan", "player"], "type": "threat", "confidence": 0.95}\n'
            '{"summary": "Player wants to upgrade their cybernetics", "entities": ["player", "cybernetics"], "type": "goal", "confidence": 0.8}\n'
            '{"summary": "BodyShop 2077 is a cybernetics store in the mall", "entities": ["BodyShop 2077"], "type": "location", "confidence": 0.75}\n'
            '{"summary": "Finnigan stalked the player near BodyShop 2077", "entities": ["MadHatter Finnigan", "BodyShop 2077"], "type": "npc", "confidence": 0.9, "npc": {"name": "MadHatter Finnigan", "aliases": ["Finnigan"], "last_seen_location": "BodyShop 2077", "intent": "hunt the player", "relationship_to_player": "hostile", "confidence": 0.9}}\n'
            "\n"
            "What NOT to store:\n"
            "- Generic atmosphere/descriptions\n"
            "- Temporary moment-to-moment actions\n"
            "\n"
            "Rules:\n"
            "- Prefer 'npc' type with an 'npc' object when a named character is involved.\n"
            "- Include last_seen_location when known; infer relationship if implied (e.g., attacks -> hostile).\n"
            "- Avoid adding generic 'player' to entities unless it adds disambiguation.\n"
            "\n"
            "Return ONLY the JSON object, nothing else."
        )

        user_message = conversation_context.get("user_message", "")
        dm_response = conversation_context.get("dm_response", "")

        user_prompt = (
            f"Player: {user_message}\n\n"
            f"DM: {dm_response}\n\n"
            "Extract any persistent facts that should be remembered. Return the JSON object:"
        )

        parsed_result, err = self._complete_json(
            system_prompt, user_prompt, "memory_analysis", expect="dict", debug=True
        )
        if not parsed_result or not isinstance(parsed_result, dict):
            return None

        required_keys = {"summary", "entities", "type", "confidence"}
        if not all(key in parsed_result for key in required_keys):
            return None

        if (
            parsed_result.get("summary") == "NO_CHANGES"
            or parsed_result.get("type") == "none"
        ):
            return None

        return parsed_result

    def summarize_world_changes(
        self, planner_json: dict, resolved_outcome: dict | None = None
    ) -> dict | None:
        """Use LLM to summarize world changes from planner responses."""
        system_prompt = (
            "You are analyzing a narrative planner's response to extract important persistent facts.\n"
            "Look for:\n"
            "- New NPCs introduced (names, relationships, hostility)\n"
            "- New locations discovered\n"
            "- Important items or goals mentioned\n"
            "- Changes to relationships or world state\n"
            "- Player objectives or threats\n"
            "\n"
            "Return ONLY a valid JSON object with this exact structure:\n"
            '{"summary": "concise fact to remember", "entities": ["entity1", "entity2"], '
            '"type": "world_change|entity_change|relationship_change|knowledge_change|location_change|other", '
            '"confidence": 0.0-1.0}\n'
            "\n"
            'If NO new persistent information is introduced, return: {"summary": "NO_CHANGES", "entities": [], "type": "none", "confidence": 0.0}\n'
            "Be generous - if the player mentions a new NPC name or important fact, that should be stored."
        )

        planner_text = self._safe_truncate(json.dumps(planner_json, indent=2), 1000)
        outcome_text = ""
        if resolved_outcome:
            outcome_text = f"\n\nResolved Outcome:\n{self._safe_truncate(json.dumps(resolved_outcome, indent=2), 500)}"

        user_prompt = (
            f"Planner Response:\n{planner_text}{outcome_text}\n\n"
            "Task: If significant world changes occurred, provide a concise summary. Focus on:\n"
            "- New information that persists\n"
            "- Changes to relationships, locations, or items\n"
            "- Security or political developments\n\n"
            "Return the JSON object:"
        )

        parsed_result, err = self._complete_json(
            system_prompt, user_prompt, "world_change_summary", expect="dict"
        )
        if not parsed_result or not isinstance(parsed_result, dict):
            return None

        required_keys = {"summary", "entities", "type", "confidence"}
        if not all(key in parsed_result for key in required_keys):
            return None

        if (
            parsed_result.get("summary") == "NO_CHANGES"
            or parsed_result.get("type") == "none"
        ):
            return None

        return parsed_result

    def get_planner_response(
        self,
        world_facts: list[dict],
        recent_scene: list[dict],
        player_action: str,
        debug: bool = False,
    ) -> dict | None:
        """Get a planner response for a player action using world context."""
        from app.world.queries import make_planner_prompt

        prompt = make_planner_prompt(world_facts, recent_scene, player_action)

        # Extract system and user messages from the prompt
        system_content = prompt[0]["content"]
        user_content = (
            prompt[1]["content"]
            if len(prompt) > 1
            else "Analyze this situation and return the JSON response."
        )

        # Complete the JSON response
        parsed_result, err = self._complete_json(
            system_content, user_content, "planner_response", expect="dict", debug=debug
        )
        return parsed_result if isinstance(parsed_result, dict) else None

    def store_world_change_from_planner(
        self,
        world_memory,
        world_facts: list[dict],
        recent_scene: list[dict],
        player_action: str,
        resolved_outcome: dict | None = None,
        debug: bool = False,
    ) -> str | None:
        """Run planner -> summarize -> store in world memory with dedupe. Returns memory id or None.

        world_memory: expected to have add_memory(summary, entities, type, ...)
        """
        try:
            planner_json = self.get_planner_response(
                world_facts, recent_scene, player_action, debug=debug
            )
        except Exception:
            planner_json = None
        if not isinstance(planner_json, dict):
            return None
        try:
            summary = self.summarize_world_changes(planner_json, resolved_outcome)
        except Exception:
            summary = None
        if not isinstance(summary, dict):
            return None
        try:
            conf = float(summary.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf <= 0.6:
            return None
        # Persist to world memory with dedupe and provenance
        try:
            source_ctx = "Planner-derived world change"
            return world_memory.add_memory(
                summary.get("summary", ""),
                summary.get("entities", []) or [],
                summary.get("type", "other"),
                npc=None,
                dedupe_check=True,
                source_context=source_ctx,
            )
        except Exception:
            return None

    def infer_player_movement(
        self,
        current_location: dict | None,
        exits: list[dict],
        user_message: str,
        dm_response: str,
        debug: bool = False,
    ) -> dict | None:
        """Ask the LLM if the player moved to a connected location.

        Returns a dict like:
          {"move": true, "target": "LocationName", "reason": "...", "confidence": 0.0-1.0}
        or {"move": false, "confidence": 0.0-1.0}
        """
        system = (
            "You are a movement detector for a text RPG.\n"
            "Given the player's message and the DM reply, decide if the player moved to one of the available exits.\n"
            "Return ONLY JSON.\n"
            'Schema: {"move": bool, "target": string|null, "reason": string, "confidence": number}'
        )
        loc_name = (current_location or {}).get("name", "unknown")
        exit_lines = []
        for e in exits:
            tgt = e.get("to_location", "")
            desc = e.get("description", "")
            if tgt:
                exit_lines.append(f"- {desc} -> {tgt}")
        exits_text = "\n".join(exit_lines) if exit_lines else "(none)"

        user = (
            f"Current location: {loc_name}\n"
            f"Available exits (description -> target):\n{exits_text}\n\n"
            f"Player: {user_message}\nDM: {dm_response}\n\n"
            "Answer with the JSON object."
        )
        parsed_result, err = self._complete_json(
            system, user, "infer_player_movement", expect="dict", debug=debug
        )
        return parsed_result if isinstance(parsed_result, dict) else None

    def extract_graph_changes(
        self,
        user_message: str,
        dm_response: str,
        current_location_name: str | None = None,
        debug: bool = False,
    ) -> dict | None:
        """Ask the LLM to suggest new locations and connections mentioned this turn.

        Returns a dict like:
          {
            "new_locations": [{"name": str, "description": str}],
            "new_connections": [{"from": str, "to": str, "description": str, "verb": str|null}],
            "confidence": number
          }
        or {"new_locations": [], "new_connections": [], "confidence": 0.0}
        """
        system = (
            "You are a world graph extractor.\n"
            "From the player's message and DM reply, extract any NEW locations and NEW connections.\n"
            "Keep it conservative; only include clear, explicit additions.\n"
            "Return ONLY JSON with keys new_locations (list), new_connections (list), confidence (0-1).\n"
            "Location names should be concise canonical names (e.g., 'Town Square').\n"
            "Connections should include a short description of the route/means (e.g., 'a creaking wooden door')."
        )
        user = (
            (
                f"Current location: {current_location_name}\n"
                if current_location_name
                else ""
            )
            + f"Player: {user_message}\nDM: {dm_response}\n\n"
            + "Return the JSON object."
        )
        parsed_result, err = self._complete_json(
            system, user, "extract_graph_changes", expect="dict", debug=debug
        )
        return parsed_result if isinstance(parsed_result, dict) else None

    def extract_memory_from_text(self, text: str, debug: bool = False) -> dict | None:
        """Extract ONE durable memory from arbitrary text.

        Returns a dict compatible with WorldMemory.add_memory input or None.
        """
        system = (
            "You analyze narrative text to extract ONE important persistent fact.\n"
            "Return ONLY a JSON object with keys summary, entities, type, confidence, and optional npc."
            ' If nothing durable, return {"summary": "NO_CHANGES", "entities": [], "type": "none", "confidence": 0.0}.'
        )
        user = text[:4000]
        parsed_result, err = self._complete_json(
            system, user, "extract_memory_from_text", expect="dict", debug=debug
        )
        return parsed_result if isinstance(parsed_result, dict) else None

    def summarize_snippet(self, text: str, debug: bool = False) -> dict | None:
        """Produce a short 1-2 sentence checkpoint summary for UI."""
        system = (
            "Summarize the following text in 1-2 concise sentences suitable for a progress checkpoint.\n"
            'Return ONLY {"summary": string}.'
        )
        user = text[:4000]
        parsed_result, err = self._complete_json(
            system, user, "summarize_snippet", expect="dict", debug=debug
        )
        return parsed_result if isinstance(parsed_result, dict) else None

    def extract_memories_from_text(
        self, text: str, max_items: int = 5, debug: bool = False
    ) -> list[dict] | None:
        """Extract up to max_items durable memories from text in one pass.

        Returns a list of memory dicts; each has keys summary, entities, type, confidence, optional npc.
        """
        system = (
            "You analyze narrative/game text to extract IMPORTANT persistent facts.\n"
            "Return ONLY a JSON array (max {max_items}) of objects with keys: summary, entities, type, confidence, optional npc.\n"
            "Types: npc|location|item|goal|threat|world_state|other.\n"
            "If nothing durable, return []."
        ).replace("{max_items}", str(max_items))
        user = text[:4000]
        parsed_result, err = self._complete_json(
            system, user, "extract_memories_from_text", expect="list", debug=debug
        )
        result = parsed_result
        if isinstance(result, list):
            # Filter to dicts
            return [m for m in result if isinstance(m, dict)]
        # Some models might return an object with list under a key
        if isinstance(result, dict):
            for key in ("memories", "items", "facts"):
                val = result.get(key)
                if isinstance(val, list):
                    return [m for m in val if isinstance(m, dict)]
        return []

    def _complete_json(
        self,
        system: str,
        user: str,
        request_type: str,
        expect: str = "dict",
        debug: bool = False,
    ) -> tuple[dict | list | None, Exception | None]:
        """Complete a prompt expecting JSON response with retry on parse failure.

        Returns (parsed_json, error). On type mismatch, returns (parsed, TypeError)
        so the caller can decide whether to adapt or treat as fatal.
        """
        logger = logging.getLogger(__name__)
        messages = cast(
            List[ChatCompletionRequestMessage],
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

        for attempt in range(2):
            try:
                raw_response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,  # Increased to handle complex planner responses
                    temperature=0.2,
                    top_p=0.8,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stream=False,
                )

                response = cast(CreateChatCompletionResponse, raw_response)
                model_text = response["choices"][0]["message"]["content"] or ""

                if debug:
                    print(f"\n[{request_type}] Attempt {attempt + 1} - Raw response:")
                    print(
                        f"  Text: {model_text[:200]}{'...' if len(model_text) > 200 else ''}"
                    )

                # Strip markdown code fences if present
                cleaned_text = model_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]  # Remove ```json
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]  # Remove ```
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]  # Remove trailing ```
                cleaned_text = cleaned_text.strip()

                # Handle multiple JSON objects - take only the first one
                # Find the first complete JSON object
                if cleaned_text.startswith("{"):
                    brace_count = 0
                    first_obj_end = -1
                    for i, char in enumerate(cleaned_text):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                first_obj_end = i + 1
                                break

                    if first_obj_end > 0:
                        cleaned_text = cleaned_text[:first_obj_end]

                # Fix common JSON formatting errors
                cleaned_text = cleaned_text.replace(
                    ' "confidence": 0. ', ' "confidence": 0.'
                )
                # Fix "confidence": 0. 95 -> "confidence": 0.95
                cleaned_text = re.sub(
                    r'"confidence"\s*:\s*0\.\s+', '"confidence": 0.', cleaned_text
                )

                # Try to parse as JSON
                try:
                    parsed = json.loads(cleaned_text)
                    if expect == "dict" and not isinstance(parsed, dict):
                        type_err = TypeError(
                            f"{request_type}: expected dict but got {type(parsed).__name__}"
                        )
                        if debug:
                            print(f"  ✗ Type mismatch: {type_err}")
                        logger.warning(
                            "%s parse type mismatch on attempt %d: %s",
                            request_type,
                            attempt + 1,
                            type_err,
                        )
                        return parsed, type_err
                    if expect == "list" and not isinstance(parsed, list):
                        type_err = TypeError(
                            f"{request_type}: expected list but got {type(parsed).__name__}"
                        )
                        if debug:
                            print(f"  ✗ Type mismatch: {type_err}")
                        logger.warning(
                            "%s parse type mismatch on attempt %d: %s",
                            request_type,
                            attempt + 1,
                            type_err,
                        )
                        return parsed, type_err

                    if debug:
                        print("  ✓ Valid JSON parsed")
                    return parsed, None
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"  ✗ JSON parse error: {e}")
                    logger.warning(
                        "%s JSON parse failed on attempt %d: %s",
                        request_type,
                        attempt + 1,
                        e,
                    )
                    if attempt == 0:
                        # Retry with correction prompt
                        correction_prompt = (
                            f"Previous response was not valid JSON: {model_text[:100]}...\n"
                            "Please return ONLY a valid JSON object with the required structure."
                        )
                        messages.append({"role": "assistant", "content": model_text})
                        messages.append({"role": "user", "content": correction_prompt})
                        continue
                    return None, e

            except Exception as e:
                if debug:
                    print(f"  ✗ Exception during generation: {e}")
                logging.getLogger(__name__).error(
                    "%s generation failed on attempt %d: %s",
                    request_type,
                    attempt + 1,
                    e,
                )
                return None, e

        return None, RuntimeError(f"{request_type}: failed to produce valid JSON")

    def _safe_truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens, preserving word boundaries."""
        if not text:
            return text

        try:
            tokens = self.llm.tokenize(text.encode("utf-8"))
            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            truncated_bytes = self.llm.detokenize(truncated_tokens)
            truncated_text = truncated_bytes.decode("utf-8", errors="ignore")

            last_space = truncated_text.rfind(" ")
            if last_space > len(truncated_text) * 0.8:
                truncated_text = truncated_text[:last_space]

            return truncated_text + "..."

        except Exception:
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."


if __name__ == "__main__":
    dm = Chatter(MODEL_PATH)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        print(dm.chat(user_input))
