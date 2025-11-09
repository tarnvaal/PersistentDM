# history.py
from . import message
from datetime import datetime
from typing import List, Dict, Any


class History:
    """
    This is used to represent the history of a chat.
    - max_history_tokens is the maximum number of tokens allowed in the active history.
    - system_prompt is the system prompt for the chat. it will always be the first message in the history.
    """

    def __init__(
        self, max_history_tokens: int, system_prompt: str, system_role: str, tokens: int
    ):
        self.max_history_tokens = max_history_tokens
        self.history = []
        self.next_id = 0

        system_message = message.Message(
            role=system_role,
            content=system_prompt,
            tokens=tokens,
            msg_id=self.next_id,
            active=True,
            timestamp=datetime.now(),
        )
        self.history.append(system_message)
        self.next_id += 1

    def add_message(self, role: str, content: str, tokens: int):
        """
        Create a new Message, assign it a unique ID, timestamp it,
        append it to history, and return it.
        """
        msg_id = self.next_id
        self.next_id += 1

        msg = message.Message(
            role=role,
            content=content,
            tokens=tokens,
            msg_id=msg_id,
            active=True,
            timestamp=datetime.now(),
        )

        self.history.append(msg)
        return msg

    def build_context(self) -> List[Dict[str, Any]]:
        """
        Build the context of the history.
        """
        selected = self._select_messages()
        return [{"role": msg.role, "content": msg.content} for msg in selected]

    def _select_messages(self):
        """
        returns the system prompt followed by new messages that are active.
        """
        if not self.history:
            return []

        system_msg = self.history[0]
        allowed_tokens = self.max_history_tokens

        total_tokens = system_msg.tokens
        chosen = []  # system message will be added later, tokens accounted for before looping

        # walk from newest to oldest, skipping index zero (system prompt)
        for msg in reversed(self.history[1:]):
            if not msg.active:
                continue
            if total_tokens + msg.tokens > allowed_tokens:
                msg.deactivate()
                continue
            chosen.append(msg)
            total_tokens += msg.tokens

        chronological = [system_msg] + list(reversed(chosen))

        return chronological
