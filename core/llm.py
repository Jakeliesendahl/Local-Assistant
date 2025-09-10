# core/llm.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import requests
import json
import time


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    """
    Minimal Ollama chat client for local, offline use.

    - Uses /api/chat (multi-turn capable).
    - Works with llama3, mistral, etc., as long as the model is pulled in Ollama.
    - Returns the final text response (streamed or non-streamed).
    """
    model: str = "llama3"
    host: str = "http://localhost:11434"
    timeout: int = 120
    default_system: Optional[str] = None
    default_options: Dict = field(default_factory=lambda: {
        # keep these small for CPU laptops; tweak as needed
        "temperature": 0.2,
        "num_ctx": 2048,
    })

    # ---------- convenience ----------
    @property
    def _chat_url(self) -> str:
        return f"{self.host}/api/chat"

    @property
    def _version_url(self) -> str:
        return f"{self.host}/api/version"

    @property
    def _tags_url(self) -> str:
        return f"{self.host}/api/tags"

    # ---------- health ----------
    def is_available(self) -> bool:
        try:
            r = requests.get(self._version_url, timeout=5)
            if r.status_code != 200:
                return False
            # optional: confirm at least one model is present
            t = requests.get(self._tags_url, timeout=5)
            return t.status_code == 200
        except requests.RequestException:
            return False

    # ---------- core chat ----------
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict] = None,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Send a prompt and return the model's response text.
        - `history`: optional list of {"role": "user"|"assistant"|"system", "content": "..."}.
        - Set `stream=True` to accumulate tokens as they arrive (reduces latency).
        """
        if not self.is_available():
            raise LLMError(
                "Ollama does not appear to be running at http://localhost:11434. "
                "Start it (e.g., `ollama serve`) and make sure your model is pulled "
                f"(e.g., `ollama pull {self.model}`)."
            )

        sys_msg = (system or self.default_system)
        msgs: List[Dict[str, str]] = []

        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})

        if history:
            msgs.extend(history)

        msgs.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": stream,
            "options": {**self.default_options, **(options or {})},
        }

        try:
            if stream:
                return self._chat_stream(payload)
            else:
                r = requests.post(self._chat_url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                # ollama /api/chat returns {"message": {"content": "..."} , ...}
                return data.get("message", {}).get("content", "").strip()
        except requests.RequestException as e:
            raise LLMError(f"Chat request failed: {e}") from e

    # ---------- streaming helper ----------
    def _chat_stream(self, payload: Dict) -> str:
        r = requests.post(self._chat_url, json=payload, stream=True, timeout=self.timeout)
        r.raise_for_status()
        buf: List[str] = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            delta = obj.get("message", {}).get("content", "")
            if delta:
                buf.append(delta)
                # optional: live print
                print(delta, end="", flush=True)
        print()  # newline after stream
        return "".join(buf).strip()
