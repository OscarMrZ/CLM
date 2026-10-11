"""Encoder side: an OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM pooling
server) plus an LRU cache of L2-normalised embeddings.

Two request shapes reach the server:

* plain texts, batched in one ``input`` list -- candidates, and text-only states
  with a text encoder (Qwen3-8B last-token pooling, the reference head):

      vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \\
           --enable-prefix-caching --max-model-len 2048 --port 8090

* ``Content`` items (a state with images, or any state when the engine runs with
  ``state_chat``), batched as conversations in one vLLM *chat embeddings* request:
  ``messages`` with image and text parts, rendered by the encoder's own chat
  template.  This is how a vision-language encoder (Qwen3-VL) sees a screenshot
  next to the question:

      vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --served-model-name qwen3-vl-8b \\
           --runner pooling --convert embed --max-model-len 4096 --port 8090

  ``instruction`` becomes the system prompt of every chat request (retrieval-tuned
  encoders such as Qwen3-VL-Embedding expect one, e.g. "Represent the user's input.").
"""
from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import requests

from .content import Content, cache_key


class EmbedderError(RuntimeError):
    pass


def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class Embedder:
    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str = "qwen3-8b",
                 max_tokens: int | None = 2048, cache_size: int = 200_000, batch: int = 32,
                 timeout: float = 300.0, api_key: str | None = None, instruction: str | None = None):
        self.url, self.model, self.max_tokens, self.batch, self.timeout = url, model, max_tokens, batch, timeout
        self.instruction = instruction
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------ requests
    def _post(self, body: dict) -> dict:
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise EmbedderError(f"embedder unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EmbedderError(f"embedder error {r.status_code}: {r.text[:300]}")
        return r.json()

    @staticmethod
    def _vector(e: Any) -> np.ndarray:
        v = np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str) else np.asarray(e, dtype=np.float32)
        return l2(v.astype(np.float32))

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        """One batched ``input`` request for plain texts."""
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "base64"}
        if self.max_tokens:
            body["truncate_prompt_tokens"] = self.max_tokens
        j = self._post(body)
        out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        for d in j["data"]:
            out[d["index"]] = self._vector(d["embedding"])
        return out, int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)

    def _fetch_chat(self, contents: list[Content]) -> tuple[list[np.ndarray], int]:
        """One chat embeddings request for a batch of conversations, one vector each."""
        body: dict[str, Any] = {"model": self.model, "messages": [c.messages(self.instruction) for c in contents],
                                "encoding_format": "base64"}
        if self.max_tokens:
            body["truncate_prompt_tokens"] = self.max_tokens
        j = self._post(body)
        out: list[np.ndarray] = [None] * len(contents)  # type: ignore[list-item]
        for d in j["data"]:
            out[d["index"]] = self._vector(d["embedding"])
        if any(v is None for v in out):
            raise EmbedderError(f"embedder returned {len(j['data'])} vectors for {len(contents)} conversations")
        return out, int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)

    # ------------------------------------------------------------------ public
    def embed(self, items: list[str | Content]) -> tuple[np.ndarray, int]:
        """-> ([n, hidden] L2-normalised embeddings, encoder tokens spent on cache misses).

        ``items`` are plain texts and/or ``Content`` (text + images); each kind goes in
        batched requests of ``batch`` items.
        """
        keys = [cache_key(x) for x in items]
        vecs: dict[str, np.ndarray] = {}
        todo: dict[str, str | Content] = {}
        with self._lock:
            for key, item in zip(keys, items):
                if key in todo or key in vecs:
                    continue
                v = self.cache.get(key)
                if v is None:
                    todo[key] = item
                else:
                    self.cache.move_to_end(key); vecs[key] = v
        tokens = 0
        for fetch, kind in ((self._fetch, str), (self._fetch_chat, Content)):
            pending = [(k, x) for k, x in todo.items() if isinstance(x, kind)]
            for i in range(0, len(pending), self.batch):
                chunk = pending[i:i + self.batch]
                got, spent = fetch([x for _, x in chunk])
                tokens += spent
                self._store([(k, v) for (k, _), v in zip(chunk, got)], vecs)
        return np.stack([vecs[k] for k in keys]), tokens

    def _store(self, pairs: list[tuple[str, np.ndarray]], vecs: dict[str, np.ndarray]) -> None:
        with self._lock:
            for k, v in pairs:
                vecs[k] = v; self.cache[k] = v
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

    def healthy(self) -> bool:
        try:
            return self.session.get(self.url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=5).status_code == 200
        except requests.RequestException:
            return False
