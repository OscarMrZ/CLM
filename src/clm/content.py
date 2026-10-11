"""Multimodal states: text plus images, and how they reach the encoder.

A state may carry images by being written as a list of **content parts**, the
shape OpenAI-style chat APIs (and vLLM's chat embeddings endpoint) already use::

    [{"type": "text", "text": "Which button submits the form?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

Every element must be an object whose ``type`` is ``text``, ``image_url`` or
``image``; any other list is still rendered as prose (``- item`` lines) by
``schema.to_text``.  Images are ``data:`` URIs or ``http(s)`` URLs -- never local
paths, so a request cannot make the server read its own disk.  ``image_part``
turns a local file into a ``data:`` URI for callers.

``Content`` is what the engine hands to the embedder for a state that needs the
encoder's chat path: the state text (question already appended) and the images.
Its ``key`` identifies it for caching: the text plus a digest of each image, under a
prefix that keeps the chat path apart from a plain text of the same words.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from typing import Any

PART_TYPES = ("text", "image_url", "image")
_SCHEMES = ("data:", "http://", "https://")


@dataclass(frozen=True)
class ImageRef:
    """One image by URL. ``key`` is short and stable: a digest of ``data:`` payloads, the URL itself otherwise."""
    url: str

    @property
    def key(self) -> str:
        if self.url.startswith("data:"):
            return "sha256:" + hashlib.sha256(self.url.encode("ascii", "ignore")).hexdigest()
        return self.url

    def part(self) -> dict:
        return {"type": "image_url", "image_url": {"url": self.url}}


@dataclass(frozen=True)
class Content:
    """A state for the encoder's chat path: text (context + question) and zero or more images."""
    text: str
    images: tuple[ImageRef, ...] = ()

    @property
    def key(self) -> str:
        """Cache identity: the text and the image digests, in order.  Prefixed, so a text-only
        ``Content`` (the chat path) never shares a cache entry with the same plain text."""
        return "\x1f".join(("chat", self.text, *(i.key for i in self.images)))

    def parts(self) -> list[dict]:
        """Content parts for a chat message: images first, then the text (how VLM recipes order them)."""
        out = [i.part() for i in self.images]
        if self.text:
            out.append({"type": "text", "text": self.text})
        return out

    def messages(self, instruction: str | None = None) -> list[dict]:
        """A one-turn conversation; ``instruction`` becomes the system prompt when given."""
        msgs = [{"role": "system", "content": instruction}] if instruction else []
        return msgs + [{"role": "user", "content": self.parts()}]


def is_content_parts(state: Any) -> bool:
    """True for a non-empty list whose every element is a typed content part."""
    return (isinstance(state, list) and bool(state)
            and all(isinstance(p, dict) and p.get("type") in PART_TYPES for p in state))


def _image_url(part: dict) -> str:
    if part["type"] == "image_url":
        u = part.get("image_url")
        u = u.get("url") if isinstance(u, dict) else u
    else:
        u = part.get("image") or part.get("url")
        if isinstance(u, dict):
            u = u.get("url")
    if not isinstance(u, str) or not u:
        raise ValueError("image part needs a URL: {'type': 'image_url', 'image_url': {'url': ...}}")
    if not u.startswith(_SCHEMES):
        raise ValueError("image URLs must be data: URIs or http(s) URLs (use clm.image_part for local files)")
    return u


def split_parts(state: Any) -> tuple[Any, list[ImageRef]]:
    """(text-only state, images) for a content-part list; (state, []) for anything else.

    The text parts are joined with blank lines in order (a single text part is
    returned as is, so an object or array inside one still renders as prose).
    """
    if not is_content_parts(state):
        return state, []
    texts, images = [], []
    for p in state:
        if p["type"] == "text":
            texts.append(p.get("text", ""))
        else:
            images.append(ImageRef(_image_url(p)))
    if len(texts) == 1:
        return texts[0], images
    from .schema import to_text
    return "\n\n".join(t for t in (to_text(x).strip() for x in texts) if t), images


def image_part(source: str | bytes, mime: str | None = None) -> dict:
    """A content part for an image: a path or bytes become a ``data:`` URI, a URL is passed through."""
    if isinstance(source, bytes):
        if not mime:
            raise ValueError("image_part(bytes) needs mime=")
        url = f"data:{mime};base64,{base64.b64encode(source).decode('ascii')}"
    elif source.startswith(_SCHEMES):
        url = source
    else:
        mime = mime or mimetypes.guess_type(source)[0]
        if not mime or not mime.startswith("image/"):
            raise ValueError(f"cannot tell the image type of {source!r}; pass mime=")
        with open(os.path.expanduser(source), "rb") as fh:
            url = f"data:{mime};base64,{base64.b64encode(fh.read()).decode('ascii')}"
    return ImageRef(url).part()


def with_images(state: Any, images: list[str | bytes | dict]) -> list[dict]:
    """``state`` (any text-renderable value) plus images -> a content-part list for the API."""
    from .schema import to_text
    parts: list[dict] = [{"type": "text", "text": state if isinstance(state, str) else to_text(state)}]
    for im in images:
        parts.append(im if isinstance(im, dict) else image_part(im))
    return parts


def cache_key(x: Any) -> str:
    """The string that identifies ``x`` in the embedding caches."""
    return x.key if isinstance(x, Content) else str(x)
