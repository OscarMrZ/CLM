"""Content parts: detection, splitting, keys, and how the schema renders them."""
import base64

import pytest

from clm.content import Content, ImageRef, cache_key, image_part, is_content_parts, split_parts, with_images
from clm.schema import build_pairs, state_input, state_text

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
DATA = f"data:image/png;base64,{PNG}"
PARTS = [{"type": "text", "text": "A login form with two buttons."},
         {"type": "image_url", "image_url": {"url": DATA}}]


def test_detection_is_strict():
    assert is_content_parts(PARTS)
    assert not is_content_parts([])
    assert not is_content_parts(["a", "b"])
    assert not is_content_parts([{"type": "text", "text": "x"}, {"name": "y"}])
    assert not is_content_parts([{"type": "button", "label": "ok"}])     # a user's own 'type' field
    assert not is_content_parts({"type": "text", "text": "x"})


def test_split_parts_keeps_text_and_images():
    text, images = split_parts(PARTS)
    assert text == "A login form with two buttons."
    assert images == [ImageRef(DATA)]
    assert images[0].key.startswith("sha256:")
    assert ImageRef("https://x.example/a.png").key == "https://x.example/a.png"


def test_split_parts_joins_several_text_parts_and_accepts_shorthand():
    text, images = split_parts([{"type": "text", "text": "first"}, {"type": "image_url", "image_url": DATA},
                                {"type": "text", "text": {"score": 3}}, {"type": "image", "image": DATA}])
    assert text == "first\n\nscore: 3"
    assert len(images) == 2


def test_rejects_local_paths_and_missing_urls():
    with pytest.raises(ValueError):
        split_parts([{"type": "image_url", "image_url": {"url": "/etc/passwd"}}])
    with pytest.raises(ValueError):
        split_parts([{"type": "image_url"}])


def test_state_text_and_state_input():
    assert state_text(PARTS, "Which button submits?") == "A login form with two buttons.\n\nWhich button submits?"
    si = state_input(PARTS, "Which button submits?")
    assert isinstance(si, Content)
    assert si.text == "A login form with two buttons.\n\nWhich button submits?"
    assert si.images == (ImageRef(DATA),)
    assert state_input("plain", "q") == "plain\n\nq"                          # text-only stays a string
    assert state_input("plain", "q", chat=True) == Content("plain\n\nq")       # unless the engine asks for chat


def test_build_pairs_with_images():
    q = {"ok": {"type": "choice", "instructions": "Which?", "criteria": {"a": "Submit", "b": "Cancel"}}}
    (si, keys, texts), = build_pairs(PARTS, q).values()
    assert isinstance(si, Content) and keys == ["a", "b"] and texts == ["Submit", "Cancel"]
    (plain, *_), = build_pairs("hello", q).values()
    assert plain == "hello\n\nWhich?"


def test_content_key_and_messages():
    c = Content("t", (ImageRef(DATA),))
    assert cache_key(c) == c.key and cache_key("t") == "t"
    assert Content("t").key != "t"                   # the chat path never shares a plain text's cache entry
    assert c.key != Content("t").key
    assert c.key != Content("t", (ImageRef("https://x/a.png"),)).key
    m = c.messages("Represent the user's input.")
    assert m[0] == {"role": "system", "content": "Represent the user's input."}
    assert m[1]["role"] == "user"
    assert [p["type"] for p in m[1]["content"]] == ["image_url", "text"]
    assert Content("t").messages()[0]["role"] == "user"


def test_image_part_and_with_images(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    part = image_part(str(p))
    assert part == {"type": "image_url", "image_url": {"url": DATA}}
    assert image_part("https://x/a.png")["image_url"]["url"] == "https://x/a.png"
    assert image_part(b"\x89PNG\r\n\x1a\nfake", mime="image/png") == part
    with pytest.raises(ValueError):
        image_part(b"x")
    state = with_images({"hp": 3}, [str(p), part])
    assert state[0] == {"type": "text", "text": "hp: 3"}
    assert state[1] == state[2] == part
    assert is_content_parts(state)
