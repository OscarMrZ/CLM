"""The embedder against a fake /v1/embeddings server: batched texts, one chat request per Content."""
import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from clm.content import Content, ImageRef
from clm.embedder import Embedder, EmbedderError

DIM = 8


def fake_vector(seed) -> np.ndarray:
    """Deterministic per input; ``seed`` is a text or a token-id list."""
    seed_text = seed if isinstance(seed, str) else json.dumps(seed)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha1(seed_text.encode()).digest()[:4], "big"))
    return rng.standard_normal(DIM).astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        self.send_response(200 if self.path.endswith("/v1/models") else 404)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append(body)
        if "messages" in body:
            convs = body["messages"] if isinstance(body["messages"][0], list) else [body["messages"]]
            vecs, tokens = [], 0
            for conv in convs:
                parts = conv[-1]["content"]
                vecs.append(fake_vector(json.dumps(parts, sort_keys=True)))
                tokens += 100 + 50 * sum(p["type"] != "text" for p in parts)
        else:
            vecs, tokens = [fake_vector(t) for t in body["input"]], 5 * len(body["input"])
        out = {"data": [{"index": i, "embedding": base64.b64encode(v.tobytes()).decode()} for i, v in enumerate(vecs)],
               "usage": {"prompt_tokens": tokens}}
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1/embeddings"
    httpd.shutdown()


IMG = ImageRef("data:image/png;base64," + base64.b64encode(b"img-bytes").decode())


def test_texts_are_batched_and_cached(server):
    Handler.requests.clear()
    emb = Embedder(server, "m", max_tokens=64, batch=2)
    vecs, tokens = emb.embed(["a", "b", "c", "a"])
    assert vecs.shape == (4, DIM)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vecs[0], vecs[3])
    assert [r["input"] for r in Handler.requests] == [["a", "b"], ["c"]]
    assert all(r["truncate_prompt_tokens"] == 64 for r in Handler.requests)
    assert tokens == 15
    _, tokens = emb.embed(["a", "b"])
    assert tokens == 0 and len(Handler.requests) == 2          # all hits


def test_content_goes_in_batched_chat_requests(server):
    Handler.requests.clear()
    emb = Embedder(server, "m", max_tokens=64, batch=8, instruction="Represent the user's input.")
    c1 = Content("state one\n\nquestion", (IMG,))
    c2 = Content("state one\n\nquestion")                       # same text, no image: a different vector
    vecs, tokens = emb.embed(["plain", c1, c2, c1])
    assert vecs.shape == (4, DIM)
    assert np.allclose(vecs[1], vecs[3]) and not np.allclose(vecs[1], vecs[2])
    chat = [r for r in Handler.requests if "messages" in r]
    plain = [r for r in Handler.requests if "input" in r]
    assert len(chat) == 1 and len(plain) == 1                    # the two Contents share one request
    convs = chat[0]["messages"]
    assert len(convs) == 2 and all(conv[0] == {"role": "system", "content": "Represent the user's input."} for conv in convs)
    assert chat[0]["truncate_prompt_tokens"] == 64 and chat[0]["encoding_format"] == "base64"
    with_img = next(conv for conv in convs if len(conv[1]["content"]) == 2)
    assert with_img[1]["content"][0] == {"type": "image_url", "image_url": {"url": IMG.url}}
    assert with_img[1]["content"][1] == {"type": "text", "text": c1.text}
    assert tokens == 5 + 150 + 100
    _, tokens = emb.embed([c1, c2])
    assert tokens == 0                                           # cached by text + image digest


def test_chat_batches_split_at_batch_size(server):
    Handler.requests.clear()
    emb = Embedder(server, "m", batch=2)
    emb.embed([Content(f"s{i}", (IMG,)) for i in range(5)])
    sizes = [len(r["messages"]) for r in Handler.requests if "messages" in r]
    assert sizes == [2, 2, 1]


def test_unreachable_server_raises():
    emb = Embedder("http://127.0.0.1:9/v1/embeddings", "m", timeout=0.5)
    with pytest.raises(EmbedderError):
        emb.embed([Content("x", (IMG,))])
    with pytest.raises(EmbedderError):
        emb.embed(["x"])
    assert not emb.healthy()
