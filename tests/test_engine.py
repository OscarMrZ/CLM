"""Engine + server with a fake embedder and tiny random heads on the CPU."""
import base64
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clm.content import Content, with_images  # noqa: E402
from clm.engine import Engine  # noqa: E402
from clm.heads import make_head  # noqa: E402

HIDDEN, PROJ = 16, 8
IMG = "data:image/png;base64," + base64.b64encode(b"shot-1").decode()
IMG2 = "data:image/png;base64," + base64.b64encode(b"shot-2").decode()


class FakeEmbedder:
    """Deterministic vectors; a Content differs from its text by its images."""

    def __init__(self):
        self.calls: list = []

    def embed(self, items):
        self.calls.append(list(items))
        out = []
        for x in items:
            seed = x.key if isinstance(x, Content) else x
            rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
            v = rng.standard_normal(HIDDEN).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out), 7 * len(items)

    def healthy(self):
        return True


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    sh, ah = make_head(32, 3, PROJ, hidden=HIDDEN, layernorm=True), make_head(32, 3, PROJ, hidden=HIDDEN, layernorm=True)
    path = tmp_path / "tiny.pt"
    torch.save({"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                "logit_scale": torch.tensor(math.log(1 / 0.07)),
                "cfg": {"width": 32, "depth": 3, "projection_dim": PROJ, "hidden_size": HIDDEN, "layernorm": True}}, path)
    return str(path)


Q = {"btn": {"type": "choice", "instructions": "Which button?", "criteria": {"submit": "Submit", "cancel": "Cancel"}},
     "ok": {"type": "noul", "instructions": "Is the form filled?"}}


def test_image_state_changes_the_answer_and_is_cached(ckpt):
    fe = FakeEmbedder()
    eng = Engine(embedder=fe, checkpoint=ckpt, device="cpu", action_cache="4MiB")
    text_only = eng.answer("A login form.", Q)
    with_img = eng.answer(with_images("A login form.", [IMG]), Q)
    other_img = eng.answer(with_images("A login form.", [IMG2]), Q)
    p = [r["answers"]["btn"]["probabilities"]["submit"] for r in (text_only, with_img, other_img)]
    assert len({round(x, 6) for x in p}) == 3                 # the image reaches the state vector
    assert with_img["usage"]["input_tokens"] > 0
    n = len(fe.calls)
    again = eng.answer(with_images("A login form.", [IMG]), Q)
    assert again["answers"] == with_img["answers"]
    assert len(fe.calls) == n and again["usage"]["input_tokens"] == 0   # arena hit: no encoder call
    stats = eng.arena.stats()
    assert stats["hit_rate"] and stats["hit_rate"] > 0


def test_state_chat_wraps_text_states(ckpt):
    fe = FakeEmbedder()
    eng = Engine(embedder=fe, checkpoint=ckpt, device="cpu", action_cache="0", state_chat=True)
    eng.answer("A login form.", {"ok": Q["ok"]})
    states = [x for call in fe.calls for x in call if isinstance(x, Content)]
    cands = [x for call in fe.calls for x in call if isinstance(x, str)]
    assert states and all(not c.images for c in states)
    assert cands                                                  # candidates stay plain text


def test_raw_model_and_rank_accept_images(ckpt):
    eng = Engine(embedder=FakeEmbedder(), checkpoint=ckpt, device="cpu", action_cache="0")
    out = eng.answer(with_images("frame", [IMG]), Q, model="clm-raw")
    assert set(out["answers"]) == {"btn", "ok"}
    ranked = eng.rank(with_images("frame", [IMG]), ["jump", "duck", "run"], "What now?")
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert abs(sum(r["prob"] for r in ranked) - 1) < 1e-6


def test_server_round_trip_with_images(ckpt):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from clm.server import create_app
    app = create_app(Engine(embedder=FakeEmbedder(), checkpoint=ckpt, device="cpu", action_cache="0"), ui=False)
    c = TestClient(app)
    r = c.post("/v1/systemone", json={"state": with_images("A login form.", [IMG]), "questions": Q})
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["btn"]["choice"] in ("submit", "cancel")
    bad = c.post("/v1/systemone", json={"state": [{"type": "image_url", "image_url": {"url": "/tmp/x.png"}}],
                                        "questions": Q})
    assert bad.status_code == 422 and "data:" in bad.json()["detail"]
    r = c.post("/v1/rank", json={"context": with_images("frame", [IMG]), "question": "Now?", "answers": ["a", "b"]})
    assert r.status_code == 200 and len(r.json()["ranked"]) == 2


def test_mock_engine_handles_images():
    from playground_mock import MockEngine
    eng = MockEngine()
    a = eng.answer("A login form.", Q)["answers"]["btn"]["probabilities"]
    b = eng.answer(with_images("A login form.", [IMG]), Q)["answers"]["btn"]["probabilities"]
    assert a != b
