"""Trainer side: image-bearing transitions reach the encoder through the chat path."""
import base64
import json
import os
import threading
import types
from http.server import HTTPServer

import numpy as np
import pytest

import adapters
import embed_utils
from test_embedder import Handler

torch = pytest.importorskip("torch")
import finetune  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1/embeddings"
    httpd.shutdown()


def test_chat_messages_shapes():
    m = embed_utils.chat_messages("look", ["https://x/a.png"], "Represent the user's input.")
    assert m[0] == {"role": "system", "content": "Represent the user's input."}
    assert m[1]["content"] == [{"type": "image_url", "image_url": {"url": "https://x/a.png"}},
                               {"type": "text", "text": "look"}]
    m = embed_utils.chat_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                                   {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}],
                                  ["https://x/b.png"])
    assert m[3]["content"][0]["type"] == "image_url" and m[3]["content"][1] == {"type": "text", "text": "u2"}
    assert m[1]["content"] == "u1"
    assert embed_utils.chat_messages({"hp": 2})[0]["content"][-1] == {"type": "text", "text": "hp: 2"}


def test_read_transitions_resolves_images(tmp_path):
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "s0.png").write_bytes(PNG)
    recs = [{"state": "frame", "action": "jump", "task_id": "t", "step_idx": 0, "images": ["shots/s0.png"]},
            {"state": "frame", "action": "run", "task_id": "t", "step_idx": 1, "images": ["https://x/a.png"]},
            {"state": "frame", "action": "run", "task_id": "t", "step_idx": 2}]
    f = tmp_path / "steps.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs))
    out = adapters.read_transitions(str(f))
    assert out[0]["images"] == [str(tmp_path / "shots" / "s0.png")]
    assert out[1]["images"] == ["https://x/a.png"] and "images" not in out[2]
    assert adapters.has_images(out)
    f.write_text(json.dumps([{**recs[2], "images": "shots/s0.png"}]))
    with pytest.raises(ValueError):
        adapters.read_transitions(str(f))


def test_server_backend_embed_chat(server):
    Handler.requests.clear()
    be = embed_utils.ServerBackend(server, "m", batch=8)
    vecs = be.embed_chat([embed_utils.chat_messages("a", ["https://x/a.png"]), embed_utils.chat_messages("a")])
    assert vecs.shape == (2, 8) and not np.allclose(vecs[0], vecs[1])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert len(Handler.requests) == 1 and len(Handler.requests[0]["messages"]) == 2   # one batched request


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        out = []
        for m in messages:
            c = m["content"]
            body = c if isinstance(c, str) else "".join("<image>" if p["type"] == "image" else p["text"] for p in c)
            out.append(f"<|{m['role']}|>{body}")
        return "".join(out) + ("<|assistant|>" if add_generation_prompt else "")


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def embed(self, prompts, use_tqdm=False):
        self.prompts = prompts
        return [types.SimpleNamespace(outputs=types.SimpleNamespace(embedding=[float(len(p["prompt"]))] * 4))
                for p in prompts]


def test_offline_backend_renders_template_and_passes_images(tmp_path):
    (tmp_path / "f.png").write_bytes(PNG)
    pytest.importorskip("PIL")
    import PIL.Image
    PIL.Image.new("RGB", (8, 8)).save(tmp_path / "f.png")
    be = embed_utils.OfflineBackend("fake", 64)
    be._tok, be._llm = FakeTokenizer(), FakeLLM()
    vecs = be.embed_chat([embed_utils.chat_messages("look", [str(tmp_path / "f.png")], "sys"),
                          embed_utils.chat_messages("plain")])
    assert vecs.shape == (2, 4)
    p0, p1 = be._llm.prompts
    assert p0["prompt"] == "<|system|>sys<|user|><image>look<|assistant|>"
    assert p0["multi_modal_data"]["image"].size == (8, 8)       # one PIL image, in order
    assert p1 == {"prompt": "<|user|>plain<|assistant|>"}


class FakeRecipe:
    def __init__(self, *a, **k):
        pass

    def text_ids(self, text, keep="head"):
        return [ord(c) for c in text]

    def state_ids(self, state):
        return [1, 2, 3]


def test_clm_embedding_dir_with_images(tmp_path, server, monkeypatch):
    monkeypatch.setattr(embed_utils, "Recipe", FakeRecipe)
    monkeypatch.setattr(Handler, "requests", [])
    (tmp_path / "s0.png").write_bytes(PNG)
    steps = [{"state": "frame 0", "action": "jump", "task_id": "t1", "step_idx": 0, "images": ["s0.png"]},
             {"state": "frame 1", "action": "duck", "task_id": "t1", "step_idx": 1}]
    data = tmp_path / "steps.json"
    data.write_text(json.dumps(steps))
    args = types.SimpleNamespace(emb_dir=None, hf_dataset=None, data=str(data), embed_cache=None,
                                 out_dir=str(tmp_path / "run"), embed_model="Qwen/Qwen3-VL-8B-Instruct-FP8",
                                 max_len=4096, embed_url=server, gpu_mem=0.8, served_model_name="qwen3-vl-8b",
                                 state_chat=False, embed_instruction="Represent the user's input.", convert=None,
                                 hf_cache=None, hf_split=None)
    out = finetune.clm_embedding_dir(args)
    meta = json.load(open(os.path.join(out, "metadata.json")))
    assert meta["state_chat"] is True and meta["images"] == 1 and meta["hidden_size"] == 8
    assert meta["instruction"] == "Represent the user's input."
    se = torch.load(os.path.join(out, "state_embeddings.pt"))
    ae = torch.load(os.path.join(out, "action_embeddings.pt"))
    assert se.shape == (2, 8) and ae.shape == (2, 8)
    chat = [r for r in Handler.requests if "messages" in r]
    convs = [conv for r in chat for conv in r["messages"]]
    assert len(convs) == 2                                    # every state through the chat path, one request
    with_img = next(conv for conv in convs if len(conv[-1]["content"]) == 2)
    url = with_img[-1]["content"][0]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(PNG).decode()   # the local file was inlined
    assert all(conv[0]["role"] == "system" for conv in convs)
    plain = [r for r in Handler.requests if "input" in r]
    assert len(plain) == 1 and plain[0]["input"] == [[ord(c) for c in "jump"], [ord(c) for c in "duck"]]
