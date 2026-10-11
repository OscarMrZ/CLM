"""The choice task on rows whose state carries an image: one chat request per such state."""
import base64
import json
import os
import threading
import types
from http.server import HTTPServer

import pytest

import embed_utils
from test_embedder import Handler

torch = pytest.importorskip("torch")
import finetune  # noqa: E402
import test_embedder  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"


class FakeRecipe:
    def __init__(self, *a, **k):
        pass

    def text_ids(self, text, keep="head"):
        return [min(ord(c), 255) for c in text][:64]


def rows(tmp_path, with_image):
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}
    out = []
    for i in range(12):
        text = f"scene {i}: the cube is {'left' if i % 2 else 'right'}"
        state = [{"type": "text", "text": text}, img] if with_image else text
        qs = {"turn": {"type": "choice", "instructions": "Which way?", "criteria": {"left": "turn left", "right": "turn right"}},
              "safe": {"type": "noul", "instructions": "Is it safe?"}}
        gold = {"turn": {"type": "choice", "label": "left" if i % 2 else "right"},
                "safe": {"type": "noul", "label": True, "probabilities": {"true": 0.8, "false": 0.2}}}
        out.append({"id": f"r{i}", "workflow": "w", "split": "", "state": json.dumps(state), "questions": json.dumps(qs),
                    "gold": json.dumps(gold), "factors": "{}", "label_agreement": "{}", "n_questions": "2"})
    return out


def write(rows_, d, split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(d, "all"), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_), os.path.join(d, "all", f"{split}-00000-of-00001.parquet"))


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(test_embedder, "DIM", 4096)
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1/embeddings"
    httpd.shutdown()


def test_choice_task_with_image_rows(tmp_path, server, monkeypatch):
    monkeypatch.setattr(embed_utils, "Recipe", FakeRecipe)
    monkeypatch.setattr(Handler, "requests", [])
    data = str(tmp_path / "typed")
    write(rows(tmp_path, True)[:8], data, "train")
    write(rows(tmp_path, True)[8:], data, "test")
    args = types.SimpleNamespace(data=data, workflow="all", hf_cache=None, seed=1, val_frac=0.25,
                                 embed_cache=None, out_dir=str(tmp_path / "run"), embed_model="Qwen/Qwen3-VL-8B-Instruct-FP8",
                                 max_len=4096, embed_url=server, gpu_mem=0.8, served_model_name="qwen3-vl-8b",
                                 state_chat=False, embed_instruction=None, convert=None, init_ckpt=None,
                                 width=64, depth=3, proj=16, epochs=2, batch=4, lr=None, weight_decay=0.0,
                                 patience=3, gpu=0, targets="soft", loss="infonce")
    os.makedirs(args.out_dir, exist_ok=True)
    res = finetune.run_choice(args)
    convs = [conv for r in Handler.requests if "messages" in r for conv in r["messages"]]
    plain = [r for r in Handler.requests if "input" in r]
    # 12 distinct scenes x 2 questions = 24 image states, batched conversations with the image part first
    assert len(convs) == 24
    assert all(conv[-1]["content"][0]["type"] == "image_url" for conv in convs)
    assert all(conv[-1]["content"][1]["type"] == "text" for conv in convs)
    # the 4 candidate texts went as plain token ids in one batched request
    assert len(plain) == 1 and len(plain[0]["input"]) == 4
    assert os.path.exists(os.path.join(args.out_dir, "best_head.pt"))
    assert 0.0 <= res["test"]["acc"] <= 1.0
    ck = torch.load(os.path.join(args.out_dir, "best_head.pt"), weights_only=False)
    assert ck["cfg"]["state_chat"] is False and ck["cfg"]["embed_model"] == "Qwen/Qwen3-VL-8B-Instruct-FP8"


def test_choice_task_text_rows_unchanged(tmp_path, server, monkeypatch):
    monkeypatch.setattr(embed_utils, "Recipe", FakeRecipe)
    monkeypatch.setattr(Handler, "requests", [])
    data = str(tmp_path / "typed")
    write(rows(tmp_path, False)[:8], data, "train")
    write(rows(tmp_path, False)[8:], data, "test")
    args = types.SimpleNamespace(data=data, workflow="all", hf_cache=None, seed=1, val_frac=0.25,
                                 embed_cache=None, out_dir=str(tmp_path / "run"), embed_model="Qwen/Qwen3-8B",
                                 max_len=2048, embed_url=server, gpu_mem=0.8, served_model_name="qwen3-8b",
                                 state_chat=False, embed_instruction=None, convert=None, init_ckpt=None,
                                 width=64, depth=3, proj=16, epochs=1, batch=4, lr=None, weight_decay=0.0,
                                 patience=3, gpu=0, targets="soft", loss="infonce")
    os.makedirs(args.out_dir, exist_ok=True)
    finetune.run_choice(args)
    assert not [r for r in Handler.requests if "messages" in r]        # text-only rows: no chat requests
    n = sum(len(r["input"]) for r in Handler.requests if "input" in r)
    assert n == 24 + 4                                                  # 24 state texts + 4 candidates, one path
