"""Data adapters for ``train/finetune.py``.

* ``read_transitions``: step transitions (``state`` chat messages, ``action``,
  ``task_id``, ``step_idx``, ``trajectory_id``, ``reward``) as written by
  the DeepSWE trace builder in the research repo's ``main`` branch, returned unchanged.
  A step may also carry ``images``: a list of image paths (relative to the
  transitions file) or URLs that belong to its state -- screenshots, camera
  frames -- embedded alongside the text by a multimodal encoder.
* ``typed_decision_examples``: System One typed questions
  (``LocalLLaMA/typed-decisions``); each (row, question) becomes a state text, option
  keys, candidate texts (``clm.schema.build_pairs``) and the gold distribution.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.content import Content  # noqa: E402
from clm.schema import build_pairs  # noqa: E402

TRANSITION_KEYS = ("state", "action", "task_id", "step_idx")


def read_transitions(path: str) -> list[dict]:
    """Load step transitions from a ``.jsonl`` (one object per line) or ``.json`` (array) file.

    Relative image paths in ``images`` are resolved against the file's directory."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            recs = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for i, r in enumerate(recs):
        missing = [k for k in TRANSITION_KEYS if k not in r]
        if missing:
            raise ValueError(f"{path}: record {i} is missing {missing}")
        images = r.get("images")
        if images is None:
            continue
        if not isinstance(images, list) or not all(isinstance(x, str) and x for x in images):
            raise ValueError(f"{path}: record {i}: images must be a list of paths or URLs")
        r["images"] = [x if "://" in x or x.startswith("data:") or os.path.isabs(x) else os.path.join(base, x)
                       for x in images]
    return recs


def has_images(recs: list[dict]) -> bool:
    return any(r.get("images") for r in recs)


def _maybe_json(x):
    """Decode JSON-encoded objects; other strings are returned as is."""
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except ValueError:
            return x
        return v if isinstance(v, (dict, list)) else x
    return x


def _key_of(label) -> str:
    if isinstance(label, bool):
        return "true" if label else "false"
    return str(label)


@dataclass
class ChoiceExample:
    qid: str
    state_text: str | Content     # a Content when the state carries images
    keys: list[str]
    candidates: list[str]
    target: list[float]      # gold distribution over ``keys``
    label: int               # index of the gold label in ``keys``
    group: str               # source row id
    workflow: str | None = None
    meta: dict = field(default_factory=dict)


def typed_decision_examples(rows: Iterable[dict]) -> Iterator[ChoiceExample]:
    """Rows with ``state``, ``questions`` (System One wire format) and ``gold``
    (``{qid: {"label", "probabilities"}}``) -> one ChoiceExample per answered question."""
    for r in rows:
        state, questions, gold = _maybe_json(r["state"]), _maybe_json(r["questions"]), _maybe_json(r["gold"])
        for qid, (stext, keys, cands) in build_pairs(state, questions).items():
            g = gold.get(qid) if isinstance(gold, dict) else None
            if not g or "label" not in g:
                continue
            label = _key_of(g["label"])
            if label not in keys:
                raise ValueError(f"row {r.get('id')}: gold label {label!r} not among options {keys} for {qid!r}")
            probs = {_key_of(k): float(v) for k, v in (g.get("probabilities") or {}).items()}
            target = [probs.get(k, 0.0) for k in keys]
            z = sum(target)
            target = [t / z for t in target] if z > 0 else [float(k == label) for k in keys]
            yield ChoiceExample(qid=qid, state_text=stext, keys=keys, candidates=cands, target=target,
                                label=keys.index(label), group=str(r.get("id", "")),
                                workflow=r.get("workflow"))
