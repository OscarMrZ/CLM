# finetuning

This is an experiment to have the LLM autonomously improve CLM fine-tuning.

## Setup

Work with the user to:

1. Agree on `clm` or `choice`, a run tag, and a fixed training and evaluation
   command.
2. Create a fresh `autofinetune/<tag>` branch.
3. Read `README.md`, `train/finetune.py`, `train/adapters.py`,
   `train/embed_utils.py`, `preprocessing/hf_embeddings.py`, and
   `evaluation/bon_eval.py`.
4. Verify the dataset, checkpoint, folds, and GPU are available.
5. Create an untracked `results.tsv` containing:

   ```text
   commit	score	status	description
   ```

6. Confirm the setup with the user, then run the unchanged baseline.

## Experimentation

Run training with the agreed command and redirect output:

```bash
python train/finetune.py ... --out-dir runs/<tag>/<commit> > run.log 2>&1
```

For `clm`, run the agreed held-out evaluation afterward:

```bash
python evaluation/bon_eval.py ... --n <N> --window <K> >> run.log 2>&1
```

**You may:**

- Modify `train/finetune.py`.
- Change the head, objective, optimizer, scheduler, batching, and
  hyperparameters.

**You may not:**

- Modify any other file.
- Change the data, embeddings, splits, folds, evaluation set, `--n`, or
  `--window`.
- Install dependencies.
- Train on evaluation data or tune on the `choice` test split.

The goal is to maximize the held-out best-of-N rate for `clm` or validation
accuracy for `choice`. Prefer simpler changes when scores are equal.

The first run must always establish the unmodified baseline.

## Logging results

Append one tab-separated row after every run:

```text
commit	score	status	description
a1b2c3d	0.796500	keep	baseline
b2c3d4e	0.805300	keep	task-blocked batches
c3d4e5f	0.787600	discard	constant learning rate
d4e5f6a	0.000000	crash	double width (OOM)
```

Do not commit `results.tsv`, `run.log`, or run outputs.

## The experiment loop

LOOP FOREVER:

1. Inspect the current commit and previous results.
2. Make one focused change to `train/finetune.py`.
3. Commit it.
4. Run training and evaluation.
5. Record the result.
6. Keep the commit if the score improved.
7. Otherwise, reset to the last kept commit.
8. Continue with another idea.

If a run crashes, inspect `tail -n 80 run.log`. Fix simple bugs and rerun;
otherwise record the crash and reset. If it exceeds the agreed timeout, stop it,
record a timeout, and reset.

Once the loop begins, do not ask whether to continue. Keep experimenting until
the user interrupts you.

## Steps with images

`--data steps.jsonl` may carry an `images` list per step (paths relative to the
file, URLs or `data:` URIs). Such steps are embedded through the encoder's chat
path with the images attached to the state, so the encoder must be multimodal:

```bash
./serve_qwen3_vl_8b.sh                                     # Qwen3-VL-8B-Instruct-FP8, --convert embed
python train/finetune.py --task clm --data data/screens/steps.jsonl --out-dir runs/screens \
    --embed-url http://127.0.0.1:8090/v1/embeddings --embed-model Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --served-model-name qwen3-vl-8b --max-len 4096 --batch 256
clm-serve --emb-model qwen3-vl-8b --state-chat --ckpt runs/screens/best_head.pt
```

`--state-chat` puts text-only states through the same template (implied when any
step has images), `--embed-instruction` sets the system prompt (Qwen3-VL-Embedding
wants `"Represent the user's input."`), and the checkpoint `cfg` records
`model_name`, `state_chat` and `instruction` so the head is served the way it was
trained. The same holds for `--task choice`: a typed-decisions row whose `state` is
a content-part list (text + `image_url`) is embedded with its image.

The released head cannot be warm-started on a different encoder: its 4096-d input
space is Qwen3-8B's. Heads for a multimodal encoder start from scratch, or from a
head trained on general image data with the same encoder.
