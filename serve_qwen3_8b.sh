#!/bin/bash
# Serve ONE Qwen3-8B pooling server as the encoder behind `clm-serve`.
# Lean settings so it coexists with other GPU work: enforce-eager, modest util,
# short max-model-len (System One states are short). LAST-token pooling + prefix cache,
# same as the precompute, so the embeddings match what the head was trained on.
#
# Usage: GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh
set -u
MODEL="${MODEL:-Qwen/Qwen3-8B}"          # Qwen/Qwen3-8B-FP8 fits a 16 GB GPU (bf16 needs ~17 GB)
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
UTIL="${UTIL:-0.35}"
MAXLEN="${MAXLEN:-2048}"
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
mkdir -p "$LOGDIR"
echo "serving $MODEL pooling on GPU $GPU port $PORT (util $UTIL)"
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$MODEL" \
    --served-model-name qwen3-8b \
    --runner pooling \
    --enforce-eager \
    --enable-prefix-caching \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs 32 \
    --port "$PORT" \
    >> "$LOGDIR/vllm_demo_8b.log" 2>&1
