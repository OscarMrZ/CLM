#!/bin/bash
# Serve a Qwen3-VL pooling server as a *multimodal* encoder behind `clm-serve --state-chat`.
#
# Two encoders make sense here; pick one with MODEL=
#   Qwen/Qwen3-VL-8B-Instruct-FP8   (default) the frozen VLM with last-token pooling, the CLM recipe
#                                   applied to a vision-language backbone (needs --convert embed)
#   Qwen/Qwen3-VL-Embedding-8B      retrieval-tuned; pass CLM_EMB_INSTRUCTION="Represent the user's input."
#                                   to clm-serve (native pooling model, no --convert)
# Dev-size: Qwen/Qwen3-VL-Embedding-2B (bf16, hidden 2048) or Qwen/Qwen3-VL-4B-Instruct (hidden 2560).
#
# Sized for one 16 GB GPU: FP8 weights ~9 GB + the 0.6B ViT, max-model-len 4096
# (a 1280x720 screenshot is ~900 visual tokens), at most 4 images per state. Text actions
# go through the same server as plain `input` texts.
#
# Usage: GPU=0 PORT=8090 UTIL=0.80 ./serve_qwen3_vl_8b.sh
set -u
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
NAME="${NAME:-qwen3-vl-8b}"
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
UTIL="${UTIL:-0.80}"
MAXLEN="${MAXLEN:-4096}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"        # ~980x1024: caps visual tokens per image (~1k)
IMAGES="${IMAGES:-4}"
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")" && pwd)/logs}"
mkdir -p "$LOGDIR"
CONVERT=()
case "$MODEL" in *Embedding*) ;; *) CONVERT=(--convert embed) ;; esac
echo "serving $MODEL pooling as $NAME on GPU $GPU port $PORT (util $UTIL, max-len $MAXLEN)"
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --runner pooling "${CONVERT[@]}" \
    --enable-prefix-caching \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --limit-mm-per-prompt "{\"image\": $IMAGES}" \
    --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
    --max-num-seqs 16 \
    --port "$PORT" \
    >> "$LOGDIR/vllm_qwen3_vl.log" 2>&1
