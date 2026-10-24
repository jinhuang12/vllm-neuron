#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the scored rungs with lm-eval against the local server, or print their commands.
#
# Usage: run_lm_eval.sh (--print-only | --rung 2|3) [--client-venv DIR] [--base-url URL]
#          [--out DIR] [--reference-runs DIR] [--hf-home DIR] [--files DIR]
#
# Rung 2 is GSM8K 200; rung 3 is GSM8K full, MMLU-Pro 500 and HumanEval+ 164. The
# arguments follow the order of the fork's accuracy runner (run_lm_eval), with the
# registered model args, generation settings and seed. Before the first request the
# three reference files under --files (default: this folder) must match their sha256
# values, or the script exits 1 and prints no command. After each task, check_samples.py
# compares its sample logs with the reference run's; its exit 2 makes the run invalid.
# --print-only prints the serve flag line and all four commands, each followed by its
# per-item check, one tab-separated "label<TAB>command" line each.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FILES=$HERE
CLIENT_VENV=${CLIENT_VENV:-}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
OUT=${OUT:-}
REFERENCE_RUNS=${REFERENCE_RUNS:-}
HF_HOME_DIR=${HF_HOME:-}
MODE=
RUNG=

usage() { sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --print-only) MODE=print; shift ;;
    --rung) MODE=run; RUNG=${2:-}; shift 2 ;;
    --client-venv) CLIENT_VENV=$2; shift 2 ;;
    --base-url) BASE_URL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --reference-runs) REFERENCE_RUNS=$2; shift 2 ;;
    --hf-home) HF_HOME_DIR=$2; shift 2 ;;
    --files) FILES=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n $MODE ]] || usage
[[ $MODE == print || $RUNG == 2 || $RUNG == 3 ]] || usage

TASK_DIR=tasks/humaneval_plus_chat
REFERENCE_FILES=(mmlu_pro_500.json "$TASK_DIR/humaneval_plus_chat.yaml" "$TASK_DIR/utils.py")
REFERENCE_SHA256=(
  199075dab36703c3fb0fb71020e12da85157f464fdf1921b0227de3c4a4e03a2
  4265e34808fac8515ef6206bc7ce18e8192bdd1fc222f7be97aa4de10a02a218
  d230583fcd8f8838bb5d9e5076859e6804920c0b921e58f5e71ab844be518b1d
)
for i in "${!REFERENCE_FILES[@]}"; do
  f=$FILES/${REFERENCE_FILES[$i]}
  got=missing
  [[ -f $f ]] && got=$(sha256sum "$f" | cut -d' ' -f1)
  if [[ $got != "${REFERENCE_SHA256[$i]}" ]]; then
    echo "error: ${REFERENCE_FILES[$i]}: sha256 $got, registered ${REFERENCE_SHA256[$i]}" >&2
    exit 1
  fi
done
for f in "$FILES/$TASK_DIR"/*; do
  case ${f##*/} in
    humaneval_plus_chat.yaml | utils.py | __pycache__) ;;
    *) echo "error: $TASK_DIR holds another file: ${f##*/}" >&2; exit 1 ;;
  esac
done

if [[ $MODE == run ]]; then
  for v in CLIENT_VENV OUT REFERENCE_RUNS HF_HOME_DIR; do
    [[ -n ${!v} ]] || { echo "error: $v is not set" >&2; exit 1; }
  done
fi

NAME=us.moonshotai.kimi-k3
PY=${CLIENT_VENV:-<client venv>}/bin/python
OUT_DIR=${OUT:-<output root>}
REF_DIR=${REFERENCE_RUNS:-<reference run folder>}
MA="model=$NAME,base_url=$BASE_URL/v1/chat/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,timeout=7200,max_length=16384,max_retries=10"
GK="max_gen_toks=2048,temperature=0,do_sample=false,reasoning_effort=none"
SEED=0,1234,1234,1234
ENV=(env "HF_HOME=${HF_HOME_DIR:-<offline cache>}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 OPENAI_API_KEY=EMPTY)
SERVE=(--reasoning-parser kimi_k3 --max-model-len 16384 --served-model-name "$NAME"
       --no-enable-prefix-caching)
SERVE_ABSENT=(--chat-template --default-chat-template-kwargs --generation-config)

# bench_cmd NAME: set CMD to the lm-eval command and CHECK to its per-item check.
bench_cmd() {
  local name=$1 task limit=() extra=() env=("${ENV[@]}") ref
  case $name in
    gsm8k_cot-200) task=gsm8k_cot; limit=(--limit 200); ref=gsm8k_cot ;;
    gsm8k_cot) task=gsm8k_cot; ref=gsm8k_cot ;;
    mmlu_pro) task=mmlu_pro; extra=(--samples "$FILES/mmlu_pro_500.json"); ref=mmlu_pro ;;
    humaneval_plus_chat)
      task=humaneval_plus_chat; env+=(HF_ALLOW_CODE_EVAL=1); ref=humaneval_plus_chat
      extra=(--include_path "$FILES/$TASK_DIR" --confirm_run_unsafe_code) ;;
  esac
  CMD=("${env[@]}" "$PY" -m lm_eval --model local-chat-completions --tasks "$task"
       --model_args "$MA" --log_samples --output_path "$OUT_DIR/$name" "${limit[@]}"
       --apply_chat_template --fewshot_as_multiturn --gen_kwargs "$GK" --seed "$SEED"
       "${extra[@]}")
  CHECK=("$PY" "$HERE/check_samples.py" --candidate "$OUT_DIR/$name"
         --reference "$REF_DIR/$ref/out" "${limit[@]}")
}

line() { printf '%s\t' "$1"; shift; printf '%q ' "$@"; printf '\n'; }

if [[ $MODE == print ]]; then
  line serve "${SERVE[@]}"
  line serve-absent "${SERVE_ABSENT[@]}"
  for name in gsm8k_cot-200 gsm8k_cot mmlu_pro humaneval_plus_chat; do
    bench_cmd "$name"
    line "$name" "${CMD[@]}"
    line "$name:check" "${CHECK[@]}"
  done
  exit 0
fi

if [[ $RUNG == 2 ]]; then NAMES=(gsm8k_cot-200); else NAMES=(gsm8k_cot mmlu_pro humaneval_plus_chat); fi
mkdir -p "$OUT"
for name in "${NAMES[@]}"; do
  bench_cmd "$name"
  line "$name" "${CMD[@]}" | tee "$OUT/$name.command.txt"
  set +e
  "${CMD[@]}" 2>&1 | tee "$OUT/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 0 ]] || { echo "error: lm_eval exited $rc on $name" >&2; exit 1; }
  set +e
  "${CHECK[@]}" | tee "$OUT/$name.check.json"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 2 ]]; then echo "error: $name: the run is invalid (per-item check)" >&2; exit 2; fi
  [[ $rc -eq 0 ]] || { echo "error: check_samples.py exited $rc on $name" >&2; exit 1; }
done
