#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the MMMU-Pro rung with lm-eval against the local server, or print its command.
#
# Usage: run_vision_eval.sh (--print-only | --run) [--client-venv DIR] [--base-url URL]
#          [--out DIR] [--reference-runs DIR] [--hf-home DIR] [--files DIR]
#
# The task is MMMU-Pro "standard (10 options)" over the 300 drawn items, with the
# registered model args (images at most 7 per prompt), generation settings and seed.
# Before anything else the two task files and the item file under --files (default:
# this folder) must match their sha256 values, or the script exits 1 and prints nothing.
# After the run, check_samples.py compares the sample logs with the reference run's;
# its exit 2 makes the run invalid. --print-only prints the serve flag line, the command
# and its per-item check, one tab-separated "label<TAB>command" line each.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FILES=$HERE
CLIENT_VENV=${CLIENT_VENV:-}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
OUT=${OUT:-}
REFERENCE_RUNS=${REFERENCE_RUNS:-}
HF_HOME_DIR=${HF_HOME:-}
MODE=

usage() { sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --print-only) MODE=print; shift ;;
    --run) MODE=run; shift ;;
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

TASK=mmmu_pro_std10
TASK_DIR=tasks/mmmu_pro
REFERENCE_FILES=("$TASK_DIR/mmmu_pro_std10.yaml" "$TASK_DIR/utils.py" mmmu_pro_300.json)
REFERENCE_SHA256=(
  734e89e67eb3a0d536109afc8bf6d56703cf24ed7ba8dcee8cf89484d24d1fe9
  726ab449faade78602fd104d3502711d61cfa6a911fca509de0daf4bedb3f1ae
  fef4887f942a048e837382a409d76c9cf17153d93503bd59dfc1cab94ea598c6
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
    mmmu_pro_std10.yaml | utils.py | __pycache__) ;;
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
MA="model=$NAME,base_url=$BASE_URL/v1/chat/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,timeout=7200,max_length=16384,max_retries=10,max_images=7"
GK="max_gen_toks=2048,temperature=0,do_sample=false,reasoning_effort=none"
SEED=0,1234,1234,1234
SERVE=(--gpu-memory-utilization 0.90 --max-model-len 131072 --reasoning-parser kimi_k3
       --no-enable-prefix-caching --served-model-name "$NAME" --limit-mm-per-prompt
       '{"image": 7}')
CMD=(env "HF_HOME=${HF_HOME_DIR:-<offline cache>}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
     OPENAI_API_KEY=EMPTY "$PY" -m lm_eval --model local-chat-completions --tasks "$TASK"
     --model_args "$MA" --log_samples --output_path "$OUT_DIR/$TASK" --apply_chat_template
     --fewshot_as_multiturn --gen_kwargs "$GK" --include_path "$FILES/$TASK_DIR"
     --seed "$SEED" --samples "$FILES/mmmu_pro_300.json")
CHECK=("$PY" "$HERE/../check_samples.py" --candidate "$OUT_DIR/$TASK" --reference "$REF_DIR/out")

line() { printf '%s\t' "$1"; shift; printf '%q ' "$@"; printf '\n'; }

if [[ $MODE == print ]]; then
  line serve "${SERVE[@]}"
  line "$TASK" "${CMD[@]}"
  line "$TASK:check" "${CHECK[@]}"
  exit 0
fi

mkdir -p "$OUT"
line "$TASK" "${CMD[@]}" | tee "$OUT/$TASK.command.txt"
set +e
"${CMD[@]}" 2>&1 | tee "$OUT/$TASK.log"
rc=${PIPESTATUS[0]}
set -e
[[ $rc -eq 0 ]] || { echo "error: lm_eval exited $rc" >&2; exit 1; }
set +e
"${CHECK[@]}" | tee "$OUT/$TASK.check.json"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -eq 2 ]]; then echo "error: $TASK: the run is invalid (per-item check)" >&2; exit 2; fi
[[ $rc -eq 0 ]] || { echo "error: check_samples.py exited $rc" >&2; exit 1; }
