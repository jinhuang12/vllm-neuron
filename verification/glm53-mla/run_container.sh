#!/usr/bin/env bash
# Build a small OS image and reuse the installed Neuron tools without changing them.
set -euo pipefail

MLA_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MLA_SOURCE=${MLA_SOURCE:-$(cd -- "$MLA_SCRIPT_DIR/../.." && pwd)}
MLA_OUTPUT=${MLA_OUTPUT:-$(dirname -- "$MLA_SOURCE")/verification}
MLA_VENV=${MLA_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
MLA_SDK=${MLA_SDK:-/opt/aws/neuron}
MLA_IMAGE=${MLA_IMAGE:-sha256:8b6d8ccd82c10303c7d2c85d76af025d16142912d8bac6642a10dadf8c776d5c}
MLA_MODE=${1:-smoke}
if (($#)); then shift; fi

case "$MLA_MODE" in
  smoke|cpu-tests|check|run) ;;
  *)
    printf 'Usage: %s {smoke|cpu-tests|check|run} [--] [command ...]\n' "$0" >&2
    exit 2
    ;;
esac

test -x "$MLA_VENV/bin/python"
test -d "$MLA_SDK"
mkdir -p "$MLA_OUTPUT/logs" "$MLA_OUTPUT/cache"
MLA_DOCKER_ARGS=(
  run --rm --read-only --network none --shm-size=1g
  --user "$(id -u):$(id -g)"
  --tmpfs /tmp:rw,exec,mode=1777
  --mount "type=bind,src=$MLA_VENV,dst=$MLA_VENV,readonly"
  --mount "type=bind,src=$MLA_SDK,dst=$MLA_SDK,readonly"
  --mount "type=bind,src=$MLA_SOURCE,dst=$MLA_SOURCE,readonly"
  --mount "type=bind,src=$MLA_OUTPUT,dst=$MLA_OUTPUT"
  --workdir "${MLA_WORKDIR:-$MLA_SOURCE}"
  --env "PATH=$MLA_VENV/bin:$MLA_SDK/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env "VIRTUAL_ENV=$MLA_VENV"
  --env "NEURON_LOGICAL_NC_CONFIG=2"
  --env "NEURON_PLATFORM_TARGET_OVERRIDE=${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  --env "LD_LIBRARY_PATH=$MLA_SDK/lib:$MLA_VENV/lib"
  --env "PYTHONPATH=$MLA_SOURCE"
  --env "XDG_CACHE_HOME=$MLA_OUTPUT/cache"
  --env "VLLM_CACHE_ROOT=$MLA_OUTPUT/cache/vllm"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "MLA_OUTPUT=$MLA_OUTPUT"
)

if [[ -n "${MLA_INPUT:-}" ]]; then
  test -d "$MLA_INPUT"
  MLA_DOCKER_ARGS+=(--mount "type=bind,src=$MLA_INPUT,dst=$MLA_INPUT,readonly")
fi

if [[ -n "${MLA_DEPS:-}" ]]; then
  test -d "$MLA_DEPS"
  MLA_DOCKER_ARGS+=(--mount "type=bind,src=$MLA_DEPS,dst=$MLA_DEPS,readonly"
                   --env "PYTHONPATH=$MLA_DEPS:$MLA_SOURCE")
fi

if [[ "$MLA_MODE" == run ]]; then
  # The lead must assign the device and logical core. No hardware is exposed
  # by the smoke and CPU-test modes.
  : "${MLA_NEURON_DEVICE:?Set the lead-assigned device, for example /dev/neuron0}"
  : "${NEURON_RT_VISIBLE_CORES:?Set the lead-assigned logical core}"
  [[ "$MLA_NEURON_DEVICE" =~ ^/dev/neuron[0-9]+$ ]] || {
    printf 'Expected one /dev/neuronN device, got %s\n' "$MLA_NEURON_DEVICE" >&2
    exit 2
  }
  test -c "$MLA_NEURON_DEVICE"
  MLA_DOCKER_ARGS+=(--device "$MLA_NEURON_DEVICE" --env "NEURON_RT_VISIBLE_CORES=$NEURON_RT_VISIBLE_CORES")
fi

if [[ "$MLA_MODE" == run || "$MLA_MODE" == check ]]; then
  [[ "${1:-}" != -- ]] || shift
  (($#)) || { printf 'The run mode needs a verification command.\n' >&2; exit 2; }
  exec docker "${MLA_DOCKER_ARGS[@]}" "$MLA_IMAGE" "$@"
elif [[ "$MLA_MODE" == cpu-tests ]]; then
  exec docker "${MLA_DOCKER_ARGS[@]}" "$MLA_IMAGE" python -m pytest -p no:cacheprovider -q \
    benchmarks/glm53_mla/test_reference.py benchmarks/glm53_mla/test_runtime.py \
    benchmarks/glm53_mla/test_run.py
else
  exec docker "${MLA_DOCKER_ARGS[@]}" "$MLA_IMAGE" python -c '
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import torch
import nki
import nrtpy
import libtorch_neuronx_lite
from benchmarks.glm53_mla.reference import make_fixture, sparse_reference
torch.set_num_threads(1)
case = make_fixture(seq=3, heads=2, latent=128, cache_rows=256, topk=128, kind="sentinel")
out = sparse_reference(case)
assert out.shape == (3, 2, 128) and out.dtype == torch.float32
assert torch.isfinite(out).all()
assert not list(pathlib.Path("/dev").glob("neuron*"))
compiler = pathlib.Path(sys.prefix) / "lib/python3.12/site-packages/neuronxcc/starfish/bin/walrus_driver"
libraries = subprocess.run(["ldd", str(compiler)], text=True, capture_output=True, check=True).stdout
assert "not found" not in libraries, libraries
print(json.dumps({"status": "PASS", "tier": "CPU import smoke", "python": sys.version,
    "versions": {name: importlib.metadata.version(name) for name in ("torch", "nki", "libtorch-neuronx-lite")},
    "devices_exposed": False, "compiler_libraries_resolved": True,
    "reference_shape": list(out.shape)}, indent=2))
'
fi
