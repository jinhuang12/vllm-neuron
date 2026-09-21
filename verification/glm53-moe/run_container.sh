#!/usr/bin/env bash
# Build a small OS image and reuse the installed Neuron tools without changing them.
set -euo pipefail

GLM53_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GLM53_SOURCE=${GLM53_SOURCE:-$(cd -- "$GLM53_SCRIPT_DIR/../.." && pwd)}
GLM53_OUTPUT=${GLM53_OUTPUT:-$(dirname -- "$GLM53_SOURCE")/verification}
GLM53_VENV=${GLM53_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
GLM53_SDK=${GLM53_SDK:-/opt/aws/neuron}
GLM53_IMAGE=${GLM53_IMAGE:-glm53-moe-verification:ubuntu24.04}
GLM53_MODE=${1:-smoke}
if (($#)); then shift; fi

case "$GLM53_MODE" in
  build)
    exec docker build --pull --tag "$GLM53_IMAGE" --file "$GLM53_SCRIPT_DIR/Dockerfile" "$GLM53_SCRIPT_DIR"
    ;;
  smoke|cpu-tests|run) ;;
  *)
    printf 'Usage: %s {build|smoke|cpu-tests|run [--] command ...}\n' "$0" >&2
    exit 2
    ;;
esac

test -x "$GLM53_VENV/bin/python"
test -d "$GLM53_SDK"
mkdir -p "$GLM53_OUTPUT/logs" "$GLM53_OUTPUT/cache"
GLM53_DOCKER_ARGS=(
  run --rm --read-only --network none --shm-size=1g
  --user "$(id -u):$(id -g)"
  --tmpfs /tmp:rw,exec,mode=1777
  --mount "type=bind,src=$GLM53_VENV,dst=$GLM53_VENV,readonly"
  --mount "type=bind,src=$GLM53_SDK,dst=$GLM53_SDK,readonly"
  --mount "type=bind,src=$GLM53_SOURCE,dst=$GLM53_SOURCE,readonly"
  --mount "type=bind,src=$GLM53_OUTPUT,dst=$GLM53_OUTPUT"
  --workdir "$GLM53_SOURCE"
  --env "PATH=$GLM53_VENV/bin:$GLM53_SDK/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env "VIRTUAL_ENV=$GLM53_VENV"
  --env "NEURON_LOGICAL_NC_CONFIG=2"
  --env "NEURON_PLATFORM_TARGET_OVERRIDE=${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  --env "LD_LIBRARY_PATH=$GLM53_SDK/lib:$GLM53_VENV/lib"
  --env "PYTHONPATH=$GLM53_SOURCE"
  --env "XDG_CACHE_HOME=$GLM53_OUTPUT/cache"
  --env "VLLM_CACHE_ROOT=$GLM53_OUTPUT/cache/vllm"
  # NKI cache entries can contain paths under /tmp. Keep the entry and its
  # referenced files in the same container lifetime.
  --env "NEURON_LIBTORCH_CACHE_ROOT=/tmp/neuron-compile-cache"
  --env "GLM53_OUTPUT=$GLM53_OUTPUT"
)

if [[ "$GLM53_MODE" == run ]]; then
  # The lead must assign the device and logical core. No hardware is exposed
  # by the smoke and CPU-test modes.
  : "${GLM53_NEURON_DEVICE:?Set the lead-assigned device, for example /dev/neuron0}"
  : "${NEURON_RT_VISIBLE_CORES:?Set the lead-assigned logical core}"
  [[ "$GLM53_NEURON_DEVICE" =~ ^/dev/neuron[0-9]+$ ]] || {
    printf 'Expected one /dev/neuronN device, got %s\n' "$GLM53_NEURON_DEVICE" >&2
    exit 2
  }
  test -c "$GLM53_NEURON_DEVICE"
  GLM53_DOCKER_ARGS+=(--device "$GLM53_NEURON_DEVICE" --env "NEURON_RT_VISIBLE_CORES=$NEURON_RT_VISIBLE_CORES")
  [[ "${1:-}" != -- ]] || shift
  (($#)) || { printf 'The run mode needs a verification command.\n' >&2; exit 2; }
  exec docker "${GLM53_DOCKER_ARGS[@]}" "$GLM53_IMAGE" "$@"
elif [[ "$GLM53_MODE" == cpu-tests ]]; then
  exec docker "${GLM53_DOCKER_ARGS[@]}" --env VLLM_NEURON_CPU_MODE=1 \
    "$GLM53_IMAGE" python -m pytest -p no:cacheprovider -q \
    benchmarks/glm53_moe/test_reference.py experiments/glm53_moe_nki/test_pack.py \
    experiments/glm53_moe_nki/test_generic_pack.py \
    experiments/glm53_moe_nki/test_model_integration.py \
    test/vllm_neuron/model/glm5_next/test_prepared_weight_release.py \
    test/vllm_neuron/model/glm5_next/test_weights_free_load_path.py \
    test/vllm_neuron/model/glm5_next/test_load_weights.py::test_blocked_the_shared_expert_prep_completes_a_load_and_the_publish_ran
else
  exec docker "${GLM53_DOCKER_ARGS[@]}" "$GLM53_IMAGE" python -c '
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import torch
import nki
import nrtpy
import libtorch_neuronx_lite
from benchmarks.glm53_moe.reference import make_fixture, routed_reference
torch.set_num_threads(1)
case = make_fixture(q=2, hidden=128, intermediate=128, experts=2)
out = routed_reference(case)[-1]
assert out.shape == (512, 128) and out.dtype == torch.float32
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
