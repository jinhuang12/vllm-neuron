# Prepared clean SDK check

Status: prepared and staged; no Docker or SDK test has run.

The source came from immutable git tree `1f1e90088a4ec65267bca0e8f7efec1302a9b722`. It is staged at `/home/ubuntu/glm53-moe-fullmodel-20260917/pr-snapshot-source`.

The 63-test selection is the prior 14 existing loader/release tests, the exact prior nine tiny tests, and the current 40 model-integration/compact-row tests. The tiny-forward file contributes only its two previously successful numerical tests. The seven-test one-graph file is the same prior selection.

The container has no Neuron devices and no network. Source, dependency overlay, SDK, and virtual environment are read-only. Scratch and caches stay in its temporary filesystem. Preflight checks the exact production tree (including pyproject/setup), all frozen runtime package versions, source import origin, and read-only mounts before pytest starts.

After the lead sends GO, run from any directory:

```sh
python3 /home/jinhun/vllm-neuron-campaigns/glm53-moe-nki-20260917/verification/glm53-moe-fullmodel/clean-sdk-r1/run-prepared.py
```

`docker-command.json` and `docker-command.sh` retain the exact Docker command. The runner creates exclusive `sdk-tests.log` and `result.json`; it does not retry.
