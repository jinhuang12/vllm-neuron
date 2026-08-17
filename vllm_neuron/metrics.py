# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for vLLM-Neuron.

Metrics are defined at module scope so they auto-register with the
``prometheus_client`` global default registry. vLLM serves this registry
at ``/metrics``, so any metric created here appears automatically.

All metrics use the ``vllm_neuron:`` prefix to avoid collisions with
built-in ``vllm:`` metrics.

Usage example::

    from vllm_neuron.metrics import NUM_SEQS_PADDING

    NUM_SEQS_PADDING.labels(
        model_name="meta-llama/Llama-3.1-8B",
        bucket_name="prefill_s128",
    ).observe(4)
"""

from prometheus_client import Counter, Gauge, Histogram

NUM_SEQS_PADDING = Histogram(
    "vllm_neuron:num_seqs_padding",
    "Number of padded batch lines processed by the model.",
    labelnames=["model_name", "bucket_name"],
    buckets=[2**i for i in range(8)],  # [1, 2, 4, 8, 16, 32, 64, 128]
)

NUM_BATCHED_TOKENS_PADDING = Histogram(
    "vllm_neuron:num_batched_tokens_padding",
    "Number of padded sequence length processed by the model.",
    labelnames=["model_name", "bucket_name"],
    buckets=[2**i for i in range(5, 15)],  # [32, 64, 128, ..., 16384]
)

NEFF_EXECUTION_COUNT = Counter(
    "vllm_neuron:neff_execution_count",
    "Number of NEFF executions.",
    labelnames=["model_name", "bucket_name"],
)

COMPILATION_TIME = Gauge(
    "vllm_neuron:compilation_time_seconds",
    "Time spent compiling Neuron graphs (FX trace + neuronxcc compile). Includes compile cache hits.",
    labelnames=["model_name", "bucket_name"],
    multiprocess_mode="max",
)

STARTUP_TIME = Gauge(
    "vllm_neuron:startup_time_seconds",
    "Total server startup time from worker spawn to ready.",
    labelnames=["model_name"],
    multiprocess_mode="max",
)

MODEL_LOAD_TIME = Gauge(
    "vllm_neuron:model_load_time_seconds",
    "Time spent loading model weights to device (host to HBM transfer).",
    labelnames=["model_name"],
    multiprocess_mode="max",
)

MODEL_LOAD_SIZE = Gauge(
    "vllm_neuron:model_load_size_bytes",
    "Size of model weights transferred to device (host to HBM transfer).",
    labelnames=["model_name"],
    multiprocess_mode="sum",
)
