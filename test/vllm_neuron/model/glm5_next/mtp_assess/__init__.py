"""MTP (multi-token prediction) assessment experiments for GLM-5.3-Flash. CPU only.

Nothing here is a test the suite collects: the files are scripts the assessment report
cites. Run them with the CPU-mode environment the tiny tests use::

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=<worktree> \
        <venv>/bin/python -m test.vllm_neuron.model.glm5_next.mtp_assess.probe_spec_config
"""
