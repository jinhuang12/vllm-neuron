# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the GLM-5.3-Flash tool-call, reasoning and response-shaping suite.

The suite exercises vLLM's own parsers and OpenAI serving code against the checkpoint's
own tokenizer and chat template, so it needs the checkpoint's tokenizer files. They are
too large to commit (``tokenizer.json`` is about 20 MB), so the suite reads them from the
directory named by ``VLLM_NEURON_TEST_GLM53F_CHECKPOINT`` and skips, naming the variable,
when it is unset. Only the files in :data:`CHECKPOINT_FILES` are read; the weights are
never opened. They are linked into a temporary directory, which is then the model path
every fixture uses, so nothing is written next to the checkpoint.

    VLLM_NEURON_TEST_GLM53F_CHECKPOINT=<checkpoint dir> PYTHONDONTWRITEBYTECODE=1 \\
        python -I -B -m pytest test/parity/p13 -q
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

import pytest

#: Names the checkpoint directory (the one holding ``tokenizer.json``). Test-only: it is
#: read here and nowhere in ``vllm_neuron``.
CHECKPOINT_ENV = "VLLM_NEURON_TEST_GLM53F_CHECKPOINT"

#: What the suite reads from the checkpoint. ``config.json`` and ``processor_config.json``
#: let vLLM build the served model config and renderer; the rest are the tokenizer, its
#: chat template and the generation config whose EOS ids end a tool-call turn.
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

#: The tiny GLM-5.3-Flash root that ``test/vllm_neuron/vllm/test_admission_http.py``
#: imports reads these at import time (``test/vllm_neuron/vllm/conftest.py`` pins them
#: for its own directory only). A value the caller set is kept.
SIMULATOR_ENV = {"NKI_SIMULATOR": "1", "NKI_PRECISE_FP": "1"}

#: The parsers ``vllm serve`` builds for this checkpoint when it is started with
#: ``--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45``.
TOOL_PARSER = "glm47"
REASONING_PARSER = "glm45"


def pytest_configure(config) -> None:
    for name, value in SIMULATOR_ENV.items():
        os.environ.setdefault(name, value)


@pytest.fixture(scope="session")
def checkpoint_dir() -> pathlib.Path:
    """A temporary model directory linking the checkpoint's tokenizer and config files."""
    source = os.environ.get(CHECKPOINT_ENV)
    if not source:
        pytest.skip(f"{CHECKPOINT_ENV} is unset; point it at the GLM-5.3-Flash checkpoint")
    source_dir = pathlib.Path(source)
    missing = [name for name in CHECKPOINT_FILES if not (source_dir / name).is_file()]
    if missing:
        pytest.skip(f"{CHECKPOINT_ENV}={source} lacks {', '.join(missing)}")
    root = pathlib.Path(tempfile.mkdtemp(prefix="glm53f-tool-parsing-"))
    for name in CHECKPOINT_FILES:
        (root / name).symlink_to(source_dir / name)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def tokenizer(checkpoint_dir):
    """The tokenizer vLLM's renderer builds for the checkpoint (``vllm.tokenizers``)."""
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(str(checkpoint_dir))


@pytest.fixture(scope="session")
def parser_cls(checkpoint_dir):
    """The unified parser class vLLM's chat server builds from the two parser flags."""
    from vllm.parser import ParserManager

    return ParserManager.get_parser(
        tool_parser_name=TOOL_PARSER,
        reasoning_parser_name=REASONING_PARSER,
        enable_auto_tools=True,
        model_name=str(checkpoint_dir),
    )
