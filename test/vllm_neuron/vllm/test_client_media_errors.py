# SPDX-License-Identifier: Apache-2.0
"""A media URL the client got wrong gets HTTP 400, not 500.

vLLM's ``MediaConnector`` raises ``RuntimeError`` for a ``file://`` URL when the server allows
no local media path, and lets a failed fetch (connection refused, HTTP error status, timeout)
propagate as an ``aiohttp``, ``requests`` or ``TimeoutError`` exception. vLLM's error handler
answers anything that is not a ``ValueError`` with HTTP 500, so a client's bad URL read as a
server fault. ``admission.install_client_media_errors`` re-raises those two cases as a
``VLLMValidationError`` naming the content part (``image_url``), which vLLM answers with 400.
Every other error keeps its type. ``NeuronPlatform.check_and_update_config`` installs it on a
server that admits multimodal input, and only there.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/test_client_media_errors.py
"""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import aiohttp
import pytest
import requests
from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import MediaConnector

from vllm_neuron.vllm import admission
from vllm_neuron.vllm.platform import NeuronPlatform

from test.vllm_neuron.vllm.test_vision_merge_factor_refusal import (  # noqa: F401 (fixtures)
    _config,
    _isolated,
    merge_factor_4,
)

pytestmark = [pytest.mark.fast]

FILE_URL = "file:///etc/hostname"
HTTP_URL = "http://media.invalid/cat.png"


@pytest.fixture
def multimodal_server(merge_factor_4):
    """A server that admits images: check_and_update_config installs the mapping."""
    cfg = _config(MultiModalConfig(limit_per_prompt={"image": 1, "video": 0}))
    NeuronPlatform.check_and_update_config(cfg)
    return cfg


class _FailingConnection:
    """An HTTP connection whose every fetch raises ``exc``."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def get_bytes(self, url, **kwargs):
        raise self.exc

    async def async_get_bytes(self, url, **kwargs):
        raise self.exc


def _assert_400(exc: VLLMValidationError, parameter: str = "image_url") -> None:
    error = create_error_response(exc).error
    assert error.code == HTTPStatus.BAD_REQUEST
    assert error.type == "BadRequestError"
    assert error.param == parameter


def test_a_file_url_gets_400_naming_image_url(multimodal_server):
    with pytest.raises(VLLMValidationError) as refused:
        asyncio.run(MediaConnector().fetch_image_async(FILE_URL))
    assert "--allowed-local-media-path" in str(refused.value)
    assert isinstance(refused.value.__cause__, RuntimeError)
    _assert_400(refused.value)


def test_a_file_url_gets_400_on_the_synchronous_path_too(multimodal_server):
    with pytest.raises(VLLMValidationError) as refused:
        MediaConnector().fetch_image(FILE_URL)
    _assert_400(refused.value)


@pytest.mark.parametrize(
    "exc",
    [
        aiohttp.ClientConnectionError("Connection refused"),
        aiohttp.ServerDisconnectedError(),
        TimeoutError(),
    ],
    ids=["connection_refused", "server_disconnected", "timeout"],
)
def test_a_failed_fetch_gets_400_naming_the_host(multimodal_server, exc):
    connector = MediaConnector(connection=_FailingConnection(exc))
    with pytest.raises(VLLMValidationError) as refused:
        asyncio.run(connector.fetch_image_async(HTTP_URL))
    message = str(refused.value)
    assert "media.invalid" in message
    assert type(exc).__name__ in message
    assert refused.value.__cause__ is exc
    _assert_400(refused.value)


@pytest.mark.parametrize(
    "exc",
    [requests.exceptions.ConnectionError("refused"), requests.exceptions.HTTPError("404")],
    ids=["connection_error", "http_error"],
)
def test_a_failed_synchronous_fetch_gets_400(multimodal_server, exc):
    connector = MediaConnector(connection=_FailingConnection(exc))
    with pytest.raises(VLLMValidationError) as refused:
        connector.fetch_image(HTTP_URL)
    _assert_400(refused.value)


def test_a_video_fetch_names_video_url(multimodal_server):
    connector = MediaConnector(connection=_FailingConnection(TimeoutError()))
    with pytest.raises(VLLMValidationError) as refused:
        asyncio.run(connector.fetch_video_async("http://media.invalid/clip.mp4"))
    _assert_400(refused.value, "video_url")


def test_a_server_error_keeps_its_type(multimodal_server):
    """Only the two client cases are mapped: anything else stays a server error."""
    connector = MediaConnector(connection=_FailingConnection(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(connector.fetch_image_async(HTTP_URL))


def test_a_value_error_is_left_as_it_is(multimodal_server):
    with pytest.raises(ValueError) as refused:
        asyncio.run(MediaConnector().fetch_image_async("ftp://media.invalid/cat.png"))
    assert not isinstance(refused.value, VLLMValidationError)


def test_a_text_only_server_leaves_the_connector_as_vllm_ships_it():
    NeuronPlatform.check_and_update_config(_config(None))
    with pytest.raises(RuntimeError):
        asyncio.run(MediaConnector().fetch_image_async(FILE_URL))


def test_installing_twice_wraps_once(multimodal_server):
    wrapped = MediaConnector.load_from_url_async
    admission.install_client_media_errors()
    assert MediaConnector.load_from_url_async is wrapped
