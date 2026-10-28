# SPDX-License-Identifier: Apache-2.0
"""Command-line options of the kernel tests."""


def pytest_addoption(parser):
    """``--dcp-groups``: the DCP groups the DCP decode attention items run side by side."""
    parser.addoption("--dcp-groups", type=int, default=1,
                     help="DCP groups the DCP decode attention items run side by side (default 1)")
