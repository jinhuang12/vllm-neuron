# SPDX-License-Identifier: Apache-2.0
"""Score the 20 rung-1 image replies with the port's coherence checks and write the
21-row on-topic sheet: the 20 replies and one planted reply, with its key.

The arguments are those of ``score_coherence.py``; only the sheet size differs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score_coherence  # noqa: E402

REPLIES = 20


def main(argv=None):
    """Run the port's scorer with a sheet of 20 replies."""
    score_coherence.SHEET_ROWS = REPLIES
    return score_coherence.main(argv)


if __name__ == "__main__":
    sys.exit(main())
