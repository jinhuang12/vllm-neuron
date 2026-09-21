# `test/` is an overlay, not an upstream tree

Upstream ships no test suite for this plugin, and the pinned release ships no
`test/` path at all. Everything under `test/` is fork-owned and is re-applied as
an overlay after each rebase onto the upstream release branch.

Three properties keep that re-apply mechanical:

1. The overlay lives entirely under `test/`. The only file outside it that
   belongs to the overlay is the `[tool.pytest.ini_options] markers` block in
   `pyproject.toml`.
2. Nothing under `test/` is imported by `vllm_neuron/`. The dependency runs one
   way, so deleting `test/` cannot break the shipped package.
3. The tree matches the configuration the release ships:
   `pyproject.toml` declares `testpaths = ["test/unit", "test/vllm_neuron"]` and
   both directories exist. `test/unit/test_layout.py` fails if a later change
   moves the tree out from under that config.

## Markers

`pyproject.toml` registers three markers and this is where their meaning lives:

- `fast` — cheap, device-free, safe to run on every change.
- `forked` — must run in its own process, because it depends on state that does
  not reset in-process (for example an environment variable read at import).
- `quarantined` — vendored from upstream ahead of the code it exercises. The
  marker gates selection, not import: `test/vllm_neuron/upstream/conftest.py`
  applies it to every item it collects there, and also skips those items, so a
  plain run reports them as skipped instead of failing against absent upstream
  behaviour.

## Vendored upstream tests

`test/vllm_neuron/upstream/` holds five test files copied from one upstream pull
request at one commit:

- Origin: [`vllm-project/vllm#53906`](https://github.com/vllm-project/vllm/pull/53906),
  head commit `878631b6079d2cf9fb80830ef9cb41b43aded098` (head repo
  `ZJY0516/vllm`, branch `glm-release`). The pull request is open and still
  moving, so the commit rather than the branch identifies what was copied.
- They are kept for two reasons: they fingerprint a moving upstream source, so a
  later rebase can tell whether the original changed; and they are the upstream
  author's own statement of the intended behaviour, which the Neuron tests under
  `test/vllm_neuron/functional/` and `test/vllm_neuron/model/` were written
  against.
- They contribute no executing assertions here. Four of them import vLLM-core
  symbols this fork does not author — a new attention backend, a new processor,
  a new model package — and the fifth compares upstream's fp8 paged indexer
  cache, which this platform does not carry (its k-pool path is bfloat16). So
  none of the five is ever un-skipped, and the last column below reads `NEVER`.

Each row records the origin path and the sha256 the file had when it was copied.
For an unchanged copy the on-disk column repeats that digest, and
`test/unit/test_layout.py` re-checks it against the bytes on every run: that is
what proves the copy is still a copy. The one adapted file carries `n/a` there,
because a digest of a file this repository edits is written in the same change as
the file and so certifies nothing. Rows are read by column position.

| Vendored path | Origin PR | Upstream path | sha256 when copied | sha256 on disk | Adopted | Un-skip when |
|---|---|---|---|---|---|---|
| `test/vllm_neuron/upstream/test_kpool_tail_slot_mapping.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_kpool_tail_slot_mapping.py` | `8a56bffb0d69a44353667ed6df79ce454bb1b16913e4663814b98da9d8fdcbdf` | `8a56bffb0d69a44353667ed6df79ce454bb1b16913e4663814b98da9d8fdcbdf` | `VERBATIM` | `NEVER` |
| `test/vllm_neuron/upstream/test_flashinfer_mla_sparse_sm90.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_flashinfer_mla_sparse_sm90.py` | `48c74334cb4035e38d1367cb2df9dd64f32d9f7286ff7a591fb22ef2c5e4b714` | `48c74334cb4035e38d1367cb2df9dd64f32d9f7286ff7a591fb22ef2c5e4b714` | `VERBATIM` | `NEVER` |
| `test/vllm_neuron/upstream/test_sparse_indexer_decode_seq_lens.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py` | `b501b93a304e62ae2208192e05303854868d946a8fff20be72caca2015242fbd` | `b501b93a304e62ae2208192e05303854868d946a8fff20be72caca2015242fbd` | `VERBATIM` | `NEVER` |
| `test/vllm_neuron/upstream/test_glm5next.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/transformers_utils/processors/test_glm5next.py` | `cfe0af2278ca47dd48ba15c06ae6974b87c2c82587ea050d6388d587d1fa2c9a` | `cfe0af2278ca47dd48ba15c06ae6974b87c2c82587ea050d6388d587d1fa2c9a` | `VERBATIM` | `NEVER` |
| `test/vllm_neuron/upstream/test_kpool_decode_update_batched.py` | `vllm-project/vllm#53906` @ `878631b6` | `tests/kernels/test_kpool_decode_update_batched.py` | `a36a2e84e381294e3a691e3722e7e414db114e8735dcb98fee656993eb0bcc25` | `n/a` | `ADAPTED` | `NEVER` |

A `VERBATIM` body is byte-identical to upstream and stays that way: editing one —
including adding an import guard or a module-level skip to make collection green
— is exactly the drift the digest columns exist to expose. That is why the
collection rules live in `test/vllm_neuron/upstream/conftest.py` instead.

`test_kpool_decode_update_batched.py` is the one `ADAPTED` file. Its only change
from the original is that the origin's import branch is replaced by a fork
adapter; every test body, `parametrize` list and device placement is left as
upstream wrote it.
