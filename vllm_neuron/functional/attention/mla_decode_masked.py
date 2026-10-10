# SPDX-License-Identifier: Apache-2.0
"""Selecting-step forms of the batched MLA decode attention that read the window once.

:mod:`mla_decode`'s selected mode gathers each query row's ``K`` selected window rows on
its own. When a request's window holds fewer rows than its ``T`` query rows gather
between them (``T * K``), reading the window once per request and masking, per row, the
columns outside that row's selection attends the same rows from fewer bytes. The kernels
of that form live in this file, beside :mod:`mla_decode` and not in it, and reuse its
helpers by import.

The file is kept out of every other graph's compile cache key
(:mod:`vllm_neuron.compile_cache_key` folds into a graph's key the kernel files the
graph's kernels reach), and three rules keep it so
(``test_mla_decode_masked_keys.py`` pins them):

* no other digested file imports it, so its source reaches only the graphs that call
  its kernels;
* it rebinds no attribute or item of a module it imports: every graph that reaches the
  patched module would fold this file;
* it imports nothing dynamically and nothing of ``nkilib``: the first makes every key
  fold the whole-package digest, the second makes every graph that calls that ``nkilib``
  kernel fold this file.
"""
