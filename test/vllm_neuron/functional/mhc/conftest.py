"""Report, from inside the pytest process, which ``vllm_neuron`` this run actually imported.

WHY THIS FILE EXISTS. An acceptance run is a pytest process, and the only reading that settles where its
code came from is one taken inside that process. A probe run beside it is a different interpreter: pytest
puts its own rootdir and test-directory entries on ``sys.path`` ahead of anything ``PYTHONPATH`` supplies,
so two processes given the same environment can still resolve one package to two different files. The
failure this guards against really happened in this campaign -- a helper ran from a scratch directory, so
the import resolved to the SHARED repository, and every check around it passed because everything reported
was true of the wrong tree.

WHY IT SITS AT THIS DIRECTORY AND NOT DEEPER. pytest loads every conftest.py from the rootdir down to the
collected test, so one file here covers all the glm5_next tests, in every lane, without a copy per test
directory. Two lane branches carrying byte-identical copies of this file fold together without a conflict;
two near-identical copies would not.

WHY THE WORK IS IN pytest_collection. Importing at module scope would run when pytest imports this
conftest, which is BEFORE any pytest_configure. That is not a style point: ``vllm_neuron/__init__.py``
initialises its backend at import time from ``VLLM_NEURON_CPU_MODE``, and the repository's own
test/conftest.py pins that variable inside pytest_configure. An import before the pin would initialise the
wrong backend and freeze the wrong FP8 clamp, and every row downstream would still look green. A
pytest_configure hook marked ``trylast`` would only REQUEST that order from the plugin manager. Collection
happening after configure is a fact about pytest's lifecycle rather than a request, so this hook takes the
reading where the order cannot be lost.

WHY BOTH A SPEC ORIGIN AND A MODULE FILE. The spec origin is where the import system says the package
WOULD come from, resolved without executing it. The module file is where the package the process actually
holds came from. They normally agree; when they disagree, something imported the package earlier from
somewhere else, and that is a finding rather than a detail. Printing one of the two would hide it.

It prints. It writes nothing, removes nothing, deletes nothing, and opens no connection.
"""

import hashlib
import importlib.util
import os

#: The package whose provenance is in question, named once. The digest is of this package's own
#: ``__init__.py`` -- the file ``__file__`` names -- not of any module under it.
PACKAGE = "vllm_neuron"


def _emit(key, value):
    """One key per line, the value alone on it.

    The acceptance driver reads these with an exact literal prefix and takes the rest of the line as the
    value, so anything sharing the line would be read as part of the value.
    """
    print("%s=%s" % (key, value), flush=True)


def pytest_collection(session):
    """Say where ``vllm_neuron`` comes from, after the environment pin and before any test module import.

    Nothing here raises. A run of these tests in some other environment should get a named reading it can
    act on, not a collection error from a file whose whole job is to report.
    """
    del session  # the report is about this interpreter, not about the session

    try:
        spec = importlib.util.find_spec(PACKAGE)
    except Exception as exc:  # noqa: BLE001
        _emit("PYTEST_MODULE_SPEC_ORIGIN", "FIND_SPEC_FAILED[%s: %s]" % (type(exc).__name__, exc))
    else:
        # A namespace package has a spec with no origin, which is a different fact from a missing spec.
        if spec is None:
            _emit("PYTEST_MODULE_SPEC_ORIGIN", "NO_SPEC")
        else:
            _emit("PYTEST_MODULE_SPEC_ORIGIN", spec.origin or "SPEC_HAS_NO_ORIGIN")

    try:
        import vllm_neuron

        path = getattr(vllm_neuron, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        # NAMED, NOT SILENT. A missing reading and a wrong reading are different defects, and the driver
        # has to tell them apart rather than seeing one absent key for both.
        _emit("PYTEST_MODULE_FILE", "IMPORT_FAILED[%s: %s]" % (type(exc).__name__, exc))
        _emit("PYTEST_MODULE_SHA", "IMPORT_FAILED")
        return

    if not path:
        _emit("PYTEST_MODULE_FILE", "NO_FILE_ATTRIBUTE")
        _emit("PYTEST_MODULE_SHA", "NO_FILE_ATTRIBUTE")
        return

    _emit("PYTEST_MODULE_FILE", path)
    # THE PATH IS NOT THE BYTES. Two trees carry this path with different content, and only a digest tells
    # them apart, so the digest is taken here from the file the process itself named.
    if not os.path.isfile(path):
        _emit("PYTEST_MODULE_SHA", "FILE_NOT_ON_DISK")
        return
    with open(path, "rb") as handle:
        _emit("PYTEST_MODULE_SHA", hashlib.sha256(handle.read()).hexdigest())
