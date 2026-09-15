"""
Utilities for the ``rig`` package.
"""

from __future__ import annotations

import functools
import locale
import os
import sys
import unittest
from typing import Sequence


class _ProgressResult(unittest.TextTestResult):
    """TextTestResult that prefixes each verbose test line with ``[N/total]``.

    ``startTest`` writes the counter and then defers to the stdlib for the
    description and `` ... ``, so the line format stays whatever this Python's
    unittest produces. The hook has the same shape on 3.7 (Maya 2022) and 3.11
    (Maya 2025). ``**kwargs`` absorbs the ``durations=`` that 3.12+ passes.
    Only active when ``showAll`` is set, i.e. verbosity 2 -- dot mode is untouched.
    """

    def __init__(self, stream, descriptions, verbosity, total=0, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)
        self._total = total
        self._width = len(str(total))

    def startTest(self, test):
        if self.showAll:
            self.stream.write(
                "[%*d/%d] " % (self._width, self.testsRun + 1, self._total)
            )
        super().startTest(test)


def _run(suite, stream, verbosity, failfast):
    """Run ``suite`` with the progress-counting result class."""
    return unittest.TextTestRunner(
        stream    = stream,
        verbosity = verbosity,
        failfast  = failfast,
        resultclass=functools.partial(
            _ProgressResult, total=suite.countTestCases()
        ),
    ).run(suite)


def _encoding_safe(stream):
    """Wrap ``stream`` so nothing written through it can raise UnicodeEncodeError.

    Text is escaped against the platform default codec -- the one a downstream
    ``open()`` or ``logging.FileHandler`` uses when given no encoding -- before
    it reaches the real stream. Everything the codec can carry passes through
    untouched; anything it cannot is written as its escaped code point. On Windows that
    codec is cp1252, so em-dashes and degree signs still render and only true
    exotics are escaped.

    Delegation keeps whatever ``sys.stderr`` really is -- Maya's Script Editor,
    a studio log tee -- in the loop, so those still receive every line, now
    guaranteed encodable. Without this, one ``->`` written as U+2192 in a test
    docstring can turn into a recursive logging flood when the tee's handler
    cannot encode it and reports the failure back through the same stream.
    """
    codec = locale.getpreferredencoding(False) or "ascii"

    class _Safe:
        def write(self, text):
            return stream.write(text.encode(codec, "backslashreplace").decode(codec))

        def flush(self):
            return stream.flush()

        def __getattr__(self, name):
            return getattr(stream, name)

    return _Safe()


def run_tests(
    target:    str | Sequence[str] | None = None,
    verbosity: int                        = 2,
    failfast:  bool                       = False,
) -> unittest.TestResult:
    """
    runs the package's unit test suite

    Args:
        target: what to run, coarsest to finest::

            None                    the whole suite
            "test_class_*.py"                       a filename glob
            "test_plug"                             one module
            "test_plug.TestPlugBasics"              one class
            "test_plug.TestPlugBasics.test_repr"    one test method

            A list or tuple runs several in one pass, which is the quick way
            to re-run a handful of failures.

            Short dotted names are resolved inside the package, so you never
            write the fully qualified ``rig._tests....`` path.  The
            search covers ``_tests`` and ``_tests/test_rig``, so a bare
            ``test_plug`` finds the nested module.
            Pasting the long form copied out of a failure line works too.
        verbosity: 0 for silent, 1 for a dot per test, 2 for a line each.
        failfast: stop on the first failure or error.

    Returns:
        The :class:`unittest.TestResult`.  ``result.wasSuccessful()`` is the
        pass/fail answer; ``result.errors`` and ``result.failures`` carry the
        detail.

    Raises:
        ValueError: a dotted target naming no such module, class or method.
            Unittest would otherwise fold that into the run as an ordinary
            test error, which reads like a real failure rather than a typo.

    Note:
        Discovery imports every matching module, so a missing optional
        dependency shows up as an error against that module rather than
        aborting the run.

        Maya is initialized on demand -- under ``mayapy`` that brings up
        standalone; inside GUI Maya it is left alone.

        >>> from rig.utils import run_tests
        >>> run_tests()                        # everything
        >>> run_tests("test_class_*.py")                      # a glob
        >>> run_tests("test_plug.TestPlugBasics")             # one class
        >>> run_tests("test_plug.TestPlugBasics.test_repr")   # one test
        >>> run_tests(["test_types", "test_spec"])            # two modules
        >>> run_tests(verbosity=1, failfast=True)
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    start_dir    = os.path.join(package_root, "_tests")

    if not os.path.isdir(start_dir):
        raise FileNotFoundError(f"no test directory at {start_dir!r}")

    # The suite's own harness owns the standalone-vs-GUI decision, so reuse it
    # rather than duplicating the guard and letting the two drift.
    from rig._tests._base import initialize_standalone

    initialize_standalone()

    # top_level_dir is the package's PARENT, so modules import as
    # ``rig._tests.<name>`` rather than as a bare ``<name>``.  The suite
    # imports its own siblings by that dotted path, so pointing this at
    # ``_tests`` would break them.
    top_level_dir = os.path.dirname(package_root)
    root          = f"{os.path.basename(package_root)}._tests."
    loader        = unittest.TestLoader()

    stream = _encoding_safe(sys.stderr)   # resolved now, so an installed stderr tee is seen

    if target is None:
        targets: list[str] = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    if not targets:
        suite = loader.discover(
            start_dir, pattern="test_*.py", top_level_dir=top_level_dir
        )
        return _run(suite, stream, verbosity, failfast)

    # Namespaces a short name may live in: ``_tests`` itself, plus any test
    # subpackage inside it.  Callers should not have to know whether a module
    # sits at the top level or one directory down.
    namespaces = [root] + [
        root + entry + "."
        for entry in sorted(os.listdir(start_dir))
        if os.path.isfile(os.path.join(start_dir, entry, "__init__.py"))
    ]

    suite = unittest.TestSuite()
    for item in targets:
        if "*" in item or "?" in item or item.endswith(".py"):
            suite.addTests(
                loader.discover(start_dir, pattern=item, top_level_dir=top_level_dir)
            )
            continue

        # accept the short "test_x...", "_tests.test_x..." and the fully
        # qualified "rig._tests.test_x..." that a failure line prints
        name = item[len("_tests.") :] if item.startswith("_tests.") else item
        candidates = (
            [name] if name.startswith(root) else [ns + name for ns in namespaces]
        )

        resolved = None
        for candidate in candidates:
            mark  = len(loader.errors)
            found = loader.loadTestsFromName(candidate)
            if len(loader.errors) == mark:
                resolved = found
                break
            del loader.errors[mark:]  # discard the miss, try the next namespace

        if resolved is None:
            raise ValueError(
                "could not resolve test target "
                + repr(item)
                + "; tried "
                + ", ".join(candidates)
            )
        suite.addTests(resolved)

    return _run(suite, stream, verbosity, failfast)


# ``mayapy -m rig.utils`` runs the suite and exits non-zero on failure, which
# is what a CI step or a pre-commit hook wants.  An optional argument narrows
# the run: ``mayapy -m rig.utils test_plug.TestPlugBasics``
if __name__ == "__main__":
    sys.exit(0 if run_tests(*sys.argv[1:2]).wasSuccessful() else 1)
