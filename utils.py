"""
Utilities for the ``rig`` package.
"""

from __future__ import annotations

import contextlib
import faulthandler
import functools
import locale
import os
import sys
import tempfile
import time
import unittest
from typing import Sequence


# the file `run_tests` logs to when it is given none, in the temp folder
_DEFAULT_LOG = "rig_run_tests.log"

# the log of the `run_tests` in progress, the one Python's fault handler writes to
_running_log = None


class _Log:
    """The file a run of `run_tests` logs to. Every write is flushed, so the line that names a
    test is on disk before the test runs, and the file is the one Python's fault handler writes
    the stack of a crash to (`fileno`). It has the ``writeln`` a unittest result wants of its
    stream."""

    def __init__(self, path):
        self.path  = path
        self._file = open(path, "w", encoding="utf-8", errors="backslashreplace", newline="\n")

    def write(self, text):
        self._file.write(text)
        self._file.flush()

    def writeln(self, text=None):
        self.write("\n" if text is None else text + "\n")

    def flush(self):
        self._file.flush()

    def fileno(self):
        return self._file.fileno()

    def close(self):
        self._file.close()


class _ProgressResult(unittest.TextTestResult):
    """TextTestResult that prefixes each verbose test line with ``[N/total]``.

    ``startTest`` writes the counter and then defers to the stdlib for the
    description and `` ... ``, so the line format stays whatever this Python's
    unittest produces. The hook has the same shape on 3.7 (Maya 2022) and 3.11
    (Maya 2025). ``**kwargs`` absorbs the ``durations=`` that 3.12+ passes.
    Only active when ``showAll`` is set, i.e. verbosity 2 -- dot mode is untouched.

    ``log`` is a second result, the one that writes the run's log file (see
    ``_run``). Every event goes to it first, so the log holds a test's line
    before the test runs and before the stream shows it.
    """

    def __init__(self, stream, descriptions, verbosity, total=0, log=None, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)
        self._total = total
        self._width = len(str(total))
        self._log   = log

    def _to_log(self, event, *args):
        if self._log is not None:
            getattr(self._log, event)(*args)

    def startTest(self, test):
        self._to_log("startTest", test)
        if self.showAll:
            self.stream.write(
                "[%*d/%d] " % (self._width, self.testsRun + 1, self._total)
            )
        super().startTest(test)

    def stopTest(self, test):
        self._to_log("stopTest", test)
        super().stopTest(test)

    def addSuccess(self, test):
        self._to_log("addSuccess", test)
        super().addSuccess(test)

    def addError(self, test, err):
        self._to_log("addError", test, err)
        super().addError(test, err)

    def addFailure(self, test, err):
        self._to_log("addFailure", test, err)
        super().addFailure(test, err)

    def addSkip(self, test, reason):
        self._to_log("addSkip", test, reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._to_log("addExpectedFailure", test, err)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._to_log("addUnexpectedSuccess", test)
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        self._to_log("addSubTest", test, subtest, err)
        super().addSubTest(test, subtest, err)


def _log_summary(result, seconds):
    """Ends the log the way ``unittest.TextTestRunner.run`` ends its stream: the
    tracebacks, the count and the time, then ``OK`` or ``FAILED`` with the numbers.
    `result` is the result that wrote the log's test lines."""
    log, ran = result.stream, result.testsRun
    result.printErrors()
    log.writeln(result.separator2)
    log.writeln("Ran %d test%s in %.3fs" % (ran, "" if ran == 1 else "s", seconds))
    log.writeln()
    counts = (
        ("failures",             len(result.failures)),
        ("errors",               len(result.errors)),
        ("skipped",              len(result.skipped)),
        ("expected failures",    len(result.expectedFailures)),
        ("unexpected successes", len(result.unexpectedSuccesses)),
    )
    infos = ", ".join("%s=%d" % (name, count) for name, count in counts if count)
    log.writeln(("OK" if result.wasSuccessful() else "FAILED") + (" (%s)" % infos if infos else ""))


def _run(suite, stream, verbosity, failfast, log=None):
    """Run ``suite`` with the progress-counting result class.

    `log` is the run's `_Log` (None: no log). It gets what verbosity 2 prints,
    whatever ``verbosity`` is: a line per test, one line each (no docstring
    line), then the tracebacks and the summary when the run ends or is stopped."""
    total  = suite.countTestCases()
    mirror = None if log is None else _ProgressResult(log, False, 2, total=total)
    start  = time.perf_counter()
    try:
        return unittest.TextTestRunner(
            stream      = stream,
            verbosity   = verbosity,
            failfast    = failfast,
            resultclass = functools.partial(_ProgressResult, total=total, log=mirror),
        ).run(suite)
    finally:
        if mirror is not None:
            _log_summary(mirror, time.perf_counter() - start)


def _log_path(log):
    """The path a run logs to for `run_tests`' ``log`` argument, None for no log."""
    if log is False:
        return None
    if log is None or log is True:
        return os.path.join(tempfile.gettempdir(), _DEFAULT_LOG)
    return os.fspath(log)


def _same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _open_log(log, stream):
    """Opens the log of a run and writes where it is on `stream`, the run's first
    line. None when the run has no log: ``log`` is False, or the default file
    cannot be opened, and then the line says why and the run goes on. A path the
    caller gave that cannot be opened raises its ``OSError``. The log of the run
    this one is inside counts as one that cannot be opened: opening it again
    would empty it."""
    path = _log_path(log)
    if path is None:
        return None
    try:
        if _running_log is not None and _same_file(path, _running_log.path):
            raise OSError(f"{path} is the log of the run this one is inside")
        opened = _Log(path)
    except OSError as err:
        if not (log is None or log is True):
            raise
        stream.write(f"run_tests log: none ({err})\n")
        stream.flush()
        return None
    stream.write(f"run_tests log: {path}\n")
    stream.flush()
    return opened


def _close_chunks():
    """Closes the named undo chunks this call runs inside; returns their name and
    how many there were, for `_reopen_chunks`.

    Inside ``maya.exe`` the Script Editor, and ``python("...")`` from MEL, run
    their text inside one undo chunk named by that text. While it is open nothing
    done is an undo step of its own and nothing leaves the undo queue, so a test
    that undoes, or that needs an edit to leave the queue, cannot pass. Measured
    2026-11-12 in Maya 2025: ``undoInfo(query=True, chunkName=True)`` is the name
    of the outermost named chunk, ``''`` when none is open or the open one has no
    name; a nested chunk shows the outer name until every level is closed; each
    ``closeChunk`` closes one level, undo on or off. An unnamed chunk cannot be
    seen, so it stays open. ``mayapy`` opens none: this does nothing there.

    The close and the open are edits of the undo queue, so they go through
    ``cmds_edit`` as every edit rig makes does (the journal's barrier)."""
    from maya import cmds
    from rig._internal.journal import cmds_edit

    name, depth = "", 0
    while depth < 64:
        seen = cmds.undoInfo(query=True, chunkName=True)
        if not seen:
            break
        name = seen
        cmds_edit.undoInfo(closeChunk=True)
        depth += 1
    return name, depth


def _reopen_chunks(name, depth):
    """Opens again the `depth` chunks `_close_chunks` closed, each under `name`
    (the outer chunk's: an inner one's name cannot be read). The caller's own
    ``closeChunk`` calls then pair up as they did."""
    from rig._internal.journal import cmds_edit

    for _ in range(depth):
        cmds_edit.undoInfo(openChunk=True, chunkName=name)


@contextlib.contextmanager
def _faults_to(log):
    """Python's fault handler writes to the `_Log` `log` inside the block: when a
    test takes Maya down (an access violation, which no ``except`` sees), the
    Python stack of every thread lands in the log under that test's line.

    After the block the handler is as it was, as far as that can be known: on the
    log of the run this one is inside; else, when it was on (``mayapy`` turns it
    on in ``maya.standalone.initialize()``; ``maya.exe`` leaves it off), on
    ``sys.__stderr__``, Python's own default, since the file it wrote to cannot be
    read back; else off. It is never left on a closed file. With no log (None)
    the handler is not touched."""
    global _running_log
    if log is None:
        yield
        return
    was_on, outer = faulthandler.is_enabled(), _running_log
    faulthandler.enable(file=log, all_threads=True)
    _running_log = log
    try:
        yield
    finally:
        _running_log = outer
        faulthandler.disable()
        if outer is not None:
            faulthandler.enable(file=outer, all_threads=True)
        elif was_on and sys.__stderr__ is not None:
            try:
                faulthandler.enable(file=sys.__stderr__, all_threads=True)
            except (AttributeError, OSError, ValueError):  # a stream with no file descriptor
                pass


@contextlib.contextmanager
def _logged(log, stream, target):
    """What `run_tests` sets up around the suite, and puts back: the log is open
    (yielded, None for none) with its header line, the enclosing named undo
    chunks are closed (`_close_chunks`), and Python's fault handler writes to the
    log (`_faults_to`). An exception that ends the run is named on the log's last
    line."""
    opened = _open_log(log, stream)
    try:
        if opened is not None:
            from rig import __version__

            opened.writeln(
                f"run_tests({target!r}) of rig {__version__}, "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {sys.executable}"
            )
        name, depth = _close_chunks()
        try:
            with _faults_to(opened):
                yield opened
        except BaseException as err:
            if opened is not None:
                opened.writeln(f"run_tests stopped: {type(err).__name__}: {err}")
            raise
        finally:
            _reopen_chunks(name, depth)
    finally:
        if opened is not None:
            opened.close()


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


def _collect(target, package_root, start_dir):
    """The suite `run_tests` runs for its ``target``, out of the tests under
    `start_dir` (``<package_root>/_tests``)."""
    # top_level_dir is the package's PARENT, so modules import as
    # ``rig._tests.<name>`` rather than as a bare ``<name>``.  The suite
    # imports its own siblings by that dotted path, so pointing this at
    # ``_tests`` would break them.
    top_level_dir = os.path.dirname(package_root)
    root          = f"{os.path.basename(package_root)}._tests."
    loader        = unittest.TestLoader()

    if target is None:
        targets: list[str] = []
    elif isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)

    if not targets:
        return loader.discover(
            start_dir, pattern="test_*.py", top_level_dir=top_level_dir
        )

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

    return suite


def run_tests(
    target:    str  | Sequence[str] | None      = None,
    verbosity: int                              = 2,
    failfast:  bool                             = False,
    log:       str  | os.PathLike | bool | None = None,
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
        log: where the run is logged.  ``None`` writes ``rig_run_tests.log``
            in the temp folder (``tempfile.gettempdir()``), a file every run
            starts over; a path writes there; ``False`` writes no log.  The
            first line the run prints is ``run_tests log: <path>``.

            The log holds one line per test, on disk before the test runs,
            then the tracebacks and the summary, whatever ``verbosity`` is.
            When a test takes Maya down, the log's last test line names it
            and the Python stack of the crash follows it.

    Returns:
        The :class:`unittest.TestResult`.  ``result.wasSuccessful()`` is the
        pass/fail answer; ``result.errors`` and ``result.failures`` carry the
        detail.

    Raises:
        ValueError: a dotted target naming no such module, class or method.
            Unittest would otherwise fold that into the run as an ordinary
            test error, which reads like a real failure rather than a typo.
        OSError: ``log`` is a path that cannot be opened for writing.  When
            the default file cannot be opened the run goes on without a log
            and its first line says why.

    Note:
        Discovery imports every matching module, so a missing optional
        dependency shows up as an error against that module rather than
        aborting the run.

        Maya is initialized on demand -- under ``mayapy`` that brings up
        standalone; inside GUI Maya it is left alone. The suite runs in the
        session it was started in and starts no interpreter of its own: a
        test that needs one (a fresh import, a crash it must survive) or
        another Maya version is skipped, naming why (2026-11-10).

        Inside ``maya.exe`` the Script Editor, and ``python("...")`` from
        MEL, run their text inside one named undo chunk, in which nothing is
        an undo step of its own and nothing leaves the undo queue: the tests
        that undo fail there.  ``run_tests`` closes the named chunks it is
        called in for the run and opens them again when it ends, so the
        whole run is not one undo step.  A chunk with no name cannot be seen
        and stays open (2026-11-12).

        With a log, Python's fault handler writes to it for the run and is
        put back afterwards (on ``sys.__stderr__`` when it was on, as under
        ``mayapy``; off when it was off).

        >>> from rig.utils import run_tests
        >>> run_tests()                        # everything
        >>> run_tests("test_class_*.py")                      # a glob
        >>> run_tests("test_plug.TestPlugBasics")             # one class
        >>> run_tests("test_plug.TestPlugBasics.test_repr")   # one test
        >>> run_tests(["test_types", "test_spec"])            # two modules
        >>> run_tests(verbosity=1, failfast=True)
        >>> run_tests(log="D:/logs/rig_tests.log")            # log there
        >>> run_tests(log=False)                              # no log
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    start_dir    = os.path.join(package_root, "_tests")

    if not os.path.isdir(start_dir):
        raise FileNotFoundError(f"no test directory at {start_dir!r}")

    # The suite's own harness owns the standalone-vs-GUI decision, so reuse it
    # rather than duplicating the guard and letting the two drift.
    from rig._tests._base import initialize_standalone

    initialize_standalone()

    stream = _encoding_safe(sys.stderr)   # resolved now, so an installed stderr tee is seen

    # the log, the undo chunks and the fault handler are set before any test
    # module is imported: an import can take Maya down as a test can
    with _logged(log, stream, target) as opened:
        suite = _collect(target, package_root, start_dir)
        return _run(suite, stream, verbosity, failfast, opened)


# ``mayapy -m rig.utils`` runs the suite and exits non-zero on failure, which
# is what a CI step or a pre-commit hook wants.  An optional argument narrows
# the run: ``mayapy -m rig.utils test_plug.TestPlugBasics``
if __name__ == "__main__":
    sys.exit(0 if run_tests(*sys.argv[1:2]).wasSuccessful() else 1)


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "run_tests",
]
