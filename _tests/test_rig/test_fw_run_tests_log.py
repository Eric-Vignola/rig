"""`run_tests` logs its run, and runs outside the undo chunk it is called in (2026-11-12).

A test that takes Maya down leaves nothing behind: the Script Editor goes with it. So every
run writes a log (``rig_run_tests.log`` in the temp folder, or ``run_tests(log=<path>)``;
``log=False`` for none): one line per test, on disk before the test runs, then the
tracebacks and the summary, whatever ``verbosity`` is. Python's fault handler writes to the
same file for the run, so the stack of a crash lands under the line of the test that made it.

Inside ``maya.exe`` the Script Editor, and ``python("...")`` from MEL, run their text inside
one named undo chunk, in which nothing is an undo step of its own and nothing leaves the undo
queue: 247 failures and a crash in the GUI, 230 in batch, none of them in ``mayapy``.
``run_tests`` closes the named chunks it is called in for the run and opens them again after.

* `TestTheLog`: what `_run` writes to the log.
* `TestWhereTheLogGoes`: the ``log`` argument, and a file that cannot be opened.
* `TestTheFaultHandler`: on the log for the run, as it was after.
* `TestTheUndoChunk`: the enclosing named chunks, closed and opened again.
* `TestRunTests`: `run_tests` itself, on `TestTheTarget`.

No test here starts a process, and none runs more than `TestTheTarget`.
"""

import contextlib
import faulthandler
import io
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from maya import cmds

import rig
from rig import utils
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

_TARGET = "test_fw_run_tests_log.TestTheTarget"

# what `TestTheTarget` saw each time it ran: (the open named chunk, the path of the log the
# fault handler writes to, whether the handler is on)
_SEEN = []


class TestTheTarget(unittest.TestCase):
    """The one test `TestRunTests` hands to `run_tests`. It records what a test sees."""

    def test_what_a_test_sees(self):
        running = vars(utils)["_running_log"]
        _SEEN.append(
            (
                cmds.undoInfo(query=True, chunkName=True),
                None if running is None else running.path,
                faulthandler.is_enabled(),
            )
        )


def _samples(seen, path):
    """A suite of tests of every outcome, in this order: ok, a failure, an error, a skip, a
    subtest that fails, an expected failure, an unexpected success, a test with a docstring.
    The first appends to `seen` what the file `path` holds while it runs. The class is made
    here so that the loader finds none of it."""

    class Sample(unittest.TestCase):
        def test_a_ok(self):
            with open(path, encoding="utf-8") as handle:
                seen.append(handle.read())

        def test_b_fails(self):
            self.assertEqual(1, 2)

        def test_c_errors(self):
            return 1 // 0

        def test_d_skipped(self):
            self.skipTest("no reason")

        def test_e_subtests(self):
            for i in range(3):
                with self.subTest(i=i):
                    self.assertNotEqual(i, 1)

        @unittest.expectedFailure
        def test_f_expected_failure(self):
            self.assertEqual(1, 2)

        @unittest.expectedFailure
        def test_g_unexpected_success(self):
            pass

        def test_h_documented(self):
            """A first line the stream shows and the log does not."""

    return unittest.TestSuite(Sample(name) for name in sorted(unittest.getTestCaseNames(Sample, "test")))


def _stopped():
    """A suite of three tests whose second is interrupted."""

    class Sample(unittest.TestCase):
        def test_a_ok(self):
            pass

        def test_b_interrupted(self):
            raise KeyboardInterrupt

        def test_c_never_runs(self):
            pass

    return unittest.TestSuite(Sample(name) for name in sorted(unittest.getTestCaseNames(Sample, "test")))


def _lines(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read().split("\n")


class _Folder(MayaTestCase):
    """A temp folder of the test's own, gone after it."""

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_run_tests_log_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.path = os.path.join(self.folder, "run.log")


class TestTheLog(_Folder):
    """`_run`: the log gets a line per test before the test runs, then the tracebacks and the
    summary."""

    def _logged(self, suite, verbosity=0, failfast=False):
        stream, log = io.StringIO(), utils._Log(self.path)
        try:
            result = utils._run(suite, stream, verbosity, failfast, log)
        finally:
            log.close()
        return result, stream.getvalue(), _lines(self.path)

    def test_a_line_per_test(self):
        suite = _samples([], self.path)
        tests = list(suite)
        _result, _out, got = self._logged(suite)
        endings = ("ok", "FAIL", "ERROR", "skipped 'no reason'", None, "expected failure", "unexpected success", "ok")
        for i, (test, ending) in enumerate(zip(tests, endings), 1):
            with self.subTest(test=test._testMethodName):
                start = "[%d/8] %s ... " % (i, test)
                found = [line for line in got if line.startswith(start)]
                self.assertEqual(len(found), 1)
                if ending is not None:
                    self.assertEqual(found[0], start + ending)

    def test_the_line_of_a_test_is_on_disk_before_the_test_runs(self):
        seen  = []
        suite = _samples(seen, self.path)
        first = list(suite)[0]
        self._logged(suite)
        self.assertEqual(seen, ["[1/8] %s ... " % first])

    def test_the_tracebacks_and_the_summary(self):
        result, _out, got = self._logged(_samples([], self.path))
        text = "\n".join(got)
        self.assertEqual(text.count("Traceback (most recent call last):"), 3)
        self.assertIn("AssertionError: 1 != 2", text)
        self.assertIn("ZeroDivisionError", text)
        self.assertIn("(i=1)", text)
        self.assertRegex(got[-4], r"^Ran 8 tests in \d+\.\d{3}s$")
        self.assertEqual(
            got[-3:],
            ["", "FAILED (failures=2, errors=1, skipped=1, expected failures=1, unexpected successes=1)", ""],
        )
        # the log's result is a second one: the caller's is what it would be without a log
        self.assertEqual(
            (result.testsRun, len(result.failures), len(result.errors), len(result.skipped)),
            (8, 2, 1, 1),
        )
        self.assertEqual((len(result.expectedFailures), len(result.unexpectedSuccesses)), (1, 1))

    def test_the_log_is_whole_at_any_verbosity(self):
        numbered = re.compile(r"^\[\d/8\] ")
        for verbosity in (0, 1, 2):
            with self.subTest(verbosity=verbosity):
                _result, out, got = self._logged(_samples([], self.path), verbosity)
                self.assertEqual(len([line for line in got if numbered.match(line)]), 8)
                shown = [line for line in out.split("\n") if numbered.match(line)]
                self.assertEqual(len(shown), 8 if verbosity == 2 else 0)

    def test_a_docstring_is_not_in_the_log(self):
        _result, out, got = self._logged(_samples([], self.path), 2)
        self.assertIn("A first line the stream shows and the log does not.", out)
        self.assertNotIn("A first line the stream shows and the log does not.", "\n".join(got))

    def test_a_run_of_all_passes(self):
        class Sample(unittest.TestCase):
            def test_only(self):
                pass

        result, _out, got = self._logged(unittest.TestSuite([Sample("test_only")]))
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(got[0], "[1/1] %s ... ok" % Sample("test_only"))
        self.assertRegex(got[-4], r"^Ran 1 test in \d+\.\d{3}s$")
        self.assertEqual(got[-3:], ["", "OK", ""])

    def test_failfast(self):
        result, _out, got = self._logged(_samples([], self.path), failfast=True)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(len([line for line in got if line.startswith("[")]), 2)
        self.assertRegex(got[-4], r"^Ran 2 tests in ")
        self.assertEqual(got[-3:], ["", "FAILED (failures=1)", ""])

    def test_a_run_that_is_stopped_still_ends_its_log(self):
        stream, log = io.StringIO(), utils._Log(self.path)
        try:
            with self.assertRaises(KeyboardInterrupt):
                utils._run(_stopped(), stream, 0, False, log)
        finally:
            log.close()
        got = _lines(self.path)
        self.assertEqual(len([line for line in got if line.startswith("[")]), 2)
        self.assertTrue(got[1].startswith("[2/3] test_b_interrupted "))
        self.assertRegex(got[-4], r"^Ran 2 tests in ")

    def test_no_log(self):
        result = utils._run(_samples([], self.path), io.StringIO(), 0, False)
        self.assertEqual(result.testsRun, 8)
        self.assertFalse(os.path.exists(self.path))

    def test_what_no_codec_of_the_console_carries(self):
        log = utils._Log(self.path)
        log.writeln("an arrow \u2192 and a lone surrogate \udc80")
        log.close()
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), b"an arrow \xe2\x86\x92 and a lone surrogate \\udc80\n")


class TestWhereTheLogGoes(_Folder):
    """The ``log`` argument: None is the default file in the temp folder, a path is that
    path, False is no log. A default that cannot be opened is said and the run goes on; a
    path that cannot be opened raises."""

    def test_the_path(self):
        default = os.path.join(tempfile.gettempdir(), "rig_run_tests.log")
        self.assertEqual(utils._log_path(None), default)
        self.assertEqual(utils._log_path(True), default)
        self.assertIsNone(utils._log_path(False))
        self.assertEqual(utils._log_path(self.path), self.path)
        self.assertEqual(utils._log_path(pathlib.Path(self.path)), self.path)

    def test_a_path(self):
        stream = io.StringIO()
        log    = utils._open_log(self.path, stream)
        log.close()
        self.assertEqual(log.path,          self.path)
        self.assertEqual(stream.getvalue(), f"run_tests log: {self.path}\n")
        self.assertEqual(_lines(self.path), [""])

    def test_the_default(self):
        stream = io.StringIO()
        with mock.patch("tempfile.gettempdir", return_value=self.folder):
            log = utils._open_log(None, stream)
        log.close()
        path = os.path.join(self.folder, "rig_run_tests.log")
        self.assertEqual(stream.getvalue(), f"run_tests log: {path}\n")
        self.assertTrue(os.path.isfile(path))

    def test_no_log(self):
        stream = io.StringIO()
        self.assertIsNone(utils._open_log(False, stream))
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(os.listdir(self.folder), [])

    def test_a_path_that_cannot_be_opened_raises(self):
        stream = io.StringIO()
        with self.assertRaises(OSError):
            utils._open_log(os.path.join(self.folder, "none", "run.log"), stream)
        self.assertEqual(stream.getvalue(), "")

    def test_a_default_that_cannot_be_opened_is_said(self):
        stream = io.StringIO()
        with mock.patch("tempfile.gettempdir", return_value=os.path.join(self.folder, "none")):
            self.assertIsNone(utils._open_log(None, stream))
        self.assertRegex(stream.getvalue(), r"^run_tests log: none \(.+\)\n$")

    def test_the_log_of_the_run_this_one_is_inside(self):
        outer = utils._Log(self.path)
        self.addCleanup(outer.close)
        outer.writeln("the outer run")
        with utils._faults_to(outer):
            stream = io.StringIO()
            with self.assertRaisesRegex(OSError, "is the log of the run this one is inside"):
                utils._open_log(self.path, stream)
            with mock.patch("tempfile.gettempdir", return_value=self.folder), mock.patch.object(
                utils, "_DEFAULT_LOG", "run.log"
            ):
                self.assertIsNone(utils._open_log(None, stream))
            self.assertRegex(stream.getvalue(), r"^run_tests log: none \(.+ is the log of the run this one is inside\)\n$")
        self.assertEqual(_lines(self.path), ["the outer run", ""])


def _stderr_has_a_file():
    try:
        sys.__stderr__.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    return True


class TestTheFaultHandler(_Folder):
    """`_faults_to`: Python's fault handler writes to the log inside the block, and is as it
    was after it."""

    def setUp(self):
        super().setUp()
        was_on, outer = faulthandler.is_enabled(), vars(utils)["_running_log"]

        def put_back():
            utils._running_log = outer
            faulthandler.disable()
            if outer is not None:
                faulthandler.enable(file=outer, all_threads=True)
            elif was_on and _stderr_has_a_file():
                faulthandler.enable(file=sys.__stderr__, all_threads=True)

        self.log = utils._Log(self.path)
        self.addCleanup(self.log.close)
        self.addCleanup(put_back)             # first, so the handler is never on a closed log
        # as a run not inside another finds it
        utils._running_log = None

    def test_on_the_log_inside_the_block(self):
        faulthandler.disable()
        with utils._faults_to(self.log):
            self.assertTrue(faulthandler.is_enabled())
            self.assertIs(vars(utils)["_running_log"], self.log)
        self.assertIsNone(vars(utils)["_running_log"])

    def test_off_before_is_off_after(self):
        faulthandler.disable()
        with utils._faults_to(self.log):
            pass
        self.assertFalse(faulthandler.is_enabled())

    def test_on_before_is_on_after(self):
        other = utils._Log(os.path.join(self.folder, "other.log"))
        self.addCleanup(other.close)
        faulthandler.enable(file=other)
        with utils._faults_to(self.log):
            pass
        # on Python's own default; off only where that has no file to write to
        self.assertEqual(faulthandler.is_enabled(), _stderr_has_a_file())

    def test_off_after_an_exception(self):
        faulthandler.disable()
        with self.assertRaises(ZeroDivisionError), utils._faults_to(self.log):
            1 // 0
        self.assertFalse(faulthandler.is_enabled())
        self.assertIsNone(vars(utils)["_running_log"])

    def test_a_run_inside_a_run_hands_back_to_the_outer_log(self):
        faulthandler.disable()
        inner = utils._Log(os.path.join(self.folder, "inner.log"))
        self.addCleanup(inner.close)
        with utils._faults_to(self.log):
            with utils._faults_to(inner):
                self.assertIs(vars(utils)["_running_log"], inner)
            self.assertIs(vars(utils)["_running_log"], self.log)
            self.assertTrue(faulthandler.is_enabled())
        self.assertFalse(faulthandler.is_enabled())

    def test_no_log_touches_nothing(self):
        for on in (False, True):
            with self.subTest(on=on):
                faulthandler.disable()
                if on:
                    faulthandler.enable(file=self.log)
                with utils._faults_to(None):
                    self.assertEqual(faulthandler.is_enabled(), on)
                    self.assertIsNone(vars(utils)["_running_log"])
                self.assertEqual(faulthandler.is_enabled(), on)


class _NoChunk(UndoWalk, MayaTestCase):
    """Starts with no named undo chunk open (a runner other than `run_tests` may have left
    one), and puts back what it closed."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        found = utils._close_chunks()
        self.addCleanup(utils._reopen_chunks, *found)

    @staticmethod
    def chunk():
        return cmds.undoInfo(query=True, chunkName=True)

    def open_chunk(self, name=None):
        if name is None:
            cmds.undoInfo(openChunk=True)
        else:
            cmds.undoInfo(openChunk=True, chunkName=name)
        self.addCleanup(cmds.undoInfo, closeChunk=True)


class TestTheUndoChunk(_NoChunk):
    """`_close_chunks` / `_reopen_chunks`: the named chunks a call runs inside."""

    def test_none_open(self):
        self.assertEqual(utils._close_chunks(), ("", 0))
        utils._reopen_chunks("", 0)
        self.assertEqual(self.chunk(), "")

    def test_a_named_chunk(self):
        self.open_chunk("mine")
        self.assertEqual(self.chunk(),          "mine")
        self.assertEqual(utils._close_chunks(), ("mine", 1))
        self.assertEqual(self.chunk(),          "")
        utils._reopen_chunks("mine", 1)
        self.assertEqual(self.chunk(), "mine")

    def test_a_chunk_inside_a_chunk(self):
        self.open_chunk("outer")
        self.open_chunk("inner")
        self.assertEqual(self.chunk(),          "outer")
        self.assertEqual(utils._close_chunks(), ("outer", 2))
        self.assertEqual(self.chunk(),          "")
        utils._reopen_chunks("outer", 2)
        self.assertEqual(self.chunk(), "outer")
        cmds.undoInfo(closeChunk=True)
        self.assertEqual(self.chunk(), "outer")
        cmds.undoInfo(closeChunk=True)
        self.assertEqual(self.chunk(), "")
        utils._reopen_chunks("outer", 2)

    def test_a_chunk_with_no_name_cannot_be_seen(self):
        self.open_chunk()
        self.assertEqual(self.chunk(), "")
        self.assertEqual(utils._close_chunks(), ("", 0))

    def test_an_edit_is_an_undo_step_of_its_own_once_the_chunk_is_closed(self):
        node = cmds.createNode("transform", name="n")
        self.open_chunk("mine")
        found = utils._close_chunks()
        cmds.setAttr(node + ".tx", 1.0)
        cmds.setAttr(node + ".tx", 2.0)
        cmds.undo()
        self.assertEqual(cmds.getAttr(node + ".tx"), 1.0)
        utils._reopen_chunks(*found)
        self.assertEqual(self.chunk(), "mine")


class TestRunTests(_NoChunk):
    """`run_tests` on one small target: the log it writes, what a test sees, and what is put
    back."""

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_run_tests_log_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.path = os.path.join(self.folder, "run.log")
        del _SEEN[:]

    def _run_tests(self, *args, **kwargs):
        """(the result, what the run printed) of ``run_tests(*args, **kwargs)``."""
        printed = io.StringIO()
        with contextlib.redirect_stderr(printed):
            result = utils.run_tests(*args, **kwargs)
        return result, printed.getvalue()

    def test_the_log_of_a_run(self):
        result, printed = self._run_tests(_TARGET, verbosity=0, log=self.path)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(printed.split("\n")[0], f"run_tests log: {self.path}")
        got = _lines(self.path)
        self.assertRegex(
            got[0],
            r"^run_tests\(%s\) of rig %s, \d{4}-\d\d-\d\d \d\d:\d\d:\d\d, "
            % (re.escape(repr(_TARGET)), re.escape(rig.__version__)),
        )
        self.assertTrue(got[0].endswith(", " + sys.executable))
        self.assertTrue(got[1].startswith("[1/1] test_what_a_test_sees "))
        self.assertTrue(got[1].endswith(" ... ok"))
        self.assertEqual(got[-3:], ["", "OK", ""])

    def test_the_default_log(self):
        with mock.patch("tempfile.gettempdir", return_value=self.folder):
            _result, printed = self._run_tests(_TARGET, verbosity=0)
        path = os.path.join(self.folder, "rig_run_tests.log")
        self.assertEqual(printed.split("\n")[0], f"run_tests log: {path}")
        self.assertEqual(_lines(path)[-3:], ["", "OK", ""])

    def test_no_log(self):
        result, printed = self._run_tests(_TARGET, verbosity=0, log=False)
        self.assertTrue(result.wasSuccessful())
        self.assertNotIn("run_tests log", printed)
        self.assertEqual(os.listdir(self.folder), [])

    def test_a_path_that_cannot_be_opened_raises_before_any_test(self):
        with self.assertRaises(OSError):
            self._run_tests(_TARGET, verbosity=0, log=os.path.join(self.folder, "none", "run.log"))
        self.assertEqual(_SEEN, [])

    def test_a_run_that_does_not_start_says_why_in_its_log(self):
        with self.assertRaises(ValueError):
            self._run_tests("test_fw_run_tests_log.TestNoSuchClass", verbosity=0, log=self.path)
        self.assertTrue(
            _lines(self.path)[-2].startswith("run_tests stopped: ValueError: could not resolve test target ")
        )

    def test_the_run_is_outside_the_chunk_it_was_called_in(self):
        self.open_chunk("mine")
        self._run_tests(_TARGET, verbosity=0, log=self.path)
        self.assertEqual([seen[0] for seen in _SEEN], [""])
        self.assertEqual(self.chunk(), "mine")

    def test_the_chunk_is_opened_again_when_the_run_does_not_start(self):
        self.open_chunk("mine")
        with self.assertRaises(ValueError):
            self._run_tests("test_fw_run_tests_log.TestNoSuchClass", verbosity=0, log=self.path)
        self.assertEqual(self.chunk(), "mine")

    def test_the_fault_handler_is_on_the_log_for_the_run(self):
        was_on, outer = faulthandler.is_enabled(), vars(utils)["_running_log"]
        self._run_tests(_TARGET, verbosity=0, log=self.path)
        self.assertEqual([seen[1:] for seen in _SEEN], [(self.path, True)])
        self.assertIs(vars(utils)["_running_log"], outer)
        if outer is not None or not was_on:
            self.assertEqual(faulthandler.is_enabled(), was_on)
        else:
            self.assertEqual(faulthandler.is_enabled(), _stderr_has_a_file())

    def test_without_a_log_the_fault_handler_is_not_touched(self):
        was_on, outer = faulthandler.is_enabled(), vars(utils)["_running_log"]
        self._run_tests(_TARGET, verbosity=0, log=False)
        self.assertEqual([seen[1:] for seen in _SEEN], [(None if outer is None else outer.path, was_on)])
        self.assertEqual(faulthandler.is_enabled(), was_on)
