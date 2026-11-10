"""The call-site check on this interpreter (user decision 2026-10-16: an 18-call-form test
on each interpreter; fix round 2026-10-16). ``call_site_forms.py`` extracts the check from
``rig/nodetypes/_base.py`` with no Maya and no rig import, so it also ran on Maya 2027's
mayapy (Python 3.13, the 3.12+ branch: LOAD_ATTR / LOAD_SUPER_ATTR with the method bit),
which cannot import rig yet. Each run was a subprocess until 2026-11-10; the suite runs in
the session it was started in now: the probe runs in this interpreter, and the Maya 2027
test runs when this is its Python (3.13) and is skipped elsewhere."""

import contextlib
import io
import os
import runpy
import sys
import unittest
from unittest import mock

import rig

HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "call_site_forms.py")
BASE   = os.path.join(os.path.dirname(os.path.abspath(rig.__file__)), "nodetypes", "_base.py")


def _run_here():
    """``(exit code, output)`` of ``call_site_forms.py`` over rig's ``_base.py``, run in this
    interpreter as ``mayapy -I call_site_forms.py <_base.py>`` ran it in a child."""
    buffer = io.StringIO()
    with mock.patch.object(sys, "argv", [SCRIPT, BASE]), contextlib.redirect_stdout(buffer):
        try:
            runpy.run_path(SCRIPT, run_name="__main__")
        except SystemExit as exc:
            return int(exc.code or 0), buffer.getvalue()
    return 0, buffer.getvalue()


class TestCallSiteFormsOnThisInterpreter(unittest.TestCase):
    def _check(self, version_prefix):
        code, out = _run_here()
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("all exact: True", out)
        self.assertTrue(out.startswith(version_prefix), out[:200])
        self.assertNotIn("WRONG", out)
        # every form ran (the 18 of the decision, plus the descriptor kinds)
        self.assertGreaterEqual(sum(line.endswith(" ok") for line in out.splitlines()), 21, out)
        return out

    def test_this_interpreter(self):
        out = self._check("%d.%d" % sys.version_info[:2])
        if sys.version_info[:2] == (3, 11):
            self.assertIn("'LOAD_METHOD': 1", out)
        if sys.version_info[:2] >= (3, 13):
            self.assertIn("'LOAD_ATTR': 2", out)
            self.assertIn("'LOAD_SUPER_ATTR': 2", out)

    def test_maya_2027_python_3_13(self):
        if sys.version_info[:2] != (3, 13):
            self.skipTest("Maya 2027's Python (3.13): run the suite under Maya 2027; this one is %d.%d" % sys.version_info[:2])
        out = self._check("3.13")
        self.assertIn("'LOAD_ATTR': 2", out)
        self.assertIn("'LOAD_SUPER_ATTR': 2", out)


if __name__ == "__main__":
    unittest.main()
