"""The call-site check on each interpreter (user decision 2026-10-16: an 18-call-form test
on each interpreter; fix round 2026-10-16). ``call_site_forms.py`` extracts the check from
``rig/nodetypes/_base.py`` with no Maya and no rig import, so it also runs on Maya 2027's
mayapy (Python 3.13, the 3.12+ branch: LOAD_ATTR / LOAD_SUPER_ATTR with the method bit),
which cannot import rig yet. Each run is a subprocess; Maya 2027 is skipped when it is not
installed (``RIG_MAYAPY_2027`` names another mayapy)."""

import os
import subprocess
import sys
import unittest

import rig

HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "call_site_forms.py")
BASE   = os.path.join(os.path.dirname(os.path.abspath(rig.__file__)), "nodetypes", "_base.py")
MAYAPY_2027 = os.environ.get(
    "RIG_MAYAPY_2027", r"C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe"
)


def _run(python):
    env = dict(os.environ)
    env["MAYA_DISABLE_CER"] = "1"
    proc = subprocess.run(
        [python, "-I", SCRIPT, BASE], env=env, timeout=300,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


class TestCallSiteFormsOnEachInterpreter(unittest.TestCase):
    def _check(self, python, version_prefix):
        code, out = _run(python)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("all exact: True", out)
        self.assertTrue(out.startswith(version_prefix), out[:200])
        self.assertNotIn("WRONG", out)
        # every form ran (the 18 of the decision, plus the descriptor kinds)
        self.assertGreaterEqual(sum(line.endswith(" ok") for line in out.splitlines()), 21, out)
        return out

    def test_this_interpreter(self):
        out = self._check(sys.executable, "%d.%d" % sys.version_info[:2])
        if sys.version_info[:2] == (3, 11):
            self.assertIn("'LOAD_METHOD': 1", out)

    def test_maya_2027_python_3_13(self):
        if not os.path.isfile(MAYAPY_2027):
            self.skipTest(f"no Maya 2027 mayapy at {MAYAPY_2027}")
        out = self._check(MAYAPY_2027, "3.13")
        self.assertIn("'LOAD_ATTR': 2", out)
        self.assertIn("'LOAD_SUPER_ATTR': 2", out)


if __name__ == "__main__":
    unittest.main()
