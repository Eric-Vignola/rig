"""The test harness itself, ``rig._tests._base``.

* `TestOwnInterpreter`: the suite runs in the Maya session it was started in, GUI or mayapy,
  and starts no child (the user's rule of 2026-11-10; a child mayapy started from the GUI
  opened a second Maya and held the test, 2026-11-09, and 70896c8's ``child_mayapy`` is
  gone): a test that needs an interpreter of its own is skipped through `own_interpreter`,
  naming why, and no test module starts a subprocess.
"""

import os
import re
import unittest

from rig._tests import _base
from rig._tests._base import own_interpreter

HERE  = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)


class TestOwnInterpreter(unittest.TestCase):
    def test_it_skips_naming_why(self):
        with self.assertRaises(unittest.SkipTest) as caught:
            own_interpreter("a raw Maya case")
        self.assertEqual(
            str(caught.exception),
            "needs an interpreter of its own (a raw Maya case); the suite runs in this Maya session only",
        )

    def test_the_harness_starts_no_child(self):
        self.assertFalse(hasattr(_base, "child_mayapy"))
        self.assertNotIn("subprocess", open(_base.__file__, encoding="utf-8").read())

    def test_no_test_module_starts_a_subprocess(self):
        """A sweep over the test tree: a module that runs a subprocess would be starting an
        interpreter of its own (this one's call is a string, not a call)."""
        found = []
        for folder, dirs, files in os.walk(TESTS):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                path = os.path.join(folder, name)
                if not name.endswith(".py") or os.path.samefile(path, __file__):
                    continue
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                if re.search(r"subprocess\.(run|Popen|call|check_output|check_call)\(", text):
                    found.append(os.path.relpath(path, TESTS))
        self.assertEqual(found, [])
