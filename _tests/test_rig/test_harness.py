"""The test harness itself, ``rig._tests._base``.

* `TestChildMayapy`: the interpreter a test starts as a child is a mayapy everywhere. Under
  mayapy it is this one. Inside Maya ``sys.executable`` is ``maya.exe``, which would open
  another Maya and hold the test until that Maya closes (18 tests did, 2026-11-09): the
  mayapy beside it is used, and none beside it skips the test.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from rig._tests._base import child_mayapy


class TestChildMayapy(unittest.TestCase):
    def _folder(self, *names):
        """A temporary folder holding empty files of these names."""
        folder = tempfile.mkdtemp(prefix="rig_harness_")
        self.addCleanup(shutil.rmtree, folder, True)
        for name in names:
            with open(os.path.join(folder, name), "w"):
                pass
        return folder

    def test_under_mayapy_it_is_this_interpreter(self):
        folder = self._folder()
        for name in ("mayapy.exe", "mayapy"):
            with self.subTest(name=name), mock.patch.object(sys, "executable", os.path.join(folder, name)):
                self.assertEqual(child_mayapy(), os.path.join(folder, name))

    def test_inside_maya_it_is_the_mayapy_beside_maya_exe(self):
        for maya, mayapy in (("maya.exe", "mayapy.exe"), ("maya", "mayapy")):
            with self.subTest(mayapy=mayapy):
                folder = self._folder(maya, mayapy)
                with mock.patch.object(sys, "executable", os.path.join(folder, maya)):
                    self.assertEqual(child_mayapy(), os.path.join(folder, mayapy))

    def test_no_mayapy_beside_it_skips(self):
        folder = self._folder("maya.exe")
        exe    = os.path.join(folder, "maya.exe")
        with mock.patch.object(sys, "executable", exe):
            with self.assertRaises(unittest.SkipTest) as caught:
                child_mayapy()
        self.assertIn(repr(exe), str(caught.exception))
        for unknown in ("", None):  # an interpreter that cannot name its own executable
            with self.subTest(executable=unknown), mock.patch.object(sys, "executable", unknown):
                with self.assertRaises(unittest.SkipTest):
                    child_mayapy()

    def test_this_session_has_one(self):
        """mayapy or Maya: the 20 tests that start a child find a mayapy here."""
        mayapy = child_mayapy()
        self.assertTrue(os.path.isfile(mayapy), mayapy)
        self.assertTrue(os.path.basename(mayapy).startswith("mayapy"), mayapy)
