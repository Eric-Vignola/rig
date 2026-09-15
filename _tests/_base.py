"""
A unit test case base class that provides shared functionality for Maya unit tests.

Vendored from the original ``unittest`` test-harness module, with the two
environment helpers it needed (``is_standalone`` / ``initialize_standalone``)
inlined here. Both live in the test tree rather than in :mod:`rig` because
nothing in the shipped package uses them.

Deliberately NOT named ``unittest.py``: test runners put the test directory on
``sys.path``, and a local ``unittest`` module would shadow the stdlib one that
these tests import.
"""

import math
import os
import sys
import unittest
from itertools import chain
from numbers import Number

from maya import cmds, standalone


def is_standalone() -> bool:
    """Checking if is a Maya standalone session."""
    return os.path.basename(sys.executable).startswith("mayapy")


def is_initialized() -> bool:
    """Check if maya standalone is initialzed in this session."""
    # uninitialized Maya has an empty cmds module
    return len([c for c in dir(cmds) if not c.startswith("__")]) > 0


def initialize_standalone() -> None:
    """Initialize maya standalone if not already."""
    if is_standalone() and not is_initialized():
        standalone.initialize()


class MayaTestCase(unittest.TestCase):
    """
    Base Maya unit test base class.
    """

    TEST_CASE_START_NEW_SCENE = False
    TEST_START_NEW_SCENE      = False

    # -- static methods

    @staticmethod
    def new_scene():
        cmds.file(f=True, new=True)

    @staticmethod
    def is_standalone():
        return is_standalone()

    # -- setup and teardown

    @classmethod
    def setUpClass(cls):
        """Shared setup class funtionalities."""
        initialize_standalone()
        # creates a new scene on demand
        if cls.TEST_CASE_START_NEW_SCENE:
            cls.new_scene()

    @classmethod
    def tearDownClass(cls):
        """Shared tear down class funtionalities."""

    def setUp(self):
        """Shared setup functionailies."""
        if self.TEST_START_NEW_SCENE:
            self.new_scene()

    def tearDown(self):
        """Shared tear down functionailies."""

    # -- assertion

    def assert_nodes_equal(self, a, b):
        """Checks if two nodes or two node lists are equivalent.

        This works for both node name strings and node objects.
        """
        a = [str(x) for x in a] if isinstance(a, (list, tuple)) else [str(a)]
        b = [str(x) for x in b] if isinstance(b, (list, tuple)) else [str(b)]
        self.assertEqual(sorted(a), sorted(b))

    def assert_list_equal(self, a, b):
        """Checks if two lists are equivalent.

        This works for both Python lists and Maya API arrays.

        NOTE: returns a bool, it does not assert. Callers must wrap it in
        ``assertTrue``. Preserved as-is from the original harness.
        """
        if isinstance(a[0], (list, tuple)):
            a = list(chain.from_iterable(a))
        if isinstance(b[0], (list, tuple)):
            b = list(chain.from_iterable(b))

        if len(a) != len(b):
            return False

        for x, y in zip(a, b):
            if isinstance(x, Number):
                if not math.isclose(x, y, rel_tol=1e-04):
                    return False
            elif x != y:
                return False

        return True
