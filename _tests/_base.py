"""
A unit test case base class that provides shared functionality for Maya unit tests.

Vendored from the original ``unittest`` test-harness module, with the two
environment helpers it needed (``is_standalone`` / ``initialize_standalone``)
inlined here, and ``own_interpreter`` for the tests that would need a child
interpreter, which the suite does not start: it runs in the Maya session it was
started in (the user's rule of 2026-11-10), and those tests are skipped.
They live in the test tree rather than in :mod:`rig` because nothing in the
shipped package uses them.

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
    return os.path.basename(sys.executable or "").startswith("mayapy")


def is_initialized() -> bool:
    """Check if maya standalone is initialzed in this session."""
    # uninitialized Maya has an empty cmds module
    return len([c for c in dir(cmds) if not c.startswith("__")]) > 0


def initialize_standalone() -> None:
    """Initialize maya standalone if not already."""
    if is_standalone() and not is_initialized():
        standalone.initialize()


def own_interpreter(why: str) -> None:
    """Skips the test that calls it: the test needs a Python of its own (`why`: ``"a raw Maya
    case a crash must not take the suite with"``, ``"an import before
    maya.standalone.initialize()"``), and the suite runs in the Maya session it was started
    in, GUI or mayapy, never starting a child (the user's rule of 2026-11-10: a child mayapy
    started from the GUI opened a second Maya and held the test, and a child of any kind is
    not the session under test). A test for another Maya version skips the same way, with
    ``unittest.TestCase.skipTest``."""
    raise unittest.SkipTest(f"needs an interpreter of its own ({why}); the suite runs in this Maya session only")


def default_shader() -> str:
    """The live default shader, the one ``initialShadingGroup`` (and ``initialParticleSE``)
    shows: ``standardSurface1`` up to Maya 2026, ``openPBR_shader1`` from 2027, whose
    ``standardSurface1`` is still in the scene, feeding no engine."""
    return "openPBR_shader1" if int(cmds.about(majorVersion=True)) >= 2027 else "standardSurface1"


def default_shader_class() -> type:
    """rig's class of `default_shader`: ``StandardSurface``, ``OpenPBRSurface`` from Maya 2027."""
    from rig.nodetypes import OpenPBRSurface, StandardSurface

    return OpenPBRSurface if int(cmds.about(majorVersion=True)) >= 2027 else StandardSurface


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

    def undo_on(self):
        """Turns undo on with an unlimited queue for this test, and puts back the
        state and queue it found once the test is done.

        A test must not leave undo off for the next one: with undo off, Maya frees
        a deleted dynamic attribute at once instead of keeping it in the queue, so
        a later test that holds a plug across `deleteAttr` sees another scene.
        """
        state    = cmds.undoInfo(query=True, state=True)
        infinity = cmds.undoInfo(query=True, infinity=True)
        length   = cmds.undoInfo(query=True, length=True)

        def put_back():
            if infinity:
                cmds.undoInfo(infinity=True)
            else:
                cmds.undoInfo(infinity=False, length=length)
            cmds.undoInfo(state=state)

        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(put_back)

    # -- assertion

    def assert_nodes_equal(self, a, b):
        """Checks if two nodes or two node lists are equivalent.

        This works for both node name strings and node objects.
        """
        a = [str(x) for x in a] if isinstance(a, (list, tuple)) else [str(a)]
        b = [str(x) for x in b] if isinstance(b, (list, tuple)) else [str(b)]
        self.assertEqual(sorted(a), sorted(b))

    def assert_one_node(self, call, node_type):
        """``call()`` builds one ``node_type`` node (a type name, or a tuple of the
        names one type has across Maya versions) and nothing else (a rig function
        that builds one node has no container of its own: no container, hyperLayout
        or network beside it), and every plug it returns (a plug, or a tuple of them)
        is that node's. Returns what ``call()`` returned."""
        types  = (node_type,) if isinstance(node_type, str) else tuple(node_type)
        before = set(cmds.ls())
        result = call()
        made   = sorted(set(cmds.ls()) - before)
        self.assertEqual(len(made), 1, f"made {made}")
        self.assertIn(cmds.nodeType(made[0]), types, f"made {made}")
        for plug in result if isinstance(result, tuple) else (result,):
            self.assertEqual(str(plug).split(".")[0], made[0], f"{plug} is not {made[0]}'s")
        return result

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
