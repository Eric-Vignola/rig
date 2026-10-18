"""``rig.run`` in Maya: rig and cgmath shared, one undo step per run, options given back,
callbacks a run made keep their modules.

* `TestShared`: a run's rig is the session's rig (``isinstance`` holds, no callback owner
  replaced); cgmath is shared even when the file imports it, so a session ``serialize()``
  after a run still works (the evaluation's cgmath poison: ``KeyError: '_scale'``).
* `TestUndo`: a copy of ``examples/rail_spine.py`` with a typo after the build raises
  NameError, and ONE undo goes back to the start; a run inside a run joins the outer step.
* `TestOptions`, `TestCallbacks`.
"""

import os
import shutil
import sys
import tempfile
import textwrap

from maya import cmds
from maya.api import OpenMaya

import rig
from rig._internal import run as run_module
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _write(folder, name, text):
    path = os.path.join(folder, name)
    with open(path, "w", newline="\n") as fh:
        fh.write(textwrap.dedent(text).lstrip("\n"))
    return path


class _Folder(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rigrunm_")
        self.addCleanup(shutil.rmtree, self.folder, True)


class TestShared(_Folder):
    def test_rig_is_the_sessions(self):
        path = _write(self.folder, "build.py", """
            import rig
            from maya import cmds
            ROOT = rig.Node(cmds.createNode("transform", name="root"))
        """)
        modules = {n: m for n, m in sys.modules.items() if n.partition(".")[0] == "rig"}
        result = rig.run(path)
        self.assertIsInstance(result["ROOT"], rig.Node)
        self.assertEqual((result.shared.get("rig"), result.owned, result.modules), ("rig", set(), {}))
        self.assertTrue(all(sys.modules[n] is m for n, m in modules.items()))

    def test_cgmath_poison_guard(self):
        for name, body in (("plain.py", ""), ("imports.py", "import cgmath, cgmath.hierarchy\n")):
            with self.subTest(file=name):
                cmds.file(new=True, force=True)
                path = _write(self.folder, name, body + textwrap.dedent("""
                    import rig
                    from maya import cmds
                    DATA = type(rig.Node(cmds.createNode("transform", name="inrun")).serialize()).__name__
                """))
                result = rig.run(path)
                self.assertEqual(result["DATA"], "TransformData")
                self.assertNotIn("cgmath", result.owned)
                after = rig.Node(cmds.createNode("transform", name="after")).serialize()
                self.assertEqual(type(after).__name__, "TransformData")
                cmds.file(new=True, force=True)
                again = rig.Node(cmds.createNode("transform", name="again")).serialize()
                self.assertEqual(type(again).__name__, "TransformData")


class TestUndo(UndoWalk, _Folder):
    def test_a_typo_in_a_rail_spine_copy_is_one_undo(self):
        source = os.path.join(os.path.dirname(rig.__file__), "examples", "rail_spine.py")
        with open(source) as fh:
            text = fh.read()
        line = "    rail.stretch         << 0\n"
        self.assertEqual(text.count(line), 1)
        path = _write(self.folder, "rail_spine_typo.py", text.replace(line, "    rail.stretch << stretch_rati0\n"))
        before = self.scene_state()
        with self.assertRaisesRegex(NameError, "stretch_rati0") as caught:
            rig.run(path)
        self.assertIn("rail", caught.exception.rig_run)                # the build ran before the typo
        self.assertGreater(len(cmds.ls()), len(before["nodes"]) + 100)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.run rail_spine_typo.py")
        self.undo_steps(1)
        self.assertEqual(self.scene_state(), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_run_inside_a_run_joins_the_outer_step(self):
        inner = _write(self.folder, "inner.py", 'from maya import cmds\ncmds.createNode("transform", name="inner")\n')
        outer = _write(self.folder, "outer.py", f"""
            import rig
            from maya import cmds
            cmds.createNode("transform", name="outer")
            rig.run({inner.replace(os.sep, "/")!r})
            cmds.createNode("transform", name="outer2")
        """)
        before = self.scene_state()
        rig.run(outer)
        self.assertTrue(cmds.objExists("inner") and cmds.objExists("outer2"))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.run outer.py")
        self.undo_steps(1)
        self.assertEqual(self.scene_state(), before)
        self.redo_steps(1)
        self.assertTrue(cmds.objExists("inner") and cmds.objExists("outer2"))


class TestOptions(_Folder):
    def test_rig_options_given_back(self):
        before = rig.get_options()
        path = _write(self.folder, "opts.py", """
            import rig
            rig.set_options(create_containers=False, constant_folding=False)
            SEEN = rig.get_options()["create_containers"]
        """)
        result = rig.run(path)
        self.assertFalse(result["SEEN"])
        self.assertEqual(rig.get_options(), before)
        failing = _write(self.folder, "opts_fail.py", "import rig\nrig.set_options(use_shorthand=False)\n1 / 0\n")
        with self.assertRaises(ZeroDivisionError):
            rig.run(failing)
        self.assertEqual(rig.get_options(), before)


class TestCallbacks(_Folder):
    def test_a_maya_callback_made_by_a_run_keeps_its_modules(self):
        lib = os.path.join(self.folder, "lib")
        os.makedirs(os.path.join(lib, "rrcblib"))
        _write(lib, "rrcblib/__init__.py", 'VALUE = "v1"\nclass Thing:\n    pass\n')
        sys.path.insert(0, lib)
        self.addCleanup(sys.path.remove, lib)
        path = _write(self.folder, "cb.py", """
            from maya.api import OpenMaya
            import rrcblib
            OBJ = rrcblib.Thing()
            SEEN = []
            def on_added(mobject, client):
                from rrcblib import Thing, VALUE
                SEEN.append((VALUE, isinstance(OBJ, Thing)))
            ID = OpenMaya.MDGMessage.addNodeAddedCallback(on_added, "transform")
        """)
        result = rig.run(path)
        self.addCleanup(OpenMaya.MMessage.removeCallback, result["ID"])
        _write(lib, "rrcblib/__init__.py", 'VALUE = "v2"\nclass Thing:\n    pass\n')
        cmds.createNode("transform")
        self.assertEqual(result["SEEN"], [("v1", True)])
        self.assertNotIn("rrcblib", sys.modules)
        self.assertEqual(run_module._ACTIVE, [])
