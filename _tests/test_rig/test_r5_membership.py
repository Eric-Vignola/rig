"""Round 5, B0, and container membership through ``cmds.container``.

B0 (``rig.nodetypes.plugins``): `_register_api_edit` applies an ``MDGModifier`` and puts
it on the undo queue through ``rigUndoableAPICommand`` (outside a journal statement, one
unnamed undo step: undo is the modifier's ``undoIt``, redo its ``doIt``; round 5 D1
journals it, ``rig._internal.journal``); `_api_edit_ready` says whether it may run (the
command loads, Maya is not undoing or redoing). With undo off the edit is applied and
Maya keeps nothing.

C1 (eager API membership, 12f6921) is reverted (round 5 BCFIX): every member a scope
makes joins its container through ``cmds.container -edit -addNode -force``, as in
v2.0.0a4. `TestMembershipThroughCmds` pins what that command decides in the cases where
the review of C1 found that an API connect to the hyperLayout decides otherwise
(``runs\\r5\\review_bc_undo_membership``): the node types and the referenced nodes Maya
refuses in an asset (its warning, no element, a shared solver that survives the
container's delete); a hyperLayout disconnected between two statements (Maya makes a new
one); and rig's undo plug-in, which membership neither loads nor uses (an unload after a
build is not refused). An API route for members (round 5's phase D) must keep these.
"""

import os
import shutil
import tempfile
import weakref
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import container, Node
from rig._internal import journal
from rig.bridges import commands as rc
from rig.nodetypes import plugins, Transform
from rig.nodetypes.plugins import UNDOABLE_API_COMMAND, _api_edit_ready, _register_api_edit
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN = "undoable_api_command"


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _plug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _connect_modifier(src, dst):
    modifier = OpenMaya.MDGModifier()
    modifier.connect(_plug(src), _plug(dst))
    return modifier


def _layout(ctn):
    """The hyperLayout of the container ``ctn`` (its name), or None."""
    found = cmds.listConnections(f"{ctn}.hyperLayout", source=True, destination=False) or []
    return found[0] if found else None


def _elements(ctn):
    """``(numElements, [(index, member or None), ...])`` of ``ctn``'s hyperLayout."""
    layout = _layout(ctn)
    if layout is None:
        return None
    fn = OpenMaya.MFnDependencyNode(_mobject(layout))
    hyp, dn = fn.findPlug("hyperPosition", False), fn.attribute("dependNode")
    out = []
    for i in sorted(hyp.getExistingArrayAttributeIndices()):
        src = hyp.elementByLogicalIndex(i).child(dn).source()
        out.append((i, None if src.isNull else OpenMaya.MFnDependencyNode(src.node()).name()))
    return hyp.numElements(), out


def _owner(name):
    """The container of the node ``name`` (``findContainer``), None when it has none."""
    return cmds.container(query=True, findContainer=[name]) or None


class _Warnings:
    """The warnings Maya's commands print while active (``MCommandMessage``)."""

    def __enter__(self):
        self.seen = []
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(self._on_output)
        return self

    def _on_output(self, message, kind, *args):
        if kind == OpenMaya.MCommandMessage.kWarning:
            self.seen.append(message.strip())

    def __exit__(self, *exc):
        OpenMaya.MMessage.removeCallback(self._callback)
        return False


class _Calls:
    """Counts the ``cmds.loadPlugin`` / ``cmds.pluginInfo`` calls made while active."""

    def __enter__(self):
        self.counts = {"loadPlugin": 0, "pluginInfo": 0}
        self._patches = []
        for name in self.counts:
            real = getattr(cmds, name)

            def counted(*args, _name=name, _real=real, **kwargs):
                self.counts[_name] += 1
                return _real(*args, **kwargs)

            self._patches.append(mock.patch.object(cmds, name, counted))
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False


class _MembershipCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed; rig's plug-in loaded."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        cmds.flushUndo()

    def tearDown(self):
        try:
            while container._stack:
                container.__exit__(None, None, None)
            cmds.flushUndo()
        finally:
            super().tearDown()


class TestRegisterApiEdit(_MembershipCase):
    """B0: one unnamed step; the readiness invariants (the command unloaded, a load
    failure, undo off, during undo and during redo); a failing doIt registers nothing."""

    def setUp(self):
        super().setUp()
        cmds.createNode("multiplyDivide", name="a")
        cmds.createNode("multiplyDivide", name="b")
        cmds.flushUndo()

    def connected(self):
        return cmds.isConnected("a.outputX", "b.input1X")

    def test_one_unnamed_step_undone_and_redone(self):
        self.assertTrue(_api_edit_ready())
        _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        self.assertTrue(self.connected())
        self.assertEqual(self.undo_name(), "")
        self.undo_steps(1)
        self.assertFalse(self.connected())
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertTrue(self.connected())

    def test_inside_a_chunk_the_step_joins_it(self):
        cmds.undoInfo(openChunk=True)
        try:
            cmds.setAttr("a.input1X", 3.0)
            _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        finally:
            cmds.undoInfo(closeChunk=True)
        self.undo_steps(1)
        self.assertEqual((self.connected(), cmds.getAttr("a.input1X")), (False, 0.0))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_undo_off_applies_and_keeps_nothing(self):
        kept = []

        # the object registered: the journal of the edit (round 5 D1 retired _APIEdit)
        class Tracked(journal._Journal):
            __slots__ = ("__weakref__",)

            def __init__(self, ops):
                super().__init__(ops)
                kept.append(weakref.ref(self))

        cmds.undoInfo(state=False)
        try:
            self.assertTrue(_api_edit_ready())
            with mock.patch.object(journal, "_Journal", Tracked):
                _register_api_edit(_connect_modifier("a.outputX", "b.input1X"))
        finally:
            cmds.undoInfo(state=True)
        self.assertTrue(self.connected())
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0]())  # Maya kept neither the step nor the modifier
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_the_command_unloaded_is_loaded_again(self):
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))
        self.assertTrue(_api_edit_ready())
        self.assertTrue(hasattr(cmds, UNDOABLE_API_COMMAND))

    def test_a_load_failure_is_not_ready(self):
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        with mock.patch.object(cmds, "loadPlugin", return_value=None):
            self.assertFalse(_api_edit_ready())  # "did not register": no exception
        # the state a forced unload leaves: loaded, no maya.cmds function
        self.assertTrue(_api_edit_ready())
        wrapper = getattr(cmds, UNDOABLE_API_COMMAND)
        delattr(cmds, UNDOABLE_API_COMMAND)
        try:
            with mock.patch.object(cmds, "loadPlugin") as load:
                self.assertFalse(_api_edit_ready())
            load.assert_not_called()
        finally:
            setattr(cmds, UNDOABLE_API_COMMAND, wrapper)

    def test_not_ready_during_undo_and_redo(self):
        seen = []

        class Spy:
            def doIt(self):
                seen.append(("doIt", _api_edit_ready()))

            def undoIt(self):
                seen.append(("undoIt", _api_edit_ready()))

            def redoIt(self):
                seen.append(("redoIt", _api_edit_ready()))

        plugins._run_undoable(Spy())
        cmds.undo()
        cmds.redo()
        self.assertEqual(seen, [("doIt", True), ("undoIt", False), ("redoIt", False)])

    def test_a_failing_doit_registers_nothing(self):
        cmds.setAttr("b.input1Y", lock=True)
        cmds.flushUndo()
        modifier = _connect_modifier("a.outputX", "b.input1X")
        modifier.connect(_plug("a.outputY"), _plug("b.input1Y"))  # refused: locked
        with self.assertRaises(RuntimeError):
            _register_api_edit(modifier)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertTrue(self.connected())  # what doIt did before it failed ...
        modifier.undoIt()
        self.assertFalse(self.connected())  # ... the caller undoes


class TestMembershipThroughCmds(_MembershipCase):
    """What ``cmds.container -addNode`` decides for a scope's new nodes, where an API
    connect to the hyperLayout decides otherwise (the round 5 review of C1)."""

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_r5_membership_")

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)

    def two_members(self):
        container.createNode("multiplyDivide", name="a")
        container.createNode("multiplyDivide", name="b")

    def assert_only_the_two_members(self, ctn="c"):
        self.assertEqual(cmds.container(ctn, query=True, nodeList=True), ["a", "b"])
        self.assertEqual(_elements(ctn), (2, [(0, "a"), (1, "b")]))

    def walk_back_and_forth(self):
        """Undo to the empty queue and redo it all (the dangling check after each step);
        True when the redo gives the membership back as it was."""
        before = {c: cmds.container(c, query=True, nodeList=True) or []
                  for c in cmds.ls(type="container")}
        self.undo_all()
        self.redo_all()
        after = {c: cmds.container(c, query=True, nodeList=True) or []
                 for c in cmds.ls(type="container")}
        return before == after

    def save_reference_source(self, name):
        cmds.createNode("multiplyDivide", name="refmd")
        path = os.path.join(self.folder, name).replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        return path

    def test_node_types_maya_refuses_are_not_members(self):
        makers = [
            ("container.createNode", "ikRPsolver", "mySolver",
             lambda t, n: container.createNode(t, name=n)),
            ("rc.createNode", "ikSplineSolver", "splineSolver",
             lambda t, n: rc.createNode(t, name=n)),
            ("Node.create", "ikSCsolver", "scSolver", lambda t, n: Node.create(t, name=n)),
            ("rc.createNode", "reference", "myRN", lambda t, n: rc.createNode(t, name=n)),
            ("rc.createNode", "resultCurveTimeToLinear", "rcurve",
             lambda t, n: rc.createNode(t, name=n)),
            ("container.createNode", "viewColorManager", "vcm",
             lambda t, n: container.createNode(t, name=n)),
        ]
        for path, node_type, name, make in makers:
            with self.subTest(path=path, node_type=node_type):
                cmds.file(new=True, force=True)
                cmds.flushUndo()
                with container("c"):
                    self.two_members()
                    with _Warnings() as warned:
                        node = make(node_type, name)
                self.assertEqual(str(node), name)
                self.assertIsNone(_owner(name))
                self.assert_only_the_two_members()
                self.assertIn(f"Skipping '{name}'. Node cannot be added to assets.", warned.seen)
                self.assertTrue(self.walk_back_and_forth())
                self.assertIsNone(_owner(name))
                self.assert_only_the_two_members()

    def test_a_scene_solver_made_in_a_scope_survives_the_containers_delete(self):
        cmds.select(clear=True)
        start = cmds.joint(position=(0, 0, 0))
        cmds.joint(position=(0, 1, 0))
        end = cmds.joint(position=(1, 2, 0))
        cmds.select(clear=True)
        cmds.flushUndo()
        with container("c"):
            self.two_members()
            if not cmds.ls(type="ikSplineSolver"):  # the usual script: make it if missing
                rc.createNode("ikSplineSolver")
            curve = rc.curve(degree=1, point=[(0, 0, 0), (0, 1, 0), (1, 2, 0)])
            rc.ikHandle(startJoint=start, endEffector=end, solver="ikSplineSolver",
                        curve=str(curve), createCurve=False)
        self.assertEqual(cmds.ls(type="ikSplineSolver"), ["ikSplineSolver1"])
        self.assertIsNone(_owner("ikSplineSolver1"))
        self.assertIn("ikHandle1", cmds.container("c", query=True, nodeList=True))
        cmds.flushUndo()
        cmds.delete("c")
        self.assertEqual(cmds.ls(type="ikSplineSolver"), ["ikSplineSolver1"])

    def test_a_deferred_reference_is_not_a_member(self):
        path = self.save_reference_source("deferred_src.ma")
        with container("c"):
            self.two_members()
            with _Warnings() as warned:
                rc.file(path, reference=True, deferReference=True, namespace="D")
        self.assertIsNone(_owner("DRN"))
        self.assert_only_the_two_members()
        self.assertIn("Skipping 'DRN'. Node cannot be added to assets.", warned.seen)

    def test_a_non_maya_format_reference_is_not_a_member(self):
        try:
            cmds.loadPlugin("objExport", quiet=True)
        except RuntimeError:
            self.skipTest("the objExport plug-in (the OBJ translator) does not load")
        path = os.path.join(self.folder, "tri.obj").replace("\\", "/")
        with open(path, "w") as handle:
            handle.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        with container("c"):
            self.two_members()
            with _Warnings() as warned:
                result = rc.file(path, reference=True, type="OBJ", namespace="O")
        referenced = cmds.referenceQuery(str(result), nodes=True) or []
        self.assertTrue(referenced)
        for name in referenced + ["ORN"]:
            self.assertIsNone(_owner(name), name)
        self.assert_only_the_two_members()
        self.assertIn("Skipping 'ORN'. Node cannot be added to assets.", warned.seen)
        for name in referenced:
            self.assertIn(
                f"Cannot add node '{name}' to asset because it is in a different file.", warned.seen
            )

    def test_a_hyperlayout_disconnected_mid_scope_is_replaced(self):
        with container("c"):
            self.two_members()
            old = _layout("c")
            cmds.disconnectAttr(f"{old}.message", "c.hyperLayout")  # a user's edit
            container.createNode("multiplyDivide", name="d")
            container.createNode("multiplyDivide", name="e")
        new = _layout("c")
        self.assertNotIn(new, (None, old))
        self.assertEqual(cmds.container("c", query=True, nodeList=True), ["d", "e"])
        self.assertEqual(_elements("c"), (2, [(0, "d"), (1, "e")]))
        self.assertEqual((_owner("d"), _owner("e")), ("c", "c"))
        self.assertEqual(cmds.getAttr(f"{old}.hyperPosition", multiIndices=True), [0, 1])
        self.assertTrue(self.walk_back_and_forth())
        self.assertEqual(_elements("c"), (2, [(0, "d"), (1, "e")]))

    def scope_of_every_creating_path(self):
        with container("c"):
            node = container.createNode("multiplyDivide", name="a")
            Transform.create(name="t")
            rc.polyCube(name="cube")
            node.input1X << node.outputX * 2
        return cmds.container("c", query=True, nodeList=True)

    def test_membership_neither_loads_nor_queries_rigs_plugin(self):
        cmds.flushUndo()
        cmds.unloadPlugin(PLUGIN)
        with _Calls() as calls:
            members = self.scope_of_every_creating_path()
        self.assertEqual(len(members), 6)  # a, t, the cube's 3 nodes, the product
        self.assertEqual(calls.counts, {"loadPlugin": 0, "pluginInfo": 0})
        self.assertFalse(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
        self.assertFalse(hasattr(cmds, UNDOABLE_API_COMMAND))

    def test_an_unload_after_a_build_is_not_refused(self):
        self.assertTrue(cmds.pluginInfo(PLUGIN, query=True, loaded=True))  # setUp loaded it
        self.scope_of_every_creating_path()
        cmds.unloadPlugin(PLUGIN)  # nothing of the build's on the queue uses it
        self.assertFalse(cmds.pluginInfo(PLUGIN, query=True, loaded=True))
