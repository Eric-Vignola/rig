"""Round 5, B0 and C1: API edits registered through rig's undo command, and eager API
container membership.

B0 (``rig.nodetypes.plugins``): `_register_api_edit` applies an ``MDGModifier`` inside
``rigUndoableAPICommand`` (one unnamed undo step: undo is the modifier's ``undoIt``, redo
its ``doIt``); `_api_edit_ready` says whether it may run (the command loads, Maya is not
undoing or redoing). With undo off the command applies the edit and Maya keeps nothing.

C1 (``rig._internal.container``): a node rig just created (``container.createNode``, a
typed create, a bridge command) joins the leaf real container by one
``MDGModifier.connect(node.message, hyperLayout.hyperPosition[k].dependNode)``, registered
through B0, once a first member joined through ``cmds.container`` (that makes the
hyperLayout). Every guard sends the add through ``cmds.container`` instead. The contract:
the same scene, element indices, file, membership as rig and ``maya.cmds`` read it, and
undo steps (only the step's name differs in the GUI) as with every member through
``cmds.container`` (``_EAGER_MEMBERSHIP`` off), which these tests compare against.

The index rule the guards rest on (``runs\\r5\\C1\\probe_rules*.txt``): ``cmds.container``
connects a member at the index after the highest one a member was ever connected at (an
element whose member was deleted, removed or undone stays, empty, and is skipped; an
element a user made by ``setAttr`` at that index is taken; removing the top elements does
not lower it). With the elements ``0 .. k-1`` and a member at ``k-1`` that index is ``k``.
"""

import os
import shutil
import tempfile
import weakref
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import container, Node
from rig._internal import container as container_module
from rig.bridges import commands as rc
from rig.nodetypes import plugins, Transform
from rig.nodetypes.plugins import UNDOABLE_API_COMMAND, _api_edit_ready, _register_api_edit
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, scene_connections

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


def _state():
    """What rig and ``maya.cmds`` can read of the scene and its membership: the nodes,
    every connection, each container's members in nodeList's order, its elements, each
    node's container (findContainer), Maya's current container."""
    ctns = sorted(cmds.ls(type="container") or [])
    nodes = sorted((n, cmds.nodeType(n)) for n in cmds.ls(long=True) or [])
    owners = {}
    for name, _ in nodes:
        found = cmds.container(query=True, findContainer=[name]) or None
        if found:
            owners[name] = found
    current = OpenMaya.MFnContainerNode.getCurrentAsMObject()
    return {
        "nodes": nodes,
        "connections": scene_connections(),
        "members": {c: cmds.container(c, query=True, nodeList=True) or [] for c in ctns},
        "elements": {c: _elements(c) for c in ctns},
        "owners": owners,
        "current": None if current.isNull() else OpenMaya.MFnDependencyNode(current).name(),
    }


class _Routes:
    """Counts the member adds by route while active: ``api`` (`_register_api_edit` from
    the container module) and ``cmds`` (``cmds.container -edit -addNode`` calls)."""

    def __init__(self):
        self.api = self.cmds = 0
        self._patches = []

    def __enter__(self):
        real_register, real_container = container_module._register_api_edit, cmds.container

        def register(modifier):
            real_register(modifier)
            self.api += 1

        def container_cmd(*args, **kwargs):
            if kwargs.get("edit") and kwargs.get("addNode") is not None:
                self.cmds += 1
            return real_container(*args, **kwargs)

        self._patches = [
            mock.patch.object(container_module, "_register_api_edit", register),
            mock.patch.object(cmds, "container", container_cmd),
        ]
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
            container_module._EAGER_MEMBERSHIP = True
            while container._stack:
                container.__exit__(None, None, None)
            cmds.flushUndo()
        finally:
            super().tearDown()

    def run_both(self, script, walk=True):
        """Run ``script()`` in a new scene with ``_EAGER_MEMBERSHIP`` on, then off. Returns
        ``(on, off)``: per side the routes, the state after the script, the undo steps
        back to the empty queue, the state at the end of a redo of all of them (with the
        dangling check after every step, ``walk``), and the script's own result."""
        out = []
        for switch in (True, False):
            cmds.file(new=True, force=True)
            cmds.flushUndo()
            container_module._EAGER_MEMBERSHIP = switch
            try:
                with _Routes() as routes:
                    result = script()
            finally:
                container_module._EAGER_MEMBERSHIP = True
            after = _state()
            steps = redone = None
            if walk:
                steps = self.undo_all()
                redone = (self.redo_all(), _state())
            out.append({"routes": (routes.api, routes.cmds), "state": after, "steps": steps,
                        "redone": redone, "result": result})
        return out

    def assert_same(self, on, off):
        self.assertEqual(on["state"], off["state"], "state after the script")
        self.assertEqual(on["result"], off["result"], "the script's result")
        self.assertEqual(on["steps"], off["steps"], "undo steps")
        self.assertEqual(on["redone"], off["redone"], "redo steps and the state after them")
        if on["redone"] is not None:
            self.assertEqual(on["redone"][1], on["state"], "redo all gives the scene back")


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

        class Tracked(plugins._APIEdit):
            __slots__ = ("__weakref__",)

            def __init__(self, modifier):
                super().__init__(modifier)
                kept.append(weakref.ref(self))

        cmds.undoInfo(state=False)
        try:
            self.assertTrue(_api_edit_ready())
            with mock.patch.object(plugins, "_APIEdit", Tracked):
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


class TestEagerMembership(_MembershipCase):
    """C1: the API route against every member through cmds, for each creating path."""

    def test_members_join_through_the_api_after_the_first(self):
        def script():
            with container("c1") as ctn:
                a = container.createNode("multiplyDivide", name="m")
                b = container.createNode("plusMinusAverage")
                t = Transform.create(name="t")
                s = Node.create("locator", name="locShape")
                cube = rc.polyCube(name="cube")  # transform, shape and polyCube: one add
                x = a.outputX * 2
                b.input1D[0] << x
                inner = []
                with container("inner"):  # flattened: the outer container's members
                    inner.append(str(container.createNode("multiplyDivide", name="n")))
            return [str(ctn), str(t), str(s), [str(c) for c in cube], inner]

        on, off = self.run_both(script)
        self.assert_same(on, off)
        # the first member through cmds, then one API add per create: the pma, t, the
        # locator, the cube's three nodes (one add), the product, the inner n
        self.assertEqual(on["routes"], (6, 1))
        self.assertEqual(off["routes"], (0, 7))
        members = on["state"]["members"]["c1"]
        self.assertEqual(members[:2], ["m", "plusMinusAverage1"])
        self.assertIn("inner_n", members)
        self.assertEqual(on["state"]["elements"]["c1"][0], len(members))

    def test_the_same_file(self):
        folder = tempfile.mkdtemp(prefix="rig_c1_")
        try:
            prints = []
            for switch in (True, False):
                cmds.file(new=True, force=True)
                container_module._EAGER_MEMBERSHIP = switch
                try:
                    with container("f1"):
                        loc = Node(cmds.spaceLocator(name="loc")[0])
                        m = loc.tx * 3
                        doomed = container.createNode("multiplyDivide", name="doomed")
                        cmds.delete(str(doomed))  # the last member deleted: an empty element
                        loc.ty << m + 1
                        container.add(Node(cmds.createNode("transform", name="outside")))
                        loc.tz << loc.tx * loc.ty
                        Transform.create(name="kid")
                finally:
                    container_module._EAGER_MEMBERSHIP = True
                path = os.path.join(folder, f"f_{switch}.ma").replace("\\", "/")
                cmds.file(rename=path)
                cmds.file(save=True, type="mayaAscii", force=True)
                with open(path, encoding="utf-8") as handle:
                    prints.append([line for line in handle.read().splitlines()
                                   if not line.strip().startswith(("rename -uid", "//", 'fileInfo "UUID"'))])
            self.assertEqual(prints[0], prints[1])
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_existing_nodes_join_through_cmds(self):
        def script():
            other = cmds.createNode("transform", name="other")
            with container("e1"):
                container.createNode("multiplyDivide")
                container.add(Node(other))
                container.add(cmds.createNode("transform", name="other2"))
                container.createNode("multiplyDivide")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["routes"], (1, 3))

    def test_the_index_after_an_orphan_and_after_a_remove(self):
        def script():
            with container("i1") as ctn:
                nodes = [container.createNode("multiplyDivide", name=f"n{i}") for i in range(4)]
                cmds.delete(str(nodes[3]))  # the last element stays, empty
                container.createNode("multiplyDivide", name="after_orphan")
                cmds.delete(str(nodes[1]))  # an empty element inside
                container.createNode("multiplyDivide", name="after_inner_orphan")
                cmds.container(str(ctn), edit=True, removeNode=["after_inner_orphan"])
                container.createNode("multiplyDivide", name="after_remove")
                cmds.container(str(ctn), edit=True, addNode=[cmds.createNode("transform", name="user")])
                container.createNode("multiplyDivide", name="after_user_add")
            return _elements("i1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(
            on["result"][1][3:],
            [(3, None), (4, "after_orphan"), (5, None), (6, "after_remove"), (7, "user"),
             (8, "after_user_add")],
        )
        # cmds: the first member, after an empty last element (twice) and the user's add
        self.assertEqual(on["routes"], (5, 4))

    def test_a_user_edit_of_the_elements_takes_cmds(self):
        def script():
            with container("u1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                layout = _layout("u1")
                cmds.setAttr(f"{layout}.hyp[2].isCollapsed", True)  # an element without a member
                container.createNode("multiplyDivide", name="third")  # cmds takes 2
                cmds.removeMultiInstance(f"{layout}.hyp[0]", b=True)  # a hole
                container.createNode("multiplyDivide", name="fourth")
            return _elements("u1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["result"][1], [(1, "second"), (2, "third"), (3, "fourth")])
        self.assertEqual(on["routes"], (1, 3))

    def test_a_user_current_container_takes_cmds(self):
        def script():
            mine = cmds.container(name="mine")
            with container("cur1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                cmds.container(mine, edit=True, current=True)
                try:
                    container.createNode("multiplyDivide", name="adopted")
                    Transform.create(name="adopted_t")
                finally:
                    cmds.container(mine, edit=True, current=False)
                container.createNode("multiplyDivide", name="third")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["state"]["owners"]["adopted"], "cur1")
        self.assertEqual(on["routes"], (2, 3))

    def test_a_current_container_set_after_the_create_takes_cmds(self):
        # the node was made before: Maya did not adopt it, the current container guard
        # alone refuses (a connect would leave it in two containers' views)
        mine = cmds.container(name="mine")
        with container("cur2"):
            container.createNode("multiplyDivide", name="first")
            frame = container._leaf_real_frame()
            node = container.createNode("multiplyDivide", name="probe", container=False)
            cmds.container(mine, edit=True, current=True)
            try:
                self.assertFalse(container._add_eager([node], frame))
            finally:
                cmds.container(mine, edit=True, current=False)
            self.assertFalse(container._add_eager([], frame))  # nothing to add: no empty step
            self.assertTrue(container._add_eager([node], frame))
        self.assertEqual(cmds.container("cur2", query=True, nodeList=True), ["first", "probe"])

    def test_the_first_members_add_undone_takes_cmds(self):
        # undoing the add that made the hyperLayout deletes it: the next member goes
        # through cmds (a new hyperLayout), which fills the slots again
        def script():
            with container("fu1"):
                container.createNode("multiplyDivide", name="first")
                layout = _layout("fu1")
                cmds.undo()  # the first member's add
                gone = not cmds.objExists(layout)
                container.createNode("multiplyDivide", name="second")
                container.createNode("multiplyDivide", name="third")
            return gone, _elements("fu1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertTrue(on["result"][0])
        self.assertEqual(on["routes"], (1, 2))

    def test_no_element_left_takes_cmds(self):
        def script():
            with container("ne1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                layout = _layout("ne1")
                for i in (0, 1):
                    cmds.removeMultiInstance(f"{layout}.hyp[{i}]", b=True)
                container.createNode("multiplyDivide", name="third")
                container.createNode("multiplyDivide", name="fourth")
            return _elements("ne1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["result"][1], [(2, "third"), (3, "fourth")])
        self.assertEqual(on["routes"], (1, 3))  # no element, then elements from 2 on: cmds

    def test_a_locked_container_takes_cmds(self):
        def script():
            with container("lk1") as ctn:
                container.createNode("multiplyDivide", name="first")
                cmds.lockNode(str(ctn), lock=True)
                try:
                    container.createNode("multiplyDivide", name="refused")  # cmds refuses
                finally:
                    cmds.lockNode(str(ctn), lock=False)
                container.createNode("multiplyDivide", name="second")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertNotIn("refused", on["state"]["members"]["lk1"])
        self.assertEqual(on["routes"], (1, 2))

    def test_the_hyperlayout_deleted_mid_scope_takes_cmds(self):
        def script():
            with container("hl1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                cmds.delete(_layout("hl1"))  # takes the container and its members along
                try:
                    container.createNode("multiplyDivide", name="after")
                    error = None
                except Exception as exc:  # noqa: BLE001 (the same on both routes)
                    error = (type(exc).__name__, str(exc), exc.__context__ is None)
            return cmds.objExists("hl1"), cmds.objExists("after"), error

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["result"][:2], (False, True))
        self.assertEqual(on["routes"][0], 1)

    def test_the_container_deleted_apart_from_its_hyperlayout_takes_cmds(self):
        # a user disconnects the hyperLayout, then deletes the container: the layout
        # lives on, the container does not (the cmds route's failure is logged; the
        # node joins nothing, and the stray layout gets no member)
        def script():
            with container("cd1") as ctn:
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                layout = _layout("cd1")
                cmds.disconnectAttr(f"{layout}.message", f"{ctn}.hyperLayout")
                cmds.delete(str(ctn))
                try:
                    container.createNode("multiplyDivide", name="after")
                    error = None
                except Exception as exc:  # noqa: BLE001 (the same on both routes)
                    error = (type(exc).__name__, str(exc))
            return cmds.objExists(layout), error

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["result"], (True, None))
        self.assertNotIn("after", on["state"]["owners"])

    def test_a_node_already_connected_by_message_takes_cmds(self):
        frame_args = {}

        def script():
            with container("msg1"):
                container.createNode("multiplyDivide", name="first")
                held = container.createNode("multiplyDivide", name="held")
                user = cmds.createNode("network", name="user")
                cmds.addAttr(user, longName="link", attributeType="message")
                fresh = cmds.createNode("multiplyDivide", name="fresh")
                cmds.connectAttr(f"{fresh}.message", f"{user}.link")
                frame = container_module.container._leaf_real_frame()
                frame_args["result"] = container._add_eager([Node(fresh)], frame)
                frame_args["member"] = container._add_eager([held], frame)
            return frame_args["result"], frame_args["member"]

        on, off = self.run_both(script, walk=False)
        self.assertEqual(on["result"], (False, False))

    def test_a_failing_registration_is_undone_and_takes_cmds(self):
        def script():
            with container("fr1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                real = container_module._register_api_edit

                def failing(modifier):
                    # the command fails before the modifier's doIt (it is gone: a
                    # forced unload). After a connect was made Maya's next index would
                    # be one higher (the index rule), but the connect is the last thing
                    # the command does, after every guard passed
                    raise RuntimeError("registration failed (test)")

                container_module._register_api_edit = failing
                try:
                    container.createNode("multiplyDivide", name="third")
                finally:
                    container_module._register_api_edit = real
                container.createNode("multiplyDivide", name="fourth")
            return _elements("fr1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["result"][1][2], (2, "third"))  # once, at cmds' index

    def test_the_plugin_unloaded_mid_scope_is_loaded_again(self):
        def script():
            with container("p1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                cmds.flushUndo()
                cmds.unloadPlugin(PLUGIN)  # a clean unload: the next add loads it again
                container.createNode("multiplyDivide", name="third")
            return hasattr(cmds, UNDOABLE_API_COMMAND)

        on, off = self.run_both(script)
        self.assertEqual((on["result"], off["result"]), (True, False))
        on["result"] = off["result"]
        self.assert_same(on, off)
        self.assertEqual(on["routes"], (2, 1))

    def test_the_forced_unload_state_takes_cmds(self):
        def script():
            with container("p2"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                wrapper = getattr(cmds, UNDOABLE_API_COMMAND)
                delattr(cmds, UNDOABLE_API_COMMAND)  # what a forced unload and a reload leave
                try:
                    container.createNode("multiplyDivide", name="third")
                    Transform.create(name="fourth")
                finally:
                    setattr(cmds, UNDOABLE_API_COMMAND, wrapper)
                container.createNode("multiplyDivide", name="fifth")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["routes"], (2, 3))

    def test_undo_off_registers_nothing(self):
        def script():
            with container("off1"):
                container.createNode("multiplyDivide", name="first")
                container.createNode("multiplyDivide", name="second")
                cmds.undoInfo(stateWithoutFlush=False)
                try:
                    container.createNode("multiplyDivide", name="unrecorded")
                    Transform.create(name="unrecorded_t")
                finally:
                    cmds.undoInfo(stateWithoutFlush=True)
                container.createNode("multiplyDivide", name="third")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertEqual(on["routes"], (4, 1))

    def test_not_taken_during_undo_or_redo(self):
        seen = []
        with container("ur1"):
            container.createNode("multiplyDivide", name="first")
            frame = container._leaf_real_frame()
            node = container.createNode("multiplyDivide", name="probe", container=False)

            class Spy:
                def doIt(self):
                    pass

                def undoIt(self):
                    seen.append(container._add_eager([node], frame))

                def redoIt(self):
                    seen.append(container._add_eager([node], frame))

            plugins._run_undoable(Spy())
            cmds.undo()
            cmds.redo()
            with mock.patch.object(container_module, "_api_edit_ready", return_value=False):
                seen.append(container._add_eager([node], frame))
            self.assertEqual(seen, [False, False, False])
            self.assertEqual(cmds.container("ur1", query=True, nodeList=True), ["first"])
            self.assertTrue(container._add_eager([node], frame))
            self.assertEqual(cmds.container("ur1", query=True, nodeList=True), ["first", "probe"])

    def test_undo_and_redo_of_one_member(self):
        with container("one1"):
            container.createNode("multiplyDivide", name="first")
            cmds.flushUndo()
            with _Routes() as routes:
                node = container.createNode("multiplyDivide", name="x")
            self.assertEqual((routes.api, routes.cmds), (1, 0))
            member = _state()
            self.assertIn("x", member["members"]["one1"])
            # createNode, the member: two steps (re-pinned, B1: 2 fewer undo steps
            # per GC tag, approved lever; the tag, written through the API, has none)
            self.undo_steps(1)
            self.assertNotIn("x", cmds.container("one1", query=True, nodeList=True))
            self.assertTrue(cmds.objExists("x"))
            self.undo_steps(1)
            self.assertFalse(cmds.objExists("x"))
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
            self.redo_steps(2)
            self.assertEqual(_state(), member)  # redo restores the membership
            self.assertEqual(cmds.container(query=True, findContainer=[str(node)]), "one1")

    def test_undo_and_redo_inside_the_block(self):
        def script():
            with container("ub1"):
                a = container.createNode("multiplyDivide", name="a")
                b = container.createNode("multiplyDivide", name="b")
                b.input1X << a.outputX * 2
                for _ in range(5):
                    cmds.undo()
                container.createNode("multiplyDivide", name="after_undo")
                cmds.undo()
                cmds.redo()
                for _ in range(3):
                    cmds.undo()
                cmds.redo()
                c = container.createNode("multiplyDivide", name="c")
                c.input1X << b.outputX + 1
            return _elements("ub1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
        self.assertGreater(on["routes"][0], 0)

    def test_a_member_deleted_before_the_next_statement(self):
        def script():
            with container("d1"):
                loc = Node(cmds.spaceLocator(name="loc")[0])
                m1 = loc.tx * 2
                gone = Node.create("multiplyDivide", name="gone")
                cmds.delete(str(gone))
                m2 = loc.ty * 3
                loc.tz << m1 + m2
            return _elements("d1")

        on, off = self.run_both(script)
        self.assert_same(on, off)
