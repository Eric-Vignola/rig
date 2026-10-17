"""Features step 3: ``lock()`` / ``unlock()`` on nodes, plugs and List.

``x.lock()`` makes ``x.is_locked`` True and ``x.unlock()`` makes it False, or they raise
InjectionError and change nothing; an object already in the asked state writes nothing and
adds no undo step. ``x << lock`` / ``x << unlock`` are the same calls and return the same
object (the node, the plug, the List). A node locks through ``lockNode``, a plug through
``setAttr -lock``, a List checks every element first, locks the plugs and then the nodes (in
one ``lockNode``), unlocks the nodes and then the plugs, in one undo step named ``rig.lock`` /
``rig.unlock``.

Decision D3 (the user, 2026-10-15): a locked plug keeps refusing a value or a connection, even
the value it already holds, so a build that ends with ``<< lock`` raises at its first value on a
second run; the refusal names ``Plug('x.w').unlock()`` (and ``Node('x').unlock()`` first on a
locked node). A node's own lock does not block values and connections.

* `TestSpellings`: every spelling returns its left-hand side, chaining, ``is_locked`` on both
  levels, ``node.is_locked = x`` raises, the plug setter is ``lock()`` / ``unlock()``, a Maya
  attribute named ``lock`` (``animLayer.lock``) is reached through ``find_attr``.
* `TestNoOp`: a lock of a locked node, plug or List, an unlock of an unlocked one, a child under
  a locked parent: nothing on the undo queue.
* `TestRefusals`: a referenced node, a reference node's unlock, a member of a locked container,
  a plug of a locked node, a child held by a locked parent, an element that is no node or plug,
  ``node << hide``, a container locked in its own block: each raises, the scene unchanged and the
  undo queue empty.
* `TestList`: the order, one named undo step, undo / redo, a node met twice, all or nothing.
* `TestRigVerbsOnALockedNode`: rename, delete, set_parent, a new attribute, ``overwrite=True``,
  destroy, add / delete / rename_attr, a layer's and a shader's own delete / rename, astype: the
  one template, nothing changed; a re-declaration, values, connections and define still work.
* `TestReRun`: the D3 pins (a plug lock stops a re-run at its value; a node lock re-runs).
* `TestContainers`, `TestCleanup`, `TestDefaultNodesAndFiles`, `TestTheLockedPlugHint`.
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds

import rig
from rig import Float, InjectionError, List, Node, Plug
from rig._internal import journal
from rig.nodetypes import Blinn, DisplayLayer, Transform
from rig.spec import destroy, hide, lock, skip, unhide, unlock
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _empty():
    return cmds.undoInfo(query=True, undoQueueEmpty=True)


def _node_locks(*names):
    return [cmds.lockNode(n, query=True, lock=True)[0] for n in names]


def _plug_locks(*names):
    return [cmds.getAttr(n, lock=True) for n in names]


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_refused(self, call, *texts, error=InjectionError):
        """``call()`` raises ``error`` whose text holds every one of ``texts``, with the
        scene as it was (nodes, connections, every lock) and nothing on the undo queue."""
        cmds.flushUndo()
        before = self.scene_state(), self._all_locks()
        with self.assertRaises(error) as ctx:
            call()
        for text in texts:
            self.assertIn(text, str(ctx.exception))
        self.assertEqual((self.scene_state(), self._all_locks()), before)
        self.assertTrue(_empty(), cmds.undoInfo(query=True, undoName=True))
        return ctx.exception

    @staticmethod
    def _all_locks():
        nodes = cmds.ls() or []
        own   = dict(zip(nodes, cmds.lockNode(nodes, query=True, lock=True) or []))
        plugs = {}
        for name in nodes:
            if cmds.nodeType(name) in ("transform", "multiplyDivide", "network", "container"):
                for attr in cmds.listAttr(name, locked=True) or []:
                    plugs[f"{name}.{attr}"] = True
        return own, plugs


class TestSpellings(_Case):
    def test_a_node(self):
        ctl = Transform.create(name="ctl")
        self.assertIs(ctl.lock(), ctl)
        self.assertTrue(ctl.is_locked)
        self.assertEqual(_node_locks("ctl"), [True])
        self.assertIs(ctl.unlock(), ctl)
        self.assertFalse(ctl.is_locked)
        self.assertIs(ctl << lock, ctl)
        self.assertTrue(ctl.is_locked)
        self.assertIs(ctl << unlock, ctl)
        self.assertEqual(_node_locks("ctl"), [False])

    def test_a_plug(self):
        ctl = Transform.create(name="ctl")
        tx  = ctl.tx
        self.assertIs(tx.lock(), tx)
        self.assertTrue(tx.is_locked)
        self.assertIs(tx.unlock(), tx)
        self.assertFalse(tx.is_locked)
        self.assertIs(tx << lock, tx)
        self.assertEqual(_plug_locks("ctl.tx"), [True])
        self.assertIs(tx << unlock, tx)
        self.assertEqual(_plug_locks("ctl.tx"), [False])

    def test_a_list(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        nodes = List([a, b])
        self.assertIs(nodes.lock(), nodes)
        self.assertEqual(_node_locks("a", "b"), [True, True])
        self.assertEqual(list(nodes.is_locked), [True, True])
        self.assertIs(nodes.unlock(), nodes)
        self.assertIs(nodes << lock, nodes)
        self.assertEqual(_node_locks("a", "b"), [True, True])
        self.assertIs(nodes << unlock, nodes)
        self.assertEqual(_node_locks("a", "b"), [False, False])
        named = List(["a", "b"])
        self.assertIs(named.lock(), named)
        self.assertEqual(_node_locks("a", "b"), [True, True])

    def test_chaining(self):
        made = Transform.create(name="x").lock()
        self.assertEqual(made, Node("x"))
        self.assertTrue(made.is_locked)
        ctl = Transform.define("ctl").lock()
        self.assertIsInstance(ctl, Transform)
        self.assertTrue(ctl.is_locked)
        again = Transform.define("ctl")
        self.assertIs(again.lock(), again)
        lst = List([Transform.create(name="y")])
        self.assertIs(lst.lock().unlock(), lst)

    def test_the_node_has_no_setter(self):
        ctl = Transform.create(name="ctl")
        with self.assertRaises(AttributeError):
            ctl.is_locked = True
        self.assertFalse(ctl.is_locked)

    def test_the_plug_setter_is_lock_and_unlock(self):
        ctl = Transform.create(name="ctl")
        ctl.tx.is_locked = True
        self.assertTrue(cmds.getAttr("ctl.tx", lock=True))
        ctl.tx.is_locked = False
        self.assertFalse(cmds.getAttr("ctl.tx", lock=True))
        ctl.lock()
        with self.assertRaises(InjectionError):
            ctl.ty.is_locked = True
        self.assertFalse(cmds.getAttr("ctl.ty", lock=True))

    def test_a_maya_attribute_named_lock(self):
        # re-pinned (user decision 2026-10-16, name clashes): a static Maya attr
        # of that name: layer.lock is the plug, layer.lock() the method
        layer = Node.create("animLayer", name="AL")
        self.assertIsInstance(layer.lock, Plug)
        self.assertEqual(str(layer.lock), "AL.lock")
        layer.lock << True
        self.assertTrue(cmds.getAttr("AL.lock"))
        self.assertFalse(layer.is_locked)
        self.assertIs(layer.lock(), layer)
        self.assertTrue(layer.is_locked)
        layer.unlock()


class TestNoOp(_Case):
    def test_a_locked_node(self):
        ctl = Transform.create(name="ctl")
        ctl.lock()
        cmds.flushUndo()
        with mock.patch.object(cmds, "lockNode", wraps=cmds.lockNode) as probe:
            self.assertIs(ctl.lock(), ctl)
            self.assertIs(ctl << lock, ctl)
        self.assertEqual(probe.call_count, 0)
        self.assertTrue(_empty())

    def test_an_unlocked_node(self):
        ctl = Transform.create(name="ctl")
        cmds.flushUndo()
        self.assertIs(ctl.unlock(), ctl)
        self.assertIs(ctl << unlock, ctl)
        self.assertTrue(_empty())

    def test_a_plug(self):
        ctl = Transform.create(name="ctl")
        tx, ty = ctl.tx, ctl.ty
        tx.lock()
        cmds.flushUndo()
        self.assertIs(tx.lock(), tx)
        self.assertIs(tx << lock, tx)
        self.assertIs(ty.unlock(), ty)
        self.assertIs(ty << unlock, ty)
        self.assertTrue(_empty())

    def test_a_child_under_a_locked_parent(self):
        ctl = Transform.create(name="ctl")
        ctl.t.lock()
        cmds.flushUndo()
        ctl.tx.lock()
        self.assertTrue(_empty())
        # it got no own flag: the parent's unlock frees it
        ctl.t.unlock()
        self.assertFalse(cmds.getAttr("ctl.tx", lock=True))

    def test_a_list(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        List([a, b.tx]).lock()
        cmds.flushUndo()
        lst = List([a, b.tx])
        self.assertIs(lst.lock(), lst)
        self.assertIs(lst << lock, lst)
        other = List([b, a.ty])
        self.assertIs(other.unlock(), other)
        self.assertTrue(_empty())

    def test_an_empty_list(self):
        lst = List([])
        cmds.flushUndo()
        self.assertIs(lst.lock(), lst)
        self.assertTrue(_empty())


class TestRefusals(_Case):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = tempfile.mkdtemp(prefix="rig_lock_")
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="free", skipSelect=True)
        cmds.createNode("transform", name="held", skipSelect=True)
        cmds.lockNode("held", lock=True)
        cmds.file(rename=cls.source())
        cmds.file(save=True, force=True, type="mayaAscii")
        cmds.file(new=True, force=True)

    @classmethod
    def tearDownClass(cls):
        cmds.file(new=True, force=True)
        shutil.rmtree(cls.folder, ignore_errors=True)
        super().tearDownClass()

    @classmethod
    def source(cls):
        return os.path.join(cls.folder, "lock_ref_src.ma").replace("\\", "/")

    def reference(self):
        cmds.file(self.source(), reference=True, namespace="ref")
        cmds.flushUndo()

    def test_a_referenced_node(self):
        self.reference()
        free, held = Node("ref:free"), Node("ref:held")
        self.assertTrue(held.is_locked)
        self.assert_refused(free.lock, "'ref:free' is referenced", "Maya saves no lockNode edit")
        self.assert_refused(lambda: free << lock, "is referenced")
        self.assert_refused(held.unlock, "'ref:held' is referenced", "the unlock would be gone")
        mine = Transform.create(name="mine")
        self.assert_refused(lambda: List([mine, free]).lock(), "is referenced")
        self.assertFalse(mine.is_locked)
        # already in the asked state: nothing to refuse
        cmds.flushUndo()
        self.assertIs(held.lock(), held)
        self.assertIs(free.unlock(), free)
        self.assertTrue(_empty())
        # a plug lock on a referenced node is a reference edit, and allowed
        free.tx.lock()
        self.assertTrue(cmds.getAttr("ref:free.tx", lock=True))

    def test_a_reference_node(self):
        self.reference()
        rn = Node("refRN")
        self.assertTrue(rn.is_locked)
        self.assert_refused(rn.unlock, "'refRN' is a reference node, which Maya keeps locked")
        cmds.flushUndo()
        self.assertIs(rn.lock(), rn)
        self.assertTrue(_empty())

    def test_a_member_of_a_locked_container(self):
        add = Node.create("multiplyDivide", name="add1")
        cmds.container(name="ctn", addNode=["add1"], force=True)
        cmds.lockNode("ctn", lock=True)
        self.assertTrue(add.is_locked)
        self.assert_refused(
            add.unlock,
            "'add1' is locked by its container 'ctn'",
            "Node('ctn').unlock() unlocks it, and every member with it",
        )
        cmds.flushUndo()
        self.assertIs(add.lock(), add)
        self.assertTrue(_empty())
        # rig's verbs on it name the container
        self.assert_refused(lambda: add.rename("x"), "cannot rename 'add1': it is locked by its container 'ctn'; Node('ctn').unlock()")

    def test_a_plug_of_a_locked_node(self):
        ctl = Transform.create(name="a")
        ctl.ty.lock()
        ctl.lock()
        self.assert_refused(
            ctl.tx.lock,
            "'a.translateX' is on the locked node 'a', whose plugs Maya never locks: "
            "Node('a').unlock() unlocks it. Nothing was changed",
        )
        self.assert_refused(lambda: ctl.tx << lock, "whose plugs Maya never locks")
        self.assert_refused(ctl.ty.unlock, "whose plugs Maya never unlocks")
        self.assert_refused(lambda: ctl.t << [lock, skip, skip], "never locks")

    def test_a_child_held_by_a_locked_parent(self):
        ctl = Transform.create(name="a")
        ctl.t.lock()
        self.assert_refused(
            ctl.tx.unlock,
            "'a.translateX' is locked by its parent 'a.translate' (a locked parent locks its "
            "children): Plug('a.translate').unlock() unlocks it. Nothing was changed",
        )
        self.assert_refused(lambda: ctl.tx << unlock, "is locked by its parent")

    def test_a_multi_root_holds_its_elements(self):
        net = Node.create("network", name="net")
        net << Float("w", multi=True)
        element = net.w[0]
        element << 1.0
        element.lock()
        net.w.lock()
        self.assert_refused(
            element.unlock,
            "'net.w[0]' is locked by its parent 'net.w' (a locked parent locks its children): "
            "Plug('net.w').unlock() unlocks it",
        )
        net.w.unlock()
        self.assertIs(element.unlock(), element)
        self.assertFalse(cmds.getAttr("net.w[0]", lock=True))

    def test_an_element_that_is_no_node_or_plug(self):
        a = Transform.create(name="a")
        self.assert_refused(
            lambda: List([a, 3]).lock(),
            "element [1] (3) is not a node or a plug: it has no lock",
            error=TypeError,
        )
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        self.assert_refused(lambda: List([a, cube.vtx[0:2]]).lock(), "it has no lock", error=TypeError)

    def test_hide_on_a_node(self):
        a = Transform.create(name="a")
        for modifier in (hide, unhide):
            with self.subTest(modifier=modifier):
                self.assert_refused(
                    lambda: a << modifier,
                    "this modifier is for attributes: write node.<attr> << it "
                    "(a node takes lock / unlock)",
                    error=TypeError,
                )

    def test_a_container_in_its_own_block(self):
        with rig.container("arm") as arm:
            Node.create("multiplyDivide", name="md")
            for call in (arm.lock, lambda: Node("arm").lock(), lambda: arm << lock, lambda: List([arm]).lock()):
                with self.assertRaises(InjectionError) as ctx:
                    call()
                self.assertEqual(
                    str(ctx.exception),
                    "lock 'arm' after its with block: a locked container takes no new member. "
                    "Nothing was changed",
                )
                self.assertFalse(Node("arm").is_locked)
        self.assertIs(arm.lock(), arm)
        self.assertTrue(Node("md").is_locked)


class TestList(_Case):
    def test_the_order(self):
        a = Transform.create(name="a")
        # the node first in the List: its plug is still locked before it
        lst = List([a, a.tx])
        lst.lock()
        self.assertEqual(_node_locks("a"), [True])
        self.assertEqual(_plug_locks("a.tx"), [True])
        lst.unlock()
        self.assertEqual(_node_locks("a"), [False])
        self.assertEqual(_plug_locks("a.tx"), [False])
        List([a.tx, a]).lock()
        List([a.tx, a]).unlock()
        self.assertEqual((_node_locks("a"), _plug_locks("a.tx")), ([False], [False]))

    def test_one_named_undo_step(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        cmds.flushUndo()
        lst = List([a, a.tx, b, b.ry])
        lst.lock()
        self.assertEqual(self.undo_name(), "rig.lock")
        state = (_node_locks("a", "b"), _plug_locks("a.tx", "b.ry"))
        self.assertEqual(state, ([True, True], [True, True]))
        self.undo_steps(1)
        self.assertTrue(_empty())
        self.assertEqual((_node_locks("a", "b"), _plug_locks("a.tx", "b.ry")), ([False, False], [False, False]))
        self.redo_steps(1)
        self.assertEqual((_node_locks("a", "b"), _plug_locks("a.tx", "b.ry")), state)
        cmds.flushUndo()
        lst << unlock
        self.assertEqual(self.undo_name(), "rig.unlock")
        self.undo_steps(1)
        self.assertTrue(_empty())
        self.assertEqual((_node_locks("a", "b"), _plug_locks("a.tx", "b.ry")), state)

    def test_the_nodes_go_in_one_lock_node(self):
        names = [Transform.create(name=f"n{i}").name for i in range(5)]
        lst   = List(names + [Node("n0"), "n1"])
        with mock.patch.object(cmds, "lockNode", wraps=cmds.lockNode) as probe:
            lst.lock()
        edits = [c for c in probe.call_args_list if not c.kwargs.get("query") and not c.kwargs.get("q")]
        self.assertEqual(len(edits), 1)
        self.assertEqual(sorted(edits[0].args[0]), names)
        self.assertEqual(_node_locks(*names), [True] * 5)

    def test_all_or_nothing(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        b.lock()
        self.assert_refused(lambda: List([a, a.tx, b.tx]).lock(), "'b.translateX' is on the locked node 'b'")
        self.assertEqual((_node_locks("a"), _plug_locks("a.tx")), ([False], [False]))
        # an unlock of the node and its plug in one List: the node first, then the plug
        b.unlock()
        b.tx.lock()
        b.lock()
        lst = List([b.tx, b])
        lst.unlock()
        self.assertEqual((_node_locks("b"), _plug_locks("b.tx")), ([False], [False]))

    def test_a_list_broadcast_still_maps(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        List([a, b]).tx << lock
        self.assertEqual(_plug_locks("a.tx", "b.tx"), [True, True])
        self.assertEqual(list(List([a, b]).tx.is_locked), [True, True])
        a.s << [lock, skip, hide]
        self.assertEqual(_plug_locks("a.sx", "a.sy"), [True, False])
        self.assertFalse(cmds.getAttr("a.sz", keyable=True))


class TestRigVerbsOnALockedNode(_Case):
    def setUp(self):
        super().setUp()
        self.grp = Transform.create(name="grp")
        self.ctl = Transform.create(name="ctl")
        self.ctl << Float("w")
        self.ctl.lock()

    def template(self, verb, name="ctl"):
        return f"cannot {verb} '{name}': it is locked; Node('{name}').unlock() unlocks it. Nothing was changed"

    def test_the_verbs(self):
        ctl = self.ctl
        cases = (
            ("rename", lambda: ctl.rename("other"), "rename"),
            ("delete", ctl.delete, "delete"),
            ("set_parent", lambda: ctl.set_parent(self.grp), "reparent"),
            ("a new attribute", lambda: ctl << Float("new"), "add the attribute 'new' to"),
            ("overwrite=True", lambda: ctl << Float("w", overwrite=True), "replace the attribute 'w' of"),
            ("destroy", lambda: ctl << destroy("w"), "destroy the attribute 'w' of"),
            ("plug << destroy", lambda: ctl.w << destroy, "destroy the attribute 'w' of"),
            ("add_attr", lambda: ctl.add_attr("new", at="double"), "add the attribute 'new' to"),
            ("delete_attr", lambda: ctl.delete_attr("w"), "delete the attribute 'w' of"),
            ("rename_attr", lambda: ctl.rename_attr("w", "w2"), "rename the attribute 'w' of"),
        )
        for label, call, verb in cases:
            with self.subTest(case=label):
                error = self.assert_refused(call)
                self.assertEqual(str(error), self.template(verb))
                self.assertIsInstance(error, RuntimeError)
        self.assertTrue(cmds.attributeQuery("w", node="ctl", exists=True))
        self.assertEqual(cmds.listRelatives("ctl", parent=True), None)

    def test_a_list_spec_refuses_before_its_first_element(self):
        free = Transform.create(name="free")
        self.assert_refused(lambda: List([free, self.ctl]) << Float("new"), self.template("add the attribute 'new' to"))
        self.assertFalse(cmds.attributeQuery("new", node="free", exists=True))

    def test_what_still_works(self):
        ctl = self.ctl
        self.assertEqual(ctl << Float("w", max=10), Plug("ctl.w"))
        self.assertEqual(cmds.addAttr("ctl.w", query=True, maxValue=True), 10)
        ctl.w << 0.3
        ctl.tx << self.grp.ty
        self.assertEqual(cmds.listConnections("ctl.tx", plugs=True), ["grp.translateY"])
        ctl.tx << None
        self.assertEqual(Transform.define("ctl", tx=3), ctl)
        self.assertEqual(cmds.getAttr("ctl.tx"), 3)
        self.assertTrue(ctl.is_locked)

    def test_a_statement_of_sets_on_locked_nodes_is_one_step(self):
        # the journal's API route (a statement is one undo step for its API edits); the
        # cmds route gives each set a step of its own, as on unlocked nodes
        for api, steps in ((True, 1), (False, 2)):
            with self.subTest(api=api), mock.patch.object(journal, "API_EDITS", api):
                other = Transform.define("other").lock()
                cmds.flushUndo()
                List([self.ctl, other]).tx << 2
                self.assertEqual([cmds.getAttr("ctl.tx"), cmds.getAttr("other.tx")], [2, 2])
                self.undo_steps(steps)
                self.assertTrue(_empty())
                self.assertEqual([cmds.getAttr("ctl.tx"), cmds.getAttr("other.tx")], [0, 0])

    def test_a_layer_and_a_shader(self):
        layer = DisplayLayer.create(name="lay")
        layer.lock()
        self.assert_refused(layer.delete, self.template("delete", "lay"))
        self.assert_refused(lambda: layer.rename("lay2"), self.template("rename", "lay"))
        red = Blinn.create(name="red")
        red.lock()
        self.assert_refused(red.delete, self.template("delete", "red"))
        self.assert_refused(lambda: red.rename("blue"), self.template("rename", "red"))
        self.assert_refused(lambda: red.astype("phong"), self.template("convert", "red"))


class TestReRun(_Case):
    """D3: keep today's behaviour (the user, 2026-10-15)."""

    def test_a_plug_lock_stops_a_rerun_at_its_value(self):
        def build():
            ctl = Transform.define("ctl")
            ctl << Float("w") << 0.5 << lock
            return ctl

        build()
        self.assertTrue(cmds.getAttr("ctl.w", lock=True))
        error = self.assert_refused(build, "Cannot set/connect 'ctl.w': locked attribute(s) ctl.w")
        self.assertIn("Unlock first: Plug('ctl.w').unlock()", str(error))
        self.assertIn("even the one it holds", str(error))
        # unlock before re-running
        Node("ctl").w.unlock()
        build()
        self.assertEqual(cmds.getAttr("ctl.w"), 0.5)

    def test_a_node_lock_reruns(self):
        def build():
            ctl = Transform.define("ctl")
            ctl << Float("w")
            ctl.w << 0.5
            return ctl

        build().lock()
        ctl = build()
        cmds.flushUndo()
        self.assertIs(ctl.lock(), ctl)
        self.assertTrue(_empty())

    def test_a_modifier_on_a_spec_reruns(self):
        def build():
            return Blinn.define("skin", reflectivity=lock)

        build()
        cmds.flushUndo()
        build()
        self.assertTrue(cmds.getAttr("skin.reflectivity", lock=True))
        self.assertTrue(_empty())


class TestContainers(_Case):
    def test_a_scope_over_a_locked_container_makes_nothing(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        with rig.container("arm"):
            cmds.lockNode("arm", lock=True)
            try:
                nodes = sorted(cmds.ls())
                for label, call in (
                    ("a math node", lambda: a.tx + b.tx),
                    ("a typed create", lambda: Transform.create(name="inner")),
                    ("Node.create", lambda: Node.create("multiplyDivide", name="md")),
                    ("a nested container", lambda: rig.container("hand", preserve=True).__enter__()),
                ):
                    with self.subTest(case=label):
                        with self.assertRaises(InjectionError) as ctx:
                            call()
                        self.assertEqual(
                            str(ctx.exception),
                            "cannot add a node to 'arm': it is locked; Node('arm').unlock() "
                            "unlocks it. Nothing was changed",
                        )
                        self.assertEqual(sorted(cmds.ls()), nodes)
                outside = Transform.create(name="outside", container=False)
                with self.assertRaises(InjectionError):
                    rig.container.add(outside)
                self.assertIsNone(cmds.container(query=True, findContainer="outside"))
            finally:
                cmds.lockNode("arm", lock=False)

    def test_members(self):
        with rig.container("arm") as arm:
            md = Node.create("multiplyDivide", name="md")
        md.lock()
        arm.lock()
        self.assertTrue(md.is_locked)
        # Maya's rule (pinned): a container's unlock unlocks every member, its own lock too
        arm.unlock()
        self.assertFalse(md.is_locked)
        self.assertEqual(_node_locks("md"), [False])


class TestCleanup(_Case):
    def test_a_locked_empty_container_is_kept(self):
        cmds.container(name="e1")
        cmds.lockNode("e1", lock=True)
        cmds.container(name="e2")
        report = rig.cleanup()
        self.assertEqual(report["deleted_containers"], ["e2"])
        self.assertTrue(cmds.objExists("e1"))
        self.assertFalse(cmds.objExists("e2"))
        cmds.lockNode("e1", lock=False)


class TestDefaultNodesAndFiles(_Case):
    def test_default_nodes(self):
        for name in ("lambert1", "initialShadingGroup", "time1", "persp", "defaultLightSet"):
            with self.subTest(node=name):
                node = Node(name)
                try:
                    self.assertIs(node.lock(), node)
                    self.assertTrue(node.is_locked)
                    self.assertIs(node.unlock(), node)
                    self.assertFalse(node.is_locked)
                finally:
                    cmds.lockNode(name, lock=False)

    def test_a_saved_lock_comes_back(self):
        folder = tempfile.mkdtemp(prefix="rig_lock_")
        try:
            Transform.create(name="ctl").lock()
            Transform.create(name="free").tx.lock()
            path = os.path.join(folder, "locked.ma").replace("\\", "/")
            cmds.file(rename=path)
            cmds.file(save=True, force=True, type="mayaAscii")
            cmds.file(new=True, force=True)
            cmds.file(path, open=True, force=True)
            self.assertTrue(Node("ctl").is_locked)
            self.assertTrue(Node("free").tx.is_locked)
            self.assertFalse(Node("free").is_locked)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


class TestTheLockedPlugHint(_Case):
    def test_the_hint_names_the_unlock(self):
        a = Transform.create(name="a")
        a.tx.lock()
        error = self.assert_refused(lambda: a.tx << 1)
        self.assertEqual(
            str(error),
            "Cannot set/connect 'a.translateX': locked attribute(s) a.translateX. Unlock first: "
            "Plug('a.translateX').unlock() (a locked plug takes no value or connection, even "
            "the one it holds).",
        )
        self.assertNotIn("cmds.setAttr", str(error))
        a.ty.lock()
        error = self.assert_refused(lambda: a.t << [1, 2, 3])
        self.assertIn("Unlock first: List(['a.translateX', 'a.translateY']).unlock()", str(error))
        a.lock()
        error = self.assert_refused(lambda: a.tx << 1)
        self.assertIn("Unlock first: Node('a').unlock(), then Plug('a.translateX').unlock()", str(error))

    def test_the_api_path_on_a_locked_node(self):
        a, b = Transform.create(name="a"), Transform.create(name="b")
        a.lock()
        cmds.flushUndo()
        a.tx << 5
        a.ty << b.tx
        self.assertEqual(cmds.getAttr("a.tx"), 5)
        self.assertEqual(cmds.listConnections("a.ty", plugs=True), ["b.translateX"])
        self.undo_steps(2)
        self.assertEqual(cmds.getAttr("a.tx"), 0)
        self.assertEqual(cmds.listConnections("a.ty", plugs=True), None)
        self.assertTrue(a.is_locked)
