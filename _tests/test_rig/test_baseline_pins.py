"""Baseline pins for the ``<<`` / ``>>`` operators before the membership grammar.

The membership grammar (``Tag``, materials, later ``Set`` / ``VertexColor``)
adds new RHS families to ``Node``, ``Plug`` and ``PlugList`` operators. Every
test here pins what those operators do TODAY for the RHS families that
already exist, so the new branches cannot silently alter them. None of the
behaviour below is new; each test is a contract that the later steps must
keep green.

Error tests assert a zero ``cmds.ls()`` delta: a refused spelling writes
nothing to the scene.

One test pins a defect that was FIXED by the membership step: the spec
fan-out used to drop non-Plug/Node elements silently (3 in, 2 out); it is
now a ``TypeError`` naming the element, with nothing applied.
"""

from __future__ import annotations

import numpy as np
from maya import cmds
from rig import Node, Plug, PlugList
from rig.nodetypes.dg_node import DGNode
from rig.spec import Float, lock
from rig._tests._base import MayaTestCase


class _Opaque:
    """A non-Plug, non-Node, non-str element that survives PlugList."""

    def __repr__(self) -> str:
        return "_Opaque()"


class TestPlugListSpecBroadcast(MayaTestCase):
    """Attribute specs and modifiers fan out over every element."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.n1 = Node.create("transform", name="n1")
        self.n2 = Node.create("transform", name="n2")

    def test_pluglist_lshift_float_gives_one_plug_per_element(self):
        result = PlugList([self.n1, self.n2]) << Float("x")
        self.assertIsInstance(result, PlugList)
        self.assertEqual([str(p) for p in result], ["n1.x", "n2.x"])
        for node in ("n1", "n2"):
            self.assertTrue(cmds.attributeQuery("x", node=node, exists=True))

    def test_pluglist_of_plugs_lshift_float_adds_to_each_owner(self):
        result = PlugList([self.n1.tx, self.n2.ty]) << Float("y")
        self.assertEqual([str(p) for p in result], ["n1.y", "n2.y"])

    def test_pluglist_lshift_lock_locks_each(self):
        plugs  = PlugList([self.n1.tx, self.n2.tx])
        result = plugs << lock
        self.assertIsInstance(result, PlugList)
        self.assertEqual(
            [str(p) for p in result], ["n1.translateX", "n2.translateX"]
        )
        for node in ("n1", "n2"):
            self.assertTrue(cmds.getAttr(f"{node}.tx", lock=True))

    def test_pluglist_of_nodes_rshift_float_declares_output_attr_per_node(self):
        result = PlugList([self.n1, self.n2]) >> Float("q")
        self.assertIsInstance(result, PlugList)
        self.assertEqual([str(p) for p in result], ["n1.q", "n2.q"])
        for node in ("n1", "n2"):
            self.assertTrue(cmds.attributeQuery("q", node=node, exists=True))
            self.assertFalse(cmds.attributeQuery("q", node=node, writable=True))

    def test_fanout_refuses_foreign_elements(self):
        # Before the membership grammar landed, list.py's spec fan-out
        # filtered with ``isinstance(x, (Plug, Node))``, so a foreign element
        # was dropped from the result with no error -- 3 in, 2 out (this
        # test pinned that as ``..._TODAY``). The fan-out now refuses the
        # element by index BEFORE applying the spec to anything.
        pl = PlugList([self.n1, self.n2, _Opaque()])
        self.assertEqual(len(pl), 3)
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"element \[2\] \(_Opaque\(\)\)"):
            pl << Float("bcast")
        self.assertEqual(set(cmds.ls()), before)
        for node in ("n1", "n2"):
            self.assertFalse(cmds.attributeQuery("bcast", node=node, exists=True))


class TestPlugLshiftNone(MayaTestCase):
    """``plug << None`` disconnects; it is never a clear or a purge."""

    TEST_START_NEW_SCENE = True

    def test_plug_lshift_none_disconnects(self):
        src = Node.create("transform", name="src")
        dst = Node.create("transform", name="dst")
        cmds.connectAttr("src.ty", "dst.ty")
        self.assertTrue(cmds.listConnections("dst.ty", s=True, d=False))
        result = dst.ty << None
        self.assertIsInstance(result, Plug)
        self.assertEqual(str(result), "dst.translateY")
        self.assertIsNone(cmds.listConnections("dst.ty", s=True, d=False))


class TestComponentSliceBroadcast(MayaTestCase):
    """Component slices are PlugLists: ``<<`` pairs element to element."""

    TEST_START_NEW_SCENE = True

    def test_curve_cv_slice_lshift_pluglist_t_connects_per_element(self):
        crv   = cmds.curve(d=1, p=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        shape = Node(cmds.listRelatives(crv, type="nurbsCurve")[0])
        ctrls = PlugList(
            [Node.create("transform", name=f"c{i}") for i in range(3)]
        )
        ctrls.t << [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
        cvs    = shape.cv[:]
        result = cvs << ctrls.t
        self.assertIs(result, cvs)
        self.assertEqual(len(cvs), 3)
        for i, cv in enumerate(cvs):
            self.assertEqual(
                cmds.listConnections(str(cv), s=True, d=False, p=True),
                [f"c{i}.translate"],
            )
        positions = [cmds.pointPosition(str(cv), local=True) for cv in cvs]
        np.testing.assert_array_almost_equal(
            positions, [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
        )

    def test_mesh_vtx_step_slice_lshift_sets_positions(self):
        # ``[0, 0, 0]`` reaches each element through the asymmetric
        # broadcast (plug i receives ``src[min(i, 2)]``, a SCALAR that the
        # compound point fans out), so it lands every touched vertex on the
        # origin. A per-vertex vector needs the nested form, pinned below.
        cube  = cmds.polyCube(name="pc")[0]
        shape = Node(cmds.listRelatives(cube, shapes=True)[0])
        before = [
            cmds.pointPosition(f"{shape}.vtx[{i}]", local=True) for i in range(8)
        ]
        sl = shape.vtx[::2]
        self.assertEqual(len(sl), 4)
        result = sl << [0, 0, 0]
        self.assertIs(result, sl)
        for i in range(8):
            got = cmds.pointPosition(f"{shape}.vtx[{i}]", local=True)
            if i % 2 == 0:
                np.testing.assert_array_almost_equal(got, [0, 0, 0])
            else:
                np.testing.assert_array_almost_equal(got, before[i])

    def test_mesh_vtx_step_slice_lshift_nested_sets_each_position(self):
        cube    = cmds.polyCube(name="pc")[0]
        shape   = Node(cmds.listRelatives(cube, shapes=True)[0])
        targets = [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 4, 4]]
        shape.vtx[::2] << targets
        for i, target in zip(range(0, 8, 2), targets):
            np.testing.assert_array_almost_equal(
                cmds.pointPosition(f"{shape}.vtx[{i}]", local=True), target
            )


class TestNodeOperators(MayaTestCase):
    """``Node`` accepts specs and matrices; everything else is refused."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.node = Node.create("transform", name="ctrl")

    def test_node_rshift_none_is_typed_dgnode(self):
        result = self.node >> None
        self.assertIsInstance(result, DGNode)
        self.assertEqual(type(result).__name__, "Transform")
        self.assertEqual(str(result), "ctrl")

    def test_node_lshift_matrix_plug_decomposes(self):
        driver = Node.create("transform", name="driver")
        result = self.node << driver.matrix
        self.assertIs(result, self.node)
        for channel in ("t", "r", "s"):
            connections = cmds.listConnections(
                f"ctrl.{channel}", s=True, d=False
            ) or []
            self.assertIn(
                "decomposeMatrix", [cmds.nodeType(c) for c in connections]
            )

    def test_node_lshift_matrix_literal_decomposes_statically(self):
        m        = np.eye(4)
        m[3, :3] = [1, 2, 3]
        result   = self.node << m
        self.assertIs(result, self.node)
        np.testing.assert_array_almost_equal(cmds.getAttr("ctrl.t")[0], [1, 2, 3])
        self.assertIsNone(cmds.listConnections("ctrl.t", s=True, d=False))

    def test_node_lshift_spec_class_is_typeerror(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "Cannot inject type into a bare Node"):
            self.node << Float
        self.assertEqual(set(cmds.ls()), before)

    def test_node_lshift_node_is_typeerror_with_current_message(self):
        other  = Node.create("transform", name="other")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "Cannot inject Node into a bare Node"):
            self.node << other
        self.assertEqual(set(cmds.ls()), before)

    def test_node_lshift_str_is_typeerror_with_current_message(self):
        Node.create("transform", name="other")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "Cannot inject str into a bare Node"):
            self.node << "other"
        self.assertEqual(set(cmds.ls()), before)
