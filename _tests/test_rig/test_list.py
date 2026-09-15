"""Tests for ``rig._internal.list`` -- vectorised PlugList broadcasts."""

from maya import cmds
from rig import Node, Plug, PlugList
from rig._tests._base import MayaTestCase


class TestPlugListConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_empty_list(self):
        pl = PlugList()
        self.assertEqual(len(pl), 0)

    def test_from_strings(self):
        cmds.createNode("transform", name="cube1")
        cmds.createNode("transform", name="cube2")
        pl = PlugList(["cube1", "cube2"])
        self.assertEqual(len(pl), 2)
        self.assertIsInstance(pl[0], Node)
        self.assertIsInstance(pl[1], Node)

    def test_from_attribute_strings(self):
        cmds.createNode("transform", name="cube1")
        cmds.createNode("transform", name="cube2")
        pl = PlugList(["cube1.tx", "cube2.tx"])
        self.assertIsInstance(pl[0], Plug)
        self.assertIsInstance(pl[1], Plug)

    def test_from_mixed_numbers_and_nodes(self):
        node = Node.create("transform", name="cube1")
        pl   = PlugList([3.14, node, 42])
        self.assertEqual(pl[0], 3.14)
        self.assertIsInstance(pl[1], Node)
        self.assertEqual(pl[2], 42)

    def test_repr(self):
        pl = PlugList([1, 2, 3])
        r  = repr(pl)
        self.assertIn("PlugList", r)


class TestPlugListAttributeAccess(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_attr_access_broadcasts(self):
        Node.create("transform", name="cube1")
        Node.create("transform", name="cube2")
        pl    = PlugList(["cube1", "cube2"])
        plugs = pl.translateX
        self.assertIsInstance(plugs, PlugList)
        self.assertEqual(len(plugs), 2)
        for p in plugs:
            self.assertIsInstance(p, Plug)

    def test_numeric_passes_through(self):
        node  = Node.create("transform", name="cube1")
        pl    = PlugList([node, 3.14])
        plugs = pl.translateX
        # First gets .translateX, second (number) passes through.
        self.assertEqual(len(plugs), 2)
        self.assertIsInstance(plugs[0], Plug)
        self.assertEqual(plugs[1], 3.14)


class TestPlugListInjection(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lshift_scalar_broadcasts(self):
        Node.create("transform", name="cube1")
        Node.create("transform", name="cube2")
        Node.create("transform", name="cube3")
        pl = PlugList(["cube1", "cube2", "cube3"])
        pl.tx << 5.0
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 5.0)
        self.assertAlmostEqual(cmds.getAttr("cube2.tx"), 5.0)
        self.assertAlmostEqual(cmds.getAttr("cube3.tx"), 5.0)

    def test_lshift_list_pairs(self):
        Node.create("transform", name="cube1")
        Node.create("transform", name="cube2")
        pl = PlugList(["cube1", "cube2"])
        pl.tx << [1.0, 2.0]
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("cube2.tx"), 2.0)


class TestPlugListSlicing(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_slice_returns_pluglist(self):
        for i in range(4):
            Node.create("transform", name=f"c{i}")
        pl  = PlugList(["c0", "c1", "c2", "c3"])
        sub = pl[1:3]
        self.assertIsInstance(sub, PlugList)
        self.assertEqual(len(sub), 2)

    def test_int_index_returns_element(self):
        node = Node.create("transform", name="cube1")
        pl   = PlugList([node])
        self.assertIs(pl[0], node)


class TestPlugListArithmetic(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_add_broadcasts(self):
        Node.create("transform", name="a")
        Node.create("transform", name="b")
        pl_a   = PlugList(["a", "b"]).tx
        pl_b   = PlugList(["a", "b"]).ty
        result = pl_a + pl_b
        self.assertIsInstance(result, PlugList)
        self.assertEqual(len(result), 2)
        for r in result:
            # Maya 2024+ may use native ``sum`` node; older uses ``plusMinusAverage``.
            self.assertIn(
                cmds.nodeType(str(r).split(".")[0]),
                {"plusMinusAverage", "sum"},
            )

    def test_negate(self):
        Node.create("transform", name="a")
        Node.create("transform", name="b")
        pl     = PlugList(["a", "b"]).tx
        result = -pl
        self.assertIsInstance(result, PlugList)
        self.assertEqual(len(result), 2)


class TestPlugListHashable(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_pluglist_hashable(self):
        node = Node.create("transform", name="cube1")
        pl   = PlugList([node])
        # Can be used in a set / dict.
        s = {pl}
        self.assertEqual(len(s), 1)


class TestPlugListRshift(MayaTestCase):
    """``PlugList.__rshift__`` broadcasts per-element."""

    TEST_START_NEW_SCENE = True

    def test_rshift_none_returns_raw_values(self):
        import numpy as np

        cmds.createNode("transform", name="cube1")
        cmds.createNode("transform", name="cube2")
        cmds.setAttr("cube1.tx", 3.0)
        cmds.setAttr("cube2.tx", 5.0)
        pl     = PlugList(["cube1", "cube2"]).tx
        values = pl >> None
        # Numpy 1-D array of the per-element scalar values.
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (2,))
        np.testing.assert_array_almost_equal(values, [3.0, 5.0])

    def test_rshift_none_vector_stacks_to_2d(self):
        import numpy as np

        cmds.createNode("transform", name="cubeA")
        cmds.createNode("transform", name="cubeB")
        cmds.setAttr("cubeA.t", 1.0, 2.0, 3.0, type="double3")
        cmds.setAttr("cubeB.t", 4.0, 5.0, 6.0, type="double3")
        pl     = PlugList(["cubeA", "cubeB"]).t
        values = pl >> None
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (2, 3))
        np.testing.assert_array_almost_equal(values, [[1, 2, 3], [4, 5, 6]])

    def test_rshift_none_matrix_stacks_to_3d(self):
        import numpy as np

        cmds.createNode("transform", name="m1")
        cmds.createNode("transform", name="m2")
        pl     = PlugList(["m1", "m2"]).matrix
        values = pl >> None
        self.assertIsInstance(values, np.ndarray)
        # Matrix list => shape (-1, 4, 4).
        self.assertEqual(values.shape, (2, 4, 4))

    def test_rshift_none_node_list_returns_dgnodes(self):
        from rig.maya.nodetypes.dg_node import DGNode

        cmds.createNode("transform", name="nodeA")
        cmds.createNode("transform", name="nodeB")
        pl     = PlugList(["nodeA", "nodeB"])
        values = pl >> None
        # Heterogeneous (or DGNode-typed) => plain Python list.
        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 2)
        for v in values:
            self.assertIsInstance(v, DGNode)

    def test_rshift_node_clones_each_attr(self):
        from rig.spec import Float

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        c = Node.create("transform", name="c")
        a << Float("custom1") << 0.1
        b << Float("custom2") << 0.2
        srcs      = PlugList([a.custom1, b.custom2])
        new_plugs = srcs >> c
        self.assertIsInstance(new_plugs, PlugList)
        self.assertEqual(len(new_plugs), 2)
        self.assertTrue(cmds.attributeQuery("custom1", node="c", exists=True))
        self.assertTrue(cmds.attributeQuery("custom2", node="c", exists=True))

    def test_rshift_unsupported_propagates_typeerror(self):
        node = Node.create("transform", name="cube1")
        pl   = PlugList([node]).tx
        with self.assertRaises(TypeError):
            _ = pl >> 5

    def test_rshift_passes_through_non_plug_elements(self):
        # Numeric / non-Plug elements pass through unchanged (then stack
        # into a numpy array along with plug values).
        import numpy as np

        node = Node.create("transform", name="cube1")
        node.tx << 7.0
        pl     = PlugList([node.tx, 42.0])
        values = pl >> None
        self.assertIsInstance(values, np.ndarray)
        np.testing.assert_array_almost_equal(values, [7.0, 42.0])


class TestPlugListComponentRange(MayaTestCase):
    """``Node('shape').vtx[0:5]`` returns a PlugList of indexed
    controlPoints plugs -- numpy-style slicing on geometry components.

    The previous implementation built the PlugList via the non-idiomatic
    ``PlugList([f"{shape}.vtx[0:5]"])`` (string range parsing). The
    intended user-facing API is ``Node(shape).vtx[0:5]``, which goes
    through Attribute's component-aware ``__getitem__`` and returns a
    PlugList of indexed controlPoints plugs directly.
    """

    TEST_START_NEW_SCENE = True

    def test_vtx_range_returns_pluglist(self):
        cube  = cmds.polyCube(name="poly_for_range")[0]
        shape = Node(cmds.listRelatives(cube, shapes=True)[0])
        pl    = shape.vtx[0:5]
        self.assertIsInstance(pl, PlugList)
        self.assertEqual(len(pl), 5)
        # Each element is an indexed controlPoints plug (vtx is the alias).
        # On a mesh, Maya resolves ``vtx`` to the ``pnts`` spelling rather than
        # ``controlPoints``; nurbsCurve cv / lattice pt do report
        # ``controlPoints``. Both name the same attribute.
        for elt in pl:
            self.assertNotIn(":", str(elt))
            self.assertTrue(str(elt).endswith("]"))
            self.assertTrue(
                any(alias in str(elt) for alias in ("controlPoints", "pnts")),
                f"expected a controlPoints/pnts plug, got {elt}",
            )


class TestPlugListGet(MayaTestCase):
    """v3.O: ``PlugList.get()`` returns numpy-aware stacked values, mirroring
    ``pluglist >> None``. Useful for snapshot-copy idioms
    (``target.t << source.t.get()``) without building live connections.
    """

    TEST_START_NEW_SCENE = True

    def test_get_returns_stacked_array_of_scalar_values(self):
        import numpy as np

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        c = Node.create("transform", name="c")
        a.tx << 1.0
        b.tx << 2.0
        c.tx << 3.0

        result = PlugList([a.tx, b.tx, c.tx]).get()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_handles_compound_plugs(self):
        import numpy as np

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1.0, 2.0, 3.0]
        b.t << [4.0, 5.0, 6.0]

        # Compound plugs unwrap Maya's [(x, y, z)] envelope and stack
        # into a homogeneous (N, 3) array -- same shape as ``>> None``.
        result = PlugList([a.t, b.t]).get()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[1], [4.0, 5.0, 6.0])

    def test_get_passes_through_non_plug_elements(self):
        # PlugList allows numbers / None pass-through; ``_stack_values``
        # rejects ``None``-containing lists and returns the raw list.
        result = PlugList([1.0, None, 2.0]).get()
        self.assertEqual(result, [1.0, None, 2.0])

    def test_get_enables_snapshot_copy_idiom(self):
        # The motivating use case: copy values without connecting.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1.0, 2.0, 3.0]

        # Snapshot copy: no live connection, just one-shot value transfer.
        # PlugList([a.t]).get() now returns shape (1, 3); index [0] gives
        # the single (3,) vector that goes into b.t.
        b.t << PlugList([a.t]).get()[0]

        # Verify b.t got the value AND no incoming connection was made.
        self.assertAlmostEqual(cmds.getAttr("b.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("b.ty"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("b.tz"), 3.0)
        connections = cmds.listConnections("b.t", source=True, destination=False)
        self.assertIsNone(connections)


class TestGetMatchesRshiftNone(MayaTestCase):
    """Regression: ``Plug.get()`` and ``PlugList.get()`` must produce the
    same shape as their ``>> None`` counterparts. Pins the unification
    so a future refactor of either path stays in sync.
    """

    TEST_START_NEW_SCENE = True

    def test_plug_scalar_get_matches_rshift_none(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        self.assertEqual(a.tx.get(), a.tx >> None)

    def test_plug_compound_get_matches_rshift_none(self):
        import numpy as np

        a = Node.create("transform", name="a")
        a.t << [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(a.t.get(), a.t >> None)

    def test_pluglist_scalar_get_matches_rshift_none(self):
        import numpy as np

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.tx << 1.0
        b.tx << 2.0
        pl = PlugList([a.tx, b.tx])
        np.testing.assert_array_equal(pl.get(), pl >> None)

    def test_pluglist_compound_get_matches_rshift_none(self):
        import numpy as np

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1.0, 2.0, 3.0]
        b.t << [4.0, 5.0, 6.0]
        pl = PlugList([a.t, b.t])
        np.testing.assert_array_equal(pl.get(), pl >> None)


class TestPlugListPow(MayaTestCase):
    """``PlugList ** scalar`` maps the new ``Plug.__pow__`` per-element --
    so a list of quaternions yields a list of fractional rotations.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_pluglist_pow_maps_elementwise(self):
        from rig import matrix as m, quaternion

        a = cmds.spaceLocator()[0]
        b = cmds.spaceLocator()[0]
        cmds.setAttr(a + ".rotateY", 90)
        cmds.setAttr(b + ".rotateY", 30)
        quats = PlugList(
            [
                m.decompose(Node(a).worldMatrix[0]).outputQuat,
                m.decompose(Node(b).worldMatrix[0]).outputQuat,
            ]
        )
        result = quats**0.5
        self.assertEqual(len(result), 2)
        # Each element actually routed through quaternion.pow (not a no-op):
        # quats[0] is a 90deg-about-Y quaternion, so ** 0.5 -> 45deg.
        self.assertAlmostEqual(
            cmds.getAttr(f"{result[0]}Y"),
            cmds.getAttr(f"{quaternion.pow(quats[0], 0.5)}Y"),
            places=4,
        )


class TestPlugListGetInputsOutputs(MayaTestCase):
    """N-aligned connection queries. Every slot is a PlugList, so no slot
    can be a ``None`` that ``<<`` would read as "disconnect"."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="drv")
        cmds.createNode("transform", name="ctrl")
        cmds.createNode("transform", name="spare")
        cmds.connectAttr("drv.translateX", "ctrl.translateX")
        cmds.connectAttr("ctrl.translateX", "spare.translateX")

    def test_get_inputs_is_n_aligned_and_nested(self):
        got = Node("ctrl").t[:].get_inputs()
        self.assertEqual(len(got), 3)
        for slot in got:
            self.assertIsInstance(slot, PlugList)
        self.assertEqual([str(x) for x in got[0]], ["drv.translateX"])
        self.assertEqual(len(got[1]), 0)
        self.assertEqual(len(got[2]), 0)

    def test_get_outputs_is_n_aligned_and_nested(self):
        got = Node("ctrl").t[:].get_outputs()
        self.assertEqual(len(got), 3)
        self.assertEqual([str(x) for x in got[0]], ["spare.translateX"])
        self.assertEqual(len(got[1]), 0)

    def test_no_slot_is_ever_none(self):
        for got in (Node("ctrl").t[:].get_inputs(), Node("ctrl").t[:].get_outputs()):
            self.assertTrue(all(slot is not None for slot in got))

    def test_empty_list_yields_empty_result(self):
        self.assertEqual(len(PlugList().get_inputs()), 0)
        self.assertEqual(len(PlugList().get_outputs()), 0)

    def test_non_plug_element_raises(self):
        with self.assertRaises(TypeError):
            PlugList([Node("ctrl")]).get_inputs()

    def test_retired_sentinels_raise(self):
        pl = Node("ctrl").t[:]
        with self.assertRaises(TypeError):
            pl << PlugList
        with self.assertRaises(TypeError):
            pl >> PlugList


class TestPlugListContainment(MayaTestCase):
    """``in`` / ``index`` / ``count`` / ``remove`` behave like plain list
    operations and never route through ``Plug.__eq__`` (which would build a
    condition network instead of answering the question)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        for name in ("a", "b", "c"):
            cmds.createNode("transform", name=name)

    def test_in_matches_by_plug_name(self):
        pl = PlugList([Node("a").t, Node("b").t])
        self.assertIn(Plug("a.translate"), pl)
        self.assertNotIn(Plug("c.translate"), pl)

    def test_in_builds_no_nodes(self):
        pl     = PlugList([Node("a").t, Node("b").t])
        before = set(cmds.ls())
        self.assertIn(Plug("a.translate"), pl)
        self.assertNotIn(Plug("c.translate"), pl)
        self.assertEqual(set(cmds.ls()), before)

    def test_index_finds_the_correct_slot(self):
        a, b = Node("a").t, Node("b").t
        pl = PlugList([a, b])
        self.assertEqual(pl.index(a), 0)
        self.assertEqual(pl.index(b), 1)

    def test_index_missing_raises(self):
        with self.assertRaises(ValueError):
            PlugList([Node("a").t]).index(Node("c").t)

    def test_count_is_per_element(self):
        pl = PlugList([Node("a").t, Node("b").t])
        self.assertEqual(pl.count(Node("a").t), 1)
        self.assertEqual(pl.count(Node("c").t), 0)

    def test_remove_deletes_the_matching_element(self):
        pl = PlugList([Node("a").t, Node("b").t])
        pl.remove(Node("b").t)
        self.assertEqual([str(x) for x in pl], ["a.translate"])

    def test_remove_missing_raises(self):
        with self.assertRaises(ValueError):
            PlugList([Node("a").t]).remove(Node("c").t)

    def test_node_elements_match_by_name(self):
        pl = PlugList([Node("a"), Node("b")])
        self.assertIn(Node("a"), pl)
        self.assertNotIn(Node("c"), pl)
        self.assertEqual(pl.index(Node("b")), 1)

    def test_non_plug_elements_use_plain_equality(self):
        pl = PlugList([1, 2.5, 7])
        self.assertIn(2.5, pl)
        self.assertNotIn(99, pl)
        self.assertEqual(pl.index(7), 2)
        self.assertEqual(pl.count(1), 1)


class TestNestedPlugListGet(MayaTestCase):
    """``get()`` must recurse into nested results. Without it a nested slot is
    neither Plug nor Node, so it passes through unconverted -- and because Plug
    subclasses str, a rectangular nesting stacks into an array of plug NAME
    STRINGS instead of raising."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        for name in ("drv", "ctrl", "s1", "s2"):
            cmds.createNode("transform", name=name)
        cmds.connectAttr("drv.translateX", "ctrl.translateX")
        cmds.setAttr("drv.translateX", 7.0)
        for dst in ("s1", "s2"):
            cmds.connectAttr("ctrl.translateX", f"{dst}.translateX")
            cmds.connectAttr("ctrl.translateY", f"{dst}.translateY")
        cmds.setAttr("ctrl.translateY", 3.0)

    def test_rectangular_nesting_yields_numbers_not_names(self):
        import numpy as np

        got = PlugList([Node("ctrl").tx, Node("ctrl").ty]).get_outputs().get()
        self.assertIsInstance(got, np.ndarray)
        self.assertEqual(got.shape, (2, 2))
        self.assertTrue(np.issubdtype(got.dtype, np.floating), got.dtype)
        self.assertEqual(sorted(got.ravel().tolist()), [3.0, 3.0, 7.0, 7.0])

    def test_ragged_nesting_degrades_to_a_list_of_values(self):
        import numpy as np

        got = Node("ctrl").t[:].get_inputs().get()
        self.assertIsInstance(got, list)
        self.assertEqual(len(got), 3)
        # Slot 0 resolved to the driver's VALUE, not to a Plug.
        self.assertEqual(np.asarray(got[0]).ravel().tolist(), [7.0])
        self.assertEqual(len(got[1]), 0)
        self.assertEqual(len(got[2]), 0)

    def test_rshift_none_matches_get(self):
        nested = Node("ctrl").t[:].get_inputs()
        self.assertEqual(repr(nested >> None), repr(nested.get()))

    def test_flat_get_unchanged(self):
        import numpy as np

        got = Node("ctrl").t[:].get()
        self.assertIsInstance(got, np.ndarray)
        self.assertTrue(np.issubdtype(got.dtype, np.floating))
        self.assertEqual(got.tolist()[:2], [7.0, 3.0])