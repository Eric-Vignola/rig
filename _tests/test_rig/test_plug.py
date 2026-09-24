"""Tests for ``rig._internal.plug`` -- the operator-extended Attribute."""

from maya import cmds
from rig import InjectionError, lock, Node, Plug, PlugList, skip, unlock
from rig._tests._base import MayaTestCase


def _has_native_math_nodes():
    """Whether the running Maya ships the 2024+ native math node types
    (``sum``, ``equal``, ``and``, ``atan``, ...).

    Reads the ACTUAL running version, never the DSL target:
    ``set_options(maya_version=N)`` only downgrades dispatch, it cannot add
    node types Maya does not ship. On an older Maya ``createNode("sum")``
    yields an unknown placeholder with no attributes rather than raising.
    """
    from rig._internal.maya_version import get_maya_version

    return get_maya_version() >= 2024


def _skip_without_native_math_nodes(test):
    if not _has_native_math_nodes():
        test.skipTest("native 2024+ math node types require Maya 2024+")


class TestPlugBasics(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construct_from_string(self):
        cmds.createNode("transform", name="cube1")
        plug = Plug("cube1.translateX")
        self.assertEqual(str(plug), "cube1.translateX")

    def test_node_property_returns_rignode(self):
        node = Node.create("transform", name="cube1")
        plug = node.translateX
        self.assertIsInstance(plug.node, Node)


class TestPlugAttributeLookup(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_child_attribute_returns_plug(self):
        node = Node.create("transform", name="cube1")
        # node.translate is a Plug (compound); .x returns child Plug.
        child = node.translate.translateX
        self.assertIsInstance(child, Plug)


class TestPlugSiblingFallback(MayaTestCase):
    """Plug.foo falls back to a sibling attribute on the same node.

    Restores Eric Vignola's linguistic affordance:
    Plug('foo.outputTranslate').outputRotate resolves to Plug('foo.outputRotate').
    """

    TEST_START_NEW_SCENE = True

    def test_sibling_attribute_on_pma(self):
        # plusMinusAverage has both .output1D (compound) and .operation (sibling).
        # Accessing .operation via .output1D must resolve to the sibling.
        node      = Node.create("plusMinusAverage", name="pma1")
        output    = node.output1D
        operation = output.operation
        self.assertEqual(operation.alias, "operation")
        operation << 3
        self.assertEqual(cmds.getAttr("pma1.operation"), 3)

    def test_sibling_via_setattr_sugar(self):
        node   = Node.create("plusMinusAverage", name="pma2")
        output = node.output1D
        # Sugar: .operation is not a child of output1D, must fall back to sibling.
        output.operation = 2
        self.assertEqual(cmds.getAttr("pma2.operation"), 2)

    def test_sibling_on_decompose_matrix(self):
        # dec = decomposeMatrix.outputTranslate
        # dec.outputRotate must fall back to the sibling on the same node.
        from rig._internal.math_nodes import _decompose_matrix

        a   = Node.create("transform", name="src_xform")
        dec = _decompose_matrix(a.matrix)
        self.assertEqual(dec.outputRotate.alias, "outputRotate")
        # Same underlying node as dec.
        self.assertEqual(str(dec).split(".")[0], str(dec.outputRotate).split(".")[0])

    def test_unknown_name_raises_attribute_error(self):
        node = Node.create("transform", name="cube_unknown")
        plug = node.translateX
        with self.assertRaises(AttributeError) as cm:
            _ = plug.completely_made_up_attribute_xyz
        self.assertIn("no child or sibling attribute", str(cm.exception))

    def test_sibling_fallback_handles_typed_atomic_plug(self):
        """Typed-atomic plugs (matrix / mesh / message) raise TypeError --
        not AttributeError -- from MPlug.numChildren(). Sibling fallback
        must still kick in.

        Regression for: `node.matrix.ro` raising
        ``TypeError: Plug is not a compound parent``.
        """
        cube = Node.create("transform", name="cube_typed")
        # `.matrix` is a typed atomic -- drilling into it for a sibling
        # name like `ro` must return the transform's `.rotateOrder`,
        # NOT raise TypeError.
        ro = cube.matrix.ro
        self.assertEqual(ro.alias, "rotateOrder")


class TestPlugSliceReturnsPlugList(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_multi_attr_slice_returns_pluglist(self):
        # blendShape weight is a multi attr.
        cube   = cmds.polyCube(name="src")[0]
        target = cmds.polyCube(name="tgt")[0]
        bs     = cmds.blendShape(cube, target, name="bs1")[0]

        from rig import PlugList

        bs_node = Node(bs)
        sliced  = bs_node.weight[:]
        # Must be PlugList (not plain list) so chained ops work.
        self.assertIsInstance(sliced, PlugList)


class TestPlugCompoundSlicing(MayaTestCase):
    """v3.O: Compound non-multi attributes (translate, rotate, scale,
    inputQuat, etc.) support numpy-style indexing and slicing.

    ``node.translate[0]``    -> Plug for translateX
    ``node.translate[:]``    -> PlugList([tx, ty, tz])
    ``node.translate[0:2]``  -> PlugList([tx, ty])
    ``node.translate[::-1]`` -> PlugList([tz, ty, tx])

    Pre-v3.O the underlying ``Attribute.__getitem__`` raised
    ``RuntimeError: not an multi attr`` on compounds. v3.O adds a
    compound branch in ``Plug.__getitem__`` that uses the canonical
    ``MPlug.child(i)`` API.
    """

    TEST_START_NEW_SCENE = True

    def test_compound_int_index(self):
        node = Node.create("transform", name="cube1")
        tx   = node.translate[0]
        self.assertIsInstance(tx, Plug)
        self.assertEqual(str(tx), "cube1.translateX")

    def test_compound_int_index_negative(self):
        node = Node.create("transform", name="cube1")
        tz   = node.translate[-1]
        self.assertEqual(str(tz), "cube1.translateZ")

    def test_compound_full_slice(self):
        from rig import PlugList

        node     = Node.create("transform", name="cube1")
        children = node.translate[:]
        self.assertIsInstance(children, PlugList)
        self.assertEqual(len(children), 3)
        self.assertEqual(
            [str(p) for p in children],
            ["cube1.translateX", "cube1.translateY", "cube1.translateZ"],
        )

    def test_compound_partial_slice(self):
        from rig import PlugList

        node      = Node.create("transform", name="cube1")
        first_two = node.translate[0:2]
        self.assertIsInstance(first_two, PlugList)
        self.assertEqual(len(first_two),    2)
        self.assertEqual(str(first_two[0]), "cube1.translateX")
        self.assertEqual(str(first_two[1]), "cube1.translateY")

    def test_compound_reverse_slice(self):
        node              = Node.create("transform", name="cube1")
        reversed_children = node.translate[::-1]
        self.assertEqual(len(reversed_children), 3)
        self.assertEqual(
            [str(p) for p in reversed_children],
            ["cube1.translateZ", "cube1.translateY", "cube1.translateX"],
        )

    def test_compound_index_out_of_range(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(IndexError):
            node.translate[5]

    def test_compound_slice_assignment(self):
        # node.translate[:] << [1, 2, 3] -- slice form, equivalent to
        # node.translate << [1, 2, 3] (which has worked since v2).
        node = Node.create("transform", name="cube1")
        node.translate[:] << [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 3.0)


class TestPlugComponentConstruction(MayaTestCase):
    """v3.O: Plug() accepts geometry-component strings (vtx[N], cv[N],
    controlPoints[N], etc.) and translates them to the canonical
    underlying ``controlPoints[N]`` plug.

    Pre-v3.O ``Plug("pCube.vtx[0]")`` raised ``TypeError: item is not a
    plug`` because Maya's ``MSelectionList`` resolves geometry components
    as ``kComponent`` items, not plugs. v3.O detects the component via
    the API and constructs the underlying ``MPlug`` directly.
    """

    TEST_START_NEW_SCENE = True

    def test_construct_from_vtx_alias(self):
        cube  = cmds.polyCube(name="poly1")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        plug  = Plug(f"{shape}.vtx[0]")
        self.assertIsInstance(plug, Plug)
        self.assertEqual(str(plug), f"{shape}.controlPoints[0]")

    def test_construct_from_controlPoints_canonical(self):
        cube  = cmds.polyCube(name="poly1")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        plug  = Plug(f"{shape}.controlPoints[0]")
        self.assertIsInstance(plug, Plug)
        self.assertEqual(str(plug), f"{shape}.controlPoints[0]")

    def test_construct_non_component_unchanged(self):
        # A normal plug name passes through unchanged -- the component
        # detection helper only intercepts kComponent items.
        cmds.createNode("transform", name="cube1")
        plug = Plug("cube1.translateX")
        self.assertEqual(str(plug), "cube1.translateX")


class TestPlugMultiRootAutoIndex(MayaTestCase):
    """Multi-attribute roots auto-index on `<<` injection.

    ``node.input1D << X`` becomes ``node.input1D[next_free] << X``. This is
    what makes ``a.tx + b.tx`` work -- the ``_plus_minus_average`` factory
    can do ``node.input1D << x`` in a loop and have each call land on a
    fresh slot.
    """

    TEST_START_NEW_SCENE = True

    def test_multi_root_connect_auto_indexes(self):
        a   = Node.create("transform",        name="a")
        b   = Node.create("transform",        name="b")
        pma = Node.create("plusMinusAverage", name="pma1")
        pma.input1D << a.tx
        pma.input1D << b.tx
        in0 = cmds.listConnections("pma1.input1D[0]", source=True, plugs=True) or []
        in1 = cmds.listConnections("pma1.input1D[1]", source=True, plugs=True) or []
        self.assertEqual(in0, ["a.translateX"])
        self.assertEqual(in1, ["b.translateX"])

    def test_multi_root_setattr_auto_indexes(self):
        pma = Node.create("plusMinusAverage", name="pma1")
        pma.input1D << 7.0
        pma.input1D << 8.0
        self.assertAlmostEqual(cmds.getAttr("pma1.input1D[0]"), 7.0)
        self.assertAlmostEqual(cmds.getAttr("pma1.input1D[1]"), 8.0)

    def test_explicit_index_bypasses_auto_indexing(self):
        # If the user already indexed (str ends in `]`), don't re-index.
        pma = Node.create("plusMinusAverage", name="pma1")
        pma.input1D[5] << 42.0
        self.assertAlmostEqual(cmds.getAttr("pma1.input1D[5]"), 42.0)


class TestPlugInjectShapeValidation(MayaTestCase):
    """Tier A: NumPy-style strict shape validation on `<<` sequence sources.

    Mismatched shapes raise ValueError instead of silently degrading.
    """

    TEST_START_NEW_SCENE = True

    def test_scalar_dst_with_multi_element_list_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError) as cm:
            node.tx << [1, 2]
        self.assertIn("scalar", str(cm.exception))

    def test_scalar_dst_with_nested_list_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError):
            node.tx << [[1, 2, 3], [4, 5, 6]]

    def test_scalar_dst_with_size_one_list_works(self):
        node = Node.create("transform", name="cube1")
        node.tx << [7.5]
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 7.5)

    def test_vector_dst_with_wrong_size_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError) as cm:
            node.t << [1, 2]
        self.assertIn("3-channel", str(cm.exception))

    def test_vector_dst_with_too_many_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError):
            node.t << [1, 2, 3, 4]

    def test_vector_dst_with_nested_2x3_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError):
            node.t << [[1, 2, 3], [4, 5, 6]]

    def test_vector_dst_with_empty_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(ValueError):
            node.t << []

    def test_vector_dst_with_correct_size_works(self):
        node = Node.create("transform", name="cube1")
        node.t << [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 3.0)

    def test_vector_dst_with_scalar_broadcasts(self):
        node = Node.create("transform", name="cube1")
        node.s << [2.5]
        self.assertAlmostEqual(cmds.getAttr("cube1.sx"), 2.5)
        self.assertAlmostEqual(cmds.getAttr("cube1.sy"), 2.5)
        self.assertAlmostEqual(cmds.getAttr("cube1.sz"), 2.5)


class TestPlugMatrixShapeCoercion(MayaTestCase):
    """Tier B: matrix-typed attrs accept 9 / 16 / 3x3 / 4x4 inputs.

    Tested on a non-transform node so Tier C decomposition doesn't fire.
    """

    TEST_START_NEW_SCENE = True

    def test_flat_16_to_matrix_attr(self):
        from rig.spec import Matrix

        n = Node.create("network", name="net1")
        n       << Matrix("xform")
        n.xform << list(range(16))
        result = cmds.getAttr("net1.xform")
        self.assertEqual(result, list(range(16)))

    def test_4x4_numpy_to_matrix_attr(self):
        import numpy as np
        from rig.spec import Matrix

        n = Node.create("network", name="net1")
        n       << Matrix("xform")
        n.xform << np.eye(4)
        result = cmds.getAttr("net1.xform")
        self.assertEqual(result, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_nested_4x4_list_to_matrix_attr(self):
        from rig.spec import Matrix

        n = Node.create("network", name="net1")
        n << Matrix("xform")
        nested = [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [4, 5, 6, 1],
        ]
        n.xform << nested
        result = cmds.getAttr("net1.xform")
        self.assertEqual(result, [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 5, 6, 1])

    def test_3x3_to_matrix_attr_embeds_in_identity(self):
        import numpy as np
        from rig.spec import Matrix

        n = Node.create("network", name="net1")
        n       << Matrix("xform")
        n.xform << np.eye(3) * 2.0
        result = cmds.getAttr("net1.xform")
        self.assertEqual(
            result,
            [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
        )

    def test_wrong_size_matrix_raises(self):
        from rig.spec import Matrix

        n = Node.create("network", name="net1")
        n << Matrix("xform")
        with self.assertRaises(ValueError):
            n.xform << [1, 2, 3, 4, 5]


class TestPlugMatrixToTransform(MayaTestCase):
    """Tier C: matrix-shaped sources route to transform channels by alias."""

    TEST_START_NEW_SCENE = True

    def test_matrix_identity_4x4_resets_transform(self):
        import numpy as np

        cube = Node.create("transform", name="cube1")
        cube.t      << [10, 20, 30]
        cube.r      << [45, 30, 60]
        cube.s      << [2, 3, 4]
        cube.matrix << np.eye(4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 0.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 0.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 0.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.rx"), 0.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sx"), 1.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sy"), 1.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sz"), 1.0, places=4)

    def test_matrix_3x3_preserves_translation(self):
        import numpy as np

        cube = Node.create("transform", name="cube1")
        cube.t      << [10, 20, 30]
        cube.matrix << np.eye(3) * 2.0
        # Translation preserved (3x3 has no translation row).
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 10.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 20.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 30.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sx"), 2.0,  places=4)

    def test_matrix_with_translation(self):
        import numpy as np

        cube     = Node.create("transform", name="cube1")
        m        = np.eye(4)
        m[3, :3] = [1.0, 2.0, 3.0]
        cube.matrix << m
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 2.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 3.0, places=4)

    def test_translate_channel_only(self):
        import numpy as np

        cube = Node.create("transform", name="cube1")
        cube.r << [45, 0, 0]
        m        = np.eye(4)
        m[3, :3] = [5.0, 6.0, 7.0]
        cube.t << m
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 5.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 6.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 7.0, places=4)
        # Rotation untouched.
        self.assertAlmostEqual(cmds.getAttr("cube1.rx"), 45.0, places=4)

    def test_scale_channel_only(self):
        import numpy as np

        cube = Node.create("transform", name="cube1")
        cube.t << [10, 20, 30]
        m = np.diag([2.0, 3.0, 4.0, 1.0])
        cube.s << m
        self.assertAlmostEqual(cmds.getAttr("cube1.sx"), 2.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sy"), 3.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.sz"), 4.0, places=4)
        # Translation untouched.
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 10.0, places=4)

    def test_disconnect_before_set(self):
        # Setting via decomposition disconnects any incoming live driver.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << a.tx
        self.assertTrue(cmds.listConnections("b.tx", source=True, destination=False))
        import numpy as np

        m        = np.eye(4)
        m[3, :3] = [9.0, 0.0, 0.0]
        b.t << m
        self.assertFalse(cmds.listConnections("b.tx", source=True, destination=False))
        self.assertAlmostEqual(cmds.getAttr("b.tx"), 9.0, places=4)

    def test_quaternion_attr_from_matrix(self):
        import numpy as np
        from rig.spec import Quat

        n = Node.create("transform", name="cube1")
        n        << Quat("myQuat")
        n.myQuat << np.eye(4)  # identity => (0,0,0,1)
        children = cmds.attributeQuery("myQuat", node="cube1", listChildren=True)
        self.assertAlmostEqual(cmds.getAttr(f"cube1.{children[0]}"), 0.0, places=4)
        self.assertAlmostEqual(cmds.getAttr(f"cube1.{children[3]}"), 1.0, places=4)

    def test_world_matrix_apply_with_no_parent(self):
        # No parent => parentInverseMatrix is identity => same as .matrix.
        import numpy as np

        cube     = Node.create("transform", name="cube1")
        m        = np.eye(4)
        m[3, :3] = [3.0, 4.0, 5.0]
        cube.worldMatrix << m
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 3.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 4.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 5.0, places=4)


class TestPlugInjection(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_set_scalar(self):
        node = Node.create("transform", name="cube1")
        node.tx << 5.0
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 5.0)

    def test_set_vector(self):
        node = Node.create("transform", name="cube1")
        node.t << [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 1.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.ty"), 2.0)
        self.assertAlmostEqual(cmds.getAttr("cube1.tz"), 3.0)

    def test_connect_plug_to_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << a.tx
        connections = cmds.listConnections(
            "b.tx", source=True, destination=False, plugs=True
        )
        self.assertIn("a.translateX", connections)

    def test_disconnect_with_none(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << a.tx
        b.tx << None
        connections = cmds.listConnections("b.tx", source=True, destination=False)
        self.assertFalse(connections)

    def test_chain_returns_self(self):
        node   = Node.create("transform", name="cube1")
        result = node.tx << 5
        # Chains must return the LHS.
        self.assertEqual(str(result), "cube1.translateX")

    def test_set_heterogeneous_plug_and_number_list(self):
        """Regression: ``node.t << [src.tx, 0, src.tz]`` should connect
        ``src.tx -> node.tx``, set ``node.ty = 0``, connect ``src.tz -> node.tz``.

        Was silently failing because ``np.asarray([Plug, int, Plug])``
        coerces all elements to strings (``Plug.__str__`` returns the plug
        name like ``"src.translateX"``), giving a ``<U21`` string array
        rather than ``dtype=object``. The validation block then dispatched
        plug-name STRINGS into ``_set_or_connect``, which silently failed
        the setAttr against a ``double`` dst.
        """
        src = Node.create("transform", name="src")
        dst = Node.create("transform", name="dst")
        cmds.setAttr("src.tx", 11.0)
        cmds.setAttr("src.ty", 22.0)
        cmds.setAttr("src.tz", 33.0)

        dst.t << [src.tx, 0, src.tz]

        # tx should be CONNECTED to src.tx
        tx_in = (
            cmds.listConnections("dst.tx", source=True, destination=False, plugs=True)
            or []
        )
        self.assertIn("src.translateX", tx_in)
        # ty should be SET to 0
        ty_in = cmds.listConnections("dst.ty", source=True, destination=False)
        self.assertFalse(ty_in)
        self.assertAlmostEqual(cmds.getAttr("dst.ty"), 0.0)
        # tz should be CONNECTED to src.tz
        tz_in = (
            cmds.listConnections("dst.tz", source=True, destination=False, plugs=True)
            or []
        )
        self.assertIn("src.translateZ", tz_in)
        # And the resulting evaluated value reflects the connections + set.
        self.assertEqual(cmds.getAttr("dst.t"), [(11.0, 0.0, 33.0)])


class TestPlugAssignmentSugar(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_setattr_sugar_works(self):
        node    = Node.create("transform", name="cube1")
        node.tx = 7.5  # syntactic sugar for `node.tx << 7.5`
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 7.5)


class TestPlugIntrospection(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_rshift_none_returns_value(self):
        node = Node.create("transform", name="cube1")
        node.tx << 3.14
        value = node.tx >> None
        # Scalar -- Python float, NOT a numpy 0-dim array.
        self.assertAlmostEqual(value, 3.14)

    def test_rshift_rignode_clones_attr(self):
        a = Node.create("transform", name="src")
        b = Node.create("transform", name="dst")
        from rig.spec import Float

        a              << Float("custom_blend") << 0.5
        a.custom_blend >> b
        # New attr exists on b.
        self.assertTrue(cmds.attributeQuery("custom_blend", node="dst", exists=True))

    def test_rshift_unsupported_raises(self):
        node = Node.create("transform", name="cube1")
        with self.assertRaises(TypeError):
            _ = node.tx >> 5  # invalid RHS for >>


class TestPlugRshiftNamedClone(MayaTestCase):
    """``plug >> "name"`` clones onto the same node, ``plug >> "node.name"``
    onto another; both are the ``plug >> Node`` clone under a chosen name,
    plus the value. A taken name refuses: a clone never overwrites."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig.spec import Color, Float

        self.src = Node.create("transform", name="src")
        self.dst = Node.create("transform", name="dst")
        self.src << Float("blend", min=0, max=1) << 0.25
        self.src << Color("tint") << [0.1, 0.2, 0.3]

    def test_bare_name_clones_onto_the_same_node_with_the_value(self):
        before = set(cmds.ls())
        plug   = self.src.blend >> "blend2"
        self.assertIsInstance(plug, Plug)
        self.assertEqual(str(plug), "src.blend2")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.attributeQuery("blend2", node="src", attributeType=True), "double")
        self.assertAlmostEqual(cmds.getAttr("src.blend2"), 0.25)
        # the source is untouched and independent
        self.assertAlmostEqual(cmds.getAttr("src.blend"), 0.25)
        plug << 0.75
        self.assertAlmostEqual(cmds.getAttr("src.blend"), 0.25)

    def test_dotted_name_clones_onto_another_node(self):
        plug = self.src.blend >> "dst.mix"
        self.assertEqual(str(plug), "dst.mix")
        self.assertTrue(cmds.attributeQuery("mix", node="dst", exists=True))
        self.assertFalse(cmds.attributeQuery("mix", node="src", exists=True))
        self.assertAlmostEqual(cmds.getAttr("dst.mix"), 0.25)

    def test_compound_clones_as_a_compound(self):
        import numpy as np

        plug = self.src.tint >> "dst.tint"
        self.assertEqual(cmds.attributeQuery("tint", node="dst", attributeType=True), "double3")
        self.assertEqual(len(cmds.attributeQuery("tint", node="dst", listChildren=True)), 3)
        np.testing.assert_array_almost_equal(plug >> None, [0.1, 0.2, 0.3])
        same = self.src.tint >> "__tint__"
        self.assertEqual(str(same), "src.__tint__")
        self.assertEqual(same.num_children, 3)
        np.testing.assert_array_almost_equal(same >> None, [0.1, 0.2, 0.3])

    def test_underscore_names_work(self):
        plug = self.src.blend >> "__blend__"
        self.assertEqual(str(plug), "src.__blend__")
        self.assertAlmostEqual(self.src.__blend__ >> None, 0.25)
        self.assertIn("__blend__", cmds.listAttr("src", userDefined=True))

    def test_existing_name_refuses_and_writes_nothing(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "never overwrites"):
            self.src.blend >> "tint"
        with self.assertRaisesRegex(TypeError, "never overwrites"):
            self.src.blend >> "dst.translateX"
        with self.assertRaisesRegex(TypeError, "does not exist"):
            self.src.blend >> "nope.mix"
        with self.assertRaisesRegex(TypeError, "not one"):
            self.src.blend >> "1bad"
        with self.assertRaisesRegex(TypeError, "not one"):
            self.src.blend >> "dst."
        self.assertEqual(set(cmds.ls()), before)
        self.assertAlmostEqual(cmds.getAttr("src.tint")[0][0], 0.1)
        self.assertEqual(cmds.listAttr("src", userDefined=True), ["blend", "tint", "tintR", "tintG", "tintB"])

    def test_incoming_connection_is_not_moved_like_rshift_node(self):
        driver = Node.create("transform", name="driver")
        self.src.blend << driver.tx
        driver.tx      << 0.5
        named  = self.src.blend >> "blendCopy"
        cloned = self.src.blend >> self.dst
        for plug in (named, cloned):
            self.assertEqual(list(plug.get_inputs()), [])
        self.assertEqual(
            cmds.listConnections("src.blend", source=True, destination=False, plugs=True),
            ["driver.translateX"],
        )
        self.assertEqual(
            cmds.listConnections("driver.translateX", source=False, destination=True, plugs=True),
            ["src.blend"],
        )
        # the named clone carries the evaluated value, the Node clone the default
        self.assertAlmostEqual(cmds.getAttr("src.blendCopy"), 0.5)
        self.assertAlmostEqual(cmds.getAttr("dst.blend"), 0.0)


class TestPlugRshiftNumpy(MayaTestCase):
    """``Plug >> None`` returns numpy arrays for compound numeric attrs."""

    TEST_START_NEW_SCENE = True

    def test_vector_returns_1d_array(self):
        import numpy as np

        node = Node.create("transform", name="cube1")
        cmds.setAttr("cube1.t", 1.0, 2.0, 3.0, type="double3")
        value = node.t >> None
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.shape, (3,))
        np.testing.assert_array_almost_equal(value, [1.0, 2.0, 3.0])

    def test_matrix_returns_4x4_array(self):
        import numpy as np

        node  = Node.create("transform", name="cube1")
        value = node.matrix >> None
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.shape, (4, 4))

    def test_string_returns_python_str(self):
        node = Node.create("transform", name="cube1")
        from rig.spec import String

        node << String("label") << "hello"
        value = node.label >> None
        self.assertIsInstance(value, str)
        self.assertEqual(value, "hello")

    def test_scalar_returns_python_scalar(self):
        # Scalars stay as Python int/float -- not wrapped in numpy.
        node = Node.create("transform", name="cube1")
        node.tx << 2.5
        value = node.tx >> None
        self.assertNotIsInstance(value, type(__import__("numpy").array(0)))
        self.assertAlmostEqual(value, 2.5)


class TestPlugArithmetic(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_add_creates_sum(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = a.tx + b.tx
        # v3.B: native Maya 2024+ ``sum`` node replaces legacy plusMinusAverage.
        expected = "sum" if _has_native_math_nodes() else "plusMinusAverage"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)

    def test_subtract_creates_subtract(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = a.tx - b.tx
        # v3.B: native Maya 2024+ ``subtract`` node replaces legacy plusMinusAverage.
        expected = "subtract" if _has_native_math_nodes() else "plusMinusAverage"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)

    def test_multiply_creates_multiply(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = a.tx * b.tx
        # v3.B: native Maya 2024+ ``multiply`` node replaces legacy multiplyDivide.
        expected = "multiply" if _has_native_math_nodes() else "multiplyDivide"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)

    def test_negate(self):
        a      = Node.create("transform", name="a")
        result = -a.tx
        # v3.D: native Maya 2024+ ``negate`` node replaces multiplyDivide(-1).
        expected = "negate" if _has_native_math_nodes() else "multiplyDivide"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)

    def test_invert(self):
        # Pre-2024 ``~x`` is a condition network behind a published container,
        # so the node type says nothing useful --
        # ``TestPlugV3TNativeAtanAndNot.test_invert_pre_2024_uses_condition_fallback``
        # covers that path instead.
        _skip_without_native_math_nodes(self)
        a      = Node.create("transform", name="a")
        result = ~a.tx
        # v3.T: ``~x`` now uses native ``not`` node on Maya 2024+
        # (was ``subtract`` (1 - x) pre-v3.T). Proper logical-NOT
        # semantic for consistency with ``&`` / ``|`` / ``^``.
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "not")


class TestPlugComparisons(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_gt_creates_greater_than(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = a.tx > b.tx
        # v3.C: native Maya 2024+ ``greaterThan`` replaces legacy condition.
        expected = "greaterThan" if _has_native_math_nodes() else "condition"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)

    def test_eq_creates_equal(self):
        a      = Node.create("transform", name="a")
        result = a.tx == 5
        # v3.C: native Maya 2024+ ``equal`` replaces legacy condition.
        expected = "equal" if _has_native_math_nodes() else "condition"
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), expected)


class TestPlugHash(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_plug_is_hashable_in_dict(self):
        node  = Node.create("transform", name="cube1")
        plug1 = node.tx
        plug2 = node.tx
        # Same logical plug, even though Python objects may differ.
        d = {plug1: "value"}
        # plug2 should hash the same (MObjectHandle.hashCode + attr).
        self.assertEqual(d[plug2], "value")

    def test_plug_in_set(self):
        node = Node.create("transform", name="cube1")
        s    = {node.tx, node.tx}
        # Duplicate plug => size 1.
        self.assertEqual(len(s), 1)

    def test_equals_method_for_real_equality(self):
        # plug == other returns a condition node Plug, NOT a bool.
        # Use .equals() for true name-identity check.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertTrue(a.tx.equals(a.tx))
        self.assertFalse(a.tx.equals(b.tx))


class TestStringPlugConnection(MayaTestCase):
    """v2-base regression: string-typed dst should accept Plug src as a
    CONNECTION, not as a literal-string SET.

    Pre-fix behavior: ``texture.fileTextureName << shape.image`` stored
    the literal string ``"shape.image"`` (the source plug's full_name)
    as the destination's value, instead of building the live cmds
    connectAttr connection. Surfaced by Eric Vignola's
    ``perspective_image_planes`` example port.
    """

    TEST_START_NEW_SCENE = True

    def test_string_plug_to_string_plug_connects(self):
        from rig.spec import String

        # file.fileTextureName <- someShape.image  -> cmds.connectAttr
        cmds.createNode("file", name="tex1")
        shape = Node.create("transform", name="someShape")
        shape << String("image") << "/tmp/foo.jpg"

        Node("tex1").fileTextureName << shape.image

        # Verify CONNECTION (not literal value).
        connections = cmds.listConnections(
            "tex1.fileTextureName", source=True, destination=False, plugs=True
        )
        self.assertIsNotNone(connections, "expected an incoming connection")
        self.assertIn("someShape.image", connections)

        # Verify the literal value of fileTextureName is NOT "someShape.image"
        # (which is what the buggy code would have stored).
        value = cmds.getAttr("tex1.fileTextureName")
        self.assertNotEqual(
            value,
            "someShape.image",
            "dst value should reflect the live source value, not the source plug name",
        )

    def test_string_literal_to_string_plug_still_sets(self):
        # Verify the pre-existing behavior is preserved: assigning a
        # raw Python string literal still does a setAttr (not a connect).
        from rig.spec import String

        shape = Node.create("transform", name="otherShape")
        shape << String("image") << ""

        shape.image << "/path/to/asset.jpg"

        # Should be set as a literal value, no incoming connection.
        connections = cmds.listConnections(
            "otherShape.image", source=True, destination=False, plugs=True
        )
        self.assertIsNone(connections)
        self.assertEqual(cmds.getAttr("otherShape.image"), "/path/to/asset.jpg")


class TestPlugCompoundFanOut(MayaTestCase):
    """v3.P: Comparison and logical operators fan out per-channel for
    compound inputs.

    Regression: ``Plug.child(i)`` previously inherited
    ``Attribute.child`` which returns a bare :class:`Attribute`. Code
    paths in ``_condition_op`` / ``condition`` / ``_modulo`` that
    constructed a compound output and wrote per-channel results back
    into it via ``output_plugs[index] << ...`` failed with
    ``TypeError: unsupported operand type(s) for <<: 'Attribute' and 'Plug'``.

    v3.P overrides ``Plug.child(i)`` to return :class:`Plug` so all
    compound-fan-out call sites get plugs with the DSL operator overloads.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.polyCube(name="src")
        cmds.polyCube(name="dst")
        self._src = Node("src")
        self._dst = Node("dst")

    def test_logical_and_compound_scalar(self):
        # Was the user's original bug -- node.t & 5 must fan out per channel.
        result = self._src.t & 5
        self.assertIsInstance(result, Plug)
        # Result must be a 3-channel compound (matching the dst's t).
        self._dst.t << result
        # Successful injection means the connection completed without raising.

    def test_logical_and_compound_compound(self):
        result = self._src.t & self._src.r
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_logical_or_compound_scalar(self):
        result = self._src.t | 5
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_logical_xor_compound_scalar(self):
        result = self._src.t ^ 5
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_eq_compound_scalar(self):
        # node.t == 0 -- Maya returns 1.0 for matching channels, 0.0 else.
        result = self._src.t == 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_ne_compound_scalar(self):
        result = self._src.t != 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_lt_compound_scalar(self):
        result = self._src.t < 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_gt_compound_scalar(self):
        result = self._src.t > 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_le_compound_scalar(self):
        result = self._src.t <= 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_ge_compound_scalar(self):
        result = self._src.t >= 0
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_eq_compound_compound(self):
        result = self._src.t == self._src.r
        self.assertIsInstance(result, Plug)
        self._dst.t << result

    def test_plug_child_returns_plug(self):
        """Direct test of the override -- ``Plug.child(i)`` returns Plug."""
        from rig._internal.types import _is_compound

        t = self._src.t
        self.assertTrue(_is_compound(t))
        for i in range(t.num_children):
            child = t.child(i)
            self.assertIsInstance(child, Plug)


class TestPlugCompoundFanOutNativeNodes(MayaTestCase):
    """v3.P amend (Option B): compound fan-out uses Maya 2024+ native
    nodes per-channel instead of falling back to legacy expressions.

    Pre-amend bug: ``_logical_and(node.t, node.r)`` would skip the
    native ``and`` node fast-path the moment ANY input was compound,
    falling through to ``((a != 0) + (b != 0)) == 2`` which built a
    legacy condition-node network -- much bigger than necessary on
    Maya 2024+.

    After Option B refactor: each ``_logical_*`` / ``_condition_op``
    extracts a per-channel scalar helper used in BOTH the scalar
    fast path AND the compound fan-out, so a 2024+ user with compound
    input gets one native node per channel.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        _skip_without_native_math_nodes(self)
        super().setUp()
        # Force 2024+ target for these tests (same as user's repro).
        from rig import set_options

        set_options(maya_version=2026, create_containers=False)
        cmds.polyCube(name="src1")
        cmds.polyCube(name="src2")
        self._a = Node("src1")
        self._b = Node("src2")

    def tearDown(self):
        super().tearDown()
        # Reset target version to default for downstream tests.
        from rig import set_options

        set_options(maya_version=None, create_containers=True)

    def test_logical_and_compound_uses_three_native_and_nodes(self):
        # node.t & node.r -> 3 native ``and`` nodes (one per channel)
        _ = self._a.t & self._b.r
        self.assertEqual(len(cmds.ls(type="and")), 3)
        # No legacy condition fallback nodes used.
        self.assertEqual(len(cmds.ls(type="condition")), 0)

    def test_logical_or_compound_uses_three_native_or_nodes(self):
        _ = self._a.t | self._b.r
        self.assertEqual(len(cmds.ls(type="or")), 3)

    def test_greater_than_compound_uses_three_native_greaterthan_nodes(self):
        _ = self._a.t > self._b.r
        self.assertEqual(len(cmds.ls(type="greaterThan")), 3)
        self.assertEqual(len(cmds.ls(type="condition")), 0)

    def test_less_than_compound_uses_three_native_lessthan_nodes(self):
        _ = self._a.t < self._b.r
        self.assertEqual(len(cmds.ls(type="lessThan")), 3)
        self.assertEqual(len(cmds.ls(type="condition")), 0)

    def test_eq_compound_uses_three_native_equal_nodes(self):
        _ = self._a.t == self._b.r
        self.assertEqual(len(cmds.ls(type="equal")), 3)
        self.assertEqual(len(cmds.ls(type="condition")), 0)

    def test_modulo_compound_uses_three_native_modulo_nodes(self):
        _ = self._a.t % self._b.r
        self.assertEqual(len(cmds.ls(type="modulo")), 3)

    def test_logical_and_scalar_uses_one_native_and_node(self):
        # Single-channel case still works unchanged.
        _ = self._a.tx & self._b.rx
        self.assertEqual(len(cmds.ls(type="and")), 1)


class TestPlugCompoundFanOutPre2024(MayaTestCase):
    """Pre-2024 Maya targets fall back to the legacy condition-node
    expression for logical/comparison ops (no native nodes available).
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig import set_options

        set_options(maya_version=2022, create_containers=False)
        cmds.polyCube(name="src1")
        cmds.polyCube(name="src2")
        self._a = Node("src1")
        self._b = Node("src2")

    def tearDown(self):
        super().tearDown()
        from rig import set_options

        set_options(maya_version=None, create_containers=True)

    def test_logical_and_compound_uses_legacy_condition_nodes(self):
        # Pre-2024 has no native ``and`` node -- uses the
        # ``((a != 0) + (b != 0)) == 2`` expression which builds
        # condition nodes.
        _ = self._a.t & self._b.r
        self.assertEqual(len(cmds.ls(type="and")), 0)
        self.assertGreater(len(cmds.ls(type="condition")), 0)


class TestPlugV3QNodeOpDispatch(MayaTestCase):
    """v3.Q: comparison and logical operators dispatch through the
    :class:`NodeOp` framework (instead of being plain memoized functions).

    User-visible behavior is identical to v3.P (same operator overloads,
    same wrapper functions, same node networks). What changes:
    * the channel-count mismatch check in ``_node_ops._invoke`` now
      applies to comparison/logical ops too (previously only arithmetic)
    * cache + version dispatch use the NodeOp framework's machinery,
      uniform with arithmetic ops
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig import set_options

        set_options(maya_version=2026, create_containers=False)
        cmds.polyCube(name="src1")
        cmds.polyCube(name="src2")
        self._a = Node("src1")
        self._b = Node("src2")

    def tearDown(self):
        super().tearDown()
        from rig import set_options

        set_options(maya_version=None, create_containers=True)

    def test_condition_op_is_nodeop_instance(self):
        from rig._internal.math_nodes import _condition_op
        from rig._internal.node_ops import NodeOp

        self.assertIsInstance(_condition_op, NodeOp)

    def test_logical_and_is_nodeop_instance(self):
        from rig._internal.math_nodes import _logical_and
        from rig._internal.node_ops import NodeOp

        self.assertIsInstance(_logical_and, NodeOp)

    def test_logical_or_is_nodeop_instance(self):
        from rig._internal.math_nodes import _logical_or
        from rig._internal.node_ops import NodeOp

        self.assertIsInstance(_logical_or, NodeOp)

    def test_logical_xor_is_nodeop_instance(self):
        from rig._internal.math_nodes import _logical_xor
        from rig._internal.node_ops import NodeOp

        self.assertIsInstance(_logical_xor, NodeOp)

    def test_modulo_is_nodeop_instance(self):
        from rig._internal.math_nodes import _modulo
        from rig._internal.node_ops import NodeOp

        self.assertIsInstance(_modulo, NodeOp)


class TestPlugV3QChannelCountEnforcement(MayaTestCase):
    """v3.Q: NodeOp framework's channel-count mismatch check now applies
    to comparison/logical operators too (previously only arithmetic ops).

    Mismatched compound shapes (e.g. 3-vector & 4-quaternion) raise
    ``ValueError`` instead of silently truncating via ``zip()``.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig import set_options

        set_options(maya_version=2026, create_containers=False)
        cmds.polyCube(name="src1")
        # Add a 4-channel quat-shaped attribute to src1 for mismatch testing.
        cmds.addAttr(
            "src1", longName="myQuat", attributeType="compound", numberOfChildren=4
        )
        cmds.addAttr("src1", longName="qX", attributeType="double", parent="myQuat")
        cmds.addAttr("src1", longName="qY", attributeType="double", parent="myQuat")
        cmds.addAttr("src1", longName="qZ", attributeType="double", parent="myQuat")
        cmds.addAttr("src1", longName="qW", attributeType="double", parent="myQuat")
        self._a = Node("src1")

    def tearDown(self):
        super().tearDown()
        from rig import set_options

        set_options(maya_version=None, create_containers=True)

    def test_logical_and_mismatched_channel_counts_raises(self):
        # Mixing 3-vector ``t`` with 4-channel ``myQuat`` must raise.
        with self.assertRaises(ValueError):
            _ = self._a.t & self._a.myQuat

    def test_eq_mismatched_channel_counts_raises(self):
        with self.assertRaises(ValueError):
            _ = self._a.t == self._a.myQuat


class TestPlugV3TNativeAtanAndNot(MayaTestCase):
    """v3.T: native ``atan`` and ``not`` node usage.

    * ``atand(x)`` now uses Maya 2024+ native ``atan`` node (was
      angleBetween-only); compound input fans out per-channel.
    * ``~x`` now uses Maya 2024+ native ``not`` node (was ``1 - x``
      via ``subtract``); compound input fans out per-channel.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig import set_options

        set_options(maya_version=2026, create_containers=False)
        cmds.polyCube(name="src")
        self._src = Node("src")

    def tearDown(self):
        super().tearDown()
        from rig import set_options

        set_options(maya_version=None, create_containers=True)

    def test_atand_uses_native_atan_node_on_2024(self):
        _skip_without_native_math_nodes(self)
        from rig.trigonometry import atand

        _ = atand(self._src.tx)
        self.assertEqual(len(cmds.ls(type="atan")), 1)
        # No fallback angleBetween used.
        self.assertEqual(len(cmds.ls(type="angleBetween")), 0)

    def test_atand_compound_fans_out_to_three_native_atan_nodes(self):
        _skip_without_native_math_nodes(self)
        from rig.trigonometry import atand

        _ = atand(self._src.t)
        self.assertEqual(len(cmds.ls(type="atan")), 3)

    def test_atand_pre_2024_uses_angleBetween(self):
        from rig import set_options
        from rig.trigonometry import atand

        set_options(maya_version=2022, create_containers=False)
        _ = atand(self._src.tx)
        self.assertEqual(len(cmds.ls(type="atan")), 0)
        self.assertGreater(len(cmds.ls(type="angleBetween")), 0)

    def test_invert_uses_native_not_node_on_2024(self):
        _skip_without_native_math_nodes(self)
        result = ~self._src.tx
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "not")
        self.assertEqual(len(cmds.ls(type="not")), 1)

    def test_invert_compound_fans_out_to_three_native_not_nodes(self):
        _skip_without_native_math_nodes(self)
        _ = ~self._src.t
        self.assertEqual(len(cmds.ls(type="not")), 3)

    def test_invert_pre_2024_uses_condition_fallback(self):
        from rig import set_options

        set_options(maya_version=2022, create_containers=False)
        _ = ~self._src.tx
        # Pre-2024: no native `not` node -> uses condition(x != 0, 0, 1).
        self.assertEqual(len(cmds.ls(type="not")), 0)
        self.assertGreater(len(cmds.ls(type="condition")), 0)

    def test_logical_not_wrapper_matches_invert_operator(self):
        from rig import functions as f, set_options

        # Operator and wrapper must agree on whatever network the running
        # Maya builds, so drop the class-wide 2024+ target override.
        set_options(maya_version=None)
        op_result      = ~self._src.tx
        wrapper_result = f.logical_not(self._src.tx)
        self.assertEqual(
            cmds.nodeType(str(op_result).split(".")[0]),
            cmds.nodeType(str(wrapper_result).split(".")[0]),
        )


class TestPlugV3UComponentConnections(MayaTestCase):
    """v3.U fix: connecting matrix-derived translations to component
    plugs (cv, vtx, etc.) must NOT crash in absorb_unit_conversions.

    Pre-v3.U: ``rail_shape.cv[:] << ctrls.wm * rail.wim`` raised
    ``TypeError: item is not a plug`` because Attribute() can't wrap
    kComponent items. Post-v3.U: gracefully skip absorption for
    components.
    """

    TEST_START_NEW_SCENE = True

    def test_matrix_to_cv_does_not_raise(self):
        # Build a minimal curve and connect a control's translation to a CV.
        from rig import container

        ctrl = Node.create("transform", name="cv_ctrl")
        ctrl.t << [1, 2, 3]
        crv        = cmds.curve(d=1, p=[(0, 0, 0), (1, 0, 0)])
        crv_shape  = cmds.listRelatives(crv, type="nurbsCurve")[0]
        rail_shape = Node(crv_shape)

        with container("cv_test1"):
            # This previously raised -- now should connect cleanly.
            rail_shape.cv[0] << ctrl.wm * Node.create("transform", name="rail").wim

        # Verify CV ended up driven (has incoming connections via decompose chain).
        connections = cmds.listConnections(
            crv_shape + ".cv[0]", source=True, destination=False
        )
        self.assertTrue(connections)


class TestPlugV3UChoicePolymorphicLiterals(MayaTestCase):
    """v3.U fix: ``functions.choice([0, plug, plug, ...])`` with literal
    tokens must auto-wrap them in ``_constant`` so the polymorphic
    ``choice.input[i]`` slot gets a type via connection (not setAttr).

    Pre-v3.U: raised ``AttributeError: 'choice1.input[0]' has no child
    or sibling attribute 'data_type'``.
    """

    TEST_START_NEW_SCENE = True

    def test_choice_with_numeric_literal_first(self):
        from rig import functions as f

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        # Literal int as first token -- pre-v3.U this crashed
        result = f.choice([0, a.tx, b.tx], selector=a.ty)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "choice")

    def test_choice_with_vector_literal(self):
        from rig import functions as f

        a = Node.create("transform", name="a")
        # All-literal vector tokens
        result = f.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]], selector=a.tx)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "choice")

    def test_choice_mixed_literal_and_plug(self):
        from rig import functions as f

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        # Mix: int literal, Plug, int literal
        result = f.choice([0, a.tx, 5], selector=b.tx)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "choice")


class TestPlugDataTypeTypedOutput(MayaTestCase):
    """Reading ``.data_type`` on a generic ("typed"/"Tdata") attr -- e.g. a
    ``choice`` node's ``output`` -- must resolve via the owning node's
    type-aware ``_attr_data_type_fallback`` hook instead of raising.

    ``Attribute.data_type`` delegates to ``self.node._attr_data_type_fallback``
    for typed attrs, but the rig ``Node.__getattr__`` rejects every
    ``_``-prefixed name, so that hook never reached the wrapped ``DGNode``
    (the type-aware ``Choice`` implementation). ``Plug(choice.output).data_type``
    therefore raised ``AttributeError: '...output' has no child or sibling
    attribute 'data_type'``. The base ``Attribute`` path always worked because
    its ``.node`` is the typed ``DGNode``; only the rig ``Plug`` path was broken.
    """

    TEST_START_NEW_SCENE = True

    def test_choice_output_data_type_resolves(self):
        from rig.maya.nodetypes import Attribute
        from rig import functions as f

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        # A choice output is a generic ("typed") attr; with selector 0 it
        # resolves to input[0]'s source type (a.translateX -> doubleLinear).
        out = f.choice([a.tx, b.tx], selector=a.ty)

        # Must NOT raise, and the rig Plug path must agree with the base
        # Attribute path (which dispatches to the typed Choice node).
        self.assertEqual(out.data_type, "doubleLinear")
        self.assertEqual(out.data_type, Attribute(str(out)).data_type)

    def test_choice_output_data_type_via_string_plug(self):
        # The same must hold for a Plug rebuilt from a string (the path the
        # DSL takes when resolving published names / sibling fallbacks).
        from rig import functions as f

        a        = Node.create("transform", name="a")
        b        = Node.create("transform", name="b")
        out_name = str(f.choice([a.tx, b.tx], selector=a.ty))

        self.assertEqual(Plug(out_name).data_type, "doubleLinear")


class TestPlugV3USlerpDivWrites(MayaTestCase):
    """v3.U fix: ``interpolate.slerp`` must work via
    ``interpolate.sequence(method=slerp)``. Pre-v3.U the implementation
    used ``div = expr / 1; div.operation << ...`` which broke after
    v3.Q's NodeOp framework conversion (sibling-attribute fallback
    didn't fire on NodeOp-returned plugs).

    Refactored to write to the divide node's ``operation`` directly.
    """

    TEST_START_NEW_SCENE = True

    def test_slerp_two_vectors_does_not_raise(self):
        from rig import slerp

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        # This previously raised AttributeError; now should return a Plug.
        result = slerp(a.t, b.t, weight=0.5)
        self.assertIsNotNone(result)

    def test_sequence_with_custom_method(self):
        # ``sequence`` accepts a non-default cross-type interpolation verb as
        # ``method``. It publishes ``yp`` as a scalar ``double`` multi, so the
        # method is invoked on SCALAR segment endpoints -- ``elerp`` (valid for
        # scalars) exercises the custom-method plumbing. (``slerp`` is
        # rotation-only and correctly rejects the scalar endpoints under the
        # Model-A dispatch; its div-write regression is covered directly by
        # ``test_slerp_two_vectors_does_not_raise`` /
        # ``test_slerp_compound_output_per_channel_values``.)
        from rig import constant, elerp, interpolate as interp

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        c = Node.create("transform", name="c")

        result = interp.sequence(
            a.tx,
            xp     = [constant(0.0), constant(0.5), constant(1.0)],
            yp     = [a.tx, b.tx, c.tx],
            method = elerp,
        )
        self.assertIsNotNone(result)

    def test_slerp_compound_output_per_channel_values(self):
        """v3.U-amend regression: slerp must preserve compound output
        across all 3 channels -- not broadcast the X channel.

        Pre-fix, the slerp impl wrote ``input2X << sine`` and returned
        ``div_node.outputX`` (per-channel writes), which collapsed the
        result to scalar X and then broadcast that scalar to Y/Z when
        connected to a compound dst (e.g. ``c.t``). Result was always
        uniform like ``(0.3827, 0.3827, 0.3827)`` instead of the true
        per-channel slerp.

        This test pins the exact numerical output for a known geometry:
        slerp([0,1,0], [1,0,0], 0.25) -> (sin(22.5deg), cos(22.5deg), 0).
        """
        from rig import slerp

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        c = Node.create("transform", name="c")
        a.t << [0, 1, 0]
        b.t << [1, 0, 0]
        c.t << slerp(a.t, b.t, 0.25)

        # 90deg angle between [0,1,0] and [1,0,0]; slerp at 0.25 gives 22.5deg.
        # cos(22.5deg) ~= 0.92388, sin(22.5deg) ~= 0.38268
        result = cmds.getAttr("c.t")[0]
        self.assertAlmostEqual(result[0], 0.38268343, places=5)
        self.assertAlmostEqual(result[1], 0.92387950, places=5)
        self.assertAlmostEqual(result[2], 0.0,        places=5)
        # CRITICAL: not a scalar broadcast to all 3 channels
        self.assertNotAlmostEqual(result[0], result[1], places=3)


class TestChoiceCompoundPreservation(MayaTestCase):
    """v2 base fix: ``f.choice([compound_a, compound_b], ...)`` must
    preserve the compound nature of its outputs through downstream math
    operations.

    Pre-fix, the rail_spine port's rider scale came out uniform (e.g.
    ``(5, 5, 5)``) instead of the expected non-uniform pattern
    ``(5, 0.1, 5)`` because:
      1. ``f.choice`` connections downgraded src compound -> first child
         (``src.scaleX -> choice.input[0]``) due to fan-out path in
         ``_inject_value`` truncating to dst's single slot
      2. ``_is_compound(choice.output)`` returned False (generic
         attributes report 0 structural children), so downstream
         ``elerp(choice.output, ...)`` routed through scalar paths
         (``multiplyDivide.outputX`` instead of ``.output``)
      3. The scalar result connected to ``probe.s`` broadcast to all
         3 channels

    Fix in ``_is_compound`` (``_types.py``): detect generic-typed
    compound by checking ``data_type`` runtime info. Mirrors
    Eric Vignola's third_party.rig handling.
    """

    TEST_START_NEW_SCENE = True

    def test_choice_with_compound_inputs_preserves_compound_output(self):
        from rig import functions as f
        from rig._internal.types import _is_compound
        from rig.spec import Float

        a = Node.create("transform", name="a")
        a.s << [5, 0.1, 5]
        b = Node.create("transform", name="b")
        b.s << [1, 1, 1]
        sel = Node.create("transform", name="sel")
        sel << Float("w")

        result = f.choice([a.s, b.s], selector=sel.w)

        # _is_compound must recognize the choice output as compound
        # (data_type == "double3" via runtime check).
        self.assertTrue(_is_compound(result))
        self.assertEqual(result.data_type, "double3")

    def test_choice_compound_pipes_through_math_ops(self):
        from rig import functions as f
        from rig.spec import Float

        a = Node.create("transform", name="a")
        a.s << [5, 0.1, 5]
        sel = Node.create("transform", name="sel")
        sel << Float("w")

        # choice -> multiply (via __pow__ at exponent=2) -> probe
        # __pow__ uses _multiply_divide which routes scalar/compound
        # based on _is_compound. With the fix, this stays compound.
        choice_out = f.choice([a.s], selector=sel.w)
        squared    = choice_out**2

        probe = Node.create("transform", name="probe")
        probe.s << squared
        sel.w   << 0  # pick a.s = (5, 0.1, 5)
        # Expected: (25, 0.01, 25). Tolerance for FP.
        self.assertAlmostEqual(cmds.getAttr("probe.sx"), 25.0, places=3)
        self.assertAlmostEqual(cmds.getAttr("probe.sy"), 0.01, places=3)
        self.assertAlmostEqual(cmds.getAttr("probe.sz"), 25.0, places=3)
        # Critically: NOT broadcast -- sy != sx
        self.assertNotEqual(cmds.getAttr("probe.sx"), cmds.getAttr("probe.sy"))


class TestInjectSetPathCoverage(MayaTestCase):
    """Coverage for the value-set dispatch paths in ``_set_or_connect`` /
    ``_inject_value`` (matrix routing, scalar->compound broadcast, and the
    raw-string ``dst_attr is None`` defensive fallbacks)."""

    TEST_START_NEW_SCENE = True

    def test_scalar_broadcasts_across_compound_children(self):
        # End-to-end: ``<<`` fans a scalar out across a compound's channels.
        n = Node.create("transform", name="n")
        n.t << 5.0
        self.assertEqual(tuple(cmds.getAttr("n.translate")[0]), (5.0, 5.0, 5.0))

    def test_set_or_connect_scalar_into_compound_attr_broadcasts(self):
        # Direct call: a SCALAR src against a COMPOUND dst takes the
        # ``_is_compound(dst_attr)`` broadcast branch inside ``_set_or_connect``
        # (the ``<<`` operator normally pre-fans-out, so this branch is only
        # reached on a direct call).
        from rig.maya.nodetypes import Attribute
        from rig._internal import plug as plugmod

        n = Node.create("transform", name="n")
        plugmod._set_or_connect(7.0, Attribute("n.translate"))
        self.assertEqual(tuple(cmds.getAttr("n.translate")[0]), (7.0, 7.0, 7.0))

    def test_set_or_connect_numeric_raw_string_success_returns(self):
        # The ``dst_attr is None`` numeric fallback's SUCCESS ``return`` path:
        # an uncoercible raw-string dst whose ``cmds.setAttr`` succeeds (mocked)
        # must return cleanly without raising.
        from unittest import mock

        from rig._internal import plug as plugmod

        with mock.patch.object(plugmod.cmds, "setAttr") as mock_setattr:
            plugmod._set_or_connect(5.0, "bogus_node.nope")
        mock_setattr.assert_called_once_with("bogus_node.nope", 5.0)

    def test_flat16_into_matrix_attr(self):
        import numpy as np

        m = Node.create("transform", name="m")
        cmds.addAttr("m", longName="mtx", dataType="matrix")
        ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
        m.mtx << np.array(ident, dtype=float)
        self.assertEqual([round(v, 3) for v in cmds.getAttr("m.mtx")], ident)

    def test_set_or_connect_numeric_raw_string_fallback_raises(self):
        from rig._internal import plug as plugmod

        with self.assertRaises(plugmod.InjectionError):
            plugmod._set_or_connect(5.0, "bogus_node.nope")

    def test_set_or_connect_string_raw_string_fallback_raises(self):
        from rig._internal import plug as plugmod

        with self.assertRaises(plugmod.InjectionError):
            plugmod._set_or_connect("hello", "bogus_node.nope")

    def test_set_or_connect_matrix_sequence_into_attr(self):
        from rig.maya.nodetypes import Attribute
        from rig._internal import plug as plugmod

        m = Node.create("transform", name="m")
        cmds.addAttr("m", longName="mtx", dataType="matrix")
        ident = [float(v) for v in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1)]
        plugmod._set_or_connect(ident, Attribute("m.mtx"))
        self.assertEqual([round(v, 3) for v in cmds.getAttr("m.mtx")], ident)

    def test_set_or_connect_matrix_sequence_raw_string_fallback_raises(self):
        from unittest import mock

        from rig._internal import plug as plugmod

        fake_dst           = mock.MagicMock()
        fake_dst.data_type = "matrix"
        fake_dst.__str__   = lambda self: "bogus_node.nope"
        with self.assertRaises(plugmod.InjectionError):
            plugmod._set_or_connect([float(i) for i in range(16)], fake_dst)


class TestPowOperator(MayaTestCase):
    """``**`` on matrix / quaternion plugs delegates to the named
    ``matrix.pow`` / ``quaternion.pow`` verbs (fractional transform /
    rotation toward identity). Non-numeric or bool exponents raise
    ``TypeError``. The reflected form (``scalar ** matrix|quaternion``)
    is undefined and raises.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_quat_operator_matches_named_verb(self):
        from rig import matrix as m, quaternion

        loc = cmds.spaceLocator()[0]
        cmds.setAttr(loc + ".rotateY", 90)
        q = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        self.assertAlmostEqual(
            cmds.getAttr(f"{q**0.5}Y"),
            cmds.getAttr(f"{quaternion.pow(q, 0.5)}Y"),
            places=4,
        )

    def test_matrix_operator_matches_named_verb(self):
        from rig import matrix as m

        b = Node.create("transform", name="b")
        b.tx << 10.0
        mtx = b.worldMatrix[0]
        dm1 = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(mtx**0.5), f"{dm1}.inputMatrix")
        dm2 = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(m.pow(mtx, 0.5)), f"{dm2}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm1}.outputTranslate")[0][0],
            cmds.getAttr(f"{dm2}.outputTranslate")[0][0],
            places=4,
        )

    def test_nonscalar_exponent_raises(self):
        from rig import matrix as m

        loc = cmds.spaceLocator()[0]
        q   = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        with self.assertRaises(TypeError):
            _ = q**q
        with self.assertRaises(TypeError):
            _ = q**True  # bool rejected

    def test_rpow_raises_for_quat(self):
        from rig import matrix as m

        loc = cmds.spaceLocator()[0]
        q   = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        with self.assertRaises(TypeError):
            _ = 2**q

    def test_matrix_times_scalar_raises(self):
        mtx = Node.create("transform", name="a").worldMatrix[0]
        with self.assertRaises(TypeError):
            _ = mtx * 0.5
        with self.assertRaises(TypeError):
            _ = 0.5 * mtx

    def test_quaternion_times_scalar_raises(self):
        from rig import matrix as m

        loc = cmds.spaceLocator()[0]
        q   = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        with self.assertRaises(TypeError):
            _ = q * 0.5
        with self.assertRaises(TypeError):
            _ = 0.5 * q

    def test_pow_accepts_scalar_plug_exponent(self):
        # ``m ** <scalar plug>`` (a live weight plug, not a literal) must
        # route to matrix.pow, not raise -- the plug drives the blend weight.
        b = Node.create("transform", name="b")
        b.tx << 10.0
        ctrl = Node.create("transform", name="ctrl")
        ctrl.tx << 0.5  # live weight
        out = b.worldMatrix[0] ** ctrl.tx
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        # weight 0.5 -> translate halfway from identity to b: 5.0
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 5.0, places=3
        )

    def test_matrix_times_vector_still_point_transforms(self):
        # regression: matrix * vec3 (a list) must STILL do point transform, not raise
        mtx = Node.create("transform", name="a").worldMatrix[0]
        out = mtx * [1.0, 2.0, 3.0]
        self.assertIsNotNone(out)

    def test_vector_times_scalar_still_scales(self):
        # regression: vector * scalar must STILL be componentwise scale
        n   = Node.create("transform", name="n")
        out = n.translate * 0.5
        self.assertIsNotNone(out)


class TestScalarPowUnchanged(MayaTestCase):
    """Regression: ``**`` on SCALAR plugs must STILL route to the existing
    componentwise power op (``multiplyDivide`` op=3 pre-2024 / native
    ``power`` on 2024+), NOT to ``matrix.pow`` / ``quaternion.pow``.
    The ``**`` dispatch added the matrix/quaternion verbs but must leave
    scalar exponentiation alone."""

    TEST_START_NEW_SCENE = True

    def test_scalar_pow_still_componentwise(self):
        # ``scalar ** scalar`` must build a power network and yield a
        # usable plug whose evaluated value matches the math.
        n = Node.create("transform", name="n")
        n.tx << 3.0
        out = n.tx**2
        self.assertIsNotNone(out)
        # 3 ** 2 == 9
        self.assertAlmostEqual(cmds.getAttr(str(out)), 9.0, places=3)


class TestPlugGetInputsOutputs(MayaTestCase):
    """Connection queries as methods, replacing the retired
    ``plug << PlugList`` / ``plug >> PlugList`` sentinels."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="drv")
        cmds.createNode("transform", name="ctrl")
        cmds.createNode("transform", name="spare")
        cmds.connectAttr("drv.translateX",  "ctrl.translateX")
        cmds.connectAttr("ctrl.translateX", "spare.translateX")
        cmds.connectAttr("ctrl.translateX", "spare.translateY")

    def test_get_inputs_returns_the_driver(self):
        got = Node("ctrl").tx.get_inputs()
        self.assertIsInstance(got, PlugList)
        self.assertEqual([str(x) for x in got], ["drv.translateX"])

    def test_get_outputs_returns_every_destination(self):
        got = Node("ctrl").tx.get_outputs()
        self.assertIsInstance(got, PlugList)
        self.assertEqual(
            sorted(str(x) for x in got), ["spare.translateX", "spare.translateY"]
        )

    def test_unwired_yields_empty_pluglist_never_none(self):
        for got in (Node("ctrl").ty.get_inputs(), Node("ctrl").ty.get_outputs()):
            self.assertIsInstance(got, PlugList)
            self.assertEqual(len(got), 0)
            self.assertFalse(got)

    def test_compound_reports_direct_connections_only(self):
        self.assertEqual(len(Node("ctrl").t.get_inputs()), 0)
        self.assertEqual(
            [str(x) for x in Node("ctrl").t[:].get_inputs()[0]], ["drv.translateX"]
        )

    def test_empty_result_cannot_silently_disconnect(self):
        # A bare ``None`` return would be read by ``<<`` as "disconnect" and
        # tear the existing wire down. An empty PlugList raises instead.
        cmds.connectAttr("drv.translateY", "spare.translateZ")
        with self.assertRaises(ValueError):
            Node("spare").tz << Node("ctrl").ty.get_inputs()
        self.assertEqual(
            cmds.listConnections("spare.translateZ", s=True, d=False, p=True),
            ["drv.translateY"],
        )

    def test_retired_sentinels_raise(self):
        p = Node("ctrl").tx
        for expr in (
            lambda: p << PlugList,
            lambda: p >> PlugList,
            lambda: p << Plug,
            lambda: p >> Plug,
        ):
            with self.assertRaises(TypeError):
                expr()

    def test_pluglist_instance_still_connects(self):
        Node("spare").tz << PlugList([Node("drv").ty])
        self.assertEqual(
            cmds.listConnections("spare.translateZ", s=True, d=False, p=True),
            ["drv.translateY"],
        )


class TestPlugFanOutNoneAndSkip(MayaTestCase):
    """Per-channel fan-out speaks the same vocabulary a whole plug does:
    ``None`` disconnects, an ``_AttrSpec`` applies, ``skip`` is a no-op."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._build()

    def _build(self):
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="drv")
        cmds.createNode("transform", name="ctrl")
        for axis in "XYZ":
            cmds.connectAttr(f"drv.translate{axis}", f"ctrl.translate{axis}")

    def _drivers(self):
        return [
            bool(cmds.listConnections(f"ctrl.t{a}", s=True, d=False, p=True))
            for a in "xyz"
        ]

    def test_none_in_list_disconnects(self):
        Node("ctrl").t << [None, 4.0, None]
        self.assertEqual(self._drivers(), [False, False, False])
        self.assertAlmostEqual(cmds.getAttr("ctrl.ty"), 4.0, places=4)

    def test_skip_in_list_leaves_the_channel_alone(self):
        Node("ctrl").t << [skip, 4.0, skip]
        self.assertEqual(self._drivers(), [True, False, True])
        self.assertAlmostEqual(cmds.getAttr("ctrl.ty"), 4.0, places=4)

    def test_spec_in_list_applies(self):
        Node("ctrl").t << [lock, 4.0, lock]
        self.assertEqual(
            [cmds.getAttr(f"ctrl.t{a}", lock=True) for a in "xyz"],
            [True, False, True],
        )

    def test_sliced_and_unsliced_forms_agree(self):
        Node("ctrl").t << [skip, 4.0, skip]
        unsliced = self._drivers()
        self._build()
        Node("ctrl").t[:] << [skip, 4.0, skip]
        self.assertEqual(unsliced, self._drivers())

    def test_bare_skip_is_a_noop(self):
        Node("ctrl").tx << skip
        Node("ctrl").t  << skip
        self.assertEqual(self._drivers(), [True, True, True])

    def test_plain_values_still_fan_out(self):
        Node("ctrl").t << [1.0, 2.0, 3.0]
        self.assertEqual(self._drivers(), [False, False, False])
        self.assertAlmostEqual(cmds.getAttr("ctrl.tz"), 3.0, places=4)


class TestPlugContainment(MayaTestCase):
    """``in`` on a Plug is plain ``str`` containment and builds no nodes."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="ctrl")

    def test_substring_semantics_match_str(self):
        p = Plug("ctrl.translateX")
        self.assertIn(".", p)
        self.assertIn("translate", p)
        self.assertIn(Plug("ctrl.translateX"), p)
        self.assertNotIn("rotate", p)

    def test_containment_builds_no_nodes(self):
        p      = Plug("ctrl.translateX")
        before = set(cmds.ls())
        self.assertIn(".", p)
        self.assertNotIn("rotate", p)
        self.assertEqual(set(cmds.ls()), before)


class TestPlugStrMethodShadowing(MayaTestCase):
    """Plug subclasses str so ``cmds`` accepts it, but every public str
    method is routed to attribute lookup -- those names are all legal Maya
    attribute names. Cast to ``str`` for the string method."""

    TEST_START_NEW_SCENE = True

    # The 8 str method names that collide with real built-in Maya attrs.
    COLLIDING = (
        "center",
        "count",
        "format",
        "index",
        "join",
        "partition",
        "title",
        "translate",
    )

    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="ctrl")

    def test_colliding_names_resolve_to_attributes(self):
        for name in self.COLLIDING:
            if not cmds.attributeQuery(name, node="ctrl", exists=True):
                cmds.addAttr("ctrl", longName=name, attributeType="double")
        p = Plug("ctrl.rotateX")
        for name in self.COLLIDING:
            self.assertIsInstance(getattr(p, name), Plug, name)

    def test_every_public_str_method_is_routed(self):
        for name in (n for n in vars(str) if not n.startswith("_")):
            self.assertIsInstance(getattr(Plug, name), property, name)

    def test_string_methods_reachable_via_cast(self):
        self.assertEqual(str(Plug("ctrl.rotateX")).upper(), "CTRL.ROTATEX")


class TestPlugFanOutSpecSlotsAndLock(MayaTestCase):
    """A spec slot is a modifier, not a value-set, so it must not trip the
    all-or-nothing lock check -- matching what ``plug << unlock`` already does
    on a scalar, where ``__lshift__`` returns before ``_inject_value`` runs."""

    TEST_START_NEW_SCENE = True

    def _build(self, locked_axis="X"):
        cmds.file(new=True, force=True)
        for name in ("drv", "ctrl"):
            cmds.createNode("transform", name=name)
        for axis in "XYZ":
            cmds.connectAttr(f"drv.translate{axis}", f"ctrl.translate{axis}")
        if locked_axis:
            cmds.setAttr(f"ctrl.translate{locked_axis}", lock=True)

    def test_skip_slot_survives_a_locked_channel(self):
        self._build("X")
        Node("ctrl").t << [skip, 4.0, skip]
        self.assertAlmostEqual(cmds.getAttr("ctrl.ty"), 4.0, places=4)
        self.assertTrue(cmds.getAttr("ctrl.tx", lock=True))
        self.assertTrue(cmds.listConnections("ctrl.tx", s=True, d=False))

    def test_sliced_form_agrees(self):
        self._build("X")
        Node("ctrl").t[:] << [skip, 4.0, skip]
        self.assertAlmostEqual(cmds.getAttr("ctrl.ty"), 4.0, places=4)
        self.assertTrue(cmds.getAttr("ctrl.tx", lock=True))

    def test_unlock_slot_can_reach_a_locked_channel(self):
        self._build("X")
        Node("ctrl").t << [unlock, 4.0, unlock]
        self.assertFalse(cmds.getAttr("ctrl.tx", lock=True))

    def test_lock_slot_survives_an_already_locked_channel(self):
        self._build("X")
        Node("ctrl").t << [lock, 4.0, lock]
        self.assertEqual(
            [cmds.getAttr(f"ctrl.t{a}", lock=True) for a in "xyz"],
            [True, False, True],
        )

    def test_value_slot_on_a_locked_channel_still_raises(self):
        self._build("Y")
        with self.assertRaises(InjectionError):
            Node("ctrl").t << [skip, 4.0, skip]
        self.assertAlmostEqual(cmds.getAttr("ctrl.ty"), 0.0, places=4)

    def test_all_or_nothing_preserved_for_plain_values(self):
        self._build("X")
        with self.assertRaises(InjectionError):
            Node("ctrl").t << [1.0, 2.0, 3.0]
        # Nothing mutated -- the contract is all-or-nothing.
        self.assertEqual(
            [round(cmds.getAttr(f"ctrl.t{a}"), 4) for a in "xyz"], [0.0, 0.0, 0.0]
        )

    def test_all_or_nothing_preserved_for_scalar_broadcast(self):
        self._build("X")
        with self.assertRaises(InjectionError):
            Node("ctrl").t << 5.0