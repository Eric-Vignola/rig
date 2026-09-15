"""Tests for the operation-first dispatch verbs in ``rig._dispatch``.

The eleven top-level verbs (``dist`` / ``lerp`` / ``slerp`` / ``elerp`` /
``angle`` / ``angle_degrees`` / ``normalize`` / ``inverse`` / ``to_euler`` /
``to_quaternion`` / ``to_matrix``) classify their first operand via
``math_type`` and route to the per-type private impl. These tests pin the
routing table (which type -> which node), the rejected ``(verb, type)`` pairs
(``TypeError``), ``PlugList`` vectorisation, and the ``matrix.aim`` ->
conversion composition that replaced the old ``vector.to_*`` constructors.
"""

from maya import cmds
from rig import (
    angle,
    angle_degrees,
    dist,
    elerp,
    inverse,
    lerp,
    matrix,
    Node,
    normalize,
    Plug,
    PlugList,
    set_options,
    slerp,
    to_euler,
    to_matrix,
    to_quaternion,
)
from rig._internal.maya_version import is_at_least
from rig._tests._base import MayaTestCase


def _quat(node):
    """Return a quaternion-typed Plug (4 channels) from a transform's matrix."""
    return to_quaternion(node.matrix)


def _unknown_plug(name):
    """Return a plug whose ``math_type`` is ``"unknown"`` (a string attr)."""
    from rig.spec import String

    n = Node.create("transform", name=name)
    n << String("label")
    return n.label


# --------------------------------------------------------------------- #
#  Scalar short-circuit (pure-Python, no node built)
# --------------------------------------------------------------------- #


class TestDispatchScalar(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lerp_scalar(self):
        self.assertAlmostEqual(lerp(2.0, 10.0, weight=0.5), 6.0)

    def test_elerp_scalar(self):
        # elerp(2, 8, 0.5) = 2^0.5 * 8^0.5 = sqrt(16) = 4.
        self.assertAlmostEqual(elerp(2.0, 8.0, weight=0.5), 4.0)


# --------------------------------------------------------------------- #
#  Routing table -- each (verb, type) reaches the expected per-type impl.
#  Publishing is disabled so the result is the raw node (not a container
#  alias); the matrix / quaternion plugins are loaded for headless Maya.
# --------------------------------------------------------------------- #


class TestDispatchRouting(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)
        cmds.loadPlugin("matrixNodes", quiet=True)
        cmds.loadPlugin("quatNodes", quiet=True)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def _node_type(self, result):
        return cmds.nodeType(str(result).split(".")[0])

    # -- dist -- #

    def test_dist_vector(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(self._node_type(dist(a.t, b.t)), "distanceBetween")

    def test_dist_matrix(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(self._node_type(dist(a.wm, b.wm)), "distanceBetween")

    # -- lerp -- #

    def test_lerp_vector_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(lerp(a.t, b.t, weight=0.5), Plug)

    def test_lerp_matrix(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(self._node_type(lerp(a.matrix, b.matrix, 0.5)), "wtAddMatrix")

    # -- slerp -- #

    def test_slerp_vector_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(slerp(a.t, b.t, weight=0.5), Plug)

    def test_slerp_quaternion(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(self._node_type(slerp(_quat(a), _quat(b), 0.5)), "quatSlerp")

    def test_slerp_euler(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(self._node_type(slerp(a.r, b.r, 0.5)), "quatToEuler")

    def test_slerp_matrix(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertEqual(
            self._node_type(slerp(a.matrix, b.matrix, 0.5)), "composeMatrix"
        )

    # -- elerp -- #

    def test_elerp_vector_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(elerp(a.t, b.t, weight=0.5), Plug)

    # -- angle / angle_degrees -- #

    def test_angle_vector_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(angle(a.t, b.t), Plug)

    def test_angle_quaternion_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(angle(_quat(a), _quat(b)), Plug)

    def test_angle_degrees_vector_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(angle_degrees(a.t, b.t), Plug)

    def test_angle_degrees_quaternion_returns_plug(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        self.assertIsInstance(angle_degrees(_quat(a), _quat(b)), Plug)

    # -- normalize -- #

    def test_normalize_vector(self):
        a = Node.create("transform", name="a")
        self.assertIn(self._node_type(normalize(a.t)), {"normalize", "multiplyDivide"})

    def test_normalize_quaternion(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(normalize(_quat(a))), "quatNormalize")

    def test_normalize_matrix(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(normalize(a.matrix)), "composeMatrix")

    # -- inverse -- #

    def test_inverse_matrix(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(inverse(a.matrix)), "inverseMatrix")

    def test_inverse_quaternion(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(inverse(_quat(a))), "quatInvert")

    # -- conversions -- #

    def test_to_euler_quaternion(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(to_euler(_quat(a))), "quatToEuler")

    def test_to_euler_matrix(self):
        a = Node.create("transform", name="a")
        self.assertTrue(str(to_euler(a.matrix)).endswith(".outputRotate"))

    def test_to_quaternion_euler(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(to_quaternion(a.r)), "eulerToQuat")

    def test_to_quaternion_matrix(self):
        a = Node.create("transform", name="a")
        self.assertTrue(str(to_quaternion(a.matrix)).endswith(".outputQuat"))

    def test_to_matrix_euler(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(to_matrix(a.r)), "composeMatrix")

    def test_to_matrix_quaternion(self):
        a = Node.create("transform", name="a")
        self.assertEqual(self._node_type(to_matrix(_quat(a))), "composeMatrix")


# --------------------------------------------------------------------- #
#  matrix.aim -> conversion composition (replaces the old vector.to_*).
# --------------------------------------------------------------------- #


class TestDispatchAimConversions(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)
        cmds.loadPlugin("matrixNodes", quiet=True)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_to_euler_of_aim(self):
        # The old ``vector.to_euler(aim, up)`` is now ``to_euler(matrix.aim(...))``.
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = to_euler(matrix.aim(a.t, b.t))
        self.assertTrue(str(result).endswith(".outputRotate"))

    def test_to_quaternion_of_aim(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = to_quaternion(matrix.aim(a.t, b.t))
        self.assertTrue(str(result).endswith(".outputQuat"))


# --------------------------------------------------------------------- #
#  PlugList vectorisation -- a list operand classifies by element 0 and the
#  @vectorize-d impl broadcasts the whole list.
# --------------------------------------------------------------------- #


class TestDispatchVectorize(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lerp_pluglist_broadcasts(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = lerp(PlugList([a.t, b.t]), PlugList([b.t, a.t]), weight=0.5)
        self.assertIsInstance(result, PlugList)
        self.assertEqual(len(result), 2)
        for plug in result:
            self.assertIsInstance(plug, Plug)


# --------------------------------------------------------------------- #
#  Rejected (verb, type) pairs raise TypeError -- no invented math.
# --------------------------------------------------------------------- #


class TestDispatchRaises(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)
        cmds.loadPlugin("quatNodes", quiet=True)

    def test_lerp_rejects_quaternion_with_slerp_hint(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError) as cm:
            lerp(_quat(a), _quat(b), 0.5)
        self.assertIn("slerp", str(cm.exception))

    def test_lerp_rejects_euler(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError):
            lerp(a.r, b.r, 0.5)

    def test_elerp_rejects_quaternion(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError):
            elerp(_quat(a), _quat(b), 0.5)

    def test_elerp_rejects_euler(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError):
            elerp(a.r, b.r, 0.5)

    def test_to_matrix_rejects_vector_with_aim_hint(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError) as cm:
            to_matrix(a.t)
        self.assertIn("matrix.aim", str(cm.exception))

    def test_to_euler_rejects_vector(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError):
            to_euler(a.t)

    def test_to_quaternion_rejects_vector(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError):
            to_quaternion(a.t)

    def test_inverse_rejects_scalar(self):
        with self.assertRaises(TypeError):
            inverse(5.0)

    def test_normalize_rejects_scalar(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError):
            normalize(a.tx)

    def test_dist_rejects_scalar(self):
        with self.assertRaises(TypeError):
            dist(5.0, 6.0)

    def test_slerp_rejects_scalar(self):
        with self.assertRaises(TypeError):
            slerp(5.0, 6.0, 0.5)

    def test_angle_rejects_matrix(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError):
            angle(a.matrix, b.matrix)

    def test_angle_degrees_rejects_matrix(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        with self.assertRaises(TypeError):
            angle_degrees(a.matrix, b.matrix)

    def test_unknown_operand_rejected_by_every_verb(self):
        # An ``unknown``-typed operand exercises each verb's final
        # ``_unsupported`` fall-through (the no-hint TypeError path).
        unk = _unknown_plug("unk")
        for verb in (dist, lerp, slerp, elerp, angle, angle_degrees):
            with self.assertRaises(TypeError):
                verb(unk, unk)
        for verb in (normalize, inverse, to_euler, to_quaternion, to_matrix):
            with self.assertRaises(TypeError):
                verb(unk)

    def test_empty_pluglist_classifies_unknown(self):
        # An empty PlugList has no element-0 to peek, so ``_classify`` yields
        # ``"unknown"`` and the verb rejects it.
        with self.assertRaises(TypeError):
            lerp(PlugList([]), PlugList([]), 0.5)


# --------------------------------------------------------------------- #
#  Numeric correctness on FLAT LITERAL vector inputs (container publishing
#  ON -- the default). Regression: a raw ``[x, y, z]`` was mis-typed as a
#  scalar by ``_infer_attr_type``, so ``publish_input`` crashed and
#  ``slerp([0,0,1],[1,0,0],0.5)`` raised instead of returning the spherical
#  midpoint. These exercise the publish path the routing tests bypass
#  (they set ``publish_attributes=False``).
# --------------------------------------------------------------------- #


class TestDispatchNumericLiterals(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Defaults (publish path active); load the plugins slerp/lerp need.
        set_options(flatten_containers=True, publish_attributes=True)
        cmds.loadPlugin("quatNodes", quiet=True)
        cmds.loadPlugin("matrixNodes", quiet=True)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True, publish_attributes=True)

    def test_slerp_literal_vectors_is_spherical(self):
        # slerp halfway between the orthonormal (0,0,1) and (1,0,0) lies on
        # the unit arc: (sqrt(2)/2, 0, sqrt(2)/2) -- NOT the linear midpoint
        # (0.5, 0, 0.5).
        result = slerp([0, 0, 1], [1, 0, 0], 0.5)
        value  = cmds.getAttr(str(result))[0]
        h      = 2**-0.5  # 0.70710678...
        for actual, expected in zip(value, (h, 0.0, h)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_lerp_literal_vectors_is_linear(self):
        # lerp halfway IS the plain midpoint (0.5, 0, 0.5).
        result = lerp([0, 0, 1], [1, 0, 0], 0.5)
        value  = cmds.getAttr(str(result))[0]
        for actual, expected in zip(value, (0.5, 0.0, 0.5)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_lerp_scalar_plug_uses_native_node(self):
        # Scalar Plug inputs (not compound, not a sequence) take the scalar
        # fast path -- the True branch of the ``inputs_are_scalar`` guard that
        # the literal-vector fix added. The value (midpoint) holds on every
        # Maya version; the native ``lerp`` node is only built on 2024+
        # (older Maya uses blendWeighted), so gate that assertion on version.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.tx << 10.0
        b.tx << 30.0
        result = lerp(a.tx, b.tx, 0.5)
        self.assertAlmostEqual(cmds.getAttr(str(result)), 20.0)
        if is_at_least(2024):
            self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "lerp")


# --------------------------------------------------------------------- #
#  ``blend`` top-level verb -- full-transform interpolation for matrices.
#  Pins the contract that distinguishes it from ``slerp`` (which is now
#  orientation-only for matrices) and that non-matrix operands raise.
# --------------------------------------------------------------------- #


class TestBlendDispatch(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_top_level_blend_full_transform(self):
        from rig import blend

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = blend(a.worldMatrix[0], b.worldMatrix[0], 1.0)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 10.0, places=3
        )

    def test_top_level_slerp_matrix_is_orientation_only(self):
        from rig import slerp

        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = slerp(a.worldMatrix[0], b.worldMatrix[0], 1.0)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 0.0, places=3
        )

    def test_blend_quaternion_unsupported(self):
        from rig import blend

        loc = cmds.spaceLocator()[0]
        from rig import matrix as _m

        q = _m.decompose(Node(loc).worldMatrix[0]).outputQuat
        with self.assertRaises(TypeError):  # _unsupported raises TypeError
            blend(q, q, 0.5)