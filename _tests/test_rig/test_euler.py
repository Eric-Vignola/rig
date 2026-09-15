"""Smoke tests for ``rig.euler``."""

from maya import cmds
from rig import euler as e, Node, set_options, slerp, to_matrix, to_quaternion
from rig._tests._base import MayaTestCase


class TestEulerToMatrix(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Internal-node-creation test -- disable publishing so result is
        # the underlying ``composeMatrix`` plug, not the published container output.
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_to_matrix_creates_composematrix(self):
        a      = Node.create("transform", name="a")
        result = to_matrix(a.r, rotate_order=a.ro)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "composeMatrix")


class TestEulerToQuaternion(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_to_quaternion_creates_eulertoquat(self):
        a      = Node.create("transform", name="a")
        result = to_quaternion(a.r, rotate_order=a.ro)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "eulerToQuat")


class TestEulerReorder(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_reorder_with_two_orders_creates_quattoeuler(self):
        a      = Node.create("transform", name="a")
        result = e.reorder(a.r, rotate_order0=0, rotate_order1=3)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatToEuler")

    def test_reorder_missing_orders_raises(self):
        # Both rotate orders are now MANDATORY positional args -- omitting
        # them is a plain Python TypeError (no hand-rolled ValueError guard).
        a = Node.create("transform", name="a")
        with self.assertRaises(TypeError):
            e.reorder(a.r)
        with self.assertRaises(TypeError):
            e.reorder(a.r, rotate_order0=0)


class TestEulerSlerp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Internal-node test -- disable publishing so the result is the
        # underlying quatToEuler plug, not the published container output.
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_slerp_creates_quattoeuler(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = slerp(a.r, b.r, weight=0.5)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatToEuler")


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, compare against API 2.0 truth.
# --------------------------------------------------------------------- #


import math

from maya.api import OpenMaya as om


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestToMatrixValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_to_matrix_zero_is_identity(self):
        n = Node.create("transform", name="n")
        n.r << (0, 0, 0)
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << to_matrix(n.r)
        m = _eval(out.offsetParentMatrix)
        # Identity diagonals.
        self.assertAlmostEqual(m[0],  1.0, places=4)
        self.assertAlmostEqual(m[5],  1.0, places=4)
        self.assertAlmostEqual(m[10], 1.0, places=4)
        # Off-diagonals.
        self.assertAlmostEqual(m[1], 0.0, places=4)
        self.assertAlmostEqual(m[2], 0.0, places=4)
        self.assertAlmostEqual(m[4], 0.0, places=4)

    def test_to_matrix_known_matches_api_truth(self):
        n = Node.create("transform", name="n")
        n.r << (45, 30, 60)
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << to_matrix(n.r)
        m = _eval(out.offsetParentMatrix)
        truth = om.MEulerRotation(
            math.radians(45), math.radians(30), math.radians(60)
        ).asMatrix()
        truth_flat = [truth[i] for i in range(16)]
        for actual, expected in zip(m, truth_flat):
            self.assertAlmostEqual(actual, expected, places=3)


class TestToQuaternionValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_to_quaternion_zero_is_identity(self):
        n = Node.create("transform", name="n")
        n.r << (0, 0, 0)
        result = to_quaternion(n.r)
        v      = _eval(result)
        for actual, expected in zip(v, [0, 0, 0, 1]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_to_quaternion_90X(self):
        n = Node.create("transform", name="n")
        n.r << (90, 0, 0)
        result = to_quaternion(n.r)
        v      = _eval(result)
        # 90deg about X: q = (sin(45deg), 0, 0, cos(45deg)).
        expected = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
        for actual, exp in zip(v, expected):
            self.assertAlmostEqual(actual, exp, places=3)


class TestEulerSlerpValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_slerp_midpoint_90x(self):
        # Slerp identity -> 90deg about X at weight 0.5 -> 45deg about X.
        a = Node.create("transform", name="a")
        a.r << (0, 0, 0)
        b = Node.create("transform", name="b")
        b.r << (90, 0, 0)
        out = Node.create("transform", name="out")
        out.r << slerp(a.r, b.r, weight=0.5)
        for actual, expected in zip(_eval(out.r), (45, 0, 0)):
            self.assertAlmostEqual(actual, expected, places=3)


class TestEulerPublishedInterface(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def test_slerp_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        slerp(a.r, b.r, weight=0.5)
        ctn = cmds.ls("euler_slerp*", type="container") or []
        self.assertTrue(any(c.startswith("euler_slerp") for c in ctn))
        target = next(c for c in ctn if c.startswith("euler_slerp"))
        for attr in ("input1", "input2", "weight", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )


class TestEulerAll(MayaTestCase):
    def test_public_surface(self):
        # euler's public surface is ``reorder`` plus the now-public cross-type
        # per-type ops (also reachable via the top-level dispatch verbs).
        self.assertEqual(
            set(e.__all__), {"reorder", "to_matrix", "to_quaternion", "slerp"}
        )
        # euler converts FROM euler, so it defines no ``to_euler``.
        self.assertFalse(hasattr(e, "to_euler"))

    def test_cross_type_ops_public(self):
        # to_matrix / to_quaternion / slerp are now PUBLIC per-type funcs on
        # euler (callable directly), in addition to the top-level dispatch
        # verbs (``from rig import to_matrix, slerp``). The old
        # ``_``-prefixed impl names are gone.
        for name in ("to_matrix", "to_quaternion", "slerp"):
            self.assertIn(name, e.__all__)
            self.assertTrue(hasattr(e, name), f"euler.{name} should be public")
            self.assertFalse(
                hasattr(e, f"_{name}"), f"euler._{name} private alias should be gone"
            )