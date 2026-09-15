"""Smoke tests for ``rig.quaternion``."""

from maya import cmds
from rig import (
    angle,
    angle_degrees,
    inverse,
    Node,
    normalize,
    Plug,
    quaternion as q,
    set_options,
    slerp,
    to_euler,
    to_matrix,
    to_quaternion,
)
from rig._tests._base import MayaTestCase


def _quat_plug(node):
    """Helper: add a Quat attribute and return the Plug."""
    from rig.spec import Quat

    node << Quat("rotQuat")
    return node.rotQuat


class TestQuaternionArithmetic(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_add_creates_quataadd(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = q.add(_quat_plug(a), _quat_plug(b))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatAdd")

    def test_multiply_creates_quatprod(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = q.multiply(_quat_plug(a), _quat_plug(b))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatProd")

    def test_subtract_creates_quatsub(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = q.subtract(_quat_plug(a), _quat_plug(b))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatSub")


class TestQuaternionUnary(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_negate(self):
        a      = Node.create("transform", name="a")
        result = q.negate(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatNegate")

    def test_normalize(self):
        a      = Node.create("transform", name="a")
        result = normalize(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatNormalize")

    def test_inverse(self):
        a      = Node.create("transform", name="a")
        result = inverse(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatInvert")

    def test_conjugate(self):
        a      = Node.create("transform", name="a")
        result = q.conjugate(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatConjugate")


class TestQuaternionConversions(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Internal-node-creation tests -- disable publishing so result is
        # the underlying node's plug, not the published container output.
        # Published-interface contract is covered by ``TestQuaternionPublishedInterface``.
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_to_euler(self):
        a      = Node.create("transform", name="a")
        result = to_euler(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatToEuler")

    def test_to_matrix(self):
        a      = Node.create("transform", name="a")
        result = to_matrix(_quat_plug(a))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "composeMatrix")

    def test_to_vector_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = q.to_vector(_quat_plug(a))
        self.assertIsInstance(result, Plug)


class TestQuaternionGeometry(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_angle_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = angle(_quat_plug(a), _quat_plug(b))
        self.assertIsInstance(result, Plug)


class TestQuaternionInterpolate(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_slerp_creates_quatslerp(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = slerp(_quat_plug(a), _quat_plug(b), weight=0.5)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "quatSlerp")


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, compare against API 2.0 truth.
# --------------------------------------------------------------------- #


import math

from maya.api import OpenMaya as om
from rig.maya.attribute import Attribute


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


def _make_quat_node(name, euler_xyz=(0, 0, 0)):
    """Build an eulerToQuat with known input -> returns the outputQuat
    Attribute and the API-2.0 ground-truth MQuaternion."""
    cmds.loadPlugin("quatNodes", quiet=True)
    n = cmds.createNode("eulerToQuat", name=name, skipSelect=True)
    cmds.setAttr(f"{n}.inputRotateX", euler_xyz[0])
    cmds.setAttr(f"{n}.inputRotateY", euler_xyz[1])
    cmds.setAttr(f"{n}.inputRotateZ", euler_xyz[2])
    eul = om.MEulerRotation(
        math.radians(euler_xyz[0]),
        math.radians(euler_xyz[1]),
        math.radians(euler_xyz[2]),
    )
    return Attribute(f"{n}.outputQuat"), eul.asQuaternion()


class TestMultiplyValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_multiply_by_identity_returns_same(self):
        q1, q1_truth = _make_quat_node("q1", (45, 0, 0))
        q2, q2_truth = _make_quat_node("q2", (0, 0, 0))
        result = q.multiply(q1, q2)
        v      = _eval(result)
        truth  = q1_truth * q2_truth
        for actual, expected in zip(v, [truth.x, truth.y, truth.z, truth.w]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_multiply_known_matches_api_truth(self):
        q1, q1_truth = _make_quat_node("q1", (90, 0, 0))
        q2, q2_truth = _make_quat_node("q2", (0, 90, 0))
        result = q.multiply(q1, q2)
        v      = _eval(result)
        truth  = q1_truth * q2_truth
        for actual, expected in zip(v, [truth.x, truth.y, truth.z, truth.w]):
            self.assertAlmostEqual(actual, expected, places=3)


class TestNormalizeValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_normalize_2_0_0_0(self):
        n = cmds.createNode("network", name="q_input", skipSelect=True)
        cmds.addAttr(n, ln="inputQuat", at="compound", numberOfChildren=4)
        for c in "XYZW":
            cmds.addAttr(n, ln=f"inputQuat{c}", at="double", parent="inputQuat")
        cmds.setAttr(f"{n}.inputQuatX", 2.0)

        result = normalize(Attribute(f"{n}.inputQuat"))
        v      = _eval(result)
        for actual, expected in zip(v, [1, 0, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=3)


class TestConjugateValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_conjugate_flips_imaginary_parts(self):
        q1, _ = _make_quat_node("q1", (45, 30, 60))
        result = q.conjugate(q1)
        v      = _eval(result)
        truth  = _eval(q1)
        # Conjugate flips x, y, z and keeps w.
        self.assertAlmostEqual(v[0], -truth[0], places=3)
        self.assertAlmostEqual(v[1], -truth[1], places=3)
        self.assertAlmostEqual(v[2], -truth[2], places=3)
        self.assertAlmostEqual(v[3], truth[3],  places=3)


class TestAngleValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_angle_returns_radians(self):
        # Identity vs 90deg about X -> shortest-arc angle is pi/2 radians.
        q1, _ = _make_quat_node("q1", (0, 0, 0))
        q2, _ = _make_quat_node("q2", (90, 0, 0))
        result = angle(q1, q2)
        self.assertAlmostEqual(_eval(result), math.pi / 2, places=3)

    def test_angle_degrees_returns_degrees(self):
        # Same pair, degrees variant -> 90.
        q1, _ = _make_quat_node("q1", (0, 0, 0))
        q2, _ = _make_quat_node("q2", (90, 0, 0))
        result = angle_degrees(q1, q2)
        self.assertAlmostEqual(_eval(result), 90.0, places=2)


# --------------------------------------------------------------------- #
#  v3.A -- axis-angle conversions
# --------------------------------------------------------------------- #


class TestFromAxisAngleValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_x_90deg(self):
        # 90deg about X-axis: q = (sin(45deg), 0, 0, cos(45deg)).
        # Drive inputAngle directly via setAttr (bypass linear-typed tx)
        # to avoid unitConversion side-effects between linear and angle.
        ax = Node.create("transform", name="ax")
        ax.t << (1, 0, 0)

        # Build the node directly so we can setAttr the doubleAngle input.
        # ``cmds.setAttr`` on a doubleAngle attr interprets the value in the
        # current angle unit (degrees by default), so we pass degrees here.
        cmds.loadPlugin("quatNodes", quiet=True)
        n = cmds.createNode("axisAngleToQuat", name="axang1", skipSelect=True)
        cmds.connectAttr("ax.t", f"{n}.inputAxis")
        cmds.setAttr(f"{n}.inputAngle", 90.0)  # 90deg = pi/2 rad

        v        = _eval(Attribute(f"{n}.outputQuat"))
        expected = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
        for actual, exp in zip(v, expected):
            self.assertAlmostEqual(actual, exp, places=3)


class TestFromAxisAngleAPI(MayaTestCase):
    """Exercise the rig DSL API surface (returns Plug, builds correct node)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_returns_quat_plug(self):
        ax = Node.create("transform", name="ax")
        ax.t << (1, 0, 0)
        ang = Node.create("transform", name="ang")
        ang.tx << 0.5
        result = q.from_axis_angle(ax.t, ang.tx)
        self.assertTrue(str(result).endswith(".outputQuat"))
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "axisAngleToQuat")


class TestToAxisAngleValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_roundtrip_x_90deg(self):
        # Build q from (axis=X, angle=pi/2 rad), then decompose back.
        # Angle is set via setAttr to avoid unit-conversion noise.
        cmds.loadPlugin("quatNodes", quiet=True)

        ax = Node.create("transform", name="ax")
        ax.t << (1, 0, 0)
        n_q = cmds.createNode("axisAngleToQuat", name="axang1", skipSelect=True)
        cmds.connectAttr("ax.t", f"{n_q}.inputAxis")
        cmds.setAttr(f"{n_q}.inputAngle", 90.0)  # 90deg = pi/2 rad

        # Decompose back via to_axis_angle.
        axis_out, angle_out = q.to_axis_angle(Attribute(f"{n_q}.outputQuat"))
        # Probe angle directly (it's doubleAngle; getAttr returns in current
        # angle unit = degrees by default).
        angle_val_degrees = cmds.getAttr(str(angle_out))
        self.assertAlmostEqual(angle_val_degrees, 90.0, places=2)
        # Probe axis via a transform's t (compound vec3, no unit issue).
        out = Node.create("transform", name="out")
        out.t << axis_out
        for actual, expected in zip(_eval(out.t), [1, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=3)


# --------------------------------------------------------------------- #
#  v4.K -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


class TestQuaternionPublishedInterface(MayaTestCase):
    """Verify each wrapped composite quaternion function creates a
    container with the expected published attributes when
    ``flatten_containers=False``."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def _assert_attrs(self, container_name, attrs):
        ctn = cmds.ls(f"{container_name}*", type="container") or []
        self.assertTrue(
            any(c.startswith(container_name) for c in ctn),
            f"expected {container_name} container, got {ctn}",
        )
        target = next(c for c in ctn if c.startswith(container_name))
        from rig._internal.container import _resolve_published

        for attr in attrs:
            # Native publish: single attrs are real bindAttr aliases that
            # ``attributeQuery`` sees; MULTI attrs are registry-only (they live
            # on the host node), so resolve them through the container instead.
            published = cmds.attributeQuery(attr, node=target, exists=True) or (
                _resolve_published(target, attr) is not None
            )
            self.assertTrue(published, f"{target}.{attr} should be published")

    def test_angle_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        # angle(q1, q2) -- needs quat plugs; use orientation attrs.
        # Build quats via matrix.to_quaternion.
        q1 = to_quaternion(a.matrix)
        q2 = to_quaternion(b.matrix)
        angle(q1, q2)
        self._assert_attrs("quat_angle1", ["input1", "input2", "output"])

    def test_angle_degrees_publishes(self):
        a  = Node.create("transform", name="a")
        b  = Node.create("transform", name="b")
        q1 = to_quaternion(a.matrix)
        q2 = to_quaternion(b.matrix)
        angle_degrees(q1, q2)
        self._assert_attrs("quat_angle_degrees1", ["input1", "input2", "output"])

    def test_to_vector_publishes(self):
        a  = Node.create("transform", name="a")
        q1 = to_quaternion(a.matrix)
        q.to_vector(q1)
        self._assert_attrs("quat_to_vector1", ["input", "output"])

    def test_slerp_publishes(self):
        a  = Node.create("transform", name="a")
        b  = Node.create("transform", name="b")
        q1 = to_quaternion(a.matrix)
        q2 = to_quaternion(b.matrix)
        slerp(q1, q2, weight=0.5)
        self._assert_attrs("quat_slerp1", ["input1", "input2", "weight", "output"])

    def test_from_axis_angle_publishes(self):
        a = Node.create("transform", name="a")
        a.t << [1, 0, 0]
        q.from_axis_angle(a.t, a.tx)
        self._assert_attrs("from_axis_angle1", ["axis", "angle", "output"])

    def test_to_axis_angle_publishes(self):
        a  = Node.create("transform", name="a")
        q1 = to_quaternion(a.matrix)
        axis_out, angle_out = q.to_axis_angle(q1)
        # Both outputs come from the same container.
        self._assert_attrs("to_axis_angle1", ["input", "axis", "angle"])
        # Native publish returns the REAL bound inner plug (the published name
        # is an alias). The returned plugs must be exactly the plugs published
        # as ``axis`` / ``angle`` on the container.
        from rig._internal.container import _resolve_published

        self.assertEqual(
            str(_resolve_published("to_axis_angle1", "axis")), str(axis_out)
        )
        self.assertEqual(
            str(_resolve_published("to_axis_angle1", "angle")), str(angle_out)
        )


# --------------------------------------------------------------------- #
#  Public API surface (__all__)
# --------------------------------------------------------------------- #


class TestQuaternionAll(MayaTestCase):
    """``quaternion``'s public surface is the Hamilton arithmetic and the
    axis-angle / vector conversions PLUS the cross-type ops, now exposed as
    public per-type functions (also reachable through the top-level dispatch
    verbs)."""

    def test_public_surface(self):
        self.assertEqual(
            set(q.__all__),
            {
                "add",
                "multiply",
                "subtract",
                "negate",
                "normalize",
                "inverse",
                "conjugate",
                "angle",
                "angle_degrees",
                "to_euler",
                "to_matrix",
                "to_vector",
                "slerp",
                "pow",
                "from_axis_angle",
                "to_axis_angle",
            },
        )
        for name in q.__all__:
            self.assertTrue(hasattr(q, name), f"quaternion.{name} missing")

    def test_cross_type_ops_public(self):
        # The cross-type ops are now PUBLIC per-type funcs on quaternion
        # (callable directly), in addition to the top-level dispatch verbs.
        # The old ``_``-prefixed impl names are gone.
        for name in (
            "normalize",
            "inverse",
            "angle",
            "angle_degrees",
            "to_euler",
            "to_matrix",
            "slerp",
        ):
            self.assertIn(name, q.__all__)
            self.assertTrue(hasattr(q, name), f"quaternion.{name} should be public")
            self.assertFalse(
                hasattr(q, f"_{name}"),
                f"quaternion._{name} private alias should be gone",
            )


# --------------------------------------------------------------------- #
#  pow -- fractional power of a unit quaternion (sugar over slerp from identity)
# --------------------------------------------------------------------- #


from rig import matrix as m


class TestQuaternionPow(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)  # decomposeMatrix

    def test_pow_half_is_half_rotation(self):
        loc = cmds.spaceLocator()[0]
        cmds.setAttr(loc + ".rotateY", 90)
        quat = m.decompose(Node(loc).worldMatrix[0]).outputQuat  # 90deg about Y
        half = q.pow(quat, 0.5)
        # 45deg about Y => [0, sin(22.5), 0, cos(22.5)]
        self.assertAlmostEqual(
            cmds.getAttr(f"{half}Y"), math.sin(math.radians(22.5)), places=3
        )

    def test_pow_zero_is_identity(self):
        loc = cmds.spaceLocator()[0]
        cmds.setAttr(loc + ".rotateY", 90)
        quat = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        out  = q.pow(quat, 0.0)
        got  = [round(cmds.getAttr(f"{out}{a}"), 4) for a in ("X", "Y", "Z", "W")]
        self.assertEqual(got, [0.0, 0.0, 0.0, 1.0])


class TestQuaternionPowEdge(MayaTestCase):
    """``quaternion.pow`` extrapolates past the [0, 1] weight range -- no
    clamp on the underlying slerp-from-identity. ``q ** 1.5`` of a 90deg
    rotation yields a 135deg rotation."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)  # decomposeMatrix

    def test_pow_extrapolates_past_one(self):
        # q is 90deg about Y; q ** 1.5 -> 135deg => [0, sin(67.5deg), 0, cos(67.5deg)]
        loc = cmds.spaceLocator()[0]
        cmds.setAttr(loc + ".rotateY", 90)
        quat = m.decompose(Node(loc).worldMatrix[0]).outputQuat
        out  = q.pow(quat, 1.5)
        self.assertAlmostEqual(
            cmds.getAttr(f"{out}Y"), math.sin(math.radians(67.5)), places=3
        )