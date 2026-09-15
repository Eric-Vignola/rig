"""Smoke tests for ``rig.trigonometry``.

One test per public function: verify the correct Maya node type is
built and at least one input is wired. Pure-Python scalar branches
are checked numerically against ``math.*``.
"""

import math
from unittest import mock

from maya import cmds
from rig import Node, Plug, trigonometry as trig
from rig._tests._base import MayaTestCase


class TestConversions(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_degrees_scalar(self):
        self.assertAlmostEqual(trig.degrees(math.pi), 180.0)

    def test_radians_scalar(self):
        self.assertAlmostEqual(trig.radians(180.0), math.pi)

    def test_degrees_plug_creates_multiplydivide(self):
        a      = Node.create("transform", name="a")
        result = trig.degrees(a.tx)
        # Maya 2024+ may use native ``multiply`` node instead of ``multiplyDivide``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "multiply"},
        )

    def test_radians_plug_creates_multiplydivide(self):
        a      = Node.create("transform", name="a")
        result = trig.radians(a.tx)
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "multiply"},
        )


class TestSinCos(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sin_scalar(self):
        self.assertAlmostEqual(trig.sin(math.pi / 2), 1.0)

    def test_sind_scalar(self):
        self.assertAlmostEqual(trig.sind(90.0), 1.0)

    def test_cos_scalar(self):
        self.assertAlmostEqual(trig.cos(0.0), 1.0)

    def test_cosd_scalar(self):
        self.assertAlmostEqual(trig.cosd(0.0), 1.0)

    def test_sin_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.sin(a.tx)
        self.assertIsInstance(result, Plug)

    def test_cos_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.cos(a.tx)
        self.assertIsInstance(result, Plug)

    def test_sind_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.sind(a.tx)
        self.assertIsInstance(result, Plug)

    def test_cosd_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.cosd(a.tx)
        self.assertIsInstance(result, Plug)


class TestTan(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_tan_scalar(self):
        self.assertAlmostEqual(trig.tan(0.0), 0.0)

    def test_tand_scalar(self):
        self.assertAlmostEqual(trig.tand(0.0), 0.0)

    def test_tan_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.tan(a.tx)
        self.assertIsInstance(result, Plug)

    def test_tand_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.tand(a.tx)
        self.assertIsInstance(result, Plug)

    @mock.patch("rig._internal.node_ops.get_target_version", return_value=2022)
    def test_tand_legacy_fallback_returns_plug(self, _ver):
        # Pre-2024 has no native ``tan`` node, so ``_tand_op`` raises
        # RuntimeError and ``tand`` composes ``sind / cosd`` with
        # div-by-zero quieting -- ``condition(c == 0, inf(), div)``.
        # Forcing the target version below 2024 exercises that fallback.
        a      = Node.create("transform", name="a")
        result = trig.tand(a.tx)
        self.assertIsInstance(result, Plug)


class TestArcSin(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_asin_scalar(self):
        self.assertAlmostEqual(trig.asin(1.0), math.pi / 2)

    def test_asind_scalar(self):
        self.assertAlmostEqual(trig.asind(1.0), 90.0)

    def test_asin_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.asin(a.tx)
        self.assertIsInstance(result, Plug)

    def test_asind_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.asind(a.tx)
        self.assertIsInstance(result, Plug)


class TestArcCos(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_acos_scalar(self):
        self.assertAlmostEqual(trig.acos(1.0), 0.0)

    def test_acosd_scalar(self):
        self.assertAlmostEqual(trig.acosd(1.0), 0.0)

    def test_acos_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.acos(a.tx)
        self.assertIsInstance(result, Plug)

    def test_acosd_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.acosd(a.tx)
        self.assertIsInstance(result, Plug)


class TestArcTan(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_atan_scalar(self):
        self.assertAlmostEqual(trig.atan(1.0), math.pi / 4)

    def test_atand_scalar(self):
        self.assertAlmostEqual(trig.atand(1.0), 45.0)

    def test_atan_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.atan(a.tx)
        self.assertIsInstance(result, Plug)

    def test_atand_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = trig.atand(a.tx)
        self.assertIsInstance(result, Plug)


class TestArcTan2(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_atan2_scalar(self):
        self.assertAlmostEqual(trig.atan2(1.0, 1.0), math.pi / 4)

    def test_atan2d_scalar(self):
        self.assertAlmostEqual(trig.atan2d(1.0, 1.0), 45.0)

    def test_atan2_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = trig.atan2(a.tx, b.tx)
        self.assertIsInstance(result, Plug)

    def test_atan2d_plug_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = trig.atan2d(a.tx, b.tx)
        self.assertIsInstance(result, Plug)


class TestRoundTrip(MayaTestCase):
    """Round-trip identities for scalar inputs."""

    TEST_START_NEW_SCENE = True

    def test_degrees_radians_roundtrip(self):
        self.assertAlmostEqual(trig.degrees(trig.radians(45.0)), 45.0)

    def test_sin_asin_roundtrip(self):
        self.assertAlmostEqual(trig.asin(trig.sin(0.5)), 0.5)

    def test_cos_acos_roundtrip(self):
        self.assertAlmostEqual(trig.acos(trig.cos(0.5)), 0.5)

    def test_atan2_quadrant_signs(self):
        # All four quadrants
        self.assertAlmostEqual(trig.atan2(1.0, 1.0),   math.pi / 4)
        self.assertAlmostEqual(trig.atan2(1.0, -1.0),  3 * math.pi / 4)
        self.assertAlmostEqual(trig.atan2(-1.0, -1.0), -3 * math.pi / 4)
        self.assertAlmostEqual(trig.atan2(-1.0, 1.0),  -math.pi / 4)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, set inputs, getAttr, compare
#  against ground-truth math.* computation.
# --------------------------------------------------------------------- #


def _eval(plug):
    """Read a plug's current value via cmds.getAttr."""
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestSinValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sin_pi_2(self):
        n = Node.create("transform", name="probe")
        n.tx << math.pi / 2
        n.ty << trig.sin(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=3)


class TestCosValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_cos_0(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.0
        n.ty << trig.cos(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=3)

    def test_cos_pi(self):
        n = Node.create("transform", name="probe")
        n.tx << math.pi
        n.ty << trig.cos(n.tx)
        self.assertAlmostEqual(_eval(n.ty), -1.0, places=3)


class TestTanValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_tan_pi_4(self):
        n = Node.create("transform", name="probe")
        n.tx << math.pi / 4
        n.ty << trig.tan(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=3)


class TestSindValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sind_90(self):
        n = Node.create("transform", name="probe")
        n.tx << 90.0
        n.ty << trig.sind(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=3)


class TestCosdValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_cosd_180(self):
        n = Node.create("transform", name="probe")
        n.tx << 180.0
        n.ty << trig.cosd(n.tx)
        self.assertAlmostEqual(_eval(n.ty), -1.0, places=3)


class TestAsinValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_asin_1(self):
        n = Node.create("transform", name="probe")
        n.tx << 1.0
        n.ty << trig.asin(n.tx)
        self.assertAlmostEqual(_eval(n.ty), math.pi / 2, places=3)


class TestAcosValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_acos_0(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.0
        n.ty << trig.acos(n.tx)
        self.assertAlmostEqual(_eval(n.ty), math.pi / 2, places=3)


class TestAtanValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_atan_1(self):
        n = Node.create("transform", name="probe")
        n.tx << 1.0
        n.ty << trig.atan(n.tx)
        self.assertAlmostEqual(_eval(n.ty), math.pi / 4, places=3)


class TestAtan2Value(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_atan2_1_0(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0  # y
        b = Node.create("transform", name="b")
        b.tx << 0.0  # x
        out = Node.create("transform", name="out")
        out.ty << trig.atan2(a.tx, b.tx)
        self.assertAlmostEqual(_eval(out.ty), math.pi / 2, places=3)


# --------------------------------------------------------------------- #
#  v4.N -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


from rig import set_options


class TestTrigonometryPublishedInterface(MayaTestCase):
    """Verify each wrapped trig function creates a container with the
    expected published attributes when ``flatten_containers=False``.

    Tests the radians-input wraps (sin/cos/tan/asin/acos/atan) and
    the atan2/atan2_1 family. Pre-2024 fallback paths only fire on
    older Maya so are exercised via the modern path here -- confirms
    the wrap is structurally correct.
    """

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
        for attr in attrs:
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )

    def test_sin_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        trig.sin(a.tx)
        self._assert_attrs("sin1", ["input", "output"])

    def test_cos_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        trig.cos(a.tx)
        self._assert_attrs("cos1", ["input", "output"])

    def test_tan_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 0.5
        trig.tan(a.tx)
        self._assert_attrs("tan1", ["input", "output"])

    def test_asin_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 0.5
        trig.asin(a.tx)
        self._assert_attrs("asin1", ["input", "output"])

    def test_acos_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 0.5
        trig.acos(a.tx)
        self._assert_attrs("acos1", ["input", "output"])

    def test_atan_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        trig.atan(a.tx)
        self._assert_attrs("atan1", ["input", "output"])

    def test_atan2_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.tx << 1.0
        b.tx << 1.0
        trig.atan2(a.tx, b.tx)
        self._assert_attrs("atan2_1", ["y", "x", "output"])


class TestTrigAll(MayaTestCase):
    def test_all_names_resolve(self):
        # degrees/radians + sin/cos/tan/asin/acos/atan/atan2 (deg + rad).
        self.assertEqual(len(trig.__all__), 16)
        for name in trig.__all__:
            self.assertTrue(callable(getattr(trig, name)), f"{name} should be callable")


class TestPerspectiveImagePlanesExample(MayaTestCase):
    """Value test for the ``perspective_image_planes`` example's
    ``get_scale``, which composes the degree-variant trig ops ``atand`` +
    ``sind``. Pins correctness after the optimization that replaced
    ``sin(atan(x))`` (a degrees->radians->degrees round-trip) with the
    conversion-free ``sind(atand(x))``."""

    TEST_START_NEW_SCENE = True

    def test_get_scale_matches_math(self):
        from rig.examples.perspective_image_planes import get_scale

        cam = Node.create("transform", name="cam")
        for attr in ("ap", "fl", "ds"):
            cmds.addAttr(str(cam), ln=attr, at="double", keyable=True)
        out = Node.create("transform", name="out")
        out.tx << get_scale(cam.ap, cam.fl, cam.ds)

        cmds.setAttr(str(cam.ap), 1.417)
        cmds.setAttr(str(cam.fl), 50.0)
        cmds.setAttr(str(cam.ds), 100.0)

        # Reference: focal mm->in, aov = atan(ap / (2*fl_in)),
        # scale = sin(aov) * 2 * distance / cos(aov).
        fl_in    = 50.0 * 0.0393700787
        aov      = math.atan(1.417 / (2 * fl_in))
        expected = math.sin(aov) * 2 * 100.0 / ((1 - math.sin(aov) ** 2) ** 0.5)
        self.assertAlmostEqual(cmds.getAttr(str(out.tx)), expected, places=3)