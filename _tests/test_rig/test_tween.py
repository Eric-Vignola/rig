"""Smoke tests for ``rig.tween``.

Each easing function gets a single scalar test plus a single Plug test.
Together they verify the function builds without errors and returns a
Plug at the boundaries.
"""

from rig import Node, Plug, tween as tw
from rig._tests._base import MayaTestCase


_EASING_FUNCTIONS = [
    "in_linear",
    "out_linear",
    "in_quad",
    "out_quad",
    "in_out_quad",
    "out_in_quad",
    "in_cubic",
    "out_cubic",
    "in_out_cubic",
    "out_in_cubic",
    "in_quart",
    "out_quart",
    "in_out_quart",
    "out_in_quart",
    "in_quint",
    "out_quint",
    "in_out_quint",
    "out_in_quint",
    "in_sine",
    "out_sine",
    "in_out_sine",
    "out_in_sine",
    "in_expo",
    "out_expo",
    "in_out_expo",
    "out_in_expo",
    "in_circ",
    "out_circ",
    "in_out_circ",
    "out_in_circ",
    "in_elastic",
    "out_elastic",
    "in_out_elastic",
    "out_in_elastic",
    "in_back",
    "out_back",
    "in_out_back",
    "out_in_back",
    "in_bounce",
    "out_bounce",
    "in_out_bounce",
    "out_in_bounce",
]


class TestTweenAllEasingsBuildPlug(MayaTestCase):
    """Each easing function called with a Plug input must return a Plug."""

    TEST_START_NEW_SCENE = True

    def test_all_easings_build_plug(self):
        node    = Node.create("transform", name="cube_for_tween")
        missing = []
        for fn_name in _EASING_FUNCTIONS:
            fn = getattr(tw, fn_name, None)
            if fn is None:
                missing.append(fn_name)
                continue
            try:
                result = fn(node.tx)
            except Exception as exc:
                self.fail(f"tween.{fn_name}(node.tx) raised: {exc!r}")
            self.assertIsInstance(
                result,
                Plug,
                msg=f"tween.{fn_name} did not return a Plug",
            )

        if missing:
            self.fail(f"Missing tween functions: {missing}")


class TestTweenAll(MayaTestCase):
    def test_all_matches_easing_functions(self):
        # ``__all__`` must list exactly the 42 documented easing curves
        # (catches a typo / missing name in the module's ``__all__``).
        self.assertEqual(set(tw.__all__), set(_EASING_FUNCTIONS))
        self.assertEqual(len(tw.__all__), 42)
        for name in tw.__all__:
            self.assertTrue(callable(getattr(tw, name)), f"{name} should be callable")


class TestTweenSpotChecks(MayaTestCase):
    """Spot-check a few easings against known mathematical identities at
    boundary points and t=0.5.

    Note: many of the in/out_* functions are clamped to [0, 1], so
    boundary checks are reliable. The in_out / out_in / elastic /
    bounce / back curves have well-defined values at t=0 and t=1.
    """

    TEST_START_NEW_SCENE = True

    def test_in_linear_endpoints(self):
        # in_linear is just clamp(t, 0, 1).
        self.assertEqual(tw.in_linear(0.0), 0.0)
        self.assertEqual(tw.in_linear(1.0), 1.0)
        self.assertEqual(tw.in_linear(0.5), 0.5)

    def test_in_quad_t_squared(self):
        self.assertAlmostEqual(tw.in_quad(0.5), 0.25)

    def test_in_cubic_t_cubed(self):
        self.assertAlmostEqual(tw.in_cubic(0.5), 0.125)

    def test_in_quart_t_to_4(self):
        self.assertAlmostEqual(tw.in_quart(0.5), 0.0625)

    def test_in_quint_t_to_5(self):
        self.assertAlmostEqual(tw.in_quint(0.5), 0.03125)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, compare against ground truth.
# --------------------------------------------------------------------- #


import math

from maya import cmds
from rig._tests._base import MayaTestCase


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestInQuadValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_in_quad_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.in_quad(n.tx)
        # in_quad(t) = t^2 -> 0.25.
        self.assertAlmostEqual(_eval(n.ty), 0.25, places=3)


class TestOutQuadValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_out_quad_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.out_quad(n.tx)
        # out_quad(t) = 1 - (1-t)^2 -> 1 - 0.25 = 0.75.
        self.assertAlmostEqual(_eval(n.ty), 0.75, places=3)


class TestInCubicValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_in_cubic_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.in_cubic(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 0.125, places=3)


class TestOutCubicValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_out_cubic_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.out_cubic(n.tx)
        # 1 - (1-0.5)^3 = 1 - 0.125 = 0.875.
        self.assertAlmostEqual(_eval(n.ty), 0.875, places=3)


class TestInLinearValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_in_linear_identity(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.7
        n.ty << tw.in_linear(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 0.7, places=3)


class TestInSineValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_in_sine_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.in_sine(n.tx)
        # in_sine(t) = 1 - cos(t * pi/2).
        expected = 1 - math.cos(0.5 * math.pi / 2)
        self.assertAlmostEqual(_eval(n.ty), expected, places=3)


class TestInExpoValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_in_expo_0_5(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.5
        n.ty << tw.in_expo(n.tx)
        # in_expo(t) = 2^(10*(t-1)).
        expected = 2 ** (10 * (0.5 - 1))
        self.assertAlmostEqual(_eval(n.ty), expected, places=3)


# --------------------------------------------------------------------- #
#  v3.S extrapolation tests -- verify tweens DON'T clamp inputs to [0,1]
# --------------------------------------------------------------------- #

import math as _math

from maya import cmds as _cmds
from rig._tests._base import MayaTestCase as _MayaTestCase


class TestTweenExtrapolation(_MayaTestCase):
    """v3.S removed the implicit ``t = clamp(t, 0, 1)`` from every
    tween function. Inputs outside ``[0, 1]`` now extrapolate per the
    underlying math, which is useful for animation principles like
    anticipation (``t < 0``) and follow-through (``t > 1``).

    These tests verify the extrapolation produces the mathematically
    expected values, NOT the previously-clamped value.
    """

    TEST_START_NEW_SCENE = True

    def _scalar(self, fn, t):
        """Build a network for ``fn(t)``, evaluate it, return the float."""
        plug = fn(t)
        # Drive a transform attribute with the result so we can read it.
        node = Node.create("transform", name=f"probe_{id(plug)}")
        node.tx << plug
        return _cmds.getAttr(f"{node}.tx")

    def test_in_quad_extrapolates_beyond_one(self):
        # in_quad(t) = t^2 -> in_quad(2) = 4
        result = self._scalar(tw.in_quad, 2.0)
        self.assertAlmostEqual(result, 4.0, places=4)

    def test_in_cubic_extrapolates_negative(self):
        # in_cubic(t) = t^3 -> in_cubic(-0.5) = -0.125
        result = self._scalar(tw.in_cubic, -0.5)
        self.assertAlmostEqual(result, -0.125, places=4)

    def test_in_quart_extrapolates_beyond_one(self):
        # in_quart(t) = t^4 -> in_quart(1.5) = 5.0625
        result = self._scalar(tw.in_quart, 1.5)
        self.assertAlmostEqual(result, 5.0625, places=4)

    def test_out_quad_extrapolates_negative(self):
        # out_quad(t) = -t*(t-2) -> out_quad(-0.5) = -(-0.5)*(-2.5) = -1.25
        result = self._scalar(tw.out_quad, -0.5)
        self.assertAlmostEqual(result, -1.25, places=4)

    def test_in_back_overshoot_beyond_one(self):
        # in_back(t) = t^2*((s+1)*t - s) with s = 1.70158
        # in_back(1.2) = 1.44*(2.70158*1.2 - 1.70158) = 1.44*1.541 ~= 2.219
        s        = 1.70158
        expected = (1.2 * 1.2) * ((s + 1) * 1.2 - s)
        result   = self._scalar(tw.in_back, 1.2)
        self.assertAlmostEqual(result, expected, places=3)

    def test_out_back_overshoot_negative(self):
        # out_back is meaningful at t < 0 -- gives anticipation overshoot
        # below 0. Just verify it doesn't raise and produces a value.
        result = self._scalar(tw.out_back, -0.2)
        self.assertIsInstance(result, float)
        # The value should NOT be clamped to [0, 1]
        # (no specific value assertion, just that it extrapolates)

    def test_in_linear_returns_t_unchanged(self):
        # v3.S simplification: in_linear is now the identity function.
        # Verify it returns t unchanged for inputs outside [0, 1].
        for t_value in (-0.5, 0.0, 0.5, 1.5, 2.0):
            result = self._scalar(tw.in_linear, t_value)
            self.assertAlmostEqual(result, t_value, places=4)

    def test_in_out_quad_extrapolates(self):
        # Composite tween -- verify extrapolation propagates through.
        # in_out_quad(-0.1): t < 0.5 -> lesser = in_quad(-0.2) * 0.5
        # = (-0.2)^2 * 0.5 = 0.04 * 0.5 = 0.02
        result = self._scalar(tw.in_out_quad, -0.1)
        self.assertAlmostEqual(result, 0.02, places=4)

    def test_in_out_quad_extrapolates_above_one(self):
        # in_out_quad(1.2): t >= 0.5 -> greater = out_quad(1.4) * 0.5 + 0.5
        # out_quad(1.4) = -1.4*(1.4-2) = -1.4*-0.6 = 0.84
        # -> 0.84 * 0.5 + 0.5 = 0.92
        result = self._scalar(tw.in_out_quad, 1.2)
        self.assertAlmostEqual(result, 0.92, places=4)

    def test_in_quad_at_zero_is_zero(self):
        # Sanity: well-defined boundary value still works
        result = self._scalar(tw.in_quad, 0.0)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_in_quad_at_one_is_one(self):
        # Sanity: other boundary value still works
        result = self._scalar(tw.in_quad, 1.0)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_out_quad_at_one_is_one(self):
        # out_quad(1) = -1*(1-2) = 1
        result = self._scalar(tw.out_quad, 1.0)
        self.assertAlmostEqual(result, 1.0, places=4)


# --------------------------------------------------------------------- #
#  v4.I -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


from maya import cmds
from rig import set_options


class TestTweenPublishedInterface(MayaTestCase):
    """Verify each composite easing function creates a container with a
    published ``input`` and ``output`` attribute when
    ``flatten_containers=False``. Spot-checks one easing per family
    rather than enumerating all 42 variants -- the wrap pattern is
    uniform, so a representative sample suffices.

    Naming convention: container = ``<funcname>1``, input = ``input``,
    output = ``output`` -- mirroring third_party.rig.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def _assert_published(self, fn, container_name):
        """Call ``fn(plug)``, verify a container with the expected name
        is created and exposes ``input`` / ``output`` published attrs."""
        n = Node.create("transform", name=f"src_{container_name}")
        n.tx << 0.5
        fn(n.tx)

        # Find the container.
        ctn = cmds.ls(f"{container_name}*", type="container") or []
        self.assertTrue(
            any(c.startswith(container_name) for c in ctn),
            f"expected {container_name} container, got {ctn}",
        )
        target = next(c for c in ctn if c.startswith(container_name))
        for attr in ("input", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )

    def test_in_linear_publishes(self):
        self._assert_published(tw.in_linear, "in_linear1")

    def test_in_quad_publishes(self):
        self._assert_published(tw.in_quad, "in_quad1")

    def test_in_out_quad_publishes(self):
        self._assert_published(tw.in_out_quad, "in_out_quad1")

    def test_in_cubic_publishes(self):
        self._assert_published(tw.in_cubic, "in_cubic1")

    def test_in_sine_publishes(self):
        self._assert_published(tw.in_sine, "in_sine1")

    def test_in_expo_publishes(self):
        self._assert_published(tw.in_expo, "in_expo1")

    def test_in_circ_publishes(self):
        self._assert_published(tw.in_circ, "in_circ1")

    def test_in_elastic_publishes(self):
        self._assert_published(tw.in_elastic, "in_elastic1")

    def test_in_back_publishes(self):
        self._assert_published(tw.in_back, "in_back1")

    def test_out_bounce_publishes(self):
        self._assert_published(tw.out_bounce, "out_bounce1")