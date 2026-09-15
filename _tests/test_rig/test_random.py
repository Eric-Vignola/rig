"""Smoke tests for ``rig.random``.

Each function tested for: builds without error, returns a Plug,
respects seed determinism (two calls with the same seed produce the
same value at frame 1).
"""

from maya import cmds
from rig import Node, Plug, random as r
from rig._tests._base import MayaTestCase


class TestValueScalar(MayaTestCase):
    """``value()`` -- pseudo-random scalar in [0, 1)."""

    TEST_START_NEW_SCENE = True

    def test_value_returns_plug(self):
        result = r.value(seed=42)
        self.assertIsInstance(result, Plug)

    def test_value_with_trigger_returns_plug(self):
        a      = Node.create("transform", name="trig")
        result = r.value(trigger=a.tx, seed=42)
        self.assertIsInstance(result, Plug)

    def test_value_seed_determinism(self):
        # Two networks with the same seed produce the same value at
        # frame 1 (which is what `frame()` is anchored at).
        cmds.currentTime(1)
        r1 = r.value(seed=12345)
        r2 = r.value(seed=12345)
        v1 = cmds.getAttr(str(r1))
        v2 = cmds.getAttr(str(r2))
        self.assertAlmostEqual(v1, v2, places=4)

    def test_value_in_unit_range(self):
        cmds.currentTime(1)
        result = r.value(seed=7)
        v      = cmds.getAttr(str(result))
        self.assertGreaterEqual(v, 0.0)
        self.assertLess(v, 1.0)


class TestValue3D(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_value3d_returns_plug(self):
        result = r.value3D(seed=[1, 2, 3])
        self.assertIsInstance(result, Plug)


class TestUniform(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_uniform_returns_plug(self):
        result = r.uniform(0, 10, seed=42)
        self.assertIsInstance(result, Plug)

    def test_uniform_in_range(self):
        cmds.currentTime(1)
        result = r.uniform(5, 15, seed=42)
        v      = cmds.getAttr(str(result))
        self.assertGreaterEqual(v, 5.0)
        self.assertLess(v, 15.0)


class TestUniform3D(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_uniform3d_returns_plug(self):
        result = r.uniform3D(0, 10, seed=[1, 2, 3])
        self.assertIsInstance(result, Plug)


class TestRandint(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_randint_returns_plug(self):
        result = r.randint(0, 100, seed=42)
        self.assertIsInstance(result, Plug)

    def test_randint_value_is_long(self):
        cmds.currentTime(1)
        result = r.randint(0, 100, seed=42)
        v      = cmds.getAttr(str(result))
        self.assertEqual(int(v), v)
        self.assertGreaterEqual(v, 0)
        self.assertLessEqual(v, 100)


class TestRandint3D(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_randint3d_returns_plug(self):
        result = r.randint3D(0, 100, seed=[1, 2, 3])
        self.assertIsInstance(result, Plug)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- set time, query, verify range/integer.
# --------------------------------------------------------------------- #


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestValueRange(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_value_in_zero_one(self):
        cmds.currentTime(1)
        v = _eval(r.value(seed=42))
        self.assertGreaterEqual(v, 0.0)
        self.assertLess(v, 1.0)


class TestUniformValueRange(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_uniform_in_5_15(self):
        cmds.currentTime(1)
        v = _eval(r.uniform(5, 15, seed=42))
        self.assertGreaterEqual(v, 5.0)
        self.assertLess(v, 15.0)


class TestRandintValueRange(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_randint_in_1_10(self):
        cmds.currentTime(1)
        v = _eval(r.randint(1, 10, seed=42))
        self.assertGreaterEqual(v, 1)
        self.assertLessEqual(v, 10)
        self.assertEqual(v, int(v), "randint should produce integer")


class TestValueSeedMemoize(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_explicit_seed_dedupes(self):
        # Two calls with same explicit seed should return SAME plug.
        r1 = r.value(seed=42)
        r2 = r.value(seed=42)
        self.assertEqual(str(r1), str(r2))


class TestValueNoSeedIndependent(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_no_seed_produces_independent_streams(self):
        # Auto-seeded calls must NOT dedupe -- fresh stream per call.
        r1 = r.value()
        r2 = r.value()
        self.assertNotEqual(str(r1), str(r2))


# --------------------------------------------------------------------- #
#  v4.M -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


from rig import set_options


class TestRandomPublishedInterface(MayaTestCase):
    """Verify each wrapped random function creates a container with the
    expected published attributes when ``flatten_containers=False``.

    ``trigger`` is published only when explicitly passed (defaults to
    frame() and isn't a Plug arg in that case). ``seed`` is build-time
    (Python int) so never published.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def _assert_attrs(self, container_name, attrs, anti_attrs=()):
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
        for attr in anti_attrs:
            self.assertFalse(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should NOT be published",
            )

    def test_value_publishes_with_explicit_trigger(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        r.value(trigger=a.tx, seed=42)
        self._assert_attrs("value1", ["trigger", "output"])

    def test_value_no_trigger_publishes_only_output(self):
        # trigger=None defaults to frame() -- not published.
        r.value(seed=42)
        self._assert_attrs("value1", ["output"], anti_attrs=["trigger"])

    def test_uniform_publishes(self):
        r.uniform(0.0, 10.0, seed=42)
        self._assert_attrs("uniform1", ["start", "end", "output"])

    def test_randint_publishes(self):
        r.randint(0, 100, seed=42)
        self._assert_attrs("randint1", ["start", "end", "output"])

    def test_value3D_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        r.value3D(trigger=a.tx, seed=[1, 2, 3])
        self._assert_attrs("value3d1", ["trigger", "output"])

    def test_uniform3D_publishes(self):
        r.uniform3D([0, 0, 0], [10, 10, 10], seed=[1, 2, 3])
        self._assert_attrs("uniform3d1", ["start", "end", "output"])

    def test_randint3D_publishes(self):
        r.randint3D([0, 0, 0], [100, 100, 100], seed=[1, 2, 3])
        self._assert_attrs("randint3d1", ["start", "end", "output"])


class TestRandomAll(MayaTestCase):
    def test_public_surface(self):
        # ``random``->``value`` / ``random3D``->``value3D`` kill the
        # ``random.random`` module-name clash.
        self.assertEqual(
            set(r.__all__),
            {"value", "uniform", "randint", "value3D", "uniform3D", "randint3D"},
        )
        self.assertFalse(hasattr(r, "random"))
        self.assertFalse(hasattr(r, "random3D"))