"""Smoke tests for ``rig.interpolate``."""

from maya import cmds
from rig import interpolate as i, Node, Plug
from rig._tests._base import MayaTestCase


class TestSequence(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sequence_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = i.sequence(a.tx, [b.tx, c.tx, a.ty], [b.ty, c.ty, a.tz])
        self.assertIsInstance(result, Plug)


class TestSmoothstep(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_smoothstep(self):
        self.assertAlmostEqual(i.smoothstep(0.0, 1.0, weight=0.5), 0.5)
        self.assertAlmostEqual(i.smoothstep(0.0, 1.0, weight=0.0), 0.0)
        self.assertAlmostEqual(i.smoothstep(0.0, 1.0, weight=1.0), 1.0)

    def test_plug_smoothstep_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = i.smoothstep(a.tx, b.tx, weight=c.tx)
        self.assertIsInstance(result, Plug)


class TestSmootherstep(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_smootherstep(self):
        self.assertAlmostEqual(i.smootherstep(0.0, 1.0, weight=0.5), 0.5)
        self.assertAlmostEqual(i.smootherstep(0.0, 1.0, weight=0.0), 0.0)
        self.assertAlmostEqual(i.smootherstep(0.0, 1.0, weight=1.0), 1.0)

    def test_plug_smootherstep_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = i.smootherstep(a.tx, b.tx, weight=c.tx)
        self.assertIsInstance(result, Plug)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, compare against ground truth.
#  Includes Eric's flagship 3-cube lerp example.
# --------------------------------------------------------------------- #


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestSmoothstepValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_smoothstep_midpoint(self):
        a = Node.create("transform", name="a")
        a.tx << 0.0
        b = Node.create("transform", name="b")
        b.tx << 1.0
        w = Node.create("transform", name="w")
        w.tx << 0.5
        out = Node.create("transform", name="out")
        out.ty << i.smoothstep(a.tx, b.tx, weight=w.tx)
        # 3*0.5^2 - 2*0.5^3 = 0.75 - 0.25 = 0.5.
        self.assertAlmostEqual(_eval(out.ty), 0.5, places=3)


# --------------------------------------------------------------------- #
#  v3.A -- inverse_lerp
# --------------------------------------------------------------------- #


class TestInverseLerpValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_inverse_lerp_known(self):
        # x = 2.5, a = 0, b = 10  ->  t = 0.25
        a = Node.create("transform", name="a")
        a.tx << 0.0
        b = Node.create("transform", name="b")
        b.tx << 10.0
        x = Node.create("transform", name="x")
        x.tx << 2.5
        out = Node.create("transform", name="out")
        out.ty << i.inverse_lerp(a.tx, b.tx, x.tx)
        self.assertAlmostEqual(_eval(out.ty), 0.25, places=4)

    def test_inverse_lerp_endpoints(self):
        # x == a -> 0 ; x == b -> 1
        a = Node.create("transform", name="a")
        a.tx << 5.0
        b = Node.create("transform", name="b")
        b.tx << 15.0
        x_at_a = Node.create("transform", name="x0")
        x_at_a.tx << 5.0
        x_at_b = Node.create("transform", name="x1")
        x_at_b.tx << 15.0
        out0 = Node.create("transform", name="out0")
        out1 = Node.create("transform", name="out1")
        out0.ty << i.inverse_lerp(a.tx, b.tx, x_at_a.tx)
        out1.ty << i.inverse_lerp(a.tx, b.tx, x_at_b.tx)
        self.assertAlmostEqual(_eval(out0.ty), 0.0, places=4)
        self.assertAlmostEqual(_eval(out1.ty), 1.0, places=4)


# --------------------------------------------------------------------- #
#  v4.H -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


from rig import set_options


class TestInterpolatePublishedInterface(MayaTestCase):
    """Verify each composite ``interpolate`` function creates a container
    with published ``input1`` / ``input2`` / ``weight`` (or per-function
    name) inputs and a published ``output`` when
    ``flatten_containers=False``. With the default
    ``flatten_containers=True`` the wraps are no-ops -- existing
    numerical tests above already cover that path."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    # -- sequence -- #

    def test_sequence_creates_published_container(self):
        x = Node.create("transform", name="x_src")
        x.tx << 1.5
        out = i.sequence(x.tx, [0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])

        ctn = cmds.ls("sequence*", type="container") or []
        self.assertTrue(any(c.startswith("sequence") for c in ctn))
        target = next(c for c in ctn if c.startswith("sequence"))
        from rig._internal.container import _resolve_published

        for attr in ("x", "xp", "yp", "output"):
            # ``xp`` / ``yp`` are MULTI (array) attrs -- registry-only on the
            # host under native publish, so resolve them through the container
            # rather than ``attributeQuery`` (which only sees real aliases).
            published = cmds.attributeQuery(attr, node=target, exists=True) or (
                _resolve_published(target, attr) is not None
            )
            self.assertTrue(published, f"{target}.{attr} should be published")
        # sequence(1.5, [0,1,2,3], [10,20,30,40]) -> lerp(20, 30, 0.5) = 25
        self.assertAlmostEqual(_eval(out), 25.0, places=4)

    # -- smoothstep -- #

    def test_smoothstep_compound_creates_published_container(self):
        # Force composite path via compound src.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [0, 0, 0]
        b.t << [1, 1, 1]
        i.smoothstep(a.t, b.t, weight=0.5)

        ctn = cmds.ls("smoothstep*", type="container") or []
        self.assertTrue(any(c.startswith("smoothstep") for c in ctn))
        target = next(c for c in ctn if c.startswith("smoothstep"))
        for attr in ("input1", "input2", "weight", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )

    # -- smootherstep -- #

    def test_smootherstep_creates_published_container(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.tx << 0.0
        b.tx << 1.0
        i.smootherstep(a.tx, b.tx, weight=0.3)

        ctn = cmds.ls("smootherstep*", type="container") or []
        self.assertTrue(any(c.startswith("smootherstep") for c in ctn))
        target = next(c for c in ctn if c.startswith("smootherstep"))
        for attr in ("input1", "input2", "weight", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )

    # -- inverse_lerp -- #

    def test_inverse_lerp_compound_creates_published_container(self):
        # Force composite path via compound src.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        x = Node.create("transform", name="x_src")
        a.t << [0, 0, 0]
        b.t << [10, 10, 10]
        x.t << [2.5, 2.5, 2.5]
        i.inverse_lerp(a.t, b.t, x.t)

        ctn = cmds.ls("inverse_lerp*", type="container") or []
        self.assertTrue(any(c.startswith("inverse_lerp") for c in ctn))
        target = next(c for c in ctn if c.startswith("inverse_lerp"))
        for attr in ("input1", "input2", "weight", "output"):
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )


class TestInterpolateAll(MayaTestCase):
    def test_public_surface(self):
        # Full T/R/S/shear ``transform`` moved to ``matrix.blend`` (matrix
        # ``slerp`` is orientation only); scalar/vector lerp/slerp/elerp
        # live in ``vector``.
        self.assertEqual(
            set(i.__all__),
            {"sequence", "smoothstep", "smootherstep", "inverse_lerp"},
        )
        self.assertFalse(hasattr(i, "transform"))


class TestLiteralVectorInputs(MayaTestCase):
    """Regression: smoothstep / inverse_lerp must accept FLAT LITERAL vec3
    inputs. The Maya-2024+ native nodes (smoothStep / inverseLerp) are
    scalar-only; a raw ``[x, y, z]`` must route to the component-wise path
    (gated by ``_is_scalar_value``) rather than crashing on the scalar node."""

    TEST_START_NEW_SCENE = True

    def _vec(self, plug):
        val = cmds.getAttr(str(plug))
        return val[0] if isinstance(val, list) else val

    def test_smoothstep_literal_vec3(self):
        result = i.smoothstep([0, 0, 0], [1, 1, 1], weight=[0.5, 0.5, 0.5])
        for a, e in zip(self._vec(result), (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(a, e, places=4)

    def test_inverse_lerp_literal_vec3(self):
        result = i.inverse_lerp([0, 0, 0], [10, 10, 10], [5, 5, 5])
        for a, e in zip(self._vec(result), (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(a, e, places=4)