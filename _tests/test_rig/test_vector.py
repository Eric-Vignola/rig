"""Smoke tests for ``rig.vector``."""

from unittest import mock

from maya import cmds
from rig import (
    angle,
    angle_degrees,
    dist,
    elerp,
    lerp,
    Node,
    normalize,
    Plug,
    set_options,
    slerp,
    vector as v,
)
from rig._internal.maya_version import is_at_least
from rig._internal.types import _is_matrix_literal
from rig._tests._base import MayaTestCase


# Tests in this file verify INTERNAL node creation. Under the leaf-frame-
# realness publishing rule (v4.P), top-level calls to module-factory functions
# create a published container whose ``output`` is the user-facing result. To
# keep these tests focused on the internal-node contract, each affected class
# disables publishing in its setUp / tearDown. The published interface for the
# same functions is covered by the ``TestVectorPublishedInterface`` class.
#
# The vector->orientation constructor (old ``vector.to_matrix`` / ``to_euler`` /
# ``to_quaternion``) moved to ``matrix.aim`` in the Model-A rescope; its tests
# live in ``test_matrix.py``. The cross-type ops (``dist`` / ``lerp`` /
# ``slerp`` / ``elerp`` / ``normalize`` / ``angle`` / ``angle_degrees``) are now
# reached through the top-level dispatch verbs imported above.


class TestVectorTripleProduct(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_triple_product_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = v.triple_product(a.t, b.t, c.t)
        self.assertIsInstance(result, Plug)


class TestVectorAngle(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_angle_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = angle(a.t, b.t)
        self.assertIsInstance(result, Plug)

    def test_angle_degrees_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = angle_degrees(a.t, b.t)
        self.assertIsInstance(result, Plug)


class TestVectorLerp(MayaTestCase):
    """``lerp`` relocated from :mod:`rig.interpolate` (Model-A P3)."""

    TEST_START_NEW_SCENE = True

    def test_scalar_lerp(self):
        self.assertAlmostEqual(lerp(0.0, 10.0, weight=0.5), 5.0)
        self.assertAlmostEqual(lerp(0.0, 10.0, weight=0.0), 0.0)
        self.assertAlmostEqual(lerp(0.0, 10.0, weight=1.0), 10.0)

    def test_plug_lerp_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = lerp(a.tx, b.tx, weight=0.5)
        self.assertIsInstance(result, Plug)


class TestVectorElerp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_elerp(self):
        # elerp(2, 8, 0.5) = 2^0.5 * 8^0.5 = sqrt(16) = 4
        self.assertAlmostEqual(elerp(2.0, 8.0, weight=0.5), 4.0)

    def test_plug_elerp_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = elerp(a.tx, b.tx, weight=0.5)
        self.assertIsInstance(result, Plug)


class TestVectorSlerp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_slerp_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = slerp(a.t, b.t, weight=0.5)
        self.assertIsInstance(result, Plug)


class TestVectorDotCross(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dot_creates_vectorproduct(self):
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = v.dot(a.t, b.t)
        node      = str(result).split(".")[0]
        node_type = cmds.nodeType(node)
        # Maya 2024+ uses native ``dotProduct``; older uses ``vectorProduct``.
        self.assertIn(node_type, {"vectorProduct", "dotProduct"})
        if node_type == "vectorProduct":
            self.assertEqual(cmds.getAttr(f"{node}.operation"), 1)

    def test_cross_creates_vectorproduct(self):
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = v.cross(a.t, b.t)
        node      = str(result).split(".")[0]
        node_type = cmds.nodeType(node)
        self.assertIn(node_type, {"vectorProduct", "crossProduct"})
        if node_type == "vectorProduct":
            self.assertEqual(cmds.getAttr(f"{node}.operation"), 2)

    def test_dot_normalized_uses_vectorproduct(self):
        # normalize=True forces the vectorProduct fallback (op=1) on all Maya
        # versions -- the native dotProduct node has no normalize flag.
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = v.dot(a.t, b.t, normalize=True)
        node   = str(result).split(".")[0]
        self.assertEqual(cmds.nodeType(node), "vectorProduct")
        self.assertEqual(cmds.getAttr(f"{node}.operation"), 1)

    def test_cross_normalized_uses_vectorproduct(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = v.cross(a.t, b.t, normalize=True)
        node   = str(result).split(".")[0]
        self.assertEqual(cmds.nodeType(node), "vectorProduct")
        self.assertEqual(cmds.getAttr(f"{node}.operation"), 2)


class TestVectorLengthNormalizeDist(MayaTestCase):
    """Topology checks for the core vector ops ``length`` / ``normalize`` /
    ``dist`` (relocated from :mod:`rig.functions` in the Model-A
    rescope; ``mag`` -> ``length``, ``unit`` -> ``normalize``)."""

    TEST_START_NEW_SCENE = True

    def test_length_scalar_list_short_circuits(self):
        # Pure-numeric input short-circuits to a Python float (no node built).
        self.assertAlmostEqual(v.length([3.0, 4.0]), 5.0)

    def test_length_creates_distance_or_length_node(self):
        a      = Node.create("transform", name="a")
        result = v.length(a.t)
        # Maya 2024+ uses native ``length`` for vec3; older / matrix
        # inputs route through ``distanceBetween``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"distanceBetween", "length"},
        )

    def test_length_of_matrix_uses_distancebetween(self):
        # A matrix input always routes through ``distanceBetween`` (fed via
        # ``inMatrix1``), even on Maya 2024+.
        a      = Node.create("transform", name="a")
        result = v.length(a.wm)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")

    def test_length_of_scalar_uses_distancebetween(self):
        # A non-compound, non-matrix input falls through to ``distanceBetween``
        # fed via ``point1`` (the native ``length`` node needs a vec3).
        a      = Node.create("transform", name="a")
        result = v.length(a.tx)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")

    def test_normalize_creates_normalize_or_divide_network(self):
        # Disable publishing so the result is the internal node plug, not the
        # published container output (v4.P leaf-frame rule): pre-2024
        # ``normalize`` is a multi-node ``v / length(v)`` network and therefore
        # gets a real container, unlike the single native ``normalize`` node.
        set_options(publish_attributes=False)
        try:
            a         = Node.create("transform", name="a")
            result    = normalize(a.t)
            node_type = cmds.nodeType(str(result).split(".")[0])
            # 2024+ => native ``normalize``; pre-2024 => ``v / length(v)`` ->
            # the result plug points at a ``multiplyDivide`` (op=2 divide).
            self.assertIn(node_type, {"normalize", "multiplyDivide"})
        finally:
            set_options(publish_attributes=True)

    def test_dist_creates_distancebetween(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = dist(a.t, b.t)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")

    def test_dist_of_matrices_uses_inmatrix(self):
        # Matrix inputs feed the distanceBetween node's inMatrix1/inMatrix2.
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = dist(a.wm, b.wm)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")

    @mock.patch("rig.vector._is_matrix", side_effect=RuntimeError("boom"))
    def test_dist_matrix_detection_failure_falls_back_to_point(self, _mock):
        # ``dist`` guards its matrix-vs-point routing with try/except: if
        # ``_is_matrix`` ever raises, both inputs degrade gracefully to the
        # ``point1`` / ``point2`` plugs rather than propagating the error.
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = dist(a.t, b.t)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")


class TestAxisConstants(MayaTestCase):
    """Sanity check that the axis constants are what we expect."""

    def test_axes(self):
        self.assertEqual(v.X, (1, 0, 0))
        self.assertEqual(v.Y, (0, 1, 0))
        self.assertEqual(v.Z, (0, 0, 1))
        # Immutable tuples (not lists) so the shared defaults can't be mutated.
        self.assertIsInstance(v.X, tuple)
        self.assertIsInstance(v.Y, tuple)
        self.assertIsInstance(v.Z, tuple)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, set inputs, getAttr, compare.
# --------------------------------------------------------------------- #


import math


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestDotValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dot_known(self):
        a = Node.create("transform", name="a")
        a.t << (1, 2, 3)
        b = Node.create("transform", name="b")
        b.t << (4, -5, 6)
        out = Node.create("transform", name="out")
        out.ty << v.dot(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), 12.0, places=4)

    def test_dot_perpendicular_is_zero(self):
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        out = Node.create("transform", name="out")
        out.ty << v.dot(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), 0.0, places=4)

    def test_dot_known_positive(self):
        a = Node.create("transform", name="a")
        a.t << (1, 2, 3)
        b = Node.create("transform", name="b")
        b.t << (4, 5, 6)
        out = Node.create("transform", name="out")
        out.ty << v.dot(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), 32.0, places=4)


class TestCrossValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_cross_x_y_is_z(self):
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        out = Node.create("transform", name="out")
        out.t << v.cross(a.t, b.t)
        for actual, expected in zip(_eval(out.t), [0, 0, 1]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestLengthValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_length_3_4_5_triangle_is_5(self):
        a = Node.create("transform", name="a")
        a.t << (3, 4, 0)
        out = Node.create("transform", name="out")
        out.ty << v.length(a.t)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)

    def test_length_of_flat16_matrix_literal_is_translation_magnitude(self):
        # A flat-16 (row-major) matrix literal with translation (3,4,0) must
        # route to ``distanceBetween`` (inMatrix1) and yield the translation
        # magnitude 5.0 -- NOT the 16-element Frobenius norm sqrt(29)=5.385
        # that the pure-numeric short-circuit would wrongly produce.
        flat16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 0, 1]
        result = v.length(flat16)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")
        out = Node.create("transform", name="out")
        out.ty << result
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)

    def test_length_of_nested_4x4_matrix_literal_is_translation_magnitude(self):
        # Nested 4x4 literal, translation (3,4,0) -> 5.0. Pre-fix this raises
        # ValueError injecting size-16 into the 3-channel ``point1``.
        nested = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [3, 4, 0, 1]]
        out    = Node.create("transform", name="out")
        out.ty << v.length(nested)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)


class TestNormalizeValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_normalize_5_0_0_is_1_0_0(self):
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        out = Node.create("transform", name="out")
        out.t << normalize(a.t)
        for actual, expected in zip(_eval(out.t), [1.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)

    @mock.patch("rig.vector.is_at_least", return_value=False)
    def test_normalize_legacy_divide_network(self, _mock):
        # Force the pre-2024 path (``v / length(v)`` with div-by-zero
        # quieting) on any Maya version so the legacy branch is exercised
        # and its math verified locally (otherwise only Maya < 2024 runs it).
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        out = Node.create("transform", name="out")
        out.t << normalize(a.t)
        for actual, expected in zip(_eval(out.t), [1.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestDistValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dist_3_4_5_triangle_is_5(self):
        a = Node.create("transform", name="a")
        a.t << (0, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (3, 4, 0)
        out = Node.create("transform", name="out")
        out.ty << dist(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)

    def test_dist_of_flat16_matrix_literals_is_translation_distance(self):
        # dist(matrix literal A, matrix literal B): translation(A)=(3,4,0),
        # translation(B)=(0,0,0) -> distance 5.0 via inMatrix1/inMatrix2.
        # Pre-fix this raises ValueError (size-16 into 3-channel ``point1``).
        a      = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 0, 1]
        b      = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        result = v.dist(a, b)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")
        out = Node.create("transform", name="out")
        out.ty << result
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)

    def test_dist_of_nested_4x4_matrix_literals_is_translation_distance(self):
        # Nested 4x4 literals -> same 5.0 translation distance.
        a   = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [3, 4, 0, 1]]
        b   = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        out = Node.create("transform", name="out")
        out.ty << v.dist(a, b)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)


class TestIsMatrixLiteral(MayaTestCase):
    """Unit coverage for the ``_is_matrix_literal`` predicate -- it routes raw
    matrix LISTS (which the attribute-only ``_is_matrix`` misses) to the matrix
    path in ``length`` / ``dist``. Crash-safe: ``False`` on anything that is
    not a flat 9/16 or nested 3x3/4x4 sequence of plain numbers."""

    def test_flat_16_and_9_and_nested_are_matrix_literals(self):
        self.assertTrue(_is_matrix_literal([0.0] * 16))
        self.assertTrue(_is_matrix_literal([1, 0, 0, 0, 1, 0, 0, 0, 1]))  # flat 3x3
        self.assertTrue(
            _is_matrix_literal([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )

    def test_wrong_length_is_not_matrix_literal(self):
        self.assertFalse(_is_matrix_literal([1, 2, 3]))     # vec3
        self.assertFalse(_is_matrix_literal([1, 2, 3, 4]))  # vec4 / quaternion
        self.assertFalse(_is_matrix_literal([1.0] * 12))

    def test_non_numeric_or_bool_elements_excluded(self):
        # Right length but wrong element types -> not a matrix literal.
        self.assertFalse(_is_matrix_literal(["x"] * 16))
        self.assertFalse(_is_matrix_literal([True] * 16))  # bool is not a channel

    def test_non_sequence_is_not_matrix_literal(self):
        self.assertFalse(_is_matrix_literal(5.0))
        self.assertFalse(_is_matrix_literal("matrix"))
        self.assertFalse(_is_matrix_literal(None))


class TestAngleValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_angle_perpendicular_is_pi_over_2(self):
        # ``angle`` returns RADIANS (parity with quaternion.angle).
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        out = Node.create("transform", name="out")
        out.ty << angle(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), math.pi / 2, places=4)

    def test_angle_degrees_perpendicular_is_90(self):
        # ``angle_degrees`` returns DEGREES.
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        out = Node.create("transform", name="out")
        out.ty << angle_degrees(a.t, b.t)
        self.assertAlmostEqual(_eval(out.ty), 90.0, places=3)

    def test_angle_degrees_output_is_doubleAngle(self):
        # ``angle_degrees`` uses the degree-variant ``atan2d`` directly, so
        # its output is a ``doubleAngle`` plug -- idiomatic for the degree-
        # variant trig ops (``atan2d`` / ``asind`` / ``acosd`` / ``atand``
        # all return ``doubleAngle``). The value is identical to the former
        # ``degrees(radians(atan2d(...)))`` round-trip under any scene angle
        # unit; only the redundant radians->degrees conversion nodes are gone.
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        result = angle_degrees(a.t, b.t)
        self.assertEqual(cmds.getAttr(str(result), type=True), "doubleAngle")


class TestTripleProductValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_triple_product_standard_basis_is_1(self):
        # det([X, Y, Z]) = X * (Y x Z) = 1.
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        c = Node.create("transform", name="c")
        c.t << (0, 0, 1)
        out = Node.create("transform", name="out")
        out.ty << v.triple_product(a.t, b.t, c.t)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)

    @mock.patch("rig.vector.is_at_least", return_value=False)
    def test_triple_product_legacy_formula(self, _mock):
        # Force the pre-2024 expanded-determinant branch on any Maya so the
        # legacy formula is exercised and its math verified locally.
        a = Node.create("transform", name="a")
        a.t << (1, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (0, 1, 0)
        c = Node.create("transform", name="c")
        c.t << (0, 0, 1)
        out = Node.create("transform", name="out")
        out.ty << v.triple_product(a.t, b.t, c.t)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)


class TestVectorLerpValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lerp_3cube_example(self):
        # Eric's flagship example: 3 cubes, lerp the third 0.25 between
        # the other two; verify position matches the math.
        a = Node.create("transform", name="cubeA")
        a.t << (0, 0, 0)
        b = Node.create("transform", name="cubeB")
        b.t << (10, 20, 30)
        lerped = Node.create("transform", name="cubeLerp")
        lerped.t << lerp(a.t, b.t, weight=0.25)
        # Expected: a + 0.25 * (b - a) = (2.5, 5, 7.5).
        for actual, expected in zip(_eval(lerped.t), [2.5, 5.0, 7.5]):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_lerp_endpoints(self):
        a = Node.create("transform", name="a")
        a.tx << 0.0
        b = Node.create("transform", name="b")
        b.tx << 100.0
        out0 = Node.create("transform", name="out0")
        out1 = Node.create("transform", name="out1")
        out0.tx << lerp(a.tx, b.tx, weight=0.0)
        out1.tx << lerp(a.tx, b.tx, weight=1.0)
        self.assertAlmostEqual(_eval(out0.tx), 0.0, places=4)
        self.assertAlmostEqual(_eval(out1.tx), 100.0, places=4)

    @mock.patch("rig.vector.is_at_least", return_value=False)
    def test_lerp_legacy_scalar_blendweighted(self, _mock):
        # Force the pre-2024 scalar path (single blendWeighted node) so it
        # is exercised and verified locally (else only Maya < 2024 runs it).
        a = Node.create("transform", name="a")
        a.tx << 0.0
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = Node.create("transform", name="out")
        out.ty << lerp(a.tx, b.tx, weight=0.5)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)


class TestVectorSlerpValue(MayaTestCase):
    """Value tests for ``vector.slerp`` -- spherical interpolation must trace
    the ``input1 -> input2`` minor great-circle arc (NO antipodal / shortest-
    path flip) and blend magnitude, for ALL angles 0-180 deg and arbitrary
    magnitudes. These pin the legacy formula
    ``(a*sin((1-w)th) + b*sin(w*th)) / sin(th)`` that the ``quatSlerp`` trick
    silently broke past 90 deg (it normalizes its inputs and takes the
    shortest path, flipping the result into the opposite quadrant)."""

    TEST_START_NEW_SCENE = True

    def test_slerp_90deg_unequal_magnitude(self):
        # User's working baseline: perpendicular, mags 5 and 10, t=0.5.
        # (sin45*(0,0,-5) + sin45*(10,0,0)) / sin90 = (7.071, 0, -3.536).
        a = Node.create("transform", name="a")
        a.t << (0, 0, -5)
        b = Node.create("transform", name="b")
        b.t << (10, 0, 0)
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        for actual, expected in zip(_eval(out.t), [7.0711, 0.0, -3.5355]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_past_90deg_no_quadrant_flip(self):
        # 120 deg apart, mags 5 and 10, t=0.5. Legacy: c0=c1=sin60/sin120=1,
        # so result = a + b = (0, 0, 8.6603). ``quatSlerp`` takes the shortest
        # path (-60 deg) and lands in the WRONG quadrant -> fails on the broken
        # impl, passes on the legacy formula.
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (10 * math.cos(math.radians(120)), 0, 10 * math.sin(math.radians(120)))
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        for actual, expected in zip(_eval(out.t), [0.0, 0.0, 8.6603]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_user_nudge_does_not_flip_quadrant(self):
        # User's exact repro: nudging the 90-deg boundary just past 90 must NOT
        # teleport the result to the opposite quadrant. Stays ~ the 90-deg
        # answer (7.071, 0, -3.536): X positive, Z negative.
        a = Node.create("transform", name="a")
        a.t << (-0.0001, 0, -5)
        b = Node.create("transform", name="b")
        b.t << (10, 0, 0)
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        x, y, z = _eval(out.t)
        self.assertAlmostEqual(x, 7.0711,  places=2)
        self.assertAlmostEqual(y, 0.0,     places=3)
        self.assertAlmostEqual(z, -3.5355, places=2)

    def test_slerp_3d_obtuse_angle(self):
        # Genuinely-3D, 109.47 deg apart, equal mag sqrt(3). Legacy:
        # c0=c1=sin(54.735)/sin(109.47)=0.866; result =
        # 0.866*((1,1,1)+(-1,-1,1)) = (0, 0, 1.7320).
        a = Node.create("transform", name="a")
        a.t << (1, 1, 1)
        b = Node.create("transform", name="b")
        b.t << (-1, -1, 1)
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        for actual, expected in zip(_eval(out.t), [0.0, 0.0, 1.7320]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_variable_weight_plug_past_90(self):
        # Weight driven by a live plug (not a constant), 120 deg apart. At
        # t=0.5 -> (0, 0, 8.6603); proves the formula works for plug weights.
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (10 * math.cos(math.radians(120)), 0, 10 * math.sin(math.radians(120)))
        w = Node.create("transform", name="w")
        w.tx << 0.5
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=w.tx)
        for actual, expected in zip(_eval(out.t), [0.0, 0.0, 8.6603]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_literal_vectors_past_90(self):
        # Literal (non-plug) inputs, 120 deg apart -> (0, 0, 8.6603). Guards
        # the literal-input path through the legacy formula past 90 deg.
        bx  = 10 * math.cos(math.radians(120))
        bz  = 10 * math.sin(math.radians(120))
        out = Node.create("transform", name="out")
        out.t << slerp([5, 0, 0], [bx, 0, bz], weight=0.5)
        for actual, expected in zip(_eval(out.t), [0.0, 0.0, 8.6603]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_endpoints(self):
        # t=0 -> input1 exactly, t=1 -> input2 exactly (magnitude preserved).
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (-5, 0, 8.6603)
        out0 = Node.create("transform", name="out0")
        out1 = Node.create("transform", name="out1")
        out0.t << slerp(a.t, b.t, weight=0.0)
        out1.t << slerp(a.t, b.t, weight=1.0)
        for actual, expected in zip(_eval(out0.t), [5.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=3)
        for actual, expected in zip(_eval(out1.t), [-5.0, 0.0, 8.6603]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_parallel_falls_back_to_lerp(self):
        # Parallel inputs (theta=0): great-circle undefined. Robust guard
        # falls back to linear interpolation -> (3, 0, 0), all finite, no
        # div-by-zero.
        a = Node.create("transform", name="a")
        a.t << (2, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (4, 0, 0)
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        result = _eval(out.t)
        for actual in result:
            self.assertTrue(math.isfinite(actual))
        for actual, expected in zip(result, [3.0, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_slerp_antiparallel_is_finite(self):
        # Anti-parallel inputs (theta=180): sin(theta)=0 -> the legacy formula
        # divides by zero. Robust guard detects sin<=1e-6 and falls back to a
        # finite linear interpolation: 0.5*(5,0,0) + 0.5*(-10,0,0) = (-2.5,0,0).
        a = Node.create("transform", name="a")
        a.t << (5, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (-10, 0, 0)
        out = Node.create("transform", name="out")
        out.t << slerp(a.t, b.t, weight=0.5)
        result = _eval(out.t)
        for actual in result:
            self.assertTrue(math.isfinite(actual))
        for actual, expected in zip(result, [-2.5, 0.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=3)


# --------------------------------------------------------------------- #
#  v3.A -- vector.rotate
# --------------------------------------------------------------------- #


class TestRotateValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_rotate_x_by_90_about_z(self):
        # X-axis rotated 90deg about Z -> Y-axis (~= (0, 1, 0)).
        from rig import vector as vec_mod

        v = Node.create("transform", name="v")
        v.t << (1, 0, 0)
        eul = Node.create("transform", name="eul")
        eul.r << (0, 0, 90)
        out = Node.create("transform", name="out")
        out.t << vec_mod.rotate(v.t, eul.r)
        for actual, expected in zip(_eval(out.t), [0, 1, 0]):
            self.assertAlmostEqual(actual, expected, places=3)


# --------------------------------------------------------------------- #
#  v4.L -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


class TestVectorPublishedInterface(MayaTestCase):
    """Verify each wrapped composite vector function creates a container
    with the expected published attributes when ``flatten_containers=False``."""

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

    def test_angle_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 0, 0]
        b.t << [0, 1, 0]
        angle(a.t, b.t)
        self._assert_attrs("vector_angle1", ["input1", "input2", "output"])

    def test_angle_degrees_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 0, 0]
        b.t << [0, 1, 0]
        angle_degrees(a.t, b.t)
        self._assert_attrs("vector_angle_degrees1", ["input1", "input2", "output"])

    def test_triple_product_publishes(self):
        v.triple_product([1, 0, 0], [0, 1, 0], [0, 0, 1])
        self._assert_attrs(
            "vector_triple_product1", ["input1", "input2", "input3", "output"]
        )

    def test_rotate_publishes(self):
        a   = Node.create("transform", name="a")
        eul = Node.create("transform", name="eul")
        a.t << [1, 0, 0]
        v.rotate(a.t, eul.r)
        self._assert_attrs("rotate1", ["vector", "rotate", "output"])

    def test_normalize_legacy_publishes(self):
        # Pre-2024 ``normalize`` wraps ``v / length(v)`` in a published
        # container; on Maya 2024+ it uses the native node (nothing to wrap).
        if not is_at_least(2024):
            a = Node.create("transform", name="a")
            a.t << [1, 0, 0]
            normalize(a.t)
            self._assert_attrs("normalize1", ["input", "output"])

    def test_lerp_compound_publishes(self):
        # Compound inputs force the blendWeighted container path (the scalar
        # 2024+ path uses a bare ``lerp`` node with nothing to publish).
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [0, 0, 0]
        b.t << [10, 20, 30]
        lerp(a.t, b.t, weight=0.25)
        self._assert_attrs("lerp1", ["input1", "input2", "weight", "output"])

    def test_elerp_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.tx << 2.0
        b.tx << 8.0
        elerp(a.tx, b.tx, weight=0.5)
        self._assert_attrs("elerp1", ["input1", "input2", "weight", "output"])

    def test_slerp_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 0, 0]
        b.t << [0, 1, 0]
        slerp(a.t, b.t, weight=0.5)
        self._assert_attrs("slerp1", ["input1", "input2", "weight", "output"])


# --------------------------------------------------------------------- #
#  Public API surface (__all__) -- P2 rescope
# --------------------------------------------------------------------- #


class TestVectorAll(MayaTestCase):
    """``vector``'s public surface is the vector-SPECIFIC ops plus the
    cross-type ops, now exposed as public per-type functions (also reachable
    through the top-level dispatch verbs). The old MEL names are gone -- no
    shims."""

    def test_public_surface(self):
        # Vector-specific ops (products + triple_product + rotate + the X/Y/Z
        # axis constants) PLUS the now-public cross-type per-type ops.
        self.assertEqual(
            set(v.__all__),
            {
                "X",
                "Y",
                "Z",
                "triple_product",
                "angle",
                "angle_degrees",
                "dot",
                "cross",
                "length",
                "normalize",
                "dist",
                "rotate",
                "lerp",
                "slerp",
                "elerp",
            },
        )
        for name in v.__all__:
            self.assertTrue(hasattr(v, name), f"vector.{name} missing")

    def test_cross_type_ops_public(self):
        # dist/lerp/slerp/elerp/normalize/angle/angle_degrees are now PUBLIC
        # per-type funcs on vector (callable directly), in addition to the
        # top-level dispatch verbs (``from rig import lerp``). The old
        # ``_``-prefixed impl names are gone.
        for name in (
            "dist",
            "lerp",
            "slerp",
            "elerp",
            "normalize",
            "angle",
            "angle_degrees",
        ):
            self.assertIn(name, v.__all__)
            self.assertTrue(hasattr(v, name), f"vector.{name} should be public")
            self.assertFalse(
                hasattr(v, f"_{name}"), f"vector._{name} private alias should be gone"
            )

    def test_old_names_removed(self):
        for old in ("mag", "unit", "determinant", "angle_between"):
            self.assertNotIn(old, v.__all__)
            self.assertFalse(hasattr(v, old), f"vector.{old} should be renamed")