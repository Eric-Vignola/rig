"""Smoke tests for ``rig.functions``.

One test per public function: assert the correct Maya node type is built
and at least one input is wired. A few aggregator / round-trip tests
exercise list-handling. Numerical correctness is left to Eric's
upstream -- these tests guarantee the network shape only.
"""

from maya import cmds
from rig import functions as f, Node, Plug, PlugList
from rig._tests._base import MayaTestCase


class TestFrame(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_frame_creates_animcurvetl(self):
        plug = f.frame()
        self.assertIsInstance(plug, Plug)
        node_name = str(plug).split(".")[0]
        self.assertEqual(cmds.nodeType(node_name), "animCurveTL")


class TestClamp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_clamp(self):
        self.assertEqual(f.clamp(5.0, 0.0, 1.0),  1.0)
        self.assertEqual(f.clamp(-5.0, 0.0, 1.0), 0.0)
        self.assertEqual(f.clamp(0.5, 0.0, 1.0),  0.5)

    def test_plug_clamp_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = f.clamp(a.tx, 0.0, 1.0)
        self.assertIsInstance(result, Plug)


class TestAbs(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_abs(self):
        self.assertEqual(f.abs(-5.0), 5.0)
        self.assertEqual(f.abs(5.0), 5.0)

    def test_plug_abs(self):
        a      = Node.create("transform", name="a")
        result = f.abs(a.tx)
        self.assertIsInstance(result, Plug)


class TestInt(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_int(self):
        self.assertEqual(f.int(3.7), 3)
        self.assertEqual(f.int(-3.7), -3)

    def test_plug_int(self):
        a      = Node.create("transform", name="a")
        result = f.int(a.tx)
        self.assertIsInstance(result, Plug)


class TestRound(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_round(self):
        self.assertEqual(f.round(3.14159, 2), 3.14)

    def test_plug_round(self):
        a      = Node.create("transform", name="a")
        result = f.round(a.tx, 2)
        self.assertIsInstance(result, Plug)


class TestFloor(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_floor(self):
        self.assertEqual(f.floor(3.7), 3)

    def test_plug_floor(self):
        a      = Node.create("transform", name="a")
        result = f.floor(a.tx)
        self.assertIsInstance(result, Plug)


class TestCeil(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_ceil(self):
        self.assertEqual(f.ceil(3.2), 4)

    def test_plug_ceil(self):
        a      = Node.create("transform", name="a")
        result = f.ceil(a.tx)
        self.assertIsInstance(result, Plug)


class TestTrunc(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_trunc(self):
        self.assertEqual(f.trunc(3.7), 3)
        self.assertEqual(f.trunc(-3.7), -3)

    def test_plug_trunc(self):
        a      = Node.create("transform", name="a")
        result = f.trunc(a.tx)
        self.assertIsInstance(result, Plug)


class TestSum(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_sum(self):
        self.assertEqual(f.sum([1.0, 2.0, 3.0]), 6.0)

    def test_plug_sum_creates_pma(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.sum([a.tx, b.tx])
        # Maya 2024+ uses native ``sum`` node; older uses ``plusMinusAverage``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"plusMinusAverage", "sum"},
        )


class TestAvg(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_avg(self):
        self.assertEqual(f.avg([2.0, 4.0]), 3.0)

    def test_plug_avg_uses_pma_op3(self):
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = f.avg([a.tx, b.tx])
        node      = str(result).split(".")[0]
        node_type = cmds.nodeType(node)
        # Maya 2024+ uses native ``average`` node; older uses ``plusMinusAverage``
        # with operation=3 (which we verify only when on the legacy path).
        self.assertIn(node_type, {"plusMinusAverage", "average"})
        if node_type == "plusMinusAverage":
            self.assertEqual(cmds.getAttr(f"{node}.operation"), 3)


class TestMax(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_max(self):
        self.assertEqual(f.max([1.0, 5.0, 3.0]), 5.0)

    def test_plug_max(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.max([a.tx, b.tx])
        self.assertIsInstance(result, Plug)

    def test_max_too_few_args_raises(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(ValueError):
            f.max([a.tx])


class TestMin(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_min(self):
        self.assertEqual(f.min([1.0, 5.0, 3.0]), 1.0)

    def test_plug_min(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.min([a.tx, b.tx])
        self.assertIsInstance(result, Plug)


class TestExp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_exp(self):
        import math

        self.assertAlmostEqual(f.exp(1.0), math.e)

    def test_plug_exp(self):
        a      = Node.create("transform", name="a")
        result = f.exp(a.tx)
        # Maya 2024+ may use native ``power`` node via the new
        # multiply_divide 2024 dispatch; older uses ``multiplyDivide``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "power"},
        )


class TestSign(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_sign(self):
        self.assertEqual(f.sign(5.0),  1)
        self.assertEqual(f.sign(-5.0), -1)
        self.assertEqual(f.sign(0.0),  1)

    def test_plug_sign(self):
        a      = Node.create("transform", name="a")
        result = f.sign(a.tx)
        self.assertIsInstance(result, Plug)


class TestSqrt(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_sqrt(self):
        self.assertEqual(f.sqrt(4.0), 2.0)

    def test_plug_sqrt(self):
        a      = Node.create("transform", name="a")
        result = f.sqrt(a.tx)
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "power"},
        )


class TestPow(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_pow(self):
        self.assertEqual(f.pow(2.0, 3.0), 8.0)

    def test_plug_pow(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.pow(a.tx, b.tx)
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "power"},
        )


class TestChoice(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_choice_creates_choice_node(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.choice([a.tx, b.tx])
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "choice")


class TestRev(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_rev(self):
        self.assertEqual(f.rev(0.3), 0.7)

    def test_plug_rev(self):
        a      = Node.create("transform", name="a")
        result = f.rev(a.tx)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "reverse")


class TestSearchsorted(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_searchsorted_returns_plug(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = f.searchsorted([a.tx, b.tx, c.tx], 5.0)
        self.assertIsInstance(result, Plug)

    def test_invalid_side_raises(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(ValueError):
            f.searchsorted([a.tx], 5.0, side="middle")


class TestAll(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_all(self):
        self.assertTrue(f.all([1, 2, 3]))
        self.assertFalse(f.all([1, 0, 3]))

    def test_plug_all(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.all([a.tx, b.tx])
        self.assertIsInstance(result, Plug)


class TestAny(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_any(self):
        self.assertTrue(f.any([0, 0, 1]))
        self.assertFalse(f.any([0, 0, 0]))

    def test_plug_any(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.any([a.tx, b.tx])
        self.assertIsInstance(result, Plug)


class TestArgmin(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_argmin(self):
        self.assertEqual(f.argmin([5.0, 1.0, 3.0]), 1)

    def test_plug_argmin(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.argmin([a.tx, b.tx])
        self.assertIsInstance(result, Plug)


class TestArgmax(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_argmax(self):
        self.assertEqual(f.argmax([1.0, 5.0, 3.0]), 1)

    def test_plug_argmax(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = f.argmax([a.tx, b.tx])
        self.assertIsInstance(result, Plug)


class TestDiff(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_diff_returns_pluglist(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = f.diff([a.tx, b.tx, c.tx])
        self.assertIsInstance(result, PlugList)
        # n inputs => n-1 differences.
        self.assertEqual(len(result), 2)


class TestCumsum(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_cumsum_returns_pluglist(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        c      = Node.create("transform", name="c")
        result = f.cumsum([a.tx, b.tx, c.tx])
        self.assertIsInstance(result, PlugList)
        # n inputs => n running sums.
        self.assertEqual(len(result), 3)


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, set inputs, getAttr, compare
#  against ground-truth Python computation.
# --------------------------------------------------------------------- #


def _eval(plug):
    """Read a plug's current value via cmds.getAttr; flatten compound."""
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestClampValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_clamp_below(self):
        n = Node.create("transform", name="probe")
        n.tx << -5.0
        n.ty << f.clamp(n.tx, 0, 3)
        self.assertAlmostEqual(_eval(n.ty), 0.0, places=4)

    def test_clamp_above(self):
        n = Node.create("transform", name="probe")
        n.tx << 7.0
        n.ty << f.clamp(n.tx, 0, 3)
        self.assertAlmostEqual(_eval(n.ty), 3.0, places=4)

    def test_clamp_inside(self):
        n = Node.create("transform", name="probe")
        n.tx << 2.0
        n.ty << f.clamp(n.tx, 0, 3)
        self.assertAlmostEqual(_eval(n.ty), 2.0, places=4)


class TestAbsValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_abs_negative(self):
        n = Node.create("transform", name="probe")
        n.tx << -3.5
        n.ty << f.abs(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 3.5, places=4)


class TestIntValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_int_truncates(self):
        n = Node.create("transform", name="probe")
        n.tx << 3.7
        n.ty << f.int(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 3.0, places=4)


class TestFloorValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_floor_positive(self):
        n = Node.create("transform", name="probe")
        n.tx << 2.7
        n.ty << f.floor(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 2.0, places=4)

    def test_floor_negative(self):
        n = Node.create("transform", name="probe")
        n.tx << -2.3
        n.ty << f.floor(n.tx)
        self.assertAlmostEqual(_eval(n.ty), -3.0, places=4)


class TestCeilValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_ceil_positive(self):
        n = Node.create("transform", name="probe")
        n.tx << 2.3
        n.ty << f.ceil(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 3.0, places=4)


class TestTruncValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_trunc_negative(self):
        n = Node.create("transform", name="probe")
        n.tx << -2.7
        n.ty << f.trunc(n.tx)
        self.assertAlmostEqual(_eval(n.ty), -2.0, places=4)


class TestSumValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sum_three(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 2.0
        c = Node.create("transform", name="c")
        c.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << f.sum([a.tx, b.tx, c.tx])
        self.assertAlmostEqual(_eval(out.ty), 6.0, places=4)


class TestAvgValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_avg_three(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 2.0
        c = Node.create("transform", name="c")
        c.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << f.avg([a.tx, b.tx, c.tx])
        self.assertAlmostEqual(_eval(out.ty), 2.0, places=4)


class TestMaxValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_max_three(self):
        # Regression test: ``max([>2 items])`` previously returned the
        # last item due to (a) wrong gate predicate and (b) array-attr
        # writes landing on slot 0.
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 5.0
        c = Node.create("transform", name="c")
        c.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << f.max([a.tx, b.tx, c.tx])
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)


class TestMinValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_min_three(self):
        # See TestMaxValue for the regression history.
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 5.0
        c = Node.create("transform", name="c")
        c.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << f.min([a.tx, b.tx, c.tx])
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)


class TestExpValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_exp_e(self):
        import math

        n = Node.create("transform", name="probe")
        n.tx << 1.0
        n.ty << f.exp(n.tx)
        self.assertAlmostEqual(_eval(n.ty), math.e, places=3)


class TestSignValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sign_positive(self):
        n = Node.create("transform", name="probe")
        n.tx << 5.0
        n.ty << f.sign(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=4)

    def test_sign_negative(self):
        n = Node.create("transform", name="probe")
        n.tx << -3.0
        n.ty << f.sign(n.tx)
        self.assertAlmostEqual(_eval(n.ty), -1.0, places=4)


class TestSqrtValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_sqrt_16(self):
        n = Node.create("transform", name="probe")
        n.tx << 16.0
        n.ty << f.sqrt(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 4.0, places=4)


class TestPowValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_pow_2_8(self):
        n = Node.create("transform", name="probe")
        n.tx << 2.0
        n.ty << f.pow(n.tx, 8)
        self.assertAlmostEqual(_eval(n.ty), 256.0, places=4)


class TestRevValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_rev_0_3(self):
        n = Node.create("transform", name="probe")
        n.tx << 0.3
        n.ty << f.rev(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 0.7, places=4)


# --------------------------------------------------------------------- #
#  v3.A -- pi + log
# --------------------------------------------------------------------- #


class TestPiValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_pi_value(self):
        import math

        # Maya 2024+ builds a native ``pi`` node whose doubleAngle output
        # reads back through cmds.getAttr in the current angle unit
        # (degrees by default). Older Maya falls back to a ``_constant``
        # network node holding the raw radian value, with no unit conversion.
        val      = _eval(f.pi())
        expected = math.degrees(math.pi) if f.is_at_least(2024) else math.pi
        self.assertAlmostEqual(val, expected, places=4)


class TestInf(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_inf_returns_float_infinity(self):
        import math

        result = f.inf()
        self.assertEqual(result, math.inf)
        self.assertIsInstance(result, float)


class TestLogValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_log_natural(self):
        import math

        if not f.is_at_least(2024):
            self.skipTest("functions.log() requires Maya 2024+")
        n = Node.create("transform", name="probe")
        n.tx << math.e
        n.ty << f.log(n.tx)
        self.assertAlmostEqual(_eval(n.ty), 1.0, places=3)

    def test_log_base10(self):
        if not f.is_at_least(2024):
            self.skipTest("functions.log() requires Maya 2024+")
        n = Node.create("transform", name="probe")
        n.tx << 100.0
        n.ty << f.log(n.tx, base=10)
        self.assertAlmostEqual(_eval(n.ty), 2.0, places=3)

    def test_log_scalar_short_circuit(self):
        import math

        # When all args are real, returns Python float -- no node built.
        result = f.log(math.e)
        self.assertAlmostEqual(result, 1.0, places=4)


# --------------------------------------------------------------------- #
#  v3.C -- fuzzy float equality
# --------------------------------------------------------------------- #


class TestEqualValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_equal_within_eps(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 1.0001
        out = Node.create("transform", name="out")
        out.ty << f.equal(a.tx, b.tx, eps=0.001)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)

    def test_not_equal_outside_eps(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        b = Node.create("transform", name="b")
        b.tx << 2.0
        out = Node.create("transform", name="out")
        out.ty << f.equal(a.tx, b.tx, eps=0.001)
        self.assertAlmostEqual(_eval(out.ty), 0.0, places=4)

    def test_scalar_short_circuit(self):
        # All-numeric inputs short-circuit to Python bool.
        self.assertEqual(f.equal(1.0, 1.0001, eps=0.001), 1.0)
        self.assertEqual(f.equal(1.0, 2.0, eps=0.001), 0.0)


class TestComparison2024Value(MayaTestCase):
    """Confirms that Plug `>`, `<`, `==` produce the right value via the
    new 2024 native paths (greaterThan / lessThan / equal)."""

    TEST_START_NEW_SCENE = True

    def test_greater_than(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        b = Node.create("transform", name="b")
        b.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << (a.tx > b.tx)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)

    def test_less_than(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        b = Node.create("transform", name="b")
        b.tx << 3.0
        out = Node.create("transform", name="out")
        out.ty << (a.tx < b.tx)
        self.assertAlmostEqual(_eval(out.ty), 0.0, places=4)

    def test_equal_exact(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        b = Node.create("transform", name="b")
        b.tx << 5.0
        out = Node.create("transform", name="out")
        out.ty << (a.tx == b.tx)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)


# --------------------------------------------------------------------- #
#  v3.P named wrappers -- comparison/logical equivalence to operators
# --------------------------------------------------------------------- #


import maya.cmds as _cmds
from rig import functions as _f, Node as _Node, Plug as _Plug
from rig._tests._base import MayaTestCase as _MayaTestCase


class TestComparisonWrappers(_MayaTestCase):
    """v3.P: ``functions.greater_than(a, b)`` produces the same network
    as ``a > b``. Same for the other 5 comparison wrappers.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _cmds.polyCube(name="src")
        self._src = _Node("src")

    def _node_type(self, plug: _Plug) -> str:
        return _cmds.nodeType(str(plug).split(".")[0])

    def test_greater_than_matches_operator(self):
        op_result      = self._src.tx > 0
        wrapper_result = _f.greater_than(self._src.tx, 0)
        # Both should build the same kind of node.
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_less_than_matches_operator(self):
        op_result      = self._src.tx < 0
        wrapper_result = _f.less_than(self._src.tx, 0)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_not_equal_matches_operator(self):
        op_result      = self._src.tx != 0
        wrapper_result = _f.not_equal(self._src.tx, 0)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_greater_or_equal_matches_operator(self):
        op_result      = self._src.tx >= 0
        wrapper_result = _f.greater_or_equal(self._src.tx, 0)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_less_or_equal_matches_operator(self):
        op_result      = self._src.tx <= 0
        wrapper_result = _f.less_or_equal(self._src.tx, 0)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))


class TestLogicalWrappers(_MayaTestCase):
    """v3.P: ``functions.logical_and(a, b)`` produces the same network
    as ``a & b``. Same for ``logical_or`` and ``logical_xor``.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _cmds.polyCube(name="src")
        self._src = _Node("src")

    def _node_type(self, plug: _Plug) -> str:
        return _cmds.nodeType(str(plug).split(".")[0])

    def test_logical_and_matches_operator(self):
        op_result      = self._src.tx & 1
        wrapper_result = _f.logical_and(self._src.tx, 1)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_logical_or_matches_operator(self):
        op_result      = self._src.tx | 1
        wrapper_result = _f.logical_or(self._src.tx, 1)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_logical_xor_matches_operator(self):
        op_result      = self._src.tx ^ 1
        wrapper_result = _f.logical_xor(self._src.tx, 1)
        self.assertEqual(self._node_type(op_result), self._node_type(wrapper_result))

    def test_logical_and_compound_works(self):
        # Compound input must also work via the wrapper.
        result = _f.logical_and(self._src.t, 1)
        self.assertIsInstance(result, _Plug)


class TestComparisonWrappersCompound(_MayaTestCase):
    """v3.P: All comparison wrappers fan out per-channel for compound input."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _cmds.polyCube(name="src")
        self._src = _Node("src")

    def test_greater_than_compound(self):
        result = _f.greater_than(self._src.t, 0)
        self.assertIsInstance(result, _Plug)

    def test_not_equal_compound(self):
        result = _f.not_equal(self._src.t, 0)
        self.assertIsInstance(result, _Plug)

    def test_less_or_equal_compound(self):
        result = _f.less_or_equal(self._src.t, 0)
        self.assertIsInstance(result, _Plug)


# --------------------------------------------------------------------- #
#  v4.O -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


from rig import set_options


class TestFunctionsPublishedInterface(MayaTestCase):
    """Verify each wrapped composite function in functions.py creates a
    container with the expected published attributes when
    ``flatten_containers=False``. Spot-checks a representative sample
    rather than enumerating all 19 wrapped functions."""

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

    def test_clamp_compound_publishes(self):
        # Force composite path with compound input.
        a = Node.create("transform", name="a")
        a.t << [0.5, 0.5, 0.5]
        _f.clamp(a.t, 0.0, 1.0)
        self._assert_attrs("clamp1", ["input", "min", "max", "output"])

    def test_abs_compound_publishes(self):
        a = Node.create("transform", name="a")
        a.t << [-1, -2, -3]
        _f.abs(a.t)
        self._assert_attrs("abs1", ["input", "output"])

    def test_sign_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << -3.0
        _f.sign(a.tx)
        self._assert_attrs("sign1", ["input", "output"])

    def test_max_legacy_publishes(self):
        # Force legacy path with compound input.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        a.t << [1, 2, 3]
        b.t << [4, 5, 6]
        _f.max([a.t, b.t])
        self._assert_attrs("max1", ["input", "output"])

    def test_searchsorted_publishes(self):
        x = Node.create("transform", name="x")
        x.tx << 1.5
        _f.searchsorted([0.0, 1.0, 2.0, 3.0], x.tx)
        self._assert_attrs("searchsorted1", ["input"])

    def test_diff_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        a.ty << 2.0
        a.tz << 3.0
        _f.diff([a.tx, a.ty, a.tz])
        self._assert_attrs("diff1", ["input"])

    def test_cumsum_publishes(self):
        a = Node.create("transform", name="a")
        a.tx << 1.0
        a.ty << 2.0
        a.tz << 3.0
        _f.cumsum([a.tx, a.ty, a.tz])
        self._assert_attrs("cumsum1", ["input"])


# --------------------------------------------------------------------- #
#  Public API surface (__all__) -- P2 rescope
# --------------------------------------------------------------------- #


class TestFunctionsAll(MayaTestCase):
    """``functions`` is never star-imported (it shadows ~11 builtins), so its
    public surface is pinned by ``__all__``: it must enumerate exactly the
    math verbs, leak no stdlib / infra imports, and the vector ops relocated
    to :mod:`rig.vector` must be gone."""

    def test_all_is_the_expected_surface(self):
        expected = {
            "frame",
            "clamp",
            "abs",
            "int",
            "round",
            "floor",
            "ceil",
            "trunc",
            "sum",
            "avg",
            "max",
            "min",
            "exp",
            "sign",
            "sqrt",
            "pow",
            "log",
            "choice",
            "rev",
            "searchsorted",
            "all",
            "any",
            "argmin",
            "argmax",
            "diff",
            "cumsum",
            "pi",
            "inf",
            "equal",
            "not_equal",
            "greater_than",
            "less_than",
            "greater_or_equal",
            "less_or_equal",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
        }
        self.assertEqual(set(f.__all__), expected)

    def test_all_names_are_accessible(self):
        for name in f.__all__:
            self.assertTrue(hasattr(f, name), f"functions.{name} missing")

    def test_no_infra_leaks_in_all(self):
        for leaked in (
            "condition",
            "container",
            "memoize",
            "vectorize",
            "PlugList",
            "Attribute",
            "cmds",
            "math",
            "builtins",
        ):
            self.assertNotIn(leaked, f.__all__)

    def test_relocated_vector_ops_are_gone(self):
        for moved in ("mag", "dot", "cross", "unit", "dist"):
            self.assertNotIn(moved, f.__all__)
            self.assertFalse(
                hasattr(f, moved),
                f"functions.{moved} should be relocated to rig.vector",
            )


class TestLiteralVectorInputs(MayaTestCase):
    """Regression: component-wise math ops must accept FLAT LITERAL vec3
    inputs (not just compound Plugs). The Maya-2024+ native nodes are
    scalar-only; a raw ``[x, y, z]`` must route to the component-wise path
    (gated by ``_is_scalar_value``) instead of crashing on the scalar node
    with 'Cannot inject sequence of size 3 into scalar attribute'. Values are
    version-independent (a literal always takes the component path)."""

    TEST_START_NEW_SCENE = True

    def _vec(self, plug):
        val = cmds.getAttr(str(plug))
        return val[0] if isinstance(val, list) else val

    def _assert_vec(self, plug, expected):
        actual = self._vec(plug)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=4)

    def test_clamp_literal_vec3(self):
        self._assert_vec(f.clamp([0.5, -1, 2], [0, 0, 0], [1, 1, 1]), (0.5, 0.0, 1.0))

    def test_abs_literal_vec3(self):
        self._assert_vec(f.abs([-1, 2, -3]), (1.0, 2.0, 3.0))

    def test_round_literal_vec3(self):
        self._assert_vec(f.round([0.4, 1.6, 2.4]), (0.0, 2.0, 2.0))

    def test_floor_literal_vec3(self):
        self._assert_vec(f.floor([0.9, 1.2, -0.1]), (0.0, 1.0, -1.0))

    def test_ceil_literal_vec3(self):
        self._assert_vec(f.ceil([0.1, 1.2, -0.9]), (1.0, 2.0, 0.0))

    def test_trunc_literal_vec3(self):
        self._assert_vec(f.trunc([1.9, -1.9, 0.5]), (1.0, -1.0, 0.0))

    def test_equal_literal_vec3(self):
        self._assert_vec(f.equal([1, 2, 3], [1, 9, 3]), (1.0, 0.0, 1.0))

    def test_rev_literal_vec3(self):
        # ``reverse`` node HAS a compound ``input``; the literal must route
        # there (per-channel 1 - x), not to scalar ``inputX``.
        self._assert_vec(f.rev([0.25, 0.5, 0.75]), (0.75, 0.5, 0.25))