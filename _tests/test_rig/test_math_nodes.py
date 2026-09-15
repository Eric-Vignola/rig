"""Tests for ``rig._internal.math_nodes`` factories -- exercise the NodeOps
that build the actual Maya math networks.
"""

from maya import cmds
from rig import condition, constant, Node
from rig._internal.math_nodes import (
    _condition_op,
    _constant,
    _decompose_matrix,
    _matrix_inverse,
    _matrix_multiply,
    _multiply_divide_op,
    _plus_minus_average_op,
)
from rig._tests._base import MayaTestCase


class TestConstant(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_constant(self):
        plug = _constant(5.0, name="const1")
        # Plug points at <network>.value
        self.assertTrue(str(plug).endswith(".value"))
        self.assertAlmostEqual(cmds.getAttr(str(plug)), 5.0)

    def test_vector_constant(self):
        plug = _constant([1.0, 2.0, 3.0], name="vec1")
        # Compound parent -- children are valueX/Y/Z
        value = cmds.getAttr(str(plug))[0]
        self.assertEqual(list(value), [1.0, 2.0, 3.0])

    def test_constant_creates_network_node(self):
        before = cmds.ls(type="network") or []
        _constant(7.0, name="const_test")
        after = cmds.ls(type="network") or []
        self.assertEqual(len(after) - len(before), 1)

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            _constant(5, dtype="invalid_dtype")

    def test_memoized_constant_dedupes(self):
        before = cmds.ls(type="network") or []
        constant(3.14, name="memo_const")
        constant(3.14, name="memo_const")
        constant(3.14, name="memo_const")
        after = cmds.ls(type="network") or []
        # Only one new network node despite three calls (memoized).
        self.assertEqual(len(after) - len(before), 1)


class TestConstantMatrix(MayaTestCase):
    """Matrix-shaped inputs to ``constant()`` route through ``holdMatrix``
    instead of the ``network`` + ``value`` sidecar.
    """

    TEST_START_NEW_SCENE = True

    def test_4x4_numpy_constant_returns_holdmatrix(self):
        import numpy as np

        plug = constant(np.eye(4))
        # Output plug is <holdMatrix>.outMatrix
        self.assertEqual(cmds.nodeType(str(plug).split(".")[0]), "holdMatrix")
        self.assertTrue(str(plug).endswith(".outMatrix"))
        # Value round-trips through the inMatrix plug.
        flat = cmds.getAttr(f"{str(plug).split('.')[0]}.inMatrix")
        self.assertEqual(list(flat), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_3x3_constant_embeds_in_identity(self):
        import numpy as np

        plug = constant(np.eye(3) * 2.0)
        self.assertEqual(cmds.nodeType(str(plug).split(".")[0]), "holdMatrix")
        flat = cmds.getAttr(f"{str(plug).split('.')[0]}.inMatrix")
        self.assertEqual(list(flat), [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])

    def test_16_flat_list_constant_returns_holdmatrix(self):
        plug = constant(list(range(16)))
        self.assertEqual(cmds.nodeType(str(plug).split(".")[0]), "holdMatrix")
        flat = cmds.getAttr(f"{str(plug).split('.')[0]}.inMatrix")
        self.assertEqual(list(flat), list(range(16)))

    def test_nested_4x4_list_constant_returns_holdmatrix(self):
        nested = [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [4, 5, 6, 1],
        ]
        plug = constant(nested)
        self.assertEqual(cmds.nodeType(str(plug).split(".")[0]), "holdMatrix")
        flat = cmds.getAttr(f"{str(plug).split('.')[0]}.inMatrix")
        self.assertEqual(list(flat), [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 4, 5, 6, 1])

    def test_short_vector_still_uses_network(self):
        # Sizes 1-4 keep the existing network + value sidecar.
        plug = constant([0.1, 2.3, 0.0])
        self.assertEqual(cmds.nodeType(str(plug).split(".")[0]), "network")
        self.assertTrue(str(plug).endswith(".value"))

    def test_holdmatrix_is_gc_eligible(self):
        import numpy as np

        plug      = constant(np.eye(4), name="gc_test_const")
        node_name = str(plug).split(".")[0]
        # Rig-owned tag is set so cleanup() can find it.
        self.assertTrue(cmds.attributeQuery("__rig__", node=node_name, exists=True))
        self.assertTrue(cmds.getAttr(f"{node_name}.__rig__"))

    def test_matrix_constant_memoized(self):
        import numpy as np

        before = cmds.ls(type="holdMatrix") or []
        a      = constant(np.eye(4))
        b      = constant(np.eye(4))
        c      = constant(np.eye(4))
        after  = cmds.ls(type="holdMatrix") or []
        # All three calls share one holdMatrix node.
        self.assertEqual(len(after) - len(before), 1)
        self.assertEqual(str(a), str(b))
        self.assertEqual(str(a), str(c))


class TestPlusMinusAverage(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_add_creates_pma(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _plus_minus_average_op(a.tx, b.tx, operation=1)
        # Maya 2024+ uses native ``sum`` node; older uses ``plusMinusAverage``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"plusMinusAverage", "sum"},
        )

    def test_compound_add_uses_input3D(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _plus_minus_average_op(a.t, b.t, operation=1)
        # Output should be output3D (compound)
        self.assertTrue(str(result).endswith(".output3D"))


class TestPlusMinusAverageRegression(MayaTestCase):
    """Regression: ``a.tx + b.tz`` must produce a plusMinusAverage with
    BOTH operand inputs wired to ``input1D[0]`` and ``input1D[1]``,
    NOT an empty wrapper.

    Triggered by the multi-root auto-index fix (``_inject_value`` now
    auto-indexes ``node.input1D << x`` to the next free slot).
    """

    TEST_START_NEW_SCENE = True

    def test_scalar_add_connects_both_inputs(self):
        # Verify that BOTH operands land in distinct slots. The exact node
        # type and slot names differ between legacy (plusMinusAverage)
        # and Maya 2024+ (sum), so we just check that the resulting
        # network produces the correct value.
        a = Node.create("transform", name="a")
        a.tx << 7.0
        b = Node.create("transform", name="b")
        b.tz << 11.0
        out = Node.create("transform", name="out")
        out.ty << (a.tx + b.tz)
        self.assertAlmostEqual(cmds.getAttr("out.ty"), 18.0, places=4)

    def test_scalar_subtract_connects_both_inputs(self):
        # Likewise: verify by VALUE rather than by node introspection,
        # since 2024+ uses ``subtract`` (input1/input2) instead of
        # ``plusMinusAverage`` (input1D[0/1] + operation=2).
        a = Node.create("transform", name="a")
        a.tx << 10.0
        b = Node.create("transform", name="b")
        b.tz << 3.0
        out = Node.create("transform", name="out")
        out.ty << (a.tx - b.tz)
        self.assertAlmostEqual(cmds.getAttr("out.ty"), 7.0, places=4)

    def test_compound_plus_literal_vector(self):
        # Regression (compound Plug + literal vec3): must sum per-channel,
        # matching ``compound + compound``. Pre-fix this crashed with an
        # InjectionError on ``add1.input3D[0].input3Dx`` -- a bare
        # ``input3D << [x, y, z]`` spread one scalar per index instead of one
        # vec3 into a single index.
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        out = Node.create("transform", name="out")
        out.t << (n.t + [4.0, 5.0, 6.0])
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [5.0, 7.0, 9.0])

    def test_compound_minus_literal_vector(self):
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        out = Node.create("transform", name="out")
        out.t << (n.t - [1.0, 2.0, 3.0])
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [0.0, 0.0, 0.0])

    def test_literal_vector_plus_compound_operand_order(self):
        # __radd__: literal-first must equal plug-first (no index spread).
        # Pre-fix this silently returned [16, 17, 18] (literal spread across
        # indices 0/1/2, plug appended at index 3).
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        out = Node.create("transform", name="out")
        out.t << ([4.0, 5.0, 6.0] + n.t)
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [5.0, 7.0, 9.0])

    def test_literal_vector_minus_compound_operand_order(self):
        # __rsub__: literal-first subtract -> [10,10,10] - [1,2,3] = [9,8,7].
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        out = Node.create("transform", name="out")
        out.t << ([10.0, 10.0, 10.0] - n.t)
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [9.0, 8.0, 7.0])

    def test_compound_plus_scalar_literal_still_broadcasts(self):
        # Guard: compound + scalar literal must keep broadcasting (5 ->
        # (5,5,5)) via the unchanged auto-append ``<<`` path.
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        out = Node.create("transform", name="out")
        out.t << (n.t + 5.0)
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [6.0, 7.0, 8.0])

    def test_compound_plus_scalar_plug_still_broadcasts(self):
        # Guard: compound + scalar PLUG must still broadcast. A scalar Plug is
        # a str-subclass so ``_is_sequence`` is False and it keeps the
        # auto-append branch (NOT the new literal-vector index path).
        n = Node.create("transform", name="n")
        n.t << [1.0, 2.0, 3.0]
        m = Node.create("transform", name="m")
        m.tx << 100.0
        out = Node.create("transform", name="out")
        out.t << (n.t + m.tx)
        self.assertEqual(list(cmds.getAttr("out.t")[0]), [101.0, 102.0, 103.0])


class TestMultiplyDivide(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_mul(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _multiply_divide_op(a.tx, b.tx, operation=1)
        # Maya 2024+ uses native ``multiply`` node; older uses ``multiplyDivide``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"multiplyDivide", "multiply"},
        )

    def test_compound_mul(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _multiply_divide_op(a.t, b.t, operation=1)
        self.assertTrue(str(result).endswith(".output"))


class TestDecomposeMatrix(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_decompose_creates_node(self):
        a      = Node.create("transform", name="a")
        result = _decompose_matrix(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "decomposeMatrix")


class TestMatrixOps(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_matrix_multiply_creates_multmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _matrix_multiply(a.matrix, b.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "multMatrix")

    def test_matrix_inverse_creates_node(self):
        a      = Node.create("transform", name="a")
        result = _matrix_inverse(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "inverseMatrix")

    def test_point_matrix_mult(self):
        # Two-token call: one matrix + one vector => vector x matrix fast path.
        # Maya 2024+ uses native ``multiplyPointByMatrix`` (no ``local`` arg
        # => point semantics, full transform with translation).
        # Pre-2024 uses ``pointMatrixMult``; Maya 2026+ aliases it to
        # ``pointMatrixMultDL`` (deprecation warning).
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _matrix_multiply(a.matrix, b.t)
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"pointMatrixMult", "pointMatrixMultDL", "multiplyPointByMatrix"},
        )

    def test_point_matrix_mult_local(self):
        # ``local=True`` => vector semantics (rotation/scale only, no translation).
        # Maya 2024+ => ``multiplyVectorByMatrix``; pre-2024 =>
        # ``pointMatrixMult`` with ``vectorMultiply=True``.
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _matrix_multiply(a.matrix, b.t, local=True)
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"pointMatrixMult", "pointMatrixMultDL", "multiplyVectorByMatrix"},
        )


class TestConditionOp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_condition(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = _condition_op(a.tx, ">", b.tx)
        # Maya 2024+ uses native ``greaterThan``; older uses ``condition``.
        self.assertIn(
            cmds.nodeType(str(result).split(".")[0]),
            {"condition", "greaterThan"},
        )

    def test_invalid_op_raises(self):
        a = Node.create("transform", name="a")
        with self.assertRaises(ValueError):
            _condition_op(a.tx, "%%", 5)


class TestConditionPublic(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_scalar_condition_short_circuit_true(self):
        a = Node.create("transform", name="a")
        # Number condition_op short-circuits to if_true
        result = condition(1, a.tx, 0)
        # Returned the if_true plug directly.
        self.assertEqual(str(result), str(a.tx))

    def test_scalar_condition_short_circuit_false(self):
        a      = Node.create("transform", name="a")
        result = condition(0, a.tx, 5)
        self.assertEqual(result, 5)