"""A plain list where a function takes one number per call: one call per number, as a List.

``vectorize(rows=("weight",))`` names the parameters that take one number (a blend weight); a plain
list, tuple, range or 1-D array given there is read as a List (`rig._internal.memoize.rows_as_lists`),
by the function and at its `operands` entry, before rig.fuse() reads it. Before, the list went in as a
value: blend's weight became a vector whose first channel drove the one weight (every row the first
weight), vector slerp's weighted each channel by its own entry. A plain list anywhere else is still a
value (a vector, a matrix, per-channel weights). And ``List[...]`` (a type alias, ``list[int]``'s
syntax) is refused with the ``List([...])`` it meant, before anything is built.
"""
import math

import numpy as np
from maya import cmds

import rig
from rig import List, Node
from rig._internal.memoize import row_slots, rows_as_lists, vectorize
from rig._tests._base import MayaTestCase
from rig.spec import Quat

W = [0.25, 0.5, 0.75]


def _value(value):
    """A plug's, a List's or a folded literal's value, rounded"""
    if not isinstance(value, (np.ndarray, list, tuple, float, int)) or isinstance(value, List):
        value = value >> None
    return np.round(np.asarray(value, dtype=float), 4)


class TestRowsAsLists(MayaTestCase):
    """The rule itself, on plain Python functions"""

    def test_a_plain_sequence_in_a_row_parameter_is_a_list(self):
        def f(a, b, weight=0.5):
            return a

        slots = row_slots(f, ("weight",))
        self.assertEqual(slots, {"weight": 2})
        for given in ([0.25, 0.5], (0.25, 0.5), range(2), np.array([0.25, 0.5])):
            with self.subTest(given=type(given).__name__):
                args, kwargs = rows_as_lists(slots, (1, 2, given), {})
                self.assertIsInstance(args[2], List)
                self.assertEqual([float(x) for x in args[2]], [float(x) for x in given])
                args, kwargs = rows_as_lists(slots, (1, 2), {"weight": given})
                self.assertIsInstance(kwargs["weight"], List)

    def test_anything_else_is_left_as_it_is(self):
        def f(a, b, weight=0.5):
            return a

        slots = row_slots(f, ("weight",))
        listed = List([0.25, 0.5])
        for given in (0.5, listed, np.float64(0.5), np.array(0.5)):
            with self.subTest(given=repr(given)):
                args, _ = rows_as_lists(slots, (1, 2, given), {})
                self.assertIs(args[2], given)
        # the other parameters keep their plain lists: values
        args, _ = rows_as_lists(slots, ([1, 2, 3], (4, 5, 6), 0.5), {})
        self.assertEqual(args[:2], ([1, 2, 3], (4, 5, 6)))

    def test_a_parameter_it_does_not_have_raises(self):
        with self.assertRaisesRegex(TypeError, "has no parameter 'nope'"):
            vectorize(rows=("nope",))(lambda a, weight=0.5: a)

    def test_vectorize_broadcasts_a_row_parameter(self):
        @vectorize(rows=("w",))
        def f(x, w=0.5):
            return x * w

        self.assertEqual([float(v) for v in f(2, [0.25, 0.5, 0.75])], [0.5, 1.0, 1.5])
        self.assertEqual([float(v) for v in f(2, w=(0.25, 0.5))], [0.5, 1.0])
        self.assertEqual(f(2, 0.5), 1.0)                    # a number: one call, as before


class _Poles(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.a = Node.create("transform", name="a")
        self.b = Node.create("transform", name="b")
        cmds.setAttr("a.t", 2, 0, 0)
        cmds.setAttr("a.r", 0, 30, 0)
        cmds.setAttr("b.t", 10, 4, 0)
        cmds.setAttr("b.r", 0, 120, 0)
        cmds.setAttr("b.s", 2, 2, 2)
        self.p = Node.create("transform", name="p")
        self.q = Node.create("transform", name="q")
        self.p.t << [10, 0, 0]
        self.q.t << [0, 10, 0]

    def assertRows(self, plain, one):
        """``plain`` (the call with W) is a List of three, each the call with its weight (``one(w)``)"""
        self.assertIsInstance(plain, List)
        self.assertEqual(len(plain), 3)
        for row, w in zip(plain, W):
            np.testing.assert_allclose(_value(row), _value(one(w)), atol=1e-4)


class TestWeights(_Poles):
    def test_the_matrix_verbs(self):
        a, b = self.a, self.b
        for verb in (rig.matrix.blend, rig.matrix.lerp, rig.matrix.slerp):
            with self.subTest(verb=verb.__name__):
                self.assertRows(verb(a.wm, b.wm, W), lambda w: verb(a.wm, b.wm, w))
                for given in (tuple(W), np.array(W), List(W)):
                    np.testing.assert_allclose(_value(List(verb(a.wm, b.wm, given))),
                                               _value(List(verb(a.wm, b.wm, W))), atol=1e-4)
                self.assertRows(verb(a.wm, b.wm, weight=W), lambda w: verb(a.wm, b.wm, w))
        self.assertRows(rig.matrix.pow(b.wm, W), lambda w: rig.matrix.pow(b.wm, w))

    def test_blend_lands_three_transforms_between_the_poles(self):
        # the user's example: before, every row was the first weight's in-between
        mids = List([Node.create("transform", name=f"mid{i}") for i in (1, 2, 3)])
        mids << rig.blend(self.a.wm, self.b.wm, W)
        got = [(round(cmds.xform(f"mid{i}", q=True, ws=True, t=True)[0], 3), round(cmds.getAttr(f"mid{i}.ry"), 3),
                round(cmds.getAttr(f"mid{i}.sx"), 3)) for i in (1, 2, 3)]
        self.assertEqual(got, [(4.0, 52.5, 1.25), (6.0, 75.0, 1.5), (8.0, 97.5, 1.75)])
        self.assertEqual(len(cmds.ls(type="blendMatrix")), 3)

    def test_vector_slerp_turns_by_each_weight(self):
        # the user's expression: before, one vector slerped with a weight per channel
        out = rig.vector.slerp(self.p.t, self.q.t, W)
        self.assertIsInstance(out, List)
        expected = [[10 * math.cos(math.radians(90 * w)), 10 * math.sin(math.radians(90 * w)), 0.0] for w in W]
        np.testing.assert_allclose(_value(out), np.round(expected, 4), atol=1e-4)
        self.assertRows(rig.slerp(self.p.t, self.q.t, W), lambda w: rig.vector.slerp(self.p.t, self.q.t, w))

    def test_quaternion_and_euler(self):
        self.a << Quat("qa") << [0, 0, 0, 1]
        self.b << Quat("qb") << [0, 0.7071068, 0, 0.7071068]
        qa, qb = self.a.qa, self.b.qb
        self.assertRows(rig.quaternion.slerp(qa, qb, W), lambda w: rig.quaternion.slerp(qa, qb, w))
        self.assertRows(rig.quaternion.pow(qb, W), lambda w: rig.quaternion.pow(qb, w))
        self.assertRows(rig.euler.slerp(self.a.r, self.b.r, W), lambda w: rig.euler.slerp(self.a.r, self.b.r, w))

    def test_in_a_fuse_block(self):
        with rig.fuse():
            out = rig.vector.slerp(self.p.t, self.q.t, W)
        self.assertIsInstance(out, List)
        expected = [[10 * math.cos(math.radians(90 * w)), 10 * math.sin(math.radians(90 * w)), 0.0] for w in W]
        np.testing.assert_allclose(_value(out), np.round(expected, 4), atol=1e-4)


    def test_vector_lerp_and_elerp(self):
        # before: lerp gave every channel the first weight ([7.5, 2.5, 0]); elerp a weight per channel
        p, q = self.p.t, self.q.t
        out = rig.vector.lerp(p, q, W)
        np.testing.assert_allclose(_value(out), [[7.5, 2.5, 0.0], [5.0, 5.0, 0.0], [2.5, 7.5, 0.0]], atol=1e-4)
        self.assertRows(rig.lerp(p, q, W), lambda w: rig.vector.lerp(p, q, w))
        self.assertRows(rig.vector.elerp(p + 1, q + 1, W), lambda w: rig.vector.elerp(p + 1, q + 1, w))
        np.testing.assert_allclose(_value(rig.vector.lerp([10, 0, 0], [0, 10, 0], W)),
                                   [[7.5, 2.5, 0.0], [5.0, 5.0, 0.0], [2.5, 7.5, 0.0]], atol=1e-4)


    def test_another_plain_list_is_one_value_for_every_row(self):
        # in a function with row parameters, rows come from a List or a row parameter only
        out = rig.vector.lerp(List([self.p.t, self.q.t]), [0, 0, 10], 0.5)      # before: lengths {2, 3}
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0]], atol=1e-4)
        out = rig.vector.lerp(List([self.p.t, self.q.t, self.a.t]), [0, 0, 10], 0.5)   # before: rows 0, 0, 10
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0], [1.0, 0.0, 5.0]], atol=1e-4)


class TestPlainListsStayValues(_Poles):
    def test_vector_literals(self):
        out = rig.vector.cross(self.p.t, [0, 1, 0])
        self.assertNotIsInstance(out, List)
        np.testing.assert_allclose(_value(out), [0.0, 0.0, 10.0], atol=1e-4)
        np.testing.assert_allclose(_value(rig.vector.slerp([0, 0, 1], [1, 0, 0], 0.5)), [0.7071, 0.0, 0.7071],
                                   atol=1e-4)
        out = rig.vector.lerp(self.p.t, [0, 10, 0], 0.5)
        self.assertNotIsInstance(out, List)
        np.testing.assert_allclose(_value(out), [5.0, 5.0, 0.0], atol=1e-4)

    def test_a_plug_weight_is_one_value(self):
        self.p << rig.spec.Float("w") << 0.25
        out = rig.vector.slerp(self.p.t, self.q.t, self.p.w)
        self.assertNotIsInstance(out, List)


class TestListAlias(_Poles):
    def test_a_function_refuses_it_before_building(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"^rig\.vector\.slerp\(\): List\[\.\.\.\] makes a type alias, "
                                               r"not a List: build the List with List\(\[\.\.\.\]\)$"):
            rig.vector.slerp(List[self.p.t, self.p.t], self.q.t, 0.5)
        self.assertEqual(set(cmds.ls()) - before, set())

    def test_lshift_names_it(self):
        with self.assertRaisesRegex(TypeError, r"List\[\.\.\.\] makes a type alias.*into Plug\(p\.translateX\)"):
            self.p.t << List[self.q.t, self.q.t]
        with self.assertRaisesRegex(TypeError, r"List\[\.\.\.\] makes a type alias"):
            List([self.p, self.q]).tx << List[1, 2]

    def test_a_type_hint_still_works(self):
        alias = List[int]
        self.assertIs(alias.__origin__, List)
