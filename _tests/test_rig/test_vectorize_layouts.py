"""Layouts: what one call of a vectorized function takes, so a plain list is one value or rows by what
it holds (`rig._internal.layout`).

``@vectorize("(3),(3),()->(3)")`` names each parameter's core: ``vector.slerp`` takes two vectors and a
weight. A plain list, tuple, range or array is one value when it fits the core as a whole
(``[1, 0, 0]``), and rows when each of its elements does (``[a.t, b.t]``, ``[0.25, 0.5]`` for the
weight); a ``List`` is rows. Rows combine with single values and with rows of 1 or of the same length,
anything else raising before a node is built. rig.fuse() reads the same rows (the `operands` entry
makes them a List, `fuse._broadcast` walks them). ``List[...]`` (a type alias, ``list[int]``'s syntax)
is refused with the ``List([...])`` it meant, before anything is built.
"""
import math
import unittest

import numpy as np
from maya import cmds

import rig
from rig import List, Node
from rig._internal.layout import Layout, LayoutError
from rig._internal.memoize import vectorize
from rig._tests._base import MayaTestCase
from rig.spec import Quat

W = [0.25, 0.5, 0.75]


def _value(value):
    """A plug's, a List's or a folded literal's value, rounded"""
    if not isinstance(value, (np.ndarray, list, tuple, float, int)) or isinstance(value, List):
        value = value >> None
    return np.round(np.asarray(value, dtype=float), 4)


def _bound(text, func):
    """``func``'s layout ``text``, bound (what `vectorize` reads a call with)"""
    return vectorize(text)(func)._rig_layout


class TestParse(unittest.TestCase):
    """The layout strings rig reads, and the ones it refuses (at decoration)"""

    def test_the_dialect(self):
        cases = {
            "(3),(3),()->(3)": ["(3)", "(3)", "()"],
            "(n?), (n?), () -> (n?)": ["(n?)", "(n?)", "()"],
            "(4,4),()->(4,4)": ["(4,4)", "()"],
            "(3)|(4,4)->()": ["(3)|(4,4)"],
            "*(4,4)->(4,4)": ["*(4,4)"],
            "[k](n?),()->(n?)": ["[k](n?)", "()"],
            "(),[k],[k]->()": ["()", "[k]", "[k]"],
            "(3?),(3?)->(3)": ["(3?)", "(3?)"],
            "->()": [],
        }
        for text, slots in cases.items():
            with self.subTest(text=text):
                layout = Layout(text)
                self.assertEqual([s.text for s in layout.slots], slots)
                self.assertNotIn(" ", layout.text)
        self.assertEqual([c.text for c in Layout("(4)->(3),()").outputs], ["(3)", "()"])
        self.assertEqual([c.text for c in Layout("*(4,4)|(3)->(4,4)|(3)").outputs], ["(4,4)|(3)"])
        self.assertEqual([c.text for c in Layout("(4,4)->node").outputs], ["node"])  # decompose's node
        self.assertEqual(Layout("(n?),(3)->(n?)").names, {"n"})

    def test_what_it_refuses(self):
        for text, words in (
            ("(3", "expected"),
            ("(3)->", "expected"),
            ("(3),(3)", "expected '->'"),
            ("[k?]->()", "a list size cannot be optional"),
            ("(0)->()", "at least 1"),
            ("(m)->(n)", "the result's 'n' is no parameter's"),
            ("*(3),(3)->()", "only the last parameter can be variadic"),
            ("(3)$->()", "cannot read"),
            ("(3)->(3))", "unexpected"),
            ("(3)->*(3)", "a result cannot be variadic"),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(LayoutError, words):
                    Layout(text)
        with self.assertRaisesRegex(TypeError, "a layout is a str"):
            Layout(3)

    def test_the_signature_must_match(self):
        def f(a, b):
            return a

        with self.assertRaisesRegex(LayoutError, r"has 3 parameters, f\(\) takes 2"):
            vectorize("(),(),()->()")(f)
        with self.assertRaisesRegex(LayoutError, r"needs .*\*args"):
            vectorize("*(4,4)->(4,4)")(lambda a, b: a)
        with self.assertRaisesRegex(TypeError, "favor_index does not go with a layout"):
            vectorize("(3)->(3)", favor_index=0)
        # parameters past the layout are settings
        vectorize("(3)->()")(lambda vector, normalize=False: vector)

    def test_the_docstring_gains_it(self):
        def f(a, b, weight=0.5):
            """``f(a, b, weight)`` -- a test."""
            return a

        doc = vectorize("(3),(3),()->(3)")(f).__doc__
        self.assertTrue(doc.startswith("``f(a, b, weight)`` -- a test."))
        self.assertTrue(doc.rstrip().endswith("Layout ``(3),(3),()->(3)``: a (3), b (3), weight ()."))
        self.assertEqual(rig.vector.slerp.__doc__.rstrip().splitlines()[-1].strip(),
                         "Layout ``(3),(3),()->(3)``: input1 (3), input2 (3), weight ().")


class TestRowsOfWeights(unittest.TestCase):
    """A () weight given a plain sequence: rows, whatever kind of sequence (pure Python)"""

    def test_a_plain_sequence_of_weights_is_rows(self):
        @vectorize("(),()->()")
        def f(x, w=0.5):
            return x * w

        for given in ([0.25, 0.5, 0.75], (0.25, 0.5, 0.75), np.array([0.25, 0.5, 0.75]), List(W)):
            with self.subTest(given=type(given).__name__):
                self.assertEqual([float(v) for v in f(2, given)], [0.5, 1.0, 1.5])
                self.assertEqual([float(v) for v in f(2, w=given)], [0.5, 1.0, 1.5])
        self.assertEqual([float(v) for v in f(2, range(3))], [0.0, 2.0, 4.0])
        self.assertEqual(f(2, 0.5), 1.0)                         # a number: one call, as before
        self.assertEqual(f(2, np.float64(0.5)), 1.0)
        self.assertEqual(f(2, np.array(0.5)), 1.0)               # a 0-d array is one value

    def test_a_setting_is_never_rows(self):
        seen = []

        @vectorize("()->()")
        def f(x, options=None):
            seen.append(options)
            return x

        self.assertEqual(len(f([1, 2], options=[7, 8, 9])), 2)
        self.assertEqual(seen, [[7, 8, 9], [7, 8, 9]])


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


class TestClassify(_Poles):
    """What a call broadcasts: None for one call, else how many rows and which arguments give them"""

    def setUp(self):
        super().setUp()
        self.vector = _bound("(3),(3),()->(3)", lambda input1, input2, weight=0.5: None)
        self.lerp = _bound("(n?),(n?),()->(n?)", lambda input1, input2, weight=0.5: None)
        self.matrix = _bound("(4,4),(4,4),()->(4,4)", lambda input1, input2, weight=0.5: None)
        self.clamp = _bound("(),(),()->()", lambda token, low=0, high=1: None)

    def rows(self, bound, *args, **kwargs):
        plan = bound.classify(args, kwargs)
        return None if plan is None else (plan.count, sorted(map(str, plan.rows)))

    def test_one_value_or_rows_by_what_a_list_holds(self):
        a, b, p, q = self.a, self.b, self.p, self.q
        v = self.vector
        self.assertIsNone(self.rows(v, a.t, b.t, 0.5))                         # nothing to read
        self.assertIsNone(self.rows(v, [1, 0, 0], b.t, 0.5))                   # three numbers: a vector
        self.assertIsNone(self.rows(v, np.array([1.0, 0, 0]), (0, 1, 0), 0.5))
        self.assertIsNone(self.rows(v, [a.tx, a.ty, a.tz], b.t, 0.5))          # three scalar plugs
        self.assertEqual(self.rows(v, [a.t, b.t], p.t, 0.5), (2, ["0"]))      # a list of vectors: rows
        self.assertEqual(self.rows(v, [[1, 0, 0], [0, 1, 0]], p.t, 0.5), (2, ["0"]))
        self.assertEqual(self.rows(v, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), p.t, 0.5), (3, ["0"]))
        self.assertEqual(self.rows(v, [a.t], [p.t, q.t], [0.25, 0.75]), (2, ["0", "1", "2"]))
        self.assertEqual(self.rows(v, a.t, b.t, W), (3, ["2"]))               # the weight's core is ()
        self.assertEqual(self.rows(v, a.t, b.t, weight=W), (3, ["weight"]))
        self.assertEqual(self.rows(v, List([a.t, b.t]), [0, 0, 10], 0.5), (2, ["0"]))   # one target, 2 rows

    def test_rows_must_broadcast(self):
        a, b, p, q = self.a, self.b, self.p, self.q
        with self.assertRaisesRegex(ValueError, r"^rig\..*\(\): input1 has 3 rows and input2 has 2 rows; each must "
                                                r"have 1 row or 3 \(layout \(3\),\(3\),\(\)->\(3\)\)$"):
            self.rows(self.vector, [a.t, b.t, p.t], [q.t, a.t], 0.5)
        with self.assertRaisesRegex(ValueError, r"input1 has 2 rows and weight has 3 rows"):
            self.rows(self.vector, [a.t, b.t], [q.t], W)
        with self.assertRaisesRegex(ValueError, r"input1 has 3 rows and input2 has 2 rows"):
            self.rows(self.vector, List([a.t, b.t, p.t]), List([q.t, a.t]), 0.5)

    def test_what_fits_neither_way_is_refused(self):
        a, b = self.a, self.b
        with self.assertRaisesRegex(TypeError, r"input1 takes a 3-channel vector \(3\), or rows of them .*; got a "
                                               r"list of 2 \(a scalar plug\)"):
            self.rows(self.vector, [a.tx, a.ty], b.t, 0.5)
        with self.assertRaisesRegex(TypeError, r"input2 takes a 3-channel vector \(3\).*got a scalar plug"):
            self.rows(self.vector, [a.t, b.t], a.tx, 0.5)              # a single value is checked too
        with self.assertRaisesRegex(TypeError, r"input1 takes a 3-channel vector \(3\).*got a List of 3"):
            self.rows(self.vector, List([1, 0, 0]), b.t, 0.5)           # a List is rows: of three numbers
        with self.assertRaisesRegex(TypeError, "rows of rows|takes"):
            self.rows(self.vector, [[a.t, b.t], [a.t, b.t]], b.t, 0.5)  # one level only

    def test_every_call_is_checked(self):
        # decision 1A: a wrong shape raises with no rows too (it used to be spread over channels)
        a, b = self.a, self.b
        with self.assertRaisesRegex(TypeError, r"input1 takes a 3-channel vector \(3\).*got a scalar plug"):
            self.rows(self.vector, a.tx, b.t, 0.5)
        with self.assertRaisesRegex(TypeError, r"input1 takes a matrix \(4,4\).*got a 3-channel plug"):
            self.rows(self.matrix, a.t, b.wm, 0.5)
        self.assertIsNone(self.rows(self.vector, a.t, b.t, a.tx))         # a () takes any one value
        self.assertIsNone(self.rows(self.lerp, a.tx, b.t, 0.5))           # a scalar meets n

    def test_n_is_one_size(self):
        a, b = self.a, self.b
        cmds.addAttr("p", ln="uv", at="double2")
        cmds.addAttr("p", ln="uvX", at="double", p="uv")
        cmds.addAttr("p", ln="uvY", at="double", p="uv")
        uv = self.p.uv
        self.assertIsNone(self.rows(self.lerp, a.tx, b.tx, 0.5))
        self.assertIsNone(self.rows(self.lerp, [a.tx, a.ty, a.tz], b.t, 0.5))
        self.assertEqual(self.rows(self.lerp, List([a.t, b.t]), 0, 0.5), (2, ["0"]))   # a scalar meets n
        self.assertEqual(self.rows(self.lerp, List([uv, uv]), [0, 1], 0.5), (2, ["0"]))
        with self.assertRaisesRegex(ValueError, r"input1 has 3 for 'n' and input2 has 2; they must agree"):
            self.rows(self.lerp, List([a.t]), uv, 0.5)
        with self.assertRaisesRegex(ValueError, r"input1 has 3 for 'n' and input2 has 2"):
            self.rows(self.lerp, a.t, [0, 0], 0.5)
        with self.assertRaisesRegex(ValueError, r"input1 has 3 for 'n' and input2 has 2"):
            self.rows(self.lerp, a.t, uv, 0.5)                              # no rows: checked too (1A)

    def test_matrices(self):
        a, b = self.a, self.b
        m = self.matrix
        flat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for one in (np.eye(4), np.eye(4).tolist(), flat, np.eye(3).tolist(), np.eye(4).ravel()):
            with self.subTest(one=type(one).__name__):
                self.assertIsNone(self.rows(m, a.wm, one, 0.5))
        self.assertEqual(self.rows(m, [a.wm, b.wm], np.eye(4), 0.5), (2, ["0"]))
        self.assertEqual(self.rows(m, np.zeros((3, 4, 4)), b.wm, 0.5), (3, ["0"]))
        self.assertEqual(self.rows(m, [flat, flat], b.wm, 0.5), (2, ["0"]))
        with self.assertRaisesRegex(TypeError, r"input2 takes a matrix \(4,4\).*got a 3-channel plug"):
            self.rows(m, [a.wm, b.wm], a.t, 0.5)

    def test_channels_in_a_per_channel_function(self):
        a, b, p, q = self.a, self.b, self.p, self.q
        c = self.clamp
        self.assertIsNone(self.rows(c, a.t, [0, 0, 0], [1, 1, 3]))              # lines up with a.t's channels
        self.assertEqual(self.rows(c, [a.tx, b.tx, p.tx, q.tx, a.ty], 0, 1), (5, ["0"]))
        self.assertEqual(self.rows(c, [a.tx, a.ty, a.tz], 0, 1), (3, ["0"]))   # rows: nothing to line up with
        self.assertEqual(self.rows(c, [a.t, b.t], 0, 1), (2, ["0"]))           # 2 rows of vectors
        self.assertEqual(self.rows(c, [a.t, b.t], [0, 0, 0], 1), (2, ["0"]))   # low: per channel, every row
        self.assertEqual(self.rows(c, List([a.t, b.t, p.t]), [0, 0, 0], 1), (3, ["0"]))
        with self.assertRaisesRegex(ValueError, r"token has 2 rows and low has 3 rows"):
            self.rows(c, [a.tx, b.tx], [0, 0, 0], 1)

    def test_lists_taken_whole(self):
        a, b, p, q = self.a, self.b, self.p, self.q
        total = _bound("[k](n?)->(n?)", lambda tokens: None)
        self.assertIsNone(self.rows(total, [a.tx, b.tx]))
        self.assertIsNone(self.rows(total, List([a.t, b.t])))                 # a List of vectors fills [k]
        self.assertIsNone(self.rows(total, [[a.tx, b.tx], [p.tx, q.tx]]))     # a list of two 2-vectors
        self.assertEqual(self.rows(total, List([[a.tx, b.tx], [p.tx, q.tx]])), (2, ["0"]))   # a List: rows
        self.assertIsNone(self.rows(total, [a.t, 1]))                          # a scalar meets a vector
        with self.assertRaisesRegex(TypeError, r"tokens takes a list of .*\[k\]\(n\?\)"):
            self.rows(total, [a.t, [0, 1]])                                     # 3 channels and 2
        interp = _bound("(),[k],[k]->()", lambda x, xp, fp: None)
        self.assertIsNone(self.rows(interp, a.tx, [0, 1, 4], [0, 10, 0]))
        self.assertEqual(self.rows(interp, [a.tx, b.tx], [0, 1, 4], [0, 10, 0]), (2, ["0"]))
        with self.assertRaisesRegex(ValueError, r"xp has 3 for 'k' and fp has 2"):
            self.rows(interp, a.tx, [0, 1, 4], [0, 10])
        with self.assertRaisesRegex(TypeError, r"xp takes a list of values \[k\]"):
            self.rows(interp, [a.tx, b.tx], 1, [0, 10])


class TestTheTable(_Poles):
    """The table the layouts were made for: rig.vector.slerp, one value or rows (and how many nodes)"""

    ONE_ROW = ("slerp(p, q, 0.25)", "slerp([p], q, 0.25)", "slerp(p, [q], 0.25)", "slerp(p, q, [0.25])",
               "slerp([p], [q], 0.25)", "slerp([p], [q], [0.25])", "slerp([px, py, pz], q, 0.25)")
    TWO_ROWS = ("slerp(p, q, [0.25, 0.75])", "slerp([p, a], q, 0.25)", "slerp(p, [q, b], 0.25)",
                "slerp([p, a], q, [0.25, 0.75])", "slerp(p, [q, b], [0.25, 0.75])",
                "slerp([p, a], [q, b], [0.25, 0.75])")

    def count(self, code):
        """``code``'s result and the slerps it made, in a fresh scene (memoize would share them)"""
        self.setUp()                     # a new scene, new poles
        names = {"slerp": rig.vector.slerp, "p": self.p.t, "q": self.q.t, "a": self.a.t, "b": self.b.t,
                 "px": self.p.tx, "py": self.p.ty, "pz": self.p.tz}
        before = len(cmds.ls(type="angleBetween"))
        out = eval(code, names)
        return out, len(cmds.ls(type="angleBetween")) - before

    def test_one_row(self):
        for code in self.ONE_ROW:
            with self.subTest(code=code):
                out, made = self.count(code)
                self.assertEqual(made, 1)
                np.testing.assert_allclose(_value(out), [9.2388, 3.8268, 0.0], atol=1e-4)

    def test_two_rows(self):
        for code in self.TWO_ROWS:
            with self.subTest(code=code):
                out, made = self.count(code)
                self.assertIsInstance(out, List)
                self.assertEqual((len(out), made), (2, 2))

    def test_what_does_not_broadcast_builds_nothing(self):
        p, q, a, b = self.p.t, self.q.t, self.a.t, self.b.t
        slerp = rig.vector.slerp
        before = set(cmds.ls())
        for call in (lambda: slerp([p, q, a], [b, p], 0.25), lambda: slerp([p, q], a, [0.25, 0.5, 0.75]),
                     lambda: slerp([p, q], [a], [0.25, 0.5, 0.75])):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, r"^rig\.vector\.slerp\(\): .* has \d rows"):
                    call()
        self.assertEqual(set(cmds.ls()) - before, set())

    def test_the_matrix_and_quaternion_verbs_read_rows_the_same_way(self):
        a, b, p = self.a, self.b, self.p
        out = rig.matrix.blend([a.wm, b.wm], p.wm, 0.5)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(rig.matrix.blend(a.wm, [b.wm, p.wm], W[:2])), 2)
        self.a << Quat("qa") << [0, 0, 0, 1]
        self.b << Quat("qb") << [0, 0.7071068, 0, 0.7071068]
        out = rig.quaternion.slerp([self.a.qa, self.b.qb], [0, 0, 0, 1], 0.5)
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(_value(out[0]), [0, 0, 0, 1], atol=1e-4)


class TestTypedModules(_Poles):
    """Step 2: every vectorized function of vector, matrix, quaternion and euler has a layout; channel
    lists reach them as one vector; the dispatch verbs read rows as the functions do"""

    def test_every_function_has_a_layout(self):
        missing = []
        for module in (rig.vector, rig.matrix, rig.quaternion, rig.euler):
            for name in dir(module):
                func = getattr(module, name)
                if callable(func) and getattr(func, "__module__", None) == module.__name__ and \
                        getattr(func, "_rig_vectorized", False) and func._rig_layout is None:
                    missing.append(f"{module.__name__}.{name}")
        self.assertEqual(missing, [])

    def test_channels_are_one_vector(self):
        p, q = self.p, self.q
        before = len(cmds.ls(type="angleBetween"))
        out = rig.vector.slerp([p.tx, p.ty, p.tz], q.t, 0.5)           # before: InjectionError (a multi)
        self.assertEqual(len(cmds.ls(type="angleBetween")) - before, 1)
        np.testing.assert_allclose(_value(out), [7.0711, 7.0711, 0.0], atol=1e-4)
        np.testing.assert_allclose(_value(rig.vector.cross([p.tx, 0, 0], q.t)), [0, 0, 100], atol=1e-4)
        np.testing.assert_allclose(_value(rig.vector.lerp([p.tx, p.ty, p.tz], q.t, 0.5)), [5, 5, 0], atol=1e-4)

    def test_rows_in_the_typed_functions(self):
        a, b, p, q = self.a, self.b, self.p, self.q
        out = rig.vector.length([p.t, q.t])
        self.assertIsInstance(out, List)
        np.testing.assert_allclose(_value(out), [10, 10], atol=1e-4)
        self.assertAlmostEqual(rig.vector.length([3.0, 4.0]), 5.0)          # (n): any size, folded
        self.assertEqual(len(rig.matrix.axis([a.wm, b.wm])), 2)
        self.assertEqual(len(rig.matrix.multiply([a.wm, b.wm], p.wm)), 2)
        out = rig.quaternion.from_axis_angle([0, 1, 0], [0, 90, 180])     # one axis, three angles
        np.testing.assert_allclose(_value(out), [[0, 0, 0, 1], [0, 0.7071, 0, 0.7071], [0, 1, 0, 0]], atol=1e-4)
        self.assertEqual(len(rig.euler.to_quaternion([[0, 90, 0], [90, 0, 0]])), 2)
        with self.assertRaisesRegex(ValueError, r"^rig\.vector\.cross\(\): vector1 has 3 rows and vector2 has 2"):
            rig.vector.cross([p.t, q.t, a.t], [p.t, q.t])

    def test_a_wrong_shape_builds_nothing(self):
        # decision 1A: each of these gave a value without a word (scalar spread over three channels,
        # a double2 cut to [5, 0, 0], a vector read as a quaternion)
        a, b, p = self.a, self.b, self.p
        cmds.addAttr("p", ln="uv", at="double2")
        cmds.addAttr("p", ln="uvX", at="double", p="uv")
        cmds.addAttr("p", ln="uvY", at="double", p="uv")
        b << Quat("qb") << [0, 0.7071068, 0, 0.7071068]
        before = set(cmds.ls())
        for call, error in ((lambda: rig.vector.slerp(a.tx, b.t, 0.5), TypeError),
                            (lambda: rig.vector.lerp(a.t, p.uv, 0.5), ValueError),
                            (lambda: rig.matrix.blend(a.t, b.wm, 0.5), TypeError),
                            (lambda: rig.quaternion.slerp(a.t, b.qb, 0.5), TypeError),
                            (lambda: rig.slerp(a.tx, b.t, 0.5), TypeError)):
            with self.subTest(call=call):
                with self.assertRaises(error):
                    call()
        self.assertEqual(set(cmds.ls()) - before, set())

    def test_a_rotate_order_is_one_per_row(self):
        # a rotate order (a name, an int, a plug) is a () value: a list of them is rows
        a = self.a
        out = rig.euler.to_matrix(a.r, rotate_order=["xyz", "zyx"])
        self.assertEqual(len(out), 2)
        for row, order in zip(out, ("xyz", "zyx")):
            np.testing.assert_allclose(_value(row), _value(rig.euler.to_matrix(a.r, rotate_order=order)), atol=1e-4)
        self.assertEqual(len(rig.matrix.to_euler([a.wm, self.b.wm], rotate_order=["zxy", 4])), 2)
        self.assertEqual(len(rig.matrix.axis(a.wm, ["x", "z"])), 2)

    def test_the_dispatch_verbs_read_rows(self):
        a, p, q = self.a, self.p, self.q
        out = rig.slerp([p.t, q.t], a.t, 0.5)                             # before: ValueError (length 2)
        self.assertIsInstance(out, List)
        for row, one in zip(out, (p.t, q.t)):
            np.testing.assert_allclose(_value(row), _value(rig.vector.slerp(one, a.t, 0.5)), atol=1e-4)
        np.testing.assert_allclose(_value(rig.normalize([p.t, q.t])), [[1, 0, 0], [0, 1, 0]], atol=1e-4)
        self.assertEqual(len(rig.to_matrix([a.r, self.b.r])), 2)
        self.assertNotIsInstance(rig.lerp([p.tx, p.ty, p.tz], q.t, 0.5), List)   # channels: one vector
        self.assertEqual(rig.lerp(2, 4, 0.5), 3.0)

    def test_in_a_fuse_block(self):
        p, q, a = self.p, self.q, self.a
        with rig.fuse():
            out = rig.slerp([p.t, q.t], a.t, 0.5)
            axes = rig.matrix.axis([a.wm, self.b.wm])
        self.assertEqual((len(out), len(axes)), (2, 2))
        np.testing.assert_allclose(_value(out[0]), _value(rig.vector.slerp(p.t, a.t, 0.5)), atol=1e-4)


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

    def test_vector_lerp_and_elerp(self):
        # before: lerp gave every channel the first weight ([7.5, 2.5, 0]); elerp a weight per channel
        p, q = self.p.t, self.q.t
        out = rig.vector.lerp(p, q, W)
        np.testing.assert_allclose(_value(out), [[7.5, 2.5, 0.0], [5.0, 5.0, 0.0], [2.5, 7.5, 0.0]], atol=1e-4)
        self.assertRows(rig.lerp(p, q, W), lambda w: rig.vector.lerp(p, q, w))
        self.assertRows(rig.vector.elerp(p + 1, q + 1, W), lambda w: rig.vector.elerp(p + 1, q + 1, w))
        np.testing.assert_allclose(_value(rig.vector.lerp([10, 0, 0], [0, 10, 0], W)),
                                   [[7.5, 2.5, 0.0], [5.0, 5.0, 0.0], [2.5, 7.5, 0.0]], atol=1e-4)

    def test_a_vector_literal_is_one_value_for_every_row(self):
        out = rig.vector.lerp(List([self.p.t, self.q.t]), [0, 0, 10], 0.5)      # before d21d79c: lengths {2, 3}
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0]], atol=1e-4)
        out = rig.vector.lerp(List([self.p.t, self.q.t, self.a.t]), [0, 0, 10], 0.5)   # before: rows 0, 0, 10
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0], [1.0, 0.0, 5.0]], atol=1e-4)
        out = rig.vector.lerp([self.p.t, self.q.t], [0, 0, 10], 0.5)              # a plain list of vectors too
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0]], atol=1e-4)

    def test_scalar_lerps_are_one_value(self):
        a, b = self.a, self.b
        self.assertEqual(rig.vector.lerp(2, 4, 0.5), 3.0)                         # folded, as before
        out = rig.vector.lerp(a.tx, b.tx, 0.5)
        self.assertNotIsInstance(out, List)
        self.assertAlmostEqual(out >> None, 6.0, places=4)
        out = rig.vector.lerp(a.tx, b.tx, W)
        np.testing.assert_allclose(_value(out), [4.0, 6.0, 8.0], atol=1e-4)


class TestInAFuseBlock(_Poles):
    def test_weights(self):
        with rig.fuse():
            out = rig.vector.slerp(self.p.t, self.q.t, W)
        self.assertIsInstance(out, List)
        expected = [[10 * math.cos(math.radians(90 * w)), 10 * math.sin(math.radians(90 * w)), 0.0] for w in W]
        np.testing.assert_allclose(_value(out), np.round(expected, 4), atol=1e-4)

    def test_a_plain_list_of_vectors_and_a_literal(self):
        with rig.fuse():
            out = rig.vector.lerp([self.p.t, self.q.t], [0, 0, 10], 0.5)
            refused = None
            try:
                rig.vector.slerp([self.p.t, self.q.t, self.a.t], [self.b.t, self.p.t], 0.5)
            except ValueError as error:
                refused = str(error)
        np.testing.assert_allclose(_value(out), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0]], atol=1e-4)
        self.assertRegex(refused or "", r"input1 has 3 rows and input2 has 2 rows")


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
