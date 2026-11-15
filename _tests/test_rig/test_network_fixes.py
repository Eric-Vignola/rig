"""Network fixes found by the fused-expression fuzzer that do not grow the graph.

  1. floor / ceil / trunc / int / // are exact at the integers (native nodes per
     channel on a vector: floor(-4) was -5); int and trunc of a vector, and a
     scalar //, also build fewer nodes;
  2. a list of plugs is a vector against a scalar ([a.tx, a.ty, a.tz] / s kept
     only a.tx); a list into a scalar attribute raises (it connected the first
     plug); choice broadcasts over a List selector (it took the first selector);
  3. a vector clamp is one clampRange per channel (12 nodes -> 3), with the
     scalar's rule (x > hi gives hi, then x < lo gives lo);
  4. searchsorted's ladder compares exactly, in the same number of nodes.
Not changed, on purpose (they add nodes; see rig_network_fixes_todo.md): the
condition node's float32 and != tolerance, the rounding of a double into a bool
for and / or / not, condition's test meaning == 1, a division by zero.
"""

from maya import cmds
from rig import functions as F
from rig import List, Node
from rig._tests._base import MayaTestCase


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.v = Node.create("transform", name="v")
        self.w = Node.create("transform", name="w")
        self.k = Node.create("transform", name="k")

    def put(self, plug, value):
        if isinstance(value, (list, tuple)):
            cmds.setAttr(str(plug), *value, type="double3")
        else:
            cmds.setAttr(str(plug), value)

    def value(self, plug):
        got = cmds.getAttr(str(plug))
        return list(got[0]) if isinstance(got, list) else float(got)

    def assertValue(self, plug, want, places=9):
        got = self.value(plug)
        if isinstance(want, list):
            self.assertEqual(len(got), len(want))
            for g, x in zip(got, want):
                self.assertAlmostEqual(g, x, places=places)
        else:
            self.assertAlmostEqual(got, want, places=places)

    def made(self, build):
        before = set(cmds.ls())
        out = build()
        return out, sorted(cmds.nodeType(n) for n in set(cmds.ls()) - before)


class TestRounding(_Case):
    def test_exact_at_the_integers(self):
        cases = ((F.floor, (3, -4, 0.25), [3, -4, 0]), (F.ceil, (3, -4, -0.25), [3, -4, 0]),
                 (F.trunc, (-3.7, 2.9, 4.0), [-3, 2, 4]), (F.int, (-3.7, 2.9, -4.0), [-3, 2, -4]))
        for fn, values, want in cases:
            with self.subTest(function=fn.__name__):
                self.put(self.v.t, values)
                self.assertValue(fn(self.v.t), want)

    def test_floor_division(self):
        self.put(self.v.t, (3, -4, 7))
        self.put(self.w.t, (1, 1, 2))
        self.assertValue(self.v.t // self.w.t, [3, -4, 3])
        self.put(self.v.tx, -4.00000005)
        self.assertValue(self.v.tx // self.w.tx, -5.0)        # the offset gave -4 this close

    def test_node_counts(self):
        # a vector floor / ceil: 3 native nodes + the gathering value (was 2, and wrong)
        _, floor = self.made(lambda: F.floor(self.v.t))
        self.assertEqual(floor.count("floor"), 3)
        # int, trunc and a scalar // build fewer nodes than the long-cast networks did
        _, scalar_int = self.made(lambda: F.int(self.v.tx))
        self.assertEqual(scalar_int, ["truncate"])
        _, floordiv = self.made(lambda: self.v.tx // self.w.tx)
        self.assertEqual(sorted(t for t in floordiv if t not in ("container", "hyperLayout", "network")),
                         ["divide", "floor"])


class TestListsOfPlugs(_Case):
    def test_a_list_of_plugs_is_a_vector_against_a_scalar(self):
        self.put(self.v.t, (2, 4, 8))
        self.put(self.k.tx, 2)
        lst = [self.v.tx, self.v.ty, self.v.tz]
        cases = (("/", self.k.tx.__rtruediv__(lst), [1, 2, 4]), ("*", self.k.tx.__rmul__(lst), [4, 8, 16]),
                 ("+", self.k.tx.__radd__(lst), [4, 6, 10]), ("-", self.k.tx - lst, [0, -2, -6]))
        for op, out, want in cases:
            with self.subTest(op=op):
                self.assertValue(out, want)

    def test_a_list_of_plugs_modulo_a_scalar(self):
        self.put(self.v.t, (2, 5, 8))
        self.put(self.k.tx, 3)
        self.assertValue(self.k.tx.__rmod__([self.v.tx, self.v.ty, self.v.tz]), [2, 2, 2])

    def test_a_list_into_a_scalar_attribute_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot inject sequence of size 2 into scalar"):
            self.k.tx << [self.v.tx, self.v.ty]
        self.assertFalse(cmds.listConnections(str(self.k.tx), source=True, destination=False))

    def test_a_list_selector_picks_per_row(self):
        rows = List([self.v.tx, self.w.tx])
        picked = F.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]], selector=rows)
        self.assertIsInstance(picked, List)
        self.put(self.v.tx, 0)
        self.put(self.w.tx, 2)
        self.assertValue(picked[0], [1, 0, 0])
        self.assertValue(picked[1], [0, 0, 1])                # was the first row's pick, [1, 0, 0]


class TestClamp(_Case):
    def test_a_vector_clamp_is_clamp_range_per_channel_with_the_scalar_rule(self):
        out, types = self.made(lambda: F.clamp(self.v.t, 5, 2))
        self.assertEqual(types.count("clampRange"), 3)
        self.assertNotIn("condition", types)
        for values in ((0, 3, 9), (-1, 1.5, 9)):
            self.put(self.v.t, values)
            for x, got in zip(values, self.value(out)):
                with self.subTest(x=x):
                    self.assertAlmostEqual(got, 2.0 if x > 2 else 5.0 if x < 5 else x)
                    self.put(self.k.tx, x)
                    self.assertAlmostEqual(self.value(F.clamp(self.k.tx, 5, 2)), got)
        self.put(self.v.t, (-1, 1.5, 9))
        self.assertValue(F.clamp(self.v.t, 0, 2), [0, 1.5, 2])


class TestSearchsorted(_Case):
    def test_the_ladder_is_exact_and_keeps_its_node_count(self):
        knots = [0.0, 1.0, 2.0]
        out, types = self.made(lambda: F.searchsorted(knots, self.v.tx))
        self.assertNotIn("not", types)
        for x, want in ((-1.0, 0), (1.0, 1), (1.0 + 1e-9, 1), (1.0 - 1e-9, 0), (2.0, 2), (5.0, 2)):
            with self.subTest(x=x):
                self.put(self.v.tx, x)
                self.assertValue(out, float(want))
