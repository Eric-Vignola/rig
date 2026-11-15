"""``rig.interp`` is ``numpy.interp``, and ``sequence(ends=...)`` holds or extrapolates.

Every value is checked against numpy itself: literals fold to numpy's own
result, and a network built from plugs reads what numpy computes for the same
numbers, within float rounding.
"""

import numpy as np
from maya import cmds
from rig import container, elerp, get_options, interp, interpolate, List, Node, set_options, slerp
from rig._tests._base import MayaTestCase

GRID = (-3.0, -1.0, 0.0, 0.5, 1.0, 1.7, 2.0, 2.5, 3.0, 4.2, 9.0)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))

    def driver(self, name="drv"):
        return Node.create("transform", name=name)

    def sweep(self, drv, out, expected_fn, xs=GRID):
        for x in xs:
            with self.subTest(x=x):
                drv.tx << x
                self.assertAlmostEqual(float(np.asarray(out.get())), float(expected_fn(x)), places=9)


class TestInterpIsNumpys(_Case):
    XP, FP = [1.0, 2.0, 3.0], [3.0, 2.0, 0.0]

    def test_literals_fold_to_numpys_result(self):
        cases = (
            ((2.5, self.XP, self.FP), {}),
            ((0.0, self.XP, self.FP), {}),
            ((7.0, self.XP, self.FP), {}),
            ((0.0, self.XP, self.FP), {"left": -99.0}),
            ((7.0, self.XP, self.FP), {"right": 99.0}),
            (([0.0, 1.5, 2.5, 3.5], self.XP, self.FP), {}),
            (([-180, -170, -185, 185, -10, -5, 0, 365], [190, -190, 350, -350], [5, 10, 3, 4]), {"period": 360}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                got = interp(*args, **kwargs)
                want = np.interp(*args, **kwargs)
                np.testing.assert_array_equal(np.asarray(got), want)
                self.assertIsInstance(got, list if isinstance(args[0], list) else float)

    def test_a_plug_x_reads_numpys_value(self):
        drv = self.driver()
        self.sweep(drv, interp(drv.tx, self.XP, self.FP), lambda x: np.interp(x, self.XP, self.FP))

    def test_left_and_right(self):
        drv = self.driver()
        out = interp(drv.tx, self.XP, self.FP, left=-5.0, right=8.0)
        self.sweep(drv, out, lambda x: np.interp(x, self.XP, self.FP, left=-5.0, right=8.0))

    def test_left_and_right_plugs(self):
        drv, knobs = self.driver(), self.driver("knobs")
        knobs.ty << -5.0
        knobs.tz << 8.0
        out = interp(drv.tx, self.XP, self.FP, left=knobs.ty, right=knobs.tz)
        self.sweep(drv, out, lambda x: np.interp(x, self.XP, self.FP, left=-5.0, right=8.0))

    def test_xp_and_fp_plugs(self):
        drv = self.driver()
        points = []
        for k, (x, f) in enumerate(zip(self.XP, self.FP)):
            p = Node.create("transform", name=f"p{k}")
            p.tx << x
            p.ty << f
            points.append(p)
        out = interp(drv.tx, [p.tx for p in points], [p.ty for p in points])
        self.sweep(drv, out, lambda x: np.interp(x, self.XP, self.FP))

    def test_period_wraps_like_numpy(self):
        # numpy's documentation example, xp unsorted until wrapped, x negative too
        xp, fp = [190, -190, 350, -350], [5, 10, 3, 4]
        drv = self.driver()
        out = interp(drv.tx, xp, fp, period=360)
        self.sweep(drv, out, lambda x: np.interp(x, xp, fp, period=360),
                   xs=(-180, -170, -185, 185, -10, -5, 0, 365, -725, 900.5))

    def test_a_list_of_x_gives_one_value_each(self):
        drivers = [self.driver(f"d{k}") for k in range(4)]
        out = interp(List([d.tx for d in drivers]), self.XP, self.FP)
        self.assertIsInstance(out, List)
        xs = [0.0, 1.5, 2.5, 3.5]
        for d, x in zip(drivers, xs):
            d.tx << x
        np.testing.assert_allclose(np.asarray([o.get() for o in out], dtype=float),
                                   np.interp(xs, self.XP, self.FP), atol=1e-9)

    def test_one_point(self):
        drv = self.driver()
        self.sweep(drv, interp(drv.tx, [2.0], [7.0], left=1.0, right=9.0),
                   lambda x: np.interp(x, [2.0], [7.0], left=1.0, right=9.0), xs=(0.0, 2.0, 5.0))

    def test_refusals(self):
        drv, v = self.driver(), self.driver("v")
        with self.assertRaisesRegex(TypeError, "not a scalar.*sequence"):
            interp(drv.tx, [0, 1], [v.t, v.t])
        with self.assertRaisesRegex(ValueError, "not of the same length"):
            interp(drv.tx, [0, 1, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "empty"):
            interp(drv.tx, [], [])
        with self.assertRaisesRegex(ValueError, "non-zero"):
            interp(drv.tx, [0, 1], [0, 1], period=0)
        with self.assertRaisesRegex(ValueError, "numeric xp"):
            interp(drv.tx, [v.tx, 1], [0, 1], period=2)
        with self.assertRaisesRegex(TypeError, "period must be a number"):
            interp(drv.tx, [0, 1], [0, 1], period=v.tx)
        with self.assertRaisesRegex(TypeError, "str"):     # an operand like any other
            interp("drv.tx", [0, 1], [0, 1])


class TestSequenceEnds(_Case):
    XP = [0.0, 0.5, 1.0]

    def points(self):
        nodes = []
        for k, value in enumerate(([1, 2, 4], [4, 2, 1], [2, 2, 2])):
            n = Node.create("transform", name=f"p{k}")
            n.t << value
            nodes.append(n)
        return [n.t for n in nodes]

    def test_extrapolate_is_still_the_default(self):
        drv = self.driver()
        out = interpolate.sequence(drv.tx, self.XP, self.points())
        drv.tx << 1.5   # one segment past the end: 2 * yp[2] - yp[1]
        np.testing.assert_allclose(out.get(), [0, 2, 3], atol=1e-9)

    def test_hold_keeps_the_end_values_for_any_method(self):
        for name, method in (("lerp", interpolate._v_lerp), ("slerp", slerp), ("elerp", elerp)):
            for place in ("top level", "inside a container"):
                with self.subTest(method=name, place=place):
                    cmds.file(new=True, force=True)
                    drv, yp = self.driver(), self.points()
                    if place == "top level":
                        out = interpolate.sequence(drv.tx, self.XP, yp, method=method, ends="hold")
                    else:
                        with container("user1"):
                            out = interpolate.sequence(drv.tx, self.XP, yp, method=method, ends="hold")
                    for x, end in ((-2.0, [1, 2, 4]), (0.0, [1, 2, 4]), (1.0, [2, 2, 2]), (3.0, [2, 2, 2])):
                        drv.tx << x
                        np.testing.assert_allclose(out.get(), end, atol=1e-9, err_msg=f"x={x}")

    def test_a_plug_switches_live(self):
        drv, knob = self.driver(), self.driver("knob")
        out = interpolate.sequence(drv.tx, self.XP, self.points(), ends=knob.tx == 2)
        drv.tx << 1.5
        np.testing.assert_allclose(out.get(), [0, 2, 3], atol=1e-9)     # knob 0: extrapolates
        knob.tx << 2
        np.testing.assert_allclose(out.get(), [2, 2, 2], atol=1e-9)     # knob 2: holds

    def test_an_unknown_ends_raises_before_any_node(self):
        drv, yp = self.driver(), self.points()
        before = len(cmds.ls())
        with self.assertRaisesRegex(ValueError, "'extrapolate', 'hold' or a plug"):
            interpolate.sequence(drv.tx, self.XP, yp, ends="clamp")
        self.assertEqual(len(cmds.ls()), before)
