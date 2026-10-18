"""Features step 5: ``AnimCurve``, the keys value, and the curve node classes.

``plug << AnimCurve(values, times=None, **options)`` keys the plug: one animation curve per
channel, of the type Maya's rule gives the plug (``timedAnimCurveTypeForPlug``: TL, TA, TU,
TT; a ``Typed`` slot TU), values in UI units, linear tangents with a step out tangent on a bool
or an enum; a driven plug has its input replaced (D1 = A: no pairBlend); a re-run updates the
plug's own curve in place (the same UUID) and unchanged keys write nothing; a locked plug
refuses, even with the keys its curve holds (D3 = b); one undo step ``rig.animcurve``; the
statement returns its left-hand side. ``AnimCurve('x')`` refers to a curve, ``AnimCurveTL`` /
``TA`` / ``TU`` / ``TT`` are the typed classes (``create`` / ``define`` with keys), and
``get_keys`` / ``set_keys`` / ``num_keys`` / ``evaluate`` / ``AnimCurve.of`` read and write
them.

* `TestTheValue`: construction checks (shapes, times, options, refusals), the repr, pickling,
  ``isinstance(AnimCurve([0]), AnimCurve)`` is False.
* `TestDetection`, `TestUnits`, `TestShapes`, `TestOptions`: the curve type and default
  tangents per attribute kind, units under ``linear=m`` and ``time=ntsc``, the shapes and the
  per-channel forms, tangents / slopes / weights / infinity.
* `TestReturns`: what each statement returns (the OPEN (a) pins).
* `TestConnected`: a constraint, a plain wire, an expression, a unitConversion and Maya's
  pairBlend are replaced, both journal routes (the D1 pins); locks.
* `TestReRun`: the update in place, a shared curve, another type, a driven key, a referenced
  curve, the no-op, the locked plug (D3).
* `TestUndo`: one step per statement, refusals with the queue unchanged, the missing plug-in.
* `TestNodeClasses`, `TestReadBack`, `TestNaming`, `TestTheElementCrash`.
"""

import math
import os
import pickle
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

import rig
from rig import (
    AnimCurve,
    AnimCurveTA,
    AnimCurveTL,
    AnimCurveTT,
    AnimCurveTU,
    Bool,
    Enum,
    Float,
    InjectionError,
    Int,
    List,
    Node,
    NodeTypeError,
    Plug,
    Typed,
    container,
    set_options,
)
from rig.spec import Angle, Color, Euler, Quat, Time, Vector, lock, skip
from rig._internal import anim_keys, journal
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _inputs(name):
    return cmds.listConnections(name, source=True, destination=False, plugs=True) or []


def _curve_of(name):
    found = cmds.listConnections(name, source=True, destination=False, type="animCurve") or []
    return found[0] if found else None


def _values(curve):
    return cmds.keyframe(curve, query=True, valueChange=True) or []


def _times(curve):
    return cmds.keyframe(curve, query=True, timeChange=True) or []


def _cube(name="pCube1"):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _nodes():
    return set(cmds.ls())


def _empty_queue():
    return cmds.undoInfo(query=True, undoQueueEmpty=True)


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._units = (cmds.currentUnit(query=True, linear=True), cmds.currentUnit(query=True, time=True),
                       cmds.currentUnit(query=True, angle=True))

    def tearDown(self):
        try:
            linear, time, angle = self._units
            cmds.currentUnit(linear=linear, time=time, angle=angle)
            set_options(flatten_containers=True)
        finally:
            super().tearDown()

    def assertRefusesWithNothingChanged(self, exc, fn, *texts):
        """``fn()`` raises ``exc`` (its text holding ``texts``), and makes no node, no
        connection and no undo step."""
        cmds.flushUndo()
        before = self.scene_state()
        with self.assertRaises(exc) as ctx:
            fn()
        for text in texts:
            self.assertIn(text, str(ctx.exception))
        self.assert_state_equal(self.scene_state(), before)
        self.assertTrue(_empty_queue(), "a refusal left an undo step")
        return ctx.exception


# --------------------------------------------------------------------------------------- #
#  The value
# --------------------------------------------------------------------------------------- #


class TestTheValue(_Case):
    def test_keys_are_a_value_not_a_node(self):
        before = _nodes()
        k = AnimCurve([0, 1, 0, 2])
        self.assertIs(type(k), anim_keys._Keys)
        self.assertFalse(isinstance(k, AnimCurve))
        self.assertEqual(repr(k), "AnimCurve([0.0, 1.0, 0.0, 2.0], times=[0.0, 1.0, 2.0, 3.0])")
        self.assertEqual(_nodes(), before)
        np.testing.assert_array_equal(k.times, [0, 1, 2, 3])
        np.testing.assert_array_equal(k.values, [0, 1, 0, 2])
        self.assertEqual(k.num_keys, 4)
        self.assertEqual((k.pre_infinity, k.post_infinity), ("constant", "constant"))

    def test_every_sequence_kind_makes_keys(self):
        for values in ([1, 2], (1, 2), range(1, 3), np.array([1.0, 2.0]), (x for x in (1, 2))):
            with self.subTest(values=type(values).__name__):
                np.testing.assert_array_equal(AnimCurve(values).values, [1, 2])

    def test_a_number_or_nothing_is_a_type_error_naming_both_forms(self):
        for call, text in ((lambda: AnimCurve(5), "AnimCurve([5])"), (lambda: AnimCurve(), "AnimCurve([0, 1, 0])"),
                           (lambda: AnimCurve(2.5), "AnimCurve([2.5])")):
            with self.subTest(text=text):
                with self.assertRaises(TypeError) as ctx:
                    call()
                self.assertIn(text, str(ctx.exception))
                self.assertIn("AnimCurve('x') refers", str(ctx.exception))

    def test_construction_checks(self):
        plug = Node.create("transform", name="p").tx
        cases = [
            (TypeError, lambda: AnimCurve([plug, plug]), "not plugs"),
            (ValueError, lambda: AnimCurve([]), "at least one key"),
            (ValueError, lambda: AnimCurve([1, np.nan]), "finite"),
            (ValueError, lambda: AnimCurve([1, np.inf]), "finite"),
            (ValueError, lambda: AnimCurve(np.zeros((2, 2, 2))), "(N,) or (N, C)"),
            (ValueError, lambda: AnimCurve([1, 2], times=[0]), "2 values need 2 times"),
            (ValueError, lambda: AnimCurve([1, 2], times=[1, 1]), "strictly increasing"),
            (ValueError, lambda: AnimCurve([1, 2], times=[2, 1]), "strictly increasing"),
            (ValueError, lambda: AnimCurve([1, 2], times=[0, np.nan]), "finite"),
            (TypeError, lambda: AnimCurve([1, 2], tangent="step"), "unknown option 'tangent'"),
            (TypeError, lambda: AnimCurve([1, 2], tangents="bouncy"), "linear, step"),
            (TypeError, lambda: AnimCurve([1, 2], infinity="forever"), "constant, linear, cycle"),
            (ValueError, lambda: AnimCurve([1, 2], in_weights=[1, 1]), "weighted=True"),
            (ValueError, lambda: AnimCurve([1, 2], weighted=True, in_weights=[1, -1]), "positive"),
            (ValueError, lambda: AnimCurve([1, 2], in_slopes=[0, 1], in_tangents="spline"), "fixed"),
            (ValueError, lambda: AnimCurve([1, 2], out_slopes=[0, 1], tangents=["fixed", "flat"]), "fixed"),
            (ValueError, lambda: AnimCurve([1, 2], weighted=True, out_weights=[1, 2], out_tangents="linear"), "fixed"),
            (ValueError, lambda: AnimCurve([1, 2], slopes=[0, 1, 2]), "3 rows for 2 keys"),
            (ValueError, lambda: AnimCurve(np.zeros((2, 3)), slopes=np.zeros((2, 2))), "channel counts"),
            (ValueError, lambda: AnimCurve([1, 2], slopes=[0, np.inf]), "infinite"),
            (TypeError, lambda: AnimCurve([1, None]), "numbers"),
            (TypeError, lambda: AnimCurve([[1, 2], [3]]), "numbers"),
            (TypeError, lambda: AnimCurve(["on", None]), "names and ints"),
            (TypeError, lambda: AnimCurve([1, 2], weighted=1), "True or False"),
        ]
        for exc, fn, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(exc) as ctx:
                    fn()
                self.assertIn(text, str(ctx.exception))

    def test_the_options_taken(self):
        k = AnimCurve([0, 1, 2], times=[0, 0.5, 1.25], in_tangents="fixed", out_tangents=["step", "linear", "Step Next"],
                      in_slopes=[np.nan, 2.0, np.nan], infinity="Cycle_Relative", post_infinity="oscillate")
        self.assertEqual(k.in_tangents, "fixed")
        self.assertEqual(list(k.out_tangents), ["step", "linear", "stepnext"])
        self.assertEqual((k.pre_infinity, k.post_infinity), ("cycleRelative", "oscillate"))
        np.testing.assert_array_equal(k.times, [0, 0.5, 1.25])
        self.assertTrue(np.isnan(k.in_slopes[0]) and k.in_slopes[1] == 2.0)

    def test_keys_pickle(self):
        k = AnimCurve([0, 90], times=[0, 24], out_slopes=[1, 2], infinity="cycle")
        back = pickle.loads(pickle.dumps(k))
        self.assertEqual(repr(back), repr(k))
        np.testing.assert_array_equal(back.out_slopes, [1, 2])

    def test_an_anim_curve_value_given_back(self):
        k = AnimCurve([1, 2])
        self.assertIs(AnimCurve(k), k)
        with self.assertRaises(TypeError):
            AnimCurve(k, times=[0, 1])


# --------------------------------------------------------------------------------------- #
#  Detection and units
# --------------------------------------------------------------------------------------- #


class TestDetection(_Case):
    def _key(self, plug, values=(0, 1, 0)):
        plug << AnimCurve(list(values))
        curve = _curve_of(str(plug))
        self.assertIsNotNone(curve, str(plug))
        return curve, cmds.nodeType(curve), cmds.keyTangent(curve, query=True, itt=True), cmds.keyTangent(
            curve, query=True, ott=True)

    def test_the_type_and_default_tangents_per_kind(self):
        n = _cube("n")
        cmds.addAttr("n", longName="dist", attributeType="doubleLinear", keyable=True)
        n << Angle("ang") << Time("tm") << Float("dbl") << Int("cnt")
        n << Bool("flag") << Enum("mode", en="a:b:c") << Float("flt", at="float")
        lin, step = ["linear"] * 3, ["step"] * 3
        table = [
            (n.tx, "animCurveTL", lin, lin), (n.dist, "animCurveTL", lin, lin),
            (n.rx, "animCurveTA", lin, lin), (n.ang, "animCurveTA", lin, lin),
            (n.tm, "animCurveTT", lin, lin),
            (n.sx, "animCurveTU", lin, lin), (n.dbl, "animCurveTU", lin, lin), (n.cnt, "animCurveTU", lin, lin),
            (n.flt, "animCurveTU", lin, lin),
            (n.v, "animCurveTU", lin, step), (n.flag, "animCurveTU", lin, step), (n.mode, "animCurveTU", lin, step),
            (n.ro, "animCurveTU", lin, step),
        ]
        for plug, kind, itt, ott in table:
            with self.subTest(plug=str(plug)):
                _, got, gin, gout = self._key(plug, (0, 1, 0) if kind != "animCurveTT" else (0, 5, 10))
                self.assertEqual((got, gin, gout), (kind, itt, ott))
                # Maya's own rule
                ref = OpenMaya.MSelectionList()
                ref.add(str(plug))
                self.assertEqual(anim_keys._CURVE_TYPES[anim_keys._TYPE_RULE.timedAnimCurveTypeForPlug(ref.getPlug(0))], kind)

    def test_compound_specs_and_an_element(self):
        n = _cube("n")
        n << Euler("e") << Color("col") << Quat("q") << Float("arr", multi=True)
        n.arr[2] << 0
        for plug, kind in ((n.e, "animCurveTA"), (n.col, "animCurveTU"), (n.q, "animCurveTU")):
            with self.subTest(plug=str(plug)):
                plug << AnimCurve([0, 1])
                kinds = {cmds.nodeType(c) for c in AnimCurve.of(plug)}
                self.assertEqual(kinds, {kind})
                self.assertEqual(len(AnimCurve.of(plug)), plug.get_num_children())
        self.assertEqual(n.arr[2] << AnimCurve([0, 1]), Plug("n.arr[2]"))
        self.assertEqual(cmds.nodeType(_curve_of("n.arr[2]")), "animCurveTU")
        self.assertEqual(_curve_of("n.arr[2]"), "n_arr_2_")

    def test_a_typed_slot_takes_an_animcurve_tu(self):
        net = Node.create("network", name="net")
        net << Typed("mix", multi=True) << Typed("one")
        for plug in (net.mix[0], net.one):
            with self.subTest(plug=str(plug)):
                plug << AnimCurve([0, 1, 4])
                self.assertEqual(cmds.nodeType(_curve_of(str(plug))), "animCurveTU")
                cmds.currentTime(2)
                self.assertAlmostEqual(cmds.getAttr(str(plug)), 4.0)

    def test_an_int_is_linear_and_rounded_by_the_attribute(self):
        n = _cube("n")
        n << Int("cnt")
        n.cnt << AnimCurve([1, 3])
        self.assertEqual(cmds.keyTangent(_curve_of("n.cnt"), query=True, ott=True), ["linear", "linear"])
        self.assertEqual(cmds.getAttr("n.cnt", time=0.5), 2)
        n << Bool("on")
        n.on << AnimCurve([0, 1], tangents="linear")
        self.assertEqual(cmds.keyTangent(_curve_of("n.on"), query=True, ott=True), ["linear", "linear"])


class TestUnits(_Case):
    def test_degrees_in_and_out(self):
        n = _cube("n")
        n.rx << AnimCurve([0, 90], times=[0, 24])
        self.assertEqual(_values("n_rotateX"), [0.0, 90.0])
        self.assertAlmostEqual(cmds.getAttr("n.rx", time=12), 45.0)
        np.testing.assert_allclose(AnimCurve("n_rotateX").get_keys().values, [0, 90])
        np.testing.assert_allclose(AnimCurve("n_rotateX").evaluate([0, 12, 24]), [0, 45, 90])
        cmds.currentUnit(angle="rad")
        n.ry << AnimCurve([0, math.pi])
        cmds.currentUnit(angle="deg")
        self.assertAlmostEqual(_values("n_rotateY")[1], 180.0)

    def test_linear_unit_metres(self):
        cmds.currentUnit(linear="m")
        n = _cube("n")
        n.tx << AnimCurve([0, 1.5])
        self.assertAlmostEqual(cmds.getAttr("n.tx", time=1), 1.5)
        self.assertAlmostEqual(_values("n_translateX")[1], 1.5)
        np.testing.assert_allclose(AnimCurve("n_translateX").get_keys().values, [0, 1.5])
        cmds.currentUnit(linear="cm")
        self.assertAlmostEqual(cmds.getAttr("n.tx", time=1), 150.0)

    def test_time_unit_ntsc_and_fractional_frames(self):
        cmds.currentUnit(time="ntsc")
        n = _cube("n")
        n.tx << AnimCurve([0, 1, 2], times=[0, 0.5, 30])
        self.assertEqual(_times("n_translateX"), [0.0, 0.5, 30.0])
        np.testing.assert_allclose(AnimCurve("n_translateX").get_keys().times, [0, 0.5, 30])
        cmds.currentUnit(time="film")
        # the keys stay at the same seconds: 30 ntsc frames are 24 film frames
        self.assertAlmostEqual(_times("n_translateX")[2], 24.0)

    def test_a_time_curve_takes_frames(self):
        n = _cube("n")
        n << Time("tm")
        n.tm << AnimCurve([0, 10, 40], times=[0, 10, 20])
        self.assertEqual(cmds.nodeType(_curve_of("n.tm")), "animCurveTT")
        self.assertEqual(_values("n_tm"), [0.0, 10.0, 40.0])
        np.testing.assert_allclose(AnimCurve("n_tm").get_keys().values, [0, 10, 40])
        np.testing.assert_allclose(AnimCurve("n_tm").evaluate([0, 15]), [0, 25])
        with self.assertRaises(ValueError):
            n.tm << AnimCurve([0, 1], slopes=1.0)


# --------------------------------------------------------------------------------------- #
#  Shapes and options
# --------------------------------------------------------------------------------------- #


class TestShapes(_Case):
    def test_compound_shapes(self):
        n = _cube("n")
        data = np.arange(30, dtype=float).reshape(10, 3)
        self.assertEqual(n.t << AnimCurve(data), Plug("n.translate"))
        for i, axis in enumerate("XYZ"):
            self.assertEqual(_values(f"n_translate{axis}"), data[:, i].tolist())
        n.r << AnimCurve([0, 1, 0])  # (N,): the same keys on every child, N == C
        for axis in "XYZ":
            self.assertEqual(_values(f"n_rotate{axis}"), [0.0, 1.0, 0.0])
        n.s << AnimCurve([[1, 2, 3]])  # one key per child
        self.assertEqual([_values(f"n_scale{a}") for a in "XYZ"], [[1.0], [2.0], [3.0]])
        n.s << AnimCurve([[4], [5]])  # (N, 1)
        self.assertEqual([_values(f"n_scale{a}") for a in "XYZ"], [[4.0, 5.0]] * 3)

    def test_a_column_count_mismatch_is_a_value_error_with_nothing_made(self):
        n = _cube("n")
        self.assertRefusesWithNothingChanged(ValueError, lambda: n.t << AnimCurve(np.zeros((4, 2))), "(N, 2) keys for 3 channels")
        self.assertRefusesWithNothingChanged(ValueError, lambda: n.tx << AnimCurve(np.zeros((4, 3))), "for 1 channel")

    def test_the_per_channel_list(self):
        n = _cube("n")
        n.ty << 4
        self.assertEqual(n.t << [AnimCurve([5, 6]), skip, 0], Plug("n.translate"))
        self.assertEqual(_values("n_translateX"), [5.0, 6.0])
        self.assertIsNone(_curve_of("n.ty"))
        self.assertEqual(cmds.getAttr("n.ty"), 4.0)
        self.assertEqual(cmds.getAttr("n.tz"), 0.0)
        self.assertEqual(self.undo_name(), "rig.animcurve")
        self.undo_steps(1)
        self.assertIsNone(_curve_of("n.tx"))
        # a refused keys slot sets nothing, not even the slot before it
        cmds.flushUndo()
        self.assertRefusesWithNothingChanged(ValueError, lambda: n.t << [5, AnimCurve(np.zeros((2, 2)))], "for 1 channel")
        self.assertEqual(cmds.getAttr("n.tx"), 0.0)

    def test_a_multi_slice_and_a_bare_multi(self):
        n = _cube("n")
        n << Float("arr", multi=True)
        for i in range(3):
            n.arr[i] << 0
        # (re-pinned, user decision 9A, 2026-10-17: a List == List has no truth value; compare the elements)
        self.assertEqual(list(n.arr[:] << AnimCurve([0, 1])), [Plug("n.arr[0]"), Plug("n.arr[1]"), Plug("n.arr[2]")])
        self.assertEqual(sorted(cmds.ls(type="animCurveTU")), ["n_arr_0_", "n_arr_1_", "n_arr_2_"])
        self.assertRefusesWithNothingChanged(TypeError, lambda: n.arr << AnimCurve([0, 1]), "multi attribute", "re-run")

    def test_non_numeric_plugs_are_refused(self):
        n = _cube("n")
        n << rig.String("label") << rig.Message("mmsg") << rig.Matrix("mtx")
        shape = Node(cmds.listRelatives("n", shapes=True)[0])
        for plug in (n.wm[0], n.label, n.mmsg, n.mtx, n.message, shape.outMesh, shape.inMesh):
            with self.subTest(plug=str(plug)):
                self.assertRefusesWithNothingChanged(TypeError, lambda: plug << AnimCurve([0, 1]), "numeric attribute")

    def test_an_output_plug_is_an_injection_error(self):
        md = Node.create("multiplyDivide", name="md")
        self.assertRefusesWithNothingChanged(InjectionError, lambda: md.outputX << AnimCurve([0, 1]), "output")

    def test_enum_field_names(self):
        n = _cube("n")
        n << Enum("mode", en="off:on:auto")
        n.mode << AnimCurve(["off", "On", 2, "auto"])
        self.assertEqual(_values("n_mode"), [0.0, 1.0, 2.0, 2.0])
        self.assertRefusesWithNothingChanged(TypeError, lambda: n.mode << AnimCurve(["off", "sideways"]), "sideways")
        self.assertRefusesWithNothingChanged(TypeError, lambda: n.tx << AnimCurve(["off", "on"]), "enum")
        self.assertRefusesWithNothingChanged(TypeError, lambda: List([n.ty, n.mode]) << AnimCurve(["on", "off"]), "enum")


class TestOptions(_Case):
    def test_every_tangent_name(self):
        n = _cube("n")
        maya_names = {"linear": "linear", "step": "step", "stepnext": "stepnext", "flat": "flat", "spline": "spline",
                      "clamped": "clamped", "plateau": "plateau", "auto": "auto", "fixed": "fixed",
                      "autoease": "autoease", "automix": "automix"}
        for name, maya in maya_names.items():
            with self.subTest(name=name):
                n.tx << AnimCurve([0, 1, 0], tangents=name)
                self.assertEqual(cmds.keyTangent("n_translateX", query=True, itt=True), [maya] * 3)
                self.assertEqual(cmds.keyTangent("n_translateX", query=True, ott=True), [maya] * 3)

    def test_per_key_and_per_channel_tangents(self):
        n = _cube("n")
        n.tx << AnimCurve([0, 1, 0], in_tangents=["flat", "spline", "linear"], out_tangents="step")
        self.assertEqual(cmds.keyTangent("n_translateX", query=True, itt=True), ["flat", "spline", "linear"])
        self.assertEqual(cmds.keyTangent("n_translateX", query=True, ott=True), ["step"] * 3)
        n.r << AnimCurve(np.zeros((2, 3)), tangents=[["flat", "step", "linear"], ["flat", "step", "linear"]])
        self.assertEqual([cmds.keyTangent(f"n_rotate{a}", query=True, ott=True)[0] for a in "XYZ"],
                         ["flat", "step", "linear"])

    def test_slopes_make_a_side_fixed_and_unlock_differing_sides(self):
        n = _cube("n")
        n.sx << AnimCurve([0, 1, 0], in_slopes=[np.nan, 2, np.nan], out_slopes=[np.nan, -3, np.nan])
        c = "n_scaleX"
        self.assertEqual(cmds.keyTangent(c, query=True, itt=True), ["linear", "fixed", "linear"])
        self.assertEqual(cmds.keyTangent(c, query=True, ott=True), ["linear", "fixed", "linear"])
        self.assertAlmostEqual(cmds.keyTangent(c, query=True, index=(1, 1), ia=True)[0], math.degrees(math.atan(2)))
        self.assertAlmostEqual(cmds.keyTangent(c, query=True, index=(1, 1), oa=True)[0], math.degrees(math.atan(-3)))
        self.assertEqual(cmds.keyTangent(c, query=True, index=(1, 1), lock=True), [False])
        n.sy << AnimCurve([0, 1, 0], slopes=[0, 2, 0])
        self.assertEqual(cmds.keyTangent("n_scaleY", query=True, index=(1, 1), lock=True), [True])
        # degrees per frame on an angle curve
        n.rx << AnimCurve([0, 10], out_slopes=[5, np.nan])
        k = AnimCurve("n_rotateX").get_keys()
        self.assertAlmostEqual(k.out_slopes[0], 5.0, places=5)  # Maya keeps a tangent in float

    def test_weights(self):
        n = _cube("n")
        n.sx << AnimCurve([0, 1, 0], weighted=True, in_weights=[1, 2, 3], out_weights=[4, 5, 6])
        c = "n_scaleX"
        self.assertTrue(cmds.getAttr(c + ".weightedTangents"))
        np.testing.assert_allclose(cmds.keyTangent(c, query=True, iw=True), [1, 2, 3])
        np.testing.assert_allclose(cmds.keyTangent(c, query=True, ow=True), [4, 5, 6])
        n.sy << AnimCurve([0, 1], weighted=True)
        self.assertTrue(cmds.getAttr("n_scaleY.weightedTangents"))

    def test_infinity_names_with_the_enum_tolerance(self):
        n = _cube("n")
        for name, code in (("constant", 0), ("linear", 1), ("cycle", 3), ("cycle_relative", 4), ("Cycle Relative", 4),
                           ("oscillate", 5)):
            with self.subTest(name=name):
                n.tx << AnimCurve([0, 1], infinity=name)
                self.assertEqual((cmds.getAttr("n_translateX.preInfinity"), cmds.getAttr("n_translateX.postInfinity")), (code, code))
        n.tx << AnimCurve([0, 1], infinity="cycle", post_infinity="linear")
        self.assertEqual((cmds.getAttr("n_translateX.preInfinity"), cmds.getAttr("n_translateX.postInfinity")), (3, 1))


# --------------------------------------------------------------------------------------- #
#  Returns (OPEN (a))
# --------------------------------------------------------------------------------------- #


class TestReturns(_Case):
    def test_the_left_hand_side_comes_back(self):
        cube = _cube("pCube1")
        test = Node("pCube1") << Float("foo") << AnimCurve([1, 2, 3, 4])
        self.assertEqual(test, Plug("pCube1.foo"))
        self.assertIs(type(test), Plug)
        self.assertEqual(cmds.nodeType(_curve_of("pCube1.foo")), "animCurveTU")
        self.assertEqual(_times("pCube1_foo"), [0.0, 1.0, 2.0, 3.0])
        tx = cube.tx
        self.assertIs(tx << AnimCurve([0, 1]), tx)
        t = cube.t
        self.assertIs(t << AnimCurve(np.random.random((10, 3))), t)
        a, b = Node.create("transform", name="a"), Node.create("transform", name="b")
        lst = List([a, b]).ry
        self.assertIs(lst << AnimCurve([0, 45, 0]), lst)
        pairs = List([a, b]).rz
        self.assertIs(pairs << [AnimCurve([1, 2]), AnimCurve([3, 4])], pairs)
        self.assertIs(t << [AnimCurve([1, 2]), skip, 0], t)

    def test_lock_chains_after_it(self):
        _cube("pCube1")
        bar = Node("pCube1") << Float("bar") << AnimCurve([0, 1]) << lock
        self.assertEqual(bar, Plug("pCube1.bar"))
        self.assertTrue(cmds.getAttr("pCube1.bar", lock=True))
        cube = Node("pCube1")
        (cube.t << AnimCurve([0, 1])) << lock
        self.assertTrue(cmds.getAttr("pCube1.translate", lock=True))
        self.assertTrue(cmds.getAttr("pCube1.tx", lock=True))


# --------------------------------------------------------------------------------------- #
#  A plug that is driven (OPEN (b), D1 = A)
# --------------------------------------------------------------------------------------- #


class TestConnected(_Case):
    def _each_route(self, body):
        for api in (True, False):
            with self.subTest(api=api):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                with mock.patch.object(journal, "API_EDITS", api):
                    body()

    def _replaced(self, make, plug, others=()):
        def body():
            driver = make()
            kept = {o: _inputs(o) for o in others}
            nodes = _nodes()
            p = Plug(plug)
            self.assertIs(p << AnimCurve([0, 10], times=[0, 10]), p)
            curve = _curve_of(plug)
            self.assertIsNotNone(curve)
            self.assertEqual(_inputs(plug), [f"{curve}.output"])
            self.assertEqual({o: _inputs(o) for o in others}, kept)
            self.assertEqual(cmds.ls(type="pairBlend"), [] if driver != "pairblend" else cmds.ls(type="pairBlend"))
            self.assertFalse(cmds.attributeQuery("blendParent2", node=plug.split(".")[0], exists=True))
            self.assertLessEqual(nodes, _nodes(), "a node was deleted")
            self.assertEqual(sorted(_nodes() - nodes), [curve])
            self.assertAlmostEqual(cmds.getAttr(plug, time=5), 5.0)
            self.assertEqual(self.undo_name(), "rig.animcurve")
            self.undo_steps(1)
            self.assertIsNone(_curve_of(plug) if driver != "pairblend" else None)
            self.redo_steps(1)
            self.assertEqual(_inputs(plug), [f"{curve}.output"])

        self._each_route(body)

    def test_a_parent_constraint(self):
        def make():
            _cube("c").t << [3, 4, 6]
            _cube("a")
            cmds.parentConstraint("c", "a")
            cmds.flushUndo()
            return "constraint"

        self._replaced(make, "a.translateX", ["a.translateY", "a.translateZ", "a.rotateX"])

    def test_a_plain_connection_and_an_expression(self):
        def plain():
            _cube("c")
            _cube("a")
            cmds.connectAttr("c.ty", "a.tx")
            cmds.flushUndo()
            return "plain"

        def expression():
            _cube("a")
            cmds.expression(s="a.tx = frame * 2;", name="ex")
            cmds.flushUndo()
            return "expression"

        self._replaced(plain, "a.translateX")
        self._replaced(expression, "a.translateX")

    def test_a_unit_conversion(self):
        def make():
            _cube("c")
            _cube("a")
            cmds.connectAttr("c.tx", "a.rz")
            assert cmds.nodeType(cmds.listConnections("a.rz", s=True, d=False)[0]) == "unitConversion"
            cmds.flushUndo()
            return "unitConversion"

        self._replaced(make, "a.rotateZ")

    def test_a_maya_pair_blend(self):
        def make():
            _cube("c").t << [3, 4, 6]
            _cube("a")
            cmds.parentConstraint("c", "a")
            cmds.setKeyframe("a.tx", t=0, v=7)
            assert _inputs("a.tx")[0].startswith("pairBlend")
            cmds.flushUndo()
            return "pairblend"

        def body():
            make()
            pb = cmds.ls(type="pairBlend")
            nodes = _nodes()
            Plug("a.tx") << AnimCurve([0, 10])
            self.assertEqual(_inputs("a.tx"), ["a_translateX.output"])
            self.assertEqual(cmds.ls(type="pairBlend"), pb)
            self.assertTrue(_inputs("a.ty")[0].startswith("pairBlend"))
            self.assertEqual(sorted(_nodes() - nodes), ["a_translateX"])
            self.assertLessEqual(nodes, _nodes())

        self._each_route(body)

    def test_no_pair_blend_and_no_blend_parent_is_ever_made(self):
        _cube("c")
        _cube("a")
        cmds.parentConstraint("c", "a")
        Node("a").t << AnimCurve([0, 1])
        Node("a").r << AnimCurve([0, 1])
        self.assertEqual(cmds.ls(type="pairBlend"), [])
        self.assertFalse(cmds.attributeQuery("blendParent1", node="a", exists=True))
        self.assertEqual(len(AnimCurve.of(Node("a"))), 6)

    def test_a_compound_connected_whole_gives_up_that_input(self):
        _cube("c")
        a = _cube("a")
        cmds.connectAttr("c.t", "a.t")
        a.t << AnimCurve([0, 1])
        self.assertEqual(_inputs("a.t"), [])
        self.assertEqual(_inputs("a.tx"), ["a_translateX.output"])

    def test_a_locked_plug_and_a_locked_child(self):
        a = _cube("a")
        cmds.setAttr("a.tx", lock=True)
        self.assertRefusesWithNothingChanged(InjectionError, lambda: a.tx << AnimCurve([0, 1]), "Plug('a.translateX').unlock()")
        self.assertRefusesWithNothingChanged(InjectionError, lambda: a.t << AnimCurve([0, 1]), "locked")
        self.assertRefusesWithNothingChanged(InjectionError, lambda: List([a.ty, a.tx]) << AnimCurve([0, 1]), "locked")
        self.assertIsNone(_curve_of("a.ty"))


# --------------------------------------------------------------------------------------- #
#  Re-run
# --------------------------------------------------------------------------------------- #


class TestReRun(_Case):
    def test_the_own_curve_is_updated_in_place(self):
        n = _cube("n")
        n.tx << AnimCurve([1, 2, 3])
        uuid = cmds.ls("n_translateX", uuid=True)
        cmds.flushUndo()
        nodes = _nodes()
        n.tx << AnimCurve([5, 6], times=[0, 10], infinity="cycle")
        self.assertEqual(cmds.ls("n_translateX", uuid=True), uuid)
        self.assertEqual(_nodes(), nodes)
        self.assertEqual(_values("n_translateX"), [5.0, 6.0])
        self.assertEqual(_times("n_translateX"), [0.0, 10.0])
        self.assertEqual(cmds.getAttr("n_translateX.preInfinity"), 3)
        self.assertEqual(self.undo_name(), "rig.animcurve")
        self.undo_steps(1)
        self.assertEqual(_values("n_translateX"), [1.0, 2.0, 3.0])
        self.assertEqual(cmds.getAttr("n_translateX.preInfinity"), 0)
        self.redo_steps(1)
        self.assertEqual(_values("n_translateX"), [5.0, 6.0])

    def test_what_the_keys_do_not_state_is_kept(self):
        n = _cube("n")
        n.rx << AnimCurve([1, 2])
        cmds.setAttr("n_rotateX.curveColor", 1, 0, 0)
        cmds.setAttr("n_rotateX.useCurveColor", True)
        cmds.rename("n_rotateX", "spin")
        n.rx << AnimCurve([3, 4])
        self.assertEqual(_curve_of("n.rx"), "spin")
        self.assertTrue(cmds.getAttr("spin.useCurveColor"))
        np.testing.assert_allclose(_values("spin"), [3.0, 4.0])

    def test_unchanged_keys_write_nothing(self):
        n = _cube("n")
        keys = AnimCurve([0, 1, 0], times=[0, 5, 10], in_slopes=[0, np.nan, 0], weighted=False, infinity="cycle")
        n.tx << keys
        n.r << AnimCurve(np.arange(6.0).reshape(2, 3))
        n.sx << AnimCurve([0, 1], weighted=True, out_weights=[2, np.nan], in_slopes=[np.nan, 3])
        cmds.flushUndo()
        with mock.patch.object(anim_keys, "_write", side_effect=AssertionError("an edit")):
            n.tx << keys
            n.r << AnimCurve(np.arange(6.0).reshape(2, 3))
            List([n.tx]) << keys
            n.sx << AnimCurve([0, 1], weighted=True, out_weights=[2, np.nan], in_slopes=[np.nan, 3])
            n.sx << AnimCurve("n_scaleX").get_keys()
        self.assertTrue(_empty_queue())

    def test_a_shared_curve_gives_a_new_one(self):
        n = _cube("n")
        n << Float("foo") << Float("bar")
        n.foo << AnimCurve([1, 2])
        cmds.connectAttr("n_foo.output", "n.bar")
        n.foo << AnimCurve([7, 8])
        self.assertEqual(_curve_of("n.foo"), "n_foo1")
        self.assertEqual(_curve_of("n.bar"), "n_foo")
        self.assertEqual(_values("n_foo"), [1.0, 2.0])
        uuid = cmds.ls("n_foo1", uuid=True)
        n.foo << AnimCurve([9])
        self.assertEqual(cmds.ls(_curve_of("n.foo"), uuid=True), uuid)

    def test_another_type_and_a_driven_key_give_a_new_one(self):
        n = _cube("n")
        cmds.createNode("animCurveTU", name="wrong")
        cmds.connectAttr("wrong.output", "n.tx")
        n.tx << AnimCurve([1, 2])
        self.assertEqual(_curve_of("n.tx"), "n_translateX")
        self.assertTrue(cmds.objExists("wrong"))
        drv = _cube("drv")
        cmds.setDrivenKeyframe("n.ty", currentDriver="drv.tx", driverValue=0, value=0)
        driven = _curve_of("n.ty")
        n.ty << AnimCurve([1, 2])
        self.assertNotEqual(_curve_of("n.ty"), driven)
        self.assertEqual(cmds.nodeType(_curve_of("n.ty")), "animCurveTL")
        self.assertTrue(cmds.objExists(driven))

    def test_a_referenced_curve_gives_a_new_one(self):
        folder = tempfile.mkdtemp()
        try:
            path = os.path.join(folder, "keyed.ma").replace("\\", "/")
            _cube("ctl")
            cmds.setKeyframe("ctl.tx", t=0, v=1)
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            self.new_scene()
            cmds.file(path, reference=True, namespace="ref")
            curve = _curve_of("ref:ctl.tx")
            self.assertTrue(cmds.referenceQuery(curve, isNodeReferenced=True))
            Node("ref:ctl").tx << AnimCurve([5, 6])
            self.assertEqual(_curve_of("ref:ctl.tx"), "ctl_translateX")
            self.assertEqual(_values(curve), [1.0])
        finally:
            self.new_scene()
            shutil.rmtree(folder, ignore_errors=True)

    def test_d3_a_locked_plug_refuses_its_own_curve_and_a_new_connect(self):
        n = _cube("n")
        keys = AnimCurve([0, 1])
        n.tx << keys << lock
        # the keys it holds, other keys, and a plug with no curve yet: all refused (D3 = b)
        for k in (keys, AnimCurve([3, 4])):
            with self.subTest(keys=repr(k)):
                self.assertRefusesWithNothingChanged(
                    InjectionError, lambda: n.tx << k, "Plug('n.translateX').unlock()", "even the one it holds")
        n.ty << lock
        self.assertRefusesWithNothingChanged(InjectionError, lambda: n.ty << AnimCurve([0, 1]), "locked")
        self.assertEqual(_values("n_translateX"), [0.0, 1.0])
        n.tx << rig.unlock
        n.tx << AnimCurve([3, 4])
        self.assertEqual(_values("n_translateX"), [3.0, 4.0])

    def test_a_locked_curve_node_refuses(self):
        n = _cube("n")
        n.tx << AnimCurve([0, 1])
        cmds.lockNode("n_translateX", lock=True)
        self.assertRefusesWithNothingChanged(InjectionError, lambda: n.tx << AnimCurve([3, 4]), "it is locked", "unlock()")
        cmds.lockNode("n_translateX", lock=False)


# --------------------------------------------------------------------------------------- #
#  Undo
# --------------------------------------------------------------------------------------- #


class TestUndo(_Case):
    def test_one_step_per_statement(self):
        n = _cube("n")
        a, b = Node.create("transform", name="a"), Node.create("transform", name="b")
        cmds.flushUndo()
        statements = [
            lambda: n.tx << AnimCurve([0, 1]),
            lambda: n.r << AnimCurve(np.zeros((3, 3))),
            lambda: List([a, b]).ry << AnimCurve([0, 45, 0]),
            lambda: List([a, b]).rz << [AnimCurve([1, 2]), AnimCurve([3, 4])],
            lambda: n.s << [AnimCurve([1, 2]), skip, 3],
        ]
        for fn in statements:
            before = self.scene_state()
            cmds.flushUndo()
            fn()
            after = self.scene_state()
            self.assertEqual(self.undo_name(), "rig.animcurve")
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), before)
            self.assertTrue(_empty_queue())
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(), after)

    def test_redo_brings_back_the_same_nodes(self):
        n = _cube("n")
        cmds.flushUndo()
        n.t << AnimCurve([0, 1])
        uuids = sorted(cmds.ls(type="animCurveTL", uuid=True))
        self.undo_steps(1)
        self.assertEqual(cmds.ls(type="animCurveTL"), [])
        self.redo_steps(1)
        self.assertEqual(sorted(cmds.ls(type="animCurveTL", uuid=True)), uuids)
        self.assertEqual(_values("n_translateY"), [0.0, 1.0])

    def test_a_refused_row_writes_nothing(self):
        a, b = Node.create("transform", name="a"), Node.create("transform", name="b")
        cmds.setAttr("b.ry", lock=True)
        self.assertRefusesWithNothingChanged(InjectionError, lambda: List([a, b]).ry << AnimCurve([0, 1]), "locked")
        self.assertRefusesWithNothingChanged(TypeError, lambda: List([a, b]).rz << [AnimCurve([1, 2]), 5], "one '<<' per kind")
        self.assertRefusesWithNothingChanged(TypeError, lambda: List([a, b]) << AnimCurve([1, 2]), "not a plug")
        self.assertRefusesWithNothingChanged(
            ValueError, lambda: List([a.t, b.tx]) << [AnimCurve([1, 2]), AnimCurve(np.zeros((2, 3)))], "for 1 channel")

    def test_without_the_undo_plugin_nothing_is_made(self):
        n = _cube("n")
        from rig.nodetypes import plugins

        with mock.patch.object(plugins, "_fast_path_ready", return_value=False):
            self.assertRefusesWithNothingChanged(RuntimeError, lambda: n.tx << AnimCurve([0, 1]), "undo plug-in")
            self.assertRefusesWithNothingChanged(RuntimeError, lambda: AnimCurveTU.create([0, 1], name="x"), "undo plug-in")

    def test_an_error_mid_statement_keeps_undo_working(self):
        n = _cube("n")
        cmds.flushUndo()
        real = anim_keys._write
        calls = []

        def second_fails(fn, e):
            calls.append(e)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return real(fn, e)

        with mock.patch.object(anim_keys, "_write", side_effect=second_fails):
            with self.assertRaises(RuntimeError):
                n.t << AnimCurve([0, 1])
        self.assertEqual(self.undo_name(), "rig.animcurve")
        self.undo_all()
        self.assertEqual(cmds.ls(type="animCurve"), [])
        self.assert_no_dangling()


# --------------------------------------------------------------------------------------- #
#  The node classes and read-back
# --------------------------------------------------------------------------------------- #


class TestNodeClasses(_Case):
    def test_refer_and_the_cast(self):
        n = _cube("n")
        n.tx << AnimCurve([0, 1])
        self.assertIs(type(Node("n_translateX")), AnimCurveTL)
        self.assertIs(type(AnimCurve("n_translateX")), AnimCurveTL)
        self.assertIs(type(AnimCurveTL("n_translateX")), AnimCurveTL)
        with self.assertRaises(NodeTypeError) as ctx:
            AnimCurveTA("n_translateX")
        self.assertIn("AnimCurve('n_translateX') takes any curve", str(ctx.exception))
        drv = _cube("drv")
        cmds.setDrivenKeyframe("n.sy", currentDriver="drv.tx", driverValue=0, value=1)
        uu = _curve_of("n.sy")
        self.assertIs(type(Node(uu)), AnimCurve)
        self.assertIs(type(AnimCurve(uu)), AnimCurve)
        with self.assertRaises(NodeTypeError):
            AnimCurve("n")

    def test_a_typed_class_called_with_values_is_a_type_error(self):
        for cls in (AnimCurveTL, AnimCurveTA, AnimCurveTU, AnimCurveTT):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(TypeError) as ctx:
                    cls([1, 2])
                self.assertIn("AnimCurve([...]) makes keys", str(ctx.exception))
                self.assertIn(f"{cls.__name__}.create([...])", str(ctx.exception))

    def test_the_abstract_class_makes_no_curve(self):
        for fn in (lambda: AnimCurve.create([0, 1]), lambda: AnimCurve.define("x", [0, 1]),
                   lambda: AnimCurve.create(name="x")):
            self.assertRefusesWithNothingChanged(TypeError, fn, "abstract")

    def test_create(self):
        c = AnimCurveTU.create([0, 1, 0], name="blink")
        self.assertIs(type(c), AnimCurveTU)
        self.assertEqual((c.get_name(), _values("blink"), _inputs("blink.input")), ("blink", [0.0, 1.0, 0.0], []))
        self.assertEqual(cmds.listConnections("blink.output") or [], [])
        c2 = AnimCurveTU.create(AnimCurve([0, 1, 0], times=[0, 2, 4]), name="blink")
        self.assertEqual(c2.get_name(), "blink1")
        self.assertEqual(_times("blink1"), [0.0, 2.0, 4.0])
        self.assertRefusesWithNothingChanged(
            TypeError, lambda: AnimCurveTU.create(AnimCurve([0, 1]), times=[0, 1], name="x"), "holds its own times")
        self.assertRefusesWithNothingChanged(
            TypeError, lambda: AnimCurveTU.create(AnimCurve([0, 1]), tangents="step", name="x"), "holds its own")
        empty = AnimCurveTL.create(name="empty")
        self.assertEqual(empty.get_num_keys(), 0)
        by_type = Node.create("animCurveTU", name="byType")
        self.assertIs(type(by_type), AnimCurveTU)
        self.assertEqual(by_type.get_num_keys(), 0)
        # attributes are set after the keys: an attribute wins over the keys' infinity
        c3 = AnimCurveTA.create([0, 1], infinity="cycle", preInfinity=1, name="spin")
        self.assertEqual((cmds.getAttr("spin.preInfinity"), cmds.getAttr("spin.postInfinity")), (1, 3))
        self.assertRefusesWithNothingChanged(AttributeError, lambda: AnimCurveTA.create([0, 1], noSuchAttr=1), "noSuchAttr")
        self.assertRefusesWithNothingChanged(TypeError, lambda: AnimCurveTU.create(["a"]), "field names")
        cmds.flushUndo()
        AnimCurveTU.create([0, 1], name="one")
        self.assertEqual(self.undo_name(), "rig.create")
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("one"))

    def test_define(self):
        c = AnimCurveTU.define("blink", [0, 1, 0], times=[0, 2, 4], tangents="step")
        self.assertIs(type(c), AnimCurveTU)
        uuid = cmds.ls("blink", uuid=True)
        self.assertEqual(cmds.keyTangent("blink", query=True, ott=True), ["step"] * 3)
        cmds.flushUndo()
        again = AnimCurveTU.define("blink", [0, 1, 0], times=[0, 2, 4], tangents="step")
        self.assertEqual(again, c)
        self.assertTrue(_empty_queue(), "unchanged keys wrote a step")
        AnimCurveTU.define("blink", [0, 5, 0], times=[0, 2, 4])
        self.assertEqual(cmds.ls("blink", uuid=True), uuid)
        self.assertEqual(_values("blink"), [0.0, 5.0, 0.0])
        self.assertEqual(self.undo_name(), "rig.define")
        AnimCurveTU.define("blink", [9, 9], update=False)
        self.assertEqual(_values("blink"), [0.0, 5.0, 0.0])
        AnimCurveTU.define("blink", postInfinity=4)
        self.assertEqual(cmds.getAttr("blink.postInfinity"), 4)
        with self.assertRaises(NodeTypeError) as ctx:
            AnimCurveTL.define("blink", [0, 1])
        self.assertIn("AnimCurve('blink') takes any curve", str(ctx.exception))
        Node.define("animCurveTA", "spin")
        self.assertIs(type(Node("spin")), AnimCurveTA)

    def test_set_keys_num_keys_evaluate(self):
        c = AnimCurveTL.create([0, 1], name="c")
        cmds.flushUndo()
        self.assertIs(c.set_keys([0, 3], times=[0, 10]), c)
        self.assertEqual(self.undo_name(), "rig.AnimCurve.set_keys")
        self.assertEqual(c.get_num_keys(), 2)
        np.testing.assert_allclose(c.evaluate([0, 5, 10]), [0, 1.5, 3])
        self.assertIsInstance(c.evaluate(5), float)
        cmds.flushUndo()
        c.set_keys(AnimCurve([0, 3], times=[0, 10]))
        self.assertTrue(_empty_queue())
        self.undo_queue_on()
        cmds.lockNode("c", lock=True)
        self.assertRefusesWithNothingChanged(InjectionError, lambda: c.set_keys([1, 2]), "it is locked")
        cmds.lockNode("c", lock=False)
        with self.assertRaises(TypeError):
            c.set_keys(None)

    def test_of(self):
        n = _cube("n")
        n << Float("arr", multi=True)
        n.t << AnimCurve([0, 1])
        n.arr[0] << AnimCurve([0, 1])
        n.arr[3] << AnimCurve([0, 1])
        n.sx << 2
        self.assertEqual([c.get_name() for c in AnimCurve.of(n.tx)], ["n_translateX"])
        self.assertEqual([c.get_name() for c in AnimCurve.of(n.t)], ["n_translateX", "n_translateY", "n_translateZ"])
        self.assertEqual([c.get_name() for c in AnimCurve.of(n.arr)], ["n_arr_0_", "n_arr_3_"])
        self.assertEqual(len(AnimCurve.of(n.s)), 0)
        self.assertEqual(sorted(c.get_name() for c in AnimCurve.of(n)),
                         ["n_arr_0_", "n_arr_3_", "n_translateX", "n_translateY", "n_translateZ"])
        self.assertEqual(sorted(c.get_name() for c in AnimCurveTU.of(n)), ["n_arr_0_", "n_arr_3_"])
        both = AnimCurve.of(List([n.tx, n.tx, n.ty]))
        self.assertEqual([c.get_name() for c in both], ["n_translateX", "n_translateY"])
        self.assertIsInstance(both, List)
        # a curve behind a pairBlend is not looked through
        _cube("c")
        cmds.parentConstraint("c", "n")
        self.assertEqual(len(AnimCurve.of(Node("c"))), 0)

    def test_a_node_or_a_shift_with_keys_is_a_type_error(self):
        n = _cube("n")
        self.assertRefusesWithNothingChanged(TypeError, lambda: n << AnimCurve([1]), "n.<attr> << AnimCurve(...)")
        self.assertRefusesWithNothingChanged(TypeError, lambda: n.tx >> AnimCurve([1]), "not a form")
        self.assertRefusesWithNothingChanged(TypeError, lambda: n >> AnimCurve([1]), "not a form")
        self.assertRefusesWithNothingChanged(TypeError, lambda: List([n.tx]) >> AnimCurve([1]), "not a form")


class TestReadBack(_Case):
    def test_get_keys_round_trip(self):
        n = _cube("n")
        m = _cube("m")
        n.tx << AnimCurve([0, 2, 1, 3], times=[0, 4, 9, 12], in_tangents=["flat", "fixed", "spline", "step"],
                          out_tangents=["linear", "fixed", "clamped", "flat"], in_slopes=[np.nan, 0.5, np.nan, np.nan],
                          out_slopes=[np.nan, -1.25, np.nan, np.nan], pre_infinity="cycle", post_infinity="oscillate")
        n.ry << AnimCurve([0, 90, 45], weighted=True, in_weights=[np.nan, 2, np.nan], out_weights=[1.5, 3, np.nan],
                          out_slopes=[np.nan, 4, np.nan])
        for src, dst in (("n.tx", m.tx), ("n.ry", m.ry)):
            with self.subTest(src=src):
                keys = AnimCurve.of(Plug(src))[0].get_keys()
                dst << keys
                a, b = _curve_of(src), _curve_of(str(dst))
                for flag in ("valueChange", "timeChange"):
                    np.testing.assert_allclose(cmds.keyframe(a, q=True, **{flag: True}), cmds.keyframe(b, q=True, **{flag: True}))
                for flag in ("itt", "ott", "ia", "oa", "iw", "ow"):
                    got, want = cmds.keyTangent(b, q=True, **{flag: True}), cmds.keyTangent(a, q=True, **{flag: True})
                    if flag in ("itt", "ott"):
                        self.assertEqual(got, want, flag)
                    else:
                        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=flag)
                for attr in ("preInfinity", "postInfinity", "weightedTangents"):
                    self.assertEqual(cmds.getAttr(f"{a}.{attr}"), cmds.getAttr(f"{b}.{attr}"), attr)
                for t in (-3, 2, 7, 15):
                    self.assertAlmostEqual(cmds.getAttr(src, time=t), cmds.getAttr(str(dst), time=t), places=6)
                cmds.flushUndo()
                dst << keys
                self.assertTrue(_empty_queue())
        k = AnimCurve("n_translateX").get_keys()
        np.testing.assert_allclose(k.out_slopes[[1]], [-1.25])
        self.assertEqual((k.pre_infinity, k.post_infinity), ("cycle", "oscillate"))

    def test_an_empty_curve_has_no_keys_value(self):
        c = AnimCurveTU.create(name="empty")
        with self.assertRaises(ValueError):
            c.get_keys()

    def test_a_driven_key_is_not_read_by_time(self):
        n, drv = _cube("n"), _cube("drv")
        cmds.setDrivenKeyframe("n.sy", currentDriver="drv.tx", driverValue=0, value=1)
        uu = AnimCurve(_curve_of("n.sy"))
        self.assertEqual(uu.get_num_keys(), 1)
        for fn in (uu.get_keys, lambda: uu.set_keys([1]), lambda: uu.evaluate(1)):
            with self.assertRaises(TypeError):
                fn()


# --------------------------------------------------------------------------------------- #
#  Naming, containers, selection
# --------------------------------------------------------------------------------------- #


class TestNaming(_Case):
    def test_maya_names_and_a_clash(self):
        n = _cube("pCube1")
        n << Float("foo")
        n.foo << AnimCurve([0])
        cmds.createNode("transform", name="pCube1_translateX")
        n.tx << AnimCurve([0])
        self.assertEqual(_curve_of("pCube1.foo"), "pCube1_foo")
        self.assertEqual(_curve_of("pCube1.tx"), "pCube1_translateX1")
        # Maya's own names, for comparison
        m = _cube("m")
        cmds.setKeyframe("m.ty", t=0, v=0)
        self.assertEqual(_curve_of("m.ty"), "m_translateY")

    def test_the_current_namespace(self):
        cmds.namespace(add="char")
        Node(cmds.polyCube(name="char:ctl", ch=False)[0]).tx << AnimCurve([0, 1])
        self.assertEqual(_curve_of("char:ctl.tx"), "ctl_translateX")
        other = _cube("other")
        cmds.namespace(set="char")
        try:
            other.tx << AnimCurve([0, 1])
        finally:
            cmds.namespace(set=":")
        self.assertEqual(_curve_of("other.tx"), "char:other_translateX")
        # Maya's setKeyframe does the same
        cmds.namespace(set="char")
        try:
            cmds.setKeyframe("other.ty", t=0, v=0)
        finally:
            cmds.namespace(set=":")
        self.assertEqual(_curve_of("other.ty"), "char:other_translateY")

    def test_a_container_scope(self):
        set_options(flatten_containers=False)
        with container("arm"):
            ctl = Node.create("transform", name="ctl")
            ctl.tx << AnimCurve([0, 1])
        self.assertEqual(_curve_of(f"{ctl}.tx"), f"{ctl.get_name()}_translateX")
        members = cmds.container("arm", query=True, nodeList=True) or []
        self.assertIn(_curve_of(f"{ctl}.tx"), members)
        set_options(flatten_containers=True)
        with container("leg"):
            inner = Node.create("transform", name="ctl")
            inner.ty << AnimCurve([0, 1])
        self.assertEqual(_curve_of(f"{inner}.ty"), f"{inner.get_name()}_translateY")
        # a curve is data: never a cleanup() candidate
        from rig._internal import container as container_module

        self.assertFalse({"animCurveTL", "animCurveTA", "animCurveTU", "animCurveTT"} & container_module._GC_ELIGIBLE_TYPES)

    def test_the_selection_is_kept(self):
        n = _cube("n")
        cmds.select("n")
        n.t << AnimCurve([0, 1])
        AnimCurveTU.define("x", [0, 1])
        self.assertEqual(cmds.ls(selection=True), ["n"])


# --------------------------------------------------------------------------------------- #
#  The critic's crash case (probe_elem_crash.py)
# --------------------------------------------------------------------------------------- #


class TestTheElementCrash(_Case):
    def test_keys_into_a_driven_element_of_a_non_keyable_multi(self):
        for spec in (Float("w", multi=True, keyable=False), Typed("w", multi=True)):
            for api in (True, False):
                with self.subTest(spec=type(spec).__name__, api=api):
                    self.new_scene()
                    self.undo_queue_restore()
                    self.undo_queue_on()
                    with mock.patch.object(journal, "API_EDITS", api):
                        a = _cube("a")
                        net = Node.create("network", name="net")
                        net << spec
                        held = net.w[0]
                        held << a.tx
                        self.assertIs(held << AnimCurve([0, 1, 2]), held)
                        self.assertEqual(_inputs("net.w[0]"), ["net_w_0_.output"])
                        self.assertEqual(held.get_inputs()[0], Plug("net_w_0_.output"))
                        held << AnimCurve([3, 4])
                        self.assertEqual(_values("net_w_0_"), [3.0, 4.0])
                        held << a.ty
                        self.assertEqual(_inputs("net.w[0]"), ["a.translateY"])
                        self.undo_all()
                        self.redo_all()
                        self.assertEqual(_inputs("net.w[0]"), ["a.translateY"])
