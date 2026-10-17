"""``NurbsCurve.create`` builds a curve from control points, or from a cgmath
``BSplineData``.

The curve rides a recorded ``createNode`` transform, as ``Mesh.create``'s mesh
does: the shape is made under it through the API (``MFnNurbsCurve.create``),
then the transform is renamed. The whole create is one undo step named
``rig.NurbsCurve.create``; it never changes the selection and leaves no
history. Points are stored as given (Maya's internal unit), and with no ``kv``
the knots are the ones ``cmds.curve`` gives.

Undo tests check the scene (``rig._tests._undo``), never ``cmds.undo()``'s
return value.
"""

import contextlib
import math
import unittest
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig import Node, container
from rig.nodetypes import NurbsCurve, NurbsSurface
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

CHUNK = "rig.NurbsCurve.create"

# six points that lie on no line or plane
POINTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.5), (2.0, 1.0, 0.0), (3.0, 1.0, 1.0), (4.0, 0.0, 0.0), (5.0, 0.5, 1.0)]

OPEN     = OpenMaya.MFnNurbsCurve.kOpen
PERIODIC = OpenMaya.MFnNurbsCurve.kPeriodic


def _bspline_data(points, degree, periodic, **kwargs):
    from cgmath.geometry import BSplineData

    return BSplineData(points=np.asarray(points, dtype=float), degree=degree, periodic=periodic, **kwargs)


def _takes_knots() -> bool:
    """True when the installed cgmath's BSplineData takes custom ``knots``."""
    from cgmath.geometry import BSplineData

    return "knots" in BSplineData.__dataclass_fields__


class _BSplineDataWithoutKnots:
    """A BSplineData without a ``knots`` field, as cgmath 1.0.4's: what serialize gives it."""

    __dataclass_fields__ = dict.fromkeys(("points", "degree", "periodic"))

    def __init__(self, points, degree, periodic):
        self.points, self.degree, self.periodic = points, degree, periodic


def _line(count):
    """`count` points that zigzag along X."""
    return [(float(i), float(i % 2), 0.25 * i) for i in range(count)]


def _ring(count=8):
    return [(math.cos(2 * math.pi * i / count), 0.1 * i, math.sin(2 * math.pi * i / count)) for i in range(count)]


# -- queries


def curve_state(name):
    """The data of the curve `name` (a shape, or its transform), or None when
    there is none: degree, form, knots and object-space CVs, as Maya holds them."""
    sel = OpenMaya.MSelectionList()
    try:
        sel.add(str(name))
        path = sel.getDagPath(0)
    except (RuntimeError, TypeError):
        return None
    if path.apiType() == OpenMaya.MFn.kTransform:
        try:
            path.extendToShape()
        except RuntimeError:
            return None
    if not path.hasFn(OpenMaya.MFn.kNurbsCurve):
        return None
    fn = OpenMaya.MFnNurbsCurve(path)
    return {
        "degree": fn.degree,
        "form": fn.form,
        "knots": tuple(fn.knots()),
        "cvs": tuple((p.x, p.y, p.z) for p in fn.cvPositions(OpenMaya.MSpace.kObject)),
    }


def _on_curve(curve, params):
    """The object-space points of `curve` at `params`."""
    fn = curve.get_fn_set()
    return np.array([list(fn.getPointAtParam(float(u), OpenMaya.MSpace.kObject))[:3] for u in params])


def _rows(points):
    return tuple(tuple(float(x) for x in row) for row in points)


class _CurveCreateCase(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        try:
            cmds.namespace(set=":")
            self.assertFalse(container.is_active)
        finally:
            super().tearDown()

    def ready(self):
        """Start the test's own steps here: select the probe transform ``selprobe``
        (made when missing), then flush what building the sources recorded."""
        if not cmds.objExists("selprobe"):
            cmds.createNode("transform", name="selprobe")
        cmds.select("selprobe", replace=True)
        cmds.flushUndo()

    def state(self, *curves):
        """`UndoWalk.scene_state` with the data of the curves named."""
        state = self.scene_state()
        state["curves"] = {name: curve_state(name) for name in curves}
        return state

    def assert_same(self, actual, expected, msg=""):
        self.assert_state_equal(actual, expected, msg)
        self.assertEqual(actual["curves"], expected["curves"], f"{msg} curves")

    def check(self, curve, xform, shape):
        """`curve` is the NurbsCurve of `shape` under `xform`, alone under a
        transform at rest, with no history; the selection is the probe's."""
        self.assertIs(type(curve), NurbsCurve)
        self.assertEqual(str(curve), shape)
        parent = curve.get_parent()
        self.assertEqual(cmds.ls(str(parent))[0], xform)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])
        self.assertEqual(cmds.listRelatives(parent.get_long_name(), children=True, fullPath=True), [curve.get_long_name()])
        self.assertFalse(cmds.getAttr(curve.get_long_name() + ".intermediateObject"))
        self.assertEqual(cmds.listHistory(curve.get_long_name()), [cmds.ls(curve.get_long_name())[0]])
        self.assertIsNone(cmds.listConnections(curve.get_long_name(), source=True, destination=False))
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(cmds.xform(parent.get_long_name(), query=True, matrix=True, objectSpace=True), identity)


# ---------------------------------------------------------------------------------------------
class TestNurbsCurveCreatePoints(_CurveCreateCase):
    """Points, with a degree (3) and knots (Maya's default): the curve ``cmds.curve`` makes."""

    def test_default_is_the_cubic_cmds_curve_makes(self):
        self.ready()
        curve = NurbsCurve.create(POINTS)
        self.check(curve, "curve1", "curveShape1")
        state = curve_state(curve)
        self.assertEqual(state["degree"], 3)
        self.assertEqual(state["form"], OPEN)
        self.assertEqual(state["knots"], (0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0))
        self.assertEqual(state["cvs"], _rows(POINTS))
        self.assertEqual(curve.get_num_cvs(), 6)
        self.assertEqual(state, curve_state(cmds.curve(point=POINTS)))

    def test_each_degree_matches_cmds_curve(self):
        for degree in (1, 2, 3, 4, 5, 6, 7):
            for count in (degree + 1, degree + 2, degree + 5):
                with self.subTest(degree=degree, count=count):
                    cmds.file(new=True, force=True)
                    points = _line(count)
                    self.ready()
                    curve = NurbsCurve.create(points, degree=degree, name="c")
                    self.check(curve, "c", "cShape")
                    reference = curve_state(cmds.curve(point=points, degree=degree))
                    self.assertEqual(reference["degree"], degree)
                    self.assertEqual(curve_state(curve), reference)

    def test_kv(self):
        cases = (
            ("not clamped", 3, 6, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
            ("uneven", 3, 6, (0.0, 0.0, 0.0, 0.25, 2.5, 7.0, 7.0, 7.0)),
            ("not from zero", 2, 5, (5.0, 5.0, 6.5, 7.0, 9.0, 9.0)),
            ("an inner knot repeated degree times", 3, 8, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0)),
            ("degree 1", 1, 4, (0.0, 0.5, 4.0, 4.5)),
        )
        for label, degree, count, knots in cases:
            for kv in (knots, list(knots), np.array(knots)):
                with self.subTest(label, kv=type(kv).__name__):
                    cmds.file(new=True, force=True)
                    points = _line(count)
                    self.ready()
                    curve = NurbsCurve.create(points, degree=degree, kv=kv, name="c")
                    self.check(curve, "c", "cShape")
                    state = curve_state(curve)
                    self.assertEqual(state["knots"], knots)
                    self.assertEqual(state["degree"], degree)
                    self.assertEqual(state, curve_state(cmds.curve(point=points, degree=degree, knot=knots)))

    def test_point_inputs(self):
        source = cmds.curve(point=POINTS)
        expected = curve_state(source)["cvs"]
        inputs = {
            "tuples": POINTS,
            "lists": [list(point) for point in POINTS],
            "array": np.array(POINTS),
            "float32 array": np.array(POINTS, dtype=np.float32),
            "MPointArray": Node(source).get_shape().get_points(world_space=False),
            "MVectors": [OpenMaya.MVector(*point) for point in POINTS],
            "MPoints": [OpenMaya.MPoint(*point) for point in POINTS],
            "MFloatPointArray": OpenMaya.MFloatPointArray([OpenMaya.MFloatPoint(*point) for point in POINTS]),
            "homogeneous rows": [(*point, 1.0) for point in POINTS],
        }
        ints = [(0, 0, 0), (1, 0, 2), (2, 1, 0), (3, 1, 1)]
        self.ready()
        for label, points in inputs.items():
            with self.subTest(label):
                self.assertEqual(curve_state(NurbsCurve.create(points))["cvs"], expected)
        self.assertEqual(curve_state(NurbsCurve.create(ints))["cvs"], _rows(ints))
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def test_points_are_stored_as_given_in_any_linear_unit(self):
        # cmds.curve reads its points in the scene's linear unit; the API does not
        unit = cmds.currentUnit(query=True, linear=True)
        self.addCleanup(cmds.currentUnit, linear=unit)
        cmds.currentUnit(linear="m")
        self.ready()
        curve = NurbsCurve.create(POINTS, name="c")
        self.assertEqual(curve_state(curve)["cvs"], _rows(POINTS))
        points = np.array(curve.get_points(world_space=False))[:, :3]
        self.assertEqual(_rows(points), _rows(POINTS))
        by_command = curve_state(cmds.curve(point=POINTS))["cvs"]
        self.assertEqual(by_command[1], (100.0, 0.0, 50.0))  # metres, stored as centimetres

    def test_node_create_runs_it(self):
        self.ready()
        curve = Node.create("nurbsCurve", POINTS, degree=1, name="viaNode")
        self.check(curve, "viaNode", "viaNodeShape")
        self.assertEqual(curve_state(curve)["degree"], 1)
        self.assertEqual(self.undo_name(), CHUNK)


# ---------------------------------------------------------------------------------------------
class TestNurbsCurveCreateNames(_CurveCreateCase):
    """Names follow ``Mesh.create``'s rules: the transform is named, the shape follows it."""

    def test_no_name_twice(self):
        self.ready()
        self.check(NurbsCurve.create(POINTS), "curve1", "curveShape1")
        self.check(NurbsCurve.create(POINTS), "curve2", "curveShape2")

    def test_explicit_names(self):
        for name, xform, shape in (("c", "c", "cShape"), ("cShape", "c", "cShape"), ("cShape2", "c2", "cShape2")):
            with self.subTest(name=name):
                cmds.file(new=True, force=True)
                self.ready()
                self.check(NurbsCurve.create(POINTS, name=name), xform, shape)

    def test_name_taken(self):
        self.ready()
        self.check(NurbsCurve.create(POINTS, name="c"), "c", "cShape")
        self.check(NurbsCurve.create(POINTS, name="c"), "c1", "cShape1")
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="c")
        self.ready()
        self.check(NurbsCurve.create(POINTS, name="c"), "c1", "cShape1")

    def test_a_name_used_under_another_parent(self):
        cmds.createNode("transform", name="c", parent=cmds.createNode("transform", name="grp"))
        self.ready()
        curve = NurbsCurve.create(POINTS, name="c")
        self.assertEqual(curve.get_long_name(), "|c|cShape")
        self.assertEqual(curve.get_parent().get_long_name(), "|c")
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def test_namespace(self):
        cmds.namespace(add="ns")
        self.ready()
        self.check(NurbsCurve.create(POINTS, name="ns:c"), "ns:c", "ns:cShape")
        cmds.namespace(set="ns")
        self.check(NurbsCurve.create(POINTS), "ns:curve1", "ns:curveShape1")
        self.check(NurbsCurve.create(POINTS, name="d"), "ns:d", "ns:dShape")

    def test_name_in_nested_containers(self):
        self.ready()
        with container("outer") as outer:
            with container("inner"):
                curve = NurbsCurve.create(POINTS, name="c")
        self.check(curve, "inner_c", "inner_cShape")
        self.assertEqual(sorted(cmds.container(str(outer), query=True, nodeList=True)), ["inner_c", "inner_cShape"])

    def test_container_false_leaves_it_out_of_the_scope(self):
        self.ready()
        with container("outer") as outer:
            with container("inner"):
                curve = NurbsCurve.create(POINTS, name="c", container=False)
        self.check(curve, "inner_c", "inner_cShape")  # the prefix still applies
        self.assertIsNone(cmds.container(str(outer), query=True, nodeList=True))


# ---------------------------------------------------------------------------------------------
class TestNurbsCurveCreateKeywords(_CurveCreateCase):
    """The typed-create rules: ``name=`` is a keyword, the curve's command places it, and the
    keywords that are not ``degree`` / ``kv`` are the curve's attributes."""

    def test_name_is_a_keyword(self):
        self.ready()
        nodes = sorted(cmds.ls())
        with self.assertRaises(TypeError):
            NurbsCurve.create(POINTS, 3, None, "c")
        self.assertEqual(sorted(cmds.ls()), nodes)

    def test_parent_is_refused(self):
        cmds.createNode("transform", name="grp")
        self.ready()
        nodes = sorted(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"^NurbsCurve\.create\(\) takes no parent=: its command places"):
            NurbsCurve.create(POINTS, name="c", parent="grp")
        with self.assertRaisesRegex(TypeError, r"shared"):
            NurbsCurve.create(POINTS, name="c", shared=True)
        self.assertEqual(sorted(cmds.ls()), nodes)

    def test_other_keywords_are_attributes(self):
        self.ready()
        curve = NurbsCurve.create(POINTS, degree=2, name="c", lineWidth=3.0, dispCV=True)
        self.check(curve, "c", "cShape")
        self.assertEqual(curve_state(curve)["degree"], 2)  # the curve's degree, not the attribute's
        self.assertEqual(cmds.getAttr("cShape.lineWidth"), 3.0)
        self.assertTrue(cmds.getAttr("cShape.dispCV"))
        self.assertEqual(self.undo_name(), CHUNK)
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("c"))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_misspelled_attribute_makes_nothing(self):
        self.ready()
        nodes = sorted(cmds.ls())
        with self.assertRaisesRegex(AttributeError, r"NurbsCurve\.create\(\)"):
            NurbsCurve.create(POINTS, name="c", lineWidht=3.0)
        self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])


# ---------------------------------------------------------------------------------------------
class TestNurbsCurveCreateBSpline(_CurveCreateCase):
    """A BSplineData gives its points, degree, knots and form; ``uniform`` / ``registered`` are not used."""

    def test_open_data(self):
        for degree in (1, 2, 3, 5):
            with self.subTest(degree=degree):
                cmds.file(new=True, force=True)
                data = _bspline_data(POINTS, degree, False)
                self.ready()
                curve = NurbsCurve.create(data, name="c")
                self.check(curve, "c", "cShape")
                state = curve_state(curve)
                self.assertEqual(state["degree"], degree)
                self.assertEqual(state["form"], OPEN)
                self.assertEqual(state["cvs"], _rows(POINTS))
                self.assertEqual(state["knots"], tuple(float(k) for k in data.kv))
                # the knots cmds.curve gives, and the same curve as the points alone make
                self.assertEqual(state, curve_state(cmds.curve(point=POINTS, degree=degree)))
                self.assertEqual(state, curve_state(NurbsCurve.create(POINTS, degree=degree)))

    def test_periodic_data_repeats_its_first_points(self):
        for degree in (1, 2, 3):
            with self.subTest(degree=degree):
                cmds.file(new=True, force=True)
                ring = _ring(8)
                data = _bspline_data(ring, degree, True)
                self.ready()
                curve = NurbsCurve.create(data, name="ring")
                self.check(curve, "ring", "ringShape")
                state = curve_state(curve)
                self.assertEqual(state["degree"], degree)
                self.assertEqual(state["form"], PERIODIC)
                self.assertEqual(state["cvs"], _rows(ring + ring[:degree]))
                self.assertEqual(state["knots"], tuple(float(k) for k in range(1 - degree, 8 + degree)))
                self.assertEqual(curve.get_num_cvs(), 8 + degree)
                self.assertEqual(curve.get_num_weight_points(), 8)

    def test_periodic_data_rebuilds_a_maya_circle(self):
        circle = cmds.circle(sections=8, degree=3, constructionHistory=False)[0]
        source = curve_state(circle)
        self.assertEqual(source["form"], PERIODIC)
        data = _bspline_data(source["cvs"][:8], 3, True)  # the circle's own points, its wrap left out
        self.ready()
        curve = NurbsCurve.create(data, name="ring")
        self.check(curve, "ring", "ringShape")
        self.assertEqual(curve_state(curve), source)
        # a real periodic curve: a repeated CV follows its first copy
        cmds.move(0, 1, 0, "ring.cv[0]", relative=True)
        moved = curve_state(curve)["cvs"]
        self.assertEqual(moved[8], moved[0])
        self.assertNotEqual(moved[0], source["cvs"][0])

    def test_the_curve_is_the_data_s_curve(self):
        cases = [("open", _bspline_data(POINTS, 3, False)), ("open degree 2", _bspline_data(POINTS, 2, False))]
        cases += [(f"periodic degree {d}", _bspline_data(_ring(9), d, True)) for d in (1, 2, 3)]
        self.ready()
        for label, data in cases:
            with self.subTest(label):
                curve = NurbsCurve.create(data)
                params = np.linspace(0.0, float(data.max_param), 41)
                expected = np.asarray(data.compute(params)[0])
                self.assertLess(abs(_on_curve(curve, params) - expected).max(), 1e-9)

    def test_uniform_and_registered_are_not_used(self):
        self.ready()
        for periodic, points in ((False, POINTS), (True, _ring(8))):
            with self.subTest(periodic=periodic):
                plain = curve_state(NurbsCurve.create(_bspline_data(points, 3, periodic)))
                flagged = _bspline_data(points, 3, periodic, uniform=True, registered=True)
                self.assertEqual(curve_state(NurbsCurve.create(flagged)), plain)

    def test_degree_or_kv_with_data_is_refused(self):
        data = _bspline_data(POINTS, 3, False)
        self.ready()
        nodes = sorted(cmds.ls())
        for label, kwargs in (
            ("degree", dict(degree=3)),
            ("kv", dict(kv=[0, 0, 0, 1, 2, 3, 3, 3])),
            ("degree and kv", dict(degree=2, kv=[0, 0, 1, 2, 3, 4, 4])),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(TypeError, rf"got {label} with a BSplineData"):
                    NurbsCurve.create(data, **kwargs)
                self.assertEqual(sorted(cmds.ls()), nodes)

    def test_serialize_round_trip(self):
        for degree in (1, 2, 3):
            with self.subTest(degree=degree):
                cmds.file(new=True, force=True)
                source = cmds.curve(point=POINTS, degree=degree, name="src")
                cmds.xform(source, translation=(1, 2, 3), rotation=(10, 20, 30))
                expected = curve_state(source)
                data = NurbsCurve(source).serialize(world_space=False)
                self.ready()
                curve = NurbsCurve.create(data, name="rebuilt")
                self.check(curve, "rebuilt", "rebuiltShape")
                self.assertEqual(curve_state(curve), expected)

    @unittest.skipUnless(_takes_knots(), "this cgmath's BSplineData takes no knots")
    def test_data_with_its_own_knots(self):
        self.ready()
        uneven = (0.0, 0.0, 0.0, 0.5, 1.25, 3.0, 3.0, 3.0)
        data = _bspline_data(POINTS, 3, False, knots=np.array(uneven))
        curve = NurbsCurve.create(data, name="uneven")
        state = curve_state(curve)
        self.assertEqual(state["knots"], tuple(float(k) for k in data.kv))
        self.assertEqual(state["knots"], uneven)
        params = np.linspace(0.0, float(data.max_param), 41)
        self.assertLess(abs(_on_curve(curve, params) - np.asarray(data.compute(params)[0])).max(), 1e-9)

        # a periodic curve fitted through points: knots spaced by the chords
        for degree in (1, 2, 3):
            with self.subTest(degree=degree):
                fitted = _bspline_data(np.zeros((0, 3)), degree, True)
                fitted.fit(np.asarray(_ring(10)))
                curve = NurbsCurve.create(fitted)
                state = curve_state(curve)
                self.assertEqual(state["form"], PERIODIC)
                self.assertEqual(state["knots"], tuple(float(k) for k in fitted.kv))
                params = np.linspace(0.0, float(fitted.max_param), 61)
                expected = np.asarray(fitted.compute(params)[0])
                self.assertLess(abs(_on_curve(curve, params) - expected).max(), 1e-9)


# ---------------------------------------------------------------------------------------------
def _closed_form(name="closed"):
    """A curve of Maya's kClosed form (its ends meet, it is not periodic), made
    through the API, as imported curves can be."""
    points = POINTS + [POINTS[0]]
    knots  = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    xform  = cmds.createNode("transform", name=name)
    sel    = OpenMaya.MSelectionList()
    sel.add(xform)
    OpenMaya.MFnNurbsCurve().create(OpenMaya.MPointArray(points), OpenMaya.MDoubleArray(knots), 3,
                                    OpenMaya.MFnNurbsCurve.kClosed, False, False, sel.getDependNode(0))
    return xform


def _uneven_curves():
    """``{label: transform}``: Maya curves whose knots are not its default ones."""
    ep = cmds.curve(editPoint=POINTS, degree=3, name="ep")
    return {
        "EP curve": ep,
        "own knots": cmds.curve(point=POINTS, degree=3, knot=[5, 5, 5, 5.5, 7, 8, 8, 8], name="own"),
        "uneven periodic": cmds.closeCurve(cmds.curve(editPoint=POINTS, degree=3, name="loop"),
                                           preserveShape=0, replaceOriginal=True, constructionHistory=False)[0],
    }


class TestNurbsCurveSerialize(_CurveCreateCase):
    """``serialize`` holds the curve's own points and form, and its knots when they are not
    Maya's default ones, so ``create`` rebuilds the same curve."""

    def assert_same_shape(self, curve, other, samples=48):
        """The two curves trace the same points, each over its own parameter range."""
        def trace(c):
            lo, hi = c.get_fn_set().knotDomain
            return _on_curve(c, np.linspace(lo, hi, samples + 1))

        self.assertLess(abs(trace(curve) - trace(other)).max(), 1e-9)

    def test_open_curve_with_default_knots(self):
        for degree in (1, 2, 3, 5):
            with self.subTest(degree=degree):
                cmds.file(new=True, force=True)
                source = cmds.curve(point=POINTS, degree=degree, name="src")
                data = NurbsCurve(source).serialize(world_space=False)
                self.assertEqual(_rows(data.points), _rows(POINTS))
                self.assertEqual((data.degree, data.periodic), (degree, False))
                self.assertIsNone(getattr(data, "knots", None))  # nothing to carry
                self.assertEqual(curve_state(NurbsCurve.create(data)), curve_state(source))

    def test_periodic_curve_holds_its_own_points(self):
        # Maya repeats a periodic curve's first `degree` CVs at its end; BSplineData wraps
        # its points itself, so they are left out (they used to be wrapped twice)
        ring = _ring(8)
        cases = [("circle", 3, lambda: cmds.circle(sections=8, degree=3, constructionHistory=False)[0])]
        cases += [
            (f"cmds.curve(periodic=True, degree={d})", d,
             lambda d=d: cmds.curve(periodic=True, degree=d, point=ring + ring[:d], knot=list(range(1 - d, 8 + d))))
            for d in (1, 2, 3)
        ]
        for label, degree, make in cases:
            with self.subTest(label):
                cmds.file(new=True, force=True)
                periodic = make()
                source = curve_state(periodic)
                self.assertEqual((source["form"], source["degree"]), (PERIODIC, degree))
                data = NurbsCurve(periodic).serialize(world_space=False)
                self.assertEqual(data.points.shape, (8, 3))
                self.assertEqual(_rows(data.points), source["cvs"][:8])
                self.assertTrue(data.periodic)
                self.assertIsNone(getattr(data, "knots", None))  # Maya's default periodic knots
                self.assertEqual(curve_state(NurbsCurve.create(data)), source)

    def test_closed_form_is_open(self):
        closed = _closed_form()
        self.assertEqual(NurbsCurve(closed).get_fn_set().form, OpenMaya.MFnNurbsCurve.kClosed)
        data = NurbsCurve(closed).serialize(world_space=False)
        self.assertFalse(data.periodic)
        self.assertEqual(_rows(data.points), _rows(POINTS + [POINTS[0]]))  # every CV
        rebuilt = curve_state(NurbsCurve.create(data))
        source = curve_state(closed)
        self.assertEqual((rebuilt["cvs"], rebuilt["knots"], rebuilt["degree"]),
                         (source["cvs"], source["knots"], source["degree"]))
        self.assertEqual(rebuilt["form"], OPEN)  # the same curve; create makes open or periodic ones

    def test_world_space(self):
        source = cmds.curve(point=POINTS, name="src")
        cmds.xform(source, translation=(1, 2, 3), rotation=(10, 20, 30))
        data = NurbsCurve(source).serialize()
        expected = np.array(NurbsCurve(source).get_points(world_space=True))[:, :3]
        self.assertEqual(_rows(data.points), _rows(expected))

    @unittest.skipUnless(_takes_knots(), "this cgmath's BSplineData takes no knots")
    def test_uneven_knots_travel(self):
        for label, xform in _uneven_curves().items():
            with self.subTest(label):
                source = NurbsCurve(xform)
                state = curve_state(source)
                data = source.serialize(world_space=False)
                self.assertEqual(tuple(data.knots), state["knots"])  # Maya's own, as they are
                rebuilt = NurbsCurve.create(data)
                got = curve_state(rebuilt)
                self.assertEqual((got["cvs"], got["degree"], got["form"]),
                                 (state["cvs"], state["degree"], state["form"]))
                # as they are, range and all (to rounding: cgmath 2.0.0's kv is lo + t * scale)
                self.assertTrue(np.allclose(got["knots"], state["knots"], rtol=0.0, atol=1e-12))
                self.assert_same_shape(rebuilt, source)

    def test_evenly_spaced_knots_on_another_range(self):
        # rebuildCurve -keepRange 0 gives knots from 0 to 1: the same shape as the default ones
        source = cmds.curve(point=POINTS, degree=3, knot=[0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1], name="unit")
        rebuilt = NurbsCurve.create(NurbsCurve(source).serialize(world_space=False))
        self.assertEqual(curve_state(rebuilt)["cvs"], curve_state(source)["cvs"])
        self.assert_same_shape(rebuilt, NurbsCurve(source))
        if not _takes_knots():
            with self.assertNoLogs("rig.nodetypes.nurbs", level="WARNING"):
                NurbsCurve(source).serialize()

    def test_uneven_knots_on_a_cgmath_without_knots_warn(self):
        # on a cgmath whose BSplineData takes knots (2.0.0) a stand-in without them (1.0.4's)
        # is patched in, so the test runs on both
        for label, xform in _uneven_curves().items():
            with self.subTest(label):
                older = (mock.patch("cgmath.geometry.BSplineData", _BSplineDataWithoutKnots)
                         if _takes_knots() else contextlib.nullcontext())
                with older, self.assertLogs("rig.nodetypes.nurbs", level="WARNING") as logs:
                    data = NurbsCurve(xform).serialize(world_space=False)
                self.assertIn("uneven knots are dropped", logs.output[0])
                self.assertEqual(data.degree, 3)


# ---------------------------------------------------------------------------------------------
class TestNurbsCurveCreateUndo(_CurveCreateCase):
    """One undo step named rig.NurbsCurve.create; exact undo / redo walks; the same node on redo."""

    def test_one_undo_removes_transform_and_shape(self):
        self.ready()
        nodes = set(cmds.ls())
        NurbsCurve.create(POINTS, name="c")
        self.assertEqual(set(cmds.ls()) - nodes, {"c", "cShape"})
        self.assertEqual(self.undo_name(), CHUNK)
        self.undo_steps(1)
        self.assertEqual(set(cmds.ls()), nodes)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))  # one step, all of it
        self.redo_steps(1)
        self.assertEqual(set(cmds.ls()) - nodes, {"c", "cShape"})

    def test_undo_all_redo_all_exact(self):
        cases = (
            ("points", lambda: (POINTS, {})),
            ("degree and kv", lambda: (POINTS, dict(degree=2, kv=[0.0, 0.0, 0.5, 2.0, 3.0, 4.0, 4.0]))),
            ("open data", lambda: (_bspline_data(POINTS, 3, False), {})),
            ("periodic data", lambda: (_bspline_data(_ring(8), 3, True), {})),
        )
        for label, make in cases:
            with self.subTest(label):
                cmds.file(new=True, force=True)
                points, kwargs = make()
                self.ready()
                start = self.state("cShape")
                NurbsCurve.create(points, name="c", **kwargs)
                built = self.state("cShape")
                self.assertIsNotNone(built["curves"]["cShape"])
                for cycle in range(3):
                    self.assertEqual(self.undo_all(), 1)
                    self.assert_same(self.state("cShape"), start, f"{label} undo {cycle}")
                    self.assertEqual(cmds.ls(selection=True), ["selprobe"])
                    self.assertEqual(self.redo_all(), 1)
                    self.assert_same(self.state("cShape"), built, f"{label} redo {cycle}")
                    self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def test_redo_returns_the_same_node(self):
        self.ready()
        curve = NurbsCurve.create(POINTS, name="c")
        handle = OpenMaya.MObjectHandle(curve.get_mobject())
        uuid = cmds.ls("cShape", uuid=True)[0]
        points = [tuple(p) for p in curve.get_points(world_space=False)]
        for _ in range(3):
            cmds.undo()
            self.assertFalse(cmds.objExists("cShape"))
            cmds.redo()
            self.assertTrue(handle.isValid())
            self.assertEqual(cmds.ls("cShape", uuid=True)[0], uuid)
            self.assertEqual(curve.get_num_cvs(), 6)
            self.assertEqual([tuple(p) for p in curve.get_points(world_space=False)], points)
            self.assertEqual(str(curve), "cShape")

    def test_inside_a_user_chunk_with_cmds_edits(self):
        self.ready()
        start = self.state("cShape")
        cmds.undoInfo(openChunk=True, chunkName="user.step")
        try:
            cmds.setAttr("selprobe.translateX", 3.0)
            NurbsCurve.create(POINTS, name="c")
            cmds.setAttr("c.translateZ", 5.0)
            cmds.parent("c", "selprobe")
        finally:
            cmds.undoInfo(closeChunk=True)
        built = self.state("cShape")
        self.assertEqual(self.undo_name(), "user.step")
        for cycle in range(2):
            self.undo_steps(1)
            self.assert_same(self.state("cShape"), start, f"undo {cycle}")
            self.redo_steps(1)
            self.assert_same(self.state("cShape"), built, f"redo {cycle}")
        self.assertEqual(cmds.getAttr("selprobe|c.translateZ"), 5.0)

    def test_between_cmds_steps(self):
        self.ready()
        states = [self.state("cShape")]
        cmds.setAttr("selprobe.translateX", 3.0)
        states.append(self.state("cShape"))
        NurbsCurve.create(POINTS, name="c")
        states.append(self.state("cShape"))
        cmds.setAttr("c.translateZ", 5.0)
        states.append(self.state("cShape"))
        for k in (2, 1, 0):
            self.undo_steps(1)
            self.assert_same(self.state("cShape"), states[k], f"undo to {k}")
        for k in (1, 2, 3):
            self.redo_steps(1)
            self.assert_same(self.state("cShape"), states[k], f"redo to {k}")

    def test_in_a_container_scope_one_undo_reverts_membership(self):
        self.ready()
        with container("box") as box:
            start = self.state("cShape")
            members_before = sorted(cmds.container(str(box), query=True, nodeList=True) or []) \
                if cmds.objExists(str(box)) else None
            NurbsCurve.create(POINTS, name="c")
        box_name = str(box)
        built = self.state("cShape")
        self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True)), ["c", "cShape"])
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("c"))
        self.assertFalse(cmds.objExists("cShape"))
        if cmds.objExists(box_name):
            self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True) or []),
                             members_before or [])
        self.assert_same(self.state("cShape"), start, "undo")
        self.redo_steps(1)
        self.assert_same(self.state("cShape"), built, "redo")
        self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True)), ["c", "cShape"])

    def test_undo_off(self):
        cmds.undoInfo(state=False)
        try:
            curve = NurbsCurve.create(POINTS, name="c")
        finally:
            cmds.undoInfo(state=True)
        self.assertEqual(str(curve), "cShape")
        self.assertEqual(curve_state(curve)["cvs"], _rows(POINTS))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


# ---------------------------------------------------------------------------------------------
class _UnwrappedRing:
    """What `NurbsCurve.create` reads of a BSplineData, for a periodic curve whose
    CVs do not repeat: Maya refuses it, after the transform is made."""

    degree   = 3
    periodic = True
    cv       = np.array([(float(i), 0.0, float(i % 2)) for i in range(11)])
    kv       = np.arange(-2.0, 11.0)


class TestNurbsCurveCreateErrors(_CurveCreateCase):
    """Input that is no curve raises before anything is made; the next undo is the user's."""

    # label, (points, kwargs), the error, a word of its message
    CASES = (
        ("a name", ("grp", {}), TypeError, r"takes \(N, 3\) points or a BSplineData, got str"),
        ("None", (None, {}), TypeError, r"takes \(N, 3\) points or a BSplineData, got NoneType"),
        ("a number", (3.0, {}), TypeError, r"takes \(N, 3\) points or a BSplineData, got float"),
        ("ragged points", ([(0, 0, 0), (1, 0), (2, 0, 0), (3, 0, 0)], {}), TypeError, r"takes \(N, 3\) points"),
        ("no points", ([], {}), ValueError, r"must be \(N, 3\), got \(0,\)"),
        ("2-D points", ([(0, 0), (1, 1), (2, 0), (3, 1)], {}), ValueError, r"must be \(N, 3\), got \(4, 2\)"),
        ("weighted points", ([(0, 0, 0, 2)] * 4, {}), ValueError, r"non-rational curves: \(N, 4\) points take w = 1"),
        ("5-D points", ([(0, 0, 0, 1, 0)] * 4, {}), ValueError, r"must be \(N, 3\), got \(4, 5\)"),
        ("flat points", ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], {}), ValueError, r"must be \(N, 3\), got \(6,\)"),
        ("too few points", (POINTS[:3], {}), ValueError, r"degree 3 takes 4 points or more, got 3"),
        ("too few for the degree", (POINTS, dict(degree=6)), ValueError, r"degree 6 takes 7 points or more, got 6"),
        ("a nan", ([(0, 0, 0), (1, float("nan"), 0), (2, 0, 0), (3, 0, 0)], {}), ValueError, r"points must be finite"),
        ("an inf", ([(0, 0, 0), (1, float("inf"), 0), (2, 0, 0), (3, 0, 0)], {}), ValueError, r"points must be finite"),
        ("degree 0", (POINTS, dict(degree=0)), ValueError, r"degree must be 1 or more, got 0"),  # crashes Maya
        ("a negative degree", (POINTS, dict(degree=-1)), ValueError, r"degree must be 1 or more, got -1"),
        ("a float degree", (POINTS, dict(degree=2.0)), TypeError, r"degree must be an int, got 2\.0"),
        ("a bool degree", (POINTS, dict(degree=True)), TypeError, r"degree must be an int, got True"),
        ("a str degree", (POINTS, dict(degree="3")), TypeError, r"degree must be an int, got '3'"),
        ("a knot too few", (POINTS, dict(kv=[0, 0, 0, 1, 2, 3, 3])), ValueError,
         r"6 points of degree 3 take 8 knots \(Maya's layout: points \+ degree - 1\), got 7$"),
        ("a knot too many", (POINTS, dict(kv=[0, 0, 0, 1, 2, 3, 3, 3, 3])), ValueError, r"take 8 knots .* got 9$"),
        ("knots with both ends padded", (POINTS, dict(kv=[0, 0, 0, 0, 1, 2, 3, 3, 3, 3])), ValueError,
         r"got 10 \(leave out the first and the last\)"),
        ("knots in rows", (POINTS, dict(kv=[[0, 0, 0, 1], [2, 3, 3, 3]])), ValueError, r"take 8 knots .* got \(2, 4\)"),
        ("knots that decrease", (POINTS, dict(kv=[0, 0, 0, 2, 1, 3, 3, 3])), ValueError, r"kv must never decrease"),
        ("a knot repeated too often", (_line(9), dict(kv=[0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3])), ValueError,
         r"repeats 3 times at most \(Maya's layout\), got one 4 times"),
        ("end knots repeated too often", (POINTS, dict(kv=[0, 0, 0, 0, 1, 1, 1, 1])), ValueError,
         r"repeats 3 times at most"),
        ("knots of no length", (POINTS[:4], dict(kv=[0, 0, 1, 1, 2, 2])), ValueError,
         r"parameter range longer than zero"),
        ("a nan knot", (POINTS, dict(kv=[0, 0, 0, 1, float("nan"), 3, 3, 3])), ValueError, r"kv must be finite"),
        ("knots that are no numbers", (POINTS, dict(kv=["a"] * 8)), TypeError, r"kv must be numbers, got list"),
    )

    def _bad(self, points, kwargs, error, message):
        cmds.setAttr("selprobe.translateX", 7.0)  # the user's previous step
        nodes = sorted(cmds.ls())
        with self.assertRaisesRegex(error, message):
            NurbsCurve.create(points, name="c", **kwargs)
        self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def test_input_that_is_no_curve(self):
        for label, (points, kwargs), error, message in self.CASES:
            with self.subTest(label):
                cmds.file(new=True, force=True)
                self.ready()
                self._bad(points, kwargs, error, message)
                self.undo_steps(1)  # the setAttr, not an empty create step
                self.assertEqual(cmds.getAttr("selprobe.translateX"), 0.0)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_refused_create_keeps_the_redo_queue(self):
        self.ready()
        cmds.setAttr("selprobe.translateY", 3.0)
        self.undo_steps(1)  # the user's step, on the redo queue
        picked = ("a name", "no points", "degree 0", "a knot too few")
        for label, (points, kwargs), error, message in [case for case in self.CASES if case[0] in picked]:
            with self.subTest(label):
                with self.assertRaisesRegex(error, message):
                    NurbsCurve.create(points, name="c", **kwargs)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.assertFalse(cmds.undoInfo(query=True, redoQueueEmpty=True))
        self.redo_steps(1)
        self.assertEqual(cmds.getAttr("selprobe.translateY"), 3.0)

    def test_a_node_or_a_plug_is_not_points(self):
        source = Node(cmds.curve(point=POINTS)).get_shape()
        self.ready()
        nodes = sorted(cmds.ls())
        for label, value in (("a curve", source), ("its transform", source.get_parent()),
                             ("its cv plug", source.cv), ("a translate plug", source.get_parent().t)):
            with self.subTest(label):
                with self.assertRaisesRegex(TypeError, r"takes \(N, 3\) points or a BSplineData, got "):
                    NurbsCurve.create(value)
                self.assertEqual(sorted(cmds.ls()), nodes)

    def test_refused_in_a_container_scope(self):
        self.ready()
        with container("box") as box:
            with self.assertRaises(ValueError):
                NurbsCurve.create(POINTS, degree=0, name="c")
            with self.assertRaises(TypeError):
                Node.create("nurbsCurve", "grp")
        self.assertFalse(cmds.ls(type="nurbsCurve"))
        if cmds.objExists(str(box)):
            self.assertIsNone(cmds.container(str(box), query=True, nodeList=True))

    def test_error_after_the_transform_leaves_no_node(self):
        self.ready()
        cmds.setAttr("selprobe.translateX", 7.0)
        nodes = sorted(cmds.ls())
        with self.assertRaisesRegex(RuntimeError, r"Maya refused the curve \(11 CVs, degree 3, periodic\)"):
            NurbsCurve.create(_UnwrappedRing(), name="c")
        self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])
        self.undo_all()
        self.redo_all()
        self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(cmds.getAttr("selprobe.translateX"), 7.0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------------------------
#  NurbsSurface.create (round features s7, F4), its serialize (F3, F5)
# ---------------------------------------------------------------------------------------------

SURFACE_CHUNK = "rig.NurbsSurface.create"
S_OPEN, S_CLOSED, S_PERIODIC = (OpenMaya.MFnNurbsSurface.kOpen, OpenMaya.MFnNurbsSurface.kClosed,
                                OpenMaya.MFnNurbsSurface.kPeriodic)


def _surface_shape(t):
    return cmds.listRelatives(t, shapes=True)[0]


def _surface_layout(shape):
    fn = NurbsSurface(shape).get_fn_set()
    return (fn.numCVsInU, fn.numCVsInV, fn.degreeInU, fn.degreeInV, fn.formInU, fn.formInV)


def _surface_at(shape, steps=7):
    """World-space points of `shape` over a grid of its normalised parameters."""
    fn = NurbsSurface(shape).get_fn_set()
    (u0, u1), (v0, v1) = fn.knotDomainInU, fn.knotDomainInV
    return np.array([
        list(fn.getPointAtParam(u0 + s * (u1 - u0), v0 + t * (v1 - v0), OpenMaya.MSpace.kWorld))[:3]
        for s in np.linspace(0, 1, steps) for t in np.linspace(0, 1, steps)
    ])


def _closed_surface():
    """A loft of two closed (kClosed, not periodic) degree-1 curves: closed in one direction."""
    pts = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 0, 0)]
    a = cmds.curve(point=pts, degree=1)
    b = cmds.curve(point=[(x, 1, z) for x, _, z in pts], degree=1)
    t = cmds.loft(a, b, constructionHistory=False, degree=1, name="closed")[0]
    cmds.delete(a, b)
    return _surface_shape(t)


def _periodic_loft():
    """A uniform loft of three circles, closed along the loft: periodic in both directions,
    evenly spaced knots."""
    rings = [cmds.circle(constructionHistory=False, radius=r)[0] for r in (1, 2, 1)]
    for i, ring in enumerate(rings):
        cmds.move(0, 0, i, ring)
    t = cmds.loft(*rings, constructionHistory=False, close=True, uniform=True, name="loft")[0]
    cmds.delete(rings)
    return _surface_shape(t)


class TestNurbsSurfaceCreate(_CurveCreateCase):
    """``NurbsSurface.create(data)``: the ride of ``NurbsCurve.create``, from a BSplinePatchData."""

    def sources(self):
        return {
            "plane": _surface_shape(cmds.nurbsPlane(name="plane", constructionHistory=False, u=3, v=2)[0]),
            "sphere": _surface_shape(cmds.sphere(name="ball", constructionHistory=False)[0]),
            "cylinder": _surface_shape(cmds.cylinder(name="cyl", constructionHistory=False)[0]),
            "loft": _periodic_loft(),
        }

    def test_the_serialize_round_trip(self):
        for label, shape in self.sources().items():
            with self.subTest(label):
                data = NurbsSurface(shape).serialize()
                self.ready()
                surface = NurbsSurface.create(data)
                self.assertIs(type(surface), NurbsSurface)
                self.assertEqual(_surface_layout(str(surface)), _surface_layout(shape))
                np.testing.assert_allclose(
                    np.array(surface.get_points())[:, :3], np.array(NurbsSurface(shape).get_points())[:, :3], atol=1e-9
                )
                np.testing.assert_allclose(_surface_at(str(surface)), _surface_at(shape), atol=1e-9)
                self.assertEqual(cmds.listSets(object=str(surface)), ["initialShadingGroup"])
                self.assertEqual(cmds.ls(selection=True), ["selprobe"])
                self.assertEqual(cmds.listHistory(str(surface)), [str(surface)])
                self.assertEqual(self.undo_name(), SURFACE_CHUNK)

    def test_evenly_spaced_knots_come_back_on_0_to_spans(self):
        # a nurbsPlane's 0..1 comes back as 0..3: the same surface, another parameter range
        shape = self.sources()["plane"]
        surface = NurbsSurface.create(NurbsSurface(shape).serialize())
        self.assertEqual(tuple(surface.get_fn_set().knotDomainInU), (0.0, 3.0))
        self.assertEqual(tuple(NurbsSurface(shape).get_fn_set().knotDomainInU), (0.0, 1.0))

    def test_names_and_attributes(self):
        data = NurbsSurface(self.sources()["plane"]).serialize()
        self.assertEqual(str(NurbsSurface.create(data).get_parent()), "surface1")
        self.assertEqual(str(NurbsSurface.create(data, name="patch")), "patchShape")
        self.assertEqual(str(NurbsSurface.create(data, name="patchShape3").get_parent()), "patch3")
        hidden = NurbsSurface.create(data, name="h", visibility=False)
        self.assertFalse(cmds.getAttr(f"{hidden}.visibility"))
        with container("outer"):
            with container("inner"):
                self.assertEqual(str(NurbsSurface.create(data, name="s").get_parent()), "inner_s")
        made = Node.create("nurbsSurface", data, name="by_node")
        self.assertIs(type(made), NurbsSurface)

    def test_one_undo_step(self):
        data = NurbsSurface(self.sources()["sphere"]).serialize()
        self.ready()
        before = self.state()
        surface = NurbsSurface.create(data, name="s")
        after = self.state()
        uuid = cmds.ls("s", uuid=True)
        self.assertEqual(self.undo_name(), SURFACE_CHUNK)
        self.undo_steps(1)
        self.assert_same(self.state(), before)
        self.redo_steps(1)
        self.assert_same(self.state(), after)
        self.assertEqual(cmds.ls("s", uuid=True), uuid)
        self.assertIs(surface.is_valid(), True)

    def test_input_that_is_no_surface(self):
        data = NurbsSurface(self.sources()["plane"]).serialize()
        from cgmath.geometry import BSplinePatchData

        few = BSplinePatchData(points=np.zeros((2, 5, 3)), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False)
        flat = BSplinePatchData(points=np.zeros((4, 4, 2)), degree_u=3, degree_v=3, periodic_u=False, periodic_v=False)
        curve_data = NurbsCurve.create(POINTS).serialize()
        self.ready()
        for label, call, error, text in (
            ("points", lambda: NurbsSurface.create([[0, 0, 0]]), TypeError,
             "^NurbsSurface.create\\(\\) takes a BSplinePatchData, got list \\(NurbsSurface.load\\(path\\) reads one from a file\\)$"),
            ("a curve's data", lambda: NurbsSurface.create(curve_data), TypeError, "got BSplineData"),
            ("too few", lambda: NurbsSurface.create(few), ValueError,
             "^a surface of degree 3 in U takes 4 rows of points or more there, got 2$"),
            ("2D points", lambda: NurbsSurface.create(flat), ValueError, "points must be \\(Nu, Nv, 3\\), got \\(4, 4, 2\\)"),
            ("a node", lambda: NurbsSurface.create(Node("plane")), TypeError, "takes a BSplinePatchData, got Transform"),
            ("no input", lambda: Node.create("nurbsSurface"), TypeError,
             "needs the inputs a NurbsSurface is built from, after the type: Node.create\\('nurbsSurface', data, ...\\)"),
            ("a parent", lambda: NurbsSurface.create(data, parent="plane"), TypeError, "takes no parent="),
        ):
            with self.subTest(label):
                before = cmds.ls()
                with self.assertRaisesRegex(error, text):
                    call()
                self.assertEqual(cmds.ls(), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_define_names_create_and_load(self):
        with self.assertRaisesRegex(
            TypeError,
            "^NurbsSurface.define\\(\\) is refused: a nurbsSurface is built from its data: "
            "NurbsSurface.create\\(data, name='x'\\) or NurbsSurface.load\\(path, name='x'\\) makes one "
            "\\(rc.nurbsPlane\\(name='x'\\) a primitive\\); NurbsSurface\\('x'\\) refers to one$",
        ):
            NurbsSurface.define("x")


class TestNurbsSurfaceSerialize(_CurveCreateCase):
    """F3: a direction is periodic only in Maya's kPeriodic form (a kClosed one is open and
    keeps every CV; v2.0.0a5 read it periodic and trimmed its CVs). F5: uneven knots warn."""

    def test_a_closed_direction_is_open(self):
        shape = _closed_surface()
        fn = NurbsSurface(shape).get_fn_set()
        self.assertIn(S_CLOSED, (fn.formInU, fn.formInV))
        data = NurbsSurface(shape).serialize()
        self.assertEqual((bool(data.periodic_u), bool(data.periodic_v)), (False, False))
        self.assertEqual(tuple(data.points.shape[:2]), (fn.numCVsInU, fn.numCVsInV))
        surface = NurbsSurface.create(data)
        np.testing.assert_allclose(_surface_at(str(surface)), _surface_at(shape), atol=1e-9)

    def test_a_periodic_direction_trims_its_wrap(self):
        shape = _periodic_loft()
        fn = NurbsSurface(shape).get_fn_set()
        data = NurbsSurface(shape).serialize()
        periodic = (fn.formInU == S_PERIODIC, fn.formInV == S_PERIODIC)
        self.assertEqual((bool(data.periodic_u), bool(data.periodic_v)), periodic)
        self.assertTrue(any(periodic))
        wraps = (fn.degreeInU if periodic[0] else 0, fn.degreeInV if periodic[1] else 0)
        self.assertEqual(tuple(data.points.shape[:2]), (fn.numCVsInU - wraps[0], fn.numCVsInV - wraps[1]))

    def test_uneven_knots_warn(self):
        rows = [cmds.curve(point=[(0, y, 0), (1, y, 0), (2, y, 0), (3, y, 0)]) for y in (0, 1, 5)]
        shape = _surface_shape(cmds.loft(*rows, constructionHistory=False, uniform=False, name="uneven")[0])
        with self.assertLogs("rig.nodetypes.nurbs", level="WARNING") as logs:
            NurbsSurface(shape).serialize()
        self.assertEqual(len(logs.output), 1)
        self.assertIn(
            f"{shape}: this cgmath's BSplinePatchData takes no knots, so the surface's uneven knots in", logs.output[0]
        )

    def test_even_knots_do_not_warn(self):
        for shape in (_surface_shape(cmds.nurbsPlane(name="p", constructionHistory=False, u=3, v=2)[0]),
                      _surface_shape(cmds.sphere(name="s", constructionHistory=False)[0]),
                      _closed_surface(), _periodic_loft()):
            with self.subTest(shape):
                with mock.patch("rig.nodetypes.nurbs.LOGGER") as logger:
                    NurbsSurface(shape).serialize()
                logger.warning.assert_not_called()