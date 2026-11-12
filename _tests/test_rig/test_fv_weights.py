"""The weights round (the user's request of 2026-11-11, calls A to F): a rational curve's or
surface's weights travel with its CVs.

cgmath's ``NurbsCurveData`` and ``NurbsSurfaceData`` hold an optional ``weights`` (cgmath
8fb221e): one per control point, ``(N,)`` or ``(Nu, Nv)``, beside the points as they are (not
times their weight).

* ``serialize`` writes them only for a shape that holds a weight other than 1, an exact compare
  (E: equal weights other than 1 are written); any other shape leaves the field None. A
  periodic direction's repeats are trimmed as its points are. A weight cgmath evaluates no
  shape with (zero, negative: the API makes such shapes, ``cmds`` clamps at 0.0001) is a
  ValueError naming its CV (F).
* ``create`` makes a rational shape of data that holds weights, each CV's weight in its ``w``;
  data that holds none, or 1 everywhere, makes the shape it made before. ``NurbsCurve.create``
  takes ``(N, 4)`` points too, the weight in ``w`` (A).
* In place (``NurbsCurve('x').load`` / ``NurbsSurface('x').load``) the data's weights are set
  with the CVs in the one undo step; data that holds none puts every weight back to 1 (B).
* In world space a weighted CV is read and written where it is (D): Maya's
  ``cvPositions(kWorld)`` / ``setCVPositions(kWorld)`` move it by the translation times its
  weight. The geometry plugs read through ``serialize``.

Every source is made without rig (``cmds.curve(pointWeight=)``, ``cmds.surface(pointWeight=)``,
``cmds.setAttr`` on ``weights``, ``MFnNurbsCurve.create`` / ``MFnNurbsSurface.create``); what rig
makes is read back with Maya's function sets and ``cmds.pointPosition``, and evaluated with
``cmds.pointOnCurve`` / ``cmds.pointOnSurface`` against cgmath's evaluation of the data.

32 of the 36 tests fail at ab305bb (negative control, the module run on that tree with this
cgmath: no weight was written, read from data or set, and a weighted CV was read in the world
where Maya puts it). The 4 that pass there pin what the round leaves as it was: a curve or a
surface whose weights are all 1 holds none, weights of 1 make the surface made without any, and
data without weights puts every weight back to 1.
"""

import dataclasses
import json
import re
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig.nodetypes import NurbsCurve, NurbsSurface
from rig.nodetypes import nurbs as nurbs_module
from rig._tests.test_class_saveload import FORMATS, _Case

WORLD = OpenMaya.MSpace.kWorld

# a degree 2 arc, its middle CV of weight 3, and the same layout without weights
ARC    = [(0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 0.0, 3.0), (2.0, 0.0, 0.0, 1.0)]
FLAT   = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)]
KNOTS  = [0, 0, 1, 1]
MIDDLE = [1.0, 1.5, 0.0]  # the arc at 0.5: (P0 / 4 + 3 * P1 / 2 + P2 / 4) / 2
ONES   = [point[:3] + (1.0,) for point in ARC]

# a quarter circle of radius 1 is a degree 2 arc whose middle weight is sqrt(1 / 2)
S2      = float(np.sqrt(0.5))
QUARTER = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
PIPE_W  = [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0]
REPEAT  = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]  # a periodic degree 3 direction of 8: the 11 Maya holds

AT      = [0.25, 0.5, 0.75]
AROUND  = [0.3, 2.7, 6.1, 7.9]
UU      = [0.2, 0.5, 0.9]
VV      = [0.1, 0.5, 0.8]
PIPE_V  = [0.3, 3.7, 7.1]

HERE    = dict(translation=(5, 1, -2), rotation=(10, 40, 70), scale=(2, 1, 3))
THERE   = dict(translation=(-3, 2, 4), rotation=(25, -15, 60), scale=(1, 2, 0.5))

BAD = "must be finite and above zero"


# -- Maya's own reads


def _path(node):
    selection = OpenMaya.MSelectionList()
    selection.add(str(node))
    return selection.getDagPath(0)


def _fn(shape):
    """Maya's function set on `shape` (not rig's)."""
    path = _path(shape)
    if path.node().hasFn(OpenMaya.MFn.kNurbsCurve):
        return OpenMaya.MFnNurbsCurve(path)
    return OpenMaya.MFnNurbsSurface(path)


def _held(shape):
    """The CVs Maya holds, ``(N, 4)``: each one's object-space position and, in w, its weight."""
    return np.array(_fn(shape).cvPositions())


def _weights(shape):
    return _held(shape)[:, 3]


def _world(shape):
    """Each CV's world position, as ``cmds.pointPosition`` reads it."""
    fn = _fn(shape)
    if isinstance(fn, OpenMaya.MFnNurbsCurve):
        cvs = [f"{shape}.cv[{i}]" for i in range(fn.numCVs)]
    else:
        cvs = [f"{shape}.cv[{u}][{v}]" for u in range(fn.numCVsInU) for v in range(fn.numCVsInV)]
    return np.array([cmds.pointPosition(cv, world=True) for cv in cvs])


def _on_curve(shape, at):
    return np.array([cmds.pointOnCurve(str(shape), parameter=float(u), position=True) for u in at])


def _on_surface(shape, uu, vv):
    return np.array([cmds.pointOnSurface(str(shape), u=float(u), v=float(v), position=True) for u, v in zip(uu, vv)])


def _takes_weights(shape):
    """Whether Maya holds `shape` as rational, by what its ``setCVPositions`` does with a w
    (measured): a shape that is or was rational takes it, one that never was drops it."""
    fn        = _fn(shape)
    cvs       = _held(shape)
    cvs[1, 3] = 3.0
    fn.setCVPositions(OpenMaya.MPointArray(cvs.tolist()))
    if isinstance(fn, OpenMaya.MFnNurbsCurve):
        fn.updateCurve()
    else:
        fn.updateSurface()
    return float(_weights(shape)[1]) == 3.0


def _nothing_to_undo():
    """Whether the undo queue is empty. Read from ``undoQueueEmpty``, not from an empty
    ``undoName``: an entry ``cmds`` records in mayapy has no name."""
    return cmds.undoInfo(query=True, undoQueueEmpty=True)


# -- sources, made without rig


def _shape(transform):
    """The shape of `transform`, named ``<transform>Shape`` (``cmds.curve`` leaves it
    ``curveShape1``)."""
    shape = cmds.listRelatives(transform, shapes=True, fullPath=True)[0]
    return cmds.rename(shape, f"{transform}Shape")


def _arc(name="arc"):
    """A rational curve Maya makes: degree 2, its middle CV of weight 3."""
    return _shape(cmds.curve(degree=2, pointWeight=ARC, knot=KNOTS, name=name))


def _flat(name="flat"):
    """A curve of the arc's layout that never held a weight."""
    return _shape(cmds.curve(degree=2, point=FLAT, knot=KNOTS, name=name))


def _ring(name="ring"):
    """A periodic curve: 8 CVs, held as 11 (the first 3 again)."""
    return _shape(cmds.circle(name=name, sections=8, constructionHistory=False)[0])


def _api_curve(points, name="api"):
    """A rational degree 2 curve made with ``MFnNurbsCurve.create``, which takes any weight."""
    xform = cmds.createNode("transform", name=name, skipSelect=True)
    OpenMaya.MFnNurbsCurve().create(
        OpenMaya.MPointArray(points), OpenMaya.MDoubleArray(KNOTS), 2, OpenMaya.MFnNurbsCurve.kOpen, False, True,
        _path(xform).node(),
    )
    return _shape(xform)


def _quarter_cvs(weight=S2):
    """``(x, y, z, w)`` rows, V fastest: a quarter cylinder of radius 1, 3 x 2 CVs of degrees
    2 x 1, its middle row of `weight`."""
    return [(x, y, z, w) for (x, y), w in zip(QUARTER, (1.0, weight, 1.0)) for z in (0.0, 2.0)]


def _patch(name, **cvs):
    shape = cmds.surface(degreeU=2, degreeV=1, knotU=KNOTS, knotV=[0, 1], **cvs)
    return _shape(cmds.rename(cmds.listRelatives(shape, parent=True)[0], name))


def _quarter(name="quarter"):
    """A rational surface Maya makes: the quarter cylinder."""
    return _patch(name, pointWeight=_quarter_cvs())


def _bent(name="bent"):
    """A surface of the quarter's layout that never held a weight."""
    return _patch(name, point=[cv[:3] for cv in _quarter_cvs()])


def _api_surface(cvs, name="api"):
    """A rational surface of the quarter's layout made with ``MFnNurbsSurface.create``, which
    takes any weight."""
    xform = cmds.createNode("transform", name=name, skipSelect=True)
    OpenMaya.MFnNurbsSurface().create(
        OpenMaya.MPointArray(cvs), OpenMaya.MDoubleArray(KNOTS), OpenMaya.MDoubleArray([0, 1]), 2, 1,
        OpenMaya.MFnNurbsSurface.kOpen, OpenMaya.MFnNurbsSurface.kOpen, True, _path(xform).node(),
    )
    return _shape(xform)


def _pipe(name="pipe"):
    """A rational surface periodic in V, made with the API: 4 x 8 CVs of degree 3, held as
    4 x 11, each ring of the weights `PIPE_W`."""
    ring = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    cvs  = []
    for z in range(4):
        row = [(float(x), float(y), float(z), w) for (x, y), w in zip(ring, PIPE_W)]
        cvs += row + row[:3]
    xform = cmds.createNode("transform", name=name, skipSelect=True)
    OpenMaya.MFnNurbsSurface().create(
        OpenMaya.MPointArray(cvs), OpenMaya.MDoubleArray([0, 0, 0, 1, 1, 1]),
        OpenMaya.MDoubleArray(np.arange(-2.0, 11.0).tolist()), 3, 3,
        OpenMaya.MFnNurbsSurface.kOpen, OpenMaya.MFnNurbsSurface.kPeriodic, True, _path(xform).node(),
    )
    return _shape(xform)


def _ball(name="ball"):
    """A sphere: periodic in V, 7 x 8 CVs held as 7 x 11."""
    return _shape(cmds.sphere(name=name, constructionHistory=False)[0])


def _graded(rows, columns):
    """A weight for each CV of a ``(rows, columns)`` grid, no two the same: 1.0, 1.1, 1.2..."""
    return 1.0 + np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) / 10.0


class _Stand:
    """A stand-in for a NurbsCurveData (`NurbsCurve.create` reads one by its attributes): the
    flat curve, holding what cgmath's own data refuses when it is made."""

    degree      = 2
    periodic    = False
    points      = np.array(FLAT)
    cv          = points
    kv          = np.array([0.0, 0.0, 1.0, 1.0])
    geometry    = np.arange(3)
    weights     = None
    uuid        = None
    parent_uuid = None

    def __init__(self, **fields):
        vars(self).update(fields)


# ---------------------------------------------------------------------------------------------
class TestSerialize(_Case):
    """``serialize`` writes ``weights`` for a shape that holds a weight other than 1, and only
    for one."""

    def test_a_curve_whose_weights_are_all_1_holds_none(self):
        back = _arc("back")
        cmds.setAttr(f"{back}.weights[1]", 1.0)
        for label, shape in (
            ("a curve", _flat()),
            ("a circle", _ring()),
            ("rational by its flag", _api_curve(ONES)),
            ("a weight put back to 1", back),
        ):
            with self.subTest(label):
                np.testing.assert_array_equal(_weights(shape), 1.0)
                self.assertIsNone(NurbsCurve(shape).serialize().weights)

    def test_a_rational_curve_holds_its_weights_beside_its_points(self):
        arc  = _arc()
        data = NurbsCurve(arc).serialize()
        self.assertEqual((data.weights.dtype, data.weights.shape), (np.float64, (3,)))
        np.testing.assert_array_equal(data.weights, [1.0, 3.0, 1.0])
        np.testing.assert_array_equal(data.points, np.array(ARC)[:, :3])  # as they are, not times w
        np.testing.assert_allclose(_on_curve(arc, [0.5]), [MIDDLE], atol=1e-12)
        np.testing.assert_allclose(data.compute(np.array(AT))[0], _on_curve(arc, AT), atol=1e-12)

    def test_equal_weights_other_than_1_are_written(self):
        # E: "all 1" is an exact compare. Equal weights draw the curve that 1 everywhere draws,
        # and are still what the shape holds; so is a weight a hair from 1
        twos = _shape(cmds.curve(degree=2, pointWeight=[point[:3] + (2.0,) for point in ARC], knot=KNOTS, name="twos"))
        np.testing.assert_array_equal(NurbsCurve(twos).serialize().weights, [2.0, 2.0, 2.0])
        near = _arc("near")
        cmds.setAttr(f"{near}.weights[1]", 1.0 + 1e-9)
        np.testing.assert_array_equal(NurbsCurve(near).serialize().weights, [1.0, 1.0 + 1e-9, 1.0])

    def test_a_periodic_curve_holds_one_weight_for_each_cv_it_does_not_repeat(self):
        ring = _ring()
        cmds.setAttr(f"{ring}.weights[1]", 3.0)
        cmds.setAttr(f"{ring}.weights[4]", 0.5)
        np.testing.assert_array_equal(_weights(ring), [1, 3, 1, 1, 0.5, 1, 1, 1, 1, 3, 1])  # Maya's 11
        data = NurbsCurve(ring).serialize()
        self.assertEqual(np.shape(data.points), (8, 3))
        np.testing.assert_array_equal(data.weights, [1, 3, 1, 1, 0.5, 1, 1, 1])
        np.testing.assert_allclose(data.compute(np.array(AROUND))[0], _on_curve(ring, AROUND), atol=1e-12)

    def test_world_space_reads_a_weighted_cv_where_it_is(self):
        arc = _arc()
        cmds.xform("arc", **HERE)
        node  = NurbsCurve(arc)
        world = np.array(node.get_points(world_space=True))
        np.testing.assert_allclose(world[:, :3], _world(arc), atol=1e-12)
        np.testing.assert_array_equal(world[:, 3], [1.0, 3.0, 1.0])
        # D: Maya's own world read puts the CV of weight 3 two translations further
        maya = np.array(_fn(arc).cvPositions(WORLD))
        np.testing.assert_allclose(maya[1, :3] - world[1, :3], 2 * np.array(HERE["translation"]), atol=1e-12)
        np.testing.assert_array_equal(np.array(node.get_points()), ARC)

        data = node.serialize(world_space=True)
        self.assertIsNone(data.parent_uuid)
        np.testing.assert_allclose(data.points, _world(arc), atol=1e-12)
        np.testing.assert_array_equal(data.weights, [1.0, 3.0, 1.0])
        np.testing.assert_allclose(data.compute(np.array(AT))[0], _on_curve(arc, AT), atol=1e-9)

        # a curve that holds no weight is Maya's read, bit for bit
        flat = _flat()
        cmds.xform("flat", **HERE)
        np.testing.assert_array_equal(
            np.array(NurbsCurve(flat).get_points(world_space=True)), np.array(_fn(flat).cvPositions(WORLD))
        )

    def test_a_weight_no_curve_evaluates_is_refused_naming_its_cv(self):
        # F: the API takes any weight
        for label, weight in (("zero", 0.0), ("negative", -2.0)):
            shape = _api_curve([ARC[0], (1.0, 2.0, 0.0, weight), ARC[2]], name=label)
            with self.subTest(label):
                self.assertEqual(_weights(shape)[1], weight)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^NurbsCurve\('{shape}'\)\.serialize\(\): cv\[1\] holds the weight {re.escape(repr(weight))}, "
                    rf"and a rational curve's weights are finite and above zero$",
                ):
                    NurbsCurve(shape).serialize()
        # cmds clamps one at 0.0001, which a curve holds
        arc = _arc()
        cmds.setAttr(f"{arc}.weights[1]", 0.0)
        np.testing.assert_array_equal(NurbsCurve(arc).serialize().weights, [1.0, 0.0001, 1.0])

    def test_a_surface_whose_weights_are_all_1_holds_none(self):
        for label, shape in (
            ("a plane", _shape(cmds.nurbsPlane(name="plane", constructionHistory=False)[0])),
            ("a sphere", _ball()),
            ("a surface", _bent()),
            ("rational by its flag", _api_surface(_quarter_cvs(1.0))),
        ):
            with self.subTest(label):
                np.testing.assert_array_equal(_weights(shape), 1.0)
                self.assertIsNone(NurbsSurface(shape).serialize().weights)

    def test_a_rational_surface_holds_its_weights_beside_its_points(self):
        quarter = _quarter()
        cvs     = np.array(_quarter_cvs())
        data    = NurbsSurface(quarter).serialize()
        self.assertEqual((data.weights.dtype, data.weights.shape), (np.float64, (3, 2)))
        np.testing.assert_array_equal(data.weights, cvs[:, 3].reshape(3, 2))
        np.testing.assert_array_equal(data.points, cvs[:, :3].reshape(3, 2, 3))  # as they are, not times w
        on = _on_surface(quarter, UU, VV)
        np.testing.assert_allclose(np.linalg.norm(on[:, :2], axis=1), 1.0, atol=1e-12)  # a circle: the weights' work
        np.testing.assert_allclose(data.evaluate(np.array(UU), np.array(VV)), on, atol=1e-12)

    def test_a_periodic_surface_holds_one_weight_for_each_cv_it_does_not_repeat(self):
        pipe = _pipe()
        np.testing.assert_array_equal(_weights(pipe).reshape(4, 11), np.tile(np.array(PIPE_W)[REPEAT], (4, 1)))
        data = NurbsSurface(pipe).serialize()
        self.assertEqual(np.shape(data.points), (4, 8, 3))
        np.testing.assert_array_equal(data.weights, np.tile(PIPE_W, (4, 1)))
        np.testing.assert_allclose(
            data.evaluate(np.array(UU), np.array(PIPE_V)), _on_surface(pipe, UU, PIPE_V), atol=1e-12
        )
        # a sphere weighted through its plug (V fastest: cv[2][1]), which Maya repeats at cv[2][9]
        ball = _ball()
        cmds.setAttr(f"{ball}.weights[{2 * 11 + 1}]", 3.0)
        self.assertEqual(_weights(ball).reshape(7, 11)[2].tolist(), [1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1])
        expected       = np.ones((7, 8))
        expected[2, 1] = 3.0
        np.testing.assert_array_equal(NurbsSurface(ball).serialize().weights, expected)

    def test_world_space_reads_a_weighted_surface_cv_where_it_is(self):
        quarter = _quarter()
        cmds.xform("quarter", **HERE)
        node  = NurbsSurface(quarter)
        world = np.array(node.get_points(world_space=True))
        np.testing.assert_allclose(world[:, :3], _world(quarter), atol=1e-12)
        np.testing.assert_array_equal(world[:, 3], np.array(_quarter_cvs())[:, 3])
        data = node.serialize(world_space=True)
        np.testing.assert_allclose(np.reshape(data.points, (-1, 3)), _world(quarter), atol=1e-12)
        np.testing.assert_array_equal(np.ravel(data.weights), np.array(_quarter_cvs())[:, 3])
        np.testing.assert_allclose(
            data.evaluate(np.array(UU), np.array(VV)), _on_surface(quarter, UU, VV), atol=1e-9
        )

    def test_a_weight_no_surface_evaluates_is_refused_naming_its_cv(self):
        cvs    = _quarter_cvs()
        cvs[3] = cvs[3][:3] + (0.0,)  # V fastest: cv[1][1]
        shape  = _api_surface(cvs, name="zero")
        with self.assertRaisesRegex(
            ValueError,
            r"^NurbsSurface\('zeroShape'\)\.serialize\(\): cv\[1\]\[1\] holds the weight 0\.0, and a rational "
            r"surface's weights are finite and above zero$",
        ):
            NurbsSurface(shape).serialize()

    def test_the_geometry_plugs_read_the_weights(self):
        # a plug's read is the shape's serialize, in the plug's space
        arc = _arc()
        cmds.xform("arc", **HERE)
        world = NurbsCurve(arc).worldSpace[0].get()
        np.testing.assert_allclose(world.points, _world(arc), atol=1e-12)
        np.testing.assert_array_equal(world.weights, [1.0, 3.0, 1.0])
        local = NurbsCurve(arc).local.get()
        np.testing.assert_array_equal(local.points, np.array(ARC)[:, :3])
        np.testing.assert_array_equal(local.weights, [1.0, 3.0, 1.0])

        quarter = _quarter()
        cmds.xform("quarter", **HERE)
        world = NurbsSurface(quarter).worldSpace[0].get()
        np.testing.assert_allclose(np.reshape(world.points, (-1, 3)), _world(quarter), atol=1e-12)
        np.testing.assert_array_equal(np.ravel(world.weights), np.array(_quarter_cvs())[:, 3])
        self.assertIsNone(NurbsCurve(_flat()).worldSpace[0].get().weights)


# ---------------------------------------------------------------------------------------------
class TestCreate(_Case):
    """``create`` makes a rational shape of weights other than 1, and the shape it made before
    of any other input."""

    def test_data_with_weights_makes_a_rational_curve(self):
        arc  = _arc()
        data = NurbsCurve(arc).serialize()
        self.ready()
        made = NurbsCurve.create(data, name="twin", uuids=False)
        np.testing.assert_array_equal(_held("twinShape"), ARC)
        np.testing.assert_allclose(_on_curve("twinShape", AT), _on_curve(arc, AT), atol=1e-12)
        np.testing.assert_array_equal(made.serialize().weights, [1.0, 3.0, 1.0])
        self.assert_kept_selection()

    def test_points_with_a_w_hold_each_cv_s_weight(self):
        # A: (N, 4) points whose w was not 1 were a ValueError
        for label, points in (
            ("a list", ARC),
            ("an array", np.array(ARC)),
            ("MPoints", [OpenMaya.MPoint(*point) for point in ARC]),
            ("get_points", NurbsCurve(_arc()).get_points()),
        ):
            with self.subTest(label):
                made  = NurbsCurve.create(points, degree=2)
                shape = made.get_long_name()
                np.testing.assert_array_equal(_held(shape), ARC)
                np.testing.assert_allclose(_on_curve(shape, [0.5]), [MIDDLE], atol=1e-12)
                np.testing.assert_array_equal(made.serialize().weights, [1.0, 3.0, 1.0])

    def test_weights_of_1_make_the_curve_made_without_any(self):
        data = NurbsCurve(_flat()).serialize()
        for label, points, degree in (
            ("(N, 4) points", ONES, 2),
            ("data that holds none", data, None),
            ("data whose weights are 1", dataclasses.replace(data, weights=np.ones(3)), None),
        ):
            with self.subTest(label):
                made = NurbsCurve.create(points, degree, uuids=False)
                self.assertIsNone(made.serialize().weights)
                self.assertFalse(_takes_weights(made.get_long_name()))
        # the control: a rational curve takes them, were it rational by its flag alone
        self.assertTrue(_takes_weights(_api_curve(ONES)))
        self.assertTrue(_takes_weights(NurbsCurve.create(ARC, degree=2).get_long_name()))

    def test_a_periodic_curve_repeats_its_weights_as_its_points(self):
        ring    = _ring()
        weights = np.arange(1.0, 9.0)
        data    = dataclasses.replace(NurbsCurve(ring).serialize(), weights=weights)
        made    = NurbsCurve.create(data, name="made", uuids=False)
        np.testing.assert_array_equal(_weights("madeShape"),     weights[REPEAT])
        np.testing.assert_array_equal(_held("madeShape")[:, :3], _held(ring)[:, :3])
        np.testing.assert_array_equal(made.serialize().weights,  weights)
        np.testing.assert_allclose(_on_curve("madeShape", AROUND), data.compute(np.array(AROUND))[0], atol=1e-12)

    def test_the_weights_go_where_the_data_sits(self):
        arc = _arc()
        cmds.xform("arc", **HERE)
        data  = NurbsCurve(arc).serialize()
        world = _world(arc)
        on    = _on_curve(arc, AT)
        cmds.xform(cmds.createNode("transform", name="grp"), **THERE)
        for label, kwargs in (
            ("under a new transform", {}),
            ("frozen in the world", dict(world_space=True)),
            ("under a transform", dict(transform="grp", world_space=True)),
        ):
            with self.subTest(label):
                shape = NurbsCurve.create(data, uuids=False, **kwargs).get_long_name()
                np.testing.assert_array_equal(_weights(shape), [1.0, 3.0, 1.0])
                np.testing.assert_allclose(_world(shape), world, atol=1e-9)
                np.testing.assert_allclose(_on_curve(shape, AT), on, atol=1e-9)

    def test_weights_no_curve_holds_make_nothing(self):
        data = NurbsCurve(_flat()).serialize()
        self.ready()
        for label, weights, text in (
            ("too few", [1.0, 2.0], r"must hold one value per control point, shape \(3,\), got shape \(2,\)"),
            ("a zero", [1.0, 0.0, 1.0], BAD),
            ("a negative", [1.0, -2.0, 1.0], BAD),
            ("a nan", [1.0, float("nan"), 1.0], BAD),
            ("an inf", [1.0, float("inf"), 1.0], BAD),
        ):
            with self.subTest(label):
                bad = dataclasses.replace(data, weights=np.array(weights))
                self.assert_refused(
                    ValueError, rf"^NurbsCurve\.create\(\) weights {text}$", lambda: NurbsCurve.create(bad, uuids=False)
                )
                self.assertTrue(_nothing_to_undo())
        # cgmath's own data takes numbers, on (N, 3) points: a stand-in for data that does not
        self.assert_refused(
            TypeError, r"^NurbsCurve\.create\(\) weights must be numbers, got list$",
            lambda: NurbsCurve.create(_Stand(weights=["a", "b", "c"])),
        )
        self.assert_refused(
            ValueError,
            r"^NurbsCurve\.create\(\) takes a NurbsCurveData of \(N, 3\) points: this one's are \(N, 4\) with a w "
            r"other than 1 \(a rational curve's weights go in its weights field\)$",
            lambda: NurbsCurve.create(_Stand(cv=np.array(ARC))),
        )

    def test_data_with_weights_makes_a_rational_surface(self):
        quarter = _quarter()
        data    = NurbsSurface(quarter).serialize()
        self.ready()
        made = NurbsSurface.create(data, name="twin", uuids=False)
        np.testing.assert_array_equal(_held("twinShape"), _held(quarter))
        on = _on_surface("twinShape", UU, VV)
        np.testing.assert_allclose(on, _on_surface(quarter, UU, VV), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(on[:, :2], axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(made.serialize().weights, data.weights)
        self.assert_kept_selection()

    def test_weights_of_1_make_the_surface_made_without_any(self):
        data = NurbsSurface(_bent()).serialize()
        for label, plain in (
            ("data that holds none", data),
            ("data whose weights are 1", dataclasses.replace(data, weights=np.ones((3, 2)))),
        ):
            with self.subTest(label):
                made = NurbsSurface.create(plain, uuids=False)
                self.assertIsNone(made.serialize().weights)
                self.assertFalse(_takes_weights(made.get_long_name()))
        self.assertTrue(_takes_weights(_api_surface(_quarter_cvs(1.0))))

    def test_a_periodic_surface_repeats_its_weights_as_its_points(self):
        pipe = _pipe()
        NurbsSurface.create(NurbsSurface(pipe).serialize(), name="made", uuids=False)
        np.testing.assert_array_equal(_held("madeShape"), _held(pipe))
        np.testing.assert_allclose(_on_surface("madeShape", UU, PIPE_V), _on_surface(pipe, UU, PIPE_V), atol=1e-12)

        # onto the layout of a sphere, which never held a weight
        ball    = _ball()
        weights = _graded(7, 8)
        data    = dataclasses.replace(NurbsSurface(ball).serialize(), weights=weights)
        made    = NurbsSurface.create(data, name="weighted", uuids=False)
        np.testing.assert_array_equal(_weights("weightedShape").reshape(7, 11), weights[:, REPEAT])
        np.testing.assert_array_equal(_held("weightedShape")[:, :3], _held(ball)[:, :3])
        np.testing.assert_array_equal(made.serialize().weights, weights)
        uu, vv = [0.5, 2.2, 3.6], PIPE_V
        np.testing.assert_allclose(
            _on_surface("weightedShape", uu, vv), data.evaluate(np.array(uu), np.array(vv)), atol=1e-12
        )

    def test_weights_no_surface_holds_make_nothing(self):
        data = NurbsSurface(_bent()).serialize()
        self.ready()
        for label, weights, text in (
            ("a column short", [[2.0], [2.0], [2.0]],
             r"must hold one value per control point, shape \(3, 2\), got shape \(3, 1\)"),
            ("a zero", [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], BAD),
            ("a negative", [[1.0, 1.0], [1.0, -2.0], [1.0, 1.0]], BAD),
            ("a nan", [[1.0, 1.0], [1.0, 1.0], [float("nan"), 1.0]], BAD),
        ):
            with self.subTest(label):
                bad = dataclasses.replace(data, weights=np.array(weights))
                self.assert_refused(
                    ValueError, rf"^NurbsSurface\.create\(\) weights {text}$",
                    lambda: NurbsSurface.create(bad, uuids=False),
                )
                self.assertTrue(_nothing_to_undo())

    def test_each_create_is_one_undo_step_that_redo_brings_back_weighted(self):
        data = NurbsSurface(_quarter()).serialize()
        held = _held("quarterShape")
        self.ready()
        NurbsCurve.create(ARC, degree=2, name="made")
        self.assert_one_step("rig.NurbsCurve.create", lambda: cmds.objExists("made"))
        np.testing.assert_array_equal(_held("madeShape"), ARC)
        NurbsSurface.create(data, name="patch", uuids=False)
        self.assert_one_step("rig.NurbsSurface.create", lambda: cmds.objExists("patch"))
        np.testing.assert_array_equal(_held("patchShape"), held)
        self.assert_kept_selection()


# ---------------------------------------------------------------------------------------------
class TestLoadInPlace(_Case):
    """``NurbsCurve('x').load`` / ``NurbsSurface('x').load`` set the data's weights with the CVs,
    in the one undo step."""

    def walk(self, shape, before, after):
        """The load just made left `shape` holding `after`: one step named for it, the
        selection kept; undo puts back `before` (CVs and weights), redo `after`. A test walks
        before it evaluates: ``cmds.pointOnCurve`` and ``cmds.pointOnSurface`` record a step of
        their own in mayapy (measured)."""
        kind = "NurbsCurve" if isinstance(_fn(shape), OpenMaya.MFnNurbsCurve) else "NurbsSurface"
        np.testing.assert_array_equal(_held(shape), after)
        self.assert_kept_selection()
        self.assertEqual(self.undo_name(), f"rig.{kind}.load")
        self.undo_steps(1)
        np.testing.assert_array_equal(_held(shape), before)
        self.assertTrue(_nothing_to_undo())
        self.redo_steps(1)
        np.testing.assert_array_equal(_held(shape), after)

    def test_weighted_data_onto_a_curve_that_held_none(self):
        data   = NurbsCurve(_arc()).serialize()
        flat   = _flat()
        before = _held(flat)
        node   = NurbsCurve(flat)
        self.ready()
        self.assertIs(node.load(data), node)
        self.walk(flat, before, np.array(ARC))
        np.testing.assert_allclose(_on_curve(flat, [0.5]), [MIDDLE], atol=1e-12)

    def test_data_without_weights_puts_every_weight_back_to_1(self):
        data = NurbsCurve(_flat()).serialize()
        arc  = _arc()
        self.ready()
        NurbsCurve(arc).load(data)
        self.walk(arc, np.array(ARC), np.column_stack([FLAT, np.ones(3)]))
        np.testing.assert_allclose(_on_curve(arc, [0.5]), [[1.0, 0.5, 0.0]], atol=1e-12)

    def test_weighted_data_onto_a_rational_curve(self):
        arc   = _arc()
        other = dataclasses.replace(NurbsCurve(_flat()).serialize(), weights=np.array([2.0, 0.5, 4.0]))
        self.ready()
        NurbsCurve(arc).load(other)
        self.walk(arc, np.array(ARC), np.column_stack([FLAT, [2.0, 0.5, 4.0]]))
        np.testing.assert_allclose(_on_curve(arc, AT), other.compute(np.array(AT))[0], atol=1e-12)

    def test_world_space_lands_the_cvs_where_the_data_sits(self):
        arc, flat = _arc(), _flat()
        cmds.xform("arc", **HERE)
        cmds.xform("flat", **THERE)
        for label, source, target, weights in (
            ("weights onto none", arc, _flat("onto_none"), [1.0, 3.0, 1.0]),
            ("none onto weights", flat, _arc("onto_weights"), [1.0, 1.0, 1.0]),
            ("weights onto weights", arc, _api_curve([(3.0, 1.0, 2.0, 5.0)] * 3, name="onto_others"), [1.0, 3.0, 1.0]),
        ):
            with self.subTest(label):
                xform = cmds.listRelatives(target, parent=True)[0]
                cmds.xform(xform, translation=(1, -4, 2), rotation=(-30, 20, 45), scale=(0.5, 3, 2))
                matrix = cmds.xform(xform, query=True, matrix=True, worldSpace=True)
                before = _held(target)
                self.ready()
                NurbsCurve(target).load(NurbsCurve(source).serialize(), world_space=True)
                self.assertEqual(self.undo_name(), "rig.NurbsCurve.load")
                np.testing.assert_allclose(_world(target), _world(source), atol=1e-9)
                np.testing.assert_array_equal(_weights(target), weights)
                self.assertEqual(cmds.xform(xform, query=True, matrix=True, worldSpace=True), matrix)
                self.undo_steps(1)
                np.testing.assert_array_equal(_held(target), before)
                self.redo_steps(1)
                np.testing.assert_allclose(_world(target), _world(source), atol=1e-9)
                np.testing.assert_allclose(_on_curve(target, AT), _on_curve(source, AT), atol=1e-9)

    def test_a_periodic_curve(self):
        ring    = _ring()
        before  = _held(ring)
        weights = np.arange(1.0, 9.0)
        data    = dataclasses.replace(NurbsCurve(ring).serialize(), weights=weights)
        self.ready()
        NurbsCurve(ring).load(data)
        self.walk(ring, before, np.column_stack([before[:, :3], weights[REPEAT]]))
        np.testing.assert_array_equal(NurbsCurve(ring).serialize().weights, weights)
        np.testing.assert_allclose(_on_curve(ring, AROUND), data.compute(np.array(AROUND))[0], atol=1e-12)

    def test_a_surface(self):
        data   = NurbsSurface(_quarter()).serialize()
        bent   = _bent()
        before = _held(bent)
        node   = NurbsSurface(bent)
        self.ready()
        self.assertIs(node.load(data), node)
        self.walk(bent, before, _held("quarterShape"))
        # and back: data that holds none onto the surface that now holds some
        self.ready()
        NurbsSurface(bent).load(dataclasses.replace(data, weights=None))
        self.walk(bent, _held("quarterShape"), before)
        self.undo_steps(1)  # the quarter's weights again
        on = _on_surface(bent, UU, VV)
        np.testing.assert_allclose(np.linalg.norm(on[:, :2], axis=1), 1.0, atol=1e-12)

    def test_a_surface_s_weights_follow_its_cvs_order(self):
        plane   = _shape(cmds.nurbsPlane(name="plane", patchesU=2, patchesV=3, constructionHistory=False)[0])
        before  = _held(plane)
        weights = _graded(5, 6)
        data    = dataclasses.replace(NurbsSurface(plane).serialize(), weights=weights)
        self.ready()
        NurbsSurface(plane).load(data)
        self.walk(plane, before, np.column_stack([before[:, :3], weights.reshape(-1)]))
        np.testing.assert_allclose(
            _on_surface(plane, UU, VV), data.evaluate(np.array(UU), np.array(VV)), atol=1e-12
        )

    def test_a_periodic_surface(self):
        ball    = _ball()
        before  = _held(ball)
        weights = _graded(7, 8)
        data    = dataclasses.replace(NurbsSurface(ball).serialize(), weights=weights)
        self.ready()
        NurbsSurface(ball).load(data)
        self.walk(ball, before, np.column_stack([before[:, :3], weights[:, REPEAT].reshape(-1)]))
        np.testing.assert_array_equal(NurbsSurface(ball).serialize().weights, weights)
        uu, vv = [0.5, 2.2, 3.6], PIPE_V
        np.testing.assert_allclose(_on_surface(ball, uu, vv), data.evaluate(np.array(uu), np.array(vv)), atol=1e-12)

    def test_world_space_lands_a_surface_s_cvs_where_the_data_sits(self):
        quarter, bent = _quarter(), _bent()
        cmds.xform("quarter", **HERE)
        cmds.xform("bent", **THERE)
        before = _held(bent)
        self.ready()
        NurbsSurface(bent).load(NurbsSurface(quarter).serialize(), world_space=True)
        self.assertEqual(self.undo_name(), "rig.NurbsSurface.load")
        np.testing.assert_allclose(_world(bent), _world(quarter), atol=1e-9)
        np.testing.assert_array_equal(_weights(bent), _weights(quarter))
        self.undo_steps(1)
        np.testing.assert_array_equal(_held(bent), before)
        self.redo_steps(1)
        np.testing.assert_allclose(_world(bent), _world(quarter), atol=1e-9)
        np.testing.assert_allclose(_on_surface(bent, UU, VV), _on_surface(quarter, UU, VV), atol=1e-9)

    def test_data_without_weights_onto_a_shape_without_is_maya_s_one_write(self):
        # B: only weights, the data's or the shape's own, take the weighted write
        flat  = _flat()
        data  = NurbsCurve(_flat("other")).serialize()
        write = nurbs_module._SetCVsCommand
        with mock.patch.object(write, "_put_weighted", autospec=True, side_effect=write._put_weighted) as weighted:
            NurbsCurve(flat).load(data)
            weighted.assert_not_called()
            self.assertFalse(_takes_weights(flat))  # still the curve that never held one
            NurbsCurve(flat).load(dataclasses.replace(data, weights=np.array([1.0, 3.0, 1.0])))
            self.assertEqual(weighted.call_count, 1)
            NurbsCurve(flat).load(data)  # it holds weights now: back to 1 by the same write
            self.assertEqual(weighted.call_count, 2)
        np.testing.assert_array_equal(_held(flat), np.column_stack([FLAT, np.ones(3)]))

    def test_weights_no_shape_holds_change_nothing(self):
        flat, bent = _flat(), _bent()
        curve   = NurbsCurve(flat).serialize()
        surface = NurbsSurface(bent).serialize()
        self.ready()
        for label, node, data, weights, text in (
            ("a curve: too few", NurbsCurve(flat), curve, [1.0, 2.0],
             r"must hold one value per control point, shape \(3,\), got shape \(2,\)"),
            ("a curve: a zero", NurbsCurve(flat), curve, [1.0, 0.0, 1.0], BAD),
            ("a curve: a negative", NurbsCurve(flat), curve, [1.0, -2.0, 1.0], BAD),
            ("a curve: a nan", NurbsCurve(flat), curve, [1.0, float("nan"), 1.0], BAD),
            ("a surface: a column short", NurbsSurface(bent), surface, [[2.0], [2.0], [2.0]],
             r"must hold one value per control point, shape \(3, 2\), got shape \(3, 1\)"),
            ("a surface: a negative", NurbsSurface(bent), surface, [[1.0, 1.0], [1.0, -2.0], [1.0, 1.0]], BAD),
        ):
            with self.subTest(label):
                kind   = type(node).__name__
                shape  = node.get_long_name()
                before = _held(shape)
                bad    = dataclasses.replace(data, weights=np.array(weights))
                self.assert_refused(
                    ValueError, rf"^{kind}\('{node}'\)\.load: the weights {text}$",
                    lambda: node.load(bad),
                )
                np.testing.assert_array_equal(_held(shape), before)
                self.assertTrue(_nothing_to_undo())


# ---------------------------------------------------------------------------------------------
class TestFiles(_Case):
    """A file holds what ``serialize`` writes: the weights of a rational shape, none of any
    other."""

    def test_the_round_trip_in_each_format(self):
        arc, quarter = _arc(), _quarter()
        for ext in FORMATS:
            with self.subTest(ext=ext, kind="curve"):
                path = self.path(f"arc.{ext}")
                NurbsCurve(arc).save(path)
                made = NurbsCurve.load(path, uuids=False)
                np.testing.assert_array_equal(_held(made.get_long_name()), ARC)
                flat = _flat(f"flat_{ext}")
                NurbsCurve(flat).load(path)
                np.testing.assert_array_equal(_held(flat), ARC)
            with self.subTest(ext=ext, kind="surface"):
                path = self.path(f"quarter.{ext}")
                NurbsSurface(quarter).save(path)
                made = NurbsSurface.load(path, uuids=False)
                np.testing.assert_array_equal(_held(made.get_long_name()), _held(quarter))
                bent = _bent(f"bent_{ext}")
                NurbsSurface(bent).load(path)
                np.testing.assert_array_equal(_held(bent), _held(quarter))

    def test_a_json_holds_weights_for_a_rational_shape_alone(self):
        def saved(node, name):
            node.save(self.path(name))
            with open(self.path(name), encoding="utf-8") as handle:
                return json.load(handle)

        self.assertNotIn("weights", saved(NurbsCurve(_flat()), "flat.json"))
        self.assertNotIn("weights", saved(NurbsCurve(_ring()), "ring.json"))
        self.assertEqual(saved(NurbsCurve(_arc()), "arc.json")["weights"], [1.0, 3.0, 1.0])
        self.assertNotIn("weights", saved(NurbsSurface(_bent()), "bent.json"))
        self.assertEqual(saved(NurbsSurface(_quarter()), "quarter.json")["weights"], [[1.0, 1.0], [S2, S2], [1.0, 1.0]])
