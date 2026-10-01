"""
Nurbs node class for NurbsCurve and NurbsSurface
"""

from __future__ import annotations

import logging
from numbers import Integral

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig._internal.undo import _undo_chunk
from rig.nodetypes.geometry import Geometry, _shape_create_name

LOGGER = logging.getLogger(__name__)


class NurbsCurve(Geometry):
    """
    nurbsCurve node class
    """

    NATIVE_NODE_TYPE = "nurbsCurve"
    FN_SET           = OpenMaya.MFnNurbsCurve
    POINT_COMP_TYPE  = "cv"

    # built from its points (``Node.create("nurbsCurve")`` with none raises, naming them)
    _CREATE_TAKES_INPUTS = "points"

    # the keywords ``_create`` takes; ``create``'s other keywords are the
    # curve's attributes
    _CREATE_FLAGS = frozenset({"name", "degree", "kv"})

    _DEFINE_REFUSED = (
        "a nurbsCurve is built from its points: NurbsCurve.create(points, name='x') "
        "makes one; NurbsCurve('x') refers to one"
    )

    # --- creation

    @classmethod
    def _create(
        cls,
        points: ArrayLike | BSplineData,
        degree: int       | None = None,
        kv:     ArrayLike | None = None,
        name:   str       | None = None,
        **kwargs,
    ) -> str:
        """[Internal] Builds the curve of `points` (control points with their
        `degree` and `kv`, or a BSplineData) and returns the transform's name.
        `NurbsCurve.create` wraps it in its undo chunk; see there."""
        return _create_curve(points, degree, kv, name)

    @classmethod
    def create(
        cls,
        points:    ArrayLike | BSplineData,
        degree:    int       | None = None,
        kv:        ArrayLike | None = None,
        *,
        name:      str       | None = None,
        container: bool      | None = None,
        **kwargs,
    ) -> "NurbsCurve":
        """Creates a curve object from its control points, or from a
        BSplineData object.

        Args:
            points: The control points (CVs), as ``(N, 3)`` numbers (a list, an
                array, MPoints or the ``MPointArray`` `get_points` returns), in
                the curve's object space. Or a BSplineData: the curve is built
                from its ``points``, ``degree`` and ``kv``, and is periodic
                when the data is (its first ``degree`` points are then repeated
                at the end, as Maya stores a periodic curve). The data's
                ``uniform`` and ``registered`` settings have no Maya
                equivalent and are not used.
            degree: The degree of a curve given as points. None: 3.
            kv: The knots of a curve given as points, in Maya's layout:
                ``N + degree - 1`` values that never decrease, none repeated
                more than ``degree`` times. None: Maya's default, the knots
                ``cmds.curve(point=..., degree=...)`` gives (one apart, the
                end ones repeated ``degree`` times: ``0 0 0 1 2 3 3 3`` for six
                points of degree 3).
            name: The name of the curve to create, a keyword (a trailing
                ``Shape<digits>`` is dropped: ``"cShape2"`` names the transform
                ``c2`` and the shape ``cShape2``). None: ``curve<N>``.
            container: Inside ``with container()``, whether the transform and
                the shape are registered with the scope (None: yes), as for
                every typed create (see `DGNode.create`).
            kwargs: attributes of the curve, checked by type before anything
                is made and set once it exists (see `DGNode.create`).

        A BSplineData carries its own degree and knots: ``degree`` or ``kv``
        given with one is a TypeError.

        The whole call is ONE undo step named ``rig.NurbsCurve.create``, as
        `Mesh.create` is: one ``cmds.undo()`` removes the transform and the
        shape (and, in a scope, the registration), and ``cmds.redo()`` brings
        back the same node (its UUID, and any NurbsCurve object held on it,
        stay valid). The curve rides a recorded ``createNode`` transform: the
        shape is made under it through the API, so the points are stored as
        given, in Maya's internal unit (what `get_points` reads), whatever the
        scene's linear unit is (``cmds.curve`` reads its points in that unit).
        It never changes the selection and leaves no construction history.

        Input that is no curve (points that are not ``(N, 3)`` finite numbers,
        fewer than ``degree + 1`` of them, a degree under 1, knots of the wrong
        count or order) raises TypeError or ValueError before anything is
        made; an error after the transform is made deletes it again before it
        propagates.
        """
        with _undo_chunk("rig.NurbsCurve.create"):
            return super().create(
                points, degree=degree, kv=kv, name=name, container=container, **kwargs
            )

    # --- curve geometry data methods

    @property
    def num_cvs(self) -> int:
        """Returns the number of CVs."""
        return self.fn_set.numCVs

    @property
    def num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count = self.fn_set.numCVs
        if self.fn_set.form == OpenMaya.MFnNurbsCurve.kPeriodic:
            cv_count -= self.fn_set.degree
        return cv_count

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsCurve.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.cvPositions(space)

    # --- serialization

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> BSplineData:
        """Serialize this nurbsCurve to BSplineData, the inverse of `create`.

        The data is periodic when the curve is (Maya's ``kPeriodic`` form; a
        ``kClosed`` curve, whose ends only meet, is open), and holds the
        curve's own points: a periodic curve's last ``degree`` CVs repeat its
        first ones, and BSplineData wraps them back itself. Knots other than
        Maya's default ones go in its ``knots`` when the installed cgmath's
        BSplineData takes them. That cgmath keeps their spacing, so the
        curve's shape, but rescales them to its own parameter range,
        ``[0, max_param]``. An older one has no such field: the knots are
        dropped and the data has uniform ones, which is the same curve only
        when the knots were evenly spaced (a warning says so otherwise).

        Args:
            world_space: If True, query points in world space, otherwise object space.
            include_uvs: Unused; kept for signature parity with `Mesh.serialize`.

        Returns:
            BSplineData object.
        """
        from cgmath.geometry import BSplineData

        fn       = self.fn_set
        degree   = fn.degree
        periodic = fn.form == OpenMaya.MFnNurbsCurve.kPeriodic
        points   = np.array(self.get_points(world_space=world_space))[:, :3]
        if periodic:
            points = points[: len(points) - degree]

        custom  = {}
        knots   = np.array(fn.knots())
        uniform = _default_knots(len(points), degree, periodic)
        if not np.array_equal(knots, uniform):
            if "knots" in getattr(BSplineData, "__dataclass_fields__", ()):
                custom["knots"] = knots
            elif not _same_spacing(knots, uniform, degree):
                LOGGER.warning(
                    f"{self.name}: this cgmath's BSplineData takes no knots, so the "
                    f"curve's uneven knots are dropped and its data has another shape"
                )
        return BSplineData(points=points, degree=degree, periodic=periodic, **custom)


class NurbsSurface(Geometry):
    """
    nurbsSurface node class
    """

    NATIVE_NODE_TYPE = "nurbsSurface"
    FN_SET           = OpenMaya.MFnNurbsSurface
    POINT_COMP_TYPE  = "cv"

    _DEFINE_REFUSED = (
        "a nurbsSurface is a shape built from its data: rc.surface(...) or "
        "rc.nurbsPlane(name='x') makes one; NurbsSurface('x') refers to one"
    )

    @property
    def num_cvs(self) -> int:
        """Returns the number of cvs."""
        return self.fn_set.numCVsInU * self.fn_set.numCVsInV

    @property
    def num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count_u = self.fn_set.numCVsInU
        if self.fn_set.formInU == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_u -= self.fn_set.degreeInU
        cv_count_v = self.fn_set.numCVsInV
        if self.fn_set.formInV == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_v -= self.fn_set.degreeInV
        return cv_count_u * cv_count_v

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsSurface.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.cvPositions(space)

    # --- serialization

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> BSplinePatchData:
        """Serialize this nurbsSurface to BSplinePatchData.

        Args:
            world_space: If True, query points in world space, otherwise object space.
            include_uvs: Unused; kept for signature parity with `Mesh.serialize`
                and `NurbsCurve.serialize`. BSplinePatchData does not store UVs.

        Returns:
            A BSplinePatchData object.
        """
        from cgmath.geometry import BSplinePatchData

        points = self.get_points(world_space=world_space)

        num_cvs_u  = self.fn_set.numCVsInU
        num_cvs_v  = self.fn_set.numCVsInV
        degree_u   = self.fn_set.degreeInU
        degree_v   = self.fn_set.degreeInV
        periodic_u = self.fn_set.formInU != OpenMaya.MFnNurbsSurface.kOpen
        periodic_v = self.fn_set.formInV != OpenMaya.MFnNurbsSurface.kOpen

        # Maya's cvPositions() returns a flat MPointArray of length
        # numCVsInU * numCVsInV ordered with V varying fastest -- i.e.
        # flat[u * numCVsInV + v] == CV at grid position (u, v). A C-order
        # reshape into (numCVsInU, numCVsInV, 3) gives the expected
        # `grid[u, v]` layout. The trailing homogeneous-w column from the
        # MPointArray is dropped via `[:, :3]`.
        grid = np.array(points)[:, :3].reshape(num_cvs_u, num_cvs_v, 3)

        # For periodic directions Maya exposes wrapped CVs (the first
        # `degree` rows/cols repeated at the end). BSplinePatchData expects
        # only the unique CVs and re-applies the wrap internally via its
        # geometry index, so trim them here for periodic axes.
        if periodic_u:
            grid = grid[: num_cvs_u - degree_u, :, :]
        if periodic_v:
            grid = grid[:, : num_cvs_v - degree_v, :]

        return BSplinePatchData(
            points     = grid,
            degree_u   = degree_u,
            degree_v   = degree_v,
            periodic_u = periodic_u,
            periodic_v = periodic_v,
        )


def _default_knots(count: int, degree: int, periodic: bool) -> np.ndarray:
    """Maya's default knots, in its layout, for a curve of `count` points of
    `degree`: one apart, the end ones of an open curve repeated `degree` times
    (``cmds.curve``'s knots: ``0 0 0 1 2 3 3 3`` for six points of degree 3),
    or, on a periodic curve (`count` points before the wrap), running past
    both ends (``cmds.circle``'s: ``-2 -1 0 ... 9 10`` for eight points)."""
    if periodic:
        return np.arange(1.0 - degree, count + degree)
    return np.clip(np.arange(count + degree - 1) - (degree - 1.0), 0.0, count - degree)


def _same_spacing(knots: np.ndarray, other: np.ndarray, degree: int) -> bool:
    """True when two knot vectors (Maya's layout) of one curve give it the same
    shape: equal once each is scaled to its own parameter range, from its
    ``degree``-th knot to its ``degree``-th from the end."""

    def unit(k):
        return (k - k[degree - 1]) / (k[-degree] - k[degree - 1])

    return bool(np.allclose(unit(knots), unit(other), rtol=0.0, atol=1e-9))


def _curve_inputs(
    points: ArrayLike | BSplineData, degree: int | None, kv: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """`NurbsCurve.create`'s input, checked, as ``MFnNurbsCurve.create`` takes
    it: ``(cvs (N, 3), knots (N + degree - 1,), degree, periodic)``. It makes
    nothing; input that is no curve raises TypeError or ValueError.

    A BSplineData gives its ``cv`` (its points, a periodic curve's first
    ``degree`` repeated at the end), ``degree`` and ``kv``. Points take
    `degree` (None: 3) and `kv` (None: Maya's default knots)."""
    # a BSplineData, duck-typed so cgmath is never imported here; read on the
    # class, so a node or a plug (whose attributes are Maya's) is never asked
    if hasattr(type(points), "kv") and hasattr(type(points), "cv"):
        given = [key for key, value in (("degree", degree), ("kv", kv)) if value is not None]
        if given:
            raise TypeError(
                f"NurbsCurve.create() got {' and '.join(given)} with a "
                f"{type(points).__name__}, which carries its own degree and knots "
                f"(degree= and kv= describe points)"
            )
        periodic = bool(getattr(points, "periodic", False))
        degree   = points.degree
        kv       = points.kv
        points   = points.cv
    else:
        periodic = False
        if degree is None:
            degree = 3

    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError(f"NurbsCurve.create() degree must be an int, got {degree!r}")
    degree = int(degree)
    if degree < 1:  # MFnNurbsCurve.create crashes Maya on degree 0
        raise ValueError(f"NurbsCurve.create() degree must be 1 or more, got {degree}")

    try:
        cvs = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        cvs = None
    if cvs is None or cvs.ndim == 0:
        raise TypeError(
            f"NurbsCurve.create() takes (N, 3) points or a BSplineData, got "
            f"{type(points).__name__}"
        )
    if cvs.ndim == 2 and cvs.shape[1] == 4:
        # homogeneous points (MPoints, an MPointArray): a non-rational curve's w is 1
        if not np.all(cvs[:, 3] == 1.0):
            raise ValueError("NurbsCurve.create() makes non-rational curves: (N, 4) points take w = 1")
        cvs = cvs[:, :3]
    if cvs.ndim != 2 or cvs.shape[1] != 3:
        raise ValueError(f"NurbsCurve.create() points must be (N, 3), got {cvs.shape}")
    count = cvs.shape[0]
    if count < degree + 1:
        raise ValueError(
            f"a curve of degree {degree} takes {degree + 1} points or more, got {count}"
        )
    if not np.isfinite(cvs).all():
        raise ValueError("NurbsCurve.create() points must be finite (got nan or inf)")

    size = count + degree - 1
    if kv is None:
        return cvs, _default_knots(count, degree, False), degree, periodic

    try:
        knots = np.asarray(kv, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(
            f"NurbsCurve.create() kv must be numbers, got {type(kv).__name__}"
        ) from None
    if knots.ndim != 1 or knots.shape[0] != size:
        # scipy and The NURBS Book write one more knot at each end
        padded = " (leave out the first and the last)" if knots.shape == (size + 2,) else ""
        raise ValueError(
            f"{count} points of degree {degree} take {size} knots (Maya's layout: "
            f"points + degree - 1), got {knots.shape[0] if knots.ndim == 1 else knots.shape}"
            f"{padded}"
        )
    if not np.isfinite(knots).all():
        raise ValueError("NurbsCurve.create() kv must be finite (got nan or inf)")
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("NurbsCurve.create() kv must never decrease")
    repeats = int(np.unique(knots, return_counts=True)[1].max())
    if repeats > degree:
        raise ValueError(
            f"a knot of a degree {degree} curve repeats {degree} times at most "
            f"(Maya's layout), got one {repeats} times"
        )
    if not knots[size - degree] > knots[degree - 1]:
        raise ValueError("NurbsCurve.create() kv must cover a parameter range longer than zero")
    return cvs, knots, degree, periodic


def _create_curve(
    points: ArrayLike | BSplineData,
    degree: int       | None,
    kv:     ArrayLike | None,
    name:   str       | None,
) -> str:
    """`NurbsCurve._create`'s body: the ride, as `Mesh.create`'s. Returns the
    transform's name.

    In this order (`NurbsCurve.create` holds the one undo chunk around it):

    1. check the input (`_curve_inputs`): nothing is made yet;
    2. ``cmds.createNode("transform", name="curve#", skipSelect=True)``, the
       recorded node the curve rides with (``cmds.curve``'s default name);
    3. ``MFnNurbsCurve.create(..., parent=<transform>)``: the shape, under it;
    4. ``cmds.rename`` of the transform (Maya renames the shape after it).

    Any error after step 2 deletes the transform before it propagates.
    """
    cvs, knots, degree, periodic = _curve_inputs(points, degree, kv)
    name = _shape_create_name(name)
    form = OpenMaya.MFnNurbsCurve.kPeriodic if periodic else OpenMaya.MFnNurbsCurve.kOpen

    xform  = cmds.createNode("transform", name="curve#", skipSelect=True)
    sel    = OpenMaya.MSelectionList()
    sel.add(xform)
    parent = sel.getDependNode(0)
    handle = OpenMaya.MObjectHandle(parent)
    try:
        try:
            NurbsCurve.FN_SET().create(
                OpenMaya.MPointArray(cvs.tolist()),
                OpenMaya.MDoubleArray(knots.tolist()),
                degree,
                form,
                False,  # not 2D
                False,  # not rational
                parent,
            )
        except RuntimeError as error:
            # Maya says little ("Unexpected Internal Failure" for a periodic
            # curve whose CVs or knots do not repeat): name the curve
            raise RuntimeError(
                f"NurbsCurve.create(): Maya refused the curve ({len(cvs)} CVs, "
                f"degree {degree}, {'periodic' if periodic else 'open'}): {error}"
            ) from None
        if name:
            # the shape follows the transform (curveShape<N> -> <name>Shape)
            xform = cmds.rename(xform, name)
    except BaseException:
        if handle.isValid():
            cmds.delete(OpenMaya.MFnDagNode(parent).fullPathName())
        raise
    return xform
