"""
Nurbs node class for NurbsCurve and NurbsSurface
"""

from __future__ import annotations

import logging
import os
from numbers import Integral

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig.nodetypes._base import _name_of
from rig._internal.journal import cmds_edit as _cmds_edit, statement as _statement
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files
from rig.nodetypes.geometry import Geometry, _shape_create_name, deformed_refusal
from rig.nodetypes.plugins import _run_undoable

LOGGER = logging.getLogger(__name__)


_CURVE_LOAD_DOC = """Loads a curve from a cgmath file (a BSplineData): one name, two forms.

    ``NurbsCurve.load(curve, *, name=None, container=None, **attrs)``, from the
    class, CREATES a new curve, always (as `create` does)::

        ctl = NurbsCurve.load("ctl.npz", name="ctl")

    ``NurbsCurve('ctlShape').load(curve, *, world_space)``, from a curve,
    UPDATES that curve's CVs in place, the inverse of
    ``NurbsCurve('ctlShape').save(...)``::

        NurbsCurve("ctlShape").load("ctl.npz", world_space=True)

    Args:
        curve: The file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``), or the BSplineData itself.
        name, container, attrs: (creating) as `create` takes them. A
            BSplineData carries no name: None gives Maya's ``curve<N>``.
        world_space: (updating, REQUIRED: no default) the space the file was
            saved in. True: the CVs are world-space positions (``save()``'s
            default); False: object space (``save(world_space=False)``). A
            BSplineData holds no space (a MeshData holds its matrix, a
            BSplineData nothing), so the wrong one would move the CVs
            silently: a load without it, or with anything but a bool, or
            with the space given by position (``load(path, True)``), is a
            TypeError naming it, before the file is read. The transform is
            never moved.

    Returns:
        The new NurbsCurve (creating), or this NurbsCurve (updating).

    The file is read, and what it holds checked, before anything is made or
    changed: a missing file is a FileNotFoundError, a file of another class a
    TypeError naming what it holds, a file from an older cgmath cgmath's
    LegacyFileError. Updating needs the same CV layout (the CV count, the
    degree, the form and the knots) and a curve without history, else a
    ValueError naming ``NurbsCurve.load(...)`` for a new curve; a deformed
    curve (a cluster, a wire ... upstream) names the curve under its
    deformers instead, ``NurbsCurve('ctlShapeOrig').load(path,
    world_space=...)``, whose update the deformed curve follows.

    Each call is ONE undo step named ``rig.NurbsCurve.load`` and leaves the
    selection as it was.
    """

_SURFACE_LOAD_DOC = """Loads a surface from a cgmath file (a BSplinePatchData): one name, two forms.

    ``NurbsSurface.load(surface, *, name=None, container=None, **attrs)``, from
    the class, CREATES a new surface, always (as `create` does)::

        patch = NurbsSurface.load("patch.npz", name="patch")

    ``NurbsSurface('patchShape').load(surface, *, world_space)``, from a
    surface, UPDATES that surface's CVs in place, the inverse of
    ``NurbsSurface('patchShape').save(...)``::

        NurbsSurface("patchShape").load("patch.npz", world_space=True)

    Args:
        surface: The file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``), or the BSplinePatchData itself.
        name, container, attrs: (creating) as `create` takes them. A
            BSplinePatchData carries no name: None gives ``surface<N>``.
        world_space: (updating, REQUIRED: no default) the space the file was
            saved in. True: the CVs are world-space positions (``save()``'s
            default); False: object space. A BSplinePatchData holds no space
            (see `NurbsCurve.load`): a load without it, or with anything but
            a bool, or by position, is a TypeError naming it, before the file
            is read. The transform is never moved.

    Returns:
        The new NurbsSurface (creating), or this NurbsSurface (updating).

    The file is read, and what it holds checked, before anything is made or
    changed (the errors of `NurbsCurve.load`). A BSplinePatchData holds no
    knots: a new surface gets evenly spaced ones (``0..spans`` in each
    direction), and an updated one keeps its own. Updating needs the same CV
    layout (the CV counts, the degrees and the forms in U and V) and a surface
    without history, else a ValueError naming ``NurbsSurface.load(...)`` for a
    new surface (a deformed one names its ``...Orig`` shape's load, with
    ``world_space=``).

    Each call is ONE undo step named ``rig.NurbsSurface.load`` and leaves the
    selection as it was.
    """


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
        "or NurbsCurve.load(path, name='x') makes one; NurbsCurve('x') refers to one"
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

    def get_num_cvs(self) -> int:
        """Returns the number of CVs."""
        return type(self)._raw_get_fn_set(self).numCVs

    def get_num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count = type(self)._raw_get_fn_set(self).numCVs
        if type(self)._raw_get_fn_set(self).form == OpenMaya.MFnNurbsCurve.kPeriodic:
            cv_count -= type(self)._raw_get_fn_set(self).degree
        return cv_count

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsCurve.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).cvPositions(space)

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
        BSplineData takes them (cgmath 2.0.0), which keeps them as they are
        (to rounding): the same curve over the same parameter range. An
        older one (1.0.4) has no such field: the knots are dropped and the
        data has uniform ones, which is the same curve only when the knots
        were evenly spaced (a warning says so otherwise).

        Args:
            world_space: If True, query points in world space, otherwise object space.
            include_uvs: Unused; kept for signature parity with `Mesh.serialize`.

        Returns:
            BSplineData object.
        """
        from cgmath.geometry import BSplineData

        fn       = type(self)._raw_get_fn_set(self)
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
                    f"{_name_of(self)}: this cgmath's BSplineData takes no knots, so the "
                    f"curve's uneven knots are dropped and its data has another shape"
                )
        return BSplineData(points=points, degree=degree, periodic=periodic, **custom)

    # --- files

    def save(self, path: str | os.PathLike, *, world_space: bool = True) -> str:
        """Saves this curve to a cgmath file: its BSplineData (`serialize`), the
        inverse of `load`.

        Args:
            path: A str or os.PathLike (``~`` expanded). The extension sets the
                format: ``.npz`` (recommended), ``.json`` or ``.pkl`` (never
                load a ``.pkl`` you did not write).
            world_space: True: world-space CVs; False: object space. A
                BSplineData has no name and no matrix: save in world space to
                keep where the curve is. The update in place,
                ``NurbsCurve('x').load(path, world_space=...)``, takes the same
                one (it has no default).

        Returns:
            `path`, as written (``~`` expanded).

        A read of the scene: no undo step, the selection unchanged. The path is
        checked before anything is serialized (an extension cgmath does not
        save: ValueError; a missing folder: FileNotFoundError).
        """
        path = _files.check_save_path(path, "NurbsCurve.save")
        self.serialize(world_space=world_space).save(path)
        return path

    @_statement
    def _load_new(cls, curve, *, name=None, container=None, **kwargs):
        _files.creating_takes_no_space(cls.__name__, "curve", kwargs)
        from cgmath.geometry import BSplineData

        data = _files.read(curve, (BSplineData,), "NurbsCurve.load", "the curve")
        with _undo_chunk("rig.NurbsCurve.load"):
            return cls.create(data, name=name, container=container, **kwargs)

    @_statement
    def _load_into(self, curve, *args, world_space=_files.REQUIRED, **kwargs):
        from cgmath.geometry import BSplineData

        method = f"NurbsCurve({_name_of(self)!r}).load"
        _files.space_by_keyword(method, args)
        _updates_only(method, kwargs, "NurbsCurve.load(path, ...) makes a new curve")
        world_space = _files.required_space(method, "BSplineData", world_space)
        data   = _files.read(curve, (BSplineData,), method, "the curve")
        points = _curve_update(self, method, curve, data)
        with _undo_chunk("rig.NurbsCurve.load"):
            _SetCVsCommand(self, points, world_space)
        return self

    load = _files.creates_or_updates(_load_new, _load_into, _CURVE_LOAD_DOC)
    del _load_new, _load_into


class NurbsSurface(Geometry):
    """
    nurbsSurface node class
    """

    NATIVE_NODE_TYPE = "nurbsSurface"
    FN_SET           = OpenMaya.MFnNurbsSurface
    POINT_COMP_TYPE  = "cv"

    # built from its data (``Node.create("nurbsSurface")`` with none raises, naming it)
    _CREATE_TAKES_INPUTS = "data"

    # the keywords ``_create`` takes; ``create``'s other keywords are the
    # surface's attributes
    _CREATE_FLAGS = frozenset({"name"})

    _DEFINE_REFUSED = (
        "a nurbsSurface is built from its data: NurbsSurface.create(data, name='x') or "
        "NurbsSurface.load(path, name='x') makes one (rc.nurbsPlane(name='x') a primitive); "
        "NurbsSurface('x') refers to one"
    )

    # --- creation

    @classmethod
    def _create(cls, data: BSplinePatchData, name: str | None = None, **kwargs) -> str:
        """[Internal] Builds the surface of `data` (a BSplinePatchData) and returns
        the transform's name. `NurbsSurface.create` wraps it in its undo chunk;
        see there."""
        return _create_surface(data, name)

    @classmethod
    def create(
        cls,
        data:      BSplinePatchData,
        *,
        name:      str  | None = None,
        container: bool | None = None,
        **kwargs,
    ) -> "NurbsSurface":
        """Creates a surface object from a BSplinePatchData.

        Args:
            data: A BSplinePatchData: the surface is built from its control
                points (``cv``: a periodic direction's first ``degree`` rows or
                columns repeated at its end, as Maya stores it), its degrees,
                its forms (periodic or open) and its knots. A BSplinePatchData
                holds no knots: they are evenly spaced (``kv_u`` / ``kv_v``,
                the parameter range ``0..spans`` in each direction), so a
                surface whose knots were not comes back another shape. Its
                ``uniform`` and ``registered`` settings have no Maya
                equivalent and are not used.
            name: The name of the surface to create, a keyword (a trailing
                ``Shape<digits>`` is dropped: ``"sShape2"`` names the transform
                ``s2`` and the shape ``sShape2``). None: ``surface<N>``.
            container: Inside ``with container()``, whether the transform and
                the shape are registered with the scope (None: yes), as for
                every typed create (see `DGNode.create`).
            kwargs: attributes of the surface, checked by type before anything
                is made and set once it exists (see `DGNode.create`).

        The whole call is ONE undo step named ``rig.NurbsSurface.create``, as
        `NurbsCurve.create` is: the surface rides a recorded ``createNode``
        transform (the shape made under it through the API, the CVs stored as
        given, in Maya's internal unit), joins ``initialShadingGroup``, never
        changes the selection and leaves no construction history. Data that is
        no surface (not a BSplinePatchData, points that are not ``(Nu, Nv, 3)``
        finite numbers, too few for a degree, a degree under 1) raises
        TypeError or ValueError before anything is made; an error after the
        transform is made deletes it again before it propagates.
        """
        with _undo_chunk("rig.NurbsSurface.create"):
            return super().create(data, name=name, container=container, **kwargs)

    def get_num_cvs(self) -> int:
        """Returns the number of cvs."""
        return type(self)._raw_get_fn_set(self).numCVsInU * type(self)._raw_get_fn_set(self).numCVsInV

    def get_num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count_u = type(self)._raw_get_fn_set(self).numCVsInU
        if type(self)._raw_get_fn_set(self).formInU == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_u -= type(self)._raw_get_fn_set(self).degreeInU
        cv_count_v = type(self)._raw_get_fn_set(self).numCVsInV
        if type(self)._raw_get_fn_set(self).formInV == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_v -= type(self)._raw_get_fn_set(self).degreeInV
        return cv_count_u * cv_count_v

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsSurface.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).cvPositions(space)

    # --- serialization

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> BSplinePatchData:
        """Serialize this nurbsSurface to BSplinePatchData, the inverse of `create`.

        A direction is periodic when the surface is in it (Maya's ``kPeriodic``
        form); a ``kClosed`` direction, whose ends only meet, is open and keeps
        every CV. BSplinePatchData holds no knots: its knots are evenly spaced,
        which is the same surface only when this one's were (to their range);
        a warning says so otherwise.

        Args:
            world_space: If True, query points in world space, otherwise object space.
            include_uvs: Unused; kept for signature parity with `Mesh.serialize`
                and `NurbsCurve.serialize`. BSplinePatchData does not store UVs.

        Returns:
            A BSplinePatchData object.
        """
        from cgmath.geometry import BSplinePatchData

        points = self.get_points(world_space=world_space)

        fn         = type(self)._raw_get_fn_set(self)
        num_cvs_u  = fn.numCVsInU
        num_cvs_v  = fn.numCVsInV
        degree_u   = fn.degreeInU
        degree_v   = fn.degreeInV
        periodic_u = fn.formInU == OpenMaya.MFnNurbsSurface.kPeriodic
        periodic_v = fn.formInV == OpenMaya.MFnNurbsSurface.kPeriodic
        uneven = [
            axis
            for axis, knots, count, degree, periodic in (
                ("U", fn.knotsInU(), num_cvs_u, degree_u, periodic_u),
                ("V", fn.knotsInV(), num_cvs_v, degree_v, periodic_v),
            )
            if not _even_knots(np.array(knots), count - degree if periodic else count, degree, periodic)
        ]
        if uneven:
            LOGGER.warning(
                f"{_name_of(self)}: this cgmath's BSplinePatchData takes no knots, so the "
                f"surface's uneven knots in {' and '.join(uneven)} are dropped and its "
                f"data has another shape"
            )

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

    # --- files

    def save(self, path: str | os.PathLike, *, world_space: bool = True) -> str:
        """Saves this surface to a cgmath file: its BSplinePatchData
        (`serialize`), the inverse of `load`.

        Args:
            path: A str or os.PathLike (``~`` expanded). The extension sets the
                format: ``.npz`` (recommended), ``.json`` or ``.pkl`` (never
                load a ``.pkl`` you did not write).
            world_space: True: world-space CVs; False: object space. A
                BSplinePatchData has no name, no matrix and no knots (uneven
                knots are dropped, with a warning: `serialize`). The update in
                place, ``NurbsSurface('x').load(path, world_space=...)``, takes
                the same one (it has no default).

        Returns:
            `path`, as written (``~`` expanded).

        A read of the scene: no undo step, the selection unchanged. The path is
        checked before anything is serialized (an extension cgmath does not
        save: ValueError; a missing folder: FileNotFoundError).
        """
        path = _files.check_save_path(path, "NurbsSurface.save")
        self.serialize(world_space=world_space).save(path)
        return path

    @_statement
    def _load_new(cls, surface, *, name=None, container=None, **kwargs):
        _files.creating_takes_no_space(cls.__name__, "surface", kwargs)
        from cgmath.geometry import BSplinePatchData

        data = _files.read(surface, (BSplinePatchData,), "NurbsSurface.load", "the surface")
        with _undo_chunk("rig.NurbsSurface.load"):
            return cls.create(data, name=name, container=container, **kwargs)

    @_statement
    def _load_into(self, surface, *args, world_space=_files.REQUIRED, **kwargs):
        from cgmath.geometry import BSplinePatchData

        method = f"NurbsSurface({_name_of(self)!r}).load"
        _files.space_by_keyword(method, args)
        _updates_only(method, kwargs, "NurbsSurface.load(path, ...) makes a new surface")
        world_space = _files.required_space(method, "BSplinePatchData", world_space)
        data   = _files.read(surface, (BSplinePatchData,), method, "the surface")
        points = _surface_update(self, method, surface, data)
        with _undo_chunk("rig.NurbsSurface.load"):
            _SetCVsCommand(self, points, world_space)
        return self

    load = _files.creates_or_updates(_load_new, _load_into, _SURFACE_LOAD_DOC)
    del _load_new, _load_into


def _updates_only(method: str, kwargs: dict, door: str) -> None:
    """A node's ``load`` updates it: a keyword of the class's creating ``load``
    (``name=``, ``container=``, an attribute) is a TypeError naming `door`."""
    if kwargs:
        raise TypeError(
            f"{method} updates it in place: it takes the file and world_space= "
            f"(got {', '.join(f'{k}=' for k in kwargs)}); {_name_of(door)}"
        )


def _no_history(node, method: str, door: str) -> None:
    """Raises ValueError when the shape `node` has history (its ``create`` input
    connected), which would compute over the CVs an update sets."""
    fn = type(node)._raw_get_fn_set(node)
    if fn.findPlug("create", False).isDestination:
        kind = OpenMaya.MFn.kNurbsCurve if isinstance(node, NurbsCurve) else OpenMaya.MFn.kNurbsSurface
        raise ValueError(
            deformed_refusal(
                method, _name_of(node), fn.object(), kind, type(node).__name__,
                "its create input is connected", "CVs", door,
                load_args="path, world_space=...",
            )
        )


def _label(src) -> str:
    """How a load's error names its input: the path, or the data object's class."""
    return repr(os.fspath(src)) if _files.is_path(src) else f"the {type(src).__name__}"


def _cv_points(cvs: np.ndarray, method: str) -> OpenMaya.MPointArray:
    cvs = np.asarray(cvs, dtype=np.float64).reshape(-1, np.asarray(cvs).shape[-1])
    if cvs.shape[1] != 3:
        raise ValueError(f"{method}: the CVs are {cvs.shape[1]}D, not 3D")
    if not np.isfinite(cvs).all():
        raise ValueError(f"{method}: the CVs must be finite (got nan or inf)")
    return OpenMaya.MPointArray(cvs.tolist())


def _curve_update(curve: NurbsCurve, method: str, src, data) -> OpenMaya.MPointArray:
    """`NurbsCurve('x').load`'s CVs, checked before anything changes: the data's
    CV layout (count, degree, form, knots) is the curve's."""
    door = f"NurbsCurve.load({_label(src)}) makes a new curve"
    _no_history(curve, method, door)
    fn       = type(curve)._raw_get_fn_set(curve)
    cvs      = np.asarray(data.cv, dtype=np.float64)
    knots    = np.asarray(data.kv, dtype=np.float64)
    now      = np.array(fn.knots())
    periodic = bool(getattr(data, "periodic", False))
    same = (
        len(cvs) == fn.numCVs
        and int(data.degree) == fn.degree
        and periodic == (fn.form == OpenMaya.MFnNurbsCurve.kPeriodic)
        and knots.shape == now.shape
        and np.allclose(knots, now, rtol=0.0, atol=1e-9)
    )
    if not same:
        raise ValueError(
            f"{method}: {_label(src)} holds another CV layout than {_name_of(curve)!r} ({len(cvs)} CVs "
            f"of degree {int(data.degree)}, {'periodic' if periodic else 'open'}, against its "
            f"{fn.numCVs} of degree {fn.degree}, {'periodic' if fn.form == OpenMaya.MFnNurbsCurve.kPeriodic else 'open'}"
            f", or other knots); a load updates the CVs of the same layout: {_name_of(door)}"
        )
    return _cv_points(cvs, method)


def _surface_update(surface: NurbsSurface, method: str, src, data) -> OpenMaya.MPointArray:
    """`NurbsSurface('x').load`'s CVs, checked before anything changes: the data's
    CV layout (counts, degrees and forms in U and V) is the surface's; the
    surface keeps its knots (the data holds none)."""
    door = f"NurbsSurface.load({_label(src)}) makes a new surface"
    _no_history(surface, method, door)
    fn  = type(surface)._raw_get_fn_set(surface)
    cvs = np.asarray(data.cv, dtype=np.float64)
    kp  = OpenMaya.MFnNurbsSurface.kPeriodic
    same = (
        cvs.ndim == 3
        and cvs.shape[:2] == (fn.numCVsInU, fn.numCVsInV)
        and (int(data.degree_u), int(data.degree_v)) == (fn.degreeInU, fn.degreeInV)
        and (bool(data.periodic_u), bool(data.periodic_v)) == (fn.formInU == kp, fn.formInV == kp)
    )
    if not same:
        raise ValueError(
            f"{method}: {_label(src)} holds another CV layout than {_name_of(surface)!r} "
            f"({' x '.join(str(n) for n in cvs.shape[:2])} CVs of degrees {int(data.degree_u)} x "
            f"{int(data.degree_v)} against its {fn.numCVsInU} x {fn.numCVsInV} of degrees "
            f"{fn.degreeInU} x {fn.degreeInV}, or other forms); a load updates the CVs of the "
            f"same layout: {_name_of(door)}"
        )
    return _cv_points(cvs, method)


class _SetCVsCommand:
    """`NurbsCurve('x').load` / `NurbsSurface('x').load`: every CV of the shape, in
    one space (one undo step: rig's plug-in command). Undo puts back the old
    CVs."""

    UNDO_EDIT = "NurbsCurve.load / NurbsSurface.load"

    def __init__(self, node, points: OpenMaya.MPointArray, world_space: bool) -> None:
        self._path   = OpenMaya.MDagPath(type(node).get_mdagpath(node))
        self._fn_cls = type(node)._raw_FN_SET
        self._points = points
        self._space  = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        self._old    = None
        _run_undoable(self)

    def _put(self, points: OpenMaya.MPointArray) -> None:
        fn = self._fn_cls(self._path)
        fn.setCVPositions(points, self._space)
        if self._fn_cls is OpenMaya.MFnNurbsCurve:
            fn.updateCurve()
        else:
            fn.updateSurface()

    def doIt(self) -> None:  # noqa: N802
        self._old = self._fn_cls(self._path).cvPositions(self._space)
        self._put(self._points)

    def undoIt(self) -> None:  # noqa: N802
        self._put(self._old)

    def redoIt(self) -> None:  # noqa: N802
        self._put(self._points)


def _surface_inputs(data) -> tuple:
    """`NurbsSurface.create`'s input, checked, as ``MFnNurbsSurface.create`` takes
    it: ``(cvs (Nu * Nv, 3), knots U, knots V, degree U, degree V, form U, form
    V)``. It makes nothing; input that is no surface raises TypeError or
    ValueError."""
    # a BSplinePatchData, duck-typed so cgmath is never imported here; read on the
    # class, so a node or a plug (whose attributes are Maya's) is never asked
    if not (hasattr(type(data), "kv_u") and hasattr(type(data), "cv")):
        raise TypeError(
            f"NurbsSurface.create() takes a BSplinePatchData, got {type(data).__name__} "
            f"(NurbsSurface.load(path) reads one from a file)"
        )
    degrees = []
    for axis in ("u", "v"):
        degree = getattr(data, f"degree_{axis}")
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError(f"NurbsSurface.create() degree_{axis} must be an int, got {degree!r}")
        if degree < 1:
            raise ValueError(f"NurbsSurface.create() degree_{axis} must be 1 or more, got {degree}")
        degrees.append(int(degree))
    points = np.asarray(data.points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"NurbsSurface.create() points must be (Nu, Nv, 3), got {points.shape}")
    for axis, count, degree, periodic in zip("UV", points.shape[:2], degrees, (data.periodic_u, data.periodic_v)):
        # a periodic direction repeats its first `degree` rows: `degree` of them make it
        least = degree if periodic else degree + 1
        if count < least:
            raise ValueError(
                f"a{' periodic' if periodic else ''} surface of degree {degree} in {axis} takes {least} "
                f"rows of points or more there, got {count}"
            )
    if not np.isfinite(points).all():
        raise ValueError("NurbsSurface.create() points must be finite (got nan or inf)")
    cvs = np.asarray(data.cv, dtype=np.float64)
    kp, ko = OpenMaya.MFnNurbsSurface.kPeriodic, OpenMaya.MFnNurbsSurface.kOpen
    return (
        cvs.reshape(-1, 3),
        np.asarray(data.kv_u, dtype=np.float64),
        np.asarray(data.kv_v, dtype=np.float64),
        degrees[0],
        degrees[1],
        kp if data.periodic_u else ko,
        kp if data.periodic_v else ko,
    )


def _create_surface(data, name: str | None) -> str:
    """`NurbsSurface._create`'s body: the ride, as `_create_curve`'s. Returns the
    transform's name.

    In this order (`NurbsSurface.create` holds the one undo chunk around it):

    1. check the input (`_surface_inputs`): nothing is made yet;
    2. ``cmds.createNode("transform", name="surface#", skipSelect=True)``, the
       recorded node the surface rides with;
    3. ``MFnNurbsSurface.create(..., parent=<transform>)``: the shape, under it;
    4. ``cmds.sets`` into ``initialShadingGroup`` and ``cmds.rename`` of the
       transform (Maya renames the shape after it).

    Any error after step 2 deletes the transform before it propagates.
    """
    cvs, knots_u, knots_v, degree_u, degree_v, form_u, form_v = _surface_inputs(data)
    name = _shape_create_name(name)

    xform  = cmds.createNode("transform", name="surface#", skipSelect=True)
    sel    = OpenMaya.MSelectionList()
    sel.add(_name_of(xform))
    parent = sel.getDependNode(0)
    handle = OpenMaya.MObjectHandle(parent)
    try:
        try:
            NurbsSurface.FN_SET().create(
                OpenMaya.MPointArray(cvs.tolist()),
                OpenMaya.MDoubleArray(knots_u.tolist()),
                OpenMaya.MDoubleArray(knots_v.tolist()),
                degree_u,
                degree_v,
                form_u,
                form_v,
                False,  # not rational
                parent,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"NurbsSurface.create(): Maya refused the surface ({cvs.shape[0]} CVs, degrees "
                f"{degree_u} x {degree_v}): {error}"
            ) from None
        _cmds_edit.sets(xform, forceElement="initialShadingGroup")
        if name:
            # the shape follows the transform (surfaceShape<N> -> <name>Shape)
            xform = _cmds_edit.rename(xform, name)
    except BaseException:
        if handle.isValid():
            _cmds_edit.delete(OpenMaya.MFnDagNode(parent).fullPathName())
        raise
    return xform


def _default_knots(count: int, degree: int, periodic: bool) -> np.ndarray:
    """Maya's default knots, in its layout, for a curve of `count` points of
    `degree`: one apart, the end ones of an open curve repeated `degree` times
    (``cmds.curve``'s knots: ``0 0 0 1 2 3 3 3`` for six points of degree 3),
    or, on a periodic curve (`count` points before the wrap), running past
    both ends (``cmds.circle``'s: ``-2 -1 0 ... 9 10`` for eight points)."""
    if periodic:
        return np.arange(1.0 - degree, count + degree)
    return np.clip(np.arange(count + degree - 1) - (degree - 1.0), 0.0, count - degree)


def _even_knots(knots: np.ndarray, count: int, degree: int, periodic: bool) -> bool:
    """True when `knots` (Maya's layout, of a direction of `count` points before
    the wrap) are evenly spaced, to their range: the knots a BSplinePatchData
    gives back, the same shape."""
    uniform = _default_knots(count, degree, periodic)
    return knots.shape == uniform.shape and _same_spacing(knots, uniform, degree)


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
    sel.add(_name_of(xform))
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
            xform = _cmds_edit.rename(xform, name)
    except BaseException:
        if handle.isValid():
            _cmds_edit.delete(OpenMaya.MFnDagNode(parent).fullPathName())
        raise
    return xform
