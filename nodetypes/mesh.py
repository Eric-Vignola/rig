"""
Mesh node class
"""

from __future__ import annotations

import contextlib
import enum
import itertools
import os
import re
from collections import namedtuple
from numbers import Number
from typing import Any, Generator, List

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig._internal.journal import (
    cmds_edit as _cmds_edit,
    mel_eval as _mel_eval,
    statement as _statement,
)
from rig._internal.types import _is_components
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files
from rig.nodetypes._base import Attribute, _name_of
from rig.nodetypes._base import _cast
from rig.nodetypes._base import _old_handle_name, _scene_predicate
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import _positional_name, DGNode
from rig.nodetypes.geometry import (
    Geometry,
    _data_ids,
    _data_placement,
    _shape_create_name,
    deformed_refusal,
    iter_component_tokens,
)
from rig.nodetypes.geometry import _discard, _put_under, _shape_name, _shape_target, _target_space
from rig.nodetypes.object_set import ObjectSet
from rig.nodetypes.shading_engine import ShadingEngine
from rig.nodetypes.plugins import _run_undoable
from scipy.spatial import cKDTree


_MESH_LOAD_DOC = """Loads a mesh from cgmath files: one name, two forms.

    ``Mesh.load(mesh, uvs=None, *, name=None, container=None, normals=True,
    colors=True, transform=None, world_space=False, hierarchy=None, **attrs)``,
    from the class, CREATES a new mesh, always (as `create` does: a second load
    of one file makes ``body1``; with ``transform=``, a new shape under that
    transform)::

        body = Mesh.load("body_mesh.npz", "body_uv.npz", name="body")
        Mesh.load("body_mesh.npz", transform="body_ctl", world_space=True)

    ``Mesh('bodyShape').load(mesh, uvs=None, *, world_space=False, normals=True,
    colors=True, hierarchy=None)``, from a mesh, UPDATES that mesh in place, the
    inverse of ``Mesh('bodyShape').save(...)``::

        Mesh("bodyShape").load("body_mesh.npz", "body_uv.npz")

    Args:
        mesh: The mesh's file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``, or an asset cgmath's MeshData
            reads, ``.obj`` / ``.fbx`` / ``.glb`` / ``.usd``), or the MeshData
            itself.
        uvs: The UVs' file, or the UVList / UVData itself. None: no UVs (a
            new mesh has none; an updated one keeps its own).
        name, container, attrs: (creating) as `create` takes them. None: the
            data's name (``cubeShape`` names the transform ``cube``), else
            ``polySurface<N>``; with ``transform=``, ``<transform>Shape``, in
            the transform's namespace.
        transform: (creating) By keyword only, as `create` takes it: the
            existing transform (or any node derived from one) the new shape
            goes under, no transform made; read before the files are.
        world_space: By keyword only, a bool (anything else is a TypeError,
            before the files are read). Updating: False (the default): the
            data's points become the mesh's own, object space (the inverse of
            ``save()``). True: the points land where the data puts them in
            the world (where it sits: under the transform its ``parent_uuid``
            names, in ``hierarchy=`` or in the scene; the inverse of
            ``save(world_space=True)`` too); the transform is not moved. Creating (as `create` takes it): False, the data's points
            become the new shape's own; True, they land where the data puts
            them in the world: under ``transform=``, or else frozen there, the
            new transform at the origin. In both forms True locks a normal
            only where Maya computes another one on the data's own points and
            where it lands too: the source's locks, every normal through a
            mirror, and the soft ones of world-space data landing anywhere but
            where it was saved, where a non-uniform scale, a shear or float
            noise moved them (`create` says how).
        normals: (both forms) a restore flag, as `create` takes it. True: the
            data's normals bring back the hard edges and locked normals.
            False: skipped, for speed: a new mesh is all smooth, an updated
            one keeps its own hard edges and locked normals.
        colors: (both forms) a restore flag, as `create` takes it. True: the
            data's colours (``colors`` / ``color_indices``, which `serialize`
            writes from the mesh's first colour set) come back as a colour
            set: ``colorSet1`` on a new mesh, the first set of an updated one
            (its representation kept; made as ``colorSet1`` when it has none).
            Data without colours: a new mesh has no colour set, an updated one
            keeps its own. False: skipped, not read.
        hierarchy: (both forms) By keyword only, as `create` takes it: the
            HierarchyData or ClipData holding the transform the data sits
            under. None: the scene's node of that UUID, else the identity.

    Returns:
        The new Mesh (creating), or this Mesh (updating).

    Every file is read, and what it holds checked, before anything is made or
    changed: a missing file is a FileNotFoundError, a file of another class (a
    UV file as the mesh, two files swapped) a TypeError naming what it holds.

    Creating is ``create`` with the data: its normals bring back the hard edges
    and locked normals, its colours a colour set. Updating needs the same
    topology (the vertex count, the face counts and every face's vertex list,
    the holes) and a mesh without history (construction history or a deformer
    would compute over its points), else a ValueError naming
    ``Mesh.load(...)`` for a new mesh; for a deformed mesh (a skin, a
    blendShape ...) it names the mesh under its deformers,
    ``Mesh('bodyShapeOrig').load(...)``, whose load the deformed mesh follows.
    It sets every point, then, from data with normals (and ``normals=True``),
    the hard edges and locked normals, then each UV set of the file by name
    (made when the mesh has none of that name; its other sets, and the current
    set, stay), then, from data with colours (and ``colors=True``), the
    colours of its first colour set, one a face corner (its other sets stay).

    Each call is ONE undo step named ``rig.Mesh.load`` (redo brings it back)
    and leaves the selection as it was.
    """

# --- the restore flags ----------------------------------------------------------------------
#
# A part of a MeshData that a create or a load brings back on the shape, at a cost of its own,
# has a *restore flag*: one keyword named after the part, taken by every door that builds or
# updates a mesh from data (``Mesh.create``, ``Mesh.load`` from the class, ``Mesh('x').load``
# from a mesh). True, the default, brings the part back; False skips it, for speed: its data
# is not read, a new mesh gets Maya's default for it and an updated mesh keeps its own. A flag
# is a bool (anything else is a TypeError before anything is read or made), and it wins over
# a mesh attribute of the same name (``create``'s other keywords are attributes; the mesh's
# internal ``normals`` / ``colors`` arrays are never set that way).
#
# ``normals`` is the first: the hard / soft edges and the locked normals (`_shading_of`,
# `_restore_shading`). ``colors`` is the second: the data's colours as a colour set, the
# first one of the mesh (`_data_colors`, `_restore_colors`, `_update_colors`). `serialize`
# takes the same names, and ``uvs``, for the parts it writes (`_serialize_flag`):
# ``serialize(colors=False)`` leaves them out, ``create(data, colors=False)`` leaves them
# unread.
_RESTORE_FLAGS = ("normals", "colors")


def _flag(method: str, name: str, value, means: str) -> bool:
    """The bool flag `name` of `method` (``"Mesh.create()"``), checked: a bool (numpy's
    too); anything else is a TypeError naming what the flag `means`."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{method}: {name}= is {means}, not {value!r}")


def _restore_flag(method: str, name: str, value) -> bool:
    """The restore flag `name` of `method` (``"Mesh.create()"``), checked: a bool."""
    return _flag(
        method, name, value,
        f"a restore flag, True (the default: the data's {name} come back) or False (skipped)",
    )


def _serialize_flag(method: str, name: str, value) -> bool:
    """The flag `name` of a serialize (`method`, ``"Mesh.serialize()"``), checked: a bool."""
    return _flag(method, name, value, f"True (the default: the mesh's {name} are serialized) or False (left out)")


MAP_DATATYPE         = "doubleArray"
MAP_DEFAULT_VALUE    = 0.0
MAP_DEFAULT_CATEGORY = "PaintableMap"  # all maps will have this category assigned


# the rig objects an OpenMaya point never takes: a Plug is a str with a
# __getitem__ / __len__, which OpenMaya.MPoint / MVector / MPointArray read as a
# sequence of plugs (an access violation in Maya 2025); a Components selection
# (``cube.f[0]``, `_is_components`) is read through its __getitem__ the same way
_RIG_OBJECTS = (Attribute, DGNode)


def _is_rig_object(x: Any) -> bool:
    return isinstance(x, _RIG_OBJECTS) or _is_components(x)

# the exact types of the usual elements of a point sequence (none is a rig object)
_POINT_TYPES = frozenset((
    list, tuple, float, int, OpenMaya.MPoint, OpenMaya.MVector, OpenMaya.MFloatPoint,
    OpenMaya.MFloatVector,
))


def _refuse_rig_points(points: Any, method: str, many: bool = False) -> None:
    """Raise TypeError if `points` (a point, or with `many` a sequence of
    points) is, or holds, a rig plug or node: OpenMaya reads a Plug as a
    sequence and can crash Maya. The message names the fix: the plug's value
    (``plug.get()``), or a node's position. The elements are read (OpenMaya
    reads them as points), and the first row's values (rows of plugs, which
    OpenMaya refuses with its own TypeError, get the hint too)."""
    bad = None
    if _is_rig_object(points):
        bad = points
    elif isinstance(points, (list, tuple)):
        for p in points:
            if type(p) in _POINT_TYPES:  # one set lookup for the usual elements
                continue
            if _is_rig_object(p):
                bad = p
                break
        if bad is None and many and points and isinstance(points[0], (list, tuple)):
            bad = next((v for v in points[0] if _is_rig_object(v)), None)
    if bad is None:
        return
    takes = "points (MPoints, or rows of 3 numbers)" if many else "a point (an MPoint, or 3 numbers)"
    if _is_components(bad):
        what = f"the components {bad!r}"
        fix  = (
            "pass a position: a vertex's from the shape's get_points() (indexed by its "
            "id), or a vertex plug's value, .vtx[i].get()"
        )
    elif isinstance(bad, Attribute):
        name = str.__str__(bad)
        fix  = f"pass its value, {type(bad).__name__}({name!r}).get()"
        what = f"the plug {name!r}"
    else:
        fix  = (
            f"pass a position: Node({_name_of(bad)!r}).get_rotate_pivot(world_space=True), in "
            f"world space (then world_space=True here too: object space is the default), or its "
            f"translate's value, .t.get(), in its parent's space"
        )
        what = f"the node {_name_of(bad)!r}"
    raise TypeError(f"Mesh.{method} takes {takes}, not {what}: {fix}")


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2


def _axis_index(axis: object) -> int:
    """The coordinate index (0, 1, 2) of a mirror axis: rig's `Axis` (values
    0 / 1 / 2) or cgmath's ``cgmath.geometry.mesh.Axis`` (values 'x' / 'y' /
    'z'), the same three members; anything else is a TypeError."""
    if isinstance(axis, Axis):
        return axis.value
    try:
        from cgmath.geometry.mesh import Axis as _CgmathAxis
    except ImportError:  # pragma: no cover -- cgmath ships with rig
        _CgmathAxis = None
    if _CgmathAxis is not None and isinstance(axis, _CgmathAxis):
        return Axis[axis.name].value
    raise TypeError(
        f"a mirror axis is rig.nodetypes.Axis or cgmath.geometry.mesh.Axis (X, Y or Z), "
        f"not {axis!r}"
    )


class Mesh(Geometry):
    """
    Mesh node class
    """

    NATIVE_NODE_TYPE = "mesh"
    FN_SET           = OpenMaya.MFnMesh
    POINT_COMP_TYPE  = "vtx"

    # built from its mesh data (``Node.create("mesh")`` with none raises, naming it)
    _CREATE_TAKES_INPUTS = "mesh_data"

    # the keywords ``_create`` takes (the restore flags among them); ``create``'s
    # other keywords are the mesh's attributes
    _CREATE_FLAGS = frozenset({"name", "uvs", "transform", "world_space", "hierarchy", *_RESTORE_FLAGS})

    _DEFINE_REFUSED = (
        "a mesh is built from its data: Mesh.create(mesh_data, name='x') or "
        "Mesh.load(path, name='x') makes one; Mesh('x') refers to one"
    )

    def get_fn_set(self) -> OpenMaya.MFnMesh:
        """A new ``MFnMesh`` of this mesh, on every access (about 3 us).

        An MFnMesh kept across an edit of the mesh made some other way (a
        ``cmds.polyUVSet`` / ``polyColorSet`` create, a UV or colour set made
        or deleted through another MFnMesh, a colour edit that adds a node to
        its history, or the undo / redo of one) can point at freed geometry and
        crash Maya when it is used again (round U,
        runs/rU/U3/clearuvs_probe.py, stale_probe.py; runs/rU2/S3/probe). Keep
        one for a single call at most.
        """
        return OpenMaya.MFnMesh(type(self).get_mdagpath(self))

    # --- creation

    @classmethod
    def _create(
        cls,
        mesh_data:   MeshData,
        name:        str               | None = None,
        uvs:         list[UVData]      | None = None,
        normals:     bool                     = True,
        colors:      bool                     = True,
        transform:   OpenMaya.MDagPath | None = None,
        world_space: bool                     = False,
        hierarchy:   Any                      = None,
        **kwargs,
    ) -> str:
        """[Internal] Builds the mesh of `mesh_data` with the UV sets of `uvs` (a
        list, or None), its shading when `normals`, its colours when `colors`,
        its points taken in `world_space` or not (placed through `hierarchy`
        when given), and returns the transform's name; with `transform` (the DAG
        path of the transform `create` checked), the shape alone under it, and
        returns the shape's full path. `Mesh.create` wraps it in its undo chunk;
        see there."""
        return _create_mesh(mesh_data, name, uvs, normals, colors, transform, world_space, hierarchy)

    @classmethod
    def create(
        cls,
        mesh_data:   MeshData,
        uvs:         UVData   | UVList  | list[UVData] | None = None,
        *,
        name:        str      | None                          = None,
        container:   bool     | None                          = None,
        normals:     bool                                     = True,
        colors:      bool                                     = True,
        transform:   str      | DAGNode | None                = None,
        world_space: bool                                     = False,
        hierarchy:   Any                                      = None,
        **kwargs,
    ) -> "Mesh":
        """Creates a mesh object from a mesh data object.

        Args:
            mesh_data: A MeshData object: its points are the new shape's own,
                and the new transform goes where the data sits (the world
                matrix of the transform its ``parent_uuid`` names, found in
                ``hierarchy=`` or in the scene; the identity for world-space
                data: `serialize` writes it), so the mesh lands where the data
                puts it in the world.
            uvs: One or more UVData objects (a UVData, a UVList or a list; what
                `serialize` returns beside the MeshData). The first one fills
                the default set (``map1``, renamed to its name when that
                differs); each other one becomes a new set of its name, in
                order.
            name: The name of the mesh to create, a keyword (a trailing
                ``Shape<digits>`` is dropped: ``"mShape2"`` names the transform
                ``m2`` and the shape ``mShape2``). None: ``mesh_data.name``,
                else ``polySurface<N>``. With
                ``transform=`` it names the shape, as given; None:
                ``<transform>Shape`` in the transform's namespace, the data's
                name unused.
            container: Inside ``with container()``, whether the transform and
                the shape are registered with the scope (None: yes), as for
                every typed create (see `DGNode.create`).
            normals: A restore flag (see below). True: the data's normals
                bring back its hard / soft edges and locked normals. False:
                they are skipped, for speed (not read: every edge smooth,
                none locked, as for data without normals).
            colors: A restore flag (see below). True: the data's colours
                (``colors`` / ``color_indices``, which `serialize` writes from
                the mesh's first colour set) become the colour set
                ``colorSet1`` of the new mesh, RGBA and unclamped, one colour a
                face corner (a colour held one a vertex goes on each corner of
                its vertex). Data without colours makes no set. False: they
                are skipped (not read: no colour set, as for data without
                colours).
            transform: By keyword only: an existing transform (or any node
                derived from one: a joint, an ikHandle, a constraint ...), by
                name or node, that the new shape goes under; no transform is
                made (see below). None, the default: a new transform.
            world_space: By keyword only, a bool (anything else is a
                TypeError). False (the default): the data's points and normals
                become the shape's own, as they are: a new transform goes where
                the data sits, and ``transform=`` does not use it. True: the
                points land where the data puts them in the world (where it
                sits): taken into the space of ``transform=``, or without it
                frozen there, and
                the normals follow (see below). Without ``transform=``,
                world-space data, which sits at identity, comes out the same
                either way.
            hierarchy: By keyword only: a HierarchyData or ClipData (at the
                frame it is on) holding the transform the data's
                ``parent_uuid`` names: where the data sits, by cgmath's rule
                (the identity for a transform it lacks). None, the default:
                the scene's node of that UUID, under which data serialized and
                created in one scene sits as its source did, and the identity
                when the scene has none (data from another scene: save it in
                world space, or give its hierarchy). An instanced scene node
                is a ValueError.
            kwargs: attributes of the mesh, checked by type before anything
                is made and set once it exists (see `DGNode.create`).

        The whole call is ONE undo step named ``rig.Mesh.create``: one
        ``cmds.undo()`` removes the transform and the shape (with the UV sets,
        the ``initialShadingGroup`` membership, the matrix and, in a scope, the
        registration), and ``cmds.redo()`` brings back the same node (its UUID,
        and any Mesh object held on it, stay valid). The mesh rides a recorded
        ``createNode`` transform: the shape is made under it through the API,
        and its UVs are written through the API before any recorded command
        touches the mesh. It never changes the selection and leaves no
        construction history.

        **With** ``transform=`` only the shape is made, and returned (by its
        path under the target as the call named it: ``Mesh('ctl')`` would
        read the transform's first mesh). It rides the same recorded
        transform, then goes under the target (as ``parent -shape -relative``,
        through the API: the selection and the hilite list are never touched;
        under an instanced target, or without rig's undo plug-in, through the
        recorded command, which clears the hilite list), the ride is deleted
        and the shape named, in the same one undo step: ``cmds.undo()``
        removes the shape (and, in a scope, its registration; the target is
        never registered) and leaves the target as it was, ``cmds.redo()``
        brings back the same shape. The target is read before anything is
        made, by the reference rule: a missing name is a NodeNotFoundError, a
        node that is no transform (a shape, a DG node) a NodeTypeError (a
        TypeError). A locked or a referenced target is taken, as Maya's
        ``parent`` takes it (a referenced one records the ``parent`` as a
        reference edit). An instanced target gets the shape under every
        instance; with ``world_space=True`` it is a ValueError, before anything
        is made. So is a singular target with ``world_space=True`` (a scale of 0
        on it or on a parent, which Maya keeps as 1e-12: its world matrix
        scales some direction by less than 1e-9, and object points would come
        out 1e12 times out); ``world_space=False`` takes the data's points
        there. A shape under a joint is reachable by its own name:
        ``Mesh('jnt')`` reads transforms only (a NodeTypeError).

        **With** ``world_space=True`` and no ``transform=`` the mesh is frozen
        at its world positions: where the data sits goes into the points, and
        the new transform stays at the origin with identity channels, where
        ``world_space=False`` gives the same world positions from the shape's
        own points under a transform placed there.

        **Locked normals**: with ``world_space=True``, under ``transform=`` or
        frozen, as in ``Mesh('x').load(path, world_space=True)``, data holds
        no lock flag, so a normal counts as locked when Maya computes it more
        than 1e-4 from the data's on the data's own points, as a plain create
        decides it, and where it lands too (what Maya computes there needs no
        lock). Only those go through the matrix (its inverse transpose) and
        are set, which locks them, and their world normals are the source's.
        Every other normal is Maya's own for the points where they land, as
        ``makeIdentity`` leaves them, and follows a deformer: a flat face's is
        the source's, and under a non-uniform scale or a shear a soft one comes
        out off the inverse transpose (at most 0.04 degrees on a 32 x 32
        sphere at a scale of (1, 2, 1), 0.5 at (1, 2, 3), 13 at a soft corner
        of a 2 x 2 x 2 cube under a shear). The hard edges come through.
        Through a mirror (a negative scale) every normal locks instead (54 of
        54 on a 2 x 2 x 2 cube): Maya's own would face the other way, and the
        world normals stay the source's. World-space data
        (``serialize(world_space=True)``) holds normals through the source's
        matrix, which Maya does not compute from its points: back on the
        source's own points only its locks lock, anywhere else the soft normals
        do too where a non-uniform scale, a shear or float noise moved them
        (988 of 994 on that sphere at (1, 2, 3)). A cluster leaves a
        locked normal as it is, a skinCluster turns it with the skin
        (``deformUserNormals``), neither recomputes it: ``normals=False`` locks
        none, and loses the hard edges. The identity is a plain create.

        A face with holes is built as its outer loop, its hole loops added
        through the API (``MFnMesh.addHoles``, before any recorded command):
        every vertex keeps its number, the face lists its outer loop, then
        each hole loop, as ``mesh_data`` does, and each UV and each colour
        stays on its vertex.

        The data's normals (``normals`` / ``normal_indices``, which
        `serialize` writes) bring back its shading, through the API before
        any recorded command: an edge is hard where the two faces' normals
        differ at one of its ends, and a normal Maya then computes away from
        the data's is set and locked (a locked normal of the source). So
        ``Mesh.create(*m.serialize())`` gives back ``m``'s hard edges and
        locked normals, its points as ``m``'s own and its transform at ``m``'s
        world matrix. Data without normals (an ``.obj``, a MeshData built by
        hand) is all smooth. The restore costs, in a plain create, over one
        without it (2.0.0a5's): within 1.5x for a mostly smooth mesh, 1.75x to
        2.6x for a faceted or heavily locked one; with ``world_space=True``
        through a matrix, whatever differs where it lands costs a second
        compare: 2.3x on a mostly smooth 20k-face mesh, 3x on 100k.
        ``normals=False`` skips it.

        The data's colours (``colors`` / ``color_indices``, which `serialize`
        writes from the mesh's first colour set) come back as the colour set
        ``colorSet1`` (Maya's name for a first set), RGBA and unclamped, one
        colour a face corner, through the API before any recorded command, and
        ``Mesh.create(*m.serialize())`` gives back ``m``'s first colour set
        corner for corner. ``colors=False`` skips it.

        ``normals`` and ``colors`` are *restore flags*: one keyword per part of
        the data a create or a load brings back at a cost of its own, the same
        on ``Mesh.create``, ``Mesh.load`` and ``Mesh('x').load``, True by
        default and a bool (anything else is a TypeError before anything is
        made).

        Invalid mesh data (normals or colours that do not fit the faces too,
        unless ``normals=False`` / ``colors=False``) raises RuntimeError before
        anything is made; an error after the transform is made deletes what
        the call made (the transform, or the shape under the target) before it
        propagates.
        """
        from cgmath.geometry import UVData

        named = _positional_name(
            cls, (mesh_data,), needs="mesh_data, ", built_from="MeshData", noun="mesh", loads=True
        )
        if named is not None:
            raise named
        normals     = _restore_flag("Mesh.create()", "normals", normals)
        colors      = _restore_flag("Mesh.create()", "colors", colors)
        world_space = _files.space_flag("Mesh.create()", world_space, _files.CREATING_SPACE)
        if transform is not None:
            transform = _shape_target("Mesh.create()", transform, world_space)
        if uvs:
            if isinstance(uvs, UVData):
                uvs = [uvs]
        else:
            uvs = None
        with _undo_chunk("rig.Mesh.create"):
            return super().create(
                mesh_data, uvs=uvs, name=name, container=container, normals=normals, colors=colors,
                transform=transform, world_space=world_space, hierarchy=hierarchy, **kwargs
            )

    # --- mesh geometry data methods

    def get_vertex_count(self) -> int:
        """Returns the number of vertices."""
        return type(self)._raw_get_fn_set(self).numVertices

    def get_weight_point_count(self) -> int:
        """Returns the number of points a deformer weights: on a mesh, its vertices."""
        return type(self).get_vertex_count(self)

    def get_face_count(self) -> int:
        """Returns the number of faces."""
        return type(self)._raw_get_fn_set(self).numPolygons

    def get_points(self, *, world_space: bool = False) -> OpenMaya.MPointArray:
        """Get the vertex points of the mesh.

        Args:
            world_space: By keyword only. False (the default): object space;
                True: world space.

        Returns:
            A MPointArray of the mesh's vertices.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).getPoints(space)

    def set_points(
        self, points: OpenMaya.MPointArray, *, world_space: bool = False
    ) -> None:
        """Set the vertex points of the mesh: one undo step.

        Args:
            points: One point per vertex: an MPointArray, a point sequence (list
                or numpy array) or a MeshData (its points). Copied: changing it
                afterwards changes nothing.
            world_space: By keyword only. False (the default): the points are
                in object space; True: in world space.

        Raises:
            ValueError: before any edit, when there is not one point per vertex.
        """
        from cgmath.geometry import MeshData

        # if given MeshData
        if isinstance(points, MeshData):
            points = points.points

        # a point sequence (list, numpy array) or an MPointArray, copied: the undo
        # queue keeps the points of this call (the API converts a list about twice
        # as fast as an ndarray)
        if isinstance(points, np.ndarray):
            numeric = points.dtype.kind in "biuf"
            points  = points.tolist()
        else:
            numeric = False
        if not numeric:  # (an array of numbers holds no plug)
            _refuse_rig_points(points, "set_points", many=True)
        points = OpenMaya.MPointArray(points)

        count  = type(self).get_vertex_count(self)
        if len(points) != count:
            raise ValueError(
                f"Mesh.set_points: {len(points)} points for the {count} vertices of {_name_of(self)}"
            )
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        _MeshSetPointsCommand(self, points, space)

    def get_closest_point(
        self,
        point:       OpenMaya.MPoint | list[float] | np.ndarray,
        *,
        world_space: bool                                       = False,
    ) -> OpenMaya.MPoint:
        """Returns the closest mesh surface point to a given point in space.

        Args:
            point: The point, in the space `world_space` names.
            world_space: By keyword only. False (the default): `point` and the
                closest point are in object space; True: in world space.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        if not isinstance(point, OpenMaya.MPoint):
            _refuse_rig_points(point, "get_closest_point")
            point = OpenMaya.MPoint(point)
        return type(self)._raw_get_fn_set(self).getClosestPoint(point, space)

    def get_vertices_above_plane(
        self,
        point:       np.ndarray,
        normal:      np.ndarray,
        offset:      float | None = None,
        *,
        world_space: bool         = False,
    ) -> list[int]:
        """Returns list of vertex numbers that are above the provided point and normal.

        Args:
            world_space: By keyword only. False (the default): the plane is in
                object space; True: in world space.
        """
        _point = point.copy()
        if offset:
            _point += offset * normal

        # Get a direction towards the plane position foreach point in the mesh.
        points   = np.array(self.get_points(world_space=world_space))
        points   = np.delete(points, 3, axis=1)
        to_plane = points - _point
        to_plane = to_plane / np.linalg.norm(to_plane)

        # numpy for dot products and filtering out indices above plane.
        dot_products = np.dot(to_plane, normal)
        return np.where(dot_products > 0)[0]

    def get_faces_above_plane(
        self,
        point:       np.ndarray,
        normal:      np.ndarray,
        offset:      float | None = None,
        *,
        world_space: bool         = False,
    ) -> list[int]:
        """Returns list of face numbers that are above the provided point and normal.

        Args:
            world_space: By keyword only. False (the default): the plane is in
                object space; True: in world space.
        """
        vert_indices = self.get_vertices_above_plane(
            point, normal, offset=offset, world_space=world_space
        )

        # Convert indices to True/False array.
        is_vert_above_plane = np.isin(np.arange(type(self).get_vertex_count(self)), vert_indices)

        # Loop over each face and check if all vertices are above the plane.
        face_indices = []
        fn = type(self)._raw_get_fn_set(self)
        for face_idx in range(fn.numPolygons):
            if all(
                (
                    is_vert_above_plane[i]
                    for i in fn.getPolygonVertices(face_idx)
                )
            ):
                face_indices.append(face_idx)

        return face_indices

    def get_uv_at_point(
        self,
        point:       OpenMaya.MPoint | list[float] | np.ndarray,
        uv_set:      str             | None,
        *,
        world_space: bool                                       = False,
    ) -> OpenMaya.MPoint:
        """Returns the UV coordinates of the mesh at a given point.

        Args:
            point: The point, in the space `world_space` names.
            uv_set: The UV set; None: the current one.
            world_space: By keyword only. False (the default): `point` is in
                object space; True: in world space.
        """
        space  = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        uv_set = uv_set or type(self).get_current_uv_set(self)
        if not isinstance(point, OpenMaya.MPoint):
            _refuse_rig_points(point, "get_uv_at_point")
            point = OpenMaya.MPoint(point)
        return type(self)._raw_get_fn_set(self).getUVAtPoint(point, space, uv_set)

    def get_vertex_normals(
        self, angle_weighted: bool = False, *, world_space: bool = False
    ) -> OpenMaya.MFloatVectorArray:
        """Returns the vertex normals of the mesh.

        Args:
            angle_weighted: If True, each face's normal weighs by its angle at
                the vertex.
            world_space: By keyword only. False (the default): object space;
                True: world space.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self)._raw_get_fn_set(self).getVertexNormals(angle_weighted, space)

    def get_face_vertex_normals(self, *, world_space: bool = False) -> list[OpenMaya.MVector]:
        """Returns the normal of every face-vertex of the mesh, face by face.

        Args:
            world_space: By keyword only. False (the default): object space;
                True: world space.

        Returns:
            A list of MVectors.
        """

        space   = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        normals = []
        mSel    = OpenMaya.MSelectionList()
        mSel.add(_name_of(self))
        dag_path  = mSel.getDagPath(0)

        mesh_iter = OpenMaya.MItMeshFaceVertex(dag_path)
        while not mesh_iter.isDone():
            normal = mesh_iter.getNormal(space)
            normals.append(normal)

            mesh_iter.next()

        return normals

    # --- materials

    def get_materials(
        self, as_pairs: bool = False
    ) -> list[DAGNode] | list[tuple[ShadingEngine, DAGNode]]:
        """Returns a list of materials associated with this geometry.

        Args:
            as_pairs: If True, returns a list of (shading_engine, material) pairs.
                Otherwise return a list of materials.
        """
        nodes = []
        shading_engines = set(
            cmds.listConnections(_name_of(self), d=True, s=False, type="shadingEngine") or []
        )
        if not shading_engines:
            return nodes

        for sg in shading_engines:
            sg = ShadingEngine._wrap(sg)
            # skip empty shading engines
            if not sg.get_members(as_components=True):
                continue

            # skip shading engines with no material
            mat = cmds.listConnections(
                f"{_name_of(sg)}.surfaceShader", source=True, destination=False, plugs=False
            )
            if not mat:
                continue
            mat = _cast(mat[0])
            if not as_pairs:
                nodes.append(mat)
            else:
                nodes.append((sg, mat))

        return nodes

    def get_shading_engines(self) -> list[ShadingEngine]:
        """Returns a list of shading engines associated with this geometry.

        Args:
            as_pairs: If True, returns a list of (shading_engine, material) pairs.
                Otherwise return a list of shading engines.
        """
        return [x for x, _ in self.get_materials(as_pairs=True)]

    def get_material_bindings(
        self, verbose: bool = False
    ) -> ObjectSet | list[tuple[DAGNode, np.ndarray]]:
        """Get the binding relationship between this geometry and the
        associated materials.

        Returns:
            If the entire geom is bind to a single material, returns the material
            node. Otherwise return a list of (material, face_indices) pairs.

            if verbose is True the output will always be (material, face_indices) pairs.
        """
        bindings          = {}
        face_count        = type(self).get_face_count(self)
        whole_mesh_mat    = None  # used to record the material assigned to the entire mesh
        assigned_face_ids = set()
        for sg, mat in self.get_materials(as_pairs=True):
            indices = set()
            for s, comp in sg.get_members(as_components=True):
                if s == self and comp:
                    comp = OpenMaya.MFnSingleIndexedComponent(comp)
                    indices.update(comp.getElements())

            if not indices or len(indices) == face_count:
                whole_mesh_mat = mat
            else:
                assigned_face_ids.update(indices)
                bindings.setdefault(mat, set())
                bindings[mat].update(indices)

        # only a single material covers the entire mesh
        if not bindings and whole_mesh_mat and not verbose:
            return whole_mesh_mat

        # use the single material to cover missing faces, if found
        if len(assigned_face_ids) != face_count and whole_mesh_mat:
            missing_face_ids = set(range(face_count)) - assigned_face_ids
            bindings.setdefault(whole_mesh_mat, set())
            bindings[whole_mesh_mat].update(missing_face_ids)

        return [(x, np.array(sorted(bindings[x]))) for x in sorted(bindings.keys())]

    # --- UV methods

    def get_uv_sets(self) -> list[str]:
        """Returns a list of UV set names."""
        return type(self)._raw_get_fn_set(self).getUVSetNames()

    def get_current_uv_set(self) -> str:
        """Returns the current uv set."""
        return type(self)._raw_get_fn_set(self).currentUVSetName()

    def add_uv_set(self, uv_set: str) -> None:
        """Adds an empty UV set: one undo step.

        Args:
            uv_set: A uv set name to add.

        Raises:
            RuntimeError: The mesh already has a UV set of that name.
        """
        if uv_set in type(self).get_uv_sets(self):
            raise RuntimeError(f"UV set {uv_set} already exists.")
        _MeshAddUVSetCommand(self, uv_set)

    def delete_uv_set(self, uv_set: str | None = None) -> None:
        """Deletes a given uv set: one undo step, whose undo brings the set back
        with its UVs (current again if it was).

        Args:
            uv_set: A uv set name to delete. If None, use current uv set.

        Raises:
            RuntimeError: The mesh has no UV set of that name, or it is the
                mesh's default (first) set, which Maya never deletes.
        """
        uv_set = uv_set or type(self).get_current_uv_set(self)
        names  = type(self).get_uv_sets(self)
        if uv_set not in names:
            raise RuntimeError(f"UV set {uv_set} does not exist.")
        if uv_set == names[0]:
            raise RuntimeError("The default uv set cannot be deleted.")
        _MeshDeleteUVSetCommand(self, uv_set)

    def delete_map(self, map_name: str | None = None) -> None:
        """Deletes a given map.

        Args:
            map_name: A map name to delete. If None, use current map.
        """
        map_name = f"{_name_of(self)}.{map_name}"
        if cmds.objExists(map_name):
            _cmds_edit.deleteAttr(map_name)

    def rename_uv_set(self, new_name: str, uv_set: str | None = None) -> None:
        """Renames the current uv set: one undo step.

        Args:
            new_name: New uv set name.
            uv_set: A uv set name to rename. If None, use current uv set.

        Raises:
            RuntimeError: The mesh has no UV set `uv_set`, or has one named
                `new_name` already.
        """
        uv_set = uv_set or type(self).get_current_uv_set(self)
        if uv_set != new_name:
            names = type(self).get_uv_sets(self)
            if uv_set not in names:
                raise RuntimeError(f"UV set {uv_set} does not exist.")
            if new_name in names:
                raise RuntimeError(f"UV set {new_name} already exists.")
            _MeshRenameUVSetCommand(self, uv_set, new_name)

    def get_uv_coords(self, uv_set: str | None = None) -> np.ndarray:
        """Returns an array of uv coordinates.

        Args:
            uv_set: A uv set name to query. If None, use current uv set.
        """
        uv_set = uv_set or type(self).get_current_uv_set(self)
        u_vals, v_vals = type(self)._raw_get_fn_set(self).getUVs(uv_set)
        uv_coords       = np.empty((len(u_vals), 2))
        uv_coords[:, 0] = np.array(u_vals)
        uv_coords[:, 1] = np.array(v_vals)
        return uv_coords

    def get_assigned_uvs(
        self, uv_set: str | None = None
    ) -> tuple[OpenMaya.MIntArray, OpenMaya.MIntArray]:
        """Returns a tuple of (uv_counts, uv_ids)

        * uv_counts: The container for the uv counts for each polygon in the mesh
        * uv_ids: The container for the uv indices mapped to each polygon-vertex

        Args:
            uv_set: A uv set name to query. If None, use current uv set.
        """
        uv_set = uv_set or type(self).get_current_uv_set(self)
        return type(self)._raw_get_fn_set(self).getAssignedUVs(uv_set)

    def set_uv_data(self, uv_data: UVData, uv_set: str | None = None) -> None:
        """Sets the data of a UV set.

        The set's UVs and their assignment are replaced by `uv_data`'s (the UV
        count may change; data without UVs empties the set), then the set is
        renamed ``uv_data.name`` when that differs from `uv_set`. One undo
        step, whose undo puts back the old name, UVs and assignment.

        On a mesh with history (construction history or a deformer) the set is
        never cleared (Maya's ``clearUVs`` there clears the current set): data
        with fewer UVs than the set has raises, and the undo of data with more
        puts back the old UVs and assignment but keeps the extra UVs in the
        set, unassigned.

        Args:
            uv_data: A UVData object.
            uv_set: A uv set name to operate on. If None, use current uv set.

        Raises:
            RuntimeError: before any edit, when `uv_set` does not exist, another
                set is named ``uv_data.name`` already, the data does not fit the
                mesh (not one UV count per face, counts not adding up to the
                indices, an index that is not one of the UVs), or it has fewer
                UVs than the set of a mesh with history.
            TypeError: ``uv_data.name`` is not a non-empty str.
        """
        uv_set = uv_set or type(self).get_current_uv_set(self)
        _check_uv_data(OpenMaya.MFnMesh(type(self).get_mdagpath(self)), uv_set, uv_data)
        _MeshSetUVDataCommand(self, uv_set, uv_data)

    # --- deformation

    def apply_skin_data(self, skin_data: SkinData):
        """Apply skin data to this mesh: to its skinCluster (with several, the
        one nearest the shape, the first ``get_deformers`` lists), or to a new
        one made from the data when it has none of its own (the skin of a
        blendShape target or of a mesh driving this one is not its own)."""
        from cgmath.geometry import SkinData

        # avoid circular import
        from rig.nodetypes import SkinCluster

        # reuse an existing skincluster, or create one directly from the data
        skin = _skincluster_of(self)
        if skin is not None:
            skin.set_weights(skin_data)
        else:
            skin = SkinCluster.create(self, skin_data)
        return skin

    def bake_deformation(self, max_influences: int | None = None):
        """Bakes the deformation to a skincluster on this mesh.

        Args:
            max_influences: The most influences a vertex keeps. None (the
                default) is no cap: every joint of the skeleton may weigh on
                a vertex. A count of at least 1 is the cap: for fidelity the
                bake runs with ``maxInfluences`` 8 (the count, when higher),
                which leaves up to one more a vertex, then
                ``set_max_influences(max_influences)``.

        Returns:
            The baked skincluster (with several, the one nearest the shape:
            the first ``get_deformers`` lists).

        Raises:
            ValueError: before anything is done, for a cap below 1.

        One undo step named ``rig.Mesh.bake_deformation``.
        """
        if max_influences is not None and max_influences < 1:
            raise ValueError(f"bake_deformation: max_influences {max_influences}: a cap is at least 1 (None is no cap)")

        with _undo_chunk("rig.Mesh.bake_deformation"):
            # set to bind pose
            _cmds_edit.select(_name_of(self), replace=True)
            _mel_eval("gotoBindPose;")

            # bake delta mush, use a high max inf count to maintain fidelity
            # (None: as many as the skeleton has joints, which is no cap)
            if max_influences is None:
                bake_cap = len(cmds.ls("root_joint", dag=True, type="joint"))
            else:
                bake_cap = max(8, max_influences)
            _cmds_edit.bakeDeformer(
                srcSkeletonName = "root_joint",
                srcMeshName     = _name_of(self),
                dstSkeletonName = "root_joint",
                dstMeshName     = _name_of(self),
                maxInfluences   = bake_cap,
            )

            # set max inf
            skin = _skincluster_of(self)
            if max_influences is not None:
                skin.set_max_influences(max_influences)

        return skin

    # --- serialization

    def serialize_uv(self, uv_set: str | None = None) -> UVData:
        """Serialize a UV set.

        Args:
            uv_set: A uv set name to query. If None, use current uv set.

        Returns:
            A UVData object.
        """
        from cgmath.geometry import UVData

        uv_set = uv_set or type(self).get_current_uv_set(self)
        points = self.get_uv_coords(uv_set)
        uv_counts, uv_ids = self.get_assigned_uvs(uv_set)
        return UVData(
            name    = uv_set,
            indices = np.asarray(uv_ids),
            counts  = np.asarray(uv_counts),
            points  = np.asarray(points),
        )

    def serialize(
        self, *, world_space: bool = False, uvs: bool = True, normals: bool = True, colors: bool = True
    ) -> MeshData | tuple[MeshData, UVList]:
        """Serialize this mesh, the inverse of `create`.

        The data is a cgmath shape: it holds the shape's name, its ``uuid`` (the
        shape node's) and the ``parent_uuid`` of the transform it sits under, no
        matrix of its own; cgmath places it by that transform.

        Args:
            world_space: By keyword only. False (the default): the points and
                normals in object space, and in ``parent_uuid`` the UUID of the
                transform they are relative to (for an instanced shape, the
                instance the points are read through): `create` places its new
                transform where that one is. True: in world space, with no
                ``parent_uuid``: the data sits at identity.
            uvs: By keyword only, a bool. True (the default): every UV set comes
                too, a UVList beside the MeshData. False: the MeshData alone (it
                holds no UVs).
            normals: By keyword only, a bool. True (the default): Maya's normals,
                one a face corner, as ``normals`` / ``normal_indices`` (`create`
                brings back the hard edges and locked normals from them). False:
                left out, which skips their read (about half of a 20k-face
                serialize).
            colors: By keyword only, a bool. True (the default): the colours of
                the mesh's first colour set as ``colors`` / ``color_indices``:
                the distinct RGBA colours, and the one each face corner names. An
                unset corner is ``(0, 0, 0, 0)``, as Maya's own fbx exporter
                writes it; an RGB or Alpha set gives its RGBA as Maya holds it. A
                mesh with no colour set, or whose first set holds no colour,
                gives None for both. False: left out.

        A flag that is not a bool is a TypeError before anything is read.

        Returns:
            A MeshData object or (MeshData, UVList).
        """
        from cgmath.geometry import MeshData, UVList

        method  = f"{type(self).__name__}.serialize()"
        uvs     = _serialize_flag(method, "uvs",     uvs)
        normals = _serialize_flag(method, "normals", normals)
        colors  = _serialize_flag(method, "colors",  colors)
        fn      = type(self)._raw_get_fn_set(self)
        points  = self.get_points(world_space=world_space)
        vert_counts, vert_ids = fn.getVertices()

        parts = {}
        if normals:
            space      = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
            normal_ids = []
            for face_id in range(fn.numPolygons):
                normal_ids.extend(fn.getFaceNormalIds(face_id))
            parts["normals"]        = np.asarray(fn.getNormals(space))
            parts["normal_indices"] = np.asarray(normal_ids)
        if colors:
            parts["colors"], parts["color_indices"] = _first_set_colors(fn)

        mesh_data = MeshData(
            indices = np.asarray(vert_ids),
            counts  = np.asarray(vert_counts),
            points  = np.asarray(points)[:, :-1],
            name    = type(self).get_short_name(self, namespace=False),
            **parts,
            **_data_ids(self, world_space),
        )

        # extract hole data from faces with interior boundaries
        holes = fn.getHoles()
        if holes:
            hole_faces, hole_verts = zip(*holes)
            mesh_data.hole_faces  = np.asarray(hole_faces)
            mesh_data.hole_counts = np.asarray([len(hv) for hv in hole_verts])
            mesh_data.hole_indices = np.concatenate(
                [np.asarray(hv) for hv in hole_verts]
            )

        if uvs:
            uv_list = UVList([self.serialize_uv(x) for x in type(self).get_uv_sets(self)])
            if mesh_data.hole_faces is not None:
                _propagate_holes_to_uvs(mesh_data, uv_list)
            return mesh_data, uv_list
        return mesh_data

    # --- files

    def save(
        self,
        path:        str  | os.PathLike,
        uvs:         str  | os.PathLike | None = None,
        *,
        world_space: bool                      = False,
        normals:     bool                      = True,
        colors:      bool                      = True,
    ) -> str | tuple[str, str]:
        """Saves this mesh to a cgmath file: its MeshData (`serialize`), and with
        `uvs` its UV sets (a UVList) to a second file. The inverse of `load`.

        Args:
            path: The mesh's file, a str or os.PathLike (``~`` expanded). The
                extension sets the format: ``.npz`` (recommended), ``.json`` or
                ``.pkl`` (never load a ``.pkl`` you did not write: a pickle
                runs code when it is read).
            uvs: The UVs' file (same rules), another path than `path`. None:
                the UVs are not saved (a MeshData holds none).
            world_space: By keyword only. False (the default): object-space
                points, with the UUID of the transform they sit under; True:
                world-space points (as `serialize`). The update in place,
                ``Mesh('x').load(path)``, takes the same one.
            normals, colors: By keyword only, bools, as `serialize` takes them:
                True (the default) writes the normals, and the first colour
                set's colours; False leaves them out of the file.

        Returns:
            `path`, or ``(path, uvs)``, each as written (``~`` expanded).

        A read of the scene: no undo step, the selection unchanged. The flags
        and both paths are checked before anything is serialized: a flag that
        is not a bool is a TypeError, an extension cgmath does not save or the
        same path twice is a ValueError, a missing folder a FileNotFoundError,
        and nothing is written. An existing file is replaced; cgmath writes a
        file whole, so a save that fails leaves the old one as it was.
        """
        normals = _serialize_flag("Mesh.save", "normals", normals)
        colors  = _serialize_flag("Mesh.save", "colors", colors)
        path    = _files.check_save_path(path, "Mesh.save")
        if uvs is not None:
            uvs = _files.check_save_path(uvs, "Mesh.save")
            if _files.same_file(path, uvs):
                raise ValueError(
                    f"Mesh.save: the mesh and its UVs are two files: give uvs= another path than {path!r}"
                )
            mesh_data, uv_list = self.serialize(world_space=world_space, normals=normals, colors=colors)
            mesh_data.save(path)
            uv_list.save(uvs)
            return path, uvs
        self.serialize(world_space=world_space, uvs=False, normals=normals, colors=colors).save(path)
        return path

    @_statement
    def _load_new(
        cls, mesh, uvs=None, *, name=None, container=None, normals=True, colors=True, transform=None,
        world_space=False, hierarchy=None, **kwargs
    ):
        normals     = _restore_flag("Mesh.load", "normals", normals)
        colors      = _restore_flag("Mesh.load", "colors", colors)
        world_space = _files.space_flag("Mesh.load", world_space, _files.CREATING_SPACE)
        if transform is not None:
            transform = _shape_target("Mesh.load", transform, world_space)  # before the files are read
        data, uv_data = _read_mesh_files("Mesh.load", mesh, uvs)
        with _undo_chunk("rig.Mesh.load"):
            return cls.create(
                data, uv_data, name=name, container=container, normals=normals, colors=colors,
                transform=transform, world_space=world_space, hierarchy=hierarchy, **kwargs
            )

    @_statement
    def _load_into(self, mesh, uvs=None, *, world_space=False, normals=True, colors=True, hierarchy=None, **kwargs):
        method = f"Mesh({_name_of(self)!r}).load"
        if kwargs:
            raise TypeError(
                f"{method} updates the mesh in place: it takes the files, world_space=, normals=, colors= and "
                f"hierarchy= (got {', '.join(f'{k}=' for k in kwargs)}); Mesh.load(path, ...) makes a new mesh"
            )
        normals     = _restore_flag(method, "normals", normals)
        colors      = _restore_flag(method, "colors", colors)
        world_space = _files.space_flag(method, world_space)
        data, uv_data = _read_mesh_files(method, mesh, uvs)
        update = _mesh_update(self, method, mesh, data, uv_data, world_space, normals, colors, hierarchy)
        with _undo_chunk("rig.Mesh.load"):
            _MeshUpdateCommand(self, *update[:3])
            names = set(type(self).get_uv_sets(self))
            for uv in update[3]:
                if uv.name not in names:
                    self.add_uv_set(uv.name)
                self.set_uv_data(uv, uv.name)
            if update[4] is not None:
                _update_colors(self, *update[4])
        return self

    load = _files.creates_or_updates(_load_new, _load_into, _MESH_LOAD_DOC)
    del _load_new, _load_into

    # --- paintable map methods

    def get_map_names(self) -> list[str]:
        """Returns the names of every map of this mesh, paintable or not."""
        return [type(x)._raw_get_name(x) for x in self.get_map_attrs()]

    def get_map_data(self, map_name: Attribute | str, sparse: bool = True) -> MapData:
        """Get the map data from a given map attr.

        Args:
            map_name: The map attr name or plug to get the data from.
            sparse: If True, only the values that differ from the map's default,
                with their vertex ids. If False, one value per vertex, as
                `get_map_values` reads them.

        Returns:
            A  MapData object
        """
        from cgmath.geometry import MapData

        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        if sparse:
            # cap filtered ids & values using vert count, since Maya doesn't
            # auto-clear stale map values; what a short map lacks is the
            # default, which a sparse read leaves out
            ids, values = map_attr.filter_array_values(
                MAP_DEFAULT_VALUE, cap_count=type(self).get_vertex_count(self)
            )
        else:
            values = self.get_map_values(map_attr)
            ids    = range(len(values))
        # remove default category as all maps has it
        categories = set(map_attr.get_categories())
        categories.remove(MAP_DEFAULT_CATEGORY)

        # arrays, as MapData declares them (cgmath 2.0.0 casts to them; 1.0.4 kept a list)
        return MapData(
            name          = type(map_attr)._raw_get_name(map_attr),
            categories    = sorted(categories),
            indices       = np.asarray(ids, dtype=np.int32),
            values        = np.asarray(values, dtype=np.float64),
            default_value = MAP_DEFAULT_VALUE,
        )

    def serialize_maps(
        self,
        match_name: list[str] | str | None       = None,
        category:   str       | list[str] | None = None,
        sparse:     bool                         = True,
    ) -> list[MapData]:
        """Serialize maps into a dict.

        Args:
            match_name: One or more map name or regex.
                If None, serialize all maps.
            category: One or more categories to filter maps by.
                If None use MAP_DEFAULT_CATEGORY
            sparse: As `get_map_data` takes it.

        Returns:
            A list of MapData objects.
        """
        from cgmath.geometry import MapData

        data_list = []
        for map_attr in self.get_map_attrs(match_name=match_name, category=category):
            data_list.append(self.get_map_data(map_attr, sparse=sparse))
        return data_list

    def add_map(
        self,
        map_name:  str,
        paintable: bool                           = False,
        category:  str         | list[str] | None = None,
        values:    list[float] | float | None     = None,
        force:     bool                           = False,
    ) -> Attribute:
        """Adds a map attribute.

        Args:
            map_name: Name of the map attribute.
            paintable: Whether to make the map paintable.
            category: One or more categories to assign to this map (the
                caller's list is left as it is).
                MAP_DEFAULT_CATEGORY is always assigned.
            values: Values to initialize the map with. Can be a list or a single value.
            force: If True, remove existing map if found.

        Returns:
            The map attribute created.
        """
        if self.has_attr(map_name):
            if force:
                self.delete_attr(map_name)
            else:
                raise RuntimeError("{} map already exists.".format(map_name))

        # ensure MAP_DEFAULT_CATEGORY is always included, in a list of our own
        # (never the caller's)
        category = category or MAP_DEFAULT_CATEGORY
        category = [category] if isinstance(category, str) else list(category)
        if MAP_DEFAULT_CATEGORY not in category:
            category.append(MAP_DEFAULT_CATEGORY)

        map_attr = self.add_attr(
            map_name,
            storable = True,
            dataType = MAP_DATATYPE,
            category = category,
        )

        self.set_map_values(map_attr, values)
        if paintable:
            self.set_attr_paintable(map_name)
        return map_attr

    def get_map_values(self, map_name: Attribute | str) -> list[float]:
        """Returns the values of a given map, one per vertex: Maya never resizes
        a map when the topology changes, so one stored at another vertex count
        reads back trimmed or padded with the map's default, and a map never set
        reads as the default."""
        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        return self._fit_map_values(map_attr.get())

    def set_map_values(
        self, map_attr: Attribute | str, values: list[float] | float | MapData | None
    ) -> None:
        """Sets the values of a given map.

        Args:
            map_name: The map attr object or name to set the value to.
            values: value array, or a single value, or None (used MAP_DEFAULT_VALUE).
        """
        from cgmath.geometry import MapData

        attr: Attribute = self.find_map_attr(map_attr, failfast=True)
        if isinstance(values, MapData):
            vals = values.to_dense_array(type(self).get_vertex_count(self))
        else:
            vals = self._fit_map_values(values)
        attr.set(vals)

    def duplicate_map(
        self,
        map_name:     Attribute | str,
        new_map_name: str,
        paintable:    bool            = False,
    ) -> Attribute:
        """Duplicates an existing map attribute.

        Args:
            map_name: A map to duplicate.
            new_map_name: New name for the duplicated map.
            paintable: Whether to make the map paintable.

        Returns:
            The new map attribute.
        """
        if not self.has_attr(map_name):
            raise RuntimeError("{} map not found.".format(map))
        if self.has_attr(new_map_name):
            raise RuntimeError("{} map already exists.".format(new_map_name))
        values = self.get_map_values(map_name)
        return self.add_map(new_map_name, paintable=paintable, values=values)

    def find_map_attr(
        self, map_name: Attribute | str, failfast: bool = False
    ) -> Attribute | None:
        """Finds a map attr by name.

        Raises:
            ValueError: If `failfast` = True and no map is found.
        """
        attr = self.find_attr(map_name)
        if (
            attr is not None
            and type(attr)._raw_get_data_type(attr) == MAP_DATATYPE
            and attr.has_category(MAP_DEFAULT_CATEGORY)
        ):
            return attr
        if failfast:
            raise ValueError(f"{map_name} is not a map attribute.")

    def get_map_attrs(
        self,
        match_name: list[str] | str | None = None,
        category:   list[str] | str | None = None,
    ) -> list[Attribute]:
        """Returns a list of map attributes.

        Args:
            match_name: One or more map name or regex.
                If None, serialize all maps.
            category: One or more category to filter maps by.
                MAP_DEFAULT_CATEGORY is always used.
        """
        names = [match_name] if isinstance(match_name, str) else match_name
        regex = re.compile(r"|".join([rf"^{_name_of(x)}$" for x in names])) if names else None

        category = category or MAP_DEFAULT_CATEGORY
        return [
            x
            for x in self.list_attr(userDefined=True)
            if type(x)._raw_get_data_type(x) == MAP_DATATYPE
            and (not category or x.has_category(category))
            and (not regex or regex.match(type(x)._raw_get_name(x)))
        ]

    def mirror_map(
        self,
        map_name:       Attribute | str,
        mirror_axis:    Axis             = Axis.X,
        mirror_inverse: bool      | None = False,
    ) -> None:
        """Mirrors this map.

        ``mirror_axis`` is rig's ``Axis`` (``rig.nodetypes.Axis.X``) or cgmath's
        (``cgmath.geometry.mesh.Axis.X``): the same three planes. Anything else
        is a TypeError before the map is read.

        TODO code is pretty hacky at the moment, needs to be refactored for speed improvements
        Replicate closer to Maya's copySkinWeights mirrormode options
        """
        axis         = _axis_index(mirror_axis)
        cur_map_vals = self.get_map_values(map_name)
        vert_points  = self.get_points(world_space=False)
        vert_count   = len(vert_points)

        np_mesh_points_array = np.empty((vert_count, 3), dtype=float)

        for i in range(vert_count):
            np_mesh_points_array[i, 0] = vert_points[i].x
            np_mesh_points_array[i, 1] = vert_points[i].y
            np_mesh_points_array[i, 2] = vert_points[i].z

        mesh_idx_and_points_to_mirror = []

        invert_axis_val = -1.0 if mirror_inverse else 1.0
        for idx, vert_pos in enumerate(np_mesh_points_array):
            mirror_value = vert_pos[axis] * invert_axis_val
            if mirror_value > 0.0:
                mirror_pos       = np.copy(vert_pos)
                mirror_pos[axis] = mirror_pos[axis] * -1
                mesh_idx_and_points_to_mirror.append([idx, mirror_pos])

        kdtree = cKDTree(np_mesh_points_array)
        # # find mirror index
        for vert_idx, mirror_vert_pos in mesh_idx_and_points_to_mirror:
            dst, idx = kdtree.query(mirror_vert_pos)
            cur_map_vals[idx] = cur_map_vals[vert_idx]

        self.set_map_values(map_name, cur_map_vals)

    def _fit_map_values(self, values: list[float] | float | None) -> list[float]:
        """Returns a value list fitted to the vert count of this mesh.

        Args:
            values: Values to fit.
                Value list: resize it to match vert count by pruning extras
                    or padding with the map's default.
                Single value: create a list of mesh vert size of that value.
                None (a map never set): the map's default, as a single value.

        Returns:
            fitted values.
        """
        values = MAP_DEFAULT_VALUE if values is None else values

        # a single value
        vert_count = type(self).get_vertex_count(self)
        if isinstance(values, Number):
            return [float(values)] * vert_count

        val_array = np.array(values, dtype=float)
        count     = val_array.size
        if count == vert_count:
            return val_array.tolist()
        elif count > vert_count:
            return val_array[:vert_count].tolist()
        else:
            val_array = np.append(
                val_array, np.repeat(MAP_DEFAULT_VALUE, vert_count - count)
            )
            return val_array.tolist()

    def enter_paint_tool(self, map_name: Attribute | str) -> None:
        """Enters Maya's Paint Attributes Tool on the given map: the map is made
        paintable and this mesh selected. There is no off: you leave the tool
        by picking another. It writes no values."""
        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        self.set_attr_paintable(map_name)
        paint_attr = f"{type(self).NATIVE_NODE_TYPE}.{_name_of(type(self).get_long_name(self))}.{type(map_attr)._raw_get_name(map_attr)}"
        _cmds_edit.select(type(map_attr)._raw_get_node(map_attr), replace=True)
        _mel_eval(
            f'artSetToolAndSelectAttr("artAttrCtx", "{paint_attr}");artAttrInitPaintableAttr;'
        )

    # --- Color Set methods

    def get_color_sets(self) -> List[ColorSet]:
        """Returns a list of the colors sets for this object"""
        return [
            ColorSet(mesh=self, name=name) for name in type(self)._raw_get_fn_set(self).getColorSetNames()
        ]

    def add_color_set(
        self, name: str, representation: ColorSet.Representation
    ) -> ColorSet:
        """Create a color set with the given name and representation (not
        clamped): one undo step. The first colour set of a mesh becomes its
        current one.

        Raises:
            ValueError: `representation` is not a ``ColorSet.Representation``
                (or its value), or the mesh already has a colour set `name`.
        """
        # cast or throw ValueError
        representation = ColorSet.Representation(representation)

        if name in type(self)._raw_get_fn_set(self).getColorSetNames():
            raise ValueError(f"{_name_of(self)} has color set {name}!")
        _MeshAddColorSetCommand(self, name, representation.value)
        return ColorSet(mesh=self, name=name)

    # --- data transfer

    def transfer_tags(
        self, tags: str | list[str], other: Mesh | str, **kwargs
    ) -> None:
        """Transfer component tags from this mesh to the other mesh: a
        component of `other` is in a transferred tag where it lands on this
        mesh, in the transfer's `mode`.

        - A vertex is in when its sample on this mesh reads one of the tag's
          vertices with a nonzero weight (see ``_resample_map``): onto a
          finer mesh that is also the vertices on the faces around them.
        - A face is in when its centre (SPATIAL: the mean of its corners in
          space; UV: the mean of its UV corners) lands on a face of the tag:
          the closest face of this mesh to it, from the sampler the
          resampler's sample data comes from (``MeshData.sample``,
          ``UVData.sample``). A flat face whose centre is off it (a concave
          face, or a holed one: its centre is in the hole) lands by the
          centre of its first triangle in Maya's triangulation instead. A
          face without UVs is never in by UV. A centre as close to several
          faces as to one (on a corner or edge they share, as each face
          centre of a mesh is on a vertex of its subdivision) lands on one of
          them, the lowest-numbered with cgmath 2.0.0.
        - An edge is in when both its end vertices are in the vertex set the
          tag's vertices transfer to by the vertex rule. So an edge the tag
          does not hold is in when it joins two tagged vertices (a rung
          between two tagged parallel loops, say).

        Onto the same topology, in the same place (SPATIAL) or with the same
        UVs that do not overlap (UV), a vertex or face tag comes back exactly,
        and an edge tag too unless it holds both ends of an edge it does not
        hold. Where UVs overlap (stacked or mirrored shells) a component lands
        on one of the faces there, as above. One undo step
        (``rig.Mesh.transfer_tags``).

        Args:
            tags: One or more component tags to transfer.
            other: The other mesh to transfer the component tag to.
            kwargs: ``mode`` (a cgmath ``ResampleMode``, SPATIAL by default; UV
                samples through the first UV sets, the only mode that reads
                UVs) and ``method`` (a
                ``SampleMethod``, BILINEAR by default).
        """
        other     = Mesh(other)
        resampler = _resampler(self, other, kwargs.get("mode"))

        with _undo_chunk("rig.Mesh.transfer_tags"):
            for t in tags if isinstance(tags, (list, tuple)) else [tags]:
                if not other.has_tag(t):
                    other.add_tag(t)

                data = self.get_tag_data(t)

                new_data = []
                if data.indices.size > 0:
                    new_data = _land_tag(resampler, self, other, data, **kwargs)

                other.set_tag_contents(t, new_data)

    def transfer_maps(self, maps: str | list[str], other: Mesh | str, **kwargs) -> None:
        """Transfer a mesh map from this mesh to the other mesh.

        Each map is made again on `other` (``add_map(force=True)``) with its
        categories: a map of that name already there is deleted first, and
        its connections with it. Each of its values is the blend of this
        map's values that its sample on this mesh reads (see
        ``_resample_map``). One undo step (``rig.Mesh.transfer_maps``).

        Args:
            maps: One or more map names to transfer.
            other: The other mesh to transfer the map to.
            kwargs: ``mode`` (a cgmath ``ResampleMode``, SPATIAL by default; UV
                samples through the first UV sets, the only mode that reads
                UVs) and ``method`` (a
                ``SampleMethod``, BILINEAR by default).
        """
        other     = Mesh(other)
        resampler = _resampler(self, other, kwargs.get("mode"))

        with _undo_chunk("rig.Mesh.transfer_maps"):
            for m in maps if isinstance(maps, (list, tuple)) else [maps]:
                data     = self.get_map_data(m)
                new_data = _resample_map(resampler, data, **kwargs)
                other.add_map(m, category=data.categories, force=True)
                other.set_map_values(m, new_data)


def _skincluster_of(mesh):
    """The skinCluster rig's skin methods act on for `mesh` (a Mesh), or None
    when it has none of its own (``_skinclusters_of``): with several, the one
    nearest the shape, the first ``get_deformers`` lists (normally the last
    one applied). ``Mesh.apply_skin_data``, ``Mesh.bake_deformation`` and
    ``SkinCluster.transfer_from_meshes`` (each source and each target) all take
    it from here."""
    skins = _skinclusters_of(mesh)
    return skins[0] if skins else None


def _skinclusters_of(node):
    """The skinClusters of `node` (a DAGNode; of a transform, its first
    non-intermediate shape, as ``get_deformers`` reads it), nearest the shape
    first: those ``get_deformers`` lists that deform this shape, whose output
    reaches it through each deformer's input geometry. The skinCluster of a
    mesh that feeds a deformer from the side is not this one's: a blendShape's
    intermediate target (which ``get_deformers`` lists) or a skinned mesh that
    drives this one through a wrap (which ``listHistory`` reaches).
    ``SkinCluster.create`` deletes these and ``SkinCluster.rebind`` rebinds
    these, and only these."""
    skins = node.get_deformers(node_type="skinCluster")
    if not skins:
        return skins
    chain = _own_history(node)
    return [s for s in skins if any(type(s)._raw_get_mobject(s) == o for o in chain)]


def _own_history(node):
    """The nodes upstream of `node` (a DAGNode; of a transform, its first
    non-intermediate shape) that make its own geometry, as MObjects, the
    shape first: walking up its input connections, at a deformer only through
    the input geometry its output geometry deforms, and stopping at another
    non-intermediate DAG node. So a deformer's other inputs (a blendShape's
    target, a wrap's driver mesh) and what makes them are not in it, though
    ``listHistory`` reaches them; the shape's ``Orig`` and the history nodes
    and deformers between them are."""
    shape = type(node)._raw_get_mobject(node)
    if shape.hasFn(OpenMaya.MFn.kTransform):
        fn    = OpenMaya.MFnDagNode(shape)
        shape = next(
            c for c in (fn.child(i) for i in range(fn.childCount()))
            if c.hasFn(OpenMaya.MFn.kShape) and not OpenMaya.MFnDagNode(c).isIntermediateObject
        )
    # walk upstream from the shape; at a deformer, only through the input
    # geometry that its output geometry deforms
    chain = []
    seen  = set()
    stack = [(shape, None)]
    while stack:
        obj, plug = stack.pop()
        deformer  = obj.hasFn(OpenMaya.MFn.kGeometryFilt)
        if deformer and (plug is None or OpenMaya.MFnAttribute(plug.attribute()).name != "outputGeometry"):
            continue
        key = (OpenMaya.MObjectHandle(obj).hashCode(), plug.logicalIndex() if deformer else -1)
        if key in seen:
            continue
        seen.add(key)
        fn = OpenMaya.MFnDependencyNode(obj)
        if deformer:
            element = fn.findPlug("input", False).elementByLogicalIndex(plug.logicalIndex())
            inputs  = [element.child(fn.attribute("inputGeometry"))]
        elif obj != shape and obj.hasFn(OpenMaya.MFn.kDagNode) and not OpenMaya.MFnDagNode(obj).isIntermediateObject:
            continue
        else:
            inputs = [p for p in fn.getConnections() if p.isDestination]
        chain.append(obj)
        for p in inputs:
            source = p.source()
            if not source.isNull:
                stack.append((source.node(), source))
    return chain


def _resampler(src, dst, mode=None):
    """A cgmath ``MeshDataResampler`` from the mesh `src` onto the mesh `dst`
    for `mode` (a ``ResampleMode``, SPATIAL by default). Only UV mode reads
    UVs, the first UV set of each mesh; a mesh without one, or whose first
    set holds no UVs, gives None there, which cgmath refuses (RuntimeError
    "UV data not supplied.")."""
    from cgmath.geometry.resample import MeshDataResampler, ResampleMode

    # world space: cgmath samples the points as they are (it never reads a MeshData's matrix); the
    # colours are not read (the resampler may read the dst's normals)
    if mode != ResampleMode.UV:
        return MeshDataResampler(
            src.serialize(world_space=True, uvs=False, colors=False),
            dst.serialize(world_space=True, uvs=False, colors=False),
        )
    (src_data, src_uvs), (dst_data, dst_uvs) = (
        src.serialize(world_space=True, colors=False), dst.serialize(world_space=True, colors=False)
    )
    src_uv, dst_uv = (uvs[0] if uvs and np.size(uvs[0].points) else None for uvs in (src_uvs, dst_uvs))
    return MeshDataResampler(src_data, dst_data, src_uv, dst_uv)


def _resample_map(resampler, data, mode=None, method="bilinear"):
    """`data`, a cgmath MapData or GeomSubsetData of the resampler's source mesh,
    sampled onto its destination mesh in `mode` (a ``ResampleMode``, SPATIAL by
    default) with `method` (a ``SampleMethod``): each value is the weighted sum
    of the source values its sample reads, and a vertex subset keeps the
    vertices any nonzero sample reaches (face and edge tags land by
    ``_land_tag``).

    cgmath's ``MeshDataResampler.resample_map`` (1.0.4, 2.0.0) hands the data
    to ``resample_skin_weights`` as a one-influence SkinData, which normalizes
    it, so every value comes back 1.0 and every subset as the whole mesh. This
    samples that SkinData's weights with the same sample data, unnormalized.
    ``get_sample_data`` refuses ``ROBUST_BILINEAR`` (ValueError), and skin
    ``options`` (max influences) are no map option (TypeError).
    """
    from cgmath.geometry import SkinData
    from cgmath.geometry.resample import ResampleMode

    sample  = resampler.get_sample_data(mode or ResampleMode.SPATIAL, method)
    weights = sample(data.to_skin_data(resampler.src_mesh).weights)
    return type(data).from_skin_data(
        SkinData(weights=weights, influences=["inf"]),
        mesh_data      = resampler.dst_mesh,
        name           = data.name,
        component_type = data.component_type,
    )


def _land_tag(resampler, src, dst, data, mode=None, method="bilinear"):
    """The contents the tag `data` (a cgmath GeomSubsetData of the mesh `src`, the
    resampler's source) has on the mesh `dst` (its destination), by where they land
    in `mode` (see ``Mesh.transfer_tags``): a GeomSubsetData, or for an
    edge tag the edge strings ``polyListComponentConversion`` gives.

    Face and edge ids are Maya's: cgmath's ``MeshData`` numbers its edges its own
    way, so the edges go through ``MFnMesh`` / ``polyListComponentConversion``.
    """
    from cgmath.geometry import GeomSubsetData
    from cgmath.geometry.resample import ResampleMode

    mode = mode or ResampleMode.SPATIAL
    if data.component_type == "f":
        # where each face centre of dst lands: the closest face of src
        if mode == ResampleMode.SPATIAL:
            src_data, dst_data, kwargs = resampler.src_mesh, resampler.dst_mesh, {"method": method}
        elif mode == ResampleMode.UV:
            src_data, dst_data, kwargs = resampler.src_uv, resampler.dst_uv, {}
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")
        if src_data is None or dst_data is None:
            raise RuntimeError("UV data not supplied.")
        counts = np.asarray(dst_data.counts)
        points = _face_points(dst, resampler.dst_mesh.counts, counts, dst_data.points, dst_data.indices)
        landed = src_data.sample(points, **kwargs).indices
        faces  = np.nonzero((counts > 0) & np.isin(landed, data.indices))[0]
        return GeomSubsetData(name=data.name, indices=faces, component_type="f")

    if data.component_type == "e":
        fn    = OpenMaya.MFnMesh(type(src).get_mdagpath(src))
        verts = np.unique([v for e in data.indices.tolist() for v in fn.getEdgeVertices(e)])
        data  = GeomSubsetData(name=data.name, indices=verts, component_type="v")
        verts = _resample_map(resampler, data, mode, method).indices
        if verts.size == 0:
            return []
        # the edges of dst both of whose ends landed
        return cmds.polyListComponentConversion(
            [f"{_name_of(dst)}.{_name_of(x)}" for x in iter_component_tokens("vtx", verts)],
            fromVertex=True, toEdge=True, internal=True,
        ) or []

    return _resample_map(resampler, data, mode, method)


def _face_points(mesh, vert_counts, counts, points, indices):
    """The point each face of `mesh` lands by: its centre, the mean of its
    corners `points[indices]` (`counts` per face, in space or in UV), or where
    the face is flat and that centre is off it (a concave or holed face) the
    centre of its first triangle in Maya's triangulation
    (``MFnMesh.getTriangleOffsets``). A face that is not flat keeps its
    centre: on a quad that is the middle of the bilinear patch cgmath samples.
    `vert_counts` are the face vertex counts of ``MFnMesh.getVertices``, which
    the triangle offsets index; a face with no corners in `counts` (no UVs)
    keeps a centre of 0."""
    counts, indices = np.asarray(counts), np.asarray(indices)
    points  = np.asarray(points, dtype=float)
    dim     = points.shape[1]
    corners = np.pad(points, ((0, 0), (0, 3 - dim)))[indices]
    face    = np.repeat(np.arange(counts.size), counts)
    centres = np.zeros((counts.size, 3))
    np.add.at(centres, face, corners)
    centres /= np.maximum(counts, 1)[:, None]

    # each triangle's corners, offsets into getVertices moved to `indices`
    tri_counts, offsets = OpenMaya.MFnMesh(type(mesh).get_mdagpath(mesh)).getTriangleOffsets()
    tri_face = np.repeat(np.arange(counts.size), np.asarray(tri_counts))
    keep     = counts[tri_face] > 0
    vcounts  = np.asarray(vert_counts)
    starts   = (np.cumsum(counts) - counts) - (np.cumsum(vcounts) - vcounts)
    offsets  = np.asarray(offsets).reshape(-1, 3)[keep] + starts[tri_face[keep]][:, None]
    tri_face = tri_face[keep]
    a, b, c  = corners[offsets].transpose(1, 0, 2)

    # is the centre on one of its face's triangles, seen along the face normal
    area   = np.cross(b - a, c - a)
    normal = np.zeros((counts.size, 3))
    np.add.at(normal, tri_face, area)
    n, p   = normal[tri_face], centres[tri_face]
    with np.errstate(divide="ignore", invalid="ignore"):
        bary = np.stack([
            np.einsum("ij,ij->i", n, np.cross(c - b, p - b)),
            np.einsum("ij,ij->i", n, np.cross(a - c, p - c)),
            np.einsum("ij,ij->i", n, np.cross(b - a, p - a)),
        ]) / np.einsum("ij,ij->i", n, area)
    centred = np.zeros(counts.size, dtype=bool)
    np.logical_or.at(centred, tri_face, np.all(bary >= -1e-6, axis=0))

    # a face is flat when its corners are within a millionth of its size of its plane
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = normal / np.linalg.norm(normal, axis=1)[:, None]
    arm    = corners - centres[face]
    height = np.zeros(counts.size)
    size   = np.zeros(counts.size)
    np.maximum.at(height, face, np.abs(np.einsum("ij,ij->i", arm, unit[face])))
    np.maximum.at(size, face, np.linalg.norm(arm, axis=1))
    centred |= ~(height <= 1e-6 * size)

    faces, first = np.unique(tri_face, return_index=True)
    moved = ~centred[faces]
    centres[faces[moved]] = ((a + b + c) / 3.0)[first[moved]]
    return centres[:, :dim]


def _face_hole_loops(face: int, face_verts: list[int], loops: list[list[int]]) -> list[list[int]]:
    """The hole loops `loops` of the holed face `face` as its vertex list `face_verts`
    lists them: in the order and from the corner the face gives them (``MFnMesh.addHoles``
    lists a face's outer loop, then each hole loop as it is added).

    Raises RuntimeError when a hole's corners are not corners of the face.
    """
    loops   = [[int(v) for v in loop] for loop in loops]
    loop_of = {v: k for k, loop in enumerate(loops) for v in loop}
    found   = {}
    for v in face_verts:
        k = loop_of.get(v)
        if k is not None:
            found.setdefault(k, []).append(v)
    if len(found) != len(loops) or any(sorted(found[k]) != sorted(loop) for k, loop in enumerate(loops)):
        raise RuntimeError(f"MeshData: a hole of face {face} has corners that are not the face's")
    return list(found.values())


def _holed_mesh(
    points: np.ndarray, counts: list[int], indices: list[int], holes: dict[int, list[list[int]]]
) -> OpenMaya.MObject:
    """A mesh data object (``MFnMeshData``) of the faces `counts` / `indices` (each holed
    face's outer loop) with the hole loops `holes` added: every vertex keeps its number and
    its point, and each holed face lists its outer loop, then each of its hole loops.

    ``MFnMesh.addHoles`` takes a hole's corners as positions and merges each one into the
    lowest-numbered vertex at that position. So the data is made with each hole corner parked
    at a spot of its own, away from all the points (it cannot merge into another vertex at its
    point), the holes are added there and the points are put back. Built as data, not on the
    shape: ``setPoints`` on a shape would leave tweaks and a cached input mesh.
    """
    corners   = sorted({v for loops in holes.values() for loop in loops for v in loop})
    size      = float(np.abs(points).max(initial=0.0))
    step      = 1.0 + size * 1e-6  # a few float32 steps apart at any size
    far       = size + 2.0 * step
    real      = OpenMaya.MPointArray(points.tolist())
    parked    = OpenMaya.MPointArray(real)
    spot      = {}
    for k, v in enumerate(corners):
        spot[v] = parked[v] = OpenMaya.MPoint(far + step * (k % 4096), far, far + step * (k // 4096))
    data = OpenMaya.MFnMeshData().create()
    fn   = OpenMaya.MFnMesh()
    fn.create(parked, counts, indices, parent=data)
    for face, loops in holes.items():
        fn.addHoles(face, [spot[v] for loop in loops for v in loop], [len(loop) for loop in loops],
                    True, step / 4.0)
    if fn.numVertices != len(real):
        raise RuntimeError(
            f"Mesh.create: adding the holes made {fn.numVertices - len(real)} vertices"
        )
    fn.setPoints(real)
    return data


def _mesh_topology(
    mesh_data: MeshData,
) -> tuple[object, object, dict[int, list[list[int]]]]:
    """Checks `mesh_data` and returns what `MFnMesh.create` builds from it:
    ``(counts, indices, holes)``, the counts and indices as lists (the API
    converts a list 2-3x faster than an ndarray).

    Without holes: the data's own counts / indices and no holes. With holes,
    every holed face is its outer loop (the corners on none of its holes) and
    ``holes`` maps each holed face to its hole loops, as `_face_hole_loops`
    orders them (`_holed_mesh` adds them back).

    Raises RuntimeError, before anything is made, when the face counts do not
    add up to the indices, an index is not a point, a hole has fewer than 3
    corners, or the hole arrays do not match each other, the faces or the points.
    """
    counts   = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    indices  = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    n_points = len(mesh_data.points)
    if int(counts.sum()) != len(indices):
        raise RuntimeError(
            f"MeshData: the face counts add up to {int(counts.sum())} face-vertices, "
            f"but there are {len(indices)} indices"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= n_points):
        raise RuntimeError(
            f"MeshData: a face index is not one of the {n_points} points "
            f"(indices range {int(indices.min())}..{int(indices.max())})"
        )

    has_holes = mesh_data.hole_faces is not None and len(mesh_data.hole_faces) > 0
    if not has_holes:
        return counts.tolist(), indices.tolist(), {}

    if mesh_data.hole_counts is None or mesh_data.hole_indices is None:
        raise RuntimeError("MeshData: hole_faces is set without hole_counts / hole_indices")
    hole_faces   = np.asarray(mesh_data.hole_faces, dtype=np.int64).ravel()
    hole_counts  = np.asarray(mesh_data.hole_counts, dtype=np.int64).ravel()
    hole_indices = np.asarray(mesh_data.hole_indices, dtype=np.int64).ravel()
    if len(hole_faces) != len(hole_counts) or int(hole_counts.sum()) != len(hole_indices):
        raise RuntimeError(
            f"MeshData: {len(hole_faces)} hole faces, {len(hole_counts)} hole counts "
            f"adding up to {int(hole_counts.sum())}, and {len(hole_indices)} hole indices "
            "do not match"
        )
    if int(hole_faces.min()) < 0 or int(hole_faces.max()) >= len(counts):
        raise RuntimeError(f"MeshData: a hole face is not one of the {len(counts)} faces")
    if int(hole_counts.min()) < 3:
        fi = int(hole_faces[int(np.argmin(hole_counts))])
        raise RuntimeError(f"MeshData: a hole of face {fi} has fewer than 3 corners")
    if len(hole_indices) and (
        int(hole_indices.min()) < 0 or int(hole_indices.max()) >= n_points
    ):
        raise RuntimeError(f"MeshData: a hole index is not one of the {n_points} points")

    hole_map    = _build_hole_map(mesh_data)
    new_counts  = []
    new_indices = []
    holes       = {}
    offset      = 0
    for fi in range(len(counts)):
        c          = int(counts[fi])
        face_verts = indices[offset : offset + c].tolist()
        offset    += c
        if fi in hole_map:
            loops      = _face_hole_loops(fi, face_verts, hole_map[fi])
            on_holes   = {v for loop in loops for v in loop}
            face_verts = [v for v in face_verts if v not in on_holes]
            holes[fi]  = loops
        new_counts.append(len(face_verts))
        new_indices.extend(face_verts)
    return new_counts, new_indices, holes


# a UV set's data in the shape `_write_uv_set` reads (a UVData has these fields)
_UVArrays = namedtuple("_UVArrays", "name points counts indices")


def _holed_uvs(
    mesh_data: MeshData, uv_data: UVData, holes: dict[int, list[list[int]]]
) -> _UVArrays:
    """`uv_data`'s assignment for the mesh `_holed_mesh` builds: each holed
    face's UVs are handed to its corners by vertex, in the order the built face
    lists them (its outer loop, then each hole loop); every other face keeps
    its own.

    Raises RuntimeError when the UV counts do not match the faces (one count
    per face, 0 or the face's vertex count on a holed face) or the UV indices.
    """
    mesh_counts = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    mesh_ids    = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    uv_counts   = np.asarray(uv_data.counts, dtype=np.int64).ravel()
    uv_ids      = np.asarray(uv_data.indices, dtype=np.int64).ravel()
    if len(uv_counts) != len(mesh_counts) or int(uv_counts.sum()) != len(uv_ids):
        raise RuntimeError(
            f"UVData {uv_data.name!r}: {len(uv_counts)} face counts adding up to "
            f"{int(uv_counts.sum())} for {len(mesh_counts)} faces and {len(uv_ids)} indices"
        )
    mesh_starts = np.concatenate(([0], np.cumsum(mesh_counts)))
    uv_starts   = np.concatenate(([0], np.cumsum(uv_counts)))
    counts, ids = [], []
    previous    = 0
    for face in sorted(holes):
        # the faces before this one keep their assignment
        counts.append(uv_counts[previous:face])
        ids.append(uv_ids[uv_starts[previous] : uv_starts[face]])
        n_uvs = int(uv_counts[face])
        if n_uvs == int(mesh_counts[face]):
            face_verts = mesh_ids[mesh_starts[face] : mesh_starts[face + 1]].tolist()
            by_vertex  = dict(zip(face_verts, uv_ids[uv_starts[face] : uv_starts[face + 1]].tolist()))
            on_holes   = {v for loop in holes[face] for v in loop}
            built      = [v for v in face_verts if v not in on_holes]
            built     += [v for loop in holes[face] for v in loop]
            ids.append(np.array([by_vertex[v] for v in built], dtype=np.int64))
        elif n_uvs:
            raise RuntimeError(
                f"UVData {uv_data.name!r}: holed face {face} has {n_uvs} UVs for "
                f"{int(mesh_counts[face])} face-vertices"
            )
        counts.append(uv_counts[face : face + 1])
        previous = face + 1
    counts.append(uv_counts[previous:])
    ids.append(uv_ids[uv_starts[previous] :])
    return _UVArrays(uv_data.name, uv_data.points, np.concatenate(counts), np.concatenate(ids))


def _write_uv_set(
    fn: OpenMaya.MFnMesh, uv_set: str, uv_data: UVData, clear: bool = True
) -> None:
    """Replaces the UVs of the set `uv_set` of the mesh `fn` with `uv_data`'s
    (``points`` as ``(n, 2)``, ``counts``, ``indices``; a UVData or a
    `_UVArrays`). ``clearUVs`` first, so a different UV count or an empty
    set is written exactly (``clear=False`` skips it for a set known to be
    empty, a new one: about 1 ms on 10k faces); nothing more when the data
    holds no UVs. Pure API (no cmds): `Mesh.create` calls it while it builds the
    shape, before any recorded command touches it."""
    if clear:
        fn.clearUVs(uv_set)
    points  = np.asarray(uv_data.points)
    indices = np.asarray(uv_data.indices)
    if len(points) or len(indices):
        fn.setUVs(points[:, 0].tolist(), points[:, 1].tolist(), uv_set)
        fn.assignUVs(np.asarray(uv_data.counts).tolist(), indices.tolist(), uv_set)


# the flat 4x4 identity: a placement that needs no xform
_IDENTITY_16 = np.eye(4).ravel()


# --- the data's shading: hard / soft edges and locked normals ------------------------------
#
# A MeshData's normals are one per face-vertex (``normals[normal_indices]``). Its shading
# comes back from them in two parts, through the API on the shape:
#
# * the edges: an edge between two faces is smooth when the two faces' normals agree at both
#   its ends (to _SPLIT_TOL), else hard (``setEdgeSmoothings``);
# * the locked normals: once the edges are set, a normal Maya computes on the data's own points
#   that still differs from the data's (by more than _LOCK_TOL) is a locked normal of the source,
#   and is set (``setFaceVertexNormals``, which locks it).
#
# The edges are paired in numpy from the faces' loops, and Maya's edge ids are looked up only
# for the hard edges, so a mostly smooth mesh costs little more than its create.
#
# Data holds no lock flag, and on points taken through a matrix (``world_space=True``,
# `_shading_of`'s `space`) no one compare tells a locked normal:
#
# * where it lands: a non-uniform scale, a shear, or float noise on a far or a dense mesh put
#   soft normals the source never locked away from the data's (988 of 994 on a 32 x 32 sphere at
#   a scale of (1, 2, 3), about 54,000 of 89,702 on a 300 x 300 one at (100, 50, 30)), and a lock
#   keeps Maya from recomputing a normal under a deformer;
# * on the data's own points: world-space data holds normals through the source's matrix, which
#   Maya does not compute there. A file of it back on its own mesh (the README's round trip)
#   finds the same 988.
#
# So a normal is locked when it differs on both. `_lock_differing` compares on the mesh itself,
# and where nothing differs it stops: a rigid matrix near the origin, world-space data back where
# it was. Otherwise it compares on a copy of the shape, as data, put back on the data's own points
# (``_Shading.own``). Only the normals both find go through the matrix and are set; every other
# one stays Maya's own, as ``makeIdentity`` leaves them. A lock Maya computes where it lands is
# not set: it needs none, its value is there. World-space data that lands elsewhere, where a
# non-uniform scale, a shear or float noise moved it, or a source on another
# ``vertexNormalMethod`` than a new shape's, differs on both, and nothing tells it.
#
# Of the ways to Maya's normals of the data's points, on 89,702 points, each on top of the 120 ms
# their ``MPointArray`` takes from a list (which is why it waits for the first compare):
#
# * a copy of the shape takes 17 ms: 2 to copy, 3 for its points, 12 for its normals. Nothing in
#   the scene, nothing on the undo queue, and it computes as the shape does (its
#   ``vertexNormalMethod``);
# * a mesh built as data takes 28 ms and computes otherwise (960 of the 32 x 32 sphere's 994, up
#   to 0.027 away);
# * the ride before its points move takes 33 ms and leaves the shape a tweak on every point and a
#   cached input mesh (as `_holed_mesh` says).
#
# A mirror is the exception: Maya's own normals of the mirrored points face the other way.
# Nothing is copied (``own`` is None), every normal differs on the mesh itself and every one is
# set, which keeps the world normals the source's.

# two normals of one vertex further apart than this split the edge between their faces
_SPLIT_TOL = 1e-5
# a normal Maya computes this far from the data's differs (through a matrix, on both sets of
# points: above), and is a locked one
_LOCK_TOL = 1e-4


def _data_normals(mesh_data: MeshData, n_rows: int) -> tuple | None:
    """The data's normals and the normal of each of its `n_rows` face-vertices:
    ``(normals (N, 3), ids (n_rows,))``, or None when the data holds no normals.
    Raises RuntimeError (nothing is made) when they do not fit the faces."""
    normals = getattr(mesh_data, "normals", None)
    ids     = getattr(mesh_data, "normal_indices", None)
    if normals is None or ids is None:
        return None
    normals = np.asarray(normals, dtype=np.float64)
    ids     = np.asarray(ids, dtype=np.int64).ravel()
    if not len(ids) and not normals.size:
        return None
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise RuntimeError(f"MeshData: the normals are {normals.shape}, not (N, 3)")
    if len(ids) != n_rows:
        raise RuntimeError(
            f"MeshData: {len(ids)} normal indices for {n_rows} face-vertices (one each)"
        )
    if len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= len(normals)):
        raise RuntimeError(
            f"MeshData: a normal index is not one of the {len(normals)} normals "
            f"(indices range {int(ids.min())}..{int(ids.max())})"
        )
    if not np.isfinite(normals).all():
        raise RuntimeError("MeshData: the normals must be finite (got nan or inf)")
    return normals, ids


def _apart(normals: np.ndarray, i: np.ndarray, j: np.ndarray, tol: float) -> np.ndarray:
    """For each pair of normal ids ``(i[k], j[k])`` of `normals`, True when the
    two normals are further apart than `tol` (the same id never is)."""
    out = i != j
    sel = np.flatnonzero(out)
    if len(sel):
        d = normals[i[sel]] - normals[j[sel]]
        out[sel] = np.einsum("ij,ij->i", d, d) > tol * tol
    return out


def _face_loops(counts: np.ndarray, indices: np.ndarray, holes: dict) -> tuple:
    """The half-edges of the faces `counts` / `indices` (the data's lists: a holed
    face lists its outer loop's corners and its holes' corners) as two row arrays
    ``(start, end)``: each half-edge runs from the face-vertex row ``start`` to the
    row ``end`` of the next corner of its loop. A holed face has a loop for its
    outer corners and one for each of its `holes` (``{face: [hole loop, ...]}``,
    as `_mesh_topology` gives them)."""
    n_rows = len(indices)
    starts = np.cumsum(counts) - counts
    end    = np.arange(1, n_rows + 1, dtype=np.int64)
    last   = starts + counts - 1
    keep   = counts > 0
    end[last[keep]] = starts[keep]
    start = np.arange(n_rows, dtype=np.int64)
    if not holes:
        return start, end
    drop, extra_a, extra_b = [], [], []
    for face, loops in holes.items():
        first  = int(starts[face])
        rows   = range(first, first + int(counts[face]))
        row_of = {int(indices[r]): r for r in rows}
        on     = {v for loop in loops for v in loop}
        outer  = [r for r in rows if int(indices[r]) not in on]
        drop.extend(rows)
        for loop_rows in [outer] + [[row_of[v] for v in loop] for loop in loops]:
            extra_a.extend(loop_rows)
            extra_b.extend(loop_rows[1:] + loop_rows[:1])
    mask = np.ones(n_rows, dtype=bool)
    mask[drop] = False
    return (
        np.concatenate((start[mask], np.asarray(extra_a, dtype=np.int64))),
        np.concatenate((end[mask], np.asarray(extra_b, dtype=np.int64))),
    )


def _hard_edges(
    counts:   np.ndarray,
    indices:  np.ndarray,
    holes:    dict,
    normals:  np.ndarray,
    ids:      np.ndarray,
    n_points: int,
) -> tuple:
    """The hard edges of the data: each edge two faces share whose normals
    (`normals[ids]`, per face-vertex) differ at one of its ends, as
    ``(pairs (H, 2), faces (H,), corners (H,), guess (H,) or None)``: its two
    vertices, a face of it with the corner the edge starts at in the face's list
    (-1 in a holed face), and the id the edge has when a mesh numbers its edges in
    the order its faces first list them (`_first_seen_edges`; None for holed data,
    whose mesh lists its faces another way), a guess `_edge_ids` checks. An edge of
    one face (a border) or of more than two (non-manifold) is left smooth, as Maya
    makes it.

    Only the edges at a split vertex (one whose face-vertices hold more than one
    normal id) are tested: a hard edge has its normals split at an end, and a
    mostly smooth mesh has few such vertices; with none, nothing is sorted."""
    none = (
        np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    if not len(indices):
        return none
    first_id = np.empty(n_points, dtype=np.int64)
    first_id[indices] = ids
    split_rows = first_id[indices] != ids
    if not split_rows.any():
        return none
    split_vertex = np.zeros(n_points, dtype=bool)
    split_vertex[indices[split_rows]] = True
    if holes:
        return _hard_edges_holed(counts, indices, holes, normals, ids, n_points, split_vertex) + (None,)
    if 8 * int(np.count_nonzero(split_vertex[indices])) < len(indices):
        # a mostly smooth mesh: only the half-edges at its few split vertices are
        # paired (no sort of every half-edge), and `_edge_ids` finds the few ids
        # face by face
        return _hard_edges_near(counts, indices, normals, ids, n_points, split_vertex) + (None,)
    # every half-edge (face-vertex row r, from its corner to the next one), in one
    # stable sort by its edge: the two half-edges of an edge two faces share sit
    # side by side, the first one the first row that lists the edge
    n_rows = len(indices)
    starts = np.cumsum(counts) - counts
    nxt    = np.arange(1, n_rows + 1, dtype=np.int64)
    keep   = counts > 0
    nxt[(starts + counts - 1)[keep]] = starts[keep]
    va, vb = indices, indices[nxt]
    n      = np.int64(n_points)
    keys   = np.minimum(va, vb) * n + np.maximum(va, vb)
    order  = np.argsort(keys, kind="stable")
    sk     = keys[order]
    same   = sk[1:] == sk[:-1]
    # the first of exactly two half-edges with one key
    pair = same.copy()
    pair[1:] &= ~same[:-1]
    pair[:-1] &= ~same[1:]
    first  = np.flatnonzero(pair)
    h1, h2 = order[first], order[first + 1]
    near   = split_vertex[va[h1]] | split_vertex[vb[h1]]
    first, h1, h2 = first[near], h1[near], h2[near]
    face_of = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    face1, face2 = face_of[h1], face_of[h2]
    shared = face1 != face2
    if not shared.all():
        first, h1, h2, face1 = first[shared], h1[shared], h2[shared], face1[shared]
    # the other face's rows of h1's two ends (its half-edge usually runs the other way)
    forward = va[h2] == va[h1]
    at_a    = np.where(forward, h2, nxt[h2])
    at_b    = np.where(forward, nxt[h2], h2)
    split   = _apart(normals, ids[h1], ids[at_a], _SPLIT_TOL)
    split  |= _apart(normals, ids[nxt[h1]], ids[at_b], _SPLIT_TOL)
    hard    = np.flatnonzero(split)
    first, h1, face1 = first[hard], h1[hard], face1[hard]
    corners = h1 - starts[face1]
    # each unique edge's first-seen rank (its id in a mesh built from these lists)
    group   = np.flatnonzero(np.concatenate(([True], ~same)))
    rank    = np.empty(len(group), dtype=np.int64)
    rank[np.argsort(order[group], kind="stable")] = np.arange(len(group), dtype=np.int64)
    guess   = rank[np.searchsorted(group, first, side="right") - 1]
    return np.stack((va[h1], vb[h1]), axis=1), face1, corners, guess


def _hard_edges_near(counts, indices, normals, ids, n_points, split_vertex) -> tuple:
    """`_hard_edges` for data with no holed face and few split vertices:
    ``(pairs, faces, corners)``, from the half-edges from and to the corners at a
    split vertex only (once each: one from a split corner's previous corner is
    listed when that is no split corner)."""
    starts = np.cumsum(counts) - counts
    at_split = split_vertex[indices]
    rows  = np.flatnonzero(at_split)
    face  = np.searchsorted(starts, rows, side="right") - 1
    first = starts[face]
    last  = first + counts[face] - 1
    nxt   = np.where(rows == last, first, rows + 1)
    prv   = np.where(rows == first, last, rows - 1)
    lone  = ~at_split[prv]
    a = np.concatenate((rows, prv[lone]))
    b = np.concatenate((nxt, rows[lone]))
    va, vb = indices[a], indices[b]
    keys   = np.minimum(va, vb) * np.int64(n_points) + np.maximum(va, vb)
    order  = np.argsort(keys, kind="stable")
    sk     = keys[order]
    same   = sk[1:] == sk[:-1]
    pair = same.copy()
    pair[1:] &= ~same[:-1]
    pair[:-1] &= ~same[1:]
    first  = np.flatnonzero(pair)
    h1, h2 = order[first], order[first + 1]
    face1  = np.searchsorted(starts, a[h1], side="right") - 1
    face2  = np.searchsorted(starts, a[h2], side="right") - 1
    shared = face1 != face2
    if not shared.all():
        h1, h2, face1 = h1[shared], h2[shared], face1[shared]
    forward = va[h2] == va[h1]
    at_a    = np.where(forward, a[h2], b[h2])
    at_b    = np.where(forward, b[h2], a[h2])
    split   = _apart(normals, ids[a[h1]], ids[at_a], _SPLIT_TOL)
    split  |= _apart(normals, ids[b[h1]], ids[at_b], _SPLIT_TOL)
    hard    = np.flatnonzero(split)
    h1, face1 = h1[hard], face1[hard]
    return np.stack((va[h1], vb[h1]), axis=1), face1, a[h1] - starts[face1]


def _hard_edges_holed(counts, indices, holes, normals, ids, n_points, split_vertex) -> tuple:
    """`_hard_edges` for data with holed faces: ``(pairs, faces, corners)``, the
    corner -1 in a holed face."""
    starts = np.cumsum(counts) - counts
    a, b = _face_loops(counts, indices, holes)
    near = np.flatnonzero(split_vertex[indices[a]] | split_vertex[indices[b]])
    a, b = a[near], b[near]
    va, vb = indices[a], indices[b]
    keys   = np.minimum(va, vb) * np.int64(n_points) + np.maximum(va, vb)
    order  = np.argsort(keys, kind="stable")
    sk     = keys[order]
    same   = sk[1:] == sk[:-1]
    pair = same.copy()
    pair[1:] &= ~same[:-1]
    pair[:-1] &= ~same[1:]
    first  = np.flatnonzero(pair)
    h1, h2 = order[first], order[first + 1]
    face1  = np.searchsorted(starts, a[h1], side="right") - 1
    face2  = np.searchsorted(starts, a[h2], side="right") - 1
    shared = face1 != face2
    if not shared.all():
        h1, h2, face1 = h1[shared], h2[shared], face1[shared]
    forward = va[h2] == va[h1]
    at_a    = np.where(forward, a[h2], b[h2])
    at_b    = np.where(forward, b[h2], a[h2])
    split   = _apart(normals, ids[a[h1]], ids[at_a], _SPLIT_TOL)
    split  |= _apart(normals, ids[b[h1]], ids[at_b], _SPLIT_TOL)
    hard    = np.flatnonzero(split)
    h1, face1 = h1[hard], face1[hard]
    corners = a[h1] - starts[face1]
    corners[np.isin(face1, list(holes))] = -1
    return np.stack((va[h1], vb[h1]), axis=1), face1, corners


def _edge_ids(fn: OpenMaya.MFnMesh, hard: tuple, shading: "_Shading | None" = None) -> list[int]:
    """Maya's edge id of each hard edge of `hard` (`_hard_edges`) of the mesh `fn`.

    First from the mesh's face lists, in numpy (`_first_seen_edges`: Maya numbers
    the edges of a mesh it builds in the order its faces first list them), each
    id then checked by its two vertices (one ``getEdgeVertices`` per hard edge,
    through ``map``); an edge that does not check (a mesh numbered another way:
    one edited since it was built) is found as before, at its corner of its face
    (``MItMeshPolygon.getEdges``, whose edge ``k`` runs from the face's corner
    ``k``), checked by its vertices, else among the edges of its first vertex. A
    faceted 20k-face mesh: about 15 ms for its 40k hard edges, where the corner
    walk took about 100 ms. `shading` (the data's, `_Shading`) gives the face
    lists when the mesh has no hole (they are the mesh's then); else they are
    read from `fn`."""
    pairs, faces, corners, guess = hard
    if not len(pairs):
        return []
    n_edges = fn.numEdges
    if guess is not None:
        derived = guess
    elif 16 * len(pairs) < n_edges:
        derived = np.full(len(pairs), -1, dtype=np.int64)  # few: found face by face below
    else:
        lists   = (shading.counts, shading.indices) if shading is not None and not shading.holes else None
        derived = _first_seen_edges(fn, pairs, lists)
    ids     = derived.tolist()
    ok      = (derived >= 0) & (derived < n_edges)
    checked = np.flatnonzero(ok)
    if len(checked):
        ends = np.fromiter(
            itertools.chain.from_iterable(map(fn.getEdgeVertices, derived[checked].tolist())),
            dtype=np.int64, count=2 * len(checked),
        ).reshape(-1, 2)
        want = pairs[checked]
        good = ((ends[:, 0] == want[:, 0]) & (ends[:, 1] == want[:, 1])) | (
            (ends[:, 0] == want[:, 1]) & (ends[:, 1] == want[:, 0])
        )
        ok[checked[~good]] = False
    missing = np.flatnonzero(~ok)
    if not len(missing):
        return ids
    poly = OpenMaya.MItMeshPolygon(fn.object())
    vert = None
    for k in missing.tolist():
        (v0, v1), face, corner = pairs[k].tolist(), int(faces[k]), int(corners[k])
        ids[k] = -1
        if corner >= 0:
            poly.setIndex(face)
            edge = poly.getEdges()[corner]
            ends = fn.getEdgeVertices(edge)
            if (ends[0] == v0 and ends[1] == v1) or (ends[0] == v1 and ends[1] == v0):
                ids[k] = edge
                continue
        if vert is None:
            vert = OpenMaya.MItMeshVertex(fn.object())
        vert.setIndex(v0)
        for edge in vert.getConnectedEdges():
            ends = fn.getEdgeVertices(edge)
            if ends[0] == v1 or ends[1] == v1:
                ids[k] = edge
                break
    return [i for i in ids if i >= 0]


def _first_seen_edges(fn: OpenMaya.MFnMesh, pairs: np.ndarray, lists: tuple | None = None) -> np.ndarray:
    """For each vertex pair of `pairs` (H, 2), the id its edge has when the mesh
    `fn`'s edges are numbered in the order its faces first list them (face by
    face, corner ``k`` to ``k + 1``), or -1 for a pair no face lists. A guess
    `_edge_ids` checks edge by edge. `lists` is the mesh's ``(counts, indices)``
    when the caller has them (read from `fn` otherwise)."""
    if lists is None:
        counts, verts = fn.getVertices()
        counts = np.fromiter(counts, dtype=np.int64, count=len(counts))
        verts  = np.fromiter(verts, dtype=np.int64, count=len(verts))
    else:
        counts, verts = (np.asarray(x, dtype=np.int64) for x in lists)
    if not len(verts):
        return np.full(len(pairs), -1, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    nxt    = np.arange(1, len(verts) + 1, dtype=np.int64)
    keep   = counts > 0
    nxt[(starts + counts - 1)[keep]] = starts[keep]
    a, b   = verts, verts[nxt]
    n      = np.int64(max(int(verts.max()), int(pairs.max())) + 1)
    keys   = np.minimum(a, b) * n + np.maximum(a, b)
    uniq, first = np.unique(keys, return_index=True)
    rank   = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(uniq), dtype=np.int64)
    want   = np.minimum(pairs[:, 0], pairs[:, 1]) * n + np.maximum(pairs[:, 0], pairs[:, 1])
    at     = np.searchsorted(uniq, want)
    at     = np.minimum(at, len(uniq) - 1)
    return np.where(uniq[at] == want, rank[at], -1)


def _normals_now(fn: OpenMaya.MFnMesh) -> np.ndarray:
    """Maya's normals of the mesh `fn`, ``(N, 3)``, object space. API 2.0's arrays
    have no buffer: each value crosses into numpy on its own (about 4 ms for the
    20k normals of a 20k-face mesh)."""
    normals = fn.getNormals(OpenMaya.MSpace.kObject)
    flat    = np.fromiter(itertools.chain.from_iterable(normals), dtype=np.float64, count=3 * len(normals))
    return flat.reshape(-1, 3)


def _normal_ids_now(fn: OpenMaya.MFnMesh) -> np.ndarray:
    """The normal of each face-vertex of the mesh `fn`, in its face-vertex order
    (about 3 ms for the 80k face-vertices of a 20k-face mesh)."""
    ids = fn.getNormalIds()[1]
    return np.fromiter(ids, dtype=np.int64, count=len(ids))


def _rows_now(fn: OpenMaya.MFnMesh, indices: np.ndarray, counts: np.ndarray) -> np.ndarray | None:
    """For each face-vertex of the mesh `fn` (its order), the row of the data's lists
    `counts` / `indices` with its face and vertex; None when the two list them in
    one order. Only a holed face can list them in another."""
    counts_now, ids_now = fn.getVertices()
    ids_now = np.fromiter(ids_now, dtype=np.int64, count=len(ids_now))
    if np.array_equal(ids_now, indices):
        return None
    nv    = np.int64(max(int(indices.max(initial=0)), int(ids_now.max(initial=0))) + 1)
    faces = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    keys  = faces * nv + indices
    order = np.argsort(keys)
    now   = np.repeat(np.arange(len(counts_now), dtype=np.int64), list(counts_now)) * nv + ids_now
    return order[np.searchsorted(keys[order], now)]


def _face_vertices(fn: OpenMaya.MFnMesh, rows: np.ndarray, lists: tuple | None = None) -> tuple:
    """The face and the vertex of each face-vertex row `rows` of the mesh `fn`;
    `lists`, the mesh's ``(counts, indices)`` when the caller has them."""
    if lists is not None:
        counts, ids = (np.asarray(x, dtype=np.int64) for x in lists)
        return np.repeat(np.arange(len(counts), dtype=np.int64), counts)[rows], ids[rows]
    counts, ids = fn.getVertices()
    faces = np.repeat(np.arange(len(counts), dtype=np.int64), list(counts))[rows]
    verts = np.fromiter(ids, dtype=np.int64, count=len(ids))[rows]
    return faces, verts


def _set_normals(fn: OpenMaya.MFnMesh, faces, vertices, normals, hard) -> None:
    """Sets (and locks) the normal of each face-vertex ``(faces[i], vertices[i])``,
    object space. ``setFaceVertexNormals`` hardens the edges at the vertices it
    sets: those not in `hard` (edge ids) are made smooth again, which keeps the
    locked normals."""
    if not len(faces):
        return
    fn.setFaceVertexNormals(
        OpenMaya.MVectorArray(np.asarray(normals, dtype=np.float64).tolist()),
        np.asarray(faces).tolist(),
        np.asarray(vertices).tolist(),
        OpenMaya.MSpace.kObject,
    )
    fn    = OpenMaya.MFnMesh(fn.object())  # a new one after the edit (see _MeshUpdateCommand)
    # every edge's smoothing again, in one call (about 1 ms for 40k edges, where a
    # walk over the edges of the set vertices cost about 2.5 us a vertex): the
    # edges `hard` hard, every other one smooth
    flags = np.ones(fn.numEdges, dtype=bool)
    if len(hard):
        flags[np.asarray(hard, dtype=np.int64)] = False
    fn.setEdgeSmoothings(list(range(fn.numEdges)), flags.tolist())
    fn.cleanupEdgeSmoothing()
    fn.updateSurface()


def _differing(fn: OpenMaya.MFnMesh, shading: _Shading, normals: np.ndarray) -> tuple:
    """The face-vertices of the mesh `fn` whose normal Maya computes away from the
    data's `normals` (one for each of `shading`'s normal ids), as ``(rows, ids)``:
    their rows in `fn`'s face-vertex order and the data's normal of each, both empty
    when there are none.

    First normal by normal: Maya numbers a mesh's normals by its topology and its
    smooth fans, so on data a Maya mesh was serialized to (the same faces, the
    same hard edges) its normals are the data's, one for one, unless the data
    locks some. When all of them match, nothing is locked and the face-vertices'
    normal ids are never read (they cost as much again). Data numbered another
    way can only match normal for normal through normals of equal value, which
    leaves each face-vertex its own (a match through two locked normals that
    each hold the other's computed value, in data numbered another way, is the
    one case it cannot tell). Otherwise each Maya normal is compared with the
    data's normal of its face-vertices when they all have one, and each
    face-vertex on its own when they have several."""
    now = _normals_now(fn)
    if len(now) == len(normals):
        d = now - normals
        if not (np.einsum("ij,ij->i", d, d) > _LOCK_TOL * _LOCK_TOL).any():
            none = np.empty(0, dtype=np.int64)
            return none, none
    now_ids = _normal_ids_now(fn)
    rows    = _rows_now(fn, shading.indices, shading.counts) if shading.holes else None
    want    = shading.ids if rows is None else shading.ids[rows]
    link    = np.full(len(now), -1, dtype=np.int64)
    link[now_ids] = want
    one     = link[now_ids] == want
    far     = np.zeros(len(now), dtype=bool)
    known   = np.flatnonzero(link >= 0)
    if len(known):
        d          = now[known] - normals[link[known]]
        far[known] = np.einsum("ij,ij->i", d, d) > _LOCK_TOL * _LOCK_TOL
    differs = far[now_ids] & one
    if not one.all():
        other          = np.flatnonzero(~one)
        d              = now[now_ids[other]] - normals[want[other]]
        differs[other] = np.einsum("ij,ij->i", d, d) > _LOCK_TOL * _LOCK_TOL
    diff = np.flatnonzero(differs)
    return diff, want[diff]


def _lock_differing(fn: OpenMaya.MFnMesh, shading: _Shading, hard: list[int]) -> tuple:
    """Sets the data's locked normals on the mesh `fn`, its edges `hard` (ids) being
    its hard ones and no normal locked. Returns them as ``(faces, vertices)``.

    They are the face-vertices Maya computes away from the data's normals (`_differing`)
    on `fn`'s points, and with ``shading.own`` on the data's own points too: a copy of
    `fn` put back on them, only once `fn` has found something (the section's comment).
    Each takes its normal of ``shading.normals``."""
    rows, ids = _differing(fn, shading, shading.normals)
    if len(rows) and shading.own is not None:
        points, normals = shading.own
        data = OpenMaya.MFnMeshData().create()
        OpenMaya.MFnMesh().copy(fn.object(), data)
        copy = OpenMaya.MFnMesh(data)
        copy.setPoints(OpenMaya.MPointArray(points.tolist()))
        both      = np.isin(rows, _differing(copy, shading, normals)[0])
        rows, ids = rows[both], ids[both]
    if not len(rows):
        return (), ()
    faces, verts = _face_vertices(fn, rows, None if shading.holes else (shading.counts, shading.indices))
    _set_normals(fn, faces, verts, shading.normals[ids], hard)
    return faces, verts


# the data's shading, checked: its lists, its normals (where they land) and the normal of each
# face-vertex, its hard edges as vertex pairs, and `own`: the data's own points and normals, when
# what it locks is decided on a copy of the mesh at those points too; else None
_Shading = namedtuple("_Shading", "counts indices holes normals ids hard own")


def _shading_of(mesh_data: MeshData, holes: dict, space: np.ndarray | None = None) -> _Shading | None:
    """The data's shading (`_Shading`), or None for data with no normals. Raises
    RuntimeError, before anything is made, when its normals do not fit its faces.

    With `space` (4x4, row vectors), the matrix that takes the data's points where
    they land (into another mesh's object space, or the world's), the normals are
    taken through it (as normals are: its inverse transpose, normalised) and the
    data's own points and normals stay in ``own``, unless it mirrors (the section's
    comment); a `space` with no inverse is a ValueError, before anything is made.
    The identity takes nothing anywhere: the data's shading, as with no `space`."""
    counts  = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    indices = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    found   = _data_normals(mesh_data, len(indices))
    if found is None:
        return None
    normals, ids = found
    hard = _hard_edges(counts, indices, holes, normals, ids, len(mesh_data.points))
    if space is None or np.allclose(space.ravel(), _IDENTITY_16, rtol=0.0, atol=1e-12):
        return _Shading(counts, indices, holes, normals, ids, hard, None)
    own    = (np.array(mesh_data.points, dtype=np.float64).reshape(-1, 3), normals)
    linear = space[:3, :3]
    if not np.allclose(linear, np.eye(3), rtol=0.0, atol=1e-12):
        try:
            normals = normals @ np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            raise ValueError(
                "MeshData: the matrix has no inverse, which the normals go through: normals=False "
                "leaves them out"
            ) from None
        length  = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
    mirror = np.linalg.det(linear) < 0.0
    return _Shading(counts, indices, holes, normals, ids, hard, None if mirror else own)


def _restore_shading(fn: OpenMaya.MFnMesh, shading: _Shading) -> None:
    """The data's `shading` on the new mesh `fn` (all its edges smooth, no normal
    locked): its hard edges, then its locked normals. Pure API: `Mesh.create`
    calls it while it builds the shape, before any recorded command touches it."""
    hard = _edge_ids(fn, shading.hard, shading)
    if hard:
        fn.setEdgeSmoothings(hard, [False] * len(hard))
        fn.cleanupEdgeSmoothing()
    fn.updateSurface()
    _lock_differing(fn, shading, hard)


def _create_mesh(
    mesh_data:   MeshData,
    name:        str               | None,
    uv_data:     list[UVData]      | None,
    normals:     bool                     = True,
    colors:      bool                     = True,
    target:      OpenMaya.MDagPath | None = None,
    world_space: bool                     = False,
    hierarchy:   Any                      = None,
) -> str:
    """`Mesh._create`'s body: the ride. Returns the transform's name, or with
    `target` the shape's full path.

    In this order (`Mesh.create` holds the one undo chunk around it):

    1. check the data and that the UV set names differ, split the holed faces
       into their outer loop and hole loops and hand their UVs to the faces as
       they will be built (`_mesh_topology`, `_holed_uvs`); with `world_space`,
       take the points where the data puts them in the world (`_target_space`),
       in the target's space when there is one, and the normals with them
       (`_shading_of`): nothing is made yet;
    2. ``cmds.createNode("transform", name="polySurface#", skipSelect=True)``,
       the recorded node the mesh rides with (today's default name);
    3. ``MFnMesh.create(..., parent=<transform>)``: the shape, under it; with
       holes, ``MFnMesh.copy(<data>, <transform>)`` of the mesh data
       `_holed_mesh` builds (the outer loops, then ``addHoles``);
    4. the UV sets through the API (`_write_uv_set`): the first into the
       current set (renamed to its name when that differs), each other one
       into a new set of its name; then the shading, then the data's colours
       as the colour set `_COLOR_SET` (`_restore_colors`);
    5. ``cmds.sets`` into ``initialShadingGroup``, then ``cmds.rename`` of the
       transform (Maya renames the shape after it, as it always did) and
       ``cmds.xform`` of where the data sits (`_data_placement`: the world
       matrix of the transform its ``parent_uuid`` names, in `hierarchy` or
       in the scene; skipped for the identity, which the new transform
       already has, and with `world_space`: it is in the points); with
       `target` (``transform=``) instead, the shape moved under it, the
       transform deleted and the shape named (`_put_under`).

    Any error after step 2 deletes what it made (`_discard`: the transform, or
    the shape under the target) before it propagates.

    The data's normals (when it has them, and with `normals`, the restore flag)
    give the shape its hard edges and locked normals in step 4, after the UV
    sets (`_restore_shading`); its colours (when it has them, and with `colors`)
    its first colour set, after the shading (`_restore_colors`; on a holed face
    each colour stays on its vertex, as the UVs do).
    """
    counts, indices, holes = _mesh_topology(mesh_data)
    points = np.asarray(mesh_data.points)
    space  = None
    if world_space:
        # where the data puts its points in the world, in the target's space if any; the normals follow
        placed = _data_placement("Mesh.create()", mesh_data, hierarchy)
        space  = _target_space("Mesh.create()", placed, target)
        points = np.asarray(points, dtype=np.float64) @ space[:3, :3] + space[3, :3]
    shading = _shading_of(mesh_data, holes, space) if normals else None
    # the data's corners (a holed face's hole corners too, which `indices` above leaves out)
    colored = _data_colors(mesh_data, np.asarray(mesh_data.indices).size) if colors else None
    if colored is not None and holes:
        colored = (colored[0], colored[1][_built_corners(mesh_data, holes)])
    uv_sets = list(uv_data or ())
    names   = [uv.name for uv in uv_sets]
    for i, uv_name in enumerate(names):
        if uv_name in names[:i]:  # the first renames the default set, the others are new
            raise RuntimeError(f"UV set {uv_name} already exists.")
    if holes:
        uv_sets = [_holed_uvs(mesh_data, uv, holes) for uv in uv_sets]
    if target is not None:
        name = _shape_name(target, name)
    else:
        name = _shape_create_name(name or mesh_data.name)

    xform = cmds.createNode("transform", name="polySurface#", skipSelect=True)
    sel   = OpenMaya.MSelectionList()
    sel.add(_name_of(xform))
    parent = sel.getDependNode(0)
    handle = OpenMaya.MObjectHandle(parent)
    moved  = None
    try:
        if holes:
            shape = Mesh.FN_SET().copy(_holed_mesh(points, counts, indices, holes), parent)
        else:
            shape = Mesh.FN_SET().create(
                OpenMaya.MPointArray(points.tolist()), counts, indices, parent=parent
            )
        fn = OpenMaya.MFnMesh(shape)
        for i, uv in enumerate(uv_sets):
            if i == 0:
                # the default set (new, so empty)
                current = fn.currentUVSetName()
                _write_uv_set(fn, current, uv, clear=False)
                if uv.name != current:
                    fn.renameUVSet(current, uv.name)
            else:
                fn.createUVSet(uv.name)
                _write_uv_set(fn, uv.name, uv, clear=False)
        if shading is not None:
            _restore_shading(fn, shading)
        if colored is not None:
            _restore_colors(fn, colored)
        _cmds_edit.sets(xform, forceElement="initialShadingGroup")
        if target is not None:
            moved = OpenMaya.MObjectHandle(shape)
            return _put_under(shape, parent, target, name)
        if name:
            # the shape follows the transform (polySurfaceShape<N> -> <name>Shape)
            xform = _cmds_edit.rename(xform, name)
        if not world_space:
            matrix = _data_placement("Mesh.create()", mesh_data, hierarchy).ravel()
            if not np.array_equal(matrix, _IDENTITY_16):
                # a new transform is at rest already (world-space data sits at identity)
                _cmds_edit.xform(xform, matrix=matrix)
    except BaseException:
        _discard(handle, moved)
        raise
    return xform


def _read_mesh_files(method: str, mesh, uvs) -> tuple:
    """`Mesh.load`'s inputs, read and checked (`_files.read`): ``(data, uvs)``, the
    MeshData and the UVList / UVData or None."""
    from cgmath.geometry import MeshData, UVData, UVList

    data = _files.read(
        mesh, (MeshData,), method, "the mesh", refused=(UVData,),
        hints=(((UVData, UVList), "the UVs go in uvs="),),
    )
    if uvs is None:
        return data, None
    return data, _files.read(
        uvs, (UVList, UVData), method, "uvs", hints=((MeshData, "the mesh goes first"),)
    )


def _source_label(src) -> str:
    """How a load's error names its input: the path, or the data object's class."""
    return repr(os.fspath(src)) if _files.is_path(src) else f"the {type(src).__name__}"


def _mesh_update(
    mesh:        Mesh,
    method:      str,
    src,
    data,
    uvs,
    world_space: bool,
    normals:     bool = True,
    colors:      bool = True,
    hierarchy:   Any  = None,
) -> tuple:
    """`Mesh('x').load`'s edit, checked before anything changes: ``(points, space,
    shading, uv_sets, colored)`` for `_MeshUpdateCommand`, the UV sets and
    `_update_colors` (the shading None without `normals`, the colours None without
    `colors` or for data without them: the mesh keeps its own). The data's corners
    are the mesh's, in its order (the topology check), so its colours need no
    reordering. With `world_space` the points come from where the data sits
    (`_data_placement`, through `hierarchy` when given). Raises ValueError for a
    mesh with history or data of another topology, and the errors of
    `_mesh_topology` / `_data_placement` / `_target_space` / `_shading_of` /
    `_data_colors` / `_check_uv_fit`."""
    fn = OpenMaya.MFnMesh(type(mesh).get_mdagpath(mesh))
    if _has_history(fn):
        raise ValueError(_history_refusal(method, mesh, fn))
    _, _, holes = _mesh_topology(data)
    counts_now, ids_now = fn.getVertices()
    counts = np.asarray(data.counts, dtype=np.int64).ravel()
    indices = np.asarray(data.indices, dtype=np.int64).ravel()
    holes_now = [(int(face), [int(v) for v in verts]) for face, verts in fn.getHoles()]
    holes_data = []
    if data.hole_faces is not None and len(data.hole_faces):
        offset = 0
        for face, count in zip(np.asarray(data.hole_faces).tolist(), np.asarray(data.hole_counts).tolist()):
            holes_data.append((int(face), [int(v) for v in np.asarray(data.hole_indices)[offset : offset + count]]))
            offset += count
    same = (
        len(data.points) == fn.numVertices
        and np.array_equal(counts, np.fromiter(counts_now, dtype=np.int64, count=len(counts_now)))
        and np.array_equal(indices, np.fromiter(ids_now, dtype=np.int64, count=len(ids_now)))
        and sorted((f, sorted(v)) for f, v in holes_data) == sorted((f, sorted(v)) for f, v in holes_now)
    )
    if not same:
        raise ValueError(
            f"{method}: {_source_label(src)} holds another topology than {_name_of(mesh)!r} "
            f"({len(data.points)} vertices and {len(counts)} faces against its {fn.numVertices} and "
            f"{fn.numPolygons}, or other face lists or holes); a load updates the points, the shading "
            f"and the UVs of the same topology: Mesh.load({_source_label(src)}) makes a new mesh"
        )
    points = np.asarray(data.points, dtype=np.float64).reshape(-1, 3)
    taken  = None
    if world_space:
        placed = _data_placement(method, data, hierarchy)
        matrix = np.asarray(placed, dtype=np.float64).reshape(4, 4)
        points = points @ matrix[:3, :3] + matrix[3, :3]
        taken  = _target_space(method, matrix, type(mesh).get_mdagpath(mesh))
        space  = OpenMaya.MSpace.kWorld
    else:
        space = OpenMaya.MSpace.kObject
    if not np.isfinite(points).all():
        raise ValueError(f"{method}: the points must be finite (got nan or inf)")
    shading = _shading_of(data, holes, taken) if normals else None
    colored = _data_colors(data, len(indices)) if colors else None
    uv_sets = [] if uvs is None else [uvs] if _is_uv_data(uvs) else list(uvs)
    names   = [uv.name for uv in uv_sets]
    for i, uv in enumerate(uv_sets):
        if uv.name in names[:i]:
            raise ValueError(f"{method}: two UV sets of the file are named {uv.name!r}")
        _check_uv_fit(fn, uv)
    return OpenMaya.MPointArray(points.tolist()), space, shading, uv_sets, colored


def _is_uv_data(uvs) -> bool:
    from cgmath.geometry import UVData

    return isinstance(uvs, UVData)


def _check_uv_fit(fn: OpenMaya.MFnMesh, uv_data: UVData) -> None:
    """Raises RuntimeError / TypeError, before any edit, when `uv_data` does not fit
    the faces of the mesh `fn` (what `Mesh.set_uv_data` checks of its data)."""
    name = uv_data.name
    if not isinstance(name, str) or not name:
        raise TypeError(f"UVData.name must be a non-empty str, not {name!r}")
    n_uvs   = len(np.asarray(uv_data.points, dtype=float).reshape(-1, 2))
    counts  = np.asarray(uv_data.counts, dtype=np.int64).ravel()
    indices = np.asarray(uv_data.indices, dtype=np.int64).ravel()
    if not n_uvs and not len(indices):
        return
    if len(counts) != fn.numPolygons:
        raise RuntimeError(f"UVData {name!r}: {len(counts)} face counts for {fn.numPolygons} faces")
    if int(counts.sum()) != len(indices):
        raise RuntimeError(
            f"UVData {name!r}: the face counts add up to {int(counts.sum())}, "
            f"but there are {len(indices)} indices"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= n_uvs):
        raise RuntimeError(
            f"UVData {name!r}: an index is not one of the {n_uvs} UVs "
            f"(indices range {int(indices.min())}..{int(indices.max())})"
        )


def _build_hole_map(mesh_data: MeshData) -> dict[int, list[list[int]]]:
    """Builds a mapping of face_index -> list of hole vertex index lists."""
    from cgmath.geometry import MeshData

    hole_map: dict[int, list[list[int]]] = {}
    offset = 0
    for i in range(len(mesh_data.hole_faces)):
        face_idx = int(mesh_data.hole_faces[i])
        count    = int(mesh_data.hole_counts[i])
        verts    = list(mesh_data.hole_indices[offset : offset + count])
        hole_map.setdefault(face_idx, []).append(verts)
        offset += count
    return hole_map


def _propagate_holes_to_uvs(mesh_data: MeshData, uv_list: UVList) -> None:
    """Copy hole metadata from MeshData to each UVData, remapping vertex indices.

    A mapped face lists its UVs in the order it lists its vertices, so for each
    mesh hole vertex we read the UV at the same position of the face. A UV set
    that maps only some faces has fewer UVs than the mesh has face-vertices:
    each face's UVs start at its own UV offset, and a holed face the set does
    not map has no hole UVs (the set's hole arrays leave it out; a set that
    maps no holed face keeps None).
    """
    from cgmath.geometry import MeshData, UVList

    face_starts = np.concatenate([[0], np.cumsum(mesh_data.counts[:-1])])

    for uv in uv_list:
        uv_counts = np.asarray(uv.counts, dtype=np.int64)
        uv_starts = np.cumsum(uv_counts) - uv_counts

        hole_faces      = []
        hole_counts     = []
        uv_hole_indices = []
        offset          = 0
        for i in range(len(mesh_data.hole_faces)):
            face_idx = int(mesh_data.hole_faces[i])
            count    = int(mesh_data.hole_counts[i])
            mesh_hole_verts = {
                int(v) for v in mesh_data.hole_indices[offset : offset + count]
            }
            offset += count

            face_count = int(mesh_data.counts[face_idx])
            if int(uv_counts[face_idx]) != face_count:
                continue  # not mapped: no hole UVs

            start    = int(face_starts[face_idx])
            uv_start = int(uv_starts[face_idx])
            for j in range(face_count):
                if int(mesh_data.indices[start + j]) in mesh_hole_verts:
                    uv_hole_indices.append(int(uv.indices[uv_start + j]))
            hole_faces.append(face_idx)
            hole_counts.append(count)

        if hole_faces:
            uv.hole_faces   = np.array(hole_faces, dtype=mesh_data.hole_faces.dtype)
            uv.hole_counts  = np.array(hole_counts, dtype=mesh_data.hole_counts.dtype)
            uv.hole_indices = np.array(uv_hole_indices, dtype=int)


# --- rig's undoable mesh edits ------------------------------------------------------------
#
# Each class below is one edit rig puts on Maya's undo queue through its plug-in
# command (`_run_undoable`, one undo step): the caller checks the edit first, the
# constructor keeps what the edit needs and runs it, doIt makes the change (and keeps
# what undoIt needs), undoIt puts the mesh back and redoIt makes the change again.
#
# * API only: no cmds or mel in doIt, undoIt or redoIt. redo must replay the API
#   change (a cmds call there would be recorded, and flush the redo queue), and a
#   cmds call in doIt would be an undo step of its own.
# * A new MFnMesh for every call, from a copy of the mesh's MDagPath: an MFnMesh
#   kept across an edit of the mesh's sets made some other way (a polyUVSet /
#   polyColorSet, or the undo / redo of one) can point at freed geometry and crash
#   Maya when it is used again.
# * UV and colour SETS are made, renamed and deleted here too, never with
#   polyUVSet / polyColorSet: Maya undoes those by swapping the mesh's set data,
#   which, beside rig's API data edits and a later vertex edit, brings back a
#   broken set (an API write into it crashes Maya) or wrong data on redo.
# * The user's own polyUVSet / polyColorSet can still leave such a set (Maya does
#   it without rig): every set edit first puts the mesh's sets into its cached
#   input (`_sync_sets`).
# * A set keeps its element of the mesh's uvSet[] / colorSet[] array, and the
#   element its connections (a uvLink): an undone set creation frees the element it
#   made, a delete's undo makes the sets again in their elements and reconnects
#   them (`_free_elements`, `_relink`). Maya's own set commands undo by element.
# * A mesh with history (its inMesh connected: construction history or a deformer)
#   gets a node for each API set or colour edit: the edit passes an MDGModifier,
#   and its undo / redo are the modifier's.


def _has_history(fn: OpenMaya.MFnMesh) -> bool:
    """True when the mesh's ``inMesh`` is connected (construction history, a
    deformer)."""
    return fn.findPlug("inMesh", False).isDestination


def _history_refusal(method: str, mesh: Mesh, fn: OpenMaya.MFnMesh) -> str:
    """Why `Mesh('x').load` refuses a mesh with history (`deformed_refusal`)."""
    return deformed_refusal(
        method, _name_of(mesh), fn.object(), OpenMaya.MFn.kMesh, "Mesh", "its inMesh is connected",
        "points", "Mesh.load(path) makes a new mesh",
    )


def _put_points(fn: OpenMaya.MFnMesh, points: OpenMaya.MPointArray, space: int) -> None:
    fn.setPoints(points, space)
    fn.updateSurface()


def _modified(modifier: OpenMaya.MDGModifier | None) -> dict:
    """The ``modifier`` keyword of an MFnMesh call: none without one (the API
    takes no None)."""
    return {} if modifier is None else {"modifier": modifier}


def _sync_sets(fn: OpenMaya.MFnMesh) -> bool:
    """Puts the UV and colour sets of the history-free mesh `fn` into its cached
    input mesh when the two differ; True when it did (`fn` is then stale).

    A history-free mesh keeps a copy of its input (``cachedInMesh``) from its
    first vertex tweak on, and API set edits go through that copy. The undo /
    redo of Maya's polyUVSet / polyColorSet swaps the mesh's sets but not the
    copy's: after a native set edit, a vertex edit and the undo of both, the
    mesh has a set its copy has not, and any API write into that set crashes
    Maya (Maya's own polyUVSet -delete fails there). The copy takes the mesh's
    data with its own points (the tweaks stay added to them)."""
    if fn.findPlug("inMesh", False).isDestination:
        return False
    plug = fn.findPlug("cachedInMesh", False)
    try:
        cached = OpenMaya.MFnMesh(plug.asMObject())
    except RuntimeError:  # no copy (never tweaked)
        return False
    if (
        cached.getUVSetNames() == fn.getUVSetNames()
        and cached.getColorSetNames() == fn.getColorSetNames()
    ):
        return False
    if cached.numVertices != fn.numVertices:
        raise RuntimeError(
            f"{fn.fullPathName()}: Maya left the mesh out of step with its cached input "
            "(the undo of a native mesh edit after a vertex edit); flush the undo queue "
            "(cmds.flushUndo()) before editing its sets"
        )
    data = OpenMaya.MFnMeshData().create()
    OpenMaya.MFnMesh().copy(fn.findPlug("outMesh", False).asMObject(), data)
    OpenMaya.MFnMesh(data).setPoints(cached.getPoints())
    plug.setMObject(data)
    return True


def _set_element(fn: OpenMaya.MFnMesh, array: str, name: str) -> OpenMaya.MPlug | None:
    """The element of the mesh's ``uvSet`` / ``colorSet`` `array` that holds the
    set `name` (its first child is the set's name), or None."""
    plug = fn.findPlug(array, False)
    for i in range(plug.numElements()):
        element = plug.elementByPhysicalIndex(i)
        if element.child(0).asString() == name:
            return element
    return None


def _links(fn: OpenMaya.MFnMesh, array: str, name: str) -> tuple:
    """What `_relink` makes again for the set `name`: ``(element index,
    [(child, [destination plugs])])``."""
    element  = _set_element(fn, array, name)
    children = [(c, list(element.child(c).destinations())) for c in range(element.numChildren())]
    return element.logicalIndex(), children


def _free_elements(fn: OpenMaya.MFnMesh, array: str, indices) -> None:
    """Removes the EMPTY elements `indices` of `array` (an API delete empties
    the set's element and keeps it), so that the API's next sets take them
    back (it takes the first free element)."""
    plug     = fn.findPlug(array, False)
    modifier = OpenMaya.MDGModifier()
    for index in indices:
        element = plug.elementByLogicalIndex(index)
        if not element.child(0).asString():
            modifier.removeMultiInstance(element, True)
    modifier.doIt()


def _relink(fn: OpenMaya.MFnMesh, array: str, name: str, links: tuple) -> None:
    """Connects the element of the set `name` to the destinations `links` holds
    (`_links`) that nothing drives now."""
    element  = _set_element(fn, array, name)
    modifier = OpenMaya.MDGModifier()
    for child, destinations in links[1]:
        for destination in destinations:
            if not destination.isDestination:
                modifier.connect(element.child(child), destination)
    modifier.doIt()


class _MeshEdit:
    """[Internal] An undoable edit of one mesh: a copy of its MDagPath, and
    `fn`, a new MFnMesh of it on every call, its sets synced (`_sync_sets`)."""

    def __init__(self, mesh: Mesh) -> None:
        self._path = OpenMaya.MDagPath(type(mesh).get_mdagpath(mesh))

    def fn(self) -> OpenMaya.MFnMesh:
        fn = OpenMaya.MFnMesh(self._path)
        return OpenMaya.MFnMesh(self._path) if _sync_sets(fn) else fn


class _MeshSetPointsCommand(_MeshEdit):
    """`Mesh.set_points`: every point of the mesh, in one space. It edits no
    set, so it skips `_sync_sets` (setPoints keeps the sets either way)."""

    UNDO_EDIT = "Mesh.set_points"

    def __init__(self, mesh: Mesh, points: OpenMaya.MPointArray, space: int) -> None:
        super().__init__(mesh)
        self._points = points
        self._space  = space
        self._old    = None
        _run_undoable(self)

    def doIt(self) -> None:
        fn        = OpenMaya.MFnMesh(self._path)
        self._old = fn.getPoints(self._space)
        _put_points(fn, self._points, self._space)

    def undoIt(self) -> None:
        _put_points(OpenMaya.MFnMesh(self._path), self._old, self._space)

    def redoIt(self) -> None:
        _put_points(OpenMaya.MFnMesh(self._path), self._points, self._space)


def _shading_now(fn: OpenMaya.MFnMesh) -> tuple:
    """The shading of the mesh `fn` as `_put_shading` puts it back: ``(hard edge
    ids, (faces, vertices, normals) of its locked face-vertex normals)``."""
    hard   = [edge for edge in range(fn.numEdges) if not fn.isEdgeSmooth(edge)]
    locked = np.array([fn.isNormalLocked(i) for i in range(fn.numNormals)], dtype=bool)
    if not locked.any():
        return hard, ((), (), ())
    normals, ids = _normals_now(fn), _normal_ids_now(fn)
    rows = np.flatnonzero(locked[ids])
    faces, verts = _face_vertices(fn, rows)
    return hard, (faces, verts, normals[ids[rows]])


def _put_shading(fn: OpenMaya.MFnMesh, hard: list[int], unlock: tuple) -> None:
    """Unlocks the face-vertex normals ``unlock`` (``(faces, vertices)``), then makes
    the edges `hard` hard and every other edge smooth, and updates the mesh."""
    faces, verts = unlock[:2]
    if len(faces):
        fn.unlockFaceVertexNormals(np.asarray(faces).tolist(), np.asarray(verts).tolist())
    flags = np.ones(fn.numEdges, dtype=bool)
    flags[np.asarray(hard, dtype=np.int64)] = False
    fn.setEdgeSmoothings(list(range(fn.numEdges)), flags.tolist())
    fn.cleanupEdgeSmoothing()
    fn.updateSurface()


class _MeshUpdateCommand(_MeshEdit):
    """`Mesh('x').load`: every point of the mesh, in one space, then, with the
    data's `shading` (`_shading_of`), its hard edges and locked normals in place
    of the mesh's own. Undo puts back the points, then the old shading (its hard
    edges and its locked normals); redo makes the edit again."""

    UNDO_EDIT = "Mesh.load"

    def __init__(self, mesh: Mesh, points: OpenMaya.MPointArray, space: int, shading) -> None:
        super().__init__(mesh)
        self._points  = points
        self._space   = space
        self._shading = shading
        self._old     = None
        self._locked  = ((), ())
        _run_undoable(self)

    def doIt(self) -> None:
        fn = OpenMaya.MFnMesh(self._path)
        old_points = fn.getPoints(self._space)
        old_shading = _shading_now(fn) if self._shading is not None else None
        self._old = (old_points, old_shading)
        self.redoIt()

    # A new MFnMesh after each edit that updates the surface: one kept across setPoints
    # writes the shading into geometry the mesh no longer uses (the edge flags were lost)

    def redoIt(self) -> None:
        _put_points(OpenMaya.MFnMesh(self._path), self._points, self._space)
        if self._shading is not None:
            fn   = OpenMaya.MFnMesh(self._path)
            hard = _edge_ids(fn, self._shading.hard, self._shading)
            _put_shading(fn, hard, self._old[1][1])
            self._locked = _lock_differing(OpenMaya.MFnMesh(self._path), self._shading, hard)

    def undoIt(self) -> None:
        points, shading = self._old
        _put_points(OpenMaya.MFnMesh(self._path), points, self._space)
        if shading is not None:
            hard, (faces, verts, normals) = shading
            _put_shading(OpenMaya.MFnMesh(self._path), hard, self._locked)
            _set_normals(OpenMaya.MFnMesh(self._path), faces, verts, normals, hard)


class _MeshSetsEdit(_MeshEdit):
    """[Internal] An edit of the mesh's UV or colour sets. `_capture` keeps
    what the undo needs, then `_apply` makes the edit. Without history, redo
    is `_apply` again and undo is `_revert`. With history (``_modifier`` is
    set before `_capture`), `_apply` gets an MDGModifier, which records the
    nodes the API adds, and undo / redo are the modifier's. The subclass sets
    its fields, then calls this constructor, which runs it."""

    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh)
        self._modifier = None
        _run_undoable(self)

    def doIt(self) -> None:
        fn = self.fn()
        if _has_history(fn):
            self._modifier = OpenMaya.MDGModifier()
        self._capture(fn)
        self._apply(fn, self._modifier)

    def undoIt(self) -> None:
        fn = self.fn()  # first: a mesh deleted since raises here, not in the modifier
        if self._modifier is None:
            self._revert(fn)
        else:
            self._modifier.undoIt()

    def redoIt(self) -> None:
        fn = self.fn()
        if self._modifier is None:
            self._apply(fn, None)
        else:
            self._modifier.doIt()

    def _capture(self, fn: OpenMaya.MFnMesh) -> None:
        pass

    def _apply(self, fn: OpenMaya.MFnMesh, modifier: OpenMaya.MDGModifier | None) -> None:
        raise NotImplementedError

    def _revert(self, fn: OpenMaya.MFnMesh) -> None:
        raise NotImplementedError


# --- UV sets


def _uv_lists(uv_data: UVData) -> tuple[list, list, list, list]:
    """`uv_data`'s UVs as the lists the API takes: ``(u, v, counts, ids)``."""
    points = np.asarray(uv_data.points, dtype=float).reshape(-1, 2)
    return (
        points[:, 0].tolist(),
        points[:, 1].tolist(),
        np.asarray(uv_data.counts, dtype=np.int64).ravel().tolist(),
        np.asarray(uv_data.indices, dtype=np.int64).ravel().tolist(),
    )


def _uvs_of(fn: OpenMaya.MFnMesh, uv_set: str) -> tuple:
    """The UVs of the set `uv_set`: ``(u, v, counts, ids)``."""
    return (*fn.getUVs(uv_set), *fn.getAssignedUVs(uv_set))


def _put_uvs(fn: OpenMaya.MFnMesh, uv_set: str, uvs: tuple, history: bool) -> None:
    """Replaces the UVs of the set `uv_set` with `uvs` (``(u, v, counts, ids)``).

    Fewer UVs than the set has: without history the set is cleared first
    (``setUVs`` cannot shrink it); with history it is not (Maya's ``clearUVs``
    there adds a ``polyMapDel`` node that clears the CURRENT set), so the extra
    UVs stay in the set, unassigned (`Mesh.set_uv_data` refuses to write fewer
    UVs there)."""
    u, v, counts, ids = uvs
    now = fn.numUVs(uv_set)
    if len(u) < now:
        if history:
            old_u, old_v = fn.getUVs(uv_set)
            u = list(u) + list(old_u)[len(u):]
            v = list(v) + list(old_v)[len(v):]
        else:
            fn.clearUVs(uv_set)
    if len(u):
        fn.setUVs(u, v, uv_set)
        fn.assignUVs(counts, ids, uv_set)


class _MeshAddUVSetCommand(_MeshSetsEdit):
    """`Mesh.add_uv_set`: an empty UV set. Undo frees its element, so a redo
    makes it in the same one."""

    UNDO_EDIT = "Mesh.add_uv_set"

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name = name
        super().__init__(mesh)

    def _apply(self, fn, modifier):
        fn.createUVSet(self._name, **_modified(modifier))

    def _revert(self, fn):
        index = _set_element(fn, "uvSet", self._name).logicalIndex()
        fn.deleteUVSet(self._name)
        _free_elements(fn, "uvSet", [index])


class _MeshRenameUVSetCommand(_MeshEdit):
    """`Mesh.rename_uv_set`. A rename adds no node, with or without history,
    so it takes no modifier (whose undo would re-evaluate the history and drop
    the UVs written into the set through the API)."""

    UNDO_EDIT = "Mesh.rename_uv_set"

    def __init__(self, mesh: Mesh, name: str, new_name: str) -> None:
        super().__init__(mesh)
        self._name     = name
        self._new_name = new_name
        _run_undoable(self)

    def doIt(self) -> None:
        self.fn().renameUVSet(self._name, self._new_name)

    def undoIt(self) -> None:
        self.fn().renameUVSet(self._new_name, self._name)

    def redoIt(self) -> None:
        self.doIt()


class _MeshDeleteUVSetCommand(_MeshSetsEdit):
    """`Mesh.delete_uv_set`. Undo makes the set again with its UVs, in its
    place (the API adds a set last, so the sets after it are made again after
    it), each in its uvSet[] element with its connections, and the current set
    is current again. With history the modifier's undo brings the set back from
    the history, without the UVs written into it through the API: they are
    written again."""

    UNDO_EDIT = "Mesh.delete_uv_set"

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name    = name
        self._uvs     = None
        self._links   = None
        self._later   = ()
        self._current = ""
        super().__init__(mesh)

    def _capture(self, fn):
        names         = list(fn.getUVSetNames())
        self._uvs     = _uvs_of(fn, self._name)
        self._links   = _links(fn, "uvSet", self._name)
        self._later   = names[names.index(self._name) + 1 :]
        self._current = fn.currentUVSetName()

    def _apply(self, fn, modifier):
        fn.deleteUVSet(self._name, **_modified(modifier))

    def undoIt(self) -> None:
        super().undoIt()
        if self._modifier is not None:
            _put_uvs(self.fn(), self._name, self._uvs, True)

    def _revert(self, fn):
        later = [(name, _uvs_of(fn, name), _links(fn, "uvSet", name)) for name in self._later]
        for name, _, _ in later:
            fn.deleteUVSet(name)
        sets = [(self._name, self._uvs, self._links), *later]
        _free_elements(fn, "uvSet", [links[0] for _, _, links in sets])
        for name, uvs, links in sets:
            fn.createUVSet(name)
            _put_uvs(fn, name, uvs, False)
            _relink(fn, "uvSet", name, links)
        if fn.currentUVSetName() != self._current:
            fn.setCurrentUVSetName(self._current)


class _MeshSetUVDataCommand(_MeshEdit):
    """`Mesh.set_uv_data`: the UVs of one set, and its rename to the data's
    name. The UVs are data, written through the API with or without history
    (see `_put_uvs`)."""

    UNDO_EDIT = "Mesh.set_uv_data"

    def __init__(self, mesh: Mesh, uv_set: str, uv_data: UVData) -> None:
        super().__init__(mesh)
        self._uv_set  = uv_set
        self._name    = uv_data.name
        self._new     = _uv_lists(uv_data)
        self._old     = None
        self._history = False
        _run_undoable(self)

    def doIt(self) -> None:
        fn            = self.fn()
        self._history = _has_history(fn)
        self._old     = _uvs_of(fn, self._uv_set)
        try:
            _put_uvs(fn, self._uv_set, self._new, self._history)
        except BaseException:
            _put_uvs(fn, self._uv_set, self._old, self._history)  # nothing is queued
            raise
        self._rename(fn, self._uv_set, self._name)

    def undoIt(self) -> None:
        fn = self.fn()
        self._rename(fn, self._name, self._uv_set)
        _put_uvs(fn, self._uv_set, self._old, self._history)

    def redoIt(self) -> None:
        fn = self.fn()
        _put_uvs(fn, self._uv_set, self._new, self._history)
        self._rename(fn, self._uv_set, self._name)

    @staticmethod
    def _rename(fn: OpenMaya.MFnMesh, name: str, new_name: str) -> None:
        if name != new_name:
            fn.renameUVSet(name, new_name)


def _check_uv_data(fn: OpenMaya.MFnMesh, uv_set: str, uv_data: UVData) -> None:
    """Raises, before any edit, when `Mesh.set_uv_data` cannot write `uv_data`
    into the set `uv_set` of the mesh `fn` and rename it ``uv_data.name``."""
    names = fn.getUVSetNames()
    if uv_set not in names:
        raise RuntimeError(f"UV set {uv_set} does not exist.")
    name = uv_data.name
    if not isinstance(name, str) or not name:
        raise TypeError(f"UVData.name must be a non-empty str, not {name!r}")
    if name != uv_set and name in names:
        raise RuntimeError(f"UV set {name} already exists.")
    n_uvs   = len(np.asarray(uv_data.points, dtype=float).reshape(-1, 2))
    counts  = np.asarray(uv_data.counts, dtype=np.int64).ravel()
    indices = np.asarray(uv_data.indices, dtype=np.int64).ravel()
    if n_uvs < fn.numUVs(uv_set) and _has_history(fn):
        raise RuntimeError(
            f"UVData {name!r}: {n_uvs} UVs for the {fn.numUVs(uv_set)} of UV set {uv_set}: "
            "rig cannot remove UVs from a mesh with history (Maya's clearUVs there clears "
            "the current UV set); delete its history first"
        )
    if not n_uvs and not len(indices):
        return  # no UVs: the set is emptied
    if len(counts) != fn.numPolygons:
        raise RuntimeError(f"UVData {name!r}: {len(counts)} face counts for {fn.numPolygons} faces")
    if int(counts.sum()) != len(indices):
        raise RuntimeError(
            f"UVData {name!r}: the face counts add up to {int(counts.sum())}, "
            f"but there are {len(indices)} indices"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= n_uvs):
        raise RuntimeError(
            f"UVData {name!r}: an index is not one of the {n_uvs} UVs "
            f"(indices range {int(indices.min())}..{int(indices.max())})"
        )


# --- colour sets


def _color_ids(fn: OpenMaya.MFnMesh, color_set: str) -> list[int]:
    """The colour index of every face-vertex of `color_set`, face by face (-1:
    no colour)."""
    ids = []
    it  = OpenMaya.MItMeshPolygon(fn.object())
    while not it.isDone():
        try:
            ids.extend(it.getColorIndices(color_set))
        except RuntimeError:  # a face without colours
            ids.extend([-1] * it.polygonVertexCount())
        it.next()
    return ids


def _capture_colors(fn: OpenMaya.MFnMesh, color_set: str) -> tuple:
    """The colours of `color_set` as `_put_colors` puts them back: ``(pool,
    ids)``; nothing to read for a set without colours."""
    if not fn.numColors(color_set):
        return (), ()
    return fn.getColors(color_set), _color_ids(fn, color_set)


def _put_colors(fn: OpenMaya.MFnMesh, color_set: str, colors: tuple, rep: int) -> None:
    """Replaces the colours of `color_set` with `colors` (`_capture_colors`):
    ``clearColors``, then the pool and the per-face-vertex indices."""
    pool, ids = colors
    fn.clearColors(color_set)
    if len(pool):
        fn.setColors(pool, color_set, rep=rep)
        fn.assignColors(ids, color_set)


def _write_vertex_colors(
    fn:        OpenMaya.MFnMesh,
    color_set: str,
    colors:    OpenMaya.MColorArray,
    rep:       int,
    modifier:  OpenMaya.MDGModifier | None,
) -> None:
    """``setVertexColors`` of every vertex into `color_set`, made the current
    set for the write (the current set is put back after it)."""
    current = fn.currentColorSetName()
    if current != color_set:
        fn.setCurrentColorSetName(color_set)
    try:
        fn.setVertexColors(colors, range(fn.numVertices), rep=rep, **_modified(modifier))
    finally:
        if current and current != color_set:
            fn.setCurrentColorSetName(current)


# --- the data's colours: a MeshData's ``colors`` / ``color_indices`` to and from a colour set

# the colour an unset face corner serializes as: what Maya's own fbx exporter writes for one
# (``polyColorPerVertex -query`` reports it too); the API reads such a corner as (-1, -1, -1, -1)
_UNSET_COLOR = OpenMaya.MColor((0.0, 0.0, 0.0, 0.0))

# the colour set a create makes for the data's colours: Maya's name for a first set
_COLOR_SET = "colorSet1"


def _first_set_colors(fn: OpenMaya.MFnMesh) -> tuple:
    """The colours of the first colour set of the mesh `fn` as a MeshData holds them:
    ``(colors (K, 4) float64, color_indices (C,) int32)``, the distinct RGBA colours and
    the one each face corner names (an unset corner `_UNSET_COLOR`), as Maya holds them
    (an RGB set's alpha 1, an Alpha set's RGB as they are); ``(None, None)`` for a mesh
    with no colour set, or whose first set holds no colour. One API call: a set's own
    pool and indices would take a call per face."""
    names = fn.getColorSetNames()
    if not names or not fn.numColors(names[0]):
        return None, None
    held = fn.getFaceVertexColors(names[0], _UNSET_COLOR)
    rows = np.fromiter(itertools.chain.from_iterable(held), dtype=np.float64, count=4 * len(held)).reshape(-1, 4)
    rows += 0.0  # a -0.0 is 0.0 as a float, not as bytes
    # the distinct rows by their bytes: 2 to 6x faster than np.unique(rows, axis=0) on 80k corners
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * 4))).ravel()
    _, first, ids = np.unique(keys, return_index=True, return_inverse=True)
    return rows[first], ids.astype(np.int32)


def _data_colors(mesh_data: MeshData, n_rows: int) -> tuple | None:
    """The data's colours and the colour of each of its `n_rows` face corners:
    ``(colors (K, 4), ids (n_rows,))``, or None when the data holds no colours.
    cgmath's two forms: ``color_indices`` names a colour a corner; without them the
    colours are one a vertex, which each corner takes through ``indices``. Raises
    RuntimeError for colours that fit neither, before anything is made."""
    colors = getattr(mesh_data, "colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 4:
        raise RuntimeError(f"MeshData: the colors are {colors.shape}, not (N, 4) RGBA")
    if not np.isfinite(colors).all():
        raise RuntimeError("MeshData: the colors must be finite (got nan or inf)")
    ids = getattr(mesh_data, "color_indices", None)
    if ids is None:
        n_points = len(mesh_data.points)
        if colors.shape[0] != n_points:
            raise RuntimeError(
                f"MeshData: {colors.shape[0]} colors for {n_points} vertices and no color_indices "
                f"(a colour a face-vertex names its face-vertices)"
            )
        ids = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    else:
        ids = np.asarray(ids, dtype=np.int64).ravel()
        if len(ids) != n_rows:
            raise RuntimeError(f"MeshData: {len(ids)} color indices for {n_rows} face-vertices (one each)")
    if not len(ids) and not colors.size:
        return None
    if len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= len(colors)):
        raise RuntimeError(
            f"MeshData: a color index is not one of the {len(colors)} colors "
            f"(indices range {int(ids.min())}..{int(ids.max())})"
        )
    return colors, ids


def _built_corners(mesh_data: MeshData, holes: dict[int, list[list[int]]]) -> np.ndarray:
    """For each face corner of the mesh `_holed_mesh` builds, in its order, the data's
    corner it comes from: a holed face lists its outer loop, then each hole loop
    (`_holed_uvs`' rule), every other face its corners as the data does. Indexes a
    per-corner array of the data (the colour ids) into the built order."""
    counts = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    ids    = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    starts = np.concatenate(([0], np.cumsum(counts)))
    order  = np.arange(len(ids), dtype=np.int64)
    for face, loops in holes.items():
        face_verts = ids[starts[face] : starts[face + 1]].tolist()
        by_vertex  = {v: starts[face] + k for k, v in enumerate(face_verts)}
        on_holes   = {v for loop in loops for v in loop}
        built      = [v for v in face_verts if v not in on_holes] + [v for loop in loops for v in loop]
        order[starts[face] : starts[face + 1]] = [by_vertex[v] for v in built]
    return order


def _restore_colors(fn: OpenMaya.MFnMesh, colored: tuple) -> None:
    """The data's colours `colored` (`_data_colors`: ``(colors, ids)``, the ids in the
    order `fn` lists its face corners) as the colour set `_COLOR_SET` of the new mesh
    `fn`, RGBA and unclamped. Pure API: `Mesh.create` calls it while it builds the
    shape, before any recorded command touches it."""
    colors, ids = colored
    fn.createColorSet(_COLOR_SET, False, rep=OpenMaya.MFnMesh.kRGBA)
    _put_colors(fn, _COLOR_SET, (OpenMaya.MColorArray(colors.tolist()), ids.tolist()), OpenMaya.MFnMesh.kRGBA)


def _update_colors(mesh: Mesh, colors: np.ndarray, ids: np.ndarray) -> None:
    """`Mesh('x').load`'s colours: the data's (`_data_colors`) into the mesh's first
    colour set, its representation kept; a mesh without one gets `_COLOR_SET` (RGBA,
    unclamped) first. One or two undo steps, inside the load's chunk."""
    sets      = type(mesh).get_color_sets(mesh)
    color_set = sets[0] if sets else type(mesh).add_color_set(mesh, _COLOR_SET, ColorSet.Representation.RGBA)
    _ColorSetCornersCommand(
        mesh, color_set.get_name(), color_set.get_representation().value,
        OpenMaya.MColorArray(colors.tolist()), ids.tolist(),
    )


class _ColorSetCornersCommand(_MeshSetsEdit):
    """`Mesh('x').load`: the colours of a colour set replaced, one a face corner
    (`_put_colors`: the set's pool and indices). Undo puts the set's colours back as
    they were, pool and indices; the current colour set stays current."""

    UNDO_EDIT = "Mesh.load"

    def __init__(self, mesh: Mesh, name: str, rep: int, colors: OpenMaya.MColorArray, ids: list) -> None:
        self._name    = name
        self._rep     = rep
        self._colors  = (colors, ids)
        self._old     = None
        self._current = ""
        super().__init__(mesh)

    def undoIt(self) -> None:
        super().undoIt()
        self._keep_current()

    def redoIt(self) -> None:
        super().redoIt()
        self._keep_current()

    def _keep_current(self) -> None:
        fn = self.fn()
        if self._current and fn.currentColorSetName() != self._current:
            fn.setCurrentColorSetName(self._current)

    def _capture(self, fn):
        self._current = fn.currentColorSetName()
        if self._modifier is None:
            self._old = _capture_colors(fn, self._name)

    def _apply(self, fn, modifier):
        _put_colors(fn, self._name, self._colors, self._rep)

    def _revert(self, fn):
        _put_colors(fn, self._name, self._old, self._rep)


class _MeshAddColorSetCommand(_MeshSetsEdit):
    """`Mesh.add_color_set`: an empty, unclamped colour set. Undo frees its
    element, so a redo makes it in the same one."""

    UNDO_EDIT = "Mesh.add_color_set"

    def __init__(self, mesh: Mesh, name: str, rep: int) -> None:
        self._name = name
        self._rep  = rep
        super().__init__(mesh)

    def _apply(self, fn, modifier):
        fn.createColorSet(self._name, False, rep=self._rep, **_modified(modifier))

    def _revert(self, fn):
        index = _set_element(fn, "colorSet", self._name).logicalIndex()
        fn.deleteColorSet(self._name)
        _free_elements(fn, "colorSet", [index])


class _ColorSetDataCommand(_MeshSetsEdit):
    """`ColorSet.set_data`: one colour per vertex. Undo puts the set's colours back
    face-vertex by face-vertex; the current colour set stays current."""

    UNDO_EDIT = "ColorSet.set_data"

    def __init__(self, mesh: Mesh, name: str, rep: int, colors: OpenMaya.MColorArray) -> None:
        self._name    = name
        self._rep     = rep
        self._colors  = colors
        self._old     = None
        self._current = ""
        super().__init__(mesh)

    def undoIt(self) -> None:
        super().undoIt()
        self._keep_current()

    def redoIt(self) -> None:
        super().redoIt()
        self._keep_current()

    def _keep_current(self) -> None:
        # with history, the modifier's undo / redo re-evaluates the mesh
        fn = self.fn()
        if self._current and fn.currentColorSetName() != self._current:
            fn.setCurrentColorSetName(self._current)

    def _capture(self, fn):
        self._current = fn.currentColorSetName()
        if self._modifier is None:
            self._old = _capture_colors(fn, self._name)

    def _apply(self, fn, modifier):
        _write_vertex_colors(fn, self._name, self._colors, self._rep, modifier)

    def _revert(self, fn):
        _put_colors(fn, self._name, self._old, self._rep)


def _color_set_of(fn: OpenMaya.MFnMesh, color_set: str) -> tuple:
    """What `_make_color_set` makes `color_set` again from: ``(representation,
    clamped, colours)``."""
    return (
        fn.getColorRepresentation(color_set),
        fn.isColorClamped(color_set),
        _capture_colors(fn, color_set),
    )


def _make_color_set(fn: OpenMaya.MFnMesh, color_set: str, saved: tuple) -> None:
    """Makes the colour set `color_set` again from `saved` (`_color_set_of`)."""
    rep, clamped, colors = saved
    fn.createColorSet(color_set, clamped, rep=rep)
    _put_colors(fn, color_set, colors, rep)


class _ColorSetDeleteCommand(_MeshSetsEdit):
    """`ColorSet.delete`. Undo makes the set again with its representation,
    clamping and colours, in its place (the API adds a set last, so the sets
    after it are made again after it), each in its colorSet[] element with its
    connections, and the current set is current again."""

    UNDO_EDIT = "ColorSet.delete"

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name    = name
        self._saved   = None
        self._links   = None
        self._later   = ()
        self._current = ""
        super().__init__(mesh)

    def _capture(self, fn):
        if self._modifier is None:
            names         = list(fn.getColorSetNames())
            self._saved   = _color_set_of(fn, self._name)
            self._links   = _links(fn, "colorSet", self._name)
            self._later   = names[names.index(self._name) + 1 :]
            self._current = fn.currentColorSetName()

    def _apply(self, fn, modifier):
        fn.deleteColorSet(self._name, **_modified(modifier))

    def _revert(self, fn):
        later = [(name, _color_set_of(fn, name), _links(fn, "colorSet", name)) for name in self._later]
        for name, _, _ in later:
            fn.deleteColorSet(name)
        sets = [(self._name, self._saved, self._links), *later]
        _free_elements(fn, "colorSet", [links[0] for _, _, links in sets])
        for name, saved, links in sets:
            _make_color_set(fn, name, saved)
            _relink(fn, "colorSet", name, links)
        if self._current and fn.currentColorSetName() != self._current:
            fn.setCurrentColorSetName(self._current)


class ColorSet:
    """A colour set of a mesh: a handle on the scene, so every public member is
    a method (user decision 2026-10-16): ``get_name()``, ``get_mesh()``,
    ``is_valid()``, ``get_representation()``, ``get_data()`` /
    ``set_data(colours)``, ``delete()``. Made by ``mesh.get_color_sets()``,
    ``mesh.add_color_set(name, rep)`` or ``ColorSet(mesh, name)``."""

    __slots__ = ("_mesh", "_name")

    class Representation(enum.Enum):
        """
        Enum for color set representations
        """

        Alpha = OpenMaya.MFnMesh.kAlpha
        RGB   = OpenMaya.MFnMesh.kRGB
        RGBA  = OpenMaya.MFnMesh.kRGBA

    # the members before the per-object rule (user decision 2026-10-16) and the
    # method the names review retired (user decision 2026-10-30): a miss, or an
    # assignment, names the method now (`_old_handle_name`)
    _OLD_NAMES = {
        "name":           "cset.name is now cset.get_name()",
        "mesh":           "cset.mesh is now cset.get_mesh()",
        "representation": "cset.representation is now cset.get_representation()",
        "data":           "cset.data is now cset.get_data() / cset.set_data(colours)",
        # the method the names review retired
        "as_current_color_set": "cset.as_current_color_set is now cset.as_current()",
    }

    def __init__(self, mesh: Mesh, name: str):
        self._mesh = mesh
        self._name = name

    def __getattr__(self, name: str) -> Any:
        raise _old_handle_name(self, name, ColorSet._OLD_NAMES)

    def __setattr__(self, name: str, value: Any) -> None:
        if name[:1] != "_":
            raise _old_handle_name(self, name, ColorSet._OLD_NAMES)
        object.__setattr__(self, name, value)

    def __repr__(self):
        return f"ColorSet<{self._mesh}.{self._name}>"

    def get_name(self) -> str:
        """The name of the colour set (the one this handle was made with)."""
        return self._name

    def get_mesh(self) -> Mesh:
        """The mesh this colour set belongs to."""
        return self._mesh

    @_scene_predicate
    def is_valid(self, *, failfast: bool = False) -> bool:
        """Does the underlying colorset still exist? A method (named without
        brackets, ``cset.is_valid`` has no truth value: the predicate guard).
        False too once its mesh is gone.

        Args:
            failfast: By keyword only. If True, raise a RuntimeError naming this
                colour set instead of returning False. It then never returns
                False, so it is a statement, not a condition (as
                ``DGNode.is_valid(failfast=True)``).
        """
        mesh = self._mesh
        if type(mesh)._raw_is_valid(mesh) and self._name in type(mesh)._raw_get_fn_set(mesh).getColorSetNames():
            return True
        if failfast:
            # (the mesh's str, marked once it is gone: its name would raise instead)
            raise RuntimeError(f"{self._name} is not a valid color set for {mesh}!")
        return False

    def get_representation(self) -> ColorSet.Representation:
        """The representation of the color set (ie the data type Alpha|RGB|RGBA)"""
        self.is_valid(failfast=True)
        return self.Representation(type(self._mesh)._raw_get_fn_set(self._mesh).getColorRepresentation(self._name))

    def get_data(self) -> np.ndarray:
        """The colours of the set, one per vertex, as a ``(V, 4)`` numpy array.
        A read: the current colour set is left as it is, and the scene is not
        marked modified."""
        self.is_valid(failfast=True)
        return np.array(type(self._mesh)._raw_get_fn_set(self._mesh).getVertexColors(self._name))

    def set_data(self, data: np.ndarray) -> None:
        """Set the colours: one per vertex, a ``(V, 4)`` array. One undo step,
        whose undo puts back the set's colours as they were, face-vertex by
        face-vertex. The current colour set is unchanged."""
        mesh = self._mesh
        data = np.asarray(data)
        if data.shape[0] != type(mesh).get_vertex_count(mesh):
            raise RuntimeError(f"shape of data is not ({type(mesh).get_vertex_count(mesh)}, 4): !")
        rep = self.get_representation().value  # raises for a colour set that is gone
        _ColorSetDataCommand(mesh, self._name, rep, OpenMaya.MColorArray(data))

    @contextlib.contextmanager
    def as_current(self) -> Generator[None, None, None]:
        """Context manager that sets the current color set to this one"""
        self.is_valid(failfast=True)

        try:
            current_colorset = type(self._mesh)._raw_get_fn_set(self._mesh).currentColorSetName()
            type(self._mesh)._raw_get_fn_set(self._mesh).setCurrentColorSetName(self._name)
            yield
        finally:
            if current_colorset:
                type(self._mesh)._raw_get_fn_set(self._mesh).setCurrentColorSetName(current_colorset)

    def delete(self):
        """Delete this color set (nothing happens when it no longer exists): one
        undo step, whose undo brings the set back with its representation,
        clamping and colours (current again if it was)."""
        if self.is_valid():
            _ColorSetDeleteCommand(self._mesh, self._name)


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "Axis",
    "ColorSet",
    "MAP_DATATYPE",
    "MAP_DEFAULT_CATEGORY",
    "MAP_DEFAULT_VALUE",
    "Mesh",
]
