"""[Internal] The file side of the typed ``load`` / ``save`` methods.

cgmath writes and reads the files: its own ``.npz`` / ``.json`` / ``.pkl``, the
format chosen by the extension on a save; on a load an npz or a json is read
from the file's header, a pickle by its ``.pkl`` / ``.pickle`` extension. A
cgmath file names no class: a load builds it as the class the slot takes, the
first of several whose fields the file holds (`read`). ``load`` also takes the
assets the slot's cgmath class reads (``.obj``, ``.fbx``, ``.glb``, ``.usd``).
rig checks what a file holds before anything is made, and a save path before
anything is serialized. The data paths need the cgmath rig is pinned to
(`CGMATH_PIN`, `require_pinned_cgmath`): an older one is an ImportError that
names it.

`load` is one name with two forms (`creates_or_updates`): from the class it is
the creator, ``Mesh.load(path)`` (a new node, as ``create`` makes one); from a
node it is the updater, ``Mesh('body').load(path)`` (that node, in place: the
inverse of ``Mesh('body').save(path)``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from types import FunctionType, MethodType
from typing import Iterable

from rig.nodetypes._base import Attribute, _name_of

# the files cgmath's Data.save writes, by extension
OWN_TYPES = ("npz", "json", "pkl")

# the extensions of the assets cgmath's classes read
ASSET_TYPES = ("obj", "fbx", "glb", "gltf", "usd", "usda", "usdc", "usdz")

# what ``world_space=`` says on a shape's create and its class-form load, for the TypeError of a
# value that is no bool (`space_flag`)
CREATING_SPACE = "where the data's points land: in the world, or as the shape's own"

# the cgmath the data paths are written against: 4.0.0's shape model, a ShapeData's uuid and
# parent_uuid, no matrix, a MeshData's colors / color_indices and its hard_edges, a NurbsCurveData's
# and a NurbsSurfaceData's weights, renew_uuids() on a hierarchy, a clip and a Scene, a
# MaterialData's uuid and the lists of one mesh (README.md's requirements and the CHEATSHEET name
# it too)
CGMATH_PIN = "cgmath 4.0.0rc1 (02b8b75, 2026-11-11)"


def require_pinned_cgmath(method: str) -> None:
    """The data paths need the cgmath rig is pinned to (`CGMATH_PIN`) or later: one whose shapes
    are ``ShapeData`` (``uuid``, ``parent_uuid``), whose ``MeshData`` holds ``colors`` and
    ``hard_edges`` (6001ea6: the hard edges of a mesh that stores no normals), whose
    ``NurbsCurveData`` and ``NurbsSurfaceData`` hold ``weights`` (8fb221e: a rational shape's)
    and whose ``MaterialData`` holds a ``uuid`` (b0c991b, after ``renew_uuids()``). An older
    cgmath (3.0.1, 3.2.0, a commit before those) raises ImportError naming `method`, the
    pin and the cgmath found, before anything is read or made."""
    try:
        from cgmath.geometry import MaterialData, MeshData, NurbsCurveData
        from cgmath.hierarchy import ShapeData  # noqa: F401

        if not {"colors", "hard_edges"} <= set(getattr(MeshData, "__dataclass_fields__", {})):
            raise ImportError
        if "weights" not in getattr(NurbsCurveData, "__dataclass_fields__", {}):
            raise ImportError
        if "uuid" not in getattr(MaterialData, "__dataclass_fields__", {}):
            raise ImportError
    except ImportError:
        import cgmath

        raise ImportError(
            f"{method} needs {CGMATH_PIN} or later, whose shapes carry a parent_uuid, whose meshes hold colors "
            f"and hard_edges, whose curves and surfaces hold weights and whose scenes renew uuids; this cgmath is "
            f"{getattr(cgmath, '__version__', '?')} at {os.path.dirname(cgmath.__file__)}"
        ) from None


def is_path(src) -> bool:
    """True for a path: a str or an os.PathLike, never a plug (a str subclass)."""
    return isinstance(src, (str, os.PathLike)) and not isinstance(src, Attribute)


def _is_node(src) -> bool:
    from rig.nodetypes._base import Node

    return isinstance(src, (Node, Attribute))


def looks_like_file(src) -> bool:
    """True for a path that names a file: one that exists, or one ending in an
    extension a load reads (a node name never has one)."""
    if not is_path(src):
        return False
    path = os.path.expanduser(os.fspath(src))
    ext  = os.path.splitext(path)[1].lstrip(".").lower()
    return ext in OWN_TYPES or ext in ASSET_TYPES or os.path.isfile(path)


def check_order(call: str, first, second) -> None:
    """A deformer's load takes the file first, then the geometry: raises
    TypeError naming the order (`call`, ``"SkinCluster.load(skin, geometry)"``)
    when the geometry came first (a node, or a file in the geometry's place, or
    the data object itself in the geometry's place: a geometry is a node or its
    name). Nothing is read."""
    data_second = second is not None and not is_path(second) and not _is_node(second)
    if (
        _is_node(first)
        or (looks_like_file(second) and not looks_like_file(first))
        or (data_second and is_path(first) and not looks_like_file(first))
    ):
        raise TypeError(
            f"{call} takes the file first, then the geometry: got {_shown(first)} first "
            f"and {_shown(second)} second"
        )


def space_flag(method: str, world_space, meaning: str = "the space the file was saved in") -> bool:
    """``world_space=`` of a load or a create (`method`, ``"NurbsCurve('ctlShape').load"``)
    as a bool: anything but a bool (None, ``0``, ``"world"``) is a TypeError naming it
    and its `meaning`, before any file is read (the user's decision 2 of 2026-10-16:
    the wrong space moves the CVs silently)."""
    import numpy as np

    if not isinstance(world_space, (bool, np.bool_)):
        raise TypeError(f"{method}: world_space= takes True or False ({meaning}), not {world_space!r}")
    return bool(world_space)


# what ``uuids=`` says on a create from data and its class-form load (`uuid_flag`)
UUIDS_MEAN = "True (the default: a node made from data takes the data's uuid) or False (Maya's own)"

# how a shape's data takes a fresh uuid (`free_uuids`' advice for a mesh, a curve, a surface)
FRESH_UUID = "data.uuid = cgmath.hierarchy.generate_uuid()"


def uuid_flag(method: str, uuids) -> bool:
    """``uuids=`` of a create from data or its class-form load (`method`, ``"Mesh.create()"``) as a
    bool: anything but a bool (None, ``1``) is a TypeError naming it, before anything is read or
    made."""
    import numpy as np

    if not isinstance(uuids, (bool, np.bool_)):
        raise TypeError(f"{method}: uuids= is {UUIDS_MEAN}, not {uuids!r}")
    return bool(uuids)


def free_uuids(method: str, named: Iterable[tuple], renew: str) -> None:
    """The uuids the data of `method` would give the nodes it makes, checked before anything is
    made: `named` pairs ``(what, uuid)``, ``what`` naming the record for the error (``"the node
    'spine'"``; a uuid of None is Maya's own, not checked). A uuid a scene node holds, or one the
    data holds twice (two nodes cannot hold one uuid: Maya's ``rename -uuid`` refuses it) is a
    ValueError naming them, `renew` (how the data takes fresh uuids: ``"hierarchy.renew_uuids()"``)
    and ``uuids=False``. Maya spells a uuid upper-case: the check reads the data's so."""
    from maya import cmds

    seen = {}
    for what, uuid in named:
        if uuid is None:
            continue
        key = str(uuid).upper()
        if key in seen:
            raise ValueError(
                f"{method}: {seen[key]} and {what} hold one uuid, {key}, which two nodes cannot; give the data "
                f"fresh uuids ({renew}), or uuids=False keeps Maya's own"
            )
        seen[key] = what
        found     = cmds.ls(key, long=True)
        if found:
            raise ValueError(
                f"{method}: {what} has the uuid {key}, which {found[0]!r} holds, and two nodes cannot hold one; "
                f"for a second copy give the data fresh uuids first ({renew}, or a Scene's renew_uuids()); "
                f"uuids=False keeps Maya's own"
            )


def node_of_uuid(uuid) -> str | None:
    """The long name of the scene node holding `uuid`, the Maya side of a cgmath link (a shape's
    ``parent_uuid``: the transform it sits under); None for no uuid, or none holding it."""
    if uuid is None:
        return None
    from maya import cmds

    found = cmds.ls(str(uuid).upper(), long=True)
    return found[0] if found else None


def is_renderable(data) -> bool:
    """True for a ``cgmath.render.Renderable``: one mesh of a Scene with what places, deforms and
    shades it, joined by uuid (``scene.renderables()``). Read from the loaded module only: a
    Renderable exists only once ``cgmath.render.scene`` is imported, so nothing is imported here."""
    scene = sys.modules.get("cgmath.render.scene")
    return scene is not None and isinstance(data, scene.Renderable)


def adopt_uuid(node: str, uuid) -> None:
    """Gives the node `node` (a name or a path) the data's `uuid` (None: left as Maya made it): a
    recorded ``rename -uuid``, so the create's undo puts Maya's uuid back and its redo the data's
    (measured in Maya 2025: ``rename -uuid`` is undoable, the uuid survives a .ma and a .mb, and a
    uuid another node holds is refused, which `free_uuids` has ruled out before the node was
    made)."""
    if uuid is not None:
        from rig._internal.journal import cmds_edit

        cmds_edit.rename(node, str(uuid), uuid=True)


def _shown(value) -> str:
    if _is_node(value):
        return f"the node {str(value)!r}"
    if is_path(value):
        return repr(os.fspath(value))
    return f"a {type(value).__name__}"


def read(
    src,
    accepted: tuple,
    method:   str,
    slot:     str,
    refused:  tuple = (),
    hints:    tuple = (),
):
    """The data `src` names: a path is read by cgmath (its own files as the
    first class of `accepted` whose fields the file holds, since a cgmath file
    names no class, `_load_own`; an asset by the slot's first class), a data
    object is taken as it is. Anything that is not one of `accepted` (or is one
    of `refused`, a subclass the slot does not take) is a TypeError naming the
    file, what it holds and what `slot` takes, with the hint of `hints`
    (``((classes, text), ...)``) that fits what it holds. A node or a plug is a
    TypeError (a node is no file). A missing file is a FileNotFoundError. An
    older cgmath than the pin is an ImportError first (`require_pinned_cgmath`).
    Nothing is made."""
    require_pinned_cgmath(method)
    if _is_node(src):
        raise TypeError(f"{method}: {slot} is a file (a path) or its data, not the node {_name_of(src)!r}")
    if is_path(src):
        from cgmath.data import get_file_type

        path = os.path.expanduser(os.fspath(src))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{method}: {slot}: no file {path!r}")
        if get_file_type(path) in OWN_TYPES:
            data = _load_own(path, accepted, method, slot, _classes_of(refused, hints))
        else:
            data = accepted[0].load(path)  # an asset: the slot's own reader
        what = f"{path!r} holds a {type(data).__name__}"
    else:
        data = src
        what = f"got a {type(data).__name__}"
    if not isinstance(data, accepted) or isinstance(data, refused):
        names = " or a ".join(cls.__name__ for cls in accepted)
        hint  = next((f" ({text})" for classes, text in hints if isinstance(data, classes)), "")
        raise TypeError(f"{method}: {slot}: {what}; {slot} is a {names}{hint}")
    return data


def _classes_of(refused: tuple, hints: tuple) -> tuple:
    """The classes of `refused` and of `hints` (``((classes, text), ...)``, a class or a tuple
    each), once each, in order: what `_load_own` tries after the accepted ones, so that an
    error can still say what a file holds."""
    classes = list(refused)
    for group, _ in hints:
        classes.extend(group if isinstance(group, tuple) else (group,))
    return tuple(dict.fromkeys(classes))


def _known() -> tuple:
    """cgmath's own data classes, the strictest first, to name what a file holds when the slot's
    classes refuse it: ``NurbsCurveData`` last, since it builds from any points."""
    from cgmath.geometry import (
        GeomSubsetData,
        MapData,
        MeshData,
        MorphData,
        MorphList,
        NurbsCurveData,
        NurbsSurfaceData,
        SkinData,
        UVData,
        UVList,
    )
    from cgmath.hierarchy import ClipData, HierarchyData

    return (
        MeshData, SkinData, ClipData, HierarchyData, NurbsSurfaceData, UVList, UVData,
        MorphList, MorphData, MapData, GeomSubsetData, NurbsCurveData,
    )


def _load_own(path: str, accepted: tuple, method: str, slot: str, others: tuple = ()):
    """A cgmath file (``.npz`` / ``.json`` / ``.pkl``) as the first class of `accepted` whose fields
    it holds: a cgmath file names no class, and a class the file does not fit refuses it (the
    ValueError or TypeError of its ``from_dict``; here a key naming no field, which a load only
    warns about, refuses the class too, so a mesh's file is no curve's). `accepted` goes from
    the most specific class down (a ``ClipData`` before its ``HierarchyData``, which would build
    from a clip's file and drop its frames). None fits: the file is read as `others` (the slot's
    refused and hinted classes) and every class of `_known`, as strictly and quietly, so that
    `read`'s error names what it holds (a UV file in the mesh's slot). Still none: `accepted`
    again, a key naming no field allowed, as a file a later cgmath wrote loads with its warning;
    else a TypeError naming the file, what `slot` takes and the first refusal."""
    reason = None
    for cls in accepted:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                return cls.load(path)
        except (TypeError, ValueError, Warning) as error:
            reason = reason or f"{cls.__name__}.load: {error}"
    candidates = list(others) + list(_known())
    for cls in dict.fromkeys(candidates):
        if cls in accepted:
            continue
        with contextlib.suppress(Exception), warnings.catch_warnings():
            warnings.simplefilter("error")
            return cls.load(path)
    for cls in accepted:
        with contextlib.suppress(TypeError, ValueError):
            return cls.load(path)
    names = " or a ".join(cls.__name__ for cls in accepted)
    raise TypeError(f"{method}: {slot}: {path!r} holds no {names} ({reason})")


def check_save_path(path, method: str) -> str:
    """The path a save writes, checked before anything is serialized: one of
    cgmath's own types by its extension (``.npz``, ``.json``, ``.pkl``), in a
    folder that exists; ``~`` expanded."""
    if not is_path(path):
        raise TypeError(f"{method}: a path is a str or an os.PathLike, not a {type(path).__name__}")
    path = os.path.expanduser(os.fspath(path))
    ext  = os.path.splitext(path)[1].lstrip(".").lower()
    if ext not in OWN_TYPES:
        raise ValueError(f"{method}: {path!r}: cgmath saves .npz, .json or .pkl files, by extension")
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{method}: {path!r}: no folder {folder!r}")
    return path


@contextlib.contextmanager
def kept_selection():
    """The selection in its order, put back when the block changed it (a load keeps
    the selection, where ``cmds.skinCluster`` selects the influences it bound)."""
    from maya import cmds

    from rig._internal.journal import cmds_edit

    ordered = cmds.ls(orderedSelection=True, long=True) or []
    try:
        yield
    finally:
        if (cmds.ls(orderedSelection=True, long=True) or []) != ordered:
            if ordered:
                # cmds.select sorts a batch: the last one goes back last
                cmds_edit.select(ordered[:-1], replace=True)
                cmds_edit.select(ordered[-1], add=True)
            else:
                cmds_edit.select(clear=True)


def same_file(a: str, b: str) -> bool:
    """True when two checked paths name one file."""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class creates_or_updates:
    """A method with two forms, told apart by the call site: from the class it
    is `create` (``Cls.load(path)`` makes a new node), from a node it is
    `update` (``node.load(path)`` updates that node in place). Both forms
    share one docstring, which describes both."""

    def __init__(self, create, update, doc: str) -> None:
        self.create  = create
        self.update  = update
        self.owner   = None
        self.copies  = {}
        self.__doc__ = doc
        for fn in (create, update):
            fn.__doc__ = doc

    def __set_name__(self, owner, name: str) -> None:
        self.owner = owner
        for fn in (self.create, self.update):
            # the function a statement wraps too: Python's own errors for its arguments
            # name the method, ``NurbsCurve.load() takes 2 positional arguments ...``
            for named in (fn, getattr(fn, "__wrapped__", None)):
                if named is not None:
                    named.__name__     = name
                    named.__qualname__ = f"{owner.__name__}.{name}"

    def _named_for(self, cls, fn):
        """``fn``, or for a subclass of the class that defines it a copy named after
        that subclass (``Joint.load_hierarchy``, not ``Transform.load_hierarchy``),
        made once: help() and a traceback name the class the call was made on."""
        if cls is self.owner or cls is None:
            return fn
        key  = (cls, fn)
        copy = self.copies.get(key)
        if copy is None:
            copy = FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
            copy.__kwdefaults__ = fn.__kwdefaults__
            copy.__dict__.update(fn.__dict__)
            copy.__doc__        = fn.__doc__
            copy.__module__     = fn.__module__
            copy.__qualname__   = f"{cls.__name__}.{fn.__name__}"
            self.copies[key]    = copy
        return copy

    def __get__(self, obj, cls=None):
        if obj is None:
            return MethodType(self._named_for(cls, self.create), cls)
        return MethodType(self._named_for(type(obj), self.update), obj)
