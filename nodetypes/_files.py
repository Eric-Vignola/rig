"""[Internal] The file side of the typed ``load`` / ``save`` methods.

cgmath writes and reads the files: its own ``.npz`` / ``.json`` / ``.pkl``, the
format chosen by the extension on a save and read from the file's header on a
load; ``load`` also takes the assets the slot's cgmath class reads (``.obj``,
``.fbx``, ``.glb``, ``.usd``). rig checks what a file holds before anything is
made, and a save path before anything is serialized.

`load` is one name with two forms (`creates_or_updates`): from the class it is
the creator, ``Mesh.load(path)`` (a new node, as ``create`` makes one); from a
node it is the updater, ``Mesh('body').load(path)`` (that node, in place: the
inverse of ``Mesh('body').save(path)``).
"""

from __future__ import annotations

import contextlib
import os
import sys
from types import MethodType

from rig.nodetypes._base import Attribute

# the files cgmath's Data.save writes, by extension
OWN_TYPES = ("npz", "json", "pkl")

# the extensions of the assets cgmath's classes read
ASSET_TYPES = ("obj", "fbx", "glb", "gltf", "usd", "usda", "usdc", "usdz")


def is_path(src) -> bool:
    """True for a path: a str or an os.PathLike, never a plug (a str subclass)."""
    return isinstance(src, (str, os.PathLike)) and not isinstance(src, Attribute)


def _is_node(src) -> bool:
    from rig.nodetypes._base import Node

    return isinstance(src, (Node, Attribute))


def is_object(data) -> bool:
    """True for a ``cgmath.render.scene.Object`` (a mesh with its UVs and its
    transform). Read from the loaded module only: an Object exists only once
    its module is imported, so nothing is imported here."""
    scene = sys.modules.get("cgmath.render.scene")
    return scene is not None and isinstance(data, scene.Object)


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
    when the geometry came first (a node, or a file in the geometry's place).
    Nothing is read."""
    if _is_node(first) or (looks_like_file(second) and not looks_like_file(first)):
        raise TypeError(
            f"{call} takes the file first, then the geometry: got {_shown(first)} first "
            f"and {_shown(second)} second"
        )


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
    objects:  bool  = False,
):
    """The data `src` names: a path is read by cgmath (its own files whatever
    class they hold, an asset by the slot's first class), a data object is
    taken as it is. Anything that is not one of `accepted` (or is one of
    `refused`, a subclass the slot does not take) is a TypeError naming the
    file, what it holds and what `slot` takes, with the hint of `hints`
    (``((classes, text), ...)``) that fits what it holds. With `objects`, a
    ``cgmath.render.scene.Object`` is accepted too. A node or a plug is a
    TypeError (a node is no file). A missing file is a FileNotFoundError; a
    file from an older cgmath raises cgmath's own LegacyFileError (a
    ValueError). Nothing is made."""
    if _is_node(src):
        raise TypeError(f"{method}: {slot} is a file (a path) or its data, not the node {str(src)!r}")
    if is_path(src):
        from cgmath.geometry import Data
        from cgmath.geometry._base import get_file_type

        path = os.path.expanduser(os.fspath(src))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{method}: {slot}: no file {path!r}")
        if get_file_type(path) in OWN_TYPES:
            data = Data.load(path)
        else:
            data = accepted[0].load(path)  # an asset: the slot's own reader
        what = f"{path!r} holds a {type(data).__name__}"
    else:
        data = src
        what = f"got a {type(data).__name__}"
    if objects and is_object(data):
        return data
    if not isinstance(data, accepted) or isinstance(data, refused):
        names = " or a ".join(cls.__name__ for cls in accepted)
        if objects:
            names += " or an Object"
        hint = next((f" ({text})" for classes, text in hints if isinstance(data, classes)), "")
        raise TypeError(f"{method}: {slot}: {what}; {slot} is a {names}{hint}")
    return data


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
        self.__doc__ = doc
        for fn in (create, update):
            fn.__doc__ = doc

    def __set_name__(self, owner, name: str) -> None:
        for fn in (self.create, self.update):
            fn.__name__     = name
            fn.__qualname__ = f"{owner.__name__}.{name}"

    def __get__(self, obj, cls=None):
        if obj is None:
            return MethodType(self.create, cls)
        return MethodType(self.update, obj)
