"""
A cgmath Scene in Maya: :func:`create_scene` builds one, :func:`load_scene` reads a file and
builds it.

A cgmath ``Scene`` (``cgmath.render``) is what an fbx holds: the nodes at rest (``hierarchy``),
the takes over them (``clip``), and the meshes with what deforms them, joined by uuid
(``scene.renderables()``). The build is rig's own creates, in the order a rig binds:

1. the nodes, ``Transform.create_hierarchy(scene.hierarchy)``, each taking its data's uuid;
2. each mesh, ``Mesh.create(item)``, under the node its ``parent_uuid`` names;
3. its blend shape targets, ``BlendShape.create(mesh, item)``, in-betweens and weights;
4. its skin, ``SkinCluster.create(mesh, item)``, bound at the nodes' rest values;
5. the active take, ``Transform.key_hierarchy(scene.clip.active)``, keyed onto the nodes.

A scene of takes alone (no ``hierarchy``) builds its nodes from the active take, keyed; a skin
in one is refused (it holds no bind pose to bind at). Cameras, lights and materials are left
out, each with one warning naming the count (rig has no camera or light class; materials are a
later release: the shapes join ``initialShadingGroup``).

Usage::

    from rig.scene import create_scene, load_scene

    built = load_scene("hero.fbx")          # cgmath's Scene.load_fbx, then create_scene
    built = create_scene(scene)             # a Scene in memory
    scene.renew_uuids()
    again = create_scene(scene)             # a second copy beside the first

The word "scene" in these docstrings is the cgmath Scene; Maya's open file is "the Maya scene".
"""

from __future__ import annotations

import logging
import os

from maya import cmds
from rig._internal.journal import cmds_edit as _cmds_edit
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_RENEW = "scene.renew_uuids()"

# load_scene's readers, by extension: a cgmath Scene's loads, and its own file formats
_READERS = {
    ".fbx":  "load_fbx",
    ".obj":  "load_obj",
    ".glb":  "load_glb",
    ".npz":  "load",
    ".json": "load",
    ".pkl":  "load",
}


def create_scene(scene, parent=None, *, uuids: bool = True):
    """Builds a cgmath Scene in the Maya scene: its nodes, meshes, blend shapes and skins, and
    its active take keyed (the module's steps).

    Args:
        scene: A ``cgmath.render.Scene`` (``Scene.load_fbx(path)``), left as it was.
        parent: An existing node to build the roots under (``create_hierarchy``'s
            ``parent``); a mesh on no node goes under it too. None: the world.
        uuids: By keyword only, a bool (anything else is a TypeError before anything is
            made). True (the default): every node and shape takes its data's uuid, so the
            parts find one another by it, and a later ``Node(uuid)`` finds them; a uuid the
            Maya scene holds already, or one the data holds twice, is a ValueError before
            anything is made (a second copy takes ``scene.renew_uuids()`` first). False:
            Maya's own uuids; the parts are joined by what this call made.

    Returns:
        A List of what was made: the nodes, parents first, then mesh by mesh its shape, its
        blendShape when it has targets and its skinCluster when it has a skin.

    Checked before anything is made: the uuids, a mesh's targets naming two blendShape
    deformers (one blendShape takes one deformer's: ``BlendShape.create(mesh, item,
    deformer=...)`` builds each), a target with no name or at a weight Maya does not hold,
    a skin influence no node of the scene and no joint of the Maya scene carries, a node of
    the take neither the scene's nodes nor the Maya scene holds, and a skin in a scene of
    takes alone. One undo step named ``rig.create_scene``; the selection is left as it was.
    """
    from cgmath.render import Scene
    from rig._internal.list import List
    from rig.nodetypes import BlendShape, Mesh, SkinCluster, Transform

    method = "create_scene"
    if isinstance(scene, (str, os.PathLike)):
        raise TypeError(f"{method}() takes the data, not a file: load_scene({os.fspath(scene)!r}) reads one")
    if not isinstance(scene, Scene):
        raise TypeError(f"{method}() takes a cgmath Scene, not a {type(scene).__name__}")
    uuids = _files.uuid_flag(f"{method}()", uuids)
    _files.require_pinned_cgmath(method)

    take     = scene.clip.active if scene.clip is not None and len(scene.clip) else None
    skeleton = scene.hierarchy if scene.hierarchy is not None else take
    items    = scene.renderables()
    _check(method, scene, skeleton, take, items, uuids)
    _warn_left_out(scene)

    before = cmds.ls(orderedSelection=True, long=True) or []
    made   = []
    with _undo_chunk(f"rig.{method}"):
        placed = {}
        if skeleton is not None:
            nodes = Transform.create_hierarchy(skeleton, parent, uuids=uuids)
            for data, node in zip(_kept(skeleton), nodes):
                placed[data.uuid] = node
            made.extend(nodes)

        for item in items:
            transform = placed.get(getattr(item.mesh, "parent_uuid", None))
            if transform is not None:
                mesh = Mesh.create(item, transform=transform, uuids=uuids)
            else:
                mesh = Mesh.create(item, uuids=uuids)
                if parent is not None:
                    _cmds_edit.parent(mesh.get_parent().get_long_name(), parent, relative=True)
            made.append(mesh)
            if len(item.morphs):
                made.append(BlendShape.create(mesh, item))
            if item.skin is not None:
                made.append(SkinCluster.create(mesh, _bound_skin(item, placed)))

        if take is not None and skeleton is not take:
            Transform._key_found(_on_nodes(take, placed))

        if before:
            _cmds_edit.select(before, replace=True)
        elif cmds.ls(selection=True):
            _cmds_edit.select(clear=True)
    return List(made)


def load_scene(path, parent=None, *, uuids: bool = True, **flags):
    """Reads a cgmath Scene from `path` and builds it (`create_scene`).

    Args:
        path: A str or os.PathLike (``~`` expanded). The extension picks the reader:
            ``.fbx`` ``Scene.load_fbx``, ``.obj`` ``Scene.load_obj``, ``.glb``
            ``Scene.load_glb``, ``.npz`` / ``.json`` / ``.pkl`` ``Scene.load`` (a Scene
            cgmath saved; never load a ``.pkl`` you did not write).
        parent, uuids: as `create_scene` takes them.
        flags: the reader's own keywords: ``Scene.load_fbx``'s kind flags (``takes=``,
            ``meshes=``, ``uvs=``, ``skins=``, ``materials=``, ``textures=``, ``morphs=``,
            ``cameras=``, ``lights=``, ``verbose=``), ``Scene.load_glb``'s
            (``scale_factor=`` ...).

    Returns:
        `create_scene`'s List.

    A missing file is a FileNotFoundError, an extension no reader takes a ValueError, before
    anything is read.
    """
    from cgmath.render import Scene

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"load_scene() takes a file, not a {type(path).__name__}: create_scene(scene) builds the data")
    path   = os.path.expanduser(os.fspath(path))
    reader = _READERS.get(os.path.splitext(path)[1].lower())
    if reader is None:
        raise ValueError(f"load_scene: {path!r}: no reader for its extension; it reads {', '.join(_READERS)}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"load_scene: no file {path!r}")
    scene = getattr(Scene, reader)(path, **flags)
    if not isinstance(scene, Scene):
        raise TypeError(f"load_scene: {path!r} holds a {type(scene).__name__}, not a Scene")
    return create_scene(scene, parent, uuids=uuids)


# --- the checks, before anything is made


def _check(method: str, scene, skeleton, take, items, uuids: bool) -> None:
    """Every refusal of `create_scene`, read before anything is made (rig's creates check
    their own, but a later one would leave the earlier ones' nodes behind)."""
    from rig.nodetypes.blendshape import _check_ats, _check_morph_names, _one_deformer

    if scene.hierarchy is None and take is not None and scene.skins is not None and len(scene.skins):
        raise ValueError(
            f"{method}: the scene holds takes alone, no nodes at rest, so its {len(scene.skins)} skin(s) "
            f"have no bind pose to bind at; give it its hierarchy (the bind pose)"
        )
    if uuids:
        named = [] if skeleton is None else [(f"the node {n.name!r}", n.uuid) for n in _kept(skeleton)]
        named += [(f"the mesh {item.name!r}", getattr(item.mesh, "uuid", None)) for item in items]
        _files.free_uuids(f"{method}()", named, _RENEW)

    names = set() if skeleton is None else {n.name for n in _kept(skeleton)}
    for item in items:
        if len(item.morphs):
            _one_deformer(f"{method}: the mesh {item.name!r}: BlendShape.create", item.morphs, None)
            _check_morph_names(method, item.morphs)
            _check_ats(method, item.morphs)
        if item.skin is not None:
            lost = [n for n in item.skin.influences if n not in names and not cmds.ls(n, type="joint")]
            if lost:
                raise ValueError(
                    f"{method}: the skin of the mesh {item.name!r} names influence(s) {lost} no node of the "
                    f"scene and no joint of the Maya scene carries"
                )
    if take is not None and skeleton is not take:
        held = {n.uuid for n in _kept(skeleton)}
        lost = [n.name for n in take if n.uuid not in held and _files.node_of_uuid(n.uuid) is None
                and len(cmds.ls(n.name, type="transform")) != 1]
        if lost:
            raise ValueError(f"{method}: the take {take.name!r} keys node(s) {lost} the scene's nodes do not hold")


def _warn_left_out(scene) -> None:
    """One warning per kind `create_scene` leaves out, naming the count."""
    for field, why in (
        ("cameras", "rig has no camera class"),
        ("lights", "rig has no light class"),
        ("materials", "materials are a later release: the shapes join initialShadingGroup"),
    ):
        held = getattr(scene, field, None)
        if held is not None and len(held):
            LOGGER.warning(f"create_scene: the scene's {len(held)} {field[:-1]} record(s) are left out ({why})")


# --- the joins, by what the call made


def _kept(hierarchy) -> list:
    """The nodes of `hierarchy` `create_hierarchy` makes, in its order: a node of a type it
    cannot make (`SUPPORTED_NODE_TYPES`) is dropped with all under it, as it drops them."""
    from cgmath.hierarchy import SUPPORTED_NODE_TYPES

    dropped = set()
    kept    = []
    for node in hierarchy:
        parent = node.get_parent()
        if node.node_type not in SUPPORTED_NODE_TYPES or (parent is not None and id(parent) in dropped):
            dropped.add(id(node))
        else:
            kept.append(node)
    return kept


def _bound_skin(item, placed: dict):
    """A copy of ``item.skin`` whose influences are the nodes this call made for its joints
    (`placed`, by uuid), long names; an influence of no such node keeps its name."""
    nodes = {}
    for joint in item.joints:
        node = placed.get(joint.uuid)
        if node is not None:
            nodes.setdefault(joint.name, type(node).get_long_name(node))
    skin            = item.skin.copy()
    skin.influences = [nodes.get(name, name) for name in item.skin.influences]
    return skin


def _on_nodes(take, placed: dict):
    """A copy of `take` whose nodes are the Maya nodes they key: the node this call made for
    its uuid, else the scene node holding it, else the one node of its name
    (`Transform._key_found` takes the copy)."""
    from rig.nodetypes._base import _cast

    keyed = take.copy()
    for node in keyed:
        found = placed.get(node.uuid)
        if found is None:
            path  = _files.node_of_uuid(node.uuid) or cmds.ls(node.name, long=True, type="transform")[0]
            found = _cast(path)
        node.name = found
    return keyed


__all__ = [
    "create_scene",
    "load_scene",
]
