"""The bind pose a hierarchy is serialized at: ``Transform.serialize_hierarchy(
bindpose=True | "bindPose5" | Node("bindPose5"))``.

Read only: nothing here edits the scene, so no journal or undo rule applies.

A dagPose stores, per member (logical index ``i``):

    members[i]      <- the member's message
    parents[i]      <- the parent member's members[j], or the pose's world
    xformMatrix[i]     the member's LOCAL transform split (the "xform" data):
                       S, R (an Euler and its order), T, shear, pivots, the
                       rotate axis and the joint orient (quaternions), the
                       inverse parent scale, segment scale compensate
    worldMatrix[i]     its world matrix when the pose was saved
    global[i]          on for a member that is not an influence (ancestors)

The API gives the xform data as an MTransformationMatrix: S, R and its order,
T and the rotate axis are there. The joint orient is not: it is solved from
the composed local matrix, ``M = S RA R JO IS`` (row vectors), so
``inv(S RA R) M = JO IS``, whose column norms are the diagonal ``IS``.

The result is what ``serialize_hierarchy()`` gives after Maya's Go to Bind
Pose (``dagPose -restore -global``): a member takes S, R, T, the rotate axis
and the joint orient from the pose, re-expressed in the node's current rotate
order (as the restore does); everything else, and every node the pose does not
hold, comes from the scene.
"""

from __future__ import annotations

import math

import numpy as np
from maya.api import OpenMaya

from rig.nodetypes.errors import NodeTypeError

# poses that hold a node agree on it when their composed local matrices are
# this close: Maya saved both from one scene state, real disagreements are
# degrees off
_AGREE = 1e-6

# a value the pose agrees with to this keeps the scene's own number, so a
# hierarchy at its bind pose serializes bit-identical with or without bindpose
_SAME = 1e-9

_DEG = 180.0 / math.pi
_RAD = math.pi / 180.0

_SPACE = OpenMaya.MSpace.kTransform


class _Entry:
    """One member of one pose: what ``apply`` writes on its TransformData."""

    __slots__ = ("pose", "pose_obj", "tm", "local", "parent", "parent_index")

    def __init__(self, pose, pose_obj, tm, local, parent, parent_index):
        self.pose         = pose          # the dagPose's name
        self.pose_obj     = pose_obj      # its MObject
        self.tm           = tm            # MTransformationMatrix of the xform data
        self.local        = local         # the composed local MMatrix
        self.parent       = parent        # the pose parent: an MObject, None (the
                                          # world), False (a deleted member) or
                                          # Ellipsis (parents[i] not connected)
        self.parent_index = parent_index  # its logical index in the pose, or None


def _bind_poses(objs: list) -> list:
    """The dagPoses with ``bindPose`` on that hold one of ``objs`` (MObjects),
    as (name, MObject) pairs sorted by name. Each node's message is walked
    once; each pose's flag is read once, through its plug."""
    found = {}
    for obj in objs:
        msg = OpenMaya.MFnDependencyNode(obj).findPlug("message", False)
        for dst in msg.destinations():
            pose = dst.node()
            if not pose.hasFn(OpenMaya.MFn.kDagPose):
                continue
            fn   = OpenMaya.MFnDependencyNode(pose)
            name = fn.name()  # a DG node's name is unique
            if name in found:
                continue
            found[name] = pose if fn.findPlug("bindPose", False).asBool() else None
    return sorted((name, pose) for name, pose in found.items() if pose is not None)


def _named_pose(bindpose) -> tuple:
    """The (name, MObject) of the dagPose ``bindpose`` names (a name; rig's
    lookup rule, then a node-type check)."""
    from rig.nodetypes._base import _lookup

    long = _lookup(bindpose, "dagPose")
    sel  = OpenMaya.MSelectionList()
    sel.add(long)
    obj = sel.getDependNode(0)
    if not obj.hasFn(OpenMaya.MFn.kDagPose):
        raise NodeTypeError(bindpose, "dagPose", OpenMaya.MFnDependencyNode(obj).typeName)
    return OpenMaya.MFnDependencyNode(obj).name(), obj


def _index_of(wanted: dict, objs: list, node: OpenMaya.MObject):
    """The index in ``objs`` of the MObject ``node``, or None; ``wanted`` maps
    MObjectHandle hash codes to the indices that have it."""
    for k in wanted.get(OpenMaya.MObjectHandle(node).hashCode(), ()):
        if objs[k] == node:
            return k
    return None


def _read_pose(name: str, pose: OpenMaya.MObject, wanted: dict, objs: list) -> dict:
    """{index in ``objs``: _Entry} for the members of ``pose`` among ``objs``.
    The member indices are read once; a deleted member leaves a hole (its
    xform entry stays), which is skipped."""
    fn       = OpenMaya.MFnDependencyNode(pose)
    members  = fn.findPlug("members", False)
    parents  = fn.findPlug("parents", False)
    xforms   = fn.findPlug("xformMatrix", False)
    m_attr   = members.attribute()

    sources = {}
    for i in members.getExistingArrayAttributeIndices():
        src = members.elementByLogicalIndex(i).source()
        if not src.isNull:
            sources[i] = src.node()

    out = {}
    for i, node in sources.items():
        k = _index_of(wanted, objs, node)
        if k is None:
            continue
        data = OpenMaya.MFnMatrixData(xforms.elementByLogicalIndex(i).asMObject())

        parent, parent_index = Ellipsis, None
        par = parents.elementByLogicalIndex(i).source()
        if not par.isNull:
            if par.isElement and par.array().attribute() == m_attr:
                parent_index = par.logicalIndex()
                parent       = sources.get(parent_index, False)
            else:
                parent = None  # the pose's world
        out[k] = _Entry(name, pose, data.transformation(), data.matrix(), parent, parent_index)
    return out


def _path(obj: OpenMaya.MObject) -> str:
    return OpenMaya.MFnDagNode(obj).partialPathName()


def resolve(root: str, objs: list, bindpose) -> dict:
    """{index in ``objs``: _Entry} for the hierarchy ``objs`` (the MObjects the
    walk serialized, the root first) at ``bindpose`` (True, or a dagPose name).

    Every check runs before anything is written, and none of them touches the
    scene: no pose (True), a name that is no dagPose, a pose that holds none of
    the nodes, poses that disagree on a node (True), a member whose DAG parent
    is not the parent it was posed under."""
    wanted = {}
    for k, obj in enumerate(objs):
        wanted.setdefault(OpenMaya.MObjectHandle(obj).hashCode(), []).append(k)

    if bindpose is True:
        poses = _bind_poses(objs)
        if not poses:
            raise ValueError(
                f"{root!r} has no bind pose: none of the {len(objs)} nodes of its "
                f"hierarchy is a member of a dagPose with bindPose on"
            )
    else:
        poses = [_named_pose(bindpose)]

    held = {}  # index -> [_Entry, ...], in pose order
    for name, pose in poses:
        for k, entry in _read_pose(name, pose, wanted, objs).items():
            held.setdefault(k, []).append(entry)
    if not held:
        raise ValueError(
            f"{poses[0][0]!r} holds none of the {len(objs)} nodes of the hierarchy at {root!r}"
        )

    # poses that share a node must agree on it
    for k in sorted(held):
        entries = held[k]
        first   = np.array(entries[0].local)
        for other in entries[1:]:
            if np.abs(np.array(other.local) - first).max() > _AGREE:
                names = ", ".join(name for name, _ in poses)
                raise ValueError(
                    f"{root!r} is in bind poses that disagree ({names}): "
                    f"{_path(objs[k])!r} differs between {entries[0].pose!r} and "
                    f"{other.pose!r}; name one: bindpose={entries[0].pose!r}"
                )

    # a member's xform is local to the parent it was posed under
    moved = {}  # pose -> ["'head' (posed under 'chest', now under 'spine')", ...]
    for k in sorted(held):
        dag_parent = OpenMaya.MFnDagNode(objs[k]).parent(0)
        is_world   = dag_parent.hasFn(OpenMaya.MFn.kWorld)
        for entry in held[k]:
            if entry.parent is Ellipsis:
                continue  # the pose does not say
            if entry.parent is None:
                same = is_world
            elif entry.parent is False:
                same = False
            else:
                same = not is_world and entry.parent == dag_parent
            if same:
                continue
            was = ("the world" if entry.parent is None
                   else "a deleted node" if entry.parent is False
                   else repr(_path(entry.parent)))
            now = "the world" if is_world else repr(_path(dag_parent))
            moved.setdefault(entry.pose, []).append(
                f"{_path(objs[k])!r} (posed under {was}, now under {now})"
            )
    if moved:
        held_text = "; ".join(
            f"{pose!r} holds {', '.join(items)}" for pose, items in moved.items()
        )
        raise ValueError(
            f"{held_text}: a pose member is stored relative to the parent it had when "
            f"the pose was saved, restore the parent or save a new pose"
        )
    return {k: entries[0] for k, entries in held.items()}


def _close(a, b) -> bool:
    return bool(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max() <= _SAME)


def _m3(m: OpenMaya.MMatrix) -> np.ndarray:
    return np.array(m).reshape(4, 4)[:3, :3]


def _xyz_degrees(m3: np.ndarray) -> np.ndarray:
    """The XYZ Euler, in degrees, of the rotation matrix ``m3``."""
    m = OpenMaya.MMatrix([*m3[0], 0.0, *m3[1], 0.0, *m3[2], 0.0, 0.0, 0.0, 0.0, 1.0])
    e = OpenMaya.MEulerRotation.decompose(m, OpenMaya.MEulerRotation.kXYZ)
    return np.array([e.x, e.y, e.z]) * _DEG


def _euler_m3(degrees) -> np.ndarray:
    x, y, z = degrees
    return _m3(OpenMaya.MEulerRotation(x * _RAD, y * _RAD, z * _RAD).asMatrix())


def _parent_scale(entry: _Entry, obj: OpenMaya.MObject) -> np.ndarray:
    """The parent's scale the pose was saved under: the pose's own xform of the
    parent member, else the scene's parent (only read for a mirrored parent)."""
    if entry.parent_index is not None and entry.parent is not False:
        xforms = OpenMaya.MFnDependencyNode(entry.pose_obj).findPlug("xformMatrix", False)
        data   = OpenMaya.MFnMatrixData(xforms.elementByLogicalIndex(entry.parent_index).asMObject())
        return np.array(data.transformation().scale(_SPACE))
    parent = OpenMaya.MFnDagNode(obj).parent(0)
    if parent.hasFn(OpenMaya.MFn.kTransform):
        return np.array(OpenMaya.MFnTransform(parent).scale())
    return np.ones(3)


def apply(data, entry: _Entry, obj: OpenMaya.MObject) -> None:
    """Writes the pose of ``entry`` on the TransformData ``data`` of the node
    ``obj``: scale, rotate, translate, rotate axis and (a joint) joint orient.
    The rotation is re-expressed in the node's current rotate order, as
    ``dagPose -restore`` does. A value the pose agrees with keeps the scene's
    own number; the rotate axis and joint orient, which the pose holds as
    quaternions, are compared as matrices and, when they differ, written as
    their XYZ Euler."""
    tm    = entry.tm
    order = tm.rotationOrder() - 1  # MTransformationMatrix.kXYZ is 1, rotateOrder xyz is 0
    euler = tm.rotation()           # an MEulerRotation in that order
    keep  = data.rotate_order       # the node's own order: the restore keeps it

    scale     = np.array(tm.scale(_SPACE))
    rotate    = np.array([euler.x, euler.y, euler.z]) * _DEG
    translate = np.array(tm.translation(_SPACE))[:3]

    if keep != order:
        data.rotate_order = order
        data.rotate       = rotate
    elif not _close(data.rotate, rotate):
        data.rotate = rotate
    if not _close(data.scale, scale):
        data.scale = scale
    if not _close(data.translate, translate):
        data.translate = translate

    q  = tm.rotationOrientation()
    ra = q.asMatrix()
    if not _close(_euler_m3(data.rotate_axis), _m3(ra)):
        e = q.asEulerRotation()
        data.rotate_axis = np.array([e.x, e.y, e.z]) * _DEG

    if data.node_type == "joint":
        s   = OpenMaya.MMatrix([scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1])
        pre = s * ra * euler.asMatrix()
        x   = _m3(pre.inverse() * entry.local)  # = JO IS
        inv_scale = np.linalg.norm(x, axis=0)
        jo  = x / inv_scale
        if np.linalg.det(jo) < 0:
            # a mirrored parent: the inverse parent scale holds its signs
            jo = x / (inv_scale * np.where(_parent_scale(entry, obj) < 0, -1.0, 1.0))
        if not _close(_euler_m3(data.joint_orient), jo):
            data.joint_orient = _xyz_degrees(jo)

    # re-expressed in the node's own order, the pose kept, as the restore does
    if keep != order:
        data.set_rotate_order(keep)
