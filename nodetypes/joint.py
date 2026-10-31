"""
Joint node class
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from maya import cmds
from rig.nodetypes._base import _name_of
from rig._internal.journal import cmds_edit as _cmds_edit
from rig.nodetypes.dg_node import DGNode, get_short_name
from rig.nodetypes.transform import Transform


_AXIS_VECTOR = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_AXIS_INDEX  = {"x": 0, "y": 1, "z": 2}


def _split_axis(axis: str) -> tuple[float, str]:
    """Split an axis token like ``"-y"`` into ``(-1.0, "y")``."""
    sign, key = (-1.0, axis[1:]) if axis.startswith("-") else (1.0, axis)
    if key not in _AXIS_VECTOR:
        raise ValueError(
            f"invalid axis {axis!r}; expected x, y or z, optionally negated"
        )
    return sign, key


def _world_matrix(node: str) -> np.ndarray:
    """Read a node's world matrix as a 4x4 array (row-vector convention)."""
    return np.reshape(cmds.getAttr(f"{_name_of(node)}.worldMatrix[0]"), (4, 4))


def replace_suffix(name: Any, suffix: str) -> str:
    """Replaces a node name's suffix.

    Only the node's own name, after the last ``:``, is searched for the ``_``
    that starts the suffix, and the namespace is kept: ``VLR_RIG:SK:root``
    with ``test`` gives ``VLR_RIG:SK:root_test``.

    Args:
        suffix: The suffix to use. Can NOT contain `_`.

    Returns:
        New name with the suffix replaced.
    """
    if suffix.find("_") != -1:
        raise ValueError("Suffix can NOT contain '_'")
    namespace, _, name = get_short_name(name).rpartition(":")
    i = name.rfind("_")
    # if no suffix, append it
    name = f"{name}_{suffix}" if i == -1 else name[: i + 1] + suffix
    return f"{namespace}:{name}" if namespace else name


def _add_prefix(name: Any, prefix: str) -> str:
    """Puts ``prefix`` and a ``_`` in front of a node's own name, after its
    namespace: ``A:B:root`` with ``L`` gives ``A:B:L_root``."""
    namespace, _, name = get_short_name(name).rpartition(":")
    name = f"{prefix}_{name}"
    return f"{namespace}:{name}" if namespace else name


class Joint(Transform):
    """
    Joint node class
    """

    NATIVE_NODE_TYPE = "joint"

    # --- skeleton hierarchy

    def get_root_joint(self) -> Joint:
        """Returns the root joint of this joint."""
        parts = type(self).get_long_name(self).split("|")[1:-1]
        name  = ""
        for part in parts:
            name += "|" + part
            if cmds.nodeType(name) == "joint":
                return Joint._wrap(name)

    def get_parent_joint(self) -> Joint | None:
        """Returns the parent joint of this joint, if exists."""
        for each in self.iter_parents(node_type="joint"):
            return each

    def iter_joints(
        self,
        match_name:  list[str] | str | None = None,
        yield_self:  bool                   = False,
        exact_match: bool                   = False,
        _regex:      str       | None       = None,
    ) -> Iterator[Joint]:
        """A generator that traverse the skeleton hierarchy from this joint and
        yields joints meeting criteria.

        Args:
            yield_self: If True, yield self at start, if self meet criteria.
            match_name: One or more of names to match, can be regex.
            exact_match: If True, match exact names, otherwise do partial match.
        """
        regex = self._search_regex(match_name, exact_match, _regex)
        if yield_self and (not regex or regex(type(self).get_short_name(self))):
            yield self
        for joint in self.get_children(type="joint"):
            if not regex or regex(type(joint).get_short_name(joint)):
                yield joint
            for each in joint.iter_joints(yield_self=False, _regex=regex):
                yield each

    def find_joint(self, match_name: str, **kwargs) -> Joint | None:
        """Returns the first joint matching given criteria, or None if no match.

        Args:
            match_name: One or more of names to match, can be regex.
            kwargs: Keyword arguments passed to `traverse` method.
        """
        kwargs["match_name"] = match_name
        it                   = type(self).iter_joints(self, **kwargs)
        return next(it, None)

    def find_skinclusters(self, recursive: bool = False) -> list[DGNode]:
        """Finds the skincluster(s) driven by this joint, or joints below this joint
        (if recursive == True)"""
        skins = set()
        for node in self.iter_joints(yield_self=True) if recursive else [self]:
            result = node.find_connected_nodes(
                source=False, destination=True, node_type="skinCluster"
            )
            skins |= set(result)
        return sorted(skins)

    def duplicate_skeleton(
        self,
        name:            str                   | None = None,
        parent:          str                   | None = None,
        clean_rotations: bool                         = True,
        clean_scales:    bool                         = True,
        include_list:    list[str | Transform] | None = None,
        suffix:          str                   | None = None,
        prefix:          str                   | None = None,
    ) -> str:
        """Duplicates this joint and all its child joints. All non-joint children will
        be removed from the duplicated hierarchy.

        The copy is a new skeleton: its joints are named without the original's
        namespace, so they land in Maya's current namespace and a referenced
        rig's namespace is left alone.

        Args:
            name: Name for the duplicated joint, used as given.
                If None, use this joint's name without its namespace.
            parent: Parent of the duplicated joint.
                If None, parent to world.
            include_list: If not None, only include joints in this list,
                matched on their names without namespace.
            suffix: A new suffix to use by the duplicated joints.
            prefix: A prefix put in front of each duplicated joint's own name,
                joined with `_`.

        Returns:
            The duplicated joint.
        """
        name = type(self).get_short_name(self, namespace=False) if not name else name
        if suffix:
            name = replace_suffix(name, suffix)

        if prefix:
            name = _add_prefix(name, prefix)

        # duplicate this joint, set parent and rename
        dup = self.duplicate(
            parentOnly=False, renameChildren=False, returnRootsOnly=True
        )[0]
        dup.set_parent(parent)
        dup.rename(name)

        # create a set of joint names to use as a filter
        include_set = None
        if include_list:
            include_set = {get_short_name(x, namespace=False) for x in include_list}
        for child in dup.get_children(allDescendents=True, type="transform"):
            # delete non-joint xforms and unwanted joints, matched on the name
            # the copy was made with, before any suffix or prefix
            if type(child).get_node_type(child) != "joint" or (
                include_set and type(child).get_short_name(child, namespace=False) not in include_set
            ):
                p = child.get_parent()
                for each in child.get_children(type="transform"):
                    each.set_parent(p)
                child.delete()
                continue

            if suffix:
                child.rename(replace_suffix(child, suffix))

            if prefix:
                child.rename(_add_prefix(child, prefix))

            # unlock joint xform attrs
            child.set_xform_attrs_locked(False)

        # clean up skeleton
        dup.freeze(translate=False, rotate=clean_rotations, scale=clean_scales)
        return dup

    def rename_skeleton(self, suffix: str) -> None:
        """Renames this joint and all children joints by replacing their suffix.

        Args:
            suffix: The suffix to use. Can NOT contain `_`.
        """
        if not suffix.isalnum():
            raise RuntimeError(
                f"Suffix '{suffix}' contains a non-alphanumeric character"
            )
        self.rename(replace_suffix(self, suffix))

        for c in self.get_children(allDescendents=True):
            c.rename(replace_suffix(c, suffix))

    def match_hierarchy(self, target: "Joint", *, world_space: bool = False) -> None:
        """Matches the skeleton of this joint onto the target's, joint by joint
        (`match_matrix`): the root joints of both (found through any of their
        joints, a root included), then the joints below them, paired in
        ``listRelatives`` order (the skeletons must have the same layout),
        parents before children.

        Args:
            target: A joint of the skeleton to match, or its name.
            world_space: By keyword only. False (the default): each joint takes
                the local matrix of its pair, a pose copy (the skeleton lands on
                the target's when the roots' parents are at the same place).
                True: each joint lands on the world matrix of its pair, its
                parent placed first.

        Raises:
            RuntimeError: the skeletons hold different numbers of joints.

        One undo step per joint, as each is set: Maya's redo of one step around
        them all solves each joint with the inverse scale it had before the
        redo, which misplaces the joints under a parent whose scale the match
        changed.
        """

        if isinstance(target, str):
            target = Joint(target)

        # get root joints (a root joint has none above it: it is its own)
        root        = self.get_root_joint() or self
        target_root = target.get_root_joint() or target

        # every joint, parents first (listRelatives -ad lists the deepest first)
        child_joints = root.get_children(
            allDescendents=True, noIntermediate=True, type="joint"
        )
        target_child_joints = target_root.get_children(
            allDescendents=True, noIntermediate=True, type="joint"
        )

        if len(child_joints) != len(target_child_joints):
            raise RuntimeError("mismatch in number of joints")

        joints        = [root] + child_joints[::-1]
        target_joints = [target_root] + target_child_joints[::-1]
        for joint, target_joint in zip(joints, target_joints):
            joint.match_matrix(target_joint, world_space=world_space)

    def _iter_hierarchy(self) -> Iterator[Joint]:
        """The joints below this one, deepest first, then this one; a leaf joint
        has none below (``listRelatives`` answers None). Each is read as the walk
        reaches it, so a short name two joints share raises there."""
        child_joints = cmds.listRelatives(
            _name_of(self), c=True, ad=True, noIntermediate=True, type="joint"
        )
        for joint in (child_joints or []) + [_name_of(self)]:
            yield Joint._wrap(joint)

    def orients_to_rotations(self, *, hierarchy: bool = False) -> None:
        """Converts joint orientation into rotations.

        Args:
            hierarchy: By keyword only. True: this joint and every joint below it.
        """
        if hierarchy:
            for joint in self._iter_hierarchy():
                joint.orients_to_rotations()
            return

        # Imported lazily: ``cgmath.transforms`` is only needed for orientation
        # math, so importing rig must not pay for it.
        from cgmath.transforms import matrix_to_euler, XYZ

        matrix = _world_matrix(_name_of(self))
        rot    = np.degrees(matrix_to_euler(matrix, XYZ))[0].tolist()
        _cmds_edit.setAttr(f"{_name_of(self)}.jointOrient", 0, 0, 0)
        _cmds_edit.xform(_name_of(self), worldSpace=True, rotation=rot)

    def rotations_to_orients(self, *, hierarchy: bool = False) -> None:
        """Converts joint rotation into orients.

        Args:
            hierarchy: By keyword only. True: this joint and every joint below it.
        """
        if hierarchy:
            for joint in self._iter_hierarchy():
                joint.rotations_to_orients()
            return

        from cgmath.transforms import matrix_to_euler, XYZ

        local_matrix = np.reshape(cmds.getAttr(f"{_name_of(self)}.matrix"), (4, 4))
        rotation     = np.degrees(matrix_to_euler(local_matrix, XYZ))[0].tolist()

        _cmds_edit.setAttr(f"{_name_of(self)}.r", 0, 0, 0)
        _cmds_edit.setAttr(f"{_name_of(self)}.jo", rotation[0], rotation[1], rotation[2])

    def orient(
        self,
        aim_axis:   str  = "x",
        up_axis:    str  = "y",
        *,
        root_joint: bool = False,
        hierarchy:  bool = False,
    ) -> None:
        """Sets joint orient based on axis and up vector.

        Args:
            aim_axis: The axis aimed at the first child joint (``"-y"`` negates it).
            up_axis: The axis kept toward the same axis of the parent's frame, or
                of this joint's own with no parent (``"-y"`` negates it).
            root_joint: By keyword only. True: `up_axis` reads this joint's own
                frame, parent or not.
            hierarchy: By keyword only. True: this joint and every joint below it.
                This joint is the chain's root, as if `root_joint` were True; each
                joint below reads its parent's frame.
        """
        if hierarchy:
            child_joints = self.get_children(
                allDescendents=True, noIntermediate=True, type="joint"
            )
            self.orient(aim_axis, up_axis, root_joint=True)

            for c in child_joints:
                c.orient(aim_axis, up_axis)
            return

        from cgmath.transforms import vector_to_euler

        self.orients_to_rotations()

        # check if end joint
        child_joints = cmds.listRelatives(
            _name_of(self), c=True, noIntermediate=True, type="joint"
        )
        if not child_joints:
            # no children, ju
            _cmds_edit.setAttr(f"{_name_of(self)}.jo", 0, 0, 0)
            _cmds_edit.setAttr(f"{_name_of(self)}.rx", 0)
            _cmds_edit.setAttr(f"{_name_of(self)}.ry", 0)
            _cmds_edit.setAttr(f"{_name_of(self)}.rz", 0)
            print(f"{_name_of(self)} has no children, zeroing orients")
            return

        # get list of all child joint parents
        child_joint_parents = [
            cmds.listRelatives(c, parent=True)[0] for c in child_joints
        ]
        _cmds_edit.parent(child_joints, w=True)

        # get the parent matrix
        matrix       = _world_matrix(_name_of(self))
        child_matrix = _world_matrix(child_joints[0])

        # make aim vector
        aim_vec = child_matrix[3, :3] - matrix[3, :3]

        # make up vector -- the up axis carried into world space. The parent's
        # frame is the better reference when there is one; fall back to our own.
        if root_joint:
            up_matrix = matrix
        else:
            parent    = cmds.listRelatives(_name_of(self), parent=True)
            up_matrix = _world_matrix(parent[0]) if parent else matrix

        aim_sign, aim_key = _split_axis(aim_axis)
        up_sign, up_key = _split_axis(up_axis)
        up_vec = up_sign * (np.asarray(_AXIS_VECTOR[up_key]) @ up_matrix[:3, :3])

        # build the aim frame and read it back in the joint's own rotation order
        rotation_order = cmds.getAttr(f"{_name_of(self)}.ro")
        rotation = np.degrees(
            vector_to_euler(
                aim_sign * aim_vec,
                up_vec,
                _AXIS_INDEX[aim_key],
                _AXIS_INDEX[up_key],
                rotation_order,
            )
        )[0].tolist()
        _cmds_edit.xform(_name_of(self), worldSpace=True, rotation=rotation)
        self.orients_to_rotations()

        # reparent children
        for i in range(len(child_joints)):
            _cmds_edit.parent(child_joints[i], child_joint_parents[i])


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "Joint",
    "replace_suffix",
]
