"""
Joint node class
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from maya import cmds
from rig.maya.nodetypes.dg_node import DGNode, get_short_name
from rig.maya.nodetypes.transform import Transform


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
    return np.reshape(cmds.getAttr(f"{node}.worldMatrix[0]"), (4, 4))


def replace_suffix(name: Any, suffix: str) -> str:
    """Replaces a node name's suffix.

    Args:
        suffix: The suffix to use. Can NOT contain `_`.

    Returns:
        New name with the suffix replaced.
    """
    if suffix.find("_") != -1:
        raise ValueError("Suffix can NOT contain '_'")
    name = get_short_name(name)
    i    = name.rfind("_")
    # if no suffix, append it
    if i == -1:
        return f"{name}_{suffix}"
    return name[: i + 1] + suffix


class Joint(Transform):
    """
    Joint node class
    """

    NATIVE_NODE_TYPE = "joint"

    # --- skeleton hierarchy

    def get_root_joint(self) -> Joint:
        """Returns the root joint of this joint."""
        parts = self.long_name.split("|")[1:-1]
        name  = ""
        for part in parts:
            name += "|" + part
            if cmds.nodeType(name) == "joint":
                return Joint(name)

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
        if yield_self and (not regex or regex(self.short_name)):
            yield self
        for joint in self.get_children(type="joint"):
            if not regex or regex(joint.short_name):
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
        it                   = self.iter_joints(**kwargs)
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

        Args:
            name: Name for the duplicated joint.
                If None, use this joint's name.
            parent: Parent of the duplicated joint.
                If None, parent to world.
            include_list: If not None, only include joints in this list.
            suffix: A new suffix to use by the duplicated joints.

        Returns:
            The duplicated joint.
        """
        name = self.short_name if not name else name
        if suffix:
            name = replace_suffix(name, suffix)

        if prefix:
            name = f"{prefix}_{name}"

        # duplicate this joint, set parent and rename
        dup = self.duplicate(
            parentOnly=False, renameChildren=False, returnRootsOnly=True
        )[0]
        dup.set_parent(parent)
        dup.rename(name)

        # create a set of joint names to use as a filter
        include_set = None
        if include_list:
            include_set = {
                x.short_name if isinstance(x, Transform) else str(x)
                for x in include_list
            }
        for child in dup.get_children(allDescendents=True, type="transform"):
            if suffix:
                child.rename(replace_suffix(child, suffix))

            if prefix:
                child.rename(f"{prefix}_{child.short_name}")

            # delete non-joint xforms and unwanted joints
            if child.node_type != "joint" or (
                include_set and child.short_name not in include_set
            ):
                p = child.get_parent()
                for each in child.get_children(type="transform"):
                    each.set_parent(p)
                child.delete()

            # unlock joint xform attrs
            else:
                child.set_xfrom_attrs_locked(False)

        # clean up skeleton
        dup.freeze(translate=False, rotate=clean_rotations, scale=clean_scales)
        return dup

    def rename_skeleton(self, suffix: str) -> None:
        """Renames this joint and all children joints by replacing their suffix.

        Args:
            suffix: The suffix to use. Can NOT contain `_`.
        """
        if not suffix.isalnum():
            raise RuntimeError("Suffix '{suffix}' containt non-alphanumeric character")
        self.rename(replace_suffix(self, suffix))

        for c in self.get_children(allDescendents=True):
            c.rename(replace_suffix(c, suffix))

    def match_hierarchy(self, target: "Joint", world_space: bool = False) -> None:
        """Matches the matrix positions of hierarchy of two joints."""

        if isinstance(target, str):
            target = Joint(target)

        # get root joints
        root        = self.get_root_joint()
        target_root = target.get_root_joint()

        # check num of children
        child_joints = root.get_children(
            allDescendents=True, noIntermediate=True, type="joint"
        )
        target_child_joints = target_root.get_children(
            allDescendents=True, noIntermediate=True, type="joint"
        )

        if len(child_joints) != len(target_child_joints):
            raise RuntimeError("mismatch in number of joints")

        # match root matrix
        self.match_matrix(target_root, world_space=world_space)

        # match children matrices
        for i in range(len(target_child_joints)):
            child_joints[i].match_matrix(
                target_child_joints[i], world_space=world_space
            )

    def convert_orients_to_rotation(self):
        """converts joint orientation into rotations"""
        # Imported lazily: ``transforms`` is only needed for orientation math,
        # so importing rig must not pay for it.
        from transforms import matrix_to_euler, XYZ

        matrix = _world_matrix(self.name)
        rot    = np.degrees(matrix_to_euler(matrix, XYZ))[0].tolist()
        cmds.setAttr(f"{self.name}.jointOrient", 0, 0, 0)
        cmds.xform(self.name, worldSpace=True, rotation=rot)

    def convert_rotation_to_orients(self):
        """converts joint rotation into orients"""
        from transforms import matrix_to_euler, XYZ

        local_matrix = np.reshape(cmds.getAttr(f"{self.name}.matrix"), (4, 4))
        rotation     = np.degrees(matrix_to_euler(local_matrix, XYZ))[0].tolist()

        cmds.setAttr(f"{self.name}.r", 0, 0, 0)
        cmds.setAttr(f"{self.name}.jo", rotation[0], rotation[1], rotation[2])

    def hierarchy_to_rotations(self):
        """converts joint orients into rotations"""
        child_joints = cmds.listRelatives(
            self.name, c=True, ad=True, noIntermediate=True, type="joint"
        )
        child_joints.append(self.name)
        for joint in child_joints:
            Joint(joint).convert_orients_to_rotation()

    def hierarchy_to_orients(self):
        """converts joint orients into rotations"""
        child_joints = cmds.listRelatives(
            self.name, c=True, ad=True, noIntermediate=True, type="joint"
        )
        child_joints.append(self.name)
        for joint in child_joints:
            Joint(joint).convert_rotation_to_orients()

    def orient_joint(self, aim_axis="x", up_axis="y", root_joint=False):
        """sets joint orient based on axis and up vector"""
        from transforms import vector_to_euler

        self.convert_orients_to_rotation()

        # check if end joint
        child_joints = cmds.listRelatives(
            self.name, c=True, noIntermediate=True, type="joint"
        )
        if not child_joints:
            # no children, ju
            cmds.setAttr(f"{self.name}.jo", 0, 0, 0)
            cmds.setAttr(f"{self.name}.rx", 0)
            cmds.setAttr(f"{self.name}.ry", 0)
            cmds.setAttr(f"{self.name}.rz", 0)
            print(f"{self.name} has no children, zeroing orients")
            return

        # get list of all child joint parents
        child_joint_parents = [
            cmds.listRelatives(c, parent=True)[0] for c in child_joints
        ]
        cmds.parent(child_joints, w=True)

        # get the parent matrix
        matrix       = _world_matrix(self.name)
        child_matrix = _world_matrix(child_joints[0])

        # make aim vector
        aim_vec = child_matrix[3, :3] - matrix[3, :3]

        # make up vector -- the up axis carried into world space. The parent's
        # frame is the better reference when there is one; fall back to our own.
        if root_joint:
            up_matrix = matrix
        else:
            parent    = cmds.listRelatives(self.name, parent=True)
            up_matrix = _world_matrix(parent[0]) if parent else matrix

        aim_sign, aim_key = _split_axis(aim_axis)
        up_sign, up_key = _split_axis(up_axis)
        up_vec = up_sign * (np.asarray(_AXIS_VECTOR[up_key]) @ up_matrix[:3, :3])

        # build the aim frame and read it back in the joint's own rotation order
        rotation_order = cmds.getAttr(f"{self.name}.ro")
        rotation = np.degrees(
            vector_to_euler(
                aim_sign * aim_vec,
                up_vec,
                _AXIS_INDEX[aim_key],
                _AXIS_INDEX[up_key],
                rotation_order,
            )
        )[0].tolist()
        cmds.xform(self.name, worldSpace=True, rotation=rotation)
        self.convert_orients_to_rotation()

        # reparent children
        for i in range(len(child_joints)):
            cmds.parent(child_joints[i], child_joint_parents[i])

    def orient_chain(self, aim_axis="x", up_axis="y"):
        """orients chain given aim axis and up axis"""

        child_joints = self.get_children(
            allDescendents=True, noIntermediate=True, type="joint"
        )
        self.orient_joint(aim_axis, up_axis, root_joint=True)

        for c in child_joints:
            c.orient_joint(aim_axis, up_axis)