"""
SkeletonDeltaBlend node
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from maya.api import OpenMaya
from rig.maya.nodetypes.dg_node import DGNode
from rig.maya.nodetypes.joint import Joint
from rig.maya.nodetypes.transform import Transform


def _validate_joint_count(a, b):
    if len(a) != len(b):
        raise ValueError(f"mismatched child count: {a[0]} - {b[0]}.")


def _get_joints(
    root: Joint | str, sort: bool = False, out_joints: list[str] | None = None
) -> list[str]:
    joints = cmds.listRelatives(root, ad=True, type="joint", fullPath=True)
    if sort:
        joints.sort()
    if out_joints:
        _validate_joint_count(out_joints, joints)
    return joints


class SkeletonDeltaBlend(DGNode):
    """
    The Skelton delta blend node
    """

    NATIVE_NODE_TYPE = "skeletonDeltaBlend"
    PLUGIN_NAME      = "SkeletonDeltaBlend"

    @classmethod
    def _create(
        cls,
        out_root:     Joint | str,
        ref_root:     Joint | str | None = None,
        anim_root:    Joint | str | None = None,
        zero_orients: bool               = True,
        sort:         bool               = False,
    ) -> str:
        """[Internal] build skeleton delta blend node and return its name.

        Args:
            out_root: Output skeleton root joint.
            ref_root: Reference skeleton root joint.
                If None, use the output skeleton's matrices directly.
            anim_root: Anim skeleton root joint.
                If None, use the output skeleton's matrices directly.
            sort: If True, sort the joints by long names.

        Return:
            The skeleton build node.
        """

        def _connect_or_set(i, attr, in_joints, out_joint):
            if in_joints:
                cmds.connectAttr(f"{in_joints[i]}.matrix", f"{attr}[{i}]", force=True)
            else:
                mtx = cmds.getAttr(f"{out_joint}.matrix")
                cmds.setAttr(f"{attr}[{i}]", mtx, type="matrix")

        if not cmds.objExists(out_root):
            raise ValueError(f"Out skeleton not found: {out_root}")
        if ref_root and not cmds.objExists(ref_root):
            raise ValueError(f"Reference skeleton not found: {ref_root}")
        if anim_root and not cmds.objExists(anim_root):
            raise ValueError(f"Anim skeleton not found: {anim_root}")
        blend = cmds.createNode(cls.NATIVE_NODE_TYPE)

        # gather joints
        out_joints  = _get_joints(out_root, sort)
        ref_joints  = _get_joints(ref_root, sort, out_joints) if ref_root else []
        anim_joints = _get_joints(anim_root, sort, out_joints) if anim_root else []

        # make connections
        for i, out_joint in enumerate(out_joints):
            _connect_or_set(i, f"{blend}.referenceSkeleton", ref_joints, out_joint)
            _connect_or_set(i, f"{blend}.animSkeleton", anim_joints, out_joint)
            for at in ("translate", "rotate", "scale"):
                for ax in "XYZ":
                    cmds.connectAttr(
                        f"{blend}.outTransforms[{i}].{at}{ax}",
                        f"{out_joint}.{at}{ax}",
                        force=True,
                    )

            # because we are using local matrices, it doesnt consider joint orients
            # which can cause unwanted results.
            # a quick fix is to zero out the out joint orients.
            if zero_orients:
                cmds.setAttr(f"{out_joint}.jointOrient", 0, 0, 0)

        return cls(blend)

    @classmethod
    def from_output_skel(cls, out_root: Joint | str) -> SkeletonDeltaBlend | None:
        """Returns the skeleton delta blend node from a reference skeleton,
        or None if not found.

        Args:
            ref_root: Root joint of a reference skeleton.
        """
        joints = cmds.listRelatives(out_root, ad=True, type="joint", fullPath=True)
        if joints:
            joints.append(out_root)
        else:
            joints = [out_root]

        result = cmds.listConnections(
            joints, type=cls.NATIVE_NODE_TYPE, source=True, d=False, plugs=False
        )
        if result:
            return cls(result[0])

    # --- query

    @property
    def num_targets(self):
        """Returns the number of targets."""
        return self.blend.num_elements

    def get_target_name(self, i: int) -> str:
        """Returns a target name by its index."""
        return self.blend[i].alias

    def set_target_name(self, target: int, name: str) -> None:
        """Sets a target name.

        Args:
            target: The target index or name.
            name: The new target name.
        """
        if not isinstance(target, int):
            target = self.get_target_index(target)
        self.blend[target].alias = name

    def get_target_index(self, target_name: str) -> int:
        """Returns the index of a given target, or -1 if not found."""
        for i in self.blend.get_logical_indices():
            if self.get_target_name(i) == target_name:
                return i
        return -1

    def get_targets(self) -> list[str]:
        """Returns a list of target names in this node."""
        return cmds.listAttr(self.name, string="blend", multi=True)

    def get_reference_root(self) -> Joint:
        """Returns the reference skeleton's root joint."""
        roots = {
            x.get_root_joint()
            for x in self.referenceSkeleton.list_connections(
                source=True, destination=False, plugs=False
            )
        }
        return roots.pop()

    def get_output_skel_root(self) -> Joint:
        """Returns the output skeleton's root joint."""
        for joint in self.outTransforms.list_connections(
            type="joint", source=False, destination=True, plugs=False
        ):
            root = joint.get_root_joint()
            if root:
                return root

    def get_target_matrices(
        self,
        target:      int | str,
        world_space: bool      = True,
        key_as_str:  bool      = False,
    ) -> dict[Any, OpenMaya.MMatrix]:
        """Returns joint matrices of the driven skeleton at a given target.
        Using target index is faster.

        Args:
            target: The target index or name.
            world_space: If True, returns world space matrices.
                If False, return object space matrices.

        Returns:
            A dict of {output_joint: matrix} pairs.
        """
        if not isinstance(target, int):
            target = self.get_target_index(target)

        ref_attr    = self.referenceSkeleton
        out_attr    = self.outTransforms
        target_attr = self.blend[target].blendSkeleton
        dct         = {}
        for i in out_attr.get_logical_indices():
            # find the output joint
            attr     = out_attr[i].child(0)
            dst_attr = attr.get_connected_attrs(src=False, dst=True, first_only=True)
            if not dst_attr:
                raise RuntimeError(f"Output joint not connected at {attr.full_name}.")

            # find the associated target joint. If not connected, fallback the the
            # associated reference joint.
            attr     = target_attr[i]
            src_attr = attr.get_connected_attrs(src=True, dst=False, first_only=True)
            if not src_attr:
                attr = ref_attr[i]
                src_attr = attr.get_connected_attrs(
                    src=True, dst=False, first_only=True
                )
                if not src_attr:
                    raise RuntimeError(f"Ref joint not connected at {attr.full_name}.")
            tjoint = src_attr.node

            # get matrices
            key = dst_attr.node
            if key_as_str:
                key = key.short_name
            dct[key] = tjoint.get_matrix(world_space=world_space)
        return dct

    # --- target editing

    def _get_driver_attr(
        self, target_name: str, driver: str | DGNode | None = None
    ) -> Transform:
        # get or create driver
        driver = Transform(driver) if driver else self.get_output_skel_root()

        # add driver attr
        if not driver.has_attr(target_name):
            return driver.add_attr(
                target_name, at="double", keyable=True, min=0.0, max=1.0
            )
        else:
            return driver.find_attr(target_name)

    def add_target(
        self,
        target_root: str  | Joint,
        target_name: str  | None          = None,
        driver:      str  | DGNode | None = None,
        sort:        bool                 = False,
    ) -> None:
        """Adds a skeleton as a target to this blend skel node.

        Args:
            target_root: Target skeleton's root joint.
            target_name: Name of the target. Can not contain "_" or illegal characters.
                If None, use the root joint's name.
            driver: A driver node to add the driver attribute to.
            sort: If True, sort the target joints by long names.
        """
        target_root = Joint(target_root)
        target_name = target_name or target_root.short_name
        i           = self.blend.get_next_available_index()

        blend_attr       = self.blend[i]
        blend_attr.alias = target_name
        if driver:
            driver_attr = self._get_driver_attr(target_name, driver)
            driver_attr >> blend_attr.blendValue

        target_joints = target_root.get_children(allDescendents=True, type="joint")
        if sort:
            target_joints.sort(key=lambda x: x.long_name)

        driver_skel_attr = blend_attr.blendSkeleton
        for j, joint in enumerate(target_joints):
            joint.matrix >> driver_skel_attr[j]

    def remove_target(self, target: str | int) -> None:
        """Removes a target from this blend skel node.

        Args:
            target: The target index or name.
        """
        if not isinstance(target, int):
            target = self.get_target_index(target)
        blend_attr       = self.blend[target]
        driver_skel_attr = blend_attr.blendSkeleton

        # delete the driver skel
        root_joint = None
        for i in driver_skel_attr.get_logical_indices():
            joint = driver_skel_attr[i].list_connections(
                source=True, destination=False, plugs=False
            )
            if joint:
                root_joint = joint[0].get_root_joint()
                break
        if root_joint:
            root_joint.delete()

        # delete the driver attr
        driver_attr = blend_attr.blendValue.list_connections(
            source=True, destination=False, plugs=True
        )
        if driver_attr:
            node = driver_attr[0].node
            node.delete_attr(driver_attr[0])

        # delete the blend attr
        self.blend.delete_logical_index(target)

    def set_target_weight(self, target: str | int, weight: float) -> None:
        """Sets the weight of a target.

        Args:
            target: The target index or name.
            weight: The target weight.
        """
        if not isinstance(target, int):
            target = self.get_target_index(target)

        blend_value_attr = self.blend[target].blendValue

        # Check if the blendValue attribute is connected to a driver
        connected_attrs = blend_value_attr.list_connections(
            source=True, destination=False, plugs=True
        )

        if connected_attrs:
            # If connected, set the weight on the driver attribute instead
            driver_attr = connected_attrs[0]
            driver_attr.set(weight)
        else:
            # If not connected, set directly on the blendValue attribute
            blend_value_attr.set(weight)

    def empty_target_from_reference(
        self,
        target_name: str,
        driver:      str | DGNode | None = None,
    ) -> Joint:
        """Creates an empty target from the reference skeleton.

        Args:
            target_name: Name of the target. Can not contain "_" or illegal characters.
            driver: A driver node to add the driver attriute to.
        """
        ref_root    = self.get_reference_root()
        target_root = ref_root.duplicate_skeleton(suffix=target_name)
        self.add_target(target_root, target_name, driver)
        return target_root