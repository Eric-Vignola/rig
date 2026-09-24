"""
Transform node class
"""

from __future__ import annotations

import logging
import re
from typing import Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes._base import Attribute
from rig.nodetypes.dag_node import DAGNode, PyNode

LOGGER = logging.getLogger(__name__)

# the exact Maya node types serialize_hierarchy() captures,
# a locator is a "transform" holding a locator shape.
SERIALIZED_NODE_TYPES = ("transform", "joint")

# cmds.addAttr(dataType=...) names of the typed attribute data kinds
TYPED_DATA_NAMES = {
    OpenMaya.MFnData.kString:      "string",
    OpenMaya.MFnData.kStringArray: "stringArray",
    OpenMaya.MFnData.kDoubleArray: "doubleArray",
    OpenMaya.MFnData.kFloatArray:  "floatArray",
    OpenMaya.MFnData.kIntArray:    "Int32Array",
    OpenMaya.MFnData.kVectorArray: "vectorArray",
    OpenMaya.MFnData.kPointArray:  "pointArray",
    OpenMaya.MFnData.kMatrix:      "matrix",
}

# array data types older serialized hierarchies stored under "attributeType",
# cmds.addAttr() only accepts them as "dataType".
# "matrix" is left out, it is valid as both.
LEGACY_DATA_TYPES = (
    "stringArray",
    "doubleArray",
    "floatArray",
    "Int32Array",
    "vectorArray",
    "pointArray",
)


class Transform(DAGNode):
    """
    Transform node class
    """

    NATIVE_NODE_TYPE = "transform"
    FN_SET           = OpenMaya.MFnTransform

    # --- creation

    def duplicate_geometry(
        self,
        name:        str  | None             = None,
        parent:      str  | Transform | None = None,
        clean_sets:  bool                    = True,
        clean_xform: bool                    = True,
    ) -> Transform:
        """Duplicates the geometry under this node.

        Args:
            name: Name for the duplicated geometry.
                If None, use the input geometry name.
            parent: Parent of the duplicated geometry.
                If None, parent to world.
            clean_sets: If True, remove the duplicate from display layers and sets.
            clean_xform: If True, cleans xform by moving pivots
                to origin and clean xform channels.

        Returns:
            The duplicated geometry name.
        """
        dup = self.duplicate(parentOnly=False)[0]
        dup.set_parent(parent or None)
        dup.rename(name or self.short_name)
        dup.delete(ch=True)
        dup.set_xfrom_attrs_locked(False)

        # clean shapes
        nodes = [dup]
        for each in dup.get_children(children=True) or []:
            if each.has_base_type("transform"):
                cmds.delete(each)
            elif each.intermediateObject.get():
                cmds.delete(each)
            else:
                nodes.append(each)

        if clean_sets:
            # remove geom from all display layers
            cmds.editDisplayLayerMembers("defaultLayer", *nodes, noRecurse=True)
            # remove geom from all object sets
            for node in nodes:
                attrs = cmds.listConnections(
                    node,
                    source      = False,
                    destination = True,
                    connections = True,
                    plugs       = True,
                    type        = "objectSet",
                    exactType   = True,
                )
                if attrs:
                    for src, dst in zip(*[iter(attrs)] * 2):
                        cmds.disconnectAttr(src, dst)

        # clean xform
        if clean_xform:
            dup.freeze(t=True, r=True, s=True, a=True)
            dup.set_pivots((0, 0, 0))

        return dup

    # --- hierarchy

    def _search_regex(self, match_name, exact_match, _regex):
        # construct search regex, reused in recursions
        if _regex:
            return _regex
        elif match_name:
            names = [match_name] if isinstance(match_name, str) else match_name
            if exact_match:
                return re.compile(r"|".join([rf"^{x}$" for x in names])).match
            else:
                return re.compile(r"|".join([rf"{x}" for x in names])).search

    def iter_shapes(
        self,
        shape_type:   str       | list[str],
        match_name:   list[str] | str | None = None,
        exact_match:  bool                   = False,
        as_transform: bool                   = True,
        _regex:       str       | None       = None,
    ) -> "DAGNode":
        """A generator that traverse the hierarchy of this node and yields shapes
        nodes.

        Args:
            shape_type: The shape type.
            match_name: One or more of names to match, can be regex.
            exact_match: If True, match exact names, otherwise do partial match.
            as_transform: If True, returns the transform node instead of shape node.

        Yields:
            Transform nodes that meet criteria.
        """
        regex = self._search_regex(match_name, exact_match, _regex)
        found = set()
        for node in reversed(
            self.get_children(allDescendents=True, noIntermediate=True, type=shape_type)
        ):
            if as_transform:
                node = node.get_parent()
            if node not in found and (not regex or regex(node.short_name)):
                yield node

    def find_shape(self, shape_type: str, match_name: str, **kwargs) -> DAGNode:
        """Returns the first shape transform matching given criteria,
        or None if no match.

        Args:
            shape_type: The shape type.
            match_name: One or more of names to match, can be regex.
            kwargs: Keyword arguments passed to `iter_shape_xforms` method.
        """
        kwargs["match_name"] = match_name
        it                   = self.iter_shapes(shape_type, **kwargs)
        return next(it, None)

    def get_shapes(self, no_interm: bool = True) -> list[DAGNode]:
        """Returns the shape nodes under this transform.

        Args:
            no_interm: If True, skip intermediate objects.
        """
        return self.get_children(children=False, shapes=True, noIntermediate=no_interm)

    def get_shape(self) -> DAGNode | None:
        """Returns the first non-intermidiate shape node under this transform."""
        shapes = self.get_shapes(no_interm=True)
        if shapes:
            return shapes[0]

    # --- transformation

    def iter_xform_attrs(self, axis_only: bool = False) -> Attribute:
        """Iterates over all transform attrs on this node.

        Args:
            axis_only: If True, only return attrs per axis.
                Otherwise also return their parent attrs: "t", "r", "s".

        Yields:
            Transform attributes.
        """
        for at in "trs":
            if not axis_only:
                yield self.find_attr(at)
            for attr in (at + ax for ax in "xyz"):
                yield self.find_attr(attr)

    def set_xfrom_attrs_locked(self, locked: bool) -> None:
        """Locks or Unlocks all transform attributes on this node."""
        for attr in self.iter_xform_attrs():
            attr.is_locked = locked

    def freeze(self, *args, **kwargs) -> None:
        """Thin wrapper around cmds.makeIdentity() function."""
        kwargs["apply"] = kwargs.get("apply", kwargs.get("a", True))
        cmds.makeIdentity(self, *args, **kwargs)

    def get_matrix(
        self, world_space: bool = True, as_transform_matrix: bool = False
    ) -> OpenMaya.MMatrix | OpenMaya.MTransformationMatrix:
        """Returns the matrix of this node.

        Args:
            world_space: If True return world space matrix, otherwise object space.
            as_transform_matrix: If True return MTransformationMatrix.
        """
        if world_space:
            attr = self.worldMatrix[0]
        else:
            attr = self.matrix
        mat = OpenMaya.MFnMatrixData(attr.plug.asMObject()).matrix()
        if as_transform_matrix:
            return OpenMaya.MTransformationMatrix(mat)
        return mat

    def set_matrix(
        self,
        matrix:      OpenMaya.MMatrix | OpenMaya.MTransformationMatrix | list[float],
        world_space: bool                                                            = True,
    ) -> None:
        """Sets the matrix of this node.

        Args:
            matrix: The matrix to set.
            world_space: If True set world space matrix, otherwise object space.
        """
        if isinstance(matrix, OpenMaya.MTransformationMatrix):
            matrix = matrix.asMatrix()

        cmds.xform(
            self.name,
            worldSpace  = world_space,
            objectSpace = not world_space,
            matrix      = matrix,
        )

    def get_rotate_pivot(self, world_space: bool = True) -> OpenMaya.MPoint:
        """Returns the rotation pivot."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.rotatePivot(space)

    def get_scale_pivot(self, world_space: bool = True) -> OpenMaya.MPoint:
        """Returns the scale pivot."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.scalePivot(space)

    def set_pivots(
        self, pivot: OpenMaya.MVector | list[float], world_space: bool = True
    ) -> None:
        """Sets both the scale and rotate pivots."""
        cmds.xform(self.name, pivots=pivot, worldSpace=world_space)

    def match_matrix(
        self, target_transform: Transform, world_space: bool = False
    ) -> None:
        """matches the matrix of another transform node"""
        matrix = target_transform.get_matrix(world_space=world_space)
        self.set_matrix(matrix, world_space=world_space)

    # --- serialization

    def serialize(self) -> TransformData:
        """creates a TransformData object from a given string"""
        from cgmath.hierarchy import TransformData

        node   = TransformData(name=self.short_name, uuid=self.uuid)
        parent = self.get_parent()
        if parent is not None:
            node.parent_node = parent.uuid

        # if there is a locatorShape, set this type as "locator"
        node.node_type = cmds.nodeType(self.name)

        if node.node_type == "joint":
            node.joint_orient             = np.array(cmds.getAttr(f"{node}.jo")[0])
            node.segment_scale_compensate = bool(cmds.getAttr(f"{node}.ssc"))
            node.rotate_axis              = np.array(self.ra.get()[0])
            node.radius                   = self.radius.get()

        else:
            locators = cmds.listRelatives(
                self.name, c=True, ni=True, f=False, type="locator"
            )

            # the shape's localScale/localPosition are not carried, only the kind
            if locators:
                node.node_type = "locator"

        node.scale        = np.array(self.s.get()[0])
        node.rotate       = np.array(self.r.get()[0])
        node.translate    = np.array(self.t.get()[0])
        node.rotate_order = self.ro.get()
        node.rotate_axis  = np.array(self.ra.get()[0])
        node.visibility   = self.v.get()

        # capture user defined attrs
        attrs = self.list_attr(ud=True)
        if attrs:
            node.user_defined_attributes = {}
            for att in attrs:
                try:
                    node.user_defined_attributes[att.name] = self._serialize_user_attr(att)

                # gracfully pass on this one
                except Exception as e:
                    LOGGER.warning(f"Skipped user defined attr {att.full_name}: {e}")

        return node

    @staticmethod
    def _serialize_user_attr(att: Attribute) -> dict:
        """Returns the cmds.addAttr() spec of a user defined attr, with its value,
        keyable and channel box states. Compound parents carry no value, their
        children do. Multi values are [logical index, value] pairs.
        """
        data = {}

        # a multi's own plug reports "compound", addAttr() wants the element
        # type. it is read off the attribute, a getAttr on an element would
        # create it.
        if att.is_multi:
            data["multi"] = True
            if att.is_typed:
                data_type = OpenMaya.MFnTypedAttribute(att.mobject).attrType()
                if data_type not in TYPED_DATA_NAMES:
                    raise ValueError(f"Unsupported multi data type: {data_type}")
                data["dataType"] = TYPED_DATA_NAMES[data_type]
            else:
                data["attributeType"] = att.attribute_type
            data["value"] = [[int(i), att[i].get()] for i in att.get_logical_indices()]

        else:
            # typed attrs (string, stringArray, doubleArray, etc.)
            # are created with cmds.addAttr(dataType=...)
            if att.is_typed:
                data["dataType"] = att.data_type
            else:
                data["attributeType"] = att.data_type

            if att.plug.isCompound:
                data["numberOfChildren"] = att.num_children
            else:
                data["value"] = att.get()

        if data.get("attributeType") == "enum":
            data["enumName"] = ":".join(att.enums)

        parent = att.get_parent()
        if parent is not None:
            data["parent"] = parent.name

        data["keyable"]     = att.is_keyable
        data["channel_box"] = att.is_channel_box
        return data

    def serialize_hierarchy(self) -> HierarchyData:
        """Serialize a whole hierarchy.

        Returns:
            A HierarchyData object.
        """
        from cgmath.hierarchy import HierarchyData

        def _recurse(leaf):
            # this function returns a hierarchy the way it should be
            # unlike listRelatives(ad=True) which returns some really weird order
            children = (
                leaf.get_children(c=True, noIntermediate=True, type="transform") or []
            )

            for child in children:
                # type="transform" also lists every type derived from it
                # (constraints, ikHandles, etc.), a TransformData can't describe
                # those, skip them along with what is under them.
                if cmds.nodeType(child.name) not in SERIALIZED_NODE_TYPES:
                    below = cmds.listRelatives(
                        child.name, ad=True, type="transform", f=True
                    )
                    for name in [child.name] + (below or []):
                        node_type          = cmds.nodeType(name)
                        skipped[node_type] = skipped.get(node_type, 0) + 1
                    continue

                tree.append(child.serialize())
                _recurse(child)

        # serialize the root
        root             = self.serialize()
        root.parent_node = None

        # recurse through the hierarchy
        tree    = [root]
        skipped = {}
        _recurse(self)

        if skipped:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"Skipped nodes that can't be serialized: {counts}")

        return HierarchyData(tree)

    @classmethod
    def create_hierarchy(
        cls,
        hierarchy:  HierarchyData,
        parent:     Union[str, None] = None,
        world_space                  = True,
    ) -> HierarchyData:
        """Creates a hierarchy from a HierarchyData object.

        Args:
            hierarchy: A HierarchyData object.

        Returns:
            A Skeleton object with updated node names (in case of duplication)
        """
        from cgmath.hierarchy import SUPPORTED_NODE_TYPES, HierarchyData

        # make a copy of the hierarchy to handle new names
        # without affecting source data and set the desired parent.
        hierarchy = hierarchy.copy()

        # drop the nodes that can't be rebuilt from this data (constraints, ikHandles,
        # etc. found in older files) along with what is under them.
        # parents come first, same as the creation loop below expects.
        skipped = {}
        dropped = set()
        for node in hierarchy:
            if node.node_type in SUPPORTED_NODE_TYPES:
                if id(node.get_parent()) not in dropped:
                    continue
            dropped.add(id(node))
            skipped[node.node_type] = skipped.get(node.node_type, 0) + 1

        if skipped:
            hierarchy.list[:] = [x for x in hierarchy.list if id(x) not in dropped]
            counts            = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"Skipped nodes that can't be created: {counts}")

        # if a parent is specified, make set the transforms relative to it
        if parent is not None:
            if parent in hierarchy:
                for node in hierarchy.get_roots():
                    node.set_parent(parent, world_space=world_space)

            # if the parent is not in the hierarchy, add it to compute matrices
            else:
                obj = PyNode(parent).serialize()
                hierarchy.list.insert(0, obj)
                for node in hierarchy.get_roots()[1:]:
                    node.set_parent(obj, world_space=world_space)
                    node.parent_node = parent
                hierarchy.list.pop(0)

        # create the hierarchy at identity before setting any attributes,
        # this is to avoid any default attrs being set by Maya.
        for node in hierarchy:
            # use the parent's long name, the new node is created in world
            # and may shadow the parent's short name.
            parent = node.get_parent()
            if parent:
                parent = getattr(parent, "name", parent)
                parent = parent.long_name if isinstance(parent, DAGNode) else str(parent)

            # create node
            if node.node_type != "locator":
                if node.node_type == "space_transform":
                    """creates a RTR space_transform box """
                    name = node.name
                    obj  = PyNode(cmds.polyCube(name=name, w=100, h=100, d=100, ch=False)[0])
                    if parent is not None:
                        cmds.parent(obj.long_name, parent)
                else:
                    obj = PyNode.create(node.node_type, name=node, parent=parent)

            else:
                obj = PyNode(cmds.spaceLocator(name=node.name)[0])
                if parent is not None:
                    cmds.parent(obj.long_name, parent)

            # rename the node's string name wiht the PyNode object
            # in case of scene duplicate during parenting.
            node.name = obj

        # set attributes after hierarchy creation.
        for i, node in enumerate(hierarchy):
            node = node.name
            node.set_attrs(skip_missing=True, **hierarchy[i].to_attributes())

        # add user defined attrs once every transform is set,
        # so a bad attr can never leave the hierarchy at identity.
        for i, node in enumerate(hierarchy):
            node  = node.name
            specs = hierarchy[i].user_defined_attributes
            if specs:
                cls._create_user_attrs(node, specs)

        # return the created nodes
        created = [str(x.name) for x in hierarchy]
        cmds.select(created)
        return created

    @staticmethod
    def _create_user_attrs(node: DAGNode, specs: dict) -> None:
        """Adds the user defined attrs of a serialized node and sets their values.

        Every attr is added first, in file order: a compound only exists once
        its children, which follow it, are added. Values are set second, on
        the attrs that exist. A failure is logged and never stops the rest.
        """
        # pass 1: add
        failed = set()
        for name, attr in specs.items():
            attr = {k: v for k, v in attr.items() if k not in ("value", "channel_box")}

            # older files stored array data types as "attributeType"
            if attr.get("attributeType") in LEGACY_DATA_TYPES:
                attr["dataType"] = attr.pop("attributeType")

            try:
                node.add_attr(name, **attr)
            except Exception as e:
                failed.add(name)
                LOGGER.warning(f"Skipped user defined attr {node}.{name}: {e}")

        # pass 2: channel box and values
        for name, attr in specs.items():
            if name in failed:
                continue

            att = node.find_attr(name, quiet=True)
            if att is None:
                LOGGER.warning(
                    f"User defined attr {node}.{name} was not created, "
                    "a compound parent is missing children"
                )
                continue

            try:
                if attr.get("channel_box"):
                    att.is_channel_box = True

                value = attr.get("value")
                if value is None:
                    continue

                # a multi's value is [logical index, value] pairs
                if attr.get("multi"):
                    for index, element in value:
                        att[index].set(element)
                else:
                    att.set(value)

            except Exception as e:
                LOGGER.warning(f"Skipped user defined attr {node}.{name} value: {e}")