"""
Transform node class
"""

from __future__ import annotations

import logging
import os
import re
from typing import Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig._internal.journal import cmds_edit, statement
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _files
from rig.nodetypes._base import _NODE_CLASS_DICT, _cast, _is_generic_slot, Attribute
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode, _typed_creator

LOGGER = logging.getLogger(__name__)


_HIERARCHY_LOAD_DOC = """Loads a hierarchy from a cgmath file (a HierarchyData): it CREATES the nodes.

    ``Transform.load_hierarchy(hierarchy, *, parent=None, world_space=True,
    container=None)`` (and ``Joint.load_hierarchy``, the same), from the class,
    makes the file's transforms, joints and locators, as `create_hierarchy`
    does, always (a second load of one file makes ``root1``)::

        skel = Joint.load_hierarchy("skeleton.npz")        # List: the new nodes, typed, parents first
        arm  = Transform.load_hierarchy("arm.npz", parent="rig_grp")

    Args:
        hierarchy: The file (a str or os.PathLike, ``~`` expanded; a cgmath
            ``.npz`` / ``.json`` / ``.pkl``), or the HierarchyData itself (a
            ClipData too: its frames are not used).
        parent: An existing node to make the roots under (`create_hierarchy`'s
            ``parent``). None: the world.
        world_space: With `parent`: True keeps the roots' world matrices from
            the file, False their local values (`create_hierarchy`'s).
        container: Inside ``with container()``, whether the new nodes join the
            scope (None: yes), as for every typed create.

    Returns:
        A List of the new nodes, typed (``Joint`` / ``Transform``), parents first:
        ``nodes[0]`` is the first root. Maya renames a node whose name is taken.

    The file is read, and what it holds checked, before anything is made: a
    missing file is a FileNotFoundError, a file of another class a TypeError
    naming what it holds, a file from an older cgmath cgmath's
    LegacyFileError. Constraints and other types derived from transform are
    skipped with a warning (`create_hierarchy`'s).

    From a node, ``Transform('root').load_hierarchy(...)`` is a TypeError: rig
    does not update a hierarchy in place (its transforms hold the rig's
    constraints, skins and children); ``Transform.load_hierarchy(path,
    parent='root')`` makes one under it.

    Each call is ONE undo step named ``rig.Transform.load_hierarchy`` (or
    ``rig.Joint.load_hierarchy``) and leaves the selection as it was.
    """

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


def _create_nested(node_type: str, **kwargs) -> DAGNode:
    """A node of ``node_type`` made by `Transform.create_hierarchy`, a typed
    create, so the create is nested: a registered type runs its class's typed
    create, any other type is ``cmds.createNode``, cast. It takes neither
    ``Node.create``'s scope rules (the outer create registers every node the
    call made, under the hierarchy's own names) nor its ``skipSelect`` default."""
    node_cls = _NODE_CLASS_DICT.get(node_type)
    if node_cls is not None:
        return node_cls.create(**kwargs)
    return _cast(cmds.createNode(node_type, **kwargs))


def _typed_nodes(names: list[str]) -> List:
    """The List of the typed nodes `names` name (`_build_hierarchy`'s, parents
    first): `create_hierarchy`'s and `load_hierarchy`'s result."""
    from rig._internal.list import List

    return List([_cast(name) for name in names])


def _set_parent_uuid(node: TransformData, uuid: str | None) -> None:
    """Sets the uuid of the parent of the TransformData `node` (None: a root):
    its ``parent_uuid`` since cgmath 2.0.0 (where ``parent_node`` is a
    deprecated alias that takes a uuid only), its ``parent_node`` before."""
    field = "parent_uuid" if hasattr(type(node), "parent_uuid") else "parent_node"
    setattr(node, field, uuid)


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
        dup.rename(name or type(self).get_short_name(self))
        dup.delete(ch=True)
        dup.set_xfrom_attrs_locked(False)

        # clean shapes
        nodes = [dup]
        for each in dup.get_children(children=True) or []:
            if each.has_base_type("transform"):
                cmds_edit.delete(each)
            elif each.find_attr("intermediateObject").get():
                cmds_edit.delete(each)
            else:
                nodes.append(each)

        if clean_sets:
            # remove geom from all display layers
            cmds_edit.editDisplayLayerMembers("defaultLayer", *nodes, noRecurse=True)
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
                        cmds_edit.disconnectAttr(src, dst)

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
            if node not in found and (not regex or regex(type(node).get_short_name(node))):
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
            if locked:
                type(attr).lock(attr)
            else:
                type(attr).unlock(attr)

    def freeze(self, *args, **kwargs) -> None:
        """Thin wrapper around cmds.makeIdentity() function."""
        kwargs["apply"] = kwargs.get("apply", kwargs.get("a", True))
        cmds_edit.makeIdentity(self, *args, **kwargs)

    def get_matrix(
        self, world_space: bool = True, as_transform_matrix: bool = False
    ) -> OpenMaya.MMatrix | OpenMaya.MTransformationMatrix:
        """Returns the matrix of this node.

        Args:
            world_space: If True return world space matrix, otherwise object space.
            as_transform_matrix: If True return MTransformationMatrix.
        """
        if world_space:
            attr = self.find_attr("worldMatrix")[0]
        else:
            attr = self.find_attr("matrix")
        mat = OpenMaya.MFnMatrixData(attr._mplug.asMObject()).matrix()
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

        cmds_edit.xform(
            str(self),
            worldSpace  = world_space,
            objectSpace = not world_space,
            matrix      = matrix,
        )

    def get_rotate_pivot(self, world_space: bool = True) -> OpenMaya.MPoint:
        """Returns the rotation pivot."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self).get_fn_set(self).rotatePivot(space)

    def get_scale_pivot(self, world_space: bool = True) -> OpenMaya.MPoint:
        """Returns the scale pivot."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return type(self).get_fn_set(self).scalePivot(space)

    def set_pivots(
        self, pivot: OpenMaya.MVector | list[float], world_space: bool = True
    ) -> None:
        """Sets both the scale and rotate pivots."""
        cmds_edit.xform(str(self), pivots=pivot, worldSpace=world_space)

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

        node   = TransformData(name=type(self).get_short_name(self), uuid=type(self).get_uuid(self))
        parent = self.get_parent()
        if parent is not None:
            _set_parent_uuid(node, type(parent).get_uuid(parent))

        # if there is a locatorShape, set this type as "locator"
        node.node_type = cmds.nodeType(str(self))

        if node.node_type == "joint":
            node.joint_orient             = np.array(cmds.getAttr(f"{node}.jo")[0])
            node.segment_scale_compensate = bool(cmds.getAttr(f"{node}.ssc"))
            node.rotate_axis              = np.array(self.find_attr("ra").get()[0])
            node.radius                   = self.find_attr("radius").get()

        else:
            locators = cmds.listRelatives(
                str(self), c=True, ni=True, f=False, type="locator"
            )

            # the shape's localScale/localPosition are not carried, only the kind
            if locators:
                node.node_type = "locator"

        node.scale        = np.array(self.find_attr("s").get()[0])
        node.rotate       = np.array(self.find_attr("r").get()[0])
        node.translate    = np.array(self.find_attr("t").get()[0])
        node.rotate_order = self.find_attr("ro").get()
        node.rotate_axis  = np.array(self.find_attr("ra").get()[0])
        node.visibility   = self.find_attr("v").get()

        # capture user defined attrs
        attrs = self.list_attr(ud=True)
        if attrs:
            node.user_defined_attributes = {}
            for att in attrs:
                try:
                    node.user_defined_attributes[type(att).get_name(att)] = self._serialize_user_attr(att)

                # gracfully pass on this one
                except Exception as e:
                    LOGGER.warning(f"Skipped user defined attr {att}: {e}")

        return node

    @staticmethod
    def _serialize_user_attr(att: Attribute) -> dict:
        """Returns the cmds.addAttr() spec of a user defined attr, with its value,
        keyable and channel box states. Compound parents carry no value, their
        children do. Multi values are [logical index, value] pairs.
        """
        data = {}

        # a typed attr that holds any data (a Typed spec's): addAttr(attributeType=
        # "typed"), and as its value only the numbers it holds (what a connection
        # fed it is not saved: serialization carries no connections)
        if _is_generic_slot(att):
            data["attributeType"] = "typed"
            if type(att).is_multi(att):
                data["multi"] = True
                data["value"] = [
                    [int(i), att[i].get()]
                    for i in att.get_logical_indices()
                    if cmds.getAttr(str(att[i]), type=True) == "typed"
                ]
            elif cmds.getAttr(str(att), type=True) == "typed":
                data["value"] = att.get()
            parent = att.get_parent()
            if parent is not None:
                data["parent"] = type(parent).get_name(parent)
            data["keyable"]     = type(att).is_keyable(att)
            data["channel_box"] = type(att).is_channel_box(att)
            return data

        # a multi's own plug reports "compound", addAttr() wants the element
        # type. it is read off the attribute, a getAttr on an element would
        # create it.
        if type(att).is_multi(att):
            data["multi"] = True
            if type(att).is_typed(att):
                data_type = OpenMaya.MFnTypedAttribute(type(att).get_mobject(att)).attrType()
                if data_type not in TYPED_DATA_NAMES:
                    raise ValueError(f"Unsupported multi data type: {data_type}")
                data["dataType"] = TYPED_DATA_NAMES[data_type]
            else:
                data["attributeType"] = type(att).get_attribute_type(att)
            data["value"] = [[int(i), att[i].get()] for i in att.get_logical_indices()]

        else:
            # typed attrs (string, stringArray, doubleArray, etc.)
            # are created with cmds.addAttr(dataType=...)
            if type(att).is_typed(att):
                data["dataType"] = type(att).get_data_type(att)
            else:
                data["attributeType"] = type(att).get_data_type(att)

            if att._mplug.isCompound:
                data["numberOfChildren"] = type(att).get_num_children(att)
            else:
                data["value"] = att.get()

        if data.get("attributeType") == "enum":
            data["enumName"] = ":".join(type(att).get_enums(att))

        parent = att.get_parent()
        if parent is not None:
            data["parent"] = type(parent).get_name(parent)

        data["keyable"]     = type(att).is_keyable(att)
        data["channel_box"] = type(att).is_channel_box(att)
        return data

    def serialize_hierarchy(self, bindpose: bool | str | DGNode = False) -> HierarchyData:
        """Serialize a whole hierarchy: this node and the transforms and joints
        below it, parents first.

        Args:
            bindpose: ``False`` (the default) reads the scene as it is.
                ``True`` reads the hierarchy at its bind pose: every dagPose
                with ``bindPose`` on that holds one of its nodes (a skinCluster
                makes one; binding another influence set makes another). Poses
                that hold the same node must agree on it.
                A dagPose name (``"bindPose5"``, ``"char:bindPose1"``) or node
                (``Node("bindPose5")``) reads it at that one pose; any dagPose
                qualifies, a saved ``tPose`` too.

        The result is what ``serialize_hierarchy()`` would return after Maya's
        own Go to Bind Pose (``dagPose -restore -global``), and nothing is
        restored: the scene, the selection and the undo queue are unchanged.
        A pose member takes its scale, rotate, translate, rotate axis and joint
        orient from the pose, its rotation re-expressed in the node's current
        rotate order (as the restore does); everything else (segment scale
        compensate, radius, visibility, user attributes, names, parents) and
        every node the pose does not hold (an end joint that is no influence,
        a helper added after the bind) keep the scene's values. At the bind
        pose the result is bit-identical to ``bindpose=False``.

        Raises (every check runs before anything is read into the result):
            TypeError: ``bindpose`` is not False, True, a str or a node.
            NodeNotFoundError / AmbiguousNodeError: the name names no node,
                or several.
            NodeTypeError: the name is not a dagPose.
            ValueError: ``True`` and no bind pose holds a node of the
                hierarchy; the named pose holds none of its nodes; ``True``
                and the bind poses disagree on a node (name one); a member's
                DAG parent is not the parent it was posed under (a member is
                stored relative to it: restore the parent or save a new pose).

        Returns:
            A HierarchyData object.
        """
        from cgmath.hierarchy import HierarchyData

        # the argument is checked before the walk; bool before int (True is an int)
        if bindpose is not False:
            if isinstance(bindpose, bool):
                pass
            elif isinstance(bindpose, DGNode):
                bindpose = str(bindpose)
            elif not isinstance(bindpose, str) or isinstance(bindpose, Attribute):
                raise TypeError(
                    f"bindpose= takes False, True or a dagPose name, not {bindpose!r}"
                )

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
                if cmds.nodeType(str(child)) not in SERIALIZED_NODE_TYPES:
                    below = cmds.listRelatives(
                        str(child), ad=True, type="transform", f=True
                    )
                    for name in [str(child)] + (below or []):
                        node_type          = cmds.nodeType(name)
                        skipped[node_type] = skipped.get(node_type, 0) + 1
                    continue

                tree.append(child.serialize())
                objs.append(type(child).get_mobject(child))
                _recurse(child)

        # serialize the root
        root = self.serialize()
        _set_parent_uuid(root, None)

        # recurse through the hierarchy
        tree    = [root]
        objs    = [type(self).get_mobject(self)]  # the serialized nodes' MObjects, for the bind pose
        skipped = {}
        _recurse(self)

        if bindpose is not False:
            from rig.nodetypes import _bindpose

            for index, entry in _bindpose.resolve(type(self).get_short_name(self), objs, bindpose).items():
                _bindpose.apply(tree[index], entry, objs[index])

        if skipped:
            counts = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
            LOGGER.warning(f"Skipped nodes that can't be serialized: {counts}")

        return HierarchyData(tree)

    @classmethod
    @_typed_creator
    def create_hierarchy(
        cls,
        hierarchy:  HierarchyData,
        parent:     Union[str, None] = None,
        world_space                  = True,
    ) -> List:
        """Creates a hierarchy from a HierarchyData object.

        Args:
            hierarchy: A HierarchyData object (`serialize_hierarchy`); it is
                left as it was.
            parent: An existing node to make the roots under. None: the world.
            world_space: With `parent`: True keeps the roots' world matrices
                from the data, False their local values.

        Returns:
            A List of the new nodes, typed as ``Transform.create`` types its
            node (``Joint`` for a joint, ``Transform`` for a transform, a
            locator's transform and a space_transform box), parents first:
            ``nodes[0]`` is the first root. Maya renames a node whose name is
            taken (``root1``)::

                skel = Transform.create_hierarchy(Node("root").serialize_hierarchy())
                skel[1].tx.get()

        Inside ``with container()`` every node the call made joins the scope
        (``container=False`` leaves them unregistered); the nodes keep the
        hierarchy's names (the creates inside this one are nested, so they add
        no prefix).

        One undo step; it selects the nodes it made. `load_hierarchy` (from a
        file or the data, the selection kept) returns the same List.
        """
        with _undo_chunk(f"rig.{cls.__name__}.create_hierarchy"):
            return _typed_nodes(cls._build_hierarchy(hierarchy, parent, world_space, select=True))

    @classmethod
    @_typed_creator
    def _load_built(
        cls,
        hierarchy:   HierarchyData,
        parent:      Union[str, None] = None,
        world_space: bool             = True,
    ) -> list:
        """[Internal] `load_hierarchy`'s create, a typed creator (so it joins the
        container scope as `create_hierarchy` does): the new nodes, typed,
        parents first; the selection is put back as it was."""
        return _typed_nodes(cls._build_hierarchy(hierarchy, parent, world_space, select=False))

    @classmethod
    def _build_hierarchy(
        cls,
        hierarchy:   HierarchyData,
        parent:      Union[str, None],
        world_space: bool,
        select:      bool,
    ) -> list[str]:
        """[Internal] The body of `create_hierarchy` and `load_hierarchy`: makes
        the nodes of `hierarchy` and returns their names, parents first. With
        `select` they end selected; without, the selection is put back as it was
        before (in its order)."""
        from cgmath.hierarchy import SUPPORTED_NODE_TYPES, HierarchyData

        before = None if select else (cmds.ls(orderedSelection=True, long=True) or [])

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
        under = {}  # id of a root -> the long name of the node outside the data it goes under
        if parent is not None:
            if parent in hierarchy:
                for node in hierarchy.get_roots():
                    node.set_parent(parent, world_space=world_space)

            # if the parent is not in the hierarchy, add it to compute matrices
            else:
                # (the data keeps the outside node's uuid: a parent is a uuid since
                # cgmath 2.0.0, so the name it is created under is kept here)
                # its own parent is not in the data, so it stands in as a root that
                # carries its world matrix: the roots are solved against that
                outside = _cast(parent)
                obj     = outside.serialize()
                _set_parent_uuid(obj, None)
                hierarchy.list.insert(0, obj)
                obj.matrix = cmds.xform(type(outside).get_long_name(outside), q=True, matrix=True, worldSpace=True)
                for node in hierarchy.get_roots()[1:]:
                    node.set_parent(obj, world_space=world_space)
                    under[id(node)] = type(outside).get_long_name(outside)
                hierarchy.list.pop(0)

        # create the hierarchy at identity before setting any attributes,
        # this is to avoid any default attrs being set by Maya.
        for node in hierarchy:
            # use the parent's long name, the new node is created in world
            # and may shadow the parent's short name.
            parent = node.get_parent()
            if id(node) in under:
                parent = under[id(node)]
            elif parent:
                # hierarchy data by its name, which holds its node once made (a
                # node's own name would be a Maya attr lookup); a node by its
                # long name
                if not isinstance(parent, DAGNode):
                    parent = getattr(parent, "name", parent)
                parent = type(parent).get_long_name(parent) if isinstance(parent, DAGNode) else str(parent)

            # create node
            if node.node_type != "locator":
                if node.node_type == "space_transform":
                    """creates a RTR space_transform box """
                    name = node.name
                    obj  = _cast(cmds_edit.polyCube(name=name, w=100, h=100, d=100, ch=False)[0])
                    if parent is not None:
                        cmds_edit.parent(type(obj).get_long_name(obj), parent)
                else:
                    obj = _create_nested(node.node_type, name=node, parent=parent)

            else:
                obj = _cast(cmds_edit.spaceLocator(name=node.name)[0])
                if parent is not None:
                    cmds_edit.parent(type(obj).get_long_name(obj), parent)

            # rename the node's string name with the node object
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
        if select:
            cmds_edit.select(created)
        elif before:
            cmds_edit.select(before[:-1], replace=True)
            cmds_edit.select(before[-1], add=True)
        elif cmds.ls(selection=True):
            cmds_edit.select(clear=True)
        return created

    # --- files

    def save_hierarchy(self, path: str | os.PathLike, *, bindpose: bool | str | DGNode = False) -> str:
        """Saves this node's hierarchy to a cgmath file: its HierarchyData
        (`serialize_hierarchy`), the inverse of `load_hierarchy`.

        Args:
            path: A str or os.PathLike (``~`` expanded). The extension sets the
                format: ``.npz`` (recommended), ``.json`` or ``.pkl`` (never
                load a ``.pkl`` you did not write).
            bindpose: as `serialize_hierarchy` takes it: False (the scene as it
                is), True (at its bind pose(s)), or a dagPose's name or node.

        Returns:
            `path`, as written (``~`` expanded).

        A read of the scene: no undo step, the selection unchanged. The path
        and `bindpose` are checked before anything is serialized (an extension
        cgmath does not save: ValueError; a missing folder: FileNotFoundError;
        a wrong `bindpose`: `serialize_hierarchy`'s TypeError).
        """
        path = _files.check_save_path(path, f"{type(self).__name__}.save_hierarchy")
        self.serialize_hierarchy(bindpose=bindpose).save(path)
        return path

    @statement
    def _load_hierarchy_new(
        cls,
        hierarchy,
        *,
        parent:      Union[str, DAGNode, None] = None,
        world_space: bool                      = True,
        container:   bool | None               = None,
    ):
        from cgmath.hierarchy import HierarchyData

        method = f"{cls.__name__}.load_hierarchy"
        data = _files.read(hierarchy, (HierarchyData,), method, "the hierarchy")
        kwargs = {} if container is None else {"container": container}
        if isinstance(parent, DAGNode):
            parent = type(parent).get_long_name(parent)
        with _undo_chunk(f"rig.{cls.__name__}.load_hierarchy"):
            return cls._load_built(data, parent, world_space, **kwargs)

    def _load_hierarchy_refused(self, *args, **kwargs):
        name = type(self).__name__
        raise TypeError(
            f"{name}({str(self)!r}).load_hierarchy(...) would update the hierarchy at {str(self)!r} "
            f"in place, which rig does not do (its transforms hold the rig's constraints, skins "
            f"and children); {name}.load_hierarchy(path, parent={str(self)!r}) makes a new one "
            f"under it"
        )

    load_hierarchy = _files.creates_or_updates(_load_hierarchy_new, _load_hierarchy_refused, _HIERARCHY_LOAD_DOC)
    del _load_hierarchy_new, _load_hierarchy_refused

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
                    type(att).set_channel_box(att, True)

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