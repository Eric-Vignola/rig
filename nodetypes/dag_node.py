from __future__ import annotations

from maya import cmds, mel
from maya.api import OpenMaya
from rig.nodetypes._base import get_custom_type, PyNode
from rig.nodetypes.dg_node import DGNode


def _parent_valid(mobject) -> bool:
    """Check if an MObject is a world object."""
    if mobject.isNull() or mobject.apiType() in (
        OpenMaya.MFn.kWorld,
        OpenMaya.MFn.kInvalid,
        OpenMaya.MFn.kUnknown,
    ):
        return False
    return True


class DAGNode(DGNode):
    """
    Base class for DAG nodes.
    """

    NATIVE_NODE_TYPE = "dagNode"
    FN_SET           = OpenMaya.MFnDagNode

    def __init__(self, node: str | OpenMaya.MObject | OpenMaya.MDagPath) -> None:
        """Initialize an instance from a node name or a MObject."""
        if isinstance(node, DAGNode):
            self._mobject    = node._mobject
            self._mdagpath   = node._mdagpath
            self._fn_set     = node._fn_set
            self._fn_set1    = node._fn_set1
            self._objhandle1 = node._objhandle1
        else:
            if isinstance(node, OpenMaya.MObject):
                self._mobject  = node
                self._mdagpath = OpenMaya.MDagPath.getAPathTo(node)
            elif isinstance(node, OpenMaya.MDagPath):
                self._mdagpath = node
                self._mobject  = self._mdagpath.node()
            else:
                sel = OpenMaya.MSelectionList()
                try:
                    sel.add(str(node))
                except Exception:
                    raise ValueError(f"Invalid node name: {node}")
                self._mdagpath = sel.getDagPath(0)
                self._mobject  = self._mdagpath.node()
            self._fn_set = self.FN_SET(self._mdagpath)
            self._cache_api1_objects(self._fn_set.partialPathName())
        self.is_type(self.name, exact_type=False, failfast=True)
        self._attr_dict = {}  # cache queried attributes

    # --- creation

    @classmethod
    def _create(cls, parent: str | DAGNode | None = None, *args, **kwargs) -> str:
        """[Internal] Creates a node of this type.
        This class can only use Maya APIs and must return a node name string.
        """
        # resolve the parent to its long name before creating anything,
        # the new node is created in world and may shadow the parent's short name.
        if parent:
            if isinstance(parent, DAGNode):
                parent = parent.long_name
            else:
                found = cmds.ls(str(parent), long=True)
                if len(found) != 1:
                    raise ValueError(f"Parent must match one node: {parent} -> {found}")
                parent = found[0]

        node = cmds.createNode(cls.NATIVE_NODE_TYPE, **kwargs)
        sel  = OpenMaya.MSelectionList()
        sel.add(node)
        mdagpath = sel.getDagPath(0)
        if parent:
            cmds.parent(mdagpath.fullPathName(), parent)
        return mdagpath.partialPathName()

    # --- properties and utils

    @property
    def mdagpath(self) -> str:
        """Returns the MDagpath object."""
        self.ensure_valid()
        return self._mdagpath

    @property
    def name(self) -> str:
        """Returns the shortest unique name."""
        return self.fn_set.partialPathName()

    @property
    def long_name(self) -> str:
        """Returns the long name."""
        return self.fn_set.fullPathName()

    @property
    def is_shape(self) -> bool:
        """Returns whether this node is a shape node."""
        return "shape" in cmds.nodeType(self.long_name, inherited=True)

    @classmethod
    def is_type(
        cls, node_name, exact_type: bool = True, failfast: bool = False
    ) -> bool:
        """Checks if a node is a valid node of this type.

        Args:
            node_name: A node name to check.
            exact_type: If True, check if node is exactly this type.
                Otherwise allow inherited types as well.
            failfast: If True, raise error if this node is invalid.
        """
        # check for exact custom type
        ctype = get_custom_type(node_name)
        if cls.CUSTOM_NODE_TYPE and ctype and ctype == cls.CUSTOM_NODE_TYPE:
            return True

        # check for exact native type
        if (
            not cls.CUSTOM_NODE_TYPE
            and not ctype
            and cmds.nodeType(node_name) == cls.NATIVE_NODE_TYPE
        ):
            return True

        # check for inherited type only if this class is not custom
        if (
            not exact_type
            and not cls.CUSTOM_NODE_TYPE
            and cmds.objectType(node_name, isAType=cls.NATIVE_NODE_TYPE)
        ):
            return True

        if failfast:
            raise ValueError(
                f"{node_name} is not a {cls.CUSTOM_NODE_TYPE or cls.NATIVE_NODE_TYPE}"
            )
        return False

    def get_bounding_box(self, world_space: bool = True) -> OpenMaya.MBoundingBox:
        """Returns the bounding box of this dag node."""
        bbx = self.fn_set.boundingBox
        if world_space:
            parent = self.get_parent()
            if parent:
                bbx.transformUsing(parent.mdagpath.exclusiveMatrix())
        return bbx

    # --- hierarchy methods

    def get_children(self, **kwargs) -> list[DGNode]:
        """Returns a list of shape objects.

        Args:
            kwargs: kwargs supported by cmds.listRelatives()

        Returns:
            A list of child objects.
        """
        # force long name
        kwargs["fullPath"] = True
        return [PyNode(x) for x in cmds.listRelatives(self.name, **kwargs) or []]

    def get_parent(self, index: int = 0) -> DGNode | None:
        """Returns the parent object at a given index.

        Args:
            index: The parent index. 0 being the immidiate parent.

        Returns:
            The parent object or None if parented to world.
        """
        index  = max(index, 0)
        fn_set = self.fn_set
        for _ in range(index + 1):
            mobject = fn_set.parent(0)
            if not _parent_valid(mobject):
                break
            fn_set = OpenMaya.MFnDagNode(mobject)
        return PyNode(mobject) if _parent_valid(mobject) else None

    def iter_parents(self, node_type: str | list[str] | None = None) -> DGNode:
        """Iterates over parent nodes matching the given type.

        Args:
            node_type: One or more node type to match.
                If None, all parents will be returned.

        Yields:
            The parent object.
        """
        node_types = [node_type] if isinstance(node_type, str) else node_type

        fn_set = self.fn_set
        while fn_set:
            mobject = fn_set.parent(0)
            if not _parent_valid(mobject):
                fn_set = None
                break

            fn_set = OpenMaya.MFnDagNode(mobject)
            name   = fn_set.getPath().partialPathName()
            if not node_types or cmds.nodeType(name) in node_types:
                yield PyNode(mobject)

    def get_parents(self, node_type: str | list[str] | None = None) -> list[DGNode]:
        """Returns a list of parents matching the given type.

        Args:
            node_type: One or more node type to match.
                If None, all parents will be returned.

        Returns:
            The parent object list.
        """
        return list(self.iter_parents(node_type=node_type))

    def set_parent(self, parent: DAGNode | str | None = None, **kwargs) -> None:
        """Sets the parent of the current object.

        Args:
            parent: The new parent object.
                If None, parent to world.
            kwargs: kwargs supported by cmds.parent().
        """
        parent = parent or None
        cur    = self.get_parent()
        if str(cur) != str(parent):
            if self.is_shape:
                if not parent:
                    raise RuntimeError("Cannot parent shapes to world.")
                for ln, sn in zip(("relative", "shape"), ("r", "s")):
                    if sn in kwargs:
                        kwargs.pop(sn)
                    kwargs[ln] = True

            if parent:
                cmds.parent(self.name, parent, **kwargs)
            else:
                if "w" in kwargs:
                    kwargs.pop("w")
                kwargs["world"] = True
                cmds.parent(self.name, **kwargs)

    # --- deformers

    def get_deformers(
        self,
        node_type:  str | list[str] | None = None,
        skip_tweak: bool                   = True,
        first_only: bool                   = False,
    ) -> list[DGNode] | None:
        """Returns deformers associated with this node.

        Args:
            node_type: One or more deformer types to query.
                if None, return all deformers.
            skip_tweak: Skip tweak nodes?
            first_only: If True, return the first node found.
        """
        node_types = [node_type] if isinstance(node_type, str) else node_type
        deformers  = []
        for deformer in mel.eval('findRelatedDeformer "{}"'.format(self)) or []:
            typ = cmds.nodeType(deformer)
            if not node_types or typ in node_types:
                if not skip_tweak or typ != "tweak":
                    if first_only:
                        return PyNode(deformer)
                    else:
                        deformers.append(PyNode(deformer))
        return None if first_only else deformers