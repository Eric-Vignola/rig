"""
Object set class
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from maya.api import OpenMaya
from rig.maya.nodetypes._base import PyNode
from rig.maya.nodetypes.dag_node import DAGNode
from rig.maya.nodetypes.dg_node import DGNode


class ObjectSet(DGNode):
    """
    Object set class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "objectSet"

    # the OpenMaya function set for this type
    FN_SET = OpenMaya.MFnSet

    def get_members(
        self, as_components: bool = False
    ) -> list[DAGNode] | list[tuple[DAGNode, OpenMaya.MObject]]:
        """Returns a list of objects and components in this set.

        Args:
            as_components: If False, only return DAG nodes.
                Otherwise also return components.
        """
        result = []
        for each in cmds.sets(self.name, query=True) or []:
            if each.find(".") == -1:
                if as_components:
                    result.append((PyNode(each), None))
                else:
                    result.append(PyNode(each))
            elif as_components:
                sel = OpenMaya.MSelectionList()
                sel.add(each)
                mdagpath, mobject = sel.getComponent(0)
                result.append((PyNode(mdagpath), mobject))

        return result

    def add_members(self, objects: Any) -> None:
        """Adds a list of nodes to this set."""
        cmds.sets(objects, add=self.name)

    def remove_members(self, objects: Any) -> None:
        """Removes object(s) from this set."""
        cmds.sets(objects, remove=self.name)

    def force_elements(self, objects: Any) -> None:
        """Force object(s) in this set."""
        cmds.sets(objects, forceElement=self.name)

    def clear(self) -> None:
        """Clear object(s) in this set."""
        cmds.sets(clear=self.name)

    @classmethod
    def get_or_create(cls, name: str) -> ObjectSet:
        """Creates an object set or return the existing one.

        The name is looked up as given and in the current namespace (where
        ``create`` puts a new node). A node of that name that is not a set of
        this type raises a TypeError: the caller named something else, and a
        set called ``name1`` beside it would silently fork.

        Args:
            name: An object set name to find or create.

        Returns:
            An ObjectSet instance.
        """
        namespace = cmds.namespaceInfo(currentNamespace=True)
        for candidate in (name, f"{namespace}:{name}"):
            if not cmds.objExists(candidate):
                continue
            if cmds.ls(candidate, type=cls.NATIVE_NODE_TYPE):
                # PyNode picks the most derived registered class, so a
                # shadingEngine comes back as a ShadingEngine, equal to any
                # other wrapper of it.
                return PyNode(candidate)
            node_type = cmds.nodeType(cmds.ls(candidate, long=True)[0])
            raise TypeError(
                f"'{candidate}' exists and is a {node_type}, not a {cls.NATIVE_NODE_TYPE}"
            )
        return cls.create(name=name)