"""
Object set class
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes._base import _cast, _lookup
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.errors import NodeNotFoundError


class ObjectSet(DGNode):
    """
    Object set class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "objectSet"

    # the OpenMaya function set for this type
    FN_SET = OpenMaya.MFnSet

    # a scene registry, found again by name (``get_or_create``; a ShadingEngine
    # by ``for_material``): a create inside ``with container()`` stays out of
    # the scope (no prefix, not registered) unless ``container=True``
    _CONTAINER_AWARE = False

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
                    result.append((_cast(each), None))
                else:
                    result.append(_cast(each))
            elif as_components:
                sel = OpenMaya.MSelectionList()
                sel.add(each)
                mdagpath, mobject = sel.getComponent(0)
                result.append((_cast(mdagpath), mobject))

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

        The name follows the lookup rule of ``Node(x)``: a bare name is
        looked up at the root namespace and in the current one (where
        ``create`` puts a new node); both raise AmbiguousNodeError, a pattern
        NodeLookupError. A node of that name that is not a set of this type
        raises a TypeError: the caller named something else, and a set called
        ``name1`` beside it would silently fork.

        Args:
            name: An object set name to find or create.

        Returns:
            An ObjectSet instance.
        """
        try:
            found = _lookup(name)
        except NodeNotFoundError:
            return cls.create(name=name)
        if cmds.ls(found, type=cls.NATIVE_NODE_TYPE):
            # the cast picks the most derived registered class, so a
            # shadingEngine comes back as a ShadingEngine, equal to any
            # other node object of it.
            return _cast(found)
        raise TypeError(
            f"'{name}' exists and is a {cmds.nodeType(found)}, not a {cls.NATIVE_NODE_TYPE}"
        )