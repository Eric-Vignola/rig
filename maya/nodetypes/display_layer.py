"""
Object set class
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from rig.maya.nodetypes._base import PyNode
from rig.maya.nodetypes.dag_node import DAGNode
from rig.maya.nodetypes.dg_node import DGNode


class DisplayLayer(DGNode):
    """
    Display layer class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "displayLayer"

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates an empty display layer and returns the layer name.

        Args:
            args, kwargs: kwargs supported by cmds.createDisplayLayer()
        """
        return cmds.createDisplayLayer(*args, **kwargs)

    def get_members(self, no_recurse: bool = True) -> list[DAGNode]:
        """Returns a list of objects in this layer."""
        objs = cmds.editDisplayLayerMembers(
            self.name, query=True, fullNames=True, noRecurse=no_recurse
        )
        return [PyNode(x) for x in objs or []]

    def add_members(self, objects: Any, no_recurse: bool = True) -> None:
        """Adds a list of nodes to this set.

        Args:
            no_recurse: If True, do not add child objects.
        """
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        cmds.editDisplayLayerMembers(self.name, *objects, noRecurse=no_recurse)

    def remove_members(self, objects: DAGNode | str | list[DAGNode | str]) -> None:
        """Removes object(s) from this layer."""
        for each in [objects] if not isinstance(objects, (list, tuple)) else objects:
            src = f"{self.name}.drawInfo"
            dst = f"{each}.drawOverride"
            if cmds.isConnected(src, dst):
                cmds.disconnectAttr(src, dst)

    def clear(self) -> None:
        """Clear object(s) in this layer."""
        self.remove_members(self.get_members(no_recurse=True))