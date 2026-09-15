"""
Base deformer class
"""

from __future__ import annotations

from maya import cmds
from rig.maya.nodetypes._base import PyNode
from rig.maya.nodetypes.dg_node import DGNode


class Deformer(DGNode):
    """
    Base deformer class
    """

    NATIVE_NODE_TYPE = "geometryFilter"

    def get_geometries(self) -> list[DGNode]:
        """Returns a list of geometry objects (post-deformation).
        TODO support components
        """
        return [PyNode(x) for x in cmds.deformer(self.name, query=True, geometry=True)]

    def get_original_geometries(self) -> list[DGNode]:
        """Returns a list of original geometry objects (pre-deformation)."""

        geom = (
            cmds.listConnections(
                f"{self.name}.originalGeometry",
                source      = True,
                destination = False,
                plugs       = True,
            )
            or []
        )
        return [PyNode(x.split(".", 1)[0]) for x in geom]