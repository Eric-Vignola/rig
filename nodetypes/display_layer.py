"""
Display layer class

A display layer holds DAG objects, exclusively: a node is in one layer, and
``defaultLayer`` (which cannot be deleted) is where a node sits when it is
in no layer. Membership is a connection from the layer's ``drawInfo`` into
the member's ``drawOverride``, so it is read from the node side
(:meth:`for_node`) without enumerating layers; children of a member draw
with its override through the DAG without being members themselves.
Handing a component to ``editDisplayLayerMembers`` silently stores its
shape, and a DG node is refused by Maya.

Usage::

    from rig.nodetypes import DisplayLayer

    layer = DisplayLayer.get_or_create("geometry")   # exact name, then <currentNamespace>:name
    layer.add_members(["|pCube1", "|grp"])           # the nodes themselves, never the subtree
    DisplayLayer.for_node("|pCube1")                 # DisplayLayer("geometry"); None in defaultLayer
    layer.get_members()                              # [Transform("pCube1"), Transform("grp")]
    layer.remove_members("|grp")                     # back to defaultLayer
    layer.rename("geo") ; layer.clear() ; layer.delete()
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from rig.nodetypes._base import PyNode
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode


class DisplayLayer(DGNode):
    """
    Display layer class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "displayLayer"

    # the layer a node is in when it is in no layer; Maya refuses to delete it
    DEFAULT = "defaultLayer"

    # --- creation

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates an empty display layer and returns the layer name.

        Args:
            args, kwargs: kwargs supported by cmds.createDisplayLayer()
        """
        return cmds.createDisplayLayer(*args, **kwargs)

    @classmethod
    def get_or_create(cls, name: str) -> DisplayLayer:
        """Creates a display layer or returns the existing one.

        The name is looked up as given and in the current namespace (where
        ``create`` puts a new node). A node of that name that is not a display
        layer raises a TypeError: the caller named something else, and a layer
        called ``name1`` beside it would silently fork. The new layer is empty
        (the selection is never read) and does not become the current layer.

        Args:
            name: A display layer name to find or create.
        """
        namespace = cmds.namespaceInfo(currentNamespace=True)
        for candidate in (name, f"{namespace}:{name}"):
            if not cmds.objExists(candidate):
                continue
            if cmds.ls(candidate, type=cls.NATIVE_NODE_TYPE):
                return cls(candidate)
            raise TypeError(
                f"'{candidate}' exists and is a {cmds.nodeType(candidate)}, not a "
                f"{cls.NATIVE_NODE_TYPE}"
            )
        return cls.create(empty=True, name=name)

    # --- properties

    @property
    def is_default(self) -> bool:
        """Whether this is ``defaultLayer``, the layer that means no layer."""
        return self.name == self.DEFAULT

    # --- membership

    @classmethod
    def for_node(cls, node: DAGNode | str) -> DisplayLayer | None:
        """Returns the display layer holding a DAG node, read from the node's own
        ``drawOverride`` input, or None when it is in ``defaultLayer``."""
        layers = cmds.listConnections(
            f"{node}.drawOverride", source=True, destination=False,
            type=cls.NATIVE_NODE_TYPE,
        )
        return cls(layers[0]) if layers else None

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
        """Removes object(s) from this layer (they go to ``defaultLayer``);
        an object in another layer is left there."""
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        mine = [str(x) for x in objects if self.for_node(x) == self]
        if mine:
            cmds.editDisplayLayerMembers(self.DEFAULT, *mine, noRecurse=True)

    def clear(self) -> None:
        """Clear object(s) in this layer."""
        self.remove_members(self.get_members(no_recurse=True))

    def delete(self, nodes: Any = None, **kwargs) -> None:
        """Deletes this layer (its members go to ``defaultLayer``), or ``nodes``
        when given, as :meth:`DGNode.delete` does. Refuses ``defaultLayer``."""
        if nodes is not None:
            super().delete(nodes, **kwargs)
            return
        if self.is_default:
            raise TypeError(f"'{self.name}' cannot be deleted: it is the layer of no layer")
        self.clear()
        super().delete(**kwargs)