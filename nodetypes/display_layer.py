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

    layer = DisplayLayer.define("geometry")          # found, or made (empty) in the current namespace
    layer.add_members(["|pCube1", "|grp"])           # the nodes themselves, never the subtree
    DisplayLayer.for_node("|pCube1")                 # DisplayLayer("geometry"); None in defaultLayer
    layer.get_members()                              # [Transform("pCube1"), Transform("grp")]
    layer.remove_members("|grp")                     # back to defaultLayer
    layer.rename("geo") ; layer.clear() ; layer.delete()
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from rig.nodetypes._base import _cast
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode


# ``rig.membership._LayerMember``, the membership a layer node runs (its
# ``<<`` / ``>>`` / ``in`` / ``of``): set when ``rig.membership`` loads (the
# D31 pattern keeps nodetypes free of imports of the DSL modules)
_LAYER_MEMBER = None

# ``defaultLayer`` spelled absolutely: the name a command resolves while
# another namespace is current and ``namespace -relativeNames`` is on (there
# the bare ``defaultLayer`` names ``:<current>:defaultLayer``)
_DEFAULT_ABSOLUTE = ":defaultLayer"


def _layer_member() -> Any:
    """`_LAYER_MEMBER`, loading ``rig.membership`` first if it is not yet set
    (mid-import only)."""
    if _LAYER_MEMBER is None:
        import rig.membership  # noqa: F401 -- sets it
    return _LAYER_MEMBER


class DisplayLayer(DGNode):
    """
    Display layer class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "displayLayer"

    # the layer a node is in when it is in no layer; Maya refuses to delete it
    DEFAULT = "defaultLayer"

    # a scene registry, found again by name: a create inside ``with container()``
    # stays out of the scope (no prefix, not registered) unless ``container=True``
    _CONTAINER_AWARE = False

    # ``cmds.createDisplayLayer``'s flags; ``create``'s other keywords are the
    # layer's attributes (``displayType=2``, ``visibility=False``)
    _CREATE_FLAGS = frozenset(
        {"name", "n", "empty", "e", "noRecurse", "nr", "number", "num", "makeCurrent", "mc"}
    )

    # ``DisplayLayer()`` (``Layer()``) is the kind token, not a refusal: on
    # ``<<`` defaultLayer (out of every layer), on ``>>`` the enumeration
    _MEMBER_KIND = True

    # ``DisplayLayer(None)``: a failed lookup must not mean every layer
    _NONE_TEXT = "None is not a layer name; Layer() is defaultLayer (it removes from every layer)"

    # --- creation

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates a display layer and returns the layer name.

        The layer is empty unless objects are given (``args``) or the caller
        passes ``empty`` / ``noRecurse``: ``cmds.createDisplayLayer`` alone would
        move the selection (with its hierarchy) out of its layers into the new
        one. ``empty=False`` takes the selection, as the command does.

        Args:
            args, kwargs: the objects, and the ``_CREATE_FLAGS`` given to
                ``create``, for cmds.createDisplayLayer()
        """
        if not args and not any(key in kwargs for key in ("empty", "e", "noRecurse", "nr")):
            kwargs["empty"] = True
        return cmds.createDisplayLayer(*args, **kwargs)

    # --- properties

    @property
    def is_default(self) -> bool:
        """Whether this is ``defaultLayer``, the layer that means no layer
        (by its absolute name: ``namespace -relativeNames`` cannot change the
        answer)."""
        return self.fn_set.absoluteName() == _DEFAULT_ABSOLUTE

    # --- the membership grammar (rig.Layer)

    @classmethod
    def _kind(cls) -> Any:
        """``DisplayLayer()``: the kind token (see NodeMeta's reference)."""
        return _layer_member()(None)

    def _member(self) -> Any:
        """The membership this layer runs on the right of ``<<`` / ``>>``."""
        return _layer_member()(self)

    def __neg__(self) -> Any:
        """``-layer``: the removal token (``cube << -layer`` moves cube back to
        defaultLayer when it is in this layer)."""
        return _layer_member()(self, remove=True)

    def __invert__(self) -> Any:
        raise TypeError(
            f"~{self!r} is unassigned; -layer removes members and Layer() removes "
            f"them from every layer"
        )

    @classmethod
    def of(cls, x: Any) -> list[DisplayLayer]:
        """The layer holding the DAG node ``x``, ``[DisplayLayer("L")]``, or
        ``[]`` in defaultLayer (``x >> Layer()`` answers it or None)."""
        return _layer_member().of(x)

    # --- membership

    @classmethod
    def for_node(cls, node: DAGNode | str) -> DisplayLayer | None:
        """Returns the display layer holding a DAG node, read from the node's own
        ``drawOverride`` input, or None when it is in ``defaultLayer``."""
        layers = cmds.listConnections(
            f"{node}.drawOverride", source=True, destination=False,
            type=cls.NATIVE_NODE_TYPE,
        )
        return cls._wrap(layers[0]) if layers else None

    def get_members(self, no_recurse: bool = True) -> list[DAGNode]:
        """Returns a list of objects in this layer."""
        objs = cmds.editDisplayLayerMembers(
            self.name, query=True, fullNames=True, noRecurse=no_recurse
        )
        return [_cast(x) for x in objs or []]

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
            cmds.editDisplayLayerMembers(_DEFAULT_ABSOLUTE, *mine, noRecurse=True)

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