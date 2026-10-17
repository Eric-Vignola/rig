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

``DisplayLayer`` is also the DSL's ``Layer`` (``rig.Layer is
DisplayLayer``): a layer node on the right of ``<<`` / ``>>`` / ``in`` is
the membership collection (``cube << layer``, ``cube in layer``;
``layer.remove(cube)``, whose shorthand is ``cube << -layer``), and
``DisplayLayer()`` (``Layer()``) is the kind token:
defaultLayer, the layer of no layer, on ``<<``; the enumeration on ``>>``.

Usage::

    from rig.nodetypes import DisplayLayer

    layer = DisplayLayer.define("geometry")          # found, or made (empty) in the current namespace
    cube << layer ; cube in layer                    # the membership grammar (rig.Layer): the node itself
    DisplayLayer.for_node(cube)                      # DisplayLayer("geometry"); None in defaultLayer
    layer.get_members()                              # [Transform("cube")]
    layer.remove(cube)                               # back to defaultLayer; cube << -layer is the shorthand
    cube >> DisplayLayer() ; DisplayLayer.of(cube)   # the layer holding cube, or None ; [layer] or []
    layer.rename("geo") ; layer.clear() ; layer.delete()

There is one way in and out of a layer (user decision 2026-10-16): the raw
``add_members`` / ``remove_members`` an object set has raise a TypeError
naming ``cube << layer`` / ``layer.remove(cube)``.
"""

from __future__ import annotations

import re
from typing import Any

from maya import cmds
from rig._internal.journal import cmds_edit as _cmds_edit, statement as _statement
from rig.nodetypes._base import _cast, _lookup
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.errors import NodeNotFoundError
from rig._internal.undo import _undo_chunk


# ``rig.membership._LayerMember``, the membership a layer node runs (its
# ``<<`` / ``>>`` / ``in`` / ``of``): set when ``rig.membership`` loads (the
# D31 pattern keeps nodetypes free of imports of the DSL modules)
_LAYER_MEMBER = None

# A display layer name Maya keeps: identifiers joined by namespace separators.
_LAYER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)*$")

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


def _leaf(name: str) -> str:
    """The name without its namespace."""
    return name.rsplit(":", 1)[-1]


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
        one. ``empty=False`` takes the selection, as the command does. The
        objects given join as the nodes themselves, never their subtrees (as
        ``node << layer`` does): ``noRecurse=True`` unless the caller passes
        ``noRecurse`` / ``nr`` (``noRecurse=False`` takes the hierarchy).

        Args:
            args, kwargs: the objects, and the ``_CREATE_FLAGS`` given to
                ``create``, for cmds.createDisplayLayer()
        """
        recurse = any(key in kwargs for key in ("noRecurse", "nr"))
        if not args and not recurse and not any(key in kwargs for key in ("empty", "e")):
            kwargs["empty"] = True
        elif args and not recurse:
            kwargs["noRecurse"] = True
        return _cmds_edit.createDisplayLayer(*args, **kwargs)

    # --- queries

    def is_default(self) -> bool:
        """Whether this is ``defaultLayer``, the layer that means no layer
        (by its absolute name: ``namespace -relativeNames`` cannot change the
        answer)."""
        return type(self)._raw_get_fn_set(self).absoluteName() == _DEFAULT_ABSOLUTE

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

    @_statement
    def remove(self, members: Any) -> Any:
        """``layer.remove(x)``: moves ``x`` (a DAG node, or a ``List`` / list of
        them) back to defaultLayer when it is in this layer, exactly as ``x <<
        -layer`` (its shorthand) does: the same refusals, before any write (a
        component, a DG node, an attribute plug, ``defaultLayer`` itself), one
        undo step. Returns ``members``, as ``<<`` returns its left-hand side
        (user decision 2026-10-16)."""
        return (-self)._member()._run_inject(members)

    def __contains__(self, lhs: Any) -> bool:
        """``x in layer``: whether every node of ``x`` (a node, a plug, which
        stands for its node, or a list of them) is in this layer
        (``defaultLayer``: in no layer)."""
        return self._member()._run_contains(lhs)

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
            str(self), query=True, fullNames=True, noRecurse=no_recurse
        )
        return [_cast(x) for x in objs or []]

    def add_members(self, *args: Any, **kwargs: Any) -> None:
        """Refused (one way only, user decision 2026-10-16): ``cube << layer``
        puts a node in this layer, with its checks (a component or a DG node
        is refused, where ``editDisplayLayerMembers`` silently stored a
        component's whole shape)."""
        raise TypeError(
            f"layer.add_members is not rig's form: write cube << {self!r} (a List or "
            f"list of nodes too); {self!r}.remove(cube) takes them out"
        )

    def remove_members(self, *args: Any, **kwargs: Any) -> None:
        """Refused (one way only, user decision 2026-10-16):
        ``layer.remove(cube)`` (shorthand ``cube << -layer``) takes a node out."""
        raise TypeError(
            f"layer.remove_members is not rig's form: write {self!r}.remove(cube) "
            f"(shorthand cube << -layer); cube << {self!r} puts it in"
        )

    def _add_members(self, objects: Any, no_recurse: bool = True) -> None:
        """Adds a list of nodes to this layer (``editDisplayLayerMembers``, no
        checks: rig's own use).

        Args:
            no_recurse: If True, do not add child objects.
        """
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        _cmds_edit.editDisplayLayerMembers(str(self), *objects, noRecurse=no_recurse)

    def _remove_members(self, objects: DAGNode | str | list[DAGNode | str]) -> None:
        """Removes object(s) from this layer (they go to ``defaultLayer``);
        an object in another layer is left there (rig's own use)."""
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        mine = [str(x) for x in objects if self.for_node(x) == self]
        if mine:
            _cmds_edit.editDisplayLayerMembers(_DEFAULT_ABSOLUTE, *mine, noRecurse=True)

    def _guard(self, verb: str) -> str:
        """The layer's name, once it is known to take ``verb``: never
        ``defaultLayer`` (TypeError) nor a referenced layer (RuntimeError)."""
        name = str(self)
        if type(self).is_default(self):
            raise TypeError(f"'{name}' cannot be {verb}: it is the layer of no layer")
        if cmds.referenceQuery(name, isNodeReferenced=True):
            raise RuntimeError(f"'{name}' is referenced and cannot be {verb}; edit the source file")
        return name

    def clear(self) -> None:
        """Sends every member of this layer to ``defaultLayer``; the layer
        stays. One undo step. Refuses ``defaultLayer`` and a referenced
        layer."""
        self._guard("cleared")
        with _undo_chunk("rig.layer"):
            self._remove_members(self.get_members(no_recurse=True))

    def delete(self, nodes: Any = None, **kwargs) -> None:
        """Deletes this layer (its members go to ``defaultLayer``), in one undo
        step, or ``nodes`` when given, as :meth:`DGNode.delete` does. Refuses
        ``defaultLayer``, a referenced layer and a locked one."""
        if nodes is not None:
            super().delete(nodes, **kwargs)
            return
        name = self._guard("deleted")
        self._refuse_if_locked("delete")
        with _undo_chunk("rig.layer"):
            self._remove_members(self.get_members(no_recurse=True))
            super().delete(**kwargs)
        if cmds.objExists(name):
            raise RuntimeError(f"'{name}' survived its deletion")

    def rename(self, new_name: Any) -> None:
        """Renames this layer (the node object follows), in one undo step.
        Refused before any write: ``defaultLayer`` (TypeError), a referenced
        layer (RuntimeError), a locked layer (InjectionError), a name that is no
        str (TypeError), a name Maya would not keep or one a node already has
        (ValueError). Verifies Maya
        kept the name."""
        if not isinstance(new_name, str) or not new_name:
            raise TypeError(f"a layer's new name is a non-empty str, not {new_name!r}")
        if not _LAYER_NAME_RE.match(new_name):
            raise ValueError(
                f"{new_name!r} is not a display layer name Maya keeps: identifiers of "
                f"[A-Za-z0-9_] not starting with a digit, joined by ':'"
            )
        old = self._guard("renamed")
        if new_name in (old, type(self).get_short_name(self)):
            return
        self._refuse_if_locked("rename")
        try:
            taken = _lookup(new_name)
        except NodeNotFoundError:
            taken = None
        if taken is not None:
            raise ValueError(f"'{new_name}' already exists")
        with _undo_chunk("rig.layer"):
            got = _cmds_edit.rename(old, new_name)
        if _leaf(got) != _leaf(new_name):
            raise RuntimeError(f"Maya renamed '{old}' to '{got}', not '{new_name}'")
