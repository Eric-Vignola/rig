"""
:class:`PlugList` -- vectorised broadcast for the rig DSL.

A ``PlugList`` is a regular ``list`` whose attribute access propagates to
each element. Numeric / non-Plug elements pass through unchanged.

Examples::

    nodes = PlugList(["pCube1", "pCube2", "pCube3"])
    nodes.t                    # [Plug("pCube1.t"), Plug("pCube2.t"), Plug("pCube3.t")]
    nodes.t << src             # broadcast src to all .t
    nodes.t << [a, b, c]       # asymmetric: a->pCube1.t, etc.
    a, b, c = nodes.tx + nodes.ty  # element-wise addition

Asymmetric operators use :func:`sequences` to broadcast (the shorter operand
caps to its last element).

Connection queries are methods, N-aligned -- one result slot per element,
so index correspondence with the source list holds::

    nodes.tx.get_inputs()      # [PlugList([Plug]), PlugList([]), ...]
    nodes.tx.get_outputs()     # [PlugList([Plug, Plug]), PlugList([]), ...]

Every slot is a ``PlugList``, empty where nothing is wired, so a slot can
never be a ``None`` that ``<<`` would read as "disconnect". Nested results
are terminal for attribute broadcast and arithmetic -- neither recurses into
them -- but :meth:`PlugList.get` DOES, so a query result reads as values.
Both methods iterate elements directly rather than routing through
:func:`sequences`, so an empty ``PlugList`` yields an empty result.
"""

from __future__ import annotations

import numbers
from typing import Any, Iterable, Iterator, Optional, Union

from maya import cmds
from rig.maya.attribute import Attribute
from rig._internal.generators import sequences
from rig._internal.introspect import _stack_values
from rig._internal.node import Node
from rig._internal.plug import Plug
from rig._internal.types import _is_attribute_spec, _is_components, _is_member_spec


class PlugList(list):
    """List-of-Plug-or-Node that propagates attribute access and arithmetic."""

    # -- construction -- #

    def __init__(
        self,
        items:         Optional[Iterable[Any]] = None,
        _parent_multi: Optional[Any]           = None,
    ) -> None:
        super().__init__()
        # Back-reference to the parent multi attribute when this PlugList
        # was produced by ``multi[:]`` slicing. Used by ``__lshift__`` to
        # route writes through the parent when the slice was empty (so
        # callers can do ``empty_multi[:] << values`` and have the indices
        # auto-created to match the source length). Set to ``None`` for
        # PlugLists not produced by slicing a multi.
        self._parent_multi = _parent_multi
        if items:
            for x in items:
                if isinstance(x, numbers.Real) or x is None:
                    self.append(x)
                elif isinstance(x, (Plug, Node)):
                    self.append(x)
                elif isinstance(x, str) and "." in x and ":" in x.split(".")[-1]:
                    # Component / range selector (e.g. ``pCube1.vtx[0:5]``)
                    # -- flatten via cmds.ls.
                    for resolved in cmds.ls(x, fl=True) or []:
                        self.append(_lift_or_pass(resolved))
                else:
                    self.append(_lift_or_pass(x))

    def __repr__(self) -> str:
        return f"PlugList({list.__repr__(self)})"

    # -- attribute access broadcasts -- #

    def __getattr__(self, name: str) -> "PlugList":
        if name.startswith("_"):
            raise AttributeError(name)
        return PlugList(
            getattr(x, name) if isinstance(x, (Plug, Node)) else x for x in self
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Broadcast assignment => broadcast inject on each element's plug.
        self.__getattr__(name).__lshift__(value)

    def __getitem__(self, key: Any) -> Any:
        # Slice / int / list of int -- preserve normal list behaviour but
        # wrap in PlugList where appropriate.
        if isinstance(key, slice):
            return PlugList(super().__getitem__(key))
        if isinstance(key, str):
            return PlugList(
                getattr(x, key) if isinstance(x, (Plug, Node)) else x for x in self
            )
        if isinstance(key, (list, tuple)):
            # ``list.__getitem__`` explicitly: a zero-argument ``super()``
            # inside a generator expression loses its ``__class__`` cell and
            # raises ``TypeError: super(type, obj)`` instead of indexing.
            return PlugList([list.__getitem__(self, k) for k in key])
        return super().__getitem__(key)

    # -- inject broadcast -- #

    def __lshift__(self, other: Any) -> Any:
        # Retired connection-query sentinel.
        if other is PlugList:
            raise TypeError(
                "'<list> << PlugList' has been replaced by "
                "'<list>.get_inputs()'. Use '<list> << PlugList([...])' -- an "
                "INSTANCE -- to connect."
            )

        # Collection spec -- the whole list is the left-hand side (grouped
        # per node by the spec); sits BEFORE the attribute-spec fan-out so a
        # Components element is never fanned out element by element.
        if _is_member_spec(other):
            return other.inject(self)

        # Spec broadcast -- apply the same spec to every element node. An
        # element that cannot take an attribute is a TypeError naming it,
        # raised before anything is applied; nothing is silently dropped.
        if _is_attribute_spec(other):
            for i, x in enumerate(self):
                if not isinstance(x, (Plug, Node)):
                    raise TypeError(
                        f"element [{i}] ({x!r}) is not a Plug or a Node and cannot "
                        f"take an attribute spec"
                    )
            return PlugList(other.apply(x) for x in self)

        # v4.F.b: empty PlugList from ``multi[:]`` slicing on an
        # unpopulated multi + a sequence source -> route the write
        # through the parent multi so we can auto-create indices to
        # match the source length. Without this fallback,
        # ``empty_multi[:] << values`` would silently no-op (the
        # asymmetric broadcast caps at the shorter side, which is
        # empty).
        parent = getattr(self, "_parent_multi", None)
        if not self and parent is not None:
            from rig._internal.types import _is_sequence

            if isinstance(other, PlugList) or (
                _is_sequence(other) and not isinstance(other, str)
            ):
                parent << other
                return self

        # Asymmetric broadcast. A Components element dispatches too, so
        # ``PlugList([cube.f[:2], cube.f[2:]]) << [Tag("a"), Tag("b")]``
        # pairs each selection with its own spec.
        for s, o in sequences(list(self), other):
            if isinstance(s, (Plug, Node)) or _is_components(s):
                s << o
        return self

    # -- introspect (numpy-aware `get()` + `>>` operator) -- #

    def __rshift__(self, other: Any) -> Any:
        """Broadcast ``__rshift__`` across each element.

        - ``pluglist >> None`` => :meth:`get` (numpy-aware stacked value).
        - ``pluglist >> Tag("x")`` (a collection spec) => query membership of
          the whole list at once; the spec answers with a plain value.
        - ``pluglist >> Node`` => clone each plug's spec onto the target,
          returning a :class:`PlugList` of new :class:`Plug` instances.
        - Other RHS types delegate to each element's :meth:`Plug.__rshift__`,
          which raises :class:`TypeError` for unsupported pairings.

        Asymmetric broadcast follows the same :func:`sequences` rules as the
        other PlugList operators (when ``other`` is itself iterable).
        """
        # `>> None` short-circuit -- return numpy-aware stacked values.
        if other is None:
            return self.get()

        if _is_member_spec(other):
            return other.query(self)

        # Retired connection-query sentinel.
        if other is PlugList:
            raise TypeError(
                "'<list> >> PlugList' has been replaced by '<list>.get_outputs()'."
            )

        results = []
        for x, y in sequences(list(self), other):
            if isinstance(x, (Plug, Node)) or _is_components(x):
                results.append(x >> y)
            else:
                results.append(x)
        return PlugList(results)

    # -- connection queries -- #

    def _query_each(self, method: str) -> "PlugList":
        out = PlugList()
        for x in self:
            if not isinstance(x, Plug):
                raise TypeError(
                    f"'{method}()' needs Plug elements; got "
                    f"{type(x).__name__} ({x!r}). Did you mean "
                    f"'<list>.<attr>.{method}()'?"
                )
            out.append(getattr(x, method)())
        return out

    def get_inputs(self) -> "PlugList":
        """N-aligned incoming connections -- one ``PlugList`` per element.

        Defined explicitly because methods do not broadcast through
        :meth:`__getattr__`; only attribute access does.
        """
        return self._query_each("get_inputs")

    def get_outputs(self) -> "PlugList":
        """N-aligned outgoing connections -- one ``PlugList`` per element."""
        return self._query_each("get_outputs")

    # -- value snapshot -- #

    def get(self) -> Any:
        """Vectorised, numpy-aware ``cmds.getAttr`` -- same shape as ``self >> None``.

        Per-element values follow each element's ``>> None`` semantic:

        - :class:`Plug` => :meth:`Plug.get` (numpy-shaped value).
        - :class:`Node` => underlying typed ``DGNode``.
        - Plain numbers / ``None`` / strings => passthrough.

        The per-element results are then stacked into one homogeneous
        ``np.ndarray`` when shapes line up. Falls back to a plain
        Python ``list`` for heterogeneous content (mixed scalars +
        nodes, mixed shapes, ``str``/``None`` entries).

        Useful for snapshot-copy idioms where ``<<`` would otherwise
        build a live connection::

            target.t << source.t           # live cmds.connectAttr
            target.t << source.t.get()     # one-shot value snapshot
        """
        # PlugList is included so NESTED results (``get_outputs()`` and
        # friends) actually resolve to values. Without it a nested slot is
        # neither Plug nor Node, so it passes through unconverted -- and
        # because Plug subclasses str, a rectangular nesting would then stack
        # into an array of plug NAME STRINGS rather than raising.
        per_element = [
            x >> None if isinstance(x, (Plug, Node, PlugList)) else x for x in self
        ]
        return _stack_values(per_element)

    # -- arithmetic broadcast -- #

    def __add__(self, other: Any) -> "PlugList":
        return PlugList(s + o for s, o in sequences(list(self), other))

    def __radd__(self, other: Any) -> "PlugList":
        return PlugList(o + s for s, o in sequences(list(self), other))

    def __sub__(self, other: Any) -> "PlugList":
        return PlugList(s - o for s, o in sequences(list(self), other))

    def __rsub__(self, other: Any) -> "PlugList":
        return PlugList(o - s for s, o in sequences(list(self), other))

    def __mul__(self, other: Any) -> "PlugList":
        return PlugList(s * o for s, o in sequences(list(self), other))

    def __rmul__(self, other: Any) -> "PlugList":
        return PlugList(o * s for s, o in sequences(list(self), other))

    def __truediv__(self, other: Any) -> "PlugList":
        return PlugList(s / o for s, o in sequences(list(self), other))

    def __rtruediv__(self, other: Any) -> "PlugList":
        return PlugList(o / s for s, o in sequences(list(self), other))

    def __pow__(self, other: Any) -> "PlugList":
        return PlugList(s**o for s, o in sequences(list(self), other))

    def __rpow__(self, other: Any) -> "PlugList":
        return PlugList(o**s for s, o in sequences(list(self), other))

    def __floordiv__(self, other: Any) -> "PlugList":
        return PlugList(s // o for s, o in sequences(list(self), other))

    def __rfloordiv__(self, other: Any) -> "PlugList":
        return PlugList(o // s for s, o in sequences(list(self), other))

    def __mod__(self, other: Any) -> "PlugList":
        return PlugList(s % o for s, o in sequences(list(self), other))

    def __rmod__(self, other: Any) -> "PlugList":
        return PlugList(o % s for s, o in sequences(list(self), other))

    def __and__(self, other: Any) -> "PlugList":
        return PlugList(s & o for s, o in sequences(list(self), other))

    def __rand__(self, other: Any) -> "PlugList":
        return PlugList(o & s for s, o in sequences(list(self), other))

    def __or__(self, other: Any) -> "PlugList":
        return PlugList(s | o for s, o in sequences(list(self), other))

    def __ror__(self, other: Any) -> "PlugList":
        return PlugList(o | s for s, o in sequences(list(self), other))

    def __xor__(self, other: Any) -> "PlugList":
        return PlugList(s ^ o for s, o in sequences(list(self), other))

    def __rxor__(self, other: Any) -> "PlugList":
        return PlugList(o ^ s for s, o in sequences(list(self), other))

    def __neg__(self) -> "PlugList":
        return PlugList(-x for x in self)

    def __invert__(self) -> "PlugList":
        return PlugList(~x for x in self)

    # -- comparison broadcast -- #

    def __eq__(self, other: Any) -> "PlugList":
        return PlugList(s == o for s, o in sequences(list(self), other))

    def __ne__(self, other: Any) -> "PlugList":
        return PlugList(s != o for s, o in sequences(list(self), other))

    def __ge__(self, other: Any) -> "PlugList":
        return PlugList(s >= o for s, o in sequences(list(self), other))

    def __le__(self, other: Any) -> "PlugList":
        return PlugList(s <= o for s, o in sequences(list(self), other))

    def __gt__(self, other: Any) -> "PlugList":
        return PlugList(s > o for s, o in sequences(list(self), other))

    def __lt__(self, other: Any) -> "PlugList":
        return PlugList(s < o for s, o in sequences(list(self), other))

    # -- list protocol (must not route through __eq__) -- #

    def __contains__(self, other: Any) -> bool:
        return any(_same_entity(x, other) for x in self)

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        items  = list(self)
        length = len(items)
        if stop is None:
            stop = length
        if start < 0:
            start = max(length + start, 0)
        if stop < 0:
            stop = max(length + stop, 0)
        for i in range(start, min(stop, length)):
            if _same_entity(items[i], value):
                return i
        raise ValueError(f"{value!r} is not in PlugList")

    def count(self, value: Any) -> int:
        return sum(1 for x in self if _same_entity(x, value))

    def remove(self, value: Any) -> None:
        list.__delitem__(self, self.index(value))

    # -- hashable (so PlugList works in sets / dict keys) -- #

    def __hash__(self) -> int:
        try:
            return hash(tuple(hash(x) for x in self))
        except TypeError:
            return id(self)


def _lift_or_pass(obj: Any) -> Any:
    """Convert a string into a Plug/Node; pass other types through."""
    if isinstance(obj, (Plug, Node, numbers.Real)) or obj is None:
        return obj
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    return obj


def _same_entity(item: Any, probe: Any) -> bool:
    """Plain-list equality, except that plugs compare by name.

    ``list`` containment compares with ``==``, and :meth:`Plug.__eq__` returns a
    condition-node Plug -- so a plug on EITHER side must be diverted, including
    the reflected ``3.0 == plug``. Nested :class:`PlugList` compares by identity
    for the same reason.
    """
    if isinstance(item, (Attribute, Node)) or isinstance(probe, (Attribute, Node)):
        return str(item) == str(probe)
    if isinstance(item, PlugList) or isinstance(probe, PlugList):
        return item is probe
    try:
        return bool(item == probe)
    except Exception:
        return False