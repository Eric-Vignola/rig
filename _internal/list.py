"""
:class:`List` -- vectorised broadcast for the rig DSL.

A ``List`` is a regular ``list`` whose attribute access propagates to
each element. Numeric / non-Plug elements pass through unchanged. It is
``rig.List``, not ``typing.List``.

Examples::

    nodes = List(["pCube1", "pCube2", "pCube3"])
    nodes.t                    # List([Plug("pCube1.translate"), ...])
    nodes.t << src             # broadcast src to all .t
    nodes.t << [a, b, c]       # asymmetric: a->pCube1.t, etc.
    a, b, c = nodes.tx + nodes.ty  # element-wise addition
    nodes.__tag__ = 5          # a '_' Maya attr, when every element has it

A ``_`` name broadcasts only when the List is not empty and every element is
a node or plug that has a Maya attr of that name (see `_private_plugs`); any
other read is AttributeError, as Python's probes (``__deepcopy__``) expect
(an assignment: see `List.__setattr__`).

Asymmetric operators use :func:`sequences` to broadcast (the shorter operand
caps to its last element).

Connection queries are methods, N-aligned -- one result slot per element,
so index correspondence with the source list holds::

    nodes.tx.get_inputs()      # List([List([Plug(...)]), List([]), ...])
    nodes.tx.get_outputs()     # List([List([Plug(...), Plug(...)]), List([]), ...])

Every slot is a ``List``, empty where nothing is wired, so a slot can
never be a ``None`` that ``<<`` would read as "disconnect". Nested results
are terminal for attribute broadcast and arithmetic -- neither recurses into
them -- but :meth:`List.get` DOES, so a query result reads as values.
Both methods iterate elements directly rather than routing through
:func:`sequences`, so an empty ``List`` yields an empty result.

``in``, ``index``, ``count`` and ``remove`` never build a node: two plugs
match when they are one Maya plug (``Node("|T1|S").v`` and
``Node("|T2|S").v``), a node matches a str that names it (``"a"``, ``"|a"``),
and a plain str given for a plug is read as the Maya plug it names, so ``"a.tx"``,
``"a.translateX"`` and an alias all find ``a.tx`` (see
:meth:`List.__contains__`).
"""

from __future__ import annotations

import contextlib
import functools
import numbers
from types import MethodType
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes._base import (
    _check_enum_names,
    _class_attr,
    _ensure_node_valid,
    _ensure_owner_alive,
    _enum_value,
    _handle_valid,
    _holds_text,
    _is_enum_attr,
    _is_text,
    _MISSING,
    _plug_identity_name,
    _same_plug,
    Attribute,
)
from rig._internal import anim_keys as _anim_keys
from rig._internal.anim_keys import _Keys
from rig._internal.generators import sequences
from rig._internal.introspect import _stack_values
from rig._internal.journal import statement, user_call
from rig._internal.node import Node
from rig._internal.operands import (
    _CAN_HOLD_STR,
    _prepared_operand,
    _render,
    _RESHAPED,
    _SCALARS,
    _text_operand,
    operator_error,
    operator_where,
    REFLECTED,
)
from rig._internal.plug import Plug
from rig._internal.types import _is_attribute_spec, _is_components, _is_membership
from rig._internal.undo import _undo_chunk


def _operand_rows(dunder: str, items: list, other: Any) -> list:
    """The broadcast rows ``sequences(items, other)`` of a List operator,
    checked before any row builds a node.

    A row that pairs a Plug with a plain str, or with a sequence holding one,
    raises TypeError: that Plug's operator would reject the row anyway (see
    :mod:`rig._internal.operands`), but only after the rows before it had
    built their networks. A row without a Plug is left alone
    (``Node("a") == "a"`` is a plain name comparison). As in the Plug
    operator, a freed plug raises its own error first, a set, frozenset or dict
    ``other`` raises TypeError, and an iterator ``other`` is read into a list
    first (see :mod:`rig._internal.operands`).
    """
    if (
        not isinstance(other, _CAN_HOLD_STR)
        and not isinstance(other, _SCALARS)
        and isinstance(other, _RESHAPED)
    ):
        other = _prepared_operand(other, operator_where(dunder, _render(items), other))
    rows = list(sequences(list(items), other))
    for row, (mine, theirs) in enumerate(rows):
        if isinstance(mine, Plug):
            if isinstance(theirs, _CAN_HOLD_STR) and not isinstance(theirs, Attribute):
                found = _text_operand(theirs)
                if found is not None:
                    _ensure_owner_alive(mine)
                    raise operator_error(dunder, mine, theirs, found, row)
        elif isinstance(theirs, Plug) and isinstance(mine, _CAN_HOLD_STR):
            found = _text_operand(mine)
            if found is not None:
                _ensure_owner_alive(theirs)
                raise operator_error(REFLECTED[dunder], theirs, mine, found, row)
    return rows


def _enum_rows(rows: Iterable) -> list:
    """The broadcast rows ``(element, value)`` of a List ``<<``, each plain str
    given to an enum Plug replaced by the value of that field, read for every
    row before any row is set (see `_enum_value`): a wrong name raises
    TypeError, naming its row, and sets nothing. A row whose Plug is a compound
    or a multi (``pair << ["x", "y"]``) has every name its value holds read
    the same way (`_check_enum_names`), and keeps its value. Other rows are
    left as they are."""
    resolved = []
    for row, (mine, theirs) in enumerate(rows):
        if isinstance(mine, Plug) and _holds_text(theirs):
            where = f"List row {row}, {mine}"
            if _is_text(theirs) and _is_enum_attr(mine):
                theirs = _enum_value(mine.plug, theirs, where)
            else:
                _check_enum_names(mine.mobject, theirs, where, element=not mine.plug.isArray)
        resolved.append((mine, theirs))
    return resolved


class List(list):
    """List-of-Plug-or-Node that propagates attribute access and arithmetic.

    ``nodes.ro << "zxy"`` and ``nodes.ro << ["xzy", "yxz"]`` set enum plugs by
    field name; every name is read before the first set, so one wrong name
    raises TypeError and sets nothing (see :meth:`Plug.__lshift__`).
    """

    # -- construction -- #

    def __init__(
        self,
        items:         Optional[Iterable[Any]] = None,
        _parent_multi: Optional[Any]           = None,
    ) -> None:
        super().__init__()
        # Back-reference to the parent multi attribute when this List
        # was produced by ``multi[:]`` slicing. Used by ``__lshift__`` to
        # route writes through the parent when the slice was empty (so
        # callers can do ``empty_multi[:] << values`` and have the indices
        # auto-created to match the source length). Set to ``None`` for
        # Lists not produced by slicing a multi. (Stored directly: __setattr__
        # is a journal statement, which a private name does not need.)
        object.__setattr__(self, "_parent_multi", _parent_multi)
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
        return f"List({list.__repr__(self)})"

    # -- attribute access broadcasts -- #

    def __getattr__(self, name: str) -> "List":
        if name.startswith("_"):
            # a Maya attr every element has (``nodes.__tag__``), else
            # AttributeError, which Python's probes (``__deepcopy__``) expect
            found = _private_plugs(self, name)
            if found is None:
                raise AttributeError(name)
            return found
        found = List(
            getattr(x, name) if isinstance(x, (Plug, Node)) else x for x in self
        )
        if found and all(type(x) is MethodType for x in found):
            # a method of every element (``List(shapes).save``): a List of bound
            # methods, which a call refuses with the loop to write
            return _Methods(found, name)
        return found

    @statement
    def __setattr__(self, name: str, value: Any) -> None:
        """``nodes.tx = 5`` is sugar for ``nodes.tx << 5``.

        A ``_`` name is the list's own Python state when it is (``_parent_multi``,
        a name already stored on this instance, one ``List`` or ``_Methods``
        declares), else a broadcast inject when every element is a node or plug
        that has a Maya attr of that name (``nodes.__tag__ = 5``). A name only
        some elements have raises AttributeError naming the first that lacks it,
        or its ``already deleted!`` when that one was deleted or freed, in any
        order, and sets nothing. A name no element has is Python state, as on a
        node (``node.__parked__ = 4.0``). A ``List`` subclass keeps private state
        under a ``_`` name by declaring it in its class body (``_extra = None``),
        as a node class does: else, should every element have a Maya attr of
        that name, ``self._extra = x`` sets those attrs.
        """
        if name.startswith("_"):
            if (
                name in _LIST_STATE
                or name in self.__dict__
                or _class_attr(type(self), name) is not _MISSING
                or _class_attr(_Methods, name) is not _MISSING
            ):
                object.__setattr__(self, name, value)
                return
            plugs = _private_plugs(self, name)
            if plugs is None:
                # the first element lacks it: a later one that has it raises
                # (no partial broadcast); else Node's rule (`DGNode.__setattr__`,
                # its ``_`` branch): a `_` name no element has is Python state.
                # Should such a typo raise, these two branches change together.
                _refuse_partial(self, name)
                object.__setattr__(self, name, value)
                return
            plugs.__lshift__(value)
            return
        # Broadcast assignment => broadcast inject on each element's plug.
        self.__getattr__(name).__lshift__(value)

    def __getitem__(self, key: Any) -> Any:
        # Slice / int / list of int -- preserve normal list behaviour but
        # wrap in List where appropriate.
        if isinstance(key, slice):
            return List(super().__getitem__(key))
        if isinstance(key, str):
            return List(
                getattr(x, key) if isinstance(x, (Plug, Node)) else x for x in self
            )
        if isinstance(key, (list, tuple)):
            # ``list.__getitem__`` explicitly: a zero-argument ``super()``
            # inside a generator expression loses its ``__class__`` cell and
            # raises ``TypeError: super(type, obj)`` instead of indexing.
            return List([list.__getitem__(self, k) for k in key])
        return super().__getitem__(key)

    # -- inject broadcast -- #

    @statement
    def __lshift__(self, other: Any) -> Any:
        # Retired connection-query sentinel (the class itself on the right).
        if other is List:
            raise TypeError(
                "'<list> << List' has been replaced by '<list>.get_inputs()'. "
                "Use '<list> << List([...])' -- an INSTANCE -- to connect."
            )

        # Membership -- the whole list is the left-hand side (grouped per
        # node by the spec, validated before any element is written); sits
        # BEFORE the attribute-spec fan-out so a Components element is never
        # fanned out element by element.
        if _is_membership(other):
            return other._member().inject(self)

        # lock / unlock: the List's own verbs (every element checked first, the
        # plugs and the nodes in one undo step), not a fan-out per element
        if _is_attribute_spec(other):
            kargs = getattr(other, "kargs", None)
            if isinstance(kargs, dict) and kargs.keys() == {"lock"}:
                return self.lock() if kargs["lock"] else self.unlock()

        # Keys: one curve per plug (``List(ctrls).ry << AnimCurve([...])``), every
        # row planned first, then one undo step (``rig.animcurve``)
        if type(other) is _Keys:
            _anim_keys.inject_rows(list(sequences(list(self), other)))
            return self

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
            # every element's checks first: a refused one (another kind of
            # attribute, a setting that cannot apply) leaves every other as it was
            if len(self) > 1:
                for x in self:
                    user_call(other._precheck, x)
            return List(user_call(other.apply, x) for x in self)

        # v4.F.b: empty List from ``multi[:]`` slicing on an
        # unpopulated multi + a sequence source -> route the write
        # through the parent multi so we can auto-create indices to
        # match the source length. Without this fallback,
        # ``empty_multi[:] << values`` would silently no-op (the
        # asymmetric broadcast caps at the shorter side, which is
        # empty).
        parent = getattr(self, "_parent_multi", None)
        if not self and parent is not None:
            from rig._internal.types import _is_sequence

            if isinstance(other, List) or (
                _is_sequence(other) and not isinstance(other, str)
            ):
                parent << other
                return self

        # Asymmetric broadcast. A Components element dispatches too, so
        # ``List([cube.f[:2], cube.f[2:]]) << [Tag("a"), Tag("b")]``
        # pairs each selection with its own spec. Enum field names are read
        # for every row first, so a wrong one sets nothing.
        rows = sequences(list(self), other)
        if isinstance(other, (list, tuple)) and any(_is_membership(o) for o in other):
            return self._membership_rows(list(rows))
        if isinstance(other, (list, tuple)) and any(type(o) is _Keys for o in other):
            # pairs of keys: every pair planned first (a row of another kind is a
            # TypeError: one '<<' per kind), then one undo step
            _anim_keys.inject_rows(list(rows))
            return self
        if _holds_text(other):
            rows = _enum_rows(rows)
        for s, o in rows:
            if isinstance(s, (Plug, Node)) or _is_components(s):
                s << o
        return self

    def _membership_rows(self, rows: list) -> "List":
        """``List([a, b]) << [red, blue]``: every pair's checks and plan first
        (a refused pair writes nothing, not even the pairs before it), then the
        pairs in order, in one undo step (``rig.membership``). Every right-hand
        side is a membership (a collection node, a token, a Tag)."""
        for i, (s, o) in enumerate(rows):
            if not _is_membership(o):
                raise TypeError(
                    f"a pair broadcast with membership takes a membership on every right-hand "
                    f"side; [{i}] is {o!r}: write one '<<' per kind"
                )
            if not (isinstance(s, (Plug, Node)) or _is_components(s)):
                raise TypeError(f"element [{i}] ({s!r}) is not a node, a plug or components")
            o._member()._prepare(s)
        with _undo_chunk("rig.membership"):
            for s, o in rows:
                s << o
        return self

    # -- introspect (numpy-aware `get()` + `>>` operator) -- #

    @statement
    def __rshift__(self, other: Any) -> Any:
        """Broadcast ``__rshift__`` across each element.

        - ``items >> None`` => :meth:`get` (numpy-aware stacked value).
        - ``items >> Tag("x")`` (membership: a tag, a layer, shader or
          shading engine node, a kind token) => query membership of the
          whole list at once (never an element-wise clone onto a membership
          node); the collection answers with a plain value.
        - ``items >> Node`` => clone each plug's spec onto the target,
          returning a :class:`List` of new :class:`Plug` instances.
        - Other RHS types delegate to each element's :meth:`Plug.__rshift__`,
          which raises :class:`TypeError` for unsupported pairings.

        Asymmetric broadcast follows the same :func:`sequences` rules as the
        other List operators (when ``other`` is itself iterable).
        """
        # `>> None` short-circuit -- return numpy-aware stacked values.
        if other is None:
            return self.get()

        if _is_membership(other):
            return other._member().query(self)

        # Retired connection-query sentinel (the class itself on the right).
        if other is List:
            raise TypeError(
                "'<list> >> List' has been replaced by '<list>.get_outputs()'."
            )

        results = []
        for x, y in sequences(list(self), other):
            if isinstance(x, (Plug, Node)) or _is_components(x):
                results.append(x >> y)
            else:
                results.append(x)
        return List(results)

    # -- connection queries -- #

    def _query_each(self, method: str) -> "List":
        out = List()
        for x in self:
            if not isinstance(x, Plug):
                raise TypeError(
                    f"'{method}()' needs Plug elements; got "
                    f"{type(x).__name__} ({x!r}). Did you mean "
                    f"'<list>.<attr>.{method}()'?"
                )
            out.append(getattr(x, method)())
        return out

    # -- lock / unlock -- #

    def lock(self) -> "List":
        """Locks every element, nodes and plugs, and returns this List:
        ``lst.lock()`` is ``lst << lock``. All or nothing: every element is
        checked first (a refused one raises InjectionError and changes
        nothing, an element that is no node or plug a TypeError); the plugs
        are locked first, then the nodes in one ``lockNode`` (a locked node's
        plugs never lock), in one undo step (``rig.lock``). An element locked
        already is left as it is; nothing to change writes nothing and adds no
        undo step.

        Defined explicitly because methods do not broadcast through
        :meth:`__getattr__` (``lst.is_locked`` does: one bool per element)."""
        return _set_locks(self, True)

    def unlock(self) -> "List":
        """Unlocks every element and returns this List: ``lst.unlock()`` is
        ``lst << unlock``. The nodes first (one ``lockNode``), then the plugs;
        all or nothing, one undo step (``rig.unlock``), as :meth:`lock`."""
        return _set_locks(self, False)

    def get_inputs(self) -> "List":
        """N-aligned incoming connections -- one ``List`` per element.

        Defined explicitly because methods do not broadcast through
        :meth:`__getattr__`; only attribute access does.
        """
        return self._query_each("get_inputs")

    def get_outputs(self) -> "List":
        """N-aligned outgoing connections -- one ``List`` per element."""
        return self._query_each("get_outputs")

    # -- value snapshot -- #

    def get(self) -> Any:
        """Vectorised, numpy-aware ``cmds.getAttr`` -- same shape as ``self >> None``.

        Per-element values follow each element's ``>> None`` semantic:

        - :class:`Plug` => :meth:`Plug.get` (numpy-shaped value).
        - a node => the node itself (``node >> None``).
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
        # List is included so NESTED results (``get_outputs()`` and
        # friends) actually resolve to values. Without it a nested slot is
        # neither Plug nor Node, so it passes through unconverted -- and
        # because Plug subclasses str, a rectangular nesting would then stack
        # into an array of plug NAME STRINGS rather than raising.
        per_element = [
            x >> None if isinstance(x, (Plug, Node, List)) else x for x in self
        ]
        return _stack_values(per_element)

    # -- arithmetic broadcast -- #

    def __add__(self, other: Any) -> "List":
        return List(s + o for s, o in _operand_rows("__add__", self, other))

    def __radd__(self, other: Any) -> "List":
        return List(o + s for s, o in _operand_rows("__radd__", self, other))

    def __sub__(self, other: Any) -> "List":
        return List(s - o for s, o in _operand_rows("__sub__", self, other))

    def __rsub__(self, other: Any) -> "List":
        return List(o - s for s, o in _operand_rows("__rsub__", self, other))

    def __mul__(self, other: Any) -> "List":
        return List(s * o for s, o in _operand_rows("__mul__", self, other))

    def __rmul__(self, other: Any) -> "List":
        return List(o * s for s, o in _operand_rows("__rmul__", self, other))

    def __truediv__(self, other: Any) -> "List":
        return List(s / o for s, o in _operand_rows("__truediv__", self, other))

    def __rtruediv__(self, other: Any) -> "List":
        return List(o / s for s, o in _operand_rows("__rtruediv__", self, other))

    def __pow__(self, other: Any) -> "List":
        return List(s**o for s, o in _operand_rows("__pow__", self, other))

    def __rpow__(self, other: Any) -> "List":
        return List(o**s for s, o in _operand_rows("__rpow__", self, other))

    def __floordiv__(self, other: Any) -> "List":
        return List(s // o for s, o in _operand_rows("__floordiv__", self, other))

    def __rfloordiv__(self, other: Any) -> "List":
        return List(o // s for s, o in _operand_rows("__rfloordiv__", self, other))

    def __mod__(self, other: Any) -> "List":
        return List(s % o for s, o in _operand_rows("__mod__", self, other))

    def __rmod__(self, other: Any) -> "List":
        return List(o % s for s, o in _operand_rows("__rmod__", self, other))

    def __and__(self, other: Any) -> "List":
        return List(s & o for s, o in _operand_rows("__and__", self, other))

    def __rand__(self, other: Any) -> "List":
        return List(o & s for s, o in _operand_rows("__rand__", self, other))

    def __or__(self, other: Any) -> "List":
        return List(s | o for s, o in _operand_rows("__or__", self, other))

    def __ror__(self, other: Any) -> "List":
        return List(o | s for s, o in _operand_rows("__ror__", self, other))

    def __xor__(self, other: Any) -> "List":
        return List(s ^ o for s, o in _operand_rows("__xor__", self, other))

    def __rxor__(self, other: Any) -> "List":
        return List(o ^ s for s, o in _operand_rows("__rxor__", self, other))

    def __neg__(self) -> "List":
        return List(-x for x in self)

    def __invert__(self) -> "List":
        return List(~x for x in self)

    # -- comparison broadcast -- #

    def __eq__(self, other: Any) -> "List":
        return List(s == o for s, o in _operand_rows("__eq__", self, other))

    def __ne__(self, other: Any) -> "List":
        return List(s != o for s, o in _operand_rows("__ne__", self, other))

    def __ge__(self, other: Any) -> "List":
        return List(s >= o for s, o in _operand_rows("__ge__", self, other))

    def __le__(self, other: Any) -> "List":
        return List(s <= o for s, o in _operand_rows("__le__", self, other))

    def __gt__(self, other: Any) -> "List":
        return List(s > o for s, o in _operand_rows("__gt__", self, other))

    def __lt__(self, other: Any) -> "List":
        return List(s < o for s, o in _operand_rows("__lt__", self, other))

    # -- list protocol (must not route through __eq__) -- #

    def __contains__(self, other: Any) -> bool:
        """True if an element is `other`; builds no node (``list``'s own test
        is ``==``, which builds a condition network for a plug).

        Two plugs match when they are one Maya plug (see ``_same_plug``): the
        same node, attribute and logical indices, through any instance path.
        A plain str (not a Plug) given for a plug is read as the Maya plug it
        names (decision S3 Q4): ``"a.tx"``, ``"a.translateX"``, an alias,
        another instance path (``"|T1|S.v"`` finds ``Node("|T2|S").v``), a
        namespaced name and a component name (``"np.cv[1][2]"`` finds its
        ``controlPoints`` element) all find it. A str that names no single plug
        matches no plug element: no such node or attribute, a node's name, a
        pattern or a range, a name more than one object has, a node deleted to
        the undo queue (see ``_plug_named``). The str is resolved once per call,
        when the first plug element is reached. A node element matches a str
        that names it, as ``Node(text) == element`` would (``"|a"`` and ``"a"``
        find ``Node("a")``; another instance path is another DAG node object),
        and any other element as ``list`` does. A plug element of a deleted or
        freed node raises ``already deleted!``, as its name does. The probe is
        read once per call too (its name, its plug identity), at the first
        element; an element that is the probe object matches at once.

        ``index``, ``count`` and ``remove`` match elements the same way.
        """
        match = _matcher(other)
        return any(match(x) for x in self)

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        items  = list(self)
        length = len(items)
        if stop is None:
            stop = length
        if start < 0:
            start = max(length + start, 0)
        if stop < 0:
            stop = max(length + stop, 0)
        match = _matcher(value)
        for i in range(start, min(stop, length)):
            if match(items[i]):
                return i
        raise ValueError(f"{value!r} is not in List")

    def count(self, value: Any) -> int:
        match = _matcher(value)
        return sum(1 for x in self if match(x))

    def remove(self, value: Any) -> None:
        list.__delitem__(self, self.index(value))

    # -- hashable (so List works in sets / dict keys) -- #

    def __hash__(self) -> int:
        try:
            return hash(tuple(hash(x) for x in self))
        except TypeError:
            return id(self)


# each broadcast operator is one journal statement
for _operator in (
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__pow__", "__rpow__", "__floordiv__",
    "__rfloordiv__", "__mod__", "__rmod__", "__and__", "__rand__", "__or__",
    "__ror__", "__xor__", "__rxor__", "__neg__", "__invert__",
    "__eq__", "__ne__", "__ge__", "__le__", "__gt__", "__lt__",
):
    setattr(List, _operator, statement(List.__dict__[_operator]))
del _operator


def _set_locks(items: "List", state: bool) -> "List":
    """`List.lock` (``state`` True) / `List.unlock`: every check first, then the
    edits (lock: the plugs, then the nodes in one ``lockNode``; unlock: the
    nodes, then the plugs), in one undo step when there is more than one
    command. An element already in that state, and a node or plug met again,
    is left out. Returns ``items``."""
    from rig._internal.journal import cmds_edit
    from rig.nodetypes._base import _injection_error, _node_lock_refusal, _plug_lock_refusal

    nodes: list = []
    plugs: list = []
    seen_nodes: set = set()
    seen_plugs: dict = {}
    for i, x in enumerate(items):
        if isinstance(x, Attribute) and not _is_components(x):
            _ensure_owner_alive(x)
            if x.plug.isLocked == state:
                continue
            same = seen_plugs.setdefault(hash(x), [])
            if any(_same_plug(x, other) for other in same):
                continue
            same.append(x)
            plugs.append(x)
        elif isinstance(x, Node):
            fn = x.fn_set
            if fn.isLocked == state:
                continue
            uuid = str(fn.uuid())
            if uuid not in seen_nodes:
                seen_nodes.add(uuid)
                nodes.append((x, fn))
        else:
            raise TypeError(f"element [{i}] ({x!r}) is not a node or a plug: it has no lock")

    # the checks, in the order of the edits
    for x, fn in nodes:
        why = _node_lock_refusal(fn, x.name, state)
        if why:
            raise _injection_error(why)
    for x in plugs:
        owner = OpenMaya.MFnDependencyNode(x.plug.node())
        # a plug of a node this List unlocks first: only a parent's lock refuses it
        first = not state and owner.isLocked and str(owner.uuid()) in seen_nodes
        why   = _plug_lock_refusal(x, state, node=not first)
        if why:
            raise _injection_error(why)

    commands = (1 if nodes else 0) + len(plugs)
    if commands == 0:
        return items
    with _undo_chunk("rig.lock" if state else "rig.unlock") if commands > 1 else contextlib.nullcontext():
        if not state and nodes:
            cmds_edit.lockNode([x.name for x, _ in nodes], lock=False)
        for x in plugs:
            cmds_edit.setAttr(x.full_name, lock=state)
        if state and nodes:
            cmds_edit.lockNode([x.name for x, _ in nodes], lock=True)
    return items


# the List's own Python state (``_Methods`` keeps its method name): never a
# Maya attr, whatever its elements have
_LIST_STATE = frozenset(("_parent_multi", "_method"))


def _private_plugs(items: "List", name: str) -> Optional["List"]:
    """``items.<name>`` for a ``_`` name (``nodes.__tag__``): the List of each
    element's Maya attr of that name, or None (the caller's AttributeError, or,
    for an assignment, `_refuse_partial` then Python state) when the list is
    empty or its first element is no node or plug with such an attr (see
    `_private_plug`). A Python probe (``__deepcopy__``, ``__array__``) costs
    one lookup, on the first element.

    Once an element has it, every other must: a later element without it
    raises AttributeError naming it, and one deleted or freed raises its
    ``already deleted!``, before anything is read or set (no partial
    broadcast)."""
    found = []
    for x in list.__iter__(items):
        plug = _private_plug(x, name)
        if plug is not None:
            found.append(plug)
            continue
        if not found:
            return None
        _raise_partial(x, len(found), name, 0)
    return List(found) if found else None


def _refuse_partial(items: "List", name: str) -> None:
    """For an assignment `_private_plugs` found no broadcast for (its first
    element lacks `name`): raise as `_private_plugs` does for a later element
    when any element has `name`, naming the first element, which lacks it (its
    ``already deleted!`` when deleted or freed), and return when none has it,
    which leaves `name` Python state (Node's rule). So ``List([d, a]).__tag__ =
    5`` raises as ``List([a, d]).__tag__ = 5`` does, whatever the order. Every
    element is read: an assignment is no Python probe."""
    for i, x in enumerate(list.__iter__(items)):
        if i and _private_plug(x, name) is not None:
            _raise_partial(list.__getitem__(items, 0), 0, name, i)


def _raise_partial(x: Any, index: int, name: str, has: int) -> None:
    """Raise for element `index` (`x`), which lacks the ``_`` name `name`
    element `has` has: its ``already deleted!`` when it is a node or plug that
    was deleted or freed, else AttributeError naming it."""
    if isinstance(x, Plug):
        _ensure_node_valid(x)
    elif isinstance(x, Node):
        ensure = getattr(type(x), "ensure_valid", None)
        if ensure is not None:
            ensure(x)
    raise AttributeError(
        f"List element [{index}] ({x!r}) has no attribute {name!r}, which "
        f"element [{has}] has: a '_' name broadcasts only when every element has it "
        f"(nothing was read or set)"
    )


def _private_plug(x: Any, name: str) -> Any:
    """Element `x`'s Maya attr of the ``_`` name `name`, as `x` resolves it on
    its own, or None: a node through `DGNode.__getattr__` (a Maya attr of the
    live node), a plug through `Plug.__getattr__` (a compound child, else a
    Maya attr of its node), never through their Python members. Anything else
    is None.

    A plug is read from its handles first, which is safe on a node deleted or
    freed (None then: `Plug.__getattr__` would raise ``already deleted!`` for
    a node deleted to the undo queue, where a probe expects AttributeError).
    Then the ``hasAttribute`` of the node (of a plug's node) must answer True
    before the lookup: a compound child, of a multi too, is an attr of the
    node, and every node class resolves a ``_`` name only through that check
    (`DGNode.__getattr__`, which `Container.__getattr__` defers to), so a probe
    costs one ``hasAttribute``, with no raise, no child-name table, no node
    cast."""
    if isinstance(x, Plug):
        d      = x.__dict__
        handle = d.get("_handle")
        if handle is not None:
            if not handle.isValid():
                return None
            attrh = d.get("_attrh")
            if attrh is not None and not attrh.isAlive():
                return None
            # the owner's fn set when the plug shares its handle (it is the
            # plug's node), else one of the node the handle holds
            node = d.get("_node")
            fn   = None
            if node is not None:
                nd = node.__dict__
                if nd.get("_objhandle") is handle:
                    fn = nd.get("_fn_set")
            if fn is None:
                fn = OpenMaya.MFnDependencyNode(handle.object())
            if not fn.hasAttribute(name):
                return None
        else:
            attrh = d.get("_attrh")
            if attrh is not None and not attrh.isAlive():
                return None
    elif isinstance(x, Node):
        # `DGNode.__getattr__`'s own ``_`` check, without its raise
        d  = x.__dict__
        fn = d.get("_fn_set")
        if fn is None or not _handle_valid(d) or not fn.hasAttribute(name):
            return None
    else:
        return None
    try:
        return type(x).__getattr__(x, name)
    except AttributeError:
        return None


class _Methods(List):
    """``List(items).<method>``: the bound method of each element. A method does not
    broadcast over a List (only plugs and values do): calling it raises a TypeError
    that names the loop. Iterating it, or indexing it, gives the methods."""

    def __init__(self, items: Iterable[Any], name: str) -> None:
        super().__init__(items)
        object.__setattr__(self, "_method", name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name = self._method
        raise TypeError(
            f"List(...).{name}(...): {name} is a method of each element, and a method does "
            f"not broadcast over a List (plugs and values do); call it on each one: "
            f"[x.{name}(...) for x in items]"
        )


def _lift_or_pass(obj: Any) -> Any:
    """Convert a string into a Plug/Node; pass other types through."""
    if isinstance(obj, (Plug, Node, numbers.Real)) or obj is None:
        return obj
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    return obj


def _matcher(probe: Any) -> Callable[[Any], bool]:
    """The test `List.__contains__`, `index` and `count` run on each element for
    `probe`: `_NamedPlug` for a plain str (not an Attribute), `_SamePlug` for a
    plug, `_SameNode` for a node (each reads the probe once per call), else
    `_same_entity`."""
    if isinstance(probe, str):
        if isinstance(probe, Attribute):
            return _SamePlug(probe)
        return _NamedPlug(probe)
    if isinstance(probe, Node):
        return _SameNode(probe)
    return functools.partial(_same_entity, probe=probe)


class _SamePlug:
    """The element test of a plug probe, `_same_entity`'s rule with the probe
    read once, at the first element: named (a deleted or freed probe raises, as
    its name does), its node and plug identity kept. An element that is the
    probe matches; another plug element is named (a deleted one raises) and
    matches when it is the same Maya plug (see `_same_plug`); any other element
    compares by name."""

    __slots__ = ("probe", "name", "mnode", "identity")

    def __init__(self, probe: Attribute) -> None:
        self.probe = probe
        self.name  = None

    def __call__(self, item: Any) -> bool:
        probe = self.probe
        if self.name is None:
            self.name     = probe.full_name
            self.mnode    = probe.__dict__["_mplug"].node()
            self.identity = _plug_identity_name(probe)
        if item is probe:
            return True
        if isinstance(item, Attribute):
            item.full_name
            return item.__dict__["_mplug"].node() == self.mnode and (
                _plug_identity_name(item) == self.identity
            )
        if isinstance(item, List):
            return False
        return str(item) == self.name


class _SameNode:
    """The element test of a node probe, `_same_entity`'s rule with the probe's
    name read once, at the first element (a deleted or freed probe raises, as
    its name does): an element that is the probe matches, any other compares
    by name (a deleted element raises)."""

    __slots__ = ("probe", "name")

    def __init__(self, probe: Any) -> None:
        self.probe = probe
        self.name  = None

    def __call__(self, item: Any) -> bool:
        if self.name is None:
            self.name = str(self.probe)
        if item is self.probe:
            return True
        if isinstance(item, List):
            return False
        return str(item) == self.name


def _plug_named(text: str) -> Optional[Plug]:
    """The Plug of the one Maya plug `text` names, read as ``Plug(text)`` reads
    it (a component name is its ``controlPoints`` / ``uvpt`` element), or None
    when `text` names no single plug: no such node or attribute, a node's name,
    a pattern or a range (``"t*.tx"``, ``"box.vtx[0:3]"``), a name more than one
    object has (``"X.tx"`` for ``|P1|X`` and ``|P2|X``), a name read against the
    selection (``".tx"``), a node deleted to the undo queue, or anything else
    Maya refuses. Builds no node."""
    if not text or text[0] == "." or "*" in text or "?" in text:
        return None
    try:
        selection = OpenMaya.MSelectionList()
        selection.add(text)
        if selection.length() != 1:
            return None
        try:
            component = selection.getComponent(0)[1]
        except (RuntimeError, TypeError):
            component = None  # a plug, or a DG node
        if (
            component is not None
            and not component.isNull()
            and OpenMaya.MFnComponent(component).elementCount != 1
        ):
            return None
        return Plug(text)
    except Exception:
        return None


class _NamedPlug:
    """The element test of a plain str probe (see `List.__contains__`): a plug
    element matches when it is the Maya plug the str names (`_same_plug`),
    which is resolved once, when the first plug element is reached; a str
    that names none matches no plug element. A node element matches when the
    str is its name, or names it (`_node_named`, resolved once, when the first
    node element the str is not the name of is reached). Any other element is
    compared by `_same_entity`."""

    __slots__ = ("text", "plug", "pending", "node_name", "node_pending")

    def __init__(self, text: str) -> None:
        self.text         = text
        self.plug         = None
        self.pending      = True
        self.node_name    = None
        self.node_pending = True

    def __call__(self, item: Any) -> bool:
        if isinstance(item, Node):
            # its name raises for a deleted or freed node
            name = str(item)
            if name == self.text:
                return True
            if self.node_pending:
                self.node_pending = False
                self.node_name    = _node_named(self.text)
            return name == self.node_name
        if not isinstance(item, Attribute):
            return _same_entity(item, self.text)
        if self.pending:
            self.pending = False
            self.plug    = _plug_named(self.text)
        if self.plug is None:
            # a deleted or freed element raises, as its name does (and as
            # `_same_plug` names both)
            item.full_name
            return False
        return _same_plug(item, self.plug)


def _node_named(text: str) -> Optional[str]:
    """The name (``str(node)``) of the one node `text` names (``"|a"``,
    ``"ns:a"``, a uuid), or None when it names none or more than one, or names
    a plug, a component or a pattern. Builds no node."""
    if not text or "." in text or "*" in text or "?" in text:
        return None
    try:
        selection = OpenMaya.MSelectionList()
        selection.add(text)
        if selection.length() != 1:
            return None
        try:
            return selection.getDagPath(0).partialPathName()
        except TypeError:
            return OpenMaya.MFnDependencyNode(selection.getDependNode(0)).name()
    except Exception:
        return None


def _same_entity(item: Any, probe: Any) -> bool:
    """Plain-list equality, except that two plugs compare as Maya plugs and a
    plug and anything else (a node, a str element) compare by name.

    ``list`` containment compares with ``==``, and :meth:`Plug.__eq__` returns a
    condition-node Plug -- so a plug on EITHER side must be diverted, including
    the reflected ``3.0 == plug``. Two plugs are the same entity when they are
    the same Maya plug (see :meth:`Plug.equals`): ``Node("|T1|S").v`` and
    ``Node("|T2|S").v`` are, though their names differ. A node and a str
    compare by name. (A plain str probe of a plug element never reaches here:
    `_NamedPlug` reads it as the plug it names.) Nested :class:`List` compares
    by identity for the same reason.
    """
    if isinstance(item, Attribute) and isinstance(probe, Attribute):
        return _same_plug(item, probe)
    if isinstance(item, (Attribute, Node)) or isinstance(probe, (Attribute, Node)):
        return str(item) == str(probe)
    if isinstance(item, List) or isinstance(probe, List):
        return item is probe
    try:
        return bool(item == probe)
    except Exception:
        return False