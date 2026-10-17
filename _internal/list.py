"""
:class:`List` -- vectorised broadcast for the rig DSL.

A ``List`` is a regular ``list`` whose attribute access propagates to
each element. It is ``rig.List``, not ``typing.List``.

What a List holds (`_user_items`): numbers (``numbers.Real``, bool, numpy
scalars), ``None``, a value (a tuple, list or array of numbers and plugs: a
vector, a matrix, an operand row; an OpenMaya ``MVector`` / ``MMatrix`` ...),
and rig objects (nodes, plugs, components, Lists, the membership tokens,
attribute specs and AnimCurve keys); a str is cast to the node or plug it
names (a component range flattens); an OpenMaya ``MObject`` / ``MDagPath`` /
``MPlug`` to its node or plug; any other object through its ``uuid`` (cgmath
data: the node it was serialized from) or its ``str`` (``str(TransformData)``
is its node's name). An element that is none of these is a TypeError naming
it, and nothing is added. The rule applies to ``List(items)``, ``append``,
``extend``, ``insert``, ``+=`` and item / slice assignment; a List given to
them gives its elements as they are. The Lists rig builds from what it
computed (a broadcast, an operator, a slice, a ``multi[:]``, a command's
result) keep their elements (`_values_list`), and so does a copy.

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

A node deleted or freed, or a plug of one, stays in its slot; every List
operation skips it with one warning, and a List an operation derives holds
``rig.skip`` there (see "deleted elements" below).

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
import reprlib
import numbers
import operator
from types import FunctionType, MethodType
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig.nodetypes._base import (
    _GETFRAME,
    _check_enum_names,
    _class_attr,
    _user_view,
    _ensure_node_valid,
    _ensure_owner_alive,
    _enum_value,
    _handle_valid,
    _holds_text,
    _is_enum_attr,
    _is_text,
    _is_uuid,
    _METHOD_LOAD,
    _MethodAttribute,
    _MISSING,
    _GETFRAME,
    _BarePredicate,
    _is_predicate_name,
    _plug_identity_name,
    _same_plug,
    Attribute,
)
from rig._internal import anim_keys as _anim_keys
from rig._internal.anim_keys import _Keys
from rig._internal.generators import sequences
from rig._internal.introspect import _stack_values
from rig._internal.journal import _in_rig, statement, user_call
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
from rig._internal.plug import _MethodPlug, Plug
from rig._internal.types import _is_attribute_spec, _is_components, _is_membership
from rig._internal.types import _lazy as _types_lazy
from rig.nodetypes.errors import AmbiguousNodeError
from rig._internal.undo import _undo_chunk
from rig.spec.modifiers import skip as _skip_marker


# --- deleted elements (user decision 2026-10-16) ------------------------------
#
# A node deleted (to the undo queue) or freed (a new scene, a file open, a
# reference unload) stays in its List slot, and so does a plug of one and a
# Components selection on one: length, indices and zip pairing are unchanged.
# rig's own List operations skip such an element and print ONE warning per
# statement naming the skipped ones ("List element [0] obj1 was deleted;
# skipped", `_warn`); a List an operation derives (an
# attribute broadcast, an operator's results, get(), get_inputs()) holds
# ``rig.skip`` in that slot, which every List operation leaves alone, so chained
# broadcasts stay aligned and keep skipping. Every rig operation on the node or
# plug itself still raises "... already deleted!".

# ``rig.skip`` (rig.spec imports neither this module nor rig._internal.plug)
_SKIP = _skip_marker


# {an element's type: the name of the handle its __dict__ holds ("_handle" for a
# plug, "_objhandle" for a node), or None}: one dict lookup per element instead
# of two isinstance calls on classes with a metaclass
_HANDLE_KEYS: dict = {}


# the key of a Components selection (``__slots__``, no __dict__): its shape's
# handle is read (`_element_handle`)
_SHAPE_KEY = "_shape"


def _handle_key(t: type) -> Optional[str]:
    key = _HANDLE_KEYS[t] = (
        "_handle" if issubclass(t, Attribute)
        else "_objhandle" if issubclass(t, Node)
        else _SHAPE_KEY if issubclass(t, _types_lazy().members.Components)
        else None
    )
    return key


def _element_handle(x: Any, key: str) -> Any:
    """The handle `_gone` reads for `x`: its own (a node's), its owner's (a
    plug's), or its shape's (a Components')."""
    if key is _SHAPE_KEY:
        return x._shape.__dict__.get("_objhandle")
    return x.__dict__.get(key)


def _gone(x: Any) -> bool:
    """True if `x` is a node deleted to the undo queue or freed, or a plug of
    one (its handle, or its owner's, is not valid), or a Components selection
    on one. Reads only the handle, which is safe on a freed node."""
    t   = type(x)
    key = _HANDLE_KEYS[t] if t in _HANDLE_KEYS else _handle_key(t)
    if key is None:
        return False
    handle = _element_handle(x, key)
    return handle is not None and not handle.isValid()


def _gone_indices(items: Iterable[Any]) -> list:
    """The indices of the deleted or freed elements of `items` (`_gone`), in
    order; usually empty. A handle object met again (the plugs of one node
    share their owner's) is read once."""
    out  = []
    seen = None
    keys = _HANDLE_KEYS
    for i, x in enumerate(items):
        t   = type(x)
        key = keys[t] if t in keys else _handle_key(t)
        if key is None:
            continue
        handle = x.__dict__.get(key) if key is not _SHAPE_KEY else _element_handle(x, key)
        if handle is None or handle is seen:
            continue
        if handle.isValid():
            seen = handle
        else:
            out.append(i)
    return out


def _gone_name(x: Any) -> str:
    """The name a skipped element is warned by: a node's last known name, a
    plug's str buffer (the name it was built with), a Components' shape's last
    known name and its ids, all without reading the freed node."""
    if isinstance(x, Attribute):
        return str.__str__(x)
    if not isinstance(x, Node):
        return x._gone_name()  # a Components
    from rig.nodetypes.dg_node import _deleted_name

    return _deleted_name(x)


def _skipped_names(items: list, indices: list) -> list:
    """``(label, element)`` of each skipped element `indices` of `items`: the
    label a warning names it by (``[i] name``)."""
    return [(f"[{i}] {_gone_name(items[i])}", items[i]) for i in indices]


# the statement the last skip warning was for: ((id of the user's frame, its
# code, its line), its instruction offset, the ids of the elements named), or
# None. No frame and no element is kept (that would keep a user's objects
# alive): ids only.
_WARNED: list = [None]


def _user_frame() -> Any:
    """The frame of the user's code running the List operation (the first frame
    up the stack that is not rig's own code), or None (C code called it)."""
    try:
        frame = _GETFRAME(2)
    except ValueError:
        return None
    in_rig = _in_rig
    while frame is not None and in_rig(frame.f_globals.get("__name__") or ""):
        frame = frame.f_back
    return frame


def _warn(named: list) -> None:
    """ONE warning per statement naming the skipped elements `named` (``(label,
    element)``, the label ``[i] name``; user decision 2026-10-16): a statement
    whose operations skip the same element (``lst.tx << lst.ty * 2 + lst.tz``,
    ``lst.tx = lst.ty``) names it once. The statement is the user's frame and
    line running it, at an instruction past the last warning's (a loop's next
    pass is back at the same one: it warns again). Called from no user frame
    (C code), every warning is printed."""
    frame = _user_frame()
    if frame is not None:
        key  = (id(frame), frame.f_code, frame.f_lineno)
        at   = frame.f_lasti
        last = _WARNED[0]
        if last is not None and last[0] == key and at > last[1]:
            known = last[2]
            fresh = [(label, x) for label, x in named if id(x) not in known]
            known.update(id(x) for _, x in named)
            _WARNED[0] = (key, at, known)
            if not fresh:
                return
            named = fresh
        else:
            _WARNED[0] = (key, at, {id(x) for _, x in named})
    else:
        _WARNED[0] = None
    if len(named) == 1:
        cmds.warning(f"List element {named[0][0]} was deleted; skipped")
    else:
        cmds.warning(f"List elements {', '.join(label for label, _ in named)} were deleted; skipped")


def _warn_skipped(items: list, indices: list) -> None:
    """ONE warning naming the skipped elements `indices` of `items`."""
    _warn(_skipped_names(items, indices))


def _with_skips(items: Any, notes: Optional[list] = None) -> list:
    """The elements of `items` (a List) with ``rig.skip`` in the slot of each
    deleted or freed one, after one warning naming them (or, given `notes`,
    their names appended to it, for the caller's one warning); the elements as
    they are (a plain list) when none is."""
    elements = list(list.__iter__(items) if isinstance(items, list) else items)
    gone     = _gone_indices(elements)
    if not gone:
        return elements
    if notes is None:
        _warn_skipped(elements, gone)
    else:
        notes.extend(_skipped_names(elements, gone))
    skip = _SKIP
    for i in gone:
        elements[i] = skip
    return elements


def _skip_rows(items: Any, other: Any) -> tuple:
    """`items`' elements and `other` with ``rig.skip`` in the slot of every
    deleted or freed element of either (`other` when it is a List), after ONE
    warning naming them all (the operand's as ``operand [i] name``)."""
    notes: list = []
    if isinstance(other, List):
        theirs = _with_skips(other, notes)
        notes  = [(f"operand {label}", x) for label, x in notes]
        if notes:
            other = _values_list(theirs)
    mine: list = []
    elements = _with_skips(items, mine)
    if mine or notes:
        _warn(mine + notes)
    return elements, other


def _live(items: "List") -> "List":
    """`items` without its deleted or freed elements and its skip slots, a
    nested List's too (one warning names them), or `items` itself when it has
    none: for an operation on the List as a whole (membership, a query), which
    has no slot to keep."""
    notes: list = []
    live = _live_into(items, notes, "")
    if notes:
        _warn(notes)
    return live


def _live_into(items: "List", notes: list, where: str) -> "List":
    """`_live` of `items` (at `where`, ``[i]`` of its parents), the names of
    the elements left out appended to `notes`."""
    elements = list(list.__iter__(items))
    gone     = set(_gone_indices(elements))
    skip     = _SKIP
    out      = []
    changed  = False
    for i, x in enumerate(elements):
        if i in gone:
            notes.append((f"{where}[{i}] {_gone_name(x)}", x))
            changed = True
        elif x is skip:
            changed = True  # a skip slot (a derived List's) is left alone too
        elif isinstance(x, List):
            y = _live_into(x, notes, f"{where}[{i}]")
            changed = changed or y is not x
            out.append(y)
        else:
            out.append(x)
    return _values_list(out) if changed else items


def _list_repr(items: "List") -> str:
    return f"List({list.__repr__(items)})"


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
    # a deleted or freed element (of this List, or of a List operand) is a skip
    # row, whose result is ``rig.skip`` (`_row_results`)
    elements, other = _skip_rows(items, other)
    rows = list(sequences(elements, other))
    skip = _SKIP
    for row, (mine, theirs) in enumerate(rows):
        if mine is skip or theirs is skip:
            continue
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


def _live_rows(items: "List", other: Any) -> list:
    """The broadcast rows ``(element, value)`` of ``items << other``
    (``sequences``) without the rows of a deleted or freed element, of `items`
    or of a List `other` (one warning names them), or of a skip slot: each is
    left alone, as ``element << skip`` leaves it."""
    skip = _SKIP
    elements, other = _skip_rows(items, other)
    return [
        (mine, theirs) for mine, theirs in sequences(elements, other)
        if mine is not skip and theirs is not skip
    ]


# what a deleted or freed element is to ``in`` / ``index`` / ``count`` /
# ``remove``: no match, never read
_NO_MATCH = object()


def _unmatched_gone(items: "List") -> list:
    """The elements of `items` with `_NO_MATCH` in the slot of each deleted or
    freed one (one warning names them): ``in``, ``index``, ``count`` and
    ``remove`` never match nor read it."""
    elements = list(list.__iter__(items))
    gone     = _gone_indices(elements)
    if gone:
        _warn_skipped(elements, gone)
        for i in gone:
            elements[i] = _NO_MATCH
    return elements


def _row_results(rows: list, op: Callable[[Any, Any], Any]) -> "List":
    """The List of ``op(mine, theirs)`` per row of `_operand_rows`, ``rig.skip``
    for a skip row (a deleted or freed element, or a skip slot, on either side)."""
    skip = _SKIP
    return _values_list(
        skip if mine is skip or theirs is skip else op(mine, theirs) for mine, theirs in rows
    )


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
                theirs = _enum_value(mine._mplug, theirs, where)
            else:
                _check_enum_names(type(mine)._raw_get_mobject(mine), theirs, where, element=not mine._mplug.isArray)
        resolved.append((mine, theirs))
    return resolved


class List(list):
    """List-of-Plug-or-Node that propagates attribute access and arithmetic.

    It holds numbers, numeric values and rig objects; a name, an OpenMaya
    object or an object whose ``str`` names a node or plug is cast to that node
    or plug (``List(Node("root").serialize_hierarchy())`` is a List of the
    hierarchy's nodes), and anything else is a TypeError naming its index:
    see `_user_items`, which ``List(items)``, ``append``, ``extend``,
    ``insert``, ``+=`` and item / slice assignment run.

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
        if items is not None:  # (an array has no truth value)
            list.extend(self, _user_items(items, 0))

    # -- adding elements: each one cast as List(items) casts it -- #

    def append(self, value: Any) -> None:
        if type(value) in _PLAIN or isinstance(value, _KEPT):
            list.append(self, value)
            return
        list.extend(self, _user_items((value,), len(self)))

    def extend(self, values: Iterable[Any]) -> None:
        list.extend(self, _user_items(values, len(self)))

    def insert(self, index: Any, value: Any) -> None:
        if type(value) in _PLAIN or isinstance(value, _KEPT):
            list.insert(self, index, value)
            return
        items = _user_items((value,), index)
        if len(items) == 1:
            list.insert(self, index, items[0])
            return
        # a component range: its elements in its place
        start, _, _ = slice(index, index).indices(len(self))
        list.__setitem__(self, slice(start, start), items)

    def __iadd__(self, values: Iterable[Any]) -> "List":
        list.extend(self, _user_items(values, len(self)))
        return self

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            start = key.indices(len(self))[0]
            list.__setitem__(self, key, _user_items(value, start))
            return
        try:
            operator.index(key)
        except TypeError:
            list.__setitem__(self, key, value)  # list's own error
            return
        items = _user_items((value,), key)
        if len(items) != 1:
            index = operator.index(key)
            if -len(self) <= index < 0:
                index += len(self)  # the slice of a negative index (lst[-1:0] is empty)
            raise TypeError(
                f"List element [{key}]: {value!r} names {len(items)} elements; assign "
                f"them to a slice (lst[{index}:{index + 1}] = ...)"
            )
        list.__setitem__(self, key, items[0])

    def __reduce_ex__(self, protocol: Any) -> tuple:
        # copy.copy / deepcopy / pickle rebuild a list subclass through append,
        # which casts: a copy keeps the elements as they are (`_rebuild`), and
        # its state (`_parent_multi`, a `_Methods`' `_method`)
        return _rebuild, (type(self), list(self), dict(self.__dict__))

    @reprlib.recursive_repr("List([...])")
    def __repr__(self) -> str:
        # a deleted or freed element shows marked (user decision 2026-10-16:
        # ``List([Transform("obj1", deleted), Transform("obj2")])``), whoever
        # asks: its own repr raises in rig's code (`_called_from_rig`), so the
        # List renders again as a user reads it (`_user_view`), which marks a
        # node, a plug, a Components, a token or a method of one, nested too
        try:
            return f"List({list.__repr__(self)})"
        except RuntimeError:
            return _user_view(_list_repr, self)

    # -- attribute access broadcasts -- #

    def __getattr__(self, name: str) -> "List":
        if name.startswith("_"):
            # a Maya attr every element has (``nodes.__tag__``), else
            # AttributeError, which Python's probes (``__deepcopy__``) expect
            found = _private_plugs(self, name)
            if found is None:
                raise AttributeError(name)
            return found
        # a deleted or freed element: rig.skip in its slot (`_with_skips`)
        found = _values_list(
            getattr(x, name) if isinstance(x, (Plug, Node)) else x for x in _with_skips(self)
        )
        if not found:
            return found
        # the plug-method rule over a List: with brackets it is the method
        # (``List(nodes).lock()``, the caller's frame on a method load: each
        # method plug stands for its method), bare it is the plugs if every
        # element has the Maya attr
        f  = _GETFRAME(1)
        co = f.f_code.co_code
        i  = f.f_lasti
        k  = _METHOD_LOAD[co[i]]
        if k and (k & 1 or co[i + 1] & 1):
            found = _values_list(
                x.__dict__["_method"] if isinstance(x, _METHOD_PLUGS) else x for x in found
            )
        skip    = _SKIP
        methods = [
            type(x) is MethodType or type(x) is FunctionType or type(x) is _BarePredicate
            for x in found
            if x is not skip
        ]
        if methods and all(methods):
            # a method of every element (``List(shapes).save``): a List of bound
            # methods, which a call refuses with the loop to write
            return _Methods(found, name)
        if any(methods):
            plugs = [k for k, x in enumerate(found) if isinstance(x, _METHOD_PLUGS)]
            if plugs:
                # only some elements have the Maya attr: neither plugs nor methods
                raise AttributeError(_mixed_methods_text(name, plugs, len(found)))
        return found

    @statement
    def __setattr__(self, name: str, value: Any) -> None:
        """``nodes.tx = 5`` is sugar for ``nodes.tx << 5``.

        A ``_`` name is the list's own Python state when it is (``_parent_multi``,
        a name already stored on this instance, one ``List`` or ``_Methods``
        declares), else a broadcast inject when every element is a node or plug
        that has a Maya attr of that name (``nodes.__tag__ = 5``). A name only
        some elements have raises AttributeError naming the first that lacks it,
        in any order, and sets nothing; a deleted or freed element is skipped
        (one warning), as every List operation skips it. A name no element has
        raises AttributeError,
        as on a node (the '_' rule, user decision 2026-10-16: ``nodes._tga = 5``;
        an empty List has no element, so it raises too). A ``List`` subclass
        keeps private state under a ``_`` name by declaring it in its class body
        (``_extra = None``), as a node class does.
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
                # its ``_`` branch, user decision 2026-10-16): a `_` name that
                # is no List state, no class member and no element's Maya attr
                # raises
                _refuse_partial(self, name)
                raise AttributeError(
                    f"{type(self).__name__}.{name}: a '_' name on a List is its "
                    f"class's Python state or every element's Maya attr, and no "
                    f"element has {name!r}: declare it in the class body "
                    f"({name} = None) to keep Python state"
                )
            plugs.__lshift__(value)
            return
        # Broadcast assignment => broadcast inject on each element's plug.
        self.__getattr__(name).__lshift__(value)

    def __getitem__(self, key: Any) -> Any:
        # Slice / int / list of int -- preserve normal list behaviour but
        # wrap in List where appropriate.
        if isinstance(key, slice):
            return _values_list(super().__getitem__(key))
        if isinstance(key, str):
            return _values_list(
                getattr(x, key) if isinstance(x, (Plug, Node)) else x for x in _with_skips(self)
            )
        if isinstance(key, (list, tuple)):
            # ``list.__getitem__`` explicitly: a zero-argument ``super()``
            # inside a generator expression loses its ``__class__`` cell and
            # raises ``TypeError: super(type, obj)`` instead of indexing.
            return _values_list([list.__getitem__(self, k) for k in key])
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
            return other._member()._run_inject(self)

        # lock / unlock: the List's own verbs (every element checked first, the
        # plugs and the nodes in one undo step), not a fan-out per element
        if _is_attribute_spec(other):
            kargs = getattr(other, "kargs", None)
            if kargs is not None and kargs.keys() == {"lock"}:  # a dict, or a modifier's read-only mapping
                return self.lock() if kargs["lock"] else self.unlock()

        # Keys: one curve per plug (``List(ctrls).ry << AnimCurve([...])``), every
        # row planned first, then one undo step (``rig.animcurve``)
        skip = _SKIP
        if type(other) is _Keys:
            _anim_keys.inject_rows(_live_rows(self, other))
            return self

        # Spec broadcast -- apply the same spec to every element node. An
        # element that cannot take an attribute is a TypeError naming it,
        # raised before anything is applied; nothing is silently dropped (a
        # deleted or freed element, a skip slot, is left alone: rig.skip in
        # its slot of the result).
        if _is_attribute_spec(other):
            elements = _with_skips(self)
            live     = [x for x in elements if x is not skip]
            for i, x in enumerate(elements):
                if x is not skip and not isinstance(x, (Plug, Node)):
                    raise TypeError(
                        f"element [{i}] ({x!r}) is not a Plug or a Node and cannot "
                        f"take an attribute spec"
                    )
            # every element's checks first: a refused one (another kind of
            # attribute, a setting that cannot apply) leaves every other as it was
            if len(live) > 1:
                for x in live:
                    user_call(other._precheck, x)
            return _values_list(
                x if x is skip else user_call(other.apply, x) for x in elements
            )

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
        # A deleted or freed element (of this List, or of a List operand) and
        # a skip slot leave their row alone (`_live_rows`, one warning).
        rows = _live_rows(self, other)
        if isinstance(other, (list, tuple)) and any(_is_membership(o) for o in other):
            return self._membership_rows(rows)
        if isinstance(other, (list, tuple)) and any(type(o) is _Keys for o in other):
            # pairs of keys: every pair planned first (a row of another kind is a
            # TypeError: one '<<' per kind), then one undo step
            _anim_keys.inject_rows(rows)
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
            return other._member()._run_query(self)

        # Retired connection-query sentinel (the class itself on the right).
        if other is List:
            raise TypeError(
                "'<list> >> List' has been replaced by '<list>.get_outputs()'."
            )

        results  = []
        skip     = _SKIP
        elements, other = _skip_rows(self, other)
        for x, y in sequences(elements, other):
            if x is skip or y is skip:
                results.append(skip)
            elif isinstance(x, (Plug, Node)) or _is_components(x):
                results.append(x >> y)
            else:
                results.append(x)
        return _values_list(results)

    # -- connection queries -- #

    def _query_each(self, method: str) -> "List":
        out  = _values_list()
        skip = _SKIP
        for x in _with_skips(self):
            if x is skip:
                list.append(out, skip)
                continue
            if not isinstance(x, Plug):
                raise TypeError(
                    f"'{method}()' needs Plug elements; got "
                    f"{type(x).__name__} ({x!r}). Did you mean "
                    f"'<list>.<attr>.{method}()'?"
                )
            list.append(out, getattr(x, method)())
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
        :meth:`__getattr__` (``is_locked()`` is a method of a node and of a plug:
        ``[x.is_locked() for x in lst]``)."""
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
        # (a deleted or freed element gives rig.skip, `_with_skips`)
        per_element = [
            x >> None if isinstance(x, (Plug, Node, List)) else x for x in _with_skips(self)
        ]
        skip = _SKIP
        if any(x is skip for x in per_element):
            # a skipped element's slot: a List of the values with rig.skip there
            # (user decision 2026-10-16, a derived List; fix round 2026-10-17):
            # it broadcasts as every derived List does (``lst.tx.get() + 1``
            # keeps skip in that slot, ``lst.ty << lst.tx.get()`` leaves it)
            return _values_list(per_element)
        return _stack_values(per_element)

    # -- arithmetic broadcast -- #

    def __add__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__add__", self, other), lambda s, o: s + o)

    def __radd__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__radd__", self, other), lambda s, o: o + s)

    def __sub__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__sub__", self, other), lambda s, o: s - o)

    def __rsub__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rsub__", self, other), lambda s, o: o - s)

    def __mul__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__mul__", self, other), lambda s, o: s * o)

    def __rmul__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rmul__", self, other), lambda s, o: o * s)

    def __truediv__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__truediv__", self, other), lambda s, o: s / o)

    def __rtruediv__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rtruediv__", self, other), lambda s, o: o / s)

    def __pow__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__pow__", self, other), lambda s, o: s**o)

    def __rpow__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rpow__", self, other), lambda s, o: o**s)

    def __floordiv__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__floordiv__", self, other), lambda s, o: s // o)

    def __rfloordiv__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rfloordiv__", self, other), lambda s, o: o // s)

    def __mod__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__mod__", self, other), lambda s, o: s % o)

    def __rmod__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rmod__", self, other), lambda s, o: o % s)

    def __and__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__and__", self, other), lambda s, o: s & o)

    def __rand__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rand__", self, other), lambda s, o: o & s)

    def __or__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__or__", self, other), lambda s, o: s | o)

    def __ror__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__ror__", self, other), lambda s, o: o | s)

    def __xor__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__xor__", self, other), lambda s, o: s ^ o)

    def __rxor__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__rxor__", self, other), lambda s, o: o ^ s)

    def __neg__(self) -> "List":
        skip = _SKIP
        return _values_list(x if x is skip else -x for x in _with_skips(self))

    def __invert__(self) -> "List":
        skip = _SKIP
        return _values_list(x if x is skip else ~x for x in _with_skips(self))

    # -- comparison broadcast -- #

    def __eq__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__eq__", self, other), lambda s, o: s == o)

    def __ne__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__ne__", self, other), lambda s, o: s != o)

    def __ge__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__ge__", self, other), lambda s, o: s >= o)

    def __le__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__le__", self, other), lambda s, o: s <= o)

    def __gt__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__gt__", self, other), lambda s, o: s > o)

    def __lt__(self, other: Any) -> "List":
        return _row_results(_operand_rows("__lt__", self, other), lambda s, o: s < o)

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
        and any other element as ``list`` does. A deleted or freed element (a
        node, or a plug of one) matches nothing and is never read, with one
        warning naming it (user decision 2026-10-16); a deleted or freed probe
        raises ``already deleted!``, as its name does. The probe is
        read once per call too (its name, its plug identity), at the first
        element; an element that is the probe object matches at once.

        ``index``, ``count`` and ``remove`` match elements the same way.
        """
        if _gone(other):
            # a deleted or freed probe is found by identity only, never read nor
            # warned for (``ctls.remove(gone_ctl)``)
            return any(x is other for x in list.__iter__(self))
        match = _matcher(other)
        return any(x is not _NO_MATCH and match(x) for x in _unmatched_gone(self))

    def index(self, value: Any, start: int = 0, stop: Optional[int] = None) -> int:
        if _gone(value):
            # a deleted or freed probe: by identity only (see `__contains__`)
            items = list(list.__iter__(self))
            match = lambda x: x is value  # noqa: E731
        else:
            items = _unmatched_gone(self)
            match = None
        length = len(items)
        if stop is None:
            stop = length
        if start < 0:
            start = max(length + start, 0)
        if stop < 0:
            stop = max(length + stop, 0)
        if match is None:
            match = _matcher(value)
        for i in range(start, min(stop, length)):
            if items[i] is not _NO_MATCH and match(items[i]):
                return i
        raise ValueError(f"{value!r} is not in List")

    def count(self, value: Any) -> int:
        if _gone(value):
            # a deleted or freed probe: by identity only (see `__contains__`)
            return sum(1 for x in list.__iter__(self) if x is value)
        match = _matcher(value)
        return sum(1 for x in _unmatched_gone(self) if x is not _NO_MATCH and match(x))

    def remove(self, value: Any) -> None:
        list.__delitem__(self, self.index(value))

    # -- hashable (so List works in sets / dict keys) -- #

    def __hash__(self) -> int:
        try:
            return hash(tuple(hash(x) for x in self))
        except (TypeError, RuntimeError):
            # an unhashable element, or a deleted or freed node's (its name)
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
    skip = _SKIP
    # a deleted or freed element, a skip slot: left alone (one warning)
    for i, x in enumerate(_with_skips(items)):
        if x is skip:
            continue
        if isinstance(x, Attribute) and not _is_components(x):
            _ensure_owner_alive(x)
            if x._mplug.isLocked == state:
                continue
            same = seen_plugs.setdefault(hash(x), [])
            if any(_same_plug(x, other) for other in same):
                continue
            same.append(x)
            plugs.append(x)
        elif isinstance(x, Node):
            fn = type(x)._raw_get_fn_set(x)
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
        why = _node_lock_refusal(fn, str(x), state)
        if why:
            raise _injection_error(why)
    for x in plugs:
        owner = OpenMaya.MFnDependencyNode(x._mplug.node())
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
            cmds_edit.lockNode([str(x) for x, _ in nodes], lock=False)
        for x in plugs:
            cmds_edit.setAttr(str(x), lock=state)
        if state and nodes:
            cmds_edit.lockNode([str(x) for x, _ in nodes], lock=True)
    return items


# the List's own Python state (``_Methods`` keeps its method name): never a
# Maya attr, whatever its elements have
_LIST_STATE = frozenset(("_parent_multi", "_method"))


def _private_plugs(items: "List", name: str) -> Optional["List"]:
    """``items.<name>`` for a ``_`` name (``nodes.__tag__``): the List of each
    element's Maya attr of that name, or None (the caller's AttributeError, or,
    for an assignment, `_refuse_partial` then the '_' rule's raise) when the list is
    empty or its first element is no node or plug with such an attr (see
    `_private_plug`). A Python probe (``__deepcopy__``, ``__array__``) costs
    one lookup, on the first element.

    Once a live element has it, every other must: a later element without it
    raises AttributeError naming it before anything is read or set (no
    partial broadcast). A deleted or freed element is skipped: rig.skip in
    its slot, one warning when the broadcast is made."""
    found = []
    has   = None
    gone  = []
    skip  = _SKIP
    elements = list(list.__iter__(items))
    for i, x in enumerate(elements):
        # a deleted or freed element, a skip slot: rig.skip in its slot
        if x is skip:
            found.append(skip)
            continue
        if _gone(x):
            found.append(skip)
            gone.append(i)
            continue
        plug = _private_plug(x, name)
        if plug is not None:
            found.append(plug)
            if has is None:
                has = i
            continue
        if has is None:
            return None
        _raise_partial(x, i, name, has)
    if has is None:
        return None
    if gone:
        _warn_skipped(elements, gone)
    return _values_list(found)


def _refuse_partial(items: "List", name: str) -> None:
    """For an assignment `_private_plugs` found no broadcast for (its first
    element lacks `name`): raise as `_private_plugs` does for a later element
    when any element has `name`, naming the first live element, which lacks it
    (a deleted or freed element is skipped), and return when none has it
    (the caller raises: Node's '_' rule). So ``List([d, a]).__tag__ = 5``
    raises as ``List([a, d]).__tag__ = 5`` does, whatever the order. Every
    element is read: an assignment is no Python probe."""
    first = None
    skip  = _SKIP
    for i, x in enumerate(list.__iter__(items)):
        if x is skip or _gone(x):
            continue  # left alone, as every List operation leaves it
        if first is None:
            first = i
        elif _private_plug(x, name) is not None:
            _raise_partial(list.__getitem__(items, first), first, name, i)


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


# the method plugs a broadcast can gather (`List.__getattr__`)
_METHOD_PLUGS = (_MethodPlug, _MethodAttribute)


def _mixed_methods_text(name: str, plugs: list, count: int) -> str:
    """The refusal of ``List(items).<name>`` where elements `plugs` (indices) have
    a Maya attr `name` (their plugs) and the others do not (their methods)."""
    others = [k for k in range(count) if k not in plugs]
    def spell(ids: list) -> str:
        return ", ".join(f"[{k}]" for k in ids)
    return (
        f"List(...).{name}: {name} is a method of each element, and a method does not "
        f"broadcast over a List (plugs and values do); element(s) {spell(plugs)} have a "
        f"Maya attr {name!r} (their plugs), {spell(others)} do not (their methods): "
        f"give every element the attr for the plugs, or call the method on each one: "
        f"[x.{name}(...) for x in items]"
    )


class _Methods(List):
    """``List(items).<method>``: the bound method of each element. A method does not
    broadcast over a List (only plugs and values do): calling it raises a TypeError
    that names the loop. Iterating it, or indexing it, gives the methods."""

    def __init__(self, items: Iterable[Any], name: str) -> None:
        # the bound methods are kept as they are (`_values_list`), never cast
        object.__setattr__(self, "_parent_multi", None)
        if items is not None:
            _fill_values(self, items)
        object.__setattr__(self, "_method", name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name = self._method
        raise TypeError(
            f"List(...).{name}(...): {name} is a method of each element, and a method does "
            f"not broadcast over a List (plugs and values do); call it on each one: "
            f"[x.{name}(...) for x in items]"
        )

    def __bool__(self) -> bool:
        # a predicate of each element named without brackets (``if
        # List(plugs).is_locked:``) has no truth value, as an element's has none
        # (`_BarePredicate`); any other method List is a non-empty list
        name = self._method
        if _is_predicate_name(name):
            raise TypeError(
                f"List(...).{name} is a method of each element: call it on each one "
                f"([x.{name}() for x in items], all(...) / any(...) for one answer)"
            )
        return list.__len__(self) > 0


def _lift_or_pass(obj: Any) -> Any:
    """Convert a string into a Plug/Node; pass other types through."""
    if isinstance(obj, (Plug, Node, numbers.Real)) or obj is None:
        return obj
    if isinstance(obj, str):
        if "." in obj:
            return Plug(obj)
        return Node(obj)
    return obj


def _values_list(items: Optional[Iterable[Any]] = None, parent_multi: Any = None) -> "List":
    """[Internal] A List rig builds from what it computed: a broadcast
    (``nodes.tx``, ``nodes.t``), an operator's results, a slice, a
    ``multi[:]`` (with `parent_multi`), a vectorized call's or a command's
    results. Its elements are kept as they come (`_fill_values`: a str is
    still read as a name), never refused by `_user_items`' rule: a value
    (a vector tuple, a matrix, a bound method) stays a value."""
    out = list.__new__(List)
    object.__setattr__(out, "_parent_multi", parent_multi)
    if items is not None:  # (an array has no truth value)
        _fill_values(out, items)
    return out


def _rebuild(cls: type, items: list, state: dict) -> "List":
    """[Internal] `List.__reduce_ex__`'s rebuild (a copy, a deepcopy, an
    unpickle): a `cls` holding `items` as they are, with `state`."""
    out = list.__new__(cls)
    for name, value in state.items():
        object.__setattr__(out, name, value)
    list.extend(out, items)
    return out


def _fill_values(out: "List", items: Iterable[Any]) -> None:
    """Fill `out` with `items` as a List always did before the casting rule:
    numbers, None, plugs and nodes as they are, a component range str
    flattened with ``cmds.ls``, any other str lifted to its plug or node, and
    anything else kept."""
    append = list.append.__get__(out)  # list's own append: no cast
    for x in items:
        if isinstance(x, numbers.Real) or x is None:
            append(x)
        elif isinstance(x, (Plug, Node)):
            append(x)
        elif isinstance(x, str) and "." in x and ":" in x.split(".")[-1]:
            # Component / range selector (e.g. ``pCube1.vtx[0:5]``)
            # -- flatten via cmds.ls.
            for resolved in cmds.ls(x, fl=True) or []:
                append(_lift_or_pass(resolved))
        else:
            append(_lift_or_pass(x))


# the elements a List takes as they are: the rig objects every List holds and
# numbers (the numpy ones too: np.int32 is no int on Windows); `_user_items`
# checks the rest. `_PLAIN` is the exact types of the common ones, one set
# lookup (an isinstance miss costs a ``__class__`` read, and one on Node an
# ``__instancecheck__`` call: its class has a metaclass).
_PLAIN = frozenset((
    float, int, bool, type(None), np.float64, np.float32, np.int64, np.int32, np.bool_,
))
_KEPT = (Plug, Node, List, np.integer, np.floating, np.bool_, float, int)

# the containers a value comes in (a vector, a matrix, an operand row)
_SEQUENCES = (tuple, list, np.ndarray)

# the OpenMaya objects Node() / Plug() take
_API_NODES = (OpenMaya.MObject, OpenMaya.MDagPath)

# the OpenMaya values a plug takes or rig's API reads (a vector, a point, a
# matrix, a rotation, a colour, their arrays): kept as values, as a tuple is
_API_VALUES = tuple(
    getattr(OpenMaya, name) for name in (
        "MVector", "MFloatVector", "MPoint", "MFloatPoint", "MMatrix", "MFloatMatrix",
        "MTransformationMatrix", "MEulerRotation", "MQuaternion", "MColor", "MAngle",
        "MDistance", "MTime", "MBoundingBox",
        "MDoubleArray", "MFloatArray", "MIntArray", "MInt64Array", "MUintArray",
        "MUint64Array", "MPointArray", "MFloatPointArray", "MVectorArray",
        "MFloatVectorArray", "MMatrixArray", "MColorArray", "MTimeArray",
    ) if hasattr(OpenMaya, name)
)

# the leaves of a value (`_value`): a number, a plug (an operand row such as
# ``(c.tx, 1, 2)``), an OpenMaya value
_LEAVES = (numbers.Real, np.bool_, Plug) + _API_VALUES


def _user_items(items: Iterable[Any], start: Any) -> list:
    """The elements `items` gives a List (``List(items)``, ``append``,
    ``extend``, ``insert``, ``+=``, item and slice assignment), every one
    checked before any is added. `start` is the index the first one takes,
    named by an error.

    * a List: its elements as they are (a List holds what it was given or
      what rig computed);
    * a number (``numbers.Real``, bool, a numpy scalar), ``None``, a rig
      object (a node, a plug, components, a List, a membership token, an
      attribute spec, AnimCurve keys), an OpenMaya value (``MVector``,
      ``MMatrix`` ...): as it is;
    * a value, a tuple, list or array of numbers, plugs and OpenMaya values
      (nested too: a matrix; an operand row ``(c.tx, 1, 2)``): as it is;
    * a str: the plug (``"a.tx"``) or node it names, a component range
      (``"pCube1.vtx[0:5]"``) its elements; a name that names nothing raises
      as ``Node(name)`` / ``Plug(name)`` do, a range that names nothing is a
      TypeError;
    * an OpenMaya ``MPlug``: its Plug; an ``MObject`` / ``MDagPath``: its node;
    * any other object: the node of its ``uuid`` if it has one that names one
      node (cgmath data: a ``TransformData``), else what its ``str`` names,
      read as a str is (``str(TransformData)`` is its node's name).

    Anything else is a TypeError naming its index, its type and its str: a
    sequence holding anything else (a nested List is
    ``List([List([...]), ...])``), a set or a dict, a class or a function, an
    object whose ``str`` names no node or plug."""
    if isinstance(items, List):
        return list(items)
    out    = []
    append = out.append
    for i, x in enumerate(items):
        if type(x) in _PLAIN or isinstance(x, _KEPT):
            append(x)
        elif isinstance(x, str):
            if "." in x and ":" in x.split(".")[-1]:
                # Component / range selector (e.g. ``pCube1.vtx[0:5]``)
                # -- flatten via cmds.ls.
                found = cmds.ls(x, fl=True)
                if not found:
                    raise _element_error(x, _index(start, i), "it names nothing")
                for resolved in found:
                    append(_lift_or_pass(resolved))
            else:
                append(_lift_or_pass(x))
        elif isinstance(x, _SEQUENCES):
            if not _value(x):
                raise _element_error(x, _index(start, i), "a sequence holds numbers and plugs "
                                     "only (a nested List is List([List([...]), ...]))")
            append(x)
        elif (
            isinstance(x, (numbers.Real, _API_VALUES))
            or _is_components(x)
            or _is_membership(x)
            or _is_attribute_spec(x)
            or type(x) is _Keys
        ):
            append(x)
        else:
            out.extend(_cast_other(x, _index(start, i)))
    return out


def _index(start: Any, i: int) -> Any:
    """The index an error names: `start` plus `i` (`start` as given, for a
    one-element call such as ``insert(-1, x)``)."""
    return start + i if i else start


def _value(value: Any) -> bool:
    """True for a value a List keeps: an array of numbers or bools, a tuple
    or list of numbers, plugs and OpenMaya values (nested too: a matrix, an
    operand row ``(c.tx, 1, 2)``)."""
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biuf"
    for v in value:
        if type(v) is float or type(v) is int or isinstance(v, _LEAVES):
            continue
        if isinstance(v, _SEQUENCES) and not isinstance(v, List) and _value(v):
            continue
        return False
    return True


def _cast_other(x: Any, index: Any) -> list:
    """The elements an object that is no number, rig object or str gives a
    List (`_user_items`): an OpenMaya plug or node object through ``Plug()`` /
    ``Node()``; else the node of its ``uuid`` when that names one node (cgmath
    data: the node it was serialized from, renamed or not, whatever else has
    its name); else what its ``str`` names, read as a str is. A TypeError
    names it when nothing resolves."""
    if isinstance(x, OpenMaya.MPlug):
        if x.isNull:
            raise _element_error(x, index, "a null MPlug", "MPlug()")
        try:
            return [Plug(x)]
        except Exception as e:
            raise _element_error(x, index, _first_line(e)) from e
    if isinstance(x, _API_NODES):
        try:
            return [Node(x)]
        except Exception as e:
            raise _element_error(x, index, _first_line(e)) from e
    if isinstance(x, (set, frozenset, dict, type)) or callable(x):
        raise _element_error(x, index, "it has no node or plug")
    node = _node_of_uuid(x)
    if node is not None:
        return [node]
    try:
        text = str(x)
    except Exception as e:
        raise _element_error(x, index, f"its str raised {type(e).__name__}") from e
    try:
        if "." in text and ":" in text.split(".")[-1]:
            found = [_lift_or_pass(r) for r in cmds.ls(text, fl=True) or []]
        else:
            found = [_lift_or_pass(text)]
    except AmbiguousNodeError as e:
        # (a name more than one node has: the error names them)
        raise _element_error(x, index, _first_line(e), text) from e
    except Exception as e:
        raise _element_error(x, index, "its str names no node or plug", text) from e
    if not found:
        raise _element_error(x, index, "its str names no node or plug", text)
    return found


def _node_of_uuid(x: Any) -> Any:
    """The node of `x`'s ``uuid`` attribute (cgmath data carry the uuid of the
    node they were serialized from) when it is a uuid exactly one node of the
    scene has; else None (no uuid, or data from another scene: the name
    decides)."""
    try:
        uuid = getattr(x, "uuid", None)
    except Exception:
        return None
    if type(uuid) is not str or not _is_uuid(uuid):
        return None
    found = cmds.ls(uuid, long=True)
    if not found or len(found) != 1:
        return None
    return Node(found[0])


def _first_line(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


def _element_error(x: Any, index: Any, why: str, text: Optional[str] = None) -> TypeError:
    """The TypeError of an element a List does not take (`_user_items`)."""
    if text is None:
        try:
            text = str(x)
        except Exception:
            text = object.__repr__(x)
    if len(text) > 80:
        text = text[:77] + "..."
    return TypeError(
        f"List element [{index}] ({type(x).__name__} {text!r}) is no node or plug ({why}): "
        f"a List holds numbers and rig nodes / plugs, and casts a name (or an object whose "
        f"str is one) to its node or plug; nothing was added"
    )


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
    probe matches; another plug element is named (`List` passes no deleted
    one, `_unmatched_gone`) and
    matches when it is the same Maya plug (see `_same_plug`); any other element
    compares by name."""

    __slots__ = ("probe", "name", "mnode", "identity")

    def __init__(self, probe: Attribute) -> None:
        self.probe = probe
        self.name  = None

    def __call__(self, item: Any) -> bool:
        probe = self.probe
        if self.name is None:
            self.name     = str(probe)
            self.mnode    = probe.__dict__["_mplug"].node()
            self.identity = _plug_identity_name(probe)
        if item is probe:
            return True
        if isinstance(item, Attribute):
            str(item)
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
    by name (`List` passes no deleted element, `_unmatched_gone`)."""

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
            str(item)
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