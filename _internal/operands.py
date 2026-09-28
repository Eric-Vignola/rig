"""
Operand checks for the DSL's node-building entry points.

A DSL operand is a Plug (or a typed Attribute), a number, ``None``, or a
sequence of them. A *plain* str is not one. ``Plug`` and ``Attribute``
subclass ``str``, so "plain" means a str that is not an Attribute. Without a
check, ``t.tx == "cube.ty"`` builds an ``equal`` node, fails to set the
string on it and raises ``InjectionError``, and the node stays in the scene.
So every Plug / List operator and every public math function raises a
``TypeError`` for a plain str operand before it creates anything. Write
``Plug("cube.ty")`` for the plug of a name, and ``str(plug)`` or an
f-string for text.

bytes (a ``bytearray``, a numpy bytes array) are text too, and are rejected
the same way: an operand of bytes injected its first byte (``b"cube.ty"`` set
99.0).

The same checks raise a plug operand's ``"... already deleted!"`` when a new
scene, a file open or a reference unload freed its node (see
``_ensure_owner_alive``): the type predicates an operator or a function starts
with read its operands' MPlugs, which then point at freed memory.

An operand sequence is ordered. A set, a frozenset or a dict operand raises
TypeError before anything is built (``t.tx + {1, 2}``, ``functions.sum({a.tx,
b.tx})``, ``t.t + {"x": 1}``): pass a list, ``sorted(...)`` for a set. An
iterator operand (a generator, ``map``, ``zip``, ``iter(...)``, an itertools
object) is read into a list first, and then checked and used as that list:
``rig.functions.sum(x.tx for x in ctrls)`` is
``rig.functions.sum([x.tx for x in ctrls])``, one memo entry. Other iterables
(a dict view, a ``range``) are passed as they are.

A config string is not an operand. The ``"<"`` of a condition op (a NodeOp
takes its positional strs as config) and the ``side=`` / ``name=`` /
``dtype=`` / ``axis=`` / ``method=`` choices are config: :func:`operands`
lists them per function and passes them through.

A rotate order (``rotate_order``, and ``rotate_order0`` / ``rotate_order1`` of
``rig.euler.reorder``) is config too, and takes an int 0-5, a plug, or one of
the six names of Maya's ``rotateOrder`` enum: ``"xyz"`` (0), ``"yzx"`` (1),
``"zxy"`` (2), ``"xzy"`` (3), ``"yxz"`` (4), ``"zyx"`` (5). :func:`operands`
replaces a name by its int before the function runs, so ``rotate_order="zxy"``
and ``rotate_order=2`` build (and memoize) one network, and any other str
(``"XYZ"``, ``""``, ``"xy"``) raises TypeError before any node or container is
created. A list or tuple of rotate orders (a broadcast call) is mapped element
by element.
"""

from __future__ import annotations

import collections.abc
import functools
import inspect
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
from rig.nodetypes._base import _ensure_owner_alive, Attribute


# The operand types that can be, or can hold, a plain str (or bytes). Numbers,
# None, Nodes and Plugs are rejected by one isinstance call.
_CAN_HOLD_STR = (str, bytes, bytearray, list, tuple, np.ndarray)

# Longest rendering of an operand in a message before it is shortened to
# its type name.
_MAX_RENDERED = 60

# The operand types that are never reshaped (see `_prepared_operand`): numbers
# and None, passed by one isinstance call.
_SCALARS = (int, float, np.generic, type(None))

# The unordered collections an operand cannot be, and the operand types
# `_prepared_operand` refuses or reads into a list.
_UNORDERED = (set, frozenset, dict)
_RESHAPED  = (set, frozenset, dict, collections.abc.Iterator)

# The rotate-order names, as Maya's ``rotateOrder`` enum numbers them, and the
# config parameters that take a rotate order (see the module docstring).
_ROTATE_ORDERS = {"xyz": 0, "yzx": 1, "zxy": 2, "xzy": 3, "yxz": 4, "zyx": 5}
_ROTATE_ORDER_PARAMS = frozenset({"rotate_order", "rotate_order0", "rotate_order1"})


def _first_text(obj: Any, with_bytes: bool) -> Optional[Any]:
    """The first plain str in ``obj`` (or bytes, ``with_bytes``), or None.

    ``obj`` is checked itself, and so are the elements of a list, tuple or
    List and the elements of a numpy array, nested to any depth. A str
    that is an Attribute (a Plug) is not plain. A numpy str array holds
    ``np.str_`` elements, which are plain strs; a numpy bytes array holds
    ``np.bytes_`` elements.
    """
    if isinstance(obj, str):
        return None if isinstance(obj, Attribute) else obj
    if isinstance(obj, (bytes, bytearray)):
        return obj if with_bytes else None
    if isinstance(obj, (list, tuple)):
        for element in obj:
            if isinstance(element, _CAN_HOLD_STR):
                found = _first_text(element, with_bytes)
                if found is not None:
                    return found
        return None
    if isinstance(obj, np.ndarray):
        kind = obj.dtype.kind
        if kind == "U":
            return str(obj.flat[0]) if obj.size else None
        if kind == "S" and with_bytes:
            return bytes(obj.flat[0]) if obj.size else None
        if kind == "O":
            for element in obj.flat:
                if isinstance(element, _CAN_HOLD_STR):
                    found = _first_text(element, with_bytes)
                    if found is not None:
                        return found
    return None


def _plain_str(obj: Any) -> Optional[str]:
    """The first plain str in ``obj``, or None (see `_first_text`)."""
    return _first_text(obj, False)


def _text_operand(obj: Any) -> Optional[Any]:
    """The first plain str or bytes in ``obj``, or None: what an operand check
    rejects (see `_first_text`)."""
    return _first_text(obj, True)


def _live_text_operand(obj: Any) -> Optional[Any]:
    """`_text_operand(obj)`, raising the node's ``"... already deleted!"`` first
    for a plug in ``obj`` (``obj`` itself, or one in a list, tuple, List or
    object array, nested to any depth, before the first plain str) whose node
    was freed (see `_ensure_owner_alive`): the type predicates a function
    starts with read its operands' MPlugs, which then point at freed memory."""
    if isinstance(obj, str):
        if isinstance(obj, Attribute):
            _ensure_owner_alive(obj)
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        elements = obj
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        elements = obj.flat
    else:
        return _first_text(obj, True)
    for element in elements:
        if isinstance(element, _CAN_HOLD_STR):
            found = _live_text_operand(element)
            if found is not None:
                return found
    return None


def _prepared_operand(value: Any, where: str) -> Any:
    """``value``, one of the `_RESHAPED` operand types, given as an operand
    ``where`` (``"rig.functions.sum() argument 'tokens'"``,
    ``"t.translateX + {1, 2}"``), as a DSL operand: a set, a frozenset or a
    dict raises TypeError, an iterator is read into a list (see the module
    docstring). Anything else is returned as it is."""
    if isinstance(value, _UNORDERED):
        raise unordered_operand_error(where, value)
    if isinstance(value, collections.abc.Iterator):
        return list(value)
    return value


def unordered_operand_error(where: str, value: Any) -> TypeError:
    """The TypeError for the set, frozenset or dict ``value`` given as an operand
    ``where``."""
    if isinstance(value, dict):
        what, hint = "a dict is a mapping, not a sequence", "list(d.values()) for its values"
    else:
        kind = "frozenset" if isinstance(value, frozenset) else "set"
        what, hint = f"a {kind} is unordered", f"sorted(...) for a {kind}"
    return TypeError(
        f"{where}: {what}, and a DSL operand is a Plug, a number or a sequence of "
        f"them; pass a list ({hint})"
    )


def _render(obj: Any) -> str:
    """A plug's name or ``repr(obj)`` for a message, or the type name when that
    is long or cannot be read."""
    try:
        text = str(obj) if isinstance(obj, Attribute) else repr(obj)
    except Exception:  # noqa: BLE001 -- a message must not fail
        text = ""
    if not text or len(text) > _MAX_RENDERED:
        return f"<{type(obj).__name__}>"
    return text


def _looks_like_a_plug_name(text: str) -> bool:
    """True for ``"node.attr"``-like text, the one a Plug hint helps with."""
    return "." in text and "%" not in text and "{" not in text


def str_operand_error(where: str, found: Any, text_hint: bool = False) -> TypeError:
    """The TypeError for the plain str (or bytes) ``found`` given as an operand
    ``where`` (``"t.translateX == 'cube.ty'"``, ``"rig.vector.lerp() argument
    'input2'"``).

    ``text_hint`` adds the str(plug) / f-string hint, for ``+`` whose str
    operand is usually meant as text.
    """
    what = "a plain str"
    text = found
    if isinstance(found, (bytes, bytearray)):
        what = "bytes"
        text = bytes(found).decode("utf-8", "replace")
    message = (
        f"{where}: {found!r} is {what}, and a DSL operand is a Plug, a number "
        f"or a sequence of them."
    )
    if _looks_like_a_plug_name(text):
        message += f" Write Plug({text!r}) for the plug of that name."
    else:
        message += " Wrap a plug name with Plug('node.attr')."
    if text_hint:
        message += " For text, write str(plug) or an f-string."
    return TypeError(message)


def text_format_error(fmt: str, plug: str) -> TypeError:
    """The TypeError for ``fmt % plug`` (``plug`` is the Plug's name): ``%``
    with a Plug on the right is a modulo node, not text formatting."""
    message = (
        f"{_render(fmt)} % {plug}: '%' with a Plug on the right builds a modulo "
        f"node, it does not format text. For text, write str(plug) or an f-string "
        f"(f\"...{{plug}}\")."
    )
    if _looks_like_a_plug_name(fmt):
        message += f" For the modulo of the plug of that name, write Plug({fmt!r}) % plug."
    return TypeError(message)


# Plug / List operator dunder -> (symbol, reflected). The comparisons have
# no reflected form: Python reflects them onto the opposite comparison.
OPERATOR_SYMBOLS = {
    "__add__": ("+", False),       "__radd__": ("+", True),
    "__sub__": ("-", False),       "__rsub__": ("-", True),
    "__mul__": ("*", False),       "__rmul__": ("*", True),
    "__truediv__": ("/", False),   "__rtruediv__": ("/", True),
    "__pow__": ("**", False),      "__rpow__": ("**", True),
    "__floordiv__": ("//", False), "__rfloordiv__": ("//", True),
    "__mod__": ("%", False),       "__rmod__": ("%", True),
    "__and__": ("&", False),       "__rand__": ("&", True),
    "__or__": ("|", False),        "__ror__": ("|", True),
    "__xor__": ("^", False),       "__rxor__": ("^", True),
    "__eq__": ("==", False),       "__ne__": ("!=", False),
    "__ge__": (">=", False),       "__le__": ("<=", False),
    "__gt__": (">", False),        "__lt__": ("<", False),
}


# The dunder Python runs on the right operand of ``a <dunder> b`` when ``b`` is
# the Plug (``"x" + plug`` runs ``plug.__radd__``, ``"x" < plug`` runs
# ``plug.__gt__``).
_COMPARISONS = {
    "__eq__": "__eq__", "__ne__": "__ne__",
    "__ge__": "__le__", "__le__": "__ge__",
    "__gt__": "__lt__", "__lt__": "__gt__",
}
REFLECTED = {
    **{
        name: (name[:2] + name[3:] if name.startswith("__r") else "__r" + name[2:])
        for name in OPERATOR_SYMBOLS
        if name not in _COMPARISONS
    },
    **_COMPARISONS,
}


def operator_where(dunder: str, left: str, right: Any, row: Optional[int] = None) -> str:
    """``left <dunder> right`` as it was written, for a message: ``left`` is the
    name (or rendering) of the operand the dunder ran on, and a reflected dunder
    renders its ``right`` first (``'%s' % plug``). ``row`` is the row of a
    List operator the pair came from."""
    symbol, reflected = OPERATOR_SYMBOLS[dunder]
    if reflected:
        where = f"{_render(right)} {symbol} {left}"
    else:
        where = f"{left} {symbol} {_render(right)}"
    if row is not None:
        where = f"PlugList row {row}, {where}"
    return where


def operator_error(
    dunder: str, left: Any, right: Any, found: str, row: Optional[int] = None
) -> TypeError:
    """The TypeError for ``left <dunder> right``, whose ``right`` operand is,
    or holds, the plain str ``found``.

    ``left`` is the Plug the dunder ran on (see `operator_where`). Naming
    ``left`` raises for a plug whose node was deleted, as every other operator
    on it does.
    """
    plug = str(left)
    if dunder == "__rmod__" and right is found and row is None and isinstance(found, str):
        return text_format_error(found, plug)
    where = operator_where(dunder, plug, right, row)
    return str_operand_error(where, found, text_hint=OPERATOR_SYMBOLS[dunder][0] == "+")


def _rotate_order(label: str, name: str, value: Any) -> Any:
    """``value`` given to the rotate-order parameter ``name`` of the function
    ``label``, a rotate-order name replaced by its int. Any other plain str
    raises TypeError; an int, a plug (an Attribute), None or anything else is
    returned as it is. A list or tuple is mapped element by element, one level
    (a broadcast call), and returned as it is when it holds no name."""
    if isinstance(value, str):
        if isinstance(value, Attribute):
            return value
        order = _ROTATE_ORDERS.get(value)
        if order is None:
            raise TypeError(
                f"{label}() argument {name!r}: {value!r} is not a rotate order; use "
                f"'xyz', 'yzx', 'zxy', 'xzy', 'yxz' or 'zyx' (0-5), an int 0-5 or a plug"
            )
        return order
    if isinstance(value, (list, tuple)) and any(
        isinstance(element, str) and not isinstance(element, Attribute)
        for element in value
    ):
        mapped = [
            _rotate_order(label, f"{name}[{j}]", element)
            if isinstance(element, str)
            else element
            for j, element in enumerate(value)
        ]
        if type(value) is list:
            return mapped
        try:
            return type(value)(mapped)
        except Exception:  # noqa: BLE001 -- a list subclass that takes no items
            return mapped
    return value


def _rotate_order_arguments(
    label:     str,
    args:      tuple,
    kwargs:    dict,
    positions: Tuple[Tuple[int, str], ...],
    keywords:  Tuple[str, ...],
) -> Tuple[tuple, dict]:
    """``args`` / ``kwargs`` of a call of ``label``, the rotate-order names of
    the parameters at ``positions`` (``(index, name)`` pairs) and ``keywords``
    replaced by their ints (see `_rotate_order`). A tuple is rebuilt, and a
    value replaced, only where a name was mapped."""
    for i, name in positions:
        if i < len(args):
            value = args[i]
            if isinstance(value, (str, list, tuple)):
                mapped = _rotate_order(label, name, value)
                if mapped is not value:
                    args = (*args[:i], mapped, *args[i + 1:])
    if kwargs:
        for key in keywords:
            value = kwargs.get(key)
            if isinstance(value, (str, list, tuple)):
                mapped = _rotate_order(label, key, value)
                if mapped is not value:
                    kwargs[key] = mapped
    return args, kwargs


def _public_name(func: Callable[..., Any]) -> str:
    """``rig.vector.lerp`` for ``func``: its module, where a private module
    (``rig._dispatch``, ``rig._internal.math_nodes``) re-exports from ``rig``."""
    module = func.__module__ or "rig"
    if any(part.startswith("_") for part in module.split(".")):
        module = module.split(".", 1)[0]
    return f"{module}.{func.__name__}"


def operands(
    func:      Optional[Callable[..., Any]]                 = None,
    *,
    config:    Iterable[str]                                = (),
    skip_when: Optional[Callable[[tuple, dict], bool]]      = None,
) -> Any:
    """Declare a public DSL function's parameters as operands: a plain str
    argument raises TypeError before the function runs, so before it builds
    any node or container (see the module docstring).

    Every other argument is an operand: a set, frozenset or dict there raises
    TypeError, and an iterator is read into a list first (see the module
    docstring).

    ``config`` names the parameters that take config values (a str there is
    a choice, not an operand, and is passed through unchecked). A rotate-order
    parameter among them (``rotate_order``, ``rotate_order0``,
    ``rotate_order1``) takes a rotate-order name, replaced by its int before
    the function runs; another str there raises TypeError. ``skip_when``
    is a ``predicate(args, kwargs)`` that is True for a call that builds no
    node and returns its arguments as they are (``condition(1, "yes", "no")``
    picks ``"yes"`` in Python); it reads the arguments once iterators are read
    into lists (and sets / dicts refused), and such a call is not checked. Put ``@operands``
    above ``@vectorize``, so every row of a broadcast is checked before the
    first row builds.

    Examples::

        @operands
        @vectorize
        @memoize
        def lerp(input1, input2, weight=0.5): ...

        @operands(config=("return_index", "side"))
        @memoize
        def searchsorted(tokens, query, return_index=True, side="left"): ...
    """
    if func is None:
        return lambda f: operands(f, config=config, skip_when=skip_when)

    config     = frozenset(config)
    parameters = list(inspect.signature(func).parameters.values())
    names      = [
        p.name
        for p in parameters
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    unknown = config.difference(p.name for p in parameters)
    if unknown and not any(p.kind == p.VAR_KEYWORD for p in parameters):
        raise TypeError(f"operands: {func.__name__}() has no parameter {sorted(unknown)}")
    var_args   = next((p.name for p in parameters if p.kind == p.VAR_POSITIONAL), None)
    skipped    = frozenset(i for i, name in enumerate(names) if name in config)
    count      = len(names)
    label      = _public_name(func)
    # the rotate-order parameters, whose names are mapped to their ints
    rotate_keywords  = tuple(sorted(config & _ROTATE_ORDER_PARAMS))
    rotate_positions = tuple(
        (i, name) for i, name in enumerate(names) if name in rotate_keywords
    )

    def _argument(i: int) -> str:
        if i < count:
            return names[i]
        return f"{var_args}[{i - count}]"

    def _prepared_arguments(args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """``args`` / ``kwargs`` with every operand prepared (see
        `_prepared_operand`), for a `skip_when` predicate to read."""
        for i, value in enumerate(args):
            if (
                not isinstance(value, _CAN_HOLD_STR)
                and not isinstance(value, _SCALARS)
                and i not in skipped
                and isinstance(value, _RESHAPED)
            ):
                value = _prepared_operand(value, f"{label}() argument {_argument(i)!r}")
                args  = (*args[:i], value, *args[i + 1:])
        for key, value in kwargs.items():
            if (
                not isinstance(value, _CAN_HOLD_STR)
                and not isinstance(value, _SCALARS)
                and key not in config
                and isinstance(value, _RESHAPED)
            ):
                kwargs[key] = _prepared_operand(value, f"{label}() argument {key!r}")
        return args, kwargs

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if skip_when is not None:
            args, kwargs = _prepared_arguments(args, kwargs)
            if skip_when(args, kwargs):
                if rotate_keywords:
                    args, kwargs = _rotate_order_arguments(
                        label, args, kwargs, rotate_positions, rotate_keywords
                    )
                return func(*args, **kwargs)
        # A set / frozenset / dict operand raises and an iterator is read into
        # a list (see `_prepared_operand`), then checked as that list. A plug
        # operand whose node was freed raises its "already deleted!", as it
        # does in a Plug operator (see `_live_text_operand`); so does a plug
        # given to a config parameter (a rotate order).
        for i, value in enumerate(args):
            if not isinstance(value, _CAN_HOLD_STR):
                if (
                    isinstance(value, _SCALARS)
                    or i in skipped
                    or not isinstance(value, _RESHAPED)
                ):
                    continue
                value = _prepared_operand(value, f"{label}() argument {_argument(i)!r}")
                args  = (*args[:i], value, *args[i + 1:])
            if i not in skipped:
                found = _live_text_operand(value)
                if found is not None:
                    raise str_operand_error(f"{label}() argument {_argument(i)!r}", found)
            elif isinstance(value, Attribute):
                _ensure_owner_alive(value)
        for key, value in kwargs.items():
            if not isinstance(value, _CAN_HOLD_STR):
                if (
                    isinstance(value, _SCALARS)
                    or key in config
                    or not isinstance(value, _RESHAPED)
                ):
                    continue
                kwargs[key] = value = _prepared_operand(value, f"{label}() argument {key!r}")
            if key not in config:
                found = _live_text_operand(value)
                if found is not None:
                    raise str_operand_error(f"{label}() argument {key!r}", found)
            elif isinstance(value, Attribute):
                _ensure_owner_alive(value)
        if rotate_keywords:
            # after the operand checks, so a freed plug raises first
            args, kwargs = _rotate_order_arguments(
                label, args, kwargs, rotate_positions, rotate_keywords
            )
        return func(*args, **kwargs)

    wrapper._operand_config = config
    return wrapper
