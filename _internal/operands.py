"""
Operand checks for the DSL's node-building entry points.

A DSL operand is a Plug (or a typed Attribute), a number, ``None``, or a
sequence of them. A *plain* str is not one. ``Plug`` and ``Attribute``
subclass ``str``, so "plain" means a str that is not an Attribute. Without a
check, ``t.tx == "cube.ty"`` builds an ``equal`` node, fails to set the
string on it and raises ``InjectionError``, and the node stays in the scene.
So every Plug / PlugList operator and every public math function raises a
``TypeError`` for a plain str operand before it creates anything. Write
``Plug("cube.ty")`` for the plug of a name, and ``str(plug)`` or an
f-string for text.

A config string is not an operand. The ``"<"`` of a condition op (a NodeOp
takes its positional strs as config) and the ``side=`` / ``name=`` /
``dtype=`` / ``axis=`` / ``method=`` choices are config: :func:`operands`
lists them per function and passes them through. So is ``rotate_order``,
for now: it takes 0-5 or a plug, a rotate-order name like ``"xyz"`` never
worked, and it keeps failing as before, in the node it builds.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Iterable, Optional

import numpy as np
from rig.nodetypes._base import Attribute


# The operand types that can be, or can hold, a plain str. Numbers, None,
# Nodes and Plugs are rejected by one isinstance call.
_CAN_HOLD_STR = (str, list, tuple, np.ndarray)

# Longest rendering of an operand in a message before it is shortened to
# its type name.
_MAX_RENDERED = 60


def _plain_str(obj: Any) -> Optional[str]:
    """The first plain str in ``obj``, or None.

    ``obj`` is checked itself, and so are the elements of a list, tuple or
    PlugList and the elements of a numpy array, nested to any depth. A str
    that is an Attribute (a Plug) is not plain. A numpy str array holds
    ``np.str_`` elements, which are plain strs.
    """
    if isinstance(obj, str):
        return None if isinstance(obj, Attribute) else obj
    if isinstance(obj, (list, tuple)):
        for element in obj:
            if isinstance(element, _CAN_HOLD_STR):
                found = _plain_str(element)
                if found is not None:
                    return found
        return None
    if isinstance(obj, np.ndarray):
        kind = obj.dtype.kind
        if kind == "U":
            return str(obj.flat[0]) if obj.size else None
        if kind == "O":
            for element in obj.flat:
                if isinstance(element, _CAN_HOLD_STR):
                    found = _plain_str(element)
                    if found is not None:
                        return found
    return None


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


def str_operand_error(where: str, found: str, text_hint: bool = False) -> TypeError:
    """The TypeError for the plain str ``found`` given as an operand ``where``
    (``"t.translateX == 'cube.ty'"``, ``"rig.vector.lerp() argument 'input2'"``).

    ``text_hint`` adds the str(plug) / f-string hint, for ``+`` whose str
    operand is usually meant as text.
    """
    message = (
        f"{where}: {found!r} is a plain str, and a DSL operand is a Plug, a number "
        f"or a sequence of them."
    )
    if _looks_like_a_plug_name(found):
        message += f" Write Plug({found!r}) for the plug of that name."
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


# Plug / PlugList operator dunder -> (symbol, reflected). The comparisons have
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


def operator_error(
    dunder: str, left: Any, right: Any, found: str, row: Optional[int] = None
) -> TypeError:
    """The TypeError for ``left <dunder> right``, whose ``right`` operand is,
    or holds, the plain str ``found``.

    ``left`` is the Plug the dunder ran on; a reflected dunder renders its
    ``right`` first, as it was written (``'%s' % plug``). ``row`` is the row
    of a PlugList operator the pair came from. Naming ``left`` raises for a
    plug whose node was deleted, as every other operator on it does.
    """
    symbol, reflected = OPERATOR_SYMBOLS[dunder]
    plug = str(left)
    if dunder == "__rmod__" and right is found and row is None:
        return text_format_error(found, plug)
    if reflected:
        where = f"{_render(right)} {symbol} {plug}"
    else:
        where = f"{plug} {symbol} {_render(right)}"
    if row is not None:
        where = f"PlugList row {row}, {where}"
    return str_operand_error(where, found, text_hint=symbol == "+")


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

    ``config`` names the parameters that take config values (a str there is
    a choice, not an operand, and is passed through unchecked). ``skip_when``
    is a ``predicate(args, kwargs)`` that is True for a call that builds no
    node and returns its arguments as they are (``condition(1, "yes", "no")``
    picks ``"yes"`` in Python); such a call is not checked. Put ``@operands``
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

    def _argument(i: int) -> str:
        if i < count:
            return names[i]
        return f"{var_args}[{i - count}]"

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if skip_when is None or not skip_when(args, kwargs):
            for i, value in enumerate(args):
                if isinstance(value, _CAN_HOLD_STR) and i not in skipped:
                    found = _plain_str(value)
                    if found is not None:
                        raise str_operand_error(
                            f"{label}() argument {_argument(i)!r}", found
                        )
            for key, value in kwargs.items():
                if isinstance(value, _CAN_HOLD_STR) and key not in config:
                    found = _plain_str(value)
                    if found is not None:
                        raise str_operand_error(f"{label}() argument {key!r}", found)
        return func(*args, **kwargs)

    wrapper._operand_config = config
    return wrapper
