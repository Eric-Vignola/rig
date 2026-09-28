"""Enum attribute spec.

File named ``enum_attr`` instead of ``enum`` to avoid shadowing the stdlib
``enum`` module. Public class is still imported as ``Enum``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from rig.spec._base import _AttrSpec


class Enum(_AttrSpec):
    """Enum attribute.

    ``en`` (or ``enumName``) names the fields, in order (a field's value is
    its index unless one is given):

    * a colon-delimited string, used as written: ``"red:green:blue"``, or with
      values ``"red=1:green=5"``;
    * a list or a tuple of names: ``["red", "green", "blue"]`` (each name is a
      piece of that string, so ``"b=10"`` works too);
    * a dict of name -> value, in its order: ``{"red": 1, "green": 5}``, or a
      list of ``(name, value)`` pairs: ``[("red", 1), ("green", 5)]``; both are
      ``"red=1:green=5"``, and names and pairs mix (``["red", ("green", 5)]``
      is ``"red:green=5"``). A name is a non-empty str without ``:`` or ``=``;
      a value is an int (not a bool).

    A set or a frozenset raises TypeError (it has no order, so the fields'
    values would be arbitrary), as do any other type, an empty list or dict,
    and an element that is neither a name nor a ``(name, value)`` pair: all
    when the spec is made, before any edit. Default is ``"False:True:"``
    (i.e. a 2-state enum).

    ``dv`` (or ``defaultValue``) takes a field name as well as its int, with
    every ``en`` form (``Enum("mode", en="a:b:c", dv="b")`` is ``dv=1``,
    ``Enum("mode", en={"a": 1, "b": 5}, dv="b")`` is ``dv=5``), read as
    ``plug << "b"`` reads one (see ``rig.nodetypes._base._enum_value``): a
    wrong name raises TypeError naming the fields when the spec is made.
    """

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "enum"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)

        en_value = _enum_names(name, self._pop_alias(
            self.kargs, ("enumName", "en"), default="False:True:"
        ))
        self.kargs["en"] = en_value

        # a default given by field name is read against the declared fields
        for key in ("defaultValue", "dv"):
            value = self.kargs.get(key)
            if isinstance(value, str):
                from rig.nodetypes._base import _match_enum_field, _parse_enum_names

                self.kargs[key] = _match_enum_field(
                    _parse_enum_names(en_value), value, f"Enum {name!r} {key}"
                )


def _enum_names(name: str, en: Any) -> str:
    """The ``-enumName`` string of the `en` given to ``Enum(name, en=...)``, in
    the forms :class:`Enum` lists; TypeError for anything else. No Maya call.

    A str is used as written. A list or a tuple of names alone keeps its
    historical join, trailing ``:`` included (``["a", "b"]`` is ``"a:b:"``;
    the attribute clone of ``_spec_from_attribute`` passes such a list); one
    with a ``(name, value)`` pair among its elements, and a dict, join without
    it (``"a=1:b=5"``)."""
    if isinstance(en, str):
        return en
    where = f"Enum {name!r}: en="
    if isinstance(en, (set, frozenset)):
        raise TypeError(
            f"{where} takes an ordered list (['red', 'green']), a dict "
            f"({{'red': 1, 'green': 5}}) or a 'red:green' string; a set has no "
            f"order, so the field indices would be arbitrary"
        )
    if isinstance(en, dict):
        pieces = [_enum_pair(where, item) for item in en.items()]
        pairs  = True
    elif isinstance(en, (list, tuple)):
        pieces = [
            element if isinstance(element, str) else _enum_pair(where, element)
            for element in en
        ]
        pairs = not all(isinstance(element, str) for element in en)
    else:
        raise TypeError(
            f"{where} takes a 'red:green' string, a list of names or of "
            f"(name, value) pairs, or a dict of name -> value; got "
            f"{type(en).__name__} {en!r}"
        )
    if not pieces:
        raise TypeError(f"{where} {en!r} names no field; name at least one")
    return ":".join(pieces) if pairs else ":".join(pieces) + ":"


def _enum_pair(where: str, pair: Any) -> str:
    """``"name=value"`` for one ``(name, value)`` pair of an ``en=`` (a dict
    item, or a 2-item tuple or list); TypeError naming `pair` otherwise."""
    if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
        raise TypeError(
            f"{where} element {pair!r} is neither a field name (a str) nor a "
            f"(name, value) pair"
        )
    field, value = pair
    if not isinstance(field, str) or not field or ":" in field or "=" in field:
        raise TypeError(
            f"{where} element {pair!r}: a field name is a non-empty str without "
            f"':' or '='; got {field!r}"
        )
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{where} element {pair!r}: the value of field {field!r} is an int "
            f"(not a bool); got {type(value).__name__} {value!r}"
        )
    return f"{field}={int(value)}"
