"""Enum attribute spec.

File named ``enum_attr`` instead of ``enum`` to avoid shadowing the stdlib
``enum`` module. Public class is still imported as ``Enum``.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from rig.spec._base import _AttrSpec


class Enum(_AttrSpec):
    """Enum attribute.

    ``en`` (or ``enumName``) accepts either a colon-delimited string
    (``"red:green:blue"``) or a sequence (``["red", "green", "blue"]``).
    Default is ``"False:True:"`` (i.e. a 2-state enum).

    ``dv`` (or ``defaultValue``) takes a field name as well as its int
    (``Enum("mode", en="a:b:c", dv="b")`` is ``dv=1``), read as ``plug << "b"``
    reads one (see ``rig.nodetypes._base._enum_value``): a wrong name raises
    TypeError naming the fields when the spec is made.
    """

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "enum"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)

        en_value = self._pop_alias(
            self.kargs, ("enumName", "en"), default="False:True:"
        )
        if isinstance(en_value, (list, tuple, set)):
            en_value = ":".join(str(x) for x in en_value) + ":"
        self.kargs["en"] = en_value

        # a default given by field name is read against the declared fields
        for key in ("defaultValue", "dv"):
            value = self.kargs.get(key)
            if isinstance(value, str) and isinstance(en_value, str):
                from rig.nodetypes._base import _match_enum_field, _parse_enum_names

                self.kargs[key] = _match_enum_field(
                    _parse_enum_names(en_value), value, f"Enum {name!r} {key}"
                )