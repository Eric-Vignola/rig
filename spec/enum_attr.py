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