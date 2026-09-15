"""Numeric attribute specs: ``Float``, ``Int``, ``Bool``, ``Angle``, ``Time``."""

from __future__ import annotations

from typing import Any

from rig.spec._base import _AttrSpec


class Int(_AttrSpec):
    """Integer attribute (``attributeType='long'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "long"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)


class Angle(_AttrSpec):
    """Angle attribute (``attributeType='doubleAngle'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "doubleAngle"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)


class Float(_AttrSpec):
    """Floating-point attribute (``attributeType='double'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "double"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)


class Bool(_AttrSpec):
    """Boolean attribute (``attributeType='bool'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "bool"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)


class Time(_AttrSpec):
    """Time attribute (``dataType='time'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["dataType"] = "time"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)