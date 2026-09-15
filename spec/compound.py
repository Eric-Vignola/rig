"""Compound attribute specs: ``Vector``, ``Quat``, ``Color``, ``Euler``."""

from __future__ import annotations

from typing import Any

from rig.spec.numeric import Float


class Vector(Float):
    """3-channel ``double3`` (children: ``X``, ``Y``, ``Z``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.compound = ["X", "Y", "Z"]


class Quat(Float):
    """4-channel ``double4`` (children: ``X``, ``Y``, ``Z``, ``W``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.compound = ["X", "Y", "Z", "W"]


class Color(Float):
    """3-channel ``double3`` (children: ``R``, ``G``, ``B``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.compound = ["R", "G", "B"]


class Euler(Float):
    """3-channel ``double3`` with each child as ``doubleAngle``."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.compound     = ["X", "Y", "Z"]
        self.compoundType = "doubleAngle"