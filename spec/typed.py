"""Typed attribute specs: ``String``, ``Matrix``, ``MeshAttr``, ``NurbsCurveAttr``,
``NurbsSurfaceAttr``, ``Message``.

The geometry data specs carry an ``Attr`` suffix because the bare names are
the node classes (``rig.nodetypes.Mesh``, ``NurbsCurve``, ``NurbsSurface``):
no public name is both a node class and a declaration."""

from __future__ import annotations

from typing import Any

from rig.spec._base import _AttrSpec


class String(_AttrSpec):
    """String attribute (``dataType='string'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["dataType"] = "string"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)


class Matrix(_AttrSpec):
    """4x4 matrix attribute (``attributeType='matrix'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "matrix"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)


class MeshAttr(_AttrSpec):
    """Mesh data attribute (``dataType='mesh'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["dataType"] = "mesh"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)


class NurbsCurveAttr(_AttrSpec):
    """NURBS-curve data attribute (``dataType='nurbsCurve'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["dataType"] = "nurbsCurve"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)


class NurbsSurfaceAttr(_AttrSpec):
    """NURBS-surface data attribute (``dataType='nurbsSurface'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["dataType"] = "nurbsSurface"
        self.kargs.pop("dt",            None)
        self.kargs.pop("attributeType", None)
        self.kargs.pop("at",            None)


class Message(_AttrSpec):
    """Message attribute (``attributeType='message'``)."""

    def __init__(self, name: str, **kargs: Any) -> None:
        super().__init__(name, **kargs)
        self.kargs["attributeType"] = "message"
        self.kargs.pop("at",       None)
        self.kargs.pop("dataType", None)
        self.kargs.pop("dt",       None)