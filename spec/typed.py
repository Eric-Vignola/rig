"""Typed attribute specs: ``String``, ``Matrix``, ``MeshAttr``, ``NurbsCurveAttr``,
``NurbsSurfaceAttr``, ``Message``, and ``Typed``, the attribute that holds any data.

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


# what a typed attribute that holds any data has no use for: Maya takes a default,
# a range, a colour use with a warning and does nothing with them, and refuses a
# dataType next to attributeType typed ("Cannot specify both")
_TYPED_REFUSED = (
    ("defaultValue", "dv"), ("minValue", "min"), ("maxValue", "max"),
    ("softMinValue", "smn"), ("softMaxValue", "smx"), ("hasMinValue", "hnv"),
    ("hasMaxValue", "hxv"), ("hasSoftMinValue", "hsn"), ("hasSoftMaxValue", "hsx"),
    ("enumName", "en"), ("usedAsColor", "uac"), ("dataType", "dt"),
    ("attributeType", "at"),
)


class Typed(_AttrSpec):
    """A typed attribute that holds any data (``addAttr -attributeType typed``,
    Maya's kAny data, as ``choice.input``): a passthrough slot.

    * ``slot << plug`` connects the plug whole, as it is (a matrix, a message, a
      number, a double3, a mesh ...), the old input disconnected: never a
      shorthand network (no composeMatrix) and never a per-channel fan-out,
      whatever the slot held before.
    * ``slot << number`` (an int, a float, a bool, a numpy scalar) sets a number.
    * Any other value (a str, a list, an ndarray, a node) is a TypeError before
      any edit: Maya keeps no other value in such an attribute.
    * ``slot << None`` disconnects it and drops the data its old input left,
      unless that is a number (a saved file would refuse it when opened again).
    * ``slot >> None`` is what it holds; the root of a multi gives one value per
      element, in index order.

    Not keyable unless ``keyable=True`` is given (a keyable one takes an
    animCurveTU key at every ``setKeyframe`` of its node). ``dv`` / ``min`` /
    ``max`` / ``smn`` / ``smx`` / every ``has*Value`` / ``en`` / ``uac`` /
    ``dt`` / ``at`` are a TypeError when the spec is made: it has no default,
    range, enum fields, colour use or data type of its own."""

    def __init__(self, name: str, **kargs: Any) -> None:
        for flags in _TYPED_REFUSED:
            given = [flag for flag in flags if flag in kargs]
            if given:
                raise TypeError(
                    f"Typed({name!r}): {given[0]}= is no setting of a typed attribute that "
                    f"holds any data (it has no default, range, enum fields, colour use or "
                    f"data type of its own); nothing was changed"
                )
        keyable_given = "keyable" in kargs or "k" in kargs
        super().__init__(name, **kargs)
        if not keyable_given:
            self.kargs["keyable"] = False
        self.kargs["attributeType"] = "typed"

    def _kind(self) -> tuple:
        # Maya's dataType query of such an attribute: 'Tdata' (a String attribute,
        # 'string', is another kind)
        at, _, multi, children = super()._kind()
        return at, "Tdata", multi, children
