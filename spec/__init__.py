"""
The ``rig.spec`` sub-package -- declarative attribute specifications
for the rig DSL.

Usage::

    from rig import Node
    from rig.spec import Float, Vector, Enum, lock, hide

    node = Node("ctrl")
    node << Float("weight", min=0, max=1) << 0.5 << lock
    node << Vector("aim")
    node << Enum("mode", en=["off", "on", "auto"])
    node << Float("blend", multi=True)
"""

from __future__ import annotations

from rig.spec.compound import Color, Euler, Quat, Vector
from rig.spec.enum_attr import Enum
from rig.spec.modifiers import destroy, hide, lock, Note, skip, unhide, unlock
from rig.spec.numeric import Angle, Bool, Float, Int, Time
from rig.spec.typed import (
    Matrix,
    Mesh,
    Message,
    NurbsCurve,
    NurbsSurface,
    String,
)


__all__ = [
    # Numeric
    "Angle",
    "Bool",
    "Float",
    "Int",
    "Time",
    # Compound
    "Color",
    "Euler",
    "Quat",
    "Vector",
    # Typed
    "Matrix",
    "Mesh",
    "Message",
    "NurbsCurve",
    "NurbsSurface",
    "String",
    # Enum
    "Enum",
    # Modifiers
    "Note",
    "destroy",
    "hide",
    "lock",
    "skip",
    "unhide",
    "unlock",
]