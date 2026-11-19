"""rig's working units: centimetres and degrees, Maya's defaults (any frame rate).

rig's math is built and checked in them. In a scene set to others, some of it gives other
values than it does there:

* radians: ``rig.fuse()``'s nodes read and write angles in degrees, where rig's network
  works in the scene's unit, so ``node1.tx + node2.rx`` differs between the two;
* a linear unit other than cm, on Maya 2024 to 2026: those Mayas' math nodes (``sum``,
  ``multiply``, ``divide``, ``lerp`` ...) take distances and compute in centimetres, so a
  product, a quotient or a multiple of distances in rig's network is off (``a.tx * 2`` with
  ``a.tx`` at 1 m gives 200 m); fused math, and Maya 2027's nodes, give 2 m.

So rig warns once in such a scene, at its first statement (`check`, from
`rig._internal.journal.run_statement`), and again once a scene is made or opened or its
units change (Maya's ``linearUnitChanged`` / ``angularUnitChanged`` events).
"""

from __future__ import annotations

import sys
from typing import Any

from maya import cmds
from maya.api import OpenMaya

# whether this scene's units were checked since it was made or opened, or its units
# changed (a list: rig's statement entry reads it at every statement)
CHECKED: list = [False]

_THIS_MODULE = sys.modules[__name__]


def check() -> None:
    """Warns when this scene's units are not cm and degrees; then not again until a scene
    is made or opened or the units change. Never raises: a check never breaks a
    statement."""
    CHECKED[0] = True
    try:
        _ensure_callbacks()
        if (OpenMaya.MDistance.uiUnit() == OpenMaya.MDistance.kCentimeters
                and OpenMaya.MAngle.uiUnit() == OpenMaya.MAngle.kDegrees):
            return
        units = f"{cmds.currentUnit(query=True, linear=True)} and {cmds.currentUnit(query=True, angle=True)}"
        cmds.warning(f"rig: this scene works in {units}: rig's math is built for cm and degrees (Maya's "
                     f"default), and some of it gives other values in other units (rig's CHEATSHEET, Setup)")
    except Exception:  # noqa: BLE001 (see above)
        pass


def _forget(*_args: Any) -> None:
    """A scene was made or opened, or its units changed: the next statement checks them"""
    CHECKED[0] = False


def _scene_callback_specs() -> list:
    msg   = OpenMaya.MSceneMessage
    event = OpenMaya.MEventMessage.addEventCallback
    return [
        (msg.addCallback, msg.kAfterNew,           _forget),
        (msg.addCallback, msg.kAfterOpen,          _forget),
        (event,           "linearUnitChanged",     _forget),
        (event,           "angularUnitChanged",    _forget),
    ]


def _ensure_callbacks() -> None:
    from rig._internal import callbacks

    callbacks.ensure(__name__, _THIS_MODULE, _scene_callback_specs())
