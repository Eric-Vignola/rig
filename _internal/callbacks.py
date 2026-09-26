"""
The Maya scene callbacks rig registers, once per Maya session.

Two of rig's session caches depend on the scene: the memo caches
(:mod:`rig._internal.memoize`) hold the nodes a build made, and the static data
types (:mod:`rig.nodetypes._base`) hold what a plug-in defines. Their modules
register ``MSceneMessage`` callbacks through :func:`register`, which keeps two
promises:

* **A re-import replaces, it never doubles.** The callback ids are kept on the
  ``sys`` module (``sys._rig_scene_callbacks``, keyed by module name), which a
  purge of ``sys.modules['rig*']`` does not touch. A module that registers again
  first removes what its previous import registered -- after an in-place
  ``importlib.reload`` or after a purge and a fresh import -- so Maya never runs
  two copies and no longer keeps the purged copy alive. A purged copy's
  ``release`` hook runs too: its memo caches are cleared, so they stop holding
  the scene's plugs.
* **Nothing is registered before Maya is initialised.** Adding a callback in a
  mayapy that has not run ``maya.standalone.initialize()`` crashes the process,
  and ``maya.cmds`` is still empty then. Such an import leaves the registration
  pending, and :func:`ensure` makes it the first time the cache is filled, which
  can only happen once Maya is up.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from maya import cmds
from maya.api import OpenMaya

# the attribute of ``sys`` that holds {module name: registration}
REGISTRY_ATTR = "_rig_scene_callbacks"

# (the MSceneMessage add function, the message, the callback)
Spec = Tuple[Callable[..., Any], Any, Callable[..., Any]]


def maya_is_initialised() -> bool:
    """True once Maya is initialised: ``maya.cmds`` has no command before."""
    return hasattr(cmds, "ls")


def registry() -> Dict[str, Dict[str, Any]]:
    """The session's registrations, ``{module name: {"module", "ids", "release"}}``,
    kept on ``sys`` so that they outlive a purge of the rig modules."""
    found = getattr(sys, REGISTRY_ATTR, None)
    if not isinstance(found, dict):
        found = {}
        setattr(sys, REGISTRY_ATTR, found)
    return found


def registered_ids(owner: str) -> List[int]:
    """The callback ids registered for the module named ``owner``, or []."""
    entry = registry().get(owner)
    return list(entry["ids"]) if entry else []


def _remove(ids: Iterable[int]) -> None:
    for callback_id in ids:
        try:
            OpenMaya.MMessage.removeCallback(callback_id)
        except Exception:  # noqa: BLE001 -- already removed
            pass


def register(
    owner:   str,
    module:  ModuleType,
    specs:   Iterable[Spec],
    release: Optional[Callable[[], Any]] = None,
) -> bool:
    """Register ``specs`` for ``module`` (named ``owner``) in place of what an
    earlier import of ``owner`` registered.

    The earlier callbacks are removed first. If they belonged to another module
    object (the rig modules were purged from ``sys.modules`` and imported again),
    that copy's ``release`` hook runs as well. ``release`` is this copy's hook,
    kept for the next import.

    Returns:
        True once registered; False, registering nothing, while Maya is not
        initialised (see :func:`ensure`).
    """
    if not maya_is_initialised():
        return False
    found    = registry()
    previous = found.pop(owner, None)
    if previous is not None:
        _remove(previous["ids"])
        if previous["module"] is not module and previous["release"] is not None:
            try:
                previous["release"]()
            except Exception:  # noqa: BLE001 -- a purged copy must not block this one
                pass
    ids: List[int] = []
    try:
        for add, message, callback in specs:
            ids.append(add(message, callback))
    except Exception:
        _remove(ids)
        raise
    found[owner] = {"module": module, "ids": ids, "release": release}
    return True


def ensure(
    owner:   str,
    module:  ModuleType,
    specs:   Iterable[Spec],
    release: Optional[Callable[[], Any]] = None,
) -> bool:
    """Make a pending registration (``module`` was imported before Maya was
    initialised). Registers only if no import of ``owner`` has registered yet:
    a stale copy must not take the callbacks of the one that replaced it.

    Called from a cache fill, so it never raises: a registration that fails is
    left pending, and the next fill tries again.

    Returns:
        True if ``owner`` has callbacks now, False while it has none.
    """
    if owner in registry():
        return True
    try:
        return register(owner, module, specs, release)
    except Exception:  # noqa: BLE001
        return False
