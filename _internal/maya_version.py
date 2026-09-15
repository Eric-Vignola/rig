"""
Lazy Maya version detection + user-overridable target version.

Two layers:

* :func:`get_maya_version` -- the **actual** Maya version we're running on.
  Lazy: queries ``cmds.about(version=True)`` only on first call, then
  caches. Import-side-effect-free, per Meta's lazy-imports rule.

* :func:`get_target_version` -- the **target** version the rig DSL should
  build for. Defaults to :func:`get_maya_version`, but can be overridden
  via :func:`rig.set_options(maya_version=N)` so users can build rigs
  compatible with older Maya releases (the DSL will then dispatch to
  legacy node networks instead of native 2024+ nodes).

  All ``@op.impl(since=N)`` dispatch and every ``is_at_least(year)``
  check reads from :func:`get_target_version` -- so flipping the target
  changes the entire DSL's code path for FUTURE node creations.
"""

from __future__ import annotations

from typing import Optional

from maya import cmds


_cached_version: Optional[int] = None
_target_version: Optional[int] = None


def get_maya_version() -> int:
    """Return the running Maya version as an integer (e.g. ``2022``).

    Lazy: queries Maya only on the first call, then caches.
    """
    global _cached_version
    if _cached_version is None:
        _cached_version = int(cmds.about(version=True))
    return _cached_version


def get_target_version() -> int:
    """Return the **target** Maya version the rig DSL is currently
    building for.

    By default this is the actual running Maya version. Users can
    override via :func:`rig.set_options(maya_version=N)` to force
    the DSL to emit legacy-compatible node networks.
    """
    if _target_version is not None:
        return _target_version
    return get_maya_version()


def set_target_version(version: Optional[int]) -> None:
    """Override the target version, or pass ``None`` to revert to the
    actual running Maya version.

    Internal -- called by :func:`rig.set_options`. End users should
    use the :func:`rig.set_options` API, which also clears caches so the
    version flip takes effect on subsequent dedupe lookups.
    """
    global _target_version
    _target_version = version


def is_at_least(version: int) -> bool:
    """Return ``True`` if the **target** Maya version is ``>= version``.

    NOTE: this consults :func:`get_target_version`, NOT
    :func:`get_maya_version` -- so a user override via
    ``set_options(maya_version=2022)`` will make this return ``False`` for
    ``is_at_least(2024)`` even on a Maya 2026 machine.
    """
    return get_target_version() >= version


def _reset_cache_for_test() -> None:
    """Reset the cached version + target override. Test-only -- do not
    call in production code."""
    global _cached_version, _target_version
    _cached_version = None
    _target_version = None