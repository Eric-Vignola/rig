"""
Undo chunking for the membership grammar.

Every ``inject`` of a collection spec runs its scene writes inside ONE undo
chunk so a single ``cmds.undo()`` reverts the whole ``<<`` -- the sets,
component tags and shading assignments it made together, never one at a
time (verified: ``undoInfo(openChunk)`` / ``closeChunk`` around several
``cmds`` calls undoes atomically in batch too).

Usage::

    from maya import cmds
    from rig._internal.undo import _undo_chunk

    with _undo_chunk("rig.tag"):
        cmds.sets(..., add="x")
        cmds.componentTag(..., create=True)
    cmds.undo()          # both calls are gone
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from maya import cmds


@contextmanager
def _undo_chunk(name: str) -> Iterator[None]:
    """Open a named undo chunk for the block and close it on the way out,
    whether the block returned or raised."""
    cmds.undoInfo(openChunk=True, chunkName=name)
    try:
        yield
    finally:
        cmds.undoInfo(closeChunk=True)
