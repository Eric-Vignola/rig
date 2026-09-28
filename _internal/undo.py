"""
Named undo chunks: :func:`undo_chunk` (public as ``rig.undo_chunk``) and
:data:`_undo_chunk`, the same function, which rig's own operations use.

A chunk groups everything recorded inside it into ONE undo step that the Edit
menu names after the chunk: ``cmds`` commands and rig's undoable API edits
(``rig.Mesh.set_points`` ...) alike (``undoInfo(openChunk)`` / ``closeChunk``
around several ``cmds`` calls undoes atomically in batch too). Every
``inject`` of a collection spec (``rig.tag`` ...) and ``rig.Mesh.create`` run
inside one.

Usage::

    import rig
    from maya import cmds

    with rig.undo_chunk("build arm"):        # a named step
        cmds.createNode("transform", name="first")
        cmds.createNode("transform", name="second")
    cmds.undo()                              # both are gone

    @rig.undo_chunk                          # named after the function: "build_leg"
    def build_leg():
        ...

    @rig.undo_chunk("rig.leg")               # named "rig.leg"
    def build_other_leg():
        ...

rig's operations open their named chunks through this module.
"""

from __future__ import annotations

import functools
import inspect

from maya import cmds

DEFAULT_NAME = "rig.undo_chunk"


class _Chunk:
    """One named chunk: a context manager (``with``) and a decorator.

    Holds only its name, so one object can be entered again while it is open
    (Maya nests chunks: the outermost one is the undo step). The decorator form
    enters a new :class:`_Chunk` on every call of the function, so recursion and
    reentrant calls each open and close their own chunk.
    """

    __slots__ = ("name",)

    def __init__(self, name: str | None) -> None:
        self.name = name  # None: DEFAULT_NAME, or the decorated function's __qualname__

    def __repr__(self) -> str:
        return f"undo_chunk({self.name!r})"

    def __enter__(self) -> None:
        cmds.undoInfo(openChunk=True, chunkName=self.name or DEFAULT_NAME)

    def __exit__(self, exc_type, exc, tb) -> bool:
        cmds.undoInfo(closeChunk=True)
        return False

    def __call__(self, func):
        return _decorate(func, self.name)


def _decorate(func, name: str | None):
    """``func`` wrapped so that each call runs inside a new chunk named ``name``
    (``func.__qualname__`` when None)."""
    if isinstance(func, type):
        raise TypeError(
            f"rig.undo_chunk decorates a function or a method, not the class {func.__qualname__}: "
            "decorate its methods"
        )
    if isinstance(func, (staticmethod, classmethod, property)):
        raise TypeError(
            f"rig.undo_chunk cannot wrap a {type(func).__name__} object: put @rig.undo_chunk "
            f"below @{type(func).__name__}, directly above the def"
        )
    if not callable(func):
        raise TypeError(
            f"rig.undo_chunk takes a chunk name (str) or a function, not {type(func).__name__}"
        )
    if (
        inspect.isgeneratorfunction(func)
        or inspect.iscoroutinefunction(func)
        or inspect.isasyncgenfunction(func)
    ):
        raise TypeError(
            f"rig.undo_chunk cannot wrap {getattr(func, '__qualname__', func)!r}: its body runs "
            "after the call returns (a generator or a coroutine), outside the chunk; open "
            "'with rig.undo_chunk(name):' inside it instead"
        )
    if name is None:
        name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or DEFAULT_NAME

    @functools.wraps(func)
    def in_chunk(*args, **kwargs):
        with _Chunk(name):
            return func(*args, **kwargs)

    return in_chunk


def undo_chunk(name_or_func=None):
    """A named undo chunk: everything recorded inside it (``cmds`` commands
    and rig's edits) undoes and redoes as ONE step, and the Edit menu names
    the step after the chunk.

    Three forms::

        with rig.undo_chunk("build arm"):   # a context manager: the step "build arm"
            ...

        @rig.undo_chunk                     # a decorator: the step is named after
        def build_arm(): ...                # the function's __qualname__ ("build_arm",
                                            # "Arm.build" for a method)

        @rig.undo_chunk("build arm")        # a decorator with a name
        def build_arm(): ...

    ``with rig.undo_chunk():`` names the step ``"rig.undo_chunk"``;
    ``@rig.undo_chunk()`` names it after the function.

    * The chunk is opened with ``cmds.undoInfo(openChunk=True, chunkName=name)``
      and always closed with ``cmds.undoInfo(closeChunk=True)``, also when the
      block (or the function) raises; the exception propagates, and what the
      block recorded before it stays one undo step.
    * Chunks nest: inside another chunk (a caller's, or rig's own), the
      outermost chunk is the one undo step and names it.
    * The decorator keeps the function's name, docstring, signature
      (``functools.wraps``; the function is ``__wrapped__``) and return value,
      and opens a new chunk on every call, so recursive and reentrant calls are
      safe. It decorates functions and methods; put it below
      ``@staticmethod`` / ``@classmethod`` / ``@property``. A generator or
      coroutine function is refused (its body would run outside the chunk).

    Raises TypeError for anything that is not a str name, None or a function,
    ValueError for an empty name.
    """
    if name_or_func is None:
        return _Chunk(None)
    if isinstance(name_or_func, str):
        if not name_or_func:
            raise ValueError("rig.undo_chunk: the chunk name is empty; give a name, or none at all")
        return _Chunk(name_or_func)
    return _decorate(name_or_func, None)


# rig's own operations: the same function (one place that opens rig's chunks)
_undo_chunk = undo_chunk
