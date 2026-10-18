"""
The typo guard of rig's value objects (user decision 2026-10-17, as cgmath):
an assignment to a public attribute the class does not declare raises
``AttributeError: 'Tag' has no attribute 'nmae'; public attributes must be
declared``, instead of making a new attribute nobody reads.

A class declares a public attribute by having it (a method, a property, a
slot, a class attribute: anything ``dir(cls)`` lists) or by naming it in
``_FIELDS`` (the instance fields its ``__init__`` sets), along its MRO.
Private names (a leading underscore) are never checked, so Python's own
machinery (``__traceback__``, ``__notes__``, ``__cause__``) and a class's
private state keep working. No imports: any module can use it.
"""

from __future__ import annotations

from typing import Any

# class -> the frozenset of its declared public names (classes are few and live
# as long as their module)
_DECLARED: dict = {}


def _declared(cls: type) -> frozenset:
    names = _DECLARED.get(cls)
    if names is None:
        fields = set()
        for klass in cls.__mro__:
            fields.update(klass.__dict__.get("_FIELDS", ()))
        names = _DECLARED[cls] = frozenset(dir(cls)) | frozenset(fields)
    return names


def _undeclared(obj: Any, name: str) -> AttributeError:
    shown = getattr(type(obj), "_SHOWN_NAME", None) or type(obj).__name__
    return AttributeError(
        f"{shown!r} has no attribute {name!r}; public attributes must be declared"
    )


class Declared:
    """Mixin: refuses an undeclared public assignment (see the module)."""

    __slots__ = ()

    # the instance fields a subclass's __init__ sets (public names only)
    _FIELDS: tuple = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name[:1] != "_" and name not in _declared(type(self)):
            raise _undeclared(self, name)
        super().__setattr__(name, value)
