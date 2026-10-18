"""
Reference node class
"""

from __future__ import annotations

import os
from pathlib import Path

from maya import cmds
from rig._internal.journal import cmds_edit as _cmds_edit
from rig.nodetypes._base import _cast, _name_of
from rig.nodetypes.dg_node import DGNode


class Reference(DGNode):
    """
    Reference node class.
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "reference"

    # built from its file (``Node.create("reference")`` with none raises, naming
    # it)
    _CREATE_TAKES_INPUTS = "file_path, namespace"

    # ``_create`` takes its two inputs (also by keyword: ``namespace="hero"``)
    # and no flag (the file names the node); ``create``'s other keywords are
    # the reference node's attributes
    _CREATE_FLAGS = frozenset({"file_path", "namespace"})

    # a create references a whole file: inside ``with container()`` it stays out
    # of the scope (tracking would pour every referenced node into it) unless
    # ``container=True``
    _CONTAINER_AWARE = False

    _DEFINE_REFUSED = (
        "a reference is made from its file: Reference.create(file_path, namespace) "
        "references it; Reference('xRN') refers to one"
    )

    @classmethod
    def _create(cls, file_path: str | os.PathLike, namespace: str) -> str:
        """[Internal] Creates a reference node and returns its name.

        Args:
            file_path: A file to reference.
            namespace: A namespace to apply to this reference.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        refs = set(cmds.ls(type="reference"))
        _cmds_edit.file(file_path.as_posix(), reference=True, namespace=namespace)
        return (set(cmds.ls(type="reference")) - refs).pop()

    @classmethod
    def find_by_path(cls, file_path: str) -> list[Reference]:
        """Returns a list of references with a given file path."""
        path = Path(file_path).as_posix()
        refs = []
        for each in cls.find_all():
            if type(each).get_file_path(each) == path:
                refs.append(each)
        return refs

    def get_namespace(self) -> str:
        """Returns the namespace of this reference."""
        return cmds.referenceQuery(_name_of(self), namespace=True)[1:]

    def set_namespace(self, namespace: str) -> None:
        """Not supported: a reference's namespace is its file's (raises, as the
        ``namespace`` setter did)."""
        raise NotImplementedError("Cannot set namespace of a reference node.")

    def get_file_path(self, copy_number: bool = False) -> str:
        """Returns the file path of this reference (posix), with its copy number
        (``scene.ma{1}``) when `copy_number` is True."""
        path = cmds.referenceQuery(_name_of(self), filename=True, withoutCopyNumber=not copy_number)
        return Path(path).as_posix()

    def get_nodes(self) -> list[DGNode]:
        """Returns all the nodes in this reference."""
        return [_cast(x) for x in cmds.referenceQuery(_name_of(self), nodes=True) or []]

    def delete(self) -> None:
        """Deletes this reference."""
        _cmds_edit.file(type(self).get_file_path(self, copy_number=True), removeReference=True)


# the public names (user decision 2026-10-17: import leaks hidden)
__all__ = [
    "Reference",
]
