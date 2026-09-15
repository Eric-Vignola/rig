"""
Reference node class
"""

from __future__ import annotations

import os
from pathlib import Path

from maya import cmds
from rig.maya.nodetypes.dg_node import DGNode, PyNode


class Reference(DGNode):
    """
    Reference node class.
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "reference"

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
        cmds.file(file_path.as_posix(), reference=True, namespace=namespace)
        return (set(cmds.ls(type="reference")) - refs).pop()

    @classmethod
    def find_by_path(cls, file_path: str) -> list[Reference]:
        """Returns a list of references with a given file path."""
        path = Path(file_path).as_posix()
        refs = []
        for each in cls.find_all():
            if each.file_path == path:
                refs.append(each)
        return refs

    @property
    def namespace(self) -> str:
        """Returns the namespace of this reference."""
        return cmds.referenceQuery(self.name, namespace=True)[1:]

    @namespace.setter
    def namespace(self, namespace: str) -> None:
        """Sets the namespace of this reference."""
        raise NotImplementedError("Cannot set namespace of a reference node.")

    @property
    def file_path(self) -> str:
        """Returns the file path of this reference."""
        path = cmds.referenceQuery(self.name, filename=True, withoutCopyNumber=True)
        return Path(path).as_posix()

    @property
    def file_path_with_copy_number(self) -> str:
        """Returns the file path of this reference (with copy number)."""
        path = cmds.referenceQuery(self.name, filename=True, withoutCopyNumber=False)
        return Path(path).as_posix()

    def get_nodes(self) -> list[DGNode]:
        """Returns all the nodes in this reference."""
        return [PyNode(x) for x in cmds.referenceQuery(self.name, nodes=True) or []]

    def delete(self) -> None:
        """Deletes this reference."""
        cmds.file(self.file_path_with_copy_number, removeReference=True)