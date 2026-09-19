"""
Base deformer class

Usage::

    from rig.maya.nodetypes.deformer import tag_references

    tag_references("top")   # [("cluster1", 0)] -- every input whose
                            # componentTagExpression names the tag
"""

from __future__ import annotations

import fnmatch
import re

from maya import cmds
from rig.maya.nodetypes._base import PyNode
from rig.maya.nodetypes.dg_node import DGNode


# a tag-name token carrying at least one glob wildcard ('to*', '*top', 'a*b')
_GLOB_TOKEN_RE = re.compile(r"[A-Za-z0-9_:*]*\*[A-Za-z0-9_:*]*")


def tag_references(name: str) -> list[tuple[str, int]]:
    """Returns the deformer inputs whose componentTagExpression references a tag.

    Deformers read component tags by name through a plain string, so renaming
    or deleting a referenced tag silently makes the deformer move nothing.
    An expression references `name` when it holds it as a whole token
    (``top``, ``!top``, ``top + side``; never ``topside`` or ``ns:top``) or a
    glob that matches it (``to*``). The bare ``*`` means every point whatever
    the tags are called and never counts as a reference.

    Args:
        name: The component tag name to look for.

    Returns:
        A list of (deformer, input index) pairs.
    """
    exact = re.compile(rf"(?<![A-Za-z0-9_:*]){re.escape(name)}(?![A-Za-z0-9_:*])")
    found = []
    for deformer in cmds.ls(type="geometryFilter") or []:
        for i in cmds.getAttr(f"{deformer}.input", multiIndices=True) or []:
            expr = cmds.getAttr(f"{deformer}.input[{i}].componentTagExpression") or ""
            if exact.search(expr) or any(
                fnmatch.fnmatchcase(name, token)
                for token in _GLOB_TOKEN_RE.findall(expr)
                if token != "*"
            ):
                found.append((deformer, i))
    return found


class Deformer(DGNode):
    """
    Base deformer class
    """

    NATIVE_NODE_TYPE = "geometryFilter"

    def get_geometries(self) -> list[DGNode]:
        """Returns a list of geometry objects (post-deformation).
        TODO support components
        """
        return [PyNode(x) for x in cmds.deformer(self.name, query=True, geometry=True)]

    def get_original_geometries(self) -> list[DGNode]:
        """Returns a list of original geometry objects (pre-deformation)."""

        geom = (
            cmds.listConnections(
                f"{self.name}.originalGeometry",
                source      = True,
                destination = False,
                plugs       = True,
            )
            or []
        )
        return [PyNode(x.split(".", 1)[0]) for x in geom]
