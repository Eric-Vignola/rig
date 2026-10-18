"""
Choice node class
"""

from __future__ import annotations

from rig.nodetypes._base import _connected_attrs, Attribute
from rig.nodetypes.dg_node import _QUERY_KEEPING_HOOKS, DGNode


class Choice(DGNode):
    """
    Choice node class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "choice"

    def _attr_data_type_fallback(self, attr: Attribute) -> str:
        """Returns the current data type of a given attr, which depends on its
        input connections.
        """
        # for the output, use get the selected input and use its connected data type
        if type(attr)._raw_get_name(attr) == "output":
            selector = type(self)._raw_find_attr(self, "selector")
            # the int `selector.get()` reads with cmds.getAttr, read from the plug:
            # a long is never geometry, and MPlug.asInt evaluates it at the
            # current time as getAttr does, connected or not. `data_type` named
            # the attr first, so the node is valid; a direct call on a deleted
            # or freed node raises the "already deleted!" `get()` raises, and
            # never reads its plug
            type(self)._raw_ensure_valid(self)
            sel_id = selector._mplug.asInt()
            # quiet=True: an unresolvable selection must degrade to the base
            # "Tdata" answer, not raise (this hook runs inside
            # ``get_data_type()``, whose callers read None or the type).
            in_attr = type(self)._raw_find_attr(self, f"input[{sel_id}]", quiet=True)
            if in_attr is not None:
                src_attr = _connected_attrs(
                    in_attr, src=True, dst=False, first_only=True
                )
                if src_attr is not None:
                    return type(src_attr)._raw_get_data_type(src_attr)

        # for input attrs, use its connected data type
        elif type(attr)._raw_get_name(attr).startswith("input"):
            src_attr = _connected_attrs(attr, src=True, dst=False, first_only=True)
            if src_attr is not None:
                return type(src_attr)._raw_get_data_type(src_attr)

        return super()._attr_data_type_fallback(attr)


# the hook above only reads the scene before it calls DGNode's
_QUERY_KEEPING_HOOKS[Choice] = Choice._attr_data_type_fallback