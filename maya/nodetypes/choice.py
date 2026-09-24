"""
Choice node class
"""

from __future__ import annotations

from rig.maya.nodetypes._base import Attribute
from rig.maya.nodetypes.dg_node import DGNode


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
        if attr.name == "output":
            sel_id = self.selector.get()
            # quiet=True: an unresolvable selection must degrade to the base
            # "Tdata" answer, not raise. ``data_type`` is a property, so an
            # AttributeError escaping here sends Python into Plug.__getattr__,
            # which relabels it as a bogus "has no child or sibling attribute
            # 'data_type'" and buries the real cause.
            in_attr = self.find_attr(f"input[{sel_id}]", quiet=True)
            if in_attr is not None:
                src_attr = in_attr.get_connected_attrs(
                    src=True, dst=False, first_only=True
                )
                if src_attr:
                    return src_attr.data_type

        # for input attrs, use its connected data type
        elif attr.name.startswith("input"):
            src_attr = attr.get_connected_attrs(src=True, dst=False, first_only=True)
            if src_attr:
                return src_attr.data_type

        return super()._attr_data_type_fallback(attr)