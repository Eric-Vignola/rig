"""
Base geometry node class
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig.maya.attribute import Attribute
from rig.maya.node_name import iter_component_ranges
from rig.maya.nodetypes.dag_node import DAGNode


class Geometry(DAGNode):
    """
    Base geometry node class

    All sub classes must implement the following:
        - NATIVE_NODE_TYPE
        - POINT_COMP_TYPE
        - num_weight_points()
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "geometryShape"

    # point component type string
    POINT_COMP_TYPE = None

    def __init__(self, node: str | OpenMaya.MObject | OpenMaya.MDagPath) -> None:
        """Initialize an instance from a node name or a MObject.

        Supports initializing from transform nodes.
        """
        if isinstance(node, str) and cmds.nodeType(node) == "transform":
            shapes = cmds.listRelatives(
                node,
                shapes         = True,
                type           = self.NATIVE_NODE_TYPE,
                noIntermediate = True,
                fullPath       = True,
            )
            if len(shapes) == 0:
                raise ValueError(
                    f"Transform {node} has no {self.NATIVE_NODE_TYPE} shape."
                )
            node = shapes[0]

        super().__init__(node)
        self.__local_shape_attr = None
        self.__world_shape_attr = None

    # --- attr helpers

    @property
    def local_shape_attr(self) -> Attribute:
        """Returns the local shape out attribute."""
        self.ensure_valid()
        if not self.__local_shape_attr:
            attr_name               = cmds.deformableShape(self.name, localShapeOutAttr=True)[0]
            self.__local_shape_attr = self.find_attr(attr_name)
        return self.__local_shape_attr

    @property
    def world_shape_attr(self) -> Attribute:
        """Returns the world shape out attribute."""
        self.ensure_valid()
        if not self.__world_shape_attr:
            attr_name               = cmds.deformableShape(self.name, worldShapeOutAttr=True)[0]
            self.__world_shape_attr = self.find_attr(attr_name)[0]
        return self.__world_shape_attr

    def set_attr_paintable(self, attr: Attribute | str, **kwargs) -> None:
        """Set the attribute paintable states. Thin wrapper of cmds.makePaintable()

        Args:
            attr: An attribute to operate on.
            kwargs: kwargs supported by cmds.makePaintable().
        """
        attr               = self.find_attr(attr)
        args               = (self.node_type, attr.name)
        kwargs["attrType"] = attr.data_type
        cmds.makePaintable(*args, **kwargs)

    @staticmethod
    def clear_all_paintable_attrs() -> None:
        """Clears all paintable attrs."""
        cmds.makePaintable(clearAll=True)

    # --- components

    def get_component_mobject(
        self, component_type: str | None = None, indices: None | np.ndarray = None
    ) -> OpenMaya.MObject:
        """Returns a mobject hosting all components in this geom.

        Args:
            component_type: The component type. e.g. "v", "e", "f", etc
        """
        comp_type = component_type or self.POINT_COMP_TYPE
        comp_type = "vtx" if comp_type == "v" else comp_type
        sel       = OpenMaya.MSelectionList()
        if indices is None:
            sel.add(f"{self.name}.{comp_type}[*]")
        else:
            token = f"{self.name}.{comp_type}"  # node_name.component_type
            for item in iter_component_ranges(token, indices):
                sel.add(item)

        _, mobject = sel.getComponent(0)
        return mobject

    # --- component tags editing

    def add_component_tag(self, key: str, force: bool = False) -> int:
        """Adds a component tag with the given key to the object.

        Args:
            key: Component tag key.
            force: If True, remove the existing component tag.

        Returns:
            The index of the new tag added.
        """
        if self.has_component_tag(key):
            if force:
                self.remove_component_tag(key)
            else:
                cmds.error(f"Component tag '{key}' already exists.")

        # ddd to first empty component tag name index.
        i         = self.componentTags.get_next_available_index()
        name_attr = self.componentTags[i].componentTagName
        name_attr.set(key, type="string")
        return i

    def remove_component_tag(self, tag: str | int) -> None:
        """Removes a component tag from the object.

        Args:
            tag: Component tag key or index.
        """
        i = tag if isinstance(tag, int) else self.get_component_tag_index(tag)
        if i != -1:
            cmds.removeMultiInstance(f"{self.name}.componentTags[{i}]", b=True)
        else:
            cmds.warning(f"Component tag not found: {tag}")

    def rename_component_tag(self, tag: str | int, new_key: str) -> None:
        """Renames a component tag with the given key the object.

        Args:
            tag: Component tag key or index.
            new_key: New component tag key name.
        """
        i = tag if isinstance(tag, int) else self.get_component_tag_index(tag)
        if i != -1:
            name_attr = self.componentTags[i].componentTagName
            name_attr.set(new_key, type="string")
        else:
            cmds.warning(f"Component tag not found: {tag}")

    def set_component_tag_contents(
        self,
        tag:      str       | int,
        contents: ArrayLike | list[str] | list[int] | GeomSubsetData,
        category: str                                                = "v",
    ) -> None:
        """Sets the component tag with the given key to the given contents.

        Args:
            tag: Component tag key or index.
            contents: One of the following:
              - An array of indices.
              - A list of component strings or integer indices.
              - A GeomSubsetData object.
            category: The component tag category. "v", "f", "e", etc.
                Only used when contents is a list of index.
        """
        from cgmath.geometry import GeomSubsetData

        i = tag if isinstance(tag, int) else self.get_component_tag_index(tag)
        if i == -1:
            cmds.warning(f"Component tag not found: {tag}")
            return

        # convert contents to a list of component strings
        if isinstance(contents, GeomSubsetData):
            cat      = contents.component_type
            cat      = "vtx" if cat == "v" else cat
            contents = list(iter_component_ranges(cat, contents.indices))
        elif isinstance(contents, np.ndarray):
            if contents.size == 0:
                contents = []
            elif isinstance(contents[0], (int, np.int64)):
                cat      = category
                cat      = "vtx" if cat == "v" else cat
                contents = list(iter_component_ranges(cat, contents))
            else:
                cmds.error("Component tag indices must be integers.")
        elif isinstance(contents, (list, tuple)):
            if len(contents) > 0:
                if isinstance(contents[0], str):
                    if contents[0].find(".") != -1:
                        # for some reason this is auto-collapsed on set attr
                        contents = [x.rsplit(".", 1)[-1] for x in contents]
                elif isinstance(contents[0], int):
                    cat      = category
                    cat      = "vtx" if cat == "v" else cat
                    contents = list(iter_component_ranges(cat, contents))
                else:
                    cmds.error(
                        "Component tag list contents must be strings or integers."
                    )
        else:
            cmds.error(
                "Component tag contents must be either GeomSubsetData, a list of component strings, or an integer np.ndarray"
            )

        comp_attr = self.componentTags[i].componentTagContents
        comp_attr.set(len(contents), *contents, type="componentList")

    def select_component_tag_contents(self, key: str) -> None:
        """Selects component members of the component tag with the given key.

        Args:
            key: Component tag key.
        """
        contents = self.get_component_tag_contents(key, full_name=True)
        cmds.select(contents, replace=True)

    # --- component tags query

    @property
    def component_tags(self) -> list[str]:
        """Returns the geometry object's component tag keys in index order."""
        if self.componentTags.num_elements == 0:
            return []
        tags = cmds.getAttr(f"{self.name}.componentTags[*].componentTagName")
        if isinstance(tags, str):
            return [tags]
        return tags

    def has_component_tag(self, tag: str | int) -> bool:
        """Returns True if a component tag exists.

        Args:
            tag: Component tag key or index.
        """
        if isinstance(tag, int):
            return tag in self.componentTags.get_logical_indices()
        else:
            data_fn = self.local_shape_attr.get_data_fn_set()
            return data_fn.hasComponentTag(tag)

    def get_component_tag_index(self, key: str) -> int:
        """Returns the logical index of a compoennt tag, or -1 if not found."""
        for i in self.componentTags.get_logical_indices():
            name_attr = self.componentTags[i].componentTagName
            if name_attr.get() == key:
                return i
        return -1

    def get_component_tag_name(self, i: int) -> str | None:
        """Returns the name of a compoennt tag, or None if not found."""
        name_attr = self.componentTags[i].componentTagName
        return name_attr.get() or None

    def component_tag_expression_subset_state(self, expr: str) -> int:
        """Returns the subset state for the component tag expression.

        [0: no geometry points, 1: some geometry points, 2: all geometry points]

        Args:
            expr: Component tag expression. (E.g. '*', 'top', 'top+bottom', etc.)
        """
        return cmds.geometryAttrInfo(
            f"{self.local_shape_attr.full_name}",
            componentTagExpression = expr,
            subsetState            = True,
        )

    def get_component_tag_history(self) -> list[dict[str, Any]]:
        """Returns verbose descriptions of all historical component tags.

        Example:
        ```
        [{'key': 'top',
          'node': 'pSphereShape1Orig',
          'affectCount': 40,
          'fullCount': 400,
          'modified': False,
          'procedural': False,
          'editable': True,
          'category': 1},
          {...}]
        ```
        """
        return cmds.geometryAttrInfo(
            f"{self.local_shape_attr.full_name}",
            componentTagHistory=True,
        )

    def get_component_tag_category(self, tag: str | int) -> str:
        """Returns the category of a given component tag.

        [v: verts, e: edges, f: faces]

        Args:
            tag: Component tag key or index.
        """
        tag_name = self.get_component_tag_name(tag) if isinstance(tag, int) else tag
        return cmds.geometryAttrInfo(
            f"{self.local_shape_attr.full_name}",
            componentTagExpression = tag_name,
            componentTagCategory   = True,
        )

    def set_component_tag_category(self, tag: str | int, category: str) -> None:
        """Sets the category of a given component tag and convert its content.

        [v: verts, e: edges, f: faces]

        Args:
            tag: Component tag key or index.
            category: The component tag category. "v", "f", "e", etc.
        """
        import maya.internal.common.utils.componenttag as ctag_utils

        cat = self.get_component_tag_category(tag)
        if cat != category:
            if category == "f":
                c = OpenMaya.MFnGeometryData.kFaces
            elif category == "v":
                c = OpenMaya.MFnGeometryData.kVerts
            else:
                c = OpenMaya.MFnGeometryData.kEdges
            tag_name = self.get_component_tag_name(tag) if isinstance(tag, int) else tag
            ctag_utils.convertTagCategory(self.name, tag_name, c)

    def get_component_tag_contents(
        self, tag: str | int, full_name: bool = False
    ) -> list[str]:
        """Returns a component list store in a given tag.

        Args:
            tag: Component tag key or index.
            full_name: If True, returns list with shape name e.g.
                (['pSphereShape.f[1:9]']).
                Otherwise return a component list (e.g. ['f[1:9]']).
        """
        tag_name = self.get_component_tag_name(tag) if isinstance(tag, int) else tag
        components = cmds.geometryAttrInfo(
            f"{self.local_shape_attr.full_name}",
            componentTagExpression = tag_name,
            components             = True,
        )

        if full_name:
            return [f"{self.name}.{component}" for component in components]
        return components

    def is_component_tag_empty(self, tag: str | int) -> bool:
        """Returns True if a tag is empty.

        Args:
            key: Component tag key or index.
        """
        if not self.has_component_tag(tag):
            cmds.error(f"Component tag {tag} does not exist.")
        return not bool(self.get_component_tag_contents(tag))

    def _resolve_component_tag_expression(
        self, expr: str
    ) -> OpenMaya.MFnSingleIndexedComponent:
        """Returns MFnSingleIndexedComponent of the component tag expression.

        Args:
            key: Component tag key.
        """
        data_fn           = self.local_shape_attr.get_data_fn_set()
        component_mobject = data_fn.resolveComponentTagExpression(expr)
        return OpenMaya.MFnSingleIndexedComponent(component_mobject)

    # --- component tags serialization

    def get_component_tag_data(
        self, tag: str | int, category: str | None = None
    ) -> GeomSubsetData:
        """Serialize a given tag.

        Args:
            tag: Component tag key or index.
        """
        from cgmath.geometry import GeomSubsetData

        i         = tag if isinstance(tag, int) else self.get_component_tag_index(tag)
        name_attr = self.componentTags[i].componentTagName
        comp_attr = self.componentTags[i].componentTagContents

        name = name_attr.get()
        data = GeomSubsetData(
            name    = name,
            indices = np.empty((0,), dtype=int),
        )

        comp_data = comp_attr.get_data_fn_set()
        if not comp_data:
            return data

        # get component ids
        indices = []
        for i in range(comp_data.length()):
            comp = OpenMaya.MFnSingleIndexedComponent(comp_data.get(i))
            indices.extend(comp.getElements())
        indices = np.array(indices)

        data.indices        = indices
        data.component_type = self.get_component_tag_category(tag)
        return data

    def serialize_component_tags(
        self,
        match_name: list[str] | str | None = None,
        category:   str       | None       = None,
    ) -> list[GeomSubsetData]:
        """Serialize component tags into a standard dict.

        Args:
            match_name: One or more of tag names or regex to match.
                If None, serialize all tags.

        Returns:
            A list of GeomSubsetData objects.
        """
        from cgmath.geometry import GeomSubsetData

        data      = []
        names     = [match_name] if isinstance(match_name, str) else match_name
        tag_regex = re.compile(r"|".join([rf"^{x}$" for x in names])) if names else None
        for i in self.componentTags.get_logical_indices():
            name_attr = self.componentTags[i].componentTagName
            if not tag_regex or tag_regex.match(name_attr.get()):
                data.append(self.get_component_tag_data(i, category=category))

        return data