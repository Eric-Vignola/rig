"""
Base geometry node class

Usage::

    from rig.maya.nodetypes import PyNode

    mesh = PyNode("pSphereShape1")
    mesh.injection_node                          # the node holding editable tags
    mesh.add_component_tag("cap")
    mesh.set_component_tag_contents("cap", [0, 1, 2])
    mesh.get_component_tag_contents("cap")       # ['vtx[0:2]']
    mesh.get_component_tag_indices("cap")        # array([0, 1, 2])
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterator, TYPE_CHECKING

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig.maya.nodetypes._base import Attribute, PyNode
from rig.maya.nodetypes.dag_node import DAGNode

if TYPE_CHECKING:
    from cgmath.geometry import GeomSubsetData


# a bare component tag name, as opposed to an expression ('top + side', '!top')
_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*$")

# The component an EMPTY tag resolves to, by geometry data type then tag
# category: Maya hands back a null MObject for an empty tag, so the reader
# builds an empty component of the native (fn set, type) instead.
_EMPTY_COMPONENT_TYPES = {
    OpenMaya.MFn.kMeshData: {
        "v": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshVertComponent),
        "e": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshEdgeComponent),
        "f": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kMeshPolygonComponent),
    },
    OpenMaya.MFn.kNurbsCurveData: {
        "v": (OpenMaya.MFnSingleIndexedComponent, OpenMaya.MFn.kCurveCVComponent),
    },
    OpenMaya.MFn.kNurbsSurfaceData: {
        "v": (OpenMaya.MFnDoubleIndexedComponent, OpenMaya.MFn.kSurfaceCVComponent),
    },
    OpenMaya.MFn.kLatticeData: {
        "v": (OpenMaya.MFnTripleIndexedComponent, OpenMaya.MFn.kLatticeComponent),
    },
}


def iter_component_ranges(prefix: str, ids: list[int]) -> Iterator[str]:
    """A generator that converts a list of indices to a list of range strings.

    e.g.
    ```python
    iter_component_ranges("vtx", [1, 2, 3, 5, 7, 8])
    # ["vtx[1:3]", "vtx[5]", "vtx[7:8]"]
    ```
    """
    gen = itertools.groupby(enumerate(ids), lambda pair: pair[1] - pair[0])
    for _, bit in gen:
        b     = list(bit)
        start = b[0][1]
        end   = b[-1][1]
        if start == end:
            yield f"{prefix}[{start}]"
        else:
            yield f"{prefix}[{start}:{end}]"


def iter_component_tokens(prefix: str, ids: Any) -> Iterator[str]:
    """Range strings for component ids of any dimension, sorted and unique.

    (N,) ids are :func:`iter_component_ranges`; (N, 2) and (N, 3)
    coordinates are grouped on their leading axes and ranged on the last,
    the forms surfaces and lattices store.

    e.g.
    ```python
    list(iter_component_tokens("cv", [[1, 0], [1, 1], [2, 5]]))
    # ["cv[1][0:1]", "cv[2][5]"]
    ```
    """
    ids = np.unique(np.asarray(ids, dtype=int), axis=0)
    if ids.size == 0:
        return
    if ids.ndim == 1:
        yield from iter_component_ranges(prefix, ids.tolist())
        return
    leads = ids[:, :-1]
    for lead in np.unique(leads, axis=0):
        head = prefix + "".join(f"[{k}]" for k in lead)
        tail = ids[np.all(leads == lead, axis=1), -1]
        yield from iter_component_ranges(head, tail.tolist())


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

    # --- component tags location

    @property
    def injection_node(self) -> Geometry:
        """Returns the node holding this geometry's editable component tags.

        Maya writes tags to the current tag injection node
        (`cmds.deformableShape(shape, tagInjectionNode=True)`): the shape
        itself until a deformer exists, its intermediate "Orig" shape from then
        on. Every tag editor here addresses that node's componentTags multi so
        that editors and readers agree. Not cached: adding a deformer moves it.
        """
        self.ensure_valid()
        nodes = cmds.deformableShape(self.long_name, tagInjectionNode=True)
        if not nodes or cmds.ls(nodes[0], long=True)[0] == self.long_name:
            return self
        return PyNode(nodes[0])

    def _component_tag_owner(self, key: str) -> tuple[str, bool] | None:
        """Returns (node, procedural) for the entry of a tag that resolves, or None."""
        for entry in self.get_component_tag_history() or []:
            if entry["key"] == key and entry["final"]:
                return entry["node"], entry["procedural"]
        return None

    def _editable_component_tag(self, tag: str | int) -> Attribute | None:
        """Returns the injection node's componentTags element holding a tag.

        A tag that resolves on this geometry but has no element at the
        injection node cannot be edited here: it is procedural (polyCube's six
        face tags are output of polyCube1) or injected on another node in the
        history, and only its owner can change it. That raises; a name that
        does not exist at all warns and returns None.

        Args:
            tag: Component tag key or index.
        """
        injection = self.injection_node
        if isinstance(tag, int):
            i = tag if tag in injection.componentTags.get_logical_indices() else -1
        else:
            i = self.get_component_tag_index(tag)
            if i == -1 and self.has_component_tag(tag):
                owner = self._component_tag_owner(tag)
                kind  = "procedural" if owner and owner[1] else "injected"
                node  = owner[0] if owner else "another node"
                cmds.error(
                    f"Component tag '{tag}' on {self.name} is {kind} (owned by {node}) "
                    f"and cannot be edited at {injection.name}."
                )

        if i == -1:
            cmds.warning(f"Component tag not found: {tag}")
            return None
        return injection.componentTags[i]

    # --- component tags editing

    def add_component_tag(self, key: str, force: bool = False) -> int:
        """Adds a component tag with the given key to the object.

        The tag is created on the injection node, where Maya's own
        componentTag command puts it.

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
                owner = self._component_tag_owner(key)
                if owner and owner[1]:
                    cmds.error(
                        f"Component tag '{key}' already exists (procedural, owned by {owner[0]})."
                    )
                cmds.error(f"Component tag '{key}' already exists.")

        # add to first empty component tag name index.
        injection = self.injection_node
        i         = injection.componentTags.get_next_available_index()
        name_attr = injection.componentTags[i].componentTagName
        name_attr.set(key, type="string")
        return i

    def remove_component_tag(self, tag: str | int) -> None:
        """Removes a component tag from the object.

        Args:
            tag: Component tag key or index.
        """
        element = self._editable_component_tag(tag)
        if element is not None:
            cmds.removeMultiInstance(element.full_name, b=True)

    def rename_component_tag(self, tag: str | int, new_key: str) -> None:
        """Renames a component tag with the given key the object.

        Args:
            tag: Component tag key or index.
            new_key: New component tag key name.
        """
        element = self._editable_component_tag(tag)
        if element is not None:
            element.componentTagName.set(new_key, type="string")

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
              - An array of indices: (N,) native ids, or (N, 2) / (N, 3)
                coordinates for surface CVs and lattice points.
              - A list of component strings or integer indices.
              - A GeomSubsetData object.
            category: The component tag category. "v", "f", "e", etc.
                Only used when contents is a list of index.
        """
        element = self._editable_component_tag(tag)
        if element is None:
            return

        contents  = self._component_tag_tokens(contents, category)
        comp_attr = element.componentTagContents
        comp_attr.set(len(contents), *contents, type="componentList")

    def _component_tag_tokens(
        self,
        contents: ArrayLike | list[str] | list[int] | GeomSubsetData,
        category: str,
    ) -> list[str]:
        """Converts tag contents into the component strings the contents plug takes."""
        # a GeomSubsetData, duck-typed so cgmath is never imported here
        if hasattr(contents, "indices") and hasattr(contents, "component_type"):
            return self._component_strings(contents.component_type, contents.indices)

        if isinstance(contents, (list, tuple)):
            if len(contents) == 0:
                return []
            if isinstance(contents[0], str):
                # the node prefix is rejected by setAttr
                return [x.rsplit(".", 1)[-1] for x in contents]
            contents = np.asarray(contents)

        if isinstance(contents, np.ndarray):
            if contents.size == 0:
                return []
            if not np.issubdtype(contents.dtype, np.integer):
                cmds.error("Component tag indices must be integers.")
            return self._component_strings(category, contents)

        cmds.error(
            "Component tag contents must be either GeomSubsetData, a list of component strings, or an integer np.ndarray"
        )

    def _component_strings(self, category: str, indices: ArrayLike) -> list[str]:
        """Renders indices as this geometry's native component strings.

        (N,) indices become ``vtx[a:b]`` / ``cv[a:b]`` / ``f[a:b]`` ranges;
        (N, 2) and (N, 3) coordinates become the ``cv[u][v0:v1]`` and
        ``pt[s][t][u0:u1]`` forms surfaces and lattices store. Category "v"
        maps to POINT_COMP_TYPE, any other category is the token itself.
        """
        if category == "v":
            prefix = self.POINT_COMP_TYPE
            if prefix is None:
                cmds.error(
                    f"{self.node_type} has no point component type; pass component strings instead."
                )
        else:
            prefix = category

        return list(iter_component_tokens(prefix, indices))

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
        """Returns the component tag keys this geometry resolves.

        Editable tags at the injection node come first, in index order; names
        only visible on the output plug (procedural tags such as polyCube's six
        face tags, or tags injected elsewhere in the history) follow in Maya's
        sorted order.
        """
        injection = self.injection_node
        tags      = []
        for i in injection.componentTags.get_logical_indices():
            name = injection.componentTags[i].componentTagName.get()
            if name:
                tags.append(name)

        visible = cmds.geometryAttrInfo(
            f"{self.local_shape_attr.full_name}",
            componentTagNames=True,
        )
        tags.extend(name for name in visible or [] if name not in tags)
        return tags

    def has_component_tag(self, tag: str | int) -> bool:
        """Returns True if a component tag exists.

        A key is looked up on the evaluated geometry (so procedural and
        upstream tags count); an index is looked up in the injection node's
        componentTags multi.

        Args:
            tag: Component tag key or index.
        """
        if isinstance(tag, int):
            return tag in self.injection_node.componentTags.get_logical_indices()
        else:
            data_fn = self.local_shape_attr.get_data_fn_set()
            return data_fn.hasComponentTag(tag)

    def get_component_tag_index(self, key: str) -> int:
        """Returns the logical index of a component tag in the injection node's
        componentTags multi, or -1 if not found (or not editable there)."""
        injection = self.injection_node
        for i in injection.componentTags.get_logical_indices():
            name_attr = injection.componentTags[i].componentTagName
            if name_attr.get() == key:
                return i
        return -1

    def get_component_tag_name(self, i: int) -> str | None:
        """Returns the name of a component tag, or None if not found."""
        injection = self.injection_node
        if i not in injection.componentTags.get_logical_indices():
            return None
        name_attr = injection.componentTags[i].componentTagName
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

    def get_component_tag_indices(self, tag: str | int) -> np.ndarray:
        """Returns the native ids of the components in a given tag.

        Face tags give face ids and edge tags edge ids, never the vertex cast
        `geometryAttrInfo -pointIndices` applies. Shape (N,) on meshes and
        curves, (N, 2) ``(u, v)`` on surfaces, (N, 3) ``(s, t, u)`` on
        lattices; an empty tag gives an empty array of the same shape.

        Args:
            tag: Component tag key or index.
        """
        tag_name = self.get_component_tag_name(tag) if isinstance(tag, int) else tag
        if tag_name is None:
            cmds.error(f"Component tag {tag} does not exist.")

        comp_fn  = self._resolve_component_tag_expression(tag_name)
        elements = np.array(list(comp_fn.getElements()), dtype=int)
        if isinstance(comp_fn, OpenMaya.MFnTripleIndexedComponent):
            return elements.reshape(-1, 3)
        if isinstance(comp_fn, OpenMaya.MFnDoubleIndexedComponent):
            return elements.reshape(-1, 2)
        return elements

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
    ) -> (
        OpenMaya.MFnSingleIndexedComponent
        | OpenMaya.MFnDoubleIndexedComponent
        | OpenMaya.MFnTripleIndexedComponent
    ):
        """Returns the component function set a component tag expression resolves to.

        The component comes back in the geometry's native form: single-indexed
        on meshes and curves, double-indexed (u, v) on surfaces, triple-indexed
        (s, t, u) on lattices. An empty tag yields an empty component of that
        form instead of the null MObject Maya returns.

        Args:
            expr: Component tag key, or an expression ('top + side', '!top').
                A bare key that does not exist raises; unknown names inside an
                expression are left to Maya, which ignores them.
        """
        data_fn = self.local_shape_attr.get_data_fn_set()
        if _TAG_NAME_RE.match(expr) and not data_fn.hasComponentTag(expr):
            cmds.error(f"Component tag {expr} does not exist.")

        component = data_fn.resolveComponentTagExpression(expr)
        if component.isNull():
            types             = _EMPTY_COMPONENT_TYPES[data_fn.object().apiType()]
            fn_cls, comp_type = types.get(self.get_component_tag_category(expr), types["v"])
            comp_fn           = fn_cls()
            comp_fn.create(comp_type)
            return comp_fn

        if component.hasFn(OpenMaya.MFn.kTripleIndexedComponent):
            return OpenMaya.MFnTripleIndexedComponent(component)
        if component.hasFn(OpenMaya.MFn.kDoubleIndexedComponent):
            return OpenMaya.MFnDoubleIndexedComponent(component)
        return OpenMaya.MFnSingleIndexedComponent(component)

    # --- component tags serialization

    def get_component_tag_data(
        self, tag: str | int, category: str | None = None
    ) -> GeomSubsetData:
        """Serialize a given tag.

        Reads the evaluated geometry, so procedural and upstream tags
        serialize like editable ones.

        Args:
            tag: Component tag key or index.
        """
        from cgmath.geometry import GeomSubsetData

        name = self.get_component_tag_name(tag) if isinstance(tag, int) else tag
        if name is None or not self.has_component_tag(name):
            cmds.error(f"Component tag {tag} does not exist.")

        return GeomSubsetData(
            name           = name,
            indices        = self.get_component_tag_indices(name),
            component_type = self.get_component_tag_category(name),
        )

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
        data      = []
        names     = [match_name] if isinstance(match_name, str) else match_name
        tag_regex = re.compile(r"|".join([rf"^{x}$" for x in names])) if names else None
        for name in self.component_tags:
            if not tag_regex or tag_regex.match(name):
                data.append(self.get_component_tag_data(name, category=category))

        return data
