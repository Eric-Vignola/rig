from __future__ import annotations

import math
from functools import total_ordering
from numbers import Number
from typing import Any, Iterator

import numpy as np
from maya import cmds
from maya.api import OpenMaya


# TODO better solutions?
# Also some data fn sets are not in API 2.0, e.g lattice and subdiv
DATA_TYPE_TO_FN = {
    OpenMaya.MFn.kComponentListData: OpenMaya.MFnComponentListData,
    OpenMaya.MFn.kDoubleArrayData: OpenMaya.MFnDoubleArrayData,
    OpenMaya.MFn.kIntArrayData: OpenMaya.MFnIntArrayData,
    OpenMaya.MFn.kMatrixData: OpenMaya.MFnMatrixData,
    OpenMaya.MFn.kNumericData: OpenMaya.MFnNumericData,
    OpenMaya.MFn.kPointArrayData: OpenMaya.MFnPointArrayData,
    OpenMaya.MFn.kStringArrayData: OpenMaya.MFnStringArrayData,
    OpenMaya.MFn.kVectorArrayData: OpenMaya.MFnVectorArrayData,
    OpenMaya.MFn.kGeometryData: OpenMaya.MFnGeometryData,
    OpenMaya.MFn.kMeshData: OpenMaya.MFnMeshData,
    OpenMaya.MFn.kNurbsCurveData: OpenMaya.MFnNurbsCurveData,
}

ATTR_TYPE_TO_FN = {
    OpenMaya.MFn.kCompoundAttribute: OpenMaya.MFnCompoundAttribute,
    OpenMaya.MFn.kEnumAttribute: OpenMaya.MFnEnumAttribute,
    OpenMaya.MFn.kGenericAttribute: OpenMaya.MFnGenericAttribute,
    OpenMaya.MFn.kLightDataAttribute: OpenMaya.MFnLightDataAttribute,
    OpenMaya.MFn.kMatrixAttribute: OpenMaya.MFnMatrixAttribute,
    OpenMaya.MFn.kMessageAttribute: OpenMaya.MFnMessageAttribute,
    OpenMaya.MFn.kNumericAttribute: OpenMaya.MFnNumericAttribute,
    OpenMaya.MFn.kTypedAttribute: OpenMaya.MFnTypedAttribute,
    OpenMaya.MFn.kUnitAttribute: OpenMaya.MFnUnitAttribute,
}

# Geometry data type strings as reported by `cmds.getAttr(..., type=True)`.
# `cmds.getAttr` cannot serialize these to Python values (it returns None and
# emits a Maya error), so `Attribute.get()` dispatches to a smart fallback
# (connected node, or wrapping fn set) for plugs of these types.
# 'geometry' is the indeterminate type reported when a generic geometry plug
# has no concrete data yet (e.g., a bare skinCluster's outputGeometry).
GEOMETRY_DATA_TYPES = frozenset({"mesh", "nurbsCurve", "nurbsSurface", "geometry"})

# Maps a geometry data MObject's apiType() to the WORKING OpenMaya fn set
# (NOT the *Data container in DATA_TYPE_TO_FN). Used by `Attribute.get()` to
# wrap a plug's computed data when it has no downstream consumer.
GEOMETRY_FN_MAP = {
    OpenMaya.MFn.kMeshData: OpenMaya.MFnMesh,
    OpenMaya.MFn.kNurbsCurveData: OpenMaya.MFnNurbsCurve,
    OpenMaya.MFn.kNurbsSurfaceData: OpenMaya.MFnNurbsSurface,
}

# Maps a geometry data MObject's apiType() to the SHAPE node MFn that would
# natively own that data. Used by `Attribute.get()` to recognize plugs whose
# owning shape *is* the geometry (e.g. mesh.outMesh -> return the Mesh node,
# not an MFnMesh wrapping the data).
SHAPE_FN_FOR_GEOMETRY_DATA = {
    OpenMaya.MFn.kMeshData: OpenMaya.MFn.kMesh,
    OpenMaya.MFn.kNurbsCurveData: OpenMaya.MFn.kNurbsCurve,
    OpenMaya.MFn.kNurbsSurfaceData: OpenMaya.MFn.kNurbsSurface,
}


def _trace_choice_source(plug: OpenMaya.MPlug) -> OpenMaya.MPlug | None:
    """For a `choice` node's `output` plug, return the source plug of the
    currently-selected `input[selector]` element.

    A choice node has no direct connection on its `output` plug -- the value is
    computed from `input[selector]` at evaluation time. To resolve `output` to
    the upstream source shape, read the selector and follow the input plug.
    """
    fn = OpenMaya.MFnDependencyNode(plug.node())
    try:
        selector   = fn.findPlug("selector", False).asInt()
        input_plug = fn.findPlug("input", False).elementByLogicalIndex(selector)
    except RuntimeError:
        return None
    if not input_plug.isDestination:
        return None
    src = input_plug.source()
    return None if src.isNull else src


# Dispatch table for nodes whose geometry output is computed from an upstream
# input rather than directly fed by a connection. Maps the node's `typeName`
# (as reported by `MFnDependencyNode.typeName`) to a callable that takes the
# output `MPlug` and returns the source `MPlug` (or None).
#
# `Attribute.get()` consults this dispatch when no shape-owner match and no
# downstream consumer is found, before falling back to wrapping the data in an
# OpenMaya fn set.
GEOMETRY_ROUTING_TRACERS = {
    "choice": _trace_choice_source,
}

# Node types whose output data type can change at runtime depending on which
# input is currently driving the output (e.g. `choice` selects one of its
# `input[N]` plugs based on `selector`). For these nodes, `Attribute.get()`
# must re-check the data type on every call -- the per-instance type cache
# would otherwise lock in the wrong dispatch when the user flips the selector
# between, e.g., a mesh input and a matrix input.
POLYMORPHIC_OUTPUT_NODE_TYPES = frozenset({"choice"})


@total_ordering
class Attribute(str):
    """
    The attribute class that wraps OpenMaya.MPlug.

    Note: This class inherits `str` so that it can be passed into maya.cmds calls
    that expect strings. It's only necessary since "__getitem__()" is implemented.
    Maybe there is a better way to do this.

    """

    def __init__(self, name_or_mplug: str | OpenMaya.MPlug) -> None:
        """Initialize an instance from an attribute full name or a MPlug."""
        if isinstance(name_or_mplug, str):
            sel = OpenMaya.MSelectionList()
            sel.add(name_or_mplug)
            self._mplug = sel.getPlug(0)
        elif isinstance(name_or_mplug, OpenMaya.MPlug):
            self._mplug = name_or_mplug
        else:
            raise ValueError(f"{name_or_mplug} is not a string or MPlug.")
        self._mobject          = None
        self._fn_set           = None
        self._node             = None
        self.__child_name_dict = {}  # cache queried child attributes
        self.__child_id_dict   = {}  # cache queried child attributes
        self.__component_type = (
            None  # cache componet type str TODO: make a proper Component class
        )
        # cache for `_is_geometry_typed_attr`; None = not yet computed
        self._geometry_attr_cache: bool | None = None
        # cache for `_owner_is_polymorphic`; None = not yet computed
        self._polymorphic_owner_cache: bool | None = None

    # --- dunders

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.full_name}")'

    def __str__(self) -> str:
        return self.full_name

    def __hash__(self) -> int:
        return hash(self.full_name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.full_name == other.full_name

    def __gt__(self, other: Any) -> bool:
        return self.full_name > str(other)

    def __getitem__(self, key: int | slice) -> Attribute | list:
        """Return attribute at a given logical index, if this is a multi-attr.
           If given a slice, return a list of attributes.
        Example:
        ```
        attr = mesh.componentTags[1]
        attrs = blendShape.weight[1:3]
        ```
        """

        if isinstance(key, int):
            return self.element_by_logical_index(key)

        elif isinstance(key, slice):
            # figure out the maximum range of the slice
            # TODO: added special edge case component handling
            #       which should be done in a proper Component class
            comp_type = self._component_type

            # is this a geometry component?
            if comp_type in (
                "kMeshVertComponent",
                "kCurveCVComponent",
                "kSurfaceCVComponent",
            ):
                start, stop, step = key.indices(self.node.num_weight_points)

            # default behavior for a multi attribute
            else:
                # Bounded NON-NEGATIVE slice (``[:6]``, ``[3:9]``) -- honour
                # the explicit ``stop`` as the upper bound. Missing
                # elements get created on access via
                # ``elementByLogicalIndex`` (Maya's standard multi-attr
                # semantics -- same as ``plug[5]``).
                #
                # Unbounded OR negative-stop slice (``[:]``, ``[3:]``,
                # ``[:-2]``) -- use existing max+1 so the slice spans the
                # sparse logical-index range AND ``slice.indices()`` can
                # resolve the negative stop relative to a positive
                # length (``slice.indices()`` raises ``ValueError`` on a
                # negative length argument). On an empty multi this
                # returns ``[]`` instead of raising ``IndexError`` (the
                # original behaviour which broke ``plug[:6]`` on
                # freshly-created multi attrs) or ``ValueError`` (a
                # regression of the original fix which broke
                # ``plug[:-2]``).
                if key.stop is not None and key.stop >= 0:
                    upper = key.stop
                else:
                    existing = self.get_logical_indices()
                    if not existing:
                        return []
                    upper = existing[-1] + 1
                start, stop, step = key.indices(upper)

            attribute_list = []
            for index in range(start, stop, step):
                attribute_list.append(self.element_by_logical_index(index))
            return attribute_list

        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )

    def __delitem__(self, i: int) -> None:
        """Delets an attribute at a given logical index, if this is a multi-attr.

        Example:
        ```
        del mesh.componentTags[1]
        ```
        """
        return self.delete_logical_index(i)

    def __iter__(self, i: int) -> Iterator[Attribute]:
        """Iterating over attrs at each logical index, if this is a multi-attr.

        Example:
        ```
        for attr in mesh.componentTags:
            print(attr)
        ```
        """
        for i in self.get_logical_indices():
            yield self.element_by_logical_index(i)

    def __getattr__(self, attr_name: str) -> Attribute:
        """Implemented to return child attribute by name, if this is a compound attr.

        Note:
        For small number of queries, this is slower than query child by index
        `child(index)`. It becomes faster when dealing with large amount queries
        thanks to caching.

        Example:
        ```
        attr = mesh.componentTags[1].componentName
        ```
        """
        if not self.__child_name_dict:
            for i in range(self.num_children):
                child_plug = self.plug.child(i)
                name = child_plug.partialName(False, False, False, False, False, True)
                name = name.rsplit(".", 1)[-1]
                attr = Attribute(child_plug)
                self.__child_name_dict[name] = attr
                self.__child_id_dict[i]      = attr
        if attr_name in self.__child_name_dict:
            return self.__child_name_dict[attr_name]
        raise AttributeError(f"{self.full_name}.{attr_name} not found.")

    # --- properties

    @property
    def plug(self) -> OpenMaya.MPlug:
        """Returns the mplug."""
        return self._mplug

    @property
    def fn_set(self) -> OpenMaya.MFnBase:
        """Returns the attribute function set."""
        if not self._fn_set:
            mobject      = self.mobject
            api_type     = mobject.apiType()
            data_fn      = ATTR_TYPE_TO_FN.get(api_type, OpenMaya.MFnAttribute)
            self._fn_set = data_fn(mobject)
        return self._fn_set

    @property
    def mobject(self) -> OpenMaya.MObject:
        """Returns the mobject."""
        if not self._mobject:
            self._mobject = self._mplug.attribute()
        return self._mobject

    @property
    def node(self) -> Any:
        """Returns the node object of this attr."""
        if not self._node:
            # use local import here to avoid circular dependency
            from rig.maya.nodetypes._base import PyNode

            self._node = PyNode(self.plug.node())
        return self._node

    @property
    def name(self) -> str:
        """Returns the attribute name (without node name)."""
        return self.plug.partialName(False, False, False, False, False, True)

    @property
    def full_name(self) -> str:
        """Returns the full attribute name (with node name), use alias if exists.

        Note. Can't use MPlug.name() directly because it doesn't use the partial node name.
        i.e. it will error if duplicated node names exist.
        """
        return f"{self.node.name}.{self.alias}"

    @property
    def alias(self) -> str:
        """Returns the attribute alias, or the attribute name if no alias applied."""
        return self.plug.partialName(False, False, False, True, False, True)

    @alias.setter
    def alias(self, alias: str) -> None:
        """Sets the alias for this attribute. Can be set to None."""
        cur_alias = self.alias
        if cur_alias == alias:
            return
        elif not alias and cur_alias != self.name:
            cmds.aliasAttr(self.full_name, remove=True)
        else:
            cmds.aliasAttr(alias, self.full_name)

    @property
    def attribute_type(self) -> str:
        """Returns the type of this attribute."""
        return cmds.attributeQuery(self.name, node=self.node.name, attributeType=True)

    @property
    def data_type(self) -> str:
        """Returns the data type of the value hosted by this attribute."""
        typ = cmds.getAttr(self.full_name, type=True)

        # maintain consistent type string with cmds.addAttr()
        if typ == "TdataCompound":
            return "compound"

        # if cmds.getAttr fails to resolve, call the node fallback hook
        # e.g. if a choice node's inputs are message attrs, its output will be resolved
        # to "typed" rather than "message"
        if typ in ("typed", "Tdata"):
            return self.node._attr_data_type_fallback(self)

        return typ

    @property
    def is_dynamic(self) -> bool:
        """Attribute dynamic state."""
        return self.plug.isDynamic

    @property
    def is_locked(self) -> bool:
        """Attribute locked state."""
        return self.plug.isLocked

    @is_locked.setter
    def is_locked(self, state) -> None:
        """Set the locked state."""
        cmds.setAttr(self.full_name, lock=state)

    @property
    def is_keyable(self) -> bool:
        """Attribute keyable state."""
        return self.plug.isKeyable

    @is_keyable.setter
    def is_keyable(self, state) -> None:
        """Set the keyable state."""
        cmds.setAttr(self.full_name, keyable=state)

    @property
    def is_channel_box(self) -> bool:
        """Attribute channel box state."""
        return self.plug.isChannelBox

    @is_channel_box.setter
    def is_channel_box(self, state) -> None:
        """Set the channel box state."""
        cmds.setAttr(self.full_name, channelBox=state)

    # --- category methods

    def get_categories(self) -> list[str]:
        """Returns a list of categories this attribute belongs to."""
        return (
            cmds.attributeQuery(self.name, node=self.node.name, categories=True) or []
        )

    def has_category(self, category: str | list[str]) -> bool:
        """Checks if this attribute belongs to the given category(ies)."""
        if isinstance(category, str):
            category = [category]
        return len(set(category) - set(self.get_categories())) == 0

    def add_category(self, category: str | list[str]) -> None:
        """Adds the given category to this attribute."""
        if self.has_category(category):
            return
        if isinstance(category, str):
            category = [category]
        for each in category:
            cmds.addAttr(self.full_name, edit=True, category=each)

    # --- connection methods

    @property
    def is_connected(self) -> bool:
        """Attribute connected state. Both input and output connections counts."""
        return self.plug.isConnected

    @property
    def is_free_to_change(self) -> bool:
        """Attribute free_to_change state. True if this attr and all its parents are
        free to change."""
        return self.plug.isFreeToChange(True, False) == OpenMaya.MPlug.kFreeToChange

    def connect(self, other: str | Attribute, force: bool = False) -> None:
        """Connects this attr to other.

        Args:
            other: The attribute to connect to this attr.
            force: If True, break existing connection if found.
        """
        other = str(other)
        if not cmds.isConnected(self.full_name, other):
            cmds.connectAttr(self.full_name, other, force=force)

    def disconnect(self, other: str | Attribute) -> None:
        """Disconnects this attr from other.

        Args:
            other: The attribute to disconnect from this attr.
        """
        _self = self.full_name
        other = str(other)
        if cmds.isConnected(_self, other):
            cmds.disconnectAttr(_self, other)

    def iter_connected_attrs(
        self, src: bool = True, dst: bool = True, first_only: bool = False
    ) -> Iterator[Attribute]:
        """Iterates over connected attributes.

        Args:
            src: If True, include source attributes.
            dst: If True, include destination attributes.
            first_only: If True, return only the first connected attr.

        Returns:
            The connected attrs.
        """
        for each in self.plug.connectedTo(src, dst):
            yield Attribute(each)
            if first_only:
                break

    def get_connected_attrs(
        self, src: bool = True, dst: bool = True, first_only: bool = False
    ) -> Attribute | list[Attribute] | None:
        """Returns connected attributes.

        Args:
            src: If True, include source attributes.
            dst: If True, include destination attributes.
            first_only: If True, return only the first connected attr.

        Returns:
            The connected attrs.
        """
        it = self.iter_connected_attrs(src, dst, first_only=first_only)
        if first_only:
            return next(it, None)
        return list(it)

    def list_connections(self, **kwargs) -> list[Any]:
        """Lists connected attrs on this node. Thin wrapper of
        cmds.listConnections()."""
        # use local import here to avoid circular dependency
        from rig.maya.nodetypes._base import PyNode

        casted = {}
        result = []
        for each in cmds.listConnections(self.full_name, **kwargs) or []:
            obj = casted.get(each)
            if not obj:
                obj          = PyNode(each)
                casted[each] = obj
            result.append(obj)
        return result

    def find_connected_nodes(
        self,
        depth:              int              = 0,
        node_type:          str       | None = None,
        source:             bool             = True,
        destination:        bool             = True,
        exclude_node_types: list[str] | None = None,
        exclude_nodes:      list[str] | None = None,
        delete_found_nodes: bool             = False,
        connections:        bool             = False,
        _cur_depth:         int              = 0,
        _processed:         set[Any]  | None = None,
    ) -> list[Any]:
        """Find connected nodes to this attribute.

        Args:
            node_type: Node type filter.
            depth: Depth level to search. 0 means searching the immediate connections.
            source: If True, search source connections.
            destination: If True, search destination connections.

        Returns:
            A list of nodes found.
        """
        # use local import here to avoid circular dependency
        from rig.maya.nodetypes._base import PyNode

        default_excludes = ["defaultShaderList1", "time1", "renderPartition"]
        if not _processed:
            _processed = {PyNode(x) for x in default_excludes}
        if exclude_nodes:
            for x in exclude_nodes:
                if cmds.objExists(x):
                    _processed.add(PyNode(x))

        connections = cmds.listConnections(
            str(self),
            source      = source,
            destination = destination,
            connections = connections,
            plugs       = False,
        )

        processed = _processed
        result    = []
        if not connections:
            return result

        for c in connections:
            c = PyNode(c)
            if c in processed:
                continue
            processed.add(c)

            if not node_type or c.node_type == node_type:
                if exclude_node_types and c.node_type in exclude_node_types:
                    continue
                result.append(c)
            if _cur_depth < depth:
                result.extend(
                    c.find_connected_nodes(
                        depth=depth,
                        node_type=node_type,
                        source=source,
                        destination=destination,
                        exclude_node_types=exclude_node_types,
                        _cur_depth=_cur_depth + 1,
                        _processed=processed,
                    )
                )

        if delete_found_nodes:
            to_delete = []
            for i in range(len(result) - 1, -1, -1):
                node = result[i]
                if not cmds.lockNode(node, query=True)[0]:
                    to_delete.append(node)
                    result.pop(i)
            if to_delete:
                cmds.delete(to_delete)

        return result

    def break_connections(
        self,
        src: bool = True,
        dst: bool = True,
    ) -> list[Attribute]:
        """Breaks connections and returns the old connected attributes.

        Args:
            src: If True, break source connections.
            dst: If True, break destination connections.

        Returns:
            The disconnected attrs.
        """
        attrs = self.get_connected_attrs(src=src, dst=dst)
        for attr in attrs:
            attr_name = str(attr)
            if cmds.isConnected(self.full_name, attr_name):
                src_name = self.full_name
                dst_name = attr_name
            else:
                src_name = attr_name
                dst_name = self.full_name
            cmds.disconnectAttr(src_name, dst_name)
        return attrs

    def __rshift__(self, other):
        """Right shift operator ">>" to force connect."""
        self.connect(other, force=True)

    def __floordiv__(self, other):
        """Floor divide operator "//" to disconnect."""
        self.disconnect(other)

    # --- value methods

    @property
    def default_value(self) -> Any:
        """Returns the default value of this attr, if any."""
        val = cmds.attributeQuery(self.name, node=self.node, listDefault=True)
        if val is not None and len(val) == 1:
            return val[0]
        return val

    @default_value.setter
    def default_value(self, value):
        """Sets the default value of this attribute."""
        if not self.is_dynamic:
            raise RuntimeError(f"Cannot set default value for native attr: {self}")
        cmds.addAttr(self.full_name, edit=True, defaultValue=value)

    def get_data_fn_set(self) -> OpenMaya.MFnData | None:
        """Returns the proper data function set for this attr, or None if
        this attribute holds no data, or plug.asMObject() causes internal failure."""
        data_mobject = self.plug.asMDataHandle().data()
        api_type     = data_mobject.apiType()
        if api_type != OpenMaya.MFn.kInvalid:
            data_fn = DATA_TYPE_TO_FN.get(api_type, OpenMaya.MFnData)
            return data_fn(data_mobject)
        return None

    def set(self, *args, **kwargs) -> None:
        """Sets the attribute to the given value. Thin wrapper around cmds.setAttr().

        Convenience features:
            - automatically assign `type` argument, if not provided.

        Args:
            args, kwargs: args supported by cmds.setAttr()
        """
        if "type" not in kwargs:
            # typed attr requires the `type` arg to be specified.
            # the only weird one-off is `fltMatrix`, which is not typed but still
            # requires the `type` arg.
            typ = self.data_type
            if self.is_typed or typ == "matrix":
                kwargs["type"] = typ

        # special handling for compound attrs
        # - use the unpacked first argument if it is a valid iterable
        #   and attr type ends with a digit (double2, double3, etc.)
        if (
            len(args) == 1
            and isinstance(args[0], (list, tuple, np.ndarray))
            and str(self.data_type)[-1].isdigit()
        ):
            cmds.setAttr(self.full_name, *args[0], **kwargs)
        else:
            cmds.setAttr(self.full_name, *args, **kwargs)

    def get(self) -> Any:
        """Returns the attribute value.

        Mirrors `cmds.getAttr()` for any value it can resolve (numerics,
        strings, matrices, compounds, typed arrays). The connection state of
        the plug is irrelevant: a connected `translateX` still returns the
        computed number.

        For typed *geometry* data plugs (mesh, nurbsCurve, nurbsSurface, etc.)
        `cmds.getAttr` is bypassed (it would emit a Maya error and return None
        -- see `_is_geometry_typed_attr`). Resolution order is:

          1. If the owning node is a SHAPE matching the plug's geometry type,
             returns the shape itself (e.g. ``mesh.outMesh`` -> ``Mesh`` node).
          2. Otherwise, if the plug has a downstream consumer, returns that
             node (e.g. ``skinCluster.outputGeometry[0]`` -> ``Mesh`` of the
             consuming shape).
          2.5. Otherwise, if the owning node is a known geometry-routing node
             (see ``GEOMETRY_ROUTING_TRACERS``, e.g. ``choice``), walks
             upstream via the registered tracer to find the source shape and
             returns it.
          3. Otherwise, returns the matching OpenMaya fn set wrapping the
             plug's computed data (``MFnMesh``, ``MFnNurbsCurve``, ...).
          4. Otherwise returns None.

        Note: case 3 uses `MPlug.asMObject()` because it forces DG evaluation,
        ensuring the returned data contains real values rather than the
        uninitialized memory `MPlug.asMDataHandle().data()` would expose for an
        unevaluated plug. The returned fn set wraps an MObject owned by the
        source node's data block -- if the source node is deleted or its inputs
        change, the fn set becomes invalid.
        """
        # Geometry data attrs need special handling: cmds.getAttr would emit
        # `# Error: The data is not a numeric or string value...` and return
        # None. Skip it entirely for known geometry types.
        if self._is_geometry_typed_attr:
            return self._get_geometry_value()

        return cmds.getAttr(self.full_name)

    @property
    def _is_geometry_typed_attr(self) -> bool:
        """True if this attr is typed to hold geometry data.

        Used by `get()` to skip `cmds.getAttr()` for these -- it would emit a
        ``# Error: The data is not a numeric or string value...`` message to
        the script editor before returning None. Cached per Attribute instance.

        Note: uses `cmds.getAttr(..., type=True)` rather than
        `MFnTypedAttribute.attrType()` because the latter does not work for
        generic geometry attrs (e.g. `skinCluster.outputGeometry[0]` is
        declared as kGenericAttribute and resolves to type 'mesh' /
        'nurbsCurve' / etc. only via `cmds.getAttr(type=True)`).
        `cmds.getAttr(type=True)` does NOT trigger the displayError emission.

        For polymorphic-output owners (see ``POLYMORPHIC_OUTPUT_NODE_TYPES``,
        e.g. ``choice``) the cache is bypassed because the data type changes
        per-evaluation based on the active input.
        """
        if self._geometry_attr_cache is not None and not self._owner_is_polymorphic:
            return self._geometry_attr_cache

        try:
            typ = cmds.getAttr(self.full_name, type=True)
        except RuntimeError:
            result = False
        else:
            result = typ in GEOMETRY_DATA_TYPES

        if not self._owner_is_polymorphic:
            self._geometry_attr_cache = result
        return result

    @property
    def _owner_is_polymorphic(self) -> bool:
        """True if the owning node has a polymorphic-output type (e.g. choice).

        Cached per Attribute instance because the owning node never changes
        for the lifetime of an Attribute.
        """
        if self._polymorphic_owner_cache is None:
            try:
                type_name = OpenMaya.MFnDependencyNode(self.plug.node()).typeName
                self._polymorphic_owner_cache = (
                    type_name in POLYMORPHIC_OUTPUT_NODE_TYPES
                )
            except RuntimeError:
                self._polymorphic_owner_cache = False
        return self._polymorphic_owner_cache

    def _get_geometry_value(self) -> Any:
        """Resolve the value for a typed geometry data plug.

        See `get()` for the four-tier resolution order.
        """
        # Local import to avoid a circular dependency with nodetypes._base.
        from rig.maya.nodetypes._base import PyNode

        owner_obj = self.plug.node()
        is_shape  = owner_obj.hasFn(OpenMaya.MFn.kShape)

        # Eagerly fetch the data once -- used both for the shape-match check
        # below and for the final MFn wrap if we fall through.
        try:
            data = self.plug.asMObject()
        except RuntimeError:
            data = None

        expected_shape_fn = (
            SHAPE_FN_FOR_GEOMETRY_DATA.get(data.apiType())
            if data is not None and not data.isNull()
            else None
        )

        # 1. If the owning node IS the geometry of this plug's data type,
        #    return the shape itself (e.g. mesh.outMesh -> Mesh node).
        if is_shape and expected_shape_fn and owner_obj.hasFn(expected_shape_fn):
            return PyNode(owner_obj).serialize()

        # 2. Prefer a real downstream node consumer.
        dests = self.plug.destinations()
        if dests:
            return PyNode(dests[0].node()).serialize()

        # 2.5. Known geometry-routing nodes (e.g. choice node): walk upstream via
        #      the registered tracer to find the source shape.
        if expected_shape_fn:
            owner_type = OpenMaya.MFnDependencyNode(owner_obj).typeName
            tracer     = GEOMETRY_ROUTING_TRACERS.get(owner_type)
            if tracer is not None:
                src_plug = tracer(self.plug)
                if src_plug is not None and src_plug.node().hasFn(expected_shape_fn):
                    return PyNode(src_plug.node()).serialize()

        # 3. Wrap the computed data via the matching MFn fn set.
        if data is None or data.isNull() or data.apiType() == OpenMaya.MFn.kInvalid:
            return None
        fn_class = GEOMETRY_FN_MAP.get(data.apiType())
        if fn_class is None:
            return None
        # The data MObject's apiType can claim to be e.g. kMeshData while the
        # underlying buffer is empty/uninitialized; the fn set constructor
        # rejects such MObjects with ValueError -- treat that as "no data".
        try:
            return fn_class(data)
        except (ValueError, RuntimeError):
            return None

    @property
    def enums(self) -> list[str]:
        """Returns a list of enum values if this attr is an enum attr."""
        enums = cmds.attributeQuery(self.name, node=self.node, listEnum=True)
        if enums:
            return enums[0].split(":")
        return []

    # --- compound attr methods

    @property
    def num_children(self):
        """Returns this number of children of this compound attr."""
        return self.plug.numChildren()

    def get_parent(self) -> Attribute | None:
        """Returns the parent attribute, if any."""
        plug = self.plug.parent()
        if plug and not plug.attribute().isNull():
            return Attribute(plug)

    def child(self, i: int) -> Attribute:
        """Returns the child attribute at the given index."""
        attr = self.__child_id_dict.get(i)
        if not attr:
            self.__child_id_dict[i] = attr = Attribute(self.plug.child(i))
        return attr

    # --- typed attr methods

    @property
    def is_typed(self):
        """Checks if this attribute is typed."""
        return self.mobject.hasFn(OpenMaya.MFn.kTypedAttribute)

    def filter_array_values(
        self, filter_value: Any, sparse: bool = True, cap_count: int | None = None
    ) -> tuple[list[int], list[Any]]:
        """filters an value out of a typed array attr.

        Args:
            filter_value: The value to filter out.
            cap_count: If specified, caps the return ids and values to this count.

        Returns:
            [sparse_ids, sparse_values]
        """
        if not self.is_typed:
            raise RuntimeError(f"{self} is not a typed attribute.")
        if not self.data_type.lower().endswith("array"):
            raise RuntimeError(f"{self} is not a typed array attribute.")

        spase_ids    = []
        spase_values = []

        values = self.get()
        if not values:
            return spase_ids, spase_values

        is_num = isinstance(values[0], Number)
        if not isinstance(values[0], type(filter_value)):
            raise RuntimeError(
                f"Default value {filter_value} doesn't match data type in {self}"
            )

        # cap the return values to cap_count, if requested
        if cap_count is not None and len(values) > cap_count:
            values = values[:cap_count]

        if not sparse:
            ids = list(range(len(values)))
            return ids, values

        for i, val in enumerate(values):
            if (is_num and not math.isclose(val, filter_value, rel_tol=1e-4)) or (
                not is_num and val != filter_value
            ):
                spase_ids.append(i)
                spase_values.append(val)
        return spase_ids, spase_values

    # --- multi/array attr methods
    #
    # OpenMaya refers to multi attrs as arrays, which is confusing since there
    # are also typed array attrs such as `doubleArray`. We use the term `multi`
    # here to for clarity.

    @property
    def is_multi(self) -> bool:
        """Checks if this attr is a multi/array attribute."""
        return self.plug.isArray

    @property
    def num_elements(self) -> int:
        """Number of elements / physical indices in the multi attr."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        return self.plug.numElements()

    def logical_index(self) -> int:
        """Returns the logical index if this attr is a child of a multi attr."""
        return self.plug.logicalIndex()

    def get_logical_indices(self) -> OpenMaya.MIntArray:
        """Returns existing logical indices (sparse)."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        return self.plug.getExistingArrayAttributeIndices()

    def get_next_available_index(self) -> int:
        """Returns the next available logical index."""
        ids = set(self.get_logical_indices())
        for index in range(len(ids) + 1):
            if index not in ids:
                return index

    def element_by_physical_index(self, i: int) -> Attribute:
        """Returns the element attribute at the given physical index."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        return Attribute(self.plug.elementByPhysicalIndex(i))

    def element_by_logical_index(self, i: int) -> Attribute:
        """Returns the element attribute at the given logical index."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        return Attribute(self.plug.elementByLogicalIndex(i))

    def delete_logical_index(self, i: int, **kwargs) -> None:
        """Deletes the element attribute at the given logical index."""
        cmds.removeMultiInstance(self.element_by_logical_index(i).full_name, **kwargs)

    # --- component type info
    # TODO make a proper Component class

    def _get_component(self):
        """
        Returns the component object assigned to this attribute.
        eg: pCubeShape1.vtx[0] -> OpenMaya.MFn.kMeshVertComponent
        """
        try:
            # indexed component, eg: pCubeShape1.pnts[0]
            try:
                tracker = OpenMaya.MSelectionList()
                tracker.add(str(self))
                return tracker.getComponent(0)[-1]

            # non-indexed component, eg: pCubeShape1.pnts
            except TypeError:
                tracker = OpenMaya.MSelectionList()
                tracker.add(f"{self}[*]")
                return tracker.getComponent(0)[-1]

        # this is not related to a component
        except Exception:
            pass

        return None

    @property
    def _component_type(self) -> str:
        """Returns the geometry component type assigned to this attribute, or 'unknown' if this attribute doesn't interface a component."""
        if self.__component_type is None:
            comp                  = self._get_component()
            self.__component_type = comp.apiTypeStr if comp else "unknown"

        return self.__component_type