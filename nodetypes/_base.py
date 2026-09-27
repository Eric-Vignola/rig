"""
Node base classes, the Attribute wrapper and utils
"""

from __future__ import annotations

import itertools
import math
import re
import sys
import uuid
from functools import total_ordering
from numbers import Number
from typing import Any, Iterator

import numpy as np
from maya import cmds, OpenMaya as OpenMaya1
from maya.api import OpenMaya
from rig._internal import callbacks as _callbacks


CUSTOM_TYPE_ATTR = "__custom_node_type__"

# a node name MSelectionList can resolve without wildcards, plugs or components
_PLAIN_NODE_NAME = re.compile(r"^[|:\w]+$")


def is_valid_maya_uid(uid_string: str) -> bool:
    """
    Validates if a string matches Maya's unique ID format using the uuid module.
    """
    try:
        uuid.UUID(uid_string)
        return True
    except ValueError:
        return False


# `_class_attr`'s answer for a name no class in the MRO defines
_MISSING = object()


def _class_attr(cls: type, name: str) -> Any:
    """The class attribute `name` as the first class of `cls`'s MRO that defines it
    holds it (unbound: a property, function, descriptor or value), or `_MISSING`.
    Used by the `=` sugar to tell a Python member from a Maya attribute."""
    for base in cls.__mro__:
        found = base.__dict__.get(name, _MISSING)
        if found is not _MISSING:
            return found
    return _MISSING


class NodeMeta(type):
    """
    A metaclass that register node classes to the cached dict in PyNode.
    """

    def __new__(mcs, class_name, bases, attrs):
        cls_obj   = type.__new__(mcs, class_name, bases, attrs)

        # a new class can change the dispatch of any node type
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        _STATIC_DATA_TYPE.clear()
        _CANONICAL_KIND.clear()

        node_type = attrs.get("CUSTOM_NODE_TYPE")
        if not node_type:
            node_type = attrs.get("NATIVE_NODE_TYPE")
        if node_type:
            PyNode._NODE_CLASS_DICT[node_type] = cls_obj

        return cls_obj

    def __setattr__(cls, name, value):
        type.__setattr__(cls, name, value)
        # a class attribute can change which wrappers of it are canonical
        _CANONICAL_KIND.clear()

    def __delattr__(cls, name):
        type.__delattr__(cls, name)
        _CANONICAL_KIND.clear()


class PyNode:
    """
    A factory class that for casting arbitrary objects into the most suitable
    object classes.

    **If you know the type of the object you are working with, it is recommended to
    explicitly cast it into the associated class instead of going through the `PyNode`
    factory process, which yields better performance.**
    """

    _NODE_CLASS_DICT = {}
    _CLASS_BY_TYPE   = {}     # (typeName, typeId) -> class of a node without custom type
    _CASTABLE_TYPES  = set()  # (typeName, typeId) keys an MObject was cast from

    def __new__(
        cls,
        obj: str | OpenMaya.MObject | OpenMaya.MDagPath | OpenMaya.MPlug,
        *args,
        **kwargs,
    ) -> Any:
        """Cast a object (node or attr) into the most suitable class.

        Args:
            obj: Node: name string, MObject, or MDagPath.
                Attribute: name string or MPlug.

        Returns:
            An object instance.
        """
        super(PyNode, cls).__new__(cls, *args, **kwargs)

        mobj     = None
        dag_path = None
        if isinstance(obj.__class__, NodeMeta) or isinstance(obj, Attribute):
            return obj
        elif isinstance(obj, OpenMaya.MPlug):
            return Attribute(obj)
        elif isinstance(obj, OpenMaya.MObject):
            mobj = obj
            obj  = _mobject_to_str(obj)
        elif isinstance(obj, OpenMaya.MDagPath):
            dag_path = obj
            obj      = obj.partialPathName()
        elif not isinstance(obj, str):
            t = type(obj)
            raise ValueError(f"{obj} ({t}) is not a str, MObject, MDagPath, or MPlug")

        # if obj is given as a unique id, convert to string
        # (uuid.UUID() needs 32 hex digits, so a shorter string is never a uid)
        if len(obj) >= 32 and is_valid_maya_uid(obj):
            str_from_uid = cmds.ls(obj, uid=True)
            if not str_from_uid:
                raise TypeError(f"No object matches uuid: {obj}.")

            # node name is properly converted to a name string
            obj      = str_from_uid[0]
            mobj     = None
            dag_path = None

        # TODO need a better way to identify attribute strings
        if obj.rfind(".") != -1:
            return Attribute(obj)

        # without a custom type attr (or an alias of that name) the class only
        # depends on the node type, so it is looked up per (typeName, typeId);
        # anything that doesn't resolve to exactly one node takes the legacy path,
        # and so does a deleted or undone node, whose name may now be another's
        from_mobject = mobj is not None
        key          = None
        try:
            if dag_path is not None:
                mobj = dag_path.node()
            elif mobj is None and _PLAIN_NODE_NAME.match(obj):
                sel = OpenMaya.MSelectionList()
                sel.add(obj)
                if sel.length() == 1:
                    mobj = sel.getDependNode(0)
            handed_in = from_mobject or dag_path is not None
            if handed_in and not OpenMaya.MObjectHandle(mobj).isValid():
                mobj = None
            if mobj is not None:
                fn = OpenMaya.MFnDependencyNode(mobj)
                if (
                    not fn.hasAttribute(CUSTOM_TYPE_ATTR)
                    and fn.findAlias(CUSTOM_TYPE_ATTR).isNull()
                ):
                    key = (fn.typeName, fn.typeId.id())
        except (RuntimeError, ValueError, TypeError):
            key = None
        if key is None:
            return _pynode_legacy_tail(cls, obj)

        if key in cls._CLASS_BY_TYPE:
            cls_obj = cls._CLASS_BY_TYPE[key]
        else:
            cls_obj = cls._CLASS_BY_TYPE[key] = _native_node_class(cls, obj)
        if not cls_obj:
            raise ValueError(f"Failed casting {obj}")

        # a type already cast from an MObject passes the class's type check
        # again, so a base-constructor class is built without re-running it
        inst = None
        if from_mobject and key in cls._CASTABLE_TYPES:
            inst = _construct_checked_type(cls_obj, obj)
        if inst is None:
            inst = cls_obj(obj)
        if from_mobject:
            cls._CASTABLE_TYPES.add(key)
        return inst

    @classmethod
    def create(cls, node_type, *args, **kwargs) -> Any:
        """Thin wrapper around cmds.createNode()."""
        node_cls = cls._NODE_CLASS_DICT.get(node_type)
        if node_cls:
            return node_cls.create(*args, **kwargs)
        else:
            result = cmds.createNode(node_type, *args, **kwargs)
            if isinstance(result, (list, tuple)):
                return [cls(x) for x in result]
            elif result:
                return cls(result)
            return result

    @classmethod
    def find_all(cls, node_type: str, exact_type: bool = True) -> list[Any]:
        """Thin wrapper around cmds.ls()."""
        node_cls = cls._NODE_CLASS_DICT.get(node_type)
        if node_cls:
            return node_cls.find_all(exact_type=exact_type)
        else:
            raise NotImplementedError(f"Node type {node_type} not implemented")


def _native_node_class(cls, obj: str) -> Any:
    """Returns the class the node's type chain maps to, ignoring any custom type."""
    # set default node type to DAG or DG
    default_type = "dagNode" if cmds.ls(obj, dag=True) else "entity"
    cls_obj      = cls._NODE_CLASS_DICT.get(default_type)

    # override cls_obj with a defined node class if any
    for t in reversed(cmds.nodeType(obj, inherited=True)):
        if t in cls._NODE_CLASS_DICT:
            cls_obj = cls._NODE_CLASS_DICT.get(t)
            break
    return cls_obj


def _pynode_legacy_tail(cls, obj: str) -> Any:
    """Casts a node name string by querying its custom type and type chain by name."""
    cls_obj     = None
    custom_type = get_custom_type(obj)
    if custom_type:
        cls_obj = cls._NODE_CLASS_DICT.get(custom_type)
    if not cls_obj:
        cls_obj = _native_node_class(cls, obj)

    if cls_obj:
        return cls_obj(obj)

    raise ValueError(f"Failed casting {obj}")


# (DGNode.__init__, DAGNode.__init__, their is_type functions, their name
# properties, DGNode._cache_api1_objects), bound on first use since dg_node
# imports this module
_BASE_CONSTRUCTOR_PARTS = None


def _construct_checked_type(cls_obj, obj: str) -> Any:
    """Builds `cls_obj(obj)` without its `is_type` check, or returns None.

    For a class that keeps the DGNode or DAGNode constructor, type check, name
    property and API 1.0 cache, `is_type` only depends on the node type and the
    custom type attr, so it passes for every node of a type that was already
    cast. The constructor's state is set in the constructor's order. Returns
    None when the class does not qualify or a step raises, and the caller then
    runs the real constructor, which raises its own errors.
    """
    global _BASE_CONSTRUCTOR_PARTS
    if _BASE_CONSTRUCTOR_PARTS is None:
        from rig.nodetypes.dag_node import DAGNode
        from rig.nodetypes.dg_node import DGNode

        _BASE_CONSTRUCTOR_PARTS = (
            DGNode.__init__,
            DAGNode.__init__,
            (
                DGNode.__dict__["is_type"].__func__,
                DAGNode.__dict__["is_type"].__func__,
            ),
            (DGNode.__dict__["name"], DAGNode.__dict__["name"]),
            DGNode._cache_api1_objects,
        )
    dg_init, dag_init, is_type_funcs, name_props, cache_api1 = _BASE_CONSTRUCTOR_PARTS

    init = cls_obj.__init__
    if (
        (init is not dg_init and init is not dag_init)
        or cls_obj.__new__ is not object.__new__
        or cls_obj.CUSTOM_NODE_TYPE
        or getattr(cls_obj.is_type, "__func__", None) not in is_type_funcs
        or cls_obj.name not in name_props
        or cls_obj._cache_api1_objects is not cache_api1
    ):
        return None
    try:
        sel = OpenMaya.MSelectionList()
        sel.add(str(obj))
        inst = object.__new__(cls_obj)
        d    = inst.__dict__
        if init is dag_init:
            d["_mdagpath"] = sel.getDagPath(0)
            d["_mobject"]  = d["_mdagpath"].node()
            d["_fn_set"]   = cls_obj.FN_SET(d["_mdagpath"])
            inst._cache_api1_objects(d["_fn_set"].partialPathName())
        else:
            d["_mobject"] = sel.getDependNode(0)
            d["_fn_set"]  = cls_obj.FN_SET(d["_mobject"])
            inst._cache_api1_objects(d["_fn_set"].name())
        d["_attr_dict"] = {}
    except Exception:
        return None
    return inst


# The canonical-wrapper check below (`_REUSABLE_WRAPPER_PARTS` to `_copy_wrapper`,
# and NodeMeta's clearing of `_CANONICAL_KIND`) is not used by the package since
# the round-3 owner rule (a plug's node is the node object it was read from).
# It is kept, unchanged, for the TestCanonicalWrapperCheck test ids.

# ((DGNode.__init__, DAGNode.__init__, Geometry.__init__), their is_type
# functions, their name properties, DGNode._cache_api1_objects, DGNode's fn_set
# property), bound on first use since dg_node imports this module
_REUSABLE_WRAPPER_PARTS = None

# class -> how `_wrapper_is_canonical` names that class's wrappers: 0 when the
# class does not keep a base constructor, type check, name property and API 1.0
# cache, otherwise one of the kinds below. Only a class whose whole MRO is node
# classes is kept, since NodeMeta clears it when a class is created or a node
# class attribute is set or deleted
_CANONICAL_KIND = {}
_NAMED_DG       = 1  # DGNode's name property
_NAMED_DAG      = 2  # DAGNode's name property
_NAMED_DAG_PATH = 3  # DAGNode's name property, constructor and DGNode's fn_set


def _canonical_kind(cls_obj: type) -> int:
    """Returns the `_CANONICAL_KIND` kind of `cls_obj` and memoizes it when NodeMeta
    can clear it. An error of the class checks propagates and memoizes nothing.
    """
    global _REUSABLE_WRAPPER_PARTS
    if _REUSABLE_WRAPPER_PARTS is None:
        from rig.nodetypes.dag_node import DAGNode
        from rig.nodetypes.dg_node import DGNode
        from rig.nodetypes.geometry import Geometry

        _REUSABLE_WRAPPER_PARTS = (
            (DGNode.__init__, DAGNode.__init__, Geometry.__init__),
            (
                DGNode.__dict__["is_type"].__func__,
                DAGNode.__dict__["is_type"].__func__,
            ),
            (DGNode.__dict__["name"], DAGNode.__dict__["name"]),
            DGNode._cache_api1_objects,
            DGNode.__dict__["fn_set"],
        )
    inits, is_type_funcs, name_props, cache_api1, fn_set_prop = _REUSABLE_WRAPPER_PARTS

    if (
        cls_obj.__init__ not in inits
        or cls_obj.__new__ is not object.__new__
        or cls_obj.CUSTOM_NODE_TYPE
        or getattr(cls_obj.is_type, "__func__", None) not in is_type_funcs
        or cls_obj.name not in name_props
        or cls_obj._cache_api1_objects is not cache_api1
    ):
        kind = 0
    elif cls_obj.name is name_props[0]:
        kind = _NAMED_DG
    elif (
        cls_obj.__init__ is not inits[0]
        and getattr(cls_obj, "fn_set", None) is fn_set_prop
    ):
        kind = _NAMED_DAG_PATH
    else:
        kind = _NAMED_DAG
    if all(isinstance(base, NodeMeta) for base in cls_obj.__mro__[:-1]):
        _CANONICAL_KIND[cls_obj] = kind
    return kind


def _is_only_path(fn: OpenMaya.MFnDagNode, mobject: OpenMaya.MObject) -> bool:
    """Returns True if `fn` is attached to a valid path to the DAG node `mobject`
    and that node has no other path, so `_mobject_to_str(mobject)` names that
    path. False when unsure or when a step raises.
    """
    try:
        if (
            mobject.isNull()
            or mobject.apiType() in _NOT_NODE_API_TYPES
            or not mobject.hasFn(OpenMaya.MFn.kDagNode)
            or fn.isInstanced(True)
        ):
            return False
        path = fn.getPath()
        return path.isValid() and path.pathCount() == 1 and path.node() == mobject
    except Exception:
        return False


def _wrapper_is_canonical(dg_node: Any, mobject: OpenMaya.MObject) -> bool:
    """Returns True if `PyNode(mobject)` would now build a wrapper equal to `dg_node`.

    That holds when `dg_node` wraps the live node `mobject`, its class is the one
    `PyNode` dispatches the node to and keeps a base constructor, type check, name
    property and API 1.0 cache, and it has the name the cast would construct from.
    The two wrappers then differ only in their caches. Returns False in any other
    case or when a step raises, and the caller casts as usual, which raises its
    own errors.
    """
    cls_obj = type(dg_node)
    kind    = _CANONICAL_KIND.get(cls_obj)
    if kind is None:
        kind = _canonical_kind(cls_obj)
    if not kind:
        return False
    try:
        # the API 1.0 handle goes first, it is safe on a node a new scene freed
        # NW6: API 1.0 handle read (a dead helper M8 deletes)
        if not dg_node._objhandle1.isValid() or mobject != dg_node._mobject:
            return False
        # the wrapper's own fn set is attached to this node
        fn = dg_node._fn_set
        if (
            fn.hasAttribute(CUSTOM_TYPE_ATTR)
            or not fn.findAlias(CUSTOM_TYPE_ATTR).isNull()
        ):
            return False
        key = (fn.typeName, fn.typeId.id())
        if (
            PyNode._CLASS_BY_TYPE.get(key) is not cls_obj
            or key not in PyNode._CASTABLE_TYPES
        ):
            return False
        if kind == _NAMED_DG and not mobject.hasFn(OpenMaya.MFn.kDagNode):
            # both names are the fn set name of the same DG node
            name = fn.name()
        elif kind == _NAMED_DAG_PATH and _is_only_path(fn, mobject):
            # the name is the fn set's partial path name of the node's only path
            name = dg_node.name
        else:
            name = _mobject_to_str(mobject)
            if dg_node.name != name:
                return False
        return not (len(name) >= 32 and is_valid_maya_uid(name))
    except Exception:
        return False


def _copy_wrapper(dg_node: Any) -> Any:
    """Returns a new wrapper of `dg_node`'s class and node in the state a fresh cast
    leaves it: its own MObject, MDagPath and fn set, and empty caches. The API 1.0
    validation objects are shared, as the DGNode copy constructor shares them.

    Only for a wrapper `_wrapper_is_canonical` accepted, so the node is live and
    the class keeps a base constructor.
    """
    dg_init, dag_init, geometry_init = _REUSABLE_WRAPPER_PARTS[0]

    cls_obj = type(dg_node)
    inst    = object.__new__(cls_obj)
    d       = inst.__dict__
    if cls_obj.__init__ is dg_init:
        d["_mobject"] = OpenMaya.MObject(dg_node._mobject)
        d["_fn_set"]  = cls_obj.FN_SET(d["_mobject"])
    else:
        d["_mdagpath"] = OpenMaya.MDagPath(dg_node._mdagpath)
        d["_mobject"]  = d["_mdagpath"].node()
        d["_fn_set"]   = cls_obj.FN_SET(d["_mdagpath"])
    # NW6: API 1.0 handle store (a dead helper M8 deletes)
    d["_fn_set1"]    = dg_node._fn_set1
    d["_objhandle1"] = dg_node._objhandle1  # NW6: API 1.0 handle store
    d["_attr_dict"]  = {}
    if cls_obj.__init__ is geometry_init:
        d["_Geometry__local_shape_attr"] = None
        d["_Geometry__world_shape_attr"] = None
    return inst


# the API types `_mobject_to_str` refuses to name
_NOT_NODE_API_TYPES = (
    OpenMaya.MFn.kWorld,
    OpenMaya.MFn.kInvalid,
    OpenMaya.MFn.kUnknown,
)


def _mobject_to_str(mobject: OpenMaya.MObject) -> str:
    """Converts an node MObject to a name string."""
    if mobject.isNull():
        raise ValueError("MObject is null.")
    elif mobject.apiType() in _NOT_NODE_API_TYPES:
        raise ValueError("Invalid MObject API Type: " + mobject.apiTypeStr)
    elif mobject.hasFn(OpenMaya.MFn.kDagNode):
        dag_path = OpenMaya.MDagPath.getAPathTo(mobject)
        return dag_path.partialPathName()
    elif mobject.hasFn(OpenMaya.MFn.kDependencyNode):
        return OpenMaya.MFnDependencyNode(mobject).name()
    else:
        raise ValueError("MObject is not a node.")


def get_custom_type(node_name: str) -> str | None:
    """Returns the value of the custom type attr, if exists."""
    # answer "no custom type" with the API when the name resolves to one node;
    # attributeQuery(exists) also matches aliases, hasAttribute does not
    if isinstance(node_name, str) and _PLAIN_NODE_NAME.match(node_name):
        try:
            sel = OpenMaya.MSelectionList()
            sel.add(node_name)
            if sel.length() == 1:
                fn = OpenMaya.MFnDependencyNode(sel.getDependNode(0))
                if (
                    not fn.hasAttribute(CUSTOM_TYPE_ATTR)
                    and fn.findAlias(CUSTOM_TYPE_ATTR).isNull()
                ):
                    return None
        except (RuntimeError, ValueError, TypeError):
            pass
    if cmds.attributeQuery(CUSTOM_TYPE_ATTR, node=node_name, exists=True):
        return cmds.getAttr(f"{node_name}.{CUSTOM_TYPE_ATTR}")


def set_custom_type(node_name: str, custom_type: str) -> None:
    """Sets the custom type of a given node."""
    attr = f"{node_name}.{CUSTOM_TYPE_ATTR}"
    if cmds.attributeQuery(CUSTOM_TYPE_ATTR, node=node_name, exists=True):
        cmds.deleteAttr(attr)
    cmds.addAttr(node_name, ln=CUSTOM_TYPE_ATTR, dt="string")
    cmds.setAttr(attr, custom_type, type="string")
    cmds.setAttr(attr, lock=True)


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
    OpenMaya.MFn.kNurbsSurfaceData: OpenMaya.MFnNurbsSurfaceData,
    OpenMaya.MFn.kLatticeData: OpenMaya.MFnGeometryData,
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

# Attribute kinds (exact `MObject.apiType()`) whose `cmds.getAttr(..., type=True)`
# string is fixed by the kind itself. A plug of a scalar kind is never a
# compound, and a plug of any listed kind never holds a matrix, so the type
# predicates can answer without querying `data_type`. Generic and typed
# attributes are left out: their runtime type follows their data or connection.
_SCALAR_ATTR_API_TYPES = frozenset(
    {
        OpenMaya.MFn.kNumericAttribute,
        OpenMaya.MFn.kDoubleLinearAttribute,
        OpenMaya.MFn.kFloatLinearAttribute,
        OpenMaya.MFn.kDoubleAngleAttribute,
        OpenMaya.MFn.kFloatAngleAttribute,
        OpenMaya.MFn.kTimeAttribute,
        OpenMaya.MFn.kEnumAttribute,
        OpenMaya.MFn.kMessageAttribute,
    }
)
_NON_MATRIX_ATTR_API_TYPES = _SCALAR_ATTR_API_TYPES | {
    OpenMaya.MFn.kCompoundAttribute,
    OpenMaya.MFn.kAttribute2Double,
    OpenMaya.MFn.kAttribute3Double,
    OpenMaya.MFn.kAttribute4Double,
    OpenMaya.MFn.kAttribute2Float,
    OpenMaya.MFn.kAttribute3Float,
    OpenMaya.MFn.kAttribute2Short,
    OpenMaya.MFn.kAttribute3Short,
    OpenMaya.MFn.kAttribute2Int,
    OpenMaya.MFn.kAttribute3Int,
    OpenMaya.MFn.kLightDataAttribute,
}

# Attribute kinds whose `cmds.getAttr(..., type=True)` string is a property of
# the node class, so `Attribute.data_type` shares it across instances in
# `_STATIC_DATA_TYPE`, keyed by (typeName, typeId, attr long name, isArray).
# Typed attrs are shared only as a top-level array root (`_is_static_typed_root`).
# Dynamic and extension attrs and plugs under an array element are never
# cached; the cache is cleared when a plug-in is loaded or unloaded (it can
# redefine a type).
_STATIC_DATA_API_TYPES = _NON_MATRIX_ATTR_API_TYPES | {
    OpenMaya.MFn.kMatrixAttribute,
    OpenMaya.MFn.kFloatMatrixAttribute,
}
_STATIC_DATA_TYPE = {}
_STATIC_KEY_UNSET = object()  # `Attribute._static_type_key` not yet computed

# (attr, "typed" or "Tdata", node) while `Attribute.data_type` runs the fallback
# hook of attr's node: the answer its getAttr query gave, and the node whose class
# hook that call runs first (`_hook_node`), see `_queried_data_type`
_FALLBACK_QUERY = None

# the rig DSL's `Node` wrapper class, registered by `rig._internal.node` when it
# is defined, since that module imports this one. `Attribute.full_name` reads the
# name of the node an instance of exactly that class wraps, which is what its
# `__getattr__` forwards `name` to; any other owner is named as before
_NODE_WRAPPER_CLASS = None
# that class's own fallback hook, which forwards to the wrapped node's
_NODE_WRAPPER_HOOK  = None


def _fn_set_name(node: Any) -> str:
    """The fn set name of `node`, or of the node a `Node` wrapper wraps: the partial
    path name of a DAG fn set, else the node name."""
    try:
        node = object.__getattribute__(node, "_dg_node")
    except AttributeError:
        pass
    fn = node.__dict__["_fn_set"]
    return fn.partialPathName() if isinstance(fn, OpenMaya.MFnDagNode) else fn.name()


def _unwrapped(node: Any) -> Any:
    """`node`, or the node a `Node` wrapper (or a `Container`) wraps. Reads the
    wrapper's slot directly: its `__getattr__` forwards to the wrapped node."""
    wrapper = _NODE_WRAPPER_CLASS
    if wrapper is not None:
        return node._dg_node if isinstance(node, wrapper) else node
    try:
        return object.__getattribute__(node, "_dg_node")
    except AttributeError:
        return node


# The API 1.0 handle of a node (`_objhandle1`, an `OpenMaya1.MObjectHandle`) is
# the one object that is safe to read once a new scene, a file open or a
# reference unload freed the node: its API 2.0 objects then point at freed
# memory. The cold readers ask these two helpers; the hot paths read the handle
# inline, and every such site (and every `_fn_set1` reader) is marked
# "NW6: API 1.0 handle" and listed in
# `test_node_model.TestApi1HandleReaders`, so that moving the handles off API
# 1.0 (round 5, NW6) has one complete list of sites to edit.


def _handle_valid(d: dict) -> bool:
    """True if the node whose `__dict__` is `d` is valid: alive and not deleted
    to the undo queue. False for a half-built node (no handle yet)."""
    handle = d.get("_objhandle1")
    return handle is not None and handle.isValid()


def _handle_alive(d: dict) -> bool:
    """True if the node whose `__dict__` is `d` is alive: not freed by a new
    scene, a file open or a reference unload (a node deleted to the undo queue
    is alive). False for a half-built node (no handle yet)."""
    handle = d.get("_objhandle1")
    return handle is not None and handle.isAlive()


def _ensure_owner_alive(attr: Any) -> None:
    """Raise ``"... already deleted!"`` if the node that owns `attr` was freed (a
    new scene, a file open, a reference unload): `attr`'s MPlug then points at
    freed memory, and an MPlug call can read another node or crash Maya.

    Only the owner's API 1.0 handle is read, which is safe on a freed node. A node
    deleted to the undo queue is still alive and passes. An attr whose owner is
    not known yet (built from a name or an MPlug, never asked for its node) reads
    the handle of its node it took when it was built instead (`_handle1`, see
    `_node_handle`). The children, elements and parent of an attr share its owner,
    or that handle (see `_inherit_owner`).

    The attribute of a dynamic attr is also freed on its own, once a delete of
    the attr leaves the undo queue (a flush, or ten more commands at mayapy's
    default queue length), while its node lives on. Such an attr reads the API
    1.0 handle of its attribute it took too (`_attr1`, see `_attr_handle`), and
    raises ``"<plug> already deleted!"``, named by its str buffer.
    """
    d    = attr.__dict__
    node = d.get("_node")
    if node is not None:
        # `_unwrapped`, inlined for a `Node` wrapper (the DSL plugs' owner)
        node   = node._dg_node if type(node) is _NODE_WRAPPER_CLASS else _unwrapped(node)
        handle = node.__dict__.get("_objhandle1")  # NW6: API 1.0 handle read (hot)
        if handle is not None and not handle.isAlive():
            node.ensure_valid()
    else:
        handle = d.get("_handle1")
        if handle is not None and not handle.isAlive():
            _raise_deleted(attr, handle)
    handle = d.get("_attr1")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


def _ensure_node_castable(attr: Any) -> None:
    """Raise ``"... already deleted!"`` before `attr`, which has no owner yet, casts
    its node from its MPlug, if the handle of its node it took when it was built is
    no longer valid: the node was freed (the MPlug points at freed memory) or
    deleted to the undo queue (the cast would name it, and reach the new node that
    took its name). An undo brings a deleted node back, and it casts again. The
    MPlug of a freed dynamic attribute cannot be read either, its node included
    (see `_ensure_owner_alive`)."""
    d      = attr.__dict__
    handle = d.get("_handle1")
    if handle is not None and not handle.isValid():
        _raise_deleted(attr, handle)
    handle = d.get("_attr1")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


def _ensure_node_valid(attr: Any) -> None:
    """Raise ``"... already deleted!"`` if the node of `attr` was deleted or freed,
    without naming it: through its owner (`DGNode.ensure_valid`), or, with none,
    the handle of its node it took (see `_ensure_node_castable`). For a caller
    that hands `attr`'s str buffer to cmds, which would name the node that took
    the name (see `rig.bridges.commands`)."""
    node = attr.__dict__.get("_node")
    if node is None:
        _ensure_node_castable(attr)
        return
    node = _unwrapped(node)
    if not _handle_valid(node.__dict__):
        node.ensure_valid()
    handle = attr.__dict__.get("_attr1")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


def _deleted_error(name: str, freed: bool = False) -> RuntimeError:
    """The ``"... already deleted!"`` RuntimeError of the node `name`, deleted to
    the undo queue, or, `freed`, freed by a new scene, a file open or a reference
    unload: a freed node cannot be read, so `name` is then its class name (a node
    object, see `DGNode.ensure_valid`) or the name a plug with no owner was built
    with (see `_raise_deleted`). Every guard raises through it."""
    if freed:
        name = f"{name} node (freed by a new scene, a file open or a reference unload)"
    return RuntimeError(f"{name} already deleted!")


def _raise_deleted(attr: Any, handle: Any) -> None:
    """Raise the ``"... already deleted!"`` of `attr`, a plug with no owner whose
    node (`handle`, its API 1.0 handle) was deleted or freed. A deleted node is
    named as `DGNode.ensure_valid` names it. A freed one cannot be read, and the
    class it would be cast to was never known, so it is named by the node part of
    the name the plug was built with (its str buffer)."""
    if handle.isAlive():
        raise _deleted_error(OpenMaya1.MFnDependencyNode(handle.objectRef()).name())
    raise _deleted_error(str.__str__(attr).split(".", 1)[0], freed=True)


def _node_handle(name: str) -> Any:
    """An API 1.0 MObjectHandle of the node the plug, component or node name `name`
    resolves to, or None if it resolves to none (`_ensure_owner_alive` then does not
    check the plug, as before). The one a plug with no owner keeps (`_handle1`):
    the owner handles are API 1.0 too, the kind that is safe to read once a new
    scene, a file open or a reference unload freed the node. One selection list
    and one MObject are reused (a handle keeps a copy of the MObject)."""
    try:
        _NODE_HANDLE_SEL.clear()
        _NODE_HANDLE_SEL.add(name)
        _NODE_HANDLE_SEL.getDependNode(0, _NODE_HANDLE_OBJ)
        return OpenMaya1.MObjectHandle(_NODE_HANDLE_OBJ)
    except Exception:
        return None


# `_node_handle`'s selection list and MObject, reused by every call
_NODE_HANDLE_SEL = OpenMaya1.MSelectionList()
_NODE_HANDLE_OBJ = OpenMaya1.MObject()


# `_mplug_handle`: the API 1.0 handle it looked up for a node, keyed by the node's
# MObjectHandle hashCode (the same in both APIs), as (the node's API 2.0 MObject,
# that handle) for every node that has the code. An entry is the node's own while
# its handle is alive (a freed node's code and memory go to later nodes, so its
# MObject is then never compared); freed nodes' entries are dropped when their
# code is seen again, and all of them once the table outgrows `_HANDLES_PRUNE_AT`.
_NODE_HANDLES     = {}
_HANDLES_PRUNE_AT = [4096]


def _prune_node_handles() -> None:
    """Drop the handles of freed nodes from `_NODE_HANDLES`."""
    for code in list(_NODE_HANDLES):
        live = [entry for entry in _NODE_HANDLES[code] if entry[1].isAlive()]
        if live:
            _NODE_HANDLES[code] = live
        else:
            del _NODE_HANDLES[code]
    _HANDLES_PRUNE_AT[0] = max(4096, 2 * len(_NODE_HANDLES))


def _mplug_handle(mplug: OpenMaya.MPlug) -> Any:
    """`_node_handle` of the node of `mplug`, a live API 2.0 plug, found by its
    unique name (an MPlug's own name can name another node of the same short
    name), or None. The handle is looked up once per node (a name parse, about
    7 us) and found in `_NODE_HANDLES` after that (about 1 us).

    A node deleted to the undo queue has no name to look it up by, and a plug of
    it can take no handle, so it raises the round-3 ``"<node> already deleted!"``
    here (it read the node that later took the name, or another one once the
    deleted node was freed)."""
    try:
        if mplug.isNull:
            return None
        mobject = mplug.node()
        handle2 = OpenMaya.MObjectHandle(mobject)
        code    = handle2.hashCode()
    except Exception:
        return None
    if not handle2.isValid():
        raise _deleted_error(OpenMaya.MFnDependencyNode(mobject).name())
    entries = _NODE_HANDLES.get(code)
    if entries is not None:
        for known, handle in entries:
            if handle.isAlive() and known == mobject:
                return handle
    try:
        name = OpenMaya.MFnDependencyNode(mobject).uniqueName()
    except Exception:
        return None
    handle = _node_handle(name)
    if handle is None or handle.hashCode() != code:
        return None  # the name resolves to another node: take none
    if entries is None:
        if len(_NODE_HANDLES) >= _HANDLES_PRUNE_AT[0]:
            _prune_node_handles()
        entries = _NODE_HANDLES[code] = []
    else:
        entries[:] = [entry for entry in entries if entry[1].isAlive()]
    entries.append((mobject, handle))
    return handle


# `_attr_handle`: the API 1.0 handle it looked up for a dynamic attribute, keyed
# by the attribute's MObjectHandle hashCode, as `_NODE_HANDLES` keeps the nodes'.
_ATTR_HANDLES   = {}
_ATTR_PRUNE_AT  = [4096]


def _prune_attr_handles() -> None:
    """Drop the handles of freed attributes from `_ATTR_HANDLES`."""
    for code in list(_ATTR_HANDLES):
        live = [entry for entry in _ATTR_HANDLES[code] if entry[1].isAlive()]
        if live:
            _ATTR_HANDLES[code] = live
        else:
            del _ATTR_HANDLES[code]
    _ATTR_PRUNE_AT[0] = max(4096, 2 * len(_ATTR_HANDLES))


def _attr_handle(
    mplug:       OpenMaya.MPlug,
    node_handle: Any  = None,
    fn1:         Any  = None,
    name:        Any  = None,
    extension:   bool = False,
) -> Any:
    """The API 1.0 MObjectHandle of the attribute of `mplug`, a live plug, if it is
    a dynamic attribute, or an `extension` one (the caller knows its class:
    `MPlug.isDynamic` is False for it), else None: the one a plug keeps as
    `_attr1` (see `_ensure_owner_alive`). A dynamic attribute is freed once a
    delete of it leaves the undo queue, while its node lives on, and an extension
    one by `deleteExtension`; a static one lives as long as its node type. The attribute is found by name on its node, through `fn1`, the
    node's API 1.0 fn set, or `node_handle`, its API 1.0 handle, once per
    attribute (about 7 us; `name`, a name of it the caller has, saves reading
    it), and in `_ATTR_HANDLES` after that (about 1.5 us). None if the name
    finds another attribute (the plug's was deleted)."""
    if not (extension or mplug.isDynamic):
        return None
    try:
        mobject = mplug.attribute()
        code    = OpenMaya.MObjectHandle(mobject).hashCode()
    except Exception:
        return None
    entries = _ATTR_HANDLES.get(code)
    if entries is not None:
        for known, handle in entries:
            if handle.isAlive() and known == mobject:
                return handle
    try:
        if fn1 is None:
            if node_handle is None or not node_handle.isAlive():
                return None
            fn1 = OpenMaya1.MFnDependencyNode(node_handle.objectRef())
        handle = None
        if name is not None:
            try:
                handle = OpenMaya1.MObjectHandle(fn1.attribute(name))
            except Exception:
                handle = None
            # an alias, or a name another attribute has: read the plug's own name
            if handle is not None and handle.hashCode() != code:
                handle = None
        if handle is None:
            handle = OpenMaya1.MObjectHandle(
                fn1.attribute(OpenMaya.MFnAttribute(mobject).name)
            )
    except Exception:
        return None
    if handle.hashCode() != code:
        return None
    if entries is None:
        if len(_ATTR_HANDLES) >= _ATTR_PRUNE_AT[0]:
            _prune_attr_handles()
        entries = _ATTR_HANDLES[code] = []
    else:
        entries[:] = [entry for entry in entries if entry[1].isAlive()]
    entries.append((mobject, handle))
    return handle


def _attr_state(mplug: OpenMaya.MPlug, handle: Any, attr_handle: Any = None) -> dict:
    """The `__dict__` of a fresh Attribute of `mplug` with no owner, checked
    through `handle` (the API 1.0 handle of its node, or None) and `attr_handle`
    (the API 1.0 handle of its attribute, for a dynamic attr, see `_attr_handle`).
    One `__dict__` store instead of one `Plug.__setattr__` call per field; a
    subclass property named like one of these keys would be bypassed."""
    return {
        "_mplug":                      mplug,
        "_mobject":                    None,
        "_fn_set":                     None,
        "_node":                       None,
        # caches of queried child attributes
        "_Attribute__child_name_dict": {},
        "_Attribute__child_id_dict":   {},
        # cache componet type str TODO: make a proper Component class
        "_Attribute__component_type":  None,
        # cache for `_is_geometry_typed_attr`; None = not yet computed
        "_geometry_attr_cache":        None,
        # cache for `_owner_is_polymorphic`; None = not yet computed
        "_polymorphic_owner_cache":    None,
        # cache for `_static_type_key`; _STATIC_KEY_UNSET = not yet computed
        "_static_key_cache":           _STATIC_KEY_UNSET,
        # the API 1.0 handle of the node, for a plug with no owner
        "_handle1":                    handle,
        # the API 1.0 handle of the attribute, for a dynamic attr
        "_attr1":                      attr_handle,
    }


def _new_attr(
    cls: type, mplug: OpenMaya.MPlug, handle: Any = None, attr_handle: Any = None
) -> Any:
    """`cls(mplug)` without the handle of its node `Attribute.__init__` takes of an
    MPlug (a name lookup): for a plug of a node the caller holds, whose owner, or
    the handle `handle` of that node, the caller hands it (see `_inherit_owner`),
    with the handle of its attribute (`attr_handle`, see `_attr_handle`).
    `cls` is `Attribute` or `Plug`, whose `__init__` does nothing else for an
    MPlug. It builds `_attr_state`'s state inline: it is the owned plugs' hot
    path (about 26k per rail_spine build)."""
    attr = str.__new__(cls, mplug)
    attr.__dict__.update(
        {
            "_mplug":                      mplug,
            "_mobject":                    None,
            "_fn_set":                     None,
            "_node":                       None,
            "_Attribute__child_name_dict": {},
            "_Attribute__child_id_dict":   {},
            "_Attribute__component_type":  None,
            "_geometry_attr_cache":        None,
            "_polymorphic_owner_cache":    None,
            "_static_key_cache":           _STATIC_KEY_UNSET,
            "_handle1":                    handle,
            "_attr1":                      attr_handle,
        }
    )
    return attr


def _attr_mobject(attr: Any) -> OpenMaya.MObject:
    """`attr.mobject` (cached on `attr`) without its owner check, for a caller that
    made one: the attribute of a dynamic attr is freed with its node."""
    d       = attr.__dict__
    mobject = d["_mobject"]
    if not mobject:
        mobject = d["_mobject"] = d["_mplug"].attribute()
    return mobject


def _connected_attrs(
    attr: Any, src: bool = True, dst: bool = True, first_only: bool = False
) -> Any:
    """`attr.get_connected_attrs(src, dst, first_only)` for a caller that reads
    the result at once and keeps none of it (a type query, a disconnect): the
    attrs take no handle of their nodes, a lookup each (see `_mplug_handle`)
    that only a held plug needs (see `_ensure_owner_alive`)."""
    _ensure_owner_alive(attr)
    found = []
    for each in attr.plug.connectedTo(src, dst):
        found.append(_new_attr(Attribute, each))
        if first_only:
            break
    if first_only:
        return found[0] if found else None
    return found


def _inherit_owner(parent: Any, attr: Any) -> Any:
    """`attr`, a fresh child, element or parent plug of `parent` (on its node),
    owned by the node object `parent` holds, if any, and named through that
    owner's path (see `_named_through_owner`, which may hand back a copy). With
    no owner, it takes the handle of that node `parent` took (see
    `_ensure_owner_alive`)."""
    d     = attr.__dict__
    owner = parent.__dict__["_node"]
    # a child, an element or the parent of a dynamic attr is freed with it
    d["_attr1"] = parent.__dict__.get("_attr1")
    if owner is None:
        d["_handle1"] = parent.__dict__["_handle1"]
        return attr
    d["_node"] = owner
    if str.__contains__(parent, "|"):  # `_named_through_a_path`, inlined
        return _named_through_owner(attr)
    return attr


def _named_through_a_path(attr: Any) -> bool:
    """False if the str buffer of `attr` names no DAG path (``S.visibility``), so
    its owner had one path when it was built (see `_named_through_owner`): the
    plugs read from it need no check. A plug named through an instanced node's
    path (``T2|S.visibility``) has a ``|``, as have other paths, which are checked."""
    return str.__contains__(attr, "|")


def _named_through_owner(attr: Any) -> Any:
    """`attr` (its owner bound), or a copy of it whose str buffer is its full name
    when its owner is a DAG node with more than one path (instanced, or under an
    instanced parent). maya.cmds reads that buffer, not ``str()``: one made from
    the MPlug names the node's first path, so ``cmds.getAttr(plug)`` would read
    another instance than ``plug.get()`` (``T2|S.worldMatrix`` is element 1)."""
    owner = attr.__dict__["_node"]
    if owner is None or not _owner_is_instanced(owner):
        return attr
    return _full_name_buffer(attr)


def _owner_is_instanced(owner: Any) -> bool:
    """True if `owner`, a plug's node object, is a live DAG node with more than
    one path. A deleted or freed node's path is not read (naming it raises)."""
    d  = _unwrapped(owner).__dict__
    fn = d.get("_fn_set")
    if not isinstance(fn, OpenMaya.MFnDagNode):
        return False
    return _handle_valid(d) and fn.isInstanced(True)


def _full_name_buffer(attr: Any) -> Any:
    """`attr`, or a copy of it (sharing its state) whose str buffer is its full
    name, if the two differ. For a fresh attr only: the copy stands in for it."""
    name = attr.full_name
    if name == str.__str__(attr):
        return attr
    copy = str.__new__(type(attr), name)
    copy.__dict__.update(attr.__dict__)
    return copy


def _point_count(attr: Any) -> int:
    """The point count of the geometry node of `attr`, a component plug (it bounds
    a component slice). Read through the node's own class: the owner can be a
    class the user chose (``Node(DAGNode(mesh)).vtx[0:2]``), which has none."""
    node = _unwrapped(attr.node)
    if not hasattr(type(node), "num_weight_points"):
        node = PyNode(attr.plug.node())
    return node.num_weight_points


def _node_name(node: Any) -> str:
    """The name `Attribute.full_name` gives `node`, a plug's owner (a `Node` wrapper
    is named by the node it wraps). Anything but a str from the `name` property is
    the node's Maya attr of that name (a class whose `name` property raises), so the
    node is named by its fn set instead. Raises if the node is deleted."""
    if type(node) is _NODE_WRAPPER_CLASS:
        node = node._dg_node
    name = node.name
    if type(name) is not str:
        name = _fn_set_name(node)
    return name


def _instanced_element_alias(mplug: OpenMaya.MPlug, node: Any, alias: str) -> str:
    """The attr part of `Attribute.full_name` for `mplug`, an element whose alias
    (`alias`) has no index: an element of an instanced (world space) attr, or an
    aliased element. cmds resolves an instanced element named without its index
    to the element of the instance the node name's path runs through, so another
    instance's element is named with its index (`MPlug.partialName` with the
    instanced indices), which cmds honours. `node` is the owner, already named."""
    fn = _unwrapped(node).__dict__.get("_fn_set")
    if not isinstance(fn, OpenMaya.MFnDagNode) or not fn.isInstanced(True):
        return alias
    path = fn.getPath()
    if path.isValid() and mplug.logicalIndex() == path.instanceNumber():
        return alias
    return mplug.partialName(False, False, True, True, False, True)


def _is_instanced_array(mplug: OpenMaya.MPlug) -> bool:
    """True if `mplug` is an array whose elements are per instance (worldMatrix,
    instObjGroups...): Maya names such an element without its index."""
    try:
        element = mplug.elementByLogicalIndex(0)
        return not element.partialName(False, False, False, False, False, False).endswith("]")
    except RuntimeError:
        return False


def _path_instance_number(attr: Any, node: Any = None) -> int | None:
    """The instance number of the DAG path `attr`'s node is named through, if
    `attr` is an array of per-instance elements read without an index
    (``worldMatrix``, ``instObjGroups``): cmds resolves such a name to the element
    of that path's instance (``T2|S.worldMatrix`` is ``worldMatrix[1]``), so that
    element is the plug it stands for. None for any other attr. `node` is
    `attr`'s owner, unwrapped, when the caller has it. Reads the owner's name
    first, so a stale path is re-resolved and a deleted node raises."""
    mplug = attr.__dict__["_mplug"]
    if not mplug.isArray:
        return None
    if node is None:
        node = _unwrapped(attr.node)
    fn = node.__dict__.get("_fn_set")
    if not isinstance(fn, OpenMaya.MFnDagNode) or not _is_instanced_array(mplug):
        return None
    _node_name(node)
    return fn.getPath().instanceNumber()


def _deleted_instance_index(mplug: OpenMaya.MPlug) -> str:
    """The index `_plug_hash` gives `mplug`, a plug of a node deleted to the undo
    queue, whose path cannot be read: ``"[0]"`` for an array of per-instance
    elements read without an index (``worldMatrix``), the element of the first
    instance, as `_path_instance_number` reads it once an undo brings back a node
    of one path, so a dict or set key made while the node is deleted is found
    after the undo; ``""`` for any other plug. (An instanced node's plug read
    through another instance hashes apart while its node is deleted.)"""
    if mplug.isArray and _is_instanced_array(mplug):
        return "[0]"
    return ""


def _plug_identity_name(attr: Any) -> str:
    """The attr part of the identity of `attr`'s Maya plug: its alias with every
    index, the instanced ones too (``worldMatrix[1]``), whatever the path the
    node is named through. An array of per-instance elements read without an
    index is the element of its path's instance (see `_path_instance_number`).
    Reads the owner's name (which raises if the node is deleted)."""
    name  = attr.__dict__["_mplug"].partialName(False, False, True, True, False, True)
    index = _path_instance_number(attr)
    return name if index is None else f"{name}[{index}]"


def _same_plug(attr: Any, other: Any) -> bool:
    """True if Attributes `attr` and `other` are the same Maya plug: the same node,
    attribute and logical indices, whatever the instance paths they are named
    through (``Node("|T1|S").v`` and ``Node("|T2|S").v``). World space elements of
    different instances are different plugs. Both are named first, so a deleted
    or freed node raises as the names did."""
    attr.full_name
    other.full_name
    if attr is other:
        return True
    return attr.__dict__["_mplug"].node() == other.__dict__["_mplug"].node() and (
        _plug_identity_name(attr) == _plug_identity_name(other)
    )


# `_node_serial`: a number per Maya node for as long as the node lives, keyed by
# its API 1.0 MObjectHandle hashCode: the (handle, serial) of every live node
# that has the code. Freed nodes' entries are dropped when their code is seen
# again, and all of them once the table outgrows `_SERIALS_PRUNE_AT`.
_NODE_SERIALS     = {}
_NEXT_SERIAL      = itertools.count(1).__next__
_SERIALS_PRUNE_AT = [4096]


def _prune_node_serials() -> None:
    """Drop the serials of freed nodes (no node can take them again)."""
    for code in list(_NODE_SERIALS):
        live = [entry for entry in _NODE_SERIALS[code] if entry[0].isAlive()]
        if live:
            _NODE_SERIALS[code] = live
        else:
            del _NODE_SERIALS[code]
    _SERIALS_PRUNE_AT[0] = max(4096, 2 * len(_NODE_SERIALS))


def _node_serial(node: Any) -> int:
    """A number that stands for the Maya node of `node` (a live DGNode) while that
    node lives, and never for another node: the node part of `_plug_hash`.

    Maya hands a freed node's `MObjectHandle.hashCode()` to a node made later,
    and a file opened again or a reference reloaded brings its nodes back with
    the UUIDs they had, so neither tells a freed node from the next one; a plug
    of a freed node kept as a dict key would then share a hash with a live plug,
    and the lookup would compare them. A serial is never reused. A node deleted
    to the undo queue lives on, and keeps its serial when the delete is undone.
    Two wrappers of one node share it; it is cached on the wrapper.
    """
    d      = node.__dict__
    serial = d.get("_node_serial")
    if serial is None:
        # NW6: API 1.0 handle read (hot)
        serial = d["_node_serial"] = _handle_serial(d["_objhandle1"])
    return serial


def _handle_serial(handle: Any) -> int:
    """`_node_serial` of the live node of `handle`, its API 1.0 MObjectHandle (a
    plug with no owner hashes through the handle of its node it took)."""
    serial  = None
    code    = handle.hashCode()
    entries = _NODE_SERIALS.get(code)
    if entries is None:
        if len(_NODE_SERIALS) >= _SERIALS_PRUNE_AT[0]:
            _prune_node_serials()
        entries = _NODE_SERIALS[code] = []
    else:
        mobject = handle.objectRef()
        live    = []
        for other, number in entries:
            # a freed node's handle is not alive: it is dropped, never compared
            if other.isAlive():
                live.append((other, number))
                if serial is None and other.objectRef() == mobject:
                    serial = number
        entries[:] = live
    if serial is None:
        serial = _NEXT_SERIAL()
        entries.append((handle, serial))
    return serial


def _plug_hash(attr: Any) -> int:
    """The hash of the Maya plug `attr` is (see `_same_plug`), for `Plug.__hash__`
    (a typed `Attribute` salts it, see `Attribute.__hash__`).

    It is the node's serial (see `_node_serial`) and the attribute's long name
    with every logical index, the instanced ones too (``worldMatrix[1]``), so it
    is the same through every instance path of the node and it does not change
    when the node is renamed, the attribute aliased, or the node deleted to the
    undo queue: a plug stays findable in a dict or set across all of them. A node
    made later, even one that takes a freed node's hashCode or UUID, hashes
    apart. It is cached on the attr, so it survives a new scene freeing the node
    too. An array of per-instance elements read without an index
    (``T2|S.worldMatrix``) is the element of its path's instance, which a
    removed instance can change, so that hash follows the path and is not
    cached. The plug of a freed node that was never hashed hashes by its str
    buffer (its MPlug points at freed memory). A plug with no owner hashes so
    through the handle of its node it took (see `_ensure_owner_alive`) while
    that node is deleted or freed, as it cannot cast the node.
    """
    d      = attr.__dict__
    cached = d.get("_plug_hash")
    if cached is not None:
        return cached
    handle = d.get("_attr1")
    if handle is not None and not handle.isAlive():
        # its attribute was freed (see `_ensure_owner_alive`): not read
        return hash((handle.hashCode(), str.__str__(attr)))
    owner = d["_node"]
    if owner is not None:
        node = _unwrapped(owner)
    else:
        handle = d["_handle1"]
        if handle is not None and not handle.isValid():
            if not handle.isAlive():
                return hash((handle.hashCode(), str.__str__(attr)))
            mplug = d["_mplug"]
            name  = mplug.partialName(False, False, True, False, False, True)
            return hash((_handle_serial(handle), name + _deleted_instance_index(mplug)))
        node = _unwrapped(attr.node)
    handle = node.__dict__.get("_objhandle1")  # NW6: API 1.0 handle read (hot)
    if handle is None:
        # not a DGNode: named as the node's name property names it
        return hash((hash(str(node)), _plug_identity_name(attr)))
    if not handle.isAlive():
        return hash((handle.hashCode(), str.__str__(attr)))
    serial = _node_serial(node)
    name   = d["_mplug"].partialName(False, False, True, False, False, True)
    # a deleted node's path is not read, and its hash is not kept
    cache = handle.isValid()
    if cache:
        index = _path_instance_number(attr, node)
        if index is not None:
            name  = f"{name}[{index}]"
            cache = False
    else:
        name += _deleted_instance_index(d["_mplug"])
    value = hash((serial, name))
    if cache:
        d["_plug_hash"] = value
    return value


def _clear_static_data_type(*args) -> None:
    """Drops every cached static data type (MSceneMessage callback)."""
    _STATIC_DATA_TYPE.clear()


def _hook_node(node: Any) -> Any:
    """The node whose class's fallback hook `node._attr_data_type_fallback` runs
    first, or None if another hook can run first: a patch of the `Node` wrapper's
    forwarding hook, a `Node` subclass, or a hook set on the node instance."""
    try:
        if type(node) is _NODE_WRAPPER_CLASS:
            if _NODE_WRAPPER_CLASS._attr_data_type_fallback is not _NODE_WRAPPER_HOOK:
                return None
            node = node._dg_node
        elif _NODE_WRAPPER_CLASS is None or isinstance(node, _NODE_WRAPPER_CLASS):
            return None
        if "_attr_data_type_fallback" in vars(node):
            return None
        return node
    except Exception:
        return None


def _queried_data_type(attr: Any, node: Any) -> str | None:
    """The type `Attribute.data_type` just queried for `attr` before calling the
    fallback hook that is running for it, if that call runs the class hook of
    `node` first, or None outside such a call."""
    query = _FALLBACK_QUERY
    if query is not None and query[0] is attr and query[2] is node:
        return query[1]
    return None


def _plugin_callback_specs() -> list:
    msg = OpenMaya.MSceneMessage
    return [
        (msg.addStringArrayCallback, msg.kAfterPluginLoad,   _clear_static_data_type),
        (msg.addStringArrayCallback, msg.kAfterPluginUnload, _clear_static_data_type),
    ]


def _register_plugin_callbacks() -> bool:
    """Register the plug-in callbacks in place of an earlier import's, once per
    Maya session (see `rig._internal.callbacks`). False, with nothing registered,
    while Maya is not initialised: `_STATIC_DATA_TYPE` is then filled only once
    `_ensure_plugin_callbacks` has registered them."""
    global _PLUGIN_CALLBACKS_READY
    _PLUGIN_CALLBACKS_READY = _callbacks.register(
        __name__, _THIS_MODULE, _plugin_callback_specs()
    )
    return _PLUGIN_CALLBACKS_READY


def _ensure_plugin_callbacks() -> bool:
    """Make the registration an import before Maya was initialised left pending."""
    global _PLUGIN_CALLBACKS_READY
    _PLUGIN_CALLBACKS_READY = _callbacks.ensure(
        __name__, _THIS_MODULE, _plugin_callback_specs()
    )
    return _PLUGIN_CALLBACKS_READY


_THIS_MODULE            = sys.modules.get(__name__)
_PLUGIN_CALLBACKS_READY = False
_register_plugin_callbacks()

# typed array attrs cmds.setAttr() sets as (count, *items) instead of a list
COUNTED_ARRAY_TYPES = ("stringArray", "vectorArray", "pointArray")

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


def _plug_in_array(plug: OpenMaya.MPlug) -> bool:
    """True if `plug` is an array or an array element, or a child at any depth
    of one."""
    while True:
        if plug.isArray or plug.isElement:
            return True
        if not plug.isChild:
            return False
        plug = plug.parent()


def _plug_under_element(plug: OpenMaya.MPlug) -> bool:
    """True if `plug` is an array element, or a child at any depth of one."""
    while True:
        if plug.isElement:
            return True
        if not plug.isChild:
            return False
        plug = plug.parent()


def _is_static_typed_root(mplug: OpenMaya.MPlug, mobject: OpenMaya.MObject) -> bool:
    """True if `mplug` is the top-level array root of a typed attr whose
    `cmds.getAttr(..., type=True)` string is the same on every node of its type.

    A typed attr reports the data it holds, so an attr that is not an array
    root is never shared: a generic source (a choice output, a deformer's
    geometry) can hand it another type. A root reports 'TdataCompound', except
    a world space root, which reports its first element; only a matrix one that
    nothing can connect into (worldMatrix, parentInverseMatrix...) always holds
    its declared type.
    """
    if (
        mobject.apiType() != OpenMaya.MFn.kTypedAttribute
        or not mplug.isArray
        or mplug.isChild
    ):
        return False
    fn = OpenMaya.MFnTypedAttribute(mobject)
    if fn.internal:
        return False
    if fn.worldSpace:
        return fn.attrType() == OpenMaya.MFnData.kMatrix and not fn.writable
    return fn.attrType() != OpenMaya.MFnData.kInvalid


def _plug_node_fn_set(attr: Any, mplug: OpenMaya.MPlug) -> OpenMaya.MFnDependencyNode | None:
    """A fn set of the node of `mplug`, `attr`'s MPlug: the one its owner holds
    (the owner rule makes the owner the plug's node) while that node is valid,
    else a new one, as `_fixed_attr_kind` built for every call (1.5 us). None
    once that node was freed (see `_ensure_owner_alive`): `mplug` then points at
    freed memory, which the new fn set would read."""
    owner = attr.__dict__.get("_node")
    if owner is not None:
        d      = _unwrapped(owner).__dict__
        handle = d.get("_objhandle1")  # NW6: API 1.0 handle read (hot)
        fn     = d.get("_fn_set")
        if (
            fn is not None
            and handle is not None
            and handle.isValid()
            and d.get("_mobject") == mplug.node()
        ):
            return fn
    else:
        handle = attr.__dict__.get("_handle1")
    if handle is not None and not handle.isAlive():
        return None
    return OpenMaya.MFnDependencyNode(mplug.node())


def _fixed_attr_kind(attr: Attribute) -> int | None:
    """Returns the kind (exact `MObject.apiType()`) of `attr`'s attribute if a type
    predicate may answer from it instead of the by-name `data_type` query, else None.

    The by-name query follows a dynamic or extension attr that was deleted and
    re-added under the same name, which a held plug does not, and it creates an
    array element that does not exist yet, so neither case uses the kind. Nor
    does a plug whose node was freed: its MPlug points at freed memory, and the
    query raises the node's ``"already deleted!"``.
    """
    try:
        mplug = attr._mplug
        fn    = _plug_node_fn_set(attr, mplug)
        if fn is None:
            return None
        mobject = _attr_mobject(attr)
        # a deleted attr is no longer on the node, even once its name is reused
        if fn.attributeClass(mobject) == OpenMaya.MFnDependencyNode.kInvalidAttr:
            return None
        elements = []
        plug     = mplug
        while True:
            if plug.isElement:
                if plug.logicalIndex() < 0:
                    return None
                elements.append(plug)
                plug = plug.array()
            elif plug.isChild:
                plug = plug.parent()
            else:
                break
        # outermost first, so no array under a missing element is listed
        for element in reversed(elements):
            indices = element.array().getExistingArrayAttributeIndices()
            if element.logicalIndex() not in indices:
                return None
        return mobject.apiType()
    except Exception:
        return None


def _naming_dag_path(attr: Any) -> OpenMaya.MDagPath | None:
    """The DAG path whose name `Attribute.full_name` gives `attr`'s node, or None
    for an owner that is not named by its fn set's path (a subclass of the `Node`
    wrapper is named through its own `name`)."""
    node = attr.__dict__["_node"]
    if type(node) is _NODE_WRAPPER_CLASS:
        node = node._dg_node
    elif _NODE_WRAPPER_CLASS is None or isinstance(node, _NODE_WRAPPER_CLASS):
        return None
    return node._fn_set.getPath()


def _names_own_plug(attr: Any) -> bool:
    """True if the name `attr` gives itself resolves in cmds to its own MPlug.

    That needs the base naming (a component plug is named after its component),
    the attribute still on its node (a deleted dynamic or extension attr's name
    resolves to nothing, or to a same-named new attr), and an index on every
    array along the path. Call it only once `attr` is named, so its node is valid.

    A DAG node is named by its wrapper's DAG path, which must still be valid (the
    instance it runs through can be deleted while the node lives on), and an
    instanced element is named without its index, after that path's instance.
    """
    cls = type(attr)
    if (
        not isinstance(attr, Attribute)
        or cls.__str__ is not Attribute.__str__
        or cls.full_name is not Attribute.full_name
        or cls.alias is not Attribute.alias
    ):
        return False
    # a freed attribute (see `_ensure_owner_alive`) is not read
    handle = attr.__dict__.get("_attr1")
    if handle is not None and not handle.isAlive():
        return False
    try:
        mplug   = attr._mplug
        mobject = mplug.node()
        fn      = OpenMaya.MFnDependencyNode(mobject)
        invalid = OpenMaya.MFnDependencyNode.kInvalidAttr
        if fn.attributeClass(mplug.attribute()) == invalid:
            return False
        # the owner's path, valid once the attr was named: a stale path (its
        # instance removed while the node lives on in another) is re-resolved by
        # DAGNode.name; an invalid one would name the node "". Kept as a guard.
        path = None
        if mobject.hasFn(OpenMaya.MFn.kDagNode):
            path = _naming_dag_path(attr)
            if path is None or not path.isValid():
                return False
        plug = mplug
        while True:
            if plug.isElement:
                index = plug.logicalIndex()
                if index < 0:
                    return False
                # an instanced element (worldMatrix, instObjGroups...) is named
                # without its index, which resolves to its path's instance
                name = plug.partialName(False, False, False, False, False, False)
                if not name.endswith("]") and (
                    path is None or index != path.instanceNumber()
                ):
                    return False
                plug = plug.array()
            elif plug.isChild:
                plug = plug.parent()
                # a child of an array root, not of one of its elements
                if plug.isArray:
                    return False
            else:
                return True
    except Exception:
        return False


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
        """Initialize an instance from an attribute full name or a MPlug.

        It has no owner yet, so it takes an API 1.0 handle of its node, which
        tells it once that node is deleted or freed (see `_ensure_owner_alive`).
        An MPlug of a node already deleted to the undo queue raises its
        ``"... already deleted!"`` (see `_mplug_handle`).
        """
        if isinstance(name_or_mplug, str):
            sel = OpenMaya.MSelectionList()
            sel.add(name_or_mplug)
            mplug = sel.getPlug(0)
        elif isinstance(name_or_mplug, OpenMaya.MPlug):
            mplug = name_or_mplug
        else:
            raise ValueError(f"{name_or_mplug} is not a string or MPlug.")
        handle = _mplug_handle(mplug)
        self.__dict__.update(_attr_state(mplug, handle, _attr_handle(mplug, handle)))

    # --- dunders

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.full_name}")'

    def __str__(self) -> str:
        return self.full_name

    def __hash__(self) -> int:
        # the Maya plug's identity, not the name: one key through every instance
        # path, kept across a rename (see `_plug_hash`). Salted, so a typed attr
        # and a DSL Plug of one plug are two keys: a dict or set lookup that
        # found them alike would compare them with `Plug.__eq__`, which folds
        # one plug to True but, under `force_nodes()` / `constant_folding=False`,
        # builds an equal node (and raises for matrices)
        return hash(("Attribute", _plug_hash(self)))

    def __eq__(self, other: Any) -> bool:
        """True if `other` is an Attribute of the same Maya plug (node, attribute
        and logical indices), whatever the instance path each is named through.
        A str is never equal: compare `str(attr)` for names."""
        return isinstance(other, Attribute) and _same_plug(self, other)

    def __ne__(self, other: Any) -> bool:
        # not `str.__ne__`, which compares the names the two were built with
        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        return self.full_name > str(other)

    def __getitem__(
        self, key: int | slice | list | tuple | np.ndarray
    ) -> Attribute | list:
        """Return attribute at a given logical index, if this is a multi-attr.
           If given a slice, return a list of attributes.
           If given a sequence of integers (a list, tuple or ndarray of ids),
           return a list of attributes at those logical indices.
        Example:
        ```
        attr = mesh.componentTags[1]
        attrs = blendShape.weight[1:3]
        attrs = mesh.vtx[[0, 4, 7]]
        ```
        """
        _ensure_owner_alive(self)

        if isinstance(key, (int, np.integer)):
            return self.element_by_logical_index(int(key))

        elif isinstance(key, (list, tuple, np.ndarray)):
            # Fancy indexing with the ids a membership query returns
            # (``mesh.vtx[ids]``). Geometry components are range-checked
            # against the point count like a slice is bounded by it; a plain
            # multi keeps Maya's sparse semantics (missing elements are
            # created on access, same as ``plug[5]``).
            ids      = np.asarray(key)
            integral = np.issubdtype(ids.dtype, np.integer)
            if ids.dtype == bool or (ids.size and not integral):
                raise TypeError(f"Indices must be integers, not {ids.dtype}")
            if ids.ndim != 1:
                raise TypeError(
                    f"Indices must be a flat sequence, got shape {ids.shape}"
                )
            if self._component_type in (
                "kMeshVertComponent",
                "kCurveCVComponent",
                "kSurfaceCVComponent",
            ):
                count = _point_count(self)
                ids   = np.where(ids < 0, ids + count, ids)
                if ids.size and (ids.min() < 0 or ids.max() >= count):
                    raise IndexError(f"{self} index out of range for {count} points")
            return [self.element_by_logical_index(int(index)) for index in ids]

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
                start, stop, step = key.indices(_point_count(self))

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
            _ensure_owner_alive(self)
            for i in range(self.num_children):
                child_plug = self.plug.child(i)
                name = child_plug.partialName(False, False, False, False, False, True)
                name = name.rsplit(".", 1)[-1]
                attr = _inherit_owner(self, _new_attr(Attribute, child_plug))
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
        """Returns the attribute function set. Raises ``"... already deleted!"``
        once the node was freed: the attribute of a dynamic attr is freed with it
        (see `_ensure_owner_alive`)."""
        _ensure_owner_alive(self)
        # the lazy caches write `__dict__`, which skips a subclass's `__setattr__`
        if not self._fn_set:
            mobject  = _attr_mobject(self)
            api_type = mobject.apiType()
            data_fn  = ATTR_TYPE_TO_FN.get(api_type, OpenMaya.MFnAttribute)
            self.__dict__["_fn_set"] = data_fn(mobject)
        return self._fn_set

    @property
    def mobject(self) -> OpenMaya.MObject:
        """Returns the mobject. Raises ``"... already deleted!"`` once the node
        was freed: the attribute of a dynamic attr is freed with it, a cached
        one too (see `_ensure_owner_alive`)."""
        _ensure_owner_alive(self)
        return _attr_mobject(self)

    @property
    def node(self) -> Any:
        """Returns the node object of this attr."""
        d    = self.__dict__
        node = d["_node"]
        if node is None:
            _ensure_node_castable(self)
            node = d["_node"] = PyNode(d["_mplug"].node())
        return node

    @property
    def name(self) -> str:
        """Returns the attribute name (without node name). Raises ``"... already
        deleted!"`` once the node was freed (see `_ensure_owner_alive`): the name
        of a dynamic attr is read from its attribute, which is freed with it."""
        _ensure_owner_alive(self)
        return self._mplug.partialName(False, False, False, False, False, True)

    @property
    def full_name(self) -> str:
        """Returns the full attribute name (with node name), use alias if exists.

        Note. Can't use MPlug.name() directly because it doesn't use the partial node name.
        i.e. it will error if duplicated node names exist.

        An element of an instanced (world space) attr, such as ``worldMatrix[1]``,
        is named without its index when it is the element of the instance the
        node's path runs through (``T2|S.worldMatrix``), and with it otherwise
        (``T2|S.worldMatrix[0]``), so cmds resolves the name to this element.
        """
        node = self.node
        # a rig Node only forwards `name` to the node it wraps, so read it there
        if type(node) is _NODE_WRAPPER_CLASS:
            node = node._dg_node
        name = node.name
        # anything but a str is the node's Maya attr of that name (a class whose
        # `name` property raises), so the node is named by its fn set instead
        if type(name) is not str:
            name = _fn_set_name(node)
        # `alias`, read once the name proved the node is alive, and the attribute
        # of a dynamic attr (see `_ensure_owner_alive`)
        d      = self.__dict__
        mplug  = d["_mplug"]
        handle = d.get("_attr1")
        if handle is not None and not handle.isAlive():
            raise _deleted_error(str.__str__(self))
        alias = mplug.partialName(False, False, False, True, False, True)
        if mplug.isElement and alias[-1:] != "]":
            alias = _instanced_element_alias(mplug, node, alias)
        return f"{name}.{alias}"

    @property
    def alias(self) -> str:
        """Returns the attribute alias, or the attribute name if no alias applied."""
        _ensure_owner_alive(self)
        return self._mplug.partialName(False, False, False, True, False, True)

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
        global _FALLBACK_QUERY
        # the name comes first so a deleted node's plug raises as before
        full_name = self.full_name
        key       = self._static_type_key()
        if key is not None:
            typ = _STATIC_DATA_TYPE.get(key)
            if typ is not None:
                return typ

        typ = cmds.getAttr(full_name, type=True)

        # maintain consistent type string with cmds.addAttr()
        if typ == "TdataCompound":
            typ = "compound"

        # if cmds.getAttr fails to resolve, call the node fallback hook
        # e.g. if a choice node's inputs are message attrs, its output will be resolved
        # to "typed" rather than "message"
        elif typ in ("typed", "Tdata"):
            # the hook can reuse this answer instead of querying it again, if no
            # wrapper or instance hook can run before the node's class hook
            outer = _FALLBACK_QUERY
            try:
                node            = self.node
                _FALLBACK_QUERY = (self, typ, _hook_node(node))
                return node._attr_data_type_fallback(self)
            finally:
                _FALLBACK_QUERY = outer

        if (
            key is not None
            and isinstance(typ, str)
            and (_PLUGIN_CALLBACKS_READY or _ensure_plugin_callbacks())
        ):
            _STATIC_DATA_TYPE[key] = typ
        return typ

    def _static_type_key(self) -> tuple | None:
        """The `_STATIC_DATA_TYPE` key of this attr, or None if its data type is
        not a property of the node class. Cached per Attribute instance."""
        key = self._static_key_cache
        if key is _STATIC_KEY_UNSET:
            key = None
            try:
                mplug   = self._mplug
                fn      = OpenMaya.MFnDependencyNode(mplug.node())
                normal  = OpenMaya.MFnDependencyNode.kNormalAttr
                # `data_type`, the caller, named the attr first
                mobject = _attr_mobject(self)
                # a dynamic or extension attr can be re-added with another type
                if (
                    fn.attributeClass(mobject) == normal
                    and (
                        mobject.apiType() in _STATIC_DATA_API_TYPES
                        or _is_static_typed_root(mplug, mobject)
                    )
                    and not _plug_under_element(mplug)
                ):
                    key = (
                        fn.typeName,
                        fn.typeId.id(),
                        OpenMaya.MFnAttribute(mobject).name,
                        mplug.isArray,
                    )
            except Exception:
                key = None
            self.__dict__["_static_key_cache"] = key
        return key

    @property
    def is_dynamic(self) -> bool:
        """Attribute dynamic state."""
        _ensure_owner_alive(self)
        return self.plug.isDynamic

    @property
    def is_locked(self) -> bool:
        """Attribute locked state."""
        _ensure_owner_alive(self)
        return self.plug.isLocked

    @is_locked.setter
    def is_locked(self, state) -> None:
        """Set the locked state."""
        cmds.setAttr(self.full_name, lock=state)

    @property
    def is_keyable(self) -> bool:
        """Attribute keyable state."""
        _ensure_owner_alive(self)
        return self.plug.isKeyable

    @is_keyable.setter
    def is_keyable(self, state) -> None:
        """Set the keyable state."""
        cmds.setAttr(self.full_name, keyable=state)

    @property
    def is_channel_box(self) -> bool:
        """Attribute channel box state."""
        _ensure_owner_alive(self)
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
        _ensure_owner_alive(self)
        return self.plug.isConnected

    @property
    def is_free_to_change(self) -> bool:
        """Attribute free_to_change state. True if this attr and all its parents are
        free to change."""
        _ensure_owner_alive(self)
        return self.plug.isFreeToChange(True, False) == OpenMaya.MPlug.kFreeToChange

    def connect(self, other: str | Attribute, force: bool = False) -> None:
        """Connects this attr to other.

        Args:
            other: The attribute to connect to this attr.
            force: If True, break existing connection if found.
        """
        dst = str(other)
        src = self.full_name
        # cmds.isConnected is True only for a direct src -> dst connection, so it
        # is False for a dst plug that is not a destination; a name that may not
        # resolve to its plug keeps the query, and so its errors
        if (
            _names_own_plug(other)
            and not other._mplug.isDestination
            and _names_own_plug(self)
        ) or not cmds.isConnected(src, dst):
            cmds.connectAttr(src, dst, force=force)

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
        _ensure_owner_alive(self)
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
        _ensure_owner_alive(self)
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
            - a single list given to a `stringArray`, `vectorArray` or `pointArray`
              attr is expanded to the (count, *items) form cmds.setAttr() expects.

        Args:
            args, kwargs: args supported by cmds.setAttr()
        """
        _ensure_owner_alive(self)
        if "type" not in kwargs and not self._is_fixed_kind_outside_array():
            # typed attr requires the `type` arg to be specified.
            # the only weird one-off is `fltMatrix`, which is not typed but still
            # requires the `type` arg.
            typ = self.data_type
            if self.is_typed or typ == "matrix":
                kwargs["type"] = typ

        # special handling for counted array attrs
        # - cmds.setAttr() wants the item count first, then the unpacked items
        if (
            len(args) == 1
            and isinstance(args[0], (list, tuple, np.ndarray))
            and self.data_type in COUNTED_ARRAY_TYPES
        ):
            cmds.setAttr(self.full_name, len(args[0]), *args[0], **kwargs)

        # special handling for compound attrs
        # - use the unpacked first argument if it is a valid iterable
        #   and attr type ends with a digit (double2, double3, etc.)
        elif (
            len(args) == 1
            and isinstance(args[0], (list, tuple, np.ndarray))
            and str(self.data_type)[-1].isdigit()
        ):
            cmds.setAttr(self.full_name, *args[0], **kwargs)
        else:
            cmds.setAttr(self.full_name, *args, **kwargs)

    def _is_fixed_kind_outside_array(self) -> bool:
        """True if this attr's kind never takes the cmds.setAttr() `type` arg.

        A numeric, unit, enum, message or compound kind is not typed and never
        holds a matrix, so `set()` can skip its `data_type` query. Arrays,
        their elements and the children of either keep the query: their
        `data_type` can raise, and `set()` must keep raising that error. So
        does a plug whose attr was deleted, which the by-name query may find
        re-added with another kind.
        """
        try:
            return _fixed_attr_kind(self) in _NON_MATRIX_ATTR_API_TYPES and (
                not _plug_in_array(self._mplug)
            )
        except Exception:
            return False

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
        _ensure_owner_alive(self)
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
            self.__dict__["_geometry_attr_cache"] = result
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
                self.__dict__["_polymorphic_owner_cache"] = (
                    type_name in POLYMORPHIC_OUTPUT_NODE_TYPES
                )
            except RuntimeError:
                self.__dict__["_polymorphic_owner_cache"] = False
        return self._polymorphic_owner_cache

    def _get_geometry_value(self) -> Any:
        """Resolve the value for a typed geometry data plug.

        See `get()` for the four-tier resolution order.
        """
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
        _ensure_owner_alive(self)
        return self.plug.numChildren()

    def get_parent(self) -> Attribute | None:
        """Returns the parent attribute, if any."""
        _ensure_owner_alive(self)
        if not self.plug.isChild:
            return None
        plug = self.plug.parent()
        if plug and not plug.attribute().isNull():
            return _inherit_owner(self, _new_attr(Attribute, plug))

    def child(self, i: int) -> Attribute:
        """Returns the child attribute at the given index."""
        attr = self.__child_id_dict.get(i)
        if not attr:
            _ensure_owner_alive(self)
            attr = _inherit_owner(self, _new_attr(Attribute, self.plug.child(i)))
            self.__child_id_dict[i] = attr
        return attr

    # --- typed attr methods

    @property
    def is_typed(self):
        """Checks if this attribute is typed (raises as `mobject` does)."""
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

        # the getAttr list, also for a DSL Plug (whose `get()` is numpy-shaped)
        values = Attribute.get(self)
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
        _ensure_owner_alive(self)
        return self.plug.isArray

    @property
    def num_elements(self) -> int:
        """Number of elements / physical indices in the multi attr."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        return self.plug.numElements()

    def logical_index(self) -> int:
        """Returns the logical index if this attr is a child of a multi attr."""
        _ensure_owner_alive(self)
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
        element = self.plug.elementByPhysicalIndex(i)
        return _inherit_owner(self, _new_attr(Attribute, element))

    def element_by_logical_index(self, i: int) -> Attribute:
        """Returns the element attribute at the given logical index."""
        if not self.is_multi:
            raise RuntimeError(f"{self} is not an multi attr.")
        element = self.plug.elementByLogicalIndex(i)
        return _inherit_owner(self, _new_attr(Attribute, element))

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
            comp = self._get_component()
            # the lazy caches write `__dict__`, which skips a subclass's `__setattr__`
            self.__dict__["_Attribute__component_type"] = (
                comp.apiTypeStr if comp else "unknown"
            )

        return self.__component_type