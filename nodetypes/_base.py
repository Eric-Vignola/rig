"""
Node base classes, the Attribute wrapper and utils
"""

from __future__ import annotations

import dis as _dis
import itertools
import math
import re
import sys
import uuid
import weakref
from functools import total_ordering
from numbers import Number, Real
from types import FunctionType as _FunctionType
from types import MethodType as _MethodType
from typing import Any, Iterator

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig._internal import callbacks as _callbacks
from rig._internal import journal as _journal
from rig._internal.api_edits import api_connect as _api_connect
from rig._internal.api_edits import api_set as _api_set
from rig._internal.journal import cmds_edit, outside, rig_code, statement
from rig.nodetypes.errors import (
    _article,
    AmbiguousNodeError,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
)


CUSTOM_TYPE_ATTR = "__custom_node_type__"

# a node name MSelectionList can resolve without wildcards, plugs or components
_PLAIN_NODE_NAME = re.compile(r"^[|:\w]+$")

# a search pattern (`_lookup` refuses it: a pattern is never a node's name)
_PATTERN_CHARS = re.compile(r"[*?\[\]]")

# the current namespace, ":" at the root (an API call, 0.5 us: no command runs)
_current_namespace = OpenMaya.MNamespace.currentNamespace


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


# --- plug methods: a method whose name is also a Maya attribute ---------------
#
# THE RULE (user decision 2026-10-16), on a node or a plug: a bare name is the
# Maya attribute if one exists, otherwise the method; with brackets it is the
# method. ``node.serialize`` is the plug when the node has a Maya attribute
# ``serialize`` (a ``_MethodPlug``: a real Plug, whose call runs the method),
# else the bound method; ``node.serialize()`` is the method either way.
#
# `NodeMeta` makes every public method of every node class (a function, a
# classmethod, a staticmethod, a ``creates_or_updates``: subclasses, Container
# and user subclasses included) one of the descriptors below; `Attribute` and
# its subclasses do the same for their public functions (`_ChildOrMethod`, a
# compound child of that name). Class access (``Transform.serialize``) gives
# what the plain member gives.
#
# THE CALL-SITE CHECK (user decision): ``x.m(...)`` compiles the load of ``m``
# to a method load (3.11: LOAD_METHOD; 3.12+: LOAD_ATTR with its oparg's
# method bit, or LOAD_SUPER_ATTR for ``super().m()``). A descriptor reached by
# one returns the bound method without asking Maya, since a call is always the
# method. Any other access (``x.m``, ``getattr(x, "m")``, ``x.m(*a)``, which
# 3.11 loads by LOAD_ATTR) runs the full check, and so does every access on a
# Python version whose encoding is not known here (`_METHOD_LOAD` all 0).
#
# Rig's own code reaches public methods by calls (the cheap path) or through
# the class (``type(x).m(x)``, no check at all) on its hot paths: see
# test_plug_methods.TestInternalAccesses. A class-level identity constant of a
# public method (``Cls.__dict__["m"]``) holds `_raw_function` of it.


def _method_loads() -> bytes:
    """A table over the opcodes: 1 for a method load whatever its oparg (3.11's
    LOAD_METHOD), 2 for one whose oparg's bit 0 says so (3.12+: LOAD_ATTR,
    LOAD_SUPER_ATTR), 0 for anything else; all 0 on another Python (the full
    check always)."""
    version = sys.version_info[:2]
    ops     = _dis.opmap
    table   = bytearray(256)
    if version == (3, 11) and "LOAD_METHOD" in ops:
        table[ops["LOAD_METHOD"]] = 1
    elif (3, 12) <= version <= (3, 14) and "LOAD_ATTR" in ops and "LOAD_SUPER_ATTR" in ops:
        table[ops["LOAD_ATTR"]]       = 2
        table[ops["LOAD_SUPER_ATTR"]] = 2
    return bytes(table)


_METHOD_LOAD = _method_loads()
_GETFRAME    = sys._getframe


def _called(depth: int = 2) -> bool:
    """True if the frame `depth` levels up sits on a method load (the access is
    the callee of a call). The descriptors inline this; for tests and tools."""
    try:
        f = _GETFRAME(depth)
    except ValueError:
        return False
    co = f.f_code.co_code
    i  = f.f_lasti
    k  = _METHOD_LOAD[co[i]]
    return bool(k and (k & 1 or co[i + 1] & 1))


# ``(node, name, bound) -> node.<name>'s plug as a _MethodPlug running bound``,
# or `bound` when no exact Plug can be built; set by rig.nodetypes.dg_node
_METHOD_PLUG_HOOK = None

# what a method plug forwards of its method (besides __wrapped__ and the call):
# kept in its own __dict__, which wins over the class's non-data __doc__
_FORWARDED = ("__name__", "__qualname__", "__func__", "__self__", "__doc__")


def _as_method_plug(plug: Any, cls: type, bound: Any) -> Any:
    """`plug` made a method plug of class `cls` (``_MethodPlug`` /
    `_MethodAttribute`) whose call runs `bound`. The class is switched with
    ``object.__setattr__`` (``Plug.__setattr__`` is a journal statement)."""
    object.__setattr__(plug, "__class__", cls)
    d = plug.__dict__
    d["_method"]     = bound
    d["__wrapped__"] = bound
    for key in _FORWARDED:
        value = getattr(bound, key, _MISSING)
        if value is not _MISSING:
            d[key] = value
    return plug


def _node_has_attr(d: dict, name: str) -> bool:
    """True if the node of instance dict `d` is a live, built node with a Maya
    attribute `name` (its held fn set's ``hasAttribute``, after its handle's
    ``isValid``: a node a new scene freed is never asked; a deleted or
    half-built node is False)."""
    return (
        d.get("_attr_dict") is not None
        and d["_objhandle"].isValid()
        and d["_fn_set"].hasAttribute(name)
    )


class _PlugMethod:
    """A public function of a node class (see above): the function itself on the
    class, on a node the plug of its Maya attr of that name if it has one, else
    the bound method."""

    __slots__ = ("func", "name", "__dict__")

    def __init__(self, func: Any, name: str) -> None:
        self.func = func
        self.name = name
        self.__dict__.update(getattr(func, "__dict__", {}))
        for key in ("__module__", "__name__", "__qualname__", "__doc__"):
            try:
                self.__dict__[key] = getattr(func, key)
            except AttributeError:
                pass
        self.__dict__["__wrapped__"] = func

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        if inst is None:
            return self.func
        bound = _MethodType(self.func, inst)
        try:
            f = _GETFRAME(1)
        except ValueError:
            f = None
        if f is not None:
            co = f.f_code.co_code
            i  = f.f_lasti
            k  = _METHOD_LOAD[co[i]]
            if k and (k & 1 or co[i + 1] & 1):
                return bound
        d = inst.__dict__
        if (
            d.get("_attr_dict") is not None
            and d["_objhandle"].isValid()
            and d["_fn_set"].hasAttribute(self.name)
        ):
            return _METHOD_PLUG_HOOK(inst, self.name, bound)
        return bound

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # ``Cls.__dict__[name](node, ...)`` calls the function, as it did
        return self.func(*args, **kwargs)

    def __set_name__(self, owner: type, name: str) -> None:
        set_name = getattr(type(self.func), "__set_name__", None)
        if set_name is not None:
            set_name(self.func, owner, name)

    def __repr__(self) -> str:
        return f"<plug method {self.__dict__.get('__qualname__', self.name)}>"


class _PlugDescriptor(_PlugMethod):
    """`_PlugMethod` of another method descriptor (``creates_or_updates``)."""

    __slots__ = ()

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        if inst is None:
            return self.func.__get__(None, owner)
        bound = self.func.__get__(inst, owner)
        try:
            f = _GETFRAME(1)
        except ValueError:
            f = None
        if f is not None:
            co = f.f_code.co_code
            i  = f.f_lasti
            k  = _METHOD_LOAD[co[i]]
            if k and (k & 1 or co[i + 1] & 1):
                return bound
        if _node_has_attr(inst.__dict__, self.name):
            return _METHOD_PLUG_HOOK(inst, self.name, bound)
        return bound


class _PlugClassMethod(classmethod):
    """`_PlugMethod` of a classmethod; still a classmethod (``.__func__``,
    ``isinstance(x, classmethod)``, help()'s "class method")."""

    __slots__ = ("_name",)

    def __init__(self, func: Any, name: str) -> None:
        super().__init__(func)
        self._name = name

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        if inst is None:
            return _MethodType(self.__func__, owner)
        bound = _MethodType(self.__func__, type(inst))
        try:
            f = _GETFRAME(1)
        except ValueError:
            f = None
        if f is not None:
            co = f.f_code.co_code
            i  = f.f_lasti
            k  = _METHOD_LOAD[co[i]]
            if k and (k & 1 or co[i + 1] & 1):
                return bound
        if _node_has_attr(inst.__dict__, self._name):
            return _METHOD_PLUG_HOOK(inst, self._name, bound)
        return bound


class _PlugStaticMethod(staticmethod):
    """`_PlugMethod` of a staticmethod."""

    __slots__ = ("_name",)

    def __init__(self, func: Any, name: str) -> None:
        super().__init__(func)
        self._name = name

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        func = self.__func__
        if inst is None:
            return func
        try:
            f = _GETFRAME(1)
        except ValueError:
            f = None
        if f is not None:
            co = f.f_code.co_code
            i  = f.f_lasti
            k  = _METHOD_LOAD[co[i]]
            if k and (k & 1 or co[i + 1] & 1):
                return func
        if _node_has_attr(inst.__dict__, self._name):
            return _METHOD_PLUG_HOOK(inst, self._name, func)
        return func


_PLUG_DESCRIPTORS = (_PlugMethod, _PlugClassMethod, _PlugStaticMethod)

# rig.nodetypes._files.creates_or_updates, read at the first class made
_CREATES_OR_UPDATES = None


def _plug_member(name: str, value: Any) -> Any:
    """The class-dict entry `NodeMeta` keeps for the member `name` = `value` of a
    node class body: a public method becomes its plug-method descriptor; any
    other member (a property, data, a private name, a descriptor already made,
    a classmethod / staticmethod subclass someone wrote) is kept."""
    global _CREATES_OR_UPDATES
    if name[:1] == "_" or isinstance(value, _PLUG_DESCRIPTORS):
        return value
    if isinstance(value, _FunctionType):
        return _PlugMethod(value, name)
    if type(value) is classmethod:
        return _PlugClassMethod(value.__func__, name)
    if type(value) is staticmethod:
        return _PlugStaticMethod(value.__func__, name)
    if _CREATES_OR_UPDATES is None:
        from rig.nodetypes._files import creates_or_updates

        _CREATES_OR_UPDATES = creates_or_updates
    if isinstance(value, _CREATES_OR_UPDATES):
        return _PlugDescriptor(value, name)
    return value


def _raw_function(value: Any) -> Any:
    """The function of a class-dict method entry, a plug-method descriptor or
    not: what an identity constant holds (``_raw_function(DGNode.__dict__[
    "find_attr"])`` is what ``cls.find_attr`` gives on the class)."""
    if isinstance(value, (_PlugMethod, _ChildOrMethod)):
        value = value.func
    return getattr(value, "__func__", value)


def is_plug_method(cls: type, name: str) -> bool:
    """True if ``name`` on node class or plug class `cls` is a method that a Maya
    attribute of that name turns into the plug (``node.<name>``)."""
    return isinstance(_class_attr(cls, name), _PLUG_DESCRIPTORS + (_ChildOrMethod,))


# The node class registry: the node type (a class's ``CUSTOM_NODE_TYPE``, else
# its ``NATIVE_NODE_TYPE``) -> the class, filled by `NodeMeta`. `_cast` and
# `Node.create` / `Node.find_all` dispatch through it.
_NODE_CLASS_DICT: dict = {}
# `_cast`'s caches, cleared by every new node class: (typeName, typeId) -> the
# class of a node without custom type, and the keys a node was built for
_CLASS_BY_TYPE: dict = {}
_CASTABLE_TYPES: set = set()


class NodeMeta(type):
    """
    The metaclass of every node class: it registers a class for its node type
    (`_NODE_CLASS_DICT`), which the typed cast and ``Node.create`` read.

    Calling a node class names a node that already exists, and never writes
    the scene. The root :class:`Node` is the DSL node factory
    (``Node("pCube1")`` returns the typed node); any other class is the strict
    typed reference (`_refer`): ``Transform("j1")`` returns the node
    ``Node("j1")`` gives, ``Joint("j1")``, checked to be a Transform. That is the one branch
    point of a node class call. The cast core never runs it: it builds a class
    with ``type.__call__``, and the package's own sites that already know a
    node's exact class build it with ``Cls._wrap(x)``. ``x in Cls`` (the
    class, not a node) is a TypeError naming the node spelling (``x in
    Blinn('red')``; ``x >> Blinn()`` lists them).
    """

    def __contains__(cls, lhs: Any) -> bool:
        name = cls.__name__
        if getattr(cls, "_MEMBER_KIND", False):
            text = (
                f"{name} is the class: {name}() is the kind token (x >> {name}() lists "
                f"them) and {name}('x') one; write x in {name}('x')"
            )
        elif getattr(cls, "_KIND_HINT", None):
            text = f"{name} is the class: write x in {name}('x'); {cls._KIND_HINT}"
        else:
            text = (
                f"{name} is a node class: 'in' asks a membership node or Tag "
                f"(x in Layer('L'), x in Blinn('red'), x in Tag('cap'))"
            )
        raise TypeError(text)

    def __call__(cls, *args, **kwargs):
        if cls is Node:
            return _node_factory(*args, **kwargs)
        return _refer(cls, args, kwargs)

    def _wrap(cls, obj: Any) -> Any:
        """The node object of class ``cls`` for ``obj`` (a name, MObject,
        MDagPath or node object), built by the class's constructor, its
        ``is_type`` check included, without the reference's lookup and
        most-derived cast.

        Private, for the package's own sites whose node's exact class is known:
        a node just created (``post_create``), read by type (``cmds.ls(type=)``,
        ``listConnections(type=)``) or type-checked. Anything else calls
        ``Cls(x)`` / ``Node(x)``, so no exact-class wrapper of a subtype node (a
        ``Transform`` object of a joint) reaches a user. A metaclass method, so
        a node never has it. The constructor of a class that is not rig's own
        is code a user wrote, run outside the open journal statement."""
        if rig_code(cls):
            return type.__call__(cls, obj)
        with outside():
            return type.__call__(cls, obj)

    def wrap(cls, value: Any) -> Any:
        """Wrap a ``maya.cmds`` result (str / list-of-str) as a node /
        :class:`List`.

        A metaclass method, so ``Node.wrap`` (and ``Transform.wrap``) works but a
        node never has it: a Maya attr ``wrap`` (3D textures) stays reachable as
        ``node.wrap``. Every string is a name Maya returned, cast as Maya
        resolves it (the node before the first ``.``), whichever class it is
        called on: never ``Node(x)``'s lookup rule for the names a user
        writes, so a returned ``x`` stays the node Maya meant while another
        namespace is current.

        Use this when calling ``maya.cmds`` directly (instead of going through
        :mod:`rig.bridges.commands`) and you want the result back in DSL form::

            from maya import cmds
            from rig import Node

            n   = Node.wrap(cmds.createNode("transform"))     # -> Transform
            sel = Node.wrap(cmds.ls(sl=True))                  # -> List of nodes
            x   = Node.wrap(5.0)                                # -> 5.0 (passthrough)
            none = Node.wrap(None)                              # -> None

        Strings that aren't valid node names are passed through unchanged
        (so ``Node.wrap(cmds.getAttr("foo.attr", asString=True))`` won't try
        to coerce a value-string into a node).
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return _cast_node(value)
            except Exception:
                return value
        if isinstance(value, (list, tuple)):
            from rig._internal.list import _values_list

            wrapped = [Node.wrap(v) for v in value]
            try:
                return _values_list(wrapped)
            except Exception:
                return wrapped
        return value

    def __new__(mcs, class_name, bases, attrs):
        # every public method of the class body is a plug method (see
        # `_PlugMethod`): node.<name> is the plug of a Maya attr of that name
        for key, value in list(attrs.items()):
            if key[:1] != "_":
                member = _plug_member(key, value)
                if member is not value:
                    attrs[key] = member
        cls_obj   = type.__new__(mcs, class_name, bases, attrs)

        # a new class can change the dispatch of any node type
        _CLASS_BY_TYPE.clear()
        _CASTABLE_TYPES.clear()
        _STATIC_DATA_TYPE.clear()

        node_type = attrs.get("CUSTOM_NODE_TYPE")
        if not node_type:
            node_type = attrs.get("NATIVE_NODE_TYPE")
        if node_type:
            _NODE_CLASS_DICT[node_type] = cls_obj

        return cls_obj


def _cast(obj: Any, build: bool = True) -> Any:
    """The typed cast, the one cast core (D12). Private: the public doors are
    ``Node(x)`` (always a node; it calls this) and ``Attribute(x)`` /
    ``Plug(x)`` (an attribute).

    A node object or an Attribute is returned as is, an MPlug or a dotted name
    gives an Attribute, and a node name, uuid, MObject or MDagPath gives the
    typed node: an instance of the most derived class registered for its custom
    type or type chain (`_NODE_CLASS_DICT`). Anything else raises ValueError.
    The package casts through this function wherever a Maya name, MObject or
    MPlug becomes an object.

    ``build=False`` is ``Cls.exists``' probe: where the type key decides the
    class and that class keeps the DGNode or DAGNode constructor (which builds
    any live node of its type), the class is returned instead of the node
    built from it; a class with its own constructor is built, since that may
    refuse a live node (a geometry shape's: MFnMesh refuses an empty mesh). A
    plain name MSelectionList resolves to no node gives None (the cast by name
    would fail: Maya's by-name queries resolve the name alike)."""
    mobj     = None
    dag_path = None
    sel      = None
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
        if not build and sel is not None and not sel.length():
            return None
        return _cast_by_name(obj)

    if key in _CLASS_BY_TYPE:
        cls_obj = _CLASS_BY_TYPE[key]
    else:
        cls_obj = _CLASS_BY_TYPE[key] = _native_node_class(obj)
    if not cls_obj:
        raise ValueError(f"Failed casting {obj}")
    if not build and type(cls_obj) is NodeMeta and _keeps_base_constructor(cls_obj):
        return cls_obj

    # a type already cast passes the class's type check again, so a
    # base-constructor class is built without re-running it (from the
    # selection a name was just resolved in); a class of a metaclass derived
    # from NodeMeta is always built by its ``_wrap``, below
    inst = None
    if key in _CASTABLE_TYPES and type(cls_obj) is NodeMeta:
        inst = _construct_checked_type(cls_obj, obj, sel)
    if inst is None:
        # a node class is constructed without NodeMeta.__call__'s frame (the
        # reference, which casts): a metaclass derived from NodeMeta may
        # override ``_wrap``. Its constructor may be a user's
        _USER_CODE_RUNS[0] += 1
        inst = _construct(cls_obj, obj)
    _CASTABLE_TYPES.add(key)
    return inst


# The cast rule's fallback for a DG type no class is registered for (D31:
# ``rig.nodetypes.material_node`` sets it when it loads): type name -> the
# generic ``Material`` for a type Maya classifies a surface shader
# (``anisotropic``, a plug-in shader), else None; memoised per type.
_CLASSIFY = None


def _native_node_class(obj: str) -> Any:
    """Returns the class the node's type chain maps to, ignoring any custom type.

    The most derived registered type of the chain wins, except that a class
    flagged ``_EXACT_TYPE`` (the shader classes) only takes its own exact
    type: ``Lambert`` never claims a blinn. A DG node that no class claims is
    a ``Material`` when Maya classifies its type a surface shader
    (``_CLASSIFY``), else a ``DGNode``."""
    # set default node type to DAG or DG
    default_type = "dagNode" if cmds.ls(obj, dag=True) else "entity"
    default      = _NODE_CLASS_DICT.get(default_type)
    cls_obj      = default

    # override cls_obj with a defined node class if any
    chain = cmds.nodeType(obj, inherited=True)
    for depth, t in enumerate(reversed(chain)):
        found = _NODE_CLASS_DICT.get(t)
        if found is not None and not (depth and getattr(found, "_EXACT_TYPE", False)):
            cls_obj = found
            break
    if cls_obj is default and default_type == "entity" and chain and _CLASSIFY is not None:
        cls_obj = _CLASSIFY(chain[-1]) or cls_obj
    return cls_obj


def _cast_by_name(obj: str) -> Any:
    """`_cast`'s tail for a name the API does not resolve to one node without a
    custom type (a custom-typed node, a name several nodes or none have, a
    deleted or undone node): casts it by querying its custom type and type
    chain by name."""
    cls_obj     = None
    custom_type = get_custom_type(obj)
    if custom_type:
        cls_obj = _NODE_CLASS_DICT.get(custom_type)
    if not cls_obj:
        cls_obj = _native_node_class(obj)

    if cls_obj:
        # as in `_cast`: no NodeMeta.__call__ frame, and a constructor that may
        # be a user's
        _USER_CODE_RUNS[0] += 1
        return _construct(cls_obj, obj)

    raise ValueError(f"Failed casting {obj}")


def _construct(cls_obj: Any, obj: Any) -> Any:
    """The node object of ``cls_obj`` for ``obj``, as `_cast` builds it: its
    constructor (no ``NodeMeta.__call__`` frame), or the ``_wrap`` of a metaclass
    derived from NodeMeta. Code a user wrote (a class or metaclass that is not
    rig's own) runs outside the open journal statement."""
    if type(cls_obj) is NodeMeta:
        if rig_code(cls_obj):
            return type.__call__(cls_obj, obj)
        with outside():
            return type.__call__(cls_obj, obj)
    if rig_code(type(cls_obj)):
        return cls_obj._wrap(obj)
    with outside():
        return cls_obj._wrap(obj)


def _from_root(name: str) -> str:
    """A node name as the lookup rule reads it: a namespaced name that is no
    path and names no namespace from the root (``char:x``) spelled from the
    root (``:char:x``), so ``namespace -relativeNames`` cannot turn it
    relative to the current namespace; any other name as written."""
    if ":" in name and name[0] != ":" and "|" not in name:
        return f":{name}"
    return name


def _lookup(name: str, label: str = "node", probe: bool = False) -> str | None:
    """The one node ``name`` names, by the lookup rule shared by ``Node(x)``,
    the node classes and the membership lookups. Cold: ``Node(x)`` runs it
    after a failed cast, and for a bare name while a namespace other than the
    root is current.

    * A qualified name (a path or ``ns:name``) is looked up as written, a
      namespace from the root (``char:x`` is ``:char:x``, whatever
      ``namespace -relativeNames``, as ``define`` keys it).
    * A bare name is looked up at the root namespace (``:x``) and, when the
      current namespace is not the root, in it too (``:char:x``), both spelled
      absolutely so ``namespace -relativeNames`` cannot change the answer. A
      node at both raises AmbiguousNodeError ("spell the namespace").
    * A pattern (``*``, ``?``, ``[]``) raises NodeLookupError: a pattern is a
      search (``cmds.ls``, ``find_all``), never a node's name.
    * A name several DAG nodes have raises AmbiguousNodeError listing their
      paths (an instanced node is one node).
    * No node raises NodeNotFoundError, whose message names ``label``
      ("no joint named 'x'") and carries the hints (see ``errors``); with
      ``probe=True`` (``Cls.exists``) it returns None instead.

    Returns the node's name as ``cmds.ls(long=True)`` prints it (a full path
    for a DAG node), which names that one node.
    """
    if _PATTERN_CHARS.search(name):
        raise NodeLookupError(name, label)
    if ":" in name or "|" in name:
        spellings = (_from_root(name),)
    else:
        current   = _current_namespace()
        spellings = (f":{name}",) if current == ":" else (f":{name}", f"{current}:{name}")
    try:
        found = cmds.ls(*spellings, long=True) or []
    except RuntimeError:  # no name Maya can parse ('', '|', '1bad')
        found = []
    if len(found) == 1:
        return found[0]
    if not found:
        if probe:
            return None
        raise NodeNotFoundError(name, label)
    # several nodes: DAG nodes under different parents, or both namespaces
    by_spelling = [(s, cmds.ls(s, long=True) or []) for s in spellings]
    for _, nodes in by_spelling:
        if len(nodes) > 1:
            raise AmbiguousNodeError(name, nodes, label)
    raise AmbiguousNodeError(name, [s for s, nodes in by_spelling if nodes], label, namespaces=True)


def _type_label(cls: type) -> str:
    """How a lookup error names the nodes of node class ``cls``: its node type
    (``"joint"``), ``"node"`` for DGNode and ``"DAG node"`` for DAGNode, or
    the class's ``_TYPE_LABEL`` (the unregistered Container: ``"container"``)."""
    label = getattr(cls, "_TYPE_LABEL", None) or cls.CUSTOM_NODE_TYPE or cls.NATIVE_NODE_TYPE
    return {"entity": "node", "dagNode": "DAG node"}.get(label, label)


def _is_uuid(name: str) -> bool:
    """Whether ``_cast`` reads the str ``name`` as a uuid (as it does: 32 hex
    digits or more, the uuid form)."""
    return len(name) >= 32 and is_valid_maya_uid(name)


def _cast_node(name: str) -> Any:
    """The node of a name Maya returned (a ``cmds`` result, a node's own
    name): the cast core on the part before the first ``.``, resolved as Maya
    resolves it. ``Node(x)``'s lookup rule is for the names a user writes; a
    name Maya returned is exact (``namespace -relativeNames`` included) and
    must not turn ambiguous because another namespace is current. Used by
    ``container.createNode``, the ``rc`` results, ``Node.wrap`` and the
    container's uuid reads."""
    return _cast(name.split(".", 1)[0])


# (DGNode.__init__, DAGNode.__init__, their is_type functions, their name
# properties), bound on first use since dg_node imports this module
_BASE_CONSTRUCTOR_PARTS = None

# (DGNode, DAGNode), bound on first use for the same reason
_BASE_CLASSES = None


def _keeps_base_constructor(cls_obj) -> bool:
    """Whether node class ``cls_obj`` is built by the DGNode or DAGNode
    constructor (the one those classes hold now), which builds any live node
    of a type the cast gives the class. A class with its own constructor may
    refuse one: a geometry shape's function set refuses an empty shape."""
    global _BASE_CLASSES
    if _BASE_CLASSES is None:
        from rig.nodetypes.dag_node import DAGNode
        from rig.nodetypes.dg_node import DGNode

        _BASE_CLASSES = (DGNode, DAGNode)
    init = cls_obj.__init__
    return init is _BASE_CLASSES[0].__init__ or init is _BASE_CLASSES[1].__init__


def _construct_checked_type(cls_obj, obj: str, sel: Any = None) -> Any:
    """Builds `cls_obj(obj)` without its `is_type` check, or returns None.

    For a class that keeps the DGNode or DAGNode constructor, type check and
    name property, `is_type` only depends on the node type and the custom type
    attr, so it passes for every node of a type that was already cast. The
    constructor's state is set in the constructor's order, the handle from the
    MObject just resolved (see `DGNode.__init__`). ``sel`` is the
    MSelectionList the cast resolved the name ``obj`` in (one node), which is
    the one the constructor would make; None makes it. Returns None when the
    class does not qualify or a step raises, and the caller then runs the real
    constructor, which raises its own errors.
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
        )
    dg_init, dag_init, is_type_funcs, name_props = _BASE_CONSTRUCTOR_PARTS

    init = cls_obj.__init__
    if (
        (init is not dg_init and init is not dag_init)
        or cls_obj.__new__ is not object.__new__
        or cls_obj.CUSTOM_NODE_TYPE
        or getattr(cls_obj.is_type, "__func__", None) not in is_type_funcs
        or cls_obj.name not in name_props
    ):
        return None
    try:
        if sel is None:
            sel = OpenMaya.MSelectionList()
            sel.add(str(obj))
        inst = object.__new__(cls_obj)
        d    = inst.__dict__
        if init is dag_init:
            d["_mdagpath"] = sel.getDagPath(0)
            d["_mobject"]  = d["_mdagpath"].node()
            d["_fn_set"]   = cls_obj.FN_SET(d["_mdagpath"])
        else:
            d["_mobject"] = sel.getDependNode(0)
            d["_fn_set"]  = cls_obj.FN_SET(d["_mobject"])
        d["_objhandle"] = OpenMaya.MObjectHandle(d["_mobject"])
        d["_attr_dict"] = {}
    except Exception:
        return None
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
        cmds_edit.deleteAttr(attr)
    cmds_edit.addAttr(node_name, ln=CUSTOM_TYPE_ATTR, dt="string")
    cmds_edit.setAttr(attr, custom_type, type="string")
    cmds_edit.setAttr(attr, lock=True)


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
# the node class, so `Attribute.get_data_type` shares it across instances in
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

# The epoch of the kind memo a plug keeps (`_fixed_attr_kind`): a new token at
# every plug-in load and unload (`_clear_static_data_type`), which can redefine
# a node type's attributes, so a memo of an older token is computed again. A
# token, not a count: an in-place reload of this module makes a new one, which
# no memo made before it matches.
_KIND_EPOCH = [object()]
# the attribute classes `_fixed_attr_kind` tells apart
_INVALID_ATTR = OpenMaya.MFnDependencyNode.kInvalidAttr
_NORMAL_ATTR  = OpenMaya.MFnDependencyNode.kNormalAttr

# (attr, "typed" or "Tdata", node) while `Attribute.get_data_type` runs the fallback
# hook of attr's node: the answer its getAttr query gave, and the node whose class
# hook that call runs first (`_hook_node`), see `_queried_data_type`
_FALLBACK_QUERY = None

# A count of the times code a user may have written ran where a type query can
# run it: a fallback hook `Attribute.get_data_type` calls that `_keeps_query` does
# not vouch for (a node class's own, a class-level patch, a hook set on the
# node), and a node constructor a cast runs (`_cast`). `shorthand._Facts`
# reads it around its data type query, so a fact whose query ran such code is
# queried again where it was before (see `_inject_value`).
_USER_CODE_RUNS = [0]

# Bound by `rig.nodetypes.dg_node` when it loads (it imports this module):
# `_keeps_query`, and DGNode's `name`, `fn_set` and `ensure_valid` (see
# `_dg_named_owner`), read before any patch of them can exist
_KEEPS_QUERY = None
_DG_NAMING   = None


def _fn_set_name(node: Any) -> str:
    """The fn set name of `node`: the partial path name of a DAG fn set, else the
    node name. Reads the fn set directly, never `node.name`."""
    fn = node.__dict__["_fn_set"]
    return fn.partialPathName() if isinstance(fn, OpenMaya.MFnDagNode) else fn.name()


# The handle of a node (`_objhandle`, an `MObjectHandle`) is the one object
# that is safe to read once a new scene, a file open or a reference unload
# freed the node: its other API objects (the MObject, the fn sets, its plugs'
# MPlugs) then point at freed memory. It is built from the MObject a
# constructor resolved, while the node is alive (the constructors, the cast's
# replay, a copy): one built later from a held MObject cannot tell a freed
# node (after a new scene it says valid). The cold readers ask
# `_handle_valid`; the hot paths read the handle inline. A plug keeps the
# handle of its node as `_handle`: its owner's own `_objhandle` object, or the
# one it took when it was built (see `_ensure_owner_alive`).


def _handle_valid(d: dict) -> bool:
    """True if the node whose `__dict__` is `d` is valid: alive and not deleted
    to the undo queue. False for a half-built node (no handle yet)."""
    handle = d.get("_objhandle")
    return handle is not None and handle.isValid()


# --- lock / unlock (lockNode on a node, setAttr -lock on a plug)

# ``rig._internal.container`` sets it when it loads (the D31 pattern, as
# ``errors._SCOPE_HINT_HOOK``): ``uuid -> bool``, True while the container of
# that uuid has its ``with container(...)`` block open (a locked container
# takes no new member, so it is locked after its block)
_OPEN_CONTAINER_HOOK = None


def _injection_error(text: str) -> RuntimeError:
    """The InjectionError of ``text``: the one error class of a ``<<`` refusal,
    raised by ``x.lock()`` / ``x.unlock()`` (``x << lock``) and by rig's verbs
    on a locked node."""
    from rig._internal.plug import InjectionError  # plug imports this module

    return InjectionError(text)


def _node_by_name(name: str) -> Any:
    """The MObject of the node ``name``, or None."""
    try:
        sel = OpenMaya.MSelectionList()
        sel.add(name)
        return sel.getDependNode(0)
    except (RuntimeError, TypeError, ValueError):
        return None


def _locking_container(name: str) -> "str | None":
    """The outermost locked container that holds node ``name`` (a container
    locks its members, and the members of its sub-containers), or None. Read on
    a refusal's path only (``container -q``, about 13 us)."""
    holder  = None
    current = cmds.container(query=True, findContainer=name)
    seen    = set()
    while current and current not in seen:
        seen.add(current)
        mobject = _node_by_name(current)
        if mobject is not None and OpenMaya.MFnDependencyNode(mobject).isLocked:
            holder = current
        parents = cmds.container(current, query=True, parentContainer=True)
        current = parents[0] if isinstance(parents, list) and parents else parents or None
    return holder


def _unlock_way(name: str) -> tuple:
    """``(why, way)`` of the locked node ``name`` for a refusal: ``("it is
    locked", "Node('x').unlock() unlocks it")``, or, when a locked container
    holds it, that container's (whose unlock unlocks every member)."""
    holder = _locking_container(name)
    if holder is not None and holder != name:
        return f"it is locked by its container '{holder}'", f"Node({holder!r}).unlock() unlocks it"
    return "it is locked", f"Node({name!r}).unlock() unlocks it"


def _locked_node_text(verb: str, name: str) -> str:
    """The one refusal text of rig's own verbs on a locked node: ``cannot
    <verb> 'x': it is locked; Node('x').unlock() unlocks it. Nothing was
    changed``."""
    why, way = _unlock_way(name)
    return f"cannot {verb} '{name}': {why}; {way}. Nothing was changed"


def _refuse_locked_node(fn: Any, verb: str, name: "str | None" = None) -> None:
    """InjectionError (`_locked_node_text`) when the node of fn set ``fn``
    (named ``name``, else by the fn set) is locked
    (``MFnDependencyNode.isLocked``, 0.1 us: its own lock or its container's),
    before Maya's own error."""
    if fn.isLocked:
        raise _injection_error(_locked_node_text(verb, name or fn.name()))


def _node_lock_refusal(fn: Any, name: str, state: bool) -> "str | None":
    """Why ``lockNode`` must not lock (``state``) or unlock node ``name`` (fn set
    ``fn``), or None. Read only when the state changes."""
    if fn.isFromReferencedFile:
        return (
            f"'{name}' is referenced: Maya saves no lockNode edit for a referenced "
            f"node, so the {'lock' if state else 'unlock'} would be gone at the next "
            f"open; change it in its own file. Nothing was changed"
        )
    type_name = fn.typeName
    if type_name == "reference" and not state:
        return (
            f"'{name}' is a reference node, which Maya keeps locked: unlocked, it "
            f"can be deleted under its reference. Nothing was changed"
        )
    if state:
        hook = _OPEN_CONTAINER_HOOK
        if type_name == "container" and hook is not None and hook(str(fn.uuid())):
            return (
                f"lock '{name}' after its with block: a locked container takes no "
                f"new member. Nothing was changed"
            )
        return None
    holder = _locking_container(name)
    if holder is not None and holder != name:
        return (
            f"'{name}' is locked by its container '{holder}' (a locked container "
            f"locks its members): Node({holder!r}).unlock() unlocks it, and every "
            f"member with it. Nothing was changed"
        )
    return None


def _plug_lock_refusal(attr: Any, state: bool, node: bool = True) -> "str | None":
    """Why ``setAttr -lock`` must not lock (``state``) or unlock plug ``attr``,
    or None: its node is locked (Maya refuses both: "is from a locked node, so
    it cannot be locked"; ``node=False`` skips that check, for a List that
    unlocks the node first), or, for an unlock, a locked parent holds the lock
    (Maya would leave the child locked)."""
    mplug = attr._mplug
    if node:
        # the fn set of the node object that owns the plug (its node: the callers
        # checked it alive, `_ensure_owner_alive`), else one of the plug's node
        owner = attr.__dict__.get("_node")
        fn    = owner.__dict__.get("_fn_set") if owner is not None else None
        if fn is None:
            fn = OpenMaya.MFnDependencyNode(mplug.node())
        if fn.isLocked:
            name = str(attr).partition(".")[0]
            return (
                f"'{str(attr)}' is on the locked node '{name}', whose plugs Maya "
                f"never {'locks' if state else 'unlocks'}: {_unlock_way(name)[1]}. "
                f"Nothing was changed"
            )
    if state or not mplug.isLocked:
        return None
    # a compound parent's lock locks its children, a multi root's its elements
    # (Maya then leaves the child locked through an unlock of its own flag)
    holder = None
    cur    = mplug
    while cur.isChild or cur.isElement:
        cur = cur.parent() if cur.isChild else cur.array()
        if cur.isLocked:
            holder = cur
    if holder is None:
        return None
    held = str(_new_attr(Attribute, holder))
    return (
        f"'{str(attr)}' is locked by its parent '{held}' (a locked parent locks "
        f"its children): Plug({held!r}).unlock() unlocks it. Nothing was changed"
    )


def _ensure_owner_alive(attr: Any) -> None:
    """Raise ``"... already deleted!"`` if the node that owns `attr` was freed (a
    new scene, a file open, a reference unload): `attr`'s MPlug then points at
    freed memory, and an MPlug call can read another node or crash Maya.

    Only the handle of the plug's node is read (`_handle`), which is safe on a
    freed node: its owner's own handle when it has an owner (the owner's
    `_objhandle` object, bound with the owner), else the one it took when it was
    built from a name or an MPlug (see `_mplug_handle`). A node deleted to the
    undo queue is still alive and passes. A freed owner raises through its
    `ensure_valid`. The children, elements and parent of an attr share its owner
    and that handle (see `_inherit_owner`).

    The attribute of a dynamic attr is also freed on its own, once a delete of
    the attr leaves the undo queue (a flush, or ten more commands at mayapy's
    default queue length), while its node lives on. Such an attr reads the
    handle of its attribute it took too (`_attrh`, see `_attr_handle`), and
    raises ``"<plug> already deleted!"``, named by its str buffer.
    """
    d      = attr.__dict__
    handle = d.get("_handle")
    if handle is not None and not handle.isAlive():
        node = d.get("_node")
        if node is not None:
            node.ensure_valid()
        else:
            _raise_deleted(attr, handle)
    handle = d.get("_attrh")
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
    handle = d.get("_handle")
    if handle is not None and not handle.isValid():
        _raise_deleted(attr, handle)
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


def _ensure_node_valid(attr: Any) -> None:
    """Raise ``"... already deleted!"`` if the node of `attr` was deleted or freed,
    without naming it: through its owner (`DGNode.ensure_valid`), or, with none,
    the handle of its node it took (see `_ensure_node_castable`). For a caller
    that hands `attr`'s str buffer to cmds, which would name the node that took
    the name (see `rig.bridges.commands`)."""
    d    = attr.__dict__
    node = d.get("_node")
    if node is None:
        _ensure_node_castable(attr)
        return
    handle = d.get("_handle")  # the owner's
    if handle is None or not handle.isValid():
        node.ensure_valid()
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


# Python's own attribute lookup, which `_dg_named_owner` asks a class to keep
_OBJECT_GETATTRIBUTE = object.__dict__["__getattribute__"]


def _dg_named_owner(attr: Any) -> Any:
    """The owner of `attr` if naming it (`Attribute.get_full_name`) can raise only
    through `_ensure_dg_named`'s two checks, else None.

    That holds when `attr`'s class keeps Attribute's `get_full_name` and its owner's
    class keeps DGNode's `name`, `fn_set` and `ensure_valid` (and Python's own
    attribute lookup), with no `ensure_valid` set on the owner itself: the name
    is then the owner's `ensure_valid()`, the `name()` of its fn set (which a
    valid node answers), the check of a dynamic attr's attribute handle, and the
    MPlug's own name, which a valid node's plug answers, a deleted dynamic
    attribute's too. So a caller that needs only the raise runs the two checks
    and builds no name. A DAG node's class names it through its path
    (`DAGNode.name` re-binds a stale one), so it is never such an owner. The
    classes are read at every call, so a patched method is seen at once. The
    owner test is `_dg_named_node`'s, written out here (the hot path)."""
    node = attr.__dict__.get("_node")
    if node is None or type(attr).get_full_name is not _ATTRIBUTE_FULL_NAME:
        return None
    naming = _DG_NAMING
    cls    = type(node)
    try:
        if (
            naming is None
            or cls.name is not naming[0]
            or cls.fn_set is not naming[1]
            or cls.ensure_valid is not naming[2]
            or cls.__getattribute__ is not _OBJECT_GETATTRIBUTE
            or "ensure_valid" in node.__dict__
        ):
            return None
    except AttributeError:
        return None
    return node


def _dg_named_node(node: Any) -> bool:
    """True if `node` (a node object) is named as a DGNode is, the owner test of
    `_dg_named_owner`: its class keeps DGNode's `name`, `fn_set` and
    `ensure_valid` (and Python's own attribute lookup), with no `ensure_valid`
    set on the node itself. Its name is then its fn set's, which names that node
    while it is valid, and naming it raises exactly when its handle is not valid
    (`DGNode.ensure_valid`)."""
    naming = _DG_NAMING
    cls    = type(node)
    try:
        return not (
            naming is None
            or cls.name is not naming[0]
            or cls.fn_set is not naming[1]
            or cls.ensure_valid is not naming[2]
            or cls.__getattribute__ is not _OBJECT_GETATTRIBUTE
            or "ensure_valid" in node.__dict__
        )
    except AttributeError:
        return False


def _ensure_dg_named(attr: Any, owner: Any) -> None:
    """Raise what naming `attr` raises, `owner` being its `_dg_named_owner`: the
    owner's ``"... already deleted!"`` (`ensure_valid`), then the freed dynamic
    attribute's ``"<plug> already deleted!"`` (see `_ensure_owner_alive`), the
    two checks `Attribute.get_full_name` makes, in its order and with its texts."""
    _DG_NAMING[2](owner)  # its class keeps DGNode's ensure_valid (`_dg_named_owner`)
    handle = attr.__dict__.get("_attrh")
    if handle is not None and not handle.isAlive():
        raise _deleted_error(str.__str__(attr))


# The nodes a shader conversion deleted (``rig._internal.shade_convert``):
# {the old node's uuid: (its name, the type it was converted to)}, so each
# held node of a chain of conversions, in any namespace, names its own.
# `_deleted_error` reads it, on the error path only; the conversion module
# clears it before a new scene or a file open.
_CONVERTED: dict = {}


def _deleted_error(name: str, freed: bool = False, uuid: str | None = None) -> RuntimeError:
    """The ``"... already deleted!"`` RuntimeError of the node `name`, deleted to
    the undo queue, or, `freed`, freed by a new scene, a file open or a reference
    unload: a freed node cannot be read, so `name` is then its class name (a node
    object, see `DGNode.ensure_valid`) or the name a plug with no owner was built
    with (see `_raise_deleted`). Every guard raises through it. A node a shader
    conversion replaced (`_CONVERTED`, by the node's `uuid`) names the
    conversion: "'red' was converted to a phong; use the node astype() returned
    (red__rigold already deleted!)"."""
    if freed:
        name = f"{name} node (freed by a new scene, a file open or a reference unload)"
    elif _CONVERTED and uuid is not None:
        converted = _CONVERTED.get(uuid)
        if converted is not None:
            old, node_type = converted
            return RuntimeError(
                f"'{old}' was converted to {_article(node_type)} {node_type}; use the "
                f"node astype() returned ({name} already deleted!)"
            )
    return RuntimeError(f"{name} already deleted!")


def _raise_deleted(attr: Any, handle: Any) -> None:
    """Raise the ``"... already deleted!"`` of `attr`, a plug with no owner whose
    node (`handle`, the handle of it the plug took) was deleted or freed. A
    deleted node is named as `DGNode.ensure_valid` names it, through the node
    of its MPlug (alive; a handle gives no MObject for a deleted node). A freed
    one cannot be read, and the class it would be cast to was never known, so
    it is named by the node part of the name the plug was built with (its str
    buffer)."""
    if handle.isAlive():
        fn = OpenMaya.MFnDependencyNode(attr.__dict__["_mplug"].node())
        raise _deleted_error(fn.name(), uuid=fn.uuid().asString())
    raise _deleted_error(str.__str__(attr).split(".", 1)[0], freed=True)


def _mplug_handle(mplug: OpenMaya.MPlug) -> Any:
    """The handle of the node of `mplug`, a live plug, or None for a null plug:
    the one a plug with no owner keeps (`_handle`, see `_ensure_owner_alive`),
    built from the plug's node while that node is alive.

    A node deleted to the undo queue raises the round-3 ``"<node> already
    deleted!"`` here: a plug of it takes no handle (built by name, it would read
    the node that later took the name, or another one once the deleted node was
    freed)."""
    try:
        if mplug.isNull:
            return None
        mobject = mplug.node()
        handle  = OpenMaya.MObjectHandle(mobject)
    except Exception:
        return None
    if not handle.isValid():
        fn = OpenMaya.MFnDependencyNode(mobject)
        raise _deleted_error(fn.name(), uuid=str(fn.uuid()))
    return handle


def _attr_handle(mplug: OpenMaya.MPlug, extension: bool = False) -> Any:
    """The MObjectHandle of the attribute of `mplug`, a live plug, if it is a
    dynamic attribute, or an `extension` one (the caller knows its class:
    `MPlug.isDynamic` is False for it), else None: the one a plug keeps as
    `_attrh` (see `_ensure_owner_alive`). A dynamic attribute is freed once a
    delete of it leaves the undo queue, while its node lives on, and an extension
    one by `deleteExtension`; a static one lives as long as its node type.

    Every plug takes the handle of its own attribute when it is built, also from
    the MPlug of an attribute already deleted to the undo queue (whose name finds
    no attribute, or another one): once that attribute is freed, the plug raises
    instead of reading it."""
    if not (extension or mplug.isDynamic):
        return None
    try:
        return OpenMaya.MObjectHandle(mplug.attribute())
    except Exception:
        return None


def _attr_state(mplug: OpenMaya.MPlug, handle: Any, attr_handle: Any = None) -> dict:
    """The `__dict__` of a fresh Attribute of `mplug` with no owner, checked
    through `handle` (the handle of its node, or None) and `attr_handle` (the
    handle of its attribute, for a dynamic attr, see `_attr_handle`). A caller
    that binds an owner (`_node`) hands the owner's handle as `handle`. One
    `__dict__` store instead of one `Plug.__setattr__` call per field; a
    subclass property named like one of these keys would be bypassed. The
    caches of queried child attributes are made on first use (see
    `Attribute.__getattr__` and `Attribute.child`)."""
    return {
        "_mplug":                      mplug,
        "_mobject":                    None,
        "_fn_set":                     None,
        "_node":                       None,
        # cache componet type str TODO: make a proper Component class
        "_Attribute__component_type":  None,
        # cache for `_is_geometry_typed_attr`; None = not yet computed
        "_geometry_attr_cache":        None,
        # cache for `_owner_is_polymorphic`; None = not yet computed
        "_polymorphic_owner_cache":    None,
        # cache for `_static_type_key`; _STATIC_KEY_UNSET = not yet computed
        "_static_key_cache":           _STATIC_KEY_UNSET,
        # the handle of the node: the owner's, or the plug's own with no owner
        "_handle":                     handle,
        # the handle of the attribute, for a dynamic attr
        "_attrh":                      attr_handle,
    }


def _new_attr(
    cls: type, mplug: OpenMaya.MPlug, handle: Any = None, attr_handle: Any = None
) -> Any:
    """`cls(mplug)` without the handle of its node `Attribute.__init__` takes of an
    MPlug: for a plug of a node the caller holds, whose owner, or the handle
    `handle` of that node, the caller hands it (see `_inherit_owner`; a caller
    that binds an owner hands the owner's handle), with the handle of its
    attribute (`attr_handle`, see `_attr_handle`). `cls` is `Attribute` or
    `Plug`, whose `__init__` does nothing else for an MPlug. It builds
    `_attr_state`'s state inline: it is the owned plugs' hot path."""
    attr = str.__new__(cls, mplug)
    attr.__dict__.update(
        {
            "_mplug":                      mplug,
            "_mobject":                    None,
            "_fn_set":                     None,
            "_node":                       None,
            "_Attribute__component_type":  None,
            "_geometry_attr_cache":        None,
            "_polymorphic_owner_cache":    None,
            "_static_key_cache":           _STATIC_KEY_UNSET,
            "_handle":                     handle,
            "_attrh":                      attr_handle,
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


_K_TYPED_ATTR = OpenMaya.MFn.kTypedAttribute
_K_ANY_DATA   = OpenMaya.MFnData.kAny


def _is_generic_slot(attr: Any) -> bool:
    """True if ``attr`` (an Attribute) is a plug of a typed attribute that holds any
    data, a *generic slot*: ``addAttr -attributeType typed`` (a ``Typed`` spec),
    ``choice.input`` / ``output``, an objectSet's ``dnSetMembers`` ... (Maya's kAny
    data, `MFnTypedAttribute.attrType`). Its kind is fixed, its data type follows
    what feeds it. Read on the cached attribute MObject (about 1 us, no by-name
    query); False for anything else, and for a plug whose attribute cannot be read
    (its owner check raises elsewhere).

    Keys: a generic slot has no unit and takes numbers, so the curve that keys it is
    an animCurveTU (Maya's ``setKeyframe`` makes one); a curve type chosen from a
    plug maps a generic slot to TU through this test."""
    try:
        mobject = _attr_mobject(attr)
        return (
            mobject.hasFn(_K_TYPED_ATTR)
            and OpenMaya.MFnTypedAttribute(mobject).attrType() == _K_ANY_DATA
        )
    except Exception:  # noqa: BLE001
        return False


def _connected_attrs(
    attr: Any, src: bool = True, dst: bool = True, first_only: bool = False
) -> Any:
    """`attr.get_connected_attrs(src, dst, first_only)` for a caller that reads
    the result at once and keeps none of it (a type query, a disconnect): the
    attrs take no handle of their nodes (see `_mplug_handle`), which only a held
    plug needs (see `_ensure_owner_alive`)."""
    _ensure_owner_alive(attr)
    found = []
    for each in attr._mplug.connectedTo(src, dst):
        found.append(_new_attr(Attribute, each))
        if first_only:
            break
    if first_only:
        return found[0] if found else None
    return found


def _inherit_owner(parent: Any, attr: Any, index: Any = None) -> Any:
    """`attr`, a fresh child, element or parent plug of `parent` (on its node),
    owned by the node object `parent` holds, if any, and named through that
    owner's path (see `_named_through_owner`, which may hand back a copy). It
    takes the handle of that node `parent` holds: its owner's, or with no owner
    the one `parent` took (see `_ensure_owner_alive`). `index` is the logical
    index of an element of `parent` (None for a child): an element, and a child
    of a held element, is held for undo and redo (`_HELD_ELEMENTS`)."""
    d     = attr.__dict__
    pd    = parent.__dict__
    owner = pd["_node"]
    # a child, an element or the parent of a dynamic attr is freed with it
    d["_attrh"]  = pd.get("_attrh")
    d["_handle"] = pd["_handle"]
    path = _HELD_ELEMENTS.get(id(parent)) if _HELD_ELEMENTS else None
    if index is not None or path is not None:
        _hold_below(parent, attr, index, None if path is None else path.path)
    if owner is None:
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
    d  = owner.__dict__
    fn = d.get("_fn_set")
    if not isinstance(fn, OpenMaya.MFnDagNode):
        return False
    return _handle_valid(d) and fn.isInstanced(True)


def _full_name_buffer(attr: Any) -> Any:
    """`attr`, or a copy of it (sharing its state) whose str buffer is its full
    name, if the two differ. For a fresh attr only: the copy stands in for it."""
    name = str(attr)
    if name == str.__str__(attr):
        return attr
    copy = str.__new__(type(attr), name)
    copy.__dict__.update(attr.__dict__)
    return _hold_like(copy, attr)


def _point_count(attr: Any) -> int:
    """The point count of the geometry node of `attr`, a component plug (it bounds
    a component slice). Read through the node's own class: the owner can be a
    class the user chose (``DAGNode(mesh).vtx[0:2]``), which has none."""
    node = type(attr).get_node(attr)
    if not hasattr(type(node), "num_weight_points"):
        node = _cast(attr._mplug.node())
    return node.num_weight_points


def _node_name(node: Any) -> str:
    """The name `Attribute.get_full_name` gives `node`, a plug's owner. Anything but a
    str from the `name` property is the node's Maya attr of that name (a class
    whose `name` property raises), so the node is named by its fn set instead.
    Raises if the node is deleted."""
    name = node.name
    if type(name) is not str:
        name = _fn_set_name(node)
    return name


def _instanced_element_alias(mplug: OpenMaya.MPlug, node: Any, alias: str) -> str:
    """The attr part of `Attribute.get_full_name` for `mplug`, an element whose alias
    (`alias`) has no index: an element of an instanced (world space) attr, or an
    aliased element. cmds resolves an instanced element named without its index
    to the element of the instance the node name's path runs through, so another
    instance's element is named with its index (`MPlug.partialName` with the
    instanced indices), which cmds honours. `node` is the owner, already named."""
    fn = node.__dict__.get("_fn_set")
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
    `attr`'s owner, when the caller has it. Reads the owner's name
    first, so a stale path is re-resolved and a deleted node raises."""
    mplug = attr.__dict__["_mplug"]
    if not mplug.isArray:
        return None
    if node is None:
        node = type(attr).get_node(attr)
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
    str(attr)
    str(other)
    if attr is other:
        return True
    return attr.__dict__["_mplug"].node() == other.__dict__["_mplug"].node() and (
        _plug_identity_name(attr) == _plug_identity_name(other)
    )


def _element_path(attr: Any) -> Any:
    """``(top plug, steps)`` that find Attribute `attr`'s MPlug again from its top
    plug, when it is an element of an array or a plug below one; else None.

    A disconnect can leave such an MPlug dangling: an element MPlug taken while
    the element is connected and holding no value of its own (an element of a
    dynamic array that is not keyable: ``Float('w', multi=True,
    keyable=False)``, ``Matrix``, ``Message``, or after ``<< hide``) points at
    freed memory once rig removes that connection, and every later call of it
    (``node()``, ``logicalIndex()``, a connect) is an access violation in Maya.
    The top plug, taken before the disconnect, stays valid. Taken before the
    disconnect, handed to `_rebind_element` after it. A plug that is not held
    by rig (`attr` has no MPlug) gives None."""
    mplug = attr.__dict__.get("_mplug") if isinstance(attr, Attribute) else None
    if mplug is None:
        return None
    steps, cur, element = [], mplug, False
    while True:
        if cur.isElement:
            steps.append((True, cur.logicalIndex()))
            cur, element = cur.array(), True
        elif cur.isChild:
            steps.append((False, cur.attribute()))
            cur = cur.parent()
        else:
            break
    return (cur, steps[::-1]) if element else None


def _rebind_element(attr: Any, path: Any) -> None:
    """Gives Attribute `attr` its element's MPlug again, found from the top plug
    `_element_path` recorded before a disconnect (None: nothing to do). The rest
    of the statement, and any later one on the same object, then uses the fresh
    MPlug. Only `attr` itself is rebound: another Attribute object held for the
    same element, or a disconnect made outside rig, is the known limit."""
    if path is None:
        return
    cur, steps = path
    for is_element, key in steps:
        cur = cur.elementByLogicalIndex(key) if is_element else cur.child(key)
    attr.__dict__["_mplug"] = cur


# The element plugs rig hands out, found again after an undo or a redo. An
# element MPlug (or one below an element) can dangle once Maya removes the
# element's connection, and Maya's undo and redo remove connections rig does not
# see: ``p = n.w[0]; p << a.tx; cmds.undo(); p << b.tx`` used a freed MPlug (an
# access violation) on a non-keyable dynamic multi or a Typed one. Every
# Attribute at or below an element that holds a handle of its node is kept here
# by a weak reference (`_HeldRef`), keyed by its id, with the path (top
# attribute, steps) that finds its MPlug again; the Undo / Redo event callbacks
# (`_rebind_held_elements`) give each one a fresh MPlug. The top plug is above
# every element, so it never dangles. Nothing is added to the Attribute itself.
# The table lives on ``sys`` (as the callbacks' registry), so a copy of this
# module imported after a purge of the rig modules rebinds the plugs of every
# copy. Building a held element costs a weak reference and a dict entry; a plug
# that is no element costs nothing.
_HELD_ELEMENTS: dict = sys.__dict__.setdefault("_rig_held_elements", {})


class _HeldRef(weakref.ref):
    """A weak reference to a held element Attribute: its key and its path."""

    __slots__ = ("key", "path")


def _drop_held(ref: Any) -> None:
    """The weak reference's callback: its Attribute is gone."""
    held = sys.__dict__.get("_rig_held_elements")
    if held is not None and held.get(ref.key) is ref:
        del held[ref.key]


def _held_path(attr: Any) -> Any:
    """The path `_hold_path` keeps for Attribute `attr`, or None (not held)."""
    ref = _HELD_ELEMENTS.get(id(attr))
    return None if ref is None else ref.path


def _hold_path(attr: Any, path: Any) -> Any:
    """Keep Attribute `attr`, whose MPlug `path` (top attribute, steps) finds,
    among the element plugs an undo or a redo finds again. Returns `attr`."""
    ref      = _HeldRef(attr, _drop_held)
    ref.key  = key = id(attr)
    ref.path = path
    _HELD_ELEMENTS[key] = ref
    if not _HELD_CALLBACKS_READY:
        _ensure_held_callbacks()
    return attr


def _top_path(mplug: OpenMaya.MPlug) -> Any:
    """``(top attribute, steps)`` for `mplug`, a valid MPlug: the steps (element
    indices and child attributes) that find it from the plug of the first
    attribute above it that is no element and below no element (its own
    attribute, steps ``()``, when it is none), and whether a step is an element.
    The path holds attribute MObjects only: an MPlug of the array kept alive
    changes what Maya frees when the node is deleted (an element MPlug of a
    deleted node then dangles), so the top plug is made again when it is used."""
    if not (mplug.isElement or mplug.isChild):
        # a top plug (an array root's, ``mul.input`` of ``mul.input[0]``): no walk
        return (mplug.attribute(), ()), False
    steps, cur, element = [], mplug, False
    while True:
        if cur.isElement:
            steps.append((True, cur.logicalIndex()))
            cur, element = cur.array(), True
        elif cur.isChild:
            steps.append((False, cur.attribute()))
            cur = cur.parent()
        else:
            break
    if not element:
        return (mplug.attribute(), ()), False
    return (cur.attribute(), tuple(steps[::-1])), True


def _hold_if_element(attr: Any) -> Any:
    """`_hold_path` for `attr`, a fresh Attribute that holds a handle of its node,
    when its MPlug is an element or below one. Returns `attr`.

    A plug below no element (``tx``, a compound's child) answers through its
    parents' ``isElement`` / ``isChild`` alone: no path is built (`_top_path`, the
    same walk, runs for an element only). A caller that walked the plug already
    (`DGNode.find_attr`, `_multi_child_walk`) calls this for an element only."""
    d = attr.__dict__
    if d["_handle"] is None:
        return attr
    mplug = d["_mplug"]
    if not mplug.isElement:
        if not mplug.isChild:
            return attr
        # `_top_path`'s walk up to its first element (a plug that is no element goes
        # to its parent there too), without its steps
        cur = mplug.parent()
        while not cur.isElement:
            if not cur.isChild:
                return attr
            cur = cur.parent()
    path, element = _top_path(mplug)
    if element:
        _hold_path(attr, path)
    return attr


def _hold_like(attr: Any, source: Any) -> Any:
    """`attr`, a new Attribute of the plug of Attribute `source` (a copy, or a
    Plug made of it), held as `source` is. Returns `attr`."""
    ref = _HELD_ELEMENTS.get(id(source))
    if ref is not None and attr.__dict__.get("_handle") is not None:
        _hold_path(attr, ref.path)
    return attr


def _hold_below(parent: Any, attr: Any, index: Any, path: Any) -> None:
    """Hold `attr`, a fresh element (``index``, its logical index) or child (index
    None) of Attribute `parent` (held with `path`, or None), when it is an element
    or below one."""
    if attr.__dict__["_handle"] is None:
        return
    if path is None:
        if index is None:
            return
        path, _element = _top_path(parent.__dict__["_mplug"])
    top, steps = path
    step = (True, index) if index is not None else (False, _attr_mobject(attr))
    _hold_path(attr, (top, steps + (step,)))


def _rebind_held_elements(*_args: Any) -> None:
    """The Undo / Redo event callback: give every held element Attribute a fresh
    MPlug, found from its top attribute's plug. One whose node is deleted (to the
    undo queue) or freed, or whose dynamic attribute is, is left: the undo that
    brings it back rebinds it. Never raises (a Maya callback)."""
    held = sys.__dict__.get("_rig_held_elements") or {}
    for ref in list(held.values()):
        attr = ref()
        if attr is None:
            continue
        d = attr.__dict__
        try:
            handle = d.get("_handle")
            if handle is None or not handle.isValid():
                continue
            attrh = d.get("_attrh")
            if attrh is not None and not attrh.isValid():
                continue
            top, steps = ref.path
            cur = OpenMaya.MPlug(handle.object(), top)
            for is_element, key in steps:
                cur = cur.elementByLogicalIndex(key) if is_element else cur.child(key)
            d["_mplug"] = cur
        except Exception:  # noqa: BLE001 -- a callback never raises
            continue


def _held_callback_specs() -> list:
    event = OpenMaya.MEventMessage.addEventCallback
    return [
        (event, "Undo", _rebind_held_elements),
        (event, "Redo", _rebind_held_elements),
    ]


# the registration's owner (the callbacks' registry key): one of its own, beside
# this module's plug-in callbacks
_HELD_OWNER = __name__ + ".held_elements"


def _register_held_callbacks() -> bool:
    """Register the Undo / Redo callbacks in place of an earlier import's (see
    `rig._internal.callbacks`). False, with nothing registered, while Maya is not
    initialised (`_ensure_held_callbacks` makes it at the first held plug)."""
    global _HELD_CALLBACKS_READY
    _HELD_CALLBACKS_READY = _callbacks.register(_HELD_OWNER, _THIS_MODULE, _held_callback_specs())
    return _HELD_CALLBACKS_READY


def _ensure_held_callbacks() -> bool:
    """Make the registration an import before Maya was initialised left pending."""
    global _HELD_CALLBACKS_READY
    _HELD_CALLBACKS_READY = _callbacks.ensure(_HELD_OWNER, _THIS_MODULE, _held_callback_specs())
    return _HELD_CALLBACKS_READY


# `_node_serial`: a number per Maya node for as long as the node lives, keyed by
# its MObjectHandle hashCode: the (handle, MObject, serial) of every live node
# that has the code (the MObject compares the nodes: a handle gives none for a
# node deleted to the undo queue). Freed nodes' entries are dropped when their
# code is seen again, and all of them once the table outgrows
# `_SERIALS_PRUNE_AT`.
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
    Two node objects of one node share it; it is cached on the node object.
    """
    d      = node.__dict__
    serial = d.get("_node_serial")
    if serial is None:
        serial = d["_node_serial"] = _handle_serial(d["_objhandle"], d["_mobject"])
    return serial


def _handle_serial(handle: Any, mobject: Any) -> int:
    """`_node_serial` of the live node of `handle`, its MObjectHandle, and
    `mobject`, its MObject (a plug with no owner hashes through the handle of
    its node it took, and its MPlug's node)."""
    serial  = None
    code    = handle.hashCode()
    entries = _NODE_SERIALS.get(code)
    if entries is None:
        if len(_NODE_SERIALS) >= _SERIALS_PRUNE_AT[0]:
            _prune_node_serials()
        entries = _NODE_SERIALS[code] = []
    else:
        live = []
        for other, known, number in entries:
            # a freed node's handle is not alive: it is dropped, never compared
            if other.isAlive():
                live.append((other, known, number))
                if serial is None and known == mobject:
                    serial = number
        entries[:] = live
    if serial is None:
        serial = _NEXT_SERIAL()
        entries.append((handle, mobject, serial))
    return serial


def _plug_hash(attr: Any) -> int:
    """The hash of the Maya plug `attr` is (see `_same_plug`), for `Plug.__hash__`
    and `Attribute.__hash__` (a typed attr and a Plug of one plug are one key).

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
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        # its attribute was freed (see `_ensure_owner_alive`): not read
        return hash((handle.hashCode(), str.__str__(attr)))
    node = d["_node"]
    if node is None:
        handle = d["_handle"]
        if handle is not None and not handle.isValid():
            if not handle.isAlive():
                return hash((handle.hashCode(), str.__str__(attr)))
            mplug  = d["_mplug"]
            name   = mplug.partialName(False, False, True, False, False, True)
            serial = _handle_serial(handle, mplug.node())
            return hash((serial, name + _deleted_instance_index(mplug)))
        node = type(attr).get_node(attr)
    handle = node.__dict__.get("_objhandle")
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
    """Drops every cached static data type, and starts a new epoch of the plugs'
    kind memos (`_KIND_EPOCH`) (MSceneMessage callback)."""
    _STATIC_DATA_TYPE.clear()
    _KIND_EPOCH[0] = object()


def _hook_node(node: Any) -> Any:
    """The node whose class's fallback hook `node._attr_data_type_fallback` runs
    first, or None if another hook can run first: a hook set on the node instance.
    (A class override or a class-level patch is ruled out by `_keeps_query`.)"""
    try:
        return None if "_attr_data_type_fallback" in vars(node) else node
    except Exception:
        return None


def _vouched_hook(node_cls: type) -> bool:
    """True if the fallback hook of node class `node_cls` is one rig vouches only
    reads the scene (`dg_node._keeps_query`); False for any other, or before
    dg_node has loaded (see `_USER_CODE_RUNS`). Never raises."""
    keeps = _KEEPS_QUERY
    if keeps is None:
        return False
    try:
        return keeps(node_cls)
    except Exception:
        return False


def _queried_data_type(attr: Any, node: Any) -> str | None:
    """The type `Attribute.get_data_type` just queried for `attr` before calling the
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
_HELD_CALLBACKS_READY   = False
_register_held_callbacks()

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
    d      = attr.__dict__
    owner  = d.get("_node")
    handle = d.get("_handle")  # the owner's, or the plug's own
    if owner is not None:
        od = owner.__dict__
        fn = od.get("_fn_set")
        if (
            fn is not None
            and handle is not None
            and handle.isValid()
            and od.get("_mobject") == mplug.node()
        ):
            return fn
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

    The kind of a normal attribute with no array element on the plug's path is
    kept on the plug (`_kind`: the epoch it was computed in, see `_KIND_EPOCH`,
    and the kind), and answers while the plug's node is valid (`_handle`): such
    an attribute is never deleted, renamed or retyped while its node type is
    registered, and a plug's MPlug never changes, so the computation would give
    it again. A deleted or freed node, or a new epoch, computes again. Elements
    (they come and go), dynamic and extension attributes (deleted and re-added
    under their name) are never kept, nor is anything before the plug-in
    callbacks that start a new epoch are registered.
    """
    try:
        d    = attr.__dict__
        memo = d.get("_kind")
        if memo is not None and memo[0] is _KIND_EPOCH[0]:
            handle = d.get("_handle")  # the owner's, or the plug's own
            if handle is not None and handle.isValid():
                return memo[1]
        mplug = attr._mplug
        fn    = _plug_node_fn_set(attr, mplug)
        if fn is None:
            return None
        mobject = _attr_mobject(attr)
        # a deleted attr is no longer on the node, even once its name is reused
        klass = fn.attributeClass(mobject)
        if klass == _INVALID_ATTR:
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
        kind = mobject.apiType()
        if (
            klass == _NORMAL_ATTR
            and not elements
            and (_PLUGIN_CALLBACKS_READY or _ensure_plugin_callbacks())
        ):
            d["_kind"] = (_KIND_EPOCH[0], kind)
        return kind
    except Exception:
        return None


def _naming_dag_path(attr: Any) -> OpenMaya.MDagPath | None:
    """The DAG path whose name `Attribute.get_full_name` gives `attr`'s node: its
    owner's fn set path."""
    return attr.__dict__["_node"]._fn_set.getPath()


def _names_own_plug(attr: Any) -> bool:
    """True if the name `attr` gives itself resolves in cmds to its own MPlug.

    That needs the base naming (a component plug is named after its component),
    the attribute still on its node (a deleted dynamic or extension attr's name
    resolves to nothing, or to a same-named new attr), and an index on every
    array along the path. Call it only once `attr` is named, so its node is valid.

    A DAG node is named by its owner's DAG path, which must still be valid (the
    instance it runs through can be deleted while the node lives on), and an
    instanced element is named without its index, after that path's instance.

    A plug of a valid DG node that keeps a kind memo of this epoch (see
    `_fixed_attr_kind`: a normal attribute, so still on its node, with no array
    element on its path) is answered from its parents alone, as the loop below
    answers it: only a parent that is an array root fails it.
    """
    # the class keeps Attribute's naming (``__str__``, ``get_full_name``,
    # ``get_alias``: Attribute's, patched or not), read from the class dicts
    if not isinstance(attr, Attribute) or _redefines(type(attr), _NAMING_MEMBERS):
        return False
    # a freed attribute (see `_ensure_owner_alive`) is not read
    d      = attr.__dict__
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        return False
    try:
        mplug   = attr._mplug
        mobject = mplug.node()
        dag     = mobject.hasFn(OpenMaya.MFn.kDagNode)
        memo    = d.get("_kind")
        if not dag and memo is not None and memo[0] is _KIND_EPOCH[0]:
            handle = d.get("_handle")  # the owner's, or the plug's own
            if handle is not None and handle.isValid():
                plug = mplug
                while plug.isChild:
                    plug = plug.parent()
                    if plug.isArray:
                        return False
                return True
        fn      = OpenMaya.MFnDependencyNode(mobject)
        invalid = OpenMaya.MFnDependencyNode.kInvalidAttr
        if fn.attributeClass(mplug.attribute()) == invalid:
            return False
        # the owner's path, valid once the attr was named: a stale path (its
        # instance removed while the node lives on in another) is re-resolved by
        # DAGNode.name; an invalid one would name the node "". Kept as a guard.
        path = None
        if dag:
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


# --- enum field names (see `Attribute.get_enums`)


def _is_text(value: Any) -> bool:
    """True if `value` is a plain str: a str that is not a plug (an `Attribute`
    is a str too, and names a connection)."""
    return isinstance(value, str) and not isinstance(value, Attribute)


def _is_enum_attr(attr: Any) -> bool:
    """True if the Maya attribute of `attr` (an `Attribute`) is an enum.
    Raises ``"... already deleted!"`` as `Attribute.get_mobject` does."""
    _ensure_owner_alive(attr)
    return _attr_mobject(attr).hasFn(OpenMaya.MFn.kEnumAttribute)


# the widest value range `_enum_fields` scans value by value (a value with no
# field costs about 2 us); a wider one is read from the attribute's addAttr
# command instead (a sparse ``en="a=-32000:b=32000"`` would take about 0.1 s)
_ENUM_SCAN_SPAN = 1024

# the ``-enumName "..."`` flag of `MFnAttribute.getAddAttrCmd`
_ENUM_NAME_FLAG = re.compile(r'-enumName "((?:[^"\\]|\\.)*)"')

# a str that names a plug (``"t1.tx"``, ``"|grp|t1.tx"``, ``"ns:t1.tx"``),
# not a number (``"4.0"``)
_PLUG_NAME_LIKE = re.compile(r"[A-Za-z_|:][^.]*\.[A-Za-z_]")


def _enum_fields(fn: OpenMaya.MFnEnumAttribute) -> list[tuple[str, int]]:
    """The ``(field name, value)`` pairs of the enum `fn`, by value. The values
    can be sparse (``en="a=5:b=10"``), and a value with no field has no pair."""
    low, high = fn.getMin(), fn.getMax()
    if high - low > _ENUM_SCAN_SPAN:
        declared = _declared_enum_fields(fn)
        if declared is not None:
            return declared
    fields = []
    for value in range(low, high + 1):
        try:
            fields.append((fn.fieldName(value), value))
        except RuntimeError:
            pass
    return fields


def _parse_enum_names(names: str) -> list[tuple[str, int]]:
    """The ``(field name, value)`` pairs an ``-enumName`` string declares, in
    order (``"a:b=5:c"`` is a=0, b=5, c=6; an empty piece, as a trailing ``:``
    leaves, declares nothing)."""
    fields = []
    value  = 0
    for piece in names.split(":"):
        name, sep, given = piece.rpartition("=")
        if sep and given.lstrip("-").isdigit():
            value = int(given)
        else:
            name = piece
        if name:
            fields.append((name, value))
        value += 1
    return fields


def _declared_enum_fields(fn: OpenMaya.MFnEnumAttribute) -> list[tuple[str, int]] | None:
    """The fields of the enum `fn` as its ``-enumName`` flag declares them
    (``"a:b=5:c"`` is a=0, b=5, c=6), each pair kept only if `fn` gives that
    name for that value; None when the flag is missing."""
    found = _ENUM_NAME_FLAG.search(fn.getAddAttrCmd(True))
    if found is None:
        return None
    fields = []
    for name, value in _parse_enum_names(re.sub(r"\\(.)", r"\1", found.group(1))):
        try:
            if fn.fieldName(value) == name:
                fields.append((name, value))
        except RuntimeError:
            pass
    return sorted(fields, key=lambda field: field[1])


# a '-' between two word characters ("Greater-Than"): a separator the loose
# match drops. A leading or lone '-' ("-X") is a sign and stays.
_INNER_DASH = re.compile(r"(?<=\w)-(?=\w)")


def _field_key(name: str) -> str:
    """`name` compared loosely: casefolded, without spaces, ``_`` or a ``-``
    between two word characters (``"Greater-Than"`` is ``"greaterthan"``; the
    sign of ``"-x"`` stays, so ``"-x"`` is never ``"x"``)."""
    return re.sub(r"[\s_]", "", _INNER_DASH.sub("", name)).casefold()


def _match_enum_field(
    fields: list[tuple[str, int]], name: str, where: str, exact: bool = True
) -> int:
    """The value of the field `name` among `fields` (``(field name, value)``
    pairs), in three tiers, each tried only when the one before finds nothing:

    1. the exact name (skipped when `exact` is False: the caller tried it);
    2. the name with its outer spaces and its case ignored on both sides
       (``"x"`` for ``"X"``, ``"-X"`` for Maya's ``" -X"``);
    3. the loose key of `_field_key` on both sides (``"greater_than"`` for
       ``"Greater Than"``, ``"hasnoeffect"``).

    Two or more fields found by a tier, or none by any, raise TypeError naming
    `where` (the attribute, as the caller calls it) and every field with its
    value, with a hint to write ``Plug("a.b")`` when the name looks like a plug
    to connect. An empty or blank name is never a field."""
    found = []
    if exact:
        found = [value for field, value in fields if field == name]
    if not found:
        folded = name.strip().casefold()
        if folded:
            found = [value for field, value in fields if field.strip().casefold() == folded]
    if not found:
        key = _field_key(name)
        if key:
            found = [value for field, value in fields if _field_key(field) == key]
    if len(found) == 1:
        return found[0]
    listed = ", ".join(f"{field}={value}" for field, value in fields) or "none"
    if found:
        message = (
            f"{where}: {name!r} matches {len(found)} of its enum fields once case, "
            f"spaces, '_' and a '-' between letters are ignored; write one of them "
            f"exactly, or its int: {listed}"
        )
    else:
        message = (
            f"{where}: {name!r} is not one of its enum fields; write a field name "
            f"or its int: {listed}"
        )
    if _PLUG_NAME_LIKE.match(name):
        message += f"; to connect, write Plug({name!r})"
    raise TypeError(message)


def _enum_value(
    attr: OpenMaya.MObject | OpenMaya.MPlug, name: str, where: str
) -> int:
    """The value of the field `name` of the enum attribute `attr` (its attribute
    MObject, or an MPlug of it), for a field name set on an enum.

    An exact field name is found first (``"zxy"``, Maya's ``"Multiply"``);
    otherwise the one field that matches with case and outer spaces ignored
    (``"x"`` for ``"X"``, ``"-X"`` for ``" -X"``), then the one that matches
    once case, spaces, ``_`` and a ``-`` between letters are ignored on both
    sides (``"multiply"``, ``"greater_than"`` for ``"Greater Than"``,
    ``"hasnoeffect"``); a leading ``-`` is a sign (``"-x"`` is never ``"X"``).
    Anything else, an empty name included, raises TypeError before any edit
    (see `_match_enum_field`).
    """
    if isinstance(attr, OpenMaya.MPlug):
        attr = attr.attribute()
    fn = OpenMaya.MFnEnumAttribute(attr)
    try:
        value = fn.fieldValue(name)
        # Maya answers some names that are no field (fieldValue("") is the
        # first value): a hit counts only when it names that value back
        if fn.fieldName(value) == name:
            return value
    except (RuntimeError, TypeError, ValueError):
        pass
    return _match_enum_field(_enum_fields(fn), name, where, exact=False)


def _is_value_sequence(value: Any) -> bool:
    """True for a list, tuple or ndarray of values (never a str)."""
    return isinstance(value, (list, tuple, np.ndarray))


def _holds_text(value: Any) -> bool:
    """True if `value` is a plain str, or a list / tuple / ndarray holding one
    at any depth (a numeric ndarray is answered by its dtype)."""
    if isinstance(value, str):
        return not isinstance(value, Attribute)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return True
        return value.dtype == object and any(_holds_text(item) for item in value.ravel())
    if isinstance(value, (list, tuple)):
        return any(_holds_text(item) for item in value)
    return False


def _check_enum_names(
    attr: OpenMaya.MObject, value: Any, where: str, element: bool = False
) -> None:
    """Raises TypeError, before any edit, when `value` set on the attribute
    `attr` (its MObject; `element`: a plug of one element of it, when it is an
    array) would give a plain str that is no field name to an enum, the way a
    set of `value` reaches the leaves:

    * an array root: a str is the auto-appended element's, a sequence one
      value per element;
    * a compound: a str goes to every child, a sequence one value per child
      (extra values are dropped, as the fan-out drops them);
    * an enum leaf: a str, or the one str of a one-element sequence, is read
      with `_enum_value`;
    * a numeric leaf: such a str raises the InjectionError its set would raise,
      but before the leaves ahead of it are set (``t.t << [5, "abc", 7]`` sets
      nothing).

    It only checks: the set itself reads each name again."""
    fn = OpenMaya.MFnAttribute(attr)
    if fn.array and not element:
        if _is_value_sequence(value):
            for i, item in enumerate(value):
                _check_enum_names(attr, item, f"{where}[{i}]", True)
        else:
            _check_enum_names(attr, value, where, True)
        return
    if attr.hasFn(OpenMaya.MFn.kCompoundAttribute):
        compound = OpenMaya.MFnCompoundAttribute(attr)
        children = [compound.child(i) for i in range(compound.numChildren())]
        if _is_value_sequence(value):
            pairs = zip(children, value)
        else:
            pairs = ((child, value) for child in children)
        for child, item in pairs:
            _check_enum_names(child, item, f"{where}.{OpenMaya.MFnAttribute(child).name}")
        return
    if _is_value_sequence(value) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.str_):
        value = str(value)
    if not _is_text(value):
        return
    if attr.hasFn(OpenMaya.MFn.kEnumAttribute):
        _enum_value(attr, value, where)
    elif attr.hasFn(OpenMaya.MFn.kNumericAttribute) or attr.hasFn(OpenMaya.MFn.kUnitAttribute):
        from rig._internal.plug import InjectionError  # plug imports this module

        raise InjectionError(
            f"Cannot set {where!r}: a numeric attribute does not accept data of type "
            f"'string' ({value!r}); nothing was set"
        )


def _check_attrs(attrs: dict, *, node_type: str | None = None, node: str | None = None) -> None:
    """Every key of `attrs` must name an attribute of the node (by type,
    `node_type`, before a create; on the node `node` before an update), so a
    typo raises AttributeError before any write. A field name given to an enum
    attribute (``displayType="reference"``) is read the same way, so a wrong
    one raises TypeError naming the fields before the node is made (see
    `_check_enum_names`). Used by a typed create's attribute keywords and the
    membership collections (``rig._internal.members`` re-exports it)."""
    for attr, value in attrs.items():
        query = {"type": node_type} if node is None else {"node": node}
        if not cmds.attributeQuery(attr, exists=True, **query):
            where = f"a {node_type}" if node is None else f"'{node}'"
            raise AttributeError(f"{where} has no attribute '{attr}'")
        if _holds_text(value):
            attribute = _attribute_of(attr, node_type=node_type, node=node)
            if attribute is not None:
                _check_enum_names(attribute, value, f"{node or node_type}.{attr}")


def _attribute_of(attr: str, *, node_type: str | None, node: str | None):
    """The attribute MObject `attr` names on `node` (or, before a create, on
    the type `node_type`), or None when the API cannot find it."""
    try:
        if node is None:
            found = OpenMaya.MNodeClass(node_type).attribute(attr)
        else:
            sel = OpenMaya.MSelectionList()
            sel.add(node)
            found = OpenMaya.MFnDependencyNode(sel.getDependNode(0)).attribute(attr)
    except (RuntimeError, TypeError, ValueError):
        return None
    return None if found.isNull() else found


@total_ordering
class Attribute(str):
    """
    The attribute class that wraps OpenMaya.MPlug.

    Note: This class inherits `str` so that it can be passed into maya.cmds calls
    that expect strings. It's only necessary since "__getitem__()" is implemented.
    Maybe there is a better way to do this.

    """

    # the class of the elements `element_by_*` builds (D29): a Plug's elements
    # are Plugs, built once. The child caches (`child`, child by name) hold
    # Attributes, which a Plug's lookups turn into new Plugs (plugs are never
    # cached, D30), and `get_parent` returns an Attribute
    _CHILD_CLASS = None  # set to Attribute below the class

    def __init__(self, name_or_mplug: str | OpenMaya.MPlug) -> None:
        """Initialize an instance from an attribute full name or a MPlug.

        It has no owner yet, so it takes a handle of its node, which tells it
        once that node is deleted or freed (see `_ensure_owner_alive`). An MPlug
        of a node already deleted to the undo queue raises its ``"... already
        deleted!"`` (see `_mplug_handle`).
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
        self.__dict__.update(_attr_state(mplug, handle, _attr_handle(mplug)))
        _hold_if_element(self)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # a subclass's public functions are plug methods too (`_ChildOrMethod`)
        super().__init_subclass__(**kwargs)
        _wrap_plug_methods(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        # assignment is strict (user decisions 2026-10-16): a method is never
        # replaced on one plug, a bare Attribute has no ``=`` sugar, and a plug
        # has no settable members (the property ban: ``set_alias(x)``,
        # ``set_keyable(b)``, ``lock()``); a ``_`` name is its state
        if name[:1] != "_":
            hint = _OLD_PLUG_SETTERS.get(name)
            if hint is None and name in _OLD_PLUG_NAMES:
                hint = f"plug.{name} is now {_OLD_PLUG_NAMES[name]}"
            what = (
                "is a method: a method is not replaced on an attribute, and"
                if isinstance(_class_attr(type(self), name), _ChildOrMethod)
                else "cannot be set:"
            )
            raise AttributeError(
                f"{type(self).__name__}.{name} {what} a bare Attribute has no '=' "
                f"sugar (Plug({str.__str__(self)!r}).{name} << value sets the plug "
                f"of that name)" + (f"; {hint}" if hint else "")
            )
        object.__setattr__(self, name, value)

    # --- dunders

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{str(self)}")'

    def __str__(self) -> str:
        # the name `get_full_name` gives (a subclass's own, ComponentPlug's)
        return type(self).get_full_name(self)

    def __hash__(self) -> int:
        # the Maya plug's identity, not the name: one key through every instance
        # path, kept across a rename (see `_plug_hash`). A typed attr and a DSL
        # Plug of one plug compare equal (`Plug.__eq__` folds them to True, X1),
        # so they hash alike and are one dict / set key; under `force_nodes()` /
        # `constant_folding=False` a lookup that mixes them builds the equal node,
        # as a lookup through two Plugs of the plug does
        return _plug_hash(self)

    def __eq__(self, other: Any) -> bool:
        """True if `other` is an Attribute of the same Maya plug (node, attribute
        and logical indices), whatever the instance path each is named through.
        A str is never equal: compare `str(attr)` for names."""
        return isinstance(other, Attribute) and _same_plug(self, other)

    def __ne__(self, other: Any) -> bool:
        # not `str.__ne__`, which compares the names the two were built with
        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        return str(self) > str(other)

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
            return type(self).element_by_logical_index(self, int(key))

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
                attribute_list.append(type(self).element_by_logical_index(self, index))
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
        # the child caches are made on first use, so they are read from
        # `__dict__`: reading an absent one as an attribute would come back here
        d     = self.__dict__
        names = d.get("_Attribute__child_name_dict")
        if not names:
            if "_mplug" not in d:
                # a half-built attribute (``copy``) has no state to read
                raise AttributeError(attr_name)
            _ensure_owner_alive(self)
            mplug = d["_mplug"]
            if not mplug.isCompound:
                # no child to look up: AttributeError, so ``hasattr`` answers
                # (named by the str it was built with: no node read, which a
                # deleted node's plug raises from)
                raise AttributeError(
                    f"{str.__str__(self)} is not a compound: it has no child "
                    f"{attr_name!r}" + _old_name_hint(attr_name)
                )
            count = mplug.numChildren()
            names = d.setdefault("_Attribute__child_name_dict", {})
            ids   = d.setdefault("_Attribute__child_id_dict", {})
            for i in range(count):
                child_plug = self._mplug.child(i)
                name = child_plug.partialName(False, False, False, False, False, True)
                name = name.rsplit(".", 1)[-1]
                attr = _inherit_owner(self, _new_attr(Attribute, child_plug))
                names[name] = attr
                ids[i]      = attr
        if attr_name in names:
            return names[attr_name]
        raise AttributeError(f"{str(self)}.{attr_name} not found" + _old_name_hint(attr_name))

    # --- queries: every public member is a method (the property ban, user
    # decision 2026-10-16). Bare, a name is the compound child of that name when
    # the plug has one, else the method; with brackets it is the method (see
    # `_ChildOrMethod`). rig's own code reads them through the class
    # (``type(plug).get_node(plug)``) or private state (``plug._mplug``).

    def get_mplug(self) -> OpenMaya.MPlug:
        """Returns the OpenMaya.MPlug this attribute wraps."""
        return self._mplug

    def get_fn_set(self) -> OpenMaya.MFnBase:
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

    def get_mobject(self) -> OpenMaya.MObject:
        """Returns the mobject. Raises ``"... already deleted!"`` once the node
        was freed: the attribute of a dynamic attr is freed with it, a cached
        one too (see `_ensure_owner_alive`)."""
        _ensure_owner_alive(self)
        return _attr_mobject(self)

    def get_node(self) -> Any:
        """Returns the node object of this attr: the node object it was read from
        (``node.tx.get_node() is node``; children and elements share it). An attr built
        from a name or an MPlug casts its node on first use (the typed cast of its
        MPlug's node), and raises ``already deleted!`` if that node was deleted
        or freed since it was built (see `_ensure_node_castable`)."""
        d    = self.__dict__
        node = d["_node"]
        if node is None:
            _ensure_node_castable(self)
            # through `__setattr__`, as a Plug's other post-init state writes
            node = self._node = _cast(d["_mplug"].node())
            # the owner's handle from now on (see `_ensure_owner_alive`)
            d["_handle"] = node.__dict__.get("_objhandle")
        return node

    def get_name(self) -> str:
        """Returns the attribute name (without node name). Raises ``"... already
        deleted!"`` once the node was freed (see `_ensure_owner_alive`): the name
        of a dynamic attr is read from its attribute, which is freed with it."""
        _ensure_owner_alive(self)
        return self._mplug.partialName(False, False, False, False, False, True)

    def get_full_name(self) -> str:
        """Returns the full attribute name (with node name), use alias if exists.

        Note. Can't use MPlug.name() directly because it doesn't use the partial node name.
        i.e. it will error if duplicated node names exist.

        An element of an instanced (world space) attr, such as ``worldMatrix[1]``,
        is named without its index when it is the element of the instance the
        node's path runs through (``T2|S.worldMatrix``), and with it otherwise
        (``T2|S.worldMatrix[0]``), so cmds resolves the name to this element.
        """
        node = self.__dict__["_node"]
        if node is None:
            node = type(self).get_node(self)
        name = node.name
        # anything but a str is the node's Maya attr of that name (a class whose
        # `name` property raises), so the node is named by its fn set instead
        if type(name) is not str:
            name = _fn_set_name(node)
        # `alias`, read once the name proved the node is alive, and the attribute
        # of a dynamic attr (see `_ensure_owner_alive`)
        d      = self.__dict__
        mplug  = d["_mplug"]
        handle = d.get("_attrh")
        if handle is not None and not handle.isAlive():
            raise _deleted_error(str.__str__(self))
        alias = mplug.partialName(False, False, False, True, False, True)
        if mplug.isElement and alias[-1:] != "]":
            alias = _instanced_element_alias(mplug, node, alias)
        return f"{name}.{alias}"

    def get_alias(self) -> str:
        """Returns the attribute alias, or the attribute name if no alias applied."""
        _ensure_owner_alive(self)
        return self._mplug.partialName(False, False, False, True, False, True)

    def set_alias(self, alias: str | None) -> None:
        """Sets the alias of this attribute (``aliasAttr``); None or ``""``
        removes it. The same alias, or no alias to remove, is left as it is
        (no command)."""
        cur_alias = type(self).get_alias(self)
        if cur_alias == alias:
            return
        if not alias:
            if cur_alias != type(self).get_name(self):
                cmds_edit.aliasAttr(str(self), remove=True)
            return
        cmds_edit.aliasAttr(alias, str(self))

    def get_attribute_type(self) -> str:
        """Returns the type of this attribute."""
        return cmds.attributeQuery(type(self).get_name(self), node=type(self).get_node(self).name, attributeType=True)

    def get_data_type(self) -> str:
        """Returns the data type of the value hosted by this attribute."""
        global _FALLBACK_QUERY
        owner = _dg_named_owner(self)
        if owner is None:
            # the name comes first so a deleted node's plug raises as before
            full_name = str(self)
        else:
            # naming the attr would raise only from these checks (see
            # `_dg_named_owner`), so the name is built on a miss only
            _ensure_dg_named(self, owner)
            full_name = None
        key = self._static_type_key()
        if key is not None:
            typ = _STATIC_DATA_TYPE.get(key)
            if typ is not None:
                return typ
        if full_name is None:
            full_name = str(self)

        typ = cmds.getAttr(full_name, type=True)

        # maintain consistent type string with cmds.addAttr()
        if typ == "TdataCompound":
            typ = "compound"

        # if cmds.getAttr fails to resolve, call the node fallback hook
        # e.g. if a choice node's inputs are message attrs, its output will be resolved
        # to "typed" rather than "message"
        elif typ in ("typed", "Tdata"):
            # the hook can reuse this answer instead of querying it again, if no
            # hook set on the node instance runs before its class hook (a class
            # override or patch is ruled out by the hook, see `_keeps_query`)
            outer = _FALLBACK_QUERY
            try:
                node            = type(self).get_node(self)
                hook_node       = _hook_node(node)
                _FALLBACK_QUERY = (self, typ, hook_node)
                if hook_node is None or not _vouched_hook(type(node)):
                    # code a user may have written: outside the journal statement
                    _USER_CODE_RUNS[0] += 1
                    with outside():
                        return node._attr_data_type_fallback(self)
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

    def is_dynamic(self) -> bool:
        """Attribute dynamic state."""
        _ensure_owner_alive(self)
        return self._mplug.isDynamic

    def is_locked(self) -> bool:
        """True when the plug is locked: its own lock, or a locked parent's (a
        compound's for its children, a multi root's for its elements)."""
        _ensure_owner_alive(self)
        return self._mplug.isLocked

    def lock(self) -> "Attribute":
        """Locks this plug (``setAttr -lock on``) and returns it: ``plug.lock()``
        is ``plug << lock``. A locked plug (its own lock or a locked parent's)
        is left as it is: no command, no undo step.

        Refused (InjectionError, nothing changed) on a locked node, whose plugs
        Maya never locks: lock the plugs first, then the node. Values and
        connections into a locked plug raise (``<<``) until it is unlocked."""
        _ensure_owner_alive(self)
        if not self._mplug.isLocked:
            why = _plug_lock_refusal(self, True)
            if why:
                raise _injection_error(why)
            cmds_edit.setAttr(str(self), lock=True)
        return self

    def unlock(self) -> "Attribute":
        """Unlocks this plug (``setAttr -lock off``) and returns it:
        ``plug.unlock()`` is ``plug << unlock``. An unlocked plug is left as it
        is: no command, no undo step.

        Refused (InjectionError, nothing changed) on a locked node (unlock the
        node first), and when a locked compound parent holds the lock (unlock
        the parent)."""
        _ensure_owner_alive(self)
        if self._mplug.isLocked:
            why = _plug_lock_refusal(self, False)
            if why:
                raise _injection_error(why)
            cmds_edit.setAttr(str(self), lock=False)
        return self

    def is_keyable(self) -> bool:
        """Attribute keyable state."""
        _ensure_owner_alive(self)
        return self._mplug.isKeyable

    def set_keyable(self, state: bool) -> None:
        """Sets the keyable state (``setAttr -keyable``)."""
        cmds_edit.setAttr(str(self), keyable=state)

    def is_channel_box(self) -> bool:
        """Attribute channel box state."""
        _ensure_owner_alive(self)
        return self._mplug.isChannelBox

    def set_channel_box(self, state: bool) -> None:
        """Sets the channel box state (``setAttr -channelBox``)."""
        cmds_edit.setAttr(str(self), channelBox=state)

    # --- category methods

    def get_categories(self) -> list[str]:
        """Returns a list of categories this attribute belongs to."""
        return (
            cmds.attributeQuery(type(self).get_name(self), node=type(self).get_node(self).name, categories=True) or []
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
            cmds_edit.addAttr(str(self), edit=True, category=each)

    # --- connection methods

    def is_connected(self) -> bool:
        """Attribute connected state. Both input and output connections counts."""
        _ensure_owner_alive(self)
        return self._mplug.isConnected

    def is_free_to_change(self) -> bool:
        """Attribute free_to_change state. True if this attr and all its parents are
        free to change."""
        _ensure_owner_alive(self)
        return self._mplug.isFreeToChange(True, False) == OpenMaya.MPlug.kFreeToChange

    def connect(self, other: str | Attribute, force: bool = False) -> None:
        """Connects this attr to other.

        Args:
            other: The attribute to connect to this attr.
            force: If True, break existing connection if found.

        A connect into a free destination is made through the API and journaled
        (`rig._internal.api_edits.api_connect`: one undo step per statement); any
        other takes the cmds route below, which raises its errors.
        """
        if _api_connect(self, other):
            return
        dst = str(other)
        src = str(self)
        # cmds.isConnected is True only for a direct src -> dst connection, so it
        # is False for a dst plug that is not a destination; a name that may not
        # resolve to its plug keeps the query, and so its errors
        if (
            _names_own_plug(other)
            and not other._mplug.isDestination
            and _names_own_plug(self)
        ) or not cmds.isConnected(src, dst):
            cmds_edit.connectAttr(src, dst, force=force)

    def disconnect(self, other: str | Attribute) -> None:
        """Disconnects this attr from other.

        Args:
            other: The attribute to disconnect from this attr.

        An Attribute `other` that is an array element (or below one) gets its
        element's MPlug again after the disconnect (see `_element_path`), so it
        stays usable.
        """
        _self = str(self)
        name  = str(other)
        if cmds.isConnected(_self, name):
            path = _element_path(other)
            cmds_edit.disconnectAttr(_self, name)
            _rebind_element(other, path)

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
        for each in self._mplug.connectedTo(src, dst):
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
        for each in cmds.listConnections(str(self), **kwargs) or []:
            obj = casted.get(each)
            if not obj:
                obj          = _cast(each)
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
            _processed = {_cast(x) for x in default_excludes}
        if exclude_nodes:
            for x in exclude_nodes:
                if cmds.objExists(x):
                    _processed.add(_cast(x))

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
            c = _cast(c)
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
                cmds_edit.delete(to_delete)

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

        The destination of each disconnect (this attr, or an attr it drove)
        that is an array element (or below one) gets its element's MPlug again
        after it (see `_element_path`), so this attr and the attrs returned
        stay usable.
        """
        attrs = self.get_connected_attrs(src=src, dst=dst)
        for attr in attrs:
            attr_name = str(attr)
            if cmds.isConnected(str(self), attr_name):
                src_name = str(self)
                dst_name = attr_name
                dst_attr = attr
            else:
                src_name = attr_name
                dst_name = str(self)
                dst_attr = self
            path = _element_path(dst_attr)
            cmds_edit.disconnectAttr(src_name, dst_name)
            _rebind_element(dst_attr, path)
        return attrs

    def __rshift__(self, other):
        """Right shift operator ">>" to force connect."""
        self.connect(other, force=True)

    def __floordiv__(self, other):
        """Floor divide operator "//" to disconnect."""
        self.disconnect(other)

    # --- value methods

    def get_default_value(self) -> Any:
        """Returns the default value of this attr, if any."""
        val = cmds.attributeQuery(type(self).get_name(self), node=type(self).get_node(self), listDefault=True)
        if val is not None and len(val) == 1:
            return val[0]
        return val

    def set_default_value(self, value: Any) -> None:
        """Sets the default value of this dynamic attribute (``addAttr -edit
        -defaultValue``); a native attribute's raises RuntimeError."""
        if not type(self).is_dynamic(self):
            raise RuntimeError(f"Cannot set default value for native attr: {self}")
        cmds_edit.addAttr(str(self), edit=True, defaultValue=value)

    def get_data_fn_set(self) -> OpenMaya.MFnData | None:
        """Returns the proper data function set for this attr, or None if
        this attribute holds no data, or plug.asMObject() causes internal failure."""
        _ensure_owner_alive(self)
        data_mobject = self._mplug.asMDataHandle().data()
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
            - an enum attr takes a field name as well as its int:
              ``node.find_attr("ro").set("zyx")`` sets 5. The exact field name is
              found first, then the one field that matches with case and outer
              spaces ignored, then once case, spaces, ``_`` and a ``-`` between
              letters are ignored (``"multiply"`` for Maya's ``"Multiply"``; a
              leading ``-`` is a sign); any other str, an empty one included,
              raises TypeError naming the fields, and the value is left as it
              was (see `_enum_value`).
            - a typed attr that holds any data (a ``Typed`` spec's, ``choice.input``,
              `_is_generic_slot`) takes one number as numeric data, with no `type`
              flag; Maya's setAttr refuses any other value there.

        One int, float or bool value (no flag) into a free numeric, unit, enum or bool
        attr is set through the API and journaled when cmds.setAttr() would set the same
        value (`rig._internal.api_edits.api_set`: one undo step per statement); any other
        set takes cmds.setAttr(), which raises its errors.

        Args:
            args, kwargs: args supported by cmds.setAttr()
        """
        _ensure_owner_alive(self)
        # a field name given to an enum (a plain str, never a plug)
        if (
            len(args) == 1
            and isinstance(args[0], str)
            and not isinstance(args[0], Attribute)
            and "type" not in kwargs
            and _attr_mobject(self).hasFn(OpenMaya.MFn.kEnumAttribute)
        ):
            args = (_enum_value(_attr_mobject(self), args[0], str(self)),)
        # one numeric value into a plug of a fixed kind outside any array (today's route
        # makes no type query there): through the API and journaled when its guards hold
        # (`rig._internal.api_edits.api_set`), else the cmds route below
        fixed = None
        if len(args) == 1 and not kwargs and _journal.API_EDITS:
            fixed = self._is_fixed_kind_outside_array()
            if fixed and _api_set(self, args[0]):
                return
        if "type" not in kwargs and not (
            fixed if fixed is not None else self._is_fixed_kind_outside_array()
        ):
            # typed attr requires the `type` arg to be specified.
            # the only weird one-off is `fltMatrix`, which is not typed but still
            # requires the `type` arg. A generic slot (`_is_generic_slot`) takes one
            # number as numeric data, which a type flag (its data type, "Tdata"
            # when it is empty) refuses: no flag, and no type query.
            if not (len(args) == 1 and isinstance(args[0], Real) and _is_generic_slot(self)):
                typ = type(self).get_data_type(self)
                if type(self).is_typed(self) or typ == "matrix":
                    kwargs["type"] = typ

        # special handling for counted array attrs
        # - cmds.setAttr() wants the item count first, then the unpacked items
        if (
            len(args) == 1
            and isinstance(args[0], (list, tuple, np.ndarray))
            and type(self).get_data_type(self) in COUNTED_ARRAY_TYPES
        ):
            cmds_edit.setAttr(str(self), len(args[0]), *args[0], **kwargs)

        # special handling for compound attrs
        # - use the unpacked first argument if it is a valid iterable
        #   and attr type ends with a digit (double2, double3, etc.)
        elif (
            len(args) == 1
            and isinstance(args[0], (list, tuple, np.ndarray))
            and str(type(self).get_data_type(self))[-1].isdigit()
        ):
            cmds_edit.setAttr(str(self), *args[0], **kwargs)
        else:
            cmds_edit.setAttr(str(self), *args, **kwargs)

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

        return cmds.getAttr(str(self))

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
            typ = cmds.getAttr(str(self), type=True)
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
                type_name = OpenMaya.MFnDependencyNode(self._mplug.node()).typeName
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
        owner_obj = self._mplug.node()
        is_shape  = owner_obj.hasFn(OpenMaya.MFn.kShape)

        # Eagerly fetch the data once -- used both for the shape-match check
        # below and for the final MFn wrap if we fall through.
        try:
            data = self._mplug.asMObject()
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
            return _cast(owner_obj).serialize()

        # 2. Prefer a real downstream node consumer.
        dests = self._mplug.destinations()
        if dests:
            return _cast(dests[0].node()).serialize()

        # 2.5. Known geometry-routing nodes (e.g. choice node): walk upstream via
        #      the registered tracer to find the source shape.
        if expected_shape_fn:
            owner_type = OpenMaya.MFnDependencyNode(owner_obj).typeName
            tracer     = GEOMETRY_ROUTING_TRACERS.get(owner_type)
            if tracer is not None:
                src_plug = tracer(self._mplug)
                if src_plug is not None and src_plug.node().hasFn(expected_shape_fn):
                    return _cast(src_plug.node()).serialize()

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

    def get_enums(self) -> list[str]:
        """Returns a list of enum values if this attr is an enum attr.

        Each field is written as `cmds.attributeQuery(listEnum=True)` writes it
        (``"b=10"`` once the values are sparse). A field name set on an enum
        attr (``plug << "zxy"``, `set`) is read by `_enum_value`."""
        enums = cmds.attributeQuery(type(self).get_name(self), node=type(self).get_node(self), listEnum=True)
        if enums:
            return enums[0].split(":")
        return []

    # --- compound attr methods

    def get_num_children(self) -> int:
        """Returns this number of children of this compound attr."""
        _ensure_owner_alive(self)
        return self._mplug.numChildren()

    def get_parent(self) -> Attribute | None:
        """Returns the parent attribute, if any."""
        _ensure_owner_alive(self)
        if not self._mplug.isChild:
            return None
        plug = self._mplug.parent()
        if plug and not plug.attribute().isNull():
            return _inherit_owner(self, _new_attr(Attribute, plug))

    def child(self, i: int) -> Attribute:
        """Returns the child attribute at the given index."""
        # made on first use, read from `__dict__` (see `__getattr__`)
        d   = self.__dict__
        ids = d.get("_Attribute__child_id_dict")
        if ids is None:
            ids = d["_Attribute__child_id_dict"] = {}
        attr = ids.get(i)
        if not attr:
            _ensure_owner_alive(self)
            attr = _inherit_owner(self, _new_attr(Attribute, self._mplug.child(i)))
            ids[i] = attr
        return attr

    # --- typed attr methods

    def is_typed(self) -> bool:
        """Checks if this attribute is typed (raises as `mobject` does)."""
        return type(self).get_mobject(self).hasFn(OpenMaya.MFn.kTypedAttribute)

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
        if not type(self).is_typed(self):
            raise RuntimeError(f"{self} is not a typed attribute.")
        if not type(self).get_data_type(self).lower().endswith("array"):
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

    def is_multi(self) -> bool:
        """Checks if this attr is a multi/array attribute."""
        _ensure_owner_alive(self)
        return self._mplug.isArray

    def get_num_elements(self) -> int:
        """Number of elements / physical indices in the multi attr."""
        if not type(self).is_multi(self):
            raise RuntimeError(f"{self} is not an multi attr.")
        return self._mplug.numElements()

    def logical_index(self) -> int:
        """Returns the logical index if this attr is a child of a multi attr."""
        _ensure_owner_alive(self)
        return self._mplug.logicalIndex()

    def get_logical_indices(self) -> OpenMaya.MIntArray:
        """Returns existing logical indices (sparse)."""
        if not type(self).is_multi(self):
            raise RuntimeError(f"{self} is not an multi attr.")
        return self._mplug.getExistingArrayAttributeIndices()

    def get_next_available_index(self) -> int:
        """Returns the next available logical index."""
        ids = set(type(self).get_logical_indices(self))
        for index in range(len(ids) + 1):
            if index not in ids:
                return index

    def element_by_physical_index(self, i: int) -> Attribute:
        """Returns the element attribute at the given physical index: an
        instance of `_CHILD_CLASS` (a Plug's element is a Plug), owned by the
        node object this attr holds (see `_inherit_owner`)."""
        if not type(self).is_multi(self):
            raise RuntimeError(f"{self} is not an multi attr.")
        element = self._mplug.elementByPhysicalIndex(i)
        return _inherit_owner(
            self, _new_attr(self._CHILD_CLASS, element), element.logicalIndex()
        )

    def element_by_logical_index(self, i: int) -> Attribute:
        """Returns the element attribute at the given logical index: an
        instance of `_CHILD_CLASS` (a Plug's element is a Plug, which
        ``plug[i]`` returns as built), owned by the node object this attr holds
        (see `_inherit_owner`)."""
        if not type(self).is_multi(self):
            raise RuntimeError(f"{self} is not an multi attr.")
        element = self._mplug.elementByLogicalIndex(i)
        return _inherit_owner(self, _new_attr(self._CHILD_CLASS, element), i)

    def delete_logical_index(self, i: int, **kwargs) -> None:
        """Deletes the element attribute at the given logical index."""
        cmds_edit.removeMultiInstance(str(self.element_by_logical_index(i)), **kwargs)

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


Attribute._CHILD_CLASS = Attribute


# --- plug methods on plugs: a compound child named like a method --------------
#
# ``plug.<name>`` (no brackets) is the compound child of that name when the plug
# has one, else the bound method; ``plug.<name>()`` is always the method (see
# the rule above `_PlugMethod`). The child comes back as a method plug
# (``_MethodPlug``, `_MethodAttribute` for a bare Attribute): a real child plug
# whose call runs the method it shadows. Every public function of Attribute and
# of each subclass (Plug, ComponentPlug, a user's) is a `_ChildOrMethod`
# (`Attribute.__init_subclass__`); a plug has no public properties (the
# property ban, user decision 2026-10-16: ``plug.get_node()``,
# ``plug.is_locked()``). Children match by long name, as ``plug.<child>``
# resolves them; an array root keeps its methods (its unindexed children are no
# plugs to read or write); the str method names already resolve to plugs on a
# Plug (``rig._internal.plug._StrRoute``).


# The plug properties rig had before the property ban (user decision
# 2026-10-16), and their spelling now: the miss path of ``Plug.__getattr__`` and
# ``Attribute.__getattr__`` names it (``plug.get_node()`` is a plug lookup, which
# misses unless the node has an attr ``node``); the hot path never reads it
_OLD_PLUG_NAMES = {
    "plug":           "plug.get_mplug()",
    "full_name":      "str(plug) or plug.get_full_name()",
    "node":           "plug.get_node()",
    "name":           "plug.get_name()",
    "alias":          "plug.get_alias() / plug.set_alias(name)",
    "attribute_type": "plug.get_attribute_type()",
    "data_type":      "plug.get_data_type()",
    "default_value":  "plug.get_default_value() / plug.set_default_value(value)",
    "enums":          "plug.get_enums()",
    "fn_set":         "plug.get_fn_set()",
    "mobject":        "plug.get_mobject()",
    "num_children":   "plug.get_num_children()",
    "num_elements":   "plug.get_num_elements()",
    "next_index":     "plug.get_next_available_index()",
}

# ``plug.<predicate> = state``, a setter before the ban: its spelling now
_OLD_PLUG_SETTERS = {
    "is_locked":      "plug.is_locked = state is now plug.lock() / plug.unlock()",
    "is_keyable":     "plug.is_keyable = state is now plug.set_keyable(state)",
    "is_channel_box": "plug.is_channel_box = state is now plug.set_channel_box(state)",
}


def _old_name_hint(name: str) -> str:
    """The end of a lookup miss's AttributeError: ``"; plug.node is now
    plug.get_node()"`` for an old plug property name, else ``"."``."""
    new = _OLD_PLUG_NAMES.get(name)
    return f"; plug.{name} is now {new}" if new else "."


def _clash_child(inst: Any, name: str) -> Any:
    """The compound child Attribute of `inst` named `name` (by long name), or
    None. The handles come first: a freed node's MPlug is never read (an MPlug
    call on freed memory is an access violation)."""
    d     = inst.__dict__
    mplug = d.get("_mplug")
    if mplug is None:
        return None
    handle = d.get("_handle")
    if handle is not None and not handle.isAlive():
        return None
    handle = d.get("_attrh")
    if handle is not None and not handle.isAlive():
        return None
    if not mplug.isCompound or mplug.isArray:
        return None
    owner = d.get("_node")
    fn    = owner.__dict__.get("_fn_set") if owner is not None else None
    if fn is None:
        fn = OpenMaya.MFnDependencyNode(mplug.node())
    if not fn.hasAttribute(name):
        return None
    # the node has an attribute of that name: a child of this compound? Read
    # from the child cache `Attribute.__getattr__` keeps, built here without its
    # raise on a miss (the raise formats full_name: ~5 us)
    names = d.get("_Attribute__child_name_dict")
    if not names:
        names = d.setdefault("_Attribute__child_name_dict", {})
        ids   = d.setdefault("_Attribute__child_id_dict", {})
        for i in range(mplug.numChildren()):
            child_plug = mplug.child(i)
            child_name = child_plug.partialName(False, False, False, False, False, True)
            child_name = child_name.rsplit(".", 1)[-1]
            attr = _inherit_owner(inst, _new_attr(Attribute, child_plug))
            names[child_name] = attr
            ids[i]            = attr
    return names.get(name)


# the Plug class and its method plug (rig._internal.plug sets them): a child of a
# Plug is a ``_MethodPlug``, of a bare Attribute a `_MethodAttribute`
_PLUG_FOR_METHODS        = None
_METHOD_PLUG_FOR_METHODS = None


def _is_predicate_name(name: str) -> bool:
    """True for a predicate's name (``is_locked``, ``has_category``): its bare
    access is guarded (`_BarePredicate`)."""
    return name.startswith(("is_", "has_"))


class _BarePredicate:
    """A predicate named without brackets where no Maya attr of that name answers
    (``plug.is_locked``): the bound method, whose truth value raises, so ``if
    plug.is_locked:`` cannot pass silently (user decision 2026-10-16: a bare
    predicate is guarded). A call runs the method; ``__name__``, ``__func__``,
    ``__self__``, ``__doc__`` and ``__wrapped__`` (``inspect.signature``) are the
    method's."""

    __slots__ = ("_bound",)

    def __init__(self, bound: Any) -> None:
        object.__setattr__(self, "_bound", bound)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._bound(*args, **kwargs)

    def __bool__(self) -> bool:
        name = self._bound.__name__
        raise TypeError(f"{name} is a method: call it ({name}())")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)

    @property
    def __doc__(self) -> Any:  # noqa: D105 (the method's)
        return self._bound.__doc__

    @property
    def __wrapped__(self) -> Any:
        return self._bound

    def __repr__(self) -> str:
        return f"<predicate {self._bound!r}: call it>"


class _ChildOrMethod:
    """A public function of an Attribute class: on an instance, the compound child
    of the same name if there is one (a method plug, whose call runs the
    method), else the bound method (a predicate's guarded when named without
    brackets, `_BarePredicate`); on the class, the function."""

    __slots__ = ("func", "name", "guard", "__dict__")

    def __init__(self, func: Any, name: str) -> None:
        self.func  = func
        self.name  = name
        self.guard = _is_predicate_name(name)
        self.__dict__.update(getattr(func, "__dict__", {}))
        for key in ("__module__", "__name__", "__qualname__", "__doc__"):
            try:
                self.__dict__[key] = getattr(func, key)
            except AttributeError:
                pass
        self.__dict__["__wrapped__"] = func

    def __get__(self, inst: Any, owner: Any = None) -> Any:
        if inst is None:
            return self.func
        bound = _MethodType(self.func, inst)
        try:
            f = _GETFRAME(1)
        except ValueError:
            f = None
        if f is not None:
            co = f.f_code.co_code
            i  = f.f_lasti
            k  = _METHOD_LOAD[co[i]]
            if k and (k & 1 or co[i + 1] & 1):
                return bound
        try:
            child = _clash_child(inst, self.name)
            if child is None:
                return _BarePredicate(bound) if self.guard else bound
            plug_cls = _PLUG_FOR_METHODS
            if plug_cls is not None and isinstance(inst, plug_cls):
                plug = _inherit_owner(inst, _new_attr(plug_cls, child._mplug))
                return _as_method_plug(plug, _METHOD_PLUG_FOR_METHODS, bound)
            plug = _inherit_owner(inst, _new_attr(Attribute, child._mplug))
            return _as_method_plug(plug, _MethodAttribute, bound)
        except Exception:
            # no plug to build (a freed attribute, an MPlug that refuses):
            # the method, never an error
            return _BarePredicate(bound) if self.guard else bound

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # ``Cls.__dict__[name](plug, ...)`` calls the function, as it did
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<plug method {self.__dict__.get('__qualname__', self.name)}>"


def _wrap_plug_methods(cls: type) -> None:
    """Make every public function `cls` defines a `_ChildOrMethod` (Attribute
    and each subclass, through `Attribute.__init_subclass__`), and refuse any
    other public member its body declares: a property, a cached_property or
    another descriptor, lowercase data, a nested class (the property ban, user
    decision 2026-10-16: on a plug every public member is a method, so a bare
    name is a child of that name or the method). A classmethod or staticmethod
    passes, and so does an UPPER_CASE declaration (``UNITS = "cm"``)."""
    for name, value in list(vars(cls).items()):
        if name[:1] == "_":
            continue
        if isinstance(value, _FunctionType):
            type.__setattr__(cls, name, _ChildOrMethod(value, name))
        elif isinstance(value, (classmethod, staticmethod, _ChildOrMethod)):
            continue
        elif name.isupper() and not hasattr(type(value), "__get__") and not isinstance(value, type):
            continue
        else:
            kind = (
                "property" if isinstance(value, property)
                else "class" if isinstance(value, type)
                else "descriptor" if hasattr(type(value), "__get__")
                else "data"
            )
            raise TypeError(
                f"{cls.__name__}.{name}: a plug class has no public {kind} (every "
                f"public member of a plug is a method, so that a bare name is the "
                f"compound child of that name or the method): write a method "
                f"(def get_{name}(self)), or keep state under _{name}"
            )


class _MethodAttribute(Attribute):
    """A bare Attribute's compound child named like an Attribute method
    (``attr.get`` with a child ``get``): the child, whose call runs the method
    of its parent (``attr.get()``)."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__dict__["_method"](*args, **kwargs)

    def __repr__(self) -> str:
        return f'Attribute("{str(self)}")'


_wrap_plug_methods(Attribute)

# the naming `_dg_named_owner` finds a plug's class keeps (class access gives
# the function)
_ATTRIBUTE_FULL_NAME = Attribute.get_full_name

# the members `_names_own_plug` asks a plug's class to keep as Attribute holds
# them (patched or not): read from the class dicts by `_redefines`, since a plug
# method's class access (``cls.get_full_name``) runs `_ChildOrMethod.__get__`, a
# Python call, and the check runs on every API edit
_NAMING_MEMBERS = ("__str__", "get_full_name", "get_alias")


def _redefines(cls: type, names: tuple) -> bool:
    """True if a class of `cls`'s MRO before Attribute defines one of `names`
    (or Attribute is not in it): then `cls` does not keep Attribute's member of
    that name. Read from the class dicts, so no descriptor runs; Plug, the common
    case, is one dict."""
    if cls is Attribute:
        return False
    if cls is _PLUG_FOR_METHODS:
        d = cls.__dict__
        for name in names:
            if name in d:
                return True
        return False
    for base in cls.__mro__:
        if base is Attribute:
            return False
        d = base.__dict__
        for name in names:
            if name in d:
                return True
    return True


# `Node.create`'s body: ``rig._internal.container`` sets it when it loads, to
# ``_node_create(node_type, args, kwargs)`` (the D31 pattern keeps nodetypes free
# of ``rig._internal`` imports; None only mid-import)
_NODE_CREATE_HOOK = None


def _shared_refused(
    call: str, node_type: str, name: Any = None, node_cls: type | None = None
) -> TypeError:
    """The TypeError of ``shared=`` on a creator (``Cls.create``,
    ``Node.create``, ``rn.<type>``), raised before anything is made: create
    always makes a new node, and ``define`` finds or makes one (CC-6). `call`
    spells the call (``"Transform.create(shared=...)"``); `name` is its name,
    if any, for the hint; the class named is `node_cls`, else the one
    registered for `node_type`, else ``Node.define(type, ...)``."""
    key      = name if isinstance(name, str) and name else "x"
    node_cls = node_cls or _NODE_CLASS_DICT.get(node_type)
    if node_cls is not None and getattr(node_cls, "_DEFINE_REFUSED", None):
        # a class built from data or inputs (Mesh, SkinCluster ...) has no define
        return TypeError(
            f"{call}: create always makes a new node; "
            f"{node_cls.__name__}({key!r}) refers to an existing one"
        )
    define   = (
        f"{node_cls.__name__}.define({key!r})"
        if node_cls is not None
        else f"Node.define({node_type!r}, {key!r})"
    )
    return TypeError(
        f"{call}: create always makes a new node; {define} finds or makes it"
    )


class Node(metaclass=NodeMeta):
    """The root of every node class, and the DSL node factory: the one entry
    point from a Maya name or object to a node.

    ``Node(x)`` returns the node object of ``x``, always a node:

    * a node object: itself (``Node(x) is x``);
    * an attribute (a typed :class:`Attribute`, a :class:`Plug`, a component
      plug) or an MPlug: the node it belongs to (a plug's own node object);
    * a str with a ``.``: the node named before the FIRST ``.``
      (``Node("a.tx")`` is node ``a``; paths and namespaces are kept);
    * a node name, uuid, MObject or MDagPath: the typed node (``Transform``,
      ``Mesh``, ...: the most derived class registered for its custom type or
      type chain).

    A name follows one lookup rule (the node classes and the membership
    lookups share it): a path or ``ns:name`` is looked up as written; a bare
    name at the root namespace and, while another namespace is current, in it
    too (``Node("root")`` finds ``char:root`` while ``char`` is current). A
    name that gives no single node raises the lookup family of
    :mod:`rig.nodetypes.errors`, every one also a TypeError and a ValueError:
    NodeNotFoundError (with hints: did you mean, the name in another
    namespace, the scope prefix), AmbiguousNodeError (two DAG paths: use a
    path; ``:x`` and ``:char:x``: spell the namespace), NodeLookupError for a
    pattern (``"red*"`` is a search: ``cmds.ls``). Anything that is not a
    name or node raises ``ValueError``. The attribute a ``"node.attr"``
    string names is ``Attribute("a.tx")`` (typed) or ``Plug("a.tx")`` (DSL).
    ``isinstance(x, Node)`` is True for every node object. :class:`DGNode` and
    its subclasses carry the typed API and the DSL: attribute access returns
    :class:`Plug` instances owned by the node (``node.tx.get_node() is node``),
    ``<<`` / ``>>`` inject and introspect, ``node.tx = 5`` is ``node.tx << 5``.

    Every other node class is a reference to a node that already exists
    (naming never creates): ``Joint("spine_01")`` takes what ``Node(x)`` takes,
    by the same lookup rule, and returns the node ``Node(x)`` gives, which
    must be an instance of the class. D9 amended: ``Cls(x)`` returns the
    canonical typed object, the most derived class, so ``Transform("j1")``
    gives ``Joint("j1")``, equal to ``Node("j1")`` both ways, and
    ``Cls(x) is x`` for a node object already of the class. A node of another type raises
    NodeTypeError (``'grp' is a transform, not a joint; Node('grp') is
    Transform("grp")``), with nothing written; the DG classes check the type
    too (``DisplayLayer("red")`` on a blinn raises). Subtypes pass: an
    engine is an objectSet (``ObjectSet("initialShadingGroup")`` is a
    ``ShadingEngine``). A geometry class takes a transform for its first
    non-intermediate shape of the class's type (``Mesh("body")`` gives
    ``Mesh("bodyShape")``). A class with no node type of its own (a user
    wrapper subclass) wraps a node its ``is_type`` accepts. A class call takes
    exactly one argument: ``Transform()``, ``Transform(None)`` and
    ``Transform("x", tx=1)`` raise TypeError (``Transform.create(name="x",
    ...)`` makes a node). ``Cls.exists(x)`` answers whether ``Cls(x)`` would
    return a node.

    ``Node.create(type, ...)`` makes a node (see :meth:`create`) and
    ``Node.find_all(type)`` lists the nodes of a type. On a node class other
    than ``Node`` and ``Container``, and so on any node (``node.create``),
    ``create`` and ``find_all`` are that class's typed versions
    (``Transform.create(...)``, ``Joint.find_all()``). ``Node.wrap`` lives on
    the metaclass, so a node's Maya attr ``wrap`` stays reachable.
    """

    @classmethod
    @statement
    def create(cls, node_type: str, *args: Any, **kwargs: Any) -> Any:
        """Create a new node of ``node_type`` and return the typed node.

        * A type a node class is registered for (its ``NATIVE_NODE_TYPE`` or
          ``CUSTOM_NODE_TYPE``: ``transform``, ``joint``, ``choice``, ``mesh``,
          ``nurbsCurve``, ``skinCluster``, ``blendShape``, ``displayLayer``,
          ``objectSet``, ``shadingEngine``, ``reference``, a user class, ...)
          runs that class's typed create with the arguments, as
          ``Joint.create(name="j")`` runs for ``Node.create("joint", name="j")``
          and ``SkinCluster.create(mesh, joints)`` for
          ``Node.create("skinCluster", mesh, joints)``. Inside
          ``with container()`` it joins the scope by the typed-create rules
          (D13); the scene registries (display layers, sets and shading
          engines, references) stay out of it unless ``container=True``.
          ``parent=`` puts a DAG node in its parent's space. A type whose class
          makes just its node (``transform``, ``joint``, ``choice`` ...) takes
          keyword arguments only; one built from inputs (``skinCluster``,
          ``blendShape``, ``mesh``, ``nurbsCurve``, ``reference``) raises
          TypeError without them, before anything is made. A display layer is
          empty unless objects are given.
        * A surface shader type makes the shader's network (the shader, its
          ``<shader>SG`` engine and materialInfo) outside the scope unless
          ``container=True``: ``Node.create("blinn", name="red")`` is
          ``Blinn.create(name="red")``, and a type Maya classifies a surface
          shader without a class of its own (``"anisotropic"``, a plug-in
          shader) is ``Material.create(type=..., ...)``. (``rn.blinn()``
          makes the bare shader, in the scope.)
        * Any other type is made by the container scope's ``createNode``
          (``Node.create("multiplyDivide", name="md")``): the node joins the
          active scope, an explicit ``name=`` takes the flattened scope's
          prefix (after any namespace), a GC-eligible utility type is tagged
          for ``cleanup()``, and ``container=False`` leaves the node out of the
          scope. No positional argument may follow the type.

        Either way a keyword that is no flag of the create (``createNode``'s
        ``name`` / ``n``, ``parent`` / ``p``, ``skipSelect`` / ``ss`` and
        ``container=``; a typed create's ``_CREATE_FLAGS``) is an attribute of
        the new node, checked on the type before the node is made (a typo is
        an AttributeError, a wrong enum field name a TypeError, nothing made)
        and set with ``<<`` once it exists:
        ``Node.create("multiplyDivide", operation="divide", input1X=3)``,
        ``Node.create("transform", name="t", tx=1)``. ``shared=`` is refused
        (TypeError, nothing made): create always makes a new node.

        Either way ``skipSelect`` defaults to ``ContainerOptions.skip_selection``
        (for a typed create, when its ``_create`` forwards the flag to
        ``cmds.createNode``), so the new node is not selected unless asked; a
        direct typed create outside a scope (``Joint.create()``) keeps Maya's
        default and selects its node. ``container=False`` has one meaning on
        both: the node is not registered with the scope (the prefix, the tag
        and ``skipSelect`` still apply).
        """
        hook = _NODE_CREATE_HOOK
        if hook is None:
            # mid-import only: loading the container module installs the hook
            import rig._internal.container  # noqa: F401

            hook = _NODE_CREATE_HOOK
        return hook(node_type, args, kwargs)

    @classmethod
    @statement
    def define(
        cls,
        node_type: str,
        name:      str,
        *,
        parent:    Any         = None,
        update:    bool        = True,
        container: bool | None = None,
        **kwargs:  Any,
    ) -> Any:
        """Find the node of ``node_type`` at the key ``name`` names, or make it
        there (see ``DGNode.define``): ``Node.define("transform", "rig")`` is
        ``Transform.define("rig")``.

        A type a node class is registered for runs that class's ``define``
        with the arguments. Any other type is keyed and guarded the same way;
        a node found at the key must be of exactly ``node_type``
        (NodeTypeError otherwise), and a missing one is made by the container
        scope's ``createNode`` (the flattened scope's prefix, the tag, the
        registration) with ``createNode``'s flags, then the attribute keywords
        (checked on the type before any lookup):
        ``Node.define("multiplyDivide", "knee_md", operation="divide")``. A
        shape type (``"locator"``) is refused: Maya makes a shape under a new
        transform, so no key names it. A surface shader type makes its network
        (``Node.define("blinn", "red")`` is ``Blinn.define("red")``,
        ``Node.define("anisotropic", "ani")`` is ``Material.define("ani",
        type="anisotropic")``)."""
        node_cls = _NODE_CLASS_DICT.get(node_type)
        if node_cls is not None:
            return node_cls.define(name, parent=parent, update=update, container=container, **kwargs)
        shader = _CLASSIFY(node_type) if _CLASSIFY is not None else None
        if shader is not None:
            # a surface shader without an exact class: its network
            return shader.define(
                name, parent=parent, update=update, container=container, type=node_type, **kwargs
            )
        from rig.nodetypes.dg_node import _define_untyped  # dg_node imports this module

        return _define_untyped(node_type, name, parent, update, container, kwargs)

    @classmethod
    def find_all(cls, node_type: str, exact_type: bool = True) -> list[Any]:
        """The nodes of ``node_type`` in the scene, as a list of typed nodes:
        the ``find_all`` of the class registered for the type
        (``Node.find_all("joint")`` is ``Joint.find_all()``), else the cast of
        every node ``cmds.ls`` lists for it (``Node.find_all("multiplyDivide")``).
        ``exact_type=False`` also lists the types derived from it. A name that is
        no Maya node type raises ValueError."""
        node_cls = _NODE_CLASS_DICT.get(node_type)
        if node_cls:
            return node_cls.find_all(exact_type=exact_type)
        try:
            known = cmds.nodeType(node_type, isTypeName=True)
        except RuntimeError:
            known = False
        if not known:
            raise ValueError(f"{node_type!r} is not a Maya node type")
        found = cmds.ls(**{"exactType" if exact_type else "type": node_type})
        return [_cast(name) for name in found or ()]


def _node_factory(obj: Any) -> Any:
    """``Node(obj)``: the node object of ``obj`` (see :class:`Node`)."""
    if isinstance(obj, Node):
        return obj
    if isinstance(obj, Attribute):
        # ``Node(plug)`` strips the attribute: the node object the plug holds
        return type(obj).get_node(obj)
    if isinstance(obj, str):
        # "pCube.tx", "pCube.translate", "pCubeShape.vtx[0]": the node, split on
        # the FIRST "." so paths and namespaces are kept (Maya node names have
        # no ".")
        if "." in obj:
            obj = obj.split(".", 1)[0]
        return _node_from_str(obj)
    if isinstance(obj, OpenMaya.MPlug):
        obj = obj.node()
    # the cast core itself (D12)
    result = _cast(obj)
    # defensive: an exotic input the cast resolves to an attribute gives its node
    return type(result).get_node(result) if isinstance(result, Attribute) else result


def _node_from_str(name: str, label: str = "node", probe: bool = False) -> Any:
    """``Node(name)`` for a node name or uuid: the cast core on a hit, the
    lookup rule (`_lookup`) around it. ``label`` names what the name was to
    be in the lookup errors (a node class's reference passes its type:
    "no joint named 'x'").

    A bare name costs one API call more than the cast (the current namespace):
    at the root namespace the cast decides; in another one the name may name a
    node there too (``:x`` and ``:char:x``), so `_lookup` decides. A pattern
    is refused before the cast. When the cast raises, `_lookup` gives the
    error of the family (not found with its hints, ambiguous); when it finds
    the one node the cast failed on, the cast's own error is raised. A uuid no
    node has raises NodeNotFoundError naming it.

    ``probe=True`` is ``Cls.exists``' mode of the same rule: a name no node
    has gives None instead of raising NodeNotFoundError, and the cast's probe
    (``_cast(name, build=False)``) gives the class of a node its type key
    decides instead of the node built from it."""
    spelled = name
    if name.isidentifier():
        if _current_namespace() != ":" and not _is_uuid(name):
            found = _lookup(name, label, probe)
            return None if found is None else _cast(found, not probe)
    elif _PATTERN_CHARS.search(name):
        raise NodeLookupError(name, label)
    elif ":" in name:
        spelled = _from_root(name)
    try:
        found = _cast(spelled, not probe)
    except (TypeError, ValueError, RuntimeError) as exc:
        error = exc
    else:
        if found is not None:
            return found
        # the probe: no node has the name as the cast spells it, so the cast
        # fails and the lookup decides; the one node it may find gives what
        # the reference gives, the cast's node or its error
        found = _lookup(name, label, True)
        return None if found is None else _cast(spelled)
    if _is_uuid(name):
        if cmds.ls(name, uid=True):
            raise error
        if probe:
            return None
        raise NodeNotFoundError(name, label, uuid=True)
    if _lookup(name, label, probe) is None:
        return None
    # the lookup found the one node the cast failed on: its own error
    raise error


def _typed(cls: Any, node: Any) -> Any:
    """The node ``cls(x)`` returns for the node ``Node(x)`` gives: ``node``
    when it is an instance of ``cls``, else what the class's ``_coerce`` hook
    gives, or None (the reference then raises NodeTypeError)."""
    return node if isinstance(node, cls) else cls._coerce(node)


def _refer(cls: Any, args: tuple, kwargs: dict) -> Any:
    """``Cls(x)`` for a node class other than :class:`Node`: the strict typed
    reference. It never writes the scene.

    ``x`` is anything ``Node(x)`` takes, by the same lookup rule and errors
    (a miss by name also names the class's ``define`` when that makes the
    node from the name: "no joint named 'spnie_01' (did you mean
    'spine_01'?); Joint.define('spnie_01') finds or makes it", read when the
    error is printed, see ``DGNode._define_door``), and the result is the
    node ``Node(x)`` gives: returned when it is an instance of ``cls`` (the most
    derived class; ``Cls(x) is x`` for a node object of the class), else the
    class's ``_coerce`` hook decides (a geometry class takes a transform's
    shape, an unregistered class wraps what its ``is_type`` accepts), else
    NodeTypeError. No argument, ``None``, a second argument or any keyword
    raises TypeError: a class call names one node that exists; except that a
    membership class (``_MEMBER_KIND``: ``DisplayLayer``) called with no
    argument is its kind token (``Layer()``), and its ``None`` refusal names
    it (``_NONE_TEXT``); and that a class whose call also makes a value
    (``_KEYS_CALL``: ``AnimCurve([0, 1, 0])`` returns keys) answers first."""
    keys_call = getattr(cls, "_KEYS_CALL", None)
    if keys_call is not None:
        # a class whose call also makes a value (``AnimCurve([0, 1])``: keys)
        made = keys_call(args, kwargs)
        if made is not None:
            return made
    label = _type_label(cls)
    if kwargs or len(args) != 1:
        if not args and not kwargs and cls._MEMBER_KIND:
            return cls._kind()
        raise TypeError(_refer_call_error(cls, label, args, kwargs))
    obj = args[0]
    if isinstance(obj, cls):
        return obj
    if obj is None:
        text = cls._NONE_TEXT or f"None is not {_article(label)} {label} name"
        raise TypeError(f"{text}; {cls._KIND_HINT}" if cls._KIND_HINT else text)
    if isinstance(obj, str) and not isinstance(obj, Attribute):
        name = obj.split(".", 1)[0] if "." in obj else obj
        try:
            node = _node_from_str(name, label)
        except NodeNotFoundError as error:
            error.node_class = cls  # the message names its define (read when printed)
            raise
    else:
        # a plug gives its node, as in Node(x)
        node = _node_factory(obj)
        name = None
    found = _typed(cls, node)
    if found is not None:
        return found
    if name is None:
        name = node.name
    raise NodeTypeError(
        name, label, node.node_type, f"Node({name!r}) is {node!r}{cls._mismatch_hint(node)}"
    )


def _refer_call_error(cls: Any, label: str, args: tuple, kwargs: dict) -> str:
    """The message of a class call that names no node (``Transform()``) or
    passes more than the node (``Transform("x", tx=1)``)."""
    kind    = cls.__name__
    defines = not getattr(cls, "_DEFINE_REFUSED", None)  # Mesh, SkinCluster ... refuse it
    if not args and not kwargs:
        return (
            f"{kind}() names no node: {kind}('x') refers to x; "
            + (f"{kind}.define('x') finds or makes it; " if defines else "")
            + f"{kind}.create(name='x') makes one"
            + (f"; {cls._KIND_HINT}" if cls._KIND_HINT else "")
        )
    name = args[0] if args and isinstance(args[0], str) else kwargs.get("name")
    name = name if isinstance(name, str) else "x"
    return (
        f"{kind}({name!r}, ...) refers to an existing {label} and takes no attributes; "
        + (f"{kind}.define({name!r}, ...) finds or makes it, " if defines else "")
        + f"{kind}.create(name={name!r}, ...) makes a new one"
    )
