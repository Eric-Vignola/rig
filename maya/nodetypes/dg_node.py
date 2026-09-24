from __future__ import annotations

import re
from functools import total_ordering
from typing import Any, Sequence

from maya import cmds, OpenMaya as OpenMaya1
from maya.api import OpenMaya
from rig.maya.nodetypes._base import (
    Attribute,
    get_custom_type,
    NodeMeta,
    PyNode,
    set_custom_type,
)


# Maya recognises these short names in cmds and MSelectionList parsing,
# but ``MFnDependencyNode.findPlug()`` does not -- they need explicit
# translation to the underlying plug name. Used as a fast filter in
# :meth:`DGNode.find_attr` to gate the alias-resolution fallback;
# the actual canonical name is looked up per-node-type via
# ``cmds.listAttr(f"{node}.{alias}[0]")`` so that aliases which are
# invalid for a given node type (e.g. ``mesh.cv``, ``curve.vtx``,
# ``lattice.pnts``) are correctly rejected.
#
# Edge / face component types (``e``, ``f``) intentionally have no entry
# here -- they're component types, not plugs, and have no settable /
# connectable underlying attribute.
_COMPONENT_ALIASES = frozenset(
    {
        "vtx",  # mesh vertices
        "cv",  # nurbsCurve / nurbsSurface CVs
        "pt",  # lattice points / generic point alias
        "pnts",  # alternative spelling
        "map",  # mesh UV components
        "uv",  # mesh UV components (alternative spelling)
    }
)


# Maya 2026 changed the canonical plug name that ``cmds.listAttr`` reports for
# some mesh component aliases:
#   * ``vtx`` / ``pt``  -> ``"pnts"``               (was ``"controlPoints"``)
#   * ``map`` / ``uv``  -> ``"uvSet.uvSetPoints"``  (was ``"uvpt"``)
# Normalize those back to the stable, cross-version canonical the rest of the
# DSL uses (mirrors ``rig._internal.plug._COMPONENT_TYPE_TO_PLUG_ATTR``,
# which the ``Plug("mesh.vtx[0]")`` construction path already uses) so that
# Node attribute access (``node.vtx``) and Plug construction resolve a
# component to the SAME plug regardless of Maya version. Both targets are
# valid, LIVE attributes on 2026: ``controlPoints`` mirrors ``pnts`` for
# vertex position, and ``uvpt`` is the attribute that actually drives the
# ``map`` UV component (the same-named ``uvSet.uvSetPoints`` array is
# independent of the live UVs that ``cmds.polyEditUV`` reflects).
_CANONICAL_COMPONENT_PLUG = {
    "pnts": "controlPoints",
    "uvSetPoints": "uvpt",
    "uvSet.uvSetPoints": "uvpt",
}


def _is_unresolved_multi_child(plug: OpenMaya.MPlug) -> bool:
    """True if ``plug`` (or an ancestor) is a multi element with an unresolved
    logical index (-1).

    ``MFnDependencyNode.findPlug`` returns such a plug for a bare child-of-multi
    name -- e.g. a blendShape's ``weights`` (child of the ``weightList`` multi)
    resolves to ``weightList[-1].weights``. That plug cannot be set/connected,
    and slicing it calls ``MPlug.getExistingArrayAttributeIndices()`` which
    aborts Maya with a ``TDEbadMultiIndex`` C++ exception.
    """
    cur = plug
    while True:
        if cur.isElement and cur.logicalIndex() < 0:
            return True
        if cur.isChild:
            cur = cur.parent()
        elif cur.isElement:
            cur = cur.array()
        else:
            return False


def get_short_name(name: Any) -> str:
    """Returns the short name of a given node."""
    return str(name).rsplit("|", 1)[-1]


def get_clean_name(name: Any) -> str:
    """Returns clean name of a given node (no namespace)"""
    return get_short_name(name).rsplit(":", 1)[-1]


@total_ordering
class DGNode(metaclass=NodeMeta):
    """Base class for DG nodes.

    This is a pymel-style class that maintains API handles to objects (no need to worry
    about name and hierarchy changes), it also provides rich methods for interacting
    with node properites, attributes, and connections.

    It can also be extended to implement custom object types.

    Instances of this class can be used as dict keys and passed into functions that
    expect string names."""

    # the maya native node type string
    NATIVE_NODE_TYPE = "entity"

    # custom node type string
    CUSTOM_NODE_TYPE = None

    # the OpenMaya function set for this type
    FN_SET = OpenMaya.MFnDependencyNode

    def __init__(self, node: str | OpenMaya.MObject | DGNode) -> None:
        """Initialize an instance from a node name or a MObject."""
        if isinstance(node, DGNode):
            self._mobject    = node._mobject
            self._fn_set     = node._fn_set
            self._fn_set1    = node._fn_set1
            self._objhandle1 = node._objhandle1
        else:
            if isinstance(node, OpenMaya.MObject):
                self._mobject = node
            else:
                sel = OpenMaya.MSelectionList()
                try:
                    sel.add(str(node))
                except Exception:
                    raise ValueError(f"Invalid node name: {node}")
                self._mobject = sel.getDependNode(0)
            self._fn_set = self.FN_SET(self._mobject)
            self._cache_api1_objects(self._fn_set.name())
        self.is_type(self.name, exact_type=False, failfast=True)
        self._attr_dict = {}  # cache queried attributes

    def _cache_api1_objects(self, name):
        # cache a API 1.0 MFnDependencyNode for validation purpose
        # 2.0 mobjects crash maya after new scene...
        # TODO check if maya fixed this in 2023+
        sel = OpenMaya1.MSelectionList()
        sel.add(name)
        mobject1 = OpenMaya1.MObject()
        sel.getDependNode(0, mobject1)
        self._fn_set1    = OpenMaya1.MFnDependencyNode(mobject1)
        self._objhandle1 = OpenMaya1.MObjectHandle(mobject1)

    # --- dunders

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}")'

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.long_name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __gt__(self, other: Any) -> bool:
        return self.name > str(other)

    def __getattr__(self, attr_name):
        """Implemented to return attribute by name.

        Example:
        ```
        node.my_attr.set(value)
        ```
        """
        return self.find_attr(attr_name, quiet=False)

    # --- properties & utils

    @property
    def mobject(self) -> OpenMaya.MObject:
        """Returns the mobject."""
        self.ensure_valid()
        return self._mobject

    @property
    def node_type(self) -> str:
        """Returns the node type."""
        if self.CUSTOM_NODE_TYPE:
            return self.CUSTOM_NODE_TYPE
        return cmds.nodeType(self.name)

    def has_base_type(self, base_type: str) -> bool:
        """Checks if this node inherits from the given base type."""
        return cmds.objectType(self.name, isAType=base_type)

    @property
    def fn_set(self) -> OpenMaya.MFnBase:
        """Returns the OpenMaya function set."""
        self.ensure_valid()
        return self._fn_set

    @property
    def is_valid(self) -> bool:
        """Check if this node is valid (not deleted)."""
        return self._objhandle1.isValid()

    def ensure_valid(self) -> None:
        """Raise erros if an object is deleted."""
        if not self._objhandle1.isValid():
            name = self._fn_set1.name()
            raise RuntimeError(f"{name} already deleted!")

    @property
    def name(self) -> str:
        """Returns the node name."""
        return self.fn_set.name()

    @property
    def uuid(self) -> str:
        """Returns the node's universally unique identifyer as a string."""
        return str(self.fn_set.uuid())

    @property
    def short_name(self) -> str:
        """Returns the short name."""
        return get_short_name(self.name)

    # for consistency with DAGNode
    long_name = name

    @property
    def clean_name(self) -> str:
        """Returns the clean node name (no namespace)."""
        return get_clean_name(self.name)

    def rename(self, new_name: Any) -> None:
        """Renames the node."""
        new_name = str(new_name)
        if self.short_name != new_name:
            cmds.rename(self.name, new_name)

    @classmethod
    def is_type(cls, node_name, failfast: bool = False, **kwargs) -> bool:
        """Checks if a node is a valid node of this type.

        Args:
            node_name: A node name to check.
            failfast: If True, raise error if this node is invalid.
        """
        # a node is dg unless it's a custom type
        if not cls.CUSTOM_NODE_TYPE:
            return True
        ctyp = get_custom_type(node_name)
        if ctyp and ctyp == cls.CUSTOM_NODE_TYPE:
            return True

        if failfast:
            raise ValueError(f"{node_name} is not a {cls.CUSTOM_NODE_TYPE}")
        return False

    @classmethod
    def exists(cls, name: str) -> bool:
        """Checks if a node of this type exists."""
        return name and cmds.objExists(name) and cls.is_type(name)

    @classmethod
    def find_all(cls, *args, exact_type: bool = True, **kwargs) -> list["DGNode"]:
        """Returns a list of objects of this type in the scene.

        Args:
            exact_type: If True, return nodes of this exact type. Otherwise return
                inherited types as well.
        """
        nodes    = []
        type_key = "exactType" if exact_type else "type"
        kwargs.setdefault(type_key, cls.NATIVE_NODE_TYPE)

        for node in cmds.ls(*args, **kwargs):
            if cls.CUSTOM_NODE_TYPE:
                if get_custom_type(node) == cls.CUSTOM_NODE_TYPE:
                    nodes.append(cls(node))
            else:
                nodes.append(
                    cls(node)
                    if cmds.nodeType(node) == cls.NATIVE_NODE_TYPE
                    else PyNode(node)
                )
        return nodes

    # --- creation & deletion

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates a new node of this type. Can be overridden by subclasses.
        This class can only use Maya APIs and must return a node name string.
        """
        name = kwargs.get("name", kwargs.get("n"))
        name = name or (cls.CUSTOM_NODE_TYPE or cls.NATIVE_NODE_TYPE)
        return cmds.createNode(cls.NATIVE_NODE_TYPE, name=name)

    @classmethod
    def create(cls, *args, **kwargs) -> "DGNode":
        """Creates a new node of this type. Can NOT be overridden by subclasses."""
        new_node = cls._create(*args, **kwargs)
        return cls.post_create(new_node, *args, **kwargs)

    @classmethod
    def post_create(cls, new_node_name: str, *args, **kwargs) -> "DGNode":
        """Post creation operations. Can be overridden by subclasses."""
        if cls.CUSTOM_NODE_TYPE:
            set_custom_type(new_node_name, cls.CUSTOM_NODE_TYPE)
        return cls(new_node_name)

    def duplicate(self, *args, **kwargs) -> list[DGNode]:
        """Thin wrapper around `cmds.duplicate()`."""
        old_nodes = set(cmds.ls(dagObjects=True, long=True))
        cmds.duplicate(self.name, *args, **kwargs)
        if kwargs.get("returnRootsOnly", kwargs.get("rr", False)):
            for node in cmds.ls(dagObjects=True, long=True):
                if node not in old_nodes:
                    return [PyNode(node)]
            return []
        return [
            PyNode(x) for x in cmds.ls(dagObjects=True, long=True) if x not in old_nodes
        ]

    def delete(
        self, nodes: str | DGNode | Sequence[str | DGNode] | None = None, **kwargs
    ) -> None:
        """Thin wrapper around `cmds.delete()`.
        If `nodes` is given, delete them. Otherwise delete this node."""
        cmds.delete(nodes or self.name, **kwargs)

    # --- namespace

    @property
    def namespace(self) -> str:
        """Returns the namespace of this node."""
        return self._fn_set.namespace

    @namespace.setter
    def namespace(self, namespace: str) -> None:
        """Sets the namespace of this node."""
        if not cmds.namespace(exists=namespace):
            cmds.namespace(add=namespace)
        self.rename(f"{namespace}:{self.clean_name}")

    # --- attribute methods

    def find_attr(
        self,
        attr:      Attribute | str | int,
        category:  str       | None      = None,
        data_type: str       | None      = None,
        quiet:     bool                  = False,
    ) -> Attribute | None:
        """Returns an attribute instance that matches the given criteria.

        Args:
            attr: The name or id of the attribute to search for.
            category: The attribute category.
            data_type: The attribute type string.
            quiet: If True, don't raise errors on failure.

        Returns:
            The attribute instance or None if no match was found.
        """
        # pass through
        if isinstance(attr, Attribute):
            if attr.plug.node() != self.mobject:
                if quiet:
                    return None
                raise AttributeError(f"{attr} doesn't belong to {self}.")
            return attr

        # return cached attr
        attr_obj = self._attr_dict.get(attr)
        if attr_obj:
            return attr_obj

        # find mplug
        try:
            if re.match(r"^.*[\[\.].*$", str(attr)):
                sel = OpenMaya.MSelectionList()
                sel.add(f"{self.name}.{attr}")
                plug = sel.getPlug(0)
            else:
                plug = self.fn_set.findPlug(attr, False)
        except RuntimeError:
            # Fallback: component-alias resolution. Maya's
            # ``MFnDependencyNode.findPlug()`` doesn't recognise short
            # component aliases like ``vtx`` / ``cv`` / ``pt``, but
            # ``cmds.listAttr`` does -- and crucially it respects node
            # type, so e.g. ``mesh.cv`` (a nurbsCurve / nurbsSurface
            # alias) is rejected here rather than silently resolving to
            # ``controlPoints``. The static set acts as a fast filter so
            # we only invoke ``cmds.listAttr`` for known aliases, not
            # for arbitrary failed plug lookups (typos etc.).
            canonical = None
            if isinstance(attr, str) and attr in _COMPONENT_ALIASES:
                try:
                    resolved = cmds.listAttr(f"{self.name}.{attr}[0]")
                except (RuntimeError, ValueError):
                    resolved = None
                if resolved:
                    # cmds.listAttr returns the canonical name, possibly
                    # with an element index suffix (e.g. ``"pnts[0]"``).
                    canonical = resolved[0].split("[")[0]
                    # Normalize Maya 2026's renamed mesh-component plugs back
                    # to the DSL's stable canonical so Node attribute access
                    # matches Plug construction and stays on the live attr.
                    canonical = _CANONICAL_COMPONENT_PLUG.get(canonical, canonical)
            if canonical is not None:
                try:
                    plug = self.fn_set.findPlug(canonical, False)
                except RuntimeError:
                    # ``cmds.listAttr`` enumerated the alias but
                    # ``findPlug`` rejected the canonical name. Honour
                    # the ``quiet`` contract and surface the resolution
                    # mismatch in the error so it's debuggable.
                    if quiet:
                        return None
                    raise AttributeError(
                        f"Attribute not found: {self}.{attr} "
                        f"(alias resolved to {canonical!r}, but findPlug failed)"
                    )
            else:
                # ``findPlug`` doesn't resolve node-level aliases (e.g. a
                # blendShape target weight aliased to ``smile``); ``MSelectionList``
                # does -- and returns the concrete element (``weight[0]``) whose
                # ``name()`` reports the alias. Fall back to it before giving up.
                try:
                    sel = OpenMaya.MSelectionList()
                    sel.add(f"{self.name}.{attr}")
                    plug = sel.getPlug(0)
                except RuntimeError:
                    if quiet:
                        return None
                    # nicer error message
                    raise AttributeError(f"Attribute not found: {self}.{attr}")

        # Component aliases that name a *real* attribute (mesh ``pnts`` and
        # its short name ``pt``) resolve via the primary ``findPlug`` branch
        # above, skipping the ``cmds.listAttr`` normalization; re-canonicalize
        # so ``Node`` access matches the ``Plug(...)`` path on every geometry
        # type and Maya version.
        plug = self._canonicalize_component_alias_plug(attr, plug)

        # A bare child-of-multi name resolves to a plug whose enclosing array
        # element is unresolved (logicalIndex -1); it is unusable and aborts
        # Maya when sliced. Reject it like any other missing attribute.
        if _is_unresolved_multi_child(plug):
            if quiet:
                return None
            raise AttributeError(
                f"Attribute not found: {self}.{attr} "
                f"(resolves to {plug.name()!r}, a child of an unindexed multi; "
                f"index the parent multi element first)"
            )

        attr_obj = Attribute(plug)
        # cache non-dynamic attrs to boost performance
        # dynamics attrs can be renamed so caching them is not reliable
        if not attr_obj.is_dynamic:
            ln                  = plug.partialName(False, False, False, False, False, True)
            sn                  = plug.partialName(False, False, False, False, False, False)
            self._attr_dict[ln] = attr_obj
            self._attr_dict[sn] = attr_obj

        # filter by category and type
        if (not category or attr_obj.has_category(category)) and (
            not data_type or attr_obj.data_type == data_type
        ):
            return attr_obj

    def _canonicalize_component_alias_plug(
        self, attr: Attribute | str | int, plug: "OpenMaya.MPlug"
    ) -> "OpenMaya.MPlug":
        """Normalize a component alias that ``findPlug`` resolved to a
        version-specific plug name back to the DSL's stable canonical.

        ``pnts`` (and its short name ``pt``) are *real* mesh attributes, so
        ``MFnDependencyNode.findPlug`` resolves them directly in
        :meth:`find_attr`'s primary branch -- bypassing the ``cmds.listAttr``
        fallback that normalizes the other aliases. Left alone,
        ``node.pnts`` / ``node.pt`` would diverge from ``Plug("mesh.pnts[0]")``
        (which Maya parses as a vertex component and maps to
        ``controlPoints``). Re-resolve so both paths agree and stay on the
        stable, cross-version canonical that mirrors the live geometry.

        No-op for non-aliases or plugs whose name isn't a renamed component
        plug, so it never perturbs ordinary attribute access.
        """
        if not (isinstance(attr, str) and attr in _COMPONENT_ALIASES):
            return plug
        name      = plug.partialName(False, False, False, False, False, True)
        canonical = _CANONICAL_COMPONENT_PLUG.get(name)
        if canonical is None:
            return plug
        try:
            return self.fn_set.findPlug(canonical, False)
        except RuntimeError:
            return plug

    def find_alias(self, alias: str, quiet: bool = False) -> Attribute | None:
        """Returns an attribute that has the given alias.
        Not using MFnDependencyNode.findAlias() as it doesn't work for multi attrs.

        Args:
            alias: An alias to search.
            quiet: If True, don't raise errors on failure.

        Returns:
            The attribute instance or None if no match was found.
        """
        alias_list = cmds.aliasAttr(f"{self.name}", query=True) or []
        for i in range(0, len(alias_list), 2):
            if alias_list[i] == alias:
                sel       = OpenMaya.MSelectionList()
                attr_name = alias_list[i + 1]
                sel.add(f"{self.name}.{attr_name}")
                return Attribute(sel.getPlug(0))
        if not quiet:
            raise RuntimeError(f"Alias not found: {self}.{alias}")

    def has_attr(self, attr: Attribute | str) -> bool:
        """Checks if an attribute exists."""
        if isinstance(attr, Attribute):
            return not attr.mobject.isNull()
        return self.fn_set.hasAttribute(attr)

    def set_attrs(self, skip_missing=False, **kwargs) -> None:
        """Gracefully sets attributes on a node from given kwargs."""

        # skip missing attributes (except for notes which are a special case)
        if skip_missing:
            kwargs = {
                k: v for k, v in kwargs.items() if (self.has_attr(k) or k == "notes")
            }

        for attr, value in kwargs.items():
            # special handling for notes which are visible in the attribute editor
            # but require an addAttr call if setting it for the first time.
            if attr == "notes" and not self.has_attr(attr):
                self.add_attr(attr, dt="string")

            # set the attribute
            a = self.find_attr(attr)
            if a:
                a.set(value)

    def add_attr(self, long_name: str, **kwargs) -> Attribute | None:
        """Adds an attribute of a given name. Thin wrapper of cmds.addAttr().

        Args:
            long_name: The attribute long name.
            kwargs: Keyword arguments passed to cmds.addAttr().

        Returns:
            The attribute object created or None if a compound attr is requested.
        """
        # enforce long name
        kwargs["longName"] = long_name
        if "ln" in kwargs:
            kwargs.pop("ln")

        # add attribute
        cmds.addAttr(self.name, **kwargs)

        # compound attr are not actually created until all child attrs are added
        # there fore quiet is set to True so that find_attr won't fail
        return self.find_attr(long_name, quiet=True)

    def delete_attr(self, attr: Attribute | str) -> bool:
        """Deletse an attribute of a given name. Thin wrapper of cmds.deleteAttr()."""
        if self.has_attr(attr):
            if isinstance(attr, Attribute):
                attr = attr.name
            cmds.deleteAttr(self.name, attribute=attr)
            return True
        return False

    def rename_attr(self, old_name: Attribute | str, new_name: str) -> Attribute:
        """Renames a given attribute. Thin wrapper of cmds.renameAttr()."""
        if not self.has_attr(old_name):
            raise RuntimeError(f"Attribute not found: {self.name}.{old_name}")
        elif self.has_attr(new_name):
            raise RuntimeError(f"Attribute already exists: {self.name}.{new_name}")

        if old_name != new_name:
            if isinstance(old_name, Attribute):
                if old_name.node != self:
                    raise RuntimeError(
                        f"Attribute {old_name} does not belong to {self}"
                    )
                cmds.renameAttr(old_name.full_name, new_name)
            else:
                cmds.renameAttr(f"{self.name}.{old_name}", new_name)

        return self.find_attr(new_name)

    def list_attr(self, *args, **kwargs) -> list[Attribute]:
        """Lists attributes on this node. Thin wrapper of cmds.listAttr().

        Compound child names returned by ``cmds.listAttr`` for multi-of-
        compound attributes (e.g. ``publishedNodeInfo.publishedNode``)
        cannot be resolved into a plug without an element index -- the
        underlying ``MSelectionList.add(node.publishedNodeInfo.publishedNode)``
        call requires ``publishedNodeInfo[N].publishedNode``. These
        unresolvable child names are silently skipped (via
        ``find_attr(quiet=True)``) so that ``list_attr()`` returns the
        list of usable Attribute objects rather than raising.
        """
        attrs = [
            self.find_attr(x, quiet=True)
            for x in cmds.listAttr(self.name, *args, **kwargs) or []
        ]
        return [a for a in attrs if a is not None]

    def list_connections(self, **kwargs) -> list[DGNode | Attribute]:
        """Lists connected attrs on this node. Thin wrapper of
        cmds.listConnections()."""
        casted = {}
        result = []
        for each in cmds.listConnections(self.name, **kwargs) or []:
            obj = casted.get(each)
            if not obj:
                obj          = PyNode(each)
                casted[each] = obj
            result.append(obj)
        return result

    # copy this implementation from Attribute
    find_connected_nodes = Attribute.find_connected_nodes

    def _attr_data_type_fallback(self, attr: Attribute) -> str:
        """A hook for node classes to resolve the data type of an given attribute on this node.

        This is called by Attribute.data_type, when it fails to resolve the data type
        using cmds.getAttr(attr_name, type=True).

        cmds.getAttr should always work but it's not the case in practice. This method
        gives node classes a chance to correct any undesired Maya behavior.
        """
        typ = cmds.getAttr(attr.full_name, type=True)
        # maintain consistent type string with cmds.addAttr()
        if typ == "TdataCompound":
            return "compound"
        return typ

    # --- misc

    def remove_from_all_sets(self) -> None:
        """Removes this node from all object sets."""
        result = cmds.listConnections(
            self.name,
            source      = False,
            destination = True,
            type        = "objectSet",
            plugs       = True,
            connections = True,
        )
        if result:
            for i in range(0, len(result), 2):
                cmds.disconnectAttr(result[i], result[i + 1])