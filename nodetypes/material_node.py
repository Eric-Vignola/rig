"""
Surface shader classes

A surface shader is a node class like ``Transform``: ``Blinn("red")`` refers
to a blinn that exists, and never writes the scene.

The classes are flat siblings, one per exact Maya type: ``Lambert``,
``Blinn``, ``Phong``, ``PhongE``, ``SurfaceShader``, ``StandardSurface`` and
``OpenPBRSurface``. Maya derives blinn and phong from lambert; rig does not
(``_EXACT_TYPE``): ``Node("red")`` of a blinn is ``Blinn("red")``, and
``Lambert("red")`` on it raises NodeTypeError. ``Material`` is the generic
class: the reference that takes any surface shader (``Material("red")`` is
``Blinn("red")``), and the class of every surface type without an exact class
(``anisotropic``, ``rampShader``, a plug-in shader Maya classifies
``shader/surface``): ``Node("ani")`` is ``Material("ani")``.

Usage::

    from rig.nodetypes import Blinn, Lambert, Material

    red = Blinn("red")                             # Blinn("red"): the reference
    Material("red")                                # Blinn("red"): any surface shader
    Lambert("red")                                 # NodeTypeError: 'red' is a blinn, not a lambert; ...
    red.color << (0, 1, 0)                         # a plain plug: red is the node
    Lambert.find_all(exact_type=False)             # lamberts, and the blinns and phongs typed Blinn / Phong
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from rig.nodetypes import _base
from rig.nodetypes._base import _cast
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.errors import _article


# The classification every surface shader satisfies.
_SURFACE = "shader/surface"

# node type -> whether Maya classifies it a surface shader: one
# ``getClassification`` per type per session (a type Maya does not know yet,
# such as a plug-in's before it loads, is not remembered)
_IS_SURFACE: dict = {}


def _is_surface_shader(node_type: str) -> bool:
    """Whether Maya classifies ``node_type`` a surface shader
    (``shader/surface``), memoised per type."""
    known = _IS_SURFACE.get(node_type)
    if known is not None:
        return known
    surface = bool(cmds.getClassification(node_type, satisfies=_SURFACE))
    if surface or _is_type_name(node_type):
        _IS_SURFACE[node_type] = surface
    return surface


def _is_type_name(node_type: str) -> bool:
    try:
        return bool(cmds.nodeType(node_type, isTypeName=True))
    except RuntimeError:
        return False


def _classify(node_type: str) -> type | None:
    """The cast rule's fallback (``_base._CLASSIFY``): ``Material`` for a
    type Maya classifies a surface shader, else None."""
    return Material if _is_surface_shader(node_type) else None


def _gate_type(node_type: str) -> None:
    """Refuse a type Maya cannot build a surface shader from (``createNode``
    of an unregistered type silently makes an ``unknown`` node)."""
    legal = cmds.listNodeTypes(_SURFACE) or []
    if node_type in legal and _is_surface_shader(node_type):
        return
    classification = [c for c in cmds.getClassification(node_type) or [] if c]
    if classification:
        raise TypeError(
            f"'{node_type}' is not a surface shader ({classification[0]}); "
            f"legal types: {legal}"
        )
    raise ValueError(
        f"'{node_type}' is not a registered surface shader (load its plugin "
        f"first); legal types: {legal}"
    )


def _own_type(cls: type) -> str | None:
    """The exact node type of a shader class, through its MRO (a user's
    ``CUSTOM_NODE_TYPE`` subclass of ``Blinn`` inherits ``blinn``); None for
    the generic ``Material``, whose ``NATIVE_NODE_TYPE`` is DGNode's."""
    node_type = cls.NATIVE_NODE_TYPE
    return None if node_type == DGNode.NATIVE_NODE_TYPE else node_type


class Material(DGNode):
    """A surface shader: the generic class (any surface shader) and the base
    of the exact classes (``Blinn``, ``Lambert`` ...). See the module
    docstring.

    ``Material("x")`` refers to any surface shader and returns it typed
    (``Blinn("x")``); a surface type without an exact class
    (``anisotropic``, a plug-in shader) is cast to ``Material``. The exact
    classes are flat: each takes only its own Maya type (``Lambert`` never a
    blinn).
    """

    # no NATIVE_NODE_TYPE: unregistered, the class of the surface types
    # without an exact class (``_base._CLASSIFY``)

    # an exact class never claims a Maya subtype in the cast (a blinn is no
    # Lambert); the generic class takes any surface shader
    _EXACT_TYPE = True

    # a scene registry, found again by name: a create stays out of
    # ``with container()`` (never prefixed) unless ``container=True``
    _CONTAINER_AWARE = False

    # how a lookup error names the generic class's nodes; each exact class
    # names its type (None: the node type)
    _TYPE_LABEL = "surface shader"

    @classmethod
    def is_type(cls, node_name, failfast: bool = False, **kwargs) -> bool:
        """Whether ``node_name`` is a node of this class: its exact type (a
        subtype is not: a blinn is no ``Lambert``), or for the generic class
        any surface shader; a ``CUSTOM_NODE_TYPE`` subclass also checks the
        custom type. ``failfast`` raises ValueError instead of False."""
        if not super().is_type(node_name, failfast=failfast):
            return False
        try:
            kind = cmds.nodeType(str(node_name))
        except RuntimeError:
            kind = None
        own = _own_type(cls)
        if kind is not None and (kind == own if own else _is_surface_shader(kind)):
            return True
        if failfast:
            label = own or "surface shader"
            raise ValueError(f"{node_name} is not {_article(label)} {label} (it is {_article(str(kind))} {kind})")
        return False

    @classmethod
    def _mismatch_hint(cls, node: Any) -> str:
        """The reference's hint for a shader of another type: the generic
        reference, and the conversion."""
        if not isinstance(node, Material) or cls is Material:
            return ""
        name = node.name
        hint = f"; Material({name!r}) takes any surface shader"
        if node.node_type != _own_type(cls):
            hint += f"; {type(node).__name__}({name!r}).astype({cls.__name__}) converts it"
        return hint

    @classmethod
    def find_all(cls, *args, exact_type: bool = True, **kwargs) -> list["Material"]:
        """The shaders of this class in the scene, typed: an exact class's
        type (``exact_type=False``, Maya's ``ls -type``, also lists the types
        Maya derives from it, typed by their own class:
        ``Lambert.find_all(exact_type=False)`` includes the blinns, as
        ``Blinn`` objects); for the generic class every node whose type Maya
        classifies a surface shader (the nodes ``is_type`` accepts)."""
        if _own_type(cls) is not None:
            return super().find_all(*args, exact_type=exact_type, **kwargs)
        # ``ls -type`` of the surface types, then each node's own type
        # (``ls -exactType`` given this list drops its first type)
        types = cmds.listNodeTypes(_SURFACE) or []
        found = cmds.ls(*args, type=types, **kwargs) if types else []
        return [node for node in map(_cast, found or ()) if cls.is_type(node.name)]


class Lambert(Material):
    """A ``lambert`` shader (never a blinn or phong: see :class:`Material`)."""

    NATIVE_NODE_TYPE = "lambert"
    _TYPE_LABEL      = None


class Blinn(Material):
    """A ``blinn`` shader."""

    NATIVE_NODE_TYPE = "blinn"
    _TYPE_LABEL      = None


class Phong(Material):
    """A ``phong`` shader."""

    NATIVE_NODE_TYPE = "phong"
    _TYPE_LABEL      = None


class PhongE(Material):
    """A ``phongE`` shader."""

    NATIVE_NODE_TYPE = "phongE"
    _TYPE_LABEL      = None


class SurfaceShader(Material):
    """A ``surfaceShader`` (a flat colour)."""

    NATIVE_NODE_TYPE = "surfaceShader"
    _TYPE_LABEL      = None


class StandardSurface(Material):
    """A ``standardSurface`` shader."""

    NATIVE_NODE_TYPE = "standardSurface"
    _TYPE_LABEL      = None


class OpenPBRSurface(Material):
    """An ``openPBRSurface`` shader."""

    NATIVE_NODE_TYPE = "openPBRSurface"
    _TYPE_LABEL      = None


# the cast rule's fallback for a surface type without an exact class (D31:
# _base reads it, this module sets it)
_base._CLASSIFY = _classify
