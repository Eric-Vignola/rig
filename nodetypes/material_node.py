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

``Blinn.define("red", color=...)`` finds a blinn or makes it, and
``Blinn.create(name="red", ...)`` always makes a new one. Making a shader
builds its network, as Hypershade does: the shader
(``cmds.shadingNode(asShader=True)``, listed in ``defaultShaderList1``), its
``<shader>SG`` shading engine and the engine's materialInfo. A material is a
shared, scene-level asset, so the network stays out of an active ``with
container()`` scope and is never prefixed, unless ``container=True`` (then the
shader, engine and materialInfo join the scope). The selection never changes.

Usage::

    from rig import Node
    from rig.bridges import nodes as rn
    from rig.nodetypes import Blinn, Lambert, Material

    Blinn.define("red", color=(1, 0, 0))           # Blinn("red"): made now (red, redSG, materialInfo1), or found
    Blinn.create(name="red")                       # Blinn("red1"): always a new network
    Material.create(type="anisotropic", name="ani")    # Material("ani"): a type without an exact class
    Node.create("blinn", name="b")                 # Blinn("b"): the network, as Blinn.create
    rn.blinn(name="bare")                          # the bare shader, joining the scope (no engine yet)

    red = Blinn("red")                             # Blinn("red"): the reference
    Material("red")                                # Blinn("red"): any surface shader
    Lambert("red")                                 # NodeTypeError: 'red' is a blinn, not a lambert; ...
    red.color << (0, 1, 0)                         # a plain plug: red is the node
    Lambert.find_all(exact_type=False)             # lamberts, and the blinns and phongs typed Blinn / Phong
"""

from __future__ import annotations

from typing import Any

from maya import cmds
from rig._internal.undo import _undo_chunk
from rig.nodetypes import _base
from rig.nodetypes._base import (
    _NODE_CLASS_DICT,
    _cast,
    _shared_refused,
    set_custom_type,
)
from rig.nodetypes.dg_node import (
    _DEFINE_OWN,
    _attribute_keywords,
    _define,
    _got,
    DGNode,
)
from rig.nodetypes.errors import _article
from rig.nodetypes.shading_engine import ShadingEngine


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

    ``Material.create(type=..., ...)`` / ``Material.define(name, type=...)``
    make any surface type, a plug-in's included
    (``type="aiStandardSurface"``).
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

    # ``cmds.shadingNode``'s name and skipSelect (True unless given); every
    # other keyword of ``create`` / ``define`` is an attribute of the shader
    _CREATE_FLAGS = frozenset({"name", "n", "skipSelect", "ss"})

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
    def _shader_type(cls, node_type: Any, verb: str, call: str) -> str:
        """The node type a create / define makes: the class's own, or the
        generic class's ``type=`` (checked to be a surface shader Maya can
        make)."""
        own = _own_type(cls)
        if own is None:
            if node_type is None:
                raise TypeError(
                    f"{call} takes type=, the surface shader's node type (type='anisotropic', "
                    f"type='aiStandardSurface'); Blinn, Lambert, Phong ... make their own type"
                )
            if not isinstance(node_type, str):
                raise TypeError(f"{call}: type= is a node type name, not {node_type!r}")
            _gate_type(node_type)
            return node_type
        if node_type is not None and node_type != own:
            raise TypeError(
                f"{call} makes {_article(own)} {own}, not {node_type!r}: "
                f"Material.{verb}(..., type={node_type!r}) makes one"
            )
        return own

    @classmethod
    def create(
        cls,
        *inputs:   Any,
        name:      str | None  = None,
        parent:    Any         = None,
        container: bool | None = None,
        type:      str | None  = None,
        **kwargs:  Any,
    ) -> "Material":
        """Makes a new shader network, always: the shader
        (``cmds.shadingNode(asShader=True)``, in ``defaultShaderList1``), its
        ``<shader>SG`` engine and materialInfo, then the attribute keywords,
        in one undo step. Maya picks the final name (a second
        ``Blinn.create(name="red")`` is ``red1``); without ``name=`` it is the
        type (``blinn``). Returns the typed shader (``Material.create(type=
        "blinn")`` is a ``Blinn``).

        ``Material.create`` takes ``type=``, any type Maya classifies a
        surface shader (``type="aiStandardSurface"`` once its plug-in is
        loaded); an exact class makes its own type. Every keyword but
        ``name`` / ``n``, ``skipSelect`` / ``ss`` and ``container`` is an
        attribute (``color=(1, 0, 0)`` sets, a plug connects). Refused before
        any write (TypeError / ValueError / AttributeError): a positional
        argument, ``parent=``, ``shared=``, a type that is no surface shader
        (``type="multiplyDivide"``, ``"ramp"``), an attribute the type lacks.

        The network stays out of an active ``with container()`` scope and is
        never prefixed; ``container=True`` adds the shader, engine and
        materialInfo to the scope. The selection never changes.
        """
        call      = f"{cls.__name__}.create()"
        node_type = cls._shader_type(type, "create", call)
        if "shared" in kwargs:
            exact = cls if _own_type(cls) else _NODE_CLASS_DICT.get(node_type)
            raise _shared_refused(f"{cls.__name__}.create(shared=...)", node_type, name, exact)
        if inputs:
            raise TypeError(f"{call} takes name= as a keyword (got {_got(inputs)})")
        if parent is not None:
            raise TypeError(f"{call} takes no parent=: {_article(node_type)} {node_type} is a DG node")
        flags = cls._CREATE_FLAGS
        attrs = {}
        if kwargs and not flags.issuperset(kwargs):
            attrs = _attribute_keywords(node_type, kwargs, flags, call)
        if name is None:
            name = kwargs.get("n")
        skip = kwargs.get("skipSelect", kwargs.get("ss", True))
        with _undo_chunk("rig.create"):
            shader = cmds.shadingNode(
                node_type, asShader=True, name=name or cls.CUSTOM_NODE_TYPE or node_type,
                skipSelect=skip,
            )
            engine = ShadingEngine.for_material(shader, create=True)
            if container is True:
                # a material is a shared, scene-level asset: it joins the
                # active scope only when asked
                from rig._internal.container import container as scope

                scope.add([shader, engine.name, *engine.get_material_info()])
            if cls.CUSTOM_NODE_TYPE:
                set_custom_type(shader, cls.CUSTOM_NODE_TYPE)
            # the node just made is of this class (the generic class: its cast)
            node = cls._wrap(shader) if _own_type(cls) else _cast(shader)
            for attr, value in attrs.items():
                getattr(node, attr) << value
            fed = engine.get_material()
            if fed is None or fed.name != shader:
                raise RuntimeError(
                    f"{engine.name}.surfaceShader does not read '{shader}' after the "
                    f"build (it reads {fed})"
                )
        return node

    @classmethod
    def define(
        cls,
        name:      str,
        *,
        parent:    Any         = None,
        update:    bool        = False,
        container: bool | None = None,
        type:      str | None  = None,
        **kwargs:  Any,
    ) -> "Material":
        """Finds the shader at the key ``name`` names, or makes its network
        there (see ``DGNode.define``; ``create`` makes it): ``Blinn.define(
        "red", color=(1, 0, 0))``. A found shader is returned as it is (no
        engine is built for it: the first assignment does that), and its
        attributes are set only with ``update=True``; a shader of another
        type raises NodeTypeError, whose hint names the conversion.

        ``Material.define(name, type=...)`` is the define of that type: of
        its exact class (``type="blinn"`` is ``Blinn.define``), else of any
        surface type, a plug-in's included, with an exact type check on a
        hit (``Node.define(type, name)`` of a surface type comes here)."""
        call      = f"{cls.__name__}.define({name!r})"
        node_type = cls._shader_type(type, "define", call)
        if _own_type(cls) is not None:
            return super().define(name, parent=parent, update=update, container=container, **kwargs)
        exact = _NODE_CLASS_DICT.get(node_type)
        if exact is not None:
            return exact.define(name, parent=parent, update=update, container=container, **kwargs)
        # a surface type without a class: define's rule, keyed by the name,
        # with an exact type check on a hit
        own = _DEFINE_OWN.intersection(kwargs)
        if own:
            raise TypeError(
                f"Material.define() takes its name first (got {', '.join(f'{key}=' for key in sorted(own))})"
            )
        if parent is not None:
            raise TypeError(f"Material.define() takes no parent=: {_article(node_type)} {node_type} is a DG node")
        flags = cls._CREATE_FLAGS
        attrs = {}
        if kwargs and not flags.issuperset(kwargs):
            attrs = _attribute_keywords(node_type, kwargs, flags, "Material.define()")

        def accept(found):
            return found if found.node_type == node_type else None

        def make(made_name, _parent):
            return cls.create(type=node_type, name=made_name, container=container, **kwargs, **attrs)

        return _define(
            lambda args: f"Material.define({args}, type={node_type!r})", "Material", node_type,
            name, None, dag=False, update=update, attrs=attrs, accept=accept, make=make,
            aware=False, joins=container, typed=True, hint=cls._mismatch_hint,
        )

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
