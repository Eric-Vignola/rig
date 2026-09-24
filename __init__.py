"""
``rig`` -- Pythonic node-network DSL for Maya.


Two user-facing classes carry the language:

  * :class:`Node` -- wraps any Maya node (any DG / DAG / typed subclass).
    Attribute access (``node.translate``) returns a :class:`Plug`.
    Assignment (``node.tx = 5``) is sugar for ``node.tx << 5``.
    Spec injection (``node << Float("blend")``) adds an attribute.

  * :class:`Plug` -- wraps a single MPlug. All math, comparison, injection,
    and introspection operators live here. ``plug.foo`` looks up a
    *child* attribute first; if not found, falls back to a *sibling* on
    the same node -- so ``decompose(node.matrix).outputRotate`` works
    even if the decomposeMatrix was returned via its ``.outputTranslate``.

  * :class:`Container` -- a subclass of :class:`Node` for Maya container
    nodes. Adds the (DORMANT in v1) publish API on top of Node.

  * :class:`PlugList` -- vectorised broadcast list (NumPy-style strict).

Operator conventions:
    * ``<<`` (right-to-left) -- inject: setAttr / connectAttr / disconnect /
      addAttr-spec / type-shorthand. Chains because it returns the LHS.
    * ``>>`` (left-to-right) -- introspect: ``plug >> None`` reads the value;
      ``plug >> Node`` clones the attr-spec onto another node.
    * ``<<`` with a collection spec (``Tag("x")``) makes the LHS a member
      and returns the LHS, so collections chain: ``cube.f[:3] << Tag("a")
      << Tag("b")``. An attribute spec returns the new plug instead (a
      value goes next): ``<<`` returns what the next ``<<`` should target.
    * ``-Spec("x")`` removes the LHS from that collection; ``Spec()`` (the
      same object as ``Spec(None)``) names no particular one: on ``<<`` it
      purges, removing the LHS from every collection of that kind; on
      ``>>`` it enumerates, ``cube >> Tag()`` being ``Tag.of(cube)``.
    * ``>>`` with a collection spec queries a plain value (the native ids
      of the LHS that are members); a missing collection is a ValueError.
      ``node >> Float("x")`` DECLARES an output attr, ``node >> Tag("x")``
      QUERIES: the RHS family decides.
    * An attribute plug on the left stands for its node, for every kind:
      ``cube.tx << Layer("x")`` and ``cube.t >> Layer()`` act on ``cube``.
      Component plugs (``cube.vtx[:3]``) keep their own meaning, and a
      deformer's ``componentTagExpression`` plug receives a ``Tag``'s name.
    * A node on the left of a per-node kind means the collection itself:
      ``cube << Tag("x")`` creates the tag, ``cube << -Tag("x")`` deletes
      it; its components (``cube.vtx[:5]``, ``cube.f``) mean membership.
    * Materials (``rig.shade``) are exclusive: ``cube << Blinn("x")`` moves
      the LHS into x's shading engine (built on first use);
      ``-Material("x")`` carves the LHS out; ``Material()`` leaves it in
      no engine (green); ``Default()`` reverts to initialShadingGroup.
    * ``Layer`` (display layers) is exclusive and holds objects only, never
      components: ``cube << Layer("x")`` moves cube (its children follow
      through the DAG without joining); ``-Layer("x")`` and ``Layer()``
      both land in defaultLayer, which reads as "no layer" (``cube >>
      Layer()`` is ``None`` there).
    * ``+ - * / ** // %`` -- Pythonic math (build ``plusMinusAverage``,
      ``multiplyDivide``, ``modulo``, ...). Matrix and quaternion are
      detected and routed to the right node type.
    * ``& | ^`` -- logical AND/OR/XOR networks.
    * ``== != < <= > >=`` -- build ``condition`` nodes (returns the output
      Plug, NOT a bool). ``__hash__`` is overridden via ``MObjectHandle``
      so dict / set membership still works.

Examples::

    from rig import Node, container, set_options, get_options
    from rig.spec import Float, Vector, lock

    obj1 = Node.create("transform", name="cube1")
    obj2 = Node.create("transform", name="cube2")
    obj3 = Node.create("transform", name="cube3")

    obj3 << Float("weight", min=0, max=1) << 0.5 << lock

    with container("ye_olde_lerp"):
        obj3.t << (obj2.t - obj1.t) * obj3.weight + obj1.t

    obj2.t << obj1.wm        # auto-decompose matrix
    val = obj1.tx >> None    # cmds.getAttr equivalent

    # Linguistic sibling-attribute fallback
    result = obj1.tx + 5     # Plug('plusMinusAverage1.output1D')
    result.operation = 3     # falls back to sibling 'operation'
"""

from __future__ import annotations

__version__ = "2.0.0a2"

# Function-library submodules -- imported AFTER the core types so they
# can reference Node / Plug / etc. when their public functions run.
from rig import (
    euler,
    functions,
    interpolate,
    matrix,
    membership,
    quaternion,
    random,
    shade,
    trigonometry,
    tween,
    vector,
)

# Cross-type dispatch verbs (Model A) -- the operation-first public form for
# cross-type math (each datatype submodule also exposes the same op as a
# public per-type function). Implementations live in the private
# ``_dispatch`` module; imported here AFTER the datatype submodules above so
# its module-scope impl imports resolve.
from rig._dispatch import (
    angle,
    angle_degrees,
    blend,
    dist,
    elerp,
    inverse,
    lerp,
    normalize,
    slerp,
    to_euler,
    to_matrix,
    to_quaternion,
)

# Core types -- must come first.
from rig._internal.container import (
    cleanup,
    Container,
    container,
    ContainerOptions,
    force_nodes,
    get_options,
    set_options,
)
from rig._internal.generators import arguments, sequences
from rig._internal.list import PlugList
from rig._internal.math_nodes import condition, constant
from rig._internal.memoize import memoize, prune_memoize_caches, vectorize
from rig._internal.node import lift, Node
from rig._internal.node_ops import NodeOp
from rig._internal.plug import InjectionError, Plug
from rig.membership import Components, Layer, Tag

# v4.T: Re-export the spec submodule's public API for top-level
# ergonomic imports. Spec types are value types used inline
# (e.g. ``obj << Float("foo")``), so the ``spec.*`` prefix adds
# verbosity without semantic clarity. The ``rig.spec.*`` paths
# remain valid for backward compatibility; this just adds shorter
# top-level aliases.
from rig.spec import (
    Angle,
    Bool,
    Color,
    destroy,
    Enum,
    Euler,
    Float,
    hide,
    Int,
    lock,
    Matrix,
    Mesh,
    Message,
    Note,
    NurbsCurve,
    NurbsSurface,
    Quat,
    skip,
    String,
    Time,
    unhide,
    unlock,
    Vector,
)

__all__ = [
    # Core types
    "Node",
    "Plug",
    "PlugList",
    "Container",
    "InjectionError",
    # Membership (collection specs and the component carrier)
    "Components",
    "Layer",
    "Tag",
    # Container management
    "container",
    "set_options",
    "get_options",
    "ContainerOptions",
    "force_nodes",
    # Lifting
    "lift",
    # Helpers
    "condition",
    "constant",
    # Iteration helpers
    "sequences",
    "arguments",
    # Decorators
    "memoize",
    "vectorize",
    # Version-dispatch framework
    "NodeOp",
    # Function libraries (v2.A)
    "functions",
    "trigonometry",
    # Function libraries (v2.B)
    "matrix",
    "vector",
    "quaternion",
    "euler",
    # Function libraries (v2.C)
    "interpolate",
    "tween",
    # Function libraries (v2.D)
    "random",
    # Membership grammar
    "membership",
    "shade",
    # Cross-type dispatch verbs (Model A)
    "dist",
    "lerp",
    "slerp",
    "blend",
    "elerp",
    "normalize",
    "inverse",
    "angle",
    "angle_degrees",
    "to_euler",
    "to_quaternion",
    "to_matrix",
    # Spec types (v4.T -- re-exported from rig.spec for ergonomic imports)
    "Angle",
    "Bool",
    "Color",
    "Enum",
    "Euler",
    "Float",
    "Int",
    "Matrix",
    "Mesh",
    "Message",
    "Note",
    "NurbsCurve",
    "NurbsSurface",
    "Quat",
    "String",
    "Time",
    "Vector",
    # Spec modifiers (v4.T)
    "destroy",
    "hide",
    "lock",
    "skip",
    "unhide",
    "unlock",
]