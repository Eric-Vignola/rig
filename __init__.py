"""
``rig`` -- Pythonic node-network DSL for Maya.


Two user-facing classes carry the language:

  * :class:`Node` -- the node factory and the root of every node class:
    ``Node("pCube1")`` returns the typed node (``Transform``, ``Mesh``,
    ...), which carries both the typed API and the DSL; ``Node(x) is x``
    for a node object, and ``isinstance(x, Node)`` holds for every node.
    Attribute access (``node.translate``) returns a :class:`Plug`.
    Assignment (``node.tx = 5``) is sugar for ``node.tx << 5``.
    Spec injection (``node << Float("blend")``) adds an attribute.

  * :class:`Plug` -- wraps a single MPlug. All math, comparison, injection,
    and introspection operators live here. ``plug.foo`` looks up a
    *child* attribute first; if not found, falls back to a *sibling* on
    the same node -- so ``decompose(node.matrix).outputRotate`` works
    even if the decomposeMatrix was returned via its ``.outputTranslate``.
    ``plug.node`` is the node the plug was read from (``node.tx.node is
    node``), and an instanced node names its plugs through that node's path.

  * :class:`Container` -- a node class (a ``DGNode`` subclass) for Maya
    container nodes. Adds the (DORMANT in v1) publish API.

  * :class:`List` -- vectorised broadcast list (NumPy-style strict).

Operator conventions:
    * ``<<`` (right-to-left) -- inject: setAttr / connectAttr / disconnect /
      addAttr-spec / type-shorthand. Chains because it returns the LHS. An
      enum plug takes a field name as well as its int (``t.ro << "zxy"``,
      ``md.operation << "divide"``); a wrong name raises TypeError.
    * ``>>`` (left-to-right) -- introspect: ``plug >> None`` reads the value;
      ``plug >> Node`` clones the attr-spec onto another node.
    * ``<<`` with a collection (``Tag("x")``, a layer node) makes the LHS a
      member and returns the LHS, so collections chain: ``cube.f[:3] <<
      Tag("a") << Tag("b")``. An attribute spec returns the new plug instead
      (a value goes next): ``<<`` returns what the next ``<<`` should target.
    * ``-x`` removes the LHS from that collection (``-Tag("x")``, ``-layer``);
      the kind token ``Tag()`` / ``Layer()`` names no particular one: on
      ``<<`` it purges, removing the LHS from every collection of that kind;
      on ``>>`` it enumerates, ``cube >> Tag()`` being ``Tag.of(cube)``.
      ``Tag(None)`` / ``Layer(None)`` are TypeErrors (a failed lookup must
      not mean every collection).
    * ``lhs in x`` / ``lhs not in x`` asks yes or no, with all-members
      semantics: ``cube.vtx[:3] in Tag("cap")`` is True when all three are
      in it, ``cube in layer`` when cube is; a plug stands for its node there
      (``cube.tx in layer``). A tag the node does not have answers False.
    * ``>>`` with a collection answers ids (the native ids of the LHS that
      are members); ``node >> Float("x")`` DECLARES an output attr, ``node
      >> Tag("x")`` QUERIES: the RHS family decides. A layer holds whole
      objects and has no ids (``cube >> layer`` raises, naming ``in``).
    * An attribute plug on the left of a membership ``<<`` / ``>>`` / ``of``
      is refused (the node is the member: ``cube << layer``; to connect,
      name a plug). Component plugs (``cube.vtx[:3]``) are members, and a
      deformer's ``componentTagExpression`` plug receives a ``Tag``'s name.
    * A node on the left of a per-node kind means the collection itself:
      ``cube << Tag("x")`` creates the tag, ``cube << -Tag("x")`` deletes
      it; its components (``cube.vtx[:5]``, ``cube.f``) mean membership.
    * Materials (``rig.shade``) are exclusive: ``cube << Blinn("x")`` moves
      the LHS into x's shading engine (built on first use);
      ``-Material("x")`` carves the LHS out; ``Material()`` leaves it in
      no engine (green); ``Default()`` reverts to initialShadingGroup.
    * ``Layer`` is the display layer node class (``Layer is DisplayLayer``):
      ``Layer("x")`` refers to an existing layer, ``Layer.define("x")`` finds
      or makes it. Layers are exclusive and hold objects only, never
      components: ``cube << layer`` moves cube (its children follow through
      the DAG without joining); ``-layer`` and ``Layer()`` both land in
      defaultLayer, which reads as "no layer" (``cube >> Layer()`` is
      ``None`` there).
    * ``+ - * / ** // %`` -- Pythonic math (build ``plusMinusAverage``,
      ``multiplyDivide``, ``modulo``, ...). Matrix and quaternion are
      detected and routed to the right node type. A plain ``str`` operand
      raises ``TypeError`` before anything is built: write ``Plug("a.tx")``.
    * ``& | ^`` -- logical AND/OR/XOR networks.
    * ``== != < <= > >=`` -- build ``condition`` nodes (returns the output
      Plug, NOT a bool), except that ``==`` / ``!=`` of two objects of one
      Maya plug fold to ``True`` / ``False`` with no node (unless
      ``force_nodes()``). An ordering result (``<`` ...) has no truth value:
      ``if plug > 0:`` and ``sorted(plugs)`` raise TypeError. ``__hash__`` is
      overridden to key the Maya plug (its node, attribute and logical
      indices, kept across a rename) so dict / set membership still works.

Undo:
    * rig's edits are undoable. ``Mesh.create``, ``SkinCluster.create`` and
      each membership edit (``cube << Tag("x")``, ...) are one undo step,
      named after the operation in the Edit menu (``rig.Mesh.create``,
      ``rig.tag``); each of rig's API edits (``Mesh.set_points``, the UV and
      colour set edits, ``SkinCluster.set_weights``, ...) is one undo step.
    * :func:`undo_chunk` makes several statements one named step:
      ``with rig.undo_chunk("build arm"):``, or ``@rig.undo_chunk`` on a
      function (the step is named after its ``__qualname__``), or
      ``@rig.undo_chunk("build arm")``.

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

__version__ = "2.0.0a3"

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
from rig._internal.list import List
from rig._internal.math_nodes import condition, constant
from rig._internal.memoize import memoize, prune_memoize_caches, vectorize
from rig._internal.node import lift, Node
from rig._internal.node_ops import NodeOp
from rig._internal.plug import InjectionError, Plug
from rig._internal.undo import undo_chunk
from rig.membership import Components, Layer, Tag
from rig.nodetypes.errors import (
    AmbiguousNodeError,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
)

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
    "List",
    "Container",
    "InjectionError",
    # Lookup errors (one family; each is also a TypeError and a ValueError)
    "NodeLookupError",
    "NodeNotFoundError",
    "AmbiguousNodeError",
    "NodeTypeError",
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
    # Undo
    "undo_chunk",
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