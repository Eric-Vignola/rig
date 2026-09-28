"""
:class:`Plug` -- the rig DSL extension of :class:`rig.nodetypes.Attribute`.

Plugs are the working unit of the DSL: they wrap an MPlug, retain all of
``Attribute``'s introspection, and add operator overloads that **build
node-networks** (``+`` => ``plusMinusAverage``, ``<<`` => set/connect,
``==`` => ``condition``, etc.).

Operator conventions:

================ ===========================================================
Op               Meaning
================ ===========================================================
``a << b``       Inject: ``a`` RECEIVES from ``b`` -- setAttr / connectAttr /
                 disconnect / addAttr-spec / type-shorthand. Returns ``a``
                 so it chains.
``a >> b``       Introspect: ``a >> None`` => ``cmds.getAttr(a)``;
                 ``a >> Node`` => clone ``a``'s spec onto ``b``,
                 return new Plug.
``a + b``        plusMinusAverage / matrix add / quat add
``a - b``        plusMinusAverage(op=2) / matrix subtract / quat subtract
``a * b``        multiplyDivide / multMatrix / pointMatrixMult / quatProd
``a / b``        multiplyDivide(op=2)
``a ** b``       multiplyDivide(op=3)
``a // b``       integer floor-div (NOT PyMel-style disconnect)
``a % b``        modulo
``a & b``        logical AND network
``a | b``        logical OR network
``a ^ b``        logical XOR network
``-a``           negate
``~a``           logical NOT network
``a == b``       condition node (returns its output, NOT a bool!), except
                 that one Maya plug compared with itself folds to ``True``
``a != b``       condition node (one plug with itself: ``False``)
``<``, ``<=``,   condition nodes
``>``, ``>=``
================ ===========================================================

A plain str (one that is not an Attribute) is never an operand: an operator
given one, alone or inside a list, raises ``TypeError`` before it builds
anything (``t.tx == "cube.ty"``, ``"%s" % t.tx``; see
:mod:`rig._internal.operands`). Write ``Plug("cube.ty")`` for the plug.

Connection queries are METHODS, not operators -- ``a.get_inputs()`` and
``a.get_outputs()``, each always a ``List`` (empty when nothing is
wired). Direct connections only: a compound whose children are driven
reports nothing, so slice it (``a[:].get_inputs()``) to query per-child.

A node's attribute lookup gives Plugs (``Node("a").tx``, a typed node's
too: every node class is a :class:`Node`); its typed API (``find_attr`` ...)
reads :class:`Attribute` instances. ``plug.node`` is the node object the plug
was read from (``node.tx.node is node``; children and elements share it), and
a plug read through a node with more than one DAG path is named through that
node's path. A plug built from a string or an MPlug casts its node on first
access (``Attribute.node``: the typed cast of its MPlug's node), and is named as
Maya names it (``Plug("|T2|S.v")`` is ``T1|S.visibility``). Until then it checks the API
1.0 handle of its node it took when it was built (its children and elements
share it), so once that node is deleted or freed it raises ``already
deleted!`` as a plug with an owner does.

``Plug`` overrides ``__hash__`` (a serial of the node, never reused for another
node, and the attribute with its logical indices) and the truth value of an
``==`` / ``!=`` result (whether both operands are the same Maya plug) so that
comparison-as-condition does not break dict / set usage. A dict or set confirms
a hash match with ``==``, and ``==`` / ``!=`` of two objects of one Maya plug
fold to ``True`` / ``False`` with no node (as literal math folds; under
``set_options(constant_folding=False)`` / ``force_nodes()`` they build the
node, whose truth value is the same), so a lookup through a second Plug object
of the same plug builds nothing, matrices included. A plain ``list`` /
``tuple`` scan (``plug in [a, b, plug]``, ``.index``) still compares every
DIFFERENT plug it passes with the DSL ``==``: an equal node each (and
``InjectionError`` for matrices); a ``List`` compares without building.
A plug's identity follows the Maya plug: ``Node("|T1|S").v`` and
``Node("|T2|S").v``, one plug read through two instance paths, are one key
(their names still differ, each is named through the path it was read from),
and a rename or an alias keeps the key. A plain str is not a plug's key:
``{plug: 1}["a.tx"]`` misses; a ``List`` reads a plain str as the Maya
plug it names (``"a.tx" in List([a.translateX])``).
"""

from __future__ import annotations

import functools
import itertools
import logging
import numbers
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
from maya import cmds, OpenMaya as OpenMaya1
from maya.api import OpenMaya
from rig.nodetypes._base import (
    _MISSING,
    _attr_handle,
    _attr_state,
    _class_attr,
    _ensure_owner_alive,
    _inherit_owner,
    _new_attr,
    _path_instance_number,
    _plug_hash,
    _same_plug,
    Attribute,
    Node,
)
from rig._internal.generators import sequences
from rig._internal.introspect import _to_numpy
from rig._internal.maya_version import is_at_least
from rig._internal.operands import (
    _CAN_HOLD_STR,
    _prepared_operand,
    _render as _render_operand,
    _RESHAPED,
    _SCALARS,
    _text_operand,
    operator_error as _operator_error,
    operator_where as _operator_where,
)
from rig.spec._base import _clone_attribute


LOGGER = logging.getLogger(__name__)

# The right-hand sides of ``<<`` that are never reshaped (see `_prepared_operand`):
# numbers, None, plugs and the plain sequences, passed by one isinstance call.
_NOT_RESHAPED = _SCALARS + (Attribute, list, tuple, np.ndarray, str)

# An attribute name Maya keeps: what ``plug >> "name"`` clones under.
_ATTR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The rig._internal modules the hot ``<<`` / ``Plug.node`` paths use; see
# :func:`_lazy`.
_LAZY_MODULES: SimpleNamespace | None = None


def _lazy() -> SimpleNamespace:
    """Return the ``container`` / ``list`` / ``node`` / ``shorthand`` /
    ``types`` modules of :mod:`rig._internal`, imported on first use.

    They all import this module at load, so they cannot be imported at the
    top. Binding them once saves a per-call ``from ... import`` in the hot
    paths; binding the MODULES (not their functions) keeps runtime rebinding
    and ``mock.patch.object(module, ...)`` effective at every call.
    """
    global _LAZY_MODULES
    if _LAZY_MODULES is None:
        from rig._internal import container, list, node, shorthand, types

        _LAZY_MODULES = SimpleNamespace(
            container = container,
            list      = list,
            node      = node,
            shorthand = shorthand,
            types     = types,
        )
    return _LAZY_MODULES


class InjectionError(RuntimeError):
    """Raised when ``<<`` cannot set / connect / disconnect a :class:`Plug`.

    The DSL's injection operator (``<<``) asserts the *desired state* of the
    destination plug, breaking an incoming connection if one is in the way.
    A **locked** attribute is the one inviolable barrier: setting a value or
    connecting into it raises this error (instead of the previous behavior of
    swallowing the Maya ``RuntimeError`` to ``LOGGER.debug`` and silently
    no-op'ing). Subclasses :class:`RuntimeError` so existing callers that
    wrap injection in ``except RuntimeError`` keep working.
    """


# Geometry component apiType strings that must be translated to the
# canonical underlying plug name when a user constructs ``Plug(...)`` from
# a component string like ``"pCube.vtx[0]"``. Maya's ``MSelectionList``
# resolves these as ``MFn.kComponent`` items, NOT plugs -- so
# ``Attribute.__init__`` raises ``TypeError: item is not a plug`` when
# fed them directly. The translation here uses the API to extract the
# component index, then reconstructs ``<dag>.controlPoints[<idx>]`` (or
# ``.uvpt`` for UV maps) so that ``Attribute.__init__`` succeeds.
_COMPONENT_TYPE_TO_PLUG_ATTR = {
    "kMeshVertComponent":  "controlPoints",
    "kCurveCVComponent":   "controlPoints",
    "kSurfaceCVComponent": "controlPoints",
    "kLatticeComponent":   "controlPoints",
    "kMeshMapComponent":   "uvpt",
}


def _maybe_translate_component(name: str) -> Any:
    """If ``name`` is a Maya geometry component string (``vtx[N]``, ``cv[N]``,
    ``controlPoints[N]``, ``cps[N]``, ``pnts[N]``, ``map[N]``, ``uvpt[N]``,
    or the bare-without-index variants), return the canonical underlying
    :class:`MPlug`. Otherwise return ``name`` unchanged.

    Examples::

        "pCube1Shape.vtx[0]"           -> MPlug for pCube1Shape.controlPoints[0]
        "pCube1Shape.controlPoints[0]" -> MPlug for pCube1Shape.controlPoints[0]
        "pCube1Shape.vtx"              -> MPlug for pCube1Shape.controlPoints
        "pCube1.tx"                    -> "pCube1.tx"  (unchanged -- not a component)

    Returning an :class:`MPlug` (not a translated string) is required
    because Maya's :class:`MSelectionList` resolves ``controlPoints[N]``
    as a ``kComponent`` item too -- so a string round-trip fails. The
    direct ``MPlug`` construction (via :class:`MFnDependencyNode` +
    :meth:`MPlug.elementByLogicalIndex`) bypasses that.

    Falls back to returning ``name`` unchanged if any step raises --
    letting :class:`Attribute.__init__` surface the original error
    message for malformed strings.
    """
    try:
        sel = OpenMaya.MSelectionList()
        sel.add(name)
    except RuntimeError:
        return name  # malformed / nonexistent -- let Attribute raise

    try:
        dag, comp = sel.getComponent(0)
    except (TypeError, RuntimeError):
        return name  # not a component -- pass through to plug resolution

    plug_attr = _COMPONENT_TYPE_TO_PLUG_ATTR.get(comp.apiTypeStr)
    if plug_attr is None:
        return name  # unrecognised component kind -- let Attribute decide

    # Construct the underlying MPlug directly via API (string round-trip
    # via "controlPoints[N]" would also fail as kComponent).
    try:
        node_mfn     = OpenMaya.MFnDependencyNode(dag.node())
        attr_mobj    = node_mfn.attribute(plug_attr)
        parent_mplug = OpenMaya.MPlug(dag.node(), attr_mobj)
    except RuntimeError:
        return name  # API failure -- let Attribute raise

    # Bare (un-indexed) component -- return the parent multi MPlug.
    if not name.endswith("]"):
        return parent_mplug

    # Indexed component -- extract the logical index via MItGeometry.
    try:
        idx = OpenMaya.MItGeometry(dag, comp).index()
        return parent_mplug.elementByLogicalIndex(idx)
    except RuntimeError:
        return name  # API failure -- let Attribute raise


def _named_plug(
    name: str, node: Any = None, attr_name: Any = None, known: Any = None
) -> "Plug":
    """``Plug(name)``, for a plug of ``node``, a node object the caller holds:
    checked through that node's API 1.0 handle (see
    ``_ensure_owner_alive``) instead of one ``Attribute.__init__`` looks up by
    name. The name resolves as ``Plug.__init__`` resolves it (a component name
    to its ``controlPoints`` / ``uvpt`` element). Without a node, ``Plug(name)``.
    ``attr_name`` is the name of its attribute, and ``known`` an Attribute of the
    same attribute the caller has (the one a spec's ``add_attr`` found), whose
    attribute handle it takes (see ``_attr_handle``)."""
    held   = None if node is None else node.__dict__
    # NW6: API 1.0 handle read (the handle is kept, see `_attr_state`)
    handle = None if held is None else held.get("_objhandle1")
    if handle is None:
        return Plug(name)
    mplug = _maybe_translate_component(name)
    if not isinstance(mplug, OpenMaya.MPlug):
        sel = OpenMaya.MSelectionList()
        sel.add(mplug)
        mplug = sel.getPlug(0)
    plug  = str.__new__(Plug, name)
    attr1 = None if known is None else known.__dict__.get("_attr1")
    if (
        attr1 is None
        or not attr1.isAlive()
        or known.__dict__["_mplug"].attribute() != mplug.attribute()
    ):
        attr1 = _attr_handle(mplug, handle, held.get("_fn_set1"), attr_name)  # NW6: API 1.0 handle read
    plug.__dict__.update(_attr_state(mplug, handle, attr1))
    return plug


class Plug(Attribute):
    """Operator-extended :class:`Attribute`.

    Construct from:

    * a name string (``Plug("pCube1.tx")``)
    * a Maya geometry component string (``Plug("pCube1Shape.vtx[0]")`` ->
      translated to ``pCube1Shape.controlPoints[0]``)
    * an :class:`MPlug`

    Lookups via ``.<child>`` and ``[<index>]`` return ``Plug`` instances
    (not bare ``Attribute``s) so the DSL propagates through compound and
    multi attributes. They are owned as the plug is (the node object it holds,
    or the handle of its node it took).
    """

    # elements (``element_by_*``, so ``plug[i]``) are Plugs, built once (D29)
    _CHILD_CLASS = None  # set to Plug below the class

    # -- construction / lookup -- #

    def __init__(self, name_or_mplug: Any) -> None:
        # Translate geometry-component strings (vtx[N], cv[N], etc.) to
        # their canonical underlying MPlug BEFORE handing to
        # ``Attribute.__init__``, which uses ``MSelectionList.getPlug(0)``
        # -- that raises ``TypeError`` on ``kComponent`` items (which
        # includes ``controlPoints[N]`` itself, not just the alias forms).
        if isinstance(name_or_mplug, OpenMaya.MPlug):
            super().__init__(name_or_mplug)
            return
        if isinstance(name_or_mplug, Attribute):
            # the plug `name_or_mplug` stands for, read through the node object it
            # holds: ``Plug(Node("|T2|S").find_attr("v"))`` is T2's, as the
            # str buffer ``str.__new__`` took from its name is; with no owner, it
            # takes the handle of its node `name_or_mplug` took
            _ensure_owner_alive(name_or_mplug)
            source = name_or_mplug.__dict__
            state  = _attr_state(
                source["_mplug"], source["_handle1"], source.get("_attr1")
            )
            state["_node"] = source["_node"]
            self.__dict__.update(state)
            return
        if isinstance(name_or_mplug, str):
            name_or_mplug = _maybe_translate_component(name_or_mplug)
        super().__init__(name_or_mplug)

    def __getattr__(self, attr_name: str) -> "Plug":
        if attr_name[:1] == "_":
            # Python probes private and dunder names constantly: a half-built
            # plug (``copy``) has none, nor has a plug of a node a new scene
            # freed (its MPlug points at freed memory), and otherwise only a
            # compound child or a Maya attr of the node resolves -- never the
            # node's Python state, and never through the container query.
            if "_mplug" not in self.__dict__ or not _owner_alive(self):
                raise AttributeError(attr_name)
            try:
                child = super().__getattr__(attr_name)
                return _inherit_owner(self, _new_attr(Plug, child.plug))
            except (AttributeError, TypeError):
                pass
            node = self.node
            return type(node).__getattr__(node, attr_name)
        # a freed node's MPlug points at freed memory: "already deleted!"
        _ensure_owner_alive(self)
        # 1) Try child-attribute lookup first (compound children): the cached
        #    child Attribute names the MPlug, the Plug is a new one each time
        #    (plugs are never cached), owned as its parent is (`_inherit_owner`)
        try:
            child = super().__getattr__(attr_name)
            return _inherit_owner(self, _new_attr(Plug, child.plug))
        except (AttributeError, TypeError):
            # AttributeError -> no such child.
            # TypeError -> MPlug.numChildren() raised because the plug isn't
            # a compound parent (typed-atomic attrs like .matrix / .message).
            # Either way, fall through to the sibling lookup below.
            pass

        # 2) Fall back to sibling on the same node -- restores Eric's
        #    `dec.outputRotate` linguistic affordance:
        #    ``Plug('foo.outputTranslate').outputRotate`` => ``Plug('foo.outputRotate')``.
        #    A Maya attribute (or component) of the node only, through the node
        #    class's ``__getattr__``: never the node's Python members, so
        #    ``node.tx.delete()`` / ``.rename()`` / ``.uuid`` do not reach the node.
        node = self.node
        try:
            return type(node).__getattr__(node, attr_name)
        except AttributeError:
            pass

        # 3) Container-published sibling. When this plug is the result of a
        #    NodeOp whose public interface lives on the owning container
        #    (``(a % b).input1`` / ``.output``), the operator result is the
        #    inner output plug, so steps 1-2 miss. Resolve ``attr_name``
        #    against the container the plug's node belongs to: ``Container``'s
        #    ``__getattr__`` handles BOTH native publishName/bindAttr aliases
        #    (which ``findPlug`` cannot resolve directly) and genuine
        #    on-container attrs.
        try:
            owner = cmds.container(query=True, findContainer=str(self.node))
        except (
            RuntimeError,
            ValueError,
        ):
            owner = None
        if owner:
            from rig._internal.container import Container

            # a published name or an attribute of the container, never its
            # Python members (``cleanup``, ``name`` ...)
            try:
                return Container.__getattr__(Container(owner), attr_name)
            except AttributeError:
                pass

        raise AttributeError(
            f"{self.full_name!r} has no child or sibling attribute {attr_name!r}."
        )

    def __getitem__(self, key: Any) -> "Plug":
        # Multi attrs and geometry components are handled by Attribute's
        # __getitem__ (which already supports both numeric indexing and
        # the kMeshVertComponent / kCurveCVComponent / kSurfaceCVComponent
        # special-case slice bounds); its elements are already Plugs
        # (``_CHILD_CLASS``) owned as this plug is, built once (D29). A list
        # result becomes a List.
        _ensure_owner_alive(self)
        is_indexable_via_attribute = self.is_multi or self._component_type != "unknown"
        if is_indexable_via_attribute:
            result = super().__getitem__(key)
            if isinstance(result, list):
                # Wrap in List so chained DSL operations work on the slice
                # (e.g. ``node.input[:].t << src``).  Lazy-bound to avoid the
                # circular dep with ``_list`` at module load.
                List = _lazy().list.List

                # Tag the returned List with a back-reference to this
                # multi attr -- enables the ``empty_multi[:] << values``
                # idiom by letting :meth:`List.__lshift__` route writes
                # through the parent when the slice was empty (auto-create
                # indices to match the source length).
                parent = self if self.is_multi and isinstance(key, slice) else None
                return List(result, _parent_multi=parent)
            return result

        # Compound non-multi (e.g. ``transform.translate``, ``.rotate``,
        # ``.scale``, ``inputQuat``) -- slice into children using the
        # canonical :meth:`MPlug.child` API. NumPy-style: ``int`` returns
        # a single Plug, ``slice`` returns a List of children.
        try:
            n = self.num_children
        except (RuntimeError, TypeError):
            n = 0

        if n > 0:
            if isinstance(key, int):
                if key < 0:
                    key += n
                if not 0 <= key < n:
                    raise IndexError(
                        f"{self} child index {key} out of range (num_children={n})"
                    )
                return _inherit_owner(self, _new_attr(Plug, self.plug.child(key)))
            if isinstance(key, slice):
                List = _lazy().list.List

                mplug    = self.plug
                indices  = range(*key.indices(n))
                children = [
                    _inherit_owner(self, _new_attr(Plug, mplug.child(i))) for i in indices
                ]
                return List(children)

        # Not multi, not component, not compound -- let Attribute raise the
        # canonical "is not an multi attr" error message.
        return super().__getitem__(key)

    def child(self, i: int) -> "Plug":
        """Return the compound child plug at index ``i`` as a :class:`Plug`.

        Overrides :meth:`Attribute.child` (which returns a bare
        :class:`Attribute`) so that compound-fan-out call sites -- most
        notably :func:`rig._internal.types._get_compound` -- receive children
        with the DSL operator overloads (``<<``, ``+``, ``==``, etc.).

        Without this override, code paths like::

            output_plugs = _get_compound(constant_plug)
            output_plugs[index] << node.outColorR  # <- AttributeError

        fail because ``_get_compound`` calls ``self.child(i)`` and
        ``Attribute.child`` returns a bare ``Attribute`` that lacks
        ``__lshift__``. This override fixes per-channel fan-out for all
        comparison / logical / condition operators that internally
        construct a compound output plug and write per-channel results
        back into it.
        """
        _ensure_owner_alive(self)
        # the cached child Attribute names the MPlug; the Plug is a new one
        # each time (plugs are never cached), owned as this plug is
        return _inherit_owner(self, _new_attr(Plug, super().child(i).plug))

    # -- assignment via attribute syntax -- #

    def __setattr__(self, name: str, value: Any) -> Any:
        """``plug.x = 5`` is sugar for ``plug.x << 5``.

        Internal state (names starting with ``_``) bypasses to normal
        ``__setattr__``, and so does a class attribute that can be set: a
        property with a setter (``plug.alias = ...``, ``plug.is_locked = ...``)
        or another descriptor with ``__set__``. A property without one (such as
        the ``str`` method names routed to plugs, ``plug.center``) stays sugar.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        found = _class_attr(type(self), name)
        if isinstance(found, property):
            settable = found.fset is not None
        else:
            settable = found is not _MISSING and hasattr(type(found), "__set__")
        if settable:
            object.__setattr__(self, name, value)
            return
        # Sugar for inject.
        self.__getattr__(name).__lshift__(value)

    # -- hashing / equality -- #

    def __hash__(self) -> int:
        """Hash by the Maya plug, so the Plug remains a valid dict / set key even
        though ``__eq__`` is overloaded to build a condition-node network.

        The hash is a serial of the node (never reused, see ``_node_serial``)
        plus the attribute's long name with its logical indices (see
        ``_plug_hash``): a plug read through either instance path of a node
        (``Node("|T1|S").v``, ``Node("|T2|S").v``) hashes the same, a rename, an
        alias or a delete to the undo queue does not change it, and the elements
        of different instances (``worldMatrix[0]``, ``worldMatrix[1]``), different
        plugs, hash apart (``==`` on two matrices raises), as do the plugs of a
        freed node and of a node made after it. A plug never hashes as its name:
        a plain str is not a key for it. A typed Attribute of the plug is the
        same key (it compares equal, see ``__eq__``).
        """
        return _plug_hash(self)

    def equals(self, other: Any) -> bool:
        """True equality (plug identity) -- use this when ``==`` would
        accidentally build a condition-node network.

        Two plugs (a ``Plug`` or a typed ``Attribute``) are equal when they are
        the same Maya plug (node, attribute and logical indices): a plug read
        through either instance path of a node, a plug held across a rename, and
        a resolved ComponentPlug element (displayed ``cv[u][v]``) and the
        ``controlPoints[k]`` Plug of its storage, keeping ``equals`` consistent
        with ``__hash__``. Anything else (a plain str) compares by name.
        """
        if isinstance(other, Attribute):
            return _same_plug(self, other)
        return self.full_name == str(other)

    # -- multi-attribute helpers (v4.F.b) -- #

    @property
    def next_index(self) -> int:
        """Return the next available logical index for this multi attribute.

        Raises :class:`ValueError` if this plug is not a multi.

        Useful as the explicit form of the implicit auto-index that
        happens when you do ``multi << scalar`` (which appends to
        ``multi[next_index]``)::

            multi[multi.next_index] << value      # explicit append
            multi.append(value)                   # equivalent helper
        """
        if not self.is_multi:
            raise ValueError(f"{self} is not a multi attribute")
        return self.get_next_available_index()

    def append(self, value: Any) -> "Plug":
        """Append ``value`` at the next available index of this multi attribute.

        Equivalent to ``multi[multi.next_index] << value`` -- explicit
        form of the implicit auto-append that happens when you do
        ``multi << scalar``. Returns the new element :class:`Plug`.

        Raises :class:`ValueError` if this plug is not a multi.
        """
        if not self.is_multi:
            raise ValueError(f"{self} is not a multi attribute")
        idx  = self.get_next_available_index()
        elem = self[idx]
        elem << value
        return elem

    # -- inject (the workhorse `<<` operator) -- #

    def __lshift__(self, other: Any) -> Any:
        """``plug << other`` -- set, connect, disconnect, or add an attribute.

        Returns ``self`` so chaining works:
        ``node << Float("x") << 5 << lock``.
        """
        lazy = _lazy()
        # a freed node's MPlug points at freed memory
        _ensure_owner_alive(self)
        if isinstance(other, Attribute):
            _ensure_owner_alive(other)

        # Disconnect.
        if other is None:
            _disconnect_incoming(self)
            return self

        # Retired connection-query sentinels (the List class or Plug itself
        # on the right). '<<' means "receives from", but a query flows the
        # other way, so the arrow pointed at the wrong end.
        if other is lazy.list.List or other is Plug:
            raise TypeError(
                "'plug << List' has been replaced by 'plug.get_inputs()', "
                "which always returns a List (empty when nothing drives the "
                "plug). Use 'plug << List([...])' -- an INSTANCE -- to connect."
            )

        # Collection spec: a component plug becomes a member (the spec
        # decides what a non-component plug means); returns this plug.
        if lazy.types._is_member_spec(other):
            return other.inject(self)

        # Attribute spec.
        if lazy.types._is_attribute_spec(other):
            return other.apply(self)

        # The operand shapes (F14): a set, frozenset or dict raises TypeError
        # (unordered), an iterator is read into a list. The common right-hand
        # sides are passed by one isinstance on concrete types, before the
        # Iterator ABC check.
        if not isinstance(other, _NOT_RESHAPED) and isinstance(other, _RESHAPED):
            other = _prepared_operand(other, f"{self} << {_render_operand(other)}")

        # Type-shorthand (matrix->transform, quat->euler, ...).
        if lazy.shorthand.shorthand(other, self):
            return self

        # Standard injection.
        _inject_value(self, other)
        return self

    # -- introspect (numpy-aware `get()` + `>>` operator) -- #
    def get(self) -> Any:
        """Numpy-aware ``cmds.getAttr`` -- same shape as ``self >> None``.

        Compound plugs unwrap Maya's ``[(x, y, z)]`` envelope into
        ``np.array([x, y, z])``. Matrix plugs reshape to ``(4, 4)``
        (or ``(-1, 4, 4)`` for matrix multis). Scalars / strings /
        opaque types pass through unchanged.

        For the raw ``cmds.getAttr`` shape, call
        ``cmds.getAttr(self.full_name)`` directly.
        """
        try:
            dt = self.data_type
        except Exception:
            dt = None
        return _to_numpy(super().get(), data_type=dt)

    def get_inputs(self) -> Any:
        """Return the plug driving this one, as a ``List`` of 0 or 1.

        Always a ``List`` -- never a bare ``Plug``, never ``None``. A
        bare-``Plug`` result would answer ``len()`` / ``[0]`` / ``in`` as a
        string, and a ``None`` result fed back into ``<<`` would silently
        disconnect the destination instead of rewiring it.

        Direct connections only: a compound whose children are driven
        reports nothing. Slice it (``plug[:].get_inputs()``) to go
        per-child.
        """
        return _query_connections(self, source=True)

    def get_outputs(self) -> Any:
        """Return every plug this one drives, as a ``List`` of 0..N.

        Same guarantees and same direct-connections-only rule as
        :meth:`get_inputs`.
        """
        return _query_connections(self, source=False)

    def __rshift__(self, other: Any) -> Any:
        """``plug >> None`` => :meth:`get` (numpy-aware value).

        ``plug >> Node`` => clone this plug's spec onto the target node,
        return the new Plug.

        ``plug >> "newName"`` => clone this plug's spec onto the SAME node
        under ``newName``; ``plug >> "other.newName"`` onto another node.
        Both go through the same :func:`_clone_attribute` call as
        ``plug >> Node`` (multi indices mirrored, no connection made) and
        then copy the current value; a name the target already has is a
        ``TypeError`` (a clone never overwrites). Parking spells it:
        ``node.cosinePower >> "__cosinePower__"``.

        ``plug >> container_node`` (when target is a Maya ``container``):
        publish this plug onto the container. Auto-routes to
        ``publish_input`` if the source plug is writable (a knob/input
        attr like ``transform.tx``) or ``publish_output`` if the source
        is read-only (a result attr like ``multiplyDivide.output``).
        Auto-names the published attr from the source's leaf alias
        (already in Maya's camelCase convention). No-op when
        ``flatten_containers=True`` (the default).

        ``plug >> container_singleton`` (the active scope's
        ``rig.container`` instance): same as above, dispatched
        to the active container scope.

        ``vtx[:8] >> Tag("x")`` (a collection spec) => query membership:
        the native ids of these components that are in the collection.

        ``plug >> other_plug`` => ``TypeError`` naming ``other_plug << plug``
        (a plug is a str too, but never a clone name).

        Anything else => ``TypeError`` (use ``<<`` to connect).
        """
        from rig._internal.list import List
        from rig._internal.node import Node
        from rig._internal.types import _is_member_spec

        # a freed node's MPlug points at freed memory
        _ensure_owner_alive(self)
        if other is None:
            return self.get()

        if _is_member_spec(other):
            return other.query(self)

        # Retired connection-query sentinels (the List class or Plug itself
        # on the right).
        if other is List or other is Plug:
            raise TypeError(
                "'plug >> List' has been replaced by 'plug.get_outputs()', "
                "which always returns a List (empty when the plug drives "
                "nothing)."
            )

        if isinstance(other, Node):
            # Container detection: publish instead of clone.
            try:
                node_type = cmds.nodeType(str(other))
            except Exception:
                node_type = None
            if node_type == "container":
                from rig._internal.container import (
                    _publish_to_container,
                    ContainerOptions,
                )

                # Gating: explicit-target form. User passed an explicit
                # container Node -- they're saying "publish onto THIS".
                # No leaf-frame check needed (target is explicit, scope
                # doesn't matter). Just enforce the global on/off flags.
                # Unified passthrough: any "no-publish" case returns the
                # source plug so downstream code keeps working uniformly.
                if (
                    not ContainerOptions.create_containers
                    or not ContainerOptions.publish_attributes
                ):
                    return self

                # Auto-route via attribute writability (uses MFnAttribute,
                # since MPlug doesn't expose isWritable in API 2.0).
                try:
                    import maya.api.OpenMaya as om

                    fn_attr     = om.MFnAttribute(self.plug.attribute())
                    is_writable = bool(fn_attr.writable)
                except Exception:
                    is_writable = True  # default to input on inspection failure

                direction = "input" if is_writable else "output"
                # Auto-name from source's leaf alias (camelCase per Maya convention).
                auto_name = self.alias
                return _publish_to_container(
                    self, other, direction=direction, name=auto_name
                )
            return _clone_attribute(self, other)

        # Singleton shortcut: ``plug >> container`` with the module-level
        # singleton resolves to the active scope's leaf real container.
        from rig._internal.container import container as _container

        if other is _container:
            from rig._internal.container import ContainerOptions

            # Singleton-target form: dispatch to the active scope's leaf
            # container, applying the same leaf-frame-realness rule as
            # publish_input/publish_output. Passthrough cases:
            #   * create_containers=False or publish_attributes=False
            #   * No active scope at all
            #   * Leaf scope is flattened (nested under flatten=True default)
            if (
                not ContainerOptions.create_containers
                or not ContainerOptions.publish_attributes
            ):
                return self
            if not _container._stack:
                return self
            leaf_frame = _container._stack[-1]
            if leaf_frame.container_node is None:
                return self
            return self.__rshift__(leaf_frame.container_node)

        # a plug is a str too, but not a clone name: '>>' never connects
        if isinstance(other, Attribute):
            raise TypeError(
                f"'>>' does not connect plugs: write {other} << {self}, or "
                f"{self}.connect({other}, force=True)"
            )

        if isinstance(other, str):
            return self._clone_as(other)

        raise TypeError(
            f"'>>' from Plug({self}) to {type(other).__name__} is not supported. "
            f"Use '<<' to connect, or '>> None' to get the value."
        )

    def _clone_as(self, target: str) -> "Plug":
        """``plug >> "name"`` / ``plug >> "node.name"``: the named clone."""
        from rig._internal.node import Node

        node_name, sep, attr_name = target.rpartition(".")
        if not sep:
            node_name, attr_name = str(self.node), target
        if not _ATTR_NAME_RE.fullmatch(attr_name):
            raise TypeError(
                f"'>>' clones an attribute under a plain name; {attr_name!r} is not "
                f"one (identifiers of [A-Za-z0-9_] not starting with a digit)"
            )
        if not node_name or not cmds.objExists(node_name):
            raise TypeError(
                f"'>>' clones onto an existing node and '{node_name}' does not exist; "
                f"write plug >> 'name' for the same node or plug >> 'node.name'"
            )
        dst_node = Node(node_name)
        if dst_node.has_attr(attr_name):
            raise TypeError(
                f"'{node_name}' already has an attribute '{attr_name}': '>>' clones, "
                f"it never overwrites. Pick another name, or destroy it first "
                f"({node_name} << destroy({attr_name!r}))"
            )
        new_plug = _clone_attribute(self, dst_node, attr_name=attr_name)
        try:
            if not self.is_multi:
                new_plug << self.get()
        except Exception as e:
            LOGGER.debug("could not copy the value of %s onto %s: %s", self, new_plug, e)
        return new_plug

    # -- arithmetic operators (each delegates to a NodeOp in _math_nodes) -- #

    def __add__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_add_op,
            _plus_minus_average_op,
            _quaternion_add_op,
        )
        from rig._internal.types import _is_matrix, _is_quaternion

        if _is_matrix(self):
            return _matrix_add_op(self, other)
        if _is_quaternion(self):
            return _quaternion_add_op(self, other)
        return _plus_minus_average_op(self, other, operation=1, name="add1")

    def __radd__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_add_op,
            _plus_minus_average_op,
            _quaternion_add_op,
        )
        from rig._internal.types import _is_matrix, _is_quaternion

        if _is_matrix(self):
            return _matrix_add_op(other, self)
        if _is_quaternion(self):
            return _quaternion_add_op(other, self)
        return _plus_minus_average_op(other, self, operation=1, name="add1")

    def __sub__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_inverse_op,
            _matrix_multiply_op,
            _plus_minus_average_op,
            _quaternion_subtract_op,
        )
        from rig._internal.types import _is_matrix, _is_quaternion

        if _is_matrix(self):
            return _matrix_multiply_op(self, _matrix_inverse_op(other))
        if _is_quaternion(self):
            return _quaternion_subtract_op(self, other)
        return _plus_minus_average_op(self, other, operation=2, name="sub1")

    def __rsub__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_inverse_op,
            _matrix_multiply_op,
            _plus_minus_average_op,
            _quaternion_subtract_op,
        )
        from rig._internal.types import _is_matrix, _is_quaternion

        if _is_matrix(self):
            return _matrix_multiply_op(_matrix_inverse_op(other), self)
        if _is_quaternion(self):
            return _quaternion_subtract_op(other, self)
        return _plus_minus_average_op(other, self, operation=2, name="sub1")

    def __mul__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_multiply_op,
            _multiply_divide_op,
            _quaternion_multiply_op,
        )
        from rig._internal.types import (
            _is_matrix,
            _is_quaternion,
            _is_scalar_value,
        )

        # ``_is_scalar_value`` is True for matrix plugs too (matrix is a
        # typed-atomic, neither compound nor sequence), so the "is this a
        # plain scalar?" predicate must exclude matrix / quaternion plugs.
        def _is_plain_scalar(obj: Any) -> bool:
            return (
                _is_scalar_value(obj)
                and not _is_matrix(obj)
                and not _is_quaternion(obj)
            )

        if _is_matrix(self) or _is_matrix(other):
            # matrix * scalar is undefined -- ``**`` does fractional transform
            # toward identity instead. (``matrix * vec3``, ``matrix * matrix``
            # stay valid: point-transform and composition respectively.)
            if (_is_matrix(self) and _is_plain_scalar(other)) or (
                _is_matrix(other) and _is_plain_scalar(self)
            ):
                raise TypeError(
                    "matrix * scalar is undefined; use ** for fractional "
                    "transform toward identity"
                )
            return _matrix_multiply_op(self, other)
        if _is_quaternion(self) or _is_quaternion(other):
            # quaternion * scalar is undefined -- ``**`` does fractional
            # rotation toward identity instead.
            if (_is_quaternion(self) and _is_plain_scalar(other)) or (
                _is_quaternion(other) and _is_plain_scalar(self)
            ):
                raise TypeError(
                    "quaternion * scalar is undefined; use ** for fractional "
                    "rotation toward identity"
                )
            return _quaternion_multiply_op(self, other)
        return _multiply_divide_op(self, other, operation=1, name="mul1")

    def __rmul__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import (
            _matrix_multiply_op,
            _multiply_divide_op,
            _quaternion_multiply_op,
        )
        from rig._internal.types import (
            _is_matrix,
            _is_quaternion,
            _is_scalar_value,
        )

        def _is_plain_scalar(obj: Any) -> bool:
            return (
                _is_scalar_value(obj)
                and not _is_matrix(obj)
                and not _is_quaternion(obj)
            )

        if _is_matrix(self) or _is_matrix(other):
            if (_is_matrix(self) and _is_plain_scalar(other)) or (
                _is_matrix(other) and _is_plain_scalar(self)
            ):
                raise TypeError(
                    "matrix * scalar is undefined; use ** for fractional "
                    "transform toward identity"
                )
            return _matrix_multiply_op(other, self)
        if _is_quaternion(self) or _is_quaternion(other):
            if (_is_quaternion(self) and _is_plain_scalar(other)) or (
                _is_quaternion(other) and _is_plain_scalar(self)
            ):
                raise TypeError(
                    "quaternion * scalar is undefined; use ** for fractional "
                    "rotation toward identity"
                )
            return _quaternion_multiply_op(other, self)
        return _multiply_divide_op(other, self, operation=1, name="mul1")

    def __truediv__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _multiply_divide_op

        return _multiply_divide_op(self, other, operation=2, name="div1")

    def __rtruediv__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _multiply_divide_op

        return _multiply_divide_op(other, self, operation=2, name="div1")

    def __pow__(self, other: Any) -> "Plug":
        # Dispatch on ``self`` only: the exponent ``other`` is a scalar.
        # For matrix / quaternion self, route to the named ``pow`` verb
        # (fractional transform / rotation toward identity). Otherwise
        # keep the legacy scalar/vector multiplyDivide(operation=3) path.
        from rig._internal.types import (
            _is_matrix,
            _is_quaternion,
            _is_scalar_value,
        )

        if _is_quaternion(self) or _is_matrix(self):
            # ``bool`` is an ``int`` subclass (so ``_is_scalar_value`` accepts
            # it), but ``q ** True`` is nonsense -- reject explicitly. A scalar
            # plug exponent (``m ** weightPlug``) is fine: it's a scalar value.
            # NB: ``_is_scalar_value`` is also True for matrix plugs (typed-
            # atomic, neither compound nor sequence), so exclude those too.
            is_plain_scalar = (
                _is_scalar_value(other)
                and not _is_matrix(other)
                and not _is_quaternion(other)
                and not isinstance(other, bool)
            )
            if not is_plain_scalar:
                raise TypeError(
                    "** exponent must be a numeric scalar (number or scalar "
                    f"plug), not {type(other).__name__}"
                )
            if _is_quaternion(self):
                from rig.quaternion import pow as _q_pow

                return _q_pow(self, other)
            from rig.matrix import pow as _m_pow

            return _m_pow(self, other)
        from rig._internal.math_nodes import _multiply_divide_op

        return _multiply_divide_op(self, other, operation=3, name="pow1")

    def __rpow__(self, other: Any) -> "Plug":
        # ``scalar ** matrix / quaternion`` is undefined (the only sensible
        # transform-toward-identity form is ``matrix|quaternion ** scalar``).
        from rig._internal.types import _is_matrix, _is_quaternion

        if _is_matrix(self) or _is_quaternion(self):
            raise TypeError(
                "scalar ** matrix/quaternion is undefined; "
                "did you mean (matrix|quaternion) ** scalar?"
            )
        from rig._internal.math_nodes import _multiply_divide_op

        return _multiply_divide_op(other, self, operation=3, name="pow1")

    def __floordiv__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _floor_div

        return _floor_div(self, other)

    def __rfloordiv__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _floor_div

        return _floor_div(other, self)

    def __mod__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _modulo

        return _modulo(self, other)

    def __rmod__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _modulo

        return _modulo(other, self)

    def __and__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_and

        return _logical_and(self, other)

    def __rand__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_and

        return _logical_and(other, self)

    def __or__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_or

        return _logical_or(self, other)

    def __ror__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_or

        return _logical_or(other, self)

    def __xor__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_xor

        return _logical_xor(self, other)

    def __rxor__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _logical_xor

        return _logical_xor(other, self)

    def __neg__(self) -> "Plug":
        from rig._internal.container import container
        from rig._internal.math_nodes import _multiply_divide_op
        from rig._internal.types import _is_compound

        # Maya 2024+ scalar fast-path: native ``negate`` node (single-input,
        # purpose-built -- same node count as the legacy multiplyDivide(-1)
        # but cleaner node-type semantics in the editor).
        if is_at_least(2024) and not _is_compound(self):
            node = container.createNode("negate", name="negate1")
            node.input << self
            return node.output

        return _multiply_divide_op(self, -1, operation=1, name="negate1")

    def __invert__(self) -> "Plug":
        from rig._internal.math_nodes import _logical_not

        return _logical_not(self)

    # -- comparison operators (build condition nodes) -- #
    #
    # NOTE: __eq__ on a Plug returns a *condition-node Plug*, NOT a bool,
    # unless both operands are the same Maya plug: then it folds to ``True``
    # (``__ne__``: ``False``) and builds nothing, as literal math folds
    # (``constant_folding``). See the module docstring. ``__hash__`` is
    # overridden above so dict/set membership still works.

    def __eq__(self, other: Any) -> "Plug | bool":
        """``self == other``: the output of an ``equal`` (``condition``) node
        comparing the two, whose truth value says whether they are the same
        Maya plug (see ``_identity``).

        When ``other`` is the same Maya plug as ``self`` (the same plug read
        twice, ``node.tx == node.tx``, or through two instance paths, an alias,
        a typed Attribute, or a ComponentPlug element and the
        ``controlPoints[k]`` plug of its storage; see ``_same_plug``), the
        comparison folds to ``True`` and builds no node, so a dict or set
        lookup through a second object of a plug builds nothing. Under
        ``set_options(constant_folding=False)`` / ``with force_nodes():`` it
        builds the node as for two plugs (its truth value is then True). A
        deleted or freed plug raises as it did before the fold. A node is never
        a plug: ``plug == node`` is ``False`` and builds nothing (a node has no
        value to compare), so ``node in [plug, node]`` finds the node.
        """
        from rig._internal.math_nodes import _condition_op

        if isinstance(other, Attribute):
            # names both first: a deleted node raises as the names do
            same = _same_plug(self, other)
            if same and _lazy().container.ContainerOptions.constant_folding:
                return True
            result = _condition_op(self, "==", other)
            if isinstance(result, Plug):
                result._identity = same
            return result
        if isinstance(other, Node):
            return False
        return _condition_op(self, "==", other)

    def __ne__(self, other: Any) -> "Plug | bool":
        """``self != other``: the output of a ``condition`` node comparing the
        two, whose truth value says whether they are different Maya plugs (see
        ``_identity``). Two objects of one Maya plug fold to ``False`` and build
        no node, by the rules ``__eq__`` folds to ``True`` with; ``plug != node``
        is ``True`` and builds nothing."""
        from rig._internal.math_nodes import _condition_op

        if isinstance(other, Attribute):
            # names both first: a deleted node raises as the names do
            same = _same_plug(self, other)
            if same and _lazy().container.ContainerOptions.constant_folding:
                return False
            result = _condition_op(self, "!=", other)
            if isinstance(result, Plug):
                result._identity = not same
            return result
        if isinstance(other, Node):
            return True
        return _condition_op(self, "!=", other)

    # ``list`` / ``set`` / ``dict`` containment calls ``PyObject_IsTrue()`` on
    # the ``__eq__`` result, so this is the only hook that can answer them.
    # Ordinary plugs stay truthy; only a comparison RESULT reports whether its
    # two operands denote the same Maya plug (the same node, attribute and
    # logical indices, whatever the instance path each is named through; see
    # ``_same_plug``). With constant folding on, a comparison of one plug with
    # itself returns the bool and no result exists, so a result built with it
    # on is always False (``!=``: True). The class default keeps ``bool(plug)`` off
    # ``__getattr__`` (a child / sibling / container lookup). An ordering result
    # (``<``, ``<=``, ``>``, ``>=``) holds None: it has no truth value (see
    # ``__bool__``).
    _identity = True

    def __bool__(self) -> bool:
        """A plug is true, and a ``==`` / ``!=`` result says whether its two
        operands are the same Maya plug (see ``_identity``). An ordering result
        (``a.tx > 0``) raises TypeError, as a numpy array does: its value is
        only known when Maya evaluates it, so ``if plug > 0:``, ``sorted(plugs)``,
        ``min`` / ``max`` and chained comparisons would silently take it as
        true."""
        identity = getattr(self, "_identity", True)
        if identity is None:
            raise TypeError(
                f"the truth value of {self.full_name!r}, the output of an ordering "
                f"comparison (<, <=, >, >=), is only known when Maya evaluates it; "
                f"read it with .get(), pick with rig.condition(...), or order plugs "
                f"by name with sorted(plugs, key=str)"
            )
        return identity

    def __ge__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _condition_op

        return _ordering(_condition_op(self, ">=", other))

    def __le__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _condition_op

        return _ordering(_condition_op(self, "<=", other))

    def __gt__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _condition_op

        return _ordering(_condition_op(self, ">", other))

    def __lt__(self, other: Any) -> "Plug":
        from rig._internal.math_nodes import _condition_op

        return _ordering(_condition_op(self, "<", other))


def _ordering(result: Any) -> Any:
    """`result`, the output of an ordering comparison, marked as having no
    truth value when it is a Plug (see `Plug.__bool__`); a folded number or a
    List of results is returned as it is."""
    if isinstance(result, Plug):
        result.__dict__["_identity"] = None
    return result


# ``Plug`` inherits ``str`` so ``cmds`` calls accept it directly, but that also
# means every public ``str`` method wins normal attribute lookup and
# ``__getattr__`` never runs. Those names are all legal Maya attribute names --
# ``center`` and ``translate`` alone exist on 289 and 222 built-in node types,
# and user attrs and blendShape aliases can use any of them -- so route them all
# to the attribute lookup. Use ``str(plug).upper()`` for the string method.
def _attr_over_str_method(name: str) -> property:
    def getter(self: "Plug") -> Any:
        return Plug.__getattr__(self, name)

    getter.__name__     = name
    getter.__qualname__ = f"Plug.{name}"
    return property(getter)


for _str_method in (n for n in vars(str) if not n.startswith("_")):
    setattr(Plug, _str_method, _attr_over_str_method(_str_method))
del _str_method

Plug._CHILD_CLASS = Plug


def _owner_alive(plug: Any) -> bool:
    """False if the node that owns `plug` was freed, or the attribute of a dynamic
    attr (see `_ensure_owner_alive`)."""
    d     = plug.__dict__
    owner = d.get("_node")
    if owner is None:
        handle = d.get("_handle1")
    else:
        handle = owner.__dict__.get("_objhandle1")  # NW6: API 1.0 handle read (hot)
    if handle is not None and not handle.isAlive():
        return False
    handle = d.get("_attr1")
    return handle is None or handle.isAlive()


def _checking_operands(method: Any) -> Any:
    """`method`, a Plug operator, run once its operands are checked, before it
    builds anything. Its plug operands must be alive: the type predicates an
    operator starts with read the operands' MPlugs, which point at freed memory
    once a new scene freed their node. And no operand may be or hold a plain str
    (see `rig._internal.operands`): `t.tx == "cube.ty"` raises TypeError instead
    of building an equal node it cannot set. A set, frozenset or dict operand
    raises TypeError too, and an iterator operand is read into a list, checked
    and passed to `method` as that list (a Plug operator takes one operand at
    most). This is the one frame that checks a Plug operator's operands."""
    dunder = method.__name__

    @functools.wraps(method)
    def checked(self: "Plug", *other: Any) -> Any:
        _ensure_owner_alive(self)
        for operand in other:
            if isinstance(operand, Attribute):
                _ensure_owner_alive(operand)
            elif isinstance(operand, _CAN_HOLD_STR):
                found = _text_operand(operand)
                if found is not None:
                    raise _operator_error(dunder, self, operand, found)
            elif not isinstance(operand, _SCALARS) and isinstance(operand, _RESHAPED):
                listed = _prepared_operand(operand, _operator_where(dunder, str(self), operand))
                found  = _text_operand(listed)
                if found is not None:
                    raise _operator_error(dunder, self, listed, found)
                return method(self, listed)
        return method(self, *other)

    return checked


for _operator in (
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__pow__", "__rpow__", "__floordiv__",
    "__rfloordiv__", "__mod__", "__rmod__", "__and__", "__rand__", "__or__",
    "__ror__", "__xor__", "__rxor__", "__neg__", "__invert__",
    "__eq__", "__ne__", "__ge__", "__le__", "__gt__", "__lt__",
):
    setattr(Plug, _operator, _checking_operands(Plug.__dict__[_operator]))
del _operator


# --------------------------------------------------------------------- #
#  Multi-dimensional geometry components (node.cv[u][v], node.pt[s][t][u])
# --------------------------------------------------------------------- #

# Node types whose CVs are addressed by MORE than one index, paired with
# ``(ndims, {aliases})``. Maya flattens these into a single ``controlPoints``
# array, but the flatten order differs per type (surface: ``u*V + v``;
# lattice: ``s + t*S + u*S*T``) and only Maya's component API knows the
# spans/degree needed to compute it -- so resolution is delegated to the
# proven ``Plug("shape.<alias>[i][j]...")`` string path, never hand-rolled.
# Only these (node-type, alias) pairs get the numpy-style :class:`ComponentPlug`
# handle; every other component (mesh ``vtx``, curve ``cv``, ``map``/``uv`` ...)
# is 1-D and stays a plain ``Plug`` -- a single index selects the element.
_MULTIDIM_COMPONENTS = {
    "nurbsSurface": (2, frozenset({"cv"})),
    "lattice":      (3, frozenset({"pt"})),
}

# Every alias any multi-dimensional spec answers to. Lets
# :func:`_maybe_component_plug` skip the node-type probe for the other names.
_MULTIDIM_ALIASES = frozenset().union(
    *(aliases for _, aliases in _MULTIDIM_COMPONENTS.values())
)


class ComponentPlug(Plug):
    """A :class:`Plug` for a *multi-dimensional* geometry component -- a
    NURBS-surface ``cv`` (2-D: ``u, v``) or a lattice ``pt`` (3-D:
    ``s, t, u``) -- with numpy-style per-axis indexing.

    ``node.cv`` / ``node.pt`` return the *handle* form (``_comp_coords is
    None``), which wraps the flat ``controlPoints`` multi. Indexing the handle
    resolves a numpy-style selection:

    * an all-integer key (``cv[1, 2]``, ``pt[1, 2, 0]``) returns a single
      **element** :class:`ComponentPlug`;
    * any key containing a slice -- or a bare/partial int, which pads the
      remaining axes with ``:`` (numpy ``arr[i] == arr[i, :, ...]``) -- returns
      a :class:`List` of element plugs in row-major order: ``cv[0]`` /
      ``cv[0, :]`` row, ``cv[:, 1]`` column, ``cv[:]`` full grid, ``cv[1:3, 2]``
      range, ``pt[:, :, 0]`` plane, and so on. Chained ``cv[i][j]`` therefore
      still works (``cv[i]`` is the row :class:`List`; ``[j]`` picks it).

    Element plugs *display* (``str`` / ``repr`` / :attr:`full_name`) in the
    component form ``cv[u][v]`` / ``pt[s][t][u]`` -- both valid Maya component
    strings -- while their underlying :class:`MPlug` is the real flat
    ``controlPoints[k]`` storage, so set / connect / get operate normally. Flat
    element access stays exclusively on ``controlPoints[k]``.

    Each coordinate's resolution is delegated to the proven
    ``Plug("shape.<alias>[i][j]...")`` -> :func:`_maybe_translate_component`
    path; the per-axis *sizes* needed to expand slices come from
    :meth:`_axis_sizes` (NURBS spans/degree with periodic wrap; lattice
    divisions). No flatten arithmetic is hand-rolled here.
    """

    # an element by index (``element_by_*`` of the handle: the flat
    # ``controlPoints[k]``) is a Plug, as a Plug's is (D29)
    _CHILD_CLASS = Plug

    def __new__(
        cls,
        mplug:       Any,
        comp_node:   str          = "",
        comp_alias:  str          = "",
        comp_ndims:  int          = 0,
        comp_coords: tuple | None = None,
    ) -> "ComponentPlug":
        # ``Attribute`` subclasses ``str`` and defines no ``__new__`` -- the
        # base ``str.__new__`` only accepts the value arg. For an element the
        # string value is the component form (a valid Maya string, so direct
        # cmds use works); for the bare handle it is built from the MPlug, as
        # ``Plug(mplug)`` does. ``__init__`` records the component metadata.
        if comp_coords is not None:
            coords = "".join(f"[{c}]" for c in comp_coords)
            return super().__new__(cls, f"{comp_node}.{comp_alias}{coords}")
        return super().__new__(cls, mplug)

    def __init__(
        self,
        mplug:       Any,
        comp_node:   str          = "",
        comp_alias:  str          = "",
        comp_ndims:  int          = 0,
        comp_coords: tuple | None = None,
    ) -> None:
        super().__init__(mplug)
        self._set_component(comp_node, comp_alias, comp_ndims, comp_coords)

    def _set_component(
        self,
        comp_node:   str,
        comp_alias:  str,
        comp_ndims:  int,
        comp_coords: tuple | None,
    ) -> None:
        """Record the component metadata (``__init__`` and ``_component_plug``)."""
        # ``_comp_node`` is the node name captured at construction. It is the
        # str BUFFER source for an element only -- NOT used for live resolution
        # (``_axis_sizes``/``_element`` read ``self.node.name`` so a held handle
        # stays rename-safe). Do not reintroduce ``cmds`` lookups on this field.
        self._comp_node  = comp_node
        self._comp_alias = comp_alias
        self._comp_ndims = comp_ndims
        # ``None`` -> bare handle (wraps controlPoints multi);
        # a tuple of one index per axis -> a resolved element.
        self._comp_coords = tuple(comp_coords) if comp_coords is not None else None

    # -- display: element plugs render in component (cv[u][v]) form -- #

    @property
    def full_name(self) -> str:
        if self._comp_coords is None:
            return super().full_name  # bare handle -> controlPoints
        coords = "".join(f"[{c}]" for c in self._comp_coords)
        return f"{self.node.name}.{self._comp_alias}{coords}"

    # -- indexing -- #

    def __getitem__(self, key: Any) -> Any:
        if self._comp_coords is not None:
            # An already-resolved element behaves as a normal Plug (compound
            # children x/y/z, etc.).
            return super().__getitem__(key)
        return self._resolve(key)

    def _resolve(self, key: Any) -> Any:
        ndims = self._comp_ndims
        specs = key if isinstance(key, tuple) else (key,)
        if len(specs) > ndims:
            raise IndexError(
                f"{self._comp_alias} is {ndims}-D but {len(specs)} indices given"
            )
        # numpy: a missing trailing axis means "all of it" (arr[i] == arr[i, :]).
        specs     = specs + (slice(None),) * (ndims - len(specs))
        sizes     = self._axis_sizes()
        per_axis  = []
        has_slice = False
        for axis, (spec, size) in enumerate(zip(specs, sizes)):
            if isinstance(spec, slice):
                per_axis.append(list(range(*spec.indices(size))))
                has_slice = True
            elif isinstance(spec, bool):
                # ``bool`` subclasses ``int``; reject it before the int branch
                # so ``cv[True, 0]`` gives a clear error, not an obscure Maya
                # one from a ``cv[True][0]`` component string downstream.
                raise TypeError(
                    f"component index must be int or slice, got {type(spec).__name__}"
                )
            elif isinstance(spec, int):
                resolved = spec + size if spec < 0 else spec
                if not 0 <= resolved < size:
                    raise IndexError(
                        f"{self._comp_alias} axis {axis} index {spec} out of "
                        f"range [0, {size})"
                    )
                per_axis.append([resolved])
            else:
                raise TypeError(
                    f"component index must be int or slice, got {type(spec).__name__}"
                )
        elements = [self._element(c) for c in itertools.product(*per_axis)]
        if not has_slice:
            return elements[0]  # fully specified single coordinate -> element
        from rig._internal.list import List

        return List(elements)

    def _element(self, coords: tuple) -> "ComponentPlug":
        """Build the resolved element plug for ``coords`` (one index per axis).

        The underlying MPlug is the flat ``controlPoints[k]`` (via the proven
        component-translation path); the plug *displays* in ``cv[u][v]`` form.

        The node name is read from the *live* MPlug (:attr:`node`) at call
        time -- never from the cached ``_comp_node`` -- so a handle held across
        a rename still resolves (MObjectHandle-based identity, the same source
        :attr:`full_name` uses).
        """
        node_name = self.node.name
        name      = f"{node_name}.{self._comp_alias}" + "".join(f"[{c}]" for c in coords)
        mplug     = _maybe_translate_component(name)
        element   = _component_plug(
            mplug, node_name, self._comp_alias, self._comp_ndims, coords, None
        )
        # owned by the node object the handle holds (``self.node`` bound it)
        element.__dict__["_node"] = self.__dict__["_node"]
        return element

    def _axis_sizes(self) -> tuple:
        """Per-axis count of DISTINCT control points.

        Lattice: ``s/t/uDivisions``. NURBS surface: ``numCVsInU/V`` minus the
        degree on any periodic axis -- the trailing ``degree`` CVs are wraps of
        the first ``degree`` (e.g. a sphere is periodic in V, so its 11 V-CVs
        collapse to 8 distinct).

        Reads the *live* node name from the MPlug (rename-safe) -- not the
        cached ``_comp_node``.
        """
        node_name = self.node.name
        if cmds.nodeType(node_name) == "lattice":
            return (
                cmds.getAttr(f"{node_name}.sDivisions"),
                cmds.getAttr(f"{node_name}.tDivisions"),
                cmds.getAttr(f"{node_name}.uDivisions"),
            )
        sel = OpenMaya.MSelectionList()
        sel.add(node_name)
        fn       = OpenMaya.MFnNurbsSurface(sel.getDagPath(0))
        periodic = OpenMaya.MFnNurbsSurface.kPeriodic
        u        = fn.numCVsInU - (fn.degreeInU if fn.formInU == periodic else 0)
        v        = fn.numCVsInV - (fn.degreeInV if fn.formInV == periodic else 0)
        return (u, v)


def _maybe_component_plug(
    attr_name: str, attr_obj: Attribute
) -> "ComponentPlug | None":
    """Return a :class:`ComponentPlug` *handle* if ``attr_name`` addresses a
    multi-dimensional component on ``attr_obj``'s node (NURBS-surface ``cv``,
    lattice ``pt``), else ``None``.

    Called from :meth:`Node.__getattr__` to upgrade ``node.cv`` / ``node.pt``
    on those node types so ``node.cv[u, v]`` / ``node.pt[s, t, u]`` index in
    numpy style. ``None`` (the common case) leaves the caller's plain
    :class:`Plug` wrap untouched -- 1-D components (mesh ``vtx``, curve ``cv``,
    mesh ``map``/``uv``) and flat ``controlPoints`` access are unchanged.
    """
    if attr_name not in _MULTIDIM_ALIASES:
        return None
    try:
        node_name = attr_obj.node.name
        node_type = cmds.nodeType(node_name)
    except (RuntimeError, AttributeError):
        return None
    spec = _MULTIDIM_COMPONENTS.get(node_type)
    if spec is None:
        return None
    ndims, aliases = spec
    if attr_name not in aliases:
        return None
    # the caller binds ``attr_obj``'s owner; with none, the handle of its node
    # ``attr_obj`` took checks it
    handle = attr_obj.__dict__.get("_handle1")
    return _component_plug(attr_obj.plug, node_name, attr_name, ndims, None, handle)


def _component_plug(
    mplug:       Any,
    comp_node:   str,
    comp_alias:  str,
    comp_ndims:  int,
    comp_coords: tuple | None,
    handle:      Any,
) -> "ComponentPlug":
    """``ComponentPlug(mplug, comp_node, comp_alias, comp_ndims, comp_coords)`` of
    a node the caller holds, without the handle of its node ``__init__`` takes of
    an MPlug (a name lookup): the caller binds its owner, or ``handle`` (the API
    1.0 handle of that node, see ``_ensure_owner_alive``) checks it. A name the
    component translation left (not an MPlug) is built by ``__init__``."""
    if not isinstance(mplug, OpenMaya.MPlug):
        return ComponentPlug(mplug, comp_node, comp_alias, comp_ndims, comp_coords)
    plug = ComponentPlug.__new__(
        ComponentPlug, mplug, comp_node, comp_alias, comp_ndims, comp_coords
    )
    plug.__dict__.update(_attr_state(mplug, handle))
    plug._set_component(comp_node, comp_alias, comp_ndims, comp_coords)
    return plug


# --------------------------------------------------------------------- #
#  Module-level helpers used by `<<` and `_shorthand`
# --------------------------------------------------------------------- #


def _disconnect_incoming(dst: Any) -> None:
    """Disconnect any source plug currently driving ``dst``.

    Uses :meth:`Attribute.get_connected_attrs` (MPlug-based, rename-safe)
    and :meth:`Attribute.disconnect` instead of raw ``cmds`` calls.

    Disconnection is permitted even when ``dst`` is **locked** -- lock guards
    *setting / connecting*, not *removing* a driver. A locked ``dst`` is
    temporarily unlocked for the disconnect and then re-locked, preserving
    the user's lock intent. A genuine disconnect failure raises
    :class:`InjectionError` instead of being swallowed to ``LOGGER.debug``.
    """
    if not isinstance(dst, Attribute):
        # Allow raw strings only as a fallback path.
        try:
            dst = Attribute(str(dst))
        except Exception as e:
            LOGGER.debug("could not coerce %s to Attribute: %s", dst, e)
            return

    try:
        sources = dst.get_connected_attrs(src=True, dst=False) or []
    except RuntimeError as e:
        raise InjectionError(
            f"Cannot inspect incoming connections on {str(dst)!r}: {e}"
        ) from e

    if not sources:
        return

    _disconnect_sources_respecting_lock(dst, sources)


def _query_connections(plug: Any, source: bool) -> Any:
    """Return plugs connected to ``plug`` as a ``List`` of ``Plug``.

    ``source=True`` returns the incoming driver (0 or 1 element);
    ``source=False`` returns every outgoing destination. Direct
    connections only -- a compound whose children are driven reports
    nothing; slice it with ``[:]`` to query per-child.
    """
    from rig._internal.list import List

    # a freed node's MPlug points at freed memory
    _ensure_owner_alive(plug)
    return List([Plug(mp) for mp in plug.plug.connectedTo(source, not source)])


def _lock_chain_names(dst: Attribute) -> list:
    """Return cmds-usable plug names for ``dst`` and each ancestor, leaf -> root.

    Walks the MPlug parent chain -- compound children via ``parent()``, array
    elements via ``array()`` -- so multi-level compounds and array elements are
    handled uniformly. Each plug is named via :class:`Attribute` (partial node
    name + alias) rather than raw ``MPlug.name()`` so duplicate short node names
    and attribute aliases resolve to a name ``cmds`` accepts. ``dst`` is an
    already-resolved Attribute whose connections the caller just inspected, so
    its plug chain is valid.
    """
    names = [dst.full_name]
    cur   = dst.plug
    while True:
        parent = None
        if cur.isChild:
            parent = cur.parent()
        elif cur.isElement:
            parent = cur.array()
        if parent is None:
            break
        # named at once (``dst``'s node was just read), never held: no handle
        names.append(_new_attr(Attribute, parent).full_name)
        cur = parent
    return names


def _relock_chain(saved: list) -> None:
    """Re-lock every plug :func:`_unlock_lock_chain` had to unlock.

    A re-lock failure is logged, never raised: the disconnect already succeeded,
    and masking an in-flight exception with a cleanup error would hide the real
    failure.
    """
    for name, was_locked in saved:
        if was_locked:
            try:
                cmds.setAttr(name, lock=True)
            except RuntimeError as relock_err:
                LOGGER.error(
                    "could not re-lock %s after disconnect (left unlocked): %s",
                    name,
                    relock_err,
                )


def _unlock_lock_chain(dst: Attribute) -> list:
    """Make ``dst`` effective-unlocked for a disconnect; return restore info.

    An effective-locked plug may owe its lock to its OWN flag, to a locked
    ancestor compound / array, or to several levels at once. Only the plugs
    along ``dst``'s parent chain that actually carry an OWN lock are unlocked --
    read top-down (root -> leaf) so each plug's own flag is visible once its
    ancestors are already unlocked. Returns ``[(plug_name, was_own_locked),
    ...]`` so :func:`_relock_chain` can restore the lock topology byte-for-byte;
    an inherited lock is therefore NEVER converted into an explicit own-lock. If
    a needed unlock fails, any unlock already done is rolled back and
    :class:`InjectionError` is raised.
    """
    chain = _lock_chain_names(dst)
    saved: list = []
    try:
        for name in reversed(chain):  # root -> leaf: ancestors unlocked first
            own_locked = bool(cmds.getAttr(name, lock=True))
            saved.append((name, own_locked))
            if own_locked:
                cmds.setAttr(name, lock=False)
    except RuntimeError as e:
        _relock_chain(saved)  # never leave the chain partially unlocked
        raise InjectionError(
            f"Cannot unlock {str(dst)!r} to break its driver: {e}"
        ) from e
    return saved


def _lock_chain_is_unlocked(dst: Attribute) -> bool:
    """Return ``True`` when no plug on ``dst``'s lock chain is locked.

    Walks the same MPlug parent chain as :func:`_lock_chain_names`, reading
    each plug's ``isLocked`` (which matches ``cmds.getAttr(lock=True)`` per
    plug). The leaf alone is not enough: a locked array root blocks the
    disconnect of its elements, which do not always report it themselves.
    """
    cur = dst.plug
    while True:
        if cur.isLocked:
            return False
        if cur.isChild:
            cur = cur.parent()
        elif cur.isElement:
            cur = cur.array()
        else:
            return True


def _disconnect_sources_respecting_lock(dst: Attribute, sources: list) -> None:
    """Remove every ``src -> dst`` connection, honoring the lock contract.

    ``<< None`` disconnects even a locked dst -- lock guards *set / connect*,
    not *removal*. The dst, plus any locked ancestor that makes it
    effective-locked, is temporarily unlocked, every driver removed, then the
    exact lock topology restored. The behavior is identical whether the lock
    sits on the leaf's own flag or is inherited from an ancestor compound, and a
    child's inherited lock is never converted into an own-lock. A genuine
    disconnect failure raises :class:`InjectionError`; the unlock / re-lock
    toggles are guarded so a lock-toggle failure becomes a clean
    ``InjectionError`` (unlock) or a logged warning (re-lock) that never masks
    an in-flight error.
    """
    if _lock_chain_is_unlocked(dst):
        # nothing to unlock / relock: skip the by-name chain walk
        try:
            for src_attr in sources:
                src_attr.disconnect(dst)
        except RuntimeError as e:
            raise InjectionError(
                f"Cannot disconnect {str(dst)!r} from its driver: {e}"
            ) from e
        return

    saved = _unlock_lock_chain(dst)
    try:
        for src_attr in sources:
            src_attr.disconnect(dst)
    except RuntimeError as e:
        raise InjectionError(
            f"Cannot disconnect {str(dst)!r} from its driver: {e}"
        ) from e
    finally:
        _relock_chain(saved)


def _locked_channels(dst: Any, leaves: list | None = None) -> list:
    """Return the locked leaf plug name(s) among ``dst``'s channels.

    For a compound plug this is every locked child; for a scalar it is
    ``dst`` itself if locked. Used to enforce the **all-or-nothing** rule:
    a compound set with any locked child must raise before mutating any
    channel.

    ``leaves`` is ``dst``'s channel list when the caller already built it
    (:func:`_inject_value` does); ``None`` resolves it here.
    """
    types = _lazy().types

    if not isinstance(dst, Attribute):
        return []
    if leaves is None:
        try:
            leaves = types._get_compound(dst) if types._is_compound(dst) else [dst]
        except Exception:
            leaves = [dst]

    locked = []
    for leaf in leaves:
        leaf_attr = leaf if isinstance(leaf, Attribute) else None
        if leaf_attr is None:
            try:
                leaf_attr = Attribute(str(leaf))
            except Exception:
                continue
        try:
            if leaf_attr.is_locked:
                locked.append(str(leaf_attr))
        except Exception:
            pass
    return locked


def _spec_slot_channels(dst: Any, src: Any) -> frozenset:
    """Names of ``dst``'s channels whose ``src`` slot is an ``_AttrSpec``.

    A spec slot is a modifier application (``skip`` / ``lock`` / ``unlock``
    / ``hide`` / ...), not a value-set or a connect, so it must not trip the
    all-or-nothing lock check. On a scalar it already doesn't:
    ``Plug.__lshift__`` returns at its ``_is_attribute_spec`` branch, before
    :func:`_inject_value` ever runs. This keeps the per-channel fan-out
    consistent with that, so ``ctrl.t << [skip, 4.0, skip]`` is not vetoed by
    a locked ``ctrl.tx`` the caller explicitly asked to leave alone.
    """
    types = _lazy().types

    if isinstance(src, (Attribute, str)) or not types._is_sequence(src):
        return frozenset()
    if not any(types._is_attribute_spec(x) for x in src):
        return frozenset()

    try:
        leaves = types._get_compound(dst) if types._is_compound(dst) else [dst]
    except Exception:
        leaves = [dst]

    # Mirror the fan-out's own pairing (it truncates an over-long source
    # before broadcasting) so the exemption lines up channel-for-channel.
    slots = list(src)[: len(leaves)]
    return frozenset(
        str(leaf)
        for slot, leaf in sequences(slots, leaves)
        if types._is_attribute_spec(slot)
    )


def _assert_settable(
    dst: Any, exempt: frozenset = frozenset(), leaves: list | None = None
) -> None:
    """Raise :class:`InjectionError` if ``dst`` (or any compound child) is locked.

    A locked attribute is the one inviolable barrier for ``<<`` value-set /
    connect. Checking up front (before any disconnect or per-channel write)
    guarantees the all-or-nothing contract: a compound with a single locked
    child mutates nothing, and a locked connect target keeps its existing
    connection rather than being disconnected by a doomed re-wire.

    ``exempt`` names channels that are not being set or connected at all --
    see :func:`_spec_slot_channels`. ``leaves`` is passed through to
    :func:`_locked_channels`.
    """
    locked = [name for name in _locked_channels(dst, leaves) if name not in exempt]
    if locked:
        names = locked[0] if len(locked) == 1 else locked
        raise InjectionError(
            f"Cannot set/connect {str(dst)!r}: locked attribute(s) {names}. "
            f"Unlock first (e.g. `plug << unlock` or "
            f"`cmds.setAttr(attr, lock=False)`)."
        )


def _do_set(dst_attr: Any, *set_args: Any, **set_kwargs: Any) -> None:
    """Set a value on ``dst_attr``, breaking an incoming connection if needed.

    ``<<`` asserts the destination's desired value, so an existing incoming
    connection is overwritten: the first set attempt that fails because the
    plug is connected triggers a disconnect (surfaced at ``LOGGER.info``) and
    a retry. Lock is assumed pre-checked by :func:`_assert_settable`, so a
    failure that is *not* a connection is a genuine error and raises
    :class:`InjectionError` rather than being swallowed.
    """
    try:
        dst_attr.set(*set_args, **set_kwargs)
        return
    except (RuntimeError, TypeError) as first:
        try:
            incoming = dst_attr.get_connected_attrs(src=True, dst=False) or []
        except Exception:
            incoming = []
        if incoming:
            LOGGER.info(
                "set %s: breaking incoming connection(s) %s to assign value",
                str(dst_attr),
                [str(i) for i in incoming],
            )
            _disconnect_incoming(dst_attr)
            try:
                dst_attr.set(*set_args, **set_kwargs)
                return
            except (RuntimeError, TypeError) as second:
                raise InjectionError(
                    f"Cannot set {str(dst_attr)!r}: {second}"
                ) from second
        raise InjectionError(f"Cannot set {str(dst_attr)!r}: {first}") from first


# NOTE: a former ``_attr_from_string(plug_string)`` helper was removed.
# It was equivalent to ``Attribute(plug_string)`` since the latter handles
# the MSelectionList round-trip internally. Use ``Attribute(s)`` directly.


def _do_destroy(
    plug_str: str,
    strict:   bool = False,
    silent:   bool = False,
    verbose:  bool = False,
) -> None:
    """v4.S: delete a Maya attribute, central impl for ``destroy`` DSL.

    Backs both forms of the ``destroy`` modifier:

      * ``plug << destroy`` (sentinel form) -- handled by
        :class:`rig.spec.modifiers._DestroyMarker.apply`
      * ``node << destroy("foo")`` (callable form) -- handled by
        :class:`rig.spec.modifiers._DestroySpec.apply`

    Behavior policy:

      * **Default**: matches Maya's ``cmds.deleteAttr`` convention --
        auto-disconnects any incoming/outgoing connections, then
        deletes. INFO log surfaces when connections were severed.
      * ``strict=True``: pre-checks for connections; raises
        :class:`RuntimeError` if any exist (no deletion).
      * ``silent=True``: no-op if the attr doesn't exist (cleanup-
        script idempotent).
      * ``verbose=True``: log each broken connection by name (uses
        ``LOGGER.info`` per-connection instead of a count summary).

    Edge cases:

      * **Locked attr**: let Maya raise (don't auto-unlock -- footgun).
      * **Compound parent with children**: Maya cascades.
      * **Published attr on container**: auto-unpublish via
        ``cmds.containerPublish(unpublishName=...)`` BEFORE delete to
        avoid dangling published-attr table references.
      * **Multi-element** (e.g. ``node.matrixIn[3]``): Maya supports
        per-element delete; passes through.
      * **System attr** (``translate``, etc.): Maya rejects.
      * **Driven keys**: out of scope; let Maya raise if it complains.

    Raises
    ------
    AttributeError
        If the attr doesn't exist and ``silent`` is False.
    RuntimeError
        If ``strict`` is True and the attr has any connections, OR if
        Maya itself rejects the deletion (locked, system, etc.).
    ValueError
        If both ``strict`` and ``silent`` are True (caller error).
    """
    if strict and silent:
        raise ValueError("_do_destroy: strict=True and silent=True are incompatible")

    plug_str = str(plug_str)

    # Step 0 -- native published-name teardown. With native container publishing,
    # ``container.<name>`` is a ``publishName`` / ``bindAttr`` ALIAS (single) or
    # a registry-resolved attr on the host (multi), NOT a plain attr on the
    # container. A plain ``cmds.deleteAttr`` on the alias clears the binding but
    # leaves the published NAME dangling, and a registry multi has no real
    # ``container.<name>`` at all (so the existence check below would wrongly
    # reject it). Detect a published name on a container FIRST and tear it down
    # via ``unbindAttr`` / ``unpublishName`` (+ registry/host-carrier cleanup).
    # Genuine attrs added with ``ctn << Float(...)`` are not published, so
    # ``_destroy_published_name`` returns False and they fall through below.
    node_str, sep, attr_name = plug_str.partition(".")
    if sep and "." not in attr_name and cmds.objExists(node_str):
        if cmds.nodeType(node_str) == "container":
            from rig._internal.container import _destroy_published_name

            if _destroy_published_name(node_str, attr_name):
                return

    # Step 1 -- existence check.
    if not cmds.objExists(plug_str):
        if silent:
            return
        raise AttributeError(
            f"Cannot destroy nonexistent attribute {plug_str!r} "
            f"(use destroy(..., silent=True) to skip missing attrs)"
        )

    # Step 2 -- query connections (used by both strict and verbose paths).
    incoming = (
        cmds.listConnections(plug_str, source=True, destination=False, plugs=True) or []
    )
    outgoing = (
        cmds.listConnections(plug_str, source=False, destination=True, plugs=True) or []
    )
    n_in, n_out = len(incoming), len(outgoing)

    # Step 3 -- strict mode: refuse to delete if any connection exists.
    if strict and (n_in or n_out):
        details = []
        for src in incoming:
            details.append(f"  {src} -> {plug_str}")
        for dst in outgoing:
            details.append(f"  {plug_str} -> {dst}")
        raise RuntimeError(
            f"Cannot destroy {plug_str!r}: has {n_in} incoming + "
            f"{n_out} outgoing connection(s):\n"
            + "\n".join(details)
            + f"\n(disconnect first or use destroy(..., strict=False) to "
            f"auto-disconnect)"
        )

    # Step 4 -- surface the side effect when connections will be severed.
    if n_in or n_out:
        if verbose:
            for src in incoming:
                LOGGER.info(
                    "destroy %s: disconnecting %s -> %s", plug_str, src, plug_str
                )
            for dst in outgoing:
                LOGGER.info(
                    "destroy %s: disconnecting %s -> %s", plug_str, plug_str, dst
                )
        else:
            LOGGER.info(
                "destroy %s: auto-disconnecting %d incoming + %d outgoing",
                plug_str,
                n_in,
                n_out,
            )

    # Step 5 -- note: native container publishing (``publishName`` /
    # ``bindAttr``) is handled in Step 1.5 above. Reaching here means a plain
    # attribute (inner-node attr or a genuine ``ctn << Float(...)`` attr).
    # When a published source plug is deleted here, Maya auto-clears its
    # ``bindAttr`` entry but leaves the container's published NAME in place
    # (dangling, no driver) -- preserving the published API for re-binding.
    # Users who want to remove the published name too should destroy it on the
    # container node: ``container_node << destroy("published_name")``.

    # Step 6 -- actually delete. Let Maya raise on locked/system attrs.
    cmds.deleteAttr(plug_str)


def _attribute_type_or_none(dst: Any) -> str | None:
    """``dst.attribute_type``, or None when the query raises.

    Read as the last test of :func:`_inject_value`'s bare-multi-root check,
    so the ``cmds.attributeQuery`` runs for multi roots only.
    """
    try:
        return dst.attribute_type
    except (AttributeError, RuntimeError, TypeError):
        return None


def _inject_value(dst: Any, src: Any) -> None:
    """Inject ``src`` into ``dst`` -- set / connect with compound-aware logic.

    Mirrors the original ``Container.inject`` (Eric's _language.py:1158-1294)
    but adapted to use ``Attribute.is_multi`` / ``num_children`` /
    ``data_type`` instead of his ``__data__`` namespace.

    Source-shape rules (NumPy-style strict, plus matrix-source routing):

    1. ``src is None`` => disconnect any incoming connection on ``dst``.
    2. ``dst.data_type == "string"`` => ``cmds.setAttr(..., type="string")``.
    3. Sequence/array ``src`` (numpy, list, tuple) -- try matrix-source
       routing first via :mod:`rig._internal.decompose` (handles
       ``transform.matrix`` / ``.worldMatrix`` / ``.t`` / ``.r`` / ``.s``
       / ``.shear`` / quaternion compounds and non-transform matrix
       attrs). If that routes, return.
    4. If routing didn't match: NumPy-style strict shape validation --
       the raveled source size must match dst's channel count, OR be 1
       (broadcast scalar). Mismatches raise :class:`ValueError`.
    5. Otherwise the existing compound-detection / fan-out logic handles
       the (Plug source) / (heterogeneous list with Plugs) cases.
    """
    types = _lazy().types

    # 1. Disconnect.
    if src is None:
        _disconnect_incoming(dst)
        return

    # Locked attributes are the one inviolable barrier for value-set /
    # connect (disconnect above is allowed on locked attrs). Check up front
    # so compound sets are all-or-nothing and a locked connect target keeps
    # its existing connection instead of being disconnected by a doomed
    # re-wire.
    #
    # ``dst``'s compound-ness and channels are pure queries, resolved once
    # here and reused by the lock check, the shape validation, the dispatch
    # and the fan-out below (recomputed if ``dst`` is re-indexed).
    compound_dst = types._is_compound(dst)
    try:
        leaves = (
            [dst.child(i) for i in range(dst.num_children)] if compound_dst else [dst]
        )
    except Exception:
        leaves = None
    _assert_settable(dst, exempt=_spec_slot_channels(dst, src), leaves=leaves)

    # Resolve dst data type once.
    try:
        dst_data_type = dst.data_type
    except Exception:
        dst_data_type = None

    # 2. Typed string -- trust the user.
    if dst_data_type == "string":
        # If src is a Plug/Attribute, this is a string->string CONNECTION,
        # NOT a literal-string SET. Fall through to ``_set_or_connect``
        # which handles the connection (cmds.connectAttr) correctly.
        # Without this, `texture.fileTextureName << shape.image` would
        # store the literal string ``"shape.image"`` as the dst's value
        # instead of building the live connection.
        if isinstance(src, Attribute):
            _set_or_connect(src, dst)
            return
        _do_set(dst, str(src), type="string")
        return

    # Bare multi-root with a sequence source -> element-wise write to
    # indices 0..len(src)-1. Auto-creates missing indices via single-int
    # Attribute indexing (Maya's create-on-write semantic).
    #
    # Handles both List sources (per-element connectAttr) and
    # numpy / list / tuple of values (per-element setAttr). Skipped when
    # ``src`` is a scalar / Attribute -- the existing auto-index path
    # below preserves Eric Vignola's ``multi << scalar`` auto-append idiom
    # for backwards compatibility.
    #
    # Also skipped for matrix-flavored multis (``multMatrix.matrixIn``,
    # ``addMatrix.matrixIn``, ``transform.worldMatrix`` etc.) so the
    # existing matrix-source-routing / matrix-chain auto-append paths
    # continue to handle them. Note: multi-of-matrix has
    # ``data_type == "compound"`` (the multi container type) and
    # ``attribute_type == "matrix"`` (the element type), so we must
    # check BOTH to catch every shape.
    #
    # Ambiguity for ``multi-of-double3 << [x, y, z]`` (length matches the
    # child count): we treat as multi-index write (per-index broadcast,
    # so each index gets the scalar broadcast as a vec3). Users who want
    # the old "append one compound value" semantic call ``multi.append(
    # [x, y, z])`` or ``multi[multi.next_index] << [x, y, z]``.
    #
    # Outside the query, reading ``attribute_type`` only builds the owning
    # node (and raises what a failed cast raises), so an Attribute dst builds
    # its node here and runs ``cmds.attributeQuery`` for a multi root only.
    lazy_attr_type = isinstance(dst, Attribute)
    dst_attr_type  = None
    try:
        if lazy_attr_type:
            dst.node
        else:
            dst_attr_type = dst.attribute_type
    except (AttributeError, RuntimeError, TypeError):
        lazy_attr_type = False
    try:
        is_bare_multi_root = (
            dst.is_multi
            and not str(dst).endswith("]")
            and dst_data_type != "matrix"
            and (_attribute_type_or_none(dst) if lazy_attr_type else dst_attr_type)
            != "matrix"
        )
    except (AttributeError, TypeError):
        is_bare_multi_root = False

    if is_bare_multi_root:
        List = _lazy().list.List

        # Multi src -> Multi dst -> per-element connect.
        # Iterate the source's existing logical indices and connect each
        # ``src[i]`` into ``dst[i]``. Handles sparse source indices too.
        # Without this branch, ``multi_dst << multi_src`` would fall
        # through to the auto-index path below and connect the whole src
        # multi root into a single new dst index (creating a spurious
        # extra index with default values -- see TestInjectMultiToMulti
        # for the regression).
        if isinstance(src, Attribute):
            try:
                src_is_multi = src.is_multi and not str(src).endswith("]")
            except (AttributeError, TypeError):
                src_is_multi = False
            if src_is_multi:
                try:
                    src_indices = src.get_logical_indices() or []
                except Exception:
                    src_indices = []
                for idx in src_indices:
                    dst[idx] << src[idx]
                return

        if isinstance(src, List):
            for i, elem in enumerate(src):
                dst[i] << elem
            return

        if (
            not isinstance(src, Attribute)
            and not isinstance(src, str)
            and not isinstance(src, numbers.Real)
            and types._is_sequence(src)
        ):
            for i, elem in enumerate(src):
                dst[i] << elem
            return

    # Auto-index multi roots: ``node.input1D << X`` => ``node.input1D[next] << X``.
    # Skipped when the user has already explicitly indexed (str(dst) ends in `]`).
    #
    # Uses ``rig.nodetypes.Attribute.get_next_available_index()`` which
    # returns the LOWEST free logical index (gap-fill semantics). For
    # sequential fills from an empty multi this is identical to Eric's
    # "highest physical index + 1"; only sparse arrays after deletions diverge.
    # An array of per-instance elements (``worldMatrix``) is not a collection:
    # read without an index it is the element of its path's instance, as cmds
    # resolves its name (``T2|S.worldMatrix`` is ``worldMatrix[1]``).
    try:
        if dst.is_multi and not str(dst).endswith("]"):
            next_index = _path_instance_number(dst)
            if next_index is None:
                try:
                    next_index = dst.get_next_available_index()
                except Exception:
                    next_index = 0
            dst = dst[next_index]
            try:
                dst_data_type = dst.data_type
            except Exception:
                dst_data_type = None
            compound_dst = types._is_compound(dst)
            leaves       = None
    except (AttributeError, TypeError):
        pass

    # 3 & 4: Sequence sources -- strict shape validation + matrix routing.
    if (
        not isinstance(src, Attribute)
        and not isinstance(src, str)
        and not isinstance(src, numbers.Real)
        and types._is_sequence(src)
    ):
        # If the sequence contains any Plug / Attribute references, skip
        # the numpy validation path entirely -- ``np.asarray`` would coerce
        # them to strings (because ``Plug.__str__`` returns the plug name
        # like ``"node.attr"``), giving a ``<U21`` string array rather than
        # ``dtype=object``. That silently breaks heterogeneous lists like
        # ``[node.tx, 0, node.tz]`` because the validation block then
        # dispatches plug-name STRINGS into ``_set_or_connect``, which sees
        # them as string-typed values and fails the setAttr against a
        # double dst. The fan-out logic below handles heterogeneous
        # element-by-element correctly (Plug -> connect, Real -> set).
        has_attribute_refs = any(isinstance(x, Attribute) for x in src)

        if has_attribute_refs:
            arr = None
        else:
            try:
                arr = np.asarray(src)
            except (ValueError, TypeError):
                arr = None

        if arr is not None and arr.dtype != object:
            # 3a. Matrix-source routing (transform channels, .matrix,
            #     .worldMatrix, quaternion compounds, non-transform matrix
            #     attrs). Handles 3x3 / 4x4 / 9-flat / 16-flat sources.
            from rig._internal.decompose import _try_matrix_source_routing

            if _try_matrix_source_routing(dst, arr):
                return

            # 3b. Direct matrix-typed dst with a flat-16 source -- route
            #     (via Tier B logic in _decompose). For 4x4-shaped sources
            #     the routing above would've handled it; for stray 16-flat
            #     into a non-transform we re-check here.
            flat = arr.ravel()

            if dst_data_type == "matrix":
                if flat.size == 16:
                    _do_set(dst, *flat.tolist(), type="matrix")
                    return
                raise ValueError(
                    f"Cannot inject sequence of size {flat.size} into "
                    f"matrix attribute {str(dst)!r} (expected 9 for 3x3 "
                    f"or 16 for 4x4)."
                )

            # 4. Compound dst (vector / quat / euler) -- must match channel
            #    count exactly OR be a 1-element broadcast.
            if compound_dst:
                n = dst.num_children
                if flat.size == n:
                    src = flat.tolist()
                elif flat.size == 1:
                    src = float(flat.item())  # scalar broadcast
                else:
                    raise ValueError(
                        f"Cannot inject sequence of size {flat.size} "
                        f"into {n}-channel compound attribute "
                        f"{str(dst)!r}."
                    )
            else:
                # Scalar dst -- only size-1 sequences accepted.
                if flat.size == 1:
                    src = flat.item()
                else:
                    raise ValueError(
                        f"Cannot inject sequence of size {flat.size} "
                        f"into scalar attribute {str(dst)!r} "
                        f"(expected 1 element)."
                    )
        # else: heterogeneous (list-with-Plugs etc.) -- fall through to
        # existing compound-detection logic below.

    compound_src = types._is_compound(src) and isinstance(src, Attribute)

    # compound->compound, or attr->attr.
    if (compound_src and compound_dst) or (not compound_src and not compound_dst):
        _set_or_connect(src, dst, _dst_compound=compound_dst)
        return

    # compound->generic-slot: when dst is a polymorphic untyped attribute
    # (kGenericAttribute, reports data_type as "Tdata" / "typed" before
    # being typed by its first connection), Maya's connectAttr accepts
    # the full compound and types the slot from the source. DON'T fan
    # out per-channel -- that would only connect the first compound child
    # (e.g. ``src.scaleX -> choice.input[0]``) and silently drop Y/Z data.
    # Used by ``functions.choice`` with compound inputs (matrix.decompose
    # outputScale, etc.) and any other DSL caller routing a compound
    # plug through a generic-typed multi.
    if compound_src and not compound_dst and dst_data_type in ("Tdata", "typed", None):
        _set_or_connect(src, dst)
        return

    # Otherwise, fan out per channel.
    src_channels = (
        types._get_compound(src) if compound_src or types._is_sequence(src) else [src]
    )
    if not compound_dst:
        dst_channels = [dst]
    elif leaves is not None:
        dst_channels = leaves
    else:
        dst_channels = types._get_compound(dst)

    if len(dst_channels) < len(src_channels):
        src_channels = src_channels[: len(dst_channels)]

    for s, d in sequences(src_channels, dst_channels):
        _fanout_channel(s, d)


def _fanout_channel(src: Any, dst: Any) -> None:
    """Apply one source slot to one destination channel.

    Gives a per-channel slot the same vocabulary a whole plug has: ``None``
    disconnects (as ``plug << None`` does), an ``_AttrSpec`` applies itself
    (``lock`` / ``hide`` / ``skip``), everything else is a set-or-connect.
    A collection spec has no per-channel meaning and raises.
    """
    types = _lazy().types

    if src is None:
        _disconnect_incoming(dst)
    elif types._is_attribute_spec(src):
        src.apply(dst)
    elif types._is_member_spec(src):
        raise TypeError(
            f"{src!r} is a collection spec and cannot be fanned into channel "
            f"{dst}; inject it into the node or its components"
        )
    else:
        _set_or_connect(src, dst)


def _set_or_connect(src: Any, dst: Any, _dst_compound: bool | None = None) -> None:
    """Either ``dst.set(src)`` or ``src.connect(dst, force=True)``.

    Uses :class:`rig.nodetypes.Attribute` API methods rather than raw
    ``cmds.*`` calls so that:

    * Connections / sets are rename-safe (handle MObjectHandle internally).
    * Auto-inserted ``unitConversion`` nodes (Maya inserts these for unit
      mismatches like ``doubleLinear`` <-> ``double``) are absorbed into the
      active container scope via :meth:`container.absorb_unit_conversions`.
      Without absorption these nodes orphan to scene root and break
      Node Editor container collapse.

    ``_dst_compound`` is ``_is_compound(dst)`` when the caller already
    resolved it for this same ``dst`` Attribute (:func:`_inject_value`
    does); ``None`` resolves it here.
    """
    if src is None:
        return

    # Lazy-bound to avoid circular dep at module load.
    lazy      = _lazy()
    container = lazy.container.container

    # Coerce dst to Attribute if it isn't already, so we can use the API.
    if not isinstance(dst, Attribute):
        try:
            dst_attr = Attribute(str(dst))
        except Exception:
            dst_attr = None
        # The hint describes ``dst`` itself, not a coerced copy.
        _dst_compound = None
    else:
        dst_attr = dst

    # Plug / Attribute source => connect.
    if isinstance(src, Attribute):
        if "." in str(src):
            _disconnect_incoming(dst_attr if dst_attr is not None else dst)
            try:
                if dst_attr is not None:
                    src.connect(dst_attr, force=True)
                    container.absorb_unit_conversions(dst_attr)
                else:
                    # Fallback for raw-string dst we couldn't coerce.
                    cmds.connectAttr(str(src), str(dst), force=True)
                    container.absorb_unit_conversions(str(dst))
            except RuntimeError as e:
                raise InjectionError(
                    f"Cannot connect {str(src)!r} -> {str(dst)!r}: {e}"
                ) from e
        return

    # Plain numeric.
    if isinstance(src, numbers.Real):
        if dst_attr is None:
            # Raw-string dst we couldn't coerce to Attribute.
            try:
                cmds.setAttr(str(dst), src)
            except (RuntimeError, TypeError) as e:
                raise InjectionError(f"Cannot set {str(dst)!r}: {e}") from e
            return
        # Compound-numeric dst: broadcast the scalar across children.
        # Lock was already checked upstream (all-or-nothing); failures here
        # are connection-overwrites or genuine errors, both handled by _do_set.
        if _dst_compound is None:
            _dst_compound = lazy.types._is_compound(dst_attr)
        if _dst_compound:
            for child in lazy.types._get_compound(dst_attr):
                child_attr = (
                    child if isinstance(child, Attribute) else Attribute(str(child))
                )
                _do_set(child_attr, src)
        else:
            _do_set(dst_attr, src)
        return

    # String.
    if isinstance(src, str):
        if dst_attr is not None:
            _do_set(dst_attr, src, type="string")
        else:
            try:
                cmds.setAttr(str(dst), src, type="string")
            except (RuntimeError, TypeError) as e:
                raise InjectionError(f"Cannot set {str(dst)!r}: {e}") from e
        return

    # Sequence (list / tuple).
    if lazy.types._is_sequence(src):
        # Matrix special-case.
        try:
            dt = dst.data_type
        except Exception:
            dt = None

        if dt == "matrix" and len(src) == 16:
            if dst_attr is not None:
                _do_set(dst_attr, *src, type="matrix")
            else:
                try:
                    cmds.setAttr(str(dst), *src, type="matrix")
                except (RuntimeError, TypeError) as e:
                    raise InjectionError(f"Cannot set {str(dst)!r}: {e}") from e
            return

        # Try a vector / quaternion fan-out.
        dst_channels = lazy.types._get_compound(dst)
        if len(dst_channels) < len(src):
            src = src[: len(dst_channels)]

        for s, d in sequences(list(src), dst_channels):
            _fanout_channel(s, d)
        return

    # Fall through.
    # Helpful error for the bare-Node-as-source case (Pattern D, v3.R):
    # we intentionally don't auto-promote `Node` to `node.matrix` /
    # `node.wm` because the caller has to choose between local and world
    # space -- auto-promoting would hide which one is being used. Make
    # the choice explicit.
    if isinstance(src, lazy.node.Node):
        try:
            dt = dst_attr.data_type if dst_attr is not None else None
        except Exception:
            dt = None
        if dt == "matrix":
            raise TypeError(
                f"Cannot inject bare Node {str(src)!r} into matrix Plug {str(dst)!r}. "
                f"Specify the source explicitly: "
                f"'{dst} << {src}.wm' (world matrix) or "
                f"'{dst} << {src}.matrix' (local matrix)."
            )

    raise TypeError(f"Don't know how to inject {type(src).__name__} into Plug({dst})")


# The typed nodes' attribute lookup (``DGNode.__getattr__``) returns Plugs and
# upgrades multi-dimensional components once this module has loaded (D31).
from rig.nodetypes import dg_node as _dg_node_module  # noqa: E402

_dg_node_module._PLUG_CLASS     = Plug
_dg_node_module._COMPONENT_PLUG = _maybe_component_plug
