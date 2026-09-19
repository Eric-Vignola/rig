"""
Shader conversion: the engine behind ``Phong(mat)``, ``mat.astype("phong")``
and ``shade.convert(mat, "phong")``.

Maya has no node-type mutation. Its own Attribute Editor "Type" dropdown is
``createNode`` + MEL ``replaceNode`` + ``delete``: a brand-new node, the
name lost, ``materialInfo.material`` left dead, every value and wire of an
attribute the target lacks dropped in a MEL ``catch``, dynamic attributes
and locks gone, and -- because ``cmds.delete`` cascades into every
``createNode``-made node whose only links went into the shader -- the
textures, utilities and animCurves behind those dropped wires destroyed.

This engine is the same kind of replacement done carefully, in one undo
chunk named ``rig.shade.convert``:

* SCAN (pure reads): every attribute of the old node is classified against
  the target TYPE (``attributeQuery(type=...)``, no scratch node) -- present
  with the same ``attributeType`` (and ``listEnum``, multi-ness, children)
  or not. That gate is never bypassed: ``copyAttr`` of a float into a
  ``rampShader`` ramp array segfaults mayapy. Only NON-default values cross
  (the target's own factory defaults win for untouched attributes, so
  ``blinn.specularColor`` 0.5 grey never becomes half-specular on a white
  default); wires in and out are split into carried and doomed; dynamic
  attributes, locks, the shading engines, materialInfos, the
  ``defaultShaderList1`` slot, the Maya container and the full name are
  captured.
* REPORT: a frozen :class:`Conversion`; ``bool(report)`` means something is
  parked or lost, ``str(report)`` is the warning text. ``dry_run`` returns
  it with zero writes; ``strict`` raises ``ValueError(str(report))``;
  otherwise ONE ``cmds.warning`` before anything moves.
* PARK (the default): every doomed value, wire and animCurve is cloned onto
  the SAME node as a hidden dynamic attribute of the same type
  (``cosinePower`` -> ``__cosinePower__``), the wire re-homed onto it. Parked
  attributes ride through the conversion as dynamic attributes, so the
  doomed source nodes are never disconnected, orphaned or cascade-deleted,
  and they are RESTORED -- value or wire, lock included -- the next time the
  material becomes a type that has the attribute. ``park=False`` disconnects
  the doomed wires instead (the sources survive as orphans, the values are
  lost, both named in the warning).
* PREPARE / COMMIT: raw ``createNode`` (no ``__rig__`` tag, no scope
  enrolment, no second ``defaultShaderList1`` entry), dynamic attributes
  cloned, ``copyAttr`` for values then for connections (parents union
  leaves, ``message`` included so the materialInfo, the
  ``defaultShaderList1`` index and the container membership move exactly
  once), locks re-applied, the result verified BEFORE the old node is
  deleted, then the rename back to the old name asserted.
* ABORT: the chunk is closed and undone only when it recorded something
  (``undoInfo(undoName)`` is the chunk's name); an unguarded ``cmds.undo()``
  after an empty chunk would undo the user's previous action.

Documented limits: Maya's own AE dropdown wipes dynamic attributes, so
parked state survives rig-driven round trips and scene saves but not the
artist's dropdown; ``duplicate`` copies parked attributes (harmless, a
duplicate converted later restores them too); keyable / channelBox flags of
built-in attributes are not preserved; an expression string driving a lost
attribute is not rewritten. Live ``Node`` / ``Plug`` wrappers of the old
node are poisoned (``already deleted!``) and nothing is rebound: the spec
re-resolves by name and is the surviving handle.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

from maya import cmds
from rig._internal.memoize import prune_memoize_caches
from rig._internal.node import Node
from rig._internal.plug import _disconnect_incoming, _do_destroy, Plug
from rig._internal.undo import _undo_chunk
from rig.spec._base import _spec_from_attribute


LOGGER = logging.getLogger(__name__)

CHUNK  = "rig.shade.convert"
SUFFIX = "__rigconvert"

# Housekeeping attributes that are never data (``message`` moves on its own).
PLUMBING = frozenset({
    "caching",
    "frozen",
    "nodeState",
    "binMembership",
    "isHistoricallyInteresting",
    "message",
})

# The attributeTypes a parked attribute can be born with (``addAttr -at``);
# a doomed attribute of any other type is disconnected and named instead.
_SCALARS = frozenset({
    "float", "double", "short", "long", "byte", "char", "bool", "enum",
    "doubleAngle", "doubleLinear",
})

_PARKED_RE = re.compile(r"__(.+)__")
_DSL1_RE   = re.compile(r"defaultShaderList1\.shaders\[(\d+)\]")


# ---------- The report ---------------------------------------------------- #


@dataclass(frozen=True)
class Conversion:
    """What one conversion carried, parked and lost. ``bool(report)`` is
    "something was parked or lost"; ``str(report)`` is the warning text,
    lines in a fixed order: animation, wire, value, output, meaning, opaque.

    ``lost_values`` are ``(attr, value, locked)``; ``lost_wires``
    ``(source plug, attr, source node type)``; ``lost_animation``
    ``(curve, attr)``; ``lost_outputs`` ``(attr, destination plug)``;
    ``default_shift`` ``(attr, value, source default, target default)`` for a
    non-default value that crossed between types whose factory defaults
    differ; ``opaque`` the attributes whose value could not be compared
    (ramp arrays, typed values); ``dropped_kwargs`` the pending kwargs a lazy
    retype pruned; ``parked`` the entries of the lost lists that were parked
    on the node instead of lost (their names as the lines print them);
    ``restored`` the attributes an earlier park gave back this time."""

    name:           str
    source:         str
    target:         str
    carried:        tuple = ()
    dynamic:        tuple = ()
    lost_values:    tuple = ()
    lost_wires:     tuple = ()
    lost_animation: tuple = ()
    lost_outputs:   tuple = ()
    default_shift:  tuple = ()
    opaque:         tuple = ()
    dropped_kwargs: tuple = ()
    parked:         tuple = ()
    restored:       tuple = ()

    def __bool__(self) -> bool:
        return bool(
            self.lost_values or self.lost_wires or self.lost_animation
            or self.lost_outputs or self.opaque or self.parked
        )

    def __str__(self) -> str:
        if not self and not self.default_shift:
            return ""
        parked = set(self.parked)
        verb   = "parks" if parked else "loses" if self else "notes"
        lines  = [f"rig.shade: {self.name!r} {self.source} -> {self.target} {verb}:"]
        for curve, attr in self.lost_animation:
            note = (
                f"restored when {self.name!r} next becomes a type with {attr}"
                if attr in parked
                else "curve kept in the scene with its keys, disconnected"
            )
            lines.append(f"   animation   {curve} -> {attr}   ({note})")
        for src, attr, _ in self.lost_wires:
            line = f"   wire        {src} -> {attr}"
            if attr not in parked:
                node  = src.split(".", 1)[0]
                line += f"   ({node} kept in the scene, disconnected; rig.cleanup() never sweeps it)"
            lines.append(line)
        for attr, value, locked in self.lost_values:
            lines.append(
                f"   value       {attr} = {_fmt(value)}" + (" [locked]" if locked else "")
            )
        for attr, dst in self.lost_outputs:
            lines.append(f"   output      {attr} -> {dst}   (disconnected)")
        for attr, value, before, after in self.default_shift:
            lines.append(
                f"   meaning     {attr} = {_fmt(value)} carried; default differs "
                f"({self.source} {_fmt_default(before)} -> {self.target} {_fmt_default(after)})"
            )
        for attr in self.opaque:
            lines.append(f"   opaque      {attr}")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return repr(value)


def _fmt_default(default: Any) -> str:
    """A factory default as ``listDefault`` gives it; a uniform compound
    prints as one number (``0.5`` for ``[0.5, 0.5, 0.5]``)."""
    if isinstance(default, (list, tuple)) and default and all(
        _close(v, default[0]) for v in default
    ):
        default = default[0]
    return _fmt(default)


# ---------- Reads --------------------------------------------------------- #


def _q(attr: str, flag: str, **where: Any) -> Any:
    """One ``attributeQuery`` flag, ``None`` when Maya refuses the read."""
    try:
        return cmds.attributeQuery(attr, **{flag: True}, **where)
    except RuntimeError:
        return None


def _shape(attr: str, **where: Any) -> tuple | None:
    """What makes two attributes the same kind: ``(attributeType, multi,
    child types, enum labels)``; ``None`` when ``attr`` does not exist there.
    ``where`` is ``node=`` for a live node or ``type=`` for a node type."""
    if not _q(attr, "exists", **where):
        return None
    attr_type = _q(attr, "attributeType", **where)
    if attr_type is None:
        return None
    children = _q(attr, "listChildren", **where) or []
    enums    = tuple(_q(attr, "listEnum", **where) or ()) if attr_type == "enum" else ()
    return (
        attr_type,
        bool(_q(attr, "multi", **where)),
        tuple(_q(child, "attributeType", **where) for child in children),
        enums,
    )


def _has(attr: str, old: str, dst: str) -> bool:
    """The target type holds ``attr`` writable and of the same kind -- the
    gate that keeps ``copyAttr`` off a type mismatch."""
    shape = _shape(attr, node=old)
    return (
        shape is not None
        and shape == _shape(attr, type=dst)
        and bool(_q(attr, "writable", type=dst))
    )


def _parkable(shape: tuple | None) -> bool:
    """A scalar or a three-child compound of one plain numeric type: what a
    hidden dynamic attribute of the same type can hold."""
    if shape is None:
        return False
    attr_type, multi, children, _ = shape
    if multi:
        return False
    if not children:
        return attr_type in _SCALARS
    return len(children) == 3 and len(set(children)) == 1 and children[0] in _SCALARS


def _element_type(shape: tuple) -> str:
    """The ``addAttr -at`` type of a parked attribute (its children's type
    for a compound)."""
    attr_type, _, children, _ = shape
    return children[0] if children else attr_type


def _root_of(node: str, leaf: str) -> str:
    """The top-level attribute a leaf belongs to (``transparencyG`` ->
    ``transparency``, ``color[0].color_Position`` -> ``color``)."""
    name = re.split(r"[\[.]", leaf)[0]
    while True:
        parents = _q(name, "listParent", node=node)
        if not parents:
            return name
        name = parents[0]


def _chain(node: str, leaf: str) -> set:
    """The attribute names ``copyAttr`` needs to move a wire at this level:
    the leaf and every ancestor up to the root (parents union leaves)."""
    parts = [re.sub(r"\[\d+\]", "", part) for part in leaf.split(".")]
    names = set(parts)
    name  = parts[0]
    while True:
        parents = _q(name, "listParent", node=node)
        if not parents:
            return names
        name = parents[0]
        names.add(name)


def _normalise(value: Any) -> Any:
    """``getAttr``'s shapes flattened: ``[(r, g, b)]`` -> ``(r, g, b)``."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], tuple):
        return tuple(value[0])
    if isinstance(value, list):
        return tuple(value)
    return value


def _close(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-7)
    except (TypeError, ValueError):
        return a == b


def _differs(value: Any, default: list) -> bool | None:
    """Whether a normalised value differs from a ``listDefault`` list;
    ``None`` when the two cannot be compared."""
    values = list(value) if isinstance(value, tuple) else [value]
    if len(values) != len(default):
        return None
    return not all(_close(a, b) for a, b in zip(values, default))


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == () or value == 0


def _parent_less(node: str, names: list) -> list:
    return [name for name in names if not _q(name, "listParent", node=node)]


def _own_locks(node: str, attr: str) -> list:
    """The plugs of ``attr`` carrying their OWN lock flag, read top-down: a
    locked parent covers its children and is the one name returned."""
    try:
        if cmds.getAttr(f"{node}.{attr}", lock=True):
            return [attr]
    except RuntimeError:
        return []
    locked = []
    for child in _q(attr, "listChildren", node=node) or []:
        try:
            if cmds.getAttr(f"{node}.{child}", lock=True):
                locked.append(child)
        except RuntimeError:
            pass
    return locked


def _copy_value(src: str, dst: str, shape: tuple) -> None:
    """``setAttr`` the value of one plug onto another of the same kind."""
    value = cmds.getAttr(src)
    if shape[2]:
        cmds.setAttr(dst, *value[0], type=cmds.getAttr(dst, type=True))
    else:
        cmds.setAttr(dst, value)


# ---------- Cloning ------------------------------------------------------- #


def _clone(src_plug: Plug, dst_node: str, name: str, hidden: bool | None = None) -> Plug:
    """Add to ``dst_node`` an attribute of exactly ``src_plug``'s kind under
    ``name``: the spec :func:`_spec_from_attribute` infers, with the source's
    own ``attributeType`` (a ``float`` stays a ``float``, a ``float3`` a
    ``float3``: the restore gate and the type gate compare exactly), its
    ``hidden`` flag (or the one given) and its keyable state."""
    src_node = str(src_plug.node)
    src_attr = src_plug.alias
    shape    = _shape(src_attr, node=src_node)
    spec     = _spec_from_attribute(src_plug, name, bool(src_plug.is_multi))
    if shape is not None and _element_type(shape) in _SCALARS and "dataType" not in spec.kargs:
        spec.kargs["attributeType"] = _element_type(shape)
    if hidden is None:
        hidden = bool(_q(src_attr, "hidden", node=src_node))
    keyable = False
    if not hidden:
        # A compound parent always reads non-keyable; its children carry the flag.
        children = _q(src_attr, "listChildren", node=src_node) or []
        probe    = f"{src_node}.{children[0]}" if children else str(src_plug)
        try:
            keyable = bool(cmds.getAttr(probe, keyable=True))
        except RuntimeError:
            keyable = True
    spec.kargs["hidden"]  = hidden
    spec.kargs["keyable"] = keyable
    new_plug = Node(dst_node) << spec
    if src_plug.is_multi:
        for index in src_plug.get_logical_indices() or []:
            try:
                new_plug[index] << src_plug[index].get()
            except Exception as e:
                LOGGER.debug("could not mirror %s[%s]: %s", src_plug, index, e)
    return new_plug


# ---------- The scan ------------------------------------------------------ #


@dataclass
class _Scan:
    """Everything the conversion needs, read before any write."""

    old:            str
    src:            str
    dst:            str
    values:         list = field(default_factory=list)   # carried non-default roots
    wires:          set  = field(default_factory=set)    # attr names for copyAttr connections
    outs:           set  = field(default_factory=set)    # out* names present on the target
    dynamic:        list = field(default_factory=list)   # parent-less user-defined attrs
    locks:          dict = field(default_factory=dict)   # root -> own-locked plug names on old
    lost_values:    list = field(default_factory=list)
    lost_wires:     list = field(default_factory=list)
    lost_animation: list = field(default_factory=list)
    lost_outputs:   list = field(default_factory=list)
    default_shift:  list = field(default_factory=list)
    opaque:         list = field(default_factory=list)
    park_roots:     list = field(default_factory=list)   # doomed roots to park, in order
    park_wires:     dict = field(default_factory=dict)   # root -> [(src plug, leaf)]
    parked:         list = field(default_factory=list)   # names as the warning lines print them
    engines:        list = field(default_factory=list)
    infos:          list = field(default_factory=list)
    dsl1_index:     int | None = None
    container:      str | None = None
    bindings:       list = field(default_factory=list)   # (container, leaf, published name, carried)
    relock:         list = field(default_factory=list)   # plug names to lock on the new node

    def report(self) -> Conversion:
        carried = sorted(set(self.values) | {_root_of(self.old, a) for a in self.wires})
        return Conversion(
            name=self.old,
            source=self.src,
            target=self.dst,
            carried=tuple(carried),
            dynamic=tuple(self.dynamic),
            lost_values=tuple(self.lost_values),
            lost_wires=tuple(self.lost_wires),
            lost_animation=tuple(self.lost_animation),
            lost_outputs=tuple(self.lost_outputs),
            default_shift=tuple(self.default_shift),
            opaque=tuple(self.opaque),
            parked=tuple(self.parked),
        )


def _scan(old: str, src: str, dst: str, park: bool) -> _Scan:
    scan = _Scan(old, src, dst)
    _scan_dynamic(scan)
    _scan_values(scan, park)
    _scan_wires_in(scan, park)
    _scan_wires_out(scan)
    _scan_locks(scan)
    _scan_topology(scan)
    return scan


def _scan_dynamic(scan: _Scan) -> None:
    """User-defined attributes cross whole (values and wires), so the value
    and wire passes leave them alone."""
    names = cmds.listAttr(scan.old, userDefined=True) or []
    scan.dynamic = _parent_less(scan.old, names)


def _scan_values(scan: _Scan, park: bool) -> None:
    old, dst = scan.old, scan.dst
    roots: dict = {}
    for leaf in cmds.listAttr(old, scalar=True, multi=True, read=True, visible=True) or []:
        roots.setdefault(_root_of(old, leaf), []).append(leaf)
    for root, leaves in roots.items():
        if root in PLUMBING or root in scan.dynamic:
            continue
        shape = _shape(root, node=old)
        if shape is None or not _q(root, "storable", node=old) or not _q(root, "writable", node=old):
            continue
        plug = f"{old}.{root}"
        if shape[1]:
            try:
                size = cmds.getAttr(plug, size=True)
            except RuntimeError:
                size = 1
            if size:
                scan.opaque.append(root)
            continue
        default = _q(root, "listDefault", node=old)
        try:
            value = _normalise(cmds.getAttr(plug))
        except RuntimeError:
            scan.opaque.append(root)
            continue
        if default is None:
            if not _is_empty(value):
                scan.opaque.append(root)
            continue
        differs = _differs(value, default)
        if differs is None:
            scan.opaque.append(root)
            continue
        if not differs:
            continue
        if _has(root, old, dst):
            scan.values.append(root)
            before = _q(root, "listDefault", type=scan.src)
            after  = _q(root, "listDefault", type=dst)
            if before is not None and after is not None and _differs(tuple(before), after):
                scan.default_shift.append((root, value, tuple(before), tuple(after)))
            continue
        # A driven attribute's value is whatever the wire says: the wire
        # pass reports it, and parks the value along with the wire.
        if any(cmds.connectionInfo(f"{old}.{name}", isDestination=True) for name in (root, *leaves)):
            continue
        scan.lost_values.append((root, value, bool(_own_locks(old, root))))
        if park and _parkable(shape):
            _park_root(scan, root)
            scan.parked.append(root)


def _park_root(scan: _Scan, root: str) -> None:
    if root not in scan.park_roots:
        scan.park_roots.append(root)


def _scan_wires_in(scan: _Scan, park: bool) -> None:
    old, dst = scan.old, scan.dst
    pairs = cmds.listConnections(
        old, source=True, destination=False, plugs=True, connections=True,
        skipConversionNodes=False,
    ) or []
    for dst_plug, src_plug in zip(pairs[0::2], pairs[1::2]):
        leaf = dst_plug.split(".", 1)[1]
        root = _root_of(old, leaf)
        if root in scan.dynamic:
            continue
        if _has(root, old, dst):
            scan.wires |= _chain(old, leaf)
            continue
        src_node = src_plug.split(".", 1)[0]
        kinds    = cmds.nodeType(src_node, inherited=True) or []
        if "animCurve" in kinds:
            scan.lost_animation.append((src_node, leaf))
        else:
            scan.lost_wires.append((src_plug, leaf, cmds.nodeType(src_node)))
        shape    = _shape(root, node=old)
        children = _q(root, "listChildren", node=old) or []
        if park and _parkable(shape) and (leaf == root or leaf in children):
            _park_root(scan, root)
            scan.park_wires.setdefault(root, []).append((src_plug, leaf))
            scan.parked.append(leaf)


def _scan_wires_out(scan: _Scan) -> None:
    old, dst = scan.old, scan.dst
    pairs = cmds.listConnections(
        old, source=False, destination=True, plugs=True, connections=True
    ) or []
    moved = []
    for src_plug, dst_plug in zip(pairs[0::2], pairs[1::2]):
        leaf = src_plug.split(".", 1)[1]
        if leaf == "message":
            node = dst_plug.split(".", 1)[0]
            kind = cmds.nodeType(node)
            if kind == "materialInfo" and dst_plug.endswith(".material"):
                scan.infos.append(node)
            match = _DSL1_RE.fullmatch(dst_plug)
            if match:
                scan.dsl1_index = int(match.group(1))
            continue
        root = _root_of(old, leaf)
        if root in scan.dynamic:
            continue
        if _is_published(dst_plug):
            # A container's published name is an alias, not a wire:
            # unbound before the move (copyAttr strands a compound's child
            # aliases on the old node) and bound again on the new node
            # when the attribute is carried; left unbound, named, when not.
            node, _, attr = dst_plug.partition(".")
            carried = _has(root, old, dst)
            scan.bindings.append((node, leaf, attr, carried))
            if not carried:
                scan.lost_outputs.append((leaf, dst_plug))
            continue
        if not _q(root, "writable", node=old):
            # An output (outColor, outTransparency, ...): moved when the
            # target computes it too.
            shape = _shape(root, node=old)
            if shape is not None and shape == _shape(root, type=dst):
                scan.outs |= _chain(old, leaf)
                moved.append(dst_plug)
            else:
                scan.lost_outputs.append((leaf, dst_plug))
            if leaf == "outColor" and dst_plug.endswith(".surfaceShader"):
                node = dst_plug.split(".", 1)[0]
                if cmds.nodeType(node) == "shadingEngine" and node not in scan.engines:
                    scan.engines.append(node)
        elif _has(root, old, dst):
            scan.wires |= _chain(old, leaf)
            moved.append(dst_plug)
        else:
            scan.lost_outputs.append((leaf, dst_plug))
    for dst_plug in moved:
        try:
            locked = cmds.getAttr(dst_plug, lock=True)
        except RuntimeError:
            locked = False
        if locked:
            raise RuntimeError(
                f"'{dst_plug}' is locked and the link from '{old}' into it must move "
                f"onto the new {dst}; unlock it first"
            )


def _is_published(plug: str) -> bool:
    """Whether ``plug`` names a container's published attribute (an alias
    ``listConnections`` reports as if it were a connection)."""
    node, _, attr = plug.partition(".")
    return (
        cmds.nodeType(node) == "container"
        and attr in (cmds.container(node, query=True, publishName=True) or [])
    )


def _bound(container: str) -> list:
    """The ``(plug, published name)`` pairs bound on a container."""
    pairs = cmds.container(container, query=True, bindAttr=True) or []
    return list(zip(pairs[0::2], pairs[1::2]))


def _unbind(scan: _Scan) -> None:
    """Release every published alias of the old node (a compound's parent
    releases its children with it)."""
    for node, leaf, attr, _ in scan.bindings:
        if (f"{scan.old}.{leaf}", attr) in _bound(node):
            cmds.container(node, edit=True, unbindAttr=[f"{scan.old}.{leaf}", attr])


def _rebind(scan: _Scan, name: str) -> None:
    """Bind the carried aliases to the new node (a parent binds its
    children with it)."""
    for node, leaf, attr, carried in scan.bindings:
        if carried and (f"{name}.{leaf}", attr) not in _bound(node):
            cmds.container(node, edit=True, bindAttr=[f"{name}.{leaf}", attr])


def _scan_locks(scan: _Scan) -> None:
    old   = scan.old
    roots = list(dict.fromkeys([
        *scan.values,
        *(_root_of(old, a) for a in sorted(scan.wires)),
        *scan.dynamic,
        *scan.park_roots,
    ]))
    for root in roots:
        locked = _own_locks(old, root)
        if locked:
            scan.locks[root] = locked
            if root not in scan.park_roots:
                scan.relock.extend(locked)


def _scan_topology(scan: _Scan) -> None:
    scan.container = cmds.container(query=True, findContainer=[scan.old]) or None


# ---------- The writes ---------------------------------------------------- #


def _prepare(scan: _Scan, park: bool) -> str:
    """Create the new node and give it everything the old one holds as
    values: dynamic attributes (parked ones included), non-default values.
    On failure the new node is deleted and the error re-raised: no wire has
    moved, so the old node is untouched."""
    old, dst = scan.old, scan.dst
    from rig.shade import _leaf

    new = cmds.createNode(dst, name=f"{_leaf(old)}{SUFFIX}", skipSelect=True)
    try:
        if cmds.nodeType(new) != dst:
            raise RuntimeError(f"Maya made a {cmds.nodeType(new)} for '{dst}', not a {dst}")
        for names in scan.locks.values():
            for name in names:
                cmds.setAttr(f"{old}.{name}", lock=False)
        if park:
            for root in scan.park_roots:
                _park(scan, root)
        for name in scan.dynamic:
            _clone(Plug(f"{old}.{name}"), new, name)
        names: set = set()
        for root in scan.values:
            names |= _chain(old, root)
        for name in scan.dynamic:
            names.add(name)
            names.update(_q(name, "listChildren", node=old) or [])
        if names:
            cmds.copyAttr(old, new, values=True, attribute=sorted(names))
        for name in [*scan.values, *scan.dynamic]:
            if not _q(name, "exists", node=new):
                raise RuntimeError(f"'{name}' did not arrive on the new {dst}")
        for name in scan.dynamic:
            if _shape(name, node=old) != _shape(name, node=new):
                raise RuntimeError(
                    f"cannot recreate dynamic attribute '{name}' on a {dst} with its type"
                )
    except Exception:
        if cmds.objExists(new):
            cmds.delete(new)
        raise
    return new


def _park(scan: _Scan, root: str) -> None:
    """Clone ``root`` onto the OLD node as the hidden ``__root__`` of the same
    type, copy its value, re-home its doomed wires onto it (leaf or parent
    level) and remember where its locks go on the new node."""
    old    = scan.old
    name   = f"__{root}__"
    shape  = _shape(root, node=old)
    parked = _clone(Plug(f"{old}.{root}"), old, name, hidden=True)
    _copy_value(f"{old}.{root}", str(parked), shape)
    children_src = _q(root, "listChildren", node=old) or []
    children_dst = _q(name, "listChildren", node=old) or []
    for src_plug, leaf in scan.park_wires.get(root, []):
        _disconnect_incoming(Plug(f"{old}.{leaf}"))
        target = str(parked) if leaf == root else f"{old}.{children_dst[children_src.index(leaf)]}"
        cmds.connectAttr(src_plug, target)
    for locked in scan.locks.get(root, []):
        scan.relock.append(
            name if locked == root else children_dst[children_src.index(locked)]
        )
    scan.dynamic.append(name)


def _commit(scan: _Scan, new: str, park: bool) -> str:
    """Move the wires, verify, delete the old node, take its name back,
    restore what an earlier park held, and return the final name."""
    old, dst = scan.old, scan.dst
    names    = set(scan.wires) | set(scan.outs) | {"message"}
    for name in scan.dynamic:
        names.add(name)
        names.update(_q(name, "listChildren", node=old) or [])
    _unbind(scan)
    cmds.copyAttr(
        old, new, inConnections=True, outConnections=True, attribute=sorted(names)
    )
    for name in scan.dynamic:
        if _q(name, "multi", node=old):
            _move_element_wires(old, new, name)
    for name in scan.relock:
        cmds.setAttr(f"{new}.{name}", lock=True)
    parked = set(scan.park_roots) if park else set()
    for src_plug, leaf in [*[(s, a) for s, a, _ in scan.lost_wires], *scan.lost_animation]:
        if _root_of(old, leaf) not in parked:
            _disconnect_incoming(Plug(f"{old}.{leaf}"))
    for leaf, dst_plug in scan.lost_outputs:
        if not _is_published(dst_plug):
            cmds.disconnectAttr(f"{old}.{leaf}", dst_plug)
    _verify(scan, new)
    # The old node is deleted LAST: every step that can still raise (the
    # rename, the alias rebind) runs while it exists, so a failure inside a
    # user's own undo chunk -- where the guarded rollback cannot fire --
    # leaves the material in the scene under a temporary name instead of
    # gone.
    aside = cmds.rename(old, f"{old}__rigold")
    got   = cmds.rename(new, old)
    if got != old:
        raise RuntimeError(
            f"the new {dst} could not take the name '{old}' back (Maya named it "
            f"'{got}'); the conversion is rolled back"
        )
    _rebind(scan, got)
    cmds.delete(aside)
    return got


def _move_element_wires(old: str, new: str, name: str) -> None:
    """``copyAttr`` never touches the elements of a multi: move the wires of
    a dynamic multi's elements by hand (the elements themselves were
    mirrored by the clone)."""
    for index in cmds.getAttr(f"{old}.{name}", multiIndices=True) or []:
        element = f"{name}[{index}]"
        pairs = cmds.listConnections(
            f"{old}.{element}", source=True, destination=False, plugs=True,
            connections=True, skipConversionNodes=False,
        ) or []
        for dst_plug, src_plug in zip(pairs[0::2], pairs[1::2]):
            cmds.disconnectAttr(src_plug, dst_plug)
            cmds.connectAttr(src_plug, f"{new}.{dst_plug.split('.', 1)[1]}")
        pairs = cmds.listConnections(
            f"{old}.{element}", source=False, destination=True, plugs=True,
            connections=True,
        ) or []
        for src_plug, dst_plug in zip(pairs[0::2], pairs[1::2]):
            cmds.disconnectAttr(src_plug, dst_plug)
            cmds.connectAttr(f"{new}.{src_plug.split('.', 1)[1]}", dst_plug)


def _verify(scan: _Scan, new: str) -> None:
    """Read the new topology back before the old node is deleted: whatever
    is wrong here is rolled back in one undo."""
    old, dst = scan.old, scan.dst
    if cmds.nodeType(new) != dst:
        raise RuntimeError(f"'{new}' is a {cmds.nodeType(new)}, not a {dst}")
    for engine in scan.engines:
        fed = cmds.listConnections(f"{engine}.surfaceShader", source=True, destination=False)
        if fed != [new]:
            raise RuntimeError(f"{engine}.surfaceShader reads {fed}, not the new {dst}")
    for info in scan.infos:
        fed = cmds.listConnections(f"{info}.material", source=True, destination=False)
        if fed != [new]:
            raise RuntimeError(f"{info}.material reads {fed}, not the new {dst}")
    if scan.dsl1_index is not None:
        shaders = cmds.listConnections(
            "defaultShaderList1.shaders", source=True, destination=False, plugs=True,
            connections=True,
        ) or []
        slots = [
            slot for slot, source in zip(shaders[0::2], shaders[1::2])
            if source == f"{new}.message"
        ]
        if slots != [f"defaultShaderList1.shaders[{scan.dsl1_index}]"]:
            raise RuntimeError(
                f"defaultShaderList1 lists the new {dst} at {slots}, not at the old "
                f"index {scan.dsl1_index}"
            )
    owner = cmds.container(query=True, findContainer=[new]) or None
    if owner != scan.container:
        raise RuntimeError(f"the new {dst} sits in container {owner!r}, not {scan.container!r}")
    if owner is not None:
        members = cmds.container(owner, query=True, nodeList=True) or []
        if members.count(new) != 1:
            raise RuntimeError(f"'{owner}' lists the new {dst} {members.count(new)} times")
    left = cmds.listConnections(old, plugs=True, connections=True) or []
    if left:
        raise RuntimeError(
            f"'{old}' still has connections after the move: {left}; nothing is deleted"
        )


def _restore(node: str, dst: str) -> list:
    """Give every parked ``__attr__`` whose bare name the node's type holds
    with the same kind back to the real attribute -- value or wire, at the
    level the wire was parked, lock included -- then destroy the parked
    attribute. Returns the restored names."""
    restored = []
    for name in _parent_less(node, cmds.listAttr(node, userDefined=True) or []):
        match = _PARKED_RE.fullmatch(name)
        if not match:
            continue
        bare  = match.group(1)
        shape = _shape(name, node=node)
        if (
            shape is None
            or shape != _shape(bare, type=dst)
            or not _q(bare, "writable", type=dst)
        ):
            continue
        parked, real = f"{node}.{name}", f"{node}.{bare}"
        pairs = [(parked, real)] + list(zip(
            (f"{node}.{c}" for c in _q(name, "listChildren", node=node) or []),
            (f"{node}.{c}" for c in _q(bare, "listChildren", node=node) or []),
        ))
        locked = [f"{node}.{c}" for c in _own_locks(node, name)]
        for plug in locked:
            cmds.setAttr(plug, lock=False)
        inputs = {
            p: (cmds.listConnections(
                p, source=True, destination=False, plugs=True, skipConversionNodes=False,
            ) or [None])[0]
            for p, _ in pairs
        }
        if inputs[parked] is None:
            if not any(inputs[p] for p, _ in pairs[1:]):
                _copy_value(parked, real, shape)
            else:
                for p, r in pairs[1:]:
                    if inputs[p] is None:
                        cmds.setAttr(r, cmds.getAttr(p))
        for p, r in pairs:
            if inputs[p] is not None:
                _disconnect_incoming(Plug(p))
                cmds.connectAttr(inputs[p], r, force=True)
        for plug in locked:
            cmds.setAttr(dict(pairs)[plug], lock=True)
        _do_destroy(parked)
        restored.append(bare)
    return restored


# ---------- The engine ---------------------------------------------------- #


def convert(
    spec:    Any,
    dst:     str,
    *,
    strict:  bool,
    dry_run: bool,
    park:    bool,
    attrs:   dict,
    update:  bool | None = None,
) -> Conversion:
    """Convert the material ``spec`` names to the node type ``dst``: a free
    retype when the material does not exist yet, a scene conversion in one
    undo chunk otherwise. The spec is retyped in place afterwards. On a
    material already of that type ``attrs`` follow the found-material rule:
    written only under ``update`` (the spec's own flag when not given)."""
    from rig.shade import _check_attrs, _gate_type

    _gate_type(dst)
    found = spec._locate()
    if found is None:
        return _retype_lazy(spec, dst, attrs)
    old, src = found.material, found.node_type
    if cmds.ls(old, defaultNodes=True):
        raise RuntimeError(
            f"'{old}' is a Maya default node and cannot be converted (cmds.delete "
            f"would print and leave it); convert a material of your own"
        )
    if cmds.referenceQuery(old, isNodeReferenced=True):
        raise RuntimeError(
            f"'{old}' is referenced and cannot be deleted or replaced; convert it "
            f"in the source file"
        )
    if cmds.lockNode(old, query=True)[0]:
        raise RuntimeError(f"'{old}' is locked (lockNode); unlock it first")
    _check_attrs(attrs, node_type=dst)
    if src == dst:
        if attrs and (spec._update if update is None else update):
            with _undo_chunk("rig.material"):
                for attr, value in attrs.items():
                    Plug(f"{old}.{attr}") << value
        _retype(spec, dst, attrs)
        return Conversion(old, src, dst)
    scan   = _scan(old, src, dst, park)
    report = scan.report()
    if dry_run:
        return report
    if strict and report:
        raise ValueError(str(report))
    if str(report):
        cmds.warning(str(report))
    restored = _run(scan, park, attrs)
    prune_memoize_caches()
    _retype(spec, dst, attrs)
    if spec._unique and spec._built is not None:
        spec._built = cmds.ls(old, uuid=True)[0]
    return replace(report, restored=tuple(restored))


def _run(scan: _Scan, park: bool, attrs: dict) -> list:
    """The chunk. The undo queue is on for its duration (a rollback needs
    it) and put back afterwards; a failure closes the chunk and undoes it
    only when the chunk recorded something."""
    was_on   = cmds.undoInfo(query=True, state=True)
    restored = []
    if not was_on:
        cmds.undoInfo(state=True)
    try:
        try:
            with _undo_chunk(CHUNK):
                new  = _prepare(scan, park)
                name = _commit(scan, new, park)
                restored = _restore(name, scan.dst)
                for attr, value in attrs.items():
                    Plug(f"{name}.{attr}") << value
        except Exception:
            if cmds.undoInfo(query=True, undoName=True) == CHUNK:
                cmds.undo()
            raise
    finally:
        if not was_on:
            cmds.undoInfo(state=False)
    return restored


def _retype_lazy(spec: Any, dst: str, attrs: dict) -> Conversion:
    """A material that does not exist yet: validate the pending kwargs
    against the target type, prune the ones it lacks with one warning, swap
    the class. Zero scene writes."""
    from rig.shade import _check_attrs

    _check_attrs(attrs, node_type=dst)
    dropped = [k for k in spec._attrs if not _q(k, "exists", type=dst)]
    src     = spec._type or ""
    if dropped:
        listed = ", ".join(f"{k}={spec._attrs[k]!r}" for k in dropped)
        plural = "s" if len(dropped) > 1 else ""
        cmds.warning(
            f"rig.shade: {spec._name!r} {src} -> {dst} drops kwarg{plural} {listed} "
            f"({dst} has no {', '.join(dropped)})"
        )
    for k in dropped:
        del spec._attrs[k]
    _retype(spec, dst, attrs)
    return Conversion(spec._name, src, dst, dropped_kwargs=tuple(dropped))


def _retype(spec: Any, dst: str, attrs: dict) -> None:
    from rig.shade import _BY_TYPE, Material

    spec._attrs.update(attrs)
    spec._attrs     = {k: v for k, v in spec._attrs.items() if _q(k, "exists", type=dst)}
    spec.__class__  = _BY_TYPE.get(dst, Material)
    spec._type      = dst
