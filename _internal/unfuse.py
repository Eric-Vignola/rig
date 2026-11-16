"""``rig.unfuse()``: ``rigExpression`` nodes back into rig's own network, rebuilt from
their program.

Everything a fused node computes is on the node: its program (``program``), its
numbers (``constants``) and its inputs (what drives each one, or the value it holds).
`decompile` reads them into fused values (`rig._internal.fuse.Fused`), one per
output, whose leaves are the inputs. `unfuse` makes each value with rig's own
functions and operators, as rig builds them without ``rig.fuse()``, connects it where
the output went and deletes the node. Nothing of the session that fused it is read: a
scene saved and opened again unfuses the same way.

Each operation comes back as the rig call that makes it (``a + b``, ``rig.matrix.aim``,
``rig.quaternion.slerp`` ...), and a few of the compiler's shapes as the one call they
came from (a decomposition's parts read off one ``decomposeMatrix``, ``compose``,
``fourbyfour``, a weighted matrix sum, ``pow``). A rig function the block traced
(``smoothstep``, ``sequence`` ...) is in the program as the operations it is made of,
and comes back as them. The values are rig's network's: where Maya's nodes and the
fused node differ (a ``%`` of a negative number, a division by zero ...), they follow
Maya's nodes, as rig's network does.
"""

from __future__ import annotations

import math
import types

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from rig._internal import fuse as F
from rig._internal import journal
from rig._internal.fuse import (
    BIN, FILE, IDENTITY_M, IDENTITY_Q, M_BIN_OPS, MAGIC, OP, OUT_ATTR, OUT_KINDS, Q_BIN_OPS, Q_UN_OPS, UN,
    Fused, _SLOT_ATTR,
)
from rig._internal.journal import statement

_CODE = {v: k for k, v in OP.items()}       # an op's int -> its name
_HEADER = {1: 6, 2: 12}                      # a program version -> its header's length
_SLOT_KIND = {"s": "s", "a": "s", "v": "v", "e": "e", "q": "q", "m": "m"}   # an input slot's value kind


# ------------------------------------------------------------------ the program
def _node_fn(node) -> tuple:
    """(MFnDependencyNode, name) of the rigExpression ``node`` (a name or a node)"""
    name = str(node)
    sel = om.MSelectionList()
    try:
        sel.add(name)
    except RuntimeError:
        raise ValueError(f"rig.unfuse: no node is named {name!r}") from None
    fn = om.MFnDependencyNode(sel.getDependNode(0))
    if fn.typeName != F.NODE_TYPE:
        raise TypeError(f"rig.unfuse: {name!r} is a {fn.typeName}, not a {F.NODE_TYPE} node")
    return fn, fn.name()


def _arrays(fn: om.MFnDependencyNode) -> tuple:
    """the node's program (ints) and constants (doubles), empty when it holds none"""
    out = []
    for attr, data in (("program", om.MFnIntArrayData), ("constants", om.MFnDoubleArrayData)):
        try:
            out.append(list(data(fn.findPlug(attr, False).asMObject()).array()))
        except (RuntimeError, ValueError, TypeError):
            out.append([])
    return out


def _parse(program: list, name: str) -> tuple:
    """(its instructions, its output registers {kind: [register, ...]}); ValueError when
    ``program`` is none rig.mll runs"""
    version = program[1] if len(program) > 1 and program[0] == MAGIC else None
    head = _HEADER.get(version)
    if head is None or len(program) < head:
        raise ValueError(f"rig.unfuse: {name!r} holds no rig.fuse() program")
    v2 = version == 2
    n = program[3]
    counts = dict(zip(OUT_KINDS, (program[4], program[5], *(program[8:12] if v2 else (0, 0, 0, 0)))))
    body = program[head:head + 5 * n]
    tail = program[head + 5 * n:]
    if n < 0 or len(body) != 5 * n or len(tail) != sum(counts.values()):
        raise ValueError(f"rig.unfuse: {name!r}'s program is malformed")
    instructions = [tuple(body[i:i + 5]) for i in range(0, len(body), 5)]
    registers, at = {}, 0
    for k in OUT_KINDS:
        registers[k] = tail[at:at + counts[k]]
        at += counts[k]
    return instructions, registers


class _Inputs:
    """the node's inputs as leaves: what drives an input, or the value it holds (a
    compound with channels of its own drivers is made of them)"""

    __slots__ = ("arrays", "name", "found")

    def __init__(self, fn: om.MFnDependencyNode, name: str) -> None:
        self.arrays = {slot: fn.findPlug(attr, False) for slot, attr in _SLOT_ATTR.items()}
        self.name = name
        self.found: dict = {}

    def leaf(self, slot: str, index: int) -> Fused:
        k = (slot, index)
        if k not in self.found:
            self.found[k] = self._leaf(slot, index, self.arrays[slot].elementByLogicalIndex(index))
        return self.found[k]

    def source(self, slot: str, index: int):
        """the plug driving input ``slot``[index], or None"""
        src = self.arrays[slot].elementByLogicalIndex(index).source()
        return None if src.isNull else src

    def _leaf(self, slot: str, index: int, plug: om.MPlug) -> Fused:
        from rig._internal.plug import Plug
        kind = _SLOT_KIND[slot]
        src = plug.source()
        if not src.isNull:
            return Fused(kind, "in_" + slot, extra=Plug(src), key=("unfuse", self.name, slot, index))
        if slot == "s":
            return F._constant(plug.asDouble())
        if slot == "a":
            return F._constant(plug.asMAngle().asDegrees())
        if slot == "m":
            try:
                m = om.MFnMatrixData(plug.asMObject()).matrix()
                return F._constants("m", list(m))
            except (RuntimeError, ValueError):
                return F._constants("m", IDENTITY_M)
        # a compound: whole, or channel by channel
        parts = []
        for c in range(plug.numChildren()):
            child = plug.child(c)
            csrc = child.source()
            if not csrc.isNull:
                angle = slot == "e"
                parts.append(Fused("s", "in_a" if angle else "in_s", extra=Plug(csrc),
                                   key=("unfuse", self.name, slot, index, c)))
            elif slot == "e":
                parts.append(F._constant(child.asMAngle().asDegrees()))
            else:
                parts.append(F._constant(child.asDouble()))
        if all(p.op == "const_s" for p in parts):
            return F._constants("q" if slot == "q" else "v", (p.extra for p in parts))
        vec = Fused("v", "make", tuple(parts[:3]))
        return Fused("q", "qmake", (vec, parts[3])) if slot == "q" else vec


# the value kind and op of each instruction's result (`rig._internal.fuse._compile`, rig.cpp)
_SCALAR_VECTOR = {
    "SPLAT": ("v", "splat", 1), "MAKE": ("v", "make", 3), "DOT": ("s", "dot", 2), "CROSS": ("v", "cross", 2),
    "LENGTH": ("s", "length", 1), "NORMALIZE": ("v", "normalize", 1), "DIST": ("s", "dist", 2),
    "SELECT_S": ("s", "select", 3), "SELECT_V": ("v", "select", 3), "CLAMP_S": ("s", "clamp", 3),
    "CLAMP_V": ("v", "clamp", 3), "LERP_S": ("s", "lerp", 3), "LERP_V": ("v", "lerp", 3),
}
# matrix and quaternion ops: (the op `_TAKES` names, its operand count, its operands' files)
_MQ = {
    "Q_VEC": ("qvec", "q"), "Q_SLERP": ("qslerp", "qqs"), "Q_EULER": ("qeuler", "qs"), "E_QUAT": ("equat", "vs"),
    "AXIS_Q": ("axisq", "vs"), "Q_AXIS": ("qaxis", "q"), "Q_ANGLE": ("qangle", "q"), "Q_MAKE": ("qmake", "vs"),
    "M_SCALE": ("mscale", "ms"), "M_INV": ("minv", "m"), "M_T": ("mtransp", "m"), "M_DET": ("mdet", "m"),
    "PT_MUL": ("ptmul", "vm"), "VEC_MUL": ("vecmul", "vm"), "M_ROW": ("mrow", "ms"), "M_COL": ("mcol", "ms"),
    "M_TRANS": ("mtrans", "m"), "M_SCALE_OF": ("mscaleof", "m"), "M_SHEAR_OF": ("mshear", "m"),
    "M_ROT": ("mrot", "m"), "R_EULER": ("reuler", "ms"), "R_QUAT": ("rquat", "m"), "E_MATRIX": ("ematrix", "vs"),
    "Q_MATRIX": ("qmatrix", "q"), "BLEND": ("blend", "mms"),
}
_RESULT = {"qmake": "q"}


def decompile(node) -> dict:
    """``{(kind, index): Fused}``: what each output of the ``rigExpression`` node
    computes (``kind`` one of ``s v q m e a``: ``scalarOut`` ... ``angleOut``), read off
    its program, its constants and its inputs alone"""
    fn, name = _node_fn(node)
    program, constants = _arrays(fn)
    instructions, registers = _parse(program, name)
    inputs = _Inputs(fn, name)
    files: tuple = ({}, {}, {})

    def reg(file: int, r: int) -> Fused:
        value = files[file].get(r)
        if value is None:
            raise ValueError(f"rig.unfuse: {name!r}'s program reads a register it never wrote")
        return value

    def const(at: int, width: int) -> list:
        if at < 0 or at + width > len(constants):
            raise ValueError(f"rig.unfuse: {name!r}'s program reads a constant it does not hold")
        return constants[at:at + width]

    loads = {"LOAD_S": "s", "LOAD_V": "v", "LOAD_A": "a", "LOAD_E": "e", "LOAD_Q": "q", "LOAD_M": "m"}
    i, n = 0, len(instructions)
    while i < n:
        code, dst, a, b, c = instructions[i]
        op = _CODE.get(code)
        ext = None
        if op in ("COMPOSE", "AIM"):
            if i + 1 >= n or _CODE.get(instructions[i + 1][0]) != "EXT":
                raise ValueError(f"rig.unfuse: {name!r}'s program is malformed")
            ext = instructions[i + 1]
            i += 1
        i += 1
        if op in loads:
            slot = loads[op]
            value, file = inputs.leaf(slot, a), FILE[_SLOT_KIND[slot]]
        elif op == "CONST_S":
            value, file = F._constant(const(a, 1)[0]), 0
        elif op in ("CONST_V", "CONST_Q", "CONST_M"):
            k = op[-1].lower()
            width = {"v": 3, "q": 4, "m": 16}[k]
            value, file = F._constants(k, const(a, width)), FILE[k]
        elif op == "COMP":
            value, file = Fused("s", "comp", (reg(0, a),), b), 0
        elif op in ("BIN_S", "BIN_V"):
            value, file = Fused(op[-1].lower(), "bin", (reg(0, a), reg(0, b)), BIN[c]), 0
        elif op in ("UN_S", "UN_V"):
            value, file = Fused(op[-1].lower(), "un", (reg(0, a),), UN[b]), 0
        elif op in _SCALAR_VECTOR:
            kind, name_, count = _SCALAR_VECTOR[op]
            value, file = Fused(kind, name_, tuple(reg(0, r) for r in (a, b, c)[:count])), 0
        elif op == "Q_COMP":
            value, file = Fused("s", "qcomp", (reg(1, a),), b), 0
        elif op == "Q_BIN":
            value, file = Fused("q", "qbin", (reg(1, a), reg(1, b)), Q_BIN_OPS[c]), 1
        elif op == "Q_UN":
            value, file = Fused("q", "qun", (reg(1, a),), Q_UN_OPS[b]), 1
        elif op == "M_BIN":
            value, file = Fused("m", "mbin", (reg(2, a), reg(2, b)), M_BIN_OPS[c]), 2
        elif op == "M_AXIS":
            value, file = Fused("v", "maxis", (reg(2, a),), b), 0
        elif op == "M_SETROW":
            value, file = Fused("m", "setrow", (reg(2, a), reg(0, b)), c), 2
        elif op == "COMPOSE":
            value, file = Fused("m", "compose", (reg(2, a), reg(0, b), reg(0, c), reg(0, ext[2]))), 2
        elif op == "AIM":
            value, file = Fused("m", "aim", (reg(0, a), reg(0, b), reg(0, ext[2]), reg(0, ext[3]))), 2
        elif op in _MQ:
            fused_op, takes = _MQ[op]
            kind = _RESULT.get(fused_op) or F._TAKES[fused_op][0]
            operands = tuple(reg({"m": 2, "q": 1}.get(t, 0), r) for t, r in zip(takes, (a, b, c)))
            value, file = Fused(kind, fused_op, operands), FILE[kind]
        elif op == "MOVE":
            value, file = reg(c, a), c          # a stand-in's output: its value
        elif op == "NOP":
            continue
        else:
            raise ValueError(f"rig.unfuse: {name!r}'s program has an op rig.unfuse does not know ({code})")
        files[file][dst] = value
    return {(k, idx): reg(FILE[k], r) for k in OUT_KINDS for idx, r in enumerate(registers[k])}


def describe(node) -> str:
    """the expressions of the ``rigExpression`` node as Markdown (what its Notes say),
    read off its program: names as they are now"""
    fn, name = _node_fn(node)
    values = decompile(name)
    record = types.SimpleNamespace(outs={k: [] for k in OUT_KINDS}, loads={}, inputs={})
    for (k, idx), e in sorted(values.items(), key=lambda kv: (OUT_KINDS.index(kv[0][0]), kv[0][1])):
        record.outs[k].append(e)
    inputs = _Inputs(fn, name)
    for slot, attr in _SLOT_ATTR.items():
        array = fn.findPlug(attr, False)
        record.inputs[slot] = [
            (lambda s: "(a value)" if s is None else s.name())(inputs.source(slot, array.elementByPhysicalIndex(p).logicalIndex()))
            for p in range(array.numElements())]
    return F.describe(record)


# ------------------------------------------------------------------ the rebuild
def _numbers(*values) -> bool:
    return all(isinstance(v, (float, int)) or (type(v) is list and all(isinstance(x, (float, int)) for x in v))
               for v in values)


def _np(v):
    return np.asarray(v, dtype=float)


def _out(v):
    v = np.asarray(v, dtype=float)
    return float(v) if v.ndim == 0 else [float(x) for x in v]


_FOLD_BIN = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide, "pow": np.power,
    "mod": np.mod, "floordiv": np.floor_divide, "min": np.minimum, "max": np.maximum,
    "lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal, "eq": np.equal,
    "ne": np.not_equal, "and": lambda a, b: (a != 0) & (b != 0), "or": lambda a, b: (a != 0) | (b != 0),
    "xor": lambda a, b: (a != 0) ^ (b != 0), "atan2": np.arctan2,
    "atan2d": lambda a, b: np.degrees(np.arctan2(a, b)),
}
_FOLD_UN = {
    "neg": np.negative, "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "round": np.round, "trunc": np.trunc,
    "sign": np.sign, "rev": lambda x: 1.0 - x, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
    "not": lambda x: x == 0, "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "sind": lambda x: np.sin(np.radians(x)), "cosd": lambda x: np.cos(np.radians(x)),
    "tand": lambda x: np.tan(np.radians(x)), "asind": lambda x: np.degrees(np.arcsin(x)),
    "acosd": lambda x: np.degrees(np.arccos(x)), "atand": lambda x: np.degrees(np.arctan(x)),
}


def _rig():
    import rig
    from rig import euler, functions, matrix, quaternion, trigonometry, vector
    from rig._internal import math_nodes
    return types.SimpleNamespace(rig=rig, E=euler, RF=functions, M=matrix, Q=quaternion, T=trigonometry, V=vector,
                                 mn=math_nodes)


def _binary(R, name):
    RF, T = R.RF, R.T
    return {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b, "pow": lambda a, b: a ** b, "mod": lambda a, b: a % b,
        "floordiv": lambda a, b: a // b, "min": lambda a, b: RF.min([a, b]), "max": lambda a, b: RF.max([a, b]),
        "lt": RF.less_than, "le": RF.less_or_equal, "gt": RF.greater_than, "ge": RF.greater_or_equal,
        "eq": RF.equal, "ne": RF.not_equal, "and": RF.logical_and, "or": RF.logical_or, "xor": RF.logical_xor,
        "atan2": T.atan2, "atan2d": T.atan2d,
    }[name]


def _unary(R, name):
    RF, T = R.RF, R.T
    table = {
        "neg": lambda x: -x, "abs": RF.abs, "floor": RF.floor, "ceil": RF.ceil, "round": RF.round,
        "trunc": RF.trunc, "sign": RF.sign, "rev": RF.rev, "sqrt": RF.sqrt, "exp": RF.exp, "log": RF.log,
        "not": RF.logical_not,
    }
    return table.get(name) or getattr(T, name)


def _is_identity(e: Fused, kind: str) -> bool:
    return e.op == "const_" + kind and tuple(e.extra) == (IDENTITY_M if kind == "m" else IDENTITY_Q)


def _is_const(e: Fused, values) -> bool:
    return e.op == "const_v" and tuple(e.extra) == tuple(values)


def _matrix_point(e: Fused):
    """the matrix ``m`` when ``e`` is distanceBetween's point of it (`fuse._point_of`)"""
    if e.op != "select" or len(e.args) != 3:
        return None
    test, t, divided = e.args
    if t.op != "mtrans" or divided.op != "bin" or divided.extra != "div" or divided.args[0].key != t.key:
        return None
    m = t.args[0]
    if test.op != "bin" or test.extra != "eq":
        return None
    w = test.args[0]
    return (m if w.op == "qcomp" and w.extra == 3 and w.args[0].op == "mrow" and w.args[0].args[0].key == m.key
            else None)


def _choice(e: Fused):
    """(selector, tokens) when ``e`` is how rig.fuse() makes ``choice(tokens,
    selector=s)`` (`fuse._f_choice`): the k-th token where ``s >= k - 0.5``, from the
    last token in. Three tokens or more: two are one ``s >= 0.5``, which a condition
    written so makes too (and a condition keeps its branches out of range, as the fused
    node does, where Maya's choice keeps the value it had)"""
    picked, s, x = [], None, e
    while x.op == "select":
        test = x.args[0]
        if test.op != "bin" or test.extra != "ge" or test.args[1].op != "const_s":
            break
        if s is not None and test.args[0].key != s.key:
            break
        k = test.args[1].extra + 0.5
        if k != int(k) or (picked and int(k) != picked[-1][0] - 1):
            break
        s = test.args[0]
        picked.append((int(k), x.args[1]))
        x = x.args[2]
    if s is None or picked[-1][0] != 1 or len(picked) < 2:
        return None
    return s, [x] + [token for _k, token in reversed(picked)]


_DEC_PARTS = {"mtrans": "outputTranslate", "mscaleof": "outputScale", "mshear": "outputShear"}


class _Rebuild:
    """fused values made as rig's network, each once (equal values, one network)"""

    def __init__(self, values: list) -> None:
        self.R = _rig()
        self.memo: dict = {}
        self.decompositions: dict = {}     # (matrix key, order key) -> decomposeMatrix's outputTranslate
        self.axis_angles: dict = {}
        # the parts of each matrix a decomposition gives (several parts: one decomposeMatrix),
        # and the rotate orders its eulers are read in (the first: the other parts' node)
        self.parts: dict = {}
        self.orders: dict = {}
        seen: set = set()
        todo = list(values)
        while todo:
            e = todo.pop()
            if e.key in seen:
                continue
            seen.add(e.key)
            part = self._part(e)
            if part is not None:
                m, what = part
                self.parts.setdefault(m.key, set()).add(what)
                if e.op == "reuler":
                    self.orders.setdefault(m.key, []).append(e.args[1])
            todo.extend(e.args)

    @staticmethod
    def _part(e: Fused):
        """(matrix, part) when ``e`` is a part of a matrix's decomposition"""
        if e.op in _DEC_PARTS:
            return e.args[0], e.op
        if e.op in ("reuler", "rquat") and e.args[0].op == "mrot":
            return e.args[0].args[0], (e.op, e.args[1].key if e.op == "reuler" else None)
        return None

    # -- the plan of each value: (how it is made, the values it is made of)
    def plan(self, e: Fused):
        op, x = e.op, e.extra
        a = e.args
        if op.startswith("in_"):
            return (lambda: x), ()
        if op == "const_s":
            return (lambda: float(x)), ()
        if op in ("const_v", "const_q"):
            return (lambda: [float(v) for v in x]), ()
        if op == "const_m":
            return (lambda: self.R.mn.constant([list(x[r * 4:r * 4 + 4]) for r in range(4)])), ()
        if op == "bin":
            # rig's + - * / take a scalar with a vector (one node, as rig broadcasts it)
            args = tuple(v.args[0] if v.op == "splat" and x in ("add", "sub", "mul", "div") else v for v in a)
            return (lambda p, q: self.binary(x, p, q)), args
        if op == "un":
            return (lambda p: self.unary(x, p)), a
        if op == "splat":
            return (lambda p: [p] * 3 if _numbers(p) else self.R.mn.constant([p, p, p])), a
        if op == "make":
            return (lambda *p: list(p) if _numbers(*p) else self.R.mn.constant(list(p))), a
        if op == "qmake":
            v, w = a
            if v.op == "make":
                return (lambda *p: list(p) if _numbers(*p) else self.R.mn.constant(list(p))), (*v.args, w)
            return (lambda p, q: [*p, q] if _numbers(p, q) else self.R.mn.constant([p[0], p[1], p[2], q])), a
        if op in ("comp", "qcomp"):
            return (lambda p: p[x]), a
        if op == "select":
            picked = _choice(e)
            if picked is not None:
                selector, items = picked
                return (lambda s, *p: self.R.RF.choice(list(p), selector=s)), (selector, *items)
            return (lambda t, p, q: self.condition(e, t, p, q)), a
        if op == "clamp":
            args = tuple(v.args[0] if v.op == "splat" else v for v in a)
            return (lambda v, lo, hi: _out(np.clip(_np(v), _np(lo), _np(hi))) if _numbers(v, lo, hi)
                    else self.R.RF.clamp(v, lo, hi)), args
        if op == "lerp":
            return (lambda p, q, w: (_out(_np(p) + (_np(q) - _np(p)) * w) if _numbers(p, q, w)
                                     else (self.R.rig.lerp if e.kind == "s" else self.R.V.lerp)(p, q, w))), a
        if op == "dist":
            ma, mb = _matrix_point(a[0]), _matrix_point(a[1])
            if ma is not None and mb is not None:
                return (lambda p, q: self.R.M.dist(p, q)), (ma, mb)
            return self._vector_op("dist", np.linalg.norm, 2), a
        if op in ("dot", "cross", "length", "normalize"):
            fold = {"dot": lambda p, q: float(np.dot(p, q)), "cross": lambda p, q: _out(np.cross(p, q)),
                    "length": lambda p: float(np.linalg.norm(p)),
                    "normalize": lambda p: _out(_np(p) / (np.linalg.norm(p) or 1.0))}[op]
            return self._vector_op(op, fold, len(a)), a
        return self._plan_mq(e)

    def _vector_op(self, name, fold, n):
        def make(*p):
            if _numbers(*p):
                return fold(*[_np(v) for v in p]) if name != "dist" else float(np.linalg.norm(_np(p[0]) - _np(p[1])))
            return getattr(self.R.V, name)(*p)
        return make

    def _plan_mq(self, e: Fused):
        R, op, x, a = self.R, e.op, e.extra, e.args
        M, Q, E = R.M, R.Q, R.E
        part = self._part(e)
        if part is not None:
            m, what = part
            order = a[1] if op == "reuler" else None
            if len(self.parts.get(m.key, ())) > 1 or op == "reuler" or op == "mshear":
                output = {"reuler": "outputRotate", "rquat": "outputQuat"}.get(op) or _DEC_PARTS[op]
                if order is None:
                    return (lambda mv: getattr(self.decompose(m.key, None, mv, None), output)), (m,)
                return (lambda mv, o: getattr(self.decompose(m.key, order.key, mv, o), output)), (m, order)
            single = {"mtrans": M.translation, "mscaleof": M.scale_of, "rquat": M.to_quaternion}[op]
            return (lambda mv: single(mv)), (m,)
        if op == "reuler":       # the euler of a rotation matrix made otherwise
            return (lambda mv, o: self.decompose(None, None, mv, o).outputRotate), a
        if op == "rquat":
            return (lambda mv: self.decompose(None, None, mv, None).outputQuat), a
        if op == "mrot":
            return (lambda mv: M.compose(rotate=self.decompose(None, None, mv, None).outputQuat)), a
        if op == "compose":
            return self._plan_compose(e)
        if op == "setrow":
            rows: dict = {}
            base = e
            while base.op == "setrow":
                rows.setdefault(base.extra, base.args[1])
                base = base.args[0]
            if not _is_identity(base, "m"):
                raise ValueError("rig.unfuse: a row set into a matrix that is not the identity")
            keys = sorted(rows)
            names = ("x", "y", "z", "position")
            return (lambda *vs: M.fourbyfour(**{names[k]: v for k, v in zip(keys, vs)})), tuple(rows[k] for k in keys)
        if op == "mbin" and x == "mul":
            chain = []
            todo = [e]
            while todo:
                t = todo.pop()
                if t.op == "mbin" and t.extra == "mul":
                    todo.extend((t.args[1], t.args[0]))
                else:
                    chain.append(t)
            return (lambda *ms: M.multiply(*ms)), tuple(chain)
        if op in ("mbin", "mscale"):
            return self._plan_sum(e)
        if op == "blend":
            if _is_identity(a[0], "m"):
                return (lambda mv, w: M.pow(mv, w)), (a[1], a[2])
            return (lambda p, q, w: M.blend(p, q, w)), a
        if op == "qslerp":
            if _is_identity(a[0], "q"):
                return (lambda q, w: Q.pow(q, w)), (a[1], a[2])
            return (lambda p, q, w: Q.slerp(p, q, w)), a
        if op in ("qaxis", "qangle"):
            index = 0 if op == "qaxis" else 1
            return (lambda q: self.axis_angle(a[0].key, q)[index]), a
        unary = {"minv": M.inverse, "mtransp": M.transpose, "mdet": M.determinant, "qvec": Q.to_vector,
                 "qmatrix": Q.to_matrix}
        if op in unary:
            return (lambda p: unary[op](p)), a
        if op == "qbin":
            return (lambda p, q: {"add": Q.add, "sub": Q.subtract, "mul": Q.multiply}[x](p, q)), a
        if op == "qun":
            return (lambda p: {"neg": Q.negate, "conjugate": Q.conjugate, "inverse": Q.inverse,
                               "normal": Q.normalize}[x](p)), a
        if op == "qeuler":
            return (lambda q, o: Q.to_euler(q, rotate_order=self.order(o))), a
        if op == "equat":
            return (lambda v, o: E.to_quaternion(v, rotate_order=self.order(o))), a
        if op == "ematrix":
            return (lambda v, o: E.to_matrix(v, rotate_order=self.order(o))), a
        if op == "axisq":
            return (lambda v, t: Q.from_axis_angle(v, t)), a
        if op == "ptmul":
            return (lambda v, mv: M.transform_point(v, mv)), a
        if op == "vecmul":
            return (lambda v, mv: M.transform_vector(v, mv)), a
        if op in ("mrow", "mcol"):
            f = M.row if op == "mrow" else M.column
            return (lambda mv, i: f(mv, int(i) if _numbers(i) else i)), a
        if op == "maxis":
            return (lambda mv: M.axis(mv, axis=x)), a
        if op == "aim":
            return (lambda p, q, aa, ua: M.aim(p, q, aim_axis=aa, up_axis=ua)), a
        raise ValueError(f"rig.unfuse: no rig call makes a fused {op!r}")

    def _plan_compose(self, e: Fused):
        R = self.R
        rotation, scale, shear, translate = e.args
        operands, names = [], []
        kwargs_order = None
        if rotation.op == "qmatrix":
            operands.append(rotation.args[0])
            names.append("rotate")
        elif rotation.op == "ematrix":
            operands += [rotation.args[0], rotation.args[1]]
            names += ["rotate", "rotate_order"]
            kwargs_order = True
        elif not _is_identity(rotation, "m"):
            operands.append(rotation)
            names.append("rotation matrix")
        for name, value, default in (("scale", scale, (1.0, 1.0, 1.0)), ("shear", shear, (0.0, 0.0, 0.0)),
                                     ("translate", translate, (0.0, 0.0, 0.0))):
            if not _is_const(value, default):
                operands.append(value)
                names.append(name)

        def make(*values):
            kwargs = {}
            for name, v in zip(names, values):
                if name == "rotation matrix":
                    kwargs["rotate"] = self.decompose(None, None, v, None).outputQuat
                elif name == "rotate_order":
                    kwargs[name] = self.order(v)
                else:
                    kwargs[name] = v
            if not kwargs:
                return R.mn.constant([list(IDENTITY_M[r * 4:r * 4 + 4]) for r in range(4)])
            return R.M.compose(**kwargs)

        del kwargs_order
        return make, tuple(operands)

    def _plan_sum(self, e: Fused):
        """a sum of matrices, each maybe scaled: one addMatrix, or one wtAddMatrix (a lerp's
        two weights, 1 - w and w: rig.matrix.lerp)"""
        M = self.R.M
        terms, todo = [], [e]
        while todo:
            t = todo.pop()
            if t.op == "mbin" and t.extra == "add":
                todo.extend((t.args[1], t.args[0]))
            else:
                terms.append(t)
        scaled = [t.op == "mscale" for t in terms]
        if not any(scaled):
            return (lambda *ms: M.add(*ms)), tuple(terms)
        if (len(terms) == 2 and all(scaled) and terms[0].args[1].op == "un" and terms[0].args[1].extra == "rev"
                and terms[0].args[1].args[0].key == terms[1].args[1].key):
            return (lambda p, q, w: M.lerp(p, q, w)), (terms[0].args[0], terms[1].args[0], terms[1].args[1])
        operands, weights = [], []
        for t in terms:
            if t.op == "mscale":
                operands.append(t.args[0])
                weights.append(t.args[1])
            else:
                operands.append(t)
                weights.append(None)
        n = len(operands)
        flat = operands + [w for w in weights if w is not None]

        def make(*values):
            ms, rest = list(values[:n]), iter(values[n:])
            ws = [1.0 if w is None else next(rest) for w in weights]
            return M.add(*ms, weights=ws)

        return make, tuple(flat)

    # -- the shared parts
    def decompose(self, mkey, okey, m, order):
        """rig.matrix.decompose(m) (its outputTranslate), once per matrix and rotate order;
        the parts that no rotate order changes read the matrix's first one"""
        if mkey is not None and okey is None:
            for (mk, _ok), found in self.decompositions.items():
                if mk == mkey:
                    return found
            orders = self.orders.get(mkey)
            if orders:           # (made in the rotate order its euler is read in)
                order = orders[0]
                okey, order = order.key, self.value(order)
        k = (mkey, okey)
        if mkey is not None:
            found = self.decompositions.get(k)
            if found is not None:
                return found
        made = self.R.M.decompose(m, rotate_order=self.order(order)) if order is not None else self.R.M.decompose(m)
        if mkey is not None:
            self.decompositions[k] = made
        return made

    def axis_angle(self, qkey, q):
        found = self.axis_angles.get(qkey)
        if found is None:
            found = self.axis_angles[qkey] = self.R.Q.to_axis_angle(q)
        return found

    @staticmethod
    def order(o):
        return int(o) if _numbers(o) else o

    def condition(self, e: Fused, t, p, q):
        """rig's condition; a vector one is its node's outColor (rig gives outColorR
        when no branch is a compound plug: two vectors of numbers)"""
        if _numbers(t):
            return p if t != 0 else q
        out = self.R.mn.condition(t, p, q)
        if e.kind in "ve" and type(out).__name__ != "list":
            node = type(out)._raw_get_node(out)
            if str(out).endswith(".outColorR"):
                out = node.outColor
        return out

    def binary(self, name, p, q):
        if _numbers(p, q):
            with np.errstate(all="ignore"):
                return _out(_FOLD_BIN[name](_np(p), _np(q)))
        return _binary(self.R, name)(p, q)

    def unary(self, name, p):
        if _numbers(p):
            with np.errstate(all="ignore"):
                return _out(_FOLD_UN[name](_np(p)))
        return _unary(self.R, name)(p)

    def check(self, values: list) -> None:
        """ValueError before anything is made when a value has no rig call (each value's
        plan, made of nothing)"""
        seen: set = set()
        todo = list(values)
        while todo:
            e = todo.pop()
            if e.key in seen:
                continue
            seen.add(e.key)
            todo.extend(self.plan(e)[1])

    # -- a value made: its operands first (no recursion: a long chain is a deep tree)
    def value(self, e: Fused):
        memo = self.memo
        stack = [(e, None)]
        while stack:
            x, planned = stack.pop()
            if x.key in memo:
                continue
            if planned is None:
                planned = self.plan(x)
                stack.append((x, planned))
                stack.extend((o, None) for o in reversed(planned[1]) if o.key not in memo)
                continue
            make, operands = planned
            memo[x.key] = make(*[memo[o.key] for o in operands])
        return memo[e.key]


# ------------------------------------------------------------------ unfuse
def _container_of(name: str):
    """the container that holds the node ``name``, or None (rig's route to
    ``cmds.container``: `journal.container_cmd`)"""
    found = journal.container_cmd(query=True, findContainer=[name])
    return found if found else None


def _destinations(fn: om.MFnDependencyNode, kind: str, index: int) -> list:
    """[(channel or None, the plug it drives)] of output ``kind``[index]"""
    array = fn.findPlug(OUT_ATTR[kind], False)
    if index not in array.getExistingArrayAttributeIndices():
        return []
    element = array.elementByLogicalIndex(index)
    out = [(None, d) for d in element.destinations()]
    if element.isCompound:
        for c in range(element.numChildren()):
            out += [(c, d) for d in element.child(c).destinations()]
    return out


def _unfuse_one(name: str) -> list:
    """``name`` (a rigExpression) rebuilt as rig's network: its connected outputs made,
    connected where they went, the node deleted. The UUIDs of the nodes made."""
    from rig._internal.container import Container, _StackFrame, container
    from rig._internal.plug import Plug
    fn, name = _node_fn(name)
    values = decompile(name)
    wired = {key: _destinations(fn, *key) for key in values}
    wired = {key: d for key, d in wired.items() if d}
    holder = _container_of(name)
    if holder is not None and cmds.lockNode(holder, query=True, lock=True)[0]:
        raise RuntimeError(f"rig.unfuse: {name!r} is in the locked container {holder!r}: unlock it first")
    frame = _StackFrame(name=name, container_node=Container._wrap(holder) if holder else None, preserve=False,
                        enabled=holder is not None, depth=len(container._stack))
    container._stack.append(frame)
    try:
        rebuild = _Rebuild([values[key] for key in wired])
        made = {key: rebuild.value(values[key]) for key in wired}
        # (in the scope: a unitConversion a connection makes joins the container too)
        for key, destinations in wired.items():
            value = made[key]
            for channel, dst in destinations:
                target = Plug(dst)
                journal.cmds_edit.disconnectAttr(str(Plug(dst.source())), str(target))
                target << (value if channel is None else value[channel])
    finally:
        container._stack.remove(frame)
    journal.cmds_edit.delete(name)
    return list(frame.members)


def unfuse(*nodes):
    """``rig.unfuse()`` -- the ``rigExpression`` nodes ``rig.fuse()`` made, back into
    rig's own network: what each output computes is built with rig's functions and
    operators, as without ``rig.fuse()``, and connected where the output went; the
    node is deleted. It reads the node alone (its program, constants and inputs), so a
    scene opened again unfuses too.

    No argument: every rigExpression node in the scene. Otherwise the nodes given (a
    node, a name, or a list of them). One undo step. A node in a container has its
    network made in it. Returns a List of the nodes made.

    Raises TypeError for a node that is not a rigExpression, RuntimeError inside a
    ``rig.fuse()`` block (the block would fuse the network again)."""
    from rig._internal.undo import undo_chunk
    # (the chunk around the statement: its last edits, sent as it ends, are in the step)
    with undo_chunk("rig.unfuse"):
        return _unfuse(nodes)


@statement
def _unfuse(nodes: tuple):
    """`unfuse`, in one statement: rig's own code, so nothing in it fuses"""
    from rig._internal.list import List
    from rig._internal.node import Node
    if F.FUSE.depth:
        raise RuntimeError("rig.unfuse inside rig.fuse(): the block would fuse the network again")
    if F.FUSE.plugin is None:
        F.FUSE.plugin = F._load_plugin()
    names: list = []
    for n in nodes:
        if isinstance(n, (list, tuple)) or type(n).__name__ == "List":
            names += [str(x) for x in n]
        else:
            names.append(str(n))
    if not nodes:
        names = cmds.ls(type=F.NODE_TYPE) or []
    for n in names:
        # every node read and planned before the first is changed (each is read again as
        # its turn comes: an earlier one rewires what it reads)
        values = list(decompile(n).values())
        _Rebuild(values).check(values)
    made: list = []
    for n in names:
        made += _unfuse_one(n)
    out = List()
    for uid in made:
        found = cmds.ls(uid)
        if found:
            out.append(Node(found[0]))
    return out
