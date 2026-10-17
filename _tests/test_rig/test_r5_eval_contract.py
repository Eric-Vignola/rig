"""Round 5, lead decision D1 (2026-10-02, option a): how many times and when
Maya runs upstream computes as a side effect of rig's own read-only queries is
not part of rig's behaviour.

A type query (``cmds.getAttr(type=True)``, an ``MSelectionList.add`` of a multi
root, ...) makes Maya evaluate a dirty upstream node. Phase A makes fewer such
queries: a ``<<`` into a fixed kind no longer queries its destination's data
type (A4's INJ-3), one ``<<`` queries each operand once (H4), and
``functions.choice`` writes each operand at its index without the multi root's
queries (A2). In one case it makes one more: that ``<<`` no longer fills the
static type cache, so a later ``data_type`` of another plug of the same node
type and attribute queries Maya. So an expression or a scripted node whose
compute prints, counts or sets attributes can run a different number of times
than at 2.0.0a4 (45d2c13).

The contract keeps everything else. The cases are the exactness review's
(``runs\\r5\\review_exactness``: ``cases_eval3.py``, ``cases_eval2.py``,
``cases_static_prime.py``, and ``cases_uncached.py``'s ``case_f`` with
``review_uncached_node.py``), each with an expression or a scripted node that
counts its computes (the review's prints and the expressions' ``setAttr`` side
effect are left out: the counter tells the computes). The static type cache is
emptied before each case, as in a fresh session.

* ``test_the_values_exceptions_and_connections_are_45d2c13s``: every value,
  exception (type, message, ``__context__`` / ``__cause__``), node and
  connection is the one 45d2c13 gave (``_ORACLE_45D2C13``, written by
  ``runs\\r5\\AF\\gen_oracle.py`` running ``_observations`` on rig_r5base).
* ``test_the_compute_counts_are_pinned``: the computes Maya runs during each
  counted op are the tree's (``_COMPUTES``; 45d2c13's are kept beside them in
  ``_COMPUTES_45D2C13``, for the record). The contract allows a later change of
  these counts, but a change must re-pin ``_COMPUTES``, with its reason.
"""

import builtins
import os
import shutil
import tempfile

from maya import cmds

from rig import List, Node, functions
from rig.nodetypes import _base
from rig._tests._base import MayaTestCase


# --------------------------------------------------------------------------- #
#  The compute counter (`_observations` runs on 45d2c13 too: public API only)
# --------------------------------------------------------------------------- #

_COUNTER = "RIG_R5_EVAL_COMPUTES"

# a MEL statement for an expression: one count per run of its compute
_COUNT_STATEMENT = 'python("import builtins; builtins.%s = getattr(builtins, \'%s\', 0) + 1");' % (
    _COUNTER, _COUNTER
)

# A scripted DG node whose matrix output is uncached: every pull computes it,
# and each compute counts (the review's review_uncached_node.py)
_UNCACHED_PLUGIN = '''
import builtins

import maya.api.OpenMaya as om


def maya_useNewAPI():
    pass


class RigR5Uncached(om.MPxNode):
    kName = "rigR5Uncached"
    kId   = om.MTypeId(0x0007F2A2)
    inp   = None
    out_m = None

    @staticmethod
    def creator():
        return RigR5Uncached()

    @staticmethod
    def initialize():
        n = om.MFnNumericAttribute()
        RigR5Uncached.inp = n.create("inp", "inp", om.MFnNumericData.kDouble, 0.0)
        n.keyable = True
        m = om.MFnMatrixAttribute()
        RigR5Uncached.out_m = m.create("outMatrix", "omx")
        m.writable = False
        m.storable = False
        m.cached   = False
        for attr in (RigR5Uncached.inp, RigR5Uncached.out_m):
            om.MPxNode.addAttribute(attr)
        om.MPxNode.attributeAffects(RigR5Uncached.inp, RigR5Uncached.out_m)

    def compute(self, plug, data):
        builtins.%s = getattr(builtins, "%s", 0) + 1
        if plug != RigR5Uncached.out_m:
            return None
        matrix = om.MMatrix()
        matrix.setElement(3, 0, data.inputValue(RigR5Uncached.inp).asDouble())
        data.outputValue(RigR5Uncached.out_m).setMMatrix(matrix)
        data.setClean(plug)


def initializePlugin(obj):
    om.MFnPlugin(obj).registerNode(
        RigR5Uncached.kName, RigR5Uncached.kId, RigR5Uncached.creator, RigR5Uncached.initialize
    )


def uninitializePlugin(obj):
    om.MFnPlugin(obj).deregisterNode(RigR5Uncached.kId)
''' % (_COUNTER, _COUNTER)


def _computes():
    return getattr(builtins, _COUNTER, 0)


def _chain(error):
    """[the ``__context__`` chain as [type, message] pairs, the ``__cause__``]."""
    out, seen, cur = [], set(), error.__context__
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append([type(cur).__name__, str(cur)])
        cur = cur.__context__
    cause = error.__cause__
    return [out, None if cause is None else [type(cause).__name__, str(cause)]]


def _outcome(call):
    """``["ok", type, value]`` or ``["raises", type, message, chain]``."""
    try:
        value = call()
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ["raises", type(error).__name__, str(error), _chain(error)]
    if isinstance(value, float):
        value = round(value, 9)
    elif isinstance(value, str):
        value = str.__str__(value)
    elif isinstance(value, (list, tuple)):
        value = [round(v, 9) if isinstance(v, float) else v for v in value]
    return ["ok", type(value).__name__, value]


class _Record:
    """The steps of one case (label, outcome), and the computes Maya ran during
    the counted ones."""

    def __init__(self):
        self.steps    = []
        self.computes = {}

    def op(self, label, call, counted=False):
        before = _computes()
        self.steps.append([label, _outcome(call)])
        if counted:
            self.computes[label] = _computes() - before


def _expression(driven):
    """An expression ``ex`` that drives `driven` from ``drv.input1X`` and counts
    its runs."""
    cmds.expression(name="ex", s="%s = drv.input1X;\n%s\n" % (driven, _COUNT_STATEMENT))


# --------------------------------------------------------------------------- #
#  The cases
# --------------------------------------------------------------------------- #

def _multi_scene(driven):
    """cases_eval3.py: an `<<` into a bare multi root writes elements 0..n-1;
    the expression drives element 5, which the op never touches."""
    cmds.createNode("multiplyDivide", name="drv")
    cmds.createNode("multiplyDivide", name="md")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.addAttr("md", longName="dmulti", attributeType="double", multi=True)
    cmds.setAttr("md.dmulti[0]", 0.0)
    _expression(driven)
    cmds.setAttr("drv.input1X", 3.0)  # the expression's output is dirty again


def _multi_values(rec, root):
    rec.op("elements", lambda: list(cmds.getAttr(root, multiIndices=True) or []))
    for index in (0, 1, 5):
        rec.op(f"{root}[{index}]", lambda index=index: cmds.getAttr(f"{root}[{index}]"))


def _eval3_a(rec):
    _multi_scene("md.dmulti[5]")
    rec.op("md.dmulti << [1.0, 2.0]", lambda: str(Node("md").dmulti << [1.0, 2.0]), counted=True)
    _multi_values(rec, "md.dmulti")


def _eval3_b(rec):
    _multi_scene("md.dmulti[5]")

    def op():
        md = Node("md")
        return str(md.dmulti << List([md.outputX, md.outputY]))

    rec.op("md.dmulti << List([md.outputX, md.outputY])", op, counted=True)
    _multi_values(rec, "md.dmulti")


def _eval3_c(rec):
    _multi_scene("pma.input1D[5]")
    rec.op(
        "pma.input1D << [1.0, 2.0]", lambda: str(Node("pma").input1D << [1.0, 2.0]), counted=True
    )
    _multi_values(rec, "pma.input1D")


def _eval3_d(rec):
    _multi_scene("md.dmulti[5]")
    # a number into the bare root: the auto-index (the next free index, 1)
    rec.op("md.dmulti << 4.0", lambda: str(Node("md").dmulti << 4.0), counted=True)
    _multi_values(rec, "md.dmulti")


def _wrong_size_scene(dst):
    """cases_eval2.py: an `<<` that raises after the dst's data type query (a
    sequence of the wrong size); the expression drives the dst."""
    cmds.createNode("multiplyDivide", name="drv")
    cmds.createNode("multiplyDivide", name="md")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.setAttr("pma.input3D[0].input3Dx", 1.0)
    cmds.addAttr("md", longName="dyn", attributeType="double")
    cmds.addAttr("md", longName="dynEnum", attributeType="enum", enumName="a:b:c")
    _expression(dst)
    cmds.setAttr("drv.input1X", 2.0)  # the expression's output is dirty again


def _eval2(dst, label, op):
    def case(rec):
        _wrong_size_scene(dst)
        rec.op(label, op, counted=True)
        rec.op(dst, lambda: cmds.getAttr(dst))

    return case


def _dynamic_ok_then_wrong():
    md = Node("md")
    try:
        md.dyn << [1.0, 2.0]
    except ValueError as error:
        return "ValueError: %s" % error
    return "no error"


def _static_prime_a(rec):
    """cases_static_prime.py: a `<<` into a fixed kind no longer fills the static
    type cache, so a later `data_type` of another plug of that (type, attr)
    queries Maya, and evaluates the plug's dirty driver."""
    cmds.createNode("multiplyDivide", name="drv")
    cmds.createNode("clamp", name="c1")
    cmds.createNode("clamp", name="c2")
    _expression("c2.minR")
    rec.op("c1.minR << 0.5", lambda: str(Node("c1").minR << 0.5))
    cmds.setAttr("drv.input1X", 3.0)  # the driver of c2.minR is dirty
    rec.op("c2.minR.data_type", lambda: Node("c2").minR.get_data_type(), counted=True)
    rec.op("c1.minR", lambda: cmds.getAttr("c1.minR"))
    rec.op("c2.minR", lambda: cmds.getAttr("c2.minR"))


def _static_prime_b(rec):
    cmds.createNode("multiplyDivide", name="drv")
    cmds.createNode("clamp", name="k1")
    cmds.createNode("clamp", name="k2")
    _expression("k2.maxG")
    rec.op("k1.maxG << 2.0", lambda: str(Node("k1").maxG << 2.0))
    cmds.setAttr("drv.input1X", 4.0)
    rec.op("str(k2.maxG.data_type)", lambda: str(Node("k2").maxG.get_data_type()), counted=True)
    rec.op("k1.maxG", lambda: cmds.getAttr("k1.maxG"))
    rec.op("k2.maxG", lambda: cmds.getAttr("k2.maxG"))


def _uncached_f(rec):
    """cases_uncached.py case_f: ``functions.choice`` of an uncached matrix
    output, whose type query computes it. The output's type is in the static
    type cache, as the review's earlier cases left it (another node's query
    puts it there), so only the choice's queries can pull the output."""
    cmds.createNode("rigR5Uncached", name="primer")
    rec.op("primer.outMatrix.data_type", lambda: Node("primer").outMatrix.get_data_type())
    cmds.createNode("rigR5Uncached", name="u")
    rec.op(
        "functions.choice([u.outMatrix, u.outMatrix], 0)",
        lambda: str(functions.choice([Node("u").outMatrix, Node("u").outMatrix], 0)),
        counted=True,
    )
    rec.op("choice1.selector", lambda: cmds.getAttr("choice1.selector"))
    rec.op("choice1.input", lambda: list(cmds.getAttr("choice1.input", multiIndices=True) or []))


_CASES = (
    ("eval3 a: a sequence into a dynamic multi", _eval3_a),
    ("eval3 b: a List of plugs into a dynamic multi", _eval3_b),
    ("eval3 c: a sequence into a static multi", _eval3_c),
    ("eval3 d: a number into a dynamic multi (the auto-index)", _eval3_d),
    ("eval2 a: a dynamic double, the wrong size", _eval2(
        "md.dyn", "md.dyn << [1.0, 2.0, 3.0]", lambda: Node("md").dyn << [1.0, 2.0, 3.0])),
    ("eval2 b: an element's child, the wrong size", _eval2(
        "pma.input3D[0].input3Dx", "pma.input3D[0].input3Dx << [1.0, 2.0]",
        lambda: Node("pma").input3D[0].input3Dx << [1.0, 2.0])),
    ("eval2 c: a static scalar, the wrong size", _eval2(
        "md.input1X", "md.input1X << [1.0, 2.0, 3.0]",
        lambda: Node("md").input1X << [1.0, 2.0, 3.0])),
    ("eval2 d: an element compound, the wrong size", _eval2(
        "pma.input3D[0].input3Dy", "pma.input3D[0] << [1.0, 2.0]",
        lambda: Node("pma").input3D[0] << [1.0, 2.0])),
    ("eval2 e: a dynamic double, the error caught", _eval2(
        "md.dyn", "md.dyn << [1.0, 2.0] (caught)", _dynamic_ok_then_wrong)),
    ("static prime a: a data_type after a << of that attribute", _static_prime_a),
    ("static prime b: the same through str()", _static_prime_b),
    ("uncached f: functions.choice of an uncached output", _uncached_f),
)


def _scene(defaults):
    """The nodes a case made, as [name, type], and every connection into them,
    as [source, destination], sorted."""
    nodes = sorted(set(cmds.ls()) - defaults)
    pairs = set()
    for node in nodes:
        found = cmds.listConnections(
            node, connections=True, plugs=True, source=True, destination=False
        ) or []
        pairs.update(zip(found[1::2], found[::2]))
    return [[node, cmds.nodeType(node)] for node in nodes], [list(pair) for pair in sorted(pairs)]


def _observations():
    """{case: {"steps", "nodes", "connections", "computes"}}: each case in a new
    scene with an infinite undo queue (restored after) and an empty static type
    cache, the uncached node's plug-in loaded for them (unloaded after)."""
    table  = {}
    folder = tempfile.mkdtemp(prefix="rig_r5_eval_")
    path   = os.path.join(folder, "rigR5Uncached.py")
    with open(path, "w") as handle:
        handle.write(_UNCACHED_PLUGIN)
    try:
        cmds.loadPlugin(path, quiet=True)
        for name, case in _CASES:
            cmds.file(new=True, force=True)
            cmds.undoInfo(state=True, infinity=True)
            cmds.flushUndo()
            _base._STATIC_DATA_TYPE.clear()
            defaults = set(cmds.ls())
            rec = _Record()
            case(rec)
            nodes, connections = _scene(defaults)
            table[name] = {
                "steps":       rec.steps,
                "nodes":       nodes,
                "connections": connections,
                "computes":    rec.computes,
            }
    finally:
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        cmds.undoInfo(state=True, infinity=True)
        if cmds.pluginInfo("rigR5Uncached", query=True, loaded=True):
            cmds.unloadPlugin("rigR5Uncached")
        shutil.rmtree(folder, ignore_errors=True)
    return table


# --------------------------------------------------------------------------- #
#  The tests
# --------------------------------------------------------------------------- #

class TestTheEvalContract(MayaTestCase):
    TEST_START_NEW_SCENE = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.table = _observations()

    def test_the_values_exceptions_and_connections_are_45d2c13s(self):
        self.assertEqual(sorted(self.table), sorted(_ORACLE_45D2C13))
        for name, expected in _ORACLE_45D2C13.items():
            observed = self.table[name]
            for key in ("steps", "nodes", "connections"):
                with self.subTest(case=name, part=key):
                    self.assertEqual(observed[key], expected[key])

    def test_the_compute_counts_are_pinned(self):
        """The computes Maya ran during each counted op on this tree. How many
        and when is not part of rig's behaviour (lead decision D1, 2026-10-02),
        so a later change of a count is allowed, but it must re-pin `_COMPUTES`
        here, with its reason (45d2c13's counts: `_COMPUTES_45D2C13`)."""
        self.assertEqual(sorted(_COMPUTES), sorted(self.table))
        for name, pinned in _COMPUTES.items():
            with self.subTest(case=name):
                self.assertEqual(self.table[name]["computes"], pinned)


# The values, exceptions, nodes and connections `_observations` gave on 45d2c13
# (rig_r5base), written by runs\r5\AF\gen_oracle.py: {case: {"steps": [[label,
# ["ok", type, value] or ["raises", type, message, [context chain, cause]]]],
# "nodes": [[name, type]], "connections": [[source, destination]]}}
_ORACLE_45D2C13 = {
    'eval2 a: a dynamic double, the wrong size': {
        "steps": [
            ['md.dyn << [1.0, 2.0, 3.0]',
             ['raises', 'ValueError',
              "Cannot inject sequence of size 3 into scalar attribute 'md.dyn' (expected 1 element).",
              [[], None]]],
            ['md.dyn', ['ok', 'float', 2.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.dyn'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    "eval2 b: an element's child, the wrong size": {
        "steps": [
            ['pma.input3D[0].input3Dx << [1.0, 2.0]',
             ['raises', 'ValueError',
              "Cannot inject sequence of size 2 into scalar attribute 'pma.input3D[0].input3Dx' (expected 1 element).",
              [[], None]]],
            ['pma.input3D[0].input3Dx', ['ok', 'float', 2.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'pma.input3D[0].input3Dx'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval2 c: a static scalar, the wrong size': {
        "steps": [
            ['md.input1X << [1.0, 2.0, 3.0]',
             ['raises', 'ValueError',
              "Cannot inject sequence of size 3 into scalar attribute 'md.input1X' (expected 1 element).",
              [[], None]]],
            ['md.input1X', ['ok', 'float', 2.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.input1X'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval2 d: an element compound, the wrong size': {
        "steps": [
            ['pma.input3D[0] << [1.0, 2.0]',
             ['raises', 'ValueError',
              "Cannot inject sequence of size 2 into 3-channel compound attribute 'pma.input3D[0]'.",
              [[], None]]],
            ['pma.input3D[0].input3Dy', ['ok', 'float', 2.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'pma.input3D[0].input3Dy'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval2 e: a dynamic double, the error caught': {
        "steps": [
            ['md.dyn << [1.0, 2.0] (caught)',
             ['ok', 'str',
              "ValueError: Cannot inject sequence of size 2 into scalar attribute 'md.dyn' (expected 1 element)."]],
            ['md.dyn', ['ok', 'float', 2.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.dyn'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval3 a: a sequence into a dynamic multi': {
        "steps": [
            ['md.dmulti << [1.0, 2.0]', ['ok', 'str', 'md.dmulti']],
            ['elements', ['ok', 'list', [0, 1, 5]]],
            ['md.dmulti[0]', ['ok', 'float', 1.0]],
            ['md.dmulti[1]', ['ok', 'float', 2.0]],
            ['md.dmulti[5]', ['ok', 'float', 3.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.dmulti[5]'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval3 b: a List of plugs into a dynamic multi': {
        "steps": [
            ['md.dmulti << List([md.outputX, md.outputY])', ['ok', 'str', 'md.dmulti']],
            ['elements', ['ok', 'list', [0, 1, 5]]],
            ['md.dmulti[0]', ['ok', 'float', 0.0]],
            ['md.dmulti[1]', ['ok', 'float', 0.0]],
            ['md.dmulti[5]', ['ok', 'float', 3.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.dmulti[5]'],
            ['md.outputX', 'md.dmulti[0]'],
            ['md.outputY', 'md.dmulti[1]'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval3 c: a sequence into a static multi': {
        "steps": [
            ['pma.input1D << [1.0, 2.0]', ['ok', 'str', 'pma.input1D']],
            ['elements', ['ok', 'list', [0, 1, 5]]],
            ['pma.input1D[0]', ['ok', 'float', 1.0]],
            ['pma.input1D[1]', ['ok', 'float', 2.0]],
            ['pma.input1D[5]', ['ok', 'float', 3.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'pma.input1D[5]'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'eval3 d: a number into a dynamic multi (the auto-index)': {
        "steps": [
            ['md.dmulti << 4.0', ['ok', 'str', 'md.dmulti']],
            ['elements', ['ok', 'list', [0, 1, 5]]],
            ['md.dmulti[0]', ['ok', 'float', 0.0]],
            ['md.dmulti[1]', ['ok', 'float', 4.0]],
            ['md.dmulti[5]', ['ok', 'float', 3.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['md', 'multiplyDivide'],
            ['pma', 'plusMinusAverage'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'md.dmulti[5]'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'static prime a: a data_type after a << of that attribute': {
        "steps": [
            ['c1.minR << 0.5', ['ok', 'str', 'c1.minR']],
            ['c2.minR.data_type', ['ok', 'str', 'float']],
            ['c1.minR', ['ok', 'float', 0.5]],
            ['c2.minR', ['ok', 'float', 3.0]],
        ],
        "nodes": [
            ['c1', 'clamp'],
            ['c2', 'clamp'],
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'c2.minR'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'static prime b: the same through str()': {
        "steps": [
            ['k1.maxG << 2.0', ['ok', 'str', 'k1.maxG']],
            ['str(k2.maxG.data_type)', ['ok', 'str', 'float']],
            ['k1.maxG', ['ok', 'float', 2.0]],
            ['k2.maxG', ['ok', 'float', 4.0]],
        ],
        "nodes": [
            ['drv', 'multiplyDivide'],
            ['ex', 'expression'],
            ['k1', 'clamp'],
            ['k2', 'clamp'],
        ],
        "connections": [
            ['drv.input1X', 'ex.input[0]'],
            ['ex.output[0]', 'k2.maxG'],
            ['time1.outTime', 'ex.time'],
        ],
    },
    'uncached f: functions.choice of an uncached output': {
        "steps": [
            ['primer.outMatrix.data_type', ['ok', 'str', 'matrix']],
            ['functions.choice([u.outMatrix, u.outMatrix], 0)', ['ok', 'str', 'choice1.output']],
            ['choice1.selector', ['ok', 'int', 0]],
            ['choice1.input', ['ok', 'list', [0, 1]]],
        ],
        "nodes": [
            ['choice1', 'choice'],
            ['primer', 'rigR5Uncached'],
            ['u', 'rigR5Uncached'],
        ],
        "connections": [
            ['u.outMatrix', 'choice1.input[0]'],
            ['u.outMatrix', 'choice1.input[1]'],
        ],
    },
}

# The computes Maya ran during each counted op on 45d2c13, for the record:
# {case: {label: computes}}
_COMPUTES_45D2C13 = {
    'eval2 a: a dynamic double, the wrong size': {
        'md.dyn << [1.0, 2.0, 3.0]': 1,
    },
    "eval2 b: an element's child, the wrong size": {
        'pma.input3D[0].input3Dx << [1.0, 2.0]': 1,
    },
    'eval2 c: a static scalar, the wrong size': {
        'md.input1X << [1.0, 2.0, 3.0]': 1,
    },
    'eval2 d: an element compound, the wrong size': {
        'pma.input3D[0] << [1.0, 2.0]': 1,
    },
    'eval2 e: a dynamic double, the error caught': {
        'md.dyn << [1.0, 2.0] (caught)': 1,
    },
    'eval3 a: a sequence into a dynamic multi': {
        'md.dmulti << [1.0, 2.0]': 1,
    },
    'eval3 b: a List of plugs into a dynamic multi': {
        'md.dmulti << List([md.outputX, md.outputY])': 1,
    },
    'eval3 c: a sequence into a static multi': {
        'pma.input1D << [1.0, 2.0]': 1,
    },
    'eval3 d: a number into a dynamic multi (the auto-index)': {
        'md.dmulti << 4.0': 1,
    },
    'static prime a: a data_type after a << of that attribute': {
        'c2.minR.data_type': 0,
    },
    'static prime b: the same through str()': {
        'str(k2.maxG.data_type)': 0,
    },
    'uncached f: functions.choice of an uncached output': {
        'functions.choice([u.outMatrix, u.outMatrix], 0)': 1,
    },
}

# The computes Maya runs during each counted op on this tree, pinned (see
# `test_the_compute_counts_are_pinned`; a step that changes one re-pins it here,
# with its reason): round 5 Phase A's head (cac7cdf) and step AF
_COMPUTES = {
    'eval2 a: a dynamic double, the wrong size': {
        'md.dyn << [1.0, 2.0, 3.0]': 0,
    },
    "eval2 b: an element's child, the wrong size": {
        'pma.input3D[0].input3Dx << [1.0, 2.0]': 0,
    },
    'eval2 c: a static scalar, the wrong size': {
        'md.input1X << [1.0, 2.0, 3.0]': 0,
    },
    'eval2 d: an element compound, the wrong size': {
        'pma.input3D[0] << [1.0, 2.0]': 0,
    },
    'eval2 e: a dynamic double, the error caught': {
        'md.dyn << [1.0, 2.0] (caught)': 0,
    },
    'eval3 a: a sequence into a dynamic multi': {
        'md.dmulti << [1.0, 2.0]': 0,
    },
    'eval3 b: a List of plugs into a dynamic multi': {
        'md.dmulti << List([md.outputX, md.outputY])': 0,
    },
    'eval3 c: a sequence into a static multi': {
        'pma.input1D << [1.0, 2.0]': 0,
    },
    'eval3 d: a number into a dynamic multi (the auto-index)': {
        'md.dmulti << 4.0': 0,
    },
    'static prime a: a data_type after a << of that attribute': {
        'c2.minR.data_type': 1,
    },
    'static prime b: the same through str()': {
        'str(k2.maxG.data_type)': 1,
    },
    'uncached f: functions.choice of an uncached output': {
        'functions.choice([u.outMatrix, u.outMatrix], 0)': 0,
    },
}
