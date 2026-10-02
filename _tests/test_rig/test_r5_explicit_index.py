"""Round 5, step A2: the library loops that fill a multi of the node they just
made write each operand at its own index (``node.input[i] << obj``) while every
operand so far filled one element, and keep the auto-index ``node.input << obj``
from the first other operand on: ``functions.choice`` (``choice.input``) and the
legacy ``plusMinusAverage`` (``input3D`` / ``input1D``; Maya 2024+ builds it for
compound operands). The node is new, so the lowest free index (the one the
auto-index finds) is the operand's position; the explicit form skips the root
queries the auto-index makes (``attributeQuery``, ``get_next_available_index``).

* ``TestFillsOneElement``: the operand rule (``types._fills_one_element``): a
  number, or an attribute that is not a multi; never raises.
* ``TestTheOldLoopsOutcomes``: for the token kinds of the round-5 probe
  (``runs\\r5\\P0\\probe_explicit_index.py``: plugs of DG and DAG nodes, scalar,
  compound, matrix and quaternion ones, numbers, a multi root, a mix, ``None``
  first, a List) and more (a multi root between two plugs, a sequence between two
  plugs, a generic source, an instanced shape's world matrix, a component, a
  namespaced and a renamed node, a deleted dynamic attribute, a deleted and a
  freed node's plug), each library call gives what the same call gives with the
  old loops (the rule patched to answer False, so every operand takes
  ``multi << obj``, the e9287a8 code): the result or the exception (type,
  message, context, cause), what it prints (Maya's command output, rig's log
  records, Python warnings, stdout / stderr), every node and connection, the
  multi indices and values, the selection and the number of undo steps.
* ``TestTheRootQueriesAreSkipped``: ``functions.choice([a, b, c])`` and the
  legacy ``plusMinusAverage`` of three operands make no ``attributeQuery`` and no
  ``get_next_available_index`` call (the old loops: 3 and 3).
* ``TestAHookOfItsOwn``: a choice class whose data type hook is not rig's keeps
  the auto-index for every token (the element's shorthand facts would read its
  data type once more): the hook runs as often as with the old loop.
* ``TestMultMatrixKeepsTheAutoIndex``: ``multMatrix`` / ``addMatrix`` are not
  changed: on an element shorthand reads a matrix where the root reads none, so a
  scalar, compound, quaternion or transform-channel token converts on an element
  and raises on the root (the round-5 probe: 5 token kinds differ).
"""

import contextlib
import inspect
import io
import logging
import warnings
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

import rig
from rig import List, Node, functions
from rig._internal import math_nodes, types
from rig.nodetypes._base import Attribute
from rig.nodetypes.choice import Choice
from rig.nodetypes.dg_node import _keeps_query
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, scene_connections


def _never(obj):
    """The rule answering False: every operand takes the auto-index (the old loops)."""
    return False


class _OldLoops:
    """Context: the library loops run as at e9287a8 (``multi << obj`` for every operand)."""

    def __enter__(self):
        self._patches = [
            mock.patch.object(functions, "_fills_one_element", _never),
            mock.patch.object(math_nodes, "_fills_one_element", _never),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _brief(error):
    return None if error is None else [type(error).__name__, str(error)]


def _multis():
    """{plug: [indices, values]} of the multis of the scene's choice, plusMinusAverage,
    multMatrix and addMatrix nodes (the values of the plusMinusAverage inputs)."""
    out = {}
    for name in cmds.ls(type=["choice", "plusMinusAverage", "multMatrix", "addMatrix"]) or []:
        for attr in ("input", "input1D", "input3D", "matrixIn"):
            if not cmds.attributeQuery(attr, node=name, exists=True):
                continue
            ids    = list(cmds.getAttr(f"{name}.{attr}", multiIndices=True) or [])
            values = None
            if attr in ("input1D", "input3D"):
                values = [cmds.getAttr(f"{name}.{attr}[{i}]") for i in ids]
            out[f"{name}.{attr}"] = [ids, values]
    return out


class _Output(logging.Handler):
    """What a call prints: Maya's command output (warnings, errors, info), the
    records of the ``rig`` loggers (at every level), Python warnings, and the text
    written to ``sys.stdout`` / ``sys.stderr``."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.maya    = []
        self.records = []

    def emit(self, record):
        self.records.append((record.name, record.levelname, record.getMessage()))

    def __enter__(self):
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(
            lambda message, kind, *args: self.maya.append((int(kind), message))
        )
        self._logger = logging.getLogger("rig")
        self._level  = self._logger.level
        self._logger.addHandler(self)
        self._logger.setLevel(logging.DEBUG)
        self._warnings = warnings.catch_warnings(record=True)
        self.warned    = self._warnings.__enter__()
        warnings.simplefilter("always")
        self._out, self._err = io.StringIO(), io.StringIO()
        self._streams = (contextlib.redirect_stdout(self._out), contextlib.redirect_stderr(self._err))
        for stream in self._streams:
            stream.__enter__()
        return self

    def __exit__(self, *exc):
        for stream in reversed(self._streams):
            stream.__exit__(None, None, None)
        self._warnings.__exit__(None, None, None)
        self._logger.setLevel(self._level)
        self._logger.removeHandler(self)
        OpenMaya.MMessage.removeCallback(self._callback)
        return False

    def state(self):
        return {
            "maya":     self.maya,
            "logs":     self.records,
            "warnings": [(w.category.__name__, str(w.message)) for w in self.warned],
            "stdout":   self._out.getvalue(),
            "stderr":   self._err.getvalue(),
        }


def _record(call):
    """The outcome of ``call()`` and the scene it leaves: the result (its str) or the
    exception (type, message, ``__context__`` / ``__cause__`` type and message,
    ``__suppress_context__``), what it printed (``_Output``), every node (name and
    type), every connection, the multis (``_multis``), the selection, and then the
    number of undo steps the queue holds (the scene is undone to count them)."""
    with _Output() as output:
        try:
            result = ["returns", str(call())]
        except Exception as error:  # noqa: BLE001 - the error is the thing compared
            result = [
                "raises",
                type(error).__name__,
                str(error),
                _brief(error.__context__),
                _brief(error.__cause__),
                error.__suppress_context__,
            ]
    state = {
        "result":      result,
        "output":      output.state(),
        "nodes":       sorted((n, cmds.nodeType(n)) for n in cmds.ls(long=True)),
        "connections": scene_connections(),
        "multis":      _multis(),
        "selection":   cmds.ls(selection=True, long=True),
    }
    steps  = 0
    quiet  = cmds.scriptEditorInfo(query=True, suppressInfo=True)
    cmds.scriptEditorInfo(suppressInfo=True)  # no "Undo: " line per step
    try:
        while not cmds.undoInfo(query=True, undoQueueEmpty=True) and steps < 1000:
            cmds.undo()
            steps += 1
    finally:
        cmds.scriptEditorInfo(suppressInfo=quiet)
    state["undo_steps"] = steps
    return state


# --------------------------------------------------------------------------- #
#  The token kinds (each built in a fresh scene, with fixed names)
# --------------------------------------------------------------------------- #

def _sources():
    md  = Node(cmds.createNode("multiplyDivide", name="srcmd"))
    md2 = Node(cmds.createNode("multiplyDivide", name="srcmd2"))
    t   = Node(cmds.spaceLocator(name="srcloc")[0])
    t2  = Node(cmds.spaceLocator(name="srcloc2")[0])
    md.input1X << 2.0
    md2.input1Y << 3.0
    return md, md2, t, t2


def _tokens(kind):
    """(tokens, anchors): the operands of `kind`, and a compound and a scalar plug
    (``srcmd.output`` / ``srcmd.outputZ``) a plusMinusAverage site puts first."""
    md, md2, t, t2 = _sources()
    anchors = (md.output, md.outputZ)
    if kind == "scalar_dg":
        return [md.outputX, md2.outputY], anchors
    if kind == "scalar_transform":
        return [t.tx, t2.ry], anchors
    if kind == "compound_dg":
        return [md.output, md2.output], anchors
    if kind == "compound_transform":
        return [t.t, t2.r], anchors
    if kind == "matrix_transform":
        return [t.worldMatrix[0], t2.worldMatrix[0]], anchors
    if kind == "matrix_dg":
        mm  = Node(cmds.createNode("multMatrix", name="srcmm"))
        mm2 = Node(cmds.createNode("multMatrix", name="srcmm2"))
        return [mm.matrixSum, mm2.matrixSum], anchors
    if kind == "quaternion":
        cmds.loadPlugin("matrixNodes", quiet=True)
        dm  = Node(cmds.createNode("decomposeMatrix", name="srcdm"))
        dm2 = Node(cmds.createNode("decomposeMatrix", name="srcdm2"))
        return [dm.outputQuat, dm2.outputQuat], anchors
    if kind == "number":
        # 0.0 is the default: a set that changes no value still fills its element
        return [1.5, 0.0, True], anchors
    if kind == "multi_root":
        pma = Node(cmds.createNode("plusMinusAverage", name="srcpma"))
        cmds.setAttr("srcpma.input1D[0]", 1.0)
        cmds.setAttr("srcpma.input1D[2]", 3.0)
        return [pma.input1D, md.outputX], anchors
    if kind == "mixed":
        return [md.outputX, t.t, t.worldMatrix[0], 2.0], anchors
    if kind == "none_first":
        return [None, md.outputX], anchors
    if kind == "list":
        return [List([md.outputX, md2.outputX]), md.outputY], anchors
    if kind == "multi_root_between":
        # a readable multi with elements 1 and 2: after it the lowest free index is 3,
        # not the next token's position (2)
        net = Node(cmds.createNode("network", name="srcnet"))
        cmds.addAttr("srcnet", longName="m", attributeType="double", multi=True)
        cmds.setAttr("srcnet.m[1]", 4.0)
        cmds.setAttr("srcnet.m[2]", 5.0)
        return [md.outputX, net.m, md2.outputY], anchors
    if kind == "sequence_between":
        return [md.output, [1.0, 2.0, 3.0], md2.output, 4.0], anchors
    if kind == "generic_source":
        ch = Node(cmds.createNode("choice", name="srcchoice"))
        ch.input[0] << md.output
        return [ch.output, md2.outputX], anchors
    if kind == "instanced":
        shape = cmds.listRelatives("srcloc", shapes=True, fullPath=True)[0]
        cmds.instance("srcloc", name="srcinst")
        second = Node(f"|srcinst|{shape.split('|')[-1]}")
        return [second.worldMatrix[1], second.worldMatrix, md.outputX], anchors
    if kind == "component":
        cube = cmds.polyCube(name="srccube", constructionHistory=False)[0]
        shape = Node(cmds.listRelatives(cube, shapes=True)[0])
        return [shape.vtx[3], md.outputX], anchors
    if kind == "namespaced":
        cmds.namespace(add="srcns")
        nmd = Node(cmds.createNode("multiplyDivide", name="srcns:srcmd"))
        return [nmd.outputX, nmd.output], anchors
    if kind == "renamed":
        tokens = [md.outputX, md2.output]
        cmds.rename("srcmd", "srcmd_renamed")
        return tokens, anchors
    if kind == "deleted_attribute":
        # a dynamic attribute deleted to the undo queue (its node lives on)
        cmds.addAttr("srcmd2", longName="dyn", attributeType="double")
        tokens = [md.outputX, md2.dyn, md.outputY]
        cmds.deleteAttr("srcmd2.dyn")
        return tokens, anchors
    if kind == "deleted_second":
        # deleted to the undo queue: alive, not valid
        tokens = [md.outputX, md2.outputY]
        cmds.delete("srcmd2")
        return tokens, anchors
    if kind == "deleted_first":
        tokens = [md2.output, md.outputX]
        cmds.delete("srcmd2")
        return tokens, anchors
    if kind == "freed":
        # a plug of a node a new scene freed, then the scene's own sources
        held = Node(cmds.createNode("multiplyDivide", name="gone")).outputX
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        md, md2, t, t2 = _sources()
        return [md.outputX, held], (md.output, md.outputZ)
    raise ValueError(kind)


_PUBLIC_KINDS = (
    "scalar_dg", "scalar_transform", "compound_dg", "compound_transform",
    "matrix_transform", "matrix_dg", "quaternion", "number", "multi_root", "mixed",
    "none_first", "list", "multi_root_between", "sequence_between", "generic_source",
    "instanced", "component", "namespaced", "renamed", "deleted_attribute",
    "deleted_second", "freed",
)

# the kinds a public call rejects before the loop runs (a deleted node's plug raises
# in the memo key, a freed one in the operand check), run through the loops' own
# functions as well, with a plain kind for comparison
_RAW_KINDS = (
    "scalar_dg", "compound_dg", "deleted_attribute", "deleted_second", "deleted_first", "freed",
)


def _choice(tokens, anchors):
    return functions.choice(tokens)


def _choice_selector(tokens, anchors):
    return functions.choice(tokens, selector=anchors[1])


def _sum3(tokens, anchors):
    # a compound operand first: Maya 2024+ builds the legacy plusMinusAverage
    return functions.sum([anchors[0]] + list(tokens))


def _sum1(tokens, anchors):
    # set_options(maya_version=2023): the legacy plusMinusAverage for any operands
    return functions.sum([anchors[1]] + list(tokens))


def _raw_choice(tokens, anchors):
    return inspect.unwrap(functions.choice)(tokens)


def _raw_pma3(tokens, anchors):
    return math_nodes._plus_minus_average_legacy(anchors[0], *tokens, operation=1, name="add1")


def _raw_pma1(tokens, anchors):
    return math_nodes._plus_minus_average_legacy(anchors[1], *tokens, operation=2, name="sub1")


class TestFillsOneElement(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_numbers_and_attributes_that_are_no_multi(self):
        md  = Node(cmds.createNode("multiplyDivide"))
        pma = Node(cmds.createNode("plusMinusAverage"))
        t   = Node(cmds.createNode("transform"))
        cube = Node(cmds.listRelatives(cmds.polyCube(constructionHistory=False)[0], shapes=True)[0])
        yes = [1, 2.5, 0.0, True, np.float64(2.0), np.int32(3), md.outputX, md.output,
               md.find_attr("input1X"), t.worldMatrix[0], pma.input1D[0], pma.input3D[2],
               pma.input3D[0].input3Dx, cube.vtx[1]]
        no  = [None, "md.outputX", [1, 2], (1.0,), np.array([1.0]), List([md.outputX]),
               md, pma.input1D, pma.input3D, t.worldMatrix, cube.vtx, object()]
        for obj in yes:
            with self.subTest(obj=repr(obj)):
                self.assertIs(types._fills_one_element(obj), True)
        for obj in no:
            with self.subTest(obj=repr(obj)):
                self.assertIs(types._fills_one_element(obj), False)

    def test_a_deleted_and_a_freed_nodes_plug(self):
        deleted = Node(cmds.createNode("multiplyDivide", name="del")).outputX
        cmds.delete("del")
        # alive on the undo queue: its `<<` raises "del already deleted!" in both forms
        self.assertIs(types._fills_one_element(deleted), True)
        freed = Node(cmds.createNode("multiplyDivide", name="gone")).outputX
        cmds.file(new=True, force=True)
        # `is_multi` raises "... already deleted!": not one (its `<<` raises it too)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            freed.is_multi
        self.assertIs(types._fills_one_element(freed), False)


class TestTheOldLoopsOutcomes(UndoWalk, MayaTestCase):
    """Every library call against the same call with the old loops, in fresh scenes."""

    def tearDown(self):
        try:
            rig.set_options(maya_version=None)
        finally:
            super().tearDown()

    def _run(self, site, kind, old):
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        tokens, anchors = _tokens(kind)
        if old:
            with _OldLoops():
                return _record(lambda: site(tokens, anchors))
        return _record(lambda: site(tokens, anchors))

    def _compare(self, site, kinds, maya_version=None):
        rig.set_options(maya_version=maya_version)
        for kind in kinds:
            with self.subTest(site=site.__name__, kind=kind):
                new = self._run(site, kind, old=False)
                old = self._run(site, kind, old=True)
                for key in old:
                    self.assertEqual(new[key], old[key], f"{site.__name__} {kind}: {key}")

    def test_choice(self):
        self._compare(_choice, _PUBLIC_KINDS)

    def test_choice_with_a_selector(self):
        self._compare(_choice_selector, ("scalar_dg", "mixed", "multi_root_between"))

    def test_the_legacy_plus_minus_average_of_compounds(self):
        self._compare(_sum3, _PUBLIC_KINDS)

    def test_the_legacy_plus_minus_average_input1D(self):
        self._compare(_sum1, _PUBLIC_KINDS, maya_version=2023)

    def test_a_plus_b(self):
        def add(tokens, anchors):
            return anchors[0] + tokens[0]

        self._compare(add, ("scalar_dg", "compound_dg", "matrix_transform", "quaternion", "number"))

    def test_the_loops_on_their_own(self):
        for site in (_raw_choice, _raw_pma3, _raw_pma1):
            self._compare(site, _RAW_KINDS)

    def test_the_oracle_is_the_auto_index(self):
        """With the rule patched, each operand of the loop takes ``multi << obj``."""
        cmds.file(new=True, force=True)
        tokens, anchors = _tokens("scalar_dg")
        calls = []
        orig  = Attribute.get_next_available_index

        def spy(attr):
            calls.append(str.__str__(attr))
            return orig(attr)

        with _OldLoops(), mock.patch.object(Attribute, "get_next_available_index", spy):
            functions.choice(tokens)
            _sum3(tokens, anchors)
        self.assertEqual(calls, ["choice1.input"] * 2 + ["sum1.input3D"] * 3)


class TestTheRootQueriesAreSkipped(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        try:
            rig.set_options(maya_version=None)
        finally:
            super().tearDown()

    def _counts(self, call, old=False):
        queries = []
        nexts   = []
        query   = cmds.attributeQuery
        orig    = Attribute.get_next_available_index

        def query_spy(*args, **kwargs):
            queries.append(args)
            return query(*args, **kwargs)

        def next_spy(attr):
            nexts.append(str.__str__(attr))
            return orig(attr)

        with mock.patch.object(cmds, "attributeQuery", query_spy), \
                mock.patch.object(Attribute, "get_next_available_index", next_spy):
            if old:
                with _OldLoops():
                    call()
            else:
                call()
        return len(queries), len(nexts)

    def _sources(self, n, attr):
        return [getattr(Node(cmds.createNode("multiplyDivide")), attr) for _ in range(n)]

    def test_choice(self):
        self.assertEqual(self._counts(lambda: functions.choice(self._sources(3, "outputX"))), (0, 0))
        self.assertEqual(
            self._counts(lambda: functions.choice(self._sources(3, "outputX")), old=True), (3, 3)
        )

    def test_the_legacy_plus_minus_average(self):
        self.assertEqual(self._counts(lambda: functions.sum(self._sources(3, "output"))), (0, 0))
        self.assertEqual(
            self._counts(lambda: functions.sum(self._sources(3, "output")), old=True), (3, 3)
        )
        rig.set_options(maya_version=2023)
        self.assertEqual(self._counts(lambda: functions.sum(self._sources(3, "outputX"))), (0, 0))
        self.assertEqual(
            self._counts(lambda: functions.sum(self._sources(3, "outputX")), old=True), (3, 3)
        )

    def test_after_the_first_other_token_every_token_auto_indexes(self):
        """A multi root between two plugs: the plug after it takes the auto-index."""
        net = Node(cmds.createNode("network", name="net"))
        cmds.addAttr("net", longName="m", attributeType="double", multi=True)
        cmds.setAttr("net.m[1]", 4.0)
        cmds.setAttr("net.m[2]", 5.0)
        a, b = self._sources(2, "outputX")
        # the multi root's own query, then the auto-index of the plug after it
        self.assertEqual(self._counts(lambda: functions.choice([a, net.m, b])), (2, 1))
        self.assertEqual(cmds.getAttr("choice1.input", multiIndices=True), [0, 1, 2, 3])
        self.assertEqual(
            cmds.listConnections("choice1.input[3]", source=True, plugs=True),
            [str(b)],
        )


class TestAHookOfItsOwn(UndoWalk, MayaTestCase):
    """A choice class whose hook is not rig's keeps the auto-index: the element's
    shorthand facts would read its data type (and run the hook) once more."""

    def _run(self, old):
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        tokens, _ = _tokens("compound_transform")
        calls = []
        nexts = []
        hook  = Choice._attr_data_type_fallback
        orig  = Attribute.get_next_available_index

        def own_hook(node, attr):
            calls.append(str.__str__(attr))
            return hook(node, attr)

        def next_spy(attr):
            nexts.append(str.__str__(attr))
            return orig(attr)

        with mock.patch.object(Choice, "_attr_data_type_fallback", own_hook), \
                mock.patch.object(Attribute, "get_next_available_index", next_spy):
            self.assertFalse(_keeps_query(Choice))
            if old:
                with _OldLoops():
                    state = _record(lambda: functions.choice(tokens))
            else:
                state = _record(lambda: functions.choice(tokens))
        return state, calls, nexts

    def test_the_hook_runs_as_often_as_with_the_old_loop(self):
        self.assertTrue(_keeps_query(Choice))
        new, new_calls, new_nexts = self._run(old=False)
        old, old_calls, old_nexts = self._run(old=True)
        self.assertEqual(new, old)
        self.assertEqual(new_calls, old_calls)
        self.assertEqual(new_nexts, ["choice1.input"] * 2)
        self.assertEqual(new_nexts, old_nexts)


class TestMultMatrixKeepsTheAutoIndex(MayaTestCase):
    """``multMatrix.matrixIn`` / ``addMatrix.matrixIn`` keep ``matrixIn << obj``."""

    TEST_START_NEW_SCENE = True

    def _nexts(self, call):
        nexts = []
        orig  = Attribute.get_next_available_index

        def next_spy(attr):
            nexts.append(str.__str__(attr))
            return orig(attr)

        with mock.patch.object(Attribute, "get_next_available_index", next_spy):
            call()
        return nexts

    def test_every_input_takes_the_auto_index(self):
        ms = [Node(cmds.createNode("transform")).worldMatrix[0] for _ in range(3)]
        self.assertEqual(self._nexts(lambda: rig.matrix.multiply(*ms)), ["multMatrix1.matrixIn"] * 3)
        self.assertEqual(self._nexts(lambda: rig.matrix.add(*ms[:2])), ["addMatrix1.matrixIn"] * 2)

    def test_why_an_element_is_not_the_root_there(self):
        """On an element shorthand reads a matrix where the root reads none: a
        translate converts through a composeMatrix on ``matrixIn[0]`` and raises
        on ``matrixIn``."""
        t  = Node(cmds.createNode("transform", name="t"))
        mm = Node(cmds.createNode("multMatrix", name="mm"))
        with self.assertRaises(rig.InjectionError):
            mm.matrixIn << t.t
        mm2 = Node(cmds.createNode("multMatrix", name="mm2"))
        mm2.matrixIn[0] << t.t
        self.assertEqual(cmds.nodeType(cmds.listConnections("mm2.matrixIn[0]", source=True)[0]),
                         "composeMatrix")
