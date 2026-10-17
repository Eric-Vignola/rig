"""Round 5, step A4: the inject bundle.

* INJ-3: ``_inject_value`` reads the data type of a dst whose attribute kind
  fixes it (a numeric, unit, enum, message or compound kind with every array
  element on its path existing) as the private ``_FIXED_KIND``, which equals
  no type name; the generic-slot test, the one comparison that tells a raised
  (None) type apart, queries it there. An element that does not exist yet has
  no kind, so its query, which creates it, still runs.
* H4: ``Plug.__lshift__`` makes one ``shorthand._Facts`` per operand of a plug
  source and hands both to ``shorthand`` and ``_inject_value``, which reuse
  their compound-ness and data type while their queries ran no code a user may
  have written (``_Facts._reusable``, from ``_base._USER_CODE_RUNS``: a
  fallback hook rig does not vouch for, a node constructor); the element an
  auto-index reaches gets its own facts.
* H2: ``Attribute.data_type`` answers a static hit of a plug whose naming can
  raise only through its owner's ``ensure_valid`` and its attribute-handle
  check (``_dg_named_owner``) after those two checks, without the name.
* H7: ``_set_or_connect``'s ``"." in str(src)`` runs those two checks for such
  a plug.
* ``_disconnect_incoming`` returns for a plug that is no destination, after the
  owner check, without listing its connections.

Tests:

* ``TestTheOutcomesOf619f29f``: every case of ``_CASES`` (sets, connects,
  rewires, fan-outs, auto-indexes, generic slots and the choice output, the
  shorthand conversions, strings, matrices, size mismatches, locks, messages,
  aliases, unitConversion-fed plugs, dynamic and deleted dynamic attributes,
  deleted, undone, renamed and freed nodes on either side, plugs built from
  names, user fallback hooks (a class's, at the end of a choice chain, on a
  node, a class-level patch) and the hooks' call counts) gives what 619f29f
  gave (``r5_inject_oracle_619f29f.json`` beside this file, written by
  ``runs\\r5\\A4\\gen_oracle.py`` running ``_inject_outcomes`` on the snapshot
  of 619f29f): the result or the error (type, message, ``__context__`` /
  ``__cause__``), what it printed, the nodes, connections, values and multi
  indices it changed, the selection and the undo steps.
* ``TestTheSentinel``: the sentinel never leaves ``_inject_value`` (no other
  frame ever holds it) and is compared only with "string" and "matrix"; a set
  or a connect into a fixed kind queries no data type; the phantom element of
  a failed ``pma.input3D[7] << [1, 2]`` is still made; a class with its own
  ``data_type`` is asked.
* ``TestOneSetOfFacts``: a generic slot's or the choice output's data type is
  queried once per ``<<`` (619f29f: two or three times); facts that ran a user's
  code are not reused, so the user's hook runs as often as before.
* ``TestDgNamedOwner``: the owner test (DG classes only; a DAG node, a plug
  with no owner, a component plug, a class with its own name or a patched
  method, a node with its own ``ensure_valid`` fail it, read at every call);
  ``data_type`` and ``_named_with_dot`` raise as the name does, for a deleted
  node, a freed one and a freed dynamic attribute, and build no name for a
  valid one; a deleted source raises before the destination is disconnected.
* ``TestDisconnectIncoming``: over the plug shapes (elements, compound
  parents and children, aliases, array roots, unitConversion-fed plugs, a
  deleted node's and attribute's plugs), ``isDestination`` is False exactly
  when ``connectedTo(True, False)`` is empty, so the fast path lists no
  connection for exactly those plugs; a class with its own connection reads
  takes the old path.
"""

import contextlib
import io
import json
import logging
import os
import sys
import warnings
from collections import Counter
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import List, Node, Plug, get_options, lock, set_options
from rig._internal import plug as plugmod
from rig._internal import shorthand as shorthand_module
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute, set_custom_type
from rig.nodetypes.choice import Choice
from rig.nodetypes.dg_node import DGNode
from rig._tests._base import MayaTestCase
from rig._tests._undo import scene_connections


# --------------------------------------------------------------------------- #
#  The outcomes table (`_inject_outcomes` runs on 619f29f too: public API only)
# --------------------------------------------------------------------------- #


def _brief(error):
    return None if error is None else [type(error).__name__, str(error)]


class _Output(logging.Handler):
    """What a call prints: Maya's command output, the records of the ``rig``
    loggers (every level), Python warnings, and ``sys.stdout`` / ``sys.stderr``."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.maya    = []
        self.records = []

    def emit(self, record):
        self.records.append([record.name, record.levelname, record.getMessage()])

    def __enter__(self):
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(
            lambda message, kind, *args: self.maya.append([int(kind), message])
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
            "warnings": [[w.category.__name__, str(w.message)] for w in self.warned],
            "stdout":   self._out.getvalue(),
            "stderr":   self._err.getvalue(),
        }


# the attributes whose values and multi indices a case's state lists
_READ = {
    "mdA":   ("input1", "input2", "operation", "output"),
    "pmaD":  ("input1D", "input3D", "output3D"),
    "chE":   ("input", "selector"),
    "chJ":   ("input", "selector"),
    "xfC":   ("translate", "rotate", "visibility", "notes_s", "dyn_d", "dyn_v"),
    "xfI":   ("translate", "rotate", "visibility"),
    "cmF":   ("outputMatrix", "inputTranslate"),
    "netK":  ("dynm",),
    "mmM":   ("matrixIn",),
}


def _values():
    """{plug: its value (str), or its multi indices and then each element's}"""
    out = {}
    for node, attrs in sorted(_READ.items()):
        if not cmds.objExists(node):
            continue
        for attr in attrs:
            name = f"{node}.{attr}"
            if not cmds.attributeQuery(attr, node=node, exists=True):
                continue
            try:
                ids = cmds.getAttr(name, multiIndices=True)
            except Exception:  # noqa: BLE001
                ids = None
            if ids:
                out[name] = ["indices", list(ids)]
                for i in ids:
                    try:
                        out[f"{name}[{i}]"] = str(cmds.getAttr(f"{name}[{i}]"))
                    except Exception as error:  # noqa: BLE001
                        out[f"{name}[{i}]"] = ["raises", type(error).__name__]
                continue
            try:
                out[name] = str(cmds.getAttr(name))
            except Exception as error:  # noqa: BLE001
                out[name] = ["raises", type(error).__name__]
    return out


def _scene():
    """The nodes (long name, type), the connections and the values (`_values`)."""
    return (
        {(n, cmds.nodeType(n)) for n in cmds.ls(long=True)},
        set(scene_connections()),
        _values(),
    )


def _delta(before, after):
    """What a case changed in the scene: the nodes and connections added and
    removed, and the values that changed (removed ones as None)."""
    nodes, connections, values = after
    return {
        "nodes":       [sorted(map(list, nodes - before[0])), sorted(map(list, before[0] - nodes))],
        "connections": [
            sorted(map(list, connections - before[1])),
            sorted(map(list, before[1] - connections)),
        ],
        "values": {
            name: values.get(name)
            for name in sorted(set(values) | set(before[2]))
            if values.get(name) != before[2].get(name)
        },
    }


def _state(call, hooks):
    """The outcome of ``call()`` and the scene it leaves: the result (its str) or
    the exception (type, message, ``__context__`` / ``__cause__``,
    ``__suppress_context__``), what it printed, the user hooks' calls
    (``hooks``), the nodes, connections, values and multi indices it changed in
    the scene `_build` made, the selection and the undo steps the queue then
    holds (undone to count them)."""
    before = _scene()
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
        "result":    result,
        "output":    output.state(),
        "hooks":     list(hooks),
        "scene":     _delta(before, _scene()),
        "selection": cmds.ls(selection=True, long=True),
    }
    steps = 0
    quiet = cmds.scriptEditorInfo(query=True, suppressInfo=True)
    cmds.scriptEditorInfo(suppressInfo=True)  # no "Undo: " line per step
    try:
        while not cmds.undoInfo(query=True, undoQueueEmpty=True) and steps < 1000:
            cmds.undo()
            steps += 1
    finally:
        cmds.scriptEditorInfo(suppressInfo=quiet)
    state["undo_steps"] = steps
    return state


def _build():
    """The scene every case starts from (fixed names), its nodes by key."""
    for plugin in ("matrixNodes", "quatNodes"):
        cmds.loadPlugin(plugin, quiet=True)
    N = {
        "md":  Node(cmds.createNode("multiplyDivide", name="mdA")),
        "md2": Node(cmds.createNode("multiplyDivide", name="mdB")),
        "xf":  Node(cmds.createNode("transform", name="xfC")),
        "pma": Node(cmds.createNode("plusMinusAverage", name="pmaD")),
        "ch":  Node(cmds.createNode("choice", name="chE")),
        "cm":  Node(cmds.createNode("composeMatrix", name="cmF")),
        "te":  Node(cmds.createNode("timeEditorClip", name="teG")),
        "loc": Node(cmds.spaceLocator(name="locH")[0]),
        "xf2": Node(cmds.createNode("transform", name="xfI")),
        "ch2": Node(cmds.createNode("choice", name="chJ")),
        "net": Node(cmds.createNode("network", name="netK")),
        "qp":  Node(cmds.createNode("quatProd", name="qpL")),
        "mm":  Node(cmds.createNode("multMatrix", name="mmM")),
    }
    cmds.setAttr("mdB.input1", 2, 3, 4)
    cmds.setAttr("xfC.translate", 1, 2, 3)
    cmds.setAttr("locH.translate", 4, 5, 6)
    cmds.addAttr("xfC", longName="notes_s", dataType="string")
    cmds.addAttr("xfC", longName="notes_t", dataType="string")
    cmds.setAttr("xfC.notes_t", "from t", type="string")
    cmds.addAttr("xfC", longName="dyn_d", attributeType="double")
    cmds.addAttr("xfC", longName="dyn_v", attributeType="double3")
    for axis in "XYZ":
        cmds.addAttr("xfC", longName="dyn_v" + axis, attributeType="double", parent="dyn_v")
    cmds.addAttr("netK", longName="msgIn", attributeType="message")
    cmds.addAttr("netK", longName="dynm", attributeType="double")
    return N


def _te_child(N):
    """A child of ``teG.layer[0]``, an element that does not exist (its getAttr
    -type raises: INJ-4's edge)."""
    kids = cmds.attributeQuery("layer", node="teG", listChildren=True) or []
    element = N["te"].layer[0]
    str(element)
    return getattr(element, kids[0])


def _pre_delete(N, name="mdA"):
    """The plugs read first on the live node, then the node deleted (undo queue)."""
    str(N["md"].message)
    str(N["ch"].message)
    cmds.delete(name)


def _freed(N, key, attr):
    """A plug of ``N[key]`` held across a new scene (its node freed), and the scene
    built again (new nodes of the same names)."""
    held = getattr(N[key], attr)
    str(held)
    cmds.file(new=True, force=True)
    return held, _build()


def _held(plug):
    """`plug`, named once (its str and its kind read while its node is alive)."""
    str(plug)
    return plug


def _rename_then(N, op):
    cmds.rename("mdB", "mdB2")
    return op()


def _undone_then(N, op):
    cmds.delete("mdB")
    cmds.undo()
    return op()


# user code in a type query (the hooks append the attrs they are asked about)
_HOOK_CALLS = []
_USER_CLASSES = {}


def _user_classes():
    """A Choice subclass with its own hook, and a DG class with one, each for a
    custom type (made once per process: a node class registers its type)."""
    if not _USER_CLASSES:
        class _CountingChoice(Choice):
            CUSTOM_NODE_TYPE = "r5injectCountingChoice"

            def _attr_data_type_fallback(self, attr):
                _HOOK_CALLS.append(["choice", str(attr)])
                return super()._attr_data_type_fallback(attr)

        class _CountingNet(DGNode):
            CUSTOM_NODE_TYPE = "r5injectCountingNet"

            def _attr_data_type_fallback(self, attr):
                _HOOK_CALLS.append(["net", str(attr)])
                return super()._attr_data_type_fallback(attr)

        _USER_CLASSES["choice"] = _CountingChoice
        _USER_CLASSES["net"]    = _CountingNet
    return _USER_CLASSES


def _user_choice(N):
    _user_classes()
    cmds.createNode("choice", name="uchN")
    set_custom_type("uchN", "r5injectCountingChoice")
    return Node("uchN")


def _user_net(N):
    _user_classes()
    cmds.createNode("network", name="unetO")
    cmds.addAttr("unetO", longName="generic", attributeType="typed")
    set_custom_type("unetO", "r5injectCountingNet")
    return Node("unetO")


def _counting(hook, label):
    def counted(self, attr):
        _HOOK_CALLS.append([label, str(attr)])
        return hook(self, attr)
    return counted


def _instance_hook(N):
    node = N["ch"]
    hook = type(node)._attr_data_type_fallback
    node.__dict__["_attr_data_type_fallback"] = lambda attr: (
        _HOOK_CALLS.append(["instance", str(attr)]) or hook(node, attr)
    )
    return node


def _patched_dg_hook(op):
    with mock.patch.object(
        DGNode, "_attr_data_type_fallback",
        _counting(DGNode.__dict__["_attr_data_type_fallback"], "patched DGNode"),
    ):
        return op()


def _patched_choice_hook(op):
    with mock.patch.object(
        Choice, "_attr_data_type_fallback",
        _counting(Choice.__dict__["_attr_data_type_fallback"], "patched Choice"),
    ):
        return op()


def _shorthand_off(op):
    saved = get_options()["use_shorthand"]
    set_options(use_shorthand=False)
    try:
        return op()
    finally:
        set_options(use_shorthand=saved)


def _instanced(N):
    shape = cmds.createNode("locator", name="ipShape", parent="xfI")
    cmds.parent(shape, "xfC", shape=True, addObject=True)
    return Node("xfC|ipShape")


def _namespaced(N):
    cmds.namespace(add="ns")
    cmds.createNode("multiplyDivide", name="ns:mdP")
    return Node("ns:mdP")


def _deleted_attr(N, flush=False):
    held = N["xf"].dyn_d
    str(held)
    cmds.deleteAttr("xfC.dyn_d")
    if flush:
        cmds.flushUndo()
    return held


_CASES = [
    # the 37 cases of runs\r45\api2_inject\probe_k.py (INJ-3 on K)
    ("scalar set", lambda N: N["md"].input1X << 2.5),
    ("scalar connect", lambda N: N["md"].input1X << N["md2"].outputX),
    ("compound set", lambda N: N["md"].input1 << [1, 2, 3]),
    ("compound connect", lambda N: N["md"].input1 << N["md2"].output),
    ("compound broadcast", lambda N: N["md"].input1 << 4.0),
    ("compound -> scalar fanout", lambda N: N["md"].input1X << N["xf"].t),
    ("scalar -> compound fanout", lambda N: N["md"].input1 << N["xf"].tx),
    ("enum set", lambda N: N["md"].operation << 2),
    ("bool set", lambda N: N["xf"].visibility << 0),
    ("multi auto-index scalar", lambda N: N["pma"].input1D << 1.5),
    ("multi auto-index compound", lambda N: N["pma"].input3D << N["xf"].t),
    ("existing element set", lambda N: (N["pma"].input1D[0] << 1.0, N["pma"].input1D[0] << 7.0)),
    ("new element set", lambda N: N["pma"].input1D[9] << 3.0),
    ("multi << list", lambda N: N["pma"].input1D << [1, 2, 3]),
    ("multi << List", lambda N: N["pma"].input1D << List([N["xf"].tx, N["xf"].ty])),
    ("generic slot << compound", lambda N: N["ch"].input[0] << N["xf"].t),
    ("generic slot << scalar", lambda N: N["ch"].input[1] << N["xf"].tx),
    ("choice.selector << plug", lambda N: N["ch"].selector << N["md2"].outputX),
    ("string attr set", lambda N: N["xf"].notes_s << "hello"),
    ("string attr connect", lambda N: N["xf"].notes_s << N["xf"].notes_t),
    ("matrix set", lambda N: N["cm"].outputMatrix << list(range(16))),
    ("seq size mismatch", lambda N: N["md"].input1 << [1, 2]),
    ("locked dst", lambda N: (cmds.setAttr("mdA.input2X", lock=True), N["md"].input2X << 1.0)),
    ("locked child of compound",
     lambda N: (cmds.setAttr("mdA.input2Y", lock=True), N["md"].input2 << [1, 2, 3])),
    ("message query", lambda N: N["md"].message >> None),
    ("disconnect", lambda N: (N["md"].input1X << N["md2"].outputX, N["md"].input1X << None)),
    ("timeEditor layer[0] child", lambda N: _te_child(N) << 0.5),
    ("timeEditor layer[0] child << compound", lambda N: _te_child(N) << N["xf"].t),
    ("dynamic double", lambda N: N["xf"].dyn_d << 0.25),
    ("dynamic double3 compound", lambda N: N["xf"].dyn_v << [1, 2, 3]),
    ("dynamic double3 << plug", lambda N: N["xf"].dyn_v << N["xf"].t),
    ("deleted dst scalar set", lambda N: (_pre_delete(N), N["md"].input1X << 1.0)),
    ("deleted dst connect", lambda N: (_pre_delete(N), N["md"].input1X << N["md2"].outputX)),
    ("deleted dst compound set", lambda N: (_pre_delete(N), N["md"].input1 << [1, 2, 3])),
    ("deleted dst generic << compound",
     lambda N: (_pre_delete(N, "chE"), N["ch"].input[0] << N["xf"].t)),
    ("new compound element, bad size", lambda N: N["pma"].input3D[7] << [1, 2]),
    ("new compound element set", lambda N: N["pma"].input3D[7] << [1, 2, 3]),
    # round 5 A4: rewires, the fast paths' shapes, generic sources, shorthand
    ("enum name set", lambda N: N["md"].operation << "divide"),
    ("rewire scalar",
     lambda N: (N["md"].input1X << N["md2"].outputX, N["md"].input1X << N["md2"].outputY)),
    ("rewire compound",
     lambda N: (N["md"].input1 << N["md2"].output, N["md"].input1 << N["xf"].t)),
    ("rewire a locked connected dst",
     lambda N: (N["md"].input1X << N["md2"].outputX, cmds.setAttr("mdA.input1X", lock=True),
                N["md"].input1X << N["md2"].outputY)),
    ("disconnect a locked dst",
     lambda N: (N["md"].input1X << N["md2"].outputX, cmds.setAttr("mdA.input1X", lock=True),
                N["md"].input1X << None)),
    ("connect into a child of a connected compound",
     lambda N: (N["md"].input1 << N["md2"].output, N["md"].input1X << N["xf"].tx)),
    ("connect into a compound with a connected child",
     lambda N: (N["md"].input1X << N["xf"].tx, N["md"].input1 << N["md2"].output)),
    ("none into a compound with a connected child",
     lambda N: (N["md"].input1X << N["xf"].tx, N["md"].input1 << None)),
    ("unitConversion connect", lambda N: N["xf2"].rx << N["md2"].outputX),
    ("unitConversion rewire",
     lambda N: (N["xf2"].rx << N["md2"].outputX, N["xf2"].rx << N["md2"].outputY)),
    ("unitConversion compound", lambda N: N["xf2"].r << N["md2"].output),
    ("unitConversion disconnect",
     lambda N: (N["xf2"].rx << N["md2"].outputX, N["xf2"].rx << None)),
    ("alias connect",
     lambda N: (cmds.aliasAttr("myAlias", "mdA.input2Z"), Node("mdA").myAlias << N["md2"].outputZ)),
    ("alias rewire",
     lambda N: (cmds.aliasAttr("myAlias", "mdA.input2Z"), Node("mdA").myAlias << N["md2"].outputZ,
                Node("mdA").input2Z << N["md2"].outputX)),
    ("message connect", lambda N: N["net"].msgIn << N["md"].message),
    ("message rewire",
     lambda N: (N["net"].msgIn << N["md"].message, N["net"].msgIn << N["md2"].message)),
    ("dynamic connect", lambda N: N["net"].dynm << N["md2"].outputX),
    ("dynamic as a source", lambda N: N["md"].input1X << N["net"].dynm),
    ("element connect", lambda N: N["pma"].input1D[2] << N["md2"].outputX),
    ("element rewire",
     lambda N: (N["pma"].input1D[2] << N["md2"].outputX, N["pma"].input1D[2] << N["md2"].outputY)),
    ("compound element connect", lambda N: N["pma"].input3D[1] << N["md2"].output),
    ("child of a missing element set", lambda N: N["pma"].input3D[4].input3Dy << 2.0),
    ("child of a missing element connect", lambda N: N["pma"].input3D[4].input3Dy << N["md2"].outputY),
    ("auto-index generic << compound", lambda N: N["ch"].input << N["xf"].t),
    ("auto-index generic << scalar", lambda N: N["ch"].input << N["md2"].outputX),
    ("auto-index generic twice",
     lambda N: (N["ch"].input << N["xf"].t, N["ch"].input << N["md2"].output)),
    ("auto-index generic << number", lambda N: N["ch"].input << 3.0),
    ("auto-index compound twice",
     lambda N: (N["pma"].input3D << N["xf"].t, N["pma"].input3D << N["md2"].output)),
    ("generic slot rewire",
     lambda N: (N["ch"].input[0] << N["xf"].t, N["ch"].input[0] << N["md2"].output)),
    ("generic slot << number", lambda N: N["ch"].input[2] << 1.5),
    ("generic slot << None", lambda N: (N["ch"].input[0] << N["xf"].t, N["ch"].input[0] << None)),
    ("choice output -> compound",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["md"].input1 << N["ch"].output)),
    ("choice output -> scalar",
     lambda N: (N["ch"].input[0] << N["md2"].outputX, N["md"].input1X << N["ch"].output)),
    ("choice output (compound) -> scalar",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["md"].input1X << N["ch"].output)),
    ("choice output -> translate",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["xf2"].t << N["ch"].output)),
    ("choice output -> generic slot",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["ch2"].input[0] << N["ch"].output)),
    ("choice chain",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["ch2"].input[0] << N["ch"].output,
                N["md"].input1 << N["ch2"].output)),
    ("choice output with nothing selected", lambda N: N["md"].input1 << N["ch"].output),
    ("choice output, selector set",
     lambda N: (N["ch"].input[0] << N["md2"].output, N["ch"].input[1] << N["xf"].t,
                N["ch"].selector << 1, N["md"].input1 << N["ch"].output)),
    ("choice output, a matrix selected",
     lambda N: (N["ch"].input[0] << N["loc"].worldMatrix[0], N["xf2"].t << N["ch"].output)),
    ("matrix -> translate", lambda N: N["xf2"].t << N["loc"].worldMatrix[0]),
    ("matrix -> transform node", lambda N: N["xf2"] << N["loc"].worldMatrix[0]),
    ("matrix -> a transform plug no channel", lambda N: N["xf2"].visibility << N["loc"].worldMatrix[0]),
    ("matrix -> rotate", lambda N: N["xf2"].r << N["loc"].worldMatrix[0]),
    ("matrix -> vector", lambda N: N["md"].input1 << N["loc"].worldMatrix[0]),
    ("matrix -> matrix element", lambda N: N["mm"].matrixIn[0] << N["loc"].worldMatrix[0]),
    ("matrix auto-index", lambda N: N["mm"].matrixIn << N["loc"].worldMatrix[0]),
    ("quaternion -> euler", lambda N: N["xf2"].r << N["qp"].outputQuat),
    ("quaternion -> vector", lambda N: N["md"].input1 << N["qp"].outputQuat),
    ("vector -> matrix element", lambda N: N["mm"].matrixIn[1] << N["xf"].t),
    ("rotate -> matrix element", lambda N: N["mm"].matrixIn[1] << N["xf"].r),
    ("vector -> quaternion", lambda N: N["qp"].input1Quat << N["xf"].t),
    ("matrix into a scalar", lambda N: N["md"].input1X << N["loc"].worldMatrix[0]),
    ("scalar into a matrix element", lambda N: N["mm"].matrixIn[0] << N["md2"].outputX),
    ("world matrix root as a source", lambda N: N["mm"].matrixIn[0] << N["loc"].worldMatrix),
    ("List into a compound", lambda N: N["md"].input1 << List([N["xf"].tx, N["xf"].ty, N["xf"].tz])),
    ("plugs and numbers into a compound", lambda N: N["md"].input1 << [N["xf"].tx, 2, N["xf"].tz]),
    ("spec slots into a compound", lambda N: N["md"].input1 << [lock, 2.0, 3.0]),
    ("src is dst", lambda N: N["md"].input1X << N["md"].input1X),
    ("instanced shape connect", lambda N: _instanced(N).localPositionX << N["md2"].outputX),
    ("instanced shape rewire",
     lambda N: (_instanced(N).localPositionX << N["md2"].outputX,
                Node("xfI|ipShape").localPositionX << N["md2"].outputY)),
    ("namespaced node connect", lambda N: _namespaced(N).input1X << N["md2"].outputX),
    ("namespaced node as a source", lambda N: N["md"].input1X << _namespaced(N).outputX),
    ("name-built source", lambda N: N["md"].input1X << Plug("mdB.outputX")),
    ("name-built dst connect", lambda N: Plug("mdA.input1X") << N["md2"].outputX),
    ("name-built dst set", lambda N: Plug("mdA.input1X") << 3.0),
    ("name-built generic dst", lambda N: Plug("chE.input[3]") << N["xf"].t),
    ("renamed source", lambda N: _rename_then(N, lambda: N["md"].input1X << N["md2"].outputX)),
    ("undone source", lambda N: _undone_then(N, lambda: N["md"].input1X << N["md2"].outputX)),
    ("deleted source, the dst connected",
     lambda N: (N["md"].input1X << N["xf"].tx, cmds.delete("mdB"),
                N["md"].input1X << N["md2"].outputX)),
    ("deleted compound source",
     lambda N: (N["md"].input1 << N["xf"].t, cmds.delete("mdB"), N["md"].input1 << N["md2"].output)),
    ("deleted source into a generic slot",
     lambda N: (cmds.delete("mdB"), N["ch"].input[0] << N["md2"].output)),
    ("deleted DAG source",
     lambda N: (N["md"].input1X << N["md2"].outputX, cmds.delete("xfC"),
                N["md"].input1X << N["xf"].tx)),
    ("deleted DAG dst", lambda N: (cmds.delete("xfC"), N["xf"].tx << 1.0)),
    ("held source deleted, the dst connected",
     lambda N: (lambda src: (N["md"].input1X << N["xf"].tx, cmds.delete("mdB"),
                             N["md"].input1X << src))(_held(N["md2"].outputX))),
    ("held compound source deleted",
     lambda N: (lambda src: (N["md"].input1 << N["xf"].t, cmds.delete("mdB"),
                             N["md"].input1 << src))(_held(N["md2"].output))),
    ("held source deleted, into a generic slot",
     lambda N: (lambda src: (cmds.delete("mdB"), N["ch"].input[0] << src))(_held(N["md2"].output))),
    ("held DAG source deleted",
     lambda N: (lambda src: (N["md"].input1X << N["md2"].outputX, cmds.delete("xfC"),
                             N["md"].input1X << src))(_held(N["xf"].tx))),
    ("held source deleted, undone",
     lambda N: (lambda src: (cmds.delete("mdB"), cmds.undo(), N["md"].input1X << src))(
         _held(N["md2"].outputX))),
    ("deleted dynamic source", lambda N: N["md"].input1X << _deleted_attr(N)),
    ("deleted dynamic dst set", lambda N: _deleted_attr(N) << 1.0),
    ("deleted dynamic dst connect", lambda N: _deleted_attr(N) << N["md2"].outputX),
    ("freed dynamic source", lambda N: N["md"].input1X << _deleted_attr(N, flush=True)),
    ("freed dynamic dst", lambda N: _deleted_attr(N, flush=True) << N["md2"].outputX),
    ("freed source",
     lambda N: (lambda held, M: M["md"].input1X << held)(*_freed(N, "md2", "outputX"))),
    ("freed dst",
     lambda N: (lambda held, M: held << M["md2"].outputX)(*_freed(N, "md", "input1X"))),
    ("freed DAG source",
     lambda N: (lambda held, M: M["md"].input1X << held)(*_freed(N, "xf", "tx"))),
    ("shorthand off: compound connect", lambda N: _shorthand_off(lambda: N["md"].input1 << N["md2"].output)),
    ("shorthand off: matrix -> vector",
     lambda N: _shorthand_off(lambda: N["md"].input1 << N["loc"].worldMatrix[0])),
    ("shorthand off: generic slot", lambda N: _shorthand_off(lambda: N["ch"].input << N["xf"].t)),
    # user code in a type query: its calls are listed (`hooks`)
    ("user choice hook: generic slot", lambda N: _user_choice(N).input[0] << N["xf"].t),
    ("user choice hook: auto-index", lambda N: _user_choice(N).input << N["xf"].t),
    ("user choice hook: output",
     lambda N: (lambda u: (u.input[0] << N["md2"].output, N["md"].input1 << u.output))(_user_choice(N))),
    ("user hook at a choice chain's end",
     lambda N: (lambda u: (cmds.connectAttr("unetO.generic", "chE.input[0]"),
                           N["md"].input1 << N["ch"].output))(_user_net(N))),
    ("instance hook: generic slot", lambda N: _instance_hook(N).input[0] << N["xf"].t),
    ("instance hook: output",
     lambda N: (lambda c: (c.input[0] << N["md2"].output, N["md"].input1 << c.output))(_instance_hook(N))),
    ("patched DGNode hook",
     lambda N: _patched_dg_hook(lambda: (N["ch"].input[0] << N["xf"].t, N["md"].input1 << N["ch"].output))),
    ("patched Choice hook",
     lambda N: _patched_choice_hook(lambda: (N["ch"].input << N["xf"].t, N["md"].input1 << N["ch"].output))),
    # data_type (H2) on the plugs of a DG node, live, deleted, freed
    ("data_type live", lambda N: N["md"].input1X.get_data_type()),
    ("data_type of a deleted node", lambda N: (lambda p: (cmds.delete("mdA"), p.get_data_type()))(N["md"].input1X)),
    ("data_type of a freed node",
     lambda N: (lambda held, M: held.get_data_type())(*_freed(N, "md", "input1X"))),
    ("data_type of a freed dynamic attr", lambda N: _deleted_attr(N, flush=True).get_data_type()),
    ("data_type of a deleted dynamic attr", lambda N: _deleted_attr(N).get_data_type()),
]


def _inject_outcomes():
    """{case name: its state (`_state`)}, each case in a new scene, the undo queue
    on and infinite (restored after)."""
    saved = (
        cmds.undoInfo(query=True, state=True),
        cmds.undoInfo(query=True, infinity=True),
        cmds.undoInfo(query=True, length=True),
    )
    cmds.undoInfo(state=True, infinity=True)
    table = {}
    try:
        for name, case in _CASES:
            cmds.file(new=True, force=True)
            N = _build()
            del _HOOK_CALLS[:]
            table[name] = _state(lambda: case(N), _HOOK_CALLS)
    finally:
        cmds.file(new=True, force=True)
        if not saved[1]:
            cmds.undoInfo(infinity=False, length=saved[2])
        cmds.undoInfo(state=saved[0])
    return table


# --------------------------------------------------------------------------- #
#  Counting calls (sys.setprofile on the functions' code objects: nothing is
#  patched, so every fast path's class tests still pass)
# --------------------------------------------------------------------------- #

# 619f29f's outcomes of `_CASES` (runs\r5\A4\gen_oracle.py on its snapshot)
_ORACLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "r5_inject_oracle_619f29f.json"
)

_DATA_TYPE_CODE = Attribute.get_data_type.__code__
_FULL_NAME_CODE = Attribute.get_full_name.__code__
# re-pinned (user decision 2026-10-16, name clashes): the class dict holds a
# plug method; class access is the function
_LISTING_CODE   = Attribute.get_connected_attrs.__code__
_CHOICE_HOOK    = Choice.__dict__["_attr_data_type_fallback"].__code__


class _Calls:
    """While active, the calls of the functions whose code objects ``codes``
    names ({code: label}) by label (``counts``), and by the str buffer of their
    first argument (``by_arg[label]``)."""

    def __init__(self, codes):
        self.codes  = codes
        self.counts = Counter()
        self.by_arg = {label: Counter() for label in codes.values()}

    def _profile(self, frame, event, arg):
        if event == "call":
            label = self.codes.get(frame.f_code)
            if label is not None:
                self.counts[label] += 1
                code  = frame.f_code
                value = frame.f_locals.get(code.co_varnames[0]) if code.co_argcount else None
                if isinstance(value, str):
                    self.by_arg[label][str.__str__(value)] += 1

    def __enter__(self):
        self._saved = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._saved)
        return False


def _type_queries(call):
    """The ``cmds.getAttr(..., type=True)`` calls ``call()`` makes."""
    with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
        call()
    return [c for c in probe.call_args_list if c.kwargs.get("type")]


def _raised(call):
    """(type name, message, the context's [type name, message] or None) of what
    ``call()`` raises, or None."""
    try:
        call()
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return (type(error).__name__, str(error), _brief(error.__context__))
    return None


# Re-pinned (B1: 2 fewer undo steps per GC tag, approved lever): the cases whose op
# makes one GC-tagged node through ``container.createNode`` (a decomposeMatrix, a
# composeMatrix, a quatToEuler, a constant's network) hold 2 undo steps fewer than on
# 619f29f: the tag is written through the API, with no addAttr / setAttr step.
_B1_FEWER_UNDO_STEPS = dict.fromkeys(
    (
        "choice output, a matrix selected",
        "matrix -> a transform plug no channel",
        "matrix -> rotate",
        "matrix -> transform node",
        "matrix -> translate",
        "matrix -> vector",
        "quaternion -> euler",
        "quaternion -> vector",
        "rotate -> matrix element",
        "vector -> matrix element",
    ),
    2,
)


# Re-pinned (D2: connects through the API, journaled; approved differences, user 2026-10-02):
# * one undo step per statement for its API edits: these cases' ops hold fewer undo steps (their
#   connects join one journal, on top of B1's GC tags);
# * one extra Maya ``Error:`` line on an incompatible-type connect: the API connect's ``doIt``
#   prints it before the cmds route runs and raises (the rest of the output is unchanged).
_D2_FEWER_UNDO_STEPS = {
    "List into a compound": 2,
    "choice output with nothing selected": 2,
    "choice output, a matrix selected": 2,
    "matrix -> a transform plug no channel": 1,
    "matrix -> rotate": 2,
    "matrix -> transform node": 5,
    "matrix -> translate": 2,
    "matrix -> vector": 1,
    "multi << List": 1,
    "quaternion -> euler": 2,
    "quaternion -> vector": 3,
    "rotate -> matrix element": 2,
    "scalar -> compound fanout": 2,
    "user hook at a choice chain's end": 2,
    "vector -> matrix element": 1,
}
# Re-pinned (D3: single numeric value sets through the API, journaled; the same approved
# difference): these cases' value sets join their statement's journal (the steps on top of
# B1's and D2's).
_D3_FEWER_UNDO_STEPS = {
    "compound broadcast": 2,
    "compound set": 2,
    "dynamic double3 compound": 2,
    "plugs and numbers into a compound": 2,
    "quaternion -> vector": 3,
    "spec slots into a compound": 1,
}
# Re-pinned (review fix, features run): a statement that removes a plug's input and then
# writes it is one undo step (the disconnect and the write share the statement's chunk,
# `journal.one_step`): these rewires make one step fewer.
_FIX_FEWER_UNDO_STEPS = {
    "alias rewire":           1,
    "element rewire":         1,
    "generic slot rewire":    1,
    "instanced shape rewire": 1,
    "message rewire":         1,
    "rewire compound":        1,
    "rewire scalar":          1,
    "unitConversion rewire":  1,
}
# Re-pinned (review fix, features run): a connect into a child of a compound connected whole
# splits the compound's input into one per child first, then connects the child (619f29f left
# both, ``mdB.output -> mdA.input1`` and ``xfC.translateX -> mdA.input1X``: two drivers on one
# channel); the INFO line names the split. The values and the undo steps are unchanged.
_FIX_SPLIT = {
    "connect into a child of a connected compound": (
        [[["mdB.outputY", "mdA.input1Y"], ["mdB.outputZ", "mdA.input1Z"],
          ["xfC.translateX", "mdA.input1X"]], []],
        [["rig._internal.plug", "INFO",
          "connect mdA.input1X: splitting the input mdB.output of mdA.input1 into one per child"]],
    ),
}

# Re-pinned (one driver per channel, user decision 2026-10-16): a plug or None into a compound
# also removes the inputs its children have of their own, logged at INFO, in the statement's
# one undo step. 619f29f left ``xfC.translateX -> mdA.input1X`` under the compound's new wire
# (two drivers on one channel: the viewport read the leaf, getAttr the compound), and
# ``mdA.input1 << None`` changed nothing. Each case -> (connections, values, logs, undo steps
# more): the None case is one step more (its disconnect; 619f29f made no edit), and the child's
# value is no longer held by the wire it read.
_OD_ONE_DRIVER = {
    "connect into a compound with a connected child": (
        [[["mdB.output", "mdA.input1"]], []],
        None,
        [["rig._internal.plug", "INFO",
          "connect mdA.input1: removing the input xfC.translateX of its child mdA.input1X "
          "(one driver per channel)"]],
        0,
    ),
    "none into a compound with a connected child": (
        [[], []],
        {},
        [["rig._internal.plug", "INFO",
          "disconnect mdA.input1: removing the input xfC.translateX of its child mdA.input1X "
          "(one driver per channel)"]],
        1,
    ),
}

# Re-pinned (features step 3, lock): the locked-plug refusal names the unlock rig spells,
# ``Plug('<plug>').unlock()``, and no cmds call (the text 619f29f gave ended "Unlock first (e.g.
# `plug << unlock` or `cmds.setAttr(attr, lock=False)`)."); the error, the scene, the output
# and the undo steps are unchanged. The case -> the locked plug the text names.
_FS3_LOCK_HINT_OLD = "Unlock first (e.g. `plug << unlock` or `cmds.setAttr(attr, lock=False)`)."
_FS3_LOCK_HINT = {
    "locked child of compound":      "mdA.input2Y",
    "locked dst":                    "mdA.input2X",
    "rewire a locked connected dst": "mdA.input1X",
}


def _fs3_lock_hint(plug):
    return (
        f"Unlock first: Plug({plug!r}).unlock() (a locked plug takes no value or "
        f"connection, even the one it holds)."
    )


# Re-pinned (features step 4, Typed: the generic-slot rule, the user's default "every kAny
# attribute, choice included"): a choice.input element (a typed attribute that holds any data,
# as a Typed one) never reads its data type as a destination, so these rows were regenerated
# on that step's tree (`_FS4_GENERIC_PATH`, beside this file); every other row is 619f29f's:
# * "generic slot << number", "auto-index generic << number": the number is set (619f29f: the
#   setAttr's type flag 'Tdata' was refused), one setAttr step more;
# * "choice output (compound) -> scalar": a generic source holding compound data is connected
#   whole (619f29f: 'Plug is not a compound parent' from the per-channel split), one step more;
# * "held source deleted, into a generic slot": the source's "already deleted" is raised
#   before the destination is queried, so no element is made (619f29f: chE.input[0] made);
# * the choice-hook rows ("instance hook: ...", "patched ... hook", "user choice hook: ..."):
#   a slot as the destination asks no fallback hook (619f29f: 1 to 3 calls), the result, the
#   scene and the undo steps unchanged.
_FS4_GENERIC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "r5_inject_fs4_generic_slots.json"
)
_FS4_GENERIC_ROWS = frozenset((
    "auto-index generic << number",
    "choice output (compound) -> scalar",
    "generic slot << number",
    "held source deleted, into a generic slot",
    "instance hook: generic slot",
    "instance hook: output",
    "patched Choice hook",
    "patched DGNode hook",
    "user choice hook: auto-index",
    "user choice hook: generic slot",
    "user choice hook: output",
))


_D2_ERROR_LINES = {
    "matrix -> a transform plug no channel": (
        "Connection not made: 'locH.worldMatrix[0]' -> 'xfI.visibility'."
        "  Data types of source and destination are not compatible."
    ),
    "matrix into a scalar": (
        "Connection not made: 'locH.worldMatrix[0]' -> 'mdA.input1X'."
        "  Data types of source and destination are not compatible."
    ),
    "scalar into a matrix element": (
        "Connection not made: 'mdB.outputX' -> 'mmM.matrixIn[0]'."
        "  Data types of source and destination are not compatible."
    ),
    "shorthand off: matrix -> vector": (
        "Connection not made: 'locH.worldMatrix[0]' -> 'mdA.input1X'."
        "  Data types of source and destination are not compatible."
    ),
    "timeEditor layer[0] child << compound": (
        "Connection not made: 'xfC.translate' -> 'teG.layer[0].layerName'."
        "  Data types of source and destination are not compatible."
    ),
    "vector -> quaternion": (
        "Connection not made: 'xfC.translate' -> 'qpL.input1Quat'."
        "  Data types of source and destination are not compatible."
    ),
}


class TestTheOutcomesOf619f29f(MayaTestCase):
    """Every case of `_CASES` gives what 619f29f gave (`_ORACLE_PATH`), but for
    the undo steps of the GC tags (`_B1_FEWER_UNDO_STEPS`), and of the journaled
    connects and the incompatible-type connect's ``Error:`` line (`_D2_FEWER_UNDO_STEPS`,
    `_D2_ERROR_LINES`), and of the journaled value sets (`_D3_FEWER_UNDO_STEPS`), and the
    locked-plug refusal's unlock hint (`_FS3_LOCK_HINT`), and the generic-slot rows
    regenerated for Typed (`_FS4_GENERIC_ROWS`), and the review fix's one-step rewires
    (`_FIX_FEWER_UNDO_STEPS`) and its split of a whole connection (`_FIX_SPLIT`), and one
    driver per channel on a compound (`_OD_ONE_DRIVER`)."""

    def test_every_case_as_619f29f(self):
        with open(_ORACLE_PATH) as fh:
            oracle = json.load(fh)
        with open(_FS4_GENERIC_PATH) as fh:
            regenerated = json.load(fh)
        self.assertEqual(set(regenerated), _FS4_GENERIC_ROWS)
        self.assertLessEqual(_FS4_GENERIC_ROWS, set(oracle))
        for name in _FS4_GENERIC_ROWS:
            # each row differs from 619f29f's
            self.assertNotEqual(regenerated[name], oracle[name], name)
        oracle.update(regenerated)
        table = json.loads(json.dumps(_inject_outcomes()))
        self.assertEqual(sorted(table), sorted(oracle))
        self.assertLessEqual(set(_B1_FEWER_UNDO_STEPS), set(oracle))
        self.assertLessEqual(set(_D2_FEWER_UNDO_STEPS) | set(_D2_ERROR_LINES), set(oracle))
        self.assertLessEqual(set(_D3_FEWER_UNDO_STEPS), set(oracle))
        self.assertLessEqual(set(_FS3_LOCK_HINT), set(oracle))
        self.assertLessEqual(set(_FIX_FEWER_UNDO_STEPS) | set(_FIX_SPLIT), set(oracle))
        self.assertLessEqual(set(_OD_ONE_DRIVER), set(oracle))
        for name in sorted(oracle):
            with self.subTest(case=name):
                if name in _FS4_GENERIC_ROWS:
                    self.assertEqual(table[name], oracle[name])
                    continue
                expected = dict(oracle[name])
                expected["undo_steps"] -= _B1_FEWER_UNDO_STEPS.get(name, 0)
                expected["undo_steps"] -= _D2_FEWER_UNDO_STEPS.get(name, 0)
                expected["undo_steps"] -= _D3_FEWER_UNDO_STEPS.get(name, 0)
                expected["undo_steps"] -= _FIX_FEWER_UNDO_STEPS.get(name, 0)
                if name in _FIX_SPLIT:
                    connections, logs = _FIX_SPLIT[name]
                    self.assertEqual(expected["output"]["logs"], [])
                    expected["scene"] = dict(expected["scene"], connections=connections)
                    expected["output"] = dict(expected["output"], logs=logs)
                if name in _OD_ONE_DRIVER:
                    connections, values, logs, more = _OD_ONE_DRIVER[name]
                    self.assertEqual(expected["output"]["logs"], [])
                    # 619f29f: the child's own wire under the compound's (or no) wire
                    self.assertIn(["xfC.translateX", "mdA.input1X"], expected["scene"]["connections"][0])
                    expected["scene"] = dict(expected["scene"], connections=connections)
                    if values is not None:
                        expected["scene"]["values"] = values
                    expected["output"] = dict(expected["output"], logs=logs)
                    expected["undo_steps"] += more
                if name in _D2_ERROR_LINES:
                    self.assertEqual(expected["output"]["maya"], [])
                    expected["output"] = dict(expected["output"], maya=[[4, _D2_ERROR_LINES[name]]])
                if name in _FS3_LOCK_HINT:
                    result = list(expected["result"])
                    self.assertTrue(result[2].endswith(_FS3_LOCK_HINT_OLD), result[2])
                    result[2] = result[2][: -len(_FS3_LOCK_HINT_OLD)] + _fs3_lock_hint(_FS3_LOCK_HINT[name])
                    expected["result"] = result
                self.assertEqual(table[name], expected)

    def test_the_user_code_cases_ran_it(self):
        # the cases that list the hooks' calls ran them, on 619f29f too
        with open(_ORACLE_PATH) as fh:
            oracle = json.load(fh)
        for name, _ in _CASES:
            if "hook" in name:
                with self.subTest(case=name):
                    self.assertTrue(oracle[name]["hooks"])


class TestTheSentinel(MayaTestCase):
    """INJ-3: `_FIXED_KIND` stands in for a fixed kind's data type."""

    TEST_START_NEW_SCENE = True

    def _ops(self, N):
        return [
            lambda: N["md"].input1X << 2.5,
            lambda: N["md"].input1 << [1, 2, 3],
            lambda: N["md"].input1 << 4.0,
            lambda: N["md"].operation << "divide",
            lambda: N["xf"].visibility << 0,
            lambda: N["md"].input1X << N["md2"].outputX,
            lambda: N["md"].input1 << N["md2"].output,
            lambda: N["md"].input1X << N["xf"].t,
            lambda: N["md"].input1 << N["xf"].tx,
            lambda: N["pma"].input1D << 1.5,
            lambda: N["pma"].input3D << N["xf"].t,
            lambda: N["pma"].input1D[0] << 7.0,
            lambda: N["xf"].dyn_v << [1, 2, 3],
            lambda: N["net"].msgIn << N["md"].message,
            lambda: N["md"].input1 << [1, 2],
            lambda: N["pma"].input3D[7] << [1, 2],
            lambda: (cmds.delete("mdB"), N["md2"].input1X << N["xf"].t),
        ]

    def test_the_sentinel_never_leaves_inject_value(self):
        compared = []

        class _Spy:
            def __eq__(self, other):
                compared.append(("==", other))
                return self is other

            def __ne__(self, other):
                compared.append(("!=", other))
                return self is not other

            __hash__ = object.__hash__

        spy     = _Spy()
        allowed = {
            plugmod._inject_value.__code__,
            plugmod._dst_data_type.__code__,
            plugmod._resolved_data_type.__code__,
            _Spy.__eq__.__code__,
            _Spy.__ne__.__code__,
        }
        leaked  = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code not in allowed:
                if any(value is spy for value in frame.f_locals.values()):
                    leaked.append(frame.f_code.co_name)

        N = _build()
        with mock.patch.object(plugmod, "_FIXED_KIND", spy):
            for op in self._ops(N):
                saved = sys.getprofile()
                sys.setprofile(profile)
                try:
                    try:
                        op()
                    except Exception:  # noqa: BLE001 - some ops raise, as before
                        pass
                finally:
                    sys.setprofile(saved)
        self.assertEqual(leaked, [])
        self.assertIn(("==", "string"), compared)
        self.assertIn(("!=", "matrix"), compared)
        self.assertLessEqual(set(compared), {("==", "string"), ("!=", "matrix"), ("==", "matrix")})

    def test_a_fixed_kind_dst_queries_no_data_type(self):
        N = _build()
        for label, op in (
            ("md.input1X << 1.0", lambda: N["md"].input1X << 1.0),
            ("md.input1 << [1, 2, 3]", lambda: N["md"].input1 << [1, 2, 3]),
            ("md.input1X << md2.outputX", lambda: N["md"].input1X << N["md2"].outputX),
            ("md.input1 << md2.output", lambda: N["md"].input1 << N["md2"].output),
            ("md.operation << 2", lambda: N["md"].operation << 2),
            ("xf.tx << md2.outputX", lambda: N["xf"].tx << N["md2"].outputX),
        ):
            with self.subTest(op=label):
                # 619f29f: one data type query (a type query or a static hit) each
                with _Calls({_DATA_TYPE_CODE: "data_type"}) as calls:
                    self.assertEqual(_type_queries(op), [])
                self.assertEqual(calls.counts["data_type"], 0)

    def test_an_element_that_does_not_exist_is_queried(self):
        # the query creates it, so it runs; Attribute.set queries it once more, as
        # before
        N = _build()
        with _Calls({_DATA_TYPE_CODE: "data_type"}) as calls:
            N["pma"].input1D[4] << 2.0
        self.assertEqual(calls.by_arg["data_type"]["pmaD.input1D[4]"], 2)  # 619f29f: 3
        self.assertEqual(cmds.getAttr("pmaD.input1D", multiIndices=True), [4])
        # once it exists, it has a kind: only Attribute.set queries
        with _Calls({_DATA_TYPE_CODE: "data_type"}) as calls:
            N["pma"].input1D[4] << 3.0
        self.assertEqual(calls.by_arg["data_type"]["pmaD.input1D[4]"], 1)  # 619f29f: 2
        self.assertEqual(cmds.getAttr("pmaD.input1D[4]"), 3.0)

    def test_the_phantom_element_is_still_made(self):
        # the type query of a missing compound element creates it before the size
        # check raises (INJ-3b, which would skip it, is not taken)
        N = _build()
        with self.assertRaises(ValueError):
            N["pma"].input3D[7] << [1, 2]
        self.assertEqual(cmds.getAttr("pmaD.input3D", multiIndices=True), [7])

    def test_a_class_with_its_own_data_type_is_asked(self):
        N    = _build()
        asks = []
        real = Attribute.get_data_type

        def asked(attr):
            asks.append(str.__str__(attr))
            return real(attr)

        with mock.patch.object(Attribute, "get_data_type", asked):
            N["md"].input1X << 1.0
            self.assertEqual(cmds.getAttr("mdA.input1X"), 1.0)
            N["md"].input1 << N["md2"].output
        self.assertEqual(asks, ["mdA.input1X", "mdA.input1"])
        self.assertIn(("mdB.output", "mdA.input1"), scene_connections())

    def test_a_raised_type_is_read_at_the_generic_slot_test(self):
        # a compound source into a scalar dst of a node deleted to the undo queue:
        # its kind is fixed, its data type raises (None), which only the
        # generic-slot test reads
        N    = _build()
        held = N["md"].input1X
        str(held)
        cmds.delete("mdA")
        seen = []
        real = plugmod._resolved_data_type

        def recording(dst, data_type):
            answer = real(dst, data_type)
            seen.append((data_type, answer))
            return answer

        with mock.patch.object(plugmod, "_resolved_data_type", recording):
            raised = _raised(lambda: held << N["xf"].t)
        # the connect raises, and naming the dst in its InjectionError raises again
        self.assertEqual(
            raised,
            ("RuntimeError", "mdA already deleted!", ["RuntimeError", "mdA already deleted!"]),
        )
        self.assertEqual(seen, [(plugmod._FIXED_KIND, None)])


class TestOneSetOfFacts(MayaTestCase):
    """H4: one `_Facts` per operand of a ``<<`` with a plug source."""

    TEST_START_NEW_SCENE = True

    def _count(self, op):
        with _Calls({_DATA_TYPE_CODE: "data_type", _CHOICE_HOOK: "choice hook"}) as calls:
            op()
        return calls

    def test_a_generic_slot_is_queried_once(self):
        # re-pinned (features step 4, Typed): a generic slot as the destination never
        # reads its data type, and asks no hook (R5: once; 619f29f: 3 / 2 times)
        N     = _build()
        calls = self._count(lambda: N["ch"].input[0] << N["xf"].t)
        self.assertEqual(calls.by_arg["data_type"]["chE.input[0]"], 0)  # 619f29f: 3
        self.assertEqual(calls.counts["choice hook"], 0)                # 619f29f: 3
        calls = self._count(lambda: N["ch"].input[1] << N["md2"].outputX)
        self.assertEqual(calls.by_arg["data_type"]["chE.input[1]"], 0)  # 619f29f: 2
        self.assertIn(("xfC.translate", "chE.input[0]"), scene_connections())
        self.assertIn(("mdB.outputX", "chE.input[1]"), scene_connections())

    def test_the_choice_output_is_queried_once(self):
        N = _build()
        N["ch"].input[0] << N["md2"].output
        calls = self._count(lambda: N["md"].input1 << N["ch"].output)
        self.assertEqual(calls.by_arg["data_type"]["chE.output"], 1)    # 619f29f: 2
        self.assertIn(("chE.output", "mdA.input1"), scene_connections())
        # re-pinned (features step 4, Typed): into a generic slot no shorthand runs, so
        # neither operand's data type is read (R5: once each)
        calls = self._count(lambda: N["ch2"].input[0] << N["ch"].output)
        self.assertEqual(calls.by_arg["data_type"]["chE.output"], 0)    # 619f29f: 2
        self.assertEqual(calls.by_arg["data_type"]["chJ.input[0]"], 0)  # 619f29f: 2

    def test_an_auto_indexed_element_gets_its_own_facts(self):
        N     = _build()
        calls = self._count(lambda: N["ch"].input << N["xf"].t)
        self.assertEqual(calls.by_arg["data_type"]["chE.input"], 1)     # 619f29f: 3
        # re-pinned (features step 4, Typed): the element is a generic slot, whose data
        # type is never read (R5: once)
        self.assertEqual(calls.by_arg["data_type"]["chE.input[0]"], 0)  # 619f29f: 2
        # the whole compound is connected
        self.assertIn(("xfC.translate", "chE.input[0]"), scene_connections())

    def test_facts_are_reused_while_no_user_code_ran(self):
        N = _build()
        N["ch"].input[0] << N["md2"].output
        for label, plug in (
            ("rig's Choice and DGNode hooks", N["ch"].output),
            ("a static type, no hook", N["md"].input1X),
            ("an unconnected generic slot", N["ch2"].input[0]),
        ):
            with self.subTest(case=label):
                facts = shorthand_module._Facts(plug)
                facts._get_data_type()
                self.assertTrue(facts._reusable)
        user = _user_choice(N)
        user.input[0] << N["md2"].output
        for label, plug in (
            ("a class's own hook", user.input[5]),
            ("an instance hook", _instance_hook(N).input[3]),
        ):
            with self.subTest(case=label):
                facts = shorthand_module._Facts(plug)
                facts._get_data_type()
                self.assertFalse(facts._reusable)
        hook = _counting(DGNode.__dict__["_attr_data_type_fallback"], "patched")
        with mock.patch.object(DGNode, "_attr_data_type_fallback", hook):
            facts = shorthand_module._Facts(N["ch2"].input[1])
            facts._get_data_type()
            self.assertFalse(facts._reusable)  # a class-level patch of DGNode's

    def test_a_users_hook_runs_as_often_as_before(self):
        N    = _build()
        user = _user_choice(N)
        # re-pinned (features step 4, Typed): a generic slot as the destination asks no
        # hook (619f29f and R5: 3, 2, 2 and 0 runs)
        for label, op, runs in (
            ("a generic slot", lambda: user.input[0] << N["xf"].t, 0),         # 619f29f: 3
            ("an auto-index", lambda: user.input << N["md2"].output, 0),       # 619f29f: 2
            ("a scalar into a slot", lambda: user.input[4] << N["md2"].outputX, 0),  # 619f29f: 2
            ("a rewire (the slot is typed now)", lambda: user.input[0] << N["md2"].output, 0),
        ):
            with self.subTest(case=label):
                del _HOOK_CALLS[:]
                op()
                self.assertEqual(len(_HOOK_CALLS), runs)

    def test_a_cast_that_runs_a_constructor_counts_as_user_code(self):
        cmds.createNode("multiplyDivide", name="castMe")
        _base._CASTABLE_TYPES.clear()
        before = _base._USER_CODE_RUNS[0]
        Node("castMe")
        self.assertEqual(_base._USER_CODE_RUNS[0], before + 1)
        Node("castMe")  # the replay: no constructor runs
        self.assertEqual(_base._USER_CODE_RUNS[0], before + 1)

    def test_other_callers_pass_no_facts(self):
        N = _build()
        self.assertIs(shorthand_module.shorthand(N["md2"].output, N["md"].input1), False)
        plugmod._inject_value(N["md"].input1, N["md2"].output)
        self.assertIn(("mdB.output", "mdA.input1"), scene_connections())


class TestDgNamedOwner(MayaTestCase):
    """H2 / H7: `_dg_named_owner` and the two checks that stand in for the name."""

    TEST_START_NEW_SCENE = True

    def test_the_owner_test(self):
        N = _build()
        self.assertIs(_base._dg_named_owner(N["md"].input1X), N["md"])
        self.assertIs(_base._dg_named_owner(N["md"].input1.child(0)), N["md"])
        self.assertIs(_base._dg_named_owner(N["pma"].input3D[0]), N["pma"])
        self.assertIs(_base._dg_named_owner(N["ch"].output), N["ch"])
        self.assertIs(_base._dg_named_owner(N["md"].find_attr("input1X")), N["md"])
        loose = Plug("mdA.input1Y")
        for label, attr in (
            ("a DAG node's plug", N["xf"].tx),
            ("a shape's plug", Node("locHShape").worldMatrix[0]),
            ("a plug with no owner yet", loose),
        ):
            with self.subTest(case=label):
                self.assertIsNone(_base._dg_named_owner(attr))
        loose.get_node()  # casts its node: owned now
        self.assertIsNotNone(_base._dg_named_owner(loose))

        class _OwnName(DGNode):
            CUSTOM_NODE_TYPE = "r5injectOwnName"

            @property
            def name(self):
                return self.fn_set.name()

        cmds.createNode("multiplyDivide", name="ownName")
        set_custom_type("ownName", "r5injectOwnName")
        self.assertIsNone(_base._dg_named_owner(Node("ownName").input1X))
        node = N["md2"]
        node.__dict__["ensure_valid"] = node.ensure_valid
        self.assertIsNone(_base._dg_named_owner(node.input1X))

    def test_a_component_plug_is_never_such_a_plug(self):
        surface = cmds.nurbsPlane(constructionHistory=False, name="plane")[0]
        handle  = Node(surface).cv
        self.assertIsNone(_base._dg_named_owner(handle))
        self.assertIsNone(_base._dg_named_owner(handle[0, 0]))

    def test_a_patched_class_fails_the_test_at_once(self):
        # the classes are read at every call: a class-level patch of one of the
        # methods is seen while it lasts, and the patched method runs
        N        = _build()
        plug     = N["md"].input1X
        checked  = []
        original = DGNode.__dict__["ensure_valid"]

        def ensure_valid(node):
            checked.append(node)
            return original(node)

        self.assertIs(_base._dg_named_owner(plug), N["md"])
        with mock.patch.object(DGNode, "ensure_valid", ensure_valid):
            self.assertIsNone(_base._dg_named_owner(plug))
            self.assertEqual(plug.get_data_type(), "float")
            self.assertTrue(checked)  # through the name, as before
            del checked[:]
            N["md"].input1X << N["md2"].outputX
            self.assertTrue(checked)
        self.assertIs(_base._dg_named_owner(plug), N["md"])
        for owner, name, value in (
            (DGNode, "fn_set", property(lambda node: node._fn_set)),
            (DGNode, "name", property(lambda node: node._fn_set.name())),
            (type(N["md"]), "__getattribute__", lambda node, name: object.__getattribute__(node, name)),
            (Attribute, "get_full_name", lambda attr, f=Attribute.get_full_name: f(attr)),
        ):
            with self.subTest(patched=f"{owner.__name__}.{name}"):
                with mock.patch.object(owner, name, value):
                    self.assertIsNone(_base._dg_named_owner(plug))
                self.assertIs(_base._dg_named_owner(plug), N["md"])

    def _named_and_typed(self, attr):
        """(what naming `attr` raises, what its data_type raises, what
        `_named_with_dot` raises)."""
        return (
            _raised(lambda: attr.get_full_name()),
            _raised(lambda: attr.get_data_type()),
            _raised(lambda: plugmod._named_with_dot(attr)),
        )

    def test_the_checks_raise_as_the_name_does(self):
        N       = _build()
        deleted = N["md"].input1X
        str(deleted)
        cmds.delete("mdA")
        named, typed, dotted = self._named_and_typed(deleted)
        self.assertEqual(named, ("RuntimeError", "mdA already deleted!", None))
        self.assertEqual(typed, named)
        self.assertEqual(dotted, named)
        cmds.undo()
        self.assertEqual(self._named_and_typed(deleted), (None, None, None))

        freed_dyn = N["net"].dynm
        str(freed_dyn)
        cmds.deleteAttr("netK.dynm")
        cmds.flushUndo()  # the dynamic attribute is freed, its node lives on
        named, typed, dotted = self._named_and_typed(freed_dyn)
        self.assertEqual(named, ("RuntimeError", "netK.dynm already deleted!", None))
        self.assertEqual(typed, named)
        self.assertEqual(dotted, named)

        held = N["md2"].outputX
        str(held)
        cmds.file(new=True, force=True)
        named = _raised(lambda: held.get_full_name())
        self.assertEqual(named[0], "RuntimeError")
        self.assertTrue(named[1].endswith(
            "(freed by a new scene, a file open or a reference unload) already deleted!"
        ))
        self.assertEqual(_raised(lambda: held.get_data_type()), named)
        self.assertEqual(_raised(lambda: plugmod._named_with_dot(held)), named)

    def test_a_static_hit_builds_no_name(self):
        N = _build()
        for label, attr, named in (
            ("a DG plug", N["md"].input1X, False),           # 619f29f: named
            ("a DG compound", N["md"].input1, False),        # 619f29f: named
            ("a DAG plug", N["xf"].tx, True),
            ("a generic slot (no static type)", N["ch"].input[0], True),
        ):
            attr.get_data_type()  # a first query keeps the static type
            with self.subTest(case=label):
                with _Calls({_FULL_NAME_CODE: "full_name"}) as calls:
                    attr.get_data_type()
                self.assertEqual(calls.counts["full_name"] > 0, named)
        for label, attr, names in (
            ("a DG plug", N["md"].outputX, 0),  # 619f29f: 1
            ("a DAG plug", N["xf"].tx, 1),
        ):
            with self.subTest(dot=label):
                with _Calls({_FULL_NAME_CODE: "full_name"}) as calls:
                    self.assertIs(plugmod._named_with_dot(attr), True)
                self.assertEqual(calls.counts["full_name"], names)

    def test_a_deleted_source_raises_before_the_dst_is_disconnected(self):
        N   = _build()
        src = _held(N["md2"].outputX)
        N["md"].input1X << N["xf"].tx
        cmds.delete("mdB")
        with mock.patch.object(
            plugmod, "_disconnect_incoming", wraps=plugmod._disconnect_incoming
        ) as disconnect:
            raised = _raised(lambda: N["md"].input1X << src)
        self.assertEqual(raised, ("RuntimeError", "mdB already deleted!", None))
        self.assertEqual(disconnect.call_count, 0)
        self.assertIn(("xfC.translateX", "mdA.input1X"), scene_connections())

        held = N["net"].dynm
        str(held)
        N["md"].input2X << N["xf"].ty
        cmds.deleteAttr("netK.dynm")
        cmds.flushUndo()
        with mock.patch.object(
            plugmod, "_disconnect_incoming", wraps=plugmod._disconnect_incoming
        ) as disconnect:
            raised = _raised(lambda: plugmod._set_or_connect(held, N["md"].input2X))
        self.assertEqual(raised, ("RuntimeError", "netK.dynm already deleted!", None))
        self.assertEqual(disconnect.call_count, 0)
        self.assertIn(("xfC.translateY", "mdA.input2X"), scene_connections())


def _shapes():
    """The scene `_build` makes, wired, and (label, plug) for every plug shape
    `_disconnect_incoming` meets in it."""
    N = _build()
    for src, dst in (
        ("mdB.outputX", "mdA.input1X"),
        ("mdB.output", "mdA.input2"),
        ("mdB.outputY", "pmaD.input1D[0]"),
        ("mdB.outputZ", "pmaD.input1D[3]"),
        ("xfC.translate", "pmaD.input3D[0]"),
        ("mdB.outputX", "pmaD.input3D[1].input3Dy"),
        ("mdB.outputX", "xfI.rotateX"),
        ("xfC.translate", "chE.input[0]"),
        ("locH.worldMatrix[0]", "mmM.matrixIn[0]"),
        ("mdA.message", "netK.msgIn"),
        ("mdB.outputY", "netK.dynm"),
    ):
        cmds.connectAttr(src, dst)
    cmds.aliasAttr("myAlias", "mdA.input1Z")
    cmds.connectAttr("mdB.outputZ", "mdA.myAlias")
    cmds.setAttr("mdA.input1X", lock=True)
    md, pma, ch = N["md"], N["pma"], N["ch"]
    return N, [
        ("a connected scalar (locked)", md.input1X),
        ("a connected alias", md.find_attr("myAlias")),
        ("an unconnected scalar", md.input1Y),
        ("a connected compound", md.input2),
        ("a child of a connected compound", md.input2.child(0)),
        ("a compound with a connected child", md.input1),
        ("an array root with connected elements", pma.input1D),
        ("a connected element", pma.input1D[0]),
        ("a sparse connected element", pma.input1D[3]),
        ("a missing element", pma.input1D[1]),
        ("a connected compound element", pma.input3D[0]),
        ("a child of a connected compound element", pma.input3D[0].child(0)),
        ("a compound element with a connected child", pma.input3D[1]),
        ("a connected child of an element", pma.input3D[1].child(1)),
        ("a unitConversion-fed plug", N["xf2"].rx),
        ("the parent of a unitConversion-fed plug", N["xf2"].r),
        ("a connected generic slot", ch.input[0]),
        ("a missing generic slot", ch.input[5]),
        ("the choice output", ch.output),
        ("a connected matrix element", N["mm"].matrixIn[0]),
        ("a message dst", N["net"].msgIn),
        ("a connected dynamic attr", N["net"].dynm),
        ("an unconnected dynamic attr", N["xf"].dyn_d),
        ("a world matrix element (a source)", Node("locHShape").worldMatrix[0]),
        ("a plug built from a name", Plug("pmaD.input1D[0]")),
    ]


class TestDisconnectIncoming(MayaTestCase):
    """`_disconnect_incoming` returns for a plug that is no destination."""

    TEST_START_NEW_SCENE = True

    def test_is_destination_is_false_exactly_when_nothing_drives_the_plug(self):
        _, shapes = _shapes()
        held = dict(shapes)
        for label, plug in shapes:
            with self.subTest(shape=label):
                mplug = plug.get_mplug()
                self.assertEqual(bool(mplug.isDestination), bool(mplug.connectedTo(True, False)))
        # a node deleted to the undo queue, then its delete undone
        for step in ("delete", "undo"):
            if step == "delete":
                cmds.delete("mdA")
            else:
                cmds.undo()
            for label in ("a connected scalar (locked)", "a connected compound",
                          "a child of a connected compound", "an unconnected scalar"):
                with self.subTest(step=step, shape=label):
                    mplug = held[label].get_mplug()
                    self.assertEqual(bool(mplug.isDestination), bool(mplug.connectedTo(True, False)))
        # a connected dynamic attribute deleted to the undo queue
        cmds.deleteAttr("netK.dynm")
        mplug = held["a connected dynamic attr"].get_mplug()
        self.assertFalse(mplug.isDestination)
        self.assertEqual(len(mplug.connectedTo(True, False)), 0)

    def _disconnect(self, plug):
        """What `_disconnect_incoming(plug)` does: (what it raises, the connections
        then), and whether it listed the plug's connections."""
        with _Calls({_LISTING_CODE: "listing"}) as calls:
            raised = _raised(lambda: plugmod._disconnect_incoming(plug))
        return (raised, scene_connections()), calls.counts["listing"]

    def test_a_plug_that_is_no_destination_lists_no_connection(self):
        count = len(_shapes()[1])
        for index in range(count):
            cmds.file(new=True, force=True)
            label, plug = _shapes()[1][index]
            destination = bool(plug.get_mplug().isDestination)
            # re-pinned (review fix, features run): a child below a compound connected
            # whole has no input of its own, but the compound's input is split into one
            # per child first (`_free_of_whole_wire`), then the child's own is listed and
            # removed; both paths make that same scene, each listing more than once
            whole = not destination and plugmod._below_a_wire(plug.get_mplug())
            # one driver per channel (user decision 2026-10-16) is `<<`'s (``one_driver=True``:
            # a compound's children lose their own inputs too, `test_od_one_driver`); this
            # default call, rig's own moves of a wire (a shader conversion, a blendShape
            # load, keys), removes the plug's own input only, so the compound shapes with a
            # connected child ("a compound with a connected child", "a compound element with
            # a connected child", "the parent of a unitConversion-fed plug") are unchanged
            with self.subTest(shape=label):
                new, listed = self._disconnect(plug)
                if not whole:
                    self.assertEqual(listed, int(destination))
                # the old path (the class test fails): the connections listed
                cmds.file(new=True, force=True)
                plug = _shapes()[1][index][1]
                with mock.patch.object(plugmod, "_GET_CONNECTED_ATTRS", None):
                    old, listed = self._disconnect(plug)
                if not whole:
                    self.assertEqual(listed, 1)
                self.assertEqual(new, old)

    def test_deleted_and_freed_dsts_as_the_old_path(self):
        def outcomes(fast):
            cmds.file(new=True, force=True)
            N, shapes = _shapes()
            plugs     = dict(shapes)
            dyn       = plugs["a connected dynamic attr"]
            freed_dyn = N["xf"].dyn_d
            str(freed_dyn)
            if fast:
                patch = contextlib.nullcontext()
            else:
                patch = mock.patch.object(plugmod, "_GET_CONNECTED_ATTRS", None)
            out = []
            with patch:
                cmds.deleteAttr("netK.dynm")
                out.append(self._disconnect(dyn)[0])
                cmds.deleteAttr("xfC.dyn_d")
                cmds.flushUndo()
                out.append(self._disconnect(freed_dyn)[0])
                cmds.delete("mdA")
                for label in ("a connected scalar (locked)", "a connected compound",
                              "an unconnected scalar"):
                    out.append(self._disconnect(plugs[label])[0])
                held = plugs["a connected element"]
                cmds.file(new=True, force=True)
                out.append(self._disconnect(held)[0])
            return out

        fast = outcomes(True)
        self.assertEqual(fast, outcomes(False))
        self.assertEqual(fast[1][0][:2], ("RuntimeError", "xfC.dyn_d already deleted!"))

    def test_a_class_with_its_own_reads_takes_the_old_path(self):
        N = _build()
        with mock.patch.object(Attribute, "get_connected_attrs", side_effect=RuntimeError("boom")):
            self.assertEqual(
                _raised(lambda: plugmod._disconnect_incoming(N["md"].input1X)),
                (
                    "InjectionError",
                    "Cannot inspect incoming connections on 'mdA.input1X': boom",
                    ["RuntimeError", "boom"],
                ),
            )
        plug = N["md"].input1Y
        plug.__dict__["iter_connected_attrs"] = lambda *a, **k: iter([N["md2"].outputX])
        with mock.patch.object(Attribute, "disconnect") as disconnect:
            plugmod._disconnect_incoming(plug)
        self.assertEqual(disconnect.call_count, 1)  # the old path read the instance's
