"""Round 5 D3: single numeric value sets through the API (`rig._internal.api_edits.api_set`).

``attr.set(value)`` (and ``attr << value``) with one int, float or bool value, no flag, into a
free numeric, unit, enum or bool attribute outside any array is an ``MDGModifier``, journaled,
when ``cmds.setAttr`` would set the same value (CR-4's table): a statement's sets are one undo
step with its connects. Every other set takes the cmds route, which raises today's errors.
Locks, keyable and channelBox flags stay on cmds (plan_D H22). The approved difference (user,
2026-10-02): one undo step per statement for its API edits.

* `TestTheCoercionMatrix` (CR-4): 6 kinds x 13 values, more kinds and values (float, char,
  the unit kinds, enums with gaps and a negative field, hard limits, numpy scalars, a bool
  into a double, ints past 32 bits), and the units: the API on and off give the same value or
  the same error and printed lines, and a ``cmds.setAttr`` spy counts the cmds-routed ones.
* `TestTheRoute`: which sets take the API, and every guard case gives the cmds route's
  outcome (a connected dst, locked plugs and parents, a proxy, an output, deleted and freed
  nodes, an array element with today's type query, typed kinds, flags, a user's classes ...).
* `TestStatementGranularity` (V6, the dangling check after every undo / redo step): a
  statement's sets are one step; sets and connects in one statement; user cmds interleaved;
  a disconnect-and-retry; CR-5 with a user's ``post_create``; undo off.
* `TestReferenceEdits`: a set on a referenced node records the same reference edits on both
  routes, and its undo removes them.
* `TestPluginLoading` (U1): without rig's plug-in, sets take cmds, with the one warning.
"""

import contextlib
import math
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig import List, Node
from rig._internal import api_edits, journal
from rig._internal.plug import Plug
from rig.nodetypes import _base, plugins
from rig.nodetypes.transform import Transform
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

PLUGIN = "undoable_api_command"

_VALUES = [2.7, -2.7, 2.5, 3.5, 2, -1, 0.4, 5, 300, 40000, 1e10, float("nan"), float("inf")]


class _SetCase(UndoWalk, MayaTestCase):
    """A new scene, the undo queue on, infinite and flushed, rig's plug-in loaded; the
    transforms ``a``, ``b`` and the multiplyDivide ``m``. The journal is closed and empty
    after every test."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        plugins._ensure_loaded(PLUGIN, True)
        self.make_nodes()
        cmds.flushUndo()

    def make_nodes(self):
        for name in ("a", "b"):
            cmds.createNode("transform", name=name)
        cmds.createNode("multiplyDivide", name="m")

    def tearDown(self):
        try:
            self.assertEqual((journal._S.open, journal._S.pending), (False, []))
        finally:
            journal._S.open, journal._S.pending = False, []
            cmds.flushUndo()
            cmds.currentUnit(linear="cm", angle="deg")
            plugins._ensure_loaded(PLUGIN, True)
            super().tearDown()

    def queue_empty(self):
        return cmds.undoInfo(query=True, undoQueueEmpty=True)

    def count_steps(self):
        """Undoes the whole queue (the dangling check after each step), redoes it, and
        returns the number of steps."""
        steps = 0
        while not self.queue_empty():
            self.undo_steps(1)
            steps += 1
            self.assertLess(steps, 1000)
        self.redo_all()
        return steps

    @contextlib.contextmanager
    def watch(self, api):
        """``API_EDITS`` = ``api``; yields a record: ``routes`` (what each `api_set` call
        returned), ``cmds`` (the ``cmds.setAttr`` calls), ``printed`` (Maya's Error: and
        Warning: lines)."""
        record = {"routes": [], "printed": [], "cmds": None}
        real = api_edits.api_set

        def spy(attr, value):
            record["routes"].append(real(attr, value))
            return record["routes"][-1]

        def on_output(message, kind, *_):
            if kind in (OpenMaya.MCommandMessage.kError, OpenMaya.MCommandMessage.kWarning):
                record["printed"].append((kind, message.strip()))

        callback = OpenMaya.MCommandMessage.addCommandOutputCallback(on_output)
        try:
            with mock.patch.object(journal, "API_EDITS", api), mock.patch.object(_base, "_api_set", spy), \
                    mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
                record["cmds"] = set_attr
                yield record
        finally:
            OpenMaya.MMessage.removeCallback(callback)


def _brief(error, depth=0):
    """An exception and its chain, as comparable data (type, text, __suppress_context__)."""
    if error is None or depth > 5:
        return None
    return (type(error).__name__, str(error), error.__suppress_context__,
            _brief(error.__context__, depth + 1), _brief(error.__cause__, depth + 1))


def _same(x, y):
    """Equal values, NaN equal to NaN, and of the same type."""
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return type(x) is type(y) and x == y


def _attrs(node, specs):
    for name, kwargs in specs:
        cmds.addAttr(node, longName=name, **kwargs)


# (long name, addAttr kwargs) of the coercion matrix's attributes
_SIX = [
    ("qlg", {"attributeType": "long"}),
    ("qsh", {"attributeType": "short"}),
    ("qbo", {"attributeType": "bool"}),
    ("qen", {"attributeType": "enum", "enumName": "a:b:c"}),
    ("qby", {"attributeType": "byte"}),
    ("qdb", {"attributeType": "double"}),
]
_MORE = [
    ("qfl", {"attributeType": "float"}),
    ("qch", {"attributeType": "char"}),
    ("qdl", {"attributeType": "doubleLinear"}),
    ("qda", {"attributeType": "doubleAngle"}),
    ("qfli", {"attributeType": "floatLinear"}),
    ("qfa", {"attributeType": "floatAngle"}),
    ("qeg", {"attributeType": "enum", "enumName": "a=0:b=5:c=10"}),
    ("qen2", {"attributeType": "enum", "enumName": "x=-3:y=2"}),
    ("qmn", {"attributeType": "double", "minValue": -1.0, "maxValue": 4.0}),
    ("qmi", {"attributeType": "long", "minValue": 0, "maxValue": 10}),
    ("qmo", {"attributeType": "double", "minValue": 0.0}),
    ("qby2", {"attributeType": "byte", "minValue": 2, "maxValue": 10}),
    ("qdlm", {"attributeType": "doubleLinear", "minValue": 0.0}),
    ("qsm", {"attributeType": "double", "softMinValue": 0.0, "softMaxValue": 1.0}),
]


def _int_in(lo, hi):
    return lambda v: type(v) is int and lo <= v <= hi


def _float_or_int32(v):
    return type(v) is float or (type(v) is int and -(2 ** 31) <= v <= 2 ** 31 - 1)


# {attribute: the values the API takes} (CR-4's table, written out independently)
_TAKES = {
    "qlg":  _int_in(-(2 ** 31), 2 ** 31 - 1),
    "qsh":  _int_in(-32768, 32767),
    "qbo":  lambda v: type(v) in (int, bool) and v in (0, 1),
    "qen":  _int_in(0, 2),
    "qby":  _int_in(0, 255),
    "qdb":  _float_or_int32,
    "qfl":  _float_or_int32,
    "qch":  _int_in(0, 127),
    "qdl":  _float_or_int32,
    "qda":  _float_or_int32,
    "qfli": _float_or_int32,
    "qfa":  _float_or_int32,
    "qeg":  _int_in(0, 10),
    "qen2": _int_in(-3, 2),
    "qmn":  lambda v: False,  # a hard min / max: cmds
    "qmi":  lambda v: False,
    "qmo":  lambda v: False,
    "qby2": _int_in(0, 255),  # Maya keeps a byte's own 0..255, not addAttr's min / max
    "qdlm": lambda v: False,
    "qsm":  _float_or_int32,  # soft limits are no limits
}

_MORE_VALUES = _VALUES + [
    0, 1, True, False, -0.0, 127, 128, 255, 256, -128, 32767, 32768, -32768, -32769,
    2 ** 31 - 1, 2 ** 31, -(2 ** 31), -(2 ** 31) - 1, 2 ** 53, -2.5, 4, 4.0, 4.5, -1.5, 10, 11,
    np.float32(2.5), np.float64(2.5), np.int64(3), np.int32(3), np.bool_(True),
]


class TestTheCoercionMatrix(_SetCase):
    """CR-4: the API takes a value only where cmds.setAttr sets the same one."""

    def outcome(self, plug, value, api):
        """(what ``plug`` reads after ``Node.set(value)`` or the error, the printed lines,
        the route, the cmds.setAttr calls), from 0 (or the value it had, if it refuses 0)."""
        try:
            cmds.setAttr(plug, 0)
        except RuntimeError:
            pass
        attr = Node(plug.split(".")[0]).find_attr(plug.split(".")[1])
        with self.watch(api) as record:
            try:
                attr.set(value)
                result = ("value", cmds.getAttr(plug))
            except Exception as error:  # noqa: BLE001 (the error is what is compared)
                result = ("error", _brief(error))
        return result, record["printed"], record["routes"], record["cmds"].call_count

    def compare(self, specs, values, attrs=None):
        node = cmds.createNode("transform", name="V")
        _attrs(node, specs)
        cmds.flushUndo()
        api_count = 0
        for name, _ in specs:
            if attrs is not None and name not in attrs:
                continue
            plug = f"{node}.{name}"
            for value in values:
                with self.subTest(attr=name, value=repr(value)):
                    on, printed_on, routes, cmds_on = self.outcome(plug, value, True)
                    off, printed_off, _, cmds_off = self.outcome(plug, value, False)
                    if on[0] == "value" and off[0] == "value":
                        self.assertTrue(_same(on[1], off[1]), (on, off))
                    else:
                        self.assertEqual(on, off)
                    self.assertEqual(printed_on, printed_off)
                    takes = _TAKES[name](value)
                    self.assertEqual(routes, [takes] if routes else [])
                    if type(value) in (int, float, bool):
                        self.assertEqual(routes, [takes])
                    # the cmds-routed values reach cmds.setAttr, the API-routed ones do not
                    self.assertEqual((cmds_on, cmds_off), (0 if takes else 1, 1))
                    api_count += takes
        return api_count

    def test_six_kinds_by_thirteen_values(self):
        # long 2 -1 5 300 40000, short 2 -1 5 300, enum 2, byte 2 5, double all 13
        self.assertEqual(self.compare(_SIX, _VALUES), 5 + 4 + 1 + 2 + 13)

    def test_more_kinds_and_values(self):
        self.assertGreater(self.compare(_SIX + _MORE, _MORE_VALUES), 300)

    def test_an_enum_past_its_max_raises_cmds_text(self):
        for api in (True, False):
            cmds.file(new=True, force=True)
            node = cmds.createNode("transform", name="V")
            _attrs(node, _SIX)
            with self.subTest(api=api), self.watch(api) as record:
                with self.assertRaisesRegex(RuntimeError, "past its maximum") as raised:
                    Node("V").qen.set(3)
                self.assertEqual(record["routes"], [False] if api else [])
                self.assertIsNone(raised.exception.__context__)
                self.assertEqual(cmds.getAttr("V.qen"), 0)

    def test_hard_limits_inside_at_and_past(self):
        self.assertEqual(self.compare(_MORE, [-1.5, -1, -1.0, 0, 0.0, 2, 2.5, 4, 4.0, 4.5, 10, 11, 1e10],
                                      attrs={"qmn", "qmi", "qmo", "qdlm"}), 0)
        # a byte's addAttr min / max are not kept: its own 0..255 (cmds sets 0 and 11 too)
        self.assertEqual(self.compare(_MORE, [0, 1, 2, 10, 11, 255, 256, -1], attrs={"qby2"}), 6)

    def test_numpy_scalars_and_a_bool_into_a_double_take_cmds(self):
        values = [np.float32(2.5), np.float64(2.5), np.int64(3), np.int32(3), np.bool_(True), True, False]
        self.assertEqual(self.compare(_SIX, values, attrs={"qdb"}), 0)

    def test_the_units(self):
        values = [2.7, -2.7, 0.4, 90, 300, 1e10, float("nan"), float("inf")]
        for linear, angle in (("cm", "deg"), ("m", "rad"), ("in", "deg")):
            cmds.file(new=True, force=True)
            cmds.currentUnit(linear=linear, angle=angle)
            with self.subTest(units=(linear, angle)):
                self.assertEqual(self.compare(_MORE, values, attrs={"qdl", "qda", "qfli", "qfa"}), 32)
                # a transform's translate and rotate (internal cm / rad, read in the UI unit)
                for plug in ("V.tx", "V.rz"):
                    for value in (2.5, 90, -1e3):
                        on = self.outcome(plug, value, True)
                        off = self.outcome(plug, value, False)
                        self.assertEqual(on[:2], off[:2])
                        self.assertEqual(on[2:], ([True], 0))
                        internal = OpenMaya.MSelectionList().add(plug).getPlug(0).asDouble()
                        unit = (OpenMaya.MDistance(value, OpenMaya.MDistance.uiUnit()).asCentimeters()
                                if plug == "V.tx" else
                                OpenMaya.MAngle(value, OpenMaya.MAngle.uiUnit()).asRadians())
                        self.assertAlmostEqual(internal, unit)
        cmds.currentUnit(linear="cm", angle="deg")


class _UserPlug(Plug):
    """A user's Plug subclass: not an exact rig class, so the cmds route."""


class _UserNamed(Transform):
    """A user's node class with its own naming: the cmds route."""

    @property
    def name(self):
        return self.fn_set.partialPathName()


def _guard_cases():
    """(label, the set, the route the API_EDITS=True run's `api_set` calls return: a list)"""

    def connected_dst_shift():
        cmds.connectAttr("a.tx", "b.ty")
        Node("b").ty << 2.5

    def connected_dst_set():
        cmds.connectAttr("a.tx", "b.ty")
        Node("b").ty.set(2.5)

    def connected_parent():
        cmds.connectAttr("a.t", "b.t")
        Node("b").tx << 2.5

    def locked_parent():
        cmds.setAttr("b.t", lock=True)
        Node("b").tx.set(2.5)

    def proxy():
        cmds.addAttr("a", longName="orig", attributeType="double")
        cmds.addAttr("b", longName="prox", proxy="a.orig")
        Node("b").prox.set(3.0)

    def lock_node():
        cmds.lockNode("b", lock=True)
        try:
            Node("b").tx.set(2.5)
        finally:
            cmds.lockNode("b", lock=False)

    def deleted():
        p = Node("b").ty
        cmds.delete("b")
        p.set(2.5)

    def freed():
        p = Node("m").input1X
        cmds.file(new=True, force=True)
        cmds.createNode("multiplyDivide", name="m")
        p.set(2.5)

    def deleted_dynamic():
        cmds.addAttr("b", longName="dyn", attributeType="double")
        p = Node("b").dyn
        cmds.deleteAttr("b.dyn")
        p.set(2.5)

    def readded_dynamic():
        cmds.addAttr("b", longName="dyn", attributeType="double")
        p = Node("b").dyn
        cmds.deleteAttr("b.dyn")
        cmds.addAttr("b", longName="dyn", attributeType="long")
        p.set(2.5)

    def non_writable():
        cmds.addAttr("b", longName="nw", attributeType="double", writable=False)
        Node("b").nw.set(2.5)

    def array_element():
        cmds.createNode("plusMinusAverage", name="p")
        Node("p").input1D[3].set(2.5)

    def element_child():
        cmds.createNode("plusMinusAverage", name="p")
        Node("p").input3D[1].input3Dy.set(2.5)

    def element_child_locked_root():
        cmds.createNode("plusMinusAverage", name="p")
        cmds.setAttr("p.input3D[0].input3Dx", 1.0)
        cmds.setAttr("p.input3D", lock=True)
        Node("p").input3D[0].input3Dx.set(2.5)

    def alias():
        cmds.aliasAttr("hey", "m.input1X")
        Node("m").hey.set(2.5)

    def time_kind():
        cmds.addAttr("b", longName="tm", attributeType="time")
        Node("b").tm.set(2.5)

    def string_kind():
        cmds.addAttr("b", longName="st", dataType="string")
        Node("b").st.set("x")

    def compound_list():
        Node("b").t.set((1, 2, 3))

    def scalar_into_compound():
        Node("b").t.set(2.5)

    def message():
        Node("b").message.set(1)

    def bool_from_float():
        Node("b").v.set(0.0)

    def enum_name():
        Node("b").ro << "zxy"

    def str_value():
        Node("b").tx.set("2.5")

    def user_plug():
        _UserPlug("b.tx").set(2.5)

    def user_naming():
        _UserNamed("b").tx.set(2.5)

    def instanced():
        cmds.createNode("transform", name="c", parent="a")
        cmds.instance("a", name="a2")
        Node("a2|c").tx.set(2.5)

    def namespaced():
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:d")
        Node("ns:d").ty << 2.5

    return [
        ("free", lambda: Node("b").tx.set(2.5), [True]),
        ("shift", lambda: Node("b").tx << 2.5, [True]),
        ("int into a double", lambda: Node("b").tx.set(2), [True]),
        ("enum", lambda: Node("b").ro.set(3), [True]),
        ("an enum's name", enum_name, [True]),
        ("bool", lambda: Node("b").v.set(0), [True]),
        ("alias", alias, [True]),
        ("lockNode", lock_node, [True]),
        ("an instanced node", instanced, [True]),
        ("a namespace", namespaced, [True]),
        ("a connected dst, <<: disconnect and retry", connected_dst_shift, [False, True]),
        ("a connected dst, set", connected_dst_set, [False]),
        ("a connected parent, <<", connected_parent, [False]),
        ("locked", lambda: (cmds.setAttr("b.tx", lock=True), Node("b").tx.set(2.5)), [False]),
        ("locked parent", locked_parent, [False]),
        ("a proxy", proxy, [False]),
        ("an output (cmds sets it)", lambda: Node("m").outputX.set(2.5), [False]),
        ("non-writable", non_writable, [False]),
        ("deleted node", deleted, [False]),
        ("freed node", freed, []),
        ("deleted dynamic attr", deleted_dynamic, []),
        ("re-added dynamic attr", readded_dynamic, []),
        ("array element", array_element, []),
        ("element child", element_child, []),
        ("element child under a locked root", element_child_locked_root, []),
        ("time", time_kind, [False]),
        ("string", string_kind, []),
        ("compound, a list", compound_list, [False]),
        ("a scalar into a compound", scalar_into_compound, [False]),
        ("message", message, [False]),
        ("a float into a bool", bool_from_float, [False]),
        ("a str value", str_value, [False]),
        ("flags", lambda: Node("b").tx.set(2.5, lock=True), []),
        ("a user's Plug class", user_plug, [False]),
        ("a user's node naming", user_naming, [False]),
    ]


class TestTheRoute(_SetCase):
    """Which sets take the API, and that the guard cases give the cmds route's outcome."""

    def test_a_set_runs_no_cmds_set_attr(self):
        with mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").tx << 2.5
            Node("a").ry.set(45)
        self.assertEqual(set_attr.call_count, 0)
        self.assertEqual((cmds.getAttr("a.tx"), cmds.getAttr("a.ry")), (2.5, 45.0))
        self.assertEqual(self.count_steps(), 2)
        self.assertEqual((cmds.getAttr("a.tx"), cmds.getAttr("a.ry")), (2.5, 45.0))

    def test_the_switch_off_takes_cmds(self):
        with mock.patch.object(journal, "API_EDITS", False), \
                mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").tx << 2.5
        self.assertEqual(set_attr.call_count, 1)
        self.assertEqual(cmds.getAttr("a.tx"), 2.5)

    def state(self):
        found = []
        for node in cmds.ls(type=("transform", "multiplyDivide", "plusMinusAverage")):
            if node in ("persp", "top", "front", "side"):
                continue
            for attr in cmds.listAttr(node, keyable=True, multi=True) or []:
                try:
                    found.append((node + "." + attr, repr(cmds.getAttr(node + "." + attr))))
                except Exception:  # noqa: BLE001
                    pass
            found.extend(cmds.listConnections(node, connections=True, plugs=True, source=True,
                                              destination=False) or [])
        return found

    def outcome(self, case, api):
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.flushUndo()
        with self.watch(api) as record:
            try:
                case()
                error = None
            except Exception as exc:  # noqa: BLE001 (the error is what is compared)
                error = _brief(exc)
            types = [c for c in record["cmds"].call_args_list]
        state = (error, self.state(), sorted(cmds.ls()))
        undone = (self.undo_all(), self.state(), sorted(cmds.ls()))[1:]
        return state, undone, record["routes"], record["printed"], len(types)

    def test_every_guard_case_gives_the_cmds_outcome(self):
        for label, case, routes in _guard_cases():
            with self.subTest(case=label):
                on, on_undone, routes_on, printed_on, cmds_on = self.outcome(case, True)
                off, off_undone, _, printed_off, cmds_off = self.outcome(case, False)
                self.assertEqual(on, off)
                self.assertEqual(on_undone, off_undone)
                self.assertEqual(printed_on, printed_off)
                self.assertEqual(routes_on, routes)
                self.assertEqual(cmds_on, cmds_off - routes.count(True))

    def test_an_array_element_keeps_its_type_query(self):
        # today's query (getAttr -type), made once on both routes; the API is not asked
        cmds.createNode("plusMinusAverage", name="p")
        counts = []
        for api in (True, False):
            with self.watch(api) as record, mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as get:
                Node("p").input1D[2].set(1.5)
            counts.append((len([c for c in get.call_args_list if c.kwargs.get("type")]), record["routes"]))
        self.assertEqual(counts[0][0], counts[1][0])
        self.assertGreater(counts[0][0], 0)
        self.assertEqual(counts[0][1], [])
        self.assertEqual(cmds.getAttr("p.input1D[2]"), 1.5)

    def test_a_failed_registration_takes_cmds(self):
        # the modifier is undone and the cmds route sets the value; no API error is chained
        def refuse(_op):
            raise RuntimeError("no registration")

        with mock.patch.object(api_edits, "_add", refuse), \
                mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").tx.set(2.5)
        self.assertEqual(set_attr.call_count, 1)
        self.assertEqual(cmds.getAttr("a.tx"), 2.5)
        self.assertEqual(self.count_steps(), 1)

    def test_a_failed_modifier_call_takes_cmds(self):
        class Refusing:
            def __init__(self):
                self.done = False

            def newPlugValueDouble(self, *_):  # noqa: N802
                raise RuntimeError("refused")

        with mock.patch.object(api_edits, "_MDGModifier", Refusing), \
                mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").sx.set(2.5)
        self.assertEqual(set_attr.call_count, 1)
        self.assertEqual(cmds.getAttr("a.sx"), 2.5)

    def test_locks_keyable_and_channel_box_stay_cmds(self):
        with self.watch(True) as record:
            Node("a").tx.set(lock=True)
            Node("a").ty.set(keyable=False)
            Node("a").tz.set(channelBox=True)
            Node("a").tx.is_locked = False
        self.assertEqual(record["routes"], [])
        self.assertEqual(record["cmds"].call_count, 4)


class TestStatementGranularity(_SetCase):
    """V6: a statement's value sets join its journal, with the dangling check after each step."""

    def test_a_compound_value_is_one_step(self):
        # R5 D2 pinned three cmds steps (test_r5_api_edits.test_value_sets_stay_cmds_steps)
        Node("a").t << (1, 2, 3)
        self.assertEqual(cmds.getAttr("a.t"), [(1.0, 2.0, 3.0)])
        self.assertEqual(self.count_steps(), 1)
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("a.t"), [(0.0, 0.0, 0.0)])
        self.redo_steps(1)
        self.assertEqual(cmds.getAttr("a.t"), [(1.0, 2.0, 3.0)])

    def fanout(self, api):
        nodes = [Node(cmds.createNode("transform", name=f"n{i}")) for i in range(50)]
        cmds.flushUndo()
        with mock.patch.object(journal, "API_EDITS", api):
            List(nodes).tx << 1.5
        self.assertEqual([cmds.getAttr(f"n{i}.tx") for i in range(50)], [1.5] * 50)
        return self.count_steps()

    def test_a_list_fanout_of_fifty_is_one_step(self):
        self.assertEqual(self.fanout(api=False), 50)
        cmds.file(new=True, force=True)
        self.make_nodes()
        self.assertEqual(self.fanout(api=True), 1)
        self.undo_steps(1)
        self.assertEqual({cmds.getAttr(f"n{i}.tx") for i in range(50)}, {0.0})

    def walk(self, statements, api, probe):
        """Runs ``statements`` (each a callable) with ``API_EDITS`` = ``api``; returns the
        states after each, the number of undo steps, and the states undo then gives step by
        step (the dangling check after each)."""
        states = [probe()]
        with mock.patch.object(journal, "API_EDITS", api):
            for statement in statements:
                statement()
                states.append(probe())
        walked = []
        while not self.queue_empty():
            self.undo_steps(1)
            walked.append(probe())
        self.redo_all()
        return states, walked, probe()

    def probe(self):
        return ([cmds.getAttr(p) for p in ("a.tx", "a.ty", "b.tz", "m.input1X", "m.input2X")],
                cmds.listConnections("m", connections=True, plugs=True, source=True, destination=False))

    def run_both(self, statements):
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.flushUndo()
        off = self.walk(statements, False, self.probe)
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.flushUndo()
        on = self.walk(statements, True, self.probe)
        return on, off

    def test_sets_and_connects_in_one_statement(self):
        def network():
            Node("m").input1X << (Node("a").tx + 2.5) * 0.5

        def mixed():
            List([Node("m").input2X, Node("a").ty]) << 3.5

        on, off = self.run_both([network, mixed])
        self.assertEqual(on[0], off[0])
        self.assertEqual(on[2], off[2])
        self.assertLess(len(on[1]), len(off[1]))
        self.assertEqual(on[1][-1], off[1][-1])

    def test_user_cmds_interleaved(self):
        statements = [
            lambda: Node("a").tx << 1.5,
            lambda: cmds.setAttr("a.ty", 3.0),   # a user's cmds edit
            lambda: Node("b").tz << 2,
            lambda: cmds.setAttr("a.tx", 4.0),   # a user's cmds edit of a journaled plug
            lambda: Node("a").ty << -1.0,
        ]
        on, off = self.run_both(statements)
        # each statement is one step either way: the same states, step by step
        self.assertEqual(on, off)
        self.assertEqual(len(on[1]), 5)

    def test_a_disconnect_and_retry(self):
        def setup():
            cmds.connectAttr("a.tx", "m.input1X")

        def overwrite():
            Node("m").input1X << 2.5

        on, off = self.run_both([setup, overwrite])
        self.assertEqual(on, off)  # the disconnect is a cmds step, the retried set one more

    def test_a_post_create_set_of_a_journaled_plug(self):
        # CR-5: a user's post_create (run outside the statement) sets the plug the statement
        # set through the API just before: undo walks the user's set first, then the journal
        class UsersTransform(Transform):
            @classmethod
            def post_create(cls, new_node_name, *args, **kwargs):
                cmds.setAttr("a.tz", 9.0)
                return super().post_create(new_node_name, *args, **kwargs)

        def body():
            Node("a").tz << 1.0
            UsersTransform.create(name="x")
            Node("x").ty << 2.0

        def probe():
            return (cmds.getAttr("a.tz"), cmds.objExists("x") and cmds.getAttr("x.ty"))

        walks = []
        for api in (False, True):
            cmds.file(new=True, force=True)
            self.make_nodes()
            cmds.flushUndo()
            with mock.patch.object(journal, "API_EDITS", api):
                journal.run_statement(body, (), {})
            walked = [probe()]
            while not self.queue_empty():
                self.undo_steps(1)
                walked.append(probe())
            self.redo_all()
            walks.append((walked, probe()))
        self.assertEqual(walks[0][0][0], (9.0, 2.0))
        self.assertEqual(walks[1][1], (9.0, 2.0))
        # the same states in the same order; the API run has fewer steps (its two sets are in
        # two journals: the user's call-out registers the first)
        api_states = [w for i, w in enumerate(walks[1][0]) if i == 0 or w != walks[1][0][i - 1]]
        cmds_states = [w for i, w in enumerate(walks[0][0]) if i == 0 or w != walks[0][0][i - 1]]
        self.assertEqual(api_states, cmds_states)
        self.assertIn((1.0, False), api_states)

    def test_undo_off(self):
        for api in (True, False):
            cmds.file(new=True, force=True)
            self.make_nodes()
            cmds.flushUndo()
            cmds.undoInfo(stateWithoutFlush=False)
            try:
                with mock.patch.object(journal, "API_EDITS", api):
                    Node("a").tx << 2.5
                    Node("a").t << (1, 2, 3)
            finally:
                cmds.undoInfo(stateWithoutFlush=True)
            with self.subTest(api=api):
                self.assertEqual(cmds.getAttr("a.t"), [(1.0, 2.0, 3.0)])
                self.assertTrue(self.queue_empty())


class TestReferenceEdits(_SetCase):
    """A set on a referenced node records the same reference edits on both routes."""

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rig_d3_ref_")
        self.path = os.path.join(self.folder, "d3_ref.ma").replace("\\", "/")
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="rl")
        cmds.addAttr("rl", longName="lk", attributeType="double")
        cmds.setAttr("rl.lk", lock=True)
        cmds.file(rename=self.path)
        cmds.file(save=True, force=True, type="mayaAscii")

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(self.folder, ignore_errors=True)

    def edits(self, api):
        cmds.file(new=True, force=True)
        self.make_nodes()
        cmds.file(self.path, reference=True, namespace="R")
        cmds.flushUndo()
        with mock.patch.object(journal, "API_EDITS", api):
            Node("R:rl").tx << 2.5
            Node("R:rl").ro << 3
            Node("R:rl").t << (4, 5, 6)
            try:
                Node("R:rl").lk << 1.0  # locked in the file: refused on both routes
            except Exception as error:  # noqa: BLE001
                refused = _brief(error)
        node = cmds.referenceQuery(self.path, referenceNode=True)
        after = sorted(cmds.referenceQuery(node, editStrings=True))
        values = cmds.getAttr("R:rl.t") + [cmds.getAttr("R:rl.ro")]
        self.undo_all()
        undone = sorted(cmds.referenceQuery(node, editStrings=True))
        self.redo_all()
        return after, values, undone, sorted(cmds.referenceQuery(node, editStrings=True)), refused

    def test_the_same_edits_on_both_routes(self):
        on, off = self.edits(True), self.edits(False)
        self.assertEqual(on, off)
        self.assertTrue([e for e in on[0] if e.startswith("setAttr")])
        self.assertEqual([e for e in on[2] if e.startswith("setAttr")], [])
        self.assertEqual(on[3], on[0])


class TestPluginLoading(_SetCase):
    """U1: without rig's plug-in, value sets take cmds, with the one warning of a session."""

    NOT_FOUND = 'Plug-in, "undoable_api_command", was not found on MAYA_PLUG_IN_PATH.'

    def test_missing_sets_take_cmds_with_one_warning(self):
        cmds.flushUndo()
        if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
            cmds.unloadPlugin(PLUGIN)
        load = mock.MagicMock(side_effect=RuntimeError(self.NOT_FOUND))
        with mock.patch.object(plugins, "bundled_plugin_path", return_value=None), \
                mock.patch.object(cmds, "loadPlugin", load), \
                mock.patch.object(plugins, "_WARNED", False), \
                mock.patch.object(plugins, "_LOAD_FAILED", None), \
                mock.patch.object(cmds, "warning") as warning, \
                mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").tx << 2.5
            Node("a").ty << 3.5
            Node("a").tz << Node("b").tx
        self.assertEqual(load.call_count, 1)
        self.assertEqual(set_attr.call_count, 2)
        self.assertEqual(warning.call_count, 1)
        self.assertIn("connections and value sets take the cmds route", warning.call_args[0][0])
        self.assertEqual((cmds.getAttr("a.tx"), cmds.getAttr("a.ty")), (2.5, 3.5))
        self.assertEqual(self.count_steps(), 3)

    def test_the_first_set_loads_it(self):
        cmds.flushUndo()
        if cmds.pluginInfo(PLUGIN, query=True, loaded=True):
            cmds.unloadPlugin(PLUGIN)
        with mock.patch.object(cmds, "loadPlugin", wraps=cmds.loadPlugin) as load, \
                mock.patch.object(cmds, "setAttr", wraps=cmds.setAttr) as set_attr:
            Node("a").tx << 2.5
        self.assertEqual(load.call_count, 1)
        self.assertEqual(set_attr.call_count, 0)
        self.assertEqual(self.count_steps(), 1)
