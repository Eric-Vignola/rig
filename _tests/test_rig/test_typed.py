"""Features step 4: ``Typed``, the attribute that holds any data, and the generic-slot rule.

``node << Typed("mix", multi=True)`` adds ``addAttr -attributeType typed``: an
``MFnTypedAttribute`` of Maya's kAny data, not keyable unless ``keyable=True`` (D4). The rule
covers every such attribute, a *generic slot* (``choice.input`` / ``output``, an objectSet's
member slots ... included):

* as a destination it never looks at what it holds: ``slot << plug`` is one whole connect (no
  shorthand node, no per-channel split), ``slot << number`` sets a number, anything else is a
  TypeError before any edit; a multi root keeps the multi rules (an item per element, an
  append, multi to multi), every item checked first;
* ``slot << None`` disconnects it and drops the data the old input left, unless that is a
  number (a saved file refuses it when opened again), a journaled API edit;
* as a source it reads as what it holds (a matrix builds the shorthand's decomposeMatrix),
  compound data (a double3) is connected whole;
* ``root >> None`` is the per-element list; a clone is a Typed; ``serialize`` keeps the numbers.

* `TestDeclaration`, `TestRedeclaration`: the attribute, the keyable default, the refused flags,
  List, ``>>``; the kinds a re-declaration matches and the text it names.
* `TestConnectEachKind`, `TestRewireMatrix`: each source kind into an element and a single slot,
  and every rewire between kinds, keyable and not: one connect, no node made.
* `TestValues`, `TestRootForms`: numbers, refusals with nothing changed, the root forms.
* `TestNoneDrop`: the data drop, .ma / .mb reopen with no ``Error:`` line, undo / redo, locks.
* `TestUndo`, `TestSources`, `TestClone`, `TestContainers`, `TestSerialize`, `TestChoice`.
"""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

import rig
from rig import Float, InjectionError, List, Node, Plug, Typed, container, set_options
from rig.spec import Matrix, Message, String, Vector, lock
from rig._internal import api_edits
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, scene_connections


def _kind(plug):
    """(attributeType, dataType, MFnTypedAttribute data type) of ``plug``'s attribute."""
    node, _, attr = plug.partition(".")
    attr = attr.split("[")[0]
    at = cmds.addAttr(f"{node}.{attr}", query=True, attributeType=True)
    dt = cmds.addAttr(f"{node}.{attr}", query=True, dataType=True)
    sel = OpenMaya.MSelectionList()
    sel.add(f"{node}.{attr}")
    mobject = sel.getPlug(0).attribute()
    return at, dt, OpenMaya.MFnTypedAttribute(mobject).attrType()


def _inputs(plug):
    return cmds.listConnections(plug, source=True, destination=False, plugs=True) or []


def _indices(plug):
    return cmds.getAttr(plug, multiIndices=True) or []


def _nodes():
    return set(cmds.ls())


def _empty_queue():
    return cmds.undoInfo(query=True, undoQueueEmpty=True)


def _mplug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


class _Errors:
    """The ``Error:`` lines Maya prints while active (``MCommandMessage``)."""

    def __enter__(self):
        self.lines = []
        self._callback = OpenMaya.MCommandMessage.addCommandOutputCallback(self._on_output)
        return self

    def _on_output(self, message, kind, *_args):
        if kind == OpenMaya.MCommandMessage.kError:
            self.lines.append(message.strip())

    def __exit__(self, *exc):
        OpenMaya.MMessage.removeCallback(self._callback)
        return False


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.net  = Node.create("network", name="net")
        self.objA = Node.create("transform", name="objA")
        self.objB = Node.create("transform", name="objB")
        self.objC = Node.create("transform", name="objC")
        self.objA.t << [1, 2, 3]
        self.objA.rx << 30
        cube = cmds.polyCube(name="cube", constructionHistory=False)[0]
        self.shape = Node(cmds.listRelatives(cube, shapes=True)[0])
        crv = cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0)])
        self.crv_shape = Node(cmds.rename(cmds.listRelatives(crv, shapes=True)[0], "crvShape"))
        cmds.flushUndo()

    def assert_unchanged(self, call, error, *texts):
        """``call()`` raises ``error`` holding every one of ``texts``, with the scene (nodes,
        connections, values read by ``scene_state``) and the multi indices as they were, and
        nothing put on the undo queue."""
        cmds.flushUndo()
        indices = {p: _indices(p) for p in ("net.mix",) if cmds.objExists(p)}
        before  = self.scene_state()
        with self.assertRaises(error) as ctx:
            call()
        for text in texts:
            self.assertIn(text, str(ctx.exception))
        self.assertEqual(self.scene_state(), before)
        self.assertEqual({p: _indices(p) for p in indices}, indices)
        self.assertTrue(_empty_queue())
        return ctx.exception


# --------------------------------------------------------------------------- declaration


class TestDeclaration(_Case):
    def test_the_attribute_maya_makes(self):
        mix = self.net << Typed("mix", multi=True)
        self.assertIsInstance(mix, Plug)
        self.assertEqual(str(mix), "net.mix")
        self.assertIs(mix.get_node(), self.net)
        self.assertEqual(_kind("net.mix"), ("typed", ["Tdata"], OpenMaya.MFnData.kAny))
        self.assertTrue(cmds.addAttr("net.mix", query=True, multi=True))
        one = self.net << Typed("one")
        self.assertEqual(str(one), "net.one")
        self.assertEqual(_kind("net.one"), ("typed", ["Tdata"], OpenMaya.MFnData.kAny))
        self.assertFalse(cmds.addAttr("net.one", query=True, multi=True))

    def test_it_is_not_keyable_unless_asked(self):
        self.net << Typed("one") << Typed("mix", multi=True)
        self.assertFalse(cmds.getAttr("net.one", keyable=True))
        self.assertFalse(cmds.addAttr("net.mix", query=True, keyable=True))
        self.net << Typed("k", keyable=True) << Typed("k2", k=True)
        self.assertTrue(cmds.getAttr("net.k", keyable=True))
        self.assertTrue(cmds.getAttr("net.k2", keyable=True))
        # Maya's setKeyframe keys the keyable one only
        cmds.setKeyframe("net")
        self.assertTrue(cmds.listConnections("net.k", source=True, type="animCurveTU"))
        self.assertFalse(cmds.listConnections("net.one", source=True))

    def test_the_names_it_is_exported_under(self):
        self.assertIs(rig.Typed, rig.spec.Typed)
        self.assertIn("Typed", rig.__all__)
        self.assertIn("Typed", rig.spec.__all__)
        self.assertEqual(len(rig.spec.__all__), 24)

    def test_the_refused_flags(self):
        for flags in (
            {"dv": 1}, {"defaultValue": 1}, {"min": 0}, {"minValue": 0}, {"max": 1},
            {"maxValue": 1}, {"smn": 0}, {"softMinValue": 0}, {"smx": 1}, {"softMaxValue": 1},
            {"hnv": True}, {"hasMinValue": True}, {"hxv": True}, {"hasMaxValue": True},
            {"hsn": True}, {"hasSoftMinValue": True}, {"hsx": True}, {"hasSoftMaxValue": True},
            {"en": "a:b"}, {"enumName": "a:b"}, {"uac": True}, {"usedAsColor": True},
            {"dt": "matrix"}, {"dataType": "matrix"}, {"at": "double"},
            {"attributeType": "double"},
        ):
            with self.subTest(flags=flags):
                flag = next(iter(flags))
                with self.assertRaises(TypeError) as ctx:
                    Typed("bad", **flags)
                self.assertIn(f"Typed('bad'): {flag}= is no setting", str(ctx.exception))
                self.assertIn("nothing was changed", str(ctx.exception))
        self.assertFalse(cmds.attributeQuery("bad", node="net", exists=True))
        # an unknown keyword is the specs' own TypeError
        with self.assertRaises(TypeError):
            Typed("bad", update=True)

    def test_the_flags_taken(self):
        self.net << Typed("one", niceName="The One", hidden=True, category="c")
        self.assertEqual(cmds.attributeQuery("one", node="net", niceName=True), "The One")
        self.assertTrue(cmds.attributeQuery("one", node="net", hidden=True))

    def test_a_list_broadcasts(self):
        result = List([self.objA, self.objB]) << Typed("slot")
        self.assertEqual([str(p) for p in result], ["objA.slot", "objB.slot"])
        for name in ("objA.slot", "objB.slot"):
            self.assertEqual(_kind(name)[0], "typed")

    def test_an_output(self):
        out = self.net >> Typed("outT")
        self.assertEqual(str(out), "net.outT")
        self.assertFalse(cmds.attributeQuery("outT", node="net", writable=True))
        self.assertEqual(_kind("net.outT")[0], "typed")


class TestRedeclaration(_Case):
    def test_declared_again_it_keeps_its_value_and_wires(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        mix[1] << 2.5
        again = self.net << Typed("mix", multi=True)
        self.assertEqual(str(again), "net.mix")
        self.assertEqual(_inputs("net.mix[0]"), ["objA.worldMatrix"])
        self.assertEqual(cmds.getAttr("net.mix[1]"), 2.5)

    def test_the_settings_apply(self):
        self.net << Typed("one")
        self.net << Typed("one", niceName="One", keyable=True)
        self.assertEqual(cmds.attributeQuery("one", node="net", niceName=True), "One")
        self.assertTrue(cmds.getAttr("net.one", keyable=True))
        self.net << Typed("one", hidden=True)
        self.assertTrue(cmds.attributeQuery("one", node="net", hidden=True))
        # rig's own default (keyable=False) is no setting given: keyable stays
        self.net << Typed("one")
        self.assertTrue(cmds.getAttr("net.one", keyable=True))

    def test_another_kind_onto_it_raises(self):
        self.net << Typed("one") << Typed("mix", multi=True)
        for spec, text in (
            (Float("one"), "'net.one' exists as a typed (any data), not a double"),
            (String("one"), "'net.one' exists as a typed (any data), not a string"),
            (Matrix("one"), "'net.one' exists as a typed (any data), not a matrix"),
            (Float("mix", multi=True), "'net.mix' exists as a multi typed (any data), not a multi double"),
        ):
            with self.subTest(spec=type(spec).__name__):
                self.assert_unchanged(lambda: self.net << spec, TypeError, text, "overwrite=True")

    def test_it_onto_another_kind_raises(self):
        self.net << String("label") << Matrix("mtx") << Message("msgIn")
        for name, text in (
            ("label", "'net.label' exists as a string, not a typed (any data)"),
            ("mtx", "'net.mtx' exists as a matrix, not a typed (any data)"),
            ("msgIn", "'net.msgIn' exists as a message, not a typed (any data)"),
        ):
            with self.subTest(name=name):
                self.assert_unchanged(lambda: self.net << Typed(name), TypeError, text)

    def test_multi_cannot_change(self):
        self.net << Typed("one")
        self.assert_unchanged(
            lambda: self.net << Typed("one", multi=True), TypeError,
            "exists as a typed (any data), not a multi typed (any data)",
        )

    def test_overwrite_replaces_it(self):
        self.net << Typed("one")
        self.net.one << self.objA.wm
        self.net << Typed("one", overwrite=True, multi=True)
        self.assertTrue(cmds.addAttr("net.one", query=True, multi=True))
        self.assertEqual(_inputs("net.one"), [])

    def test_size_writes_numbers(self):
        mix = self.net << Typed("mix", multi=True, size=2)
        self.assertEqual(_indices("net.mix"), [0, 1])
        self.assertEqual(mix >> None, [0.0, 0.0])


# --------------------------------------------------------------------------- connects


class TestConnectEachKind(_Case):
    def _sources(self):
        mtx = self.objB << Matrix("mtx")
        tslot = self.objB << Typed("tslot")
        tslot << 4.0
        curve = cmds.createNode("animCurveTU", name="curveTU")
        cmds.setKeyframe(curve, time=1, value=2.0)
        return [
            ("a matrix attribute", mtx, "objB.mtx"),
            ("worldMatrix", self.objA.wm, "objA.worldMatrix"),
            ("offsetParentMatrix", self.objA.opm, "objA.offsetParentMatrix"),
            ("message", self.objA.message, "objA.message"),
            ("translateX", self.objA.tx, "objA.translateX"),
            ("rotateX", self.objA.rx, "objA.rotateX"),
            ("an enum", self.objA.rotateOrder, "objA.rotateOrder"),
            ("a bool", self.objA.visibility, "objA.visibility"),
            ("translate", self.objA.t, "objA.translate"),
            ("rotate", self.objA.r, "objA.rotate"),
            ("outMesh", self.shape.outMesh, "cubeShape.outMesh"),
            ("worldSpace", self.crv_shape.worldSpace, "crvShape.worldSpace"),
            ("a typed slot", tslot, "objB.tslot"),
            ("an animCurveTU", Plug("curveTU.output"), "curveTU.output"),
        ]

    def test_each_kind_into_an_element_and_a_single_slot(self):
        mix = self.net << Typed("mix", multi=True)
        for i, (label, src, name) in enumerate(self._sources()):
            single = self.net << Typed(f"one{i}")
            with self.subTest(source=label):
                before = _nodes()
                self.assertEqual(str(mix[i] << src), f"net.mix[{i}]")
                self.assertEqual(str(single << src), f"net.one{i}")
                self.assertEqual(_nodes(), before)  # no unitConversion, no shorthand node
                self.assertEqual(_inputs(f"net.mix[{i}]"), [name])
                self.assertEqual(_inputs(f"net.one{i}"), [name])

    def test_a_compound_is_never_split(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.t
        self.assertEqual(cmds.getAttr("net.mix[0]", type=True), "double3")
        self.assertEqual(_inputs("net.mix[0]"), ["objA.translate"])
        np.testing.assert_allclose(mix[0] >> None, [1, 2, 3])
        self.assertEqual(mix[0].get_data_type(), "double3")


_KINDS = ("matrix", "double3", "number", "message", "mesh", "empty")


class TestRewireMatrix(_Case):
    """Every (old kind, new kind) pair, on a non-keyable and a keyable multi: no node made,
    the connection is the new one (an empty one: none), and Maya keeps running."""

    def _source(self, kind):
        return {
            "matrix":  self.objA.wm,
            "double3": self.objA.t,
            "number":  self.objA.tx,
            "message": self.objA.message,
            "mesh":    self.shape.outMesh,
            "empty":   None,
        }[kind]

    _NAMES = {
        "matrix":  ["objA.worldMatrix"],
        "double3": ["objA.translate"],
        "number":  ["objA.translateX"],
        "message": ["objA.message"],
        "mesh":    ["cubeShape.outMesh"],
        "empty":   [],
    }

    def test_every_rewire(self):
        for keyable in (False, True):
            mix = self.net << Typed(f"mix{int(keyable)}", multi=True, keyable=keyable)
            i = 0
            for old in _KINDS:
                for new in _KINDS:
                    with self.subTest(keyable=keyable, old=old, new=new):
                        name = f"net.mix{int(keyable)}[{i}]"
                        if old != "empty":
                            mix[i] << self._source(old)
                        before = _nodes()
                        self.assertEqual(str(mix[i] << self._source(new)), name)
                        self.assertEqual(_nodes(), before)
                        self.assertEqual(_inputs(name), self._NAMES[new])
                        # the plug held across both statements and one taken fresh
                        self.assertEqual(mix[i].get_full_name(), name)
                    i += 1

    def test_a_held_element_rewired_twice(self):
        mix = self.net << Typed("mix", multi=True)
        slot = mix[0]
        slot << self.objA.wm
        slot << self.objA.t
        slot << self.objA.tx
        self.assertEqual(_inputs("net.mix[0]"), ["objA.translateX"])
        self.assertEqual(slot >> None, 1.0)


# --------------------------------------------------------------------------- values


class TestValues(_Case):
    def test_numbers_set(self):
        mix = self.net << Typed("mix", multi=True)
        for i, (value, read) in enumerate(
            ((2.5, 2.5), (3, 3.0), (True, 1.0), (np.float32(1.5), 1.5), (np.int64(7), 7.0),
             (np.bool_(False), 0.0))
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(str(mix[i] << value), f"net.mix[{i}]")
                self.assertEqual(cmds.getAttr(f"net.mix[{i}]", type=True), "typed")
                self.assertEqual(mix[i] >> None, read)
                self.assertIsInstance(mix[i] >> None, float)
        one = self.net << Typed("one")
        one << 4
        self.assertEqual(one >> None, 4.0)

    def test_a_number_over_an_input_replaces_it(self):
        one = self.net << Typed("one")
        one << self.objA.wm
        one << 2.0
        self.assertEqual(_inputs("net.one"), [])
        self.assertEqual(one >> None, 2.0)
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.t
        mix[0] << 3
        self.assertEqual(_inputs("net.mix[0]"), [])
        self.assertEqual(cmds.getAttr("net.mix[0]"), 3.0)

    def test_attribute_set_takes_a_number(self):
        one = self.net << Typed("one")
        one.set(5.5)
        self.assertEqual(cmds.getAttr("net.one"), 5.5)

    def test_any_other_value_raises_with_nothing_changed(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        one = self.net << Typed("one")
        one << self.objA.t
        for label, value, extra in (
            ("a str", "hello", "write Plug('hello')"),
            ("a plug name", "objA.tx", "write Plug('objA.tx')"),
            ("a list", [1, 2, 3], "not list [1, 2, 3]"),
            ("a tuple", (1.0,), "not tuple"),
            ("an ndarray", np.eye(4), "not ndarray"),
            ("a node", self.objB, "not Transform"),
            ("a complex", 1j, "not complex"),
        ):
            for slot, name in ((mix[0], "net.mix[0]"), (mix[5], "net.mix[5]"), (one, "net.one")):
                with self.subTest(value=label, slot=name):
                    self.assert_unchanged(
                        lambda: slot << value, TypeError,
                        f"'{name}' is a typed attribute that holds any data: it takes a plug",
                        extra, "Nothing was changed",
                    )
        self.assertEqual(_indices("net.mix"), [0])  # mix[5] was never made
        self.assertEqual(_inputs("net.mix[0]"), ["objA.worldMatrix"])
        self.assertEqual(_inputs("net.one"), ["objA.translate"])

    def test_a_root_list_with_a_bad_item_raises_before_the_first_edit(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        self.assert_unchanged(
            lambda: mix << [1, self.objA.tx, "x"], TypeError, "'net.mix[2]' is a typed attribute",
        )
        self.assert_unchanged(
            lambda: mix << List([self.objA.tx, self.objB]), TypeError, "'net.mix[1]'", "not Transform",
        )
        self.assertEqual(_inputs("net.mix[0]"), ["objA.worldMatrix"])

    def test_a_locked_slot_refuses(self):
        one = self.net << Typed("one")
        one << self.objA.wm
        one.lock()
        for value in (self.objA.t, 2.0):
            with self.subTest(value=repr(value)):
                self.assert_unchanged(
                    lambda: one << value, InjectionError, "Plug('net.one').unlock()",
                )


class TestRootForms(_Case):
    def test_a_list_goes_item_by_item(self):
        mix = self.net << Typed("mix", multi=True)
        mix << List([self.objA.wm, self.objA.message, self.objA.t])
        self.assertEqual(
            [_inputs(f"net.mix[{i}]") for i in range(3)],
            [["objA.worldMatrix"], ["objA.message"], ["objA.translate"]],
        )

    def test_numbers_none_and_plugs(self):
        mix = self.net << Typed("mix", multi=True)
        mix[1] << self.objA.wm
        mix << [1, None, self.objA.tx]
        self.assertEqual(cmds.getAttr("net.mix[0]"), 1.0)
        self.assertEqual(_inputs("net.mix[1]"), [])
        self.assertEqual(cmds.getAttr("net.mix[1]", type=True), "Tdata")  # the matrix dropped
        self.assertEqual(_inputs("net.mix[2]"), ["objA.translateX"])
        mix << np.array([4.0, 5.0])
        self.assertEqual([cmds.getAttr(f"net.mix[{i}]") for i in (0, 1)], [4.0, 5.0])

    def test_a_root_appends(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << 1.0
        mix[1] << self.objA.t
        self.assertEqual(str(mix << self.objB.wm), "net.mix")
        self.assertEqual(_inputs("net.mix[2]"), ["objB.worldMatrix"])
        mix << 2.5
        self.assertEqual(cmds.getAttr("net.mix[3]"), 2.5)
        mix << self.objA.message
        self.assertEqual(_inputs("net.mix[4]"), ["objA.message"])
        self.assertEqual(_inputs("net.mix[0]"), [])  # never rewired
        self.assertEqual(_indices("net.mix"), [0, 1, 2, 3, 4])

    def test_the_same_edits_as_an_explicit_index(self):
        mix = self.net << Typed("mix", multi=True)
        cmds.flushUndo()
        mix << self.objA.t
        mix[1] << self.objA.t
        self.assertEqual(_inputs("net.mix[0]"), _inputs("net.mix[1]"))

    def test_a_multi_into_a_multi(self):
        src = self.objB << Typed("src", multi=True)
        src[1] << self.objA.wm
        src[4] << self.objA.t
        src[6] << 2.0
        mix = self.net << Typed("mix", multi=True)
        mix << src
        self.assertEqual(_indices("net.mix"), [1, 4, 6])
        self.assertEqual(_inputs("net.mix[1]"), ["objB.src[1]"])
        self.assertEqual(_inputs("net.mix[4]"), ["objB.src[4]"])
        self.assertEqual(_inputs("net.mix[6]"), ["objB.src[6]"])

    def test_the_root_reads_as_a_list_in_index_order(self):
        mix = self.net << Typed("mix", multi=True)
        mix[3] << self.objA.t
        mix[0] << self.objA.wm
        mix[1] << 2.0
        mix[7] << self.objA.message
        values = mix >> None
        self.assertEqual(len(values), 4)
        np.testing.assert_allclose(values[0], np.reshape(cmds.getAttr("objA.worldMatrix[0]"), (4, 4)))
        self.assertEqual(values[1], 2.0)
        np.testing.assert_allclose(values[2], [1, 2, 3])
        self.assertIsNone(values[3])
        self.assertEqual(len(mix.get()), 4)
        self.assertEqual(mix.get_data_type(), "compound")
        self.assertEqual([mix[i].get_data_type() for i in (0, 1, 3, 7)], ["matrix", "typed", "double3", "Tdata"])


# --------------------------------------------------------------------------- << None


class TestNoneDrop(_Case):
    def test_the_data_an_input_left_is_dropped(self):
        mix = self.net << Typed("mix", multi=True)
        sources = [self.objA.wm, self.objA.t, self.shape.outMesh, self.objA.tx, self.objA.message]
        for i, src in enumerate(sources):
            mix[i] << src
        for i in range(len(sources)):
            mix[i] << None
        self.assertEqual(
            [cmds.getAttr(f"net.mix[{i}]", type=True) for i in range(len(sources))],
            ["Tdata", "Tdata", "Tdata", "typed", "Tdata"],  # a number is kept
        )
        self.assertEqual(cmds.getAttr("net.mix[3]"), 1.0)
        self.assertEqual([_inputs(f"net.mix[{i}]") for i in range(len(sources))], [[]] * 5)

    def test_a_single_slot_too(self):
        one = self.net << Typed("one")
        one << self.objA.wm
        self.assertIs(one << None, one)
        self.assertEqual(cmds.getAttr("net.one", type=True), "Tdata")

    def test_a_slot_never_made_stays_unmade(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << 1.0
        mix[3] << None
        self.assertEqual(_indices("net.mix"), [0])

    def test_a_disconnect_outside_rig_keeps_the_data(self):
        # Maya's own behaviour, which rig cannot see (documented)
        one = self.net << Typed("one")
        one << self.objA.wm
        cmds.disconnectAttr("objA.worldMatrix[0]", "net.one")
        self.assertEqual(cmds.getAttr("net.one", type=True), "matrix")
        one << None  # rig's own drop takes it
        self.assertEqual(cmds.getAttr("net.one", type=True), "Tdata")

    def test_undo_and_redo(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        cmds.flushUndo()
        connected = self.scene_state()
        mix[0] << None
        dropped = self.scene_state()
        self.assertEqual(cmds.getAttr("net.mix[0]", type=True), "Tdata")
        # re-pinned by the review fix (two steps at s4): the disconnect and the drop are
        # one step, so one undo never leaves the slot disconnected with the stale data
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(), connected)
        self.assertEqual(cmds.getAttr("net.mix[0]", type=True), "matrix")
        self.assertEqual(_inputs("net.mix[0]"), ["objA.worldMatrix"])
        self.assertTrue(_empty_queue())
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(), dropped)
        self.assertEqual(cmds.getAttr("net.mix[0]", type=True), "Tdata")
        self.assertEqual(_inputs("net.mix[0]"), [])

    def test_a_locked_slot_is_disconnected_and_dropped_and_stays_locked(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        mix[1] << self.objA.wm
        one = self.net << Typed("one")
        one << self.objA.wm
        mix[0].lock()
        one.lock()
        mix[0] << None
        one << None
        for name in ("net.mix[0]", "net.one"):
            with self.subTest(slot=name):
                self.assertEqual(_inputs(name), [])
                self.assertEqual(cmds.getAttr(name, type=True), "Tdata")
                self.assertTrue(cmds.getAttr(name, lock=True))
        # a locked root locks its elements: they are dropped too, the root's own lock kept
        cmds.setAttr("net.mix", lock=True)
        mix[1] << None
        self.assertEqual(cmds.getAttr("net.mix[1]", type=True), "Tdata")
        self.assertTrue(cmds.getAttr("net.mix", lock=True))
        cmds.setAttr("net.mix", lock=False)
        self.assertFalse(_mplug("net.mix[1]").isLocked)  # no lock of its own was made

    def test_without_rigs_undo_command_the_data_stays(self):
        one = self.net << Typed("one")
        one << self.objA.wm
        api_edits._bind()
        with mock.patch.object(api_edits, "_ready", lambda: False):
            one << None
        self.assertEqual(_inputs("net.one"), [])
        self.assertEqual(cmds.getAttr("net.one", type=True), "matrix")

    def _build_and_save(self, folder, ext, kind, drop):
        """A network whose slots held a matrix, a double3, a mesh (each dropped by rig's
        ``<< None`` when ``drop``, else disconnected by cmds), a number, and a connected
        matrix; saved to ``folder``."""
        cmds.file(new=True, force=True)
        net  = Node.create("network", name="net")
        objA = Node.create("transform", name="objA")
        cube = cmds.polyCube(name="cube", constructionHistory=False)[0]
        shape = Node(cmds.listRelatives(cube, shapes=True)[0])
        mix = net << Typed("mix", multi=True)
        one = net << Typed("one")
        mix[0] << objA.wm
        mix[1] << objA.t
        mix[2] << shape.outMesh
        mix[3] << 2.5
        mix[4] << objA.wm   # stays connected
        one << objA.wm
        for slot in (mix[0], mix[1], mix[2], one):
            if drop:
                slot << None
            else:
                cmds.disconnectAttr(_inputs(slot.get_full_name())[0], slot.get_full_name())
        path = os.path.join(folder, f"typed_{int(drop)}{ext}")
        cmds.file(rename=path)
        cmds.file(save=True, type=kind, force=True)
        return path

    def test_saved_files_open_with_no_error(self):
        folder = tempfile.mkdtemp(prefix="rig_typed_")
        try:
            for ext, kind in ((".ma", "mayaAscii"), (".mb", "mayaBinary")):
                for drop in (True, False):
                    with self.subTest(file=ext, drop=drop):
                        path = self._build_and_save(folder, ext, kind, drop)
                        if ext == ".ma":
                            with open(path) as fh:
                                network = fh.read().split('createNode network -n "net"')[1]
                            network = network.split("createNode")[0]
                            self.assertEqual('-type "matrix"' in network, not drop)
                        cmds.file(new=True, force=True)
                        with _Errors() as errors:
                            cmds.file(path, open=True, force=True)
                        kinds = [cmds.getAttr(f"net.mix[{i}]", type=True) for i in range(5)]
                        if drop:
                            self.assertEqual(errors.lines, [])
                            self.assertEqual(kinds, ["Tdata", "Tdata", "Tdata", "typed", "matrix"])
                            self.assertEqual(cmds.getAttr("net.one", type=True), "Tdata")
                            self.assertEqual(cmds.getAttr("net.mix[3]"), 2.5)
                        elif ext == ".ma":
                            # the negative control: Maya's own disconnect keeps the data,
                            # and the .ma refuses it when opened (the slots come back empty)
                            self.assertEqual(len(errors.lines), 4, errors.lines)
                            self.assertIn("does not accept data of type 'matrix'", errors.lines[0])
                            self.assertEqual(kinds[:3], ["Tdata", "Tdata", "Tdata"])
                        else:
                            # a .mb keeps it, with no error
                            self.assertEqual(errors.lines, [])
                            self.assertEqual(kinds[:2], ["matrix", "double3"])
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


# --------------------------------------------------------------------------- undo


class TestUndo(_Case):
    def test_a_list_into_a_root_is_one_step(self):
        mix = self.net << Typed("mix", multi=True)
        cmds.flushUndo()
        before = self.scene_state()
        mix << List([self.objA.wm, self.objA.message, self.objA.t, self.objA.tx])
        after = self.scene_state()
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(), before)
        self.assertTrue(_empty_queue())
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(), after)

    def test_a_number_is_one_step(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << 1.0
        cmds.flushUndo()
        mix[0] << 2.0
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("net.mix[0]"), 1.0)
        self.assertTrue(_empty_queue())

    def test_a_rewire_takes_the_steps_of_a_float_multi(self):
        def steps(make):
            cmds.file(new=True, force=True)
            node = Node.create("network", name="n")
            a = Node.create("transform", name="a")
            slot = (node << make())[0]
            slot << a.tx
            cmds.flushUndo()
            slot << a.ty
            count = 0
            while not _empty_queue():
                cmds.undo()
                count += 1
            return count, _inputs("n.w[0]")

        self.assertEqual(
            steps(lambda: Typed("w", multi=True)),
            steps(lambda: Float("w", multi=True, keyable=False)),
        )


# --------------------------------------------------------------------------- sources


class TestSources(_Case):
    def test_a_double3_slot_into_matrix_scalar_and_compound(self):
        one = self.net << Typed("one")
        one << self.objA.t
        for dst, name in (
            (self.objC.opm, "objC.offsetParentMatrix"),
            (self.objC.tx, "objC.translateX"),
            (self.objB.r, "objB.rotate"),
        ):
            with self.subTest(dst=name):
                before = _nodes()
                dst << one  # no TypeError: connected whole, Maya decides
                self.assertEqual(_inputs(name), ["net.one"])
                self.assertEqual(_nodes(), before)

    def test_a_matrix_slot_reads_as_a_matrix(self):
        one = self.net << Typed("one")
        one << self.objA.wm
        before = _nodes()
        self.objC.r << one
        made = _nodes() - before
        self.assertEqual([cmds.nodeType(n) for n in made], ["decomposeMatrix"])
        self.assertEqual(_inputs("objC.rotate"), [f"{made.pop()}.outputRotate"])

    def test_a_number_slot_broadcasts(self):
        one = self.net << Typed("one")
        one << 2.0
        self.objC.s << one
        self.assertEqual([_inputs(f"objC.scale{a}") for a in "XYZ"], [["net.one"]] * 3)
        self.assertEqual(cmds.getAttr("objC.s"), [(2.0, 2.0, 2.0)])

    def test_a_slot_into_a_slot(self):
        one = self.net << Typed("one")
        two = self.net << Typed("two")
        one << self.objA.t
        two << one
        self.assertEqual(_inputs("net.two"), ["net.one"])
        np.testing.assert_allclose(two >> None, [1, 2, 3])


# --------------------------------------------------------------------------- clone


class TestClone(_Case):
    def test_the_root_clones_as_a_typed_multi(self):
        mix = self.net << Typed("mix", multi=True)
        mix[0] << self.objA.wm
        mix[2] << 4.0
        mix[3] << self.objA.t
        other = Node.create("network", name="other")
        cloned = mix >> other
        self.assertEqual(str(cloned), "other.mix")
        self.assertEqual(_kind("other.mix"), ("typed", ["Tdata"], OpenMaya.MFnData.kAny))
        self.assertTrue(cmds.addAttr("other.mix", query=True, multi=True))
        # the numbers are mirrored; a clone never connects
        self.assertEqual(cmds.getAttr("other.mix[2]"), 4.0)
        self.assertEqual(cmds.listConnections("other", source=True, destination=False) or [], [])

    def test_an_element_clones_as_a_single_typed(self):
        mix = self.net << Typed("mix", multi=True)
        mix[3] << self.objA.t
        cloned = mix[3] >> "objB.slot"
        self.assertEqual(str(cloned), "objB.slot")
        self.assertEqual(_kind("objB.slot"), ("typed", ["Tdata"], OpenMaya.MFnData.kAny))
        self.assertFalse(cmds.addAttr("objB.slot", query=True, multi=True))


# --------------------------------------------------------------------------- containers


class TestContainers(_Case):
    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True, native_multi_publish=False)

    def test_a_single_slot_publishes_natively(self):
        with container("box"):
            inner = Node.create("network", name="inner")
            slot = inner << Typed("slot")
            container.publish_input(slot, "slotIn")
        self.assertIn("slotIn", cmds.container("box", query=True, publishName=True) or [])
        Node("box").slotIn << self.objA.wm
        self.assertEqual(cmds.getAttr("inner.slot", type=True), "matrix")

    def test_a_multi_stays_on_its_node(self):
        with container("box"):
            inner = Node.create("network", name="inner")
            mix = inner << Typed("mix", multi=True)
            container.publish_input(mix, "mixIn")
        self.assertNotIn("mixIn", cmds.container("box", query=True, publishName=True) or [])

    def test_an_external_slot_clones_as_typed(self):
        one = self.net << Typed("one")
        one << 3.0
        with container("box"):
            published = container.publish_input(one, "oneIn")
        self.assertEqual(_kind(str(published))[0], "typed")


# --------------------------------------------------------------------------- serialize


class TestSerialize(_Case):
    def test_the_round_trip_keeps_the_numbers(self):
        xf = Node.create("transform", name="ser")
        mix = xf << Typed("mix", multi=True)
        mix[1] << 2.0
        mix[2] << self.objA.wm
        mix[4] << 7.0
        xf << Typed("one") << 5.0
        xf << Typed("two", keyable=True)
        data = xf.serialize()
        attrs = data.user_defined_attributes
        self.assertEqual(attrs["mix"], {
            "attributeType": "typed", "multi": True, "value": [[1, 2.0], [4, 7.0]],
            "keyable": False, "channel_box": False,
        })
        self.assertEqual(attrs["one"], {
            "attributeType": "typed", "value": 5.0, "keyable": False, "channel_box": False,
        })
        self.assertEqual(attrs["two"], {
            "attributeType": "typed", "keyable": True, "channel_box": False,
        })
        from rig.nodetypes.transform import Transform

        Transform._create_user_attrs(Node.create("transform", name="ser2"), attrs)
        for name in ("mix", "one", "two"):
            self.assertEqual(_kind(f"ser2.{name}")[0], "typed")
        self.assertEqual(_indices("ser2.mix"), [1, 4])
        self.assertEqual([cmds.getAttr(f"ser2.mix[{i}]") for i in (1, 4)], [2.0, 7.0])
        self.assertEqual(cmds.getAttr("ser2.one"), 5.0)
        self.assertTrue(cmds.getAttr("ser2.two", keyable=True))


# --------------------------------------------------------------------------- choice


class TestChoice(_Case):
    def test_its_slots_take_the_rule(self):
        ch = Node.create("choice", name="ch")
        before = _nodes()
        ch.input[0] << self.objA.wm
        ch.input[0] << self.objA.t           # 858ceab: a composeMatrix
        ch.input[1] << self.objA.t
        ch.input[1] << self.objA.wm          # 858ceab: 'Plug is not a compound parent'
        ch.input[2] << 2.0                   # 858ceab: setAttr's type 'Tdata' refused
        self.assertEqual(_nodes(), before)
        self.assertEqual(_inputs("ch.input[0]"), ["objA.translate"])
        self.assertEqual(_inputs("ch.input[1]"), ["objA.worldMatrix"])
        self.assertEqual(cmds.getAttr("ch.input[2]"), 2.0)
        ch.selector << 1
        self.objC.opm << ch.output            # what is selected: a matrix
        self.assertEqual(_inputs("objC.offsetParentMatrix"), ["ch.output"])

    def test_functions_choice_is_unchanged(self):
        out = rig.functions.choice([self.objA.t, self.objB.t], self.objA.tx)
        self.assertEqual(cmds.nodeType(str(out).partition(".")[0]), "choice")


if __name__ == "__main__":
    import unittest

    unittest.main()
