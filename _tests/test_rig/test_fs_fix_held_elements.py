"""Review fix (features run): an element plug held across Maya's undo or redo.

Step 1 made rig find an element's MPlug again after each disconnect rig makes. Maya's undo
and redo remove connections too, and rig did not see those: ``p = n.w[0]; p << a.tx;
cmds.undo(); p << b.tx`` used the freed MPlug of ``p``, an access violation in mayapy, on a
non-keyable dynamic multi (``Float`` / ``Vector`` with ``keyable=False``), a ``Typed`` multi
(``keyable=False`` by default, D4) or after ``<< hide``; also through ``p >> None`` or
``p.plug``, on the API route and the cmds one (review_undo_speed probe_p0_undo: 8 of 14 Typed
runs, 7 of 14 Float, 6 of 14 Vector crashed at 8286cb8; Float and Vector on 858ceab too).

rig now keeps every element Attribute it hands out (and every plug below one) by a weak
reference, with the path that finds its MPlug again from its top plug, and the Undo / Redo
event callbacks give each one a fresh MPlug (``rig.nodetypes._base._HELD_ELEMENTS``).

Every test of `TestHeldAcrossUndo` crashed mayapy at 8286cb8 or failed, except the guards
(named so in their docstrings). Run them one per process on the old tree.
"""

import gc
from unittest import mock

from maya import cmds

from rig import Float, List, Matrix, Message, Node, Plug, Typed, Vector, hide
from rig._internal import journal
from rig.nodetypes import _base
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _setup(kind):
    """``(n, s1, s2, name1, name2)`` for a multi of ``kind`` on node ``n``; the undo queue
    is flushed after it."""
    n = Node.create("network", name="n")
    a = Node.create("transform", name="a")
    b = Node.create("transform", name="b")
    if kind == "Float":
        n << Float("w", multi=True, keyable=False)
        srcs = (a.tx, b.tx, "a.translateX", "b.translateX")
    elif kind == "Vector":
        n << Vector("w", multi=True, keyable=False)
        srcs = (a.t, b.t, "a.translate", "b.translate")
    elif kind == "Typed":
        n << Typed("w", multi=True)
        srcs = (a.wm, b.wm, "a.worldMatrix", "b.worldMatrix")
    elif kind == "Matrix":
        n << Matrix("w", multi=True)
        srcs = (a.wm, b.wm, "a.worldMatrix", "b.worldMatrix")
    elif kind == "Message":
        n << Message("w", multi=True)
        srcs = (a.message, b.message, "a.message", "b.message")
    elif kind == "hide":
        n << Float("w", multi=True)
        n.w << hide
        srcs = (a.tx, b.tx, "a.translateX", "b.translateX")
    else:
        raise ValueError(kind)
    cmds.flushUndo()
    return (n,) + srcs


def _input(name="n.w[0]"):
    found = cmds.listConnections(name, source=True, destination=False, plugs=True) or []
    return [x.split("[")[0] for x in found]


KINDS = ("Float", "Vector", "Typed", "Matrix", "Message", "hide")


class TestHeldAcrossUndo(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def each_route(self, body):
        for api in (True, False):
            with self.subTest(api=api):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                with mock.patch.object(journal, "API_EDITS", api):
                    body()

    def each_kind(self, body):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.each_route(lambda kind=kind: body(*_setup(kind)))

    def test_undo_then_rewire(self):
        """``n.w[0] << a; held = n.w[0]; cmds.undo(); held << b``: b drives the element."""
        def body(n, s1, s2, name1, name2):
            n.w[0] << s1
            held = n.w[0]
            held.get_mplug().isDestination
            cmds.undo()
            self.assertEqual(_input(), [])
            held << s2
            self.assertEqual(_input(), [name2])
            self.assertEqual(held.get_mplug().logicalIndex(), 0)
        self.each_kind(body)

    def test_undo_then_none_and_get(self):
        """After the undo, ``held >> None``, ``held.plug`` and ``held << None`` work."""
        def body(n, s1, s2, name1, name2):
            n.w[0] << s1
            held = n.w[0]
            held.get_mplug().isDestination
            cmds.undo()
            if name1 != "a.message":  # a message plug has no value to read
                held >> None
            self.assertFalse(held.get_mplug().isDestination)
            self.assertEqual(held.get_mplug().name(), "n.w[0]")
            held << None
            held << s1
            self.assertEqual(_input(), [name1])
        self.each_kind(body)

    def test_undo_then_value(self):
        """A number over the element after the undo (Float and Typed take numbers)."""
        def body(kind):
            n, s1, _s2, _n1, _n2 = _setup(kind)
            n.w[0] << (s1 if kind == "Float" else Node("a").tx)
            held = n.w[0]
            held.get_mplug().isDestination
            cmds.undo()
            held << 3.0
            self.assertEqual(cmds.getAttr("n.w[0]"), 3.0)
        for kind in ("Float", "Typed"):
            with self.subTest(kind=kind):
                self.each_route(lambda kind=kind: body(kind))

    def test_redo_walk_then_rewire(self):
        """Two statements, undone twice, redone twice, undone once, then a rewire on the
        plug object held all along."""
        def body(n, s1, s2, name1, name2):
            n.w[0] << s1
            n.w[0] << s2
            held = n.w[0]
            held.get_mplug().isDestination
            cmds.undo()
            cmds.undo()
            cmds.redo()
            cmds.redo()
            cmds.undo()
            self.assertEqual(_input(), [name1])
            held << s1
            self.assertEqual(_input(), [name1])
            held << s2
            self.assertEqual(_input(), [name2])
        self.each_kind(body)

    def test_a_child_below_an_element(self):
        """guard (8286cb8 survived it): ``n.w[0].wX`` of a Vector multi, held across the
        undo of its connect."""
        def body():
            n, _s1, _s2, _n1, _n2 = _setup("Vector")
            a = Node("a")
            n.w[0].wX << a.tx
            held = n.w[0].wX
            held.get_mplug().isDestination
            cmds.undo()
            held << a.ty
            self.assertEqual(_input("n.w[0].wX"), ["a.translateY"])
        self.each_route(body)

    def test_a_list_and_find_attr(self):
        """guard (8286cb8 survived it): the elements of a held List, and an element
        ``find_attr`` returned."""
        def body():
            n, s1, s2, name1, name2 = _setup("Typed")
            items = List([n.w[0], n.w[1]])
            items << [s1, s2]
            found = n.find_attr("w[2]")  # an Attribute
            s1.connect(found)
            for x in list(items) + [found]:
                x.get_mplug().isDestination
            cmds.undo()
            cmds.undo()
            self.assertFalse(found.get_mplug().isDestination)
            items << [s2, s1]
            Plug(found) << s2
            self.assertEqual(_input("n.w[0]"), [name2])
            self.assertEqual(_input("n.w[1]"), [name1])
            self.assertEqual(_input("n.w[2]"), [name2])
        self.each_route(body)

    def test_a_static_generic_slot(self):
        """guard: ``choice.input[0]`` (a static generic slot) held across the undo."""
        def body():
            ch = Node.create("choice", name="ch")
            a = Node.create("transform", name="a")
            b = Node.create("transform", name="b")
            cmds.flushUndo()
            ch.input[0] << a.wm
            held = ch.input[0]
            held.get_mplug().isDestination
            cmds.undo()
            held << b.wm
            self.assertEqual(_input("ch.input[0]"), ["b.worldMatrix"])
        self.each_route(body)

    def test_a_deleted_node_undone(self):
        """A held element of a node deleted (its connection removed with it) and brought
        back by undo works once the undo brings the node back."""
        def body():
            n, s1, s2, name1, name2 = _setup("Float")
            n.w[0] << s1
            held = n.w[0]
            held.get_mplug().isDestination
            cmds.delete("n")
            cmds.undo()
            self.assertEqual(_input(), [name1])
            held << s2
            self.assertEqual(_input(), [name2])
        self.each_route(body)

    def test_the_registry_holds_no_dead_plug(self):
        """An element plug dropped leaves no entry; a plug that is no element makes none."""
        n, _s1, _s2, _n1, _n2 = _setup("Float")
        gc.collect()
        before = len(_base._HELD_ELEMENTS)
        held = [n.w[i] for i in range(50)]
        self.assertEqual(len(_base._HELD_ELEMENTS), before + 50)
        n.w, Node("a").t, Node("a").tx, Node("a").wm  # no element: nothing kept
        self.assertEqual(len(_base._HELD_ELEMENTS), before + 50)
        del held
        gc.collect()
        self.assertEqual(len(_base._HELD_ELEMENTS), before)
