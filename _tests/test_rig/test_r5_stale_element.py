"""Features step 1 (P0): rewiring an element of a non-keyable dynamic multi.

An element MPlug taken while the element is connected, and holding no value of its own,
points at freed memory once that connection is removed (an element of a dynamic multi that
is not keyable: ``Float`` / ``Matrix`` / ``Message`` / ``Vector`` with ``keyable=False``, or
after ``<< hide``). In 2.0.0a5 ``n.w[0] << a.tx; n.w[0] << a.ty`` took the element's MPlug,
disconnected ``a.tx``, then connected on that MPlug: an access violation in
``MPlug::node``, on the API route and the cmds one (``journal.API_EDITS = False``). rig now
finds the element's MPlug again from its top plug after each disconnect it makes
(``_element_path`` / ``_rebind_element``: ``<<``, ``Attribute.disconnect``,
``Attribute.break_connections``).

Every test here crashed mayapy on 858ceab (run them through
``rig_perf\\tools\\run_rig_tests.py`` with the sweeper running) or failed
(``test_matrix_held_connected``), except the guards: the ``*_held`` ones (a plug taken
before the first connect never dangled) and ``test_a_child_of_an_element``.

* `TestRewireAnElement`: each multi kind rewired in statements, with the Plug object held
  from before the first connect, held from while the element is connected, or taken fresh
  each time; through ``API_EDITS`` True and False. The new connection holds, the plug
  object stays usable, and undo / redo walk the statements back and forth.
* `TestOtherEdits`: a value set over the connection, ``<< None`` then a rewire,
  ``Attribute.disconnect`` and ``break_connections`` then a rewire on the plug held across
  them (and on the attrs ``break_connections`` returns).
"""

from unittest import mock

from maya import cmds

import rig
from rig import Float, Matrix, Message, Node, Vector, hide
from rig._internal import journal
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _setup(kind):
    """The scene for one multi kind: ``(n, [source plugs], [their names], attr)``; the undo
    queue is flushed after it."""
    a = Node.create("transform", name="objA")
    b = Node.create("transform", name="objB")
    n = Node.create("network", name="net")
    if kind == "float":
        n << Float("w", multi=True, keyable=False)
        srcs, names = [a.tx, a.ty], ["objA.translateX", "objA.translateY"]
    elif kind == "matrix":
        n << Matrix("w", multi=True, keyable=False)
        srcs, names = [a.matrix, b.matrix], ["objA.matrix", "objB.matrix"]
    elif kind == "message":
        n << Message("w", multi=True, keyable=False)
        srcs, names = [a.message, b.message], ["objA.message", "objB.message"]
    elif kind == "vector":
        n << Vector("w", multi=True, keyable=False)
        srcs, names = [a.t, b.t], ["objA.translate", "objB.translate"]
    elif kind == "hide":
        n << Float("w", multi=True)
        n.w << hide
        srcs, names = [a.tx, a.ty], ["objA.translateX", "objA.translateY"]
    else:
        raise ValueError(kind)
    cmds.flushUndo()
    return n, srcs, names


def _inputs(name="net.w[0]"):
    return cmds.listConnections(name, source=True, destination=False, plugs=True) or []


KINDS = ("float", "matrix", "message", "vector", "hide")


class TestRewireAnElement(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def rewire(self, kind, how, api):
        """Three statements, each one undo chunk: ``w[0] << s1``, ``w[0] << s2``,
        ``w[0] << s1``. ``how``: ``held`` (one Plug object from before the first connect),
        ``held_connected`` (one Plug object taken after the first connect, while the element
        is connected), ``fresh`` (``n.w[0]`` each time)."""
        with mock.patch.object(journal, "API_EDITS", api):
            n, (s1, s2), (name1, name2) = _setup(kind)
            slot = n.w[0] if how == "held" else None
            state0 = self.scene_state()

            def dst():
                return n.w[0] if slot is None else slot

            with rig.undo_chunk("first"):
                dst() << s1
            self.assertEqual(_inputs(), [name1])
            state1 = self.scene_state()
            if how == "held_connected":
                slot = n.w[0]

            with rig.undo_chunk("second"):
                dst() << s2
            self.assertEqual(_inputs(), [name2])
            state2 = self.scene_state()
            # the plug object is alive, and names its element
            self.assertEqual(dst().plug.name(), "net.w[0]")
            self.assertEqual(dst().plug.logicalIndex(), 0)
            self.assertEqual(cmds.getAttr("net.w", multiIndices=True), [0])

            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state1, "undo second")
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo first")
            self.redo_steps(2)
            self.assert_state_equal(self.scene_state(), state2, "redo both")

            # a third statement on the same (or a fresh) plug object after the walk
            with rig.undo_chunk("third"):
                dst() << s1
            self.assertEqual(_inputs(), [name1])
            state3 = self.scene_state()
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state2, "undo third")
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(), state3, "redo third")
            self.assertEqual(dst().plug.name(), "net.w[0]")

    def check(self, kind, how):
        for api in (True, False):
            with self.subTest(api=api):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                self.rewire(kind, how, api)


def _add_rewire_tests():
    for kind in KINDS:
        for how in ("held", "held_connected", "fresh"):
            def test(self, kind=kind, how=how):
                self.check(kind, how)
            test.__name__ = f"test_{kind}_{how}"
            guard = "guard (a plug taken before the first connect never dangled): " if how == "held" else ""
            test.__doc__ = f"{guard}A {kind} multi element rewired, the plug {how.replace('_', ' ')}."
            setattr(TestRewireAnElement, test.__name__, test)


_add_rewire_tests()


class TestOtherEdits(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def each_route(self, body):
        """``body()`` in a new scene, with ``API_EDITS`` True, then False."""
        for api in (True, False):
            with self.subTest(api=api):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                with mock.patch.object(journal, "API_EDITS", api):
                    body()

    def test_a_value_over_the_connection(self):
        """``w[0] << a.tx; w[0] << 2.5``: the value replaces the input; then a rewire."""
        def body():
            n, (s1, s2), _names = _setup("float")
            n.w[0] << s1
            n.w[0] << 2.5
            self.assertEqual(_inputs(), [])
            self.assertEqual(cmds.getAttr("net.w[0]"), 2.5)
            n.w[0] << s2
            self.assertEqual(_inputs(), ["objA.translateY"])
        self.each_route(body)

    def test_none_then_a_rewire(self):
        """``slot << None`` on the plug held while connected, then ``slot << a.ty``."""
        def body():
            n, (s1, s2), _names = _setup("float")
            n.w[0] << s1
            slot = n.w[0]
            slot << None
            self.assertEqual(_inputs(), [])
            slot << s2
            self.assertEqual(_inputs(), ["objA.translateY"])
            self.assertEqual(slot.plug.name(), "net.w[0]")
        self.each_route(body)

    def test_disconnect_then_a_rewire(self):
        """``a.tx.disconnect(slot)`` (rig's own disconnect), then ``slot << a.ty``."""
        def body():
            for kind in ("float", "matrix", "message"):
                self.new_scene()
                n, (s1, s2), (_name1, name2) = _setup(kind)
                n.w[0] << s1
                slot = n.w[0]
                s1.disconnect(slot)
                self.assertEqual(_inputs(), [])
                slot << s2
                self.assertEqual(_inputs(), [name2], kind)
        self.each_route(body)

    def test_break_connections_then_a_rewire(self):
        """``slot.break_connections()`` then ``slot << a.ty``; and ``a.tx.break_connections()``
        returns the element it drove (an Attribute), usable for a connect."""
        def body():
            n, (s1, s2), _names = _setup("float")
            n.w[0] << s1
            slot = n.w[0]
            slot.break_connections()
            self.assertEqual(_inputs(), [])
            slot << s2
            self.assertEqual(_inputs(), ["objA.translateY"])

            n.w[1] << s1
            (driven,) = s1.break_connections()
            self.assertEqual(_inputs("net.w[1]"), [])
            self.assertEqual(driven.plug.name(), "net.w[1]")
            s2.connect(driven)
            self.assertEqual(_inputs("net.w[1]"), ["objA.translateY"])
        self.each_route(body)

    def test_a_child_of_an_element(self):
        """guard: a child below an element (``w[0].wX``) rewired, the plug held while
        connected (858ceab survived this one)."""
        def body():
            n, (s1, _s2), _names = _setup("vector")
            a = Node("objA")
            n.w[0].wX << a.tx
            slot = n.w[0].wX
            slot << a.ty
            self.assertEqual(_inputs("net.w[0].wX"), ["objA.translateY"])
            slot << a.tz
            self.assertEqual(_inputs("net.w[0].wX"), ["objA.translateZ"])
        self.each_route(body)
