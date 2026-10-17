"""Features step 2 (decision D1): ``plug << value`` replaces whatever drives the plug.

In 2.0.0a5 a value over a plug driven by an animCurve, a constraint or a pairBlend was
silently lost: Maya's setAttr accepts the value and the DG overrides it at its next
evaluation (a keyed ``tx << 5`` read 4.52 a frame later, a constrained one 3.0). Only a plain
connection or an expression, which make setAttr fail, were broken and the value set. rig now
removes the input of a destination before the set (`rig._internal.plug._do_set`), as
``plug << plug`` replaces it, logs the INFO line ``breaking incoming connection(s)`` it logged
for a plain connection, and the disconnect and the set are one undo step.

The guards (they passed on 858ceab too): a plain connection, an expression, a locked plug, a
child of a connected compound, and ``Attribute.set``, which still refuses a connected plug.

* `TestAValueReplacesTheDriver`: ``<< 5`` on a keyed plug (linear and angular), a
  parentConstraint-driven plug, a pairBlend-driven plug (Maya's setKeyframe over a
  constraint), a plain connection and an expression, through ``API_EDITS`` True and False:
  after a dgdirty and a time change the value reads 5 and ``get_inputs()`` is empty; the
  statement returns the plug; the constraint and the pairBlend keep their other channels;
  one undo brings the driver back and redo removes it again.
* `TestOtherValues`: an enum name, a bool and a compound list over keyed channels; a list
  over a compound whose children are driven by a curve and a constraint; a value over an
  element of a multi fed by a curve.
* `TestTheGuards`: a locked keyed plug raises InjectionError with nothing changed; a child
  of a connected compound raises as before; ``Attribute.set`` keeps refusing a driven plug;
  a List broadcast over keyed plugs is one undo step; the INFO line names the driver.
"""

import logging
from unittest import mock

from maya import cmds

import rig
from rig import InjectionError, List, Node, Float
from rig._internal import journal
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _inputs(name):
    return cmds.listConnections(name, source=True, destination=False, plugs=True) or []


def _settle():
    """Evaluate the scene at another frame: a value the DG overrides is gone after it. The
    time changes and the dgdirty are kept off the undo queue (``stateWithoutFlush``), so the
    undo walks below step over the statements only."""
    cmds.undoInfo(stateWithoutFlush=False)
    try:
        cmds.currentTime(4)
        cmds.dgdirty(allPlugs=True)
        cmds.currentTime(5)
    finally:
        cmds.undoInfo(stateWithoutFlush=True)


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _driven(kind):
    """``(node, plug name, the value the driver gives at frame 5, [other plugs it drives])``
    for one kind of driver; the undo queue is flushed after it."""
    cmds.currentTime(0)
    if kind in ("curve", "curve_angle"):
        n = _cube("a")
        attr = "tx" if kind == "curve" else "rx"
        cmds.setKeyframe("a." + attr, t=0, v=1)
        cmds.setKeyframe("a." + attr, t=10, v=11)
        name, others = "a." + ("translateX" if kind == "curve" else "rotateX"), []
    elif kind == "constraint":
        c = _cube("c")
        n = _cube("a")
        c.t << [3, 4, 6]
        cmds.parentConstraint("c", "a")
        name, others = "a.translateX", ["a.translateY", "a.translateZ", "a.rotateX"]
    elif kind == "pairblend":
        c = _cube("c")
        n = _cube("a")
        c.t << [3, 4, 6]
        cmds.parentConstraint("c", "a")
        cmds.setKeyframe("a.tx", t=0, v=7)
        assert _inputs("a.tx")[0].startswith("pairBlend"), _inputs("a.tx")
        name, others = "a.translateX", ["a.translateY", "a.translateZ", "a.rotateX"]
    elif kind == "plain":
        c = _cube("c")
        n = _cube("a")
        c.ty << 4
        cmds.connectAttr("c.ty", "a.tx")
        name, others = "a.translateX", []
    elif kind == "expression":
        n = _cube("a")
        cmds.expression(s="a.tx = frame * 2;", name="ex")
        name, others = "a.translateX", []
    else:
        raise ValueError(kind)
    cmds.flushUndo()
    _settle()
    return n, name, cmds.getAttr(name), {o: (_inputs(o), cmds.getAttr(o)) for o in others}


KINDS = ("curve", "curve_angle", "constraint", "pairblend", "plain", "expression")
GUARDS = ("plain", "expression")


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def each_route(self, body):
        """``body()`` in a new scene with an empty undo queue, with ``API_EDITS`` True, then
        False."""
        for api in (True, False):
            with self.subTest(api=api):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                with mock.patch.object(journal, "API_EDITS", api):
                    body()


class TestAValueReplacesTheDriver(_Case):
    def check(self, kind):
        def body():
            n, name, driven, others = _driven(kind)
            attr = name.split(".")[1]
            plug = getattr(n, attr)
            self.assertTrue(plug.get_inputs())
            self.assertNotEqual(driven, 5)
            state0 = self.scene_state()

            with rig.undo_chunk("value"):
                result = plug << 5
            self.assertEqual(str(result), name)
            _settle()
            self.assertEqual(cmds.getAttr(name), 5)
            self.assertEqual(list(plug.get_inputs()), [])
            self.assertEqual(_inputs(name), [])
            # the driver keeps its other channels
            for other, (inputs, value) in others.items():
                self.assertEqual(_inputs(other), inputs, other)
                self.assertAlmostEqual(cmds.getAttr(other), value, 6, other)
            state1 = self.scene_state()

            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo")
            _settle()
            self.assertAlmostEqual(cmds.getAttr(name), driven, 6)
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(), state1, "redo")
            _settle()
            self.assertEqual(cmds.getAttr(name), 5)

            # a re-run of the statement sets the value again, with nothing to break
            with rig.undo_chunk("again"):
                plug << 5
            _settle()
            self.assertEqual(cmds.getAttr(name), 5)
            self.assert_state_equal(self.scene_state(), state1, "re-run")
        self.each_route(body)


def _add_value_tests():
    for kind in KINDS:
        def test(self, kind=kind):
            self.check(kind)
        test.__name__ = f"test_{kind}"
        guard = "guard (858ceab broke this driver too): " if kind in GUARDS else ""
        test.__doc__ = f"{guard}``<< 5`` on a plug driven by: {kind}."
        setattr(TestAValueReplacesTheDriver, test.__name__, test)


_add_value_tests()


class TestOtherValues(_Case):
    def test_an_enum_name_over_a_keyed_enum(self):
        """``n.mode << 'on'`` on a keyed enum attribute sets the field's int."""
        def body():
            n = _cube("a")
            cmds.addAttr("a", ln="mode", at="enum", en="off:on:auto", k=True)
            cmds.setKeyframe("a.mode", t=0, v=0)
            cmds.setKeyframe("a.mode", t=10, v=2)
            cmds.flushUndo()
            with rig.undo_chunk("enum"):
                n.mode << "on"
            _settle()
            self.assertEqual(cmds.getAttr("a.mode"), 1)
            self.assertEqual(_inputs("a.mode"), [])
            self.undo_steps(1)
            self.assertEqual(_inputs("a.mode"), ["a_mode.output"])
        self.each_route(body)

    def test_a_bool_over_keyed_visibility(self):
        def body():
            n = _cube("a")
            cmds.setKeyframe("a.v", t=0, v=1)
            cmds.setKeyframe("a.v", t=10, v=1)
            cmds.flushUndo()
            with rig.undo_chunk("bool"):
                n.v << False
            _settle()
            self.assertFalse(cmds.getAttr("a.v"))
            self.assertEqual(_inputs("a.v"), [])
            self.undo_steps(1)
            self.assertEqual(_inputs("a.v"), ["a_visibility.output"])
        self.each_route(body)

    def test_a_list_over_driven_children(self):
        """``a.t << [5, 6, 7]``: tx keyed, ty from a constraint, tz free; one undo step."""
        def body():
            c = _cube("c")
            n = _cube("a")
            c.t << [3, 4, 6]
            cmds.pointConstraint("c", "a", skip=["x", "z"])
            cmds.setKeyframe("a.tx", t=0, v=1)
            cmds.setKeyframe("a.tx", t=10, v=2)
            cmds.flushUndo()
            state0 = self.scene_state()
            with rig.undo_chunk("list"):
                result = n.t << [5, 6, 7]
            self.assertEqual(str(result), "a.translate")
            _settle()
            self.assertEqual(cmds.getAttr("a.t")[0], (5, 6, 7))
            for ch in "XYZ":
                self.assertEqual(_inputs("a.translate" + ch), [], ch)
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo")
        self.each_route(body)

    def test_a_scalar_over_a_driven_compound(self):
        """``a.s << 2``: every child keyed; each gives up its curve."""
        def body():
            n = _cube("a")
            for ch in "xyz":
                cmds.setKeyframe("a.s" + ch, t=0, v=1)
                cmds.setKeyframe("a.s" + ch, t=10, v=3)
            cmds.flushUndo()
            with rig.undo_chunk("scalar"):
                n.s << 2
            _settle()
            self.assertEqual(cmds.getAttr("a.s")[0], (2, 2, 2))
            self.assertEqual(cmds.listConnections("a", type="animCurve", s=True, d=False) or [], [])
        self.each_route(body)

    def test_a_value_over_a_keyed_multi_element(self):
        """``net.w[1] << 5`` on an element of a non-keyable Float multi fed by a curve
        (step 1's rebind keeps the element's plug usable for the set)."""
        def body():
            n = Node.create("network", name="net")
            n << Float("w", multi=True, keyable=False)
            n.w[1] << 0
            curve = cmds.createNode("animCurveTU", name="crv")
            cmds.setKeyframe(curve, t=0, v=1)
            cmds.setKeyframe(curve, t=10, v=3)
            cmds.connectAttr(curve + ".output", "net.w[1]")
            cmds.flushUndo()
            slot = n.w[1]
            with rig.undo_chunk("element"):
                slot << 5
            _settle()
            self.assertEqual(cmds.getAttr("net.w[1]"), 5)
            self.assertEqual(_inputs("net.w[1]"), [])
            self.assertEqual(slot.get_mplug().name(), "net.w[1]")
            self.undo_steps(1)
            self.assertEqual(_inputs("net.w[1]"), ["crv.output"])
            self.redo_steps(1)
            slot << 6
            self.assertEqual(cmds.getAttr("net.w[1]"), 6)
        self.each_route(body)


class TestTheGuards(_Case):
    def test_a_locked_keyed_plug_raises_with_nothing_changed(self):
        """guard: the lock check comes before the disconnect."""
        def body():
            n, name, driven, _others = _driven("curve")
            cmds.setAttr(name, lock=True)
            cmds.flushUndo()
            state0 = self.scene_state()
            with self.assertRaises(InjectionError):
                n.tx << 5
            self.assert_state_equal(self.scene_state(), state0, "locked")
            _settle()
            self.assertAlmostEqual(cmds.getAttr(name), driven, 6)
            self.assertTrue(cmds.getAttr(name, lock=True))
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.each_route(body)

    def test_a_locked_constraint_driven_plug_raises(self):
        """guard."""
        def body():
            n, name, driven, others = _driven("constraint")
            cmds.setAttr(name, lock=True)
            state0 = self.scene_state()
            with self.assertRaises(InjectionError):
                n.tx << 5
            self.assert_state_equal(self.scene_state(), state0, "locked")
        self.each_route(body)

    def test_a_child_of_a_connected_compound_splits_its_input(self):
        """Re-pinned by the review fix (it raised InjectionError at s2): ``g.tx`` has no
        input of its own when ``g.t`` is connected whole; ``g.tx << 5`` gives each child
        its own input from ``c.t``'s children, then replaces ``g.tx``'s, as ``g.tx <<
        c.ty`` and ``g.tx << AnimCurve(...)`` do: ty and tz keep following c. One undo
        brings the whole connection back."""
        def body():
            c = _cube("c")
            g = _cube("g")
            cmds.setAttr("c.t", 4, 5, 6)
            cmds.connectAttr("c.t", "g.t")
            cmds.flushUndo()
            state0 = self.scene_state()
            g.tx << 5
            self.assertEqual(_inputs("g.t"), [])
            self.assertEqual(_inputs("g.tx"), [])
            self.assertEqual(_inputs("g.ty"), ["c.translateY"])
            self.assertEqual(_inputs("g.tz"), ["c.translateZ"])
            self.assertEqual(cmds.getAttr("g.t")[0], (5.0, 5.0, 6.0))
            c.ty << 9
            self.assertEqual(cmds.getAttr("g.ty"), 9.0)
            self.undo_steps(2)
            self.assert_state_equal(self.scene_state(), state0, "one undo per statement")
        self.each_route(body)

    def test_attribute_set_still_refuses_a_driven_plug(self):
        """guard: only ``<<`` replaces the driver; ``Attribute.set`` is Maya's setAttr."""
        def body():
            n, name, _driven_value, _others = _driven("plain")
            with self.assertRaises(RuntimeError):
                n.tx.set(5)
            self.assertEqual(_inputs(name), ["c.translateY"])
        self.each_route(body)

    def test_a_list_broadcast_is_one_undo_step(self):
        def body():
            xs = [_cube("l%d" % i) for i in range(3)]
            for x in xs:
                cmds.setKeyframe(str(x) + ".ry", t=0, v=0)
                cmds.setKeyframe(str(x) + ".ry", t=10, v=50)
            cmds.flushUndo()
            _settle()
            state0 = self.scene_state()
            with rig.undo_chunk("broadcast"):
                List(xs).ry << 5
            _settle()
            for i in range(3):
                self.assertEqual(cmds.getAttr("l%d.ry" % i), 5)
                self.assertEqual(_inputs("l%d.ry" % i), [])
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo")
            _settle()
            for i in range(3):
                self.assertAlmostEqual(cmds.getAttr("l%d.ry" % i), 25, 6)
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.each_route(body)

    def test_the_info_line_names_the_driver(self):
        def body():
            n, name, _value, _others = _driven("constraint")
            with self.assertLogs("rig._internal.plug", level=logging.INFO) as logs:
                n.tx << 5
            self.assertEqual(len(logs.records), 1)
            self.assertIn("breaking incoming connection(s)", logs.output[0])
            self.assertIn("a_parentConstraint1.constraintTranslateX", logs.output[0])
        self.each_route(body)
