"""One driver per channel (the user's decision, 2026-10-16).

Maya takes a compound connected whole and an input on one of its children at once (``c.t``
into ``g.t`` and ``d.ty`` into ``g.tx``: Maya's setKeyframe on ``g.tx`` makes it, so do older
scenes and raw cmds), and resolves such a double drive by the reader's pull path: the viewport
and the evaluation manager read the leaf, getAttr, MPlug, rig's get() and bakeResults read the
compound. rig keeps one driver per channel, so the leaf supersedes in the graph itself:

* a child write over a compound connected whole splits the whole input first (as before);
* ``compound << plug`` and ``compound << None`` also remove the inputs its children have of
  their own (as ``compound << [values]`` already did), logged at INFO;
* an existing double is normalised by any write to a leaf, a value included: the whole input
  is split, the written leaf's own input replaced, and a sibling with an input of its own
  keeps it (the leaf one) and takes no part of the whole wire. On 3e34727 a value over such a
  leaf removed its own input and then raised "locked or connected" (a half edit);
* every such statement is one undo step, and its undo gives back the original state, the
  double included.

Negative control: on 3e34727 (runs\\features\\onedriver\\negctl.txt) 14 of the 15 tests fail; the
guard ``test_a_list_into_the_compound_is_unchanged`` passes there too.
"""

import logging

from maya import cmds

from rig import AnimCurve, InjectionError, Node, Vector
from rig._tests.test_rig.test_r5_driven_value import _Case, _cube, _inputs, _settle


def _wired(double=()):
    """``c`` (t = 4, 5, 6) connected whole into ``g.t``, ``d`` (t = 1, 2, 3); ``double`` names
    (g child, source) wires made next to it with cmds (two drivers); the undo queue flushed."""
    c = _cube("c")
    g = _cube("g")
    d = _cube("d")
    cmds.setAttr("c.t", 4, 5, 6)
    cmds.setAttr("d.t", 1, 2, 3)
    cmds.connectAttr("c.t", "g.t")
    for leaf, src in double:
        cmds.connectAttr(src, "g." + leaf)
    cmds.flushUndo()
    return c, g, d


def _doubles(node):
    """Every (destination, its compound ancestor that is a destination too) of ``node``."""
    from maya.api import OpenMaya

    found = []
    pairs = cmds.listConnections(node, source=True, destination=False, connections=True, plugs=True) or []
    for name in pairs[0::2]:
        sel = OpenMaya.MSelectionList()
        sel.add(name)
        cur = sel.getPlug(0)
        while cur.isChild or cur.isElement:
            cur = cur.parent() if cur.isChild else cur.array()
            if cur.isDestination and not cur.isArray:
                found.append((name, cur.name()))
                break
    return found


def _connections(node):
    pairs = cmds.listConnections(node, source=True, destination=False, connections=True, plugs=True) or []
    return sorted(zip(pairs[1::2], pairs[0::2]))


class _Rig(logging.Handler):
    """The INFO lines of the ``rig`` logger while active."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        if record.levelno == logging.INFO:
            self.lines.append(record.getMessage())

    def __enter__(self):
        self.logger = logging.getLogger("rig")
        self.level  = self.logger.level
        self.logger.addHandler(self)
        self.logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self.level)
        return False


class _OneDriver(_Case):
    def check_one_step(self, state0, undone=None):
        """The statement just run is one undo step: its undo gives ``state0`` back (then
        ``undone()`` runs), its redo the state after it."""
        statement_state = self.scene_state()
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(), state0, "one undo")
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True), "one step")
        if undone is not None:
            undone()
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(), statement_state, "redo")


class TestACompoundWrite(_OneDriver):
    """``g.t << plug`` and ``g.t << None`` remove the children's own inputs."""

    def test_a_plug_after_a_split(self):
        def body():
            c, g, d = _wired()
            g.tx << c.ty  # split: c.tx -> g.tx gone, c.ty -> g.tx, c.tY / c.tZ per child
            cmds.flushUndo()
            state0 = self.scene_state()
            with _Rig() as log:
                result = g.t << d.t
            self.assertEqual(str(result), "g.translate")
            self.assertEqual(_connections("g"), [("d.translate", "g.translate")])
            self.assertEqual(_doubles("g"), [])
            self.assertEqual(
                log.lines,
                [
                    "connect g.translate: removing the input c.translateY of its child "
                    "g.translateX (one driver per channel)",
                    "connect g.translate: removing the input c.translateY of its child "
                    "g.translateY (one driver per channel)",
                    "connect g.translate: removing the input c.translateZ of its child "
                    "g.translateZ (one driver per channel)",
                ],
            )
            _settle()
            self.assertEqual(cmds.getAttr("g.t")[0], (1.0, 2.0, 3.0))
            self.check_one_step(state0)
        self.each_route(body)

    def test_none_after_a_split(self):
        """``g.t << None`` frees every channel (it left the per-child wires)."""
        def body():
            c, g, d = _wired()
            g.tx << d.tz
            cmds.flushUndo()
            state0 = self.scene_state()
            with _Rig() as log:
                g.t << None
            self.assertEqual(_connections("g"), [])
            self.assertEqual(len(log.lines), 3)
            self.check_one_step(state0)
        self.each_route(body)

    def test_an_operator_network_after_a_split(self):
        def body():
            c, g, d = _wired()
            g.ty << d.tx
            cmds.flushUndo()
            g.t << d.t * 2
            self.assertEqual(len(_connections("g")), 1)
            self.assertEqual(_doubles("g"), [])
            _settle()
            self.assertEqual(cmds.getAttr("g.t")[0], (2.0, 4.0, 6.0))
        self.each_route(body)

    def test_a_key_on_one_channel(self):
        """Maya's setKeyframe on ``g.tx`` (no whole wire), then ``g.t << d.t``: the curve is
        removed with the statement, one undo gives it back."""
        def body():
            c, g, d = _cube("c"), _cube("g"), _cube("d")
            cmds.setKeyframe("g.tx", t=0, v=0)
            cmds.setKeyframe("g.tx", t=10, v=10)
            curve = _inputs("g.tx")[0]
            cmds.flushUndo()
            state0 = self.scene_state()
            g.t << d.t
            self.assertEqual(_connections("g"), [("d.translate", "g.translate")])
            self.assertEqual(_doubles("g"), [])
            self.check_one_step(state0, lambda: self.assertEqual(_inputs("g.tx"), [curve]))
        self.each_route(body)

    def test_a_vector_multi_element(self):
        """``n.w[0] << None`` frees the per-child wires a split left below the element."""
        def body():
            c = _cube("c")
            n = Node.create("network", name="n")
            n << Vector("w", multi=True)
            n.w[0] << c.t
            n.w[0].wX << c.ty
            cmds.flushUndo()
            state0 = self.scene_state()
            n.w[0] << None
            self.assertEqual(_connections("n"), [])
            self.check_one_step(state0)
        self.each_route(body)

    def test_a_list_into_the_compound_is_unchanged(self):
        """guard: ``g.t << [1, 2, 3]`` replaced each channel's input already (3e34727 too)."""
        def body():
            c, g, d = _wired(double=(("tx", "d.ty"),))
            g.t << [1, 2, 3]
            self.assertEqual(_connections("g"), [])
            self.assertEqual(cmds.getAttr("g.t")[0], (1.0, 2.0, 3.0))
        self.each_route(body)

    def test_a_double_on_the_compound(self):
        """``g.t << d.t`` and ``g.t << None`` over a double: one driver, or none, per channel;
        one undo gives the double back."""
        def body():
            for rhs, want in (("plug", [("d.translate", "g.translate")]), (None, [])):
                with self.subTest(rhs=rhs):
                    self.new_scene()
                    c, g, d = _wired(double=(("tx", "d.ty"),))
                    state0 = self.scene_state()
                    g.t << (d.t if rhs == "plug" else None)
                    self.assertEqual(_connections("g"), want)
                    self.check_one_step(
                        state0,
                        lambda: self.assertEqual(_doubles("g"), [("g.translateX", "g.translate")]),
                    )
        self.each_route(body)


class TestAnExistingDouble(_OneDriver):
    """``c.t`` into ``g.t`` and ``d.ty`` into ``g.tx`` (two drivers on g.tx): a write to a leaf
    leaves one driver per channel."""

    def test_each_right_hand_side_on_the_leaf(self):
        def body():
            for label, rhs, want_tx in (
                ("value", lambda d: 5, []),
                ("plug", lambda d: d.tz, ["d.translateZ"]),
                ("keys", lambda d: AnimCurve([0, 10], times=[0, 10]), ["g_translateX.output"]),
                ("None", lambda d: None, []),
            ):
                with self.subTest(rhs=label):
                    self.new_scene()
                    c, g, d = _wired(double=(("tx", "d.ty"),))
                    state0 = self.scene_state()
                    with _Rig() as log:
                        result = g.tx << rhs(d)
                    self.assertEqual(str(result), "g.translateX")
                    self.assertEqual(_inputs("g.t"), [])
                    self.assertEqual(_inputs("g.tx"), want_tx)
                    self.assertEqual(_inputs("g.ty"), ["c.translateY"])
                    self.assertEqual(_inputs("g.tz"), ["c.translateZ"])
                    self.assertEqual(_doubles("g"), [])
                    self.assertTrue(
                        any("splitting the input c.translate of g.translate" in line for line in log.lines),
                        log.lines,
                    )
                    if label == "value":
                        _settle()
                        self.assertEqual(cmds.getAttr("g.tx"), 5.0)
                    self.check_one_step(
                        state0,
                        lambda: self.assertEqual(_doubles("g"), [("g.translateX", "g.translate")]),
                    )
        self.each_route(body)

    def test_a_sibling_keeps_its_own_input(self):
        """``g.ty << 5``: the whole input is split, g.tx keeps d.ty (the leaf supersedes) and
        takes no c.tx; the INFO line names it."""
        def body():
            c, g, d = _wired(double=(("tx", "d.ty"),))
            state0 = self.scene_state()
            with _Rig() as log:
                g.ty << 5
            self.assertEqual(_inputs("g.t"), [])
            self.assertEqual(_inputs("g.tx"), ["d.translateY"])
            self.assertEqual(_inputs("g.ty"), [])
            self.assertEqual(_inputs("g.tz"), ["c.translateZ"])
            self.assertIn(
                "set g.translateY: splitting the input c.translate of g.translate into one per "
                "child (g.translateX keeps its own input)",
                log.lines,
            )
            _settle()
            self.assertEqual(cmds.getAttr("g.t")[0], (2.0, 5.0, 6.0))
            self.check_one_step(state0)
        self.each_route(body)

    def test_a_locked_sibling_with_its_own_input(self):
        """A locked g.tx with an input of its own takes no connect: the split is made."""
        def body():
            c, g, d = _wired(double=(("tx", "d.ty"),))
            cmds.setAttr("g.tx", lock=True)
            cmds.flushUndo()
            state0 = self.scene_state()
            g.tz << 9
            self.assertEqual(_inputs("g.tx"), ["d.translateY"])
            self.assertEqual(_inputs("g.ty"), ["c.translateY"])
            self.assertTrue(cmds.getAttr("g.tx", lock=True))
            self.check_one_step(state0)
        self.each_route(body)

    def test_a_split_that_cannot_be_made_changes_nothing(self):
        """``c.r`` into ``g.t`` (a unitConversion) and a double on g.tx: a value, a plug or
        None on g.tx raises naming ``g.t << None`` before any edit (3e34727 removed d.ty
        first, then raised for a value, and left the double for a plug or None)."""
        def body():
            c, g, d = _cube("c"), _cube("g"), _cube("d")
            cmds.connectAttr("c.r", "g.t")
            cmds.connectAttr("d.ty", "g.tx")
            cmds.flushUndo()
            state0 = self.scene_state()
            for rhs in (5, d.tz, None):
                with self.subTest(rhs=repr(rhs)):
                    with self.assertRaises(InjectionError) as caught:
                        g.tx << rhs
                    self.assertIn("g.translate << None", str(caught.exception))
                    self.assert_state_equal(self.scene_state(), state0, "refused")
            g.t << None
            self.assertEqual(_connections("g"), [])
        self.each_route(body)


class TestMayasSetKeyframe(_OneDriver):
    """Maya's setKeyframe on ``g.tx`` below ``c.t`` into ``g.t`` keys the child next to the
    whole wire."""

    def keyed(self):
        c, g, d = _wired()
        cmds.setKeyframe("g.tx", t=0, v=0)
        cmds.setKeyframe("g.tx", t=10, v=10)
        curve = _inputs("g.tx")[0]
        self.assertEqual(_doubles("g"), [("g.translateX", "g.translate")])
        cmds.flushUndo()
        return c, g, d, curve

    def test_a_value(self):
        """``g.tx << 5`` sets 5 (3e34727 removed the curve and raised "locked or connected")."""
        def body():
            c, g, d, curve = self.keyed()
            state0 = self.scene_state()
            g.tx << 5
            _settle()
            self.assertEqual(cmds.getAttr("g.tx"), 5.0)
            self.assertEqual(_connections("g"), [("c.translateY", "g.translateY"), ("c.translateZ", "g.translateZ")])
            self.check_one_step(state0, lambda: self.assertEqual(_inputs("g.tx"), [curve]))
        self.each_route(body)

    def test_the_same_keys(self):
        """``g.tx << AnimCurve`` with the keys the curve holds keeps the curve and splits the
        whole wire (3e34727 replaced the curve with c.tx: the keys were lost)."""
        def body():
            c, g, d, curve = self.keyed()
            state0 = self.scene_state()
            g.tx << AnimCurve([0, 10], times=[0, 10])
            self.assertEqual(_inputs("g.tx"), [curve])
            self.assertEqual(_inputs("g.t"), [])
            self.assertEqual(_doubles("g"), [])
            self.check_one_step(state0)
            for frame in (0, 5):
                cmds.currentTime(frame)
                self.assertAlmostEqual(cmds.getAttr("g.tx"), float(frame))
        self.each_route(body)

    def test_a_sibling_and_a_reopen(self):
        """``g.ty << 3`` keeps the curve on g.tx; the scene saved and opened again reads the
        same at every frame (one driver per channel: no reader disagrees)."""
        def body():
            c, g, d, curve = self.keyed()
            g.ty << 3
            self.assertEqual(_inputs("g.tx"), [curve])
            self.assertEqual(_doubles("g"), [])
            values = []
            for frame in (0, 5, 10):
                cmds.currentTime(frame)
                values.append((cmds.getAttr("g.t")[0], tuple(cmds.xform("g", q=True, ws=True, t=True))))
            path = cmds.internalVar(userTmpDir=True) + "od_one_driver_keys.ma"
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, open=True, force=True)
            again = []
            for frame in (0, 5, 10):
                cmds.currentTime(frame)
                again.append((cmds.getAttr("g.t")[0], tuple(cmds.xform("g", q=True, ws=True, t=True))))
            self.assertEqual(again, values)
            self.assertEqual([round(v, 6) for v in values[1][0]], [5.0, 3.0, 6.0])
        self.each_route(body)

    def test_the_compound_none(self):
        def body():
            c, g, d, curve = self.keyed()
            g.t << None
            self.assertEqual(_connections("g"), [])
        self.each_route(body)
