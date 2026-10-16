"""Review fix (features run): one rule for every right-hand side over a driven plug, and one
undo step for a statement that replaces a driver.

D1 says ``plug << value`` replaces whatever drives the plug, as ``plug << plug`` does. The
review found three places where the forms disagreed, or the result was wrong:

* a child whose compound is connected whole (``c.t`` into ``g.t``): ``g.tx << AnimCurve`` made
  a curve next to the whole input (two drivers: frame 0 read the wire, the keys were lost on
  reopen), ``g.tx << c2.ty`` did the same through Maya's connectAttr -f, and ``g.tx << 5``
  raised "locked or connected". The compound's input is now split into one per child first
  (``c.tx`` into ``g.tx`` ...; `rig._internal.plug._free_of_whole_wire`), then the child's is
  replaced: the other children keep following ``c``. A split that cannot be made (a
  unitConversion feeds the compound whole, a sibling is locked) raises InjectionError naming
  ``g.t << None``, with nothing changed. ``g.tx << None`` frees that child alone.
* a value on the compound itself connected whole (``g.t << [1, 2, 3]``, ``g.t << 1``) raised;
  it now replaces the input, as ``g.t << c2.t`` and ``g.t << AnimCurve(...)`` do. A list with
  a skip or a None splits instead, so the skipped channels keep their driver.
* a value over a keyed, constrained or pairBlend-driven plug was two undo steps (the
  disconnect, then the set): one undo left the plug with no driver. A statement that removes
  an input and then writes is now one undo step (``rig.replace``, `journal.one_step`); so is a
  rewire (``x << b`` over ``x << a``) and ``slot << None`` with its data drop.

Every test of this module failed at 8286cb8 (negative control), except the guards.
"""

from maya import cmds

from rig import AnimCurve, InjectionError, List, Node, Typed, Vector, skip
from rig._internal import journal
from rig._tests.test_rig.test_r5_driven_value import _Case, _cube, _driven, _inputs, _settle


def _wired():
    """``c`` (t = 4, 5, 6) connected whole into ``g.t``; the undo queue flushed."""
    c = _cube("c")
    g = _cube("g")
    cmds.setAttr("c.t", 4, 5, 6)
    cmds.connectAttr("c.t", "g.t")
    cmds.flushUndo()
    return c, g


def _steps(statement):
    """How many undo steps ``statement()`` made (read by undoing them all)."""
    cmds.flushUndo()
    statement()
    count = 0
    while not cmds.undoInfo(query=True, undoQueueEmpty=True):
        cmds.undo()
        count += 1
    return count


class TestAChildOfAWholeWire(_Case):
    def check_split(self, g, keep=("ty", "tz")):
        self.assertEqual(_inputs("g.t"), [])
        for ch in keep:
            self.assertEqual(_inputs("g." + ch), ["c.translate" + ch[-1].upper()])

    def test_keys(self):
        """``g.tx << AnimCurve`` keys tx alone; the scene agrees at every frame, and after a
        save and a reopen (the keys were lost there at 8286cb8)."""
        def body():
            c, g = _wired()
            state0 = self.scene_state()
            result = g.tx << AnimCurve([0, 10], times=[0, 10])
            self.assertEqual(str(result), "g.translateX")
            self.check_split(g)
            self.assertEqual(_inputs("g.tx"), ["g_translateX.output"])
            keyed = self.scene_state()
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "one undo")
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(), keyed, "redo")
            for frame in (0, 5, 10):
                cmds.currentTime(frame)
                self.assertAlmostEqual(cmds.getAttr("g.tx"), float(frame))
                self.assertEqual(cmds.xform("g", q=True, ws=True, t=True)[0], float(frame))
            path = cmds.internalVar(userTmpDir=True) + "fs_fix_child_keys.ma"
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, open=True, force=True)
            cmds.currentTime(5)
            self.assertAlmostEqual(cmds.getAttr("g.tx"), 5.0)
            self.assertEqual(_inputs("g.tx"), ["g_translateX.output"])
            self.assertEqual(_inputs("g.t"), [])
        self.each_route(body)

    def test_a_plug(self):
        """``g.tx << d.ty``: the same split; ty and tz keep following c."""
        def body():
            c, g = _wired()
            d = _cube("d")
            state0 = self.scene_state()
            g.tx << d.ty
            self.check_split(g)
            self.assertEqual(_inputs("g.tx"), ["d.translateY"])
            c.tz << 30
            d.ty << 7
            self.assertEqual(cmds.getAttr("g.t")[0], (7.0, 5.0, 30.0))
            self.undo_steps(3)
            self.assert_state_equal(self.scene_state(), state0, "one undo per statement")
        self.each_route(body)

    def test_a_value_and_none(self):
        """``g.tx << 2`` and ``g.ty << None``: each child freed alone."""
        def body():
            c, g = _wired()
            state0 = self.scene_state()
            g.tx << 2
            self.check_split(g)
            self.assertEqual(_inputs("g.tx"), [])
            self.assertEqual(cmds.getAttr("g.tx"), 2.0)
            g.ty << None
            self.assertEqual(_inputs("g.ty"), [])
            self.assertEqual(_inputs("g.tz"), ["c.translateZ"])
            self.assertEqual(cmds.getAttr("g.t")[0], (2.0, 5.0, 6.0))
            self.undo_steps(2)
            self.assert_state_equal(self.scene_state(), state0, "one undo per statement")
        self.each_route(body)

    def test_a_skip_list_on_the_compound(self):
        """``g.t << [9, skip, skip]``: only tx is replaced; ``g.t << [AnimCurve, skip, 0]``
        keys tx, keeps ty, sets tz, in one step."""
        def body():
            c, g = _wired()
            state0 = self.scene_state()
            g.t << [9, skip, skip]
            self.check_split(g)
            self.assertEqual(cmds.getAttr("g.tx"), 9.0)
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo")
            g.t << [AnimCurve([1, 2]), skip, 0]
            self.check_split(g, keep=("ty",))
            self.assertEqual(_inputs("g.tx"), ["g_translateX.output"])
            self.assertEqual(_inputs("g.tz"), [])
            self.assertEqual(cmds.getAttr("g.tz"), 0.0)
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(), state0, "undo keys")
        self.each_route(body)

    def test_a_list_row(self):
        """``List([g.tx, h.tx]) << AnimCurve(...)``: each row splits its own compound."""
        def body():
            c, g = _wired()
            h = _cube("h")
            cmds.connectAttr("c.t", "h.t")
            cmds.flushUndo()
            List([g.tx, h.tx]) << AnimCurve([0, 1])
            for x in ("g", "h"):
                self.assertEqual(_inputs(x + ".t"), [])
                self.assertEqual(_inputs(x + ".ty"), ["c.translateY"])
            self.assertEqual(_inputs("h.tx"), ["h_translateX.output"])
        self.each_route(body)

    def test_an_element_of_a_vector_multi(self):
        """A Vector multi element connected whole: ``n.w[0].wX << 3`` splits its input."""
        def body():
            c = _cube("c")
            cmds.setAttr("c.t", 4, 5, 6)
            n = Node.create("network", name="n")
            n << Vector("w", multi=True)
            n.w[0] << c.t
            cmds.flushUndo()
            n.w[0].wX << 3
            self.assertEqual(_inputs("n.w[0]"), [])
            self.assertEqual(_inputs("n.w[0].wY"), ["c.translateY"])
            self.assertEqual(cmds.getAttr("n.w[0]")[0], (3.0, 5.0, 6.0))
        self.each_route(body)

    def test_a_split_that_cannot_be_made_raises(self):
        """A unitConversion feeds the compound whole (``c.r`` into ``g.t``), or a sibling is
        locked: InjectionError naming ``g.t << None``, nothing changed, for a value, a plug
        and keys."""
        def body():
            c = _cube("c")
            g = _cube("g")
            d = _cube("d")
            cmds.connectAttr("c.r", "g.t")
            cmds.flushUndo()
            state0 = self.scene_state()
            for rhs in (5, d.ty, AnimCurve([0, 1]), None):
                with self.subTest(rhs=repr(rhs)):
                    with self.assertRaises(InjectionError) as caught:
                        g.tx << rhs
                    self.assertIn("g.translate << None", str(caught.exception))
                    self.assertIn("unitConversion", str(caught.exception))
                    self.assert_state_equal(self.scene_state(), state0, "refused")
            cmds.disconnectAttr(cmds.listConnections("g.t", s=True, p=True)[0], "g.t")
            cmds.connectAttr("c.t", "g.t")
            cmds.setAttr("g.tz", lock=True)
            cmds.flushUndo()
            state1 = self.scene_state()
            with self.assertRaises(InjectionError) as caught:
                g.tx << 5
            self.assertIn("Plug('g.translateZ').unlock()", str(caught.exception))
            self.assert_state_equal(self.scene_state(), state1, "locked sibling")
        self.each_route(body)


class TestAValueOnAWholeWiredCompound(_Case):
    def test_a_list_and_a_scalar(self):
        """``g.t << [1, 2, 3]`` and ``g.t << 1`` replace the whole input (they raised)."""
        def body():
            for rhs, want in (([1, 2, 3], (1.0, 2.0, 3.0)), (1, (1.0, 1.0, 1.0))):
                with self.subTest(rhs=rhs):
                    self.new_scene()
                    c, g = _wired()
                    state0 = self.scene_state()
                    self.assertEqual(str(g.t << rhs), "g.translate")
                    self.assertEqual(_inputs("g.t"), [])
                    self.assertEqual(_inputs("g.tx"), [])
                    self.assertEqual(cmds.getAttr("g.t")[0], want)
                    self.undo_steps(1)
                    self.assert_state_equal(self.scene_state(), state0, "one undo")
                    self.assertEqual(_inputs("g.t"), ["c.translate"])
        self.each_route(body)

    def test_a_vector_multi_element(self):
        """``n.w[0] << [1, 2, 3]`` over ``n.w[0] << c.t``."""
        def body():
            c = _cube("c")
            n = Node.create("network", name="n")
            n << Vector("w", multi=True)
            n.w[0] << c.t
            n.w[0] << [1, 2, 3]
            self.assertEqual(_inputs("n.w[0]"), [])
            self.assertEqual(cmds.getAttr("n.w[0]")[0], (1.0, 2.0, 3.0))
        self.each_route(body)


class TestOneUndoStep(_Case):
    """Bare statements (no undo chunk around them): one undo brings the driver back."""

    def test_a_value_over_each_driver(self):
        def body():
            for kind in ("curve", "constraint", "pairblend", "plain", "expression"):
                with self.subTest(kind=kind):
                    self.new_scene()
                    n, name, _value, _others = _driven(kind)
                    before = _inputs(name)
                    self.assertEqual(_steps(lambda: n.tx << 5), 1)
                    self.assertEqual(_inputs(name), before)
        self.each_route(body)

    def test_a_vector_and_a_list(self):
        def body():
            n, _name, _value, _others = _driven("constraint")
            before = [_inputs("a." + c) for c in ("tx", "ty", "tz")]
            self.assertEqual(_steps(lambda: n.t << [1, 2, 3]), 1)
            self.assertEqual([_inputs("a." + c) for c in ("tx", "ty", "tz")], before)
            self.new_scene()
            xs = [_cube("l%d" % i) for i in range(3)]
            for x in xs:
                cmds.setKeyframe(str(x) + ".ry", t=0, v=0)
                cmds.setKeyframe(str(x) + ".ry", t=10, v=50)
            self.assertEqual(_steps(lambda: List(xs).ry << 5), 1)
            for x in xs:
                self.assertEqual(len(_inputs(str(x) + ".ry")), 1)
        self.each_route(body)

    def test_a_rewire(self):
        """``a.tx << c.ty`` over ``a.tx << c.tx``: one step, and the undo gives c.tx back."""
        def body():
            c = _cube("c")
            a = _cube("a")
            a.tx << c.tx
            self.assertEqual(_steps(lambda: a.tx << c.ty), 1)
            self.assertEqual(_inputs("a.tx"), ["c.translateX"])
        self.each_route(body)

    def test_a_typed_slot_none(self):
        """``slot << None`` on a Typed slot that held a matrix: one step, the connection and
        the data back together."""
        def body():
            c = _cube("c")
            n = Node.create("network", name="n")
            mix = n << Typed("mix", multi=True)
            mix[0] << c.wm
            self.assertEqual(_steps(lambda: mix[0] << None), 1)
            self.assertEqual(_inputs("n.mix[0]"), ["c.worldMatrix"])
            self.assertEqual(cmds.getAttr("n.mix[0]", type=True), "matrix")
        self.each_route(body)

    def test_a_free_plug_is_unchanged(self):
        """guard: a set and a connect on free plugs stay one step (a compound's sets are
        one step on the API route; the cmds route makes one per setAttr, as before)."""
        def body():
            c = _cube("c")
            a = _cube("a")
            self.assertEqual(_steps(lambda: a.tx << 3), 1)
            self.assertEqual(_steps(lambda: a.tx << c.ty), 1)
            if journal.API_EDITS:
                self.assertEqual(_steps(lambda: a.t << [1, 2, 3]), 1)
        self.each_route(body)
