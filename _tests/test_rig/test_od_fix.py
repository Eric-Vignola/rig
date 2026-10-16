"""One driver per channel, the verification's regression tests (OD_FIX, 2026-10-16).

* `TestADynamicSource`: a custom double3 (``src.vec``) connected whole into ``cube.t``, then a
  value, a plug, keys or None on ``cube.ty``. The split read the children of the networked
  MPlug ``sourceWithConversion()`` gave, after the disconnect that frees it: an access
  violation in Maya (773641c), or ``TypeError: Plug is not a compound parent`` with half the
  split made (3e34727). Dynamic to dynamic and a nested dynamic compound too.
* `TestANestedLock`: a split that walks into a nested compound (a double below it) or splits
  down to the written plug refuses a locked plug it would connect into before any edit
  (773641c removed the whole wire, then raised).
* `TestANestedDouble`: ``dst.top << plug`` / ``<< None`` also frees a double below a child
  that has a whole input of its own (773641c left ``o.tx -> dst.vecX``).
* `TestNoneUnderAnUnsplittableWire`: ``cube.rx << None`` on a key Maya's setKeyframe put
  next to a whole wire through a unitConversion (no matching children to split) removes the
  key, logged at INFO; the wire is then the channel's one driver (773641c refused it). A
  value, a plug or keys there still raise with nothing changed.
* `TestTheCheapTest`: a compound connect with nothing connected or locked below it walks
  nothing (773641c walked every child in Python on every compound connect).

Negative control: runs\\features\\onedriver\\negctl_fix.txt (on 773641c).
"""

from unittest import mock

from maya import cmds

from rig import AnimCurve, InjectionError, Node, Plug
from rig._internal import plug as _plug
from rig._tests.test_rig.test_od_one_driver import _OneDriver, _Rig, _connections, _doubles
from rig._tests.test_rig.test_r5_driven_value import _cube, _inputs, _settle


def _vec(node, name="vec", parent=None):
    """A double3 ``name`` (children ``nameX`` ...) on ``node``, below ``parent`` if given."""
    kw = {"p": parent} if parent else {}
    cmds.addAttr(node, ln=name, at="double3", **kw)
    for axis in "XYZ":
        cmds.addAttr(node, ln=name + axis, at="double", p=name)


def _nested(node):
    """``top`` (compound) holding the double3 ``vec`` and the double ``f``."""
    cmds.addAttr(node, ln="top", at="compound", nc=2)
    _vec(node, parent="top")
    cmds.addAttr(node, ln="f", at="double", p="top")


class TestADynamicSource(_OneDriver):
    def test_a_leaf_under_a_custom_double3(self):
        def body():
            for label, rhs, want in (
                ("value", lambda d: 5, []),
                ("plug", lambda d: d.tz, ["d.translateZ"]),
                ("keys", lambda d: AnimCurve([0, 10], times=[0, 10]), ["cube_translateY.output"]),
                ("None", lambda d: None, []),
            ):
                for k in range(5):
                    with self.subTest(rhs=label, k=k):
                        self.new_scene()
                        src = Node.create("transform", name="src")
                        cube = _cube("cube")
                        d = _cube("d")
                        _vec("src")
                        cmds.setAttr("src.vec", 1, 2, 3)
                        cmds.connectAttr("src.vec", "cube.t")
                        cmds.flushUndo()
                        state0 = self.scene_state()
                        cube.ty << rhs(d)
                        self.assertEqual(_inputs("cube.t"), [])
                        self.assertEqual(_inputs("cube.tx"), ["src.vecX"])
                        self.assertEqual(_inputs("cube.ty"), want)
                        self.assertEqual(_inputs("cube.tz"), ["src.vecZ"])
                        self.assertEqual(_doubles("cube"), [])
                        self.check_one_step(state0)
                        del src
        self.each_route(body)

    def test_dynamic_to_dynamic_with_a_double(self):
        def body():
            src = Node.create("transform", name="src")
            dst = Node.create("transform", name="dst")
            o = _cube("o")
            _vec("src")
            _vec("dst")
            cmds.connectAttr("src.vec", "dst.vec")
            cmds.connectAttr("o.tx", "dst.vecX")
            cmds.flushUndo()
            state0 = self.scene_state()
            dst.vecY << 5
            self.assertEqual(
                _connections("dst"), [("o.translateX", "dst.vecX"), ("src.vecZ", "dst.vecZ")]
            )
            self.assertEqual(cmds.getAttr("dst.vecY"), 5.0)
            self.check_one_step(state0, lambda: self.assertEqual(_doubles("dst"), [("dst.vecX", "dst.vec")]))
            del src, o
        self.each_route(body)

    def test_a_nested_dynamic_compound(self):
        def body():
            for leaf, rhs, double in (
                ("f", 5, False),
                ("vecY", 5, False),
                ("f", None, False),
                ("f", 5, True),
                ("vecY", 5, True),
                ("f", None, True),
            ):
                with self.subTest(leaf=leaf, rhs=rhs, double=double):
                    self.new_scene()
                    src = Node.create("transform", name="src")
                    dst = Node.create("transform", name="dst")
                    _cube("o")
                    _nested("src")
                    _nested("dst")
                    cmds.connectAttr("src.top", "dst.top")
                    if double:
                        cmds.connectAttr("o.tx", "dst.vecX")
                    cmds.flushUndo()
                    state0 = self.scene_state()
                    getattr(dst, leaf) << rhs
                    self.assertEqual(_inputs("dst.top"), [])
                    self.assertEqual(_inputs("dst." + leaf), [])
                    self.assertEqual(_doubles("dst"), [])
                    if double:
                        self.assertEqual(_inputs("dst.vecX"), ["o.translateX"])
                    self.check_one_step(state0)
                    del src
        self.each_route(body)


class TestANestedLock(_OneDriver):
    def light(self, double=True, locked="lightDirectionY"):
        """``plShape.lightData`` into ``lam.lightDataArray[0]`` (a static nested compound
        element), ``o.tx`` into lightDirectionX (a double) and ``locked`` locked."""
        cmds.shadingNode("pointLight", asLight=True)
        cmds.rename(cmds.ls(type="pointLight")[0], "plShape")
        cmds.shadingNode("lambert", asShader=True, name="lam")
        _cube("o")
        cmds.connectAttr("plShape.lightData", "lam.lightDataArray[0]", force=True)
        if double:
            cmds.connectAttr("o.tx", "lam.lightDataArray[0].lightDirectionX")
        cmds.setAttr("lam.lightDataArray[0]." + locked, lock=True)
        cmds.flushUndo()

    def refused(self, statement, locked):
        state0 = self.scene_state()
        with self.assertRaises(InjectionError) as caught:
            statement()
        self.assertIn(f"needs {locked!r} unlocked", str(caught.exception))
        self.assertIn("Nothing was changed", str(caught.exception))
        self.assert_state_equal(self.scene_state(), state0, "refused")

    def test_a_locked_grandchild_below_a_double(self):
        def body():
            self.light()
            self.refused(
                lambda: Plug("lam.lightDataArray[0].lightAmbient") << 0,
                "lam.lightDataArray[0].lightDirectionY",
            )
        self.each_route(body)

    def test_a_locked_plug_on_the_way_down(self):
        """No double: lightDirectionY << 1 splits lightData, then lightDirection; a locked
        lightDirectionZ refuses before the first split (3e34727 and 773641c split lightData,
        then raised)."""
        def body():
            self.light(double=False, locked="lightDirectionZ")
            self.refused(
                lambda: Plug("lam.lightDataArray[0].lightDirectionY") << 1,
                "lam.lightDataArray[0].lightDirectionZ",
            )
        self.each_route(body)

    def test_a_dynamic_nested_compound(self):
        def body():
            for rhs in (5, "plug", None):
                with self.subTest(rhs=rhs):
                    self.new_scene()
                    src = Node.create("transform", name="src")
                    dst = Node.create("transform", name="dst")
                    o = _cube("o")
                    _nested("src")
                    _nested("dst")
                    cmds.connectAttr("src.top", "dst.top")
                    cmds.connectAttr("o.tx", "dst.vecX")
                    cmds.setAttr("dst.vecY", lock=True)
                    cmds.flushUndo()
                    self.refused(lambda: dst.f << (o.ty if rhs == "plug" else rhs), "dst.vecY")
                    del src
        self.each_route(body)


class TestANestedDouble(_OneDriver):
    def test_a_double_below_a_wired_child(self):
        def body():
            for rhs in ("plug", None):
                with self.subTest(rhs=rhs):
                    self.new_scene()
                    src = Node.create("transform", name="src")
                    dst = Node.create("transform", name="dst")
                    Node.create("transform", name="s2")
                    for n in ("src", "dst", "s2"):
                        _nested(n)
                    _cube("o")
                    cmds.connectAttr("s2.vec", "dst.vec")
                    cmds.connectAttr("o.tx", "dst.vecX")
                    cmds.flushUndo()
                    state0 = self.scene_state()
                    with _Rig() as log:
                        dst.top << (src.top if rhs == "plug" else None)
                    want = [("src.top", "dst.top")] if rhs == "plug" else []
                    self.assertEqual(_connections("dst"), want)
                    self.assertEqual(len(log.lines), 2, log.lines)
                    self.check_one_step(
                        state0, lambda: self.assertEqual(_doubles("dst"), [("dst.vecX", "dst.vec")])
                    )
        self.each_route(body)


class TestNoneUnderAnUnsplittableWire(_OneDriver):
    def keyed(self):
        _cube("b")
        cube = _cube("cube")
        cmds.connectAttr("b.t", "cube.r")  # a unitConversion: no matching children
        cmds.setKeyframe("cube.rx", t=0, v=0)
        cmds.setKeyframe("cube.rx", t=10, v=10)
        curve = _inputs("cube.rx")[0]
        self.assertEqual(_doubles("cube"), [("cube.rotateX", "cube.rotate")])
        cmds.flushUndo()
        return cube, curve

    def test_none_removes_the_key(self):
        def body():
            cube, curve = self.keyed()
            wire = _inputs("cube.r")
            state0 = self.scene_state()
            with _Rig() as log:
                cube.rx << None
            self.assertEqual(_inputs("cube.rx"), [])
            self.assertEqual(_inputs("cube.r"), wire)
            self.assertEqual(_doubles("cube"), [])
            self.assertTrue(any("stays its one driver" in line for line in log.lines), log.lines)
            self.check_one_step(state0, lambda: self.assertEqual(_inputs("cube.rx"), [curve]))
        self.each_route(body)

    def test_a_value_a_plug_or_keys_still_raise(self):
        def body():
            cube, curve = self.keyed()
            state0 = self.scene_state()
            for rhs in (5, Node("b").ty, AnimCurve([0, 1], times=[0, 10])):
                with self.subTest(rhs=repr(rhs)):
                    with self.assertRaises(InjectionError) as caught:
                        cube.rx << rhs
                    self.assertIn("cube.rotate << None", str(caught.exception))
                    self.assert_state_equal(self.scene_state(), state0, "refused")
        self.each_route(body)


class TestTheCheapTest(_OneDriver):
    def test_a_fresh_compound_connect_walks_nothing(self):
        def body():
            c, g, n = _cube("c"), _cube("g"), Node.create("network", name="n")
            _nested("n")
            with mock.patch.object(_plug, "_driven_below", wraps=_plug._driven_below) as walk, \
                    mock.patch.object(_plug, "_wired_ancestor", wraps=_plug._wired_ancestor) as up:
                g.t << c.t
                g.t << c.t
                g.t << None
                n.top << None
                n.vec << c.t
            self.assertEqual(walk.call_count, 0)
            self.assertEqual(up.call_count, 1)  # n.vec is a child: its parent is looked at
            self.assertEqual(_connections("g"), [])
            # and a keyed child, a driven grandchild are still found
            cmds.setKeyframe("g.tx", t=0, v=0)
            g.t << c.t
            self.assertEqual(_connections("g"), [("c.translate", "g.translate")])
            cmds.connectAttr("c.ty", "n.vecX")
            n.top << None
            self.assertEqual(_connections("n"), [])
        self.each_route(body)
