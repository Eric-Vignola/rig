"""``rig.fuse()``: scalar and vector math compiled into rigExpr nodes (prototype).

Needs the rigExpr plug-in (``RIG_EXPR_PLUGIN``, a path to its .mll, or on Maya's
plug-in path); without it, every test here is skipped but the fallback one.
Values are checked against numpy, which the fused node follows.
"""
import os
import tempfile

import numpy as np
from maya import cmds

import rig
from rig import condition, functions as F, get_options, interpolate, List, Node, set_options
from rig import trigonometry as T, vector as V
from rig._internal import fuse as _fuse
from rig._tests._base import MayaTestCase


def _plugin_loads():
    if cmds.pluginInfo("rigExpr", query=True, loaded=True):
        return True
    for candidate in (os.environ.get("RIG_EXPR_PLUGIN"), "rigExpr"):
        if candidate:
            try:
                cmds.loadPlugin(candidate, quiet=True)
                return True
            except RuntimeError:
                pass
    return False


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _plugin_loads():
            raise __import__("unittest").SkipTest("the rigExpr plug-in is not available (RIG_EXPR_PLUGIN)")

    def setUp(self):
        super().setUp()
        options = dict(get_options())
        self.addCleanup(lambda: set_options(**options))
        _fuse.FUSE.plugin = None          # found again (a test may have refused it)
        self.a = Node.create("transform", name="a")
        self.b = Node.create("transform", name="b")
        self.out = Node.create("transform", name="out")
        self.a.t << [1.5, -2.0, 3.25]
        self.b.t << [0.5, 4.0, -1.0]
        for attr, value in (("w", 0.25), ("k", 2.0)):
            cmds.addAttr("a", longName=attr, attributeType="double", keyable=True)
            cmds.setAttr(f"a.{attr}", value)

    def value(self, plug):
        got = cmds.getAttr(str(plug))
        return np.array(got[0], dtype=float) if isinstance(got, list) else float(got)

    def fused_nodes(self):
        return cmds.ls(type="rigExpr")


class TestValues(_Case):
    """the fused node computes what numpy computes"""

    def check(self, build, expected):
        with rig.fuse():
            value = build()
        self.assertIsInstance(value, rig.Fused)
        plug = _fuse.materialize(value)
        np.testing.assert_allclose(self.value(plug), expected, rtol=1e-12, atol=1e-12)

    def test_scalars(self):
        a = np.array([1.5, -2.0, 3.25])
        b = np.array([0.5, 4.0, -1.0])
        w, k = 0.25, 2.0
        A, B = self.a, self.b
        cases = {
            "arith": (lambda: (A.tx * 2 + B.ty) / A.k - A.tz ** 2, (a[0] * 2 + b[1]) / k - a[2] ** 2),
            "mod": (lambda: A.ty % 3, np.remainder(a[1], 3)),
            "floordiv": (lambda: A.ty // 3, np.floor_divide(a[1], 3)),
            "neg": (lambda: -A.tx, -a[0]),
            "compare": (lambda: (A.tx > B.tx) & (A.ty < 0), 1.0),
            "condition": (lambda: condition(A.tx > 2, A.ty, B.ty), b[1]),
            "clamp": (lambda: F.clamp(A.tz, 0, 2), 2.0),
            "round_half_even": (lambda: F.round(A.k + 0.5), 2.0),
            "abs_sign": (lambda: F.abs(A.ty) * F.sign(A.ty), -2.0),
            "sqrt_exp_log": (lambda: F.sqrt(A.k) + F.exp(A.w) + F.log(A.k), np.sqrt(k) + np.exp(w) + np.log(k)),
            "log_base": (lambda: F.log(A.k, 10), np.log(k) / np.log(10)),
            "rev": (lambda: F.rev(A.w), 1 - w),
            "min_max": (lambda: F.max([A.tx, B.tx, 1]) - F.min([A.tx, B.tx]), 1.5 - 0.5),
            "lerp": (lambda: rig.lerp(A.tx, B.tx, A.w), a[0] + (b[0] - a[0]) * w),
            "trig": (lambda: T.sind(A.tx * 20) + T.atan2(A.ty, B.ty), np.sin(np.deg2rad(30)) + np.arctan2(a[1], b[1])),
            "div_zero": (lambda: A.tx / (B.tx - 0.5), np.inf),
            "dot_length": (lambda: V.dot(A.t, B.t) + V.length(A.t - B.t), a @ b + np.linalg.norm(a - b)),
            "dist": (lambda: V.dist(A.t, B.t), np.linalg.norm(a - b)),
            "component": (lambda: (A.t * 2).y, a[1] * 2),
        }
        for name, (build, expected) in cases.items():
            with self.subTest(name):
                self.check(build, expected)

    def test_vectors(self):
        a = np.array([1.5, -2.0, 3.25])
        b = np.array([0.5, 4.0, -1.0])
        A, B = self.a, self.b
        cases = {
            "arith": (lambda: A.t * 2 - B.t / 4, a * 2 - b / 4),
            "list_vector": (lambda: [A.tx, B.ty, 1] * A.k, np.array([a[0], b[1], 1]) * 2),
            "condition": (lambda: condition(A.tx > 0, A.t, B.t), a),
            "normalize": (lambda: V.normalize(A.t - B.t) * F.clamp(A.k, 0, 5), (a - b) / np.linalg.norm(a - b) * 2),
            "cross": (lambda: V.cross(A.t, B.t), np.cross(a, b)),
            "cross_normalized": (lambda: V.cross(A.t, B.t, normalize=True), np.cross(a, b) / np.linalg.norm(np.cross(a, b))),
            "lerp": (lambda: rig.lerp(A.t, B.t, 0.25), a + (b - a) * 0.25),
            "abs_floor": (lambda: F.floor(F.abs(A.t)), np.floor(np.abs(a))),
            "clamp_vector": (lambda: F.clamp(A.t, -1, [0, 1, 2]), np.clip(a, -1, [0, 1, 2])),
        }
        for name, (build, expected) in cases.items():
            with self.subTest(name):
                self.check(build, expected)


class TestNodes(_Case):
    """when a fused value becomes a node, and how many"""

    def test_a_chain_is_one_node(self):
        with rig.fuse():
            u = (self.a.tx * 2 + self.b.ty) * F.rev(self.a.w) - self.a.k
            self.assertEqual(self.fused_nodes(), [])          # nothing is built yet
            self.out.tx << u
        self.assertEqual(len(self.fused_nodes()), 1)
        self.assertEqual(cmds.nodeType(cmds.listConnections("out.tx", source=True)[0]), "rigExpr")

    def test_values_built_on_a_result_join_its_node(self):
        with rig.fuse("demo"):
            u = self.a.tx * 2 + self.b.ty
            self.out.tx << u
            self.out.ty << condition(u > 0, self.a.ty, u * 3)
            self.out.t << [u, u + 1, self.a.tz]       # a list of fused values: one node
        self.assertEqual(self.fused_nodes(), ["demo_fused1"])
        self.assertEqual(cmds.getAttr("out.t")[0], (7.0, 8.0, 3.25))

    def test_unrelated_values_get_their_own_node(self):
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.out.ty << self.b.ty * 2
        self.assertEqual(len(self.fused_nodes()), 2)

    def test_no_cycle_through_a_node_downstream(self):
        # out.tx is driven by the first node; a value reading out's matrix cannot join it
        with rig.fuse():
            u = self.a.tx * 2
            self.out.tx << u
            mp = Node.create("plusMinusAverage", name="pma")
            mp.input1D[0] << self.out.tx
            self.out.ty << u + mp.output1D
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertEqual(cmds.getAttr("out.ty"), 6.0)
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

    def test_a_value_shared_by_two_nodes_gets_its_own_node(self):
        with rig.fuse():
            shared = self.a.tx * self.a.k + self.a.w      # made once, used by every pass
            for i in range(3):
                driver = Node.create("transform", name=f"d{i}")
                driver.tx << i
                driven = Node.create("transform", name=f"o{i}")
                u = driver.tx * 2
                driven.tx << u                             # the pass's node
                driven.ty << u + shared                    # joins it (it uses u)
        # pass 0 computes `shared` inside its node; pass 1 gives it a node of its own,
        # which pass 1 and pass 2 read: 3 pass nodes and 1 shared node
        self.assertEqual(len(self.fused_nodes()), 4)
        self.assertEqual(cmds.getAttr("o2.ty"), 4 + (1.5 * 2 + 0.25))

    def test_a_value_using_what_the_last_node_computes_joins_it(self):
        with rig.fuse():
            s = self.a.tx * self.a.k
            self.out.tx << s + 1        # s is computed inside this node
            self.out.ty << s * 3        # the last node computes s: this joins it
        self.assertEqual(len(self.fused_nodes()), 1)
        self.assertEqual(cmds.getAttr("out.ty"), 9.0)

    def test_numbers_fold(self):
        with rig.fuse():
            self.assertEqual(F.abs(-5), 5)
            self.assertEqual(2 * 3, 6)
        self.assertEqual(self.fused_nodes(), [])

    def test_notes(self):
        with rig.fuse():
            self.out.tx << self.a.tx * 2 + 1
        notes = cmds.getAttr(self.fused_nodes()[0] + ".notes")
        self.assertIn("scalarIn[0] <- a.translateX", notes)
        self.assertIn("scalarOut[0] = ((a.translateX * 2) + 1)", notes)


class TestRig(_Case):
    """a fused value goes wherever a plug goes"""

    def test_non_fusable_functions_take_its_plug(self):
        with rig.fuse():
            x = self.a.tx * 0.1
            out = interpolate.sequence(x, [0, 1], [[0, 0, 0], [10, 20, 30]])
            self.out.t << out
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], [1.5, 3.0, 4.5], rtol=1e-6)

    def test_a_user_callback_returning_a_fused_value(self):
        with rig.fuse():
            out = interpolate.sequence(self.a.tx * 0.1, [0, 1], [[0, 0, 0], [10, 20, 30]],
                                       method=lambda a, b, weight=0.5: a + (b - a) * weight)
            self.out.t << out
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], [1.5, 3.0, 4.5], rtol=1e-6)

    def test_matrices_fall_back(self):
        with rig.fuse():
            m = self.a.worldMatrix[0] * self.b.worldInverseMatrix[0]
            self.assertNotIsInstance(m, rig.Fused)
            self.out << m
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], [1.0, -6.0, 4.25], atol=1e-9)

    def test_lists(self):
        with rig.fuse():
            items = List([self.a.tx * 2, self.b.tx])
            self.assertNotIsInstance(items[0], rig.Fused)
            items.append(self.a.ty + 1)
            doubled = items * 2                         # a List operator: rig's nodes
        self.assertEqual([cmds.getAttr(str(p)) for p in doubled], [6.0, 1.0, -2.0])

    def test_truth(self):
        with rig.fuse():
            with self.assertRaises(TypeError):
                bool(self.a.tx > 0)
            self.assertTrue(self.a.tx in [self.b.tx, self.a.tx])     # no node built
            self.assertTrue(self.a.tx == self.a.tx)
            self.assertFalse(self.a.tx == self.b.tx)
        self.assertEqual(self.fused_nodes(), [])

    def test_truth_messages_are_a_plugs(self):
        with rig.fuse():
            for value, pattern in ((self.a.tx > 0, r"ordering comparison .*\.get\(\)"),
                                   (self.a.tx == 5, "the output of a comparison with a value"),
                                   (self.a.tx + 1, r"^a plug has no truth value \(")):
                with self.subTest(pattern), self.assertRaisesRegex(TypeError, pattern):
                    bool(value)

    def test_an_iterator_fuses_as_a_list(self):
        with rig.fuse():
            total = F.sum(iter([self.a.tx, self.b.tx, 1]))
            self.assertIsInstance(total, rig.Fused)
            self.out.tx << total
        self.assertEqual(cmds.getAttr("out.tx"), 3.0)

    def test_a_rejected_call_builds_nothing(self):
        with rig.fuse():
            with self.assertRaises(TypeError):
                condition(self.a.tx > 0, "yes", "no")       # a str operand: rig refuses it
            with self.assertRaises(TypeError):
                self.a.tx * [self.a.ty + 1, "b.tx", 1]      # a str in a list operand
        self.assertEqual(self.fused_nodes(), [])

    def test_an_attribute_deleted_and_added_again(self):
        set_options(fuse=True)
        cmds.addAttr("a", longName="spare", attributeType="double")
        cmds.setAttr("a.spare", -2.0)
        self.out.tx << F.abs(self.a.spare)
        cmds.deleteAttr("a.spare")
        cmds.addAttr("a", longName="spare", attributeType="double")
        cmds.setAttr("a.spare", -5.0)
        self.out.ty << F.abs(self.a.spare)          # the new attribute, not the old node's output
        self.assertEqual(cmds.getAttr("out.ty"), 5.0)

    def test_an_input_deleted_before_the_value_is_used(self):
        cmds.addAttr("a", longName="spare2", attributeType="double")
        with rig.fuse():
            v = self.a.spare2 * 2
            cmds.deleteAttr("a.spare2")              # (to the undo queue: still alive)
            with self.assertRaisesRegex(RuntimeError, "already deleted"):
                self.out.tx << v
        self.assertEqual(self.fused_nodes(), [])

    def test_another_target_maya_does_not_fuse(self):
        from rig._internal.maya_version import get_maya_version
        set_options(maya_version=get_maya_version() - 1)
        with rig.fuse():
            self.assertNotIsInstance(self.a.tx * 2, rig.Fused)
        set_options(maya_version=None)
        with rig.fuse():
            self.assertIsInstance(self.a.tx * 2, rig.Fused)

    def test_two_objects_of_one_plug(self):
        with rig.fuse():
            p1, p2 = self.a.tx, rig.Plug("a.tx")
            self.assertTrue(p1 == p2)                  # rig's rule: the same plug
            self.assertFalse(p1 != p2)
            self.assertEqual(len({p1, p2}), 1)
        self.assertEqual(self.fused_nodes(), [])

    def test_another_object_of_a_plug_is_the_same_value(self):
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.out.ty << rig.Plug("a.tx") * 2       # another object of a.tx: the same value
        self.assertEqual(len(self.fused_nodes()), 1)
        self.assertEqual(cmds.listConnections("out.ty", plugs=True), cmds.listConnections("out.tx", plugs=True))

    def test_no_cost_once_no_fused_value_is_left(self):
        if _fuse.FUSE.option:
            self.skipTest("the fuse option keeps the caches of an endless block")
        import gc

        gc.collect()                    # (earlier tests' values held in reference cycles)
        before = _fuse.FUSE.alive
        with rig.fuse():
            v = self.a.tx * 2
            self.assertGreater(_fuse.FUSE.alive, before)
        del v
        gc.collect()
        self.assertEqual(_fuse.FUSE.alive, before)    # rig's hooks look for none now

    def test_math_type_and_channels(self):
        from rig._internal.types import math_type
        with rig.fuse():
            v = self.a.t * 2
            self.assertEqual(math_type(v), "vector")
            self.assertEqual(math_type(v.x), "scalar")
            with self.assertRaises(TypeError):
                iter(v)

    def test_after_the_block(self):
        with rig.fuse():
            late = self.a.tx + self.b.tx
        self.out.tx << late                    # still a value: its own node
        self.assertEqual(cmds.getAttr("out.tx"), 2.0)
        x = late * 3                           # outside a block: rig's own node
        self.assertNotIsInstance(x, rig.Fused)
        self.assertEqual(cmds.getAttr(str(x)), 6.0)

    def test_str_materializes(self):
        with rig.fuse():
            v = self.a.tx * 2
            name = str(v)
        self.assertTrue(name.endswith("scalarOut[0]"), name)
        self.assertEqual(cmds.getAttr(name), 3.0)


class TestScene(_Case):

    def test_one_undo_step(self):
        with rig.fuse():
            u = self.a.tx * 2
            self.out.tx << u
            self.out.ty << u + 1
        self.assertEqual(cmds.getAttr("out.ty"), 4.0)
        cmds.undo()
        self.assertEqual(self.fused_nodes(), [])
        self.assertEqual(cmds.getAttr("out.ty"), 0.0)
        cmds.redo()
        self.assertEqual(cmds.getAttr("out.ty"), 4.0)

    def test_save_and_reopen(self):
        with rig.fuse():
            self.out.t << V.normalize(self.a.t) * self.a.k
        expected = cmds.getAttr("out.t")[0]
        path = os.path.join(tempfile.mkdtemp(), "fused.ma")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, open=True, force=True)
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], expected)
        cmds.setAttr("a.k", 4.0)
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], np.array(expected) * 2)

    def test_option(self):
        set_options(fuse=True)
        self.assertTrue(get_options()["fuse"])
        v = self.a.tx * 2
        self.assertIsInstance(v, rig.Fused)
        self.out.tx << v
        set_options(fuse=False)
        self.assertNotIsInstance(self.a.tx * 2, rig.Fused)
        self.assertEqual(cmds.getAttr("out.tx"), 3.0)


class TestNoPlugin(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_falls_back_with_one_warning(self):
        from unittest import mock

        state = _fuse.FUSE
        saved = state.plugin, state.warned
        self.addCleanup(lambda: setattr(state, "plugin", saved[0]) or setattr(state, "warned", saved[1]))
        state.plugin, state.warned = None, False
        a = Node.create("transform", name="a")
        with mock.patch.object(_fuse, "PLUGIN", "rigExprNotThere"),                 mock.patch.dict(os.environ, {"RIG_EXPR_PLUGIN": ""}),                 mock.patch.object(_fuse.cmds, "warning") as warning:
            with rig.fuse():
                v = a.tx * 2
            with rig.fuse():
                w = a.ty * 2
        self.assertNotIsInstance(v, rig.Fused)
        self.assertNotIsInstance(w, rig.Fused)
        self.assertEqual(cmds.ls(type="rigExpr"), [])
        self.assertEqual(warning.call_count, 1)
        self.assertIn("rigExpr", warning.call_args[0][0])
