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
        self.inputs()

    def inputs(self):
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

    def test_a_block_is_one_node(self):
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.out.ty << self.b.ty * 2                  # unrelated: the block's node all the same
        self.assertEqual(len(self.fused_nodes()), 1)
        with rig.fuse():
            self.out.tz << self.a.tz * 2                  # another block: another node
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertEqual([cmds.getAttr(f"out.t{c}") for c in "xyz"], [3.0, 8.0, 6.5])

    def test_a_setting_on_a_node_the_block_drives_joins_its_node(self):
        # out.k is a setting nothing drives: reading it closes no ring with out.tx
        cmds.addAttr("out", longName="k", attributeType="double", keyable=True)
        cmds.setAttr("out.k", 4.0)
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.out.ty << self.out.k * self.a.tx
        self.assertEqual(len(self.fused_nodes()), 1)
        self.assertEqual(cmds.getAttr("out.ty"), 6.0)
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

    def test_a_driven_attribute_on_a_node_the_block_drives_starts_a_node(self):
        # out.k driven by b.tx: the node would drive out and read out (a ring the
        # evaluation manager evaluates serially)
        cmds.addAttr("out", longName="k", attributeType="double", keyable=True)
        cmds.connectAttr("b.tx", "out.k")
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.out.ty << self.out.k * self.a.tx
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertEqual(cmds.getAttr("out.ty"), 0.75)

    def test_a_world_input_below_a_transform_the_block_drives_starts_a_node(self):
        # the child's world position follows its parent's matrix: no join
        loc = cmds.spaceLocator(name="loc")[0]
        cmds.parent(loc, "out")
        cmds.setAttr("loc.ty", 1.0)
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.b.r << Node("locShape").worldPosition[0] * 2
        self.assertEqual(len(self.fused_nodes()), 2)
        np.testing.assert_allclose(cmds.getAttr("b.r")[0], [6.0, 2.0, 0.0])
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

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

    def test_a_block_per_pass_is_a_node_per_pass(self):
        shared = None
        with rig.fuse():
            shared = self.a.tx * self.a.k + self.a.w      # made once, used by every pass
            for i in range(3):
                driver = Node.create("transform", name=f"d{i}")
                driver.tx << i
                driven = Node.create("transform", name=f"o{i}")
                with rig.fuse():                           # the pass's own node
                    u = driver.tx * 2
                    driven.tx << u
                    driven.ty << u + shared                # computes `shared` itself
        self.assertEqual(len(self.fused_nodes()), 3)
        self.assertEqual(cmds.getAttr("o2.ty"), 4 + (1.5 * 2 + 0.25))

    def test_the_graph_is_watched_only_while_fusing(self):
        import sys

        self.assertIsNone(getattr(sys, "_rig_fuse_watch", None))
        with rig.fuse():
            self.out.tx << self.a.tx * 2
            self.assertIsNotNone(sys._rig_fuse_watch)
        self.assertIsNone(sys._rig_fuse_watch)
        self.assertEqual([len(c) for c in _fuse._GRAPH.values()], [0, 0, 0])

    def test_the_kept_graph_answers_as_a_fresh_one(self):
        # what the block learns of the graph, kept up to date by Maya's messages (a
        # connection, a reparent, a removal), answers every ring test as a walk from scratch
        tests = []
        original = _fuse._loops

        def both(record, plugs):
            kept = original(record, plugs)
            saved = {k: dict(v) for k, v in _fuse._GRAPH.items()}
            _fuse._forget_graph()
            fresh = original(record, plugs)
            for k, v in saved.items():
                _fuse._GRAPH[k].clear()
                _fuse._GRAPH[k].update(v)
            tests.append((kept, fresh))
            return kept

        from unittest import mock
        with mock.patch.object(_fuse, "_loops", both):
            with rig.fuse():
                self.out.tx << self.a.tx * 2
                mp = Node.create("plusMinusAverage", name="pma")
                mp.input1D[0] << self.out.tx
                self.out.ty << self.a.ty + mp.output1D       # a ring: a second node
                grp = Node.create("transform", name="grp")
                cmds.parent("b", "grp")
                grp.tx << self.a.tx * 3                       # the block drives b's parent
                self.out.tz << Node("b").tx + self.a.tz       # b.tx is a setting: joins
                cmds.disconnectAttr("out.tx", "pma.input1D[0]")   # the ring is cut: pma is a setting
                self.out.sx << mp.output1D + 1
        self.assertGreaterEqual(len(tests), 3)
        self.assertEqual({kept for kept, _ in tests}, {True, False})   # a ring found, and a join
        self.assertEqual([kept for kept, _ in tests], [fresh for _, fresh in tests])
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

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
        far = Node.create("transform", name="far")
        with rig.fuse():
            u = self.a.tx * 2
            self.out.tx << u * u + 1
            far.t << self.a.t * u
        self.assertEqual(self.fused_nodes(), ["fused1"])
        self.assertEqual(cmds.getAttr("fused1.notes"), "\n".join([   # Markdown
            "# rig.fuse",
            "",
            "One `rigExpr` node computes these expressions.",
            "",
            "## Inputs",
            "",
            "- `scalarIn[0] <- a.translateX`",
            "- `vectorIn[0] <- a.translate`",
            "",
            "## Shared terms",
            "",
            "- `t1 = (a.translateX * 2)`",
            "",
            "## Outputs",
            "",
            "- `scalarOut[0] = ((t1 * t1) + 1)`",
            "- `vectorOut[0] = (a.translate * t1)`",
        ]))


class TestLists(_Case):
    """a List operator, and a function given a List, fuse row by row into the block's node"""

    def setUp(self):
        super().setUp()
        # the reference is rig's network (fuse off, even under the option), built without
        # containers: rig's own network of a List in one can fail to connect
        # (rig_network_fixes_todo.md, item 8)
        set_options(fuse=False, create_containers=False)

    def inputs(self):
        super().inputs()
        self.l0, self.l1, self.targets = List(), List(), List()
        for i in range(4):
            v, w = 0.5 * i + 0.25, 3.1 - i
            for lst, name, t in ((self.l0, "p", [v, v * 2, -v]), (self.l1, "q", [w, -w, w * 0.5])):
                node = Node.create("transform", name=f"{name}{i}")
                node.t << t
                lst.append(node)
            self.targets.append(Node.create("transform", name=f"t{i}"))

    def built(self, build, read="t"):
        """(the node types `build` makes, the targets' ``read`` values), rig's network then
        fused, each in a new scene"""
        out = []
        for fused in (False, True):
            cmds.file(new=True, force=True)     # (a new scene clears memoize: both build)
            self.inputs()
            before = set(cmds.ls())
            if fused:
                with rig.fuse():
                    build()
            else:
                build()
            made = sorted({cmds.nodeType(n) for n in set(cmds.ls()) - before})
            values = [cmds.getAttr(f"{t}.{read}") for t in self.targets]
            out.append((made, np.array([v[0] if isinstance(v, list) and read == "t" else v for v in values])))
        return out

    def test_an_operator_on_lists_is_one_node(self):
        (net, expected), (fused, got) = self.built(lambda: self.targets.ty << self.l0.tx + self.l1.tx)
        self.assertEqual(net, ["sum"])
        self.assertEqual(fused, ["rigExpr"])
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(got, expected)

    def test_a_chain_on_a_lists_result_stays_in_the_node(self):
        # the first operator's plugs are its node's outputs: the next one computes on
        # their expressions in that node, it does not read them
        (_, expected), (fused, got) = self.built(
            lambda: self.targets.ty << rig.condition((self.l0.tx + self.l1.tx) * 2 > 5, self.l0.ty, -self.l1.ty))
        self.assertEqual(fused, ["rigExpr"])
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(got, expected)

    def test_a_function_given_a_list(self):
        for name, build in (
            ("vectorized", lambda: self.targets.t << V.normalize(self.l0.t - self.l1.t) * 2),
            ("dispatch", lambda: self.targets.t << rig.normalize(self.l1.t)),
            ("elementwise", lambda: self.targets.ty << F.clamp(self.l0.tx - 1, 0, 0.75)
             + F.logical_not(self.l1.tx > 2)),
        ):
            with self.subTest(name):
                (_, expected), (fused, got) = self.built(build)
                self.assertEqual(fused, ["rigExpr"])
                np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_rows_that_do_not_fuse_build_rigs_nodes(self):
        # matrices: no row fuses, rig's network as without the block
        (net, expected), (fused, got) = self.built(
            lambda: self.targets.offsetParentMatrix << self.l0.worldMatrix[0] * self.l1.worldInverseMatrix[0],
            read="offsetParentMatrix")
        self.assertEqual(fused, net)
        self.assertNotIn("rigExpr", fused)
        np.testing.assert_allclose(got, expected)


class TestRig(_Case):
    """a fused value goes wherever a plug goes"""

    def test_non_fusable_functions_take_its_plug(self):
        with rig.fuse():
            x = self.a.tx * 0.1
            steps = F.diff([x, x * 2, 5])               # diff builds rig's nodes
            self.assertNotIsInstance(steps[0], rig.Fused)
            self.out.tx << steps[0]
            self.out.ty << steps[1]
        self.assertEqual(len(self.fused_nodes()), 1)    # x and x * 2: one node
        np.testing.assert_allclose([cmds.getAttr("out.tx"), cmds.getAttr("out.ty")], [0.15, 4.7], rtol=1e-6)

    def test_a_user_callback_returning_a_fused_value(self):
        from unittest import mock

        # sequence's network (not its traced body): rig takes the callback's fused value as a plug
        with mock.patch.dict(_fuse.RAW):
            del _fuse.RAW["rig.interpolate.sequence"]
            with rig.fuse():
                out = interpolate.sequence(self.a.tx * 0.1, [0, 1], [[0, 0, 0], [10, 20, 30]],
                                           method=lambda a, b, weight=0.5: a + (b - a) * weight)
                self.assertNotIsInstance(out, rig.Fused)
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


class TestRigFunctions(_Case):
    """rig's own functions fuse inside (decision 5): their bodies run on fused values"""

    def setUp(self):
        super().setUp()
        set_options(fuse=False)        # rig's network outside a block, even under the option

    def inputs(self):
        super().inputs()
        self.a.s << [0.5, 2.0, 1.25]
        self.b.s << [3.0, 0.75, 1.5]
        cmds.addAttr("a", longName="u", attributeType="double", keyable=True)
        cmds.setAttr("a.u", 1.4)

    def calls(self):
        from rig import tween
        A, B = self.a, self.b
        return {
            "in_out_elastic": lambda: tween.in_out_elastic(A.w),
            "out_bounce": lambda: tween.out_bounce(A.w * 3),
            "in_out_sine": lambda: tween.in_out_sine(A.u - 1),
            "smoothstep": lambda: interpolate.smoothstep(B.tx, A.k, A.u),
            "smoothstep_vector": lambda: interpolate.smoothstep(B.t, A.t, A.w),
            "smootherstep": lambda: interpolate.smootherstep(0, A.k, A.u, normalize=True),
            "inverse_lerp": lambda: interpolate.inverse_lerp(B.tx, A.k, A.u),
            "interp": lambda: interpolate.interp(A.u, [0, 1, 2], [A.tx, B.tx, A.k], right=B.ty),
            "sequence": lambda: interpolate.sequence(A.u, [0, 1, 2], [A.t, B.t, A.s]),
            "sequence_two_points": lambda: interpolate.sequence(A.w, [0, 1], [A.t, B.t]),
            "sequence_hold": lambda: interpolate.sequence(A.u * 3, [0, 1, 2], [A.t, B.t, A.s], ends="hold"),
            "sequence_live_ends": lambda: interpolate.sequence(A.u * 3, [0, 1, 2], [A.t, B.t, A.s], ends=A.k > 1),
            "sequence_elerp": lambda: interpolate.sequence(A.u, [0, 1, 2], [A.s, B.s, A.s * 2], method=rig.elerp),
            "sequence_slerp": lambda: interpolate.sequence(A.u, [0, 1, 2], [A.t, B.t, -A.t], method=rig.slerp),
            "slerp": lambda: rig.slerp(A.t, B.t, A.w),
            "elerp": lambda: V.elerp(A.s, B.s, A.w),
            "angle": lambda: rig.angle(A.t, B.t),
            "angle_degrees": lambda: V.angle_degrees(A.t, B.t),
            "triple_product": lambda: V.triple_product(A.t, B.t, A.s),
            "choice": lambda: F.choice([A.tx, B.tx, A.k], selector=A.u),
            "searchsorted": lambda: F.searchsorted([0, 1, 2], A.u),
            "all_any": lambda: F.all([A.tx, B.tx]) + F.any([A.tx - 1.5, 0]),
            "argmin_argmax": lambda: F.argmin([A.tx, B.tx, A.k]) * 10 + F.argmax([A.tx, B.tx, A.k]),
        }

    def test_each_fuses_into_one_node_and_computes_what_rigs_network_computes(self):
        for name, build in self.calls().items():
            with self.subTest(name):
                expected = self.value(build())              # rig's network
                before = set(cmds.ls())
                with rig.fuse():
                    value = build()
                    self.assertIsInstance(value, rig.Fused)
                    self.assertEqual(set(cmds.ls()), before)  # nothing built yet
                    plug = _fuse.materialize(value)
                made = [cmds.nodeType(n) for n in set(cmds.ls()) - before]
                self.assertEqual(made, ["rigExpr"])
                # rig's condition nodes hold floats: equal within float rounding
                np.testing.assert_allclose(self.value(plug), expected, rtol=2e-6, atol=1e-6)

    def test_no_container_and_nothing_published(self):
        set_options(create_containers=True)
        with rig.fuse():
            self.out.t << interpolate.sequence(self.a.u, [0, 1, 2], [self.a.t, self.b.t, self.a.s])
        self.assertEqual(cmds.ls(type="container"), [])
        self.assertEqual(len(self.fused_nodes()), 1)

    def test_choice_reads_its_selector_as_maya_does(self):
        with rig.fuse():
            self.out.tx << F.choice([10, 20, 30], selector=self.a.u)
        for selector, expected in ((0.49, 10), (0.5, 20), (1.49, 20), (1.5, 30), (2.4, 30),
                                   (-3, 10), (7, 30)):        # out of range: the first or last
            cmds.setAttr("a.u", selector)
            with self.subTest(selector):
                self.assertEqual(cmds.getAttr("out.tx"), expected)

    def test_a_live_ends_plug_switches(self):
        with rig.fuse():
            self.out.tx << interpolate.sequence(self.a.u, [0, 1], [0, 10], ends=self.a.k > 1)
        self.assertEqual(cmds.getAttr("out.tx"), 10.0)      # a.k = 2: held at the end
        cmds.setAttr("a.k", 0)
        self.assertAlmostEqual(cmds.getAttr("out.tx"), 14.0)   # extrapolated

    def test_a_list_broadcasts_row_by_row(self):
        from rig import tween
        build = lambda: tween.in_quad(List([self.a.tx * 0.2, self.b.tx * 0.5]))  # noqa: E731
        network = [self.value(p) for p in build()]
        cmds.file(new=True, force=True)
        self.inputs()
        before = set(cmds.ls())
        with rig.fuse():
            fused = build()
        self.assertEqual(sorted({cmds.nodeType(n) for n in set(cmds.ls()) - before}), ["rigExpr"])
        np.testing.assert_allclose([self.value(p) for p in fused], network, rtol=1e-6)

    def test_what_does_not_fuse_builds_rigs_network(self):
        cases = {
            "matrices": lambda: interpolate.sequence(self.a.w, [0, 1], [self.a.worldMatrix[0], self.b.worldMatrix[0]],
                                                     method=rig.blend),
            "a choice without a selector": lambda: F.choice([self.a.tx, self.b.tx]),
        }
        for name, build in cases.items():
            with self.subTest(name):
                cmds.file(new=True, force=True)     # (a new scene clears memoize: both build)
                self.inputs()
                before = set(cmds.ls())
                network = build()
                net = sorted(cmds.nodeType(n) for n in set(cmds.ls()) - before)
                cmds.file(new=True, force=True)
                self.inputs()
                before = set(cmds.ls())
                with rig.fuse():
                    fused = build()
                self.assertNotIsInstance(fused, rig.Fused)
                self.assertEqual(type(fused), type(network))
                # the dropped trace built nothing: the same nodes as rig's network
                self.assertEqual(sorted(cmds.nodeType(n) for n in set(cmds.ls()) - before), net)

    def test_errors_are_rigs_and_build_nothing(self):
        with rig.fuse():
            with self.assertRaisesRegex(ValueError, "argument 'ends'"):
                interpolate.sequence(self.a.u, [0, 1], [0, 1], ends="sideways")
            with self.assertRaisesRegex(TypeError, "is not callable"):
                interpolate.sequence(self.a.u, [0, 1], [0, 1], method="lerp")
            with self.assertRaisesRegex(ValueError, "side must be"):
                F.searchsorted([0, 1], self.a.u, side="middle")
            # rig's operand checks run before the body (where a str would read as a value)
            with self.assertRaisesRegex(TypeError, "is a plain str"):
                F.all([self.a.tx, "b.tx"])
        self.assertEqual(cmds.ls(type="rigExpr"), [])
        self.assertEqual(cmds.ls(type="container"), [])

    def test_a_vector_sequence_gives_its_points_exactly(self):
        points = [[1.1, 1.3, 1.7], [0.3, -2.9, 5.1], [-4.7, 0.1, 2.2]]
        for i, p in enumerate(points):
            cmds.addAttr("a", longName=f"p{i}", attributeType="double3")
            for ch in "XYZ":
                cmds.addAttr("a", longName=f"p{i}{ch}", attributeType="double", parent=f"p{i}")
            cmds.setAttr(f"a.p{i}", *p)
        with rig.fuse():
            self.out.t << interpolate.sequence(self.a.u, [0, 1, 2], [self.a.p0, self.a.p1, self.a.p2])
        for x in (0, 1, 2):                     # the last one too: weight 1 of the last segment
            cmds.setAttr("a.u", x)
            with self.subTest(x=x):
                self.assertEqual(list(cmds.getAttr("out.t")[0]), points[x])

    def test_a_user_method_fuses_too(self):
        from rig import tween
        with rig.fuse():
            out = interpolate.sequence(self.a.w, [0, 1], [self.a.t, self.b.t],
                                       method=lambda y0, y1, weight: rig.lerp(y0, y1, tween.in_out_sine(weight)))
            self.assertIsInstance(out, rig.Fused)
            self.out.t << out
        a, b = np.array([1.5, -2.0, 3.25]), np.array([0.5, 4.0, -1.0])
        eased = -0.5 * (np.cos(np.pi * 0.25) - 1)
        np.testing.assert_allclose(cmds.getAttr("out.t")[0], a + (b - a) * eased, rtol=1e-6)
        self.assertEqual(len(self.fused_nodes()), 1)


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
