"""``rig.fuse()``: scalar and vector math compiled into rigExpression nodes (prototype).

Needs rig's plug-in, rig.mll (``RIG_PLUGIN``, its path, or on Maya's plug-in
path); without it, every test here is skipped but the fallback one.
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
    if cmds.pluginInfo("rig", query=True, loaded=True):
        return True
    for candidate in (os.environ.get("RIG_PLUGIN"), "rig"):
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
            raise __import__("unittest").SkipTest("the rig plug-in (rig.mll) is not available (RIG_PLUGIN)")

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
        return cmds.ls(type="rigExpression")


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
        self.assertEqual(cmds.nodeType(cmds.listConnections("out.tx", source=True)[0]), "rigExpression")

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

    def source(self, plug):
        return cmds.listConnections(plug, source=True, destination=False)[0]

    def test_a_value_whose_destination_feeds_the_node_starts_a_node(self):
        # the block's node reads pma: a value connected into pma cannot join it (a DG cycle)
        mp = Node.create("plusMinusAverage", name="pma")
        with rig.fuse():
            self.out.tx << mp.output1D * 2 + self.a.tx
            mp.input1D[0] << self.a.ty * 3
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertEqual(cmds.getAttr("out.tx"), -2.0 * 3 * 2 + 1.5)
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

    def test_a_value_that_loops_through_its_destination_gets_a_loop_node(self):
        # a.u from a's own channel, which an animator keys: a ring whichever node computes
        # it (rail_spine's controls: each one's u from the distances between them). It gets
        # a node of its own, so the evaluation manager's cluster holds a small node, not
        # the block's; the values after it still join the block's node
        cmds.addAttr("a", longName="u", attributeType="double", keyable=True)
        with rig.fuse():
            self.out.tx << self.a.tx * 2 + self.b.tx
            self.a.u << self.a.tx * 3
            self.out.ty << self.b.ty * 2
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertNotEqual(self.source("a.u"), self.source("out.tx"))
        self.assertEqual(self.source("out.ty"), self.source("out.tx"))
        self.assertEqual(cmds.getAttr("a.u"), 4.5)
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

    def test_a_channel_nobody_can_key_is_a_setting(self):
        # a's translate locked: it cannot change, so a.u from it makes no ring and joins
        cmds.addAttr("a", longName="u", attributeType="double", keyable=True)
        for axis in "xyz":
            cmds.setAttr(f"a.t{axis}", lock=True)
        with rig.fuse():
            self.out.tx << self.b.tx * 2
            self.a.u << self.a.tx * 3
        self.assertEqual(len(self.fused_nodes()), 1)
        self.assertEqual(cmds.getAttr("a.u"), 4.5)

    def test_an_output_connected_back_to_its_inputs_is_computed_again_in_a_loop_node(self):
        # a List operator's rows are plugs of the block's node. One connected later into a
        # node that node reads would make it part of a ring: the destination reads the row
        # computed again in a loop node, and the block's node keeps computing it for the rest
        cmds.addAttr("a", longName="u", attributeType="double", keyable=True)
        with rig.fuse():
            rows = List([self.a, self.b]).tx * 2
            self.out.tx << rows[0] + rows[1]
            self.a.u << rows[0]
        self.assertEqual(len(self.fused_nodes()), 2)
        self.assertNotEqual(self.source("a.u"), self.source("out.tx"))
        self.assertEqual(cmds.getAttr("a.u"), 3.0)
        self.assertEqual(cmds.getAttr("out.tx"), 4.0)
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
                # b.tx is keyable (an animator's channel: it can change) and b moves with
                # grp, which the block drives: a ring
                self.out.tz << Node("b").tx + self.a.tz
                cmds.disconnectAttr("out.tx", "pma.input1D[0]")   # the ring is cut: pma is a setting
                self.out.sx << mp.output1D + 1
        self.assertGreaterEqual(len(tests), 2)
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
            "One `rigExpression` node computes these expressions.",
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
        self.assertEqual(fused, ["rigExpression"])
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(got, expected)

    def test_a_chain_on_a_lists_result_stays_in_the_node(self):
        # the first operator's plugs are its node's outputs: the next one computes on
        # their expressions in that node, it does not read them
        (_, expected), (fused, got) = self.built(
            lambda: self.targets.ty << rig.condition((self.l0.tx + self.l1.tx) * 2 > 5, self.l0.ty, -self.l1.ty))
        self.assertEqual(fused, ["rigExpression"])
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
                self.assertEqual(fused, ["rigExpression"])
                np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_rows_of_matrices_fuse(self):
        (net, expected), (fused, got) = self.built(
            lambda: self.targets.offsetParentMatrix << self.l0.worldMatrix[0] * self.l1.worldInverseMatrix[0],
            read="offsetParentMatrix")
        self.assertIn("multMatrix", net)
        # (a decomposeMatrix rig's shorthand makes, unused, for a matrix into a transform's
        # offsetParentMatrix: rig's, in both builds)
        self.assertEqual([t for t in fused if t != "decomposeMatrix"], ["rigExpression"])
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_a_decomposed_lists_siblings_fuse(self):
        # decompose is vectorized: its rows come back as plugs, and they keep their
        # decomposeMatrix's siblings
        (net, expected), (fused, got) = self.built(
            lambda: self.targets.s << rig.matrix.decompose(self.l0.worldMatrix).outputScale, read="s")
        self.assertEqual(fused, ["rigExpression"])
        np.testing.assert_allclose(got, expected, atol=1e-9)


class _MatrixCase(_Case):
    """inputs with rotations, scales (one negative), a shear and quaternion plugs; a build
    made twice, rig's network then fused, each in a new scene (`built`)"""

    def setUp(self):
        super().setUp()
        set_options(fuse=False)          # the reference is rig's network, even under the option

    def inputs(self):
        super().inputs()
        cmds.setAttr("a.r", 30.0, -50.0, 110.0)
        cmds.setAttr("a.s", 1.2, 0.8, -1.5)           # a negative scale
        cmds.setAttr("a.shear", 0.2, -0.1, 0.3)
        cmds.setAttr("a.rotateOrder", 2)
        cmds.setAttr("b.r", -70.0, 15.0, 40.0)
        cmds.setAttr("b.s", 0.7, 1.4, 1.1)
        cmds.loadPlugin("matrixNodes", quiet=True)
        for name, source in (("qa", "a"), ("qb", "b")):     # quaternion plugs
            cmds.createNode("decomposeMatrix", name=name)
            cmds.connectAttr(f"{source}.worldMatrix[0]", f"{name}.inputMatrix")

    def read(self, plugs):
        """the values of a plug's name (or a tuple of them), flat"""
        if type(plugs) is tuple:
            return np.concatenate([self.read(p) for p in plugs])
        got = cmds.getAttr(plugs)
        if isinstance(got, list):
            got = [x for item in got for x in (item if isinstance(item, tuple) else (item,))]
        return np.atleast_1d(np.asarray(got, dtype=float))

    def built(self, build):
        """(rig's network's values, the fused ones, the node types the fused build made);
        the fused value made a plug in its block"""
        names = (lambda r: tuple(str(x) for x in r) if type(r) is tuple else str(r))
        cmds.file(new=True, force=True)         # (a new scene clears memoize)
        self.inputs()
        network = self.read(names(build()))
        cmds.file(new=True, force=True)
        self.inputs()
        before = set(cmds.ls())
        with rig.fuse():
            plugs = names(build())
        made = sorted({cmds.nodeType(n) for n in set(cmds.ls()) - before})
        return network, self.read(plugs), made



class TestMatrices(_MatrixCase):
    """matrices, quaternions and eulers: every rig function and operator on them, fused,
    against rig's network (Maya's nodes, whose rules rig.mll's ops follow)"""

    def test_each_function_fuses_and_gives_rigs_values(self):
        M, Q, Eu = rig.matrix, rig.quaternion, rig.euler
        cases = {
            "matrix * matrix": lambda: self.a.worldMatrix[0] * self.b.worldInverseMatrix[0],
            "matrix - matrix": lambda: self.a.worldMatrix[0] - self.b.worldMatrix[0],
            "matrix + matrix": lambda: self.a.worldMatrix[0] + self.b.worldMatrix[0],
            "matrix ** weight": lambda: self.a.worldMatrix[0] ** self.a.w,
            "vector * matrix": lambda: self.b.t * self.a.worldMatrix[0],
            "matrix * vector": lambda: self.a.worldMatrix[0] * self.b.t,
            "multiply, a chain": lambda: M.multiply(self.a.worldMatrix[0], self.b.worldInverseMatrix[0], self.b.worldMatrix[0]),
            "multiply, local": lambda: M.multiply(self.b.t, self.a.worldMatrix[0], local=True),
            "inverse": lambda: M.inverse(self.a.worldMatrix[0]),
            "transpose": lambda: M.transpose(self.a.worldMatrix[0]),
            "determinant": lambda: M.determinant(self.a.worldMatrix[0]),
            "translation": lambda: M.translation(self.a.worldMatrix[0]),
            "rotation": lambda: M.rotation(self.a.worldMatrix[0]),
            "scale_of": lambda: M.scale_of(self.a.worldMatrix[0]),
            "axis y": lambda: M.axis(self.a.worldMatrix[0], axis="y"),
            "row 3": lambda: M.row(self.a.worldMatrix[0], 3),
            "row of a plug": lambda: M.row(self.a.worldMatrix[0], self.a.k),
            "column 1": lambda: M.column(self.a.worldMatrix[0], 1),
            "transform_point": lambda: M.transform_point(self.b.t, self.a.worldMatrix[0]),
            "transform_vector": lambda: M.transform_vector(self.b.t, self.a.worldMatrix[0]),
            "dist": lambda: M.dist(self.a.worldMatrix[0], self.b.worldMatrix[0]),
            "decompose": lambda: M.decompose(self.a.worldMatrix[0]),
            "decompose, a rotate order": lambda: M.decompose(self.a.worldMatrix[0], rotate_order=4).outputRotate,
            "decompose, an order plug": lambda: M.decompose(self.a.worldMatrix[0], rotate_order=self.a.rotateOrder).outputRotate,
            "decompose's scale": lambda: M.decompose(self.a.worldMatrix[0]).outputScale,
            "decompose's shear": lambda: M.decompose(self.a.worldMatrix[0]).outputShear,
            "decompose's quaternion": lambda: M.decompose(self.a.worldMatrix[0]).outputQuat,
            "decompose's quaternion w": lambda: M.decompose(self.a.worldMatrix[0]).outputQuatW,
            "decompose's rotate x": lambda: M.decompose(self.a.worldMatrix[0]).outputRotateX,
            "to_quaternion": lambda: M.to_quaternion(self.a.worldMatrix[0]),
            "to_euler": lambda: M.to_euler(self.a.worldMatrix[0], rotate_order="yxz"),
            "compose": lambda: M.compose(scale=self.a.t, rotate=self.a.r, translate=self.b.t, shear=[0.1, 0.2, -0.3], rotate_order=1),
            "compose, a quaternion": lambda: M.compose(rotate=Node("qa").outputQuat, translate=self.b.t),
            "fourbyfour": lambda: M.fourbyfour(x=self.b.t, y=self.a.t, position=self.b.s),
            "aim": lambda: M.aim(self.b.t, self.a.t),
            "aim, axes": lambda: M.aim(self.b.t, self.a.t, aim_axis=(0, 0, 1), up_axis=(1, 0, 0)),
            "add": lambda: M.add(self.a.worldMatrix[0], self.b.worldMatrix[0], self.b.worldInverseMatrix[0]),
            "add, weights": lambda: M.add(self.a.worldMatrix[0], self.b.worldMatrix[0], weights=[self.a.w, 0.75]),
            "lerp": lambda: M.lerp(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "blend": lambda: M.blend(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "slerp": lambda: M.slerp(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "pow": lambda: M.pow(self.a.worldMatrix[0], self.a.w),
            "normalize": lambda: M.normalize(self.a.worldMatrix[0]),
            "quaternion * quaternion": lambda: Node("qa").outputQuat * Node("qb").outputQuat,
            "quaternion + quaternion": lambda: Node("qa").outputQuat + Node("qb").outputQuat,
            "quaternion - quaternion": lambda: Node("qa").outputQuat - Node("qb").outputQuat,
            "quaternion ** weight": lambda: Node("qa").outputQuat ** self.a.w,
            "quaternion add": lambda: Q.add(Node("qa").outputQuat, Node("qb").outputQuat),
            "quaternion multiply": lambda: Q.multiply(Node("qa").outputQuat, Node("qb").outputQuat),
            "quaternion subtract": lambda: Q.subtract(Node("qa").outputQuat, Node("qb").outputQuat),
            "quaternion negate": lambda: Q.negate(Node("qa").outputQuat),
            "quaternion normalize": lambda: Q.normalize(Node("qa").outputQuat + Node("qb").outputQuat),
            "quaternion inverse": lambda: Q.inverse(Node("qa").outputQuat),
            "quaternion conjugate": lambda: Q.conjugate(Node("qa").outputQuat),
            "quaternion angle": lambda: Q.angle(Node("qa").outputQuat, Node("qb").outputQuat),
            "quaternion angle_degrees": lambda: Q.angle_degrees(Node("qa").outputQuat, Node("qb").outputQuat),
            "quaternion to_euler": lambda: Q.to_euler(Node("qa").outputQuat, rotate_order=5),
            "quaternion to_matrix": lambda: Q.to_matrix(Node("qa").outputQuat),
            "quaternion to_vector": lambda: Q.to_vector(Node("qa").outputQuat),
            "quaternion slerp": lambda: Q.slerp(Node("qa").outputQuat, Node("qb").outputQuat, self.a.w),
            "quaternion pow": lambda: Q.pow(Node("qa").outputQuat, self.a.w),
            "from_axis_angle": lambda: Q.from_axis_angle(self.b.t, self.a.k),
            "from_axis_angle, an angle plug": lambda: Q.from_axis_angle(self.b.t, self.b.rx),
            "from_axis_angle, a number": lambda: Q.from_axis_angle(self.b.t, 1.2),
            "to_axis_angle": lambda: Q.to_axis_angle(Node("qa").outputQuat),
            "euler to_matrix": lambda: Eu.to_matrix(self.a.r, rotate_order=2),
            "euler to_quaternion": lambda: Eu.to_quaternion(self.a.r, rotate_order=3),
            "euler reorder": lambda: Eu.reorder(self.a.r, 2, 4),
            "euler slerp": lambda: Eu.slerp(self.a.r, self.b.r, self.a.w),
            "rig.inverse, a matrix": lambda: rig.inverse(self.a.worldMatrix[0]),
            "rig.inverse, a quaternion": lambda: rig.inverse(Node("qa").outputQuat),
            "rig.lerp, matrices": lambda: rig.lerp(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "rig.slerp, quaternions": lambda: rig.slerp(Node("qa").outputQuat, Node("qb").outputQuat, self.a.w),
            "rig.slerp, matrices": lambda: rig.slerp(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "rig.slerp, eulers": lambda: rig.slerp(self.a.r, self.b.r, self.a.w),
            "rig.normalize, a quaternion": lambda: rig.normalize(Node("qa").outputQuat + Node("qb").outputQuat),
            "rig.normalize, a matrix": lambda: rig.normalize(self.a.worldMatrix[0]),
            "rig.to_euler, a quaternion": lambda: rig.to_euler(Node("qa").outputQuat, rotate_order=1),
            "rig.to_euler, a matrix": lambda: rig.to_euler(self.a.worldMatrix[0]),
            "rig.to_quaternion, an euler": lambda: rig.to_quaternion(self.a.r, rotate_order=2),
            "rig.to_quaternion, a matrix": lambda: rig.to_quaternion(self.a.worldMatrix[0]),
            "rig.to_matrix, an euler": lambda: rig.to_matrix(self.a.r),
            "rig.to_matrix, a quaternion": lambda: rig.to_matrix(Node("qa").outputQuat),
            "rig.dist, matrices": lambda: rig.dist(self.a.worldMatrix[0], self.b.worldMatrix[0]),
            "rig.blend": lambda: rig.blend(self.a.worldMatrix[0], self.b.worldMatrix[0], self.a.w),
            "rig.angle, quaternions": lambda: rig.angle(Node("qa").outputQuat, Node("qb").outputQuat),
            "a rider's twist": lambda: rig.matrix.compose(rotate=Q.pow(Node("qa").outputQuat, self.a.w)) * self.b.worldInverseMatrix[0],
            "the euler of a product": lambda: M.decompose(self.a.worldMatrix[0] * self.b.worldInverseMatrix[0]).outputRotate,
        }
        loose = {"lerp": 1e-5, "rig.lerp, matrices": 1e-5, "add, weights": 1e-5}   # rig's reverse node: floats
        for name, build in cases.items():
            with self.subTest(name):
                network, fused, made = self.built(build)
                self.assertEqual(made, ["rigExpression"])
                np.testing.assert_allclose(fused, network, atol=loose.get(name, 1e-6), rtol=0)

    def test_an_aim_with_an_opposite_axis(self):
        """the up axis exactly opposite the up turns half a turn about the aim, as
        aimMatrix does (limb_twist's end frame at a half turn, and its slerp); the aim
        axis exactly opposite the aim, too"""
        M = rig.matrix
        for aim, up in (([2, 0, 0], [0, -3, 0]), ([-2, 0, 0], [0.5, 4, -1])):
            def build():
                cmds.setAttr("out.t", *aim)
                cmds.setAttr("out.s", *up)
                return M.aim(self.out.t, self.out.s)

            def twist():
                build()
                start = M.aim(self.out.t, [0, 0, 1])
                return M.compose(rotate=rig.slerp(rig.to_quaternion(start), rig.to_quaternion(build()), 0.25))

            for name, case in (("aim", build), ("twist", twist)):
                with self.subTest(name, aim=aim, up=up):
                    network, fused, made = self.built(case)
                    self.assertEqual(made, ["rigExpression"])
                    np.testing.assert_allclose(fused, network, atol=1e-9, rtol=0)

    def test_a_rotation_drives_a_rotate_without_a_unit_conversion(self):
        for fused in (False, True):
            cmds.file(new=True, force=True)
            self.inputs()
            with rig.fuse() if fused else _nothing():
                d = rig.matrix.decompose(self.a.worldMatrix[0] * self.b.worldInverseMatrix[0])
                self.out.r << d.outputRotate
                self.out.ry << self.a.tx * 10          # a scalar of degrees into an angle
                self.out.t << d
            values = [cmds.getAttr("out.r")[0], cmds.getAttr("out.t")[0]]
            if not fused:
                expected = values
        self.assertEqual(cmds.ls(type="unitConversion"), [])
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(values, expected, atol=1e-6)

    def test_a_decompositions_outputs_share_its_node(self):
        with rig.fuse():
            d = rig.matrix.decompose(self.a.worldMatrix[0])
            self.out.t << d
            self.out.r << d.outputRotate
            self.out.s << d.outputScale
            self.b.offsetParentMatrix << rig.matrix.compose(rotate=d.outputQuat, translate=d)
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(cmds.getAttr("out.s")[0], [1.2, 0.8, -1.5], atol=1e-9)


class TestStandIns(_MatrixCase):
    """a math node made by hand in a block (``rn.composeMatrix()``): computed in the
    block's node while its inputs are set before its outputs are read; the node itself
    once used as one, the values that read it then reading the node"""

    def test_a_compose_matrix_made_by_hand_fuses(self):
        from rig.bridges import nodes as rn

        def build():
            cm = rn.composeMatrix()
            cm.inputTranslate << self.a.t * 2
            cm.inputRotate << self.a.r
            cm.inputScale << self.b.s
            cm.inputRotateOrder << 3
            self.out << cm.outputMatrix * self.b.worldInverseMatrix[0]
            return self.out.worldMatrix[0]

        network, fused, made = self.built(build)
        self.assertEqual(made, ["rigExpression"])
        np.testing.assert_allclose(fused, network, atol=1e-9)

    def test_a_decompose_matrix_made_by_hand_fuses(self):
        from rig.bridges import nodes as rn

        def build():
            dm = rn.decomposeMatrix(inputRotateOrder=4)
            dm.inputMatrix << self.a.worldMatrix[0] * self.b.worldInverseMatrix[0]
            self.out.t << dm.outputTranslate
            self.out.r << dm.outputRotate
            self.out.sx << dm.outputScaleY
            return self.out.worldMatrix[0]

        network, fused, made = self.built(build)
        self.assertEqual(made, ["rigExpression"])
        np.testing.assert_allclose(fused, network, atol=1e-9)

    def test_an_input_set_after_an_output_was_read_makes_the_node(self):
        from rig.bridges import nodes as rn

        with rig.fuse():
            cm = rn.composeMatrix()
            cm.inputRotate << [0, 0, 30]
            self.out.offsetParentMatrix << cm.outputMatrix * self.b.worldMatrix[0]
            cm.inputTranslate << self.a.t            # after the read: the node itself
        self.assertEqual(len(cmds.ls(type="composeMatrix")), 1)
        self.assertEqual(len(self.fused_nodes()), 1)
        for t in ([1.5, -2.0, 3.25], [4.0, 1.0, -1.0]):     # it follows the input, as rig's node
            self.a.t << t
            expected = (np.array(cmds.getAttr(f"{cmds.ls(type='composeMatrix')[0]}.outputMatrix")).reshape(4, 4)
                        @ np.array(cmds.getAttr("b.worldMatrix[0]")).reshape(4, 4))
            np.testing.assert_allclose(np.array(cmds.getAttr("out.offsetParentMatrix")).reshape(4, 4), expected,
                                       atol=1e-9)
        self.assertFalse(cmds.cycleCheck(all=True, list=True) or [])

    def test_a_stand_in_used_as_a_node_is_the_node(self):
        from rig.bridges import nodes as rn

        with rig.fuse():
            cm = rn.composeMatrix(name="handmade")
            cm.inputTranslate << self.a.t
            self.assertEqual(str(cm), "handmade")              # its name: the node
            self.out.offsetParentMatrix << cm.outputMatrix
        self.assertEqual(cmds.ls(type="composeMatrix"), ["handmade"])
        self.assertEqual(cmds.listConnections("out.offsetParentMatrix", source=True), ["handmade"])

    def test_after_the_block_a_stand_in_is_the_node(self):
        from rig.bridges import nodes as rn

        with rig.fuse():
            cm = rn.composeMatrix()
            cm.inputTranslate << self.a.t * 2
        self.assertEqual(cmds.ls(type="composeMatrix"), [])   # nothing read it in the block
        self.out.offsetParentMatrix << cm.outputMatrix         # read after it: the node
        self.assertEqual(len(cmds.ls(type="composeMatrix")), 1)
        np.testing.assert_allclose(cmds.xform("out", query=True, worldSpace=True, translation=True),
                                   [3.0, -4.0, 6.5], atol=1e-9)


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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

    def test_matrices_fuse(self):
        with rig.fuse():
            m = self.a.worldMatrix[0] * self.b.worldInverseMatrix[0]
            self.assertIsInstance(m, rig.Fused)
            self.assertEqual(m.math_type, "matrix")
            self.out.offsetParentMatrix << m
        self.assertEqual(len(self.fused_nodes()), 1)
        np.testing.assert_allclose(cmds.xform("out", query=True, worldSpace=True, translation=True),
                                   [1.0, -6.0, 4.25], atol=1e-9)

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
                self.assertEqual(made, ["rigExpression"])
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
        self.assertEqual(sorted({cmds.nodeType(n) for n in set(cmds.ls()) - before}), ["rigExpression"])
        np.testing.assert_allclose([self.value(p) for p in fused], network, rtol=1e-6)

    def test_what_does_not_fuse_builds_rigs_network(self):
        cases = {
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
        self.assertEqual(cmds.ls(type="rigExpression"), [])
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
        with mock.patch.object(_fuse, "PLUGIN", "rigNotThere"), \
                mock.patch.dict(os.environ, {"RIG_PLUGIN": ""}), \
                mock.patch.object(_fuse.cmds, "warning") as warning:
            with rig.fuse():
                v = a.tx * 2
            with rig.fuse():
                w = a.ty * 2
        self.assertNotIsInstance(v, rig.Fused)
        self.assertNotIsInstance(w, rig.Fused)
        self.assertEqual(cmds.ls(type="rigExpression"), [])
        self.assertEqual(warning.call_count, 1)
        self.assertIn("rig.mll", warning.call_args[0][0])
