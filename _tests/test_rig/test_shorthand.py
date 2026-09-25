"""Tests for ``rig._internal.shorthand`` -- type-aware translation on connect."""

from collections import Counter
from unittest import mock

from maya import cmds
from rig import Node, set_options
from rig._internal.container import ContainerOptions
from rig._internal.shorthand import _Facts, shorthand
from rig._internal.types import _is_compound, _is_matrix, _is_quaternion, _is_vector
from rig._tests._base import MayaTestCase
from rig.nodetypes._base import Attribute


def _node_type_of(plug_or_node):
    """Return the Maya type of the node owning ``plug_or_node``."""
    s = str(plug_or_node)
    return cmds.nodeType(s.split(".")[0])


class TestMatrixToTransform(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_matrix_to_translate_inserts_decompose(self):
        a = Node.create("transform", name="src")
        b = Node.create("transform", name="dst")
        b.t << a.wm
        # b.t should have an incoming connection from a decomposeMatrix.
        connections = cmds.listConnections("dst.t", source=True, destination=False)
        self.assertTrue(connections)
        types = [cmds.nodeType(c) for c in connections]
        self.assertIn("decomposeMatrix", types)

    def test_matrix_to_rotate_inserts_decompose(self):
        a = Node.create("transform", name="src")
        b = Node.create("transform", name="dst")
        b.r << a.wm
        connections = cmds.listConnections("dst.r", source=True, destination=False)
        self.assertTrue(connections)
        types = [cmds.nodeType(c) for c in connections]
        self.assertIn("decomposeMatrix", types)


class TestVectorToMatrix(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_vector_to_matrix_inserts_compose(self):
        a = Node.create("transform", name="src")
        b = Node.create("network", name="dst_net")
        from rig.spec import Matrix

        b       << Matrix("xform")
        b.xform << a.t
        connections = cmds.listConnections(
            "dst_net.xform", source=True, destination=False
        )
        self.assertTrue(connections)
        types = [cmds.nodeType(c) for c in connections]
        self.assertIn("composeMatrix", types)


class TestTransformToMatrix(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_bare_node_to_matrix_attr_raises_with_helpful_message(self):
        """v3.R Pattern D: ``b.xform << a`` (a is bare transform Node)
        intentionally raises rather than auto-promoting. The user has
        to choose between local (`a.matrix`) and world (`a.wm`) space --
        auto-promoting would hide which one is being used.

        The error message must mention BOTH explicit alternatives.
        """
        a = Node.create("transform", name="src")
        b = Node.create("network", name="dst_net")
        from rig.spec import Matrix

        b << Matrix("xform")
        with self.assertRaises(TypeError) as ctx:
            b.xform << a
        msg = str(ctx.exception)
        self.assertIn(".wm",       msg)
        self.assertIn(".matrix",   msg)
        self.assertIn("bare Node", msg)

    def test_explicit_worldmatrix_to_matrix_attr_works(self):
        """v3.R Pattern D: explicit ``b.xform << a.wm`` works."""
        a = Node.create("transform", name="src")
        b = Node.create("network", name="dst_net")
        from rig.spec import Matrix

        b       << Matrix("xform")
        b.xform << a.wm[0]
        connections = cmds.listConnections(
            "dst_net.xform", source=True, destination=False
        )
        self.assertTrue(connections)

    def test_explicit_localmatrix_to_matrix_attr_works(self):
        """v3.R Pattern D: explicit ``b.xform << a.matrix`` works."""
        a = Node.create("transform", name="src")
        b = Node.create("network", name="dst_net")
        from rig.spec import Matrix

        b       << Matrix("xform")
        b.xform << a.matrix
        connections = cmds.listConnections(
            "dst_net.xform", source=True, destination=False
        )
        self.assertTrue(connections)


class TestShorthandToggle(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_shorthand_can_be_disabled(self):
        original = ContainerOptions.use_shorthand
        try:
            set_options(use_shorthand=False)
            a = Node.create("transform", name="src")
            b = Node.create("transform", name="dst")
            # With shorthand off, this attempts a direct matrix->vector
            # connect. The attribute types are incompatible, so the
            # underlying _inject_value will try a fan-out; the connection
            # may or may not succeed but the key point is no
            # decomposeMatrix gets inserted.
            try:
                b.t << a.wm
            except Exception:
                pass
            connections = cmds.listConnections("dst.t", source=True, destination=False)
            types       = [cmds.nodeType(c) for c in connections] if connections else []
            self.assertNotIn("decomposeMatrix", types)
        finally:
            set_options(use_shorthand=original)


class TestMatrixToMatrixViaTransform(MayaTestCase):
    """Regression: ``node1.matrix << node2.matrix`` must route through
    decomposeMatrix shorthand without raising
    ``TypeError: Plug is not a compound parent`` on rotate-order lookup.
    """

    TEST_START_NEW_SCENE = True

    def test_matrix_to_matrix_inserts_decompose(self):
        a = Node.create("transform", name="src_xform")
        b = Node.create("transform", name="dst_xform")
        b.matrix << a.matrix
        # b.t / b.r / b.s should each have a decomposeMatrix in the
        # incoming chain.
        for channel in ("t", "r", "s"):
            connections = (
                cmds.listConnections(
                    f"dst_xform.{channel}", source=True, destination=False
                )
                or []
            )
            types = [cmds.nodeType(c) for c in connections]
            self.assertIn(
                "decomposeMatrix",
                types,
                msg=f"dst_xform.{channel} missing decomposeMatrix in driver chain",
            )


class TestShorthandSameTypePassThrough(MayaTestCase):
    """Same-type connections should NOT trigger shorthand."""

    TEST_START_NEW_SCENE = True

    def test_translate_to_translate_no_decompose(self):
        a = Node.create("transform", name="src")
        b = Node.create("transform", name="dst")
        b.t << a.t
        # v3.* shorthand: same-type compound connect uses a single
        # ``src.t -> dst.t`` connection (no per-channel fan-out, no
        # decomposeMatrix in the chain). Query the compound to confirm.
        connections = cmds.listConnections("dst.t", source=True, destination=False)
        self.assertTrue(connections)
        for c in connections:
            self.assertNotEqual(cmds.nodeType(c), "decomposeMatrix")


class TestShorthandFacts(MayaTestCase):
    """``shorthand`` answers its type tests from one ``_Facts`` per operand,
    which must agree with the ``rig._internal.types`` predicates."""

    TEST_START_NEW_SCENE = True

    def _operands(self):
        cmds.loadPlugin("matrixNodes", quiet=True)
        cmds.loadPlugin("quatNodes", quiet=True)
        xform     = Node.create("transform", name="xform")
        decompose = Node(cmds.createNode("decomposeMatrix"))
        quat_prod = Node(cmds.createNode("quatProd"))
        vec_ch    = Node(cmds.createNode("choice", name="vec_ch"))
        mtx_ch    = Node(cmds.createNode("choice", name="mtx_ch"))
        open_ch   = Node(cmds.createNode("choice", name="open_ch"))
        cmds.connectAttr(str(xform.translate), f"{vec_ch}.input[0]")
        cmds.connectAttr(f"{xform}.worldMatrix[0]", f"{mtx_ch}.input[0]")

        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="vec", dt="double3")
        cmds.addAttr(net, ln="mtx", dt="matrix")
        cmds.addAttr(net, ln="quat", at="compound", numberOfChildren=4)
        for axis in "XYZW":
            cmds.addAttr(net, ln=f"quat{axis}", at="double", p="quat")
        cmds.addAttr(net, ln="dbls", at="double", multi=True)
        cmds.setAttr(f"{net}.dbls[0]", 1.0)
        net = Node(net)

        plugs = [
            xform.matrix,
            xform.worldMatrix[0],
            xform.worldMatrix,
            xform.t,
            xform.tx,
            xform.r,
            xform.ro,
            decompose.outputQuat,
            decompose.outputTranslate,
            decompose.inputMatrix,
            quat_prod.outputQuat,
            vec_ch.output,
            mtx_ch.output,
            open_ch.output,
            net.vec,
            net.mtx,
            net.quat,
            net.quatW,
            net.dbls,
            net.dbls[0],
        ]
        return plugs + [1.5, 2, [1, 2, 3], (0, 0, 0, 1), "xform.t", None, xform]

    def test_shorthand_facts_match_predicates(self):
        for operand in self._operands():
            with self.subTest(operand=repr(operand)):
                facts = _Facts(operand)
                self.assertIs(facts.is_matrix, _is_matrix(operand))
                self.assertIs(facts.is_compound, _is_compound(operand))
                self.assertIs(facts.is_quaternion, _is_quaternion(operand))
                self.assertIs(facts.is_vector, _is_vector(operand))

    def test_shorthand_facts_query_data_type_once(self):
        data_type = Attribute.data_type
        queries   = Counter()

        def counted(attr):
            queries[id(attr)] += 1
            return data_type.fget(attr)

        operands = [op for op in self._operands() if isinstance(op, Attribute)]
        with mock.patch.object(Attribute, "data_type", property(counted)):
            for operand in operands:
                queries.clear()
                facts = _Facts(operand)
                for _ in range(2):
                    facts.is_matrix
                    facts.is_quaternion
                    facts.is_vector
                    facts.is_compound
                with self.subTest(operand=str(operand)):
                    self.assertLessEqual(queries[id(operand)], 1)

    def test_shorthand_non_attribute_source_returns_false(self):
        dst = Node.create("transform", name="dst")
        for src in (1.5, 3, [1, 2, 3], (0, 0, 0, 1), "xform.t", None, dst):
            with self.subTest(src=repr(src)):
                with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
                    self.assertIs(shorthand(src, dst.t), False)
                    self.assertIs(shorthand(src, dst.worldMatrix[0]), False)
                self.assertEqual(probe.call_count, 0)