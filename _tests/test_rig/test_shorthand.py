"""Tests for ``rig._internal.shorthand`` -- type-aware translation on connect."""

from maya import cmds
from rig import Node, set_options
from rig._internal.container import ContainerOptions
from rig._tests._base import MayaTestCase


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