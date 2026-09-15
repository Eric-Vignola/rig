"""Tests for ``Node.__rshift__(spec)`` -- output-only attribute declaration.

Mirror of ``Node.__lshift__(spec)`` (which adds a writable=True input
attr): ``node >> spec`` adds a writable=False output attr. The mnemonic
is "the operator points the way data flows" -- `<<` flows in, `>>`
flows out.

This is the rig-side of the v6.21 DSL alignment fix. The
matching bifrost-side semantic is `bf >> spec` declaring a graph
output port (a Maya output attr on the bifrostGraphShape, filled by
the bifrost compute engine).
"""

from __future__ import annotations

import maya.cmds as cmds
from rig._internal.node import Node
from rig.spec import Float, Matrix, Vector
from rig._tests._base import MayaTestCase


class TestNodeRshiftSpec(MayaTestCase):
    """``node >> spec`` declares an output-only (writable=False) attr."""

    def setUp(self) -> None:
        super().setUp()
        cmds.file(new=True, force=True)
        self.node = Node.create("transform", name="rshift_test")

    # ---------- Basic shape ----------

    def test_returns_plug(self) -> None:
        plug = self.node >> Float("my_output")
        # Plug-like -- has a string repr that contains the long name.
        self.assertIn("my_output", str(plug))

    def test_attribute_is_added(self) -> None:
        self.node >> Float("my_output")
        self.assertTrue(
            cmds.attributeQuery("my_output", node=str(self.node), exists=True)
        )

    # ---------- The point: writable=False ----------

    def test_attribute_is_writable_false(self) -> None:
        """The defining trait -- `>>` declares a non-writable attr."""
        self.node >> Float("out_attr")
        is_writable = cmds.attributeQuery(
            "out_attr", node=str(self.node), writable=True
        )
        self.assertFalse(is_writable)

    def test_lshift_attribute_is_writable_true(self) -> None:
        """Sanity: `<<` (the input form) keeps writable=True (default)."""
        self.node << Float("in_attr")
        is_writable = cmds.attributeQuery("in_attr", node=str(self.node), writable=True)
        self.assertTrue(is_writable)

    # ---------- Spec instance is NOT mutated (re-use safety) ----------

    def test_caller_spec_instance_untouched(self) -> None:
        """`>>` should shallow-copy the spec so the caller can reuse it
        for an unrelated `<<` declaration."""
        m_spec = Matrix("foo")
        self.node >> m_spec
        # Re-use the same spec on another node as a NORMAL (writable=True)
        # attribute -- proves the original spec wasn't permanently stamped
        # with writable=False.
        other = Node.create("transform", name="other_node")
        other << Matrix("foo")  # Use a fresh spec for the input side.
        is_writable = cmds.attributeQuery("foo", node=str(other), writable=True)
        self.assertTrue(is_writable)
        # And the original spec still has no writable kwarg set.
        self.assertNotIn("writable", m_spec.kargs)

    # ---------- Different spec types ----------

    def test_works_with_vector(self) -> None:
        self.node >> Vector("vec_out")
        self.assertTrue(
            cmds.attributeQuery("vec_out", node=str(self.node), exists=True)
        )
        self.assertFalse(
            cmds.attributeQuery("vec_out", node=str(self.node), writable=True)
        )

    def test_works_with_matrix(self) -> None:
        self.node >> Matrix("mat_out")
        self.assertTrue(
            cmds.attributeQuery("mat_out", node=str(self.node), exists=True)
        )
        self.assertFalse(
            cmds.attributeQuery("mat_out", node=str(self.node), writable=True)
        )

    # ---------- Existing semantics preserved ----------

    def test_rshift_none_still_returns_dgnode(self) -> None:
        """Pre-existing semantic: `node >> None` -> typed DGNode."""
        result = self.node >> None
        # The DGNode wrapper exposes a `.name` or string conversion that
        # matches the underlying node.
        self.assertEqual(str(result), str(self.node))

    def test_rshift_non_spec_non_none_raises(self) -> None:
        """Pre-existing semantic: anything else still raises."""
        with self.assertRaises(TypeError):
            _ = self.node >> 42