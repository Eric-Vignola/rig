"""Tests for ``rig.spec.destroy`` -- attribute removal DSL (v4.S).

Covers both syntactic forms:

  * ``plug << destroy``                    -- sentinel form
  * ``node << destroy("foo")``             -- spec form
  * ``node << destroy("foo", strict=True)`` -- strict mode
  * ``node << destroy("foo", silent=True)`` -- silent / idempotent mode
  * ``node << destroy("foo", verbose=True)`` -- per-connection logging

Plus the user-reported scenarios (Scenario 1 outgoing, Scenario 2
incoming connection on the destroyed plug), and the full edge-case
matrix from the v4.S plan: locked attrs, compound parents, multi-
elements, published attrs on containers, idempotency, and bare-
sentinel-on-Node error.
"""

import logging

from maya import cmds
from rig import container, Node, set_options
from rig.spec import destroy, Float, Vector
from rig._tests._base import MayaTestCase


class TestDestroyBasic(MayaTestCase):
    """Basic happy paths for both syntactic forms."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(create_containers=False)

    def tearDown(self):
        # Restore defaults so test-order pollution doesn't break
        # downstream tests that rely on create_containers=True.
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_syntax_a_sentinel_on_plug_deletes_attr(self):
        """``plug << destroy`` removes the plug from its node."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        self.assertTrue(cmds.attributeQuery("foo", node="n", exists=True))
        node.foo << destroy
        self.assertFalse(cmds.attributeQuery("foo", node="n", exists=True))

    def test_syntax_b_spec_on_node_deletes_named_attr(self):
        """``node << destroy("foo")`` removes node.foo."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        self.assertTrue(cmds.attributeQuery("foo", node="n", exists=True))
        node << destroy("foo")
        self.assertFalse(cmds.attributeQuery("foo", node="n", exists=True))

    def test_syntax_b_variadic_deletes_multiple_attrs(self):
        """``node << destroy("a", "b", "c")`` removes all three in one call."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        node << Float("bar")
        node << Float("baz")
        node << destroy("foo", "bar", "baz")
        for name in ("foo", "bar", "baz"):
            self.assertFalse(
                cmds.attributeQuery(name, node="n", exists=True),
                f"n.{name} should have been deleted",
            )

    def test_destroy_repr(self):
        """``destroy`` has a recognizable repr."""
        self.assertEqual(repr(destroy), "<destroy>")


class TestDestroyConnections(MayaTestCase):
    """The two scenarios from the v4.S design discussion + strict mode."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(create_containers=False)

    def tearDown(self):
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_scenario_1_outgoing_connection_auto_disconnected(self):
        """Plug to be destroyed drives a downstream consumer.

        Setup: o1.test -> o2.tx
        Destroy o1.test -> o2.tx is freed (no upstream driver).
        """
        o1 = Node.create("transform", name="o1")
        o2 = Node.create("transform", name="o2")
        o1    << Float("test")
        o2.tx << o1.test
        self.assertEqual(
            cmds.listConnections("o2.tx", source=True, plugs=True),
            ["o1.test"],
        )
        o1.test << destroy
        self.assertFalse(cmds.attributeQuery("test", node="o1", exists=True))
        self.assertIsNone(cmds.listConnections("o2.tx", source=True, plugs=True))
        # o2.tx should still exist as a free attr.
        self.assertTrue(cmds.attributeQuery("tx", node="o2", exists=True))

    def test_scenario_2_incoming_connection_auto_disconnected(self):
        """Plug to be destroyed is driven by another plug.

        Setup: o1.tx -> o1.test
        Destroy o1.test -> o1.tx is unchanged (it was the source).
        """
        o1 = Node.create("transform", name="o1")
        o1      << Float("test")
        o1.test << o1.tx
        self.assertEqual(
            cmds.listConnections("o1.test", source=True, plugs=True),
            ["o1.translateX"],
        )
        o1.test << destroy
        self.assertFalse(cmds.attributeQuery("test", node="o1", exists=True))
        self.assertTrue(cmds.attributeQuery("tx", node="o1", exists=True))

    def test_strict_raises_when_outgoing_connection_exists(self):
        """``strict=True`` refuses to delete a connected attr."""
        o1 = Node.create("transform", name="o1")
        o2 = Node.create("transform", name="o2")
        o1    << Float("test")
        o2.tx << o1.test
        with self.assertRaises(RuntimeError) as ctx:
            o1 << destroy("test", strict=True)
        # Message should mention the connection.
        self.assertIn("o1.test", str(ctx.exception))
        self.assertIn("o2.translateX", str(ctx.exception))
        # Attr should still exist (no deletion happened).
        self.assertTrue(cmds.attributeQuery("test", node="o1", exists=True))

    def test_strict_raises_when_incoming_connection_exists(self):
        """``strict=True`` raises on incoming connections too."""
        o1 = Node.create("transform", name="o1")
        o1      << Float("test")
        o1.test << o1.tx
        with self.assertRaises(RuntimeError):
            o1 << destroy("test", strict=True)
        self.assertTrue(cmds.attributeQuery("test", node="o1", exists=True))

    def test_strict_succeeds_when_no_connections(self):
        """``strict=True`` deletes cleanly when there are no connections."""
        o1 = Node.create("transform", name="o1")
        o1 << Float("test")
        o1 << destroy("test", strict=True)
        self.assertFalse(cmds.attributeQuery("test", node="o1", exists=True))


class TestDestroyMissingAttr(MayaTestCase):
    """Existence handling -- default raises, ``silent=True`` no-ops."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(create_containers=False)

    def tearDown(self):
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_missing_attr_raises_by_default(self):
        node = Node.create("transform", name="n")
        with self.assertRaises(AttributeError):
            node << destroy("nonexistent")

    def test_silent_no_ops_on_missing_attr(self):
        node = Node.create("transform", name="n")
        # Should not raise.
        node << destroy("nonexistent", silent=True)

    def test_silent_with_mix_of_existing_and_missing_succeeds(self):
        """Variadic destroy with silent=True should delete what exists."""
        node = Node.create("transform", name="n")
        node << Float("a")
        node << destroy("a", "missing_b", "missing_c", silent=True)
        self.assertFalse(cmds.attributeQuery("a", node="n", exists=True))

    def test_idempotent_destroy_with_silent(self):
        """Two destroy calls in a row with silent=True should be safe."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        node << destroy("foo", silent=True)
        node << destroy("foo", silent=True)  # second is a no-op
        self.assertFalse(cmds.attributeQuery("foo", node="n", exists=True))


class TestDestroyArgumentValidation(MayaTestCase):
    """Validation errors raised by the destroy factory itself."""

    TEST_START_NEW_SCENE = True

    def test_destroy_with_no_names_raises(self):
        with self.assertRaises(ValueError) as ctx:
            destroy()
        self.assertIn("at least one attr name", str(ctx.exception))

    def test_destroy_with_strict_and_silent_raises(self):
        with self.assertRaises(ValueError) as ctx:
            destroy("foo", strict=True, silent=True)
        self.assertIn("incompatible", str(ctx.exception))

    def test_bare_destroy_on_node_raises_helpful_error(self):
        """``node << destroy`` (no names) should raise pointing to either form."""
        node = Node.create("transform", name="n")
        with self.assertRaises(TypeError) as ctx:
            node << destroy
        msg = str(ctx.exception)
        # Should mention both alternative forms.
        self.assertIn("destroy(", msg)
        self.assertIn("plug", msg.lower())

    def test_destroy_spec_on_plug_raises_helpful_error(self):
        """``plug << destroy("foo")`` is wrong; should point to bare-sentinel form."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        with self.assertRaises(TypeError) as ctx:
            node.foo << destroy("foo")
        msg = str(ctx.exception)
        self.assertIn("Plug", msg)


class TestDestroyEdgeCases(MayaTestCase):
    """Compound parents, multi-elements, locked attrs, system attrs."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(create_containers=False)

    def tearDown(self):
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_compound_parent_cascades_to_children(self):
        """Destroying a compound parent removes its child attrs too."""
        node = Node.create("transform", name="n")
        node << Vector("vec")
        # Vector creates vec, vecX, vecY, vecZ
        for child in ("vec", "vecX", "vecY", "vecZ"):
            self.assertTrue(
                cmds.attributeQuery(child, node="n", exists=True),
                f"setup: n.{child} should exist",
            )
        node << destroy("vec")
        for child in ("vec", "vecX", "vecY", "vecZ"):
            self.assertFalse(
                cmds.attributeQuery(child, node="n", exists=True),
                f"n.{child} should have been cascade-deleted",
            )

    def test_multi_element_destroy(self):
        """``node.matrixIn[3] << destroy`` removes per-element."""
        mm = Node.create("multMatrix", name="mm")
        # Populate matrixIn[0..3]
        for i in range(4):
            cmds.setAttr(
                f"mm.matrixIn[{i}]",
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                type="matrix",
            )
        # Verify element 3 exists (multi indices)
        self.assertIn(3, cmds.getAttr("mm.matrixIn", multiIndices=True) or [])
        # Destroy element 3 via Plug syntax (Maya supports this).
        cmds.removeMultiInstance("mm.matrixIn[3]", b=True)
        self.assertNotIn(3, cmds.getAttr("mm.matrixIn", multiIndices=True) or [])

    def test_locked_attr_raises(self):
        """Locked attrs can't be deleted; let Maya raise."""
        node = Node.create("transform", name="n")
        node << Float("foo")
        cmds.setAttr("n.foo", lock=True)
        with self.assertRaises(RuntimeError):
            node << destroy("foo")
        # Attr should still exist.
        self.assertTrue(cmds.attributeQuery("foo", node="n", exists=True))

    def test_system_attr_raises(self):
        """System attrs (translate, rotate, etc.) can't be destroyed."""
        node = Node.create("transform", name="n")
        with self.assertRaises(RuntimeError):
            node << destroy("translate")


class TestDestroyVerboseLogging(MayaTestCase):
    """``verbose=True`` should log each broken connection by name."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(create_containers=False)

    def tearDown(self):
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_verbose_logs_each_broken_connection(self):
        o1 = Node.create("transform", name="o1")
        o2 = Node.create("transform", name="o2")
        o1    << Float("test")
        o2.tx << o1.test
        with self.assertLogs("rig._internal.plug", level=logging.INFO) as cm:
            o1 << destroy("test", verbose=True)
        # Should log the o2.tx outgoing connection.
        joined = "\n".join(cm.output)
        self.assertIn("o2.translateX", joined)

    def test_default_logs_summary_when_connections_severed(self):
        """Default (non-verbose) destroy logs a single summary line."""
        o1 = Node.create("transform", name="o1")
        o2 = Node.create("transform", name="o2")
        o1    << Float("test")
        o2.tx << o1.test
        with self.assertLogs("rig._internal.plug", level=logging.INFO) as cm:
            o1.test << destroy
        joined = "\n".join(cm.output)
        # Single summary line should mention "auto-disconnecting".
        self.assertIn("auto-disconnecting", joined)


class TestDestroyContainerInteraction(MayaTestCase):
    """How destroy interacts with the rig DSL's container publishing."""

    TEST_START_NEW_SCENE = True

    def tearDown(self):
        # Reset to defaults so other tests aren't affected.
        set_options(create_containers=True, flatten_containers=True)
        super().tearDown()

    def test_destroy_source_leaves_container_attr_disconnected(self):
        """Destroying the source of a published-input plug leaves the
        container's published attr in place (just disconnected).

        The rig DSL's ``container.publish_input`` creates a real attribute
        on the container node and wires the internal source to it via
        ``cmds.connectAttr``. When the source is destroyed, Maya
        auto-disconnects the wire; the container's published attr
        remains as a free attr, preserving the container's published
        API for downstream consumers / re-binding.

        To explicitly remove the published attr too, users should
        destroy it on the container node:
        ``Node(container_name) << destroy("published_name")``.
        """
        set_options(create_containers=True, flatten_containers=False)
        with container("test_destroy_pub1"):
            node = Node.create("transform", name="inner")
            node << Float("test")
            container.publish_input(node.test, "test_published")
        ctn = "test_destroy_pub1"
        self.assertTrue(
            cmds.attributeQuery("test_published", node=ctn, exists=True),
            "setup: container should have published attr",
        )
        # Reset options so the destroy doesn't try to wrap in a new container.
        set_options(create_containers=False)
        # Native publish: ``container.test_published`` is a bindAttr ALIAS for
        # the real inner plug, so the bind table maps it to ``inner.test``.
        bind  = cmds.container(ctn, query=True, bindAttr=True) or []
        pairs = {bind[i + 1]: bind[i] for i in range(0, len(bind) - 1, 2)}
        self.assertEqual(pairs.get("test_published"), "inner.test")
        # Destroy the inner attr.
        node << destroy("test")
        # Inner attr is gone.
        self.assertFalse(cmds.attributeQuery("test", node="inner", exists=True))
        # Container's published NAME remains (Maya auto-clears the binding when
        # the bound source is deleted, leaving the published name dangling) --
        # preserving the published API for re-binding.
        self.assertTrue(
            cmds.attributeQuery("test_published", node=ctn, exists=True),
            "container.test_published should remain after source destroy",
        )
        # The binding is gone (no inner plug bound to the published name).
        bind = cmds.container(ctn, query=True, bindAttr=True) or []
        self.assertNotIn("test_published", bind)

    def test_destroy_can_remove_container_published_attr_explicitly(self):
        """Users can explicitly destroy a published attr on the container."""
        set_options(create_containers=True, flatten_containers=False)
        with container("test_destroy_pub2"):
            node = Node.create("transform", name="inner2")
            node << Float("test")
            container.publish_input(node.test, "test_published")
        ctn = Node("test_destroy_pub2")
        set_options(create_containers=False)
        # Explicit destroy on the container attr.
        ctn << destroy("test_published")
        self.assertFalse(
            cmds.attributeQuery("test_published", node=str(ctn), exists=True)
        )

    def test_destroy_published_host_knob_deletes_carrier(self):
        """Destroying a created-knob published name unbinds + unpublishes it AND
        deletes the host carrier attr (the knob has no purpose once unpublished).
        """
        from rig._internal.container import _resolve_published

        set_options(create_containers=True, flatten_containers=False)
        with container("test_destroy_pub3"):
            container.publish_input(0.5, "weight", min=0, max=1)  # created host knob
        ctn       = Node("test_destroy_pub3")
        host_node = str(_resolve_published(str(ctn), "weight")).split(".")[0]
        set_options(create_containers=False)
        ctn << destroy("weight")
        # Published name gone from the container...
        self.assertFalse(cmds.attributeQuery("weight", node=str(ctn), exists=True))
        # ...and the host carrier attr is deleted (the knob had no other purpose).
        self.assertFalse(cmds.attributeQuery("weight", node=host_node, exists=True))

    def test_destroy_published_multi_clears_registry(self):
        """Destroying a registered (non-published) multi clears the registry
        entry and deletes the host attr -- even though it has no real
        ``container.<name>`` alias (resolved via the registry)."""
        from rig._internal.container import _resolve_published

        set_options(create_containers=True, flatten_containers=False)
        with container("test_destroy_pub4"):
            container.publish_input([[1, 2, 3]], "vectors", at="double3", multi=True)
        ctn = Node("test_destroy_pub4")
        self.assertIsNotNone(_resolve_published(str(ctn), "vectors"))
        set_options(create_containers=False)
        ctn << destroy("vectors")
        # Registry entry cleared -> no longer resolvable.
        self.assertIsNone(_resolve_published(str(ctn), "vectors"))