"""Tests for ``rig.cleanup`` (v2.E garbage collection).

Covers the four-filter safety model:
    1. Per-node ``__rig__`` ownership tag.
    2. ``_GC_ELIGIBLE_TYPES`` whitelist.
    3. Downstream-connection check (iterative leaf-orphan deletion).
    4. Container empty-promotion + single ``cmds.delete`` call.

Also covers ``cleanup_on_exit`` (per-block + global override) and
memoize cache pruning.
"""

from maya import cmds
from rig import cleanup, container, Node, prune_memoize_caches, set_options
from rig._internal.container import (
    _GC_ELIGIBLE_TYPES,
    _RIG_TAG,
    ContainerOptions,
)
from rig._tests._base import MayaTestCase


class TestCleanupOwnership(MayaTestCase):
    """Filter 1 -- only nodes carrying the ``__rig__`` tag are eligible."""

    TEST_START_NEW_SCENE = True

    def test_rig_created_math_node_is_tagged(self):
        with container("ctn"):
            n = Node.create("multiplyDivide", name="mul1")
        self.assertTrue(
            cmds.attributeQuery(_RIG_TAG, node=n.name, exists=True),
            f"{n.name} missing __rig__ tag",
        )

    def test_rig_created_transform_is_NOT_tagged(self):
        # Transforms are user-facing -- never tagged, never GC-eligible.
        with container("ctn"):
            n = Node.create("transform", name="ctrl1")
        self.assertFalse(
            cmds.attributeQuery(_RIG_TAG, node=n.name, exists=True),
            f"{n.name} should not be tagged (it's a transform)",
        )

    def test_user_created_node_is_NOT_tagged(self):
        # If the user makes a node OUTSIDE the rig DSL, it has no tag.
        cmds.createNode("multiplyDivide", name="user_mul")
        self.assertFalse(cmds.attributeQuery(_RIG_TAG, node="user_mul", exists=True))
        # cleanup() should leave it alone even though it has no consumers.
        cleanup()
        self.assertTrue(cmds.objExists("user_mul"))


class TestCleanupTypeWhitelist(MayaTestCase):
    """Filter 2 -- only types in the whitelist are GC-eligible."""

    TEST_START_NEW_SCENE = True

    def test_transform_in_container_NEVER_deleted(self):
        # Even if we explicitly add a transform to a rig container with NO
        # downstream connections, it must NOT be deleted.
        with container("ctn"):
            ctrl = Node.create("transform", name="elbow_ctrl")
        self.assertTrue(cmds.objExists("elbow_ctrl"))
        report = cleanup(container="ctn")
        self.assertTrue(
            cmds.objExists("elbow_ctrl"),
            "transform deleted by cleanup -- should be untouched",
        )
        # Container itself stays (still has elbow_ctrl).
        self.assertEqual(report["deleted_containers"], [])

    def test_orphan_math_node_is_deleted(self):
        with container("ctn"):
            mul = Node.create("multiplyDivide", name="orphan_mul")
        # No connections from orphan_mul.
        report = cleanup(container="ctn")
        self.assertFalse(
            cmds.objExists("orphan_mul"),
            "orphan multiplyDivide should be deleted",
        )
        self.assertIn("orphan_mul", report["by_owner"].get("ctn", []))


class TestCleanupConnectionCheck(MayaTestCase):
    """Filter 3 -- only nodes with no downstream consumers are deleted."""

    TEST_START_NEW_SCENE = True

    def test_connected_math_node_is_preserved(self):
        # mul1 -> mul2 -> ty(some transform).  mul1 has a consumer.
        with container("ctn"):
            ctrl = Node.create("transform",      name="ctrl1")
            mul1 = Node.create("multiplyDivide", name="mul1")
            mul2 = Node.create("multiplyDivide", name="mul2")
        cmds.connectAttr("mul1.outputX", "mul2.input1X")
        cmds.connectAttr("mul2.outputX", "ctrl1.tx")
        cleanup(container="ctn")
        # All three live.
        self.assertTrue(cmds.objExists("mul1"))
        self.assertTrue(cmds.objExists("mul2"))
        self.assertTrue(cmds.objExists("ctrl1"))

    def test_iterative_chain_collapses(self):
        # Chain: mul1 -> mul2 -> mul3, dangling at mul3 (no consumer).
        # All three should iteratively delete.
        with container("ctn"):
            mul1 = Node.create("multiplyDivide", name="mul1")
            mul2 = Node.create("multiplyDivide", name="mul2")
            mul3 = Node.create("multiplyDivide", name="mul3")
        cmds.connectAttr("mul1.outputX", "mul2.input1X")
        cmds.connectAttr("mul2.outputX", "mul3.input1X")
        cleanup(container="ctn")
        self.assertFalse(cmds.objExists("mul1"))
        self.assertFalse(cmds.objExists("mul2"))
        self.assertFalse(cmds.objExists("mul3"))


class TestCleanupSceneWide(MayaTestCase):
    """Cleanup MUST work even when ``create_containers=False``."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Reset to defaults at start of each test.
        set_options(create_containers=True, cleanup_on_exit=False)

    def tearDown(self):
        set_options(create_containers=True, cleanup_on_exit=False)
        super().tearDown()

    def test_orphan_at_scene_root_is_caught(self):
        # No `with container():` block -- node sits at scene root.
        # But it's still rig-created, so it carries the tag.
        set_options(create_containers=False)
        with container("scope"):
            Node.create("multiplyDivide", name="root_orphan")
        # No real container was made; node is at scene root.
        self.assertTrue(cmds.objExists("root_orphan"))
        report = cleanup()
        self.assertFalse(cmds.objExists("root_orphan"))
        self.assertIn(
            "root_orphan",
            report["by_owner"].get("(scene root)", []),
        )

    def test_no_container_block_at_all(self):
        # No with-block whatsoever. Direct Node.create.
        set_options(create_containers=False)
        Node.create("multiplyDivide", name="bare_mul")
        cleanup()
        self.assertFalse(cmds.objExists("bare_mul"))


class TestCleanupEmptyContainerPromotion(MayaTestCase):
    """When all members of a container are GC'd, the container itself is too."""

    TEST_START_NEW_SCENE = True

    def test_empty_container_deleted(self):
        with container("ephemeral"):
            Node.create("multiplyDivide", name="m1")
            Node.create("multiplyDivide", name="m2")
        # No consumers -- both will GC. Container becomes empty.
        report = cleanup(container="ephemeral")
        self.assertFalse(cmds.objExists("ephemeral"))
        self.assertIn("ephemeral", report["deleted_containers"])

    def test_partial_container_NOT_deleted(self):
        # Half the container is connected; half is orphan.
        with container("partial"):
            ctrl = Node.create("transform", name="surviving_ctrl")
            mul  = Node.create("multiplyDivide", name="orphan_mul")
        cleanup(container="partial")
        # Container still exists (transform survived).
        self.assertTrue(cmds.objExists("partial"))
        self.assertTrue(cmds.objExists("surviving_ctrl"))
        self.assertFalse(cmds.objExists("orphan_mul"))


class TestCleanupDryRun(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dry_run_reports_but_does_not_delete(self):
        with container("ctn"):
            Node.create("multiplyDivide", name="mul1")
            Node.create("multiplyDivide", name="mul2")
        report = cleanup(container="ctn", dry_run=True)
        # Nothing was actually deleted.
        self.assertTrue(cmds.objExists("mul1"))
        self.assertTrue(cmds.objExists("mul2"))
        self.assertTrue(cmds.objExists("ctn"))
        # But the report tells us what WOULD go.
        self.assertTrue(report["dry_run"])
        self.assertIn("mul1", report["by_owner"].get("ctn", []))
        self.assertIn("mul2", report["by_owner"].get("ctn", []))


class TestCleanupOnExit(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(cleanup_on_exit=False)

    def tearDown(self):
        set_options(cleanup_on_exit=False)
        super().tearDown()

    def test_per_block_cleanup_on_exit(self):
        with container("ctn", cleanup_on_exit=True):
            Node.create("multiplyDivide", name="mul1")
        # On exit, mul1 (orphan) should be gone.
        self.assertFalse(cmds.objExists("mul1"))

    def test_global_cleanup_on_exit(self):
        set_options(cleanup_on_exit=True)
        with container("ctn"):
            Node.create("multiplyDivide", name="mul2")
        self.assertFalse(cmds.objExists("mul2"))

    def test_per_block_overrides_global_off(self):
        set_options(cleanup_on_exit=True)
        with container("ctn", cleanup_on_exit=False):
            Node.create("multiplyDivide", name="mul3")
        # Per-block False wins -> mul3 survives.
        self.assertTrue(cmds.objExists("mul3"))

    def test_default_is_no_cleanup(self):
        # Without any flag, orphans persist after the with-block.
        with container("ctn"):
            Node.create("multiplyDivide", name="mul4")
        self.assertTrue(cmds.objExists("mul4"))


class TestContainerCleanupMethod(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_container_cleanup_method(self):
        with container("ctn") as ctn:
            Node.create("multiplyDivide", name="mul1")
        self.assertTrue(cmds.objExists("mul1"))
        ctn.cleanup()
        self.assertFalse(cmds.objExists("mul1"))


class TestMemoizeCachePrune(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_prune_drops_stale_entries(self):
        from rig import constant

        # Use the constant memoize cache.
        plug      = constant(42.0)
        node_name = str(plug).split(".")[0]
        # Cache should now have at least one entry.
        cache  = constant._cache
        before = len(cache)
        self.assertGreater(before, 0)

        # Delete the node -- entry becomes stale.
        cmds.delete(node_name)
        dropped = prune_memoize_caches()
        self.assertGreater(dropped, 0, "no entries dropped after delete")
        self.assertLess(len(cache), before, "cache size should shrink after prune")


class TestEligibleTypesWhitelist(MayaTestCase):
    """Sanity-check the whitelist contents."""

    def test_common_math_types_listed(self):
        for t in (
            "multiplyDivide",
            "plusMinusAverage",
            "condition",
            "decomposeMatrix",
            "composeMatrix",
            "blendColors",
            "unitConversion",
        ):
            self.assertIn(t, _GC_ELIGIBLE_TYPES, f"{t} missing from whitelist")

    def test_user_facing_types_excluded(self):
        # If any of these slip into the whitelist it's a SAFETY BUG.
        for t in (
            "transform",
            "joint",
            "mesh",
            "nurbsCurve",
            "nurbsSurface",
            "locator",
            "camera",
            "directionalLight",
            "objectSet",
            "displayLayer",
            "shadingEngine",
            "lambert",
            "container",
        ):
            self.assertNotIn(
                t,
                _GC_ELIGIBLE_TYPES,
                f"{t} must NOT be in whitelist",
            )