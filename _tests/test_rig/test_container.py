"""Tests for ``rig._internal.container``.

Covers visual-grouping, flattening, name-prefix breadcrumbs, the
``rig.set_options`` / ``rig.get_options`` toggles, and the dormant
publish stubs (which return ``plug`` unchanged in v1).
"""

import gc

from maya import cmds
from rig import Container, container, get_options, Node, set_options
from rig._internal.container import _StackFrame, ContainerOptions
from rig._tests._base import MayaTestCase


class TestContainerCreation(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_top_level_creates_container(self):
        before = cmds.ls(type="container") or []
        with container("test1") as ctn:
            self.assertIsInstance(ctn, Container)
        after = cmds.ls(type="container") or []
        self.assertEqual(len(after) - len(before), 1)

    def test_nested_default_flattens(self):
        before = cmds.ls(type="container") or []
        with container("outer"):
            with container("inner") as ctn:
                # Nested ctn is None -- flattened.
                self.assertIsNone(ctn)
        after = cmds.ls(type="container") or []
        # Only ONE container created -- the inner was flattened.
        self.assertEqual(len(after) - len(before), 1)

    def test_nested_preserve_creates_real_container(self):
        before = cmds.ls(type="container") or []
        with container("outer"):
            with container("inner", preserve=True) as ctn:
                self.assertIsInstance(ctn, Container)
        after = cmds.ls(type="container") or []
        # Both outer and inner created.
        self.assertEqual(len(after) - len(before), 2)

    def test_enabled_false_creates_no_container(self):
        before = cmds.ls(type="container") or []
        with container("test", enabled=False) as ctn:
            self.assertIsNone(ctn)
        after = cmds.ls(type="container") or []
        self.assertEqual(len(after), len(before))

    def test_options_disable_creation_globally(self):
        original = ContainerOptions.create_containers
        try:
            set_options(create_containers=False)
            before = cmds.ls(type="container") or []
            with container("test") as ctn:
                self.assertIsNone(ctn)
            after = cmds.ls(type="container") or []
            self.assertEqual(len(after), len(before))
        finally:
            set_options(create_containers=original)


class TestContainerNodeAddition(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_node_added_to_container(self):
        with container("ctn1") as ctn:
            node = Node.create("transform", name="cube1")
        members = cmds.container(str(ctn), q=True, nodeList=True) or []
        self.assertIn(str(node), members)


class TestContainerNamePrefix(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_flattened_block_prefixes_names(self):
        with container("outer"):
            with container("inner"):
                node = Node.create("transform", name="cube1")
        # Inner was flattened -- its node should be prefixed `inner_*`.
        self.assertIn("inner_", str(node))

    def test_top_level_no_prefix(self):
        with container("outer"):
            node = Node.create("transform", name="cube1")
        # Top-level node -- no prefix.
        self.assertEqual(str(node), "cube1")


class TestContainerStack(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_stack_empty_outside_with_block(self):
        self.assertFalse(container.is_active)
        self.assertEqual(container.stack, [])

    def test_stack_populated_inside_with_block(self):
        with container("ctn1"):
            self.assertTrue(container.is_active)
            self.assertEqual(len(container.stack), 1)
        self.assertFalse(container.is_active)

    def test_stack_pops_on_exception(self):
        try:
            with container("ctn_err"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Stack must be empty after exception bubble-up.
        self.assertFalse(container.is_active)


class TestOptionsReturnsDict(MayaTestCase):
    """v2 base: ``get_options()`` returns a snapshot dict of current
    settings; ``set_options(...)`` mutates them. Mirrors numpy's
    ``np.set_printoptions`` / ``np.get_printoptions`` convention.

    Use cases:
      * ``vals = get_options()`` -- introspect current settings
      * ``get_options()`` at the REPL -- auto-prints the dict
      * ``set_options(...)`` -- mutate (returns None per numpy convention)
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Capture current state so we can restore in tearDown.
        self._saved = get_options()

    def tearDown(self):
        super().tearDown()
        # Restore (only the keys present at this diff level).
        set_options(
            create_containers = self._saved["create_containers"],
            use_shorthand     = self._saved["use_shorthand"],
            skip_selection    = self._saved["skip_selection"],
        )

    def test_get_options_returns_snapshot_dict(self):
        result = get_options()
        self.assertIsInstance(result, dict)
        self.assertIn("create_containers", result)
        self.assertIn("use_shorthand",     result)
        self.assertIn("skip_selection",    result)

    def test_set_options_returns_none(self):
        # Numpy parity: set_options is a pure setter.
        result = set_options(create_containers=False)
        self.assertIsNone(result)

    def test_set_then_get_reflects_change(self):
        set_options(use_shorthand=False)
        snapshot = get_options()
        self.assertEqual(snapshot["use_shorthand"], False)


class TestScopeKeyHardening(MayaTestCase):
    """``_StackFrame._scope_key`` must stay unique across separate builds.

    The memoize layer keys on the outermost frame's ``_scope_key``. When
    that key embedded ``id(self)`` (a CPython address recycled after the
    popped frame was GC'd), two sequential builds sharing ``(name, depth)``
    -- e.g. two ``create_rail`` calls under ``with container("railNode1")``
    -- could be handed the same scope key, so the second build's memoize
    lookups collided with the first's stale entries. A process-monotonic
    uid must never repeat, even when CPython recycles the frame address.
    """

    def test_recycled_frame_address_yields_distinct_scope_key(self):
        seen = {}
        for _ in range(5000):
            frame    = _StackFrame("railNode1", None, False, True, 0)
            frame_id = id(frame)
            key      = frame._scope_key
            if frame_id in seen:
                # CPython recycled the address. With an id()-based scope
                # key this frame's key equals the prior frame's key
                # (collision); with a monotonic uid it must differ.
                self.assertNotEqual(
                    key,
                    seen[frame_id],
                    "scope key collided after the frame address was recycled",
                )
                return
            seen[frame_id] = key
            del frame
            gc.collect()
        self.fail("CPython never recycled a frame address in 5000 iterations")

    def test_scope_key_uid_strictly_increases(self):
        first  = _StackFrame("x", None, False, True, 0)._scope_key
        second = _StackFrame("x", None, False, True, 0)._scope_key
        # Shape stays (name, depth, uid); the uid component must advance.
        self.assertEqual(first[:2], second[:2])
        self.assertLess(first[2], second[2])