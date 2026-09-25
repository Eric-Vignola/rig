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

def _uuid(node) -> str:
    return cmds.ls(str(node), uuid=True)[0]


class TestFrameMembers(MayaTestCase):
    """``_StackFrame.members`` records every added node's uuid on every frame,
    once per frame, in add order; a flattened frame's exit then appends its
    whole list to the parent, duplicates included. The golden lists below
    replay that algorithm from ``cmds.ls(uuid=True)``.
    """

    TEST_START_NEW_SCENE = True

    def test_frame_members_order_and_duplicates_unchanged(self):
        golden: list = []

        def record(*nodes):
            for node in nodes:
                uid = _uuid(node)
                for members in golden:
                    if uid not in members:
                        members.append(uid)

        def check(frames):
            for frame, members in zip(frames, golden):
                self.assertEqual(frame.members, members)
                self.assertEqual(frame._member_set, set(members))

        outside = Node(cmds.createNode("transform", name="outside"))
        with container("outer"):
            outer = container.stack[-1]
            golden.append([])
            a = Node.create("multiplyDivide", name="a")
            record(a)
            with container("mid"):
                mid = container.stack[-1]
                golden.append([])
                b = Node.create("plusMinusAverage", name="b")
                record(b)
                container.add(a)
                record(a)
                with container("inner"):
                    inner = container.stack[-1]
                    golden.append([])
                    c = Node.create("condition", name="c")
                    container.add([outside, str(b)])
                    record(c, outside, b)
                    check([outer, mid, inner])
                golden[1].extend(golden.pop())
                check([outer, mid, inner])
                d = Node.create("multiplyDivide", name="d")
                record(d)
            golden[0].extend(golden.pop())
            check([outer, mid])
            container.add(d)
            record(d)
            check([outer])
        # The flattened exits leave duplicates on the parents.
        self.assertGreater(len(outer.members), len(set(outer.members)))
        self.assertEqual(outer.members, golden[0])

    def test_add_mixed_node_and_string_inputs(self):
        node_a = Node(cmds.createNode("transform", name="mixedA"))
        name_b = cmds.createNode("transform", name="mixedB")
        with container("mixed") as ctn:
            frame = container.stack[-1]
            container.add([node_a, name_b])
            self.assertEqual(frame.members, [_uuid(node_a), _uuid(name_b)])
        members = cmds.container(str(ctn), q=True, nodeList=True) or []
        self.assertIn(str(node_a), members)
        self.assertIn(name_b,      members)

    def test_node_input_skips_name_lookup(self):
        from rig._internal import container as container_module

        original = container_module._node_uuid
        calls    = []

        def counting(name):
            calls.append(name)
            return original(name)

        container_module._node_uuid = counting
        try:
            with container("lookups"):
                frame = container.stack[-1]
                node  = Node.create("multiplyDivide", name="lookupA")
                self.assertEqual(calls, [])
                name_b = cmds.createNode("transform", name="lookupB")
                container.add(name_b)
                self.assertEqual(calls, [name_b])
        finally:
            container_module._node_uuid = original
        self.assertEqual(frame.members, [_uuid(node), _uuid(name_b)])


def _attribute_query(attr, node):
    """``("ok", bool)`` or ``("err", type, message)`` of ``attributeQuery``."""
    try:
        return ("ok", bool(cmds.attributeQuery(attr, node=node, exists=True)))
    except Exception as e:
        return ("err", type(e), str(e))


class TestTagQueries(MayaTestCase):
    """``_is_host`` / ``_is_rig_owned`` answer with the API and give the same
    result as ``attributeQuery(exists)``, aliases and failures included."""

    TEST_START_NEW_SCENE = True

    def _build(self):
        net = cmds.createNode("network", name="tq_net")
        cmds.addAttr(net, longName="__rl_host__", attributeType="bool", hidden=True)
        cmds.addAttr(net, longName="__rig__", attributeType="bool", hidden=True)
        xf = cmds.createNode("transform", name="tq_xf")
        cmds.addAttr(xf, longName="__rl_host__", attributeType="bool", hidden=True)
        cmds.addAttr(xf, longName="__rig__", attributeType="bool", hidden=True)
        cmds.createNode("transform", name="tq_plain")
        al = cmds.createNode("transform", name="tq_alias")
        cmds.addAttr(al, longName="foo", attributeType="double")
        cmds.aliasAttr("__rl_host__", f"{al}.foo")
        cmds.aliasAttr("__rig__", f"{al}.translateX")
        g1 = cmds.createNode("transform", name="tq_g1")
        g2 = cmds.createNode("transform", name="tq_g2")
        cmds.createNode("transform", name="tq_dup", parent=g1)
        cmds.createNode("transform", name="tq_dup", parent=g2)
        cmds.addAttr("|tq_g1|tq_dup", longName="__rig__", attributeType="bool")
        return [
            "tq_net", "tq_xf", "tq_plain", "tq_alias", "tq_g1|tq_dup",
            "tq_g2|tq_dup", "tq_dup", "tq_*", "nonexistent", "", "tq_net.foo",
        ]

    def test_is_host_matches_attribute_query(self):
        from rig._internal import container as _C

        for name in self._build():
            legacy = _attribute_query(_C._HOST_MARKER, name)
            self.assertEqual(
                _C._is_host(name), legacy == ("ok", True), repr(name)
            )
        self.assertTrue(_C._is_host("tq_net"))
        self.assertTrue(_C._is_host("tq_xf"))
        self.assertTrue(_C._is_host("tq_alias"))
        self.assertFalse(_C._is_host("tq_plain"))
        self.assertFalse(_C._is_host("nonexistent"))

    def test_is_rig_owned_matches_attribute_query(self):
        from rig._internal import container as _C

        for name in self._build():
            legacy = _attribute_query(_C._RIG_TAG, name)
            try:
                new = ("ok", _C._is_rig_owned(name))
            except Exception as e:
                new = ("err", type(e), str(e))
            if legacy[0] == "err" and legacy[1] in (RuntimeError, ValueError):
                legacy = ("ok", False)
            self.assertEqual(new, legacy, repr(name))
        self.assertTrue(_C._is_rig_owned("tq_xf"))
        self.assertTrue(_C._is_rig_owned("tq_alias"))
        self.assertTrue(_C._is_rig_owned("tq_g1|tq_dup"))
        self.assertFalse(_C._is_rig_owned("tq_g2|tq_dup"))

    def test_tag_queries_skip_attribute_query_for_plain_names(self):
        from unittest import mock

        from rig._internal import container as _C

        self._build()
        with mock.patch.object(
            cmds, "attributeQuery", side_effect=cmds.attributeQuery
        ) as spy:
            for name in ("tq_net", "tq_xf", "tq_plain", "tq_alias", "tq_g1|tq_dup"):
                _C._is_host(name)
                _C._is_rig_owned(name)
        self.assertEqual(spy.call_count, 0)

    def test_host_scan_after_cache_clear(self):
        from rig._internal import container as _C

        set_options(flatten_containers=False)
        try:
            with container("tq_ctn"):
                container.publish_input(1.0, "knobA")
                host = _C._get_or_create_host("tq_ctn")
                _C._HOST_CACHE.clear()
                self.assertEqual(_C._get_or_create_host("tq_ctn").name, host.name)
                container.publish_input(2.0, "knobB")
        finally:
            set_options(flatten_containers=True)
        hosts = [
            m
            for m in cmds.container("tq_ctn", query=True, nodeList=True) or []
            if cmds.attributeQuery("__rl_host__", node=m, exists=True)
        ]
        self.assertEqual(hosts, [host.name])
