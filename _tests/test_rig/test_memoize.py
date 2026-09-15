"""Tests for ``rig._internal.memoize``.

Covers both ``@memoize`` (caching with MObjectHandle staleness check)
and ``@vectorize`` (NumPy-style strict broadcasting).
"""

from unittest import mock

from maya import cmds, OpenMaya as om1
from rig import Node, PlugList
from rig._internal.memoize import (
    _broadcast_len,
    _node_identity,
    _stable_key,
    memoize,
    vectorize,
)
from rig._tests._base import MayaTestCase


class TestBroadcastLen(MayaTestCase):
    def test_scalars(self):
        self.assertEqual(_broadcast_len(5),        1)
        self.assertEqual(_broadcast_len(3.14),     1)
        self.assertEqual(_broadcast_len(None),     1)
        self.assertEqual(_broadcast_len("foo"),    1)
        self.assertEqual(_broadcast_len(b"bytes"), 1)

    def test_sequences(self):
        self.assertEqual(_broadcast_len([1, 2, 3]), 3)
        self.assertEqual(_broadcast_len((1,)),      1)
        self.assertEqual(_broadcast_len([]),        0)


class TestVectorizeStrictBroadcast(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_no_pluglist_calls_func_directly(self):
        # No PlugList among the args => no vectorisation, raw call.
        @vectorize
        def f(a, b):
            return (a, b)

        # Plain list isn't a trigger.
        self.assertEqual(f(1, 2), (1, 2))
        self.assertEqual(f([1, 2, 3], 5), ([1, 2, 3], 5))

    def test_equal_length_pluglist_broadcasts(self):
        @vectorize
        def f(a, b):
            return a + b

        nodes_a = PlugList([1, 2, 3])
        result  = f(nodes_a, [10, 20, 30])
        self.assertEqual(list(result), [11, 22, 33])

    def test_scalar_broadcasts_against_pluglist(self):
        @vectorize
        def f(a, b):
            return a + b

        nodes_a = PlugList([1, 2, 3])
        result  = f(nodes_a, 5)
        self.assertEqual(list(result), [6, 7, 8])

    def test_length_one_list_broadcasts(self):
        @vectorize
        def f(a, b):
            return a + b

        nodes_a = PlugList([1, 2, 3])
        result  = f(nodes_a, [100])
        self.assertEqual(list(result), [101, 102, 103])

    def test_mismatched_lengths_raises(self):
        @vectorize
        def f(a, b):
            return a + b

        nodes_a = PlugList([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError) as cm:
            f(nodes_a, [10, 20, 30])
        msg = str(cm.exception)
        self.assertIn("cannot broadcast", msg)
        self.assertIn("3", msg)
        self.assertIn("5", msg)

    def test_mismatched_kwargs_raises(self):
        @vectorize
        def f(a, b=None):
            return (a, b)

        nodes_a = PlugList([1, 2, 3])
        with self.assertRaises(ValueError) as cm:
            f(nodes_a, b=[10, 20])
        msg = str(cm.exception)
        self.assertIn("cannot broadcast", msg)
        # Make sure the offending kwarg name appears.
        self.assertIn("b=", msg)

    def test_empty_list_with_non_empty_raises(self):
        @vectorize
        def f(a, b):
            return (a, b)

        nodes_a = PlugList([])
        # Length-0 vs length-3 must error (NumPy-style strict).
        with self.assertRaises(ValueError):
            f(nodes_a, [1, 2, 3])

    def test_single_result_unwrapped(self):
        @vectorize
        def f(a):
            return a * 2

        result = f(PlugList([5]))
        # One result => returned directly, not wrapped in PlugList.
        self.assertEqual(result, 10)

    def test_multiple_results_wrapped_in_pluglist(self):
        @vectorize
        def f(a):
            return a * 2

        result = f(PlugList([1, 2, 3]))
        self.assertIsInstance(result, PlugList)
        self.assertEqual(list(result), [2, 4, 6])

    def test_favor_index_caps_iteration(self):
        @vectorize(favor_index=0)
        def f(a, b):
            return (a, b)

        # Even with longer matching lists, favor_index=0 caps to len(args[0]).
        nodes_a = PlugList([1, 2])
        result  = f(nodes_a, [10, 20])  # both len 2 anyway
        self.assertEqual(list(result), [(1, 10), (2, 20)])

    def test_zero_args_calls_func(self):
        @vectorize
        def f():
            return "no-args"

        self.assertEqual(f(), "no-args")

    def test_falsy_returns_preserved(self):
        # Distinguishes mine from Eric's behaviour -- `0` is preserved.
        @vectorize
        def f(a):
            return 0 if a < 5 else a

        result = f(PlugList([1, 2, 10]))
        self.assertEqual(list(result), [0, 0, 10])

    def test_none_returns_dropped(self):
        @vectorize
        def f(a):
            return None if a < 5 else a

        result = f(PlugList([1, 2, 10]))
        # None is filtered; only the qualifying result remains.
        self.assertEqual(result, 10)


class TestMemoize(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_same_args_return_same_result(self):
        call_count = {"n": 0}

        @memoize
        def add(a, b):
            call_count["n"] += 1
            return a + b

        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(2, 3), 5)
        # Cached after first call.
        self.assertEqual(call_count["n"], 1)

    def test_different_args_different_result(self):
        @memoize
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(3, 4), 7)

    def test_int_and_float_collapse_to_same_key(self):
        # Memoize converts ints to floats when keying.
        call_count = {"n": 0}

        @memoize
        def add(a, b):
            call_count["n"] += 1
            return a + b

        add(2, 3)
        add(2.0, 3.0)
        # Same cache hit.
        self.assertEqual(call_count["n"], 1)

    def test_memoize_with_plug_args_caches(self):
        node       = Node.create("transform", name="cube1")
        call_count = {"n": 0}

        @memoize
        def double(plug):
            call_count["n"] += 1
            return str(plug)

        result1 = double(node.translateX)
        result2 = double(node.translateX)
        self.assertEqual(result1, result2)
        self.assertEqual(call_count["n"], 1)


class TestMemoizeNodeKeyHardening(MayaTestCase):
    """Node/attribute cache keys must survive Maya hashCode recycling.

    Maya recycles ``MObjectHandle.hashCode()`` after a node is deleted (the
    freed MObject slot is reused by the next node). A hashCode-only key
    therefore collides across a delete+recreate -- exactly the create_rail
    proxy-curve churn that produced last session's non-deterministic
    wrong-orient bug. The composite ``(uuid, hashCode)`` key distinguishes
    them because a fresh node always gets a fresh, never-recycled UUID.
    """

    TEST_START_NEW_SCENE = True

    def _force_hashcode_recycle(self, key_of):
        """Churn create+delete until Maya hands a fresh node the hashCode a
        deleted node held; return (old_node_key, recycled_node_key).

        Maya only reuses a freed MObject slot once nothing references it, and a
        live undo queue does: with the GUI default (undo on, infinite) 200
        cycles produce 200 distinct hashCodes and this helper cannot do its
        job. Flushing the queue after each delete releases the slot, so the
        recycle shows up within a handful of cycles regardless of the
        session's undo settings.
        """
        seen = {}
        for i in range(200):
            name = "recyc%d" % i
            node = Node.create("transform", name=name)
            sel  = om1.MSelectionList()
            sel.add(name)
            mobj = om1.MObject()
            sel.getDependNode(0, mobj)
            hash_code = om1.MObjectHandle(mobj).hashCode()
            key       = key_of(node)
            if hash_code in seen:
                old_key = seen[hash_code]
                cmds.delete(node)
                cmds.flushUndo()
                return old_key, key
            seen[hash_code] = key
            cmds.delete(node)
            cmds.flushUndo()
        self.fail("Maya never recycled a hashCode in 200 create/delete cycles")

    def test_node_key_survives_hashcode_recycling(self):
        old_key, new_key = self._force_hashcode_recycle(_stable_key)
        self.assertNotEqual(
            old_key,
            new_key,
            "node key collided after Maya recycled the MObjectHandle hashCode",
        )

    def test_attribute_key_survives_hashcode_recycling(self):
        old_key, new_key = self._force_hashcode_recycle(
            lambda n: _stable_key(n.translateX)
        )
        self.assertNotEqual(
            old_key,
            new_key,
            "attribute key collided after Maya recycled the hashCode",
        )

    def test_node_key_embeds_uuid(self):
        node = Node.create("transform", name="cube1")
        key  = _stable_key(node)
        # Shape is ("node", (uuid, hashCode)).
        self.assertEqual(key[0], "node")
        self.assertEqual(key[1][0], node.uuid)

    def test_attribute_key_embeds_uuid(self):
        node = Node.create("transform", name="cube1")
        key  = _stable_key(node.translateX)
        # Shape is ((uuid, hashCode), alias).
        self.assertEqual(key[0][0], node.uuid)

    def test_node_identity_unresolvable_raises_typeerror(self):
        # An unresolvable node must raise TypeError so the @memoize wrapper
        # falls through its existing un-hashable bypass and recomputes,
        # rather than synthesising a name-based (collision-prone) key.
        with self.assertRaises(TypeError):
            _node_identity("no_such_node_zzz_12345")

    def test_node_identity_falls_back_to_cmds_when_api_uuid_unavailable(self):
        # Older Maya may lack API 1.0 MUuid; the resolver must fall back to
        # cmds.ls(uuid=True) and still return the correct (uuid, hashCode).
        Node.create("transform", name="cube1")
        expected_uuid = cmds.ls("cube1", uuid=True)[0]
        with mock.patch.object(
            om1, "MFnDependencyNode", side_effect=AttributeError("no uuid")
        ):
            uuid_str, hash_code = _node_identity("cube1")
        self.assertEqual(uuid_str, expected_uuid)
        self.assertIsInstance(hash_code, int)