"""Round 6 S7 (rig_round6_plan item 10): the memo key of a plug of a DG node object reads
its node identity (``(uuid, hashCode)``) once per statement.

`rig._internal.memoize._named_dg_identity` keeps the identity it read in
``journal.NODE_IDS`` while a statement is open, keyed by the node object (the object is
kept in the entry, so its ``id`` is not reused), and uses it again only while the node
object holds the same handle and that handle is valid. ``NODE_IDS`` is emptied at the
outermost statement's start and end, at `outside` (a user call), at every
``cmds_edit.rename`` and at the Undo / Redo and scene events.

Every test compares the key made inside a statement with the key the same plug gets
with the identity read afresh (`_fresh`): the two must always be equal, or raise alike.

* ``TestOneStatement``: the identity is kept and reused within one statement, the key
  is the fresh one, and the statement's end empties it (also when it raises).
* ``TestUuidChanges``: ``rename -uuid`` between two statements, inside a user callback
  (``user_call``), through ``cmds_edit.rename`` inside the statement; a plain rename.
* ``TestDeleteAndRecreate``: a node deleted and a new node made in one statement (with
  Maya's hashCode reuse forced), and a node deleted to the undo queue.
* ``TestReference``: a referenced node unloaded and reloaded inside one statement
  (the D1 case): the old node object's key raises as the fresh one does.
* ``TestClearPoints``: the Undo / Redo and scene callbacks, ``outside``.
* ``TestMemoHit``: ``@memoize`` across a uuid change inside a user callback.
"""

import os
import tempfile

from maya import cmds
from maya.api import OpenMaya

from rig import Node
from rig._internal import journal
from rig._internal.journal import cmds_edit, outside, statement, user_call
from rig._internal.memoize import _attribute_key, _stable_key, memoize
from rig._tests._base import MayaTestCase

_UUID_A = "0E4C1D2A-0000-4000-8000-00000000A001"
_UUID_B = "0E4C1D2A-0000-4000-8000-00000000B002"


def _fresh(fn, *args):
    """``fn(*args)`` with the identity read afresh: no statement is open (the cache is
    neither read nor filled), then the statement state is put back."""
    was, journal._S.open = journal._S.open, False
    try:
        return fn(*args)
    finally:
        journal._S.open = was


def _uuid(node_name):
    return cmds.ls(node_name, uuid=True)[0]


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        journal.NODE_IDS.clear()

    def tearDown(self):
        journal.NODE_IDS.clear()
        cmds.undoInfo(state=True)
        super().tearDown()

    def assertFresh(self, plug):
        """The key of ``plug`` now equals its fresh key, and returns it."""
        key = _attribute_key(plug)
        self.assertEqual(key, _fresh(_attribute_key, plug))
        return key


class TestOneStatement(_Case):
    def test_kept_and_reused_within_a_statement(self):
        node = Node.create("multiplyDivide", name="md1")
        seen = []

        @statement
        def body():
            seen.append(self.assertFresh(node.input1X))
            self.assertIn(id(node), journal.NODE_IDS)
            self.assertIs(journal.NODE_IDS[id(node)][0], node)
            seen.append(self.assertFresh(node.input2Y))
            seen.append(_stable_key(node.input1X))

        body()
        self.assertEqual(seen[0], seen[2])
        self.assertEqual(seen[0][0], (_uuid("md1"), seen[0][0][1]))
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertEqual(journal.NODE_IDS, {})

    def test_not_filled_outside_a_statement(self):
        node = Node.create("multiplyDivide")
        _attribute_key(node.input1X)
        self.assertEqual(journal.NODE_IDS, {})

    def test_emptied_when_the_statement_raises(self):
        node = Node.create("multiplyDivide")

        @statement
        def body():
            _attribute_key(node.input1X)
            self.assertTrue(journal.NODE_IDS)
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            body()
        self.assertEqual(journal.NODE_IDS, {})

    def test_a_dag_node_is_not_kept(self):
        node = Node.create("transform")

        @statement
        def body():
            self.assertFresh(node.translateX)
            self.assertEqual(journal.NODE_IDS, {})

        body()


class TestUuidChanges(_Case):
    def test_rename_uuid_between_two_statements(self):
        node = Node.create("multiplyDivide", name="md1")
        keys = []

        @statement
        def body():
            keys.append(self.assertFresh(node.input1X))

        body()
        cmds.rename("md1", _UUID_A, uuid=True)
        body()
        self.assertEqual(keys[1][0][0], _UUID_A)
        self.assertNotEqual(keys[0], keys[1])

    def test_rename_uuid_inside_a_user_callback(self):
        node = Node.create("multiplyDivide", name="md1")
        keys = []

        def callback():
            cmds.rename("md1", _UUID_A, uuid=True)

        @statement
        def body():
            keys.append(self.assertFresh(node.input1X))
            user_call(callback)
            keys.append(self.assertFresh(node.input1X))

        body()
        self.assertNotEqual(keys[0][0][0], _UUID_A)
        self.assertEqual(keys[1][0][0], _UUID_A)

    def test_rename_uuid_through_cmds_edit_inside_a_statement(self):
        node = Node.create("multiplyDivide", name="md1")
        keys = []

        @statement
        def body():
            keys.append(self.assertFresh(node.input1X))
            cmds_edit.rename("md1", _UUID_B, uuid=True)
            keys.append(self.assertFresh(node.input1X))

        body()
        self.assertEqual(keys[1][0][0], _UUID_B)
        self.assertNotEqual(keys[0], keys[1])

    def test_plain_rename_inside_a_statement(self):
        node = Node.create("multiplyDivide", name="md1")
        keys = []

        @statement
        def body():
            keys.append(self.assertFresh(node.input1X))
            node.rename("md2")
            keys.append(self.assertFresh(node.input1X))
            keys.append(self.assertFresh(Node("md2").input1X))

        body()
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], keys[2])


class TestDeleteAndRecreate(_Case):
    def test_delete_and_new_node_with_a_reused_hashcode(self):
        """One statement: a node is keyed, deleted (undo off, so its slot is freed) and new
        nodes are made until Maya hands one the deleted node's hashCode. The new node's key
        is its fresh one, and the deleted node's raises as its fresh one does."""
        cmds.undoInfo(state=False)
        out = {}

        @statement
        def body():
            first = Node.create("network", name="nw0")
            out["old"] = self.assertFresh(first.message)
            old_hash = out["old"][0][1]
            cmds_edit.delete("nw0")
            for i in range(200):
                node = Node.create("network", name="nw%d" % (i + 1))
                key = self.assertFresh(node.message)
                if key[0][1] == old_hash:
                    out["new"] = key
                    break
                cmds_edit.delete(str(node))
            for fn in (_attribute_key, lambda p: _fresh(_attribute_key, p)):
                with self.assertRaises(RuntimeError):
                    fn(first.message)

        body()
        if "new" not in out:
            self.skipTest("Maya reused no hashCode in 200 create/delete cycles")
        self.assertEqual(out["new"][0][1], out["old"][0][1])
        self.assertNotEqual(out["new"], out["old"])

    def test_node_deleted_to_the_undo_queue(self):
        node = Node.create("multiplyDivide", name="md1")

        @statement
        def body():
            self.assertFresh(node.input1X)
            cmds_edit.delete("md1")
            with self.assertRaises(RuntimeError):
                _attribute_key(node.input1X)
            with self.assertRaises(RuntimeError):
                _fresh(_attribute_key, node.input1X)

        body()


class TestReference(_Case):
    def test_unload_and_reload_inside_a_statement(self):
        """The D1 case: a referenced node keyed, its reference unloaded and reloaded in
        the same statement. The old node object's key raises as its fresh one does; a new
        node object of the reloaded node is keyed afresh."""
        folder = tempfile.mkdtemp(prefix="rig_s7_")
        path = os.path.join(folder, "ref_md.ma").replace("\\", "/")
        cmds.createNode("multiplyDivide", name="refmd")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        ref_file = cmds.file(path, reference=True, namespace="R")
        ref_node = cmds.referenceQuery(ref_file, referenceNode=True)
        old = Node("R:refmd")
        keys = []

        @statement
        def body():
            keys.append(self.assertFresh(old.input1X))
            cmds_edit.file(unloadReference=ref_node)
            cmds_edit.file(loadReference=ref_node)
            for fn in (_attribute_key, lambda p: _fresh(_attribute_key, p)):
                with self.assertRaises(RuntimeError):
                    fn(old.input1X)
            keys.append(self.assertFresh(Node("R:refmd").input1X))

        body()
        self.assertEqual(keys[0][0][0], keys[1][0][0])  # the uuid is kept across the reload


class TestClearPoints(_Case):
    def _fill(self):
        node = Node.create("multiplyDivide")
        journal.NODE_IDS[id(node)] = (node, node._objhandle, ("x", 0))
        return node

    def test_undo_redo_and_scene_callbacks(self):
        for callback in (journal._undone_or_redone, journal._settle, journal._discard):
            self._fill()
            callback()
            self.assertEqual(journal.NODE_IDS, {}, callback.__name__)

    def test_outside(self):
        @statement
        def body():
            self._fill()
            with outside():
                self.assertEqual(journal.NODE_IDS, {})
                self._fill()
            self.assertEqual(journal.NODE_IDS, {})

        body()

    def test_a_stale_entry_is_never_read(self):
        """An entry whose handle is not the node object's own is passed over."""
        node = Node.create("multiplyDivide")

        @statement
        def body():
            other = OpenMaya.MObjectHandle(node._mobject)
            journal.NODE_IDS[id(node)] = (node, other, (_UUID_A, 0))
            self.assertNotEqual(self.assertFresh(node.input1X)[0][0], _UUID_A)

        body()


class TestMemoHit(_Case):
    def test_memo_misses_after_a_uuid_change_in_a_user_callback(self):
        calls = []

        @memoize
        def probe(plug):
            calls.append(1)
            return len(calls)

        node = Node.create("multiplyDivide", name="md1")
        got = []

        @statement
        def body():
            got.append(probe(node.input1X))
            got.append(probe(node.input1X))
            user_call(cmds.rename, "md1", _UUID_A, uuid=True)
            got.append(probe(node.input1X))

        body()
        self.assertEqual(got, [1, 1, 2])
