"""Regression tests for the fixes of the round 3 review (step R3_FIX).

Each class covers one finding of the review of perf/efficiency at 29a4128; the
docstring of each test says what it pinned before the fix.
"""

import gc
import os
import shutil
import tempfile
from unittest import mock

from maya import cmds, OpenMaya as om1
from maya.api import OpenMaya
from rig import Node, Plug
from rig.nodetypes import PyNode, _base
from rig.nodetypes._base import Attribute
from rig._internal.node_ops import NodeOp
from rig._tests._base import MayaTestCase


def _hash_code(name):
    """The API 1.0 MObjectHandle hashCode of the node `name`."""
    sel = om1.MSelectionList()
    sel.add(name)
    mobject = om1.MObject()
    sel.getDependNode(0, mobject)
    return om1.MObjectHandle(mobject).hashCode()


def _instanced_locator():
    """Locator shape ``S`` instanced under ``T1`` (instance 0, tx 0) and ``T2``
    (instance 1, tx 7)."""
    cmds.loadPlugin("matrixNodes", quiet=True)
    cmds.createNode("transform", name="T1")
    cmds.createNode("locator", name="S", parent="T1")
    cmds.createNode("transform", name="T2")
    cmds.parent("|T1|S", "T2", add=True, shape=True)
    cmds.setAttr("T2.tx", 7)


class TestSmallFixes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_world_mobject_error_names_its_api_type(self):
        # `apiTypeStr` is a property in API 2.0: calling it raised
        # TypeError "'str' object is not callable" instead of this ValueError
        root = OpenMaya.MItDag().root()
        with self.assertRaisesRegex(ValueError, r"^Invalid MObject API Type: kWorld$"):
            _base._mobject_to_str(root)

    def test_component_type_cache_skips_plug_setattr(self):
        # the lazy cache wrote through `Plug.__setattr__`
        cmds.polyCube(name="pc", constructionHistory=False)
        plug = Node("pcShape").vtx
        with mock.patch.object(Plug, "__setattr__", autospec=True, wraps=Plug.__setattr__) as spy:
            self.assertEqual(plug._component_type, "kMeshVertComponent")
        names = [call.args[1] for call in spy.call_args_list]
        self.assertNotIn("_Attribute__component_type", names)
        self.assertEqual(plug.__dict__["_Attribute__component_type"], "kMeshVertComponent")

    def test_nodeop_fall_through_keeps_no_traceback_cycle(self):
        # the NotImplementedError of a fall-through kept its traceback, whose
        # frame held the error: a cycle per fall-through
        op = NodeOp("r3fallthrough")

        @op.impl(since=2024)
        def _native(value):
            raise NotImplementedError("native cannot")

        @op.impl(since=0)
        def _legacy(value):
            return value

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        gc.set_debug(gc.DEBUG_SAVEALL)
        try:
            self.assertEqual(op(3.0), 3.0)
            gc.collect()
            leaked = [obj for obj in gc.garbage if isinstance(obj, NotImplementedError)]
        finally:
            gc.set_debug(0)
            del gc.garbage[:]
            if was_enabled:
                gc.enable()
        self.assertEqual(leaked, [])

    def test_nodeop_all_impls_failing_still_names_the_last_error(self):
        op = NodeOp("r3allfail")

        @op.impl(since=0)
        def _legacy(value):
            raise NotImplementedError("legacy cannot")

        with self.assertRaisesRegex(RuntimeError, r"raised NotImplementedError\. Last: legacy cannot$"):
            op(1.0)


class TestPlugKeysOfFreedNodes(MayaTestCase):
    """A plug of a freed node kept as a dict or set key never shares a hash with a
    plug of a node made later. At 29a4128 the node part of the hash was the API
    1.0 hashCode, which Maya hands to the next node it makes, so the lookup
    compared the two plugs and the freed one raised "... already deleted!"."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._undo = cmds.undoInfo(query=True, state=True)

    def tearDown(self):
        cmds.undoInfo(state=self._undo)
        super().tearDown()

    def _assert_misses(self, keys, probe):
        self.assertNotIn(probe, keys)
        self.assertIsNone(dict.fromkeys(keys).get(probe, None))
        set(keys).discard(probe)

    def test_undo_off_delete_then_a_new_node(self):
        cmds.undoInfo(state=False)
        cmds.createNode("transform", name="keep")
        # until Maya hands the deleted node's hashCode to the new node
        for _ in range(200):
            cmds.createNode("transform", name="proxy")
            code  = _hash_code("proxy")
            held  = [Node("proxy").tx, Node("keep").tx]
            typed = [PyNode("proxy").find_attr("tx"), PyNode("keep").find_attr("tx")]
            seen, typed_seen = set(held), set(typed)
            cmds.delete("proxy")
            cmds.createNode("transform", name="fresh")
            if _hash_code("fresh") == code:
                break
            cmds.delete("fresh")
        else:
            self.fail("Maya never recycled a hashCode in 200 create/delete cycles")
        self._assert_misses(seen, Node("fresh").tx)
        self._assert_misses(typed_seen, PyNode("fresh").find_attr("tx"))
        # the held keys are still found, the live one by a new spelling too
        self.assertIn(held[0], seen)
        self.assertIn(Node("keep").tx, seen)
        self.assertIn(PyNode("keep").find_attr("tx"), typed_seen)

    def test_flushed_undo_then_a_new_node(self):
        cmds.undoInfo(state=True)
        cmds.createNode("transform", name="proxy")
        seen = {Node("proxy").tx}
        cmds.delete("proxy")
        cmds.flushUndo()
        cmds.createNode("transform", name="fresh")
        self._assert_misses(seen, Node("fresh").tx)

    def test_registry_held_across_a_new_scene(self):
        names = [cmds.createNode("transform", name=f"a{i}") for i in range(100)]
        codes = {_hash_code(name) for name in names}
        table = {Node(name).tx: name for name in names}
        typed = {PyNode(name).find_attr("ty"): name for name in names}
        cmds.file(new=True, force=True)
        recycled = 0
        for i in range(200):
            name = cmds.createNode("transform", name=f"c{i}")
            recycled += _hash_code(name) in codes
            self.assertNotIn(Node(name).tx, table)
            self.assertNotIn(PyNode(name).find_attr("ty"), typed)
        self.assertGreater(recycled, 0)
        self.assertEqual(len(table), 100)

    def test_registry_held_across_a_reopen_and_a_reference_reload(self):
        # a file opened again, or a reference reloaded, brings its nodes back with
        # the UUIDs they had, and often at the hashCode they had
        folder = tempfile.mkdtemp(prefix="rig_r3_keys_")
        path   = os.path.join(folder, "keys.ma").replace(os.sep, "/")
        try:
            names = [cmds.createNode("transform", name=f"n{i}") for i in range(50)]
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            table = {Node(name).tx: name for name in names}
            cmds.file(path, open=True, force=True)
            for name in names:
                self.assertNotIn(Node(name).tx, table)
                self.assertNotIn(PyNode(name).find_attr("tx"), table)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            table = {Node(f"ref:{name}").tx: name for name in names}
            ref = cmds.referenceQuery(path, referenceNode=True)
            cmds.file(unloadReference=ref)
            cmds.file(loadReference=ref)
            for name in names:
                self.assertNotIn(Node(f"ref:{name}").tx, table)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_node_keeps_its_key_across_a_delete_and_its_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        held = Node(cmds.createNode("transform", name="a")).tx
        seen = {held}
        cmds.delete("a")
        cmds.createNode("transform", name="a")
        self.assertNotIn(Node("a").tx, seen)
        cmds.undo()
        cmds.undo()
        self.assertIn(Node("a").tx, seen)
        # two wrappers of one node share the serial
        self.assertEqual(
            _base._node_serial(PyNode("a")), _base._node_serial(Node("a")._dg_node)
        )

    def test_the_serial_table_drops_freed_nodes(self):
        for i in range(20):
            hash(Node(cmds.createNode("transform", name=f"p{i}")).tx)
        cmds.file(new=True, force=True)
        _base._prune_node_serials()
        for entries in _base._NODE_SERIALS.values():
            self.assertTrue(all(handle.isAlive() for handle, _ in entries))


class TestTypedAttributeAndPlugKeys(MayaTestCase):
    """A typed Attribute and a DSL Plug of one Maya plug are two keys again (as on
    d6ad8b2). At 29a4128 they hashed alike, so a dict or set lookup mixing them
    ran Plug.__eq__, which built an equal node, and raised for matrix plugs."""

    TEST_START_NEW_SCENE = True

    def test_mixed_lookups_miss_and_build_nothing(self):
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        cmds.connectAttr("a.tx", "b.tx")
        before = sorted(cmds.ls())
        typed_tx = PyNode("a").find_attr("tx")
        typed_wm = PyNode("a").find_attr("worldMatrix")[0]
        self.assertIsNone({typed_tx: 1}.get(Node("a").tx))
        self.assertIsNone({Node("a").tx: 1}.get(typed_tx))
        self.assertNotIn(Node("a").worldMatrix[0], {typed_wm})
        self.assertNotIn(typed_wm, {Node("a").worldMatrix[0]})
        self.assertNotIn(Node("a").tx, set(PyNode("a").list_attr(keyable=True)))
        dst = set(typed_tx.get_connected_attrs(src=False, dst=True))
        self.assertNotIn(Node("b").tx, dst)
        self.assertEqual(sorted(cmds.ls()), before)
        # one plug all the same, and a typed probe finds the typed key
        self.assertTrue(Node("a").tx.equals(typed_tx))
        self.assertIn(PyNode("b").find_attr("tx"), dst)

    def test_each_layer_is_one_key_through_two_instance_paths(self):
        _instanced_locator()
        plugs = [Node("|T1|S").v, Node("|T2|S").v]
        typed = [PyNode("|T1|S").find_attr("v"), PyNode("|T2|S").find_attr("v")]
        self.assertEqual(len(set(plugs)), 1)
        self.assertEqual(len(set(typed)), 1)
        self.assertEqual(len(set(plugs) | set(typed)), 2)


class TestAttributeInequality(MayaTestCase):
    """Attribute != is the negation of ==. It was str.__ne__, which compares the
    names the two were built with ('a.tx' and 'a.translateX')."""

    TEST_START_NEW_SCENE = True

    def test_ne_negates_eq(self):
        cmds.createNode("transform", name="a")
        by_name, found = Attribute("a.tx"), PyNode("a").find_attr("tx")
        self.assertTrue(by_name == found)
        self.assertFalse(by_name != found)
        other = PyNode("a").find_attr("ty")
        self.assertFalse(found == other)
        self.assertTrue(found != other)
        # a str is never equal, so it is always unequal
        self.assertFalse(found == "a.translateX")
        self.assertTrue(found != "a.translateX")
        cmds.rename("a", "m")
        renamed = PyNode("m").find_attr("translateX")
        self.assertTrue(found == renamed)
        self.assertFalse(found != renamed)

    def test_ne_through_instance_paths(self):
        _instanced_locator()
        a, b = PyNode("|T1|S").find_attr("v"), PyNode("|T2|S").find_attr("v")
        self.assertFalse(a != b)
        wm = PyNode("|T1|S").find_attr("worldMatrix")
        self.assertTrue(wm[0] != wm[1])
        self.assertFalse(wm[1] != PyNode("|T2|S").find_attr("worldMatrix")[1])


class TestFindAttrCacheOfInstancedElements(MayaTestCase):
    """find_attr of a per-instance element (``worldMatrix[1]``) is cached under its
    own name. It was cached under the array's name (partialName drops the
    instanced index), so a later find_attr('worldMatrix') or node.worldMatrix
    returned element 1, now also named and connected as element 1."""

    TEST_START_NEW_SCENE = True

    def test_element_lookup_does_not_shadow_the_array(self):
        _instanced_locator()
        for path in ("|T1|S", "|T2|S"):
            with self.subTest(path=path):
                node = PyNode(path)
                element = node.find_attr("worldMatrix[1]")
                self.assertTrue(element.plug.isElement)
                self.assertEqual(element.plug.logicalIndex(), 1)
                whole = node.find_attr("worldMatrix")
                self.assertTrue(whole.plug.isArray)
                self.assertIs(node.find_attr("worldMatrix[1]"), element)
                self.assertTrue(node.find_attr("iog[1]").plug.isElement)
                self.assertTrue(node.find_attr("instObjGroups").plug.isArray)
        wrapper = Node(PyNode("|T1|S"))
        wrapper._dg_node.find_attr("worldMatrix[1]")
        self.assertTrue(wrapper.worldMatrix.plug.isArray)
        self.assertEqual(str(wrapper.worldMatrix), "T1|S.worldMatrix")

    def test_single_instance_element_lookup_then_index(self):
        cmds.createNode("transform", name="a")
        node = PyNode("a")
        node.find_attr("worldMatrix[0]")
        self.assertEqual(str(Node(node).worldMatrix[0]), "a.worldMatrix")
