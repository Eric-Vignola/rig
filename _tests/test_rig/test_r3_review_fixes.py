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
from rig.nodetypes import _base
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
            typed = [Node("proxy").find_attr("tx"), Node("keep").find_attr("tx")]
            seen, typed_seen = set(held), set(typed)
            cmds.delete("proxy")
            cmds.createNode("transform", name="fresh")
            if _hash_code("fresh") == code:
                break
            cmds.delete("fresh")
        else:
            self.fail("Maya never recycled a hashCode in 200 create/delete cycles")
        self._assert_misses(seen, Node("fresh").tx)
        self._assert_misses(typed_seen, Node("fresh").find_attr("tx"))
        # the held keys are still found, the live one by a new spelling too
        self.assertIn(held[0], seen)
        self.assertIn(Node("keep").tx, seen)
        self.assertIn(Node("keep").find_attr("tx"), typed_seen)

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
        typed = {Node(name).find_attr("ty"): name for name in names}
        cmds.file(new=True, force=True)
        recycled = 0
        for i in range(200):
            name = cmds.createNode("transform", name=f"c{i}")
            recycled += _hash_code(name) in codes
            self.assertNotIn(Node(name).tx, table)
            self.assertNotIn(Node(name).find_attr("ty"), typed)
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
                self.assertNotIn(Node(name).find_attr("tx"), table)
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
        # two node objects of one node share the serial
        self.assertEqual(_base._node_serial(Node("a")), _base._node_serial(Node("a")))

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
        typed_tx = Node("a").find_attr("tx")
        typed_wm = Node("a").find_attr("worldMatrix")[0]
        self.assertIsNone({typed_tx: 1}.get(Node("a").tx))
        self.assertIsNone({Node("a").tx: 1}.get(typed_tx))
        self.assertNotIn(Node("a").worldMatrix[0], {typed_wm})
        self.assertNotIn(typed_wm, {Node("a").worldMatrix[0]})
        self.assertNotIn(Node("a").tx, set(Node("a").list_attr(keyable=True)))
        dst = set(typed_tx.get_connected_attrs(src=False, dst=True))
        self.assertNotIn(Node("b").tx, dst)
        self.assertEqual(sorted(cmds.ls()), before)
        # one plug all the same, and a typed probe finds the typed key
        self.assertTrue(Node("a").tx.equals(typed_tx))
        self.assertIn(Node("b").find_attr("tx"), dst)

    def test_each_layer_is_one_key_through_two_instance_paths(self):
        _instanced_locator()
        plugs = [Node("|T1|S").v, Node("|T2|S").v]
        typed = [Node("|T1|S").find_attr("v"), Node("|T2|S").find_attr("v")]
        self.assertEqual(len(set(plugs)), 1)
        self.assertEqual(len(set(typed)), 1)
        self.assertEqual(len(set(plugs) | set(typed)), 2)


class TestAttributeInequality(MayaTestCase):
    """Attribute != is the negation of ==. It was str.__ne__, which compares the
    names the two were built with ('a.tx' and 'a.translateX')."""

    TEST_START_NEW_SCENE = True

    def test_ne_negates_eq(self):
        cmds.createNode("transform", name="a")
        by_name, found = Attribute("a.tx"), Node("a").find_attr("tx")
        self.assertTrue(by_name == found)
        self.assertFalse(by_name != found)
        other = Node("a").find_attr("ty")
        self.assertFalse(found == other)
        self.assertTrue(found != other)
        # a str is never equal, so it is always unequal
        self.assertFalse(found == "a.translateX")
        self.assertTrue(found != "a.translateX")
        cmds.rename("a", "m")
        renamed = Node("m").find_attr("translateX")
        self.assertTrue(found == renamed)
        self.assertFalse(found != renamed)

    def test_ne_through_instance_paths(self):
        _instanced_locator()
        a, b = Node("|T1|S").find_attr("v"), Node("|T2|S").find_attr("v")
        self.assertFalse(a != b)
        wm = Node("|T1|S").find_attr("worldMatrix")
        self.assertTrue(wm[0] != wm[1])
        self.assertFalse(wm[1] != Node("|T2|S").find_attr("worldMatrix")[1])


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
                node = Node(path)
                element = node.find_attr("worldMatrix[1]")
                self.assertTrue(element.plug.isElement)
                self.assertEqual(element.plug.logicalIndex(), 1)
                whole = node.find_attr("worldMatrix")
                self.assertTrue(whole.plug.isArray)
                self.assertIs(node.find_attr("worldMatrix[1]"), element)
                self.assertTrue(node.find_attr("iog[1]").plug.isElement)
                self.assertTrue(node.find_attr("instObjGroups").plug.isArray)
        node = Node(Node("|T1|S"))
        node.find_attr("worldMatrix[1]")
        self.assertTrue(node.worldMatrix.plug.isArray)
        self.assertEqual(str(node.worldMatrix), "T1|S.worldMatrix")

    def test_single_instance_element_lookup_then_index(self):
        cmds.createNode("transform", name="a")
        node = Node("a")
        node.find_attr("worldMatrix[0]")
        self.assertEqual(str(Node(node).worldMatrix[0]), "a.worldMatrix")


def _surface(name="ball"):
    """A NURBS sphere; returns its shape name."""
    surf = cmds.sphere(name=name, constructionHistory=False)[0]
    return cmds.listRelatives(surf, shapes=True)[0]


class TestComponentPlugOwner(MayaTestCase):
    """The owner rule (D-A) reaches ComponentPlugs (NURBS-surface cv, lattice pt).
    At 29a4128 a Node-read ComponentPlug and its elements had no owner: their
    node was cast again (first-path naming), and the freed-node guards missed
    them (garbage names, or a Maya crash, after a new scene)."""

    TEST_START_NEW_SCENE = True

    def test_handle_rows_elements_and_children_share_the_node(self):
        node = Node(_surface())
        self.assertIs(node.cv.node, node)
        self.assertIs(node.cv[1, 2].node, node)
        self.assertIs(node.cv[1][2].node, node)
        self.assertTrue(all(p.node is node for p in node.cv[1]))
        self.assertIs(node.cv[1, 2].xValue.node, node)
        self.assertIs(node.cv[1, 2].child(0).node, node)
        lattice = cmds.lattice(cmds.polyCube(name="box", ch=False)[0], name="ffd")[1]
        pts = Node(cmds.listRelatives(lattice, shapes=True)[0])
        self.assertIs(pts.pt.node, pts)
        self.assertIs(pts.pt[1, 0, 1].node, pts)

    def test_an_instanced_surface_is_named_through_the_held_path(self):
        shape = _surface()
        cmds.createNode("transform", name="G2")
        cmds.parent(f"|ball|{shape}", "G2", add=True, shape=True)
        second = Node(f"|G2|{shape}")
        self.assertEqual(str(second.visibility), f"G2|{shape}.visibility")
        self.assertEqual(str(second.cv), f"G2|{shape}.controlPoints")
        self.assertEqual(str(second.cv[1, 2]), f"G2|{shape}.cv[1][2]")
        self.assertEqual(second.cv[1, 2].node.long_name, f"|G2|{shape}")
        first = Node(f"|ball|{shape}")
        self.assertEqual(str(first.cv[1, 2]), f"ball|{shape}.cv[1][2]")
        # one Maya plug all the same
        self.assertTrue(second.cv[1, 2].equals(first.cv[1, 2]))
        self.assertEqual(hash(second.cv[1, 2]), hash(first.cv[1, 2]))


class TestTypedChildrenShareTheOwner(MayaTestCase):
    """A typed Attribute's children, elements and parent are owned by the node
    object it holds (D-A in the typed layer). They had no owner: named through
    the first path, and not guarded once their node was freed."""

    TEST_START_NEW_SCENE = True

    def test_children_elements_and_parent(self):
        _instanced_locator()
        cmds.addAttr("|T1|S", longName="arr", attributeType="double", multi=True)
        cmds.setAttr("|T1|S.arr[3]", 1.0)
        node = Node("|T2|S")
        lp = node.find_attr("localPosition")
        for label, attr, name in (
            ("child(0)", lp.child(0), "T2|S.localPositionX"),
            ("child by name", lp.localPositionY, "T2|S.localPositionY"),
            ("parent", node.find_attr("lpz").get_parent(), "T2|S.localPosition"),
            ("logical element", node.find_attr("worldMatrix").element_by_logical_index(1),
             "T2|S.worldMatrix"),
            ("physical element", node.find_attr("arr").element_by_physical_index(0),
             "T2|S.arr[3]"),
            ("index", node.find_attr("instObjGroups")[1], "T2|S.instObjGroups"),
        ):
            with self.subTest(attr=label):
                self.assertIs(attr.node, node)
                self.assertEqual(str(attr), name)
        # a node with one path, and an attr of another node, as before
        cmds.createNode("plusMinusAverage", name="pma")
        element = Node("pma").find_attr("input3D")[1]
        self.assertEqual(str(element.input3Dx), "pma.input3D[1].input3Dx")
        cmds.createNode("transform", name="dst")
        cmds.connectAttr("T1.tx", "dst.tx")
        (source,) = Node("dst").find_attr("tx").get_connected_attrs(dst=False)
        self.assertIsNone(source.__dict__["_node"])
        self.assertEqual(str(source), "T1.translateX")


class TestLiftAndPlugOfATypedAttribute(MayaTestCase):
    """lift(attr), Plug(attr) and List([attr]) of a typed Attribute keep the node
    object the attr holds. They re-resolved its MPlug (or its name), so a plug of
    an instanced node switched to the first path, and a per-instance array
    connected the first instance's element."""

    TEST_START_NEW_SCENE = True

    def test_the_instance_asked_for_is_kept(self):
        from rig import lift, List

        _instanced_locator()
        typed = Node("|T2|S").find_attr("worldMatrix")
        for label, make in (
            ("lift", lift),
            ("Plug", Plug),
            ("List", lambda attr: List([attr])[0]),
        ):
            with self.subTest(via=label):
                plug = make(typed)
                self.assertIs(type(plug), Plug)
                self.assertEqual(str(plug), "T2|S.worldMatrix")
                self.assertTrue(plug.equals(typed))
                self.assertIs(plug.node, typed.node)
                dst = cmds.createNode("multMatrix")
                Node(dst).matrixIn[0] << plug
                sel = OpenMaya.MSelectionList()
                sel.add(f"{dst}.matrixIn[0]")
                self.assertEqual(sel.getPlug(0).source().logicalIndex(), 1)
                self.assertEqual(cmds.getAttr(f"{dst}.matrixSum")[12], 7.0)
        visibility = Node("|T2|S").find_attr("v")
        self.assertEqual(str(lift(visibility)), "T2|S.visibility")
        self.assertEqual(lift(visibility).node.long_name, "|T2|S")

    def test_a_node_of_a_plug_is_its_owner(self):
        from rig.nodetypes import Transform

        node = Node(cmds.createNode("transform", name="a"))
        self.assertIs(Node(node.tx), node)
        self.assertIs(type(Node(node.tx) >> None), Transform)
        self.assertIs(type(Node(Node("a").find_attr("tx")) >> None), Transform)


class TestHeldAcrossAFreeRaise(MayaTestCase):
    """Component plugs, typed children and elements, and a typed attr lifted into
    the DSL, all held across a new scene, a file open or a reference unload and
    never named before it, raise "... already deleted!". They read freed memory:
    another node's name, a ValueError, or a Maya crash."""

    TEST_START_NEW_SCENE = True

    _FREED = r"^\w+ node \(freed by a new scene, a file open or a reference unload\) already deleted!$"

    def _build(self):
        cmds.nurbsPlane(name="np", patchesU=3, patchesV=3, constructionHistory=False)
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("transform", name="held")

    def _hold(self, prefix):
        surface = Node(f"{prefix}npShape")
        typed   = Node(f"{prefix}pma").find_attr("input3D")
        return {
            "cv handle":         surface.cv,
            "cv row":            surface.cv[1][0],
            "cv element":        surface.cv[2][1],
            "cv element child":  surface.cv[1, 2].xValue,
            "typed element":     typed[1],
            "typed child":       typed[1].input3Dx,
            "typed get_parent":  Node(f"{prefix}held").find_attr("tx").get_parent(),
            "typed attr":        Node(f"{prefix}held").find_attr("ty"),
        }

    def _assert_freed(self, held):
        from rig import lift, List

        common = {"str": str, "get": lambda p: p.get(), "Plug": Plug}
        plug_ops = {"plus 1": lambda p: p + 1, "xValue << 5": lambda p: p.xValue << 5}
        typed_ops = {"lift": lift, "List": lambda p: List([p])}
        for name, attr in held.items():
            ops = dict(common)
            ops.update(plug_ops if isinstance(attr, Plug) else typed_ops)
            if name in ("cv handle", "typed attr"):
                ops["index"] = lambda p: p[0]
            for label, op in ops.items():
                with self.subTest(held=name, op=label):
                    with self.assertRaisesRegex(RuntimeError, self._FREED):
                        op(attr)

    def test_across_a_new_scene(self):
        self._build()
        held = self._hold("")
        cmds.file(new=True, force=True)
        for _ in range(200):
            cmds.createNode("multiplyDivide")
        self._assert_freed(held)

    def test_across_a_file_open_of_the_same_names(self):
        folder = tempfile.mkdtemp(prefix="rig_r3_open_")
        path   = os.path.join(folder, "held.ma").replace(os.sep, "/")
        try:
            self._build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = self._hold("")
            cmds.file(path, open=True, force=True)
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_across_a_reference_unload(self):
        folder = tempfile.mkdtemp(prefix="rig_r3_ref_")
        path   = os.path.join(folder, "held_ref.ma").replace(os.sep, "/")
        try:
            self._build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            held = self._hold("ref:")
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)


class TestCmdsReadsTheHeldInstance(MayaTestCase):
    """maya.cmds reads a plug's str buffer, not str(): a plug read through an
    instanced node's path now carries that path in its buffer, so cmds and rig
    act on the same instance. The buffer was built from the MPlug, which names
    the first path: cmds.getAttr(Node('|T2|S').worldMatrix) read T1's."""

    TEST_START_NEW_SCENE = True

    def test_cmds_and_rig_agree(self):
        _instanced_locator()
        second = Node("|T2|S")
        for label, plug, name, tx in (
            ("wm", second.worldMatrix, "T2|S.worldMatrix", 7.0),
            ("wm[0]", second.worldMatrix[0], "T2|S.worldMatrix[0]", 0.0),
            ("wm[1]", second.worldMatrix[1], "T2|S.worldMatrix", 7.0),
            ("typed wm", Node("|T2|S").find_attr("worldMatrix"), "T2|S.worldMatrix", 7.0),
            ("typed wm[1]", Node("|T2|S").find_attr("worldMatrix")[1], "T2|S.worldMatrix", 7.0),
        ):
            with self.subTest(plug=label):
                self.assertEqual(str.__str__(plug), name)
                self.assertEqual(cmds.getAttr(plug)[12], tx)
                self.assertEqual(f"{plug}", name)
                self.assertEqual("-".join([plug, "x"]), f"{name}-x")
        for label, plug in (
            ("v", second.v),
            ("lp child", second.lp[0]),
            ("lpx", second.localPositionX),
            ("iog[1].og", second.instObjGroups[1].objectGroups),
        ):
            with self.subTest(plug=label):
                self.assertEqual(str.__str__(plug), str(plug))
                self.assertTrue(str(plug).startswith("T2|S."))
        mm = cmds.createNode("multMatrix")
        cmds.connectAttr(second.worldMatrix, f"{mm}.matrixIn[0]")
        self.assertEqual(
            cmds.listConnections(f"{mm}.matrixIn[0]", plugs=True), ["T2|S.worldMatrix"]
        )

    def test_a_node_with_one_path_keeps_the_mplug_name(self):
        cmds.createNode("transform", name="A")
        cmds.createNode("transform", name="ctrl", parent="A")
        for plug in (Node("ctrl").tx, Node("A|ctrl").t[0], Node("ctrl").find_attr("ty")):
            with self.subTest(plug=str(plug)):
                self.assertEqual(str.__str__(plug), plug.plug.name())


class TestComponentSliceOnAChosenClass(MayaTestCase):
    """A component slice on a shape wrapped in a generic class reads the point
    count through the node's own class. The owner rule (C2) keeps the class the
    user chose, which has no num_weight_points: slices raised AttributeError
    'Attribute not found: pcShape.num_weight_points'."""

    TEST_START_NEW_SCENE = True

    def test_slices_and_fancy_indexing(self):
        from rig.nodetypes import DAGNode, DGNode

        cmds.polyCube(name="pc", constructionHistory=False)
        cmds.nurbsPlane(name="np", patchesU=1, patchesV=1, constructionHistory=False)
        expected = ["pcShape.controlPoints[0]", "pcShape.controlPoints[1]"]
        for cls in (DAGNode, DGNode):
            with self.subTest(cls=cls.__name__):
                node = Node(cls("pcShape"))
                self.assertIs(type(node.vtx.node), cls)
                self.assertEqual([str(p) for p in node.vtx[0:2]], expected)
                self.assertEqual([str(p) for p in node.vtx[[0, 1]]], expected)
                self.assertEqual([str(p) for p in node.controlPoints[0:2]], expected)
                self.assertEqual(len(node.vtx[:]), 8)
                with self.assertRaisesRegex(IndexError, "index out of range for 8 points"):
                    node.vtx[[8]]
        surface = Node(DAGNode("npShape"))
        self.assertEqual(
            [str(p) for p in surface.controlPoints[0:2]],
            ["npShape.controlPoints[0]", "npShape.controlPoints[1]"],
        )


class TestMemoOfARebuiltAttribute(MayaTestCase):
    """A memoized network is rebuilt when the dynamic attribute it reads was
    deleted and added again, or renamed while a new attribute took its name.
    The key names the attribute, so such a call handed back the old network:
    disconnected, or reading the renamed attribute (the same on v2.0.0a2)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._undo = cmds.undoInfo(query=True, state=True)
        cmds.createNode("transform", name="n")

    def tearDown(self):
        cmds.undoInfo(state=self._undo)
        super().tearDown()

    def _calls(self):
        from rig import functions, random as rrandom

        return {
            "@memoize abs": lambda plug: functions.abs(plug),
            "NodeOp +": lambda plug: plug + 1,
            "random.value": lambda plug: rrandom.value(plug, seed=3),
        }

    def _readd(self, name, undo):
        cmds.undoInfo(state=undo)
        cmds.addAttr("n", longName=name, attributeType="double")
        firsts = {
            label: call(getattr(Node("n"), name)) for label, call in self._calls().items()
        }
        cmds.deleteAttr(f"n.{name}")
        cmds.addAttr("n", longName=name, attributeType="double")
        return firsts

    def test_deleted_and_added_again(self):
        for undo in (True, False):
            name = "foo" if undo else "fooOff"
            firsts = self._readd(name, undo)
            for label, call in self._calls().items():
                with self.subTest(call=label, undo=undo):
                    again = call(getattr(Node("n"), name))
                    self.assertNotEqual(str(again), str(firsts[label]))
                    # the new network reads the new attribute, and is shared
                    self.assertEqual(str(call(getattr(Node("n"), name))), str(again))
        from rig import functions

        cmds.setAttr("n.foo", -3)
        self.assertEqual(cmds.getAttr(str(functions.abs(Node("n").foo))), 3.0)

    def test_renamed_while_a_new_attribute_takes_the_name(self):
        from rig import functions

        cmds.undoInfo(state=True)
        cmds.addAttr("n", longName="bar", attributeType="double")
        first = functions.abs(Node("n").bar)
        plus  = Node("n").bar + 1
        cmds.renameAttr("n.bar", "baz")
        cmds.addAttr("n", longName="bar", attributeType="double")
        again = functions.abs(Node("n").bar)
        self.assertNotEqual(str(again), str(first))
        self.assertNotEqual(str(Node("n").bar + 1), str(plus))
        cmds.setAttr("n.bar", -5)
        cmds.setAttr("n.baz", -1)
        self.assertEqual(cmds.getAttr(str(again)), 5.0)
        self.assertEqual(cmds.getAttr(str(first)), 1.0)

    def test_unchanged_attributes_keep_their_network(self):
        from rig import functions

        cmds.addAttr("n", longName="knob", attributeType="double")
        for plug in (Node("n").knob, Node("n").tx):
            with self.subTest(plug=str(plug)):
                self.assertEqual(str(functions.abs(plug)), str(functions.abs(plug)))
                self.assertEqual(str(plug + 1), str(plug + 1))
        # an alias names the attribute anew: a new key, as before
        cmds.aliasAttr("dial", "n.knob")
        self.assertEqual(str(functions.abs(Node("n").dial)), str(functions.abs(Node("n").dial)))


class TestTextOperandsBeyondStr(MayaTestCase):
    """bytes, bytearray and numpy bytes arrays are rejected as DSL operands like a
    plain str (D-C), before any node is built. They got past the check: the node
    was built and the first byte was injected (b'cube.ty' set 99.0), or the
    node was left behind."""

    TEST_START_NEW_SCENE = True

    def test_bytes_operands_raise_before_any_node(self):
        import numpy as np
        from rig import functions, vector

        t = Node(cmds.createNode("transform", name="t"))
        before = sorted(cmds.ls())
        for label, call in (
            ("== bytes", lambda: t.tx == b"cube.ty"),
            ("* bytearray", lambda: t.tx * bytearray(b"x")),
            ("+ numpy bytes", lambda: t.tx + np.array([b"x"])),
            ("vector + [1, bytes, 2]", lambda: t.t + [1, b"x", 2]),
            ("functions.abs", lambda: functions.abs(b"cube.ty")),
            ("vector.lerp", lambda: vector.lerp(t.t, b"cube.ty")),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(TypeError, r" is bytes, and a DSL operand is a Plug"):
                    call()
        self.assertEqual(sorted(cmds.ls()), before)
        with self.assertRaisesRegex(TypeError, r"Write Plug\('cube\.ty'\) for the plug of that name"):
            t.tx == b"cube.ty"

    def test_sequence_method_must_be_callable(self):
        from rig import interpolate

        t = Node(cmds.createNode("transform", name="t"))
        before = sorted(cmds.ls())
        for method in ("lerp", None):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                    TypeError, r"^rig\.interpolate\.sequence\(\) argument 'method': .* is not callable"
                ):
                    interpolate.sequence(t.tx, [0, 1], [0, 1], method=method)
        self.assertEqual(sorted(cmds.ls()), before)


class TestCheapCommonPaths(MayaTestCase):
    """The common paths of this step's checks read what the plug already holds."""

    TEST_START_NEW_SCENE = True

    def test_fixed_attr_kind_reads_the_owners_fn_set(self):
        node  = Node(cmds.createNode("transform", name="a"))
        owned = node.tx
        self.assertIs(_base._plug_node_fn_set(owned, owned.plug), node._fn_set)
        self.assertEqual(_base._fixed_attr_kind(owned), OpenMaya.MFn.kDoubleLinearAttribute)
        loose = Plug("a.tx")
        self.assertIsNot(_base._plug_node_fn_set(loose, loose.plug), node._fn_set)
        self.assertEqual(_base._fixed_attr_kind(loose), OpenMaya.MFn.kDoubleLinearAttribute)
        # a deleted owner's fn set is not read
        cmds.undoInfo(state=True, infinity=True)
        cmds.delete("a")
        self.assertIsNot(_base._plug_node_fn_set(owned, owned.plug), node._fn_set)
        cmds.undo()

    def test_only_a_path_named_parent_is_checked_for_instancing(self):
        _instanced_locator()
        cmds.createNode("transform", name="a")
        self.assertTrue(_base._named_through_a_path(Node("|T2|S").v))
        self.assertTrue(_base._named_through_a_path(Node("|T1|S").lp))
        self.assertFalse(_base._named_through_a_path(Node("a").t))
        # a node instanced after its node object cached an attr keeps the first
        # path's name, which cmds resolves to that node object's own (first)
        # instance
        cmds.createNode("transform", name="G1")
        cmds.createNode("transform", name="X", parent="G1")
        first = Node("|G1|X")
        first.worldMatrix
        cmds.createNode("transform", name="G2")
        cmds.parent("|G1|X", "G2", add=True)
        cmds.setAttr("G2.tx", 5)
        self.assertEqual(cmds.getAttr(first.worldMatrix)[12], 0.0)
        self.assertEqual(cmds.getAttr(Node("|G2|X").worldMatrix)[12], 5.0)

    def test_memo_checks_only_dynamic_attributes(self):
        from rig._internal.memoize import _entry_handles

        node = Node(cmds.createNode("transform", name="a"))
        cmds.addAttr("a", longName="knob", attributeType="double")
        self.assertEqual(_entry_handles(None, (node.tx, 1.0, [node.ty]), {}), [])
        checks = _entry_handles(None, (node.tx, [2.0, (node.knob,)]), {"w": node.knob})
        self.assertEqual([check.name for check in checks], ["knob", "knob"])
        self.assertTrue(all(check.isAlive() and check.isValid() for check in checks))
