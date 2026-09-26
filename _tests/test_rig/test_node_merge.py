"""Tests for the merge of the DSL ``Node`` wrapper into the ``rig.nodetypes``
hierarchy. Each class names the step of the merge it belongs to:

* S0: Plug property setters, ``bool(plug)``, ``find_attr`` filters and caching,
  ``rename_attr`` with a Plug.
* S1: the owner rule (a plug's node is the node object it was read from),
  foreign plugs, stale DAG paths, held plugs across delete / reuse / undo.

The merge itself was not landed (round 3, decision D-A: the owner rule only, on
the two classes). The fixes of the prototype's review that apply to the two
classes were ported (``TestReviewFixes``, same ids as on proto/node-merge), and
decision D-B gives an instanced plug one identity (``TestInstancedPlugIdentity``).
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Container, Node, Plug, container
from rig.nodetypes import PyNode, Transform
from rig._internal.list import PlugList
from rig._internal.members import Components
from rig._internal.memoize import _attribute_key
from rig._tests._base import MayaTestCase


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _source_index(dst):
    """The logical index of the element connected into the plug named `dst`."""
    sel = OpenMaya.MSelectionList()
    sel.add(dst)
    return sel.getPlug(0).source().logicalIndex()


def _instanced_locator():
    """Locator shape ``S`` instanced under ``T1`` (instance 0, tx 0) and ``T2``
    (instance 1, tx 7)."""
    cmds.loadPlugin("matrixNodes", quiet=True)
    cmds.createNode("transform", name="T1")
    cmds.createNode("locator", name="S", parent="T1")
    cmds.createNode("transform", name="T2")
    cmds.parent("|T1|S", "T2", add=True, shape=True)
    cmds.setAttr("T2.tx", 7)


class TestPlugQuickWins(MayaTestCase):
    """S0: the Plug and ``find_attr`` fixes that do not depend on the merge."""

    TEST_START_NEW_SCENE = True

    def test_property_setters_work_on_a_plug(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="knob", at="double", keyable=True)
        plug = Node(net).knob
        plug.alias = "dial"
        self.assertEqual(Plug("net.knob").alias, "dial")
        plug.alias = None
        self.assertEqual(Plug("net.knob").alias, "knob")
        plug.is_keyable = False
        self.assertFalse(cmds.getAttr("net.knob", keyable=True))
        plug.is_channel_box = True
        self.assertTrue(cmds.getAttr("net.knob", channelBox=True))
        plug.default_value = 2.5
        self.assertEqual(cmds.addAttr("net.knob", query=True, defaultValue=True), 2.5)
        plug.is_locked = True
        self.assertTrue(cmds.getAttr("net.knob", lock=True))
        # none of them became a Python attribute of the plug
        for name in ("alias", "is_keyable", "is_channel_box", "default_value", "is_locked"):
            self.assertNotIn(name, vars(plug))

    def test_str_method_names_stay_sugar(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="center", at="double")
        cmds.addAttr(net, ln="other", at="double")
        plug = Plug("net.other")
        plug.center = 5
        self.assertEqual(cmds.getAttr("net.center"), 5.0)
        self.assertNotIn("center", vars(plug))

    def test_bool_makes_no_container_query(self):
        cmds.createNode("transform", name="a")
        plugs = (Node("a").tx, Plug("a.t"), Node("a").t[0])
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            self.assertEqual([bool(plug) for plug in plugs], [True] * 3)
        self.assertEqual(probe.call_count, 0)
        # a comparison result still reports whether its operands match
        self.assertTrue(Node("a").tx == Plug("a.tx"))
        self.assertFalse(Node("a").tx == Node("a").ty)

    def test_find_attr_filters_a_cache_hit(self):
        cmds.createNode("transform", name="a")
        dg     = PyNode("a")
        cached = dg.find_attr("tx")
        self.assertIsNone(dg.find_attr("tx", data_type="string"))
        self.assertIsNone(dg.find_attr("translateX", category="noSuchCategory"))
        self.assertIs(dg.find_attr("tx", data_type="doubleLinear"), cached)
        self.assertIs(dg.find_attr("translateX"), cached)

    def test_readded_extension_attr_through_the_same_wrapper(self):
        cmds.createNode("transform", name="a")
        dg = PyNode("a")
        try:
            cmds.addExtension(nodeType="transform", longName="mergeExt", at="double")
            self.assertEqual(dg.find_attr("mergeExt").data_type, "double")
            cmds.deleteExtension(
                nodeType="transform", attribute="mergeExt", forceDelete=True
            )
            cmds.addExtension(nodeType="transform", longName="mergeExt", dataType="string")
            attr = dg.find_attr("mergeExt")
            self.assertEqual(str(attr), "a.mergeExt")
            self.assertEqual(attr.data_type, "string")
            self.assertNotIn("mergeExt", dg._attr_dict)
            # normal attrs are still cached
            dg.find_attr("tx")
            self.assertIn("translateX", dg._attr_dict)
        finally:
            if cmds.attributeQuery("mergeExt", type="transform", exists=True):
                cmds.deleteExtension(
                    nodeType="transform", attribute="mergeExt", forceDelete=True
                )

    def test_rename_attr_with_a_plug_creates_no_node(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="foo", at="double")
        dg     = PyNode(net)
        before = sorted(cmds.ls())
        result = dg.rename_attr(Node(net).foo, "bar")
        self.assertEqual(sorted(cmds.ls()), before)
        self.assertTrue(cmds.attributeQuery("bar", node=net, exists=True))
        self.assertEqual(str(result), "net.bar")


class TestOwnerRule(MayaTestCase):
    """S1: a plug's node is the node object it was read from."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._registered = dict(PyNode._NODE_CLASS_DICT)

    def tearDown(self):
        PyNode._NODE_CLASS_DICT.clear()
        PyNode._NODE_CLASS_DICT.update(self._registered)
        PyNode._CLASS_BY_TYPE.clear()
        PyNode._CASTABLE_TYPES.clear()
        super().tearDown()

    def test_plugs_children_and_elements_share_the_node(self):
        node = Node(cmds.createNode("transform", name="a"))
        pma  = Node(cmds.createNode("plusMinusAverage", name="pma"))
        for plug in (
            node.tx,
            node.t[0],
            node.t.tx,
            node.t[:][1],
            node.translate.child(2),
        ):
            with self.subTest(plug=str(plug)):
                self.assertIs(plug.node, node)
        for plug in (pma.input1D[3], pma.input3D[1], pma.input3D[1].input3Dx):
            with self.subTest(plug=str(plug)):
                self.assertIs(plug.node, pma)
        self.assertIs(node.find_attr("tx").node, node._dg_node)
        self.assertIs(node.find_attr("tx"), node.find_attr("translateX"))

    def test_foreign_plug_gets_its_own_node(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        curve  = cmds.curve(point=points, name="crv")
        shape  = cmds.listRelatives(curve, shapes=True)[0]
        plug   = Node(curve).controlPoints
        self.assertEqual(plug.node.name, shape)
        self.assertTrue(plug.node.mobject == _mobject(shape))
        self.assertEqual(str(plug), f"{shape}.controlPoints")
        # the transform's own attrs are still its own
        self.assertEqual(Node(curve).tx.node.name, curve)

    def test_stale_path_names_the_surviving_instance(self):
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        node = Node("|T2|S")
        held = node.visibility
        self.assertEqual(str(held), "T2|S.visibility")
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(node.name, "S")
        self.assertEqual(node.long_name, "|T1|S")
        self.assertTrue(node.mdagpath.isValid())
        self.assertEqual(str(held), "S.visibility")
        self.assertEqual(str(node.localPositionX), "S.localPositionX")

    def test_held_plug_across_delete_reuse_and_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        node = Node(cmds.createNode("transform", name="held"))
        plug = node.tx
        cmds.delete("held")
        for func in (str, lambda p: p.get()):
            with self.assertRaises(RuntimeError) as ctx:
                func(plug)
            self.assertEqual(str(ctx.exception), "held already deleted!")
        cmds.createNode("transform", name="held")
        with self.assertRaises(RuntimeError) as ctx:
            str(plug)
        self.assertEqual(str(ctx.exception), "held already deleted!")
        # a plug built from the name finds the new node
        self.assertEqual(str(Plug("held.tx")), "held.translateX")
        cmds.undo()
        cmds.undo()
        self.assertEqual(str(plug), "held.translateX")
        plug << 3.0
        self.assertEqual(cmds.getAttr("held.tx"), 3.0)

    def test_owner_soundness_sweep(self):
        md         = cmds.createNode("multiplyDivide", name="md")
        xform      = cmds.createNode("transform", name="xf")
        jnt        = cmds.createNode("joint", name="jnt")
        cube       = cmds.polyCube(name="cube")[0]
        mesh       = cmds.listRelatives(cube, shapes=True)[0]
        surf       = cmds.sphere(name="ball", constructionHistory=False)[0]
        surf_shape = cmds.listRelatives(surf, shapes=True)[0]
        lattice    = cmds.lattice(cube, name="lat")[1]
        lat_shape  = cmds.listRelatives(lattice, shapes=True)[0]
        target     = cmds.polyCube(name="target")[0]
        base       = cmds.polyCube(name="base")[0]
        bs         = cmds.blendShape(target, base, name="bs")[0]
        pma        = cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:n")
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "slide")
        self.assertIsInstance(box, Container)

        plugs = {
            "dg": lambda: Node(md).input1X,
            "dg_child": lambda: Node(md).input1[1],
            "transform": lambda: Node(xform).tx,
            "transform_child": lambda: Node(xform).t.ty,
            "world_matrix": lambda: Node(xform).worldMatrix[0],
            "joint": lambda: Node(jnt).jointOrientX,
            "mesh": lambda: Node(mesh).outMesh,
            "mesh_vtx": lambda: Node(mesh).vtx[3],
            "vtx_via_transform": lambda: Node(cube).vtx[3],
            "surface_cv": lambda: Node(surf_shape).cv[1, 2],
            "surface_cv_handle": lambda: Node(surf_shape).cv,
            "surface_cv_via_transform": lambda: Node(surf).cv[1, 2],
            "lattice_pt": lambda: Node(lat_shape).pt[0, 1, 0],
            "blendshape_alias": lambda: getattr(Node(bs), target),
            "blendshape_weight": lambda: Node(bs).weight[0],
            "container_genuine": lambda: box.blackBox,
            "container_published": lambda: box.slide,
            "instanced": lambda: Node("|T2|S").visibility,
            "namespaced": lambda: Node("ns:n").tx,
            "element": lambda: Node(pma).input3D[1],
            "element_child": lambda: Node(pma).input3D[1].input3Dx,
            "string": lambda: Plug("xf.tx"),
        }
        for label, factory in plugs.items():
            with self.subTest(pattern=label):
                plug = factory()
                self.assertTrue(plug.node.mobject == plug.plug.node(), str(plug))
                self.assertIsInstance(plug.node, Node)
        for kind in ("f", "e"):
            with self.subTest(pattern=kind):
                components = getattr(Node(cube), kind)
                self.assertIsInstance(components, Components)
                self.assertTrue(components.shape.mobject == _mobject(mesh))

    def test_name_property_that_raises_names_by_fn_set(self):
        class _NameRaises(Transform):
            NATIVE_NODE_TYPE = "mergeNameRaisesProbe"

            @property
            def name(self):
                raise AttributeError("no name")

        cmds.createNode("transform", name="w")
        cmds.addAttr("w", longName="name", dataType="string")
        wrapper = object.__new__(_NameRaises)
        vars(wrapper).update(vars(PyNode(_mobject("w"))))
        vars(wrapper)["_attr_dict"] = {}
        for plug in (Node(wrapper).tx, wrapper.find_attr("tx"), Node(wrapper).t[0]):
            with self.subTest(plug=type(plug).__name__):
                self.assertEqual(str(plug), "w.translateX")
                self.assertEqual(plug.full_name, "w.translateX")


class TestReviewFixes(MayaTestCase):
    """The fixes the review of the merge prototype found that apply to the two
    classes, ported with the prototype's test ids."""

    TEST_START_NEW_SCENE = True

    _FREED = (
        r"^Transform node \(freed by a new scene, a file open or a reference unload\) "
        r"already deleted!$"
    )

    def _held(self, names):
        held = []
        for name in names:
            node     = Node(name)
            compound = node.t
            compound.tx  # a cached child
            typed = PyNode(name)
            typed.tx  # a cached typed attr
            held.append(
                (node, node.tx, compound, node.worldMatrix[0], node.find_attr("ty"), typed)
            )
        return held

    def _assert_freed(self, held):
        persp = Node("persp")
        ops = {
            "node.tx (cached)":  lambda n, p, t, w, a, d: n.tx,
            "node.ty":           lambda n, p, t, w, a, d: n.ty,
            "str(node)":         lambda n, p, t, w, a, d: str(n),
            "repr(node)":        lambda n, p, t, w, a, d: repr(n),
            "node == node":      lambda n, p, t, w, a, d: n == n,
            "hash(node)":        lambda n, p, t, w, a, d: hash(n),
            "node.tx = 1":       lambda n, p, t, w, a, d: setattr(n, "tx", 1),
            "str(plug)":         lambda n, p, t, w, a, d: str(p),
            "plug.get()":        lambda n, p, t, w, a, d: p.get(),
            "plug.set(1)":       lambda n, p, t, w, a, d: p.set(1),
            "plug << 1":         lambda n, p, t, w, a, d: p << 1,
            "plug << None":      lambda n, p, t, w, a, d: p << None,
            "plug >> None":      lambda n, p, t, w, a, d: p >> None,
            "plug.alias":        lambda n, p, t, w, a, d: p.alias,
            "plug.is_connected": lambda n, p, t, w, a, d: p.is_connected,
            "plug.is_locked":    lambda n, p, t, w, a, d: p.is_locked,
            "plug.get_inputs()": lambda n, p, t, w, a, d: p.get_inputs(),
            "plug + 1":          lambda n, p, t, w, a, d: p + 1,
            "plug == plug":      lambda n, p, t, w, a, d: p == p,
            "plug.equals(plug)": lambda n, p, t, w, a, d: p.equals(t),
            "plug.equals(self)": lambda n, p, t, w, a, d: p.equals(p),
            "compound.tx":       lambda n, p, t, w, a, d: t.tx,
            "compound[0]":       lambda n, p, t, w, a, d: t[0],
            "compound.child(0)": lambda n, p, t, w, a, d: t.child(0),
            "element.get()":     lambda n, p, t, w, a, d: w.get(),
            "attribute.get()":   lambda n, p, t, w, a, d: a.get(),
            "attribute parent":  lambda n, p, t, w, a, d: a.get_parent(),
            "persp.tx << plug":  lambda n, p, t, w, a, d: persp.tx << p,
            "memo key":          lambda n, p, t, w, a, d: _attribute_key(p),
            "str(typed)":        lambda n, p, t, w, a, d: str(d),
            "str(typed.tx)":     lambda n, p, t, w, a, d: str(d.tx),
            "typed.tx.get()":    lambda n, p, t, w, a, d: d.tx.get(),
            "typed.rx":          lambda n, p, t, w, a, d: d.rx,
        }
        for label, op in ops.items():
            with self.subTest(op=label):
                for entry in held:
                    with self.assertRaisesRegex(RuntimeError, self._FREED):
                        op(*entry)
        for node, plug, *_ in held:
            self.assertFalse(node.is_valid)
            self.assertIsInstance(hash(plug), int)
            self.assertFalse(hasattr(plug, "__array__"))
            self.assertFalse(hasattr(node, "__deepcopy__"))
            self.assertIs(plug.node, node)
            self.assertIs(Node(node)._dg_node, node._dg_node)
            self.assertIs(type(node >> None), Transform)

    def test_held_nodes_and_plugs_across_a_new_scene_raise(self):
        # a freed node's fn sets and MPlugs point at freed memory: reading them
        # named another node or crashed Maya; nothing reads them now
        held = self._held([cmds.createNode("transform", name=f"held{i}") for i in range(3)])
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")  # reuse the freed memory
        self._assert_freed(held)

    def test_held_nodes_and_plugs_across_a_reference_unload_raise(self):
        folder = tempfile.mkdtemp(prefix="rig_freed_ref_")
        path   = os.path.join(folder, "freed_ref.ma").replace("\\", "/")
        try:
            for i in range(2):
                cmds.createNode("transform", name=f"held{i}")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            held = self._held([f"ref:held{i}" for i in range(2)])
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_held_nodes_and_plugs_across_a_file_open_raise(self):
        # the opened file brings nodes of the same names: a held node is freed,
        # it neither names nor retargets to them
        folder = tempfile.mkdtemp(prefix="rig_freed_open_")
        path   = os.path.join(folder, "freed_open.ma").replace("\\", "/")
        try:
            names = [cmds.createNode("transform", name=f"held{i}") for i in range(2)]
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = self._held(names)
            cmds.file(path, open=True, force=True)
            self.assertTrue(all(cmds.objExists(name) for name in names))
            self._assert_freed(held)
            self.assertEqual(str(Node("held0").tx), "held0.translateX")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_deleted_node_in_the_undo_queue_keeps_its_name(self):
        cmds.undoInfo(state=True, infinity=True)
        node = Node(cmds.createNode("transform", name="gone"))
        plug = node.tx
        typed = node >> None
        cmds.delete("gone")
        for op in (
            lambda: str(plug),
            lambda: str(node.tx),
            lambda: node.ty,
            lambda: plug.get(),
            lambda: str(typed),
        ):
            with self.assertRaisesRegex(RuntimeError, "^gone already deleted!$"):
                op()
        cmds.undo()
        self.assertEqual(str(plug), "gone.translateX")

    def test_instanced_world_space_elements_connect_the_element_asked_for(self):
        _instanced_locator()
        for path, index, name in (
            ("|T2|S", 0, "T2|S.worldMatrix[0]"),
            ("|T2|S", 1, "T2|S.worldMatrix"),
            ("|T1|S", 0, "T1|S.worldMatrix"),
            ("|T1|S", 1, "T1|S.worldMatrix[1]"),
        ):
            with self.subTest(path=path, index=index):
                plug = Node(path).worldMatrix[index]
                self.assertEqual(str(plug), name)
                self.assertEqual(plug.get()[3][0], 7.0 if index else 0.0)
                dst = cmds.createNode("multMatrix")
                Node(dst).matrixIn[0] << plug
                self.assertEqual(_source_index(dst + ".matrixIn[0]"), index)
                self.assertEqual(cmds.getAttr(dst + ".matrixSum")[12], 7.0 if index else 0.0)
        # without an index: the element of the path's own instance, as in cmds
        dst = cmds.createNode("multMatrix")
        Node(dst).matrixIn[0] << Node("|T2|S").worldMatrix
        self.assertEqual(_source_index(dst + ".matrixIn[0]"), 1)
        # one memo key per element, whatever the path it is read through
        second, first = Node("|T2|S"), Node("|T1|S")
        self.assertNotEqual(
            _attribute_key(second.worldMatrix[0]), _attribute_key(second.worldMatrix[1])
        )
        self.assertEqual(
            _attribute_key(first.worldMatrix[1]), _attribute_key(second.worldMatrix[1])
        )
        # so a memoized function reads the element asked for
        from rig.matrix import decompose

        tx = [
            cmds.getAttr(str(decompose(second.worldMatrix[index])).split(".")[0] + ".outputTranslateX")
            for index in (0, 1)
        ]
        self.assertEqual(tx, [0.0, 7.0])
        # a node with one instance is named as before
        single = Node(cmds.createNode("transform", name="single"))
        self.assertEqual(str(single.worldMatrix[0]), "single.worldMatrix")

    def test_stale_path_read_through_mdagpath_first_and_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="T1")
        cmds.createNode("transform", name="T2")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.parent("T1|S", "T2", add=True, shape=True, relative=True)
        node = Node("|T2|S")
        held = node.visibility
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertTrue(node.mdagpath.isValid())
        self.assertEqual(node.mdagpath.fullPathName(), "|T1|S")
        self.assertEqual(str(node.visibility), "S.visibility")
        # undoing the removal: the node names the path it was taken through again
        cmds.undo()
        self.assertEqual(node.long_name, "|T2|S")
        self.assertEqual(str(node.visibility), "T2|S.visibility")
        self.assertEqual(str(held), "T2|S.visibility")
        self.assertEqual(node.mdagpath.fullPathName(), "|T2|S")
        # and a second removal re-resolves it again
        cmds.parent("T2|S", removeObject=True, shape=True)
        self.assertEqual(str(held), "S.visibility")
        cmds.undo()
        self.assertEqual(str(held), "T2|S.visibility")

    def test_rshift_between_plugs_says_how_to_connect(self):
        a = Node(cmds.createNode("transform", name="a"))
        b = Node(cmds.createNode("transform", name="b"))
        cmds.addAttr("b", longName="fresh", attributeType="double")
        for target in (b.ty, b.find_attr("ty"), b.fresh, Plug("b.tz")):
            with self.subTest(target=f"{type(target).__name__} {target}"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^'>>' does not connect plugs: write {target} << a\.translateX, "
                    rf"or a\.translateX\.connect\({target}, force=True\)$",
                ):
                    a.tx >> target
        self.assertIsNone(cmds.listConnections("b", source=True, destination=False))
        self.assertEqual(
            sorted(cmds.listAttr("a", userDefined=True) or []), [],
        )
        # a plain name is still a clone target
        self.assertEqual(str(a.tx >> "txCopy"), "a.txCopy")

    def test_shape_attr_read_through_a_transform_follows_the_shape(self):
        xf   = cmds.polyCube(name="c", ch=False)[0]
        node = Node(xf)
        self.assertEqual(str(node.outMesh), "cShape.outMesh")
        self.assertNotIn("outMesh", node._dg_node._attr_dict)
        cmds.delete("cShape")
        tmp = cmds.polySphere(name="tmp", ch=False)[0]
        cmds.parent(cmds.listRelatives(tmp, shapes=True)[0], xf, shape=True, relative=True)
        self.assertEqual(str(node.outMesh), "tmpShape.outMesh")
        self.assertEqual(node.outMesh.node.name, "tmpShape")
        # typed access too, and the transform's own attrs are still cached
        self.assertEqual(str((node >> None).find_attr("outMesh")), "tmpShape.outMesh")
        node.tx
        self.assertIn("translateX", node._dg_node._attr_dict)


class TestInstancedPlugIdentity(MayaTestCase):
    """Decision D-B: a plug's identity follows the Maya plug (node, attribute and
    logical indices), whatever the instance path it is named through."""

    TEST_START_NEW_SCENE = True

    def test_one_plug_through_two_paths_is_one_key(self):
        _instanced_locator()
        first, second = Node("|T1|S"), Node("|T2|S")
        for label, get in (
            ("v", lambda n: n.v),
            ("lp[0]", lambda n: n.localPosition[0]),
            ("lpx", lambda n: n.localPositionX),
        ):
            with self.subTest(plug=label):
                a, b = get(first), get(second)
                # each is still named through the path it was read from
                self.assertTrue(str(a).startswith("T1|S."))
                self.assertTrue(str(b).startswith("T2|S."))
                self.assertEqual(hash(a), hash(b))
                self.assertEqual(len({a: 1, b: 2}), 1)
                self.assertEqual(len({a, b}), 1)
                self.assertIn(b, [a])
                self.assertTrue(a.equals(b))
                self.assertTrue(bool(a == b))
                self.assertFalse(bool(a != b))
                self.assertEqual(_attribute_key(a), _attribute_key(b))
        # a plug of another attribute or another node is still another key
        self.assertEqual(len({first.v: 1, second.lodVisibility: 2, Node("T1").v: 3}), 3)
        self.assertFalse(bool(first.v == Node("T1").v))

    def test_world_space_elements_of_different_instances_are_distinct(self):
        _instanced_locator()
        first, second = Node("|T1|S"), Node("|T2|S")
        pairs = {
            # the same element read through either path: one key
            "wm[1] via T1 and T2": (first.worldMatrix[1], second.worldMatrix[1], True),
            "wm[0] via T1 and T2": (first.worldMatrix[0], second.worldMatrix[0], True),
            # an unindexed world space array is its path's element, as in cmds
            "T2 wm and wm[1]": (second.worldMatrix, first.worldMatrix[1], True),
            "T1 wm and wm[0]": (first.worldMatrix, second.worldMatrix[0], True),
            "Plug('T2|S.worldMatrix') and wm[1]": (
                Plug("T2|S.worldMatrix"), second.worldMatrix[1], True,
            ),
            # elements of different instances are different plugs
            "wm[0] and wm[1]": (second.worldMatrix[0], second.worldMatrix[1], False),
            "T1 wm and T2 wm": (first.worldMatrix, second.worldMatrix, False),
            "wim[0] and wim[1]": (
                first.worldInverseMatrix[0], first.worldInverseMatrix[1], False,
            ),
            # instObjGroups is per instance too, and its children name the index
            "iog[1] via T1 and T2": (first.instObjGroups[1], second.instObjGroups[1], True),
            "iog[0] and iog[1]": (second.instObjGroups[0], second.instObjGroups[1], False),
            "iog[1].og via T1 and T2": (
                first.instObjGroups[1].objectGroups,
                second.instObjGroups[1].objectGroups,
                True,
            ),
        }
        for label, (a, b, same) in pairs.items():
            with self.subTest(pair=label):
                self.assertEqual(a.equals(b), same)
                self.assertEqual(hash(a) == hash(b), same)
                self.assertEqual(_attribute_key(a) == _attribute_key(b), same)
        # distinct elements are distinct set members (their hashes differ, so no
        # `==`, which cannot compare two matrices, is needed)
        self.assertEqual(len({first.worldMatrix, second.worldMatrix}), 2)
        # a memoized function builds one node per element
        from rig.matrix import decompose

        via_first  = decompose(first.worldMatrix)
        via_second = decompose(second.worldMatrix)
        self.assertNotEqual(str(via_first), str(via_second))
        self.assertEqual(str(decompose(second.worldMatrix[0])), str(via_first))
        self.assertEqual(str(decompose(first.worldMatrix[1])), str(via_second))
        tx = [cmds.getAttr(str(d).split(".")[0] + ".outputTranslateX") for d in (via_first, via_second)]
        self.assertEqual(tx, [0.0, 7.0])

    def test_single_instance_world_matrix_spellings_stay_one_key(self):
        # v2.0.0a2 names both 'a.worldMatrix', and cmds resolves that to [0]
        node = Node(cmds.createNode("transform", name="a"))
        whole, element = node.worldMatrix, node.worldMatrix[0]
        self.assertEqual(str(whole), str(element))
        self.assertEqual(hash(whole), hash(element))
        self.assertTrue(whole.equals(element))
        self.assertEqual(_attribute_key(whole), _attribute_key(element))

    def test_component_element_and_its_storage_are_one_key(self):
        plane = cmds.nurbsPlane(name="np", degree=3, patchesU=1, patchesV=1, ch=False)[0]
        shape = cmds.listRelatives(plane, shapes=True)[0]
        element = Node(shape).cv[1, 2]
        storage = Plug(f"{shape}.controlPoints[6]")
        self.assertEqual(str(element), f"{shape}.cv[1][2]")
        self.assertEqual(len({element: 1, storage: 2}), 1)
        self.assertIn(storage, [element])
        self.assertNotIn(Node(shape).cv[1, 3], [element])

    def test_plug_hash_follows_the_plug_through_a_stale_path(self):
        _instanced_locator()
        cmds.undoInfo(state=True, infinity=True)
        second = Node("|T2|S")
        held   = second.v
        key    = hash(Node("|T1|S").v)
        self.assertEqual(hash(held), key)
        cmds.parent("T2|S", removeObject=True, shape=True)
        # the held node re-resolves to the surviving path, the plug is unchanged
        self.assertEqual(str(held), "S.visibility")
        self.assertEqual(hash(held), hash(Node("S").v))

    # -- round 3 step S2: the rest of the plug world follows D-B -- #

    def test_typed_attributes_through_two_paths_are_one_key(self):
        _instanced_locator()
        first, second = PyNode("|T1|S"), PyNode("|T2|S")
        a, b = first.find_attr("v"), second.find_attr("v")
        self.assertEqual((str(a), str(b)), ("T1|S.visibility", "T2|S.visibility"))
        self.assertEqual(hash(a), hash(b))
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertEqual(len({a: 1, b: 2}), 1)
        self.assertEqual(len({a, b}), 1)
        self.assertIn(b, [a])
        self.assertEqual([a].index(b), 0)
        # through a rig Node, and against the Plug of the same plug
        self.assertTrue(Node("|T1|S").find_attr("v") == Node("|T2|S").find_attr("v"))
        plug = Node("|T2|S").v
        self.assertEqual(hash(plug), hash(a))
        self.assertTrue(plug.equals(a))
        # another attribute, another node, another instance's element: another key
        self.assertFalse(a == second.find_attr("lodVisibility"))
        self.assertFalse(a == PyNode("T1").find_attr("v"))
        wm0, wm1 = first.find_attr("worldMatrix")[0], second.find_attr("worldMatrix")[1]
        self.assertEqual(len({wm0, wm1}), 2)
        self.assertFalse(wm0 == wm1)
        self.assertTrue(wm1 == first.find_attr("worldMatrix")[1])
        # a typed attr is not equal to its name
        self.assertFalse(a == "T1|S.visibility")

    def test_every_spelling_of_a_plug_is_one_key(self):
        # equals() and hash agree for every way of reaching one plug
        _instanced_locator()
        pma = cmds.createNode("plusMinusAverage", name="pma")
        spellings = [
            (Node("|T1|S").lpx, Node("|T2|S").localPosition[0], Plug("T2|S.localPositionX"),
             Node("|T1|S").lp.localPositionX, PyNode("|T2|S").find_attr("lpx")),
            (Node(pma).input3D[1].input3Dx, Plug(f"{pma}.input3D[1].input3Dx"),
             Node(pma).input3D[1][0], PyNode(pma).find_attr("input3D")[1].child(0)),
            (Node("|T2|S").worldMatrix, Node("|T1|S").worldMatrix[1],
             Plug("T2|S.worldMatrix"), PyNode("|T1|S").find_attr("worldMatrix")[1]),
        ]
        for group in spellings:
            with self.subTest(plug=str(group[0])):
                for other in group[1:]:
                    self.assertTrue(group[0].equals(other), str(other))
                    self.assertEqual(hash(group[0]), hash(other), str(other))
        # distinct plugs of the groups hash apart
        firsts = [group[0] for group in spellings] + [Node(pma).input3D[2].input3Dx]
        self.assertEqual(len({hash(plug) for plug in firsts}), len(firsts))

    def test_pluglist_membership_follows_the_maya_plug(self):
        _instanced_locator()
        plane  = cmds.nurbsPlane(name="np", degree=3, patchesU=1, patchesV=1, ch=False)[0]
        shape  = cmds.listRelatives(plane, shapes=True)[0]
        first, second = Node("|T1|S"), Node("|T2|S")
        before = sorted(cmds.ls())
        pl = PlugList([second.v, second.lodVisibility])
        self.assertIn(first.v, pl)
        self.assertEqual(pl.index(first.v), 0)
        self.assertEqual(PlugList([second.v, first.v, first.lodv]).count(first.v), 2)
        pl.remove(first.v)
        self.assertEqual([str(p) for p in pl], ["T2|S.lodVisibility"])
        self.assertNotIn(Node("T1").v, PlugList([first.v]))
        # world space elements of different instances are different plugs
        self.assertNotIn(first.worldMatrix, PlugList([second.worldMatrix]))
        self.assertIn(first.worldMatrix[1], PlugList([second.worldMatrix]))
        # a component element and the Plug of its storage are one plug too
        self.assertIn(Plug(f"{shape}.controlPoints[6]"), PlugList([Node(shape).cv[1, 2]]))
        self.assertNotIn(Plug(f"{shape}.controlPoints[7]"), PlugList([Node(shape).cv[1, 2]]))
        # a str is a name: it finds the plug named so, through that path only
        self.assertIn("T2|S.visibility", PlugList([second.v]))
        self.assertNotIn("T1|S.visibility", PlugList([second.v]))
        self.assertNotIn("T2|S.v", PlugList([second.v]))
        # none of it built a node
        self.assertEqual(sorted(cmds.ls()), before)

    def test_plug_key_survives_rename_alias_and_delete(self):
        cmds.undoInfo(state=True, infinity=True)
        node  = Node(cmds.createNode("transform", name="a"))
        held  = node.tx
        typed = PyNode("a").find_attr("tx")
        table, members, typed_set = {held: "x"}, {held}, {typed}
        key, typed_key = hash(held), hash(typed)
        cmds.rename("a", "b")
        self.assertEqual(str(held), "b.translateX")
        self.assertEqual((hash(held), hash(typed)), (key, typed_key))
        self.assertEqual(table[held], "x")
        self.assertEqual(table[Node("b").tx], "x")
        self.assertIn(PyNode("b").find_attr("translateX"), typed_set)
        members.add(held)
        members.add(Node("b").tx)
        self.assertEqual(len(members), 1)
        # an alias names the plug anew; it is still the same key
        cmds.addAttr("b", longName="knob", attributeType="double")
        knob     = Node("b").knob
        knob_key = hash(knob)
        cmds.aliasAttr("dial", "b.knob")
        self.assertEqual(str(knob), "b.dial")
        self.assertEqual(hash(knob), knob_key)
        self.assertEqual(hash(Node("b").dial), knob_key)
        # deleted to the undo queue: the held key is still found, and hashing
        # does not raise, while naming it does
        cmds.delete("b")
        self.assertEqual(hash(held), key)
        self.assertEqual(table[held], "x")
        members.discard(held)
        self.assertEqual(len(members), 0)
        with self.assertRaisesRegex(RuntimeError, "^b already deleted!$"):
            str(held)
        # a new node of the same name is another plug, another key
        cmds.createNode("transform", name="b")
        fresh = Plug("b.tx")
        self.assertNotEqual(hash(fresh), key)
        self.assertNotIn(fresh, table)
        cmds.undo()
        cmds.undo()
        self.assertEqual(table[Node("b").tx], "x")

    def test_plug_key_survives_a_new_scene(self):
        node   = Node(cmds.createNode("transform", name="a"))
        held   = node.tx
        typed  = PyNode("a").find_attr("ty")
        keys   = (hash(held), hash(typed))
        table  = {held: 1, typed: 2}
        unseen = node.tz  # never hashed before the free
        cmds.file(new=True, force=True)
        self.assertEqual((hash(held), hash(typed)), keys)
        self.assertEqual((table[held], table[typed]), (1, 2))
        self.assertIsInstance(hash(unseen), int)
        del table[held]
        self.assertEqual(list(table.values()), [2])

    def test_a_plain_str_is_a_name_not_a_plug_key(self):
        node  = Node(cmds.createNode("transform", name="a"))
        typed = PyNode("a").find_attr("tx")
        before = sorted(cmds.ls())
        for key in (node.tx, typed):
            with self.subTest(key=type(key).__name__):
                self.assertNotEqual(hash(key), hash("a.translateX"))
                self.assertIsNone({key: 1}.get("a.translateX"))
                self.assertNotIn("a.translateX", {key})
        self.assertTrue(node.tx.equals("a.translateX"))
        self.assertFalse(node.tx.equals("a.tx"))
        self.assertEqual(sorted(cmds.ls()), before)

    def test_memoized_networks_are_shared_through_two_paths(self):
        from rig import functions, random as rrandom
        from rig.matrix import decompose

        _instanced_locator()
        cmds.createNode("transform", name="G1")
        cmds.createNode("transform", name="G2")
        cmds.createNode("transform", name="X", parent="G1")
        cmds.parent("|G1|X", "G2", add=True)
        x1, x2 = Node("|G1|X"), Node("|G2|X")
        s1, s2 = Node("|T1|S"), Node("|T2|S")
        pairs = {
            "decompose(matrix)": (decompose(x1.matrix), decompose(x2.matrix)),
            "abs(lpx)": (functions.abs(s1.lpx), functions.abs(s2.lpx)),
            "lpx + 1": (s1.lpx + 1, s2.lpx + 1),
            "random.value(seed)": (
                rrandom.value(s1.lpx, seed=3), rrandom.value(s2.lpx, seed=3),
            ),
        }
        for label, (a, b) in pairs.items():
            with self.subTest(call=label):
                self.assertEqual(str(a), str(b))
        # a world space matrix is per instance: one network each
        self.assertNotEqual(
            str(decompose(x1.worldMatrix)), str(decompose(x2.worldMatrix))
        )
        # the shared network survives a rename of the instanced node
        cmds.rename("|G1|X", "Y")
        self.assertEqual(
            str(decompose(Node("|G2|Y").matrix)), str(pairs["decompose(matrix)"][0])
        )

    def test_unindexed_world_space_array_is_its_paths_element(self):
        # a static world matrix and a container publish pick the element of the
        # path the array was read through, as a connection and cmds do
        import numpy as np

        cmds.createNode("transform", name="G1")
        cmds.createNode("transform", name="G2")
        cmds.setAttr("G2.tx", 10)
        cmds.createNode("transform", name="X", parent="G1")
        cmds.parent("|G1|X", "G2", add=True)
        world = np.eye(4)
        world[3, 0] = 15.0
        for label, get, tx in (
            ("G2 wm", lambda x: x.worldMatrix, 5.0),
            ("G2 wm[1]", lambda x: x.worldMatrix[1], 5.0),
            ("G2 wm[0]", lambda x: x.worldMatrix[0], 15.0),
            ("G1 wm", lambda x: Node("|G1|X").worldMatrix, 15.0),
        ):
            with self.subTest(plug=label):
                cmds.setAttr("|G1|X.tx", 0)
                get(Node("|G2|X")) << world
                self.assertAlmostEqual(cmds.getAttr("|G1|X.tx"), tx)
        _instanced_locator()
        with container("box"):
            published = container.publish_input(Node("|T2|S").worldMatrix, "wmIn")
            first     = container.publish_input(Node("|T1|S").worldMatrix, "wmFirst")
        sources = [
            cmds.listConnections(str(p), source=True, destination=False, plugs=True)
            for p in (published, first)
        ]
        self.assertEqual(sources, [["T2|S.worldMatrix"], ["T1|S.worldMatrix"]])
        self.assertEqual(_source_index(str(published)), 1)
        # a node with one instance, as before
        single = Node(cmds.createNode("transform", name="single"))
        with container("box2"):
            one = container.publish_input(single.worldMatrix, "wmSingle")
        self.assertEqual(_source_index(str(one)), 0)
