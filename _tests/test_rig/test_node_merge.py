"""Tests for the merge of the DSL ``Node`` wrapper into the ``rig.nodetypes``
hierarchy. Each class names the step of the merge it belongs to:

* S0: Plug property setters, ``bool(plug)``, ``find_attr`` filters and caching,
  ``rename_attr`` with a Plug.
* S1: the owner rule (a plug's node is the node object it was read from),
  foreign plugs, stale DAG paths, held plugs across delete / reuse / undo.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Container, Node, Plug, container
from rig.nodetypes import PyNode, Transform
from rig._internal.members import Components
from rig._tests._base import MayaTestCase


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


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
