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

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Container, Node, Plug, container
from rig.nodetypes import PyNode, Transform
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
