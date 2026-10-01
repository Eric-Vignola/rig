"""Round 4a, review fixes (step FIX).

* A scope registers only the nodes a call made for itself: a deformer's
  ``...ShapeOrig`` under the user's mesh, a history shape and a skin's shared
  ``bindPose`` stay out, so deleting the container leaves the mesh at rest and
  a later skin of the same joints keeps its bind pose.
* ``Node.create`` of a registered type (the typed create): ``parent=`` puts a
  DAG node in its parent's space at identity (``cmds.createNode(parent=)``,
  never world-then-parent), a type that makes just its node takes keyword
  arguments only, a type built from inputs refuses a call without them, and a
  display layer is empty unless given objects (the selection is never read).
* ``container=False`` has ``createNode``'s meaning on every creator: not
  registered; the flattened prefix and the GC tag still apply.
* The flattened prefix goes on the leaf of a namespaced name (``ns:inner_x``).
* A typed create inside a scope is one undo step.
* A typed Attribute and a Plug of one plug are one dict / set key.
* The ``=`` sugar: a data default a node class declares is Python state of the
  instance; the advice for a Maya attr a class member shadows works
  (``Plug(node.find_attr(name)) << value``); an unknown name says how to keep
  Python state.
* A plug's sibling lookup gives Maya attributes only (``node.tx.delete`` is no
  node method), for the node and for its container.
* An ordering comparison (``<`` ...) has no truth value.
* ``plug == node`` is False and builds nothing.
* ``Container.find_all`` is the root's, and a metaclass name (``wrap``) can be
  published.
* ``List`` reads its probe once per call, and a str probe finds the node it
  names.
* ``Node.find_all`` lists a type no class is registered for.
* ``plug << x`` follows the operand-shape rule (F14).
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import container, force_nodes, functions as F, List, Node
from rig._internal import list as list_module
from rig._internal.container import Container, ContainerOptions
from rig._internal.plug import Plug
from rig.bridges import commands as rc
from rig.nodetypes import (
    DGNode,
    DisplayLayer,
    Follicle,
    Joint,
    Mesh,
    SkinCluster,
    Transform,
)
from rig.nodetypes._base import Attribute
from rig._tests._base import MayaTestCase


def _members(ctn):
    """The nodes of a Maya container, as it lists them."""
    return sorted(cmds.container(str(ctn), query=True, nodeList=True) or [])


def _heights(shape):
    """The y of the first four points of mesh `shape`."""
    sel = OpenMaya.MSelectionList()
    sel.add(shape)
    return [round(p.y, 3) for p in OpenMaya.MFnMesh(sel.getDagPath(0)).getPoints()][:4]


class _SceneCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._options = {
            key: getattr(ContainerOptions, key)
            for key in ("skip_selection", "create_containers", "flatten_containers")
        }
        cmds.select(clear=True)

    def tearDown(self):
        for key, value in self._options.items():
            setattr(ContainerOptions, key, value)
        cmds.namespace(set=":")
        self.assertFalse(container.is_active)
        super().tearDown()


class TestScopeRegistersItsOwnNodes(_SceneCase):
    def _body(self, name="body"):
        cube  = cmds.polyCube(name=name, ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True, fullPath=True)[0]
        return cube, shape

    def test_deleting_the_container_of_a_skin_keeps_the_rest_shape(self):
        cube, shape = self._body()
        cmds.createNode("joint", name="j1")
        rest = _heights(shape)
        with container("rigA") as rig_a:
            skin = SkinCluster.create(cube, "j1")
        self.assertEqual(_members(rig_a), [str(skin)])
        cmds.setAttr("j1.ty", 5)
        self.assertNotEqual(_heights(shape), rest)
        cmds.delete(str(rig_a))
        cmds.setAttr("j1.ty", 0)
        self.assertFalse(cmds.ls(type="skinCluster"))
        self.assertEqual(_heights(shape), rest)

    def test_a_later_skin_keeps_the_shared_bind_pose(self):
        first, _ = self._body("a")
        second, _ = self._body("b")
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            SkinCluster.create(first, "j1")
        outside = SkinCluster.create(second, "j1")
        pose = cmds.listConnections(f"{outside}.bindPose")
        self.assertTrue(pose)
        cmds.delete(str(rig_a))
        self.assertEqual(cmds.listConnections(f"{outside}.bindPose"), pose)

    def test_the_bridges_and_node_create(self):
        target, _ = self._body("tgt")
        base, _ = self._body("base")
        cluster_base, _ = self._body("cbase")
        skinned, _ = self._body("skinned")
        extruded, _ = self._body("extruded")
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            blend = Node.create("blendShape", target, base)
            handle = rc.cluster(cluster_base)
            skin = rc.skinCluster(skinned, "j1")
            extrude = rc.polyExtrudeFacet(f"{extruded}.f[0]")
        members = _members(rig_a)
        self.assertIn(str(blend), members)
        self.assertIn(str(skin[0]), members)
        self.assertIn(str(extrude[0]), members)
        # the nodes the calls made for the user's meshes stay out
        for name in ("baseShapeOrig", "cbaseShapeOrig", "skinnedShapeOrig"):
            self.assertTrue(cmds.objExists(name), name)
            self.assertNotIn(name, members)
        self.assertFalse([m for m in members if cmds.nodeType(m) == "dagPose"])
        history = [
            m for m in members
            if cmds.nodeType(m) == "mesh" and cmds.getAttr(f"{m}.intermediateObject")
        ]
        self.assertEqual(history, [])
        # the cluster and its handle transform and shape are the call's own
        self.assertEqual([str(x) for x in handle], ["cluster1", "cluster1Handle"])
        for name in ("cluster1", "cluster1Handle", "cluster1HandleShape"):
            self.assertIn(name, members)

    def test_a_node_the_call_returns_is_its_own(self):
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            pose = rc.dagPose("j1", save=True, name="pose")
        self.assertEqual(cmds.nodeType(str(pose)), "dagPose")
        self.assertIn(str(pose), _members(rig_a))


def _local(node, attr):
    return [round(v, 3) for v in cmds.getAttr(f"{node}.{attr}")[0]]


def _world_translation(node):
    return [
        round(v, 3)
        for v in cmds.xform(str(node), query=True, worldSpace=True, translation=True)
    ]


class TestNodeCreateTypedDispatch(_SceneCase):
    """Node.create of a registered type runs the typed create (the former
    PyNode.create), with the DSL's placement and without reading the selection."""

    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="grp")
        cmds.setAttr("grp.t", 5, 0, 0)
        cmds.setAttr("grp.r", 0, 45, 0)

    def test_a_parent_puts_the_node_in_its_space(self):
        made = [
            Node.create("transform", name="t", parent="grp"),
            Node.create("transform", name="p", p="grp"),
            Transform.create(name="typed", parent=Node("grp")),
            Node.create("joint", name="j", parent="grp"),
        ]
        for node in made:
            with self.subTest(node=str(node)):
                self.assertEqual(cmds.listRelatives(str(node), parent=True), ["grp"])
                self.assertEqual(_local(node, "t"), [0.0, 0.0, 0.0])
                self.assertEqual(_local(node, "r"), [0.0, 0.0, 0.0])
                self.assertEqual(_world_translation(node), [5.0, 0.0, 0.0])
        self.assertEqual(_local(made[-1], "jo"), [0.0, 0.0, 0.0])
        self.assertIs(type(made[-1]), Joint)

    def test_a_world_node_of_the_same_name_does_not_rename_it(self):
        cmds.createNode("transform", name="t")
        node = Node.create("transform", name="t", parent="grp")
        self.assertEqual(node.long_name, "|grp|t")
        self.assertEqual(str(node), "grp|t")
        joint = Joint.create(name="t", parent="grp|t")
        self.assertEqual(joint.long_name, "|grp|t|t")

    def test_inside_a_scope(self):
        with container("box") as box:
            with container("inner"):
                node = Node.create("transform", name="t", parent="grp")
        self.assertEqual(node.long_name, "|grp|inner_t")
        self.assertEqual(_local(node, "t"), [0.0, 0.0, 0.0])
        self.assertEqual(_members(box), ["inner_t"])

    def test_a_missing_or_ambiguous_parent_makes_nothing(self):
        cmds.createNode("transform", name="twin", parent="grp")
        cmds.createNode("transform", name="twin")
        before = set(cmds.ls(long=True))
        for parent in ("nosuch", "twin"):
            with self.subTest(parent=parent):
                with self.assertRaises(ValueError):
                    Node.create("joint", name="orphan", parent=parent)
                self.assertEqual(set(cmds.ls(long=True)), before)

    def test_a_node_that_takes_no_input_takes_no_positional_argument(self):
        before = set(cmds.ls(long=True))
        for node_type in ("transform", "joint", "choice", "nurbsSurface", "multiplyDivide"):
            with self.subTest(node_type=node_type):
                with self.assertRaisesRegex(TypeError, r"keyword arguments only"):
                    Node.create(node_type, "grp")
                self.assertEqual(set(cmds.ls(long=True)), before)

    def test_a_node_built_from_inputs_needs_them(self):
        cube = cmds.polyCube(name="cube", ch=False)[0]
        cmds.select(cube)
        before = set(cmds.ls(long=True))
        for node_type, inputs in (
            ("blendShape", r"\*targets, base"),
            ("skinCluster", "geom, influences"),
            ("mesh", "mesh_data"),
            ("nurbsCurve", "points"),
            ("reference", "file_path, namespace"),
        ):
            with self.subTest(node_type=node_type):
                with self.assertRaisesRegex(TypeError, rf"needs the inputs .*{inputs}"):
                    Node.create(node_type, name="made")
                self.assertEqual(set(cmds.ls(long=True)), before)
        # the selected mesh was never deformed
        self.assertEqual(cmds.listHistory(cube), ["cubeShape"])
        # with its inputs the typed create runs, as before
        target = cmds.polyCube(name="target", ch=False)[0]
        blend  = Node.create("blendShape", target, cube)
        self.assertEqual(cmds.nodeType(str(blend)), "blendShape")
        cmds.createNode("joint", name="j1")
        skinned = cmds.polyCube(name="skinned", ch=False)[0]
        self.assertIs(type(Node.create("skinCluster", skinned, "j1")), SkinCluster)

    def test_a_display_layer_is_empty_unless_given_objects(self):
        cmds.createNode("transform", name="sel")
        cmds.createDisplayLayer(name="L0", empty=True)
        cmds.editDisplayLayerMembers("L0", "sel", noRecurse=True)
        cmds.select("sel")
        layers = [
            Node.create("displayLayer", name="L1"),
            DisplayLayer.create(name="L2"),
            DisplayLayer.get_or_create("L3"),
        ]
        with container("box"):
            layers.append(Node.create("displayLayer", name="L4"))
        for layer in layers:
            with self.subTest(layer=str(layer)):
                self.assertIs(type(layer), DisplayLayer)
                self.assertIsNone(cmds.editDisplayLayerMembers(str(layer), query=True))
        self.assertEqual(cmds.editDisplayLayerMembers("L0", query=True), ["sel"])
        self.assertEqual(cmds.ls(selection=True), ["sel"])
        # objects given, or empty=False (the selection), fill it
        cmds.createNode("transform", name="other")
        given = Node.create("displayLayer", "other", name="given")
        self.assertEqual(cmds.editDisplayLayerMembers(str(given), query=True), ["other"])
        cmds.select("sel")
        chosen = DisplayLayer.create(name="chosen", empty=False)
        self.assertEqual(cmds.editDisplayLayerMembers(str(chosen), query=True), ["sel"])


class TestContainerFalseHasOneMeaning(_SceneCase):
    """container=False leaves the node unregistered, on every creator; the
    flattened prefix and an eligible type's GC tag still apply."""

    def _mesh_data(self):
        cube = cmds.polyCube(name="src", ch=False)[0]
        data = Mesh(cmds.listRelatives(cube, shapes=True)[0]).serialize(include_uvs=False)
        cmds.delete(cube)
        return data

    def test_every_creator(self):
        data = self._mesh_data()
        with container("box") as box:
            with container("inner"):
                made = [
                    Node.create("transform", name="t", container=False),
                    Transform.create(name="typed", container=False),
                    Node.create("joint", name="j", container=False),
                    Node.create("multiplyDivide", name="m", container=False),
                    Mesh.create(data, name="mesh", container=False),
                    Node.create("mesh", data, name="nmesh", container=False),
                ]
                kept = Mesh.create(data, name="kept")
        self.assertEqual(
            [str(x) for x in made[:4]], ["inner_t", "inner_typed", "inner_j", "inner_m"]
        )
        self.assertEqual(str(made[4].get_parent()), "inner_mesh")
        self.assertEqual(str(made[5].get_parent()), "inner_nmesh")
        self.assertEqual(_members(box), sorted([str(kept.get_parent()), str(kept)]))
        self.assertTrue(cmds.attributeQuery("__rig__", node="inner_m", exists=True))
        # outside a scope nothing changes
        self.assertEqual(str(Transform.create(name="free", container=False)), "free")
        self.assertEqual(
            str(Node.create("multiplyDivide", name="fm", container=False)), "fm"
        )


class TestFlattenedPrefixKeepsTheNamespace(_SceneCase):
    def test_the_prefix_goes_on_the_leaf(self):
        cmds.namespace(add="ns")
        cmds.namespace(add="b", parent="ns")
        plane = cmds.polyPlane(name="plane", ch=False)[0]
        ref   = cmds.spaceLocator(name="ref")[0]
        before = sorted(cmds.namespaceInfo(":", listOnlyNamespaces=True, recurse=True))
        with container("outer"):
            with container("inner"):
                made = [
                    Transform.create(name="ns:x"),
                    Node.create("transform", name="ns:y"),
                    Node.create("multiplyDivide", name="ns:m"),
                    container.createNode("multiplyDivide", name="ns:b:deep"),
                    Transform.create(name=":root"),
                    Transform.create(n="ns:short"),
                ]
                follicle = Follicle.create_on_mesh(plane, ref, "ns:fol")
        self.assertEqual(
            [str(x) for x in made],
            [
                "ns:inner_x", "ns:inner_y", "ns:inner_m", "ns:b:inner_deep", "inner_root",
                "ns:inner_short",
            ],
        )
        self.assertEqual(str(follicle.get_parent()), "ns:inner_fol")
        # no namespace was made
        self.assertEqual(
            sorted(cmds.namespaceInfo(":", listOnlyNamespaces=True, recurse=True)), before
        )


class TestTypedCreateIsOneUndoStep(_SceneCase):
    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)

    def test_one_undo_reverts_the_create(self):
        with container("rig") as rig:
            cmds.flushUndo()
            Joint.create(name="j")
            Transform.create(name="t")
            self.assertEqual(_members(rig), ["j", "t"])
            cmds.undo()
            self.assertFalse(cmds.objExists("t"))
            self.assertTrue(cmds.objExists("j"))
            cmds.undo()
            self.assertFalse(cmds.objExists("j"))
            cmds.redo()
            self.assertTrue(cmds.objExists("j"))
            self.assertEqual(_members(rig), ["j"])

    def test_a_tracked_create(self):
        cube = cmds.polyCube(name="cube", ch=False)[0]
        cmds.createNode("joint", name="j1")
        with container("rig") as rig:
            cmds.flushUndo()
            skin = SkinCluster.create(cube, "j1")
            name = str(skin)
            self.assertEqual(_members(rig), [name])
            cmds.undo()
            self.assertFalse(cmds.objExists(name))
            self.assertEqual(cmds.listHistory(cube), ["cubeShape"])


class _PlugCase(_SceneCase):
    def setUp(self):
        super().setUp()
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        self.a, self.b = Node("a"), Node("b")

    def tearDown(self):
        ContainerOptions.constant_folding = True
        super().tearDown()


class TestTypedAttributeAndPlugAreOneKey(_PlugCase):
    def test_equal_objects_hash_alike(self):
        a = self.a
        typed = a.find_attr("tx")
        self.assertIsInstance(typed, Attribute)
        self.assertNotIsInstance(typed, Plug)
        before = set(cmds.ls())
        self.assertTrue(typed == a.tx)
        self.assertTrue(a.tx == typed)
        self.assertEqual(hash(typed), hash(a.tx))
        self.assertIn(typed, {a.tx, a.ty})
        self.assertIn(a.tx, {typed})
        self.assertEqual({typed: 1}[a.tx], 1)
        self.assertIn(a.tx, set(a.list_attr(keyable=True)))
        self.assertEqual(len({typed, a.tx, a.find_attr("translateX")}), 1)
        self.assertEqual(set(cmds.ls()), before)
        # another plug is another key
        self.assertNotIn(a.find_attr("ty"), {a.tx})
        self.assertNotEqual(hash(a.find_attr("ty")), hash(a.tx))

    def test_under_force_nodes_a_mixed_lookup_builds_as_two_plugs_do(self):
        a = self.a
        typed = a.find_attr("tx")
        with force_nodes():
            before = set(cmds.ls())
            self.assertIn(typed, {a.tx})
            built = set(cmds.ls()) - before
            # the equal node two Plugs of the plug build (memoized: the same one)
            self.assertIn(a.tx, {Node("a").tx})
            self.assertEqual(set(cmds.ls()) - before, built)
        self.assertEqual(len(built), 1)


class TestSetattrSugarState(_SceneCase):
    def test_a_declared_default_is_python_state(self):
        class _Ctl(Transform):
            CUSTOM_NODE_TYPE = "r4aReviewCtl"
            side  = None
            cache = ()

            def __init__(self, node):
                super().__init__(node)
                self.side = "L"

            def remember(self, value):
                self.cache = (value,)

        ctl = _Ctl.create(name="hand_ctl")
        self.assertEqual(ctl.side, "L")
        self.assertEqual(vars(ctl)["side"], "L")
        self.assertIsNone(_Ctl.side)
        ctl.side = "R"
        self.assertEqual(ctl.side, "R")
        ctl.remember(3)
        self.assertEqual(ctl.cache, (3,))
        # the node casts again (its __init__ runs), and a Maya attr is no state
        again = Node("hand_ctl")
        self.assertIs(type(again), _Ctl)
        self.assertEqual(again.side, "L")
        self.assertFalse(cmds.attributeQuery("side", node="hand_ctl", exists=True))
        ctl.tx = 2
        self.assertEqual(cmds.getAttr("hand_ctl.tx"), 2.0)
        self.assertNotIn("tx", vars(ctl))

    def test_an_undeclared_name_says_how_to_keep_state(self):
        node = Node(cmds.createNode("transform", name="x"))
        with self.assertRaisesRegex(
            AttributeError, r"^Attribute not found: .*declare a default in its class body, "
            r"label = None, or use a '_' name"
        ):
            node.label = 1
        self.assertNotIn("label", vars(node))
        # the constants of the rig classes stay constants of the class
        layer = DisplayLayer.create(name="L")
        layer.DEFAULT = "other"
        self.assertEqual(DisplayLayer.DEFAULT, "defaultLayer")

    def test_the_advice_for_a_shadowed_maya_attr_works(self):
        curve = cmds.curve(point=[(0, 0, 0), (1, 0, 0)], degree=1)
        shape = Node(cmds.listRelatives(curve, shapes=True)[0])
        maker = Node(cmds.createNode("makeNurbCircle", name="mk"))
        with self.assertRaisesRegex(
            AttributeError, r"use Plug\(node\.find_attr\('create'\)\) << value"
        ):
            shape.create = maker.outputCurve
        # the advised line
        Plug(shape.find_attr("create")) << maker.outputCurve
        self.assertEqual(
            cmds.listConnections(f"{shape}.create", source=True, plugs=True),
            ["mk.outputCurve"],
        )
        cmds.addAttr("mk", longName="rename", attributeType="double")
        with self.assertRaisesRegex(AttributeError, r"Plug\(node\.find_attr\('rename'\)\)"):
            maker.rename = 3
        Plug(maker.find_attr("rename")) << 3
        self.assertEqual(cmds.getAttr("mk.rename"), 3.0)


class TestSiblingLookupGivesMayaAttributes(_PlugCase):
    def test_no_node_member_through_a_plug(self):
        a = self.a
        for name in ("delete", "rename", "uuid", "exists", "node_type", "duplicate",
                     "find_attr", "create", "namespace", "list_attr"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AttributeError, "no child or sibling attribute"):
                    getattr(a.tx, name)
        self.assertTrue(cmds.objExists("a"))
        # a sibling attribute and a compound child still resolve
        decompose = Node(cmds.createNode("decomposeMatrix", name="dm"))
        self.assertEqual(str(decompose.outputTranslate.outputRotate), "dm.outputRotate")
        self.assertEqual(str(a.t.tx), "a.translateX")
        self.assertEqual(str(a.tx.ty), "a.translateY")
        # a Maya attr a node member shadows is reached through a sibling plug
        curve = cmds.curve(point=[(0, 0, 0), (1, 0, 0)], degree=1)
        shape = Node(cmds.listRelatives(curve, shapes=True)[0])
        self.assertEqual(str(shape.local.create), f"{shape}.create")

    def test_the_container_sibling_gives_published_attributes_only(self):
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "blend")
        self.assertIsInstance(box, Container)
        self.assertEqual(str(inner.ty.blend), str(box.blend))
        for name in ("cleanup", "delete", "uuid"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    getattr(inner.ty, name)


class TestOrderingHasNoTruthValue(_PlugCase):
    def test_the_idioms_raise(self):
        a, b = self.a, self.b
        cmds.setAttr("a.tx", -5)
        for label, idiom in (
            ("if", lambda: bool(a.tx > 0)),
            ("sorted", lambda: sorted([a.tz, a.tx, a.ty])),
            ("max", lambda: max(a.tx, b.tx)),
            ("chained", lambda: 0 < a.tx < 1),
            ("not", lambda: not (a.tx <= b.tx)),
        ):
            with self.subTest(idiom=label):
                with self.assertRaisesRegex(TypeError, r"ordering comparison .*\.get\(\)"):
                    idiom()
        # the result is a plug, for the network
        result = a.tx >= b.tx
        self.assertIsInstance(result, Plug)
        self.assertIsInstance(F.greater_than(a.tx, b.tx), Plug)
        # == / != keep their identity truth, and a plain plug is true
        self.assertTrue(a.tx)
        self.assertFalse(bool(a.tx == b.tx))
        self.assertTrue(bool(a.tx != b.tx))
        # numbers fold, and a List of results is a list
        self.assertEqual(sorted([a.tx, b.tx], key=str), [a.tx, b.tx])
        self.assertTrue(len(List([a.tx, b.tx]) > 0), 2)


class TestPlugEqualsNode(_PlugCase):
    def test_a_node_is_never_the_plug(self):
        a, b = self.a, self.b
        before = set(cmds.ls())
        self.assertIs(a.tx == b, False)
        self.assertIs(a.tx != b, True)
        self.assertIn(b, [a.tx, b])
        self.assertEqual([a.tx, b].index(b), 1)
        self.assertNotIn(b, [a.tx])
        with container("box"):
            self.assertIn(b, [a.tx, b])
        self.assertEqual(set(cmds.ls()) - before, {"box"})


class TestContainerFindAllAndPublishNames(_SceneCase):
    def test_find_all_is_the_roots(self):
        cmds.createNode("joint", name="j1")
        self.assertIs(Container.find_all.__func__, Node.find_all.__func__)
        self.assertEqual([str(x) for x in Container.find_all("joint")], ["j1"])

    def test_a_metaclass_name_can_be_published(self):
        with container("box") as box:
            inner = Node.create("transform", name="inner")
            container.publish_input(inner.tx, "wrap")
            with self.assertRaisesRegex(ValueError, r"'cleanup' is a Container attribute"):
                container.publish_input(inner.ty, "cleanup")
        box.wrap = 2.0
        self.assertEqual(cmds.getAttr("inner.tx"), 2.0)
        self.assertIsInstance(box.wrap, Plug)


class TestListProbes(_PlugCase):
    def test_a_str_probe_finds_the_node_it_names(self):
        cmds.namespace(add="ns")
        cmds.createNode("transform", name="ns:c")
        nodes = List([self.a, Node("ns:c")])
        for text, found in (("a", True), ("|a", True), ("ns:c", True), (":ns:c", True),
                            ("|ns:c", True), ("c", False), ("nosuch", False), ("a.tx", False)):
            with self.subTest(text=text):
                self.assertIs(text in nodes, found)
        self.assertEqual(nodes.index("|ns:c"), 1)
        self.assertEqual(nodes.count("|a"), 1)
        # another instance path is another DAG node object, as for ==
        shape = cmds.createNode("locator", name="S", parent="a")
        cmds.parent(shape, "b", shape=True, addObject=True)
        self.assertNotIn("|b|S", List([Node("|a|S")]))
        self.assertIn("|a|S", List([Node("|a|S")]))

    def test_the_probe_is_read_once(self):
        a = self.a
        nodes = List([Node(cmds.createNode("transform", name=f"n{i}")) for i in range(6)] + [a])
        plugs = List([Node(f"n{i}").tx for i in range(6)] + [a.tx])
        # results: a hit, a miss, the probe object itself, another object of it
        self.assertIn(a, nodes)
        self.assertIn(Node("a"), nodes)
        self.assertNotIn(self.b, nodes)
        self.assertEqual(nodes.index(a), 6)
        self.assertIn(a.tx, plugs)
        self.assertIn(Node("a").translateX, plugs)
        self.assertIn(a.find_attr("tx"), plugs)
        self.assertNotIn(a.ty, plugs)
        self.assertEqual(plugs.count(a.tx), 1)
        # the probe is named once per call, whatever the number of elements
        probe_node, probe_plug = self.b, self.b.tx
        calls     = []
        node_str  = type(probe_node).__str__
        plug_name = Attribute.full_name.fget

        def counted_str(node):
            if node is probe_node:
                calls.append("node")
            return node_str(node)

        def counted_name(attr):
            if attr is probe_plug:
                calls.append("plug")
            return plug_name(attr)

        with mock.patch.object(type(probe_node), "__str__", counted_str), mock.patch.object(
            Plug, "full_name", property(counted_name)
        ):
            self.assertNotIn(probe_node, nodes)
            self.assertNotIn(probe_plug, plugs)
            self.assertNotIn(probe_plug, nodes)
        self.assertEqual(calls, ["node", "plug", "plug"])

    def test_a_deleted_probe_or_element_raises(self):
        cmds.undoInfo(state=True, infinity=True)
        gone = Node(cmds.createNode("transform", name="gone"))
        plug = gone.tx
        cmds.delete("gone")
        for probe in (gone, plug):
            with self.subTest(probe=type(probe).__name__):
                with self.assertRaisesRegex(RuntimeError, "already deleted!"):
                    probe in List([self.a, self.a.tx])
        for elements in (List([self.a, gone]), List([self.a.tx, plug])):
            with self.assertRaisesRegex(RuntimeError, "already deleted!"):
                self.b in elements
            with self.assertRaisesRegex(RuntimeError, "already deleted!"):
                self.b.tx in elements
        self.assertNotIn(gone, List([]))


class TestFindAllOfAnyType(_SceneCase):
    def test_an_unregistered_type(self):
        cmds.createNode("multiplyDivide", name="md1")
        cmds.createNode("multiplyDivide", name="md2")
        found = Node.find_all("multiplyDivide")
        self.assertEqual(sorted(str(x) for x in found), ["md1", "md2"])
        self.assertTrue(all(type(x) is DGNode for x in found))
        # exact_type=False lists derived types
        cmds.createNode("transform", name="t")
        wide = {str(x) for x in Node.find_all("dagNode", exact_type=False)}
        self.assertIn("t", wide)
        with self.assertRaisesRegex(ValueError, "not a Maya node type"):
            Node.find_all("nosuchType")


class TestLshiftOperandShapes(_PlugCase):
    def test_sets_and_dicts_are_refused_and_iterators_read(self):
        a = self.a
        for value, text in (
            ({3, 1, 2}, r"a\.translate << \{1, 2, 3\}: a set is unordered"),
            (frozenset({1, 2, 3}), r"a frozenset is unordered"),
            ({"x": 1}, r"a dict is a mapping, not a sequence"),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(TypeError, text):
                    a.t << value
        self.assertEqual(cmds.getAttr("a.t")[0], (0.0, 0.0, 0.0))
        a.t << (x for x in (1, 2, 3))
        self.assertEqual(cmds.getAttr("a.t")[0], (1.0, 2.0, 3.0))
        a.t << iter([4, 5, 6])
        self.assertEqual(cmds.getAttr("a.t")[0], (4.0, 5.0, 6.0))
        a.t = map(float, (7, 8, 9))
        self.assertEqual(cmds.getAttr("a.t")[0], (7.0, 8.0, 9.0))
        # a generator of plugs connects
        a.t << (p for p in (self.b.tx, self.b.ty, self.b.tz))
        self.assertEqual(
            cmds.listConnections("a.tx", source=True, plugs=True), ["b.translateX"]
        )
        # a set on the right of a DSL operator was already refused
        with self.assertRaises(TypeError):
            a.tx + {1}
