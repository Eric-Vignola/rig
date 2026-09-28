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
"""

from maya import cmds
from maya.api import OpenMaya

from rig import container, Node
from rig._internal.container import ContainerOptions
from rig.bridges import commands as rc
from rig.nodetypes import DisplayLayer, Follicle, Joint, Mesh, SkinCluster, Transform
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
        for node_type in ("transform", "joint", "choice", "nurbsCurve", "multiplyDivide"):
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
