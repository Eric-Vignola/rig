import numpy as np
from maya import cmds
from rig.nodetypes import Joint, Node, SkinCluster
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


class TestSkinCluster(MayaTestCase):
    """
    Unit test for SkinCluster node class.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a joint chain
        self.joint1 = Node.create("joint", name="cube_1_joint")
        self.joint2 = Node.create("joint", name="cube_2_joint", parent=self.joint1)
        self.joint2.t.set(0, 1, 0)

        # make a cube
        self.cube = Node(cmds.polyCube(name="base", height=2, ch=False)[0])

        # create a skincluster
        self.skin = Node.create("skinCluster", self.cube, (self.joint1, self.joint2))
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[0:3]", transformValue=[(self.joint1, 1)]
        )
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[4:7]", transformValue=[(self.joint2, 1)]
        )

    def test_properties(self):
        self.assertEqual(self.skin.get_influence_objects(), [self.joint1, self.joint2])

    def test_add_influence_objects_keeps_the_selection(self):
        """Maya selects an influence it adds singly, discarding the pick."""
        cmds.selectPref(trackSelectionOrder=True)
        extra_joint = Joint.create()
        # One at a time, the way a user clicks: a bulk select coalesces the
        # components into ranges and sorts them, leaving no order to keep.
        cmds.select(clear=True)
        for index in (0, 3, 1):
            cmds.select(f"{self.cube}.vtx[{index}]", add=True)
        before = cmds.ls(orderedSelection=True, long=True)

        self.skin.add_influence_objects(extra_joint)

        self.assertEqual(cmds.ls(orderedSelection=True, long=True), before)
        # The displayed component of an order-tracking picker is the last one.
        self.assertTrue(
            cmds.ls(orderedSelection=True)[-1].endswith(".vtx[1]"),
            msg=str(cmds.ls(orderedSelection=True)),
        )
        self.assertIn(extra_joint, self.skin.get_influence_objects())

    def test_add_influence_objects_leaves_an_empty_selection_empty(self):
        extra_joint = Joint.create()
        cmds.select(clear=True)

        self.skin.add_influence_objects(extra_joint)

        self.assertEqual(cmds.ls(selection=True), [])

    def test_set_weights_keeps_the_selection(self):
        """set_weights routes joints it has to add through the same helper."""
        data = self.skin.serialize()
        self.skin.delete()
        skin = Node.create("skinCluster", self.cube, self.joint1)

        cmds.select([f"{self.cube}.vtx[0]", f"{self.cube}.vtx[3]"])
        before = cmds.ls(selection=True, long=True)

        # One joint short of the data, so exactly one influence gets added.
        skin.set_weights(data)

        self.assertEqual(cmds.ls(selection=True, long=True), before)

    def test_weights(self):
        org_data = self.skin.serialize()
        cmds.skinPercent(
            self.skin, f"{self.cube}.vtx[0]", transformValue=[(self.joint2, 1)]
        )
        data = self.skin.serialize()
        self.assertNotEqual(org_data, data)

        self.skin.set_weights(org_data)
        data = self.skin.serialize()
        self.assertEqual(org_data, data)

        # test apply skin data to a skincluster with less influences
        self.skin.delete()
        skin = Node.create("skinCluster", self.cube, self.joint1)
        skin.set_weights(org_data)
        data = skin.serialize()
        self.assertEqual(org_data, data)

        # test apply skin data to a skincluster with more influences (non-additive)
        skin.delete()
        extra_joint = Joint.create()
        skin = Node.create(
            "skinCluster", self.cube, [self.joint1, self.joint2, extra_joint]
        )
        skin.set_weights(org_data)
        data = skin.serialize()
        self.assertEqual(org_data, data)
        self.assertFalse(extra_joint.clean_name in data.influences)

        # test apply skin data to a skincluster with more influences (additive)
        skin.delete()
        skin = Node.create(
            "skinCluster", self.cube, [self.joint1, self.joint2, extra_joint]
        )
        skin.set_weights(org_data, additive=True)
        data = skin.serialize()
        self.assertNotEqual(org_data, data)
        self.assertTrue(extra_joint.clean_name in data.influences)

    def test_create_from_skin_data(self):
        # rebuild a skincluster directly from a SkinData object: influences are
        # taken from the data and weights are applied during creation
        data = self.skin.serialize()
        self.skin.delete()

        skin = SkinCluster.create(self.cube, data)
        self.assertEqual(skin.get_influence_objects(), [self.joint1, self.joint2])
        self.assertEqual(skin.serialize(), data)

    # --- copy_skincluster

    def _skin_of(self, node):
        return node.get_shape().get_deformers(node_type="skinCluster")[0]

    def test_copy_skincluster_adds_the_influences_the_target_lacks(self):
        """The target's skin gains the source joints; the vertices not copied keep
        their weights."""
        extra  = Joint.create(name="extra_joint")
        target = Node(cmds.polyCube(name="target", height=2, ch=False)[0])
        SkinCluster.create(target, extra)

        SkinCluster.copy_skincluster(str(self.cube), [f"{target}.vtx[0:3]"])

        skin = self._skin_of(target)
        self.assertEqual(skin.get_influence_objects(), [extra, self.joint1, self.joint2])
        np.testing.assert_allclose(
            skin.get_weights(), [[0, 1, 0]] * 4 + [[1, 0, 0]] * 4, rtol=0, atol=1e-12
        )

    def test_copy_skincluster_onto_an_unskinned_mesh(self):
        target = Node(cmds.polyCube(name="target", height=2, ch=False)[0])

        SkinCluster.copy_skincluster([str(self.cube)], [f"{target}.vtx[0:7]"])

        skin = self._skin_of(target)
        self.assertEqual(skin.get_influence_objects(), [self.joint1, self.joint2])
        np.testing.assert_allclose(skin.get_weights(), self.skin.get_weights(), rtol=0, atol=1e-12)

    def test_copy_skincluster_onto_a_mesh_by_name_copies_every_vertex(self):
        """A mesh name without components stands for every vertex (not the first ten)."""
        source = Node(cmds.polySphere(name="source", ch=False)[0])
        target = Node(cmds.polySphere(name="target", ch=False)[0])
        source_skin = SkinCluster.create(source, [self.joint1, self.joint2])
        weights     = np.random.default_rng(3).random((source_skin.get_weights().shape[0], 2))
        source_skin.set_weights(weights / weights.sum(axis=1, keepdims=True))
        SkinCluster.create(target, [self.joint1, self.joint2])

        SkinCluster.copy_skincluster(str(source), [str(target)])

        np.testing.assert_allclose(
            self._skin_of(target).get_weights(), source_skin.get_weights(), rtol=0, atol=1e-12
        )

    # --- several skinClusters on one mesh: the one nearest the shape (runs/r5/SKIN)

    def _skin_on_top(self, node, joint, name):
        """A second skinCluster on `node`, bound to `joint` alone: Maya puts it
        nearest the shape, above the first one."""
        skin = Node(cmds.skinCluster(str(joint), str(node), multi=True, toSelectedBones=True, name=name)[0])
        self.assertEqual(node.get_shape().get_deformers(node_type="skinCluster")[0], skin)
        return skin

    def test_copy_skincluster_reads_the_source_skin_nearest_the_shape(self):
        top    = Joint.create(name="top_joint")
        self._skin_on_top(self.cube, top, "top_skin")
        target = Node(cmds.polyCube(name="target", height=2, ch=False)[0])

        SkinCluster.copy_skincluster(str(self.cube), [f"{target}.vtx[0:7]"])

        skin = self._skin_of(target)
        self.assertEqual(skin.get_influence_objects(), [top])
        np.testing.assert_allclose(skin.get_weights(), np.ones((8, 1)), rtol=0, atol=1e-12)

    def test_copy_skincluster_writes_the_target_skin_nearest_the_shape(self):
        under, over = Joint.create(name="under_joint"), Joint.create(name="over_joint")
        target      = Node(cmds.polyCube(name="target", height=2, ch=False)[0])
        first       = SkinCluster.create(target, under)
        nearest     = self._skin_on_top(target, over, "nearest_skin")

        SkinCluster.copy_skincluster(str(self.cube), [f"{target}.vtx[0:3]"])

        self.assertEqual(nearest.get_influence_objects(), [over, self.joint1, self.joint2])
        np.testing.assert_allclose(
            nearest.get_weights(), [[0, 1, 0]] * 4 + [[1, 0, 0]] * 4, rtol=0, atol=1e-12
        )
        self.assertEqual(first.get_influence_objects(), [under])
        np.testing.assert_allclose(first.get_weights(), np.ones((8, 1)), rtol=0, atol=1e-12)

    def test_apply_skin_data_sets_the_skin_nearest_the_shape(self):
        top     = Joint.create(name="top_joint")
        nearest = self._skin_on_top(self.cube, top, "top_skin")
        before  = self.skin.get_weights()
        data    = self.skin.serialize()
        data.weights[:] = data.weights[:, ::-1]  # j1 and j2 swapped

        self.assertEqual(self.cube.get_shape().apply_skin_data(data), nearest)

        self.assertEqual(nearest.get_influence_objects(), [self.joint1, self.joint2])
        np.testing.assert_allclose(nearest.get_weights(), before[:, ::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.skin.get_weights(), before, rtol=0, atol=0)

    # --- a mesh with no skinCluster of its own (runs/r5/SKIN/review)

    def _driven(self, how):
        """A cube driven by the skinned cube through a blendShape or a wrap: it
        has no skinCluster of its own, but the cube's is in its history."""
        driven = Node(cmds.polyCube(name=f"driven_{how}", height=2, ch=False)[0])
        if how == "blendShape":
            cmds.blendShape(str(self.cube), str(driven), weight=(0, 1))
        else:
            cmds.select(str(driven), str(self.cube))
            cmds.CreateWrap()
        self.assertIsNone(driven.get_shape().get_deformers(node_type="skinCluster", first_only=True))
        return driven

    def _assert_own_skin(self, mesh, skin):
        self.assertEqual(mesh.get_shape().get_deformers(node_type="skinCluster"), [skin])

    def test_apply_skin_data_onto_a_driven_mesh_keeps_the_drivers_skin(self):
        """SkinCluster.create deleted every skinCluster of listHistory: the
        skinned mesh's that drives this one too."""
        data = self.skin.serialize()
        for how in ("blendShape", "wrap"):
            with self.subTest(how=how):
                driven = self._driven(how)

                skin = driven.get_shape().apply_skin_data(data)

                self.assertTrue(cmds.objExists(self.skin.name))
                self._assert_own_skin(self.cube, self.skin)
                self._assert_own_skin(driven, skin)

    def test_copy_skincluster_onto_a_driven_mesh_keeps_the_drivers_skin(self):
        source = Node(cmds.polyCube(name="source", height=2, ch=False)[0])
        SkinCluster.create(source, self.joint2)
        for how in ("blendShape", "wrap"):
            with self.subTest(how=how):
                driven = self._driven(how)

                SkinCluster.copy_skincluster(str(source), [str(driven)])

                self.assertTrue(cmds.objExists(self.skin.name))
                self._assert_own_skin(self.cube, self.skin)
                skin = self._skin_of(driven)
                self._assert_own_skin(driven, skin)
                self.assertEqual(skin.get_influence_objects(), [self.joint2])

    def _skinned_intermediate_target(self):
        """A cube with no skinCluster of its own whose blendShape target is a
        skinned intermediate shape: get_deformers lists the target's skin."""
        target      = Node(cmds.polyCube(name="target", height=2, ch=False)[0])
        target_skin = SkinCluster.create(target, self.joint1)
        mesh        = Node(cmds.polyCube(name="blended", height=2, ch=False)[0])
        cmds.blendShape(str(target), str(mesh), weight=(0, 1))
        target.get_shape().intermediateObject.set(True)
        self.assertEqual(mesh.get_shape().get_deformers(node_type="skinCluster"), [target_skin])
        return mesh, target_skin

    def test_apply_skin_data_leaves_the_skin_of_a_blendshape_target_alone(self):
        """The target's skinCluster was picked and given this mesh's weights."""
        mesh, target_skin = self._skinned_intermediate_target()

        skin = mesh.get_shape().apply_skin_data(self.skin.serialize())

        self.assertNotEqual(skin, target_skin)
        self.assertTrue(cmds.objExists(target_skin.name))
        self.assertEqual(target_skin.get_influence_objects(), [self.joint1])
        self.assertEqual(skin.get_influence_objects(), [self.joint1, self.joint2])
        self.assertCountEqual(mesh.get_shape().get_deformers(node_type="skinCluster"), [skin, target_skin])

    def test_copy_skincluster_reads_no_skin_from_a_blendshape_target(self):
        mesh, _ = self._skinned_intermediate_target()
        other   = Node(cmds.polyCube(name="other", height=2, ch=False)[0])

        with self.assertRaisesRegex(RuntimeError, "No skinCluster found on"):
            SkinCluster.copy_skincluster(str(mesh), [str(other)])

    def test_copy_skincluster_needs_the_target_vertices(self):
        """No default: None was a TypeError from sorted(None) (runs/r5/XFER/review/probe_skin.py)."""
        before = cmds.ls()
        with self.assertRaisesRegex(TypeError, "target_vertices"):
            SkinCluster.copy_skincluster(str(self.cube))
        self.assertEqual(cmds.ls(), before)


class TestBakeDeformation(MayaTestCase):
    """``Mesh.bake_deformation(max_influences)``: None is no cap, a count is
    the cap (runs/r5/SKIN: None was an UnboundLocalError, a count gave 4)."""

    TEST_START_NEW_SCENE = True
    JOINTS = 10

    def setUp(self):
        super().setUp()
        self.build()

    def build(self):
        """A cylinder skinned to a 10-joint chain from root_joint, under a deltaMush."""
        self.new_scene()
        body = cmds.polyCylinder(name="body", height=10, subdivisionsY=40, subdivisionsX=12, ch=False)[0]
        cmds.select(clear=True)
        joints = [
            cmds.joint(name="root_joint" if i == 0 else f"j{i}", position=(0, -5 + 10.0 * i / (self.JOINTS - 1), 0))
            for i in range(self.JOINTS)
        ]
        cmds.skinCluster(joints, body, maximumInfluences=6, dropoffRate=0.5, toSelectedBones=True)
        cmds.deltaMush(body, smoothingIterations=20)
        self.mesh = Node(body).get_shape()

    def most_influences(self, skin):
        return int((np.asarray(skin.get_weights()) > 0).sum(axis=1).max())

    def test_bake_deformation_without_a_cap_keeps_every_influence(self):
        skin = self.mesh.bake_deformation()

        self.assertEqual(skin, self.mesh.get_deformers(node_type="skinCluster")[0])
        self.assertEqual(self.most_influences(skin), self.JOINTS)

    def test_bake_deformation_caps_at_the_count_given(self):
        for cap in (2, 6, 9):
            with self.subTest(cap=cap):
                self.build()
                skin = self.mesh.bake_deformation(max_influences=cap)

                self.assertEqual(skin, self.mesh.get_deformers(node_type="skinCluster")[0])
                self.assertEqual(self.most_influences(skin), cap)


# --- undo (round U2, step S3) ------------------------------------------------------------


class TestSkinClusterUndo(UndoWalk, MayaTestCase):
    """``SkinCluster.set_weights`` and ``SkinCluster.create`` are one undo step each:
    the weights are one ``MFnSkinCluster.setWeights`` (normalised) through rig's
    plug-in command; with a SkinData the influence edits join it in a step named
    ``rig.SkinCluster.set_weights``, and a create is a step named
    ``rig.SkinCluster.create``. Checked on the scene, never on ``cmds.undo()``'s return."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.select(clear=True)
        self.j1 = cmds.joint(name="j1", position=(0, -1, 0))
        cmds.select(clear=True)
        self.j2 = cmds.joint(name="j2", position=(0, 1, 0))
        cmds.select(clear=True)
        self.j3 = cmds.joint(name="j3", position=(0, 2, 0))
        cmds.select(clear=True)
        self.cube = cmds.polyCube(name="c", height=2, constructionHistory=False)[0]
        cmds.flushUndo()

    def tearDown(self):
        try:
            cmds.undoInfo(state=True)
            cmds.flushUndo()
        finally:
            super().tearDown()

    def state(self):
        return self.scene_state(meshes=["cShape"], skins=["c_skincluster"])

    def skin(self, influences=None):
        """A skincluster on the cube from joints (not undone by the tests)."""
        skin = SkinCluster.create(self.cube, influences or [self.j1, self.j2])
        cmds.flushUndo()
        return skin

    def walk(self, before, after, cycles=2):
        """One undo step gives `before` back and one redo `after`, `cycles` times."""
        for cycle in range(cycles):
            self.undo_steps(1)
            self.assert_state_equal(self.state(), before, f"cycle {cycle} undo")
            self.redo_steps(1)
            self.assert_state_equal(self.state(), after, f"cycle {cycle} redo")
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)

    def test_set_weights_full(self):
        skin = self.skin()
        before = self.state()
        skin.set_weights(np.tile([0.25, 0.75], (8, 1)))
        after = self.state()
        self.assertNotEqual(before["skins"], after["skins"])
        self.walk(before, after)
        self.assertTrue(np.allclose(skin.get_weights(), np.tile([0.25, 0.75], (8, 1))))

    def test_set_weights_with_indices(self):
        skin = self.skin()
        before = self.state()
        skin.set_weights(np.array([[0.5, 0.5], [0.1, 0.9]]), indices=np.array([1, 3]))
        after = self.state()
        weights = skin.get_weights()
        self.assertTrue(np.allclose(weights[[1, 3]], [[0.5, 0.5], [0.1, 0.9]]))
        self.assertTrue(np.allclose(np.delete(weights, [1, 3], axis=0), np.delete(
            np.array(before["skins"]["c_skincluster"]["weights"]), [1, 3], axis=0)))
        self.walk(before, after)

    def test_set_weights_skin_data_adding_and_removing_influences_is_one_step(self):
        from cgmath.geometry import SkinData

        skin = self.skin()
        before = self.state()
        data = SkinData(influences=[self.j1, self.j3], weights=np.tile([0.6, 0.4], (8, 1)))
        skin.set_weights(data)  # adds j3, sets the weights, removes j2 (not additive)
        after = self.state()
        self.assertEqual([n.split("|")[-1] for n in after["skins"]["c_skincluster"]["influences"]], ["j1", "j3"])
        self.assertEqual(self.undo_name(), "rig.SkinCluster.set_weights")
        self.walk(before, after)
        self.assertEqual(skin.get_influence_objects(), [Joint(self.j1), Joint(self.j3)])

    def test_create_from_skin_data_undo_all_redo_all(self):
        from cgmath.geometry import SkinData

        data = SkinData(influences=[self.j1, self.j2], weights=np.tile([0.75, 0.25], (8, 1)))
        before = self.state()
        skin = SkinCluster.create(self.cube, data)
        after = self.state()
        self.assertEqual(self.undo_name(), "rig.SkinCluster.create")
        for cycle in range(2):
            self.assertEqual(self.undo_all(), 1)
            self.assert_state_equal(self.state(), before, f"undo_all {cycle}")
            self.assertEqual(self.redo_all(), 1)
            self.assert_state_equal(self.state(), after, f"redo_all {cycle}")
        # the held SkinCluster still answers, with the data's weights
        self.assertTrue(np.allclose(skin.get_weights(), data.weights))
        self.assertEqual(skin.serialize(), SkinCluster("c_skincluster").serialize())

    def test_create_replacing_a_skin_is_one_step(self):
        from cgmath.geometry import SkinData

        for label, influences in (
            ("joints", [self.j1, self.j3]),
            ("skin data", SkinData(influences=[self.j1, self.j2], weights=np.tile([0.5, 0.5], (8, 1)))),
        ):
            with self.subTest(label):
                self.skin([self.j1, self.j2, self.j3])
                before = self.state()
                SkinCluster.create(self.cube, influences)
                after = self.state()
                self.assertEqual(self.undo_name(), "rig.SkinCluster.create")
                self.walk(before, after)

    def test_a_wrong_weight_count_raises_before_any_edit(self):
        from cgmath.geometry import SkinData

        skin = self.skin()
        before = self.state()
        cases = {
            "fewer rows": lambda: skin.set_weights(np.tile([0.25, 0.75], (5, 1))),
            "more rows": lambda: skin.set_weights(np.tile([0.25, 0.75], (9, 1))),
            "a missing column": lambda: skin.set_weights(np.ones((8, 1))),
            "rows for other indices": lambda: skin.set_weights(np.ones((3, 2)), indices=np.array([0, 1])),
            "skin data rows": lambda: skin.set_weights(
                SkinData(influences=[self.j1, self.j3], weights=np.tile([0.6, 0.4], (5, 1)))
            ),
        }
        for label, call in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, r"^SkinCluster\.set_weights: "):
                    call()
                self.assert_state_equal(self.state(), before)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_create_from_skin_data_of_the_wrong_size_raises_before_any_edit(self):
        # the check ran after the delete of the current skincluster: with undo off its
        # painted weights were lost (runs\rU2\review_undo\p_skinbad.py)
        from cgmath.geometry import SkinData

        SkinCluster.create(self.cube, SkinData(influences=[self.j1, self.j2], weights=np.tile([0.9, 0.1], (8, 1))))
        cmds.flushUndo()
        before = self.state()
        cases  = {
            "fewer rows": np.tile([0.5, 0.5], (5, 1)),
            "more rows": np.tile([0.5, 0.5], (9, 1)),
            "a missing column": np.ones((8, 1)),
        }
        for undo in (True, False):
            for label, weights in cases.items():
                with self.subTest(label, undo=undo):
                    cmds.undoInfo(state=undo)
                    try:
                        with self.assertRaisesRegex(ValueError, r"^SkinCluster\.create: .* 8 points x 2 influences of c$"):
                            SkinCluster.create(self.cube, SkinData(influences=[self.j1, self.j2], weights=weights))
                    finally:
                        cmds.undoInfo(state=True)
                    self.assert_state_equal(self.state(), before)
                    self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_inside_a_user_undo_chunk(self):
        import rig

        skin = self.skin()
        before = self.state()
        with rig.undo_chunk("rig.test.skin"):
            skin.set_weights(np.tile([0.25, 0.75], (8, 1)))
            cmds.setAttr("c_skincluster.envelope", 0.5)
            skin.set_weights(np.tile([0.5, 0.5], (8, 1)), indices=None)
        after = self.state()
        self.assertEqual(self.undo_name(), "rig.test.skin")
        self.walk(before, after)

    def test_undo_off_sets_the_weights_and_queues_nothing(self):
        skin = self.skin()
        cmds.undoInfo(state=False)
        try:
            skin.set_weights(np.tile([0.25, 0.75], (8, 1)))
            skin.set_weights(np.tile([0.5, 0.5], (8, 1)))
        finally:
            cmds.undoInfo(state=True)
        self.assertTrue(np.allclose(skin.get_weights(), 0.5))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_new_scene_with_weights_queued(self):
        skin = self.skin()
        skin.set_weights(np.tile([0.25, 0.75], (8, 1)))
        skin.set_weights(np.tile([0.5, 0.5], (8, 1)))
        self.undo_steps(1)
        cmds.file(new=True, force=True)
        cube = cmds.polyCube(name="c", constructionHistory=False)[0]
        cmds.select(clear=True)
        j = cmds.joint(name="j1")
        skin = SkinCluster.create(cube, [j])
        self.assertTrue(np.allclose(skin.get_weights(), 1.0))
    # --- copy / transfer: one step each (runs\r5\XFER\review\probe_undo.py: copy_skincluster
    # onto an unskinned mesh was two rig steps and three findRelatedDeformer entries, and
    # one undo left the new skin bound to the first joint only)

    def one_step(self, name, edit, meshes, skins):
        """`edit()` is ONE undo step named `name`: one undo gives the scene back and
        empties the queue, one redo gives the edit back, twice over."""
        cmds.flushUndo()
        before = self.scene_state(meshes=meshes, skins=skins)
        edit()
        after = self.scene_state(meshes=meshes, skins=skins)
        self.assertNotEqual(before["skins"], after["skins"])
        self.assertEqual(self.undo_name(), name)
        for cycle in range(2):
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(meshes=meshes, skins=skins), before, f"cycle {cycle} undo")
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True), f"cycle {cycle}: more than one step")
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(meshes=meshes, skins=skins), after, f"cycle {cycle} redo")

    def source(self):
        skin = self.skin()
        skin.set_weights(np.tile([0.25, 0.75], (8, 1)))
        target = cmds.polyCube(name="t", height=2, constructionHistory=False)[0]
        return skin, target

    def test_copy_skincluster_onto_an_unskinned_mesh_is_one_step(self):
        _, target = self.source()
        self.one_step(
            "rig.SkinCluster.copy_skincluster",
            lambda: SkinCluster.copy_skincluster(self.cube, [f"{target}.vtx[0:7]"]),
            meshes=["tShape"], skins=["tShape_skincluster"],
        )

    def test_copy_skincluster_adding_influences_is_one_step(self):
        _, target = self.source()
        SkinCluster.create(target, [self.j3])
        self.one_step(
            "rig.SkinCluster.copy_skincluster",
            lambda: SkinCluster.copy_skincluster(self.cube, [f"{target}.vtx[0:3]"]),
            meshes=["tShape"], skins=["t_skincluster"],
        )

    def test_transfer_to_mesh_is_one_step(self):
        skin, target = self.source()
        self.one_step(
            "rig.SkinCluster.transfer_to_mesh",
            lambda: skin.transfer_to_mesh(f"{target}Shape"),
            meshes=["tShape"], skins=["tShape_skincluster"],
        )
