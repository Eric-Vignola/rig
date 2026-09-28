"""Smoke tests of the undo test helper ``rig._tests._undo`` (cmds-only walks).

The helper is shared by the round-U undo tests; these tests pin what it
records and how it walks, with Maya's own commands only (no rig edit).
"""

from maya import cmds
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, dangling_connections, mesh_state, skin_state


class TestUndoWalkQueue(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_setup_turns_the_queue_on_infinite_and_empty(self):
        self.assertTrue(cmds.undoInfo(query=True, state=True))
        self.assertTrue(cmds.undoInfo(query=True, infinity=True))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertTrue(cmds.undoInfo(query=True, redoQueueEmpty=True))

    def test_restore_puts_back_state_infinity_and_length(self):
        # step out of the setUp's queue, then run on / restore from a known finite queue
        found = self._undo_prev
        self.undo_queue_restore()
        try:
            cmds.undoInfo(state=True, infinity=False, length=23)
            self.undo_queue_on()
            self.assertTrue(cmds.undoInfo(query=True, infinity=True))
            cmds.createNode("transform", name="queued")
            self.undo_queue_restore()
            self.assertTrue(cmds.undoInfo(query=True, state=True))
            self.assertFalse(cmds.undoInfo(query=True, infinity=True))
            self.assertEqual(cmds.undoInfo(query=True, length=True), 23)
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))  # flushed
            self.undo_queue_restore()  # a second restore is a no-op
            self.assertEqual(cmds.undoInfo(query=True, length=True), 23)
        finally:
            # hand the setUp's saved queue back to tearDown
            self._undo_prev = found

    def test_undo_steps_fails_when_the_queue_runs_out(self):
        cmds.createNode("transform", name="one")
        with self.assertRaises(AssertionError):
            self.undo_steps(2)
        self.assertFalse(cmds.objExists("one"))  # the first step ran
        with self.assertRaises(AssertionError):
            self.redo_steps(2)
        self.assertTrue(cmds.objExists("one"))

    def test_undo_all_and_redo_all_count_every_entry(self):
        # plain cmds entries have an empty undoName in mayapy; the walk still reaches them
        cmds.createNode("transform", name="a")
        cmds.setAttr("a.tx", 1.0)
        cmds.undoInfo(openChunk=True, chunkName="rig.test.named")
        cmds.setAttr("a.ty", 2.0)
        cmds.undoInfo(closeChunk=True)
        self.assertEqual(self.undo_name(), "rig.test.named")
        self.assertEqual(self.undo_all(), 3)
        self.assertFalse(cmds.objExists("a"))
        self.assertEqual(self.redo_all(), 3)
        self.assertEqual((cmds.getAttr("a.tx"), cmds.getAttr("a.ty")), (1.0, 2.0))


class TestUndoWalkSceneState(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    MESHES = ("cubeShape",)

    def state(self):
        return self.scene_state(meshes=self.MESHES)

    def test_cmds_create_and_set_walk_is_exact(self):
        states = [self.state()]
        # each edit changes something the snapshot records (nodes, connections, mesh data)
        edits = (
            lambda: cmds.polyCube(name="cube", constructionHistory=False),
            lambda: cmds.move(0.0, 0.5, 0.0, "cube.vtx[0]", relative=True, objectSpace=True),
            lambda: cmds.polyUVSet("cube", create=True, uvSet="uv2"),
            lambda: cmds.polyColorSet("cube", create=True, colorSet="cs1", representation="RGBA"),
            lambda: cmds.createNode("transform", name="other"),
            lambda: cmds.connectAttr("cube.tx", "other.ty"),
        )
        for edit in edits:
            edit()
            states.append(self.state())
        # every edit changed the snapshot
        for before, after in zip(states, states[1:]):
            self.assertNotEqual(before, after)
        mesh = states[-1]["meshes"]["cubeShape"]
        self.assertEqual(mesh["uv_sets"], ["map1", "uv2"])
        self.assertEqual(mesh["color_sets"], ["cs1"])
        self.assertEqual(mesh["points"][0], (-0.5, 0.0, 0.5))  # vertex 0 (-0.5, -0.5, 0.5) moved up 0.5
        self.assertIn(("cube.translateX", "other.translateY"), states[-1]["connections"])
        # one step at a time, back and forth
        for k in range(len(edits), 0, -1):
            self.undo_steps(1)
            self.assert_state_equal(self.state(), states[k - 1], f"undo to {k - 1}")
        self.assertIsNone(self.state()["meshes"]["cubeShape"])
        for k in range(1, len(edits) + 1):
            self.redo_steps(1)
            self.assert_state_equal(self.state(), states[k], f"redo to {k}")
        # whole walks
        self.assertEqual(self.undo_all(), len(edits))
        self.assert_state_equal(self.state(), states[0], "undo_all")
        self.assertEqual(self.redo_all(), len(edits))
        self.assert_state_equal(self.state(), states[-1], "redo_all")

    def test_node_uuids_are_part_of_the_state(self):
        cmds.createNode("transform", name="n")
        before = self.scene_state()
        cmds.delete("n")
        cmds.createNode("transform", name="n")  # same name and type, a new node
        after = self.scene_state()
        self.assertEqual([x[:2] for x in before["nodes"]], [x[:2] for x in after["nodes"]])
        self.assertNotEqual(before["nodes"], after["nodes"])

    def test_missing_nodes_read_none(self):
        state = self.scene_state(meshes=["nope"], skins=["nope"])
        self.assertIsNone(state["meshes"]["nope"])
        self.assertIsNone(state["skins"]["nope"])
        cmds.createNode("transform", name="plain")
        self.assertIsNone(mesh_state("plain"))
        self.assertIsNone(skin_state("plain"))

    def test_mesh_state_by_transform_equals_by_shape(self):
        cmds.polyCube(name="cube", constructionHistory=False)
        self.assertEqual(mesh_state("cube"), mesh_state("cubeShape"))
        self.assertEqual(mesh_state("cube")["vertices"][0], (4, 4, 4, 4, 4, 4))

    def test_holes_are_recorded(self):
        cmds.polyCreateFacet(
            point=[(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4), (), (1, 0, 1), (3, 0, 1), (3, 0, 3), (1, 0, 3)],
            name="facet",
            constructionHistory=False,
        )
        holes = mesh_state("facet")["holes"]
        self.assertEqual(len(holes), 1)
        self.assertEqual(sorted(holes[0][1]), [4, 5, 6, 7])

    def test_no_dangling_in_a_wired_scene_and_its_walk(self):
        cmds.createNode("transform", name="src")
        cmds.createNode("transform", name="dst")
        cmds.connectAttr("src.tx", "dst.tx")
        cmds.connectAttr("src.worldMatrix[0]", "dst.offsetParentMatrix")
        self.assertEqual(dangling_connections(), [])
        self.assert_no_dangling()
        self.undo_all()
        self.redo_all()
        self.assertEqual(cmds.listConnections("dst.tx", source=True, plugs=True), ["src.translateX"])


class TestUndoWalkSkin(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_cmds_skin_walk_is_exact(self):
        cmds.polyCube(name="cube", constructionHistory=False)
        cmds.select(clear=True)
        cmds.joint(name="j1", position=(0, -1, 0))
        cmds.joint(name="j2", position=(0, 1, 0))
        cmds.select(clear=True)
        cmds.flushUndo()
        names = dict(meshes=["cubeShape"], skins=["skin"])
        states = [self.scene_state(**names)]
        cmds.skinCluster("j1", "j2", "cube", name="skin", toSelectedBones=True)
        states.append(self.scene_state(**names))
        cmds.skinPercent("skin", "cube.vtx[0]", transformValue=[("j1", 0.25), ("j2", 0.75)])
        states.append(self.scene_state(**names))
        skin = states[-1]["skins"]["skin"]
        self.assertEqual(skin["influences"], ["|j1", "|j1|j2"])
        self.assertEqual(skin["weights"][0], (0.25, 0.75))
        self.assertEqual(len(skin["weights"]), 8)
        self.assertIsNone(states[0]["skins"]["skin"])
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(**names), states[1], "undo skinPercent")
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(**names), states[0], "undo skinCluster")
        self.redo_steps(2)
        self.assert_state_equal(self.scene_state(**names), states[2], "redo both")
