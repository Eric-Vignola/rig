"""``Transform.key_hierarchy`` and ``rig.create_scene`` / ``rig.load_scene`` (the Renderable
round, 5 of 6; the plan of 2026-11-09 and the user's decisions of 2026-11-10).

* ``key_hierarchy(clip)`` keys a ClipData onto the nodes it describes, found by uuid, else by
  name; a miss or a name of several nodes is a ValueError before any edit; a plain
  HierarchyData poses, no key. One undo step.
* ``create_scene(scene)``: the nodes, each mesh under its node, its blendShape, its skin bound at
  rest, then the active take keyed; a List of what was made, one undo step, the selection kept.
  Every refusal before anything is made: a uuid the Maya scene holds, two deformers on a mesh,
  a skin in a scene of takes alone. ``uuids=False`` joins the parts by what the call made.
  Cameras, lights and materials left out with a warning.
* ``load_scene(path)`` reads by extension and builds.
"""

import importlib.util
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
import rig
from cgmath.geometry import MaterialData, MorphData, SkinData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.render import Scene
from maya import cmds
from rig import create_scene, load_scene
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig._tests.test_class_hierarchy_clip import _clip, _moving_curves, _worlds
from rig.nodetypes import Mesh, Transform


def _uuid(node):
    return cmds.ls(str(node), long=True, uuid=True)[0]


def _weights():
    """A cube's 8 points: the bottom 4 on root, the top 4 on tip."""
    t = cmds.polyCube(name="probe_cube", constructionHistory=False)[0]
    y = np.array(cmds.xform(f"{t}.vtx[*]", query=True, translation=True, objectSpace=True)).reshape(-1, 3)[:, 1]
    cmds.delete(t)
    top = (y > 0).astype(float)
    return np.stack([1.0 - top, top], axis=1)


def _scene(take=True, skinned=True, morphs=True):
    """A Scene: joints root > tip, a transform geo (3 along X) holding a cube's mesh, skinned to
    root and tip, with a target and its in-between; a take of 4 frames moving tip up."""
    t    = cmds.polyCube(name="src", constructionHistory=False)[0]
    data = Mesh(t).serialize(world_space=False, uvs=False)
    cmds.delete(t)
    scene = Scene("hero")
    root  = TransformData("root", node_type="joint")
    geo   = TransformData("geo", translate=(3, 0, 0))
    scene.add(root)                                   # a node takes its uuid as it is added
    scene.add(TransformData("tip", node_type="joint", parent_uuid=root.uuid, translate=(0, 2, 0)))
    scene.add(geo)
    data.name, data.parent_uuid = None, geo.uuid
    scene.add(data)
    if skinned:
        scene.add(SkinData(weights=_weights(), influences=["root", "tip"], name="geo_skin", mesh_uuid=data.uuid))
    if morphs:
        for at in (0.5, 1.0):
            scene.add(MorphData("smile", offsets=np.array([[0, at, 0]]), indices=np.array([0]), at=at, weight=0.25,
                                deformer="face_bs", mesh_uuid=data.uuid))
    if take:
        clip                           = ClipData(scene.hierarchy, frames=4, start_frame=1)
        clip.frames["tip"].translate_y = np.array([2.0, 3.0, 4.0, 5.0])
        scene.clip                     = clip
    return scene


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_nothing_made(self, error, pattern, call):
        before = cmds.ls(long=True)
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(long=True), before)

    def folder(self):
        folder = tempfile.mkdtemp(prefix="rig_scene_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return folder


# ---------------------------------------------------------------------------------------------
class TestKeyHierarchy(_Case):
    def test_keys_the_nodes_it_describes(self):
        clip  = _clip()
        nodes = Transform.create_hierarchy(clip)          # wired, keyed; the keys taken off
        cmds.delete(cmds.ls(type="animCurve"))
        keyed = Transform.key_hierarchy(clip)
        self.assertEqual([n.get_long_name() for n in keyed], [n.get_long_name() for n in nodes])
        self.assertEqual(sorted(cmds.ls(type="animCurve")), _moving_curves())
        self.assertLessEqual(np.abs(_worlds(nodes, clip) - clip.frames.world_matrix).max(), 1e-9)

    def test_found_by_name_when_no_node_holds_the_uuid(self):
        clip  = _clip()
        nodes = Transform.create_hierarchy(clip, uuids=False)
        cmds.delete(cmds.ls(type="animCurve"))
        Transform.key_hierarchy(clip)
        self.assertEqual(sorted(cmds.ls(type="animCurve")), _moving_curves())
        self.assertLessEqual(np.abs(_worlds(nodes, clip) - clip.frames.world_matrix).max(), 1e-9)

    def test_a_miss_is_refused_before_any_edit(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        cmds.delete(cmds.ls(type="animCurve"))
        cmds.delete("probe")
        before = cmds.getAttr("root.translate")
        self.assert_nothing_made(
            ValueError, r"^Transform\.key_hierarchy: no node, or several, holds the uuid or the name of 1 node\(s\) of the data: 'probe' \(none\)",
            lambda: Transform.key_hierarchy(clip),
        )
        self.assertEqual(cmds.getAttr("root.translate"), before)

    def test_a_name_of_two_nodes_is_refused(self):
        clip = _clip()
        Transform.create_hierarchy(clip, uuids=False)
        Transform.create_hierarchy(clip, uuids=False)
        with self.assertRaisesRegex(ValueError, r"'spine' \(\|root1\|spine, \|root\|spine\)"):
            Transform.key_hierarchy(clip)

    def test_a_hierarchy_poses_no_key(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        cmds.delete(cmds.ls(type="animCurve"))
        cmds.setAttr("root.translate", 9, 9, 9)
        pose = clip.at(clip.start_frame + 2)
        Transform.key_hierarchy(pose)
        self.assertEqual(cmds.ls(type="animCurve"), [])
        np.testing.assert_allclose(cmds.getAttr("root.translate")[0], pose["root"].translate, atol=1e-9)

    def test_one_undo_step_and_the_selection(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        cmds.delete(cmds.ls(type="animCurve"))
        cmds.select("grp")
        self.undo_queue_on()
        Transform.key_hierarchy(clip)
        self.assertEqual(self.undo_name(), "rig.Transform.key_hierarchy")
        self.assertEqual(cmds.ls(selection=True), ["grp"])
        self.undo_steps(1)
        self.assertEqual(cmds.ls(type="animCurve"), [])
        self.redo_steps(1)
        self.assertEqual(sorted(cmds.ls(type="animCurve")), _moving_curves())

    def test_the_guards(self):
        with self.assertRaisesRegex(TypeError, r"^Transform\.key_hierarchy\(\) takes the data, not a file"):
            Transform.key_hierarchy("walk.npz")
        with self.assertRaisesRegex(TypeError, r"^Transform\.key_hierarchy\(\) takes a ClipData or a HierarchyData, not a int$"):
            Transform.key_hierarchy(3)


# ---------------------------------------------------------------------------------------------
class TestCreateScene(_Case):
    def test_the_whole_scene(self):
        scene = _scene()
        item  = scene.renderables()[0]
        cmds.select(clear=True)
        made = create_scene(scene)
        self.assertEqual(
            [repr(n) for n in made],
            ['Joint("root")', 'Joint("tip")', 'Transform("geo")', 'Mesh("geoShape")', 'BlendShape("face_bs")', 'SkinCluster("geo_skin")'],
        )
        self.assertEqual(cmds.ls(selection=True), [])
        # the nodes and the mesh take the data's uuids; the mesh sits under its node
        for data in scene.hierarchy:
            self.assertEqual(_uuid(cmds.ls(data.uuid.upper())[0]), data.uuid.upper())
        self.assertEqual(_uuid(made[3]), item.mesh.uuid.upper())
        self.assertEqual(made[3].get_parent().get_long_name(), "|geo")
        # the targets, the skin bound at rest, then the take keyed
        self.assertEqual(made[4].serialize(), item.morphs)
        self.assertEqual(cmds.skinCluster("geo_skin", query=True, influence=True), ["root", "tip"])
        np.testing.assert_allclose(made[5].get_weights(), item.skin.weights, atol=1e-9)
        self.assertEqual(cmds.ls(type="animCurve"), ["tip_translateY"])
        self.assertEqual([cmds.getAttr("tip.ty", time=f) for f in (1, 4)], [2.0, 5.0])

    def test_one_undo_step(self):
        self.undo_queue_on()
        create_scene(_scene())
        self.assertEqual(self.undo_name(), "rig.create_scene")
        self.undo_steps(1)
        self.assertEqual(cmds.ls(type=("joint", "mesh", "blendShape", "skinCluster", "animCurve")), [])
        self.redo_steps(1)
        self.assertEqual(len(cmds.ls(type="skinCluster")), 1)

    def test_a_second_copy_takes_fresh_uuids(self):
        scene = _scene()
        create_scene(scene)
        self.assert_nothing_made(
            ValueError, r"^create_scene\(\): the node 'root' has the uuid .*, which '\|root' holds", lambda: create_scene(scene)
        )
        scene.renew_uuids()
        again = create_scene(scene)
        self.assertEqual(again[0].get_long_name(), "|root1")
        self.assertEqual(cmds.skinCluster(str(again[5]), query=True, influence=True), ["root1", "root1|tip"])

    def test_uuids_false_joins_by_what_it_made(self):
        scene = _scene()
        create_scene(scene)
        twin = create_scene(scene, uuids=False)
        self.assertEqual(twin[3].get_parent().get_long_name(), "|geo1")
        self.assertEqual(cmds.skinCluster(str(twin[5]), query=True, influence=True), ["root1", "root1|tip"])
        self.assertEqual(sorted(cmds.ls(type="animCurve")), ["tip_translateY", "tip_translateY1"])

    def test_under_a_parent(self):
        scene                       = _scene(take=False)
        scene.meshes[0].parent_uuid = None
        cmds.setAttr(cmds.createNode("transform", name="top") + ".ty", 10)
        made = create_scene(scene, parent="top")
        self.assertEqual(made[0].get_long_name(), "|top|root")
        self.assertEqual(made[3].get_parent().get_parent().get_long_name(), "|top")

    def test_a_scene_of_takes_alone_builds_from_its_take(self):
        scene           = _scene(skinned=False)
        scene.hierarchy = None
        made            = create_scene(scene)
        self.assertEqual([repr(n) for n in made][:3], ['Joint("root")', 'Joint("tip")', 'Transform("geo")'])
        self.assertEqual(cmds.ls(type="animCurve"), ["tip_translateY"])

    def test_a_skin_in_a_scene_of_takes_alone_is_refused(self):
        scene           = _scene()
        scene.hierarchy = None
        self.assert_nothing_made(
            ValueError, r"^create_scene: the scene holds takes alone, no nodes at rest, so its 1 skin\(s\) have no bind pose",
            lambda: create_scene(scene),
        )

    def test_two_deformers_on_a_mesh_are_refused(self):
        scene                    = _scene()
        scene.morphs[1].deformer = "other_bs"
        self.assert_nothing_made(ValueError, r"name 2 deformers \('face_bs', 'other_bs'\)", lambda: create_scene(scene))

    def test_an_influence_nothing_carries_is_refused(self):
        scene                     = _scene(take=False)
        scene.skins[0].influences = ["root", "elbow"]
        self.assert_nothing_made(
            ValueError, r"^create_scene: the skin of the mesh 'geo' names influence\(s\) \['elbow'\]", lambda: create_scene(scene)
        )

    def test_what_is_left_out_is_named(self):
        scene = _scene(take=False)
        scene.add(MaterialData(name="skin", mesh_uuid=scene.meshes[0].uuid, color=(1, 0, 0)))
        with self.assertLogs("rig.scene", logging.WARNING) as logs:
            create_scene(scene)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("the scene's 1 material record(s) are left out", logs.output[0])

    def test_a_scene_without_nodes_or_deformers(self):
        scene = Scene("bare")
        t     = cmds.polyCube(name="src", constructionHistory=False)[0]
        data  = Mesh(t).serialize(world_space=True, uvs=False)
        cmds.delete(t)
        data.name = "box"
        scene.add(data)
        self.assertEqual([repr(n) for n in create_scene(scene)], ['Mesh("boxShape")'])

    def test_the_guards(self):
        with self.assertRaisesRegex(TypeError, r"^create_scene\(\) takes the data, not a file: load_scene\('a\.fbx'\) reads one$"):
            create_scene("a.fbx")
        with self.assertRaisesRegex(TypeError, r"^create_scene\(\) takes a cgmath Scene, not a HierarchyData$"):
            create_scene(HierarchyData())
        with self.assertRaisesRegex(TypeError, r"uuids= is True"):
            create_scene(_scene(), uuids=1)

    def test_rig_exports_them(self):
        self.assertIn("create_scene", rig.__all__)
        self.assertIn("load_scene", rig.__all__)
        self.assertIs(rig.create_scene, rig.scene.create_scene)


# ---------------------------------------------------------------------------------------------
class TestLoadScene(_Case):
    def test_a_cgmath_file(self):
        path = os.path.join(self.folder(), "hero.npz")
        _scene().save(path)
        made = load_scene(path)
        self.assertEqual(repr(made[-1]), 'SkinCluster("geo_skin")')

    @unittest.skipUnless(importlib.util.find_spec("fbx"), "needs the FBX SDK's Python module (fbx)")
    def test_an_fbx(self):
        path  = _scene().save_fbx(os.path.join(self.folder(), "hero.fbx"))
        made  = load_scene(path, verbose=False)
        kinds = [type(n).__name__ for n in made]
        self.assertEqual(sorted(kinds[:3]), ["Joint", "Joint", "Transform"])
        self.assertEqual(kinds[3:], ["Mesh", "BlendShape", "SkinCluster"])
        self.assertEqual([(m.name, m.at) for m in made[4].serialize()], [("smile", 0.5), ("smile", 1.0)])
        self.assertEqual(cmds.ls(type="animCurve"), ["tip_translateY"])

    def test_the_refusals(self):
        with self.assertRaisesRegex(ValueError, r"^load_scene: '.*hero\.abc': no reader for its extension"):
            load_scene("hero.abc")
        with self.assertRaisesRegex(FileNotFoundError, r"^load_scene: no file '.*nowhere\.fbx'$"):
            load_scene(os.path.join(self.folder(), "nowhere.fbx"))
        with self.assertRaisesRegex(TypeError, r"^load_scene\(\) takes a file, not a Scene"):
            load_scene(_scene())


# ---------------------------------------------------------------------------------------------
@unittest.skipUnless(os.environ.get("RIG_SCENE_FBX"), "set RIG_SCENE_FBX to an fbx to build (retarget/test_assets/1-maya.fbx)")
class TestARealFile(_Case):
    def test_it_builds(self):
        scene = Scene.load_fbx(os.environ["RIG_SCENE_FBX"], verbose=False)
        made  = create_scene(scene)
        items = scene.renderables()
        self.assertEqual(len(cmds.ls(type="mesh", noIntermediate=True)), len(items))
        self.assertEqual(len(cmds.ls(type="skinCluster")), sum(item.skin is not None for item in items))
        self.assertGreater(len(made), len(items))
