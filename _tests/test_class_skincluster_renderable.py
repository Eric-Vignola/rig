"""``SkinCluster.create(item)``: one mesh of a cgmath Scene, a ``Renderable`` (the Renderable round,
4 of 6; the plan of 2026-11-09 and the user's decisions of 2026-11-10).

* ``create(item)`` binds ``item.skin`` to the scene node holding the mesh's uuid (the shape a
  ``Mesh.create(item)`` made); ``create(mesh, item)`` to ``mesh``, which wins over the uuid.
* Each influence, a joint name, is the scene node holding the uuid of its node in
  ``item.joints`` (a ``Transform.create_hierarchy`` of the scene made it), else the joint of that
  name: a second copy of a scene binds to its own joints, whose short names repeat the first's.
* A SkinData's ``name`` names the node unless ``name=`` does; ``serialize`` writes ``mesh_uuid``.
* An item with no skin, or a mesh no node holds: ValueError; an item with influences of its own,
  or a geometry with none: TypeError; all before anything is made.
"""

import numpy as np
from cgmath.geometry import SkinData
from cgmath.hierarchy import TransformData
from cgmath.render import Scene
from maya import cmds
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.nodetypes import Mesh, SkinCluster, Transform


def _uuid(node):
    return cmds.ls(str(node), long=True, uuid=True)[0]


def _weights():
    """A cube's 8 points: the bottom 4 on root, the top 4 on tip."""
    t = cmds.polyCube(name="probe", constructionHistory=False)[0]
    y = np.array(cmds.xform(f"{t}.vtx[*]", query=True, translation=True, objectSpace=True)).reshape(-1, 3)[:, 1]
    cmds.delete(t)
    top = (y > 0).astype(float)
    return np.stack([1.0 - top, top], axis=1)


def _scene(skinned=True, skin_name="body_skin"):
    """A Scene: joints root > tip (2 up), a cube's data on no node, skinned to both."""
    t    = cmds.polyCube(name="src", constructionHistory=False)[0]
    data = Mesh(t).serialize(world_space=False, uvs=False)
    cmds.delete(t)
    data.name        = "hero_geo"
    data.parent_uuid = None
    scene            = Scene("s")
    root             = TransformData("root", node_type="joint")
    scene.add(root)
    scene.add(TransformData("tip", node_type="joint", parent_uuid=root.uuid, translate=(0, 2, 0)))
    scene.add(data)
    if skinned:
        scene.add(SkinData(weights=_weights(), influences=["root", "tip"], name=skin_name, mesh_uuid=data.uuid))
    return scene


def _item(scene):
    items = scene.renderables()
    assert len(items) == 1, items
    return items[0]


def _influences(skin):
    return [cmds.ls(i, long=True)[0] for i in cmds.skinCluster(str(skin), query=True, influence=True)]


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_nothing_made(self, error, pattern, call):
        before = cmds.ls(long=True)
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(long=True), before)

    def built(self, scene=None):
        scene = scene or _scene()
        item  = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        return scene, item, Mesh.create(item)


# ---------------------------------------------------------------------------------------------
class TestCreateFromARenderable(_Case):
    def test_the_mesh_is_found_by_its_uuid(self):
        _, item, mesh = self.built()
        skin = SkinCluster.create(item)
        self.assertEqual(str(skin), "body_skin")
        self.assertEqual([str(g) for g in skin.get_geometries()], [str(mesh)])
        self.assertEqual(_influences(skin), ["|root", "|root|tip"])
        np.testing.assert_allclose(skin.get_weights(), item.skin.weights, atol=1e-9)
        got = skin.serialize()
        self.assertEqual(got.mesh_uuid, item.mesh.uuid)
        self.assertEqual(got, item.skin)

    def test_a_mesh_given_wins(self):
        _, item, _ = self.built()
        other = cmds.polyCube(name="other", constructionHistory=False)[0]
        skin  = SkinCluster.create(other, item)
        self.assertEqual([str(g.get_parent()) for g in skin.get_geometries()], ["other"])
        self.assertEqual(skin.serialize().mesh_uuid, _uuid(cmds.listRelatives(other, shapes=True, fullPath=True)[0]))
        np.testing.assert_allclose(skin.get_weights(), item.skin.weights, atol=1e-9)

    def test_a_second_copy_binds_to_its_own_joints(self):
        scene, item, _ = self.built()
        SkinCluster.create(item)
        scene.renew_uuids()
        again = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        self.assertEqual(len(cmds.ls("tip")), 2)           # tip names two joints now: by name, ambiguous
        Mesh.create(again)
        skin = SkinCluster.create(again)
        self.assertEqual(_influences(skin), ["|root1", "|root1|tip"])

    def test_a_joint_the_scene_does_not_hold_is_found_by_name(self):
        scene = _scene()
        item  = _item(scene)
        Mesh.create(item)
        cmds.select(clear=True)
        cmds.joint(name="root")
        cmds.joint(name="tip", position=(0, 2, 0))
        skin = SkinCluster.create(item)
        self.assertEqual(_influences(skin), ["|root", "|root|tip"])

    def test_the_mesh_s_transform_names_a_skin_the_data_does_not(self):
        _, item, _ = self.built(_scene(skin_name=None))
        self.assertEqual(str(SkinCluster.create(item)), "hero_geo_skincluster")

    def test_name_wins(self):
        _, item, _ = self.built()
        self.assertEqual(str(SkinCluster.create(item, name="mine")), "mine")

    def test_a_skin_data_s_name_names_its_node(self):
        _, item, mesh = self.built()
        self.assertEqual(str(SkinCluster.create(mesh, item.skin)), "body_skin")

    def test_one_undo_step(self):
        _, item, _ = self.built()
        self.undo_queue_on()
        SkinCluster.create(item)
        self.assertEqual(self.undo_name(), "rig.SkinCluster.create")
        self.undo_steps(1)
        self.assertEqual(cmds.ls(type="skinCluster"), [])
        self.redo_steps(1)
        self.assertEqual(cmds.ls(type="skinCluster"), ["body_skin"])


# ---------------------------------------------------------------------------------------------
class TestTheRefusals(_Case):
    def test_no_skin(self):
        _, item, _ = self.built(_scene(skinned=False))
        self.assert_nothing_made(
            ValueError, r"^SkinCluster\.create: the Renderable 'hero_geo' holds no skin \(item\.skin is None\): nothing to bind$",
            lambda: SkinCluster.create(item),
        )

    def test_a_mesh_no_node_holds(self):
        scene = _scene()
        item  = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        self.assert_nothing_made(
            ValueError,
            r"^SkinCluster\.create: no scene node holds the uuid [0-9A-F-]{36} of the Renderable 'hero_geo''s mesh: "
            r"Mesh\.create\(item\) builds it, or give the mesh: SkinCluster\.create\(mesh, item\)$",
            lambda: SkinCluster.create(item),
        )

    def test_an_item_with_influences(self):
        _, item, _ = self.built()
        self.assert_nothing_made(
            TypeError, r"^SkinCluster\.create\(item, \.\.\.\): a Renderable carries its skin's influences",
            lambda: SkinCluster.create(item, ["root"]),
        )

    def test_a_geometry_with_no_influences(self):
        _, _, mesh = self.built()
        self.assert_nothing_made(
            TypeError, r"^SkinCluster\.create\('hero_geoShape'\): a geometry takes its influences, or a SkinData",
            lambda: SkinCluster.create(mesh),
        )
