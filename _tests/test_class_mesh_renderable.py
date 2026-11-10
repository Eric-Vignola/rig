"""``Mesh.create(item)``: one mesh of a cgmath Scene, a ``Renderable`` (the Renderable round,
2 of 6; the plan of 2026-11-09 and the user's decisions of 2026-11-10).

* ``item.mesh`` is built with ``item.uvs`` as the UV sets, named ``item.name`` (the mesh's name,
  else its node's) unless ``name=`` says; the shape takes ``item.mesh.uuid``.
* Where it goes: under the scene node holding ``item.mesh.parent_uuid`` (the node a
  ``Transform.create_hierarchy`` of the same scene made, which took the data's uuid), as
  ``transform=`` puts a shape; when the scene holds the node and Maya does not, a new transform
  at the node's world matrix; with no node, at identity. ``transform=`` wins over the node.
* ``uvs=`` or ``hierarchy=`` with an item is a TypeError before anything is made: the item
  carries them. The other keywords apply (``name=``, ``normals=``, ``colors=``, ``uuids=``,
  ``world_space=``, attributes).
* Materials, skin, joints and blend shape targets are left alone: the shape joins
  ``initialShadingGroup``, no deformer is made (a later release / the deformers' creates).
* ``Mesh.load(item)`` is refused as any object that is no MeshData: a Renderable is no file.
"""

import numpy as np
from cgmath.geometry import MaterialData, SkinData
from cgmath.hierarchy import TransformData
from cgmath.render import Scene
from maya import cmds
from rig import Node
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.nodetypes import Mesh, Transform


def _uuid(node):
    return cmds.ls(node.get_long_name(), uuid=True)[0]


def _scene(on_node=True, named=True):
    """A Scene: ``grp`` (translated 10 up) holding ``hero`` (5 along X, turned 90 about Y), and a
    cube's data on hero (``on_node``) with its UV set; the cube deleted. ``named=False`` leaves
    the mesh unnamed, as Maya's fbx export does."""
    t = cmds.polyCube(name="src", constructionHistory=False)[0]
    data, uvs = Mesh(t).serialize(world_space=False)
    cmds.delete(t)
    scene = Scene("s")
    grp   = TransformData("grp", translate=(0, 10, 0))
    scene.add(grp)
    hero = TransformData("hero", parent_uuid=grp.uuid, translate=(5, 0, 0), rotate=(0, 90, 0))
    scene.add(hero)
    data.name        = "hero_geo" if named else None
    data.parent_uuid = hero.uuid if on_node else None
    for uv in uvs:
        uv.mesh_uuid = data.uuid
    scene.add(data, *uvs)
    return scene


def _item(scene):
    items = scene.renderables()
    assert len(items) == 1, items
    return items[0]


def _world(shape):
    return np.array(shape.get_points(world_space=True))[:, :3]


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_nothing_made(self, error, pattern, call):
        before = cmds.ls(long=True)
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(long=True), before)


# ---------------------------------------------------------------------------------------------
class TestWhereItGoes(_Case):
    def test_under_the_scene_node_maya_holds(self):
        scene = _scene()
        item  = _item(scene)
        nodes = Transform.create_hierarchy(scene.hierarchy)
        self.assertEqual([n.get_long_name() for n in nodes], ["|grp", "|grp|hero"])
        made = Mesh.create(item)
        self.assertEqual((repr(made), made.get_parent().get_long_name()), ('Mesh("heroShape")', "|grp|hero"))
        self.assertEqual(_uuid(made), item.mesh.uuid)
        self.assertEqual(list(made.get_uv_sets()), [uv.name for uv in item.uvs])
        np.testing.assert_allclose(np.array(made.get_points(world_space=False))[:, :3], np.asarray(item.mesh.points), atol=1e-6)
        # where hero puts it: hero's world matrix is the scene's
        np.testing.assert_allclose(
            _world(made), np.asarray(item.mesh.points) @ np.asarray(item.node.world_matrix)[:3, :3] + np.asarray(item.node.world_matrix)[3, :3],
            atol=1e-5,
        )
        # the data comes back: the shape's uuid and the node's as parent_uuid
        again = made.serialize(uvs=False)
        self.assertEqual((again.uuid, again.parent_uuid), (item.mesh.uuid, item.node.uuid))
        self.assertEqual(repr(Node(item.mesh.uuid)), 'Mesh("heroShape")')

    def test_a_new_transform_where_the_scene_s_node_is_when_maya_lacks_it(self):
        scene = _scene()
        item  = _item(scene)
        made  = Mesh.create(item)
        self.assertEqual((repr(made), str(made.get_parent())), ('Mesh("hero_geoShape")', "hero_geo"))
        self.assertEqual(_uuid(made), item.mesh.uuid)
        np.testing.assert_allclose(
            np.reshape(cmds.xform("hero_geo", query=True, matrix=True, worldSpace=True), (4, 4)), np.asarray(item.node.world_matrix), atol=1e-9,
        )
        np.testing.assert_allclose(np.array(made.get_points(world_space=False))[:, :3], np.asarray(item.mesh.points), atol=1e-6)

    def test_no_node_sits_at_identity(self):
        scene = _scene(on_node=False)
        item  = _item(scene)
        self.assertIsNone(item.node)
        made = Mesh.create(item)
        np.testing.assert_array_equal(np.reshape(cmds.xform("hero_geo", query=True, matrix=True, worldSpace=True), (4, 4)), np.eye(4))

    def test_transform_wins_over_the_node(self):
        scene = _scene()
        item  = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        ctl  = Transform.create(name="ctl", t=(1, 2, 3))
        made = Mesh.create(item, transform=ctl)
        self.assertEqual((repr(made), made.get_parent().get_long_name()), ('Mesh("ctlShape")', "|ctl"))
        self.assertEqual(_uuid(made), item.mesh.uuid)
        self.assertEqual(cmds.listRelatives("hero", shapes=True), None)

    def test_world_space_true_lands_the_same_under_the_node(self):
        scene = _scene()
        item  = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        own = Mesh.create(item)
        put = Mesh.create(item, world_space=True, uuids=False)
        np.testing.assert_allclose(_world(put), _world(own), atol=1e-5)

    def test_an_unnamed_mesh_takes_its_node_s_name(self):
        scene = _scene(named=False)
        item  = _item(scene)
        self.assertEqual((item.mesh.name, item.name), (None, "hero"))
        made = Mesh.create(item)
        self.assertEqual((repr(made), str(made.get_parent())), ('Mesh("heroShape")', "hero"))
        Transform.create_hierarchy(scene.hierarchy)   # |grp|hero beside |hero: the item's uuid points at it
        under = Mesh.create(item, uuids=False)
        self.assertEqual(under.get_long_name(), "|grp|hero|heroShape")

    def test_name_overrides(self):
        scene = _scene()
        item  = _item(scene)
        self.assertEqual(str(Mesh.create(item, name="geo").get_parent()), "geo")
        Transform.create_hierarchy(scene.hierarchy)
        self.assertEqual(Mesh.create(item, name="petal", uuids=False).get_long_name(), "|grp|hero|petal")


# ---------------------------------------------------------------------------------------------
class TestTheKeywords(_Case):
    def test_uvs_and_hierarchy_are_the_item_s(self):
        scene = _scene()
        item  = _item(scene)
        self.assert_nothing_made(
            TypeError, r"^Mesh\.create\(\): a Renderable carries its mesh's UV sets \(item\.uvs\): uvs= goes with a MeshData$",
            lambda: Mesh.create(item, uvs=list(item.uvs)),
        )
        self.assert_nothing_made(
            TypeError, r"^Mesh\.create\(\): a Renderable carries its node \(item\.node\): hierarchy= goes with a MeshData$",
            lambda: Mesh.create(item, hierarchy=scene.hierarchy),
        )

    def test_the_uuid_rule_holds(self):
        scene = _scene()
        item  = _item(scene)
        made  = Mesh.create(item)
        self.assert_nothing_made(
            ValueError, r"^Mesh\.create\(\): the MeshData 'hero_geo' has the uuid [0-9A-F-]{36}, which '\|hero_geo\|hero_geoShape' holds",
            lambda: Mesh.create(item),
        )
        twin = Mesh.create(item, uuids=False)
        self.assertNotEqual(_uuid(twin), item.mesh.uuid)
        self.assertEqual(_uuid(made), item.mesh.uuid)
        # a second copy the scene's way: fresh uuids, the links rewired, the node not in Maya any more
        scene.renew_uuids()
        again = Mesh.create(item)
        self.assertEqual(_uuid(again), item.mesh.uuid)
        self.assertEqual(str(again.get_parent()), "hero_geo2")

    def test_the_restore_flags_and_attributes_apply(self):
        scene = _scene()
        item  = _item(scene)
        made  = Mesh.create(item, normals=False, colors=False, visibility=False)
        self.assertFalse(cmds.getAttr(f"{made}.visibility"))
        self.assertEqual(made.get_color_sets(), [])

    def test_the_rest_of_the_item_is_left_alone(self):
        scene = _scene()
        item  = _item(scene)
        scene.add(MaterialData(name="skin", mesh_uuid=item.mesh.uuid, color=(1, 0, 0)))
        scene.add(SkinData(weights=np.ones((8, 1)), influences=["hero"], name="bodySkin", mesh_uuid=item.mesh.uuid))
        item = _item(scene)
        self.assertEqual((len(item.materials), item.skin is not None), (1, True))
        Transform.create_hierarchy(scene.hierarchy)
        made = Mesh.create(item)
        self.assertEqual([str(e) for e in made.get_shading_engines()], ["initialShadingGroup"])
        self.assertEqual((cmds.ls(type="skinCluster"), cmds.ls(type="blendShape")), ([], []))

    def test_one_undo_step_and_the_selection(self):
        scene = _scene()
        item  = _item(scene)
        Transform.create_hierarchy(scene.hierarchy)
        cmds.select("grp")
        self.undo_queue_on()
        made = Mesh.create(item)
        self.assertEqual(cmds.ls(selection=True), ["grp"])
        self.assertEqual(self.undo_name(), "rig.Mesh.create")
        self.undo_steps(1)
        self.assertEqual((cmds.listRelatives("hero", shapes=True), cmds.ls(item.mesh.uuid)), (None, []))
        self.redo_steps(1)
        self.assertEqual(_uuid(made), item.mesh.uuid)

    def test_load_takes_no_item(self):
        scene = _scene()
        item  = _item(scene)
        self.assert_nothing_made(
            TypeError, r"^Mesh\.load: the mesh: got a Renderable; the mesh is a MeshData$", lambda: Mesh.load(item),
        )
