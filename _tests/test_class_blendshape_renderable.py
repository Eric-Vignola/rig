"""BlendShape in-betweens and ``BlendShape.create(item)`` (the Renderable round, 3 of 6; the plan
of 2026-11-09 and the user's decisions of 2026-11-10).

* A target's shapes are its inputTargetItem items, each at 5000 + 1000 x the weight it is reached
  at: a MorphData's ``at`` (1 its full target, 6000; an in-between below or above). Maya holds a
  weight to three decimals; a fourth is a ValueError before any edit.
* ``serialize`` gives a MorphList of the mesh (``mesh_uuid``), target by target, each target's
  shapes by ``at``, with the target's weight, ``deformer`` the node's name, ``mesh_uuid`` the
  mesh's. A target a call adds takes its morphs' weight; a target it sets keeps its own.
* ``BlendShape.create(item)``: the mesh is the scene node holding ``item.mesh.uuid``; a mesh
  given wins. Morphs of two deformers are a ValueError unless ``deformer=`` picks one; a single
  deformer's name names the node unless ``name=`` does.
"""

import importlib.util
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from cgmath.geometry import MorphData, MorphList
from cgmath.render import Scene
from maya import cmds
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.nodetypes import BlendShape, Mesh


def _uuid(node):
    return cmds.ls(str(node), long=True, uuid=True)[0]


def _morph(name, offset, index=0, **fields):
    return MorphData(name, offsets=np.array([offset], dtype=float), indices=np.array([index]), **fields)


def _cube(name="base"):
    return cmds.polyCube(name=name, constructionHistory=False)[0]


def _point(mesh, index=0):
    return np.array(cmds.xform(f"{mesh}.vtx[{index}]", query=True, translation=True, objectSpace=True))


def _shapes(morphs):
    """`morphs` target by target, each target's shapes by ``at``, as `serialize` orders them."""
    names = list(dict.fromkeys(m.name for m in morphs))
    return MorphList(sorted(morphs, key=lambda m: (names.index(m.name), m.at)))


def _items(bs, target=0):
    return cmds.getAttr(f"{bs}.inputTarget[0].inputTargetGroup[{target}].inputTargetItem", multiIndices=True)


def _scene(*morphs, deformer="face_bs"):
    """A Scene of a cube's data and `morphs` on it, each given the mesh's uuid and `deformer`
    (None leaves each its own); the cube deleted."""
    t    = _cube("src")
    data = Mesh(t).serialize(world_space=False, uvs=False)
    cmds.delete(t)
    data.name        = "hero_geo"
    data.parent_uuid = None
    for m in morphs:
        m.mesh_uuid = data.uuid
        if deformer is not None:
            m.deformer = deformer
    scene = Scene("s")
    scene.add(data, *morphs)
    return scene


def _smile():
    """A full target (weight 0.25) and its in-between at 0.5, and a second target."""
    return (
        _morph("smile", (0, 2, 0), weight=0.25),
        _morph("smile", (1, 0, 0), at=0.5, weight=0.25),
        _morph("brow", (0, 0, 1), index=1),
    )


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_nothing_made(self, error, pattern, call):
        before = cmds.ls(long=True)
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(long=True), before)


# ---------------------------------------------------------------------------------------------
class TestInBetweens(_Case):
    def test_a_morph_at_below_1_is_an_in_between(self):
        base = _cube()
        bs   = BlendShape.create(MorphList(list(_smile())), base)
        self.assertEqual(_items(bs), [5500, 6000])
        # Maya evaluates it as its own -inBetween: the in-between at 0.5, the target at 1
        cmds.setAttr(f"{bs}.smile", 0.0)
        start = _point(base)
        cmds.setAttr(f"{bs}.smile", 0.5)
        np.testing.assert_allclose(_point(base) - start, (1, 0, 0), atol=1e-6)
        cmds.setAttr(f"{bs}.smile", 1.0)
        np.testing.assert_allclose(_point(base) - start, (0, 2, 0), atol=1e-6)

    def test_above_1_too(self):
        base = _cube()
        bs   = BlendShape.create(MorphList([_morph("t", (0, 1, 0)), _morph("t", (0, 0, 3), at=2.0)]), base)
        self.assertEqual(_items(bs), [6000, 7000])
        self.assertEqual([m.at for m in bs.serialize()], [1.0, 2.0])

    def test_serialize_carries_each_shape_with_its_fields(self):
        base  = _cube()
        bs    = BlendShape.create(MorphList(list(_smile())), base, name="face_bs")
        got   = bs.serialize()
        shape = cmds.listRelatives(base, shapes=True, fullPath=True)[0]
        self.assertEqual(got.mesh_uuid, _uuid(shape))
        self.assertEqual(
            [(m.name, m.at, m.weight, m.deformer, m.mesh_uuid) for m in got],
            [
                ("smile", 0.5, 0.25, "face_bs", _uuid(shape)),
                ("smile", 1.0, 0.25, "face_bs", _uuid(shape)),
                ("brow", 1.0, 0.0, "face_bs", _uuid(shape)),
            ],
        )
        self.assertEqual([m.offsets.tolist() for m in got], [[[1, 0, 0]], [[0, 2, 0]], [[0, 0, 1]]])
        # get_target_data reads one shape by at
        half = bs.get_target_data("smile", at=0.5)
        self.assertEqual((half.at, half.offsets.tolist(), half.weight), (0.5, [[1, 0, 0]], 0.25))

    def test_maya_s_own_in_between_reads_back(self):
        base = _cube()
        full = _cube("full")
        half = _cube("half")
        cmds.move(0, 2, 0, f"{full}.vtx[0]", relative=True)
        cmds.move(1, 0, 0, f"{half}.vtx[0]", relative=True)
        bs = cmds.blendShape(full, base, name="bs")[0]
        cmds.blendShape(bs, edit=True, inBetween=True, target=(base, 0, half, 0.5))
        cmds.delete(full, half)
        got = BlendShape(bs).serialize()
        self.assertEqual([(m.name, m.at) for m in got], [("full", 0.5), ("full", 1.0)])
        self.assertEqual([m.offsets.tolist() for m in got], [[[1, 0, 0]], [[0, 2, 0]]])

    def test_a_round_trip_onto_another_mesh(self):
        bs   = BlendShape.create(MorphList(list(_smile())), _cube())
        got  = bs.serialize()
        copy = BlendShape.create(got, _cube("other"))
        self.assertEqual(copy.serialize(), got)
        self.assertEqual(cmds.getAttr(f"{copy}.smile"), 0.25)

    def test_a_save_and_load_round_trip(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        bs   = BlendShape.create(MorphList(list(_smile())), _cube())
        path = bs.save(os.path.join(folder, "face.npz"))
        made = BlendShape.load(path, _cube("other"))
        self.assertEqual(made.serialize(), bs.serialize())
        self.assertEqual(_items(made), [5500, 6000])

    def test_a_fourth_decimal_is_refused_before_any_edit(self):
        base = _cube()
        self.assert_nothing_made(
            ValueError, r"^BlendShape\.create: the morph 'q' is reached at weight 0\.3333; Maya holds a target's weights to three decimal places",
            lambda: BlendShape.create(_morph("q", (0, 1, 0), at=0.3333), base),
        )
        bs = BlendShape.create(_morph("q", (0, 1, 0)), base)
        with self.assertRaisesRegex(ValueError, r"^BlendShape\.set_target_data: the morph 'r' is reached at weight 0\.0005"):
            bs.set_target_data(MorphList([_morph("r", (0, 1, 0)), _morph("r", (1, 1, 0), at=0.0005)]))
        self.assertEqual(bs.get_targets(), ["q"])

    def test_a_new_target_takes_its_weight_a_set_one_keeps_its_own(self):
        bs = BlendShape.create(_morph("a", (0, 1, 0), weight=0.5), _cube())
        self.assertEqual(cmds.getAttr(f"{bs}.a"), 0.5)
        cmds.setAttr(f"{bs}.a", 0.75)
        bs.set_target_data(MorphList([_morph("a", (0, 3, 0), weight=0.1), _morph("b", (0, 3, 0), at=0.4, weight=0.3), _morph("b", (0, 4, 0), weight=0.2)]))
        self.assertEqual(cmds.getAttr(f"{bs}.a"), 0.75)
        # its full target's weight, wherever the full target is in the list
        self.assertAlmostEqual(cmds.getAttr(f"{bs}.b"), 0.2, places=6)
        self.assertEqual(_items(bs, 1), [5400, 6000])

    def test_an_in_between_onto_a_target_it_has(self):
        bs = BlendShape.create(_morph("a", (0, 1, 0)), _cube())
        bs.set_target_data("a", _morph("ignored", (2, 0, 0), at=0.25))
        self.assertEqual(_items(bs), [5250, 6000])
        self.assertEqual(bs.get_target_data("a", at=0.25).offsets.tolist(), [[2, 0, 0]])

    def test_a_shape_at_0_or_below_is_left_out_with_a_warning(self):
        bs   = BlendShape.create(_morph("a", (0, 1, 0)), _cube())
        item = f"{bs}.inputTarget[0].inputTargetGroup[0].inputTargetItem[4500]"
        cmds.setAttr(f"{item}.inputPointsTarget", 1, (1, 0, 0, 1), type="pointArray")
        cmds.setAttr(f"{item}.inputComponentsTarget", 1, "vtx[0]", type="componentList")
        with self.assertLogs("rig.nodetypes.blendshape", logging.WARNING) as logs:
            got = bs.serialize()
        self.assertEqual([m.at for m in got], [1.0])
        self.assertIn("holds shapes at weight -0.5, at or below 0", logs.output[0])

    def test_one_undo_step(self):
        bs = BlendShape.create(_morph("a", (0, 1, 0)), _cube())
        self.undo_queue_on()
        bs.set_target_data(MorphList([_morph("a", (1, 0, 0), at=0.5), _morph("b", (0, 1, 0), weight=0.5)]))
        self.assertEqual(self.undo_name(), "rig.BlendShape.set_target_data")
        self.undo_steps(1)
        # the undo of a new target leaves its weight element, named by none (serialize leaves it out)
        self.assertEqual((_items(bs), [m.name for m in bs.serialize()]), ([6000], ["a"]))
        self.redo_steps(1)
        self.assertEqual((_items(bs), cmds.getAttr(f"{bs}.b")), ([5500, 6000], 0.5))


# ---------------------------------------------------------------------------------------------
class TestCreateFromARenderable(_Case):
    def test_the_mesh_is_found_by_its_uuid(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        mesh  = Mesh.create(item)
        bs    = BlendShape.create(item)
        self.assertEqual(str(bs), "face_bs")
        self.assertEqual([str(g) for g in bs.get_geometries()], [str(mesh)])
        self.assertEqual(bs.get_targets(), ["smile", "brow"])
        self.assertEqual(cmds.getAttr(f"{bs}.smile"), 0.25)
        got = bs.serialize()
        self.assertEqual(got.mesh_uuid, item.mesh.uuid)
        self.assertEqual(got, _shapes(item.morphs))

    def test_a_mesh_given_wins(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        Mesh.create(item)
        other = _cube("other")
        bs    = BlendShape.create(other, item)
        self.assertEqual([str(g.get_parent()) for g in bs.get_geometries()], ["other"])
        self.assertEqual(bs.serialize(), _shapes(item.morphs))

    def test_no_targets_is_refused(self):
        scene = _scene()
        item  = scene.renderables()[0]
        Mesh.create(item)
        self.assert_nothing_made(
            ValueError, r"^BlendShape\.create: the Renderable 'hero_geo' holds no blend shape targets", lambda: BlendShape.create(item)
        )

    def test_a_mesh_no_node_holds_is_refused(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        self.assert_nothing_made(
            ValueError,
            r"^BlendShape\.create: no scene node holds the uuid [0-9A-F-]{36} of the Renderable 'hero_geo''s mesh: "
            r"Mesh\.create\(item\) builds it, or give the mesh",
            lambda: BlendShape.create(item),
        )

    def test_two_renderables_are_refused(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        Mesh.create(item)
        self.assert_nothing_made(
            TypeError, r"^BlendShape\.create: a create takes one Renderable, its mesh's targets; got 2$",
            lambda: BlendShape.create(item, item),
        )

    def test_name_wins_over_the_deformer_s(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        Mesh.create(item)
        self.assertEqual(str(BlendShape.create(item, name="mine")), "mine")

    def test_one_undo_step(self):
        scene = _scene(*_smile())
        item  = scene.renderables()[0]
        Mesh.create(item)
        self.undo_queue_on()
        BlendShape.create(item)
        self.assertEqual(self.undo_name(), "rig.BlendShape.create")
        self.undo_steps(1)
        self.assertEqual(cmds.ls(type="blendShape"), [])
        self.redo_steps(1)
        self.assertEqual(cmds.ls(type="blendShape"), ["face_bs"])


# ---------------------------------------------------------------------------------------------
class TestOneDeformer(_Case):
    def _two(self):
        a = _morph("a", (0, 1, 0), deformer="first")
        b = _morph("b", (1, 0, 0), deformer="second")
        return MorphList([a, b])

    def test_two_deformers_are_refused(self):
        base = _cube()
        self.assert_nothing_made(
            ValueError,
            r"^BlendShape\.create: the targets name 2 deformers \('first', 'second'\); one blendShape takes one "
            r"deformer's targets: deformer= picks one, a create each$",
            lambda: BlendShape.create(self._two(), base),
        )

    def test_deformer_picks_one(self):
        base   = _cube()
        first  = BlendShape.create(self._two(), base, deformer="first")
        second = BlendShape.create(self._two(), base, deformer="second")
        self.assertEqual((str(first), first.get_targets()), ("first", ["a"]))
        self.assertEqual((str(second), second.get_targets()), ("second", ["b"]))

    def test_a_deformer_no_morph_names_is_refused(self):
        base = _cube()
        self.assert_nothing_made(
            ValueError, r"^BlendShape\.create: deformer='third' names no target's deformer \(they name 'first', 'second'\)$",
            lambda: BlendShape.create(self._two(), base, deformer="third"),
        )

    def test_a_renderable_of_two_deformers_too(self):
        a, b, c = _smile()
        c.deformer = "other"
        a.deformer = b.deformer = "face_bs"
        scene      = _scene(a, b, c, deformer=None)
        item       = scene.renderables()[0]
        Mesh.create(item)
        self.assert_nothing_made(ValueError, r"name 2 deformers \('face_bs', 'other'\)", lambda: BlendShape.create(item))
        self.assertEqual(BlendShape.create(item, deformer="other").get_targets(), ["brow"])

    def test_morphs_of_no_deformer_keep_maya_s_name(self):
        bs = BlendShape.create(_morph("a", (0, 1, 0)), _cube())
        self.assertEqual(str(bs), "blendShape1")


# ---------------------------------------------------------------------------------------------
@unittest.skipUnless(importlib.util.find_spec("fbx"), "needs the FBX SDK's Python module (fbx)")
class TestThroughAnFbx(_Case):
    def test_a_scene_s_in_betweens_come_back(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        scene = _scene(*_smile())
        path  = scene.save_fbx(os.path.join(folder, "face.fbx"))
        again = Scene.load_fbx(path)
        item  = again.renderables()[0]
        self.assertEqual([(m.name, m.at) for m in item.morphs], [("smile", 0.5), ("smile", 1.0), ("brow", 1.0)])
        Mesh.create(item)
        bs = BlendShape.create(item)
        self.assertEqual(str(bs), "face_bs")
        self.assertEqual(bs.serialize(), item.morphs)
