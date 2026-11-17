"""``Transform.key_hierarchy`` and ``rig.create_scene`` / ``rig.load_scene`` (the Renderable
round, 5 of 6; the plan of 2026-11-09 and the user's decisions of 2026-11-10).

* ``node.key_hierarchy(clip)`` keys a ClipData onto a node and what is below it (the user's
  decisions of 2026-11-13): by order by default, node i of the data on node i, the counts free
  to differ; ``index=False`` by name and ``uuids=True`` by uuid, both at or below the node, a
  node of the data keying its own branch alone. A node of the data with no node, an attribute
  its node lacks and a locked channel are each a ValueError before any edit, or, with
  ``strict=False``, skipped with one warning. A plain HierarchyData poses, no key. One undo
  step.
* ``create_scene(scene)``: the nodes, each mesh under its node, its blendShape, its skin bound at
  rest, then the active take keyed; a List of what was made, one undo step, the selection kept.
  The parts are joined by what the call made, so the nodes keep Maya's own uuids and one scene
  builds any number of times (the user's decision of 2026-11-13); ``uuids=True`` gives every
  node and shape its data's uuid. Every refusal before anything is made: with ``uuids=True`` a
  uuid the Maya scene holds, two deformers on a mesh, a skin in a scene of takes alone.
  Cameras, lights and materials left out with a warning.
* ``load_scene(path)`` reads by extension and builds.
"""

import importlib.util
import logging
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import rig
from cgmath.geometry import MaterialData, MorphData, SkinData
from cgmath.hierarchy import ClipData, HierarchyData, TransformData
from cgmath.render import Scene
from maya import cmds
from maya.api import OpenMaya
from rig import create_scene, load_scene
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig._tests.test_class_hierarchy_clip import _clip as _film_clip
from rig._tests.test_class_hierarchy_clip import _curves, _moving_curves, _worlds
from rig import plugins
from rig.nodetypes import Joint, Mesh, Transform


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
_LOG    = "rig.nodetypes.transform"
_JOINTS = "jointOrient, segmentScaleCompensate, radius, drawStyle"   # what a joint's data sets and a transform lacks


def _fps():
    """The scene's own rate. A clip at it logs no fps warning into what these tests read, whatever
    the session's time unit: a Maya with preferences of its own is not at film."""
    return OpenMaya.MTime(1.0, OpenMaya.MTime.kSeconds).asUnits(OpenMaya.MTime.uiUnit())


def _clip():
    """`test_class_hierarchy_clip`'s clip, at the scene's own rate."""
    return _film_clip(fps=_fps())


def _bare(clip, parent=None, **kwargs):
    """The nodes of `clip` made, the keys `create_hierarchy` gave them taken off."""
    nodes = Transform.create_hierarchy(clip, parent, **kwargs)
    cmds.delete(cmds.ls(type="animCurve"))
    return nodes


def _curves_of(*names):
    """`_moving_curves` of the nodes `names` alone."""
    return [curve for curve in _moving_curves() if curve.split("_")[0] in names]


def _transforms():
    """`_clip`'s names and places, every node a transform at its defaults; the top one."""
    made = {"root": Transform.create(name="root")}
    for name, parent in (("spine", "root"), ("arm", "spine"), ("grp", "root"), ("probe", "grp")):
        made[name] = Transform.create(name=name, parent=made[parent])
    return made["root"]


def _values():
    return [
        cmds.getAttr(f"{node}.{attr}")
        for node in cmds.ls(type="transform", long=True)
        for attr in ("translate", "rotate", "scale")
    ]


def _exact(message):
    return f"^{re.escape(message)}$"


def _wired(parent, joint):
    return cmds.isConnected(f"{parent}.scale", f"{joint}.inverseScale")


class _KeyCase(_Case):
    def warnings(self, call):
        """The warnings `call` logs, its result in ``self.keyed``."""
        with self.assertLogs(_LOG, logging.WARNING) as logs:
            self.keyed = call()
        return [record.getMessage() for record in logs.records]

    def assert_worlds(self, nodes, clip):
        """Every node's world matrix on every frame is the clip's."""
        self.assertLessEqual(np.abs(_worlds(nodes, clip) - clip.frames.world_matrix).max(), 1e-9)

    def assert_refused(self, message, call, error=ValueError):
        """`error` with `message`, before any edit: no node, no value, no undo step."""
        self.undo_queue_on()
        before = _values()
        self.assert_nothing_made(error, _exact(message), call)
        self.assertEqual(_values(), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


class TestKeyHierarchyByOrder(_KeyCase):
    """The default: node i of the data keys node i of the node and what is below it."""

    def test_node_i_of_the_data_keys_node_i(self):
        clip  = _clip()
        nodes = _bare(clip)
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("root").key_hierarchy(clip)
        self.assertIsInstance(keyed, rig.List)
        self.assertEqual(
            [repr(node) for node in keyed],
            ['Joint("root")', 'Joint("spine")', 'Joint("arm")', 'Transform("grp")', 'Transform("probe")'],
        )
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_no_name_and_no_uuid_is_read(self):
        clip  = _clip()
        nodes = _bare(clip)   # Maya's own uuids
        for name in ("probe", "grp", "arm", "spine", "root"):
            cmds.rename(name, f"x_{name}")
        keyed = Transform("x_root").key_hierarchy(clip)
        self.assertEqual([str(node) for node in keyed], ["x_root", "x_spine", "x_arm", "x_grp", "x_probe"])
        self.assertEqual(_curves(), sorted(f"x_{curve}" for curve in _moving_curves()))
        self.assert_worlds(nodes, clip)

    def test_a_second_copy_takes_the_clip_too(self):
        clip = _clip()
        one  = _bare(clip)
        two  = _bare(clip)   # Maya names its top node root1
        Transform("root").key_hierarchy(clip)
        Transform("root1").key_hierarchy(clip)
        self.assertEqual(len(_curves()), 2 * len(_moving_curves()))
        self.assert_worlds(one, clip)
        self.assert_worlds(two, clip)

    def test_more_nodes_in_the_data_than_below_the_node(self):
        clip = _clip()
        _bare(clip)
        cmds.delete("probe")
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip))
        self.assertEqual(said, [
            "Joint.key_hierarchy: the data holds 5 node(s), root and what is below it 4: the first 4 were keyed, in order",
        ])
        self.assertEqual([str(node) for node in self.keyed], ["root", "spine", "arm", "grp"])
        self.assertEqual(_curves(), _curves_of("root", "spine", "arm", "grp"))

    def test_more_nodes_below_the_node_than_in_the_data(self):
        clip  = _clip()
        nodes = _bare(clip)
        Transform.create(name="extra", parent=Transform("probe"))
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip))
        self.assertEqual(said, [
            "Joint.key_hierarchy: the data holds 5 node(s), root and what is below it 6: the first 5 were keyed, in order",
        ])
        self.assertEqual([str(node) for node in self.keyed], ["root", "spine", "arm", "grp", "probe"])
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_a_node_more_shifts_every_later_one(self):
        clip = _clip()
        _bare(clip)
        Transform.create(name="extra", parent=Transform("arm"))   # root, spine, arm, extra, grp, probe
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip))
        self.assertEqual(len(said), 1)
        self.assertEqual([str(node) for node in self.keyed], ["root", "spine", "arm", "extra", "grp"])
        self.assertEqual(
            _curves(),
            sorted(_curves_of("root", "spine", "arm") + ["extra_rotateY", "extra_scaleX", "grp_rotateZ", "grp_translateY"]),
        )

    def test_on_a_branch_the_first_node_of_the_data_lands_on_it(self):
        clip = _clip()
        _bare(clip)
        said = self.warnings(lambda: Transform("spine").key_hierarchy(clip))
        self.assertEqual(said, [
            "Joint.key_hierarchy: the data holds 5 node(s), spine and what is below it 2: the first 2 were keyed, in order",
        ])
        self.assertEqual([str(node) for node in self.keyed], ["spine", "arm"])
        self.assertEqual(   # root's columns on spine, spine's on arm
            _curves(),
            ["arm_jointOrientY", "arm_rotateX", "arm_rotateZ", "spine_rotateY", "spine_translateX", "spine_translateZ"],
        )

    def test_a_branch_takes_a_clip_of_its_own(self):
        _bare(_clip())
        branch                        = ClipData(Transform("spine").serialize_hierarchy(), frames=3, start_frame=1, fps=_fps())
        branch.frames["arm"].rotate_x = np.array([0.0, 20.0, 40.0])
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("spine").key_hierarchy(branch)
        self.assertEqual([str(node) for node in keyed], ["spine", "arm"])
        self.assertEqual(_curves(), ["arm_rotateX"])
        self.assertAlmostEqual(cmds.getAttr("arm.rotateX", time=2), 20.0, places=9)


class TestKeyHierarchyByUuid(_KeyCase):
    """``uuids=True``: a node of the data keys the node holding its uuid, at or below the node."""

    def test_each_node_keys_the_node_holding_its_uuid(self):
        clip  = _clip()
        nodes = _bare(clip, uuids=True)
        cmds.reorder("grp", front=True)   # another order, other names
        for name in ("probe", "grp", "arm", "spine", "root"):
            cmds.rename(name, f"x_{name}")
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("x_root").key_hierarchy(clip, uuids=True)
        self.assertIsInstance(keyed, rig.List)
        self.assertEqual(   # the data's order, typed
            [repr(node) for node in keyed],
            ['Joint("x_root")', 'Joint("x_spine")', 'Joint("x_arm")', 'Transform("x_grp")', 'Transform("x_probe")'],
        )
        self.assertEqual(_curves(), sorted(f"x_{curve}" for curve in _moving_curves()))
        self.assert_worlds(nodes, clip)

    def test_index_is_not_read(self):
        clip = _clip()
        _bare(clip, uuids=True)
        cmds.rename("probe", "other")   # by name, a miss
        keyed = Transform("root").key_hierarchy(clip, uuids=True, index=False)
        self.assertEqual([str(node) for node in keyed], ["root", "spine", "arm", "grp", "other"])

    def test_a_branch_keys_the_branch_alone(self):
        clip = _clip()
        _bare(clip, uuids=True)
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("spine").key_hierarchy(clip, uuids=True)
        self.assertEqual([repr(node) for node in keyed], ['Joint("spine")', 'Joint("arm")'])
        self.assertEqual(_curves(), _curves_of("spine", "arm"))

    def test_a_node_above_the_data_keys_every_node(self):
        clip  = _clip()
        nodes = _bare(clip, uuids=True)
        cmds.parent("root", str(Transform.create(name="above")))
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("above").key_hierarchy(clip, uuids=True)
        self.assertEqual([str(node) for node in keyed], ["root", "spine", "arm", "grp", "probe"])
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_a_miss_is_refused_before_any_edit(self):
        clip = _clip()
        _bare(clip)   # Maya's own uuids: no node holds the data's
        self.assert_refused(
            "Joint.key_hierarchy: 5 node(s) of the data have no one node holding their uuid at or below root: "
            "'root' (none), 'spine' (none), 'arm' (none), 'grp' (none), 'probe' (none); "
            "strict=False skips them, nothing was keyed",
            lambda: Transform("root").key_hierarchy(clip, uuids=True),
        )
        self.assertEqual(_curves(), [])

    def test_a_node_outside_is_a_miss(self):
        clip = _clip()
        _bare(clip, uuids=True)
        cmds.parent("grp", world=True)   # it holds its uuid, and is not at or below root
        self.assert_refused(
            "Joint.key_hierarchy: 2 node(s) of the data have no one node holding their uuid at or below root: "
            "'grp' (none), 'probe' (none); strict=False skips them, nothing was keyed",
            lambda: Transform("root").key_hierarchy(clip, uuids=True),
        )

    def test_strict_false_keys_the_rest(self):
        clip = _clip()
        _bare(clip, uuids=True)
        cmds.delete("probe")
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip, uuids=True, strict=False))
        self.assertEqual(said, [
            "Joint.key_hierarchy: strict=False skipped 1 node(s) of the data with no one node holding their uuid "
            "at or below root: 'probe' (none)",
        ])
        self.assertEqual([str(node) for node in self.keyed], ["root", "spine", "arm", "grp"])
        self.assertEqual(_curves(), _curves_of("root", "spine", "arm", "grp"))


class TestKeyHierarchyByName(_KeyCase):
    """``index=False``: a node of the data keys the one node of its name, at or below the node."""

    def test_each_node_keys_the_one_node_of_its_name(self):
        clip  = _clip()
        nodes = _bare(clip)
        cmds.reorder("grp", front=True)   # another order
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("root").key_hierarchy(clip, index=False)
        self.assertIsInstance(keyed, rig.List)
        self.assertEqual(
            [repr(node) for node in keyed],
            ['Joint("root")', 'Joint("spine")', 'Joint("arm")', 'Transform("grp")', 'Transform("probe")'],
        )
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_two_copies_each_take_the_clip(self):
        clip = _clip()
        Transform.create(name="a")
        below = Transform.create(name="b")
        one   = _bare(clip, "a")
        two   = _bare(clip, "b")
        with self.assertNoLogs(_LOG, level="WARNING"):
            first  = Transform("|a|root").key_hierarchy(clip, index=False)
            second = below.key_hierarchy(clip, index=False)   # from the group above
        self.assertEqual([node.get_long_name() for node in first], [node.get_long_name() for node in one])
        self.assertEqual([node.get_long_name() for node in second], [node.get_long_name() for node in two])
        self.assertEqual(len(_curves()), 2 * len(_moving_curves()))
        self.assert_worlds(one, clip)
        self.assert_worlds(two, clip)

    def test_a_branch_keys_the_branch_alone(self):
        clip = _clip()
        _bare(clip)
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = Transform("grp").key_hierarchy(clip, index=False)
        self.assertEqual([repr(node) for node in keyed], ['Transform("grp")', 'Transform("probe")'])
        self.assertEqual(_curves(), _curves_of("grp", "probe"))

    def test_a_name_no_node_or_several_have_is_refused(self):
        clip = _clip()
        _bare(clip)
        Transform.create(name="spine", parent=Transform("grp"))
        cmds.delete("probe")
        self.assert_refused(
            "Joint.key_hierarchy: 2 node(s) of the data have no one node of their name at or below root: "
            "'spine' (|root|grp|spine, |root|spine), 'probe' (none); strict=False skips them, nothing was keyed",
            lambda: Transform("root").key_hierarchy(clip, index=False),
        )
        self.assertEqual(_curves(), [])

    def test_a_name_two_nodes_of_the_data_share(self):
        top = Transform.create(name="base")
        for side in ("l", "r"):
            Transform.create(name="end", parent=Transform.create(name=side, parent=top))
        clip = ClipData(top.serialize_hierarchy(), frames=3, start_frame=1)
        self.assert_refused(
            "Transform.key_hierarchy: 2 node(s) of the data have no one node of their name at or below base: "
            "'end' (2 nodes of the data have the name), 'end' (2 nodes of the data have the name); "
            "strict=False skips them, nothing was keyed",
            lambda: top.key_hierarchy(clip, index=False),
        )
        # a branch holds the name once
        keyed = Transform("|base|l").key_hierarchy(clip, index=False)
        self.assertEqual([node.get_long_name() for node in keyed], ["|base|l", "|base|l|end"])

    def test_strict_false_keys_the_rest(self):
        clip = _clip()
        _bare(clip)
        cmds.delete("probe")
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip, index=False, strict=False))
        self.assertEqual(said, [
            "Joint.key_hierarchy: strict=False skipped 1 node(s) of the data with no one node of their name "
            "at or below root: 'probe' (none)",
        ])
        self.assertEqual([str(node) for node in self.keyed], ["root", "spine", "arm", "grp"])
        self.assertEqual(_curves(), _curves_of("root", "spine", "arm", "grp"))


class TestKeyHierarchyChannels(_KeyCase):
    """What a node cannot take: an attribute it lacks, a locked channel."""

    def test_a_joints_data_on_a_node_that_is_no_joint_is_refused(self):
        clip = _clip()
        top  = _transforms()
        self.assert_refused(
            "Transform.key_hierarchy: 3 node(s) lack attributes their data sets (a joint's data, a node that is "
            f"no joint): root ({_JOINTS}), spine ({_JOINTS}), arm ({_JOINTS}); "
            "strict=False skips them, nothing was keyed",
            lambda: top.key_hierarchy(clip),
        )
        self.assertEqual(_curves(), [])

    def test_a_transforms_data_on_a_joint_keys(self):
        top = _transforms()
        top.t.set(1, 2, 3)
        clip                             = ClipData(top.serialize_hierarchy(), frames=3, start_frame=1, fps=_fps())
        clip.frames["probe"].translate_y = np.array([0.0, 1.0, 2.0])
        cmds.delete("root")
        made = [Joint.create(name="root")]
        for name, parent in (("spine", 0), ("arm", 1), ("grp", 0), ("probe", 3)):
            made.append(Joint.create(name=name, parent=made[parent]))
        with self.assertNoLogs(_LOG, level="WARNING"):
            keyed = made[0].key_hierarchy(clip)
        self.assertEqual([repr(node) for node in keyed][::4], ['Joint("root")', 'Joint("probe")'])
        self.assertEqual(_curves(), ["probe_translateY"])
        self.assertEqual(cmds.getAttr("root.translate"), [(1.0, 2.0, 3.0)])

    def test_a_locked_channel_is_refused(self):
        clip = _clip()
        _bare(clip)
        cmds.setAttr("spine.rotateX", lock=True)
        self.assert_refused(
            "Joint.key_hierarchy: the data changes 1 locked channel(s): spine.rotateX (a skinCluster locks its "
            "mesh's transform); strict=False skips them, nothing was keyed",
            lambda: Transform("root").key_hierarchy(clip),
        )
        self.assertEqual(_curves(), [])

    def test_a_lock_the_data_leaves_alone_is_no_error(self):
        clip  = _clip()
        nodes = _bare(clip)
        cmds.setAttr("grp.translate", lock=True)    # a compound, an axis: values the nodes hold already
        cmds.setAttr("arm.translateY", lock=True)
        with self.assertNoLogs(_LOG, level="WARNING"):
            Transform("root").key_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_strict_false_skips_a_locked_axis_alone(self):
        clip = _clip()
        _bare(clip)
        cmds.setAttr("grp.translate", 5, 5, 5)
        cmds.setAttr("grp.translateY", lock=True)  # an axis the data sets
        cmds.setAttr("spine.rotateX", lock=True)   # an axis the clip keys
        said = self.warnings(lambda: Transform("root").key_hierarchy(clip, strict=False))
        self.assertEqual(said, [
            "Joint.key_hierarchy: strict=False skipped 2 locked channel(s): spine.rotateX, grp.translateY",
        ])
        self.assertEqual(len(self.keyed), 5)
        self.assertEqual(_curves(), [curve for curve in _moving_curves() if curve != "spine_rotateX"])
        np.testing.assert_allclose(cmds.getAttr("grp.translate")[0], (-2.0, 5.0, 0.0), atol=1e-9)   # x and z set

    def test_strict_false_names_both_kinds_in_one_warning(self):
        clip = _clip()
        top  = _transforms()
        cmds.setAttr("grp.rotateY", lock=True)
        said = self.warnings(lambda: top.key_hierarchy(clip, strict=False))
        self.assertEqual(said, [
            "Transform.key_hierarchy: strict=False skipped 3 node(s) lacking attributes a joint's data sets: "
            f"root ({_JOINTS}), spine ({_JOINTS}), arm ({_JOINTS}); 1 locked channel(s): grp.rotateY",
        ])
        self.assertEqual(len(self.keyed), 5)
        self.assertEqual(   # every channel the nodes have, but the locked one
            _curves(),
            [curve for curve in _moving_curves() if curve not in ("spine_jointOrientY", "grp_rotateY")],
        )
        np.testing.assert_allclose(cmds.getAttr("spine.translate")[0], (0.0, 5.0, 0.0), atol=1e-9)

    def test_what_strict_false_does_not_skip(self):
        clip = _clip()
        _bare(clip, uuids=True)
        with mock.patch.object(plugins, "_fast_path_ready", return_value=False):
            self.assert_refused(
                "rig needs its undo plug-in 'undoable_api_command' to key a clip, and it could not be loaded "
                f"({plugins._plugin_whereabouts()}); nothing was made",
                lambda: Transform("root").key_hierarchy(clip, strict=False),
                error=RuntimeError,
            )
        clip.frames["probe"].translate_y = [0, 1, np.nan, 3, 4, 5]
        for strict in (True, False):
            with self.subTest(strict=strict):
                self.assert_nothing_made(
                    ValueError, "AnimCurve values must be finite",
                    lambda: Transform("root").key_hierarchy(clip, strict=strict),
                )
        self.assertEqual(_curves(), [])
        # a node that takes no key: its NaN is not read
        Transform("spine").key_hierarchy(clip, uuids=True)
        self.assertEqual(_curves(), _curves_of("spine", "arm"))

    def test_a_compensating_joint_takes_the_scale_of_its_keyed_parent(self):
        clip = _clip()
        _bare(clip, uuids=True)
        self.assertEqual((_wired("root", "spine"), _wired("spine", "arm")), (True, True))   # create_hierarchy's
        cmds.disconnectAttr("root.scale", "spine.inverseScale")
        cmds.disconnectAttr("spine.scale", "arm.inverseScale")
        Transform("spine").key_hierarchy(clip, uuids=True)   # root takes no key: spine is left as it is
        self.assertEqual((_wired("root", "spine"), _wired("spine", "arm")), (False, True))
        Transform("root").key_hierarchy(clip)
        self.assertEqual((_wired("root", "spine"), _wired("spine", "arm")), (True, True))


class TestKeyHierarchy(_KeyCase):
    def test_a_hierarchy_poses_no_key(self):
        clip = _clip()
        _bare(clip)
        cmds.setAttr("root.translate", 9, 9, 9)
        pose  = clip.at(clip.start_frame + 2)
        keyed = Transform("root").key_hierarchy(pose)
        self.assertEqual(len(keyed), 5)
        self.assertEqual(_curves(), [])
        np.testing.assert_allclose(cmds.getAttr("root.translate")[0], pose["root"].translate, atol=1e-9)

    def test_keys_held_already_are_replaced(self):
        clip  = _clip()
        nodes = _bare(clip)
        Transform("root").key_hierarchy(clip)
        cmds.setKeyframe("arm.rotateX", time=clip.start_frame + 1, value=77.0)
        Transform("root").key_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip)

    def test_the_clip_is_left_as_it_was(self):
        clip   = _clip()
        before = ([node.name for node in clip], clip.frames.world_matrix.copy())
        _bare(clip)
        Transform("root").key_hierarchy(clip)
        self.assertEqual([node.name for node in clip], before[0])
        np.testing.assert_array_equal(clip.frames.world_matrix, before[1])

    def test_one_undo_step_and_the_selection(self):
        clip = _clip()
        _bare(clip)
        cmds.select("grp")
        self.undo_queue_on()
        Transform("root").key_hierarchy(clip)
        self.assertEqual(self.undo_name(), "rig.Joint.key_hierarchy")   # the node's class
        self.assertEqual(cmds.ls(selection=True), ["grp"])
        self.undo_steps(1)
        self.assertEqual(_curves(), [])
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.redo_steps(1)
        self.assertEqual(_curves(), _moving_curves())
        # keys a channel holds already are no edit, and no undo step: undone first
        self.undo_steps(1)
        Transform("grp").key_hierarchy(clip, index=False)
        self.assertEqual(self.undo_name(), "rig.Transform.key_hierarchy")

    def test_the_guards(self):
        top = Transform.create(name="above")
        with self.assertRaisesRegex(TypeError, r"^Transform\.key_hierarchy\(\) takes the data, not a file"):
            top.key_hierarchy("walk.npz")
        with self.assertRaisesRegex(TypeError, r"^Transform\.key_hierarchy\(\) takes a ClipData or a HierarchyData, not a int$"):
            top.key_hierarchy(3)

    def test_the_flags_take_a_bool(self):
        clip = _clip()
        _bare(clip)
        root = Transform("root")
        for flag, value in (("uuids", 1), ("index", None), ("strict", "no")):
            with self.subTest(flag=flag):
                self.assert_nothing_made(
                    TypeError,
                    rf"^Joint\.key_hierarchy: {flag}= is True \(.+\) or False \(.+\), not {re.escape(repr(value))}$",
                    lambda: root.key_hierarchy(clip, **{flag: value}),
                )
        with self.assertRaises(TypeError):   # by keyword only
            root.key_hierarchy(clip, True)
        self.assertEqual(_curves(), [])

    def test_called_on_the_class_it_is_pythons_own_typeerror(self):
        clip = _clip()
        _bare(clip)
        with self.assertRaisesRegex(TypeError, r"key_hierarchy\(\) missing 1 required positional argument: 'clip'"):
            Transform.key_hierarchy(clip)
        self.assertEqual(_curves(), [])


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
        # the nodes and the mesh keep Maya's own uuids; the mesh sits under its node all the same
        held = [data.uuid.upper() for data in scene.hierarchy] + [item.mesh.uuid.upper()]
        self.assertEqual(cmds.ls(held), [])
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

    def test_uuids_true_gives_the_data_s_uuids(self):
        scene = _scene()
        item  = scene.renderables()[0]
        made  = create_scene(scene, uuids=True)
        for data in scene.hierarchy:
            self.assertEqual(_uuid(cmds.ls(data.uuid.upper())[0]), data.uuid.upper())
        self.assertEqual(_uuid(made[3]), item.mesh.uuid.upper())
        self.assertEqual(repr(rig.Node(item.mesh.uuid)), 'Mesh("geoShape")')
        self.assertEqual(made[3].get_parent().get_long_name(), "|geo")
        self.assertEqual(cmds.ls(type="animCurve"), ["tip_translateY"])

    def test_a_second_copy_with_its_uuids_takes_fresh_uuids(self):
        scene = _scene()
        create_scene(scene, uuids=True)
        self.assert_nothing_made(
            ValueError,
            r"^create_scene\(\): the node 'root' has the uuid .*, which '\|root' holds, and two nodes cannot hold one; for a "
            r"second copy give the data fresh uuids first \(scene\.renew_uuids\(\), or a Scene's renew_uuids\(\)\); without "
            r"uuids=True the nodes keep Maya's own$",
            lambda: create_scene(scene, uuids=True),
        )
        scene.renew_uuids()
        again = create_scene(scene, uuids=True)
        self.assertEqual(again[0].get_long_name(), "|root1")
        self.assertEqual(cmds.skinCluster(str(again[5]), query=True, influence=True), ["root1", "root1|tip"])

    def test_any_number_of_copies_by_default(self):
        # no flag: each copy keeps Maya's own uuids and joins its parts by what it made
        scene = _scene()
        first = create_scene(scene)
        twin  = create_scene(scene)
        third = create_scene(scene, uuids=False)
        self.assertEqual([c[0].get_long_name() for c in (first, twin, third)], ["|root", "|root1", "|root2"])
        self.assertEqual(twin[3].get_parent().get_long_name(), "|geo1")
        self.assertEqual(cmds.skinCluster(str(twin[5]), query=True, influence=True), ["root1", "root1|tip"])
        self.assertEqual(cmds.skinCluster(str(third[5]), query=True, influence=True), ["root2", "root2|tip"])
        self.assertEqual(sorted(cmds.ls(type="animCurve")), ["tip_translateY", "tip_translateY1", "tip_translateY2"])
        uuids = [_uuid(n) for copy in (first, twin, third) for n in copy[:4]]
        self.assertEqual(len(set(uuids)), 12)
        self.assertFalse(set(uuids) & {d.uuid.upper() for d in scene.hierarchy})

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
