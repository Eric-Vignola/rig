"""``Transform.create_hierarchy`` and ``load_hierarchy`` key a ClipData (the user's decisions of
2026-10-30; rig/nodetypes/README.md, Serialisation and files).

* The type decides: a ClipData is built and keyed, with no keyword, by ``create_hierarchy`` (the
  object) and ``load_hierarchy`` (the object, or a file that holds one).
* Only what changes gets keys: a column of a node's scale, rotate, translate, rotate axis or joint
  orient with any difference across the frames gets one curve, a key on every frame; a flat one is
  a plain value, with no curve. A one-frame clip makes no curve.
* Frame for frame: key k lands on frame ``start_frame + k`` whatever the scene's rate; a clip at
  another fps logs one warning naming both, keys or none; the time unit and the playback range are
  not touched.
* A clip defines no tangents: Maya's global default, as it is when the clip is built.
* One ``List(plugs) << [AnimCurve(...), ...]``, inside the build's one undo step. The curves join a
  container as the nodes do, and the selection behaves as it does without keys.
* ``parent=`` with ``world_space=True``: the roots are solved on every frame.
* A clip's joints evaluate what its world matrices hold: one whose segment scale compensate is on
  takes its parent's scale out, wired from a joint of the clip as ``cmds.parent`` wires it.
* A plain HierarchyData (``clip.rest_pose``) builds a clip without keys.
* ``create_hierarchy(world_space=)`` takes a bool, as ``load_hierarchy`` does.

Each test failed against the source files of 39a3dbb, the commit before the feature (negative
control), except the guards.
"""

import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from cgmath.hierarchy import ClipData, HierarchyData
from maya import cmds
from rig import AnimCurve, InjectionError, List, Node, container
from rig._internal import anim_keys
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.nodetypes import Joint, Transform, plugins

# the columns `_clip` moves, by node, as a clip's frames name them
_MOVING = {
    "root":  ("rotate_y", "translate_x", "translate_z"),
    "spine": ("rotate_x", "rotate_z", "joint_orient_y"),
    "arm":   ("rotate_x", "rotate_y", "rotate_z", "rotate_axis_x"),
    "grp":   ("rotate_y", "scale_x"),
    "probe": ("rotate_z", "translate_y"),
}
_NAMES = list(_MOVING)


def _attr(column):
    """The Maya attribute of a clip's column: ``rotate_axis_x`` is rotateAxisX."""
    head, *rest = column.split("_")
    return head + "".join(word.capitalize() for word in rest)


def _skeleton():
    """root > spine > arm, joints with orients, a rotate axis and rotate orders that are not xyz;
    root > grp > probe, a transform and a locator."""
    root = Joint.create(name="root")
    root.t.set(1, 2, 3)
    root.ro.set(3)
    spine = Joint.create(name="spine", parent=root)
    spine.t.set(0, 5, 0)
    spine.jointOrient.set(10, 20, 30)
    spine.ro.set(1)
    spine.rotateAxis.set(5, 0, -5)
    arm = Joint.create(name="arm", parent=spine)
    arm.t.set(3, 0, 1)
    arm.jointOrient.set(-40, 15, 70)
    arm.ro.set(5)
    grp = Transform.create(name="grp", parent=root)
    grp.t.set(-2, 1, 0)
    grp.ro.set(4)
    grp.rotateAxis.set(10, 20, 30)
    probe = Node(cmds.spaceLocator(name="probe")[0])
    probe.set_parent(grp)
    probe.t.set(0, 1, 1)
    return root


def _clip(frames=6, start_frame=1001, fps=24.0):
    """`_skeleton` as a clip, the columns of `_MOVING` moving, the scene left empty."""
    clip = ClipData(_skeleton().serialize_hierarchy(), frames=frames, start_frame=start_frame, fps=fps)
    rng  = np.random.default_rng(3)
    for name, columns in _MOVING.items():
        for column in columns:
            values = np.cumsum(rng.uniform(-9, 9, frames))
            setattr(clip.frames[name], column, 1 + np.abs(values) / 9 if "scale" in column else values)
    cmds.delete("root")
    return clip


def _two_transforms(names=("a", "b"), frames=4):
    """A clip of transforms `names`, the second under the first, standing still, the scene left
    empty."""
    top   = Transform.create(name=names[0])
    below = Transform.create(name=names[1], parent=top)
    below.t.set(0, 2, 0)
    clip = ClipData(top.serialize_hierarchy(), frames=frames, start_frame=1)
    cmds.delete(names[0])
    return clip


def _curves():
    return sorted(cmds.ls(type="animCurve"))


def _moving_curves():
    return sorted(f"{name}_{_attr(column)}" for name, columns in _MOVING.items() for column in columns)


def _frames(curve):
    return cmds.keyframe(curve, query=True, timeChange=True)


def _worlds(nodes, clip):
    """(F, N, 4, 4): the world matrix Maya evaluates for `nodes` on each frame of `clip`."""
    names = [node.get_long_name() for node in nodes]
    return np.array([
        [np.reshape(cmds.getAttr(f"{name}.worldMatrix[0]", time=frame), (4, 4)) for name in names]
        for frame in range(clip.start_frame, clip.end_frame + 1)
    ])


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_worlds(self, nodes, clip, want=None):
        """Every node's world matrix on every frame is `want`, by default the clip's own."""
        want = clip.frames.world_matrix if want is None else want
        self.assertLessEqual(np.abs(_worlds(nodes, clip) - want).max(), 1e-9)

    def make_folder(self):
        folder = tempfile.mkdtemp(prefix="rig_clip_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return folder


# ---------------------------------------------------------------------------------------------
class TestTheTypeDecides(_Case):
    """A ClipData, the object or a file that holds one, is built and keyed, with no keyword."""

    def test_create_takes_no_path(self):
        """A file is load_hierarchy's: an error that says so, before anything is made."""
        before = cmds.ls()
        for path in ("walk.npz", pathlib.Path("walk.npz")):
            with self.subTest(type(path).__name__):
                with self.assertRaisesRegex(TypeError, r"Joint\.load_hierarchy\('walk\.npz'\) reads one"):
                    Joint.create_hierarchy(path)
        self.assertEqual(cmds.ls(), before)

    def test_the_object(self):
        calls = (
            ("Transform.create_hierarchy", Transform.create_hierarchy),
            ("Joint.create_hierarchy", Joint.create_hierarchy),
            ("Transform.load_hierarchy", Transform.load_hierarchy),
            ("Joint.load_hierarchy", Joint.load_hierarchy),
        )
        for label, call in calls:
            with self.subTest(label):
                self.new_scene()
                nodes = call(_clip())
                self.assertIsInstance(nodes, List)
                self.assertEqual([n.get_short_name() for n in nodes], _NAMES)
                self.assertEqual([type(n).__name__ for n in nodes], ["Joint"] * 3 + ["Transform"] * 2)
                self.assertEqual(_curves(), _moving_curves())

    def test_a_file_that_holds_one(self):
        folder = self.make_folder()
        for ext in ("npz", "json", "pkl"):
            with self.subTest(ext=ext):
                self.new_scene()
                clip = _clip()
                path = os.path.join(folder, f"clip.{ext}")
                clip.save(path)
                for load in (path, pathlib.Path(path)):
                    self.new_scene()
                    nodes = Transform.load_hierarchy(load)
                    self.assertEqual(_curves(), _moving_curves())
                    self.assert_worlds(nodes, clip)


# ---------------------------------------------------------------------------------------------
class TestOnlyWhatChanges(_Case):
    """A column with any difference across the frames gets one curve, a key on every frame; a flat
    one is a plain value, with no curve."""

    def test_a_curve_for_each_column_that_changes_a_key_on_every_frame(self):
        clip = _clip(frames=6)
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        for curve in _curves():
            self.assertEqual(_frames(curve), [float(f) for f in range(1001, 1007)], curve)

    def test_each_curve_holds_its_column(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        for name, columns in _MOVING.items():
            for column in columns:
                with self.subTest(f"{name}.{column}"):
                    keys = cmds.keyframe(f"{name}_{_attr(column)}", query=True, valueChange=True)
                    np.testing.assert_allclose(keys, getattr(clip.frames[name], column).ravel(), atol=1e-9)

    def test_a_flat_column_is_a_plain_value_that_holds(self):
        Transform.create_hierarchy(_clip())
        self.assertEqual(_curves(), _moving_curves())
        for plug, value in (("spine.ty", 5.0), ("spine.tx", 0.0), ("arm.tz", 1.0), ("root.ty", 2.0), ("grp.sy", 1.0)):
            with self.subTest(plug):
                self.assertIsNone(cmds.listConnections(plug, source=True, destination=False))
                held = [cmds.getAttr(plug, time=frame) for frame in (1001, 1003, 1006, 0)]
                np.testing.assert_allclose(held, value, atol=1e-12)

    def test_rotate_axis_and_joint_orient_are_keyed_only_where_they_change(self):
        Transform.create_hierarchy(_clip())
        self.assertEqual([c for c in _curves() if "jointOrient" in c], ["spine_jointOrientY"])
        self.assertEqual([c for c in _curves() if "rotateAxis" in c], ["arm_rotateAxisX"])
        self.assertAlmostEqual(cmds.getAttr("spine.jointOrientX", time=1003), 10.0, places=12)
        self.assertAlmostEqual(cmds.getAttr("arm.jointOrientZ", time=1003),   70.0, places=12)
        self.assertAlmostEqual(cmds.getAttr("grp.rotateAxisZ", time=1003),    30.0, places=12)

    def test_any_difference_is_a_key_on_every_frame(self):
        clip                         = _two_transforms(frames=5)
        clip.frames["a"].translate_x = [1, 1, 1, 1, 1 + 1e-12]
        clip.frames["b"].scale_z     = [1, 2, 1, 1, 1]
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), ["a_translateX", "b_scaleZ"])
        self.assertEqual(_frames("a_translateX"), [1, 2, 3, 4, 5])
        self.assertEqual(_frames("b_scaleZ"), [1, 2, 3, 4, 5])

    def test_a_transform_has_no_joint_orient_to_key(self):
        clip                            = _two_transforms(frames=3)
        clip.frames["b"].joint_orient_y = [0, 10, 20]
        clip.frames["b"].rotate_y       = [0, 10, 20]
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), ["b_rotateY"])

    def test_two_frames_are_enough(self):
        clip                         = _two_transforms(frames=2)
        clip.frames["a"].translate_x = [1, 2]
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), ["a_translateX"])
        self.assertEqual(_frames("a_translateX"), [1, 2])

    def test_a_one_frame_clip_makes_no_curve(self):
        """guard: a one-frame clip is its plain values."""
        clip  = _clip(frames=1)
        nodes = Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), [])
        self.assert_worlds(nodes, clip)

    def test_a_clip_that_stands_still_makes_no_curve(self):
        """guard: no column differs."""
        clip = _two_transforms(frames=5)
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), [])
        self.assertEqual(cmds.getAttr("b.ty"), 2.0)

    def test_a_clip_with_no_frames_makes_no_curve(self):
        """guard: a clip made without ``frames=`` holds its nodes' one pose."""
        root = Transform.create(name="a")
        root.tx.set(4)
        clip = ClipData(root.serialize_hierarchy())
        cmds.delete("a")
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), [])
        self.assertEqual(cmds.getAttr("a.tx"), 4.0)

    def test_a_clip_with_no_frames_under_a_parent(self):
        """guard: it has no block to read a scale from."""
        root = Transform.create(name="a")
        root.tx.set(4)
        clip = ClipData(root.serialize_hierarchy())
        cmds.delete("a")
        for world_space, value in ((False, 4.0), (True, 2.0)):
            with self.subTest(world_space=world_space):
                self.new_scene()
                up    = Transform.create(name="up", sx=2, sy=2, sz=2)
                nodes = Transform.create_hierarchy(clip, parent=up, world_space=world_space)
                self.assertEqual(_curves(), [])
                self.assertEqual(nodes[0].tx.get(), value)


# ---------------------------------------------------------------------------------------------
class TestNames(_Case):
    """The keys go by node, not by name."""

    def test_a_name_maya_took(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        again = Transform.create_hierarchy(clip)
        self.assertEqual(again[0].get_short_name(), "root1")
        self.assertEqual(len(_curves()), 2 * len(_moving_curves()))
        self.assert_worlds(again, clip)

    def test_one_short_name_twice(self):
        body = Transform.create(name="body")
        for side in ("L", "R"):
            Transform.create(name="hand", parent=Transform.create(name=side, parent=body))
        clip = ClipData(body.serialize_hierarchy(), frames=3, start_frame=1)
        self.assertEqual(clip.name, ["body", "L", "hand", "R", "hand"])
        clip.frames[[2]].translate_x = [1, 2, 3]
        clip.frames[[4]].translate_x = [7, 8, 9]
        cmds.delete("body")
        nodes = Transform.create_hierarchy(clip)
        self.assertEqual(len(_curves()), 2)
        self.assertEqual(cmds.getAttr("body|L|hand.tx", time=2), 2.0)
        self.assertEqual(cmds.getAttr("body|R|hand.tx", time=2), 8.0)
        self.assert_worlds(nodes, clip)


# ---------------------------------------------------------------------------------------------
class TestWorldMatrices(_Case):
    """Every node's world matrix on every frame is the clip's (<= 1e-9)."""

    def test_joints_and_transforms_on_every_frame(self):
        clip  = _clip(frames=12)
        nodes = Transform.create_hierarchy(clip)
        self.assert_worlds(nodes, clip)
        self.assertGreater(np.ptp(_worlds(nodes, clip)[:, 2], axis=0).max(), 1.0)  # the arm moves

    def test_every_rotate_order(self):
        tops = []
        for order in range(6):
            for maker, prefix in ((Joint, "j"), (Transform, "t")):
                top = maker.create(name=f"{prefix}{order}")
                top.ro.set(order)
                top.rotateAxis.set(11, 22, 33)
                tops.append(top)
        data               = HierarchyData([node.copy() for top in tops for node in top.serialize_hierarchy()])
        clip               = ClipData(data, frames=5, start_frame=1)
        rng                = np.random.default_rng(5)
        clip.frames.rotate = rng.uniform(-170, 170, (5, 12, 3))
        cmds.delete([str(top) for top in tops])
        self.assert_worlds(Transform.create_hierarchy(clip), clip)

    def test_on_the_frame_the_clip_is_bound_to_or_any_other(self):
        clip       = _clip()
        clip.frame = 4
        nodes      = Transform.create_hierarchy(clip)
        self.assert_worlds(nodes, clip)
        self.assertEqual(cmds.getAttr("root.ty"), 2.0)

    def test_the_scene_evaluates_them_at_its_current_time(self):
        clip  = _clip()
        nodes = Transform.create_hierarchy(clip)
        want  = clip.frames.world_matrix
        for k in range(clip.frame_count):
            cmds.currentTime(clip.start_frame + k)
            for n, node in enumerate(nodes):
                got = np.array(node.get_matrix(world_space=True)).reshape(4, 4)
                np.testing.assert_allclose(got, want[k, n], atol=1e-9)


# ---------------------------------------------------------------------------------------------
class TestFrameForFrame(_Case):
    """Key k lands on frame ``start_frame + k`` whatever the scene's rate."""

    def test_key_k_lands_on_start_frame_plus_k(self):
        for start in (1001, 0, -5):
            with self.subTest(start=start):
                self.new_scene()
                clip                         = _two_transforms(frames=3)
                clip.start_frame             = start
                clip.frames["a"].translate_x = [1, 2, 3]
                Transform.create_hierarchy(clip)
                self.assertEqual(_frames("a_translateX"), [start, start + 1, start + 2])

    def test_another_fps_logs_one_warning_naming_both(self):
        clip = _clip(fps=30.0)
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            nodes = Transform.create_hierarchy(clip)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("30 fps", logs.records[0].getMessage())
        self.assertIn("24 fps", logs.records[0].getMessage())
        self.assertEqual(_frames("root_translateX"), [float(f) for f in range(1001, 1007)])
        self.assert_worlds(nodes, clip)

    def test_load_hierarchy_warns_too(self):
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            Joint.load_hierarchy(_clip(fps=48.0))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("48 fps", logs.records[0].getMessage())

    def test_the_scene_s_own_rate_logs_nothing(self):
        clip = _clip(fps=24.0)
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())

    def test_a_clip_that_makes_no_key_warns_too(self):
        still  = _two_transforms()
        one    = _two_transforms(frames=1)
        orient = _two_transforms(frames=3)
        orient.frames["b"].joint_orient_y = [0, 10, 20]  # a transform has none to key
        for label, clip in (("flat", still), ("one frame", one), ("no plug to key", orient)):
            with self.subTest(label):
                clip.fps = 30.0
                with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
                    Transform.create_hierarchy(clip)
                self.assertEqual(len(logs.records), 1)
                self.assertIn("30 fps", logs.records[0].getMessage())
                self.assertIn("24 fps", logs.records[0].getMessage())
        self.assertEqual(_curves(), [])

    def test_a_rate_near_the_scene_s_warns(self):
        clip = _clip(fps=23.976)
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            Transform.create_hierarchy(clip)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("23.976 fps", logs.records[0].getMessage())

    def test_the_time_unit_and_the_playback_range_are_not_touched(self):
        clip = _clip(fps=30.0)
        cmds.playbackOptions(min=10, max=20, animationStartTime=5, animationEndTime=25)
        cmds.currentTime(12)
        with self.assertLogs("rig.nodetypes.transform", level="WARNING"):
            Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        self.assertEqual(cmds.currentUnit(query=True, time=True), "film")
        self.assertEqual(cmds.playbackOptions(query=True, min=True), 10)
        self.assertEqual(cmds.playbackOptions(query=True, max=True), 20)
        self.assertEqual(cmds.playbackOptions(query=True, animationStartTime=True), 5)
        self.assertEqual(cmds.playbackOptions(query=True, animationEndTime=True), 25)
        self.assertEqual(cmds.currentTime(query=True), 12)

    def test_a_scene_at_another_rate(self):
        self.addCleanup(cmds.currentUnit, time="film")
        cmds.currentUnit(time="ntsc")
        clip = _clip(fps=30.0)
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            nodes = Transform.create_hierarchy(clip)
        self.assertEqual(_frames("root_translateX"), [float(f) for f in range(1001, 1007)])
        self.assert_worlds(nodes, clip)
        clip.fps = 24.0
        with self.assertLogs("rig.nodetypes.transform", level="WARNING") as logs:
            Transform.create_hierarchy(clip)
        self.assertIn("24 fps", logs.records[0].getMessage())
        self.assertIn("30 fps", logs.records[0].getMessage())

    def test_a_rate_written_short_is_the_scene_s(self):
        self.addCleanup(cmds.currentUnit, time="film")
        cmds.currentUnit(time="23.976fps")
        clip = _clip(fps=23.976)
        with self.assertNoLogs("rig.nodetypes.transform", level="WARNING"):
            Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())


# ---------------------------------------------------------------------------------------------
class TestTangents(_Case):
    """A clip defines no tangents: Maya's global default, as it is when the clip is built."""

    def setUp(self):
        super().setUp()
        kept = {
            "inTangentType":  cmds.keyTangent(query=True, g=True, inTangentType=True)[0],
            "outTangentType": cmds.keyTangent(query=True, g=True, outTangentType=True)[0],
        }
        self.addCleanup(cmds.keyTangent, g=True, **kept)

    def assert_tangents(self, in_type, out_type):
        self.assertEqual(_curves(), _moving_curves())
        for curve in _curves():
            self.assertEqual(set(cmds.keyTangent(curve, query=True, inTangentType=True)), {in_type}, curve)
            self.assertEqual(set(cmds.keyTangent(curve, query=True, outTangentType=True)), {out_type}, curve)

    def test_the_global_default(self):
        Transform.create_hierarchy(_clip())
        self.assert_tangents(
            cmds.keyTangent(query=True, g=True, inTangentType=True)[0],
            cmds.keyTangent(query=True, g=True, outTangentType=True)[0],
        )

    def test_it_follows_a_change(self):
        for in_type, out_type in (
            ("linear", "linear"), ("spline", "clamped"), ("flat", "step"), ("plateau", "stepnext"),
            ("autoease", "automix"), ("fast", "slow"), ("autocustom", "auto"),
        ):
            with self.subTest(in_type=in_type, out_type=out_type):
                self.new_scene()
                cmds.keyTangent(g=True, inTangentType=in_type, outTangentType=out_type)
                Transform.load_hierarchy(_clip())
                self.assert_tangents(in_type, out_type)


# ---------------------------------------------------------------------------------------------
class TestOneStatement(_Case):
    """Every curve in one ``List(plugs) << [AnimCurve(...), ...]``: every row planned first, and a
    clip no key can hold refused before any node is made."""

    def test_every_curve_is_one_list_statement(self):
        clip = _clip()
        with mock.patch.object(anim_keys, "inject_rows", wraps=anim_keys.inject_rows) as rows:
            Transform.create_hierarchy(clip)
        rows.assert_called_once()
        self.assertEqual(len(rows.call_args.args[0]), len(_moving_curves()))
        self.assertEqual(_curves(), _moving_curves())

    def test_a_nan_refuses_with_nothing_made(self):
        clip                        = _clip()
        clip.frames["arm"].rotate_x = [0, 1, np.nan, 3, 4, 5]
        nodes                       = sorted(cmds.ls())
        with self.assertRaisesRegex(ValueError, "AnimCurve values must be finite"):
            Transform.create_hierarchy(clip)
        self.assertEqual(sorted(cmds.ls()), nodes)

    def test_without_the_undo_plugin_nothing_is_made(self):
        clip  = _clip()
        nodes = sorted(cmds.ls())
        with mock.patch.object(plugins, "_fast_path_ready", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "undo plug-in"):
                Transform.create_hierarchy(clip)
        self.assertEqual(sorted(cmds.ls()), nodes)


# ---------------------------------------------------------------------------------------------
class TestOneUndoStep(UndoWalk, _Case):
    """The curves and their keys are in the build's one undo step."""

    def test_one_undo_takes_nodes_and_curves_and_redo_brings_them_back(self):
        for label, call, step in (
            ("create_hierarchy", Transform.create_hierarchy, "rig.Transform.create_hierarchy"),
            ("Joint.create_hierarchy", Joint.create_hierarchy, "rig.Joint.create_hierarchy"),
            ("load_hierarchy", Transform.load_hierarchy, "rig.Transform.load_hierarchy"),
            ("Joint.load_hierarchy", Joint.load_hierarchy, "rig.Joint.load_hierarchy"),
        ):
            with self.subTest(label):
                self.new_scene()
                clip = _clip()
                cmds.flushUndo()
                before = self.scene_state()
                nodes  = call(clip)
                after  = self.scene_state()
                self.assertEqual(_curves(), _moving_curves())
                self.assertEqual(self.undo_name(), step)
                self.undo_steps(1)
                self.assert_state_equal(self.scene_state(), before)
                self.assertEqual(_curves(), [])
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.redo_steps(1)
                self.assert_state_equal(self.scene_state(), after)
                self.assert_worlds(nodes, clip)
                self.assertEqual(self.undo_name(), step)

    def test_a_walk_back_and_forth_ends_where_it_started(self):
        clip = _clip()
        cmds.flushUndo()
        nodes = Transform.create_hierarchy(clip, parent=Transform.create(name="grp1"), world_space=True)
        for _ in range(2):
            self.undo_all()
            self.assertEqual(cmds.ls(type="joint"), [])
            self.redo_all()
        self.assert_worlds(nodes, clip)


# ---------------------------------------------------------------------------------------------
class TestSelection(_Case):
    """The selection behaves as it does without keys."""

    def test_create_hierarchy_selects_its_nodes_not_the_curves(self):
        clip = _clip()
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        self.assertEqual(cmds.ls(selection=True), _NAMES)

    def test_load_hierarchy_leaves_it_as_it_was(self):
        clip = _clip()
        for name in ("c", "b", "a"):
            Transform.create(name=name)
        cmds.select("c", "a")
        Transform.load_hierarchy(clip)
        self.assertEqual(_curves(), _moving_curves())
        self.assertEqual(cmds.ls(orderedSelection=True), ["c", "a"])


# ---------------------------------------------------------------------------------------------
class TestContainer(UndoWalk, _Case):
    """The curves join a ``with container()`` scope as the nodes do."""

    def assert_members(self, box):
        members = set(cmds.container(str(box), query=True, nodeList=True))
        self.assertEqual(members, {*_NAMES, "probeShape", *_moving_curves()})

    def test_the_curves_join_with_the_nodes(self):
        for label, call in (("create_hierarchy", Transform.create_hierarchy), ("load_hierarchy", Joint.load_hierarchy)):
            with self.subTest(label):
                self.new_scene()
                clip = _clip()
                with container("box") as box:
                    nodes = call(clip)
                self.assert_members(box)
                self.assert_worlds(nodes, clip)

    def test_one_undo_takes_them_out_and_redo_brings_them_back(self):
        clip = _clip()
        with container("box") as box:
            nodes = Transform.create_hierarchy(clip)
        self.undo_steps(1)
        self.assertEqual(_curves(), [])
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True) or [], [])
        self.redo_steps(1)
        self.assert_members(box)
        self.assert_worlds(nodes, clip)

    def test_container_false_leaves_them_out_with_the_nodes(self):
        for label, call in (("create_hierarchy", Transform.create_hierarchy), ("load_hierarchy", Joint.load_hierarchy)):
            with self.subTest(label):
                self.new_scene()
                clip = _clip()
                with container("box") as box:
                    Transform.create(name="other")
                    call(clip, container=False)
                self.assertEqual(cmds.container(str(box), query=True, nodeList=True), ["other"])
                self.assertEqual(_curves(), _moving_curves())

    def test_container_false_in_a_locked_scope_makes_them_all(self):
        clip = _clip()
        with container("arm_rig"):
            cmds.lockNode("arm_rig", lock=True)
            try:
                Transform.create_hierarchy(clip, container=False)
            finally:
                cmds.lockNode("arm_rig", lock=False)
        self.assertEqual(_curves(), _moving_curves())

    def test_the_keys_of_a_create_follow_its_container_flag(self):
        with container("box") as box:
            Transform.create(name="joined", tx=AnimCurve([0, 1]))
            Transform.create(name="kept_out", container=False, tx=AnimCurve([0, 1]))
        self.assertEqual(sorted(cmds.container(str(box), query=True, nodeList=True)), ["joined", "joined_translateX"])
        self.assertEqual(_curves(), ["joined_translateX", "kept_out_translateX"])

    def test_the_keys_of_a_create_that_tracks_what_it_makes_join_with_it(self):
        data = Transform.create(name="a").serialize()
        cmds.delete("a")
        with container("box") as box:
            Transform.create(data, tx=AnimCurve([0, 1]))
        self.assertEqual(sorted(cmds.container(str(box), query=True, nodeList=True)), ["a", "a_translateX"])

    def test_keys_written_in_a_scope_join_it(self):
        """guard: outside a create, a new curve registers itself."""
        node = Transform.create(name="a")
        with container("box") as box:
            node.tx << AnimCurve([0, 1])
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True), ["a_translateX"])

    def test_a_locked_scope_takes_nothing(self):
        """guard: refused before the create."""
        clip = _clip()
        with container("arm_rig"):
            cmds.lockNode("arm_rig", lock=True)
            try:
                nodes = sorted(cmds.ls())
                with self.assertRaisesRegex(InjectionError, "cannot add a node to 'arm_rig': it is locked"):
                    Transform.create_hierarchy(clip)
                self.assertEqual(sorted(cmds.ls()), nodes)
            finally:
                cmds.lockNode("arm_rig", lock=False)


# ---------------------------------------------------------------------------------------------
class TestParent(_Case):
    """``parent=``: the clip's values under it, or with ``world_space=True`` its world matrices,
    the roots solved on every frame."""

    @staticmethod
    def _rig_grp(scale=1):
        grp = Transform.create(name="rig_grp")
        grp.t.set(10, -3, 7)
        grp.r.set(30, 40, 50)
        grp.s.set(scale, scale, scale)
        return grp

    def test_world_space_is_the_clip_s_on_every_frame(self):
        for label, call in (("create_hierarchy", Transform.create_hierarchy), ("load_hierarchy", Joint.load_hierarchy)):
            for given in ("rig_grp", "the node"):
                with self.subTest(label, parent=given):
                    self.new_scene()
                    clip  = _clip()
                    grp   = self._rig_grp()
                    nodes = call(clip, parent=grp if given == "the node" else "rig_grp", world_space=True)
                    self.assertEqual(nodes[0].get_long_name(), "|rig_grp|root")
                    self.assert_worlds(nodes, clip)

    def test_world_space_takes_the_parent_s_scale_from_the_first_root_on(self):
        """cgmath lines a new node up in a clip's blocks when a solve first reads it, and that read
        takes its scale from the blocks before the line-up: a stand-in left to it loses the
        parent's scale on the first root's solve."""
        clip                         = _two_transforms(frames=3)
        clip.frames["a"].rotate_y    = [0, 40, 80]
        clip.frames["b"].translate_x = [1, 2, 3]
        nodes                        = Transform.create_hierarchy(clip, parent=self._rig_grp(scale=2), world_space=True)
        self.assert_worlds(nodes, clip)
        self.assertAlmostEqual(cmds.getAttr(f"{nodes[0].get_long_name()}.sy", time=2), 0.5, places=9)

    def test_world_space_with_a_joint_that_does_not_compensate(self):
        clip = _clip()
        for node in clip:
            node.segment_scale_compensate = False
        nodes = Transform.create_hierarchy(clip, parent=self._rig_grp(scale=2), world_space=True)
        self.assert_worlds(nodes, clip)
        self.assertEqual(len(_curves()), 17)

    def test_world_space_keys_the_channels_as_they_were_solved(self):
        clip                      = _two_transforms(frames=3)
        clip.frames["a"].rotate_y = [0, 40, 80]
        nodes                     = Transform.create_hierarchy(clip, parent=self._rig_grp(), world_space=True)
        self.assert_worlds(nodes, clip)
        self.assertEqual(_curves(), ["a_rotateX", "a_rotateY", "a_rotateZ"])
        self.assertEqual(cmds.getAttr(f"{nodes[1].get_long_name()}.ty"), 2.0)

    def test_world_space_leaves_a_solve_s_round_off_flat(self):
        """A root that turns comes back with its scale 1 give or take 1e-15: no curve."""
        clip                    = _two_transforms(frames=40)
        clip.frames["a"].rotate = np.cumsum(np.random.default_rng(8).uniform(-30, 30, (40, 1, 3)), axis=0)
        nodes                   = Transform.create_hierarchy(clip, parent=self._rig_grp(scale=2), world_space=True)
        self.assertEqual(_curves(), ["a_rotateX", "a_rotateY", "a_rotateZ"])
        self.assert_worlds(nodes, clip)

    def test_world_space_still_keys_what_really_changes(self):
        clip                     = _two_transforms(frames=3)
        clip.frames["a"].scale_x = [1, 1 + 1e-9, 1]
        Transform.create_hierarchy(clip, parent=self._rig_grp(), world_space=True)
        self.assertEqual(_curves(), ["a_scaleX"])

    def test_the_round_off_of_a_large_value_is_flat_too(self):
        """A root under a scale of 1e-5 holds a scale of 1e5, its round-off above 1e-12."""
        clip                    = _two_transforms(frames=40)
        clip.frames["a"].rotate = np.cumsum(np.random.default_rng(8).uniform(-30, 30, (40, 1, 3)), axis=0)
        nodes                   = Transform.create_hierarchy(clip, parent=self._rig_grp(scale=1e-5), world_space=True)
        self.assertEqual(_curves(), ["a_rotateX", "a_rotateY", "a_rotateZ"])
        self.assert_worlds(nodes, clip)

    def test_only_a_solve_has_a_round_off(self):
        """Any difference still counts on a root the default takes as it is, and on a node below a
        root that is solved."""
        for label, name, world_space in (("the root, as it is", "a", False), ("below a solved root", "b", True)):
            with self.subTest(label):
                self.new_scene()
                clip                          = _two_transforms(frames=3)
                clip.frames[name].translate_z = [0, 1e-13, 0]
                Transform.create_hierarchy(clip, parent=self._rig_grp(), world_space=world_space)
                self.assertEqual(_curves(), [f"{name}_translateZ"])

    def test_the_default_takes_the_clip_s_values_under_the_parent(self):
        clip  = _clip()
        grp   = self._rig_grp(scale=2)
        nodes = Transform.create_hierarchy(clip, parent=grp)
        self.assertEqual(_curves(), _moving_curves())
        self.assert_worlds(nodes, clip, clip.frames.world_matrix @ np.reshape(cmds.getAttr("rig_grp.wm[0]"), (4, 4)))

    def test_two_roots(self):
        one, two = _two_transforms(("a", "b")), _two_transforms(("c", "d"))
        data = HierarchyData([node.copy() for node in (*one, *two)])
        clip = ClipData(data, frames=3, start_frame=1)
        clip.frames["a"].rotate_y    = [0, 40, 80]
        clip.frames["d"].translate_x = [1, 2, 3]
        nodes = Transform.create_hierarchy(clip, parent=self._rig_grp(scale=3), world_space=True)
        self.assertEqual([n.get_long_name() for n in nodes[:2]], ["|rig_grp|a", "|rig_grp|a|b"])
        self.assert_worlds(nodes, clip)

    def test_a_parent_the_clip_was_taken_from(self):
        """The node keeps its uuid in the clip: the stand-in does not take the column it holds."""
        _skeleton()
        clip                          = ClipData(Node("root").serialize_hierarchy(), frames=4, start_frame=1)
        clip.frames["spine"].rotate_y = [0, 30, 60, 90]
        clip.add_prefix("b_")
        cmds.setAttr("root.tx", 5)
        nodes = Transform.create_hierarchy(clip, parent="spine", world_space=True)
        self.assertEqual(nodes[0].get_long_name(), "|root|spine|b_root")
        self.assert_worlds(nodes, clip)

    def test_a_parent_that_moves_is_taken_where_it_is_at_the_build(self):
        clip                      = _two_transforms(frames=5)
        clip.frames["a"].rotate_y = [0, 20, 40, 60, 80]
        grp                       = self._rig_grp()
        cmds.setKeyframe("rig_grp", attribute="translateX", time=1, value=0)
        cmds.setKeyframe("rig_grp", attribute="translateX", time=5, value=8)
        cmds.currentTime(3)
        nodes = Transform.create_hierarchy(clip, parent=grp, world_space=True)
        moved = [np.reshape(cmds.getAttr("rig_grp.wm[0]", time=frame), (4, 4)) for frame in range(1, 6)]
        taken = np.linalg.inv(moved[2]) @ np.array(moved)  # how far the parent is from the build's frame
        self.assert_worlds(nodes, clip, clip.frames.world_matrix @ taken[:, None])
        self.assertGreater(np.abs(taken[0] - np.eye(4)).max(), 1.0)

    def test_a_plain_hierarchy_under_a_parent(self):
        """guard: the stand-in's new uuid changes nothing for one pose."""
        root = _skeleton()
        data = root.serialize_hierarchy()
        cmds.delete("root")
        nodes = Transform.create_hierarchy(data, parent=self._rig_grp(), world_space=True)
        for node, source in zip(nodes, data):
            world = np.reshape(cmds.getAttr(f"{node.get_long_name()}.worldMatrix[0]"), (4, 4))
            np.testing.assert_allclose(world, source.world_matrix, atol=1e-9)


# ---------------------------------------------------------------------------------------------
class TestCompensation(UndoWalk, _Case):
    """A clip's joints evaluate what its world matrices hold: one whose segment scale compensate is
    on takes its parent's scale out, on every frame."""

    @staticmethod
    def _chain(frames=6):
        """root > spine > arm > hand, joints as ``Joint.create`` makes them (they compensate), the
        spine and the arm stretching, as a clip; the scene left empty."""
        root  = Joint.create(name="root")
        spine = Joint.create(name="spine", parent=root)
        arm   = Joint.create(name="arm",   parent=spine)
        hand  = Joint.create(name="hand",  parent=arm)
        root.t.set(1, 2, 3)
        spine.t.set(0, 5, 0)
        arm.t.set(3, 0, 1)
        hand.t.set(2, 0, 0)
        spine.jointOrient.set(10, 20, 30)
        clip                          = ClipData(root.serialize_hierarchy(), frames=frames, start_frame=1)
        line                          = np.linspace(0, 1, frames)
        clip.frames["root"].rotate_y  = 120 * line
        clip.frames["spine"].rotate_x = 40 * line
        clip.frames["spine"].scale_x  = 1 + line
        clip.frames["spine"].scale_y  = 1 + 2 * line
        clip.frames["arm"].rotate_y   = 70 * line
        clip.frames["arm"].scale_z    = 1 - 0.5 * line
        cmds.delete("root")
        return clip

    @staticmethod
    def _input(plug):
        return cmds.listConnections(plug, source=True, destination=False, plugs=True)

    def _chain_without_its_root_s_compensation(self):
        clip                                  = self._chain()
        clip["root"].segment_scale_compensate = False
        return clip

    def test_a_joint_under_a_joint_whose_scale_changes(self):
        for label, call in (("create_hierarchy", Transform.create_hierarchy), ("load_hierarchy", Joint.load_hierarchy)):
            with self.subTest(label):
                self.new_scene()
                clip  = self._chain()
                nodes = call(clip)
                self.assert_worlds(nodes, clip)
                self.assertIsNone(self._input("root.inverseScale"))
                for child, up in (("spine", "root"), ("arm", "spine"), ("hand", "arm")):  # as cmds.parent wires it
                    self.assertEqual(self._input(f"{child}.inverseScale"), [f"{up}.scale"], child)

    def test_a_joint_that_does_not_compensate_is_not_wired(self):
        clip                                 = self._chain()
        clip["arm"].segment_scale_compensate = False
        nodes                                = Transform.create_hierarchy(clip)
        self.assertIsNone(self._input("arm.inverseScale"))
        self.assertEqual(self._input("hand.inverseScale"), ["arm.scale"])
        self.assert_worlds(nodes, clip)

    def test_a_joint_under_a_transform_is_left_as_maya_leaves_it(self):
        """guard: ``cmds.parent`` wires no transform's scale, so the joint lands where it was. cgmath
        takes that scale out too, so it is not at the clip's ``world_matrix``."""
        grp = Transform.create(name="grp")
        grp.s.set(2, 2, 2)
        Joint.create(name="j", parent=grp, tx=1)
        Joint.create(name="k", parent="grp|j", tx=1)
        was  = cmds.getAttr("k.worldMatrix[0]")
        clip = ClipData(grp.serialize_hierarchy(), frames=3, start_frame=1)
        cmds.delete("grp")
        Transform.create_hierarchy(clip)
        self.assertIsNone(self._input("j.inverseScale"))
        self.assertEqual(cmds.getAttr("k.worldMatrix[0]"), was)
        self.assertGreater(np.abs(np.reshape(was, (4, 4)) - clip.frames.world_matrix[0, 2]).max(), 0.5)

    def test_world_space_under_any_scale(self):
        def group(scale):
            return Transform.create(name="up", tx=10, ry=30, sx=scale[0], sy=scale[1], sz=scale[2])

        def below(scale):
            return Transform.create(name="up", tx=5, rz=20, parent=group(scale))

        def joint(scale):
            return Joint.create(name="up", tx=3, sx=scale[0], sy=scale[1], sz=scale[2])

        def below_a_joint(scale):
            return Joint.create(name="low", tx=3, parent=joint(scale))

        for label, make, scale in (
            ("no scale", group, (1, 1, 1)), ("uniform", group, (2, 2, 2)), ("small", group, (0.01, 0.01, 0.01)),
            ("uneven", group, (2, 1, 3)), ("a mirror", group, (-1, 1, 1)), ("from above", below, (2, 2, 2)),
            ("a joint", joint, (2, 2, 2)), ("below a joint", below_a_joint, (2, 2, 2)),
        ):
            with self.subTest(label):
                self.new_scene()
                clip  = self._chain()
                nodes = Joint.create_hierarchy(clip, parent=make(scale), world_space=True)
                self.assert_worlds(nodes, clip)
                self.assertEqual(self._input(f"{nodes[1].get_long_name()}.inverseScale"), ["root.scale"])

    def test_a_solved_root_holds_the_scale_its_solve_took_out(self):
        """A plain value: with the scale from above, the parent's own plug does not hold it."""
        grp = Transform.create(name="up", parent=Transform.create(name="top", sx=2, sy=2, sz=2))
        got = Joint.create_hierarchy(self._chain(), parent=grp, world_space=True)[0].get_long_name()
        self.assertIsNone(self._input(f"{got}.inverseScale"))
        np.testing.assert_allclose(cmds.getAttr(f"{got}.inverseScale")[0], (2, 2, 2), atol=1e-12)
        np.testing.assert_allclose(cmds.getAttr(f"{got}.scale")[0], (1, 1, 1), atol=1e-12)
        self.assertEqual(cmds.getAttr(f"{got}.segmentScaleCompensate"), True)

    def test_a_solved_root_that_does_not_compensate(self):
        """The first joint of an FBX: its solve takes the parent's scale into its own, so its joints
        no longer find what they take out on its plug, and hold it. Plain here: 1."""
        clip  = self._chain_without_its_root_s_compensation()
        up    = Transform.create(name="up", sx=2, sy=2, sz=2)
        nodes = Joint.create_hierarchy(clip, parent=up, world_space=True)
        self.assert_worlds(nodes, clip)
        spine = nodes[1].get_long_name()
        self.assertIsNone(self._input(f"{spine}.inverseScale"))
        self.assertEqual(cmds.getAttr(f"{spine}.inverseScale")[0], (1, 1, 1))
        self.assertAlmostEqual(cmds.getAttr(f"{nodes[0].get_long_name()}.sx"), 0.5, places=12)
        self.assertEqual(cmds.getAttr(f"{spine}.segmentScaleCompensate"), True)

    def test_its_scale_is_keyed_on_the_joints_that_take_it_out(self):
        """A flat scale is a plain value there too, and its keys go in the one statement."""
        clip                        = self._chain_without_its_root_s_compensation()
        clip.frames["root"].scale_y = 2.0
        clip.frames["root"].scale_x = 1 + np.linspace(0, 1, 6)
        up                          = Transform.create(name="up", sx=2, sy=2, sz=2)
        with mock.patch.object(anim_keys, "inject_rows", wraps=anim_keys.inject_rows) as rows:
            nodes = Joint.create_hierarchy(clip, parent=up, world_space=True)
        rows.assert_called_once()
        self.assert_worlds(nodes, clip)
        spine = nodes[1].get_long_name()
        keys  = [str(curve) for curve in AnimCurve.of(nodes[1])]
        self.assertIn("spine_inverseScaleX", keys)
        self.assertNotIn("spine_inverseScaleY", keys)
        self.assertEqual(cmds.getAttr(f"{spine}.inverseScaleY"), 2.0)
        held = cmds.keyframe("spine_inverseScaleX", query=True, valueChange=True)
        np.testing.assert_allclose(held, clip.frames["root"].scale_x.ravel())

    def test_a_held_scale_that_moves_late_takes_the_global_tangents(self):
        """Keyed on every frame, the held ones too, with the tangents every key takes."""
        kept = {
            "inTangentType":  cmds.keyTangent(query=True, g=True, inTangentType=True)[0],
            "outTangentType": cmds.keyTangent(query=True, g=True, outTangentType=True)[0],
        }
        self.addCleanup(cmds.keyTangent, g=True, **kept)
        cmds.keyTangent(g=True, inTangentType="flat", outTangentType="step")
        clip = self._chain_without_its_root_s_compensation()
        clip.frames["root"].scale_x = [1, 1, 1, 1, 1, 2]
        up    = Transform.create(name="up", sx=2, sy=2, sz=2)
        nodes = Joint.create_hierarchy(clip, parent=up, world_space=True)
        self.assert_worlds(nodes, clip)
        curve = "spine_inverseScaleX"
        self.assertEqual(cmds.keyframe(curve, query=True, valueChange=True),           [1, 1, 1, 1, 1, 2])
        self.assertEqual(set(cmds.keyTangent(curve, query=True, inTangentType=True)),  {"flat"})
        self.assertEqual(set(cmds.keyTangent(curve, query=True, outTangentType=True)), {"step"})

    def test_a_parent_that_leaves_the_scale_alone_keeps_the_wire(self):
        """Under no scale a root that does not compensate keeps its scale: its plug holds it."""
        clip  = self._chain_without_its_root_s_compensation()
        nodes = Joint.create_hierarchy(clip, parent=Transform.create(name="up", ry=40), world_space=True)
        self.assert_worlds(nodes, clip)
        self.assertEqual(self._input(f"{nodes[1].get_long_name()}.inverseScale"), ["root.scale"])

    def test_the_default_space_takes_nothing_from_the_parent(self):
        """The roots take the clip's values as they are: they hold no inverse scale, and each world
        matrix is the clip's under the parent's."""
        for label, make in (("a transform", Transform), ("a joint", Joint)):
            with self.subTest(label):
                self.new_scene()
                clip  = self._chain()
                up    = make.create(name="up", tx=3, sx=2, sy=2, sz=2)
                nodes = Joint.create_hierarchy(clip, parent=up)
                root  = nodes[0].get_long_name()
                self.assertIsNone(self._input(f"{root}.inverseScale"))
                self.assertEqual(cmds.getAttr(f"{root}.inverseScale")[0], (1, 1, 1))
                self.assert_worlds(nodes, clip, clip.frames.world_matrix @ np.reshape(cmds.getAttr("up.wm[0]"), (4, 4)))

    def test_one_undo_takes_them_out_and_redo_brings_them_back(self):
        clip = self._chain()
        grp  = Transform.create(name="up", sx=2, sy=2, sz=2)
        cmds.flushUndo()
        before = self.scene_state()
        nodes  = Joint.create_hierarchy(clip, parent=grp, world_space=True)
        after  = self.scene_state()
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(), before)
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(), after)
        self.assert_worlds(nodes, clip)
        np.testing.assert_allclose(cmds.getAttr(f"{nodes[0].get_long_name()}.inverseScale")[0], (2, 2, 2), atol=1e-12)

    def test_a_plain_hierarchy_wires_nothing(self):
        """guard: the page's gotcha stands for one pose, for a rest pose, and for a clip that holds
        no frames."""
        root = Joint.create(name="root")
        Joint.create(name="tip", parent=root, tx=1)
        data = root.serialize_hierarchy()
        cmds.delete("root")
        for label, pose in (("plain", data), ("no frames", ClipData(data))):
            with self.subTest(label):
                self.new_scene()
                up    = Transform.create(name="up", sx=2, sy=2, sz=2)
                nodes = Joint.create_hierarchy(pose, parent=up, world_space=True)
                self.assertIsNone(self._input(f"{nodes[1].get_long_name()}.inverseScale"))
                self.assertEqual(cmds.getAttr(f"{nodes[0].get_long_name()}.inverseScale")[0], (1, 1, 1))


# ---------------------------------------------------------------------------------------------
class TestTheClipIsLeftAsItWas(_Case):
    """The build works on a copy: the clip's blocks, frame and names are as they were."""

    def test_after_a_build_under_a_parent(self):
        clip       = _clip()
        clip.frame = 3
        channels   = ("scale", "rotate", "translate", "rotate_axis", "joint_orient")
        blocks     = {channel: np.array(getattr(clip.frames, channel)) for channel in channels}
        Transform.create_hierarchy(clip, parent=Transform.create(name="grp1"), world_space=True)
        self.assertEqual(len(_curves()), len(_moving_curves()))
        self.assertEqual(clip.frame,     3)
        self.assertEqual(clip.name,      _NAMES)
        self.assertIsNone(clip[0].parent_uuid)
        for channel in channels:
            np.testing.assert_array_equal(getattr(clip.frames, channel), blocks[channel], channel)


# ---------------------------------------------------------------------------------------------
class TestWithoutKeys(_Case):
    """A plain HierarchyData builds a clip without keys."""

    def test_a_locator_under_a_non_uniform_scale_takes_no_shear(self):
        """A plain build too: it goes under its parent as it is, so no solve of cmds.parent's
        leaves a shear that no value sets back (a turned parent over a non-uniform scale)."""
        root = _skeleton()
        data = root.serialize_hierarchy()
        cmds.delete("root")
        Transform.create_hierarchy(data, parent=Transform.create(name="up", sx=2, sy=1, sz=3, ry=40, rz=30))
        self.assertEqual(cmds.getAttr("probe.shear")[0], (0.0, 0.0, 0.0))
        self.assertEqual(cmds.getAttr("probe.t")[0], (0.0, 1.0, 1.0))

    @unittest.skipUnless(hasattr(ClipData, "rest_pose"), "a ClipData of this cgmath holds no rest pose")
    def test_the_rest_pose(self):
        """guard: a plain HierarchyData, as ever."""
        root = _skeleton()
        rest = root.serialize_hierarchy()
        clip = ClipData(rest, frames=3)
        clip.frames["root"].translate_x = [1, 2, 3]
        clip.rest_pose                  = rest
        cmds.delete("root")
        nodes = Transform.create_hierarchy(clip.rest_pose)
        self.assertEqual(_curves(), [])
        self.assertEqual(nodes[0].tx.get(), 1.0)

    def test_one_frame_of_the_clip(self):
        """guard: a cut of one frame is a one-frame clip."""
        clip                         = _two_transforms(frames=3)
        clip.frames["a"].translate_x = [1, 2, 3]
        nodes                        = Transform.create_hierarchy(clip.frames[1:2])
        self.assertEqual(_curves(), [])
        self.assertEqual(nodes[0].tx.get(), 2.0)


# ---------------------------------------------------------------------------------------------
class TestUnderANonUniformScale(_Case):
    """A locator of a clip goes under its parent as it is: its values are its local values."""

    def test_it_takes_no_shear_in_either_space(self):
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                self.new_scene()
                clip = _clip()
                up   = Transform.create(name="up", sx=2, sy=1, sz=3, ry=40, rz=30)
                Transform.create_hierarchy(clip, parent=up, world_space=world_space)
                self.assertEqual(cmds.getAttr("probe.shear")[0], (0.0, 0.0, 0.0))

    def test_its_world_matrix_is_the_clip_s(self):
        clip  = _clip()
        up    = Transform.create(name="up", sx=2, sy=1, sz=3, ry=40, rz=30)
        nodes = Transform.create_hierarchy(clip, parent=up, world_space=True)
        probe = [node.get_long_name().endswith("|probe") for node in nodes].index(True)
        self.assert_worlds(nodes[probe:probe + 1], clip, clip.frames.world_matrix[:, probe:probe + 1])


# ---------------------------------------------------------------------------------------------
class TestSkippedNodes(_Case):
    """A node that cannot be built leaves with its column: the others' keys stay their own."""

    def test_the_keys_stay_lined_up(self):
        top               = _two_transforms(frames=3)
        data              = HierarchyData([node.copy() for node in top])
        data[1].node_type = "parentConstraint"
        data.append(Transform.create(name="c").serialize())
        cmds.delete("c")
        clip                         = ClipData(data, frames=3, start_frame=1)
        clip.frames["a"].translate_x = [1, 2, 3]
        clip.frames["b"].translate_x = [4, 5, 6]
        clip.frames["c"].translate_x = [7, 8, 9]
        with self.assertLogs("rig.nodetypes.transform", level="WARNING"):
            Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), ["a_translateX", "c_translateX"])
        self.assertEqual(cmds.keyframe("c_translateX", query=True, valueChange=True), [7, 8, 9])

    def test_a_clip_with_no_node_left(self):
        """guard: every node skipped, nothing to key."""
        data              = Transform.create(name="a").serialize_hierarchy()
        data[0].node_type = "parentConstraint"
        clip              = ClipData(data, frames=3)
        cmds.delete("a")
        with self.assertLogs("rig.nodetypes.transform", level="WARNING"):
            self.assertEqual(len(Transform.create_hierarchy(clip)), 0)


# ---------------------------------------------------------------------------------------------
class TestUnits(_Case):
    """Keys read the clip's values as a plain value does: in the scene's units."""

    def test_a_scene_in_millimetres(self):
        self.addCleanup(cmds.currentUnit, linear="cm")
        cmds.currentUnit(linear="mm")
        clip                         = _two_transforms(frames=3)
        clip.frames["a"].translate_x = [10, 20, 30]
        clip.frames["a"].translate_y = [5, 5, 5]
        Transform.create_hierarchy(clip)
        self.assertEqual(_curves(), ["a_translateX"])
        self.assertAlmostEqual(cmds.getAttr("a.tx", time=2), 20.0, places=9)
        self.assertAlmostEqual(cmds.getAttr("a.ty"), 5.0, places=9)


# ---------------------------------------------------------------------------------------------
class TestWorldSpaceTakesABool(_Case):
    """``create_hierarchy(world_space=)`` takes a bool, through `_files.space_flag`."""

    def test_anything_else_is_a_type_error_with_nothing_made(self):
        data = Transform.create(name="a").serialize_hierarchy()
        cmds.delete("a")
        nodes = sorted(cmds.ls())
        for cls in (Transform, Joint):
            for value in (None, 1, 0, "world"):
                with self.subTest(cls=cls.__name__, value=value):
                    with self.assertRaisesRegex(
                        TypeError,
                        rf"^{cls.__name__}\.create_hierarchy: world_space= takes True or False \(how the roots' "
                        rf"values read under parent: local, or world\), not {value!r}$",
                    ):
                        cls.create_hierarchy(data, world_space=value)
                    self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(Transform.create_hierarchy(data, world_space=np.True_)[0].get_short_name(), "a")
