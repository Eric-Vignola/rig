"""Transform.serialize_hierarchy(bindpose=False | True | "name" | node): a
hierarchy serialized at its bind pose (a dagPose), read only.

The fixture is the design's probe 3 skeleton (runs/features/bindpose):

* ``grp``: rotated, scale 2, a transform member; Maya inserts ``transform1``
  between it and ``root`` when the joint is parented under the scaled group;
* ``root``: oriented (joint orient), ``spine``: rotate axis (10, 20, 30), rotate
  order xzy, rotate 370 degrees, scale (1, 1.5, 1); ``chest``: segment scale
  compensate, joint orient (12, -33, 170), radius 2.5, a user attr; ``head``:
  segment scale compensate off;
* non-members: the end joint ``head_end`` (bound with tsb=True), the locator
  ``probe`` and the transform ``offset`` under ``chest``.

A TransformData's ``==`` is uuid equality, so results are compared field by
field and by cgmath's world matrices.
"""

import inspect
import time
import unittest

import numpy as np
from maya import cmds
from rig.nodetypes import AmbiguousNodeError, Node, NodeNotFoundError, NodeTypeError, Transform
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, scene_connections, scene_nodes

FIELDS = (
    "scale", "rotate", "translate", "rotate_order", "rotate_axis", "joint_orient",
    "segment_scale_compensate", "radius", "visibility", "node_type", "name", "uuid",
    "parent_uuid", "user_defined_attributes",
)
CHANNELS = ("scale", "rotate", "translate", "rotate_axis", "joint_orient")
MEMBERS = ("grp", "transform1", "root", "spine", "chest", "head")


def build(chest_ssc=1):
    """The probe 3 skeleton, bound to ``body`` (bindPose1)."""
    cmds.file(new=True, force=True)
    cmds.createNode("transform", name="grp")
    cmds.setAttr("grp.t", 1, 2, 3)
    cmds.setAttr("grp.s", 2, 2, 2)
    cmds.setAttr("grp.ry", 15)
    cmds.select(clear=True)
    cmds.joint(name="root", p=(0, 0, 0))
    cmds.joint(name="spine", p=(0, 5, 0))
    cmds.joint(name="chest", p=(0, 10, 0))
    cmds.joint(name="head", p=(0, 14, 0))
    cmds.joint(name="head_end", p=(0, 16, 0))
    cmds.parent("root", "grp")
    cmds.joint("root", edit=True, oj="xyz", sao="yup", ch=True, zso=True)
    cmds.setAttr("spine.ra", 10, 20, 30)
    cmds.setAttr("spine.ro", 3)
    cmds.setAttr("spine.r", 15, 25, 370)
    cmds.setAttr("spine.s", 1, 1.5, 1)
    cmds.setAttr("chest.ssc", chest_ssc)
    cmds.setAttr("head.ssc", 0)
    cmds.setAttr("chest.jo", 12, -33, 170)
    cmds.setAttr("chest.radius", 2.5)
    cmds.addAttr("chest", ln="weight", at="double", k=True)
    cmds.setAttr("chest.weight", 0.25)
    loc = cmds.spaceLocator(name="probe")[0]
    cmds.parent(loc, "chest", relative=True)
    cmds.setAttr(loc + ".t", 1, 0, 0)
    cmds.createNode("transform", name="offset", parent="chest")
    cmds.setAttr("offset.t", 0, 1, 0)
    cmds.setAttr("offset.r", 5, 6, 7)
    for name in ("body", "hat", "cape"):
        cmds.polyCube(name=name)
    cmds.skinCluster(["root", "spine", "chest", "head"], "body", tsb=True, name="bodySkin")


def pose_away():
    """Every member moved away from the bind pose."""
    cmds.setAttr("grp.t", 9, 9, 9)
    cmds.setAttr("grp.r", 30, 40, 50)
    cmds.setAttr("root.r", 11, 22, 33)
    cmds.setAttr("root.t", 1, 1, 1)
    cmds.setAttr("spine.r", -40, 80, 200)
    cmds.setAttr("spine.s", 2, 0.5, 3)
    cmds.setAttr("spine.ra", 0, 0, 0)
    cmds.setAttr("chest.jo", 0, 0, 0)
    cmds.setAttr("chest.r", 1, 2, 3)
    cmds.setAttr("head.t", 0, 7, 0)
    cmds.setAttr("head.s", 3, 3, 3)


def pose_non_members():
    cmds.setAttr("head_end.t", 0, 4, 1)
    cmds.setAttr("offset.r", 50, 60, 70)
    cmds.setAttr("probe.t", 2, 3, 4)


def restored(root, pose="bindPose1"):
    """The oracle: Maya's Go to Bind Pose (dagPose -restore -global), a
    serialize, then the restore undone."""
    state = cmds.undoInfo(query=True, state=True)
    cmds.undoInfo(state=True)
    cmds.undoInfo(openChunk=True)
    try:
        cmds.dagPose(pose, restore=True, g=True)
        data = Node(root).serialize_hierarchy()
    finally:
        cmds.undoInfo(closeChunk=True)
        cmds.undo()
        cmds.undoInfo(state=state)
    return data


def by_name(data):
    return {n.name: n for n in data}


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assert_same(self, a, b, atol=1e-9, fields=FIELDS, skip=(), msg=""):
        """Two HierarchyData with the same nodes, the same field values (arrays
        within ``atol``) and the same cgmath world matrices (1e-6)."""
        self.assertEqual([n.name for n in a], [n.name for n in b], msg)
        for x, y in zip(a, b):
            if x.name in skip:
                continue
            for field in fields:
                vx, vy = getattr(x, field), getattr(y, field)
                if isinstance(vx, np.ndarray) or isinstance(vy, np.ndarray):
                    self.assertTrue(
                        np.allclose(vx, vy, atol=atol, rtol=0),
                        f"{msg} {x.name}.{field}: {np.asarray(vx).tolist()} != {np.asarray(vy).tolist()}",
                    )
                else:
                    self.assertEqual(vx, vy, f"{msg} {x.name}.{field}")
            self.assertTrue(
                np.allclose(x.world_matrix, y.world_matrix, atol=1e-6),
                f"{msg} {x.name}: world matrices differ",
            )

    def assert_bit_identical(self, a, b, msg=""):
        self.assertEqual([n.name for n in a], [n.name for n in b], msg)
        for x, y in zip(a, b):
            for field in FIELDS:
                vx, vy = getattr(x, field), getattr(y, field)
                if isinstance(vx, np.ndarray):
                    self.assertTrue(np.array_equal(vx, vy), f"{msg} {x.name}.{field}: {vx!r} != {vy!r}")
                else:
                    self.assertEqual(vx, vy, f"{msg} {x.name}.{field}")


class TestAtTheBindPose(_Case):
    """A == B: serialized at the bind pose, then posed away and read with
    bindpose=True or the pose's name; C == B: Maya's restore oracle."""

    def test_bind_equals_serialize_at_bind(self):
        for root in ("root", "grp"):
            for bindpose in (True, "bindPose1"):
                with self.subTest(root=root, bindpose=bindpose):
                    build()
                    a = Node(root).serialize_hierarchy()
                    pose_away()
                    b = Node(root).serialize_hierarchy(bindpose=bindpose)
                    self.assert_same(a, b)

    def test_the_posed_scene_differs(self):
        # the comparison is not vacuous: every member moved
        build()
        a = by_name(Node("grp").serialize_hierarchy())
        pose_away()
        d = by_name(Node("grp").serialize_hierarchy())
        for name in ("grp", "root", "spine", "chest", "head"):
            with self.subTest(name=name):
                self.assertFalse(np.allclose(a[name].world_matrix, d[name].world_matrix, atol=1e-6))

    def test_matches_restore_global(self):
        for root in ("root", "grp"):
            with self.subTest(root=root):
                build()
                pose_away()
                pose_non_members()
                b = Node(root).serialize_hierarchy(bindpose=True)
                c = restored(root)
                self.assert_same(c, b, atol=1e-6)

    def test_a_rotate_order_changed_after_the_bind(self):
        # the node's current order is kept and the pose's rotation re-expressed
        # in it, as dagPose -restore does: A and B differ in rotate_order and the
        # rotate numbers only, and C == B
        build()
        a = Node("grp").serialize_hierarchy()
        pose_away()
        cmds.setAttr("chest.ro", 5)
        cmds.setAttr("spine.ro", 1)
        b = Node("grp").serialize_hierarchy(bindpose=True)
        fields = tuple(f for f in FIELDS if f not in ("rotate_order", "rotate"))
        self.assert_same(a, b, atol=1e-6, fields=fields)
        bb = by_name(b)
        self.assertEqual((bb["chest"].rotate_order, bb["spine"].rotate_order), (5, 1))
        self.assertFalse(np.allclose(by_name(a)["spine"].rotate, bb["spine"].rotate, atol=1e-3))
        self.assert_same(restored("grp"), b, atol=1e-6)

    def test_at_bind_bit_identical(self):
        build()
        for root in ("root", "grp"):
            for bindpose in (True, "bindPose1", Node("bindPose1")):
                with self.subTest(root=root, bindpose=str(bindpose)):
                    self.assert_bit_identical(
                        Node(root).serialize_hierarchy(),
                        Node(root).serialize_hierarchy(bindpose=bindpose),
                    )

    def test_non_members_keep_scene_values(self):
        build()
        pose_away()
        pose_non_members()
        d = by_name(Node("grp").serialize_hierarchy())
        b = by_name(Node("grp").serialize_hierarchy(bindpose=True))
        for name in ("head_end", "probe", "offset"):
            with self.subTest(name=name):
                for field in FIELDS:
                    if field in ("uuid", "parent_uuid", "name", "user_defined_attributes",
                                 "node_type", "rotate_order", "segment_scale_compensate",
                                 "radius", "visibility"):
                        self.assertEqual(getattr(b[name], field), getattr(d[name], field))
                    else:
                        self.assertTrue(np.array_equal(getattr(b[name], field), getattr(d[name], field)))
        self.assertEqual(b["probe"].node_type, "locator")
        # what the scene holds and a pose does not: the scene's
        self.assertEqual(b["chest"].radius, 2.5)
        self.assertEqual(b["chest"].user_defined_attributes, d["chest"].user_defined_attributes)
        self.assertTrue(b["chest"].segment_scale_compensate)
        self.assertFalse(b["head"].segment_scale_compensate)

    def test_scene_values_a_pose_does_not_hold_come_from_the_scene(self):
        build()
        a = by_name(Node("grp").serialize_hierarchy())
        cmds.setAttr("chest.radius", 7)
        cmds.setAttr("chest.weight", 0.75)
        cmds.setAttr("head.v", 0)
        cmds.setAttr("chest.ssc", 0)
        b = by_name(Node("grp").serialize_hierarchy(bindpose=True))
        self.assertEqual(b["chest"].radius, 7)
        self.assertEqual(b["chest"].user_defined_attributes["weight"]["value"], 0.75)
        self.assertFalse(b["head"].visibility)
        self.assertFalse(b["chest"].segment_scale_compensate)
        for name in MEMBERS:
            for field in CHANNELS:
                self.assertTrue(np.allclose(getattr(a[name], field), getattr(b[name], field), atol=1e-9))

    def test_a_joint_and_a_transform(self):
        # on Transform, so a Joint has it too
        build()
        a = {"root": Node("root").serialize_hierarchy(), "grp": Node("grp").serialize_hierarchy()}
        pose_away()
        self.assertEqual(type(Node("root")).__name__, "Joint")
        self.assert_same(a["root"], Node("root").serialize_hierarchy(bindpose=True))
        self.assert_same(a["grp"], Transform("grp").serialize_hierarchy(bindpose=True))

    def test_serialize_takes_no_bindpose(self):
        # one node's serialize() is unchanged; serialize_hierarchy covers it
        self.assertEqual(list(inspect.signature(Transform.serialize).parameters), ["self"])
        params = inspect.signature(Transform.serialize_hierarchy).parameters
        self.assertIs(params["bindpose"].default, False)


class TestReadOnly(UndoWalk, _Case):
    def test_read_only(self):
        build()
        pose_away()
        cmds.select(["spine", "body"])
        cmds.flushUndo()
        cmds.setAttr("head.tx", 1.5)  # one entry in the queue
        name     = self.undo_name()
        nodes    = scene_nodes()
        wires    = scene_connections()
        values   = {p: cmds.getAttr(p) for p in ("grp.t", "spine.r", "spine.ra", "chest.jo", "head.s")}
        selected = cmds.ls(selection=True, long=True)
        for bindpose in (True, "bindPose1", Node("bindPose1")):
            with self.subTest(bindpose=str(bindpose)):
                Node("grp").serialize_hierarchy(bindpose=bindpose)
                self.assertEqual(self.undo_name(), name)
                self.assertTrue(cmds.undoInfo(query=True, redoQueueEmpty=True))
                self.assertEqual(cmds.ls(selection=True, long=True), selected)
                self.assertEqual(scene_nodes(), nodes)
                self.assertEqual(scene_connections(), wires)
                self.assertEqual({p: cmds.getAttr(p) for p in values}, values)
        # the one entry is the setAttr: nothing was added after it
        cmds.undo()
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_an_error_is_read_only_too(self):
        build()
        cmds.flushUndo()
        cmds.parent("head", "spine")
        cmds.flushUndo()
        nodes = scene_nodes()
        with self.assertRaises(ValueError):
            Node("root").serialize_hierarchy(bindpose=True)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertEqual(scene_nodes(), nodes)


class TestPoses(_Case):
    def test_named_pose_and_node(self):
        build()
        a = Node("grp").serialize_hierarchy()
        pose_away()
        for bindpose in ("bindPose1", Node("bindPose1")):
            with self.subTest(bindpose=str(bindpose)):
                self.assert_same(a, Node("grp").serialize_hierarchy(bindpose=bindpose))

    def test_namespaced_pose(self):
        build()
        a = Node("grp").serialize_hierarchy()
        pose_away()
        cmds.namespace(add="char")
        cmds.rename("bindPose1", "char:bindPose1")
        self.assert_same(a, Node("grp").serialize_hierarchy(bindpose="char:bindPose1"))
        self.assert_same(a, Node("grp").serialize_hierarchy(bindpose=True))
        with self.assertRaises(NodeNotFoundError) as caught:
            Node("grp").serialize_hierarchy(bindpose="bindPose1")
        self.assertIn("no dagPose named 'bindPose1'", str(caught.exception))
        # a bare name the root and the current namespace both have is ambiguous
        cmds.dagPose("grp", save=True, name="bindPose1")
        cmds.namespace(set="char")
        try:
            with self.assertRaises(AmbiguousNodeError):
                Node("grp").serialize_hierarchy(bindpose="bindPose1")
            self.assert_same(a, Node(":grp").serialize_hierarchy(bindpose=":char:bindPose1"))
        finally:
            cmds.namespace(set=":")

    def test_a_saved_tpose(self):
        # any dagPose can be named; bindpose=True reads the bind poses only
        build()
        a = Node("grp").serialize_hierarchy()
        cmds.setAttr("spine.r", 0, 0, 90)
        cmds.setAttr("root.t", 0, 3, 0)
        t = Node("grp").serialize_hierarchy()
        cmds.select(list(MEMBERS) + ["head_end"])
        pose = cmds.dagPose(save=True, selection=True, name="tPose")
        self.assertFalse(cmds.getAttr(pose + ".bindPose"))
        pose_away()
        pose_non_members()
        self.assert_same(t, Node("grp").serialize_hierarchy(bindpose="tPose"),
                         skip=("head_end", "probe", "offset"))
        self.assert_same(a, Node("grp").serialize_hierarchy(bindpose=True),
                         skip=("head_end", "probe", "offset"))
        self.assert_same(restored("grp", "tPose"), Node("grp").serialize_hierarchy(bindpose="tPose"),
                         atol=1e-6)

    def test_agreeing_poses_merge(self):
        # another influence set at the same pose makes a second bind pose that
        # holds the shared ancestors with the same values
        build()
        cmds.skinCluster(["head_end"], "hat", tsb=True)
        self.assertEqual(sorted(cmds.ls(type="dagPose")), ["bindPose1", "bindPose2"])
        a = Node("grp").serialize_hierarchy()
        pose_away()
        cmds.setAttr("head_end.t", 0, 5, 5)
        b = Node("grp").serialize_hierarchy(bindpose=True)
        self.assert_same(a, b, skip=("probe", "offset"))
        # head_end is in bindPose2 only: True takes it from there
        self.assertTrue(np.allclose(by_name(b)["head_end"].translate, by_name(a)["head_end"].translate))
        # bindPose1 alone leaves it at the scene's values
        b1 = by_name(Node("grp").serialize_hierarchy(bindpose="bindPose1"))
        self.assertTrue(np.allclose(b1["head_end"].translate, (0, 5, 5)))

    def test_disagreeing_poses_raise(self):
        build()
        a = Node("grp").serialize_hierarchy()
        pose_away()
        cmds.skinCluster(["root", "spine"], "cape", tsb=True)  # bound while posed: bindPose2
        with self.assertRaises(ValueError) as caught:
            Node("root").serialize_hierarchy(bindpose=True)
        self.assertEqual(
            str(caught.exception),
            "'root' is in bind poses that disagree (bindPose1, bindPose2): 'root' differs "
            "between 'bindPose1' and 'bindPose2'; name one: bindpose='bindPose1'",
        )
        # each named pose still works
        self.assert_same(a, Node("grp").serialize_hierarchy(bindpose="bindPose1"))
        posed = by_name(Node("grp").serialize_hierarchy())
        b2 = by_name(Node("grp").serialize_hierarchy(bindpose="bindPose2"))
        for name in ("root", "spine"):
            for field in CHANNELS:
                self.assertTrue(np.allclose(getattr(b2[name], field), getattr(posed[name], field), atol=1e-6))

    def test_deleted_member_hole(self):
        build()
        cmds.skinCluster("bodySkin", edit=True, ri="head")
        a = Node("root").serialize_hierarchy()
        cmds.delete("head")
        holes = [bool(cmds.listConnections(f"bindPose1.members[{i}]", s=True, d=False))
                 for i in cmds.getAttr("bindPose1.members", mi=True)]
        self.assertIn(False, holes)
        cmds.setAttr("root.r", 11, 22, 33)
        cmds.setAttr("chest.jo", 0, 0, 0)
        b = Node("root").serialize_hierarchy(bindpose=True)
        self.assert_same([n for n in a if n.name not in ("head", "head_end")], b)


class TestErrors(_Case):
    def setUp(self):
        super().setUp()
        build()

    def test_the_argument_type(self):
        for value in (None, 1, 0, 2.5, ["bindPose1"], Node("grp").t):
            with self.subTest(value=str(value)):
                with self.assertRaises(TypeError) as caught:
                    Node("grp").serialize_hierarchy(bindpose=value)
                self.assertEqual(
                    str(caught.exception),
                    f"bindpose= takes False, True or a dagPose name, not {value!r}",
                )
                self.assertNotIsInstance(caught.exception, NodeTypeError)

    def test_the_argument_is_checked_before_the_walk(self):
        calls = []
        original = Transform.serialize
        Transform.serialize = lambda self: calls.append(self) or original(self)
        try:
            with self.assertRaises(TypeError):
                Node("grp").serialize_hierarchy(bindpose=None)
        finally:
            Transform.serialize = original
        self.assertEqual(calls, [])

    def test_not_found(self):
        with self.assertRaises(NodeNotFoundError) as caught:
            Node("grp").serialize_hierarchy(bindpose="nope")
        self.assertIn("no dagPose named 'nope'", str(caught.exception))

    def test_not_a_dagpose(self):
        for value, kind in (("grp", "transform"), (Node("root"), "joint"), ("bodySkin", "skinCluster")):
            with self.subTest(value=str(value)):
                with self.assertRaises(NodeTypeError) as caught:
                    Node("grp").serialize_hierarchy(bindpose=value)
                self.assertEqual(str(caught.exception), f"'{value}' is a {kind}, not a dagPose")

    def test_no_bind_pose(self):
        cmds.select(clear=True)
        cmds.joint(name="lone")
        with self.assertRaises(ValueError) as caught:
            Node("lone").serialize_hierarchy(bindpose=True)
        self.assertEqual(
            str(caught.exception),
            "'lone' has no bind pose: none of the 1 nodes of its hierarchy is a member of a "
            "dagPose with bindPose on",
        )
        # a saved pose that is no bind pose is not one
        cmds.dagPose("lone", save=True, name="tPose")
        with self.assertRaises(ValueError):
            Node("lone").serialize_hierarchy(bindpose=True)
        Node("lone").serialize_hierarchy(bindpose="tPose")

    def test_the_pose_holds_none_of_the_nodes(self):
        cmds.select(clear=True)
        cmds.joint(name="lone")
        with self.assertRaises(ValueError) as caught:
            Node("lone").serialize_hierarchy(bindpose="bindPose1")
        self.assertEqual(
            str(caught.exception),
            "'bindPose1' holds none of the 1 nodes of the hierarchy at 'lone'",
        )

    def test_a_reparented_member(self):
        cmds.parent("head", "chest|offset")
        with self.assertRaises(ValueError) as caught:
            Node("root").serialize_hierarchy(bindpose=True)
        self.assertEqual(
            str(caught.exception),
            "'bindPose1' holds 'head' (posed under 'chest', now under 'offset'): a pose member "
            "is stored relative to the parent it had when the pose was saved, restore the "
            "parent or save a new pose",
        )
        # a non-member reparented is not checked
        build()
        cmds.parent("head_end", "chest")
        Node("root").serialize_hierarchy(bindpose=True)

    def test_a_group_inserted_above_a_member(self):
        cmds.group("root", name="inserted")
        with self.assertRaises(ValueError) as caught:
            Node("grp").serialize_hierarchy(bindpose=True)
        self.assertEqual(
            str(caught.exception),
            "'bindPose1' holds 'root' (posed under 'transform1', now under 'inserted'): a pose "
            "member is stored relative to the parent it had when the pose was saved, restore "
            "the parent or save a new pose",
        )

    def test_a_deleted_pose_parent(self):
        cmds.parent("head", world=True)
        cmds.delete("chest")
        with self.assertRaises(ValueError) as caught:
            Node("head").serialize_hierarchy(bindpose="bindPose1")
        self.assertIn("'head' (posed under a deleted node, now under the world)", str(caught.exception))

    def test_a_member_moved_to_the_world(self):
        cmds.parent("head", world=True)
        with self.assertRaises(ValueError) as caught:
            Node("head").serialize_hierarchy(bindpose=True)
        self.assertIn("'head' (posed under 'chest', now under the world)", str(caught.exception))

    def test_several_moved_members_are_named(self):
        cmds.parent("head", "chest|offset")
        cmds.parent("probe", "root")
        cmds.parent("chest", "root")
        with self.assertRaises(ValueError) as caught:
            Node("grp").serialize_hierarchy(bindpose=True)
        message = str(caught.exception)
        self.assertTrue(message.startswith("'bindPose1' holds "), message)
        # (Maya inserts a transform above a joint parented under a scaled one)
        self.assertIn("'chest' (posed under 'spine', now under '", message)
        self.assertIn("'head' (posed under 'chest', now under 'offset')", message)
        self.assertNotIn("probe", message)  # not a member


def _mirrored(axis, ssc):
    cmds.file(new=True, force=True)
    cmds.select(clear=True)
    cmds.joint(name="a", p=(0, 0, 0))
    cmds.joint(name="b", p=(3, 0, 0))
    cmds.joint(name="c", p=(6, 1, 0))
    cmds.setAttr(f"a.{axis}", -1)
    cmds.setAttr("b.jo", 20, -35, 110)
    cmds.setAttr("b.ssc", ssc)
    cmds.setAttr("c.jo", 5, 6, 7)
    cmds.polyCube(name="m")
    cmds.skinCluster(["a", "b", "c"], "m", tsb=True)


class TestMirroredParent(_Case):
    """A parent joint at scale -1: the joint orient solved from the composed
    matrix must keep det(JO) > 0 (the inverse parent scale holds the sign)."""

    def test_a_mirrored_parent(self):
        for ssc in (1, 0):
            for axis in ("sx", "sy", "sz"):
                with self.subTest(ssc=ssc, axis=axis):
                    _mirrored(axis, ssc)
                    a = Node("a").serialize_hierarchy()
                    cmds.setAttr("b.jo", 0, 0, 0)
                    cmds.setAttr("b.r", 10, 20, 30)
                    cmds.setAttr("a.r", 3, 4, 5)
                    cmds.setAttr("c.jo", 0, 0, 0)
                    b = Node("a").serialize_hierarchy(bindpose=True)
                    self.assert_same(a, b, atol=1e-6)
                    self.assert_same(restored("a"), b, atol=1e-6)

    def test_a_mirrored_parent_outside_the_hierarchy(self):
        # read from b: its pose parent a is not serialized, the pose still has it
        _mirrored("sx", 1)
        a = Node("b").serialize_hierarchy()
        cmds.setAttr("b.jo", 0, 0, 0)
        cmds.setAttr("a.sx", 1)
        b = Node("b").serialize_hierarchy(bindpose=True)
        self.assert_same(a, b, atol=1e-6)


class TestRoundTrip(_Case):
    NAMES = ("grp", "transform1", "root", "spine", "chest", "head", "head_end", "probe", "offset")

    def _rebuilt(self, data):
        cmds.file(new=True, force=True)
        created = Transform.create_hierarchy(data)
        self.assertEqual([n.split("|")[-1] for n in created], list(self.NAMES))
        return {n: cmds.xform(n, query=True, matrix=True, worldSpace=True) for n in self.NAMES}

    def test_create_hierarchy_round_trip(self):
        # create_hierarchy(serialize_hierarchy(bindpose=True)) rebuilds the
        # skeleton at its bind world matrices. (chest's segment scale
        # compensate is off here: create_hierarchy does not connect a joint's
        # inverseScale, so a compensating joint under a scaled parent comes back
        # elsewhere, from either serialize; see the next test.)
        build(chest_ssc=0)
        world = {n: cmds.xform(n, query=True, matrix=True, worldSpace=True) for n in self.NAMES}
        pose_away()
        rebuilt = self._rebuilt(Node("grp").serialize_hierarchy(bindpose=True))
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(np.allclose(rebuilt[name], world[name], atol=1e-5))

    def test_the_round_trip_equals_the_one_at_bind(self):
        build()
        at_bind = Node("grp").serialize_hierarchy()
        pose_away()
        data = Node("grp").serialize_hierarchy(bindpose=True)
        a, b = self._rebuilt(at_bind), self._rebuilt(data)
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(np.allclose(b[name], a[name], atol=1e-6))


class TestCost(_Case):
    def test_the_bind_pose_read_adds_little(self):
        # the walk's MObjects are reused and each pose is read once: at 400
        # joints the bind-pose read adds well under the prototype's 0.15 s
        # (about 0.04 s on a 0.55 s serialize here)
        cmds.select(clear=True)
        joints = [cmds.joint(name="j0")]
        for i in range(1, 400):
            cmds.select(joints[(i - 1) // 2])
            joints.append(cmds.joint(name=f"j{i}", p=(i % 7, i % 5, i % 3)))
        cmds.polyCube(name="m")
        cmds.skinCluster(joints, "m", tsb=True)
        for j in joints[::3]:
            cmds.setAttr(j + ".r", 5, 6, 7)
        plain, posed = [], []
        root = Node("j0")
        for _ in range(3):
            t0 = time.perf_counter()
            root.serialize_hierarchy()
            t1 = time.perf_counter()
            root.serialize_hierarchy(bindpose=True)
            t2 = time.perf_counter()
            plain.append(t1 - t0)
            posed.append(t2 - t1)
        self.assertLess(min(posed) - min(plain), max(0.1, 0.25 * min(plain)), (plain, posed))


if __name__ == "__main__":
    unittest.main()
