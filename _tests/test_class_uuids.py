"""A node made from data takes the data's uuid (the Renderable round, 1 of 6; the user's decision
of 2026-11-10: "on everywhere but if a target is given, the target supersedes the uuid").

* ``uuids=True``, the default, on ``Transform.create_hierarchy`` and ``load_hierarchy``,
  ``Transform.create(data)`` and ``Joint.create(data)``, and on ``create`` and the class-form
  ``load`` of ``Mesh``, ``NurbsCurve`` and ``NurbsSurface``: every node made takes its data's
  uuid (a recorded ``rename -uuid``), so ``cmds.ls(uuid)`` and ``Node(uuid)`` find it and a
  serialize of it gives the same uuids back.
* Two nodes cannot hold one uuid: a uuid a scene node holds, or one the data holds twice, is a
  ValueError before anything is made, naming both and how to go on (fresh uuids on the data,
  ``hierarchy.renew_uuids()`` or ``data.uuid = generate_uuid()``; or ``uuids=False``).
* ``uuids=False`` leaves every node the uuid Maya gives it: the twin of a live source.
* Data without a uuid (a bare TransformData, a curve's points) leaves Maya's too.
* The flag is a checked bool on every door, before anything is read or made; a node's own
  ``load`` (an update) does not take it.
* One undo step as before: undo frees the uuid, redo brings the node back holding it.

Measured in Maya 2025 first (session 9f0b3bbf, eval/probe_uuid.py): ``rename -uuid`` adopts a
uuid, spells it upper-case, refuses one another node holds ("UUID already in use."), works on a
shape, is undoable and survives a .ma and a .mb.
"""

import os
import shutil
import tempfile

import numpy as np
from cgmath.hierarchy import ClipData, TransformData, generate_uuid
from maya import cmds
from rig import Node
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig.nodetypes import Joint, Mesh, NurbsCurve, NurbsSurface, Transform
from rig.nodetypes import _files

# the text of a refusal, after the method and the record
_HELD      = r" has the uuid [0-9A-F-]{36}, which '%s' holds, and two nodes cannot hold one; for a second copy give the data fresh uuids first \(%s, or a Scene's renew_uuids\(\)\); uuids=False keeps Maya's own$"
_FRESH     = r"data\.uuid = cgmath\.hierarchy\.generate_uuid\(\)"
_RENEW     = r"hierarchy\.renew_uuids\(\)"
_FLAG      = r": uuids= is True \(the default: a node made from data takes the data's uuid\) or False \(Maya's own\), not %s$"
_NOT_BOOLS = (None, 1, "yes")


def _uuid(node):
    """The uuid of the node `node` (a rig node, by its long name)."""
    return cmds.ls(node.get_long_name(), uuid=True)[0]


def _skeleton():
    """root > spine > arm, joints, root > grp, a transform: the data, the scene left empty."""
    root  = Joint.create(name="root")
    spine = Joint.create(name="spine", parent=root)
    spine.t.set(0, 5, 0)
    arm = Joint.create(name="arm", parent=spine)
    arm.t.set(3, 0, 0)
    grp = Transform.create(name="grp", parent=root)
    grp.t.set(-2, 1, 0)
    data = root.serialize_hierarchy()
    cmds.delete("root")
    return data


def _mesh_data(name="src"):
    """A cube's data and UVs, the cube deleted."""
    t = cmds.polyCube(name=name, constructionHistory=False)[0]
    data, uvs = Mesh(t).serialize()
    cmds.delete(t)
    return data, list(uvs)


def _curve_data(name="crv"):
    """A curve's data, the curve deleted."""
    curve = NurbsCurve.create([(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name=name)
    data  = curve.serialize()
    cmds.delete(name)
    return data


def _surface_data(name="ball"):
    """A sphere's data, the sphere deleted."""
    t    = cmds.sphere(name=name, constructionHistory=False)[0]
    data = NurbsSurface(cmds.listRelatives(t, shapes=True)[0]).serialize()
    cmds.delete(t)
    return data


def _shape_data(cls):
    """`cls`'s data (a MeshData, a NurbsCurveData, a NurbsSurfaceData) from a source since deleted."""
    if cls is Mesh:
        return _mesh_data()[0]
    return _curve_data() if cls is NurbsCurve else _surface_data()


CLASSES = (Mesh, NurbsCurve, NurbsSurface)


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def folder(self):
        folder = tempfile.mkdtemp(prefix="rig_uuids_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        return folder

    def assert_nothing_made(self, error, pattern, call):
        before = cmds.ls(long=True)
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(long=True), before)


# ---------------------------------------------------------------------------------------------
class TestHierarchy(_Case):
    """``create_hierarchy`` and ``load_hierarchy``."""

    def test_every_node_takes_its_data_s_uuid(self):
        data = _skeleton()
        made = Transform.create_hierarchy(data)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])
        for node in data:
            self.assertEqual(cmds.ls(node.uuid, long=True), [Node(node.uuid).get_long_name()])
        # the hierarchy's links hold too: a shape serialized under spine names its uuid
        self.assertEqual(Node(data[1].uuid).get_short_name(), "spine")

    def test_serialize_gives_the_same_uuids_back(self):
        data  = _skeleton()
        again = Transform.create_hierarchy(data)[0].serialize_hierarchy()
        self.assertEqual([node.uuid for node in again], [node.uuid for node in data])
        self.assertEqual([node.parent_uuid for node in again], [node.parent_uuid for node in data])

    def test_a_clip_adopts_too(self):
        data = _skeleton()
        clip = ClipData(data, frames=3, start_frame=1)
        clip.frames["arm"].rotate_z = [0, 10, 20]
        made = Transform.create_hierarchy(clip)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in clip])
        self.assertEqual(cmds.ls(type="animCurve"), ["arm_rotateZ"])

    def test_load_hierarchy_adopts(self):
        data = _skeleton()
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        made = Joint.load_hierarchy(path)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])

    def test_a_uuid_a_scene_node_holds_is_refused_before_anything_is_made(self):
        data = _skeleton()
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        Transform.create_hierarchy(data)
        for label, call, head in (
            ("create", lambda: Transform.create_hierarchy(data), r"^Transform\.create_hierarchy: the node 'root'"),
            ("Joint", lambda: Joint.create_hierarchy(data), r"^Joint\.create_hierarchy: the node 'root'"),
            ("load", lambda: Transform.load_hierarchy(path), r"^Transform\.load_hierarchy: the node 'root'"),
            ("under a parent", lambda: Transform.create_hierarchy(data, parent="grp"),
             r"^Transform\.create_hierarchy: the node 'root'"),
        ):
            with self.subTest(label):
                self.assert_nothing_made(ValueError, head + _HELD % (r"\|root", _RENEW), call)

    def test_the_clash_can_be_on_any_node(self):
        # the scene holds a node with arm's uuid alone: the refusal names arm and that node
        data  = _skeleton()
        other = cmds.createNode("transform", name="other")
        cmds.rename(other, data[2].uuid, uuid=True)
        self.assert_nothing_made(
            ValueError, r"^Transform\.create_hierarchy: the node 'arm'" + _HELD % (r"\|other", _RENEW),
            lambda: Transform.create_hierarchy(data),
        )

    def test_renew_uuids_builds_a_second_copy_beside_the_first(self):
        data = _skeleton()
        made = Transform.create_hierarchy(data)
        copy = data.copy()
        copy.renew_uuids()
        again = Transform.create_hierarchy(copy)
        self.assertEqual([_uuid(node) for node in again], [node.uuid for node in copy])
        self.assertTrue(all(new.uuid != old.uuid for new, old in zip(copy, data)))
        self.assertEqual(again[1].get_parent().get_long_name(), again[0].get_long_name())  # the links rewired
        self.assertEqual((len(cmds.ls(type="joint")), len(cmds.ls("grp*"))), (6, 2))
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])     # the first untouched

    def test_false_leaves_maya_s_own(self):
        data = _skeleton()
        made = Transform.create_hierarchy(data)
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        for label, call in (
            ("create", lambda: Transform.create_hierarchy(data, uuids=False)),
            ("load", lambda: Transform.load_hierarchy(path, uuids=False)),
        ):
            with self.subTest(label):
                twin = call()
                self.assertEqual(twin[0].get_short_name().rstrip("0123456789"), "root")  # root1, root2: beside root
                self.assertNotEqual(twin[0].get_short_name(), "root")
                self.assertFalse(set(_uuid(node) for node in twin) & set(node.uuid for node in data))
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])

    def test_the_documented_twin(self):
        # the create_hierarchy docstring's example: a twin of a live hierarchy
        data = _skeleton()
        Transform.create_hierarchy(data)
        live = Node("root").serialize_hierarchy()
        self.assertEqual([node.uuid for node in live], [node.uuid for node in data])
        self.assert_nothing_made(ValueError, r"^Transform\.create_hierarchy: the node 'root' has the uuid",
                                 lambda: Transform.create_hierarchy(live))
        twin = Transform.create_hierarchy(live, uuids=False)
        self.assertEqual(twin[0].get_short_name(), "root1")

    def test_a_lower_case_uuid_is_maya_s_upper_case(self):
        data = _skeleton()
        for node in data:
            node.uuid = node.uuid.lower()
        made = Transform.create_hierarchy(data)
        self.assertEqual([_uuid(node) for node in made], [node.uuid.upper() for node in data])
        self.assert_nothing_made(ValueError, r"^Transform\.create_hierarchy: the node 'root' has the uuid",
                                 lambda: Transform.create_hierarchy(data))

    def test_undo_frees_the_uuid_and_redo_takes_it_back(self):
        data = _skeleton()
        self.undo_queue_on()
        made = Transform.create_hierarchy(data)
        self.assertEqual(self.undo_name(), "rig.Transform.create_hierarchy")
        self.undo_steps(1)
        self.assertEqual([cmds.ls(node.uuid) for node in data], [[]] * len(data))
        self.redo_steps(1)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])
        # and the uuid is free again for a second build once the first is undone
        self.undo_steps(1)
        Transform.create_hierarchy(data)

    def test_the_flag_is_a_checked_bool(self):
        data = _skeleton()
        for value in _NOT_BOOLS:
            with self.subTest(value=value):
                said = _FLAG % __import__("re").escape(repr(value))
                self.assert_nothing_made(TypeError, r"^Transform\.create_hierarchy" + said,
                                         lambda: Transform.create_hierarchy(data, uuids=value))
                # before the file is read: a missing file is not reached
                self.assert_nothing_made(TypeError, r"^Joint\.load_hierarchy" + said,
                                         lambda: Joint.load_hierarchy("no_such_file.npz", uuids=value))
        self.assertEqual(len(Transform.create_hierarchy(data, uuids=np.bool_(True))), len(data))


# ---------------------------------------------------------------------------------------------
class TestTransformCreate(_Case):
    """``Transform.create(data)`` / ``Joint.create(data)``."""

    def test_the_node_takes_the_data_s_uuid(self):
        for cls, node_type in ((Transform, "transform"), (Joint, "joint")):
            with self.subTest(cls=cls.__name__):
                data = TransformData(f"{node_type}_data", node_type=node_type, uuid=generate_uuid())
                made = cls.create(data)
                self.assertEqual(_uuid(made), data.uuid)
                self.assertEqual(repr(Node(data.uuid)), repr(made))

    def test_a_uuid_a_scene_node_holds_is_refused_before_anything_is_made(self):
        data = TransformData("x", uuid=generate_uuid())
        Transform.create(data)
        self.assert_nothing_made(
            ValueError, r"^Transform\.create\(\): the TransformData 'x'" + _HELD % (r"\|x", _FRESH),
            lambda: Transform.create(data, name="again"),
        )
        # a fresh uuid on the data, and it builds beside
        data.uuid = generate_uuid()
        self.assertEqual(_uuid(Transform.create(data, name="again")), data.uuid)

    def test_false_and_data_without_a_uuid_leave_maya_s_own(self):
        data = TransformData("x", uuid=generate_uuid())
        made = Transform.create(data)
        twin = Transform.create(data, name="twin", uuids=False)
        self.assertNotEqual(_uuid(twin), data.uuid)
        self.assertEqual(_uuid(made), data.uuid)
        bare = TransformData("bare")  # no uuid: a node gets one when it joins a hierarchy
        self.assertIsNone(bare.uuid)
        self.assertEqual(Transform.create(bare).get_short_name(), "bare")

    def test_the_documented_twin(self):
        # the cheatsheet's form: a live node's data, built beside it
        cube = Transform.create(name="cube", tx=1)
        data = cube.serialize()
        self.assertEqual(data.uuid, _uuid(cube))
        self.assert_nothing_made(ValueError, r"^Transform\.create\(\): the TransformData 'cube' has the uuid",
                                 lambda: Transform.create(data, name="twin"))
        twin = Transform.create(data, name="twin", tx=2, uuids=False)
        self.assertEqual((twin.get_short_name(), twin.tx.get()), ("twin", 2.0))

    def test_the_flag_is_a_checked_bool_and_no_attribute(self):
        data = TransformData("x", uuid=generate_uuid())
        for value in _NOT_BOOLS:
            with self.subTest(value=value):
                self.assert_nothing_made(TypeError, r"^Transform\.create\(\)" + _FLAG % __import__("re").escape(repr(value)),
                                         lambda: Transform.create(data, uuids=value))
        self.assertFalse(cmds.attributeQuery("uuids", type="transform", exists=True))
        self.assertEqual(_uuid(Joint.create(TransformData("j", node_type="joint", uuid=data.uuid), uuids=True)), data.uuid)

    def test_one_undo_step(self):
        data = TransformData("x", uuid=generate_uuid())
        self.undo_queue_on()
        made = Transform.create(data)
        self.assertEqual(self.undo_name(), "rig.Transform.create")
        self.undo_steps(1)
        self.assertEqual(cmds.ls(data.uuid), [])
        self.redo_steps(1)
        self.assertEqual(_uuid(made), data.uuid)


# ---------------------------------------------------------------------------------------------
class TestShapes(_Case):
    """``create`` and the class-form ``load`` of Mesh, NurbsCurve and NurbsSurface: the shape takes
    the data's uuid, the transform keeps Maya's."""

    def test_the_shape_takes_the_data_s_uuid(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                shape = cls.create(data, name="made")
                self.assertEqual(_uuid(shape), data.uuid)
                self.assertEqual(cmds.ls(data.uuid, long=True), [shape.get_long_name()])
                self.assertNotEqual(_uuid(shape.get_parent()), data.uuid)
                self.assertEqual(repr(Node(data.uuid)), repr(shape))

    def test_under_a_transform_too(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                ctl   = Transform.create(name=f"ctl_{cls.__name__}")
                shape = cls.create(data, transform=ctl)
                self.assertEqual(_uuid(shape), data.uuid)
                self.assertEqual(shape.get_parent().get_long_name(), ctl.get_long_name())

    def test_load_adopts_and_serialize_gives_it_back(self):
        folder = self.folder()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                path  = data.save(os.path.join(folder, f"{cls.__name__}.npz"))
                shape = cls.load(path, name="loaded")
                self.assertEqual(_uuid(shape), data.uuid)
                again = shape.serialize(uvs=False) if cls is Mesh else shape.serialize()
                self.assertEqual(again.uuid, data.uuid)
                self.assertEqual(again.parent_uuid, _uuid(shape.get_parent()))

    def test_a_uuid_a_scene_node_holds_is_refused_before_anything_is_made(self):
        folder = self.folder()
        for cls, what in ((Mesh, "the MeshData 'srcShape'"), (NurbsCurve, "the NurbsCurveData"),
                          (NurbsSurface, "the NurbsSurfaceData")):
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                path  = data.save(os.path.join(folder, f"{cls.__name__}.npz"))
                shape = cls.create(data, name="made")
                held  = _HELD % (__import__("re").escape(shape.get_long_name()), _FRESH)
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.create(data, name="again"))
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.load(path, name="again"))
                ctl = Transform.create(name=f"ctl_{cls.__name__}")
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.create(data, transform=ctl))
                # a fresh uuid on the data, and it builds beside
                data.uuid = generate_uuid()
                self.assertEqual(_uuid(cls.create(data, name="again")), data.uuid)

    def test_false_leaves_maya_s_own(self):
        folder = self.folder()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                path  = data.save(os.path.join(folder, f"{cls.__name__}.npz"))
                shape = cls.create(data, name="made")
                for twin in (cls.create(data, name="twin", uuids=False), cls.load(path, name="twin", uuids=False)):
                    self.assertNotEqual(_uuid(twin), data.uuid)
                self.assertEqual(_uuid(shape), data.uuid)

    def test_points_hold_no_uuid(self):
        curve = NurbsCurve.create([(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name="pts")
        self.assertTrue(cmds.ls(curve.get_long_name(), uuid=True))
        NurbsCurve.create([(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name="more")  # no clash to speak of

    def test_the_flag_is_a_checked_bool_before_the_file_is_read(self):
        for cls in CLASSES:
            data = _shape_data(cls)
            for value in _NOT_BOOLS:
                with self.subTest(cls=cls.__name__, value=value):
                    said = _FLAG % __import__("re").escape(repr(value))
                    self.assert_nothing_made(TypeError, rf"^{cls.__name__}\.create\(\)" + said,
                                             lambda: cls.create(data, uuids=value))
                    self.assert_nothing_made(TypeError, rf"^{cls.__name__}\.load" + said,
                                             lambda: cls.load("no_such_file.npz", uuids=value))

    def test_a_node_s_own_load_takes_no_uuids(self):
        folder = self.folder()
        data, _ = _mesh_data()
        path = data.save(os.path.join(folder, "mesh.npz"))
        mesh = Mesh.create(data, name="made")
        with self.assertRaisesRegex(TypeError, r"takes the files, world_space=, normals=, colors= and hierarchy= \(got uuids=\)"):
            mesh.load(path, uuids=False)
        self.assertEqual(_uuid(mesh), data.uuid)

    def test_one_undo_step(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data = _shape_data(cls)
                self.undo_queue_on()
                shape = cls.create(data, name="made")
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.create")
                self.undo_steps(1)
                self.assertEqual(cmds.ls(data.uuid), [])
                self.redo_steps(1)
                self.assertEqual(_uuid(shape), data.uuid)
                cmds.flushUndo()


# ---------------------------------------------------------------------------------------------
class TestTheCheck(_Case):
    """`_files.free_uuids` and `_files.adopt_uuid` themselves."""

    def test_a_uuid_the_data_holds_twice(self):
        uuid = generate_uuid()
        with self.assertRaisesRegex(
            ValueError,
            r"^X: the node 'a' and the node 'b' hold one uuid, " + uuid + r", which two nodes cannot; give the data "
            r"fresh uuids \(r\(\)\), or uuids=False keeps Maya's own$",
        ):
            _files.free_uuids("X", [("the node 'a'", uuid), ("the node 'b'", uuid)], "r()")

    def test_none_is_not_checked_and_case_does_not_matter(self):
        uuid = generate_uuid()
        _files.free_uuids("X", [("a", None), ("b", None)], "r()")
        node = cmds.createNode("transform", name="held")
        cmds.rename(node, uuid, uuid=True)
        with self.assertRaisesRegex(ValueError, r"^X: the data has the uuid " + uuid + r", which '\|held' holds"):
            _files.free_uuids("X", [("the data", uuid.lower())], "r()")

    def test_adopt_is_a_recorded_rename(self):
        uuid = generate_uuid()
        node = cmds.createNode("transform", name="n")
        self.undo_queue_on()
        _files.adopt_uuid(node, uuid)
        self.assertEqual(cmds.ls(node, uuid=True), [uuid])
        _files.adopt_uuid(node, None)  # nothing
        self.assertEqual(cmds.ls(node, uuid=True), [uuid])
        cmds.undo()
        self.assertEqual(cmds.ls(uuid), [])
