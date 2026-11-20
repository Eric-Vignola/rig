"""A node made from data takes the data's uuid when asked (the Renderable round, 1 of 6; the
user's decisions of 2026-11-10, "if a target is given, the target supersedes the uuid", and of
2026-11-13, "I want uuids to be False by default").

* ``uuids=False``, the default, on ``Transform.create_hierarchy`` and ``load_hierarchy``,
  ``Transform.create(data)`` and ``Joint.create(data)``, and on ``create`` and the class-form
  ``load`` of ``Mesh``, ``NurbsCurve`` and ``NurbsSurface``: every node made keeps the uuid Maya
  gives it, so one data builds any number of copies, and the twin of a live source.
* ``uuids=True``: every node made takes its data's uuid (a recorded ``rename -uuid``), so
  ``cmds.ls(uuid)`` and ``Node(uuid)`` find it and a serialize of it gives the same uuids back.
* Two nodes cannot hold one uuid: under ``uuids=True`` a uuid a scene node holds, or one the
  data holds twice, is a ValueError before anything is made, naming both and how to go on
  (fresh uuids on the data, ``hierarchy.renew_uuids()`` or ``data.uuid = generate_uuid()``; or
  no ``uuids=True``).
* Data without a uuid (a bare TransformData, a curve's points) leaves Maya's too.
* The flag is a checked bool or None on every door, before anything is read or made; a node's
  own ``load`` (an update) does not take it.
* None, the default, is what ``with rig.uuids():`` or ``rig.set_options(uuids=)`` says, False
  unless one does (the user, 2026-11-18): inside the block every create takes its data's uuids,
  and a create given no target finds it by uuid (TestTheBlock).
* One undo step as before: undo frees the uuid, redo brings the node back holding it.

Measured in Maya 2025 first (session 9f0b3bbf, eval/probe_uuid.py): ``rename -uuid`` adopts a
uuid, spells it upper-case, refuses one another node holds ("UUID already in use."), works on a
shape, is undoable and survives a .ma and a .mb.
"""

import os
import re
import shutil
import tempfile

import numpy as np
import rig
from cgmath.hierarchy import ClipData, TransformData, generate_uuid
from maya import cmds
from rig import Node
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk
from rig._tests.test_scene import _scene
from rig.nodetypes import BlendShape, Joint, Mesh, NurbsCurve, NurbsSurface, SkinCluster, Transform
from rig.nodetypes import _files

# the text of a refusal, after the method and the record
_HELD      = r" has the uuid [0-9A-F-]{36}, which '%s' holds, and two nodes cannot hold one; for a second copy give the data fresh uuids first \(%s, or a Scene's renew_uuids\(\)\); uuids=False keeps Maya's own$"
_FRESH     = r"data\.uuid = cgmath\.hierarchy\.generate_uuid\(\)"
_RENEW     = r"hierarchy\.renew_uuids\(\)"
_FLAG      = (r": uuids= is True \(a node made from data takes the data's uuid\), False \(Maya's own\) or None \(the "
              r"default: as rig\.uuids\(\) or rig\.set_options\(uuids=\) says, else False\), not %s$")
_NOT_BOOLS = (1, "yes")   # None is the default: what rig.uuids() says (TestTheBlock)


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
        made = Transform.create_hierarchy(data, uuids=True)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])
        for node in data:
            self.assertEqual(cmds.ls(node.uuid, long=True), [Node(node.uuid).get_long_name()])
        # the hierarchy's links hold too: a shape serialized under spine names its uuid
        self.assertEqual(Node(data[1].uuid).get_short_name(), "spine")

    def test_serialize_gives_the_same_uuids_back(self):
        data  = _skeleton()
        again = Transform.create_hierarchy(data, uuids=True)[0].serialize_hierarchy()
        self.assertEqual([node.uuid for node in again], [node.uuid for node in data])
        self.assertEqual([node.parent_uuid for node in again], [node.parent_uuid for node in data])

    def test_a_clip_adopts_too(self):
        data = _skeleton()
        clip = ClipData(data, frames=3, start_frame=1)
        clip["arm"].rotate_z = [0, 10, 20]
        made = Transform.create_hierarchy(clip, uuids=True)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in clip])
        self.assertEqual(cmds.ls(type="animCurve"), ["arm_rotateZ"])

    def test_load_hierarchy_adopts(self):
        data = _skeleton()
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        made = Joint.load_hierarchy(path, uuids=True)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])

    def test_a_uuid_a_scene_node_holds_is_refused_before_anything_is_made(self):
        data = _skeleton()
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        Transform.create_hierarchy(data, uuids=True)
        for label, call, head in (
            ("create", lambda: Transform.create_hierarchy(data, uuids=True),
             r"^Transform\.create_hierarchy: the node 'root'"),
            ("Joint", lambda: Joint.create_hierarchy(data, uuids=True), r"^Joint\.create_hierarchy: the node 'root'"),
            ("load", lambda: Transform.load_hierarchy(path, uuids=True),
             r"^Transform\.load_hierarchy: the node 'root'"),
            ("under a parent", lambda: Transform.create_hierarchy(data, parent="grp", uuids=True),
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
            lambda: Transform.create_hierarchy(data, uuids=True),
        )

    def test_renew_uuids_builds_a_second_copy_beside_the_first(self):
        data = _skeleton()
        made = Transform.create_hierarchy(data, uuids=True)
        copy = data.copy()
        copy.renew_uuids()
        again = Transform.create_hierarchy(copy, uuids=True)
        self.assertEqual([_uuid(node) for node in again], [node.uuid for node in copy])
        self.assertTrue(all(new.uuid != old.uuid for new, old in zip(copy, data)))
        self.assertEqual(again[1].get_parent().get_long_name(), again[0].get_long_name())  # the links rewired
        self.assertEqual((len(cmds.ls(type="joint")), len(cmds.ls("grp*"))), (6, 2))
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])     # the first untouched

    def test_the_default_leaves_maya_s_own(self):
        data = _skeleton()
        made = Transform.create_hierarchy(data, uuids=True)
        path = data.save(os.path.join(self.folder(), "skel.npz"))
        for label, call in (
            ("create", lambda: Transform.create_hierarchy(data)),
            ("load", lambda: Transform.load_hierarchy(path)),
            ("create, False", lambda: Transform.create_hierarchy(data, uuids=False)),
            ("load, False", lambda: Transform.load_hierarchy(path, uuids=False)),
        ):
            with self.subTest(label):
                twin = call()
                self.assertEqual(twin[0].get_short_name().rstrip("0123456789"), "root")  # root1, root2: beside root
                self.assertNotEqual(twin[0].get_short_name(), "root")
                self.assertFalse(set(_uuid(node) for node in twin) & set(node.uuid for node in data))
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])

    def test_one_data_builds_any_number_of_copies(self):
        # the user, 2026-11-13: "I want uuids to be False by default"
        data = _skeleton()
        made = [Transform.create_hierarchy(data) for _ in range(3)]
        self.assertEqual([nodes[0].get_short_name() for nodes in made], ["root", "root1", "root2"])
        held = [_uuid(node) for nodes in made for node in nodes]
        self.assertEqual(len(set(held)), 3 * len(data))
        self.assertFalse(set(held) & set(node.uuid for node in data))

    def test_the_documented_twin(self):
        # the create_hierarchy docstring's example: a twin of a live hierarchy, with no flag
        data = _skeleton()
        Transform.create_hierarchy(data, uuids=True)
        live = Node("root").serialize_hierarchy()
        self.assertEqual([node.uuid for node in live], [node.uuid for node in data])
        twin = Transform.create_hierarchy(live)
        self.assertEqual(twin[0].get_short_name(), "root1")
        # the live nodes hold the data's uuids: asking for them is the refusal
        self.assert_nothing_made(ValueError, r"^Transform\.create_hierarchy: the node 'root' has the uuid",
                                 lambda: Transform.create_hierarchy(live, uuids=True))

    def test_a_lower_case_uuid_is_maya_s_upper_case(self):
        data = _skeleton()
        for node in data:
            node.uuid = node.uuid.lower()
        made = Transform.create_hierarchy(data, uuids=True)
        self.assertEqual([_uuid(node) for node in made], [node.uuid.upper() for node in data])
        self.assert_nothing_made(ValueError, r"^Transform\.create_hierarchy: the node 'root' has the uuid",
                                 lambda: Transform.create_hierarchy(data, uuids=True))

    def test_undo_frees_the_uuid_and_redo_takes_it_back(self):
        data = _skeleton()
        self.undo_queue_on()
        made = Transform.create_hierarchy(data, uuids=True)
        self.assertEqual(self.undo_name(), "rig.Transform.create_hierarchy")
        self.undo_steps(1)
        self.assertEqual([cmds.ls(node.uuid) for node in data], [[]] * len(data))
        self.redo_steps(1)
        self.assertEqual([_uuid(node) for node in made], [node.uuid for node in data])
        # and the uuid is free again for a second build once the first is undone
        self.undo_steps(1)
        Transform.create_hierarchy(data, uuids=True)

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
                made = cls.create(data, uuids=True)
                self.assertEqual(_uuid(made), data.uuid)
                self.assertEqual(repr(Node(data.uuid)), repr(made))

    def test_a_uuid_a_scene_node_holds_is_refused_before_anything_is_made(self):
        data = TransformData("x", uuid=generate_uuid())
        Transform.create(data, uuids=True)
        self.assert_nothing_made(
            ValueError, r"^Transform\.create\(\): the TransformData 'x'" + _HELD % (r"\|x", _FRESH),
            lambda: Transform.create(data, name="again", uuids=True),
        )
        # a fresh uuid on the data, and it builds beside
        data.uuid = generate_uuid()
        self.assertEqual(_uuid(Transform.create(data, name="again", uuids=True)), data.uuid)

    def test_the_default_and_data_without_a_uuid_leave_maya_s_own(self):
        data  = TransformData("x", uuid=generate_uuid())
        made  = Transform.create(data, uuids=True)
        twins = [Transform.create(data, name="twin"), Transform.create(data, name="twin", uuids=False)]
        self.assertNotIn(data.uuid, [_uuid(twin) for twin in twins])
        self.assertEqual(len(set(_uuid(twin) for twin in twins)), 2)
        self.assertEqual(_uuid(made), data.uuid)
        bare = TransformData("bare")  # no uuid: a node gets one when it joins a hierarchy
        self.assertIsNone(bare.uuid)
        self.assertEqual(Transform.create(bare, uuids=True).get_short_name(), "bare")

    def test_the_documented_twin(self):
        # the cheatsheet's form: a live node's data, built beside it with no flag
        cube = Transform.create(name="cube", tx=1)
        data = cube.serialize()
        self.assertEqual(data.uuid, _uuid(cube))
        twin = Transform.create(data, name="twin", tx=2)
        self.assertEqual((twin.get_short_name(), twin.tx.get()), ("twin", 2.0))
        self.assertNotEqual(_uuid(twin), data.uuid)
        # the live node holds the data's uuid: asking for it is the refusal
        self.assert_nothing_made(ValueError, r"^Transform\.create\(\): the TransformData 'cube' has the uuid",
                                 lambda: Transform.create(data, name="held", uuids=True))

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
        made = Transform.create(data, uuids=True)
        self.assertEqual(self.undo_name(), "rig.Transform.create")
        self.undo_steps(1)
        self.assertEqual(cmds.ls(data.uuid), [])
        self.redo_steps(1)
        self.assertEqual(_uuid(made), data.uuid)


# ---------------------------------------------------------------------------------------------
class TestShapes(_Case):
    """``create`` and the class-form ``load`` of Mesh, NurbsCurve and NurbsSurface: with
    ``uuids=True`` the shape takes the data's uuid, the transform keeps Maya's."""

    def test_the_shape_takes_the_data_s_uuid(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                shape = cls.create(data, name="made", uuids=True)
                self.assertEqual(_uuid(shape), data.uuid)
                self.assertEqual(cmds.ls(data.uuid, long=True), [shape.get_long_name()])
                self.assertNotEqual(_uuid(shape.get_parent()), data.uuid)
                self.assertEqual(repr(Node(data.uuid)), repr(shape))

    def test_under_a_transform_too(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                ctl   = Transform.create(name=f"ctl_{cls.__name__}")
                shape = cls.create(data, transform=ctl, uuids=True)
                self.assertEqual(_uuid(shape), data.uuid)
                self.assertEqual(shape.get_parent().get_long_name(), ctl.get_long_name())

    def test_load_adopts_and_serialize_gives_it_back(self):
        folder = self.folder()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                path  = data.save(os.path.join(folder, f"{cls.__name__}.npz"))
                shape = cls.load(path, name="loaded", uuids=True)
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
                shape = cls.create(data, name="made", uuids=True)
                held  = _HELD % (__import__("re").escape(shape.get_long_name()), _FRESH)
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.create(data, name="again", uuids=True))
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.load(path, name="again", uuids=True))
                ctl = Transform.create(name=f"ctl_{cls.__name__}")
                self.assert_nothing_made(ValueError, rf"^{cls.__name__}\.create\(\): {what}" + held,
                                         lambda: cls.create(data, transform=ctl, uuids=True))
                # a fresh uuid on the data, and it builds beside
                data.uuid = generate_uuid()
                self.assertEqual(_uuid(cls.create(data, name="again", uuids=True)), data.uuid)

    def test_the_default_leaves_maya_s_own(self):
        folder = self.folder()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data  = _shape_data(cls)
                path  = data.save(os.path.join(folder, f"{cls.__name__}.npz"))
                shape = cls.create(data, name="made", uuids=True)
                for twin in (cls.create(data, name="twin"), cls.load(path, name="twin"),
                             cls.create(data, name="twin", uuids=False), cls.load(path, name="twin", uuids=False)):
                    self.assertNotEqual(_uuid(twin), data.uuid)
                self.assertEqual(_uuid(shape), data.uuid)

    def test_one_data_builds_any_number_of_copies(self):
        # the user, 2026-11-13: "I want uuids to be False by default"
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data = _shape_data(cls)
                held = [_uuid(cls.create(data, name="copy")) for _ in range(3)]
                self.assertEqual(len(set(held)), 3)
                self.assertNotIn(data.uuid, held)

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
        mesh = Mesh.create(data, name="made", uuids=True)
        with self.assertRaisesRegex(TypeError, r"takes the files, world_space=, normals=, colors= and hierarchy= \(got uuids=\)"):
            mesh.load(path, uuids=False)
        self.assertEqual(_uuid(mesh), data.uuid)

    def test_one_undo_step(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                data = _shape_data(cls)
                self.undo_queue_on()
                shape = cls.create(data, name="made", uuids=True)
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


# ---------------------------------------------------------------------------------------------
class TestTheBlock(_Case):
    """``with rig.uuids():`` (the user, 2026-11-18: "some sort of context trigger that would force
    uuids on on Mesh.create and other rig.nodetypes"): every create from data not given
    ``uuids=`` takes its data's uuids, and a create given no target finds it by uuid. The
    default stays False (the user, 2026-11-13)."""

    def setUp(self):
        super().setUp()
        self.addCleanup(rig.set_options, uuids=rig.get_options()["uuids"])

    def test_inside_the_block_every_create_takes_its_data_s_uuid(self):
        data   = _skeleton()
        lone   = TransformData("lone", uuid=generate_uuid())
        shapes = {cls: _shape_data(cls) for cls in CLASSES}
        with rig.uuids():
            made  = Transform.create_hierarchy(data)
            node  = Transform.create(lone)
            built = {cls: cls.create(shape, name=f"made_{cls.__name__}") for cls, shape in shapes.items()}
        self.assertEqual([_uuid(n) for n in made], [n.uuid for n in data])
        self.assertEqual(_uuid(node), lone.uuid)
        for cls, shape in built.items():
            self.assertEqual(_uuid(shape), shapes[cls].uuid)
        # out of the block, Maya's own again: one data builds any number of copies
        twin = Transform.create(lone, name="twin")
        self.assertNotEqual(_uuid(twin), lone.uuid)

    def test_load_hierarchy_and_create_scene_too(self):
        folder = self.folder()
        data   = _skeleton()
        path   = data.save(os.path.join(folder, "skeleton.npz"))
        with rig.uuids():
            loaded = Joint.load_hierarchy(path)
        self.assertEqual([_uuid(n) for n in loaded], [n.uuid for n in data])
        cmds.file(new=True, force=True)
        scene = _scene(take=False)
        with rig.uuids():
            rig.create_scene(scene)
        self.assertEqual([Node(n.uuid).get_short_name() for n in scene.hierarchy], ["root", "tip", "geo"])
        self.assertEqual(Node(scene.meshes[0].uuid).get_short_name(), "geoShape")

    def test_an_explicit_flag_wins(self):
        data, _ = _mesh_data()
        with rig.uuids():
            kept = Mesh.create(data, name="kept", uuids=False)
        self.assertNotEqual(_uuid(kept), data.uuid)
        with rig.uuids(False):
            taken = Mesh.create(data, name="taken", uuids=True)
        self.assertEqual(_uuid(taken), data.uuid)

    def test_blocks_nest_and_give_the_setting_back(self):
        self.assertFalse(rig.get_options()["uuids"])
        with rig.uuids() as block:
            self.assertEqual(repr(block), "rig.uuids(True)")
            self.assertTrue(rig.get_options()["uuids"])
            with rig.uuids(False):
                self.assertFalse(rig.get_options()["uuids"])
            self.assertTrue(rig.get_options()["uuids"])
        self.assertFalse(rig.get_options()["uuids"])
        with self.assertRaises(ZeroDivisionError):
            with rig.uuids():
                1 / 0
        self.assertFalse(rig.get_options()["uuids"])    # on an exception too

    def test_the_option_and_a_run(self):
        lone = TransformData("lone", uuid=generate_uuid())
        rig.set_options(uuids=True)                      # with no block
        self.assertEqual(_uuid(Transform.create(lone)), lone.uuid)
        rig.set_options(uuids=False)
        rig.exec("import rig\nrig.set_options(uuids=True)")
        self.assertFalse(rig.get_options()["uuids"])     # a run gives its options back
        with rig.uuids():
            rig.exec("import rig\nrig.set_options(uuids=False)")
            self.assertTrue(rig.get_options()["uuids"])

    def test_the_setting_is_a_checked_bool(self):
        for value in (None, 1, "yes"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, r"^rig\.uuids\(\): uuids takes True \(creates from data take "
                                            r"its uuids\) or False \(Maya's own\), not " + re.escape(repr(value)) + "$"):
                    rig.uuids(value)
        with self.assertRaisesRegex(TypeError, r"^rig\.set_options: uuids takes True"):
            rig.set_options(uuids=1)
        self.assertFalse(rig.get_options()["uuids"])
        self.assertIs(rig.uuids, rig._internal.container.uuids)
        self.assertIn("uuids", rig.__all__)

    def test_a_node_goes_under_the_node_of_its_parent_uuid(self):
        geo  = TransformData("geo", translate=(3, 0, 0), uuid=generate_uuid())
        tip  = TransformData("tip", uuid=generate_uuid(), parent_uuid=geo.uuid)
        mesh = _mesh_data()[0]
        curve = _curve_data()
        mesh.parent_uuid = curve.parent_uuid = geo.uuid
        with rig.uuids():
            node  = Transform.create(geo)
            child = Transform.create(tip)                # under geo, by parent_uuid
            shape = Mesh.create(mesh)                    # geo's shape: no transform made
            crv   = NurbsCurve.create(curve)
        for made in (child, shape, crv):
            self.assertEqual(made.get_parent().get_long_name(), node.get_long_name())
        self.assertEqual(shape.get_short_name(), "geoShape")
        self.assertEqual(cmds.getAttr("geo.tx"), 3.0)

    def test_what_is_given_wins_and_a_missing_node_is_no_error(self):
        mesh = _mesh_data()[0]
        geo  = TransformData("geo", uuid=generate_uuid())
        mesh.parent_uuid = geo.uuid
        with rig.uuids():
            Transform.create(geo)
            ctl  = Transform.create(name="ctl")
            here = Mesh.create(mesh, transform=ctl)                       # transform= wins
            mesh.uuid = generate_uuid()
            held = Mesh.create(mesh, name="held", hierarchy=_skeleton())  # hierarchy= says where it sits
            mesh.uuid, mesh.parent_uuid = generate_uuid(), generate_uuid()
            alone = Mesh.create(mesh, name="alone")                       # no node of its parent_uuid
        self.assertEqual(here.get_parent().get_short_name(), "ctl")
        self.assertEqual(held.get_parent().get_short_name(), "held")
        self.assertEqual(alone.get_parent().get_short_name(), "alone")
        # out of the block, a shape never goes under the node of its parent_uuid
        mesh.uuid, mesh.parent_uuid = generate_uuid(), geo.uuid
        self.assertEqual(Mesh.create(mesh, name="beside").get_parent().get_short_name(), "beside")

    def test_the_snippet(self):
        # the user's: the nodes, the mesh, its skin and its targets, each found by uuid
        scene = _scene(take=False)
        with rig.uuids():
            nodes = Transform.create_hierarchy(scene.hierarchy)
            mesh  = Mesh.create(scene.meshes[0])
            skin  = SkinCluster.create(scene.skins[0])
            blend = BlendShape.create(scene.morphs)
        self.assertEqual([_uuid(n) for n in nodes], [n.uuid for n in scene.hierarchy])
        self.assertEqual((mesh.get_parent().get_short_name(), _uuid(mesh)), ("geo", scene.meshes[0].uuid))
        self.assertEqual(skin.get_short_name(), "geo_skin")                    # the SkinData's name
        self.assertEqual([g.get_long_name() for g in skin.get_geometries()], [mesh.get_long_name()])
        self.assertEqual(sorted(j.get_short_name() for j in skin.get_influence_objects()), ["root", "tip"])
        np.testing.assert_allclose(skin.get_weights(), scene.skins[0].weights, atol=1e-6)
        self.assertEqual([g.get_long_name() for g in blend.get_geometries()], [mesh.get_long_name()])
        self.assertEqual(blend.get_target_count(), 1)
        self.assertEqual(blend.get_short_name(), "face_bs")                    # the morphs' deformer

    def test_a_skin_or_morphs_alone_need_the_block_and_their_mesh(self):
        scene = _scene(take=False)
        with rig.uuids():
            Transform.create_hierarchy(scene.hierarchy)
        with self.assertRaisesRegex(TypeError, r"and so does a SkinData inside rig\.uuids\(\): SkinCluster\.create\(skin\)$"):
            SkinCluster.create(scene.skins[0])
        with rig.uuids():
            with self.assertRaisesRegex(ValueError, r"^SkinCluster\.create: no scene node holds the uuid [0-9A-F-]+ of "
                                        r"the SkinData 'geo_skin''s mesh: a Mesh\.create of its mesh inside rig\.uuids\(\)"):
                SkinCluster.create(scene.skins[0])
            with self.assertRaisesRegex(ValueError, r"^BlendShape\.create: no scene node holds the uuid [0-9A-F-]+ of "
                                        r"the morphs' mesh"):
                BlendShape.create(scene.morphs)
            two = scene.morphs.copy()
            two[0].mesh_uuid = generate_uuid()
            with self.assertRaisesRegex(ValueError, r"^BlendShape\.create: inside rig\.uuids\(\) morphs alone deform "
                                        r"the mesh of their mesh_uuid; these name 2 meshes"):
                BlendShape.create(two)
        self.assertEqual(cmds.ls(type=("skinCluster", "blendShape")), [])

    def test_key_hierarchy_keys_by_uuid_in_the_block(self):
        data = _skeleton()
        clip = ClipData(data, frames=3, start_frame=1)
        clip["spine"].translate_y = np.array([5.0, 6.0, 7.0])
        with rig.uuids():
            nodes = Transform.create_hierarchy(data)
        extra = Joint.create(name="extra", parent=nodes[0])
        cmds.reorder(extra.get_long_name(), front=True)   # by order, spine's keys would go to extra
        with rig.uuids():
            nodes[0].key_hierarchy(clip)
        self.assertEqual(cmds.keyframe("spine.ty", query=True, valueChange=True), [5.0, 6.0, 7.0])
        self.assertIsNone(cmds.keyframe(extra.get_long_name() + ".ty", query=True, valueChange=True))
