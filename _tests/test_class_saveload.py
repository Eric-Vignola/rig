"""``save`` / ``load`` on rig.nodetypes (round features s7, feature 3; the user's D2 of
2026-10-15): cgmath files for Mesh, NurbsCurve, NurbsSurface, SkinCluster, BlendShape and
the hierarchy.

* A file is cgmath's own (``.npz`` / ``.json`` / ``.pkl``, by extension on a save, by its
  header on a load), or for a mesh an asset cgmath's MeshData reads (``.obj``). Every
  input is a path (str, ``pathlib.Path``, ``~``) or the data object itself.
* ``Cls.load(path, ...)`` from the class CREATES a new node, always, as ``Cls.create`` does
  (D2). ``node.load(path)`` from a node UPDATES that node in place, the inverse of
  ``node.save(path)``: a mesh of the same topology, a curve or surface of the same CV
  layout, a skinCluster's weights, a blendShape's targets by name. A hierarchy has the
  class form only: ``Transform('root').load_hierarchy(...)`` is a TypeError before
  anything is read.
* Every file is read and its class checked before anything is made or changed: a wrong
  or swapped file, or the arguments in the wrong order, is a TypeError with ``cmds.ls()``
  unchanged.
* ``save`` is a read of the scene: no undo step, the selection kept, the path checked
  before ``serialize`` (mocked here, never called).
* Each load is ONE undo step named ``rig.<Cls>.load`` / ``rig.<Cls>.load_hierarchy``,
  redo brings it back, and the selection is kept.
* A ``cgmath.render.scene.Object`` file stands for a mesh and its UVs (the user's
  addition C).

Probes: runs\\features\\saveload\\probes (P1 to P7) and runs\\features\\s7-saveload\\probes.
"""

import os
import pathlib
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from rig import List, Node, container
from rig._internal import journal
from rig.nodetypes import BlendShape, Joint, Mesh, NurbsCurve, NurbsSurface, SkinCluster, Transform
from rig.nodetypes import _files
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, mesh_state, skin_state

FORMATS = ("npz", "json", "pkl")


# -- sources


def _cube(name="body", ch=False):
    """A polyCube with a second UV set, moved: the transform."""
    t = cmds.polyCube(name=name, constructionHistory=ch)[0]
    if not ch:
        cmds.polyUVSet(t, create=True, uvSet="second")
        cmds.polyProjection(f"{t}.f[*]", type="Planar", mapDirection="y", uvSetName="second")
        cmds.delete(t, constructionHistory=True)
    cmds.xform(t, translation=(1, 2, 3), rotation=(10, 20, 30))
    return t


def _fv_normals(shape, space=OpenMaya.MSpace.kObject):
    fn = Mesh(shape).fn_set
    normals = np.array([tuple(v) for v in fn.getNormals(space)])
    return normals[np.array(list(fn.getNormalIds()[1]), dtype=int)]


def _hard_edges(shape):
    fn = Mesh(shape).fn_set
    return {frozenset(fn.getEdgeVertices(e)) for e in range(fn.numEdges) if not fn.isEdgeSmooth(e)}


def _locked_rows(shape):
    fn = Mesh(shape).fn_set
    counts, ids = fn.getNormalIds()
    _, verts = fn.getVertices()
    faces = np.repeat(np.arange(len(counts)), list(counts))
    return {(int(f), int(v)) for f, v, n in zip(faces, verts, ids) if fn.isNormalLocked(n)}


def _world_points(shape):
    return np.array(Mesh(shape).get_points(world_space=True))[:, :3]


def _cvs(shape, world_space=True):
    node = Node(shape)
    return np.array(node.get_points(world_space=world_space))[:, :3]


def _skeleton():
    """root -> spine (a joint chain with a user attr), a locator and a group: the root."""
    root = Joint.create(name="root")
    spine = Joint.create(name="spine", parent=root)
    spine.t.set(0, 5, 0)
    spine.r.set(10, 0, 30)
    cmds.addAttr("spine", longName="twist", attributeType="double", keyable=True)
    cmds.setAttr("spine.twist", 2.5)
    loc = cmds.spaceLocator(name="tip")[0]
    cmds.parent(loc, "spine")
    cmds.setAttr("tip.ty", 3)
    grp = Transform.create(name="grp", parent=root)
    grp.t.set(1, 0, 0)
    return root


def _world_matrices(nodes):
    return {Node(n).short_name: np.array(cmds.xform(str(n), query=True, matrix=True, worldSpace=True))
            for n in nodes}


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp(prefix="rig_saveload_")

    def tearDown(self):
        try:
            cmds.namespace(set=":")
            shutil.rmtree(self.dir, ignore_errors=True)
            self.assertFalse(container.is_active)
        finally:
            super().tearDown()

    def path(self, name):
        return os.path.join(self.dir, name)

    def ready(self):
        """Select the probe transform ``selprobe`` (made when missing) and flush what
        building the sources recorded: the test's own steps start here."""
        if not cmds.objExists("selprobe"):
            cmds.createNode("transform", name="selprobe")
        cmds.select("selprobe", replace=True)
        cmds.flushUndo()

    def assert_kept_selection(self):
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def assert_refused(self, error, pattern, call):
        """`call` raises `error` matching `pattern`, and the scene has the same nodes."""
        before = cmds.ls()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(cmds.ls(), before)

    def assert_one_step(self, name, exists):
        """The last edit is one undo step named `name`: undoing it makes `exists()`
        False, redoing it True again."""
        self.assertEqual(self.undo_name(), name)
        self.undo_steps(1)
        self.assertFalse(exists())
        self.redo_steps(1)
        self.assertTrue(exists())


# ---------------------------------------------------------------------------------------------
class TestTheTwoForms(MayaTestCase):
    """One name, two forms: from the class the creator, from a node the updater."""

    TEST_START_NEW_SCENE = True

    def test_the_class_form_creates_the_node_form_updates(self):
        cube = cmds.polyCube(name="c", constructionHistory=False)[0]
        circle = cmds.circle(name="o", constructionHistory=False)[0]
        surface = cmds.sphere(name="s", constructionHistory=False)[0]
        for cls, node, method in (
            (Mesh, Mesh(cube), "load"),
            (NurbsCurve, NurbsCurve(circle), "load"),
            (NurbsSurface, NurbsSurface(surface), "load"),
            (Transform, Transform(cube), "load_hierarchy"),
        ):
            with self.subTest(cls=cls.__name__):
                from_class = getattr(cls, method)
                from_node = getattr(node, method)
                self.assertIs(from_class.__self__, cls)
                self.assertIs(from_node.__self__, node)
                self.assertIsNot(from_class.__func__, from_node.__func__)
                self.assertEqual(from_class.__doc__, from_node.__doc__)
                self.assertEqual(from_class.__qualname__, f"{cls.__name__}.{method}")
                self.assertIs(from_class.__func__._rig_statement, True)
        for cls in (SkinCluster, BlendShape):
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.load.__self__, cls)
                self.assertIs(cls.load.__func__._rig_statement, True)
                self.assertIn("one name, two forms", cls.load.__doc__)

    def test_the_docstrings_name_both_forms(self):
        for method, forms in (
            (Mesh.load, ("Mesh.load(mesh", "Mesh('bodyShape').load(mesh")),
            (NurbsCurve.load, ("NurbsCurve.load(curve", "NurbsCurve('ctlShape').load(curve")),
            (NurbsSurface.load, ("NurbsSurface.load(surface", "NurbsSurface('patchShape').load(surface")),
            (SkinCluster.load, ("SkinCluster.load(skin, geometry", "SkinCluster('body_skincluster').load(skin)")),
            (BlendShape.load, ("BlendShape.load(targets, geometry", "BlendShape('face_bs').load(targets)")),
            (Transform.load_hierarchy, ("Transform.load_hierarchy(hierarchy", "Transform('root').load_hierarchy(...)")),
        ):
            for form in forms:
                with self.subTest(form=form):
                    self.assertIn(form, method.__doc__)

    def test_the_names_shadow_no_maya_attribute(self):
        # P1: no node type has an attribute of these names
        for node_type in ("mesh", "nurbsCurve", "nurbsSurface", "skinCluster", "blendShape", "transform", "joint"):
            for name in ("save", "load", "save_hierarchy", "load_hierarchy"):
                with self.subTest(node_type=node_type, name=name):
                    self.assertFalse(cmds.attributeQuery(name, type=node_type, exists=True))

    def test_a_transform_names_its_shape_and_its_branch(self):
        cube = cmds.polyCube(name="body", constructionHistory=False)[0]
        circle = cmds.circle(name="ctl", constructionHistory=False)[0]
        Joint.create(name="j")
        for node, name, text in (
            ("body", "save", "Attribute not found: body.save; 'body' is a transform: "
                             "Mesh('body').save(...) saves its mesh, save_hierarchy(...) its branch"),
            ("body", "load", "Attribute not found: body.load; 'body' is a transform: "
                             "Mesh('body').load(...) updates its mesh, Transform.load_hierarchy(path) makes a branch"),
            ("ctl", "save", "Attribute not found: ctl.save; 'ctl' is a transform: "
                            "NurbsCurve('ctl').save(...) saves its curve, save_hierarchy(...) its branch"),
            ("j", "save", "Attribute not found: j.save; 'j' is a joint: save_hierarchy(...) saves its branch"),
            ("j", "load", "Attribute not found: j.load; 'j' is a joint: Joint.load_hierarchy(path) makes a branch"),
        ):
            with self.subTest(node=node, name=name):
                with self.assertRaises(AttributeError) as caught:
                    getattr(Node(node), name)
                self.assertEqual(str(caught.exception), text)
        # another node's miss keeps its message
        with self.assertRaises(AttributeError) as caught:
            Node(cmds.createNode("multiplyDivide", name="md")).save
        self.assertEqual(str(caught.exception), "Attribute not found: md.save")
        self.assertTrue(cube and circle)


# ---------------------------------------------------------------------------------------------
class TestFiles(_Case):
    """The file side, shared by every class (`rig.nodetypes._files`)."""

    def test_a_save_path_is_checked_first(self):
        for path, error, text in (
            (self.path("x.txt"), ValueError, "cgmath saves .npz, .json or .pkl files, by extension"),
            (self.path("x"), ValueError, "cgmath saves .npz, .json or .pkl files, by extension"),
            (self.path("no_folder/x.npz"), FileNotFoundError, "no folder"),
            (5, TypeError, "a path is a str or an os.PathLike, not a int"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(error, text):
                    _files.check_save_path(path, "X.save")

    def test_a_plug_is_no_path(self):
        cube = Node(cmds.polyCube(name="c", constructionHistory=False)[0])
        self.assertFalse(_files.is_path(cube.tx))
        self.assertTrue(_files.is_path("a.npz"))
        self.assertTrue(_files.is_path(pathlib.Path("a.npz")))

    def test_a_tilde_is_expanded(self):
        with mock.patch.dict(os.environ, {"USERPROFILE": self.dir, "HOME": self.dir}):
            got = _files.check_save_path("~/x.npz", "X.save")
        self.assertEqual(os.path.normpath(got), os.path.join(self.dir, "x.npz"))


# ---------------------------------------------------------------------------------------------
class TestMesh(_Case):
    """Mesh.save / Mesh.load (the class creates) / Mesh('x').load (updates)."""

    def saved(self, ext="npz", world_space=True):
        t = _cube()
        Mesh(t).save(self.path(f"body.{ext}"), uvs=self.path(f"body_uv.{ext}"), world_space=world_space)
        return t, self.path(f"body.{ext}"), self.path(f"body_uv.{ext}")

    def assert_same_mesh(self, made, source_state, source_world, source_normals, source_hard):
        state = mesh_state(made)
        for key in ("vertices", "holes", "uv_sets", "uvs"):
            self.assertEqual(state[key], source_state[key], key)
        np.testing.assert_allclose(_world_points(made), source_world, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(_fv_normals(made, OpenMaya.MSpace.kWorld), source_normals, rtol=0.0, atol=1e-5)
        self.assertEqual(_hard_edges(made), source_hard)

    def test_the_round_trip_in_each_format(self):
        t = _cube()
        shape = cmds.listRelatives(t, shapes=True)[0]
        source = (mesh_state(shape), _world_points(shape), _fv_normals(shape, OpenMaya.MSpace.kWorld), _hard_edges(shape))
        for ext in FORMATS:
            with self.subTest(ext=ext):
                saved = Mesh(t).save(self.path(f"b.{ext}"), uvs=self.path(f"b_uv.{ext}"))
                self.assertEqual(saved, (self.path(f"b.{ext}"), self.path(f"b_uv.{ext}")))
                self.ready()
                mesh = Mesh.load(self.path(f"b.{ext}"), self.path(f"b_uv.{ext}"), name=f"m_{ext}")
                self.assertIsInstance(mesh, Mesh)
                self.assertEqual(str(mesh.get_parent()), f"m_{ext}")
                self.assert_same_mesh(str(mesh), *source)
                self.assert_kept_selection()
                # what a serialize gives back is the file's data
                data, uvs = mesh.serialize()
                from cgmath.geometry import MeshData, UVList

                file_data = MeshData.load(self.path(f"b.{ext}"))
                np.testing.assert_allclose(data.points, file_data.points, atol=1e-5)
                self.assertEqual(data.indices.tolist(), file_data.indices.tolist())
                self.assertEqual([uv.name for uv in uvs], [uv.name for uv in UVList.load(self.path(f"b_uv.{ext}"))])

    def test_save_without_uvs_writes_the_mesh_only(self):
        t = _cube()
        self.assertEqual(Mesh(t).save(self.path("only.npz")), self.path("only.npz"))
        mesh = Mesh.load(self.path("only.npz"), name="m")
        self.assertEqual(list(mesh.uv_sets), ["map1"])
        self.assertEqual(len(mesh.get_uv_coords("map1")), 0)

    def test_world_space_false_brings_back_the_transform(self):
        t, path, uvs = self.saved(world_space=False)
        world = _world_points(cmds.listRelatives(t, shapes=True)[0])
        mesh = Mesh.load(path, uvs, name="m")
        np.testing.assert_allclose(_world_points(str(mesh)), world, atol=1e-5)
        np.testing.assert_allclose(cmds.getAttr("m.t")[0], (1, 2, 3), atol=1e-6)

    def test_data_objects_paths_and_tilde(self):
        t, path, uvs = self.saved()
        data, uv_list = Mesh(t).serialize()
        made = [
            Mesh.load(data, uv_list, name="from_data"),
            Mesh.load(pathlib.Path(path), pathlib.Path(uvs), name="from_path"),
        ]
        with mock.patch.dict(os.environ, {"USERPROFILE": self.dir, "HOME": self.dir}):
            made.append(Mesh.load("~/body.npz", "~/body_uv.npz", name="from_tilde"))
        for mesh in made:
            with self.subTest(mesh=str(mesh)):
                self.assertEqual(mesh_state(str(mesh))["uvs"], mesh_state(str(made[0]))["uvs"])
        # the caller's data object is not changed
        again, _ = Mesh(t).serialize()
        np.testing.assert_array_equal(np.asarray(data.points), np.asarray(again.points))

    def test_an_obj_asset(self):
        t = _cube()
        data = Mesh(t).serialize(include_uvs=False)
        data.save_obj(self.path("body.obj"))
        mesh = Mesh.load(self.path("body.obj"), self.path("body.obj"), name="from_obj")
        self.assertEqual(mesh.num_vertices, 8)
        self.assertEqual(list(mesh.uv_sets), ["map1"])

    def test_wrong_and_swapped_inputs_make_nothing(self):
        t, path, uvs = self.saved()
        curve = self.curve_file()
        self.ready()
        q = lambda p: repr(p).replace("\\", "\\\\")  # noqa: E731
        for label, call, error, text in (
            ("swapped", lambda: Mesh.load(uvs, path), TypeError,
             f"^Mesh.load: the mesh: {q(uvs)} holds a UVList; the mesh is a MeshData or an Object \\(the UVs go in uvs=\\)$"),
            ("the uvs alone", lambda: Mesh.load(uvs), TypeError, "holds a UVList"),
            ("a mesh as uvs", lambda: Mesh.load(path, path), TypeError,
             f"^Mesh.load: uvs: {q(path)} holds a MeshData; uvs is a UVList or a UVData \\(the mesh goes first\\)$"),
            ("a UVData object", lambda: Mesh.load(Mesh(t).serialize_uv()), TypeError, "got a UVData"),
            ("a node", lambda: Mesh.load(Node(t)), TypeError,
             "^Mesh.load: the mesh is a file \\(a path\\) or its data, not the node 'body'$"),
            ("a plug", lambda: Mesh.load(Node(t).tx), TypeError, "not the node 'body.translateX'"),
            ("a curve file", lambda: Mesh.load(curve), TypeError, "holds a BSplineData"),
            ("a missing file", lambda: Mesh.load(self.path("nope.npz")), FileNotFoundError,
             f"^Mesh.load: the mesh: no file {q(self.path('nope.npz'))}$"),
            ("a missing uv file", lambda: Mesh.load(path, self.path("nope.npz")), FileNotFoundError, "Mesh.load: uvs: no file"),
        ):
            with self.subTest(label):
                self.assert_refused(error, text, call)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def curve_file(self):
        if not os.path.exists(self.path("c.npz")):
            NurbsCurve.create([(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)]).serialize().save(self.path("c.npz"))
        return self.path("c.npz")

    def test_an_older_cgmath_file_raises_cgmaths_error(self):
        from cgmath.geometry._base import LegacyFileError

        t, path, _ = self.saved()
        with mock.patch("cgmath.geometry.Data.load", side_effect=LegacyFileError("an older cgmath's file")):
            self.assert_refused(LegacyFileError, "an older cgmath's file", lambda: Mesh.load(path))

    def test_save_checks_the_paths_before_it_serializes(self):
        t = _cube()
        mesh = Mesh(t)
        for label, call, error, text in (
            ("a bad extension", lambda: mesh.save(self.path("b.txt")), ValueError, "cgmath saves .npz"),
            ("a bad uv extension", lambda: mesh.save(self.path("b.npz"), uvs=self.path("u.obj")), ValueError, "cgmath saves .npz"),
            ("a missing folder", lambda: mesh.save(self.path("nofolder/b.npz")), FileNotFoundError, "no folder"),
            ("the same path twice", lambda: mesh.save(self.path("b.npz"), uvs=self.path("b.npz")), ValueError,
             "^Mesh.save: the mesh and its UVs are two files: give uvs= another path than "),
            ("one file two spellings", lambda: mesh.save(self.path("b.npz"), uvs=self.path("x/../b.npz")), ValueError, "two files"),
        ):
            with self.subTest(label):
                with mock.patch.object(Mesh, "serialize") as serialize:
                    with self.assertRaisesRegex(error, text):
                        call()
                    serialize.assert_not_called()
                self.assertEqual(os.listdir(self.dir), [])

    def test_save_is_a_read(self):
        t = _cube()
        self.ready()
        Mesh(t).save(self.path("b.npz"), uvs=self.path("u.npz"))
        Mesh(t).save(self.path("b.npz"))  # an existing file is replaced
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assert_kept_selection()

    def test_one_undo_step_and_redo(self):
        _, path, uvs = self.saved()
        self.ready()
        mesh = Mesh.load(path, uvs, name="m")
        uuid = cmds.ls("m", uuid=True)
        self.assert_one_step("rig.Mesh.load", lambda: cmds.objExists("m"))
        self.assertEqual(cmds.ls("m", uuid=True), uuid)
        self.assertEqual(_hard_edges("mShape"), _hard_edges("bodyShape"))
        self.assertIs(mesh.is_valid, True)

    def test_the_names(self):
        _, path, uvs = self.saved()
        # the data's name, Shape dropped; Maya's suffix on a clash; name= wins
        self.assertEqual(str(Mesh.load(path).get_parent()), "body1")
        self.assertEqual(str(Mesh.load(path).get_parent()), "body2")
        self.assertEqual(str(Mesh.load(path, name="hero").get_parent()), "hero")
        data = Mesh("body").serialize(include_uvs=False)
        data.name = None
        self.assertEqual(str(Mesh.load(data).get_parent()), "polySurface1")

    def test_the_class_form_always_creates(self):
        # D2: Mesh.load() creates, as Mesh.create does (Mesh('x').load() updates)
        _, path, uvs = self.saved()
        first = Mesh.load(path, uvs, name="m")
        second = Mesh.load(path, uvs, name="m")
        self.assertEqual((str(first.get_parent()), str(second.get_parent())), ("m", "m1"))

    def test_flags_and_attributes_pass_through(self):
        _, path, uvs = self.saved()
        mesh = Mesh.load(path, uvs, name="m", visibility=False, castsShadows=False)
        self.assertEqual((cmds.getAttr("mShape.visibility"), cmds.getAttr("mShape.castsShadows")), (False, False))
        self.assert_refused(AttributeError, "nosuch", lambda: Mesh.load(path, uvs, name="x", nosuch=1))
        self.assertIsInstance(mesh, Mesh)

    def test_container(self):
        _, path, uvs = self.saved()
        self.ready()
        with container("box") as box:
            Mesh.load(path, uvs, name="m")
            Mesh.load(path, uvs, name="out", container=False)
        # one scope adds no prefix (a nested one does: test_class_mesh_create)
        members = sorted(cmds.container(str(box), query=True, nodeList=True))
        self.assertEqual(members, ["m", "mShape"])
        self.assertEqual(self.undo_name(), "rig.Mesh.load")

    # --- the node form: an update in place

    def test_the_node_form_updates_in_place(self):
        t, path, uvs = self.saved()
        source = mesh_state("bodyShape"), _world_points("bodyShape"), _fv_normals("bodyShape", OpenMaya.MSpace.kWorld)
        hard = _hard_edges("bodyShape")
        # the loaded copy is edited: points moved, every edge softened, a UV set renamed
        copy = Mesh.load(path, uvs, name="copy")
        cmds.xform("copy.vtx[0:3]", translation=(0, 4, 0), relative=True, worldSpace=True)
        cmds.polySoftEdge("copy", angle=180, constructionHistory=False)
        cmds.polyNormalPerVertex("copy.vtx[5]", xyz=(1, 0, 0))
        cmds.delete("copy", constructionHistory=True)
        copy.rename_uv_set("other", "second")
        uuid = cmds.ls("copyShape", uuid=True)
        edited = mesh_state("copyShape"), _hard_edges("copyShape"), _locked_rows("copyShape")
        self.ready()
        self.assertIs(copy.load(path, uvs), copy)
        self.assertEqual(cmds.ls("copyShape", uuid=True), uuid)
        np.testing.assert_allclose(_world_points("copyShape"), source[1], atol=1e-5)
        np.testing.assert_allclose(_fv_normals("copyShape", OpenMaya.MSpace.kWorld), source[2], atol=1e-5)
        self.assertEqual(_hard_edges("copyShape"), hard)
        self.assertEqual(_locked_rows("copyShape"), set())
        # the file's sets by name: map1 and second written, the mesh's own 'other' kept
        state = mesh_state("copyShape")
        self.assertEqual(state["uv_sets"], ["map1", "other", "second"])
        self.assertEqual(state["uvs"]["second"], source[0]["uvs"]["second"])
        self.assertEqual(state["uvs"]["map1"], source[0]["uvs"]["map1"])
        self.assert_kept_selection()
        # one step: undo puts back the edited mesh, its shading and its locks
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        after = mesh_state("copyShape"), _hard_edges("copyShape"), _locked_rows("copyShape")
        self.undo_steps(1)
        self.assertEqual((mesh_state("copyShape"), _hard_edges("copyShape"), _locked_rows("copyShape")), edited)
        self.redo_steps(1)
        self.assertEqual((mesh_state("copyShape"), _hard_edges("copyShape"), _locked_rows("copyShape")), after)

    def test_the_node_form_brings_back_the_hard_edges(self):
        # nothing locked, the mesh only softened: the edges come back hard (the update once kept
        # one MFnMesh across its setPoints and lost the edge flags)
        _, path, uvs = self.saved()
        copy = Mesh.load(path, uvs, name="copy")
        cmds.polySoftEdge("copy", angle=180, constructionHistory=False)
        self.assertEqual(_hard_edges("copyShape"), set())
        copy.load(path, uvs)
        self.assertEqual(_hard_edges("copyShape"), _hard_edges("bodyShape"))
        self.undo_steps(1)
        self.assertEqual(_hard_edges("copyShape"), set())
        self.redo_steps(1)
        self.assertEqual(_hard_edges("copyShape"), _hard_edges("bodyShape"))

    def test_the_node_form_world_space(self):
        t, path, uvs = self.saved()
        world = _world_points("bodyShape")
        Mesh.load(path, uvs, name="copy")
        cmds.xform("copy", translation=(0, 10, 0), rotation=(0, 90, 0))
        # world_space=True (the default): the points land where the file puts them
        Mesh("copy").load(path)
        np.testing.assert_allclose(_world_points("copyShape"), world, atol=1e-5)
        self.assertEqual(cmds.getAttr("copy.t")[0], (0.0, 10.0, 0.0))  # the transform is not moved
        # world_space=False: the file's points become the mesh's own
        Mesh(t).save(self.path("local.npz"), world_space=False)
        Mesh("copy").load(self.path("local.npz"), world_space=False)
        np.testing.assert_allclose(
            np.array(Mesh("copyShape").get_points(world_space=False))[:, :3],
            np.array(Mesh("bodyShape").get_points(world_space=False))[:, :3],
            atol=1e-6,
        )

    def test_the_node_form_refuses_another_topology(self):
        _, path, _ = self.saved()
        ball = Mesh(cmds.polySphere(name="ball", constructionHistory=False)[0])
        before = mesh_state("ballShape")
        q = repr(path).replace("\\", "\\\\")
        self.assert_refused(
            ValueError,
            f"^Mesh\\('ballShape'\\).load: {q} holds another topology than 'ballShape' \\(8 vertices and 6 faces "
            f"against its 382 and 400, or other face lists or holes\\); a load updates the points, the shading "
            f"and the UVs of the same topology: Mesh.load\\({q}\\) makes a new mesh$",
            lambda: ball.load(path),
        )
        self.assertEqual(mesh_state("ballShape"), before)
        # the same counts, other face lists
        data = Mesh("body").serialize(include_uvs=False)
        data.indices = np.roll(np.asarray(data.indices), 1)
        copy = Mesh.load(path, name="copy")
        self.assert_refused(ValueError, "holds another topology", lambda: copy.load(data))

    def test_the_node_form_refuses_a_mesh_with_history(self):
        _, path, _ = self.saved()
        cube = _cube("live", ch=True)
        self.assert_refused(
            ValueError,
            "^Mesh\\('liveShape'\\).load: 'liveShape' has history \\(its inMesh is connected: construction "
            "history\\), which would compute over the points it sets; delete the history, or "
            "Mesh.load\\(path\\) makes a new mesh$",
            lambda: Mesh(cube).load(path),
        )

    def test_the_node_form_takes_no_create_keywords(self):
        _, path, _ = self.saved()
        self.assert_refused(
            TypeError,
            "^Mesh\\('bodyShape'\\).load updates the mesh in place: it takes the files, world_space= and "
            "normals= \\(got name=, container=\\); Mesh.load\\(path, ...\\) makes a new mesh$",
            lambda: Mesh("body").load(path, name="x", container=False),
        )

    def test_the_node_form_checks_the_uvs_first(self):
        _, path, uvs = self.saved()
        ball_uv = Mesh(cmds.polySphere(name="ball", constructionHistory=False)[0]).serialize()[1]
        before = mesh_state("bodyShape")
        with self.assertRaisesRegex(RuntimeError, "face counts for 6 faces"):
            Mesh("body").load(path, ball_uv)
        self.assertEqual(mesh_state("bodyShape"), before)

    def test_the_node_form_without_normals_keeps_the_shading(self):
        _, path, _ = self.saved()
        data = Mesh("body").serialize(include_uvs=False)
        data.normals = None
        data.normal_indices = None
        copy = Mesh.load(path, name="copy")
        cmds.polySoftEdge("copy", angle=180, constructionHistory=False)
        Mesh("copy").load(data)
        self.assertEqual(_hard_edges("copyShape"), set())

    # --- a cgmath Object (the user's addition C)

    def objects(self):
        """``(an Object holding a UVData, one holding a UVList)``, from the moved cube."""
        from cgmath.geometry import UVList
        from cgmath.render.scene import Object

        t = _cube()
        data, uvs = Mesh(t).serialize(world_space=False)
        one = Object(name="hero", mesh=data, uv=uvs[0], translate=(5, 0, 0), rotate=(0, 45, 0))
        # the UVList the next cgmath's Object holds (one takes it today: a stand-in)
        every = Object(name="hero", mesh=data, uv=UVList(list(uvs)), translate=(5, 0, 0), rotate=(0, 45, 0))
        return one, every

    def test_an_object_file(self):
        one, every = self.objects()
        for label, obj, sets in (("UVData", one, ["map1"]), ("UVList", every, ["map1", "second"])):
            for ext in FORMATS:
                with self.subTest(label=label, ext=ext):
                    obj.save(self.path(f"hero_{label}.{ext}"))
                    self.ready()
                    mesh = Mesh.load(self.path(f"hero_{label}.{ext}"))
                    self.assertEqual(list(mesh.uv_sets), sets)
                    self.assertEqual(str(mesh.get_parent()).rstrip("0123456789"), "hero")
                    self.assertEqual(cmds.getAttr(f"{mesh.get_parent()}.t")[0], (5.0, 0.0, 0.0))
                    self.assertEqual(cmds.getAttr(f"{mesh.get_parent()}.r")[0], (0.0, 45.0, 0.0))
                    self.assertEqual(_hard_edges(str(mesh)), _hard_edges("bodyShape"))
                    self.assertEqual(self.undo_name(), "rig.Mesh.load")
                    self.assert_kept_selection()

    def test_an_object_file_updates_a_mesh(self):
        one, every = self.objects()
        every.save(self.path("hero.npz"))
        copy = Mesh.load(self.path("hero.npz"), name="copy")
        cmds.xform("copy.vtx[0]", translation=(0, 3, 0), relative=True)
        cmds.polyUVSet("copy", delete=True, uvSet="second")
        cmds.delete("copy", constructionHistory=True)
        world = _world_points("copyShape")
        copy.load(self.path("hero.npz"))
        self.assertEqual(list(copy.uv_sets), ["map1", "second"])
        # world space: the Object's transform places the points (the copy is where it is)
        self.assertFalse(np.allclose(_world_points("copyShape"), world, atol=1e-5))
        copy.load(self.path("hero.npz"), world_space=False)
        np.testing.assert_allclose(
            np.array(copy.get_points(world_space=False))[:, :3], np.asarray(one.mesh.points), atol=1e-6
        )

    def test_an_object_takes_no_uvs(self):
        one, _ = self.objects()
        one.save(self.path("hero.npz"))
        Mesh("body").save(self.path("b.npz"), uvs=self.path("u.npz"))
        for label, call, text in (
            ("class", lambda: Mesh.load(self.path("hero.npz"), self.path("u.npz")),
             "^Mesh.load: an Object carries its mesh and its UVs, so it takes no uvs= \\(uvs= goes with a MeshData\\)$"),
            ("node", lambda: Mesh("body").load(self.path("hero.npz"), self.path("u.npz")),
             "^Mesh\\('bodyShape'\\).load: an Object carries its mesh and its UVs, so it takes no uvs="),
            ("object", lambda: Mesh.load(one, self.path("u.npz")), "so it takes no uvs="),
        ):
            with self.subTest(label):
                with mock.patch.object(_files, "read", wraps=_files.read) as read:
                    self.assert_refused(TypeError, text, call)
                    self.assertEqual(read.call_count, 1)  # the UVs' file is never read


# ---------------------------------------------------------------------------------------------
class TestNurbsCurve(_Case):
    """NurbsCurve.save / NurbsCurve.load / NurbsCurve('x').load."""

    def sources(self):
        """``{label: shape}``: an open curve, a periodic circle, uneven knots, a moved one."""
        out = {
            "open": cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)], name="open_c"),
            "circle": cmds.circle(name="circ", constructionHistory=False, sections=8)[0],
            "uneven": cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)],
                                 knot=[0, 0, 0, 1, 5, 5, 5], name="uneven"),
            "moved": cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name="moved"),
        }
        cmds.xform("moved", translation=(5, 0, 0), rotation=(0, 90, 0))
        return {k: cmds.listRelatives(v, shapes=True)[0] for k, v in out.items()}

    def test_the_round_trip_in_each_format(self):
        for label, shape in self.sources().items():
            fn = NurbsCurve(shape).fn_set
            for ext in FORMATS:
                with self.subTest(label=label, ext=ext):
                    path = self.path(f"{label}.{ext}")
                    self.assertEqual(NurbsCurve(shape).save(path), path)
                    self.ready()
                    curve = NurbsCurve.load(path)
                    self.assertIsInstance(curve, NurbsCurve)
                    got = curve.fn_set
                    self.assertEqual((got.numCVs, got.degree, got.form), (fn.numCVs, fn.degree, fn.form))
                    np.testing.assert_allclose(np.array(got.knots()), np.array(fn.knots()), atol=1e-9)
                    np.testing.assert_allclose(_cvs(str(curve)), _cvs(shape), atol=1e-9)
                    self.assert_kept_selection()
                    self.assertEqual(self.undo_name(), "rig.NurbsCurve.load")

    def test_world_space_false_loads_at_the_origin(self):
        # a BSplineData has no matrix: the default world_space=True keeps where the curve is
        shape = self.sources()["moved"]
        NurbsCurve(shape).save(self.path("local.npz"), world_space=False)
        curve = NurbsCurve.load(self.path("local.npz"))
        np.testing.assert_allclose(_cvs(str(curve)), _cvs(shape, world_space=False), atol=1e-9)

    def test_data_objects_paths_and_names(self):
        shape = self.sources()["circle"]
        data = NurbsCurve(shape).serialize()
        NurbsCurve(shape).save(self.path("c.npz"))
        first = NurbsCurve.load(data)
        self.assertEqual(str(first.get_parent()), "curve1")
        self.assertEqual(str(NurbsCurve.load(pathlib.Path(self.path("c.npz"))).get_parent()), "curve2")
        self.assertEqual(str(NurbsCurve.load(self.path("c.npz"), name="ctl").get_parent()), "ctl")
        self.assertEqual(str(NurbsCurve.load(self.path("c.npz"), name="ctl").get_parent()), "ctl1")
        hidden = NurbsCurve.load(self.path("c.npz"), name="h", visibility=False)
        self.assertFalse(cmds.getAttr(f"{hidden}.visibility"))

    def test_wrong_inputs_make_nothing(self):
        shape = self.sources()["circle"]
        Mesh(cmds.polyCube(name="m", constructionHistory=False)[0]).save(self.path("m.npz"))
        for label, call, error, text in (
            ("a mesh file", lambda: NurbsCurve.load(self.path("m.npz")), TypeError, "holds a MeshData; the curve is a BSplineData$"),
            ("a node", lambda: NurbsCurve.load(Node(shape)), TypeError, "not the node"),
            ("a missing file", lambda: NurbsCurve.load(self.path("nope.npz")), FileNotFoundError, "NurbsCurve.load: the curve: no file"),
        ):
            with self.subTest(label):
                self.assert_refused(error, text, call)
        with mock.patch.object(NurbsCurve, "serialize") as serialize:
            with self.assertRaisesRegex(ValueError, "^NurbsCurve.save: .*cgmath saves .npz"):
                NurbsCurve(shape).save(self.path("c.txt"))
            with self.assertRaisesRegex(FileNotFoundError, "no folder"):
                NurbsCurve(shape).save(self.path("no/c.npz"))
            serialize.assert_not_called()

    def test_one_undo_step_and_container(self):
        shape = self.sources()["circle"]
        NurbsCurve(shape).save(self.path("c.npz"))
        self.ready()
        NurbsCurve.load(self.path("c.npz"), name="ctl")
        self.assert_one_step("rig.NurbsCurve.load", lambda: cmds.objExists("ctl"))
        with container("box") as box:
            NurbsCurve.load(self.path("c.npz"), name="ctl")
        self.assertEqual(sorted(cmds.container(str(box), query=True, nodeList=True)), ["ctl1", "ctlShape1"])

    def test_the_node_form_updates_the_cvs(self):
        shape = self.sources()["circle"]
        NurbsCurve(shape).save(self.path("c.npz"))
        source = _cvs(shape)
        copy = NurbsCurve.load(self.path("c.npz"), name="copy")
        cmds.xform("copy.cv[0:2]", translation=(0, 2, 0), relative=True)
        cmds.xform("copy", translation=(0, 0, 3))
        edited = _cvs(str(copy))
        uuid = cmds.ls(str(copy), uuid=True)
        self.ready()
        self.assertIs(copy.load(self.path("c.npz")), copy)
        np.testing.assert_allclose(_cvs(str(copy)), source, atol=1e-9)
        self.assertEqual(cmds.ls(str(copy), uuid=True), uuid)
        self.assertEqual(cmds.getAttr("copy.t")[0], (0.0, 0.0, 3.0))
        self.assert_kept_selection()
        self.assertEqual(self.undo_name(), "rig.NurbsCurve.load")
        self.undo_steps(1)
        np.testing.assert_allclose(_cvs(str(copy)), edited, atol=1e-9)
        self.redo_steps(1)
        np.testing.assert_allclose(_cvs(str(copy)), source, atol=1e-9)
        # world_space=False: the file's CVs as the curve's own
        copy.load(self.path("c.npz"), world_space=False)
        np.testing.assert_allclose(_cvs(str(copy), world_space=False), source, atol=1e-9)

    def test_the_node_form_refuses_another_layout(self):
        shapes = self.sources()
        NurbsCurve(shapes["circle"]).save(self.path("c.npz"))
        NurbsCurve(shapes["open"]).save(self.path("o.npz"))
        q = repr(self.path("c.npz")).replace("\\", "\\\\")
        name = shapes["open"]
        for label, target, path, text in (
            ("count and form", shapes["open"], self.path("c.npz"),
             f"^NurbsCurve\\('{name}'\\).load: {q} holds another CV layout than '{name}' \\(11 CVs of "
             f"degree 3, periodic, against its 5 of degree 3, open, or other knots\\); a load updates the CVs "
             f"of the same layout: NurbsCurve.load\\({q}\\) makes a new curve$"),
            ("knots", shapes["uneven"], self.path("o.npz"), f"holds another CV layout than '{shapes['uneven']}' \\(5 CVs of degree 3, open, against its 5 of degree 3, open, or other knots\\)"),
        ):
            with self.subTest(label):
                before = _cvs(target)
                self.assert_refused(ValueError, text, lambda: NurbsCurve(target).load(path))
                np.testing.assert_array_equal(_cvs(target), before)

    def test_the_node_form_refuses_history_and_create_keywords(self):
        live = cmds.circle(name="live", constructionHistory=True)[0]
        NurbsCurve(cmds.listRelatives(live, shapes=True)[0]).save(self.path("c.npz"))
        self.assert_refused(ValueError, "^NurbsCurve\\('liveShape'\\).load: 'liveShape' has history",
                            lambda: NurbsCurve("liveShape").load(self.path("c.npz")))
        self.assert_refused(TypeError, "^NurbsCurve\\('liveShape'\\).load updates it in place: it takes the file "
                            "and world_space= \\(got name=\\); NurbsCurve.load\\(path, ...\\) makes a new curve$",
                            lambda: NurbsCurve("liveShape").load(self.path("c.npz"), name="x"))


# ---------------------------------------------------------------------------------------------
class TestNurbsSurface(_Case):
    """NurbsSurface.save / NurbsSurface.load / NurbsSurface('x').load."""

    def sources(self):
        out = {
            "plane": cmds.nurbsPlane(name="plane", constructionHistory=False, u=3, v=2)[0],
            "sphere": cmds.sphere(name="ball", constructionHistory=False)[0],
            "cylinder": cmds.cylinder(name="cyl", constructionHistory=False)[0],
        }
        cmds.xform("ball", translation=(0, 3, 0))
        return {k: cmds.listRelatives(v, shapes=True)[0] for k, v in out.items()}

    def test_the_round_trip_in_each_format(self):
        for label, shape in self.sources().items():
            fn = NurbsSurface(shape).fn_set
            for ext in FORMATS:
                with self.subTest(label=label, ext=ext):
                    path = self.path(f"{label}.{ext}")
                    self.assertEqual(NurbsSurface(shape).save(path), path)
                    self.ready()
                    surface = NurbsSurface.load(path)
                    self.assertIsInstance(surface, NurbsSurface)
                    got = surface.fn_set
                    self.assertEqual(
                        (got.numCVsInU, got.numCVsInV, got.degreeInU, got.degreeInV, got.formInU, got.formInV),
                        (fn.numCVsInU, fn.numCVsInV, fn.degreeInU, fn.degreeInV, fn.formInU, fn.formInV),
                    )
                    np.testing.assert_allclose(_cvs(str(surface)), _cvs(shape), atol=1e-9)
                    self.assertEqual(cmds.listSets(object=str(surface)), ["initialShadingGroup"])
                    self.assert_kept_selection()
                    self.assertEqual(self.undo_name(), "rig.NurbsSurface.load")

    def test_names_flags_and_wrong_inputs(self):
        shape = self.sources()["plane"]
        NurbsSurface(shape).save(self.path("p.npz"))
        self.assertEqual(str(NurbsSurface.load(self.path("p.npz")).get_parent()), "surface1")
        self.assertEqual(str(NurbsSurface.load(self.path("p.npz"), name="patch").get_parent()), "patch")
        self.assertFalse(cmds.getAttr(f"{NurbsSurface.load(self.path('p.npz'), name='h', visibility=False)}.visibility"))
        NurbsCurve.create([(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)]).save(self.path("c.npz"))
        self.assert_refused(TypeError, "holds a BSplineData; the surface is a BSplinePatchData$",
                            lambda: NurbsSurface.load(self.path("c.npz")))
        self.assert_refused(FileNotFoundError, "NurbsSurface.load: the surface: no file",
                            lambda: NurbsSurface.load(self.path("none.npz")))

    def test_one_undo_step(self):
        shape = self.sources()["sphere"]
        NurbsSurface(shape).save(self.path("s.npz"))
        self.ready()
        NurbsSurface.load(self.path("s.npz"), name="copy")
        self.assert_one_step("rig.NurbsSurface.load", lambda: cmds.objExists("copy"))

    def test_the_node_form_updates_the_cvs(self):
        shape = self.sources()["sphere"]
        NurbsSurface(shape).save(self.path("s.npz"))
        source = _cvs(shape)
        copy = NurbsSurface.load(self.path("s.npz"), name="copy")
        cmds.xform("copy.cv[0][0:3]", translation=(0, 2, 0), relative=True)
        edited = _cvs(str(copy))
        self.ready()
        self.assertIs(copy.load(self.path("s.npz")), copy)
        np.testing.assert_allclose(_cvs(str(copy)), source, atol=1e-9)
        self.assertEqual(self.undo_name(), "rig.NurbsSurface.load")
        self.undo_steps(1)
        np.testing.assert_allclose(_cvs(str(copy)), edited, atol=1e-9)
        self.redo_steps(1)
        np.testing.assert_allclose(_cvs(str(copy)), source, atol=1e-9)

    def test_the_node_form_keeps_its_knots(self):
        # a BSplinePatchData holds no knots: an update keeps the surface's own
        shape = self.sources()["plane"]
        knots = np.array(NurbsSurface(shape).fn_set.knotsInU())
        NurbsSurface(shape).save(self.path("p.npz"))
        NurbsSurface(shape).load(self.path("p.npz"))
        np.testing.assert_array_equal(np.array(NurbsSurface(shape).fn_set.knotsInU()), knots)

    def test_the_node_form_refuses_another_layout(self):
        shapes = self.sources()
        NurbsSurface(shapes["sphere"]).save(self.path("s.npz"))
        q = repr(self.path("s.npz")).replace("\\", "\\\\")
        self.assert_refused(
            ValueError,
            f"^NurbsSurface\\('planeShape'\\).load: {q} holds another CV layout than 'planeShape' \\(7 x 11 CVs of "
            f"degrees 3 x 3 against its 6 x 5 of degrees 3 x 3, or other forms\\); a load updates the CVs of the "
            f"same layout: NurbsSurface.load\\({q}\\) makes a new surface$",
            lambda: NurbsSurface(shapes["plane"]).load(self.path("s.npz")),
        )


# ---------------------------------------------------------------------------------------------
class TestSkinCluster(_Case):
    """SkinCluster.save / SkinCluster.load(skin, geometry) / SkinCluster('x').load(skin)."""

    def bound(self):
        """j1 -> j2 and a cube skinned to them, the weights a gradient: ``(geo, skin, weights)``."""
        j1 = Joint.create(name="j1")
        j2 = Joint.create(name="j2", parent=j1)
        j2.t.set(0, 1, 0)
        geo = Node(cmds.polyCube(name="geo", constructionHistory=False, height=2, sy=4)[0])
        skin = SkinCluster.create(geo, [j1, j2])
        weights = skin.get_weights()
        weights[:, 0] = np.linspace(0, 1, len(weights))
        weights[:, 1] = 1 - weights[:, 0]
        skin.set_weights(weights)
        return geo, skin, skin.get_weights()

    def test_the_round_trip_in_each_format(self):
        geo, skin, weights = self.bound()
        for ext in FORMATS:
            with self.subTest(ext=ext):
                path = self.path(f"skin.{ext}")
                self.assertEqual(skin.save(path), path)
                cmds.delete(skin.name)
                self.ready()
                skin = SkinCluster.load(path, geo)
                self.assertIsInstance(skin, SkinCluster)
                np.testing.assert_array_equal(skin.get_weights(), weights)
                self.assertEqual([j.short_name for j in skin.get_influence_objects()], ["j1", "j2"])
                self.assertEqual(self.undo_name(), "rig.SkinCluster.load")

    def test_the_class_form_replaces_the_geometrys_own_skin(self):
        # D2: SkinCluster.load() creates, as create does: the geometry's own skin is deleted first
        geo, skin, weights = self.bound()
        skin.save(self.path("skin.npz"))
        old = cmds.ls("geo_skincluster", uuid=True)
        uuids = lambda: cmds.ls(cmds.ls(type="skinCluster"), uuid=True)  # noqa: E731
        self.ready()
        again = SkinCluster.load(self.path("skin.npz"), geo)
        new = cmds.ls("geo_skincluster", uuid=True)
        self.assert_kept_selection()
        self.assertEqual(cmds.ls(type="skinCluster"), ["geo_skincluster"])
        self.assertNotEqual(new, old)
        np.testing.assert_array_equal(again.get_weights(), weights)
        self.assertEqual(self.undo_name(), "rig.SkinCluster.load")
        self.undo_steps(1)
        self.assertEqual(uuids(), old)
        self.redo_steps(1)
        self.assertEqual(uuids(), new)
        np.testing.assert_array_equal(SkinCluster("geo_skincluster").get_weights(), weights)

    def test_flags_and_data_objects(self):
        geo, skin, weights = self.bound()
        data = skin.serialize()
        new = SkinCluster.load(data, geo, name="hero_skin", skinningMethod=1)
        self.assertEqual((str(new), cmds.getAttr("hero_skin.skinningMethod")), ("hero_skin", 1))
        np.testing.assert_array_equal(new.get_weights(), weights)
        skin_path = pathlib.Path(self.path("skin.json"))
        new.save(skin_path)
        self.assertIsInstance(SkinCluster.load(skin_path, "geo"), SkinCluster)

    def test_the_file_comes_first(self):
        geo, skin, _ = self.bound()
        skin.save(self.path("skin.npz"))
        q = repr(self.path("skin.npz")).replace("\\", "\\\\")
        for label, call, text in (
            ("a node first", lambda: SkinCluster.load(geo, self.path("skin.npz")),
             f"^SkinCluster.load\\(skin, geometry\\) takes the file first, then the geometry: got the node 'geo' "
             f"first and {q} second$"),
            ("a name first", lambda: SkinCluster.load("geo", self.path("skin.npz")),
             f"^SkinCluster.load\\(skin, geometry\\) takes the file first, then the geometry: got 'geo' first and {q} second$"),
            ("a Path second", lambda: SkinCluster.load("geo", pathlib.Path(self.path("skin.npz"))), "takes the file first"),
        ):
            with self.subTest(label):
                with mock.patch.object(_files, "read", wraps=_files.read) as read:
                    self.assert_refused(TypeError, text, call)
                    read.assert_not_called()

    def test_wrong_inputs_make_nothing(self):
        geo, skin, _ = self.bound()
        Mesh("geo").save(self.path("m.npz"))
        skin.save(self.path("skin.npz"))
        small = Node(cmds.polyCube(name="small", constructionHistory=False)[0])
        before = skin_state("geo_skincluster")
        for label, call, error, text in (
            ("a mesh file", lambda: SkinCluster.load(self.path("m.npz"), geo), TypeError, "holds a MeshData; the skin is a SkinData$"),
            ("a missing file", lambda: SkinCluster.load(self.path("x.npz"), geo), FileNotFoundError, "SkinCluster.load: the skin: no file"),
            ("rows against points", lambda: SkinCluster.load(self.path("skin.npz"), small), ValueError, "weights of shape"),
        ):
            with self.subTest(label):
                self.assert_refused(error, text, call)
        cmds.delete("j2")
        self.assert_refused(RuntimeError, r"Missing joints found: \['j2'\]", lambda: SkinCluster.load(self.path("skin.npz"), small))
        self.assertEqual(skin_state("geo_skincluster")["geometry"], before["geometry"])

    def test_the_node_form_updates_the_weights(self):
        geo, skin, weights = self.bound()
        skin.save(self.path("skin.npz"))
        skin.set_weights(np.full_like(weights, 0.5))
        edited = skin.get_weights()
        self.ready()
        self.assertIs(skin.load(self.path("skin.npz")), skin)
        np.testing.assert_array_equal(skin.get_weights(), weights)
        self.assertEqual(self.undo_name(), "rig.SkinCluster.load")
        self.undo_steps(1)
        np.testing.assert_array_equal(skin.get_weights(), edited)
        self.redo_steps(1)
        np.testing.assert_array_equal(skin.get_weights(), weights)
        self.assert_kept_selection()

    def test_the_node_form_adds_and_removes_influences_by_name(self):
        geo, skin, weights = self.bound()
        j3 = Joint.create(name="j3", parent="j2")
        j3.t.set(0, 1, 0)
        data = skin.serialize()
        data.influences = ["j1", "j3"]  # the file names j3, not j2
        skin.load(data)
        self.assertEqual([j.short_name for j in skin.get_influence_objects()], ["j1", "j3"])

    def test_the_node_form_takes_the_file_alone(self):
        geo, skin, _ = self.bound()
        skin.save(self.path("skin.npz"))
        self.assert_refused(
            TypeError,
            "^SkinCluster\\('geo_skincluster'\\).load updates its weights in place from the file alone \\(got "
            "Transform too\\); SkinCluster.load\\(path, geometry, ...\\) makes a new skinCluster$",
            lambda: skin.load(self.path("skin.npz"), geo),
        )
        self.assert_refused(TypeError, "\\(got name= too\\)", lambda: skin.load(self.path("skin.npz"), name="x"))


# ---------------------------------------------------------------------------------------------
class TestBlendShape(_Case):
    """BlendShape.save / BlendShape.load(targets, geometry) / BlendShape('x').load(targets)."""

    def shaped(self):
        """A moved base with two targets: ``(base, blendShape)``."""
        base = Node(cmds.polyCube(name="base", constructionHistory=False)[0])
        base.t << [5, 0, 0]
        smile = Node(cmds.polyCube(name="smile", constructionHistory=False)[0])
        cmds.xform("smile.vtx[3]", objectSpace=True, translation=(1, 1, 1), relative=True)
        frown = Node(cmds.polyCube(name="frown", constructionHistory=False)[0])
        cmds.xform("frown.vtx[5]", objectSpace=True, translation=(0, -1, 0), relative=True)
        bs = BlendShape.create(smile, frown, base, name="face_bs")
        return base, bs

    def offsets(self, bs):
        """The named targets' offsets. (The undo of a target set_target_data added leaves its
        weight element, unnamed: an older behaviour of set_target_data, left out here.)"""
        return {
            d.name: (np.asarray(d.indices).tolist(), np.round(np.asarray(d.offsets), 6).tolist())
            for d in bs.serialize() if not d.name.startswith("weight[")
        }

    def test_the_round_trip_on_a_moved_base(self):
        base, bs = self.shaped()
        want = {"smile": ([3], [[1.0, 1.0, 1.0]]), "frown": ([5], [[0.0, -1.0, 0.0]])}
        self.assertEqual(self.offsets(bs), want)
        for ext in FORMATS:
            with self.subTest(ext=ext):
                path = self.path(f"face.{ext}")
                self.assertEqual(bs.save(path), path)
                cmds.delete(bs.name)
                self.ready()
                bs = BlendShape.load(path, base, name="face_bs")
                self.assertEqual(self.undo_name(), "rig.BlendShape.load")
                self.assertEqual(bs.get_targets(), ["smile", "frown"])
                self.assertEqual(self.offsets(bs), want)
                bs.set_target_weight("smile", 1)
                np.testing.assert_allclose(np.array(base.get_shape().get_points(world_space=False))[3, :3], (1.5, 1.5, 1.5))

    def test_the_class_form_stacks_another(self):
        # D2: BlendShape.load() creates, as create does: a second blendShape on the base
        base, bs = self.shaped()
        bs.save(self.path("face.npz"))
        self.ready()
        second = str(BlendShape.load(self.path("face.npz"), base))
        self.assertEqual(sorted(cmds.ls(type="blendShape")), sorted(["face_bs", second]))
        self.assert_one_step("rig.BlendShape.load", lambda: cmds.objExists(second))
        self.assert_kept_selection()

    def test_the_data_object_is_not_changed(self):
        base, bs = self.shaped()
        data = bs.serialize()
        data[0].offsets = np.vstack([data[0].offsets, [[1e-9, 0, 0]]])
        data[0].indices = np.append(data[0].indices, 6)
        before = [(np.asarray(d.indices).copy(), np.asarray(d.offsets).copy()) for d in data]
        BlendShape.load(data, base)
        bs.load(data)
        for (indices, offsets), d in zip(before, data):
            np.testing.assert_array_equal(np.asarray(d.indices), indices)
            np.testing.assert_array_equal(np.asarray(d.offsets), offsets)

    def test_the_file_comes_first(self):
        base, bs = self.shaped()
        bs.save(self.path("face.npz"))
        self.assert_refused(TypeError, "^BlendShape.load\\(targets, geometry\\) takes the file first, then the "
                            "geometry: got the node 'base' first", lambda: BlendShape.load(base, self.path("face.npz")))
        Mesh("base").save(self.path("m.npz"))
        self.assert_refused(TypeError, "holds a MeshData; the targets is a MorphList or a MorphData$",
                            lambda: BlendShape.load(self.path("m.npz"), base))

    def test_the_node_form_sets_the_targets_by_name(self):
        base, bs = self.shaped()
        bs.save(self.path("face.npz"))
        from cgmath.geometry import MorphData, MorphList

        # the blendShape has changed: smile emptied, a third target the file does not name
        bs.set_target_data("smile", MorphData(name="smile", indices=np.array([0]), offsets=np.array([[0, 0, 2.0]])))
        bs.set_target_data("extra", MorphData(name="extra", indices=np.array([1]), offsets=np.array([[2.0, 0, 0]])))
        edited = self.offsets(bs)
        # the file also holds a new target
        data = MorphList.load(self.path("face.npz"))
        data.append(MorphData(name="brow", indices=np.array([2]), offsets=np.array([[0, 3.0, 0]])))
        data.save(self.path("face2.npz"))
        self.ready()
        self.assertIs(bs.load(self.path("face2.npz")), bs)
        self.assertEqual(bs.get_targets(), ["smile", "frown", "extra", "brow"])
        got = self.offsets(bs)
        self.assertEqual(got["smile"], ([3], [[1.0, 1.0, 1.0]]))
        self.assertEqual(got["extra"], edited["extra"])  # not in the file: kept
        self.assertEqual(got["brow"], ([2], [[0.0, 3.0, 0.0]]))
        self.assertEqual(self.undo_name(), "rig.BlendShape.load")
        self.undo_steps(1)
        self.assertEqual(self.offsets(bs), edited)
        self.redo_steps(1)
        self.assertEqual(self.offsets(bs), got)

    def test_the_node_form_takes_the_file_alone(self):
        base, bs = self.shaped()
        bs.save(self.path("face.npz"))
        self.assert_refused(TypeError, "^BlendShape\\('face_bs'\\).load updates its targets in place from the file "
                            "alone \\(got Transform too\\); BlendShape.load\\(path, geometry, ...\\) makes a new blendShape$",
                            lambda: bs.load(self.path("face.npz"), base))

    def test_save_checks_the_path_first(self):
        base, bs = self.shaped()
        with mock.patch.object(BlendShape, "serialize") as serialize:
            with self.assertRaisesRegex(ValueError, "^BlendShape.save: "):
                bs.save(self.path("face.obj"))
            serialize.assert_not_called()
        bs.save(self.path("one.npz"), match_name="smile")
        from cgmath.geometry import MorphList

        self.assertEqual([d.name for d in MorphList.load(self.path("one.npz"))], ["smile"])


# ---------------------------------------------------------------------------------------------
class TestHierarchy(_Case):
    """Transform.save_hierarchy / Transform.load_hierarchy (and Joint's): the class creates;
    a node's load_hierarchy is refused."""

    def test_the_round_trip_in_each_format(self):
        root = _skeleton()
        names = ["root", "spine", "tip", "grp"]
        matrices = _world_matrices(names)
        for ext in FORMATS:
            with self.subTest(ext=ext):
                path = self.path(f"skel.{ext}")
                self.assertEqual(root.save_hierarchy(path), path)
                cmds.delete("root")
                self.ready()
                nodes = Joint.load_hierarchy(path)
                self.assertIsInstance(nodes, List)
                self.assertEqual([n.short_name for n in nodes], ["root", "spine", "tip", "grp"])
                self.assertEqual([type(n).__name__ for n in nodes], ["Joint", "Joint", "Transform", "Transform"])
                self.assertEqual(nodes[1].long_name, "|root|spine")
                for name, matrix in _world_matrices(names).items():
                    np.testing.assert_allclose(matrix, matrices[name], atol=1e-9)
                self.assertEqual(cmds.getAttr("spine.twist"), 2.5)
                self.assertTrue(cmds.listRelatives("tip", shapes=True, type="locator"))
                self.assert_kept_selection()
                self.assertEqual(self.undo_name(), "rig.Joint.load_hierarchy")
                root = nodes[0]

    def test_one_undo_step_and_the_selection(self):
        root = _skeleton()
        root.save_hierarchy(self.path("skel.npz"))
        cmds.delete("root")
        self.ready()
        Transform.load_hierarchy(self.path("skel.npz"))
        self.assert_one_step("rig.Transform.load_hierarchy", lambda: cmds.objExists("root"))
        self.assert_kept_selection()
        # an empty selection stays empty
        cmds.select(clear=True)
        Transform.load_hierarchy(self.path("skel.npz"))
        self.assertEqual(cmds.ls(selection=True), [])

    def test_a_second_load_makes_another(self):
        root = _skeleton()
        root.save_hierarchy(self.path("skel.npz"))
        nodes = Joint.load_hierarchy(self.path("skel.npz"))
        self.assertEqual(nodes[0].short_name, "root1")
        self.assertEqual(nodes[1].long_name, "|root1|spine")

    def test_parent_and_world_space(self):
        root = _skeleton()
        root.save_hierarchy(self.path("skel.npz"))
        matrices = _world_matrices(["root", "spine"])
        rig_grp = Transform.create(name="rig_grp")
        rig_grp.t.set(10, 0, 0)
        kept = Transform.load_hierarchy(self.path("skel.npz"), parent="rig_grp")
        self.assertEqual(kept[0].long_name, "|rig_grp|root")
        np.testing.assert_allclose(_world_matrices([kept[1].long_name])["spine"], matrices["spine"], atol=1e-9)
        local = Transform.load_hierarchy(self.path("skel.npz"), parent=rig_grp, world_space=False)
        self.assertEqual(local[0].long_name, "|rig_grp|root1")
        self.assertEqual(cmds.getAttr(f"{local[0].long_name}.t")[0], (0.0, 0.0, 0.0))

    def test_data_objects_paths_and_a_clip(self):
        root = _skeleton()
        data = root.serialize_hierarchy()
        nodes = Transform.load_hierarchy(data)
        self.assertEqual(nodes[0].short_name, "root1")
        root.save_hierarchy(pathlib.Path(self.path("skel.npz")))
        self.assertEqual(Transform.load_hierarchy(pathlib.Path(self.path("skel.npz")))[0].short_name, "root2")
        from cgmath.hierarchy import ClipData

        clip = ClipData(data, frames=3)
        self.assertEqual(Transform.load_hierarchy(clip)[0].short_name, "root3")

    def test_wrong_inputs_make_nothing(self):
        Mesh(cmds.polyCube(name="m", constructionHistory=False)[0]).save(self.path("m.npz"))
        for label, call, error, text in (
            ("a mesh file", lambda: Transform.load_hierarchy(self.path("m.npz")), TypeError,
             "^Transform.load_hierarchy: the hierarchy: .* holds a MeshData; the hierarchy is a HierarchyData$"),
            ("a missing file", lambda: Joint.load_hierarchy(self.path("x.npz")), FileNotFoundError,
             "^Joint.load_hierarchy: the hierarchy: no file"),
        ):
            with self.subTest(label):
                self.assert_refused(error, text, call)

    def test_a_node_s_load_hierarchy_is_refused_before_any_read(self):
        root = _skeleton()
        root.save_hierarchy(self.path("skel.npz"))
        for node, name in ((root, "Joint"), (Transform("grp"), "Transform")):
            with self.subTest(name):
                with mock.patch.object(_files, "read") as read:
                    self.assert_refused(
                        TypeError,
                        f"^{name}\\('{node.name}'\\).load_hierarchy\\(...\\) would update the hierarchy at "
                        f"'{node.name}' in place, which rig does not do \\(its transforms hold the rig's "
                        f"constraints, skins and children\\); {name}.load_hierarchy\\(path, parent='{node.name}'\\) "
                        f"makes a new one under it$",
                        lambda: node.load_hierarchy(self.path("skel.npz")),
                    )
                    read.assert_not_called()

    def test_container(self):
        root = _skeleton()
        root.save_hierarchy(self.path("skel.npz"))
        cmds.delete("root")
        with container("box") as box:
            Joint.load_hierarchy(self.path("skel.npz"))
        members = set(cmds.container(str(box), query=True, nodeList=True))
        self.assertTrue({"root", "spine", "tip", "tipShape", "grp"} <= members, members)

    def test_save_hierarchy_checks_first_and_takes_bindpose(self):
        root = _skeleton()
        with mock.patch.object(Transform, "serialize_hierarchy") as serialize:
            with self.assertRaisesRegex(ValueError, "^Joint.save_hierarchy: .*cgmath saves .npz"):
                root.save_hierarchy(self.path("skel.txt"))
            serialize.assert_not_called()
        with self.assertRaisesRegex(TypeError, "^bindpose= takes False, True or a dagPose name, not None$"):
            root.save_hierarchy(self.path("skel.npz"), bindpose=None)
        # at the bind pose: the skin bound at rest, then posed
        geo = Node(cmds.polyCube(name="geo", constructionHistory=False, height=6, sy=6)[0])
        SkinCluster.create(geo, ["root", "spine"])
        rest = root.serialize_hierarchy()
        cmds.setAttr("spine.rz", 70)
        self.ready()
        root.save_hierarchy(self.path("bind.npz"), bindpose=True)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        from cgmath.hierarchy import HierarchyData

        saved = HierarchyData.load(self.path("bind.npz"))
        np.testing.assert_allclose(saved["spine"].rotate, rest["spine"].rotate, atol=1e-6)
        self.assertEqual(cmds.getAttr("spine.rz"), 70)


# ---------------------------------------------------------------------------------------------
class TestStatements(MayaTestCase):
    """Each load is a journal statement: its API edits join its one undo step."""

    def test_every_load_is_a_statement(self):
        for fn in (
            Mesh.load, NurbsCurve.load, NurbsSurface.load, SkinCluster.load, BlendShape.load,
            Transform.load_hierarchy, Joint.load_hierarchy,
        ):
            with self.subTest(fn.__qualname__):
                self.assertIs(fn.__func__._rig_statement, True)

    def test_a_load_s_edits_are_registered_once(self):
        cmds.file(new=True, force=True)
        cmds.polyCube(name="c", constructionHistory=False)
        directory = tempfile.mkdtemp()
        try:
            path = Mesh("c").save(os.path.join(directory, "c.npz"))
            registered = []
            real = journal._register

            def spy(ops):
                registered.append(len(ops))
                return real(ops)

            with mock.patch.object(journal, "_register", spy):
                Mesh.load(path, name="m", visibility=False)
            self.assertLessEqual(len(registered), 1)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
