"""Follow-up of the default-normals round (the user's decisions of 2026-11-11): a mesh whose
points give its normals stores none.

``Mesh.serialize`` and ``save`` write the fewest fields that say a mesh's shading, of the three
forms a cgmath MeshData takes:

* nothing, with no hard edge between two faces: the normals are the ones the points give;
* ``hard_edges`` alone: the two vertices of each hard edge that two or more faces share. A
  border's flag shades nothing and is left out (decision 7);
* ``normals`` / ``normal_indices`` alone (decision 5), where the points do not give them: a
  locked normal, also one at the value Maya computes (decision 2), a ``vertexNormalMethod``
  other than Maya's default, ``world_space=True`` through more than a rotation and one scale.

``Mesh.create`` and ``load`` bring the first two back as edges: nothing is locked, the normals
are Maya's own. Stored normals win over a ``hard_edges`` beside them, which is then not read
(decision 6); a pair that is no edge of the faces is a RuntimeError before anything is made. In
place (``Mesh('x').load``) the data's edges replace the mesh's own, its locks released
(decision 3). ``normals=False`` writes no shading and reads none.

Through a matrix (``world_space=True``) data that stores no normals locks nothing, but through
a mirror: Maya's normals of mirrored points face the other way, so the ones the points give
are set.

Each of the 26 tests fails at 3bd78fd (negative control, the module run on that tree: every
mesh stored its normals, and ``hard_edges`` was not read).
"""

import json
import re
from unittest import mock

import numpy as np
from cgmath.geometry import MeshData
from maya import cmds
from maya.api import OpenMaya

from rig.nodetypes import Mesh
from rig.nodetypes import mesh as mesh_module
from rig._tests._undo import mesh_state
from rig._tests.test_class_mesh_create import _grid, _shape, _stored
from rig._tests.test_class_saveload import FORMATS, _Case, _fv_normals, _hard_edges, _locked_rows, _world_points
from rig._tests.test_class_shape_under_transform import _sheared

WORLD   = OpenMaya.MSpace.kWorld

NO_EDGE = "hard edges name two vertices that no face lists as an edge"


def _form(data):
    """The shape of each shading field `data` holds, None for one it does not:
    ``(normals, normal_indices, hard_edges)``."""
    return tuple(
        None if field is None else np.asarray(field).shape
        for field in (data.normals, data.normal_indices, data.hard_edges)
    )


def _shared_hard(shape):
    """The hard edges of `shape` that two or more faces share, as ``hard_edges`` holds them:
    ``[lower vertex, higher vertex]`` rows, sorted."""
    fn   = Mesh(shape).get_fn_set()
    edge = OpenMaya.MItMeshEdge(fn.object())
    rows = []
    while not edge.isDone():
        if not edge.isSmooth and not edge.onBoundary():
            rows.append(sorted((edge.vertexId(0), edge.vertexId(1))))
        edge.next()
    return sorted(rows)


def _locks(shape):
    """``(locked, all)``: how many normals of `shape` are locked, and how many it has."""
    fn = Mesh(shape).get_fn_set()
    return sum(map(fn.isNormalLocked, range(fn.numNormals))), fn.numNormals


def _stored_normals(data):
    """The normal `data` stores for each face-vertex."""
    return np.asarray(data.normals)[np.asarray(data.normal_indices)]


def _cube(name="cube"):
    return cmds.polyCube(name=name, constructionHistory=False)[0]


def _box(name="box"):
    """A 2 x 2 x 2 polyCube (26 vertices, 24 hard edges, 54 normals), without history: the
    transform."""
    return cmds.polyCube(name=name, subdivisionsX=2, subdivisionsY=2, subdivisionsZ=2, constructionHistory=False)[0]


def _soft(name="soft", sx=8, sy=6):
    """A polySphere, every edge soft, without history: the transform."""
    t = cmds.polySphere(name=name, sx=sx, sy=sy, constructionHistory=False)[0]
    cmds.polySoftEdge(t, angle=180, constructionHistory=False)
    return t


def _edged(name="src"):
    """A polySphere (8 x 6), soft but for its first 8 edges, no normal locked, without history:
    the transform."""
    t = _soft(name)
    cmds.polySoftEdge(f"{t}.e[0:7]", angle=0, constructionHistory=False)
    return t


def _two_cubes(name="two"):
    """Two cubes merged along one edge, which four faces share (14 vertices, 23 edges, 12
    faces): the transform."""
    a = _cube("left")
    b = _cube("right")
    cmds.move(1, 1, 0, b)
    t = cmds.polyUnite(a, b, name=name, constructionHistory=False)[0]
    cmds.polyMergeVertex(t, distance=0.001, constructionHistory=False)
    return t


class TestSerializeWritesTheLeast(_Case):
    """``serialize`` and ``save``: the fewest fields that say the mesh's shading."""

    def test_a_soft_mesh_stores_no_shading(self):
        t    = _soft(sx=32, sy=32)
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(_form(data), (None, None, None))
        # the normals its points give are Maya's
        np.testing.assert_allclose(data.get_face_vertex_normals(), _fv_normals(_shape(t)), atol=1e-5)

    def test_a_hard_edge_is_its_two_vertices(self):
        for label, make in (
            ("cube", lambda: _cube()),
            ("cylinder", lambda: cmds.polyCylinder(name="cylinder", constructionHistory=False)[0]),
            ("sphere", lambda: cmds.polySphere(name="ball", sx=20, sy=10, constructionHistory=False)[0]),
        ):
            with self.subTest(label):
                shape = _shape(make())
                rows  = _shared_hard(shape)
                data  = Mesh(shape).serialize(uvs=False)
                self.assertTrue(rows)
                self.assertEqual(_form(data), (None, None, (len(rows), 2)))
                self.assertEqual(data.hard_edges.dtype, np.int32)
                self.assertEqual(data.hard_edges.tolist(), rows)
                np.testing.assert_allclose(data.get_face_vertex_normals(), _fv_normals(shape), atol=1e-5)
        self.assertEqual(len(Mesh("cube").serialize(uvs=False).hard_edges), 12)

    def test_a_border_is_left_out(self):
        """Decision 7. Maya flags the 16 border edges of a stock plane hard, which shades nothing:
        none is written. A hard edge between two faces of one plane is, though it splits no
        normal: it is the mesh's."""
        t     = cmds.polyPlane(name="plane", sx=4, sy=4, constructionHistory=False)[0]
        shape = _shape(t)
        fn    = Mesh(shape).get_fn_set()
        inner = [e for e in range(fn.numEdges) if fn.isEdgeSmooth(e)]
        self.assertEqual((fn.numEdges, len(inner), len(_shared_hard(shape))), (40, 24, 0))
        self.assertEqual(_form(Mesh(t).serialize(uvs=False)), (None, None, None))
        pair = sorted(fn.getEdgeVertices(inner[3]))
        cmds.polySoftEdge(f"{t}.e[{inner[3]}]", angle=0, constructionHistory=False)
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(len(_hard_edges(shape)), 17)
        self.assertEqual((pair, data.hard_edges.tolist()), ([5, 6], [[5, 6]]))

    def test_a_locked_normal_stores_them(self):
        """Decision 2: a lock means stored, also one at the value Maya computes."""
        t      = _cube()
        shape  = _shape(t)
        before = _fv_normals(shape)
        cmds.polyNormalPerVertex(f"{t}.vtx[0]", freezeNormal=True)
        cmds.delete(t, constructionHistory=True)
        self.assertEqual(_locks(shape), (3, 24))
        np.testing.assert_array_equal(_fv_normals(shape), before)
        data = Mesh(t).serialize(uvs=False)
        # decision 5: the normals only, no hard edges beside them
        self.assertEqual(_form(data), ((24, 3), (24,), None))
        np.testing.assert_allclose(_stored_normals(data), before, atol=1e-6)
        cmds.polyNormalPerVertex(f"{t}.vtx[0]", unFreezeNormal=True)
        cmds.delete(t, constructionHistory=True)
        self.assertEqual(_locks(shape), (0, 24))
        self.assertEqual(_form(Mesh(t).serialize(uvs=False)), (None, None, (12, 2)))

    def test_another_normal_method_stores_them(self):
        """cgmath derives Maya's default, angle and area weighted (3): a shape on another
        ``vertexNormalMethod`` holds normals its points do not give there."""
        t = _cube()
        for method in (0, 1, 2, 3):
            with self.subTest(method=method):
                cmds.setAttr(f"{_shape(t)}.vertexNormalMethod", method)
                want = (None, None, (12, 2)) if method == 3 else ((24, 3), (24,), None)
                self.assertEqual(_form(Mesh(t).serialize(uvs=False)), want)

    def test_world_space_stores_them_through_more_than_a_rotation_and_one_scale(self):
        t     = _cube()
        shape = _shape(t)
        for scale, stored in (
            ((1, 1, 1), False),
            ((2.5, 2.5, 2.5), False),
            ((1e-3, 1e-3, 1e-3), False),
            ((1e4, 1e4, 1e4), False),
            ((-1, -1, 1), False),  # two mirrors: a rotation
            ((1, 2, 3), True),
            ((-1, 1, 1), True),
            ((-1, -1, -1), True),
        ):
            with self.subTest(scale=scale):
                cmds.xform(t, translation=(1, 2, 3), rotation=(10, 20, 30), scale=scale)
                self.assert_world_form(shape, stored)
        # a shear: under a parent scaled unevenly
        cmds.xform(t, translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
        cmds.parent(t, _sheared(), relative=True)
        self.assert_world_form(cmds.ls(type="mesh", long=True)[0], True)

    def assert_world_form(self, shape, stored):
        """The cube `shape` serializes its world normals when `stored`, else its 12 hard edges,
        and in object space its hard edges, whatever its transform."""
        world = Mesh(shape).serialize(uvs=False, world_space=True)
        maya  = _fv_normals(shape, WORLD)
        if stored:
            self.assertEqual(_form(world), ((24, 3), (24,), None))
            np.testing.assert_allclose(_stored_normals(world), maya, atol=1e-6)
        else:
            self.assertEqual(_form(world), (None, None, (12, 2)))
            # the normals the world points give are Maya's world normals
            np.testing.assert_allclose(world.get_face_vertex_normals(), maya, atol=1e-5)
        self.assertEqual(_form(Mesh(shape).serialize(uvs=False)), (None, None, (12, 2)))

    def test_normals_false_writes_no_shading(self):
        """The flag leaves the shading out whole, the hard edges with the normals, and reads
        none of it."""
        t = _cube()
        for label, prepare in (
            ("hard edges", lambda: None),
            ("a lock", lambda: cmds.polyNormalPerVertex(f"{t}.vtx[0]", xyz=(0.0, 1.0, 0.0))),
        ):
            with self.subTest(label):
                prepare()
                with mock.patch.object(mesh_module, "_shading_fields", wraps=mesh_module._shading_fields) as fields:
                    data = Mesh(t).serialize(uvs=False, normals=False)
                fields.assert_not_called()
                self.assertEqual(_form(data), (None, None, None))

    def test_an_edge_of_more_than_two_faces(self):
        t     = _two_cubes()
        shape = _shape(t)
        fn    = Mesh(shape).get_fn_set()
        # 16 vertices and 24 edges less the merged ones: one edge holds four faces
        self.assertEqual((fn.numVertices, fn.numEdges, fn.numPolygons), (14, 23, 12))
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(_form(data), (None, None, (23, 2)))
        self.assertEqual(data.hard_edges.tolist(), _shared_hard(shape))
        made = Mesh.create(data, name="made", uuids=False)
        self.assertEqual(_shared_hard(str(made)), _shared_hard(shape))
        self.assertEqual(_locks(str(made))[0], 0)
        np.testing.assert_allclose(_fv_normals(str(made)), _fv_normals(shape), atol=1e-6)

    def test_the_files_hold_the_same(self):
        soft, edged = _soft(), _edged()
        rows = _shared_hard(_shape(edged))
        self.assertEqual(len(rows), 8)
        for ext in FORMATS:
            with self.subTest(ext=ext):
                back = MeshData.load(Mesh(soft).save(self.path(f"soft.{ext}")))
                self.assertEqual(_form(back), (None, None, None))
                path = Mesh(edged).save(self.path(f"edged.{ext}"))
                back = MeshData.load(path)
                self.assertEqual(_form(back), (None, None, (8, 2)))
                self.assertEqual(back.hard_edges.tolist(), rows)
                made = Mesh.load(path, name=f"made_{ext}", uuids=False)
                self.assertEqual(_shared_hard(str(made)), rows)
                self.assertEqual(_locks(str(made))[0], 0)
        # a field the mesh does not need is not in its file
        fields = {"normals", "normal_indices", "hard_edges"}
        for name, held in (("soft", set()), ("edged", {"hard_edges"})):
            with open(self.path(f"{name}.json"), encoding="utf-8") as handle:
                self.assertEqual(fields & set(json.load(handle)), held, name)


class TestCreateFromHardEdges(_Case):
    """``create``: data that stores no normals comes back as edges, and nothing is locked."""

    def test_the_edges_come_back_and_nothing_locks(self):
        t     = _edged()
        shape = _shape(t)
        data, uvs = Mesh(t).serialize()
        self.assertEqual(_form(data), (None, None, (8, 2)))
        self.ready()
        with mock.patch.object(mesh_module, "_lock_differing", wraps=mesh_module._lock_differing) as lock:
            made = Mesh.create(data, uvs=uvs, name="made", uuids=False)
        lock.assert_not_called()
        self.assertEqual(_hard_edges(str(made)), _hard_edges(shape))
        self.assertEqual(_locks(str(made))[0], 0)
        np.testing.assert_allclose(_fv_normals(str(made)), _fv_normals(shape), atol=1e-6)
        # and serializes as its source did
        again = made.serialize(uvs=False)
        self.assertEqual(_form(again), (None, None, (8, 2)))
        self.assertEqual(again.hard_edges.tolist(), data.hard_edges.tolist())
        self.assert_kept_selection()
        self.assert_one_step("rig.Mesh.create", lambda: cmds.objExists("made"))

    def test_data_that_names_none_is_all_soft(self):
        t               = _cube()
        data            = Mesh(t).serialize(uvs=False)
        data.hard_edges = None
        made            = Mesh.create(data, name="made", uuids=False)
        self.assertEqual((_hard_edges(str(made)), _locks(str(made))[0]), (set(), 0))
        cmds.polySoftEdge(t, angle=180, constructionHistory=False)
        np.testing.assert_allclose(_fv_normals(str(made)), _fv_normals(_shape(t)), atol=1e-6)

    def test_pairs_as_they_come(self):
        """cgmath sorts the pairs it is assigned. Data made another way may hold them in any
        order, either vertex first, an edge twice, in any integer type."""
        data = Mesh(_cube()).serialize(uvs=False)
        want = {frozenset((0, 1)), frozenset((2, 3))}
        for i, (label, pairs) in enumerate((
            ("int64, turned, one twice", np.array([[3, 2], [1, 0], [0, 1]], dtype=np.int64)),
            ("uint8", np.array([[0, 1], [2, 3]], dtype=np.uint8)),
            ("a list", [[2, 3], [0, 1]]),
        )):
            with self.subTest(label):
                object.__setattr__(data, "hard_edges", pairs)
                made = Mesh.create(data, name=f"made{i}", uuids=False)
                self.assertEqual(_hard_edges(str(made)), want)
                self.assertEqual(_locks(str(made))[0], 0)

    def test_a_border_pair_is_taken_and_shades_nothing(self):
        t    = cmds.polyPlane(name="plane", sx=4, sy=4, constructionHistory=False)[0]
        bare = Mesh(t).serialize(uvs=False)
        self.assertIsNone(bare.hard_edges)
        soft = Mesh.create(bare, name="soft", uuids=False)
        self.assertEqual(_hard_edges(str(soft)), set())
        data            = bare.copy()
        data.hard_edges = np.array([[0, 1]], dtype=np.int32)
        made            = Mesh.create(data, name="made", uuids=False)
        # Maya then flags every border (``cleanupEdgeSmoothing``), as it does a stock plane's
        self.assertIn(frozenset((0, 1)), _hard_edges(str(made)))
        self.assertEqual(_locks(str(made))[0], 0)
        np.testing.assert_array_equal(_fv_normals(str(made)), _fv_normals(str(soft)))
        self.assertEqual(_form(made.serialize(uvs=False)), (None, None, None))

    def test_stored_normals_win(self):
        """Decision 6: beside stored normals the hard edges are not read, whatever they name."""
        t = _edged()
        with _stored():
            data = Mesh(t).serialize(uvs=False)
        self.assertEqual(_form(data), ((50, 3), (176,), None))
        for i, (label, pairs) in enumerate((("a soft edge", [[8, 9]]), ("no edge", [[0, 30]]))):
            with self.subTest(label):
                both            = data.copy()
                both.hard_edges = np.array(pairs, dtype=np.int32)
                self.assertEqual(_form(both), ((50, 3), (176,), (1, 2)))
                with mock.patch.object(mesh_module, "_data_hard_edges", wraps=mesh_module._data_hard_edges) as read:
                    made = Mesh.create(both, name=f"made{i}", uuids=False)
                read.assert_not_called()
                self.assertEqual(_hard_edges(str(made)), _hard_edges(_shape(t)))
                self.assertEqual(_locks(str(made))[0], 0)

    def test_a_pair_that_is_no_edge_is_refused(self):
        """Before anything is made. A cube's vertices 0 and 7, 1 and 6 share no edge."""
        data = Mesh(_cube()).serialize(uvs=False)
        self.ready()
        for pairs, said in (
            ([[0, 7]], f"1 of the 1 {NO_EDGE} (the first: 0 and 7)"),
            ([[1, 6], [0, 1], [0, 7]], f"2 of the 3 {NO_EDGE} (the first: 0 and 7)"),
            ([[0, 999]], f"1 of the 1 {NO_EDGE} (the first: 0 and 999)"),
            ([[-1, 0]], f"1 of the 1 {NO_EDGE} (the first: -1 and 0)"),
        ):
            with self.subTest(pairs=pairs):
                bad            = data.copy()
                bad.hard_edges = np.array(pairs, dtype=np.int32)
                self.assert_refused(
                    RuntimeError, f"^MeshData: {re.escape(said)}$", lambda: Mesh.create(bad, name="bad", uuids=False)
                )
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_pairs_of_another_shape_are_refused(self):
        """cgmath refuses them where they are assigned; data made another way meets rig's check."""
        data = Mesh(_cube()).serialize(uvs=False)
        self.ready()
        for pairs, said in (
            (np.array([[0.0, 1.0], [2.0, 3.0]]), "float64 (2, 2)"),
            (np.array([0, 1, 2], dtype=np.int32), "int32 (3,)"),
            (np.array([[0, 1, 3], [2, 3, 5]], dtype=np.int32), "int32 (2, 3)"),
        ):
            with self.subTest(said):
                bad = data.copy()
                object.__setattr__(bad, "hard_edges", pairs)
                self.assert_refused(
                    RuntimeError,
                    f"^MeshData: the hard edges are {re.escape(said)}, not \\(N, 2\\) vertex indices$",
                    lambda: Mesh.create(bad, name="bad", uuids=False),
                )
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_normals_false_reads_no_hard_edges(self):
        """The restore flag skips the data's shading whole: hard edges that fit no face are
        not read."""
        data = Mesh(_cube()).serialize(uvs=False)
        self.assertEqual(len(data.hard_edges), 12)
        with mock.patch.object(mesh_module, "_data_hard_edges", wraps=mesh_module._data_hard_edges) as read:
            made = Mesh.create(data, name="made", normals=False, uuids=False)
        read.assert_not_called()
        self.assertEqual(_hard_edges(str(made)), set())
        object.__setattr__(data, "hard_edges", np.array([0, 1, 2], dtype=np.int32))
        flat = Mesh.create(data, name="flat", normals=False, uuids=False)
        self.assertEqual(_hard_edges(str(flat)), set())

    def test_a_holed_mesh(self):
        t     = _grid()
        shape = _shape(t)
        # flat: its border and its holes' edges are all it flags
        self.assertEqual(_form(Mesh(t).serialize(uvs=False)), (None, None, None))
        cmds.move(0, 0.7, 0, f"{t}.vtx[0:3]", relative=True)
        cmds.polySoftEdge(f"{t}.e[0:6]", angle=0, constructionHistory=False)
        cmds.delete(t, constructionHistory=True)
        data, uvs = Mesh(t).serialize()
        self.assertEqual(data.hard_edges.tolist(), [[1, 5], [2, 6]])
        self.assertEqual(data.hard_edges.tolist(), _shared_hard(shape))
        made = Mesh.create(data, uvs=uvs, name="made", uuids=False)
        self.assertEqual(_shared_hard(str(made)), [[1, 5], [2, 6]])
        self.assertEqual(_locks(str(made))[0], 0)
        np.testing.assert_allclose(_fv_normals(str(made)), _fv_normals(shape), atol=1e-6)
        # an edge of a hole is an edge of the faces
        hole            = data.copy()
        hole.hard_edges = np.array([[16, 17]], dtype=np.int32)
        self.assertIn(frozenset((16, 17)), _hard_edges(str(Mesh.create(hole, uvs=uvs, name="hole", uuids=False))))
        # a diagonal of a face is none
        bad            = data.copy()
        bad.hard_edges = np.array([[0, 5]], dtype=np.int32)
        self.assert_refused(
            RuntimeError,
            f"^MeshData: {re.escape(f'1 of the 1 {NO_EDGE} (the first: 0 and 5)')}$",
            lambda: Mesh.create(bad, uvs=uvs, name="bad", uuids=False),
        )


class TestThroughAMatrix(_Case):
    """``world_space=True`` takes the data's points through a matrix. Data that stores no normals
    has none to take with them: its edges come through, and the normals are Maya's own where
    the points land. But through a mirror, where those face the other way."""

    def source(self):
        """``(shape, data)`` of a box with 20 hard edges, moved, rotated and scaled unevenly,
        and its data: no normals."""
        t = _box("src")
        cmds.polySoftEdge(f"{t}.e[0:5]", angle=180, constructionHistory=False)
        cmds.xform(t, translation=(4, 5, 6), rotation=(10, 20, 30), scale=(2, 1, 0.5))
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(_form(data), (None, None, (20, 2)))
        return _shape(t), data

    def test_a_mirror_sets_the_normals_the_points_give(self):
        src = _box("src")
        cmds.xform(src, translation=(2, 0, 0), scale=(-1, 1, 1))
        shape = _shape(src)
        data  = Mesh(src).serialize(uvs=False)
        self.assertEqual(_form(data), (None, None, (24, 2)))
        # frozen: every normal is set, and the world normals are the source's
        made = Mesh.create(data, name="frozen", world_space=True, uuids=False)
        self.assertEqual(_locks(str(made)), (54, 54))
        self.assertEqual(_shared_hard(str(made)), _shared_hard(shape))
        np.testing.assert_allclose(_fv_normals(str(made), WORLD), _fv_normals(shape, WORLD), atol=1e-6)
        np.testing.assert_allclose(_world_points(str(made)), _world_points(shape), atol=1e-6)
        # placed (the default): the points are the mesh's own under the mirrored transform
        placed = Mesh.create(data, name="placed", uuids=False)
        self.assertEqual(_locks(str(placed))[0], 0)
        np.testing.assert_allclose(_fv_normals(str(placed), WORLD), _fv_normals(shape, WORLD), atol=1e-6)

    def test_a_mirror_sets_them_for_data_with_no_hard_edge_too(self):
        src = _box("src")
        cmds.xform(src, translation=(2, 0, 0), scale=(-1, 1, 1))
        data            = Mesh(src).serialize(uvs=False)
        data.hard_edges = None
        made            = Mesh.create(data, name="frozen", world_space=True, uuids=False)
        self.assertEqual(_locks(str(made)), (26, 26))
        self.assertEqual(_shared_hard(str(made)), [])
        cmds.polySoftEdge(src, angle=180, constructionHistory=False)
        np.testing.assert_allclose(_fv_normals(str(made), WORLD), _fv_normals(_shape(src), WORLD), atol=1e-6)

    def test_under_a_mirrored_target(self):
        src = _cube("src")
        cmds.xform(src, translation=(1, 2, 3), rotation=(10, 20, 30))
        data = Mesh(src).serialize(uvs=False)
        self.assertEqual(_form(data), (None, None, (12, 2)))
        target = cmds.createNode("transform", name="mirror")
        cmds.xform(target, translation=(3, 0, 0), rotation=(0, 30, 0), scale=(-1, 1, 1))
        made = Mesh.create(data, transform=target, world_space=True, uuids=False)
        self.assertEqual(_locks(str(made)), (24, 24))
        self.assertEqual(_shared_hard(str(made)), _shared_hard(_shape(src)))
        np.testing.assert_allclose(_fv_normals(str(made), WORLD), _fv_normals(_shape(src), WORLD), atol=1e-6)
        np.testing.assert_allclose(_world_points(str(made)), _world_points(_shape(src)), atol=1e-6)

    def test_any_other_matrix_locks_nothing(self):
        shape, data = self.source()
        with _stored():
            stored = Mesh(shape).serialize(uvs=False)
        self.assertEqual(_form(stored), ((48, 3), (96,), None))
        for label, least_to, stored_to in (
            ("frozen", {"name": "least"}, {"name": "stored"}),
            ("under a sheared target", {"transform": _sheared("a")}, {"transform": _sheared("b")}),
        ):
            with self.subTest(label):
                with mock.patch.object(mesh_module, "_lock_differing", wraps=mesh_module._lock_differing) as lock:
                    made = Mesh.create(data, world_space=True, uuids=False, **least_to)
                lock.assert_not_called()
                self.assertEqual(_locks(str(made))[0], 0)
                self.assertEqual(_shared_hard(str(made)), _shared_hard(shape))
                np.testing.assert_allclose(_world_points(str(made)), _world_points(shape), atol=1e-5)
                # as the stored normals of a mesh that locks none land: Maya's own of the points
                old = Mesh.create(stored, world_space=True, uuids=False, **stored_to)
                self.assertEqual(_locks(str(old))[0], 0)
                self.assertEqual(_shared_hard(str(old)), _shared_hard(shape))
                np.testing.assert_allclose(_fv_normals(str(made), WORLD), _fv_normals(str(old), WORLD), atol=1e-6)

    def test_an_update_through_a_matrix_locks_nothing(self):
        shape, data = self.source()
        dst = cmds.duplicate(shape.rsplit("|", 1)[0], name="dst")[0]
        cmds.xform(dst, translation=(0, 0, 0), rotation=(50, 0, 10), scale=(1, 3, 1))
        cmds.polySoftEdge(dst, angle=0, constructionHistory=False)
        self.assertEqual(len(_shared_hard(_shape(dst))), 48)
        Mesh(dst).load(data, world_space=True)
        self.assertEqual(_locks(_shape(dst))[0], 0)
        self.assertEqual(_shared_hard(_shape(dst)), _shared_hard(shape))
        np.testing.assert_allclose(_world_points(_shape(dst)), _world_points(shape), atol=1e-5)


class TestInPlace(_Case):
    """``Mesh('x').load``: the data's shading replaces the mesh's own (decision 3)."""

    def box(self):
        """``(shape, data)``: a cube with a locked normal (12 hard edges, 3 locked face-vertices),
        and its data as it serialized before the lock: no normals."""
        t    = _cube("box")
        data = Mesh(t).serialize(uvs=False)
        cmds.polyNormalPerVertex(f"{t}.vtx[5]", xyz=(1.0, 0.0, 0.0))
        cmds.delete(t, constructionHistory=True)
        shape = _shape(t)
        self.assertEqual((_form(data), len(_hard_edges(shape)), len(_locked_rows(shape))), ((None, None, (12, 2)), 12, 3))
        return shape, data

    def test_the_data_s_edges_replace_the_mesh_s_own(self):
        shape, data     = self.box()
        data.hard_edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        two             = {frozenset((0, 1)), frozenset((2, 3))}
        before          = mesh_state(shape)
        self.ready()
        mesh = Mesh(shape)
        self.assertIs(mesh.load(data), mesh)
        self.assertEqual((_hard_edges(shape), _locked_rows(shape)), (two, set()))
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        self.assert_kept_selection()
        self.undo_steps(1)
        self.assertEqual((len(_hard_edges(shape)), len(_locked_rows(shape))), (12, 3))
        self.assertEqual(mesh_state(shape), before)
        self.redo_steps(1)
        self.assertEqual((_hard_edges(shape), _locked_rows(shape)), (two, set()))
        # data that names none: every edge soft
        data.hard_edges = None
        mesh.load(data)
        self.assertEqual((_hard_edges(shape), _locked_rows(shape)), (set(), set()))

    def test_normals_false_keeps_the_mesh_s_own(self):
        shape, data     = self.box()
        own             = _hard_edges(shape), _locked_rows(shape)
        data.hard_edges = np.array([[0, 1]], dtype=np.int32)
        with mock.patch.object(mesh_module, "_data_hard_edges", wraps=mesh_module._data_hard_edges) as read:
            Mesh(shape).load(data, normals=False)
        read.assert_not_called()
        self.assertEqual((_hard_edges(shape), _locked_rows(shape)), own)

    def test_a_pair_that_is_no_edge_changes_nothing(self):
        shape, data     = self.box()
        data.hard_edges = np.array([[0, 7]], dtype=np.int32)
        before          = mesh_state(shape), _hard_edges(shape), _locked_rows(shape)
        self.ready()
        self.assert_refused(
            RuntimeError,
            f"^MeshData: {re.escape(f'1 of the 1 {NO_EDGE} (the first: 0 and 7)')}$",
            lambda: Mesh(shape).load(data),
        )
        self.assertEqual((mesh_state(shape), _hard_edges(shape), _locked_rows(shape)), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
