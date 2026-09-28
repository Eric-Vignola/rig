"""``Mesh.create`` rides a recorded ``createNode`` (round U, step U1).

The whole create is one undo step named ``rig.Mesh.create``: a
``cmds.createNode`` transform, the shape made under it through the API
(``MFnMesh.create(parent=)``), the UV sets through the API, then ``sets`` /
``rename`` / ``xform``. Undoing a named create used to segfault Maya (the old
``runUndoableAPICommand`` wrapper); redo used to make a new node. A holed face
is rebuilt from its triangulation by one ``polyDelEdge`` without construction
history, after the UVs, so each UV stays on its vertex.

The names, the selection and the topology are pinned to what the create made
before the ride (runs\\rU\\U0\\mesh_reference.json). Undo tests check the scene
(``rig._tests._undo``), never ``cmds.undo()``'s return value.
"""

from maya import cmds
from maya.api import OpenMaya
from rig import container
from rig.nodetypes import Mesh
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, mesh_state

CHUNK = "rig.Mesh.create"

# the holed face of each fixture as today's create (polyDelEdge) orders it:
# every loop starts at another vertex than the source's (U0 reference)
FACET_SOURCE_VERTICES = ((8,), (0, 1, 2, 3, 4, 7, 6, 5))
FACET_SOURCE_HOLES    = ((0, (4, 7, 6, 5)),)
FACET_VERTICES        = ((8,), (3, 0, 1, 2, 7, 6, 5, 4))
FACET_HOLES           = ((0, (7, 6, 5, 4)),)
GRID_FACE4_SOURCE     = (5, 6, 10, 9, 16, 17, 18, 19, 20, 21, 22, 23)
GRID_FACE4            = (10, 9, 5, 6, 19, 16, 17, 18, 21, 22, 23, 20)
GRID_HOLES            = ((4, (19, 16, 17, 18)), (4, (21, 22, 23, 20)))


# -- sources


def _cube_data(name=None):
    """A polyCube (moved and rotated) serialized in object space, then deleted:
    ``(mesh data named `name`, [map1])``."""
    t = cmds.polyCube(name="src", constructionHistory=False)[0]
    cmds.xform(t, translation=(1, 2, 3), rotation=(10, 20, 30))
    data, uvs = Mesh(t).serialize(world_space=False)
    cmds.delete(t)
    data.name = name
    return data, list(uvs)


def _uv_sets(map1):
    """``[custom, uv2]``: map1 renamed ``custom``, and a second, scaled set ``uv2``."""
    custom = map1.copy()
    custom.name = "custom"
    uv2 = map1.copy()
    uv2.name = "uv2"
    uv2.points = uv2.points * 0.5 + 0.25
    return [custom, uv2]


def _facet(name="facet"):
    """A face with one square hole (polyCreateFacet makes its map1): the transform."""
    return cmds.polyCreateFacet(
        point=[(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4), (), (1, 0, 1), (3, 0, 1), (3, 0, 3), (1, 0, 3)],
        name=name,
        constructionHistory=False,
    )[0]


def _grid(name="grid"):
    """A 3x3 plane whose centre face carries two holes, planar-mapped: the transform."""
    t = cmds.polyPlane(name=name, sx=3, sy=3, w=3, h=3, constructionHistory=False)[0]
    shape = cmds.listRelatives(t, shapes=True, fullPath=True)[0]
    sel = OpenMaya.MSelectionList()
    sel.add(shape)
    loops = [(-0.4, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-0.4, 0.2), (0.1, -0.2), (0.4, -0.2), (0.4, 0.2), (0.1, 0.2)]
    OpenMaya.MFnMesh(sel.getDagPath(0)).addHoles(4, [OpenMaya.MPoint(x, 0, z) for x, z in loops], [4, 4])
    cmds.polyPlanarProjection(t + ".f[*]", mapDirection="y")
    cmds.delete(t, constructionHistory=True)
    return t


def _serialized(t):
    """``(mesh data, [uv data])`` of the transform `t` in object space."""
    data, uvs = Mesh(t).serialize(world_space=False)
    return data, list(uvs)


# -- queries


def _shape(xform):
    return cmds.listRelatives(xform, shapes=True, fullPath=True)[0]


def _uv_placement(name, uv_set="map1"):
    """``{(face, vertex): (u, v)}`` of the mesh `name`: where each UV sits,
    whatever order the faces list their vertices in."""
    state = mesh_state(name)
    counts, ids = state["vertices"]
    u, v, uv_counts, uv_ids = state["uvs"][uv_set]
    out, off, uv_off = {}, 0, 0
    for face, count in enumerate(counts):
        for k in range(count):
            out[(face, ids[off + k])] = (
                (u[uv_ids[uv_off + k]], v[uv_ids[uv_off + k]]) if uv_counts[face] else None
            )
        off += count
        uv_off += uv_counts[face]
    return out


def _loops(state):
    """Each face's loops (outer loop, then each hole loop) as sets of cyclic
    rotations: two meshes whose loops start at other vertices compare equal."""
    counts, ids = state["vertices"]
    holes = {}
    for face, loop in state["holes"]:
        holes.setdefault(face, []).append(loop)
    faces, off = [], 0
    for face, count in enumerate(counts):
        verts = ids[off : off + count]
        off += count
        hole_verts = {v for loop in holes.get(face, ()) for v in loop}
        outer = tuple(v for v in verts if v not in hole_verts)
        loops = [outer] + list(holes.get(face, ()))
        faces.append(tuple(frozenset(loop[i:] + loop[:i] for i in range(len(loop))) for loop in loops))
    return faces


class _MeshCreateCase(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        try:
            cmds.namespace(set=":")
            self.assertFalse(container.is_active)
        finally:
            super().tearDown()

    def ready(self):
        """Start the test's own steps here: select the probe transform ``selprobe``
        (made when missing), then flush what building the sources recorded."""
        if not cmds.objExists("selprobe"):
            cmds.createNode("transform", name="selprobe")
        cmds.select("selprobe", replace=True)
        cmds.flushUndo()

    def assert_no_history(self, mesh):
        shape = _shape(mesh.get_parent()) if isinstance(mesh, Mesh) else _shape(mesh)
        self.assertEqual(cmds.listHistory(shape), [cmds.ls(shape)[0]])
        shapes = cmds.listRelatives(cmds.listRelatives(shape, parent=True, fullPath=True)[0], shapes=True,
                                    fullPath=True)
        self.assertEqual(shapes, [shape])  # no Orig / intermediate shape
        self.assertFalse(cmds.getAttr(shape + ".intermediateObject"))
        self.assertEqual(cmds.ls(type="polyDelEdge"), [])


# ---------------------------------------------------------------------------------------------
class TestMeshCreateNames(_MeshCreateCase):
    """Names, selection, shading, history and matrix equal today's create (U0 reference)."""

    def _check(self, mesh, xform, shape, data):
        self.assertEqual(str(mesh), shape)
        self.assertEqual(cmds.ls(str(mesh.get_parent()))[0], xform)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])
        self.assertEqual(
            sorted(set(cmds.listConnections(mesh.long_name, type="shadingEngine") or [])),
            ["initialShadingGroup"],
        )
        self.assert_no_history(mesh)
        matrix = cmds.xform(mesh.get_parent().long_name, query=True, matrix=True, objectSpace=True)
        for got, want in zip(matrix, data.matrix.ravel()):
            self.assertAlmostEqual(got, float(want), places=6)

    def test_no_name_twice(self):
        data, _ = _cube_data()
        self.ready()
        self._check(Mesh.create(data), "polySurface1", "polySurfaceShape1", data)
        self._check(Mesh.create(data), "polySurface2", "polySurfaceShape2", data)

    def test_explicit_names(self):
        data, _ = _cube_data()
        for name, xform, shape in (("m", "m", "mShape"), ("mShape", "m", "mShape"), ("mShape2", "m2", "mShape2")):
            with self.subTest(name=name):
                cmds.file(new=True, force=True)
                self.ready()
                self._check(Mesh.create(data, name=name), xform, shape, data)

    def test_name_taken(self):
        data, _ = _cube_data()
        self.ready()
        self._check(Mesh.create(data, name="m"), "m", "mShape", data)
        self._check(Mesh.create(data, name="m"), "m1", "mShape1", data)
        cmds.file(new=True, force=True)
        cmds.createNode("transform", name="m")
        self.ready()
        self._check(Mesh.create(data, name="m"), "m1", "mShape1", data)

    def test_namespace(self):
        data, _ = _cube_data()
        cmds.namespace(add="ns")
        self.ready()
        self._check(Mesh.create(data, name="ns:m"), "ns:m", "ns:mShape", data)

    def test_name_from_the_data(self):
        for data_name, xform, shape in (
            ("srcShape", "src", "srcShape"),
            ("fromData", "fromData", "fromDataShape"),
            ("fromDataShape3", "fromData3", "fromDataShape3"),
        ):
            with self.subTest(data_name=data_name):
                cmds.file(new=True, force=True)
                data, _ = _cube_data(data_name)
                self.ready()
                self._check(Mesh.create(data), xform, shape, data)

    def test_name_in_nested_containers(self):
        data, _ = _cube_data()
        self.ready()
        with container("outer") as outer:
            with container("inner"):
                mesh = Mesh.create(data, name="m")
        self._check(mesh, "inner_m", "inner_mShape", data)
        self.assertEqual(sorted(cmds.container(str(outer), query=True, nodeList=True)), ["inner_m", "inner_mShape"])

    def test_uv_sets_and_order(self):
        data, (map1,) = _cube_data()
        custom, uv2 = _uv_sets(map1)
        for uvs, names in (
            ([map1], ["map1"]),
            (map1, ["map1"]),  # a single UVData
            ([map1, uv2], ["map1", "uv2"]),
            ([custom], ["custom"]),
            ([custom, uv2], ["custom", "uv2"]),
        ):
            with self.subTest(names=names, single=not isinstance(uvs, list)):
                cmds.file(new=True, force=True)
                self.ready()
                mesh = Mesh.create(data, uv_data=uvs, name="m")
                self._check(mesh, "m", "mShape", data)
                self.assertEqual(list(mesh.uv_sets), names)
                self.assertEqual(mesh.current_uv_set, names[0])
                for name, uv in zip(names, uvs if isinstance(uvs, list) else [uvs]):
                    written = mesh.serialize_uv(name)
                    self.assertEqual(written.indices.tolist(), uv.indices.tolist())
                    self.assertEqual(written.counts.tolist(), uv.counts.tolist())
                    self.assertTrue(abs(written.points - uv.points).max() < 1e-6)

    def test_world_space_data_leaves_the_transform_at_rest(self):
        # world-space data carries the identity matrix: no xform, the points are world points
        t = cmds.polyCube(name="src", constructionHistory=False)[0]
        cmds.xform(t, translation=(1, 2, 3), rotation=(10, 20, 30))
        data, uvs = Mesh(t).serialize()
        cmds.delete(t)
        self.ready()
        mesh = Mesh.create(data, uv_data=uvs, name="m")
        self._check(mesh, "m", "mShape", data)
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(cmds.xform("m", query=True, matrix=True, worldSpace=True), identity)
        points = mesh.get_points(world_space=True)
        for i, row in enumerate(data.points):
            for got, want in zip((points[i].x, points[i].y, points[i].z), row):
                self.assertAlmostEqual(got, float(want), places=6)
        self.assertEqual(self.undo_name(), CHUNK)
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("m"))

    def test_no_uvs_leaves_map1_empty(self):
        data, _ = _cube_data()
        mesh = Mesh.create(data, uv_data=[], name="m")
        self.assertEqual(list(mesh.uv_sets), ["map1"])
        self.assertEqual(len(mesh.get_uv_coords("map1")), 0)


# ---------------------------------------------------------------------------------------------
class TestMeshCreateUndo(_MeshCreateCase):
    """One undo step named rig.Mesh.create; exact undo / redo walks; the same node on redo."""

    def test_named_create_undo_does_not_crash(self):
        # the old wrapper segfaulted Maya here (exit 139 in TcommandContainer::undoIt)
        data, uvs = _cube_data()
        self.ready()
        Mesh.create(data, uv_data=uvs, name="m")
        self.assertEqual(self.undo_name(), CHUNK)
        cmds.undo()
        self.assertFalse(cmds.objExists("m"))
        self.assertFalse(cmds.ls(type="mesh"))
        cmds.redo()
        self.assertEqual(cmds.ls(type="mesh"), ["mShape"])

    def test_serialize_create_round_trip_then_undo(self):
        t = cmds.polyCube(name="body", constructionHistory=False)[0]
        cmds.xform(t, translation=(0, 1, 0))
        data, uvs = _serialized(t)
        self.assertEqual(data.name, "bodyShape")
        source = mesh_state("bodyShape")
        cmds.delete(t)
        self.ready()
        before = self.scene_state(meshes=["bodyShape"])
        mesh = Mesh.create(data, uv_data=uvs)  # the name stored in the data
        self.assertEqual((str(mesh), cmds.ls(str(mesh.get_parent()))[0]), ("bodyShape", "body"))
        self.assertEqual(mesh_state("bodyShape"), source)
        self.assertEqual(cmds.getAttr("body.translateY"), 1.0)
        after = self.scene_state(meshes=["bodyShape"])
        self.undo_steps(1)
        self.assert_state_equal(self.scene_state(meshes=["bodyShape"]), before, "undo")
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(meshes=["bodyShape"]), after, "redo")

    def test_one_undo_removes_transform_and_shape(self):
        data, uvs = _cube_data()
        self.ready()
        nodes = set(cmds.ls())
        Mesh.create(data, uv_data=_uv_sets(uvs[0]), name="m")
        self.assertEqual(set(cmds.ls()) - nodes, {"m", "mShape"})
        self.assertEqual(self.undo_name(), CHUNK)
        self.undo_steps(1)
        self.assertEqual(set(cmds.ls()), nodes)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))  # one step, all of it

    def test_undo_all_redo_all_exact(self):
        cases = (
            ("cube", None, False),
            ("cube_two_sets", "two", False),
            ("facet", None, True),
            ("facet_uvs", "own", True),
            ("grid_uvs", "own", True),
        )
        for label, uv_kind, holed in cases:
            with self.subTest(label):
                cmds.file(new=True, force=True)
                if not holed:
                    data, uvs = _cube_data()
                    uvs = _uv_sets(uvs[0]) if uv_kind == "two" else None
                else:
                    t = _grid() if label.startswith("grid") else _facet()
                    data, uvs = _serialized(t)
                    cmds.delete(t)
                    uvs = uvs if uv_kind == "own" else None
                self.ready()
                names = dict(meshes=["mShape"])
                start = self.scene_state(**names)
                Mesh.create(data, uv_data=uvs, name="m")
                built = self.scene_state(**names)
                self.assertIsNotNone(built["meshes"]["mShape"])
                for cycle in range(3):
                    self.assertEqual(self.undo_all(), 1)
                    self.assert_state_equal(self.scene_state(**names), start, f"{label} undo {cycle}")
                    self.assertEqual(self.redo_all(), 1)
                    self.assert_state_equal(self.scene_state(**names), built, f"{label} redo {cycle}")

    def test_redo_returns_the_same_node(self):
        data, uvs = _cube_data()
        self.ready()
        mesh = Mesh.create(data, uv_data=uvs, name="m")
        handle = OpenMaya.MObjectHandle(mesh.mobject)
        uuid = cmds.ls("mShape", uuid=True)[0]
        points = [tuple(p) for p in mesh.get_points(world_space=False)]
        for _ in range(3):
            cmds.undo()
            self.assertFalse(cmds.objExists("mShape"))
            cmds.redo()
            self.assertTrue(handle.isValid())
            self.assertEqual(cmds.ls("mShape", uuid=True)[0], uuid)
            self.assertEqual(mesh.num_vertices, 8)
            self.assertEqual([tuple(p) for p in mesh.get_points(world_space=False)], points)
            self.assertEqual(str(mesh), "mShape")

    def test_inside_a_user_chunk_with_cmds_edits(self):
        data, uvs = _cube_data()
        self.ready()
        names = dict(meshes=["mShape"])
        start = self.scene_state(**names)
        cmds.undoInfo(openChunk=True, chunkName="user.step")
        try:
            cmds.setAttr("selprobe.translateX", 3.0)
            Mesh.create(data, uv_data=uvs, name="m")
            cmds.setAttr("m.translateZ", 5.0)
            cmds.parent("m", "selprobe")
        finally:
            cmds.undoInfo(closeChunk=True)
        built = self.scene_state(**names)
        self.assertEqual(self.undo_name(), "user.step")
        for cycle in range(2):
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(**names), start, f"undo {cycle}")
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(**names), built, f"redo {cycle}")
        self.assertEqual(cmds.getAttr("selprobe|m.translateZ"), 5.0)

    def test_between_cmds_steps(self):
        data, uvs = _cube_data()
        self.ready()
        names = dict(meshes=["mShape"])
        states = [self.scene_state(**names)]
        cmds.setAttr("selprobe.translateX", 3.0)
        states.append(self.scene_state(**names))
        Mesh.create(data, uv_data=uvs, name="m")
        states.append(self.scene_state(**names))
        cmds.setAttr("m.translateZ", 5.0)
        states.append(self.scene_state(**names))
        for k in (2, 1, 0):
            self.undo_steps(1)
            self.assert_state_equal(self.scene_state(**names), states[k], f"undo to {k}")
        for k in (1, 2, 3):
            self.redo_steps(1)
            self.assert_state_equal(self.scene_state(**names), states[k], f"redo to {k}")

    def test_in_a_container_scope_one_undo_reverts_membership(self):
        data, uvs = _cube_data()
        self.ready()
        names = dict(meshes=["mShape"])
        with container("box") as box:
            start = self.scene_state(**names)
            members_before = sorted(cmds.container(str(box), query=True, nodeList=True) or []) \
                if cmds.objExists(str(box)) else None
            Mesh.create(data, uv_data=uvs, name="m")
        box_name = str(box)
        built = self.scene_state(**names)
        self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True)), ["m", "mShape"])
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("m"))
        self.assertFalse(cmds.objExists("mShape"))
        if cmds.objExists(box_name):
            self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True) or []),
                             members_before or [])
        self.assert_state_equal(self.scene_state(**names), start, "undo")
        self.redo_steps(1)
        self.assert_state_equal(self.scene_state(**names), built, "redo")
        self.assertEqual(sorted(cmds.container(box_name, query=True, nodeList=True)), ["m", "mShape"])

    def test_undo_off(self):
        data, uvs = _cube_data()
        cmds.undoInfo(state=False)
        try:
            mesh = Mesh.create(data, uv_data=uvs, name="m")
        finally:
            cmds.undoInfo(state=True)
        self.assertEqual(str(mesh), "mShape")
        self.assertEqual(list(mesh.uv_sets), ["map1"])
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


# ---------------------------------------------------------------------------------------------
class TestMeshCreateErrors(_MeshCreateCase):
    """Invalid data raises RuntimeError before anything is made; the next undo is the user's."""

    def _bad(self, data, uvs=None, name="m"):
        cmds.setAttr("selprobe.translateX", 7.0)  # the user's previous step
        nodes = sorted(cmds.ls())
        with self.assertRaises(RuntimeError):
            Mesh.create(data, uv_data=uvs, name=name)
        self.assertEqual(sorted(cmds.ls()), nodes)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])
        return nodes

    def test_invalid_mesh_data(self):
        for label in ("counts", "index_negative", "index_past_the_points", "hole_counts", "hole_face",
                      "hole_index", "hole_arrays_missing"):
            with self.subTest(label):
                cmds.file(new=True, force=True)
                if label.startswith("hole"):
                    t = _facet()
                    data, uvs = _serialized(t)
                    cmds.delete(t)
                else:
                    data, uvs = _cube_data()
                if label == "counts":
                    data.counts = data.counts.copy()
                    data.counts[0] += 1
                elif label == "index_negative":
                    data.indices = data.indices.copy()
                    data.indices[3] = -1
                elif label == "index_past_the_points":
                    data.indices = data.indices.copy()
                    data.indices[3] = len(data.points)
                elif label == "hole_counts":
                    data.hole_counts = data.hole_counts + 1
                elif label == "hole_face":
                    data.hole_faces = data.hole_faces + 5
                elif label == "hole_index":
                    data.hole_indices = data.hole_indices.copy()
                    data.hole_indices[0] = len(data.points) + 3
                elif label == "hole_arrays_missing":
                    data.hole_indices = None
                self.ready()
                self._bad(data, uvs)
                self.undo_steps(1)  # the setAttr, not an empty create step
                self.assertEqual(cmds.getAttr("selprobe.translateX"), 0.0)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_holed_face_uv_count_mismatch(self):
        t = _facet()
        data, uvs = _serialized(t)
        cmds.delete(t)
        uvs[0].counts = uvs[0].counts.copy()
        uvs[0].counts[0] = 5
        uvs[0].indices = uvs[0].indices[:5].copy()
        self.ready()
        self._bad(data, uvs)
        self.undo_steps(1)
        self.assertEqual(cmds.getAttr("selprobe.translateX"), 0.0)

    def test_error_after_the_transform_leaves_no_node(self):
        # a UV write failing after the transform is made (duplicate UV set names, the
        # case this test first used, are refused before it since round U2's FIX)
        from rig.nodetypes import mesh as mesh_module

        data, (map1,) = _cube_data()
        second = map1.copy()
        second.name = "uv2"

        def failing(fn, uv_set, uv_data, clear=True):
            raise RuntimeError("a UV write failed")

        original = mesh_module._write_uv_set
        mesh_module._write_uv_set = failing
        try:
            self.ready()
            self._bad(data, [map1, second])
        finally:
            mesh_module._write_uv_set = original
        self.assertFalse(cmds.ls(type="mesh"))
        self.undo_all()
        self.redo_all()
        self.assertFalse(cmds.ls(type="mesh"))
        self.assertEqual(cmds.getAttr("selprobe.translateX"), 7.0)

    def test_duplicate_uv_set_names_raise_before_anything_is_made(self):
        # they raised after the transform was made: the error left an empty create step
        # and flushed the redo queue (runs\rU2\review_undo\p_empty.py)
        data, (map1,) = _cube_data()
        for names in (("map1", "map1"), ("a", "b", "a")):
            with self.subTest(names):
                uvs = []
                for uv_name in names:
                    uv = map1.copy()
                    uv.name = uv_name
                    uvs.append(uv)
                self.ready()
                cmds.setAttr("selprobe.translateY", 3.0)
                self.undo_steps(1)  # the user's step on the redo queue
                with self.assertRaisesRegex(RuntimeError, rf"^UV set {names[-1]} already exists\.$"):
                    Mesh.create(data, uv_data=uvs, name="m")
                self.assertFalse(cmds.ls(type="mesh"))
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.redo_steps(1)
                self.assertEqual(cmds.getAttr("selprobe.translateY"), 3.0)
                cmds.setAttr("selprobe.translateY", 0.0)


# ---------------------------------------------------------------------------------------------
class TestMeshCreateHoles(_MeshCreateCase):
    """A holed face comes back with its loops, its points and each UV on its vertex, with no
    polyDelEdge node and no Orig shape; the loops start where today's create started them."""

    def test_facet_hole_round_trip(self):
        t = _facet()
        source = mesh_state(_shape(t))
        placed = _uv_placement(_shape(t))
        data, uvs = _serialized(t)
        cmds.delete(t)
        self.assertEqual(source["vertices"], FACET_SOURCE_VERTICES)
        self.assertEqual(source["holes"], FACET_SOURCE_HOLES)
        self.ready()
        mesh = Mesh.create(data, uv_data=uvs, name="copy")
        got = mesh_state(str(mesh))
        self.assertEqual(got["points"], source["points"])
        self.assertEqual(_loops(got), _loops(source))
        self.assertEqual(got["vertices"], FACET_VERTICES)  # today's order (U0 reference)
        self.assertEqual(got["holes"], FACET_HOLES)
        self.assertEqual(_uv_placement(str(mesh)), placed)  # each UV on its vertex
        self.assert_no_history(mesh)
        self.assertEqual(cmds.ls(selection=True), ["selprobe"])

    def test_grid_two_holes_round_trip(self):
        t = _grid()
        source = mesh_state(_shape(t))
        placed = _uv_placement(_shape(t))
        data, uvs = _serialized(t)
        cmds.delete(t)
        self.assertEqual(source["vertices"][1][16:28], GRID_FACE4_SOURCE)
        self.ready()
        mesh = Mesh.create(data, uv_data=uvs)
        self.assertEqual(str(mesh), "gridShape")
        got = mesh_state(str(mesh))
        self.assertEqual(got["points"], source["points"])
        self.assertEqual(got["vertices"][0], source["vertices"][0])
        self.assertEqual(_loops(got), _loops(source))
        self.assertEqual(got["vertices"][1][16:28], GRID_FACE4)  # today's order (U0 reference)
        self.assertEqual(got["vertices"][1][:16], source["vertices"][1][:16])
        self.assertEqual(got["vertices"][1][28:], source["vertices"][1][28:])
        self.assertEqual(got["holes"], GRID_HOLES)
        self.assertEqual(_uv_placement(str(mesh)), placed)
        self.assert_no_history(mesh)

    def test_holed_create_without_uvs(self):
        t = _facet()
        data, _ = _serialized(t)
        cmds.delete(t)
        mesh = Mesh.create(data, name="copy")
        got = mesh_state(str(mesh))
        self.assertEqual(got["vertices"], FACET_VERTICES)
        self.assertEqual(got["holes"], FACET_HOLES)
        self.assertEqual(got["uvs"]["map1"][:2], ((), ()))
        self.assert_no_history(mesh)

    def test_holed_undo_redo_keeps_the_uvs_on_their_vertices(self):
        t = _grid()
        placed = _uv_placement(_shape(t))
        data, uvs = _serialized(t)
        cmds.delete(t)
        self.ready()
        mesh = Mesh.create(data, uv_data=uvs, name="m")
        handle = OpenMaya.MObjectHandle(mesh.mobject)
        for _ in range(3):
            self.undo_steps(1)
            self.assertFalse(cmds.objExists("mShape"))
            self.redo_steps(1)
            self.assertTrue(handle.isValid())
            self.assertEqual(_uv_placement("mShape"), placed)
            self.assertEqual(mesh_state("mShape")["holes"], GRID_HOLES)
        self.assertEqual(cmds.ls(type="polyDelEdge"), [])
