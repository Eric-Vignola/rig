"""Undo of rig's mesh edits (round U2, step S3).

Every mesh edit is one undo step through rig's plug-in command
(``rig.nodetypes.plugins._run_undoable``): ``Mesh.set_points``, the UV set edits
(``add_uv_set``, ``rename_uv_set``, ``delete_uv_set``, ``set_uv_data``) and the colour
set edits (``add_color_set``, the ``ColorSet.data`` setter, ``ColorSet.delete``). Each
is API only, sets included (never ``polyUVSet`` / ``polyColorSet``, whose undo beside
API data edits and a later vertex edit restores broken sets: runs\\rU2\\S3\\probe).
The user's own polyUVSet / polyColorSet beside them: TestNativeSetEditsBesideRig and
TestSetElementsAndLinks (round U2, FIX: runs\\rU2\\FIX).

Undo tests check the scene (``rig._tests._undo``), never ``cmds.undo()``'s return
value. The walks run on a history-free cube, one with vertex tweaks, one with
construction history and a skinned one where the edit is exact there.
"""

import ast
import inspect

import numpy as np
from maya import cmds
from maya.api import OpenMaya

import rig
from rig.nodetypes import Mesh, SkinCluster
from rig.nodetypes import mesh as mesh_module
from rig.nodetypes import skincluster as skincluster_module
from rig.nodetypes.mesh import ColorSet
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, mesh_state

RGBA  = ColorSet.Representation.RGBA
RGB   = ColorSet.Representation.RGB
ALPHA = ColorSet.Representation.Alpha

# the kinds of mesh the walks run on (see make_mesh)
MESH_KINDS    = ("free", "tweaked", "history", "skinned")
HISTORY_FREE  = ("free", "tweaked")


def make_mesh(kind="free", name="c"):
    """A cube ``name`` of the given kind, as a Mesh: ``free`` polyCube(ch=False);
    ``tweaked`` the same with a vertex moved (a ``pnts`` tweak); ``history``
    polyCube(ch=True); ``skinned`` a history-free cube bound to two joints."""
    t = cmds.polyCube(name=name, constructionHistory=(kind == "history"))[0]
    if kind == "tweaked":
        cmds.move(0.0, 0.5, 0.0, f"{t}.vtx[0]", relative=True, objectSpace=True)
    elif kind == "skinned":
        cmds.select(clear=True)
        j1 = cmds.joint(name=f"{name}_j1", position=(0, -1, 0))
        cmds.select(clear=True)
        j2 = cmds.joint(name=f"{name}_j2", position=(0, 1, 0))
        cmds.skinCluster([j1, j2], t, toSelectedBones=True)
        cmds.select(clear=True)
    return Mesh(t)


def points_of(mesh, world=False):
    return np.array(mesh.get_points(world_space=world))[:, :3]


def lifted(mesh):
    """The mesh's object-space points, one unit up."""
    return points_of(mesh) + [0.0, 1.0, 0.0]


def move_vertices(mesh):
    """A user's native vertex edit (recorded ``move``)."""
    cmds.move(0.0, 1.0, 0.0, f"{mesh.long_name}.vtx[*]", relative=True)


def uv_data(mesh, uv_set="map1", name=None, scale=1.0):
    data = mesh.serialize_uv(uv_set)
    if name is not None:
        data.name = name
    data.points = np.asarray(data.points) * scale
    return data


def per_face_vertex_uvs(mesh, name="map1"):
    """A UVData with one UV per face-vertex of `mesh` (24 on a cube)."""
    data   = uv_data(mesh, "map1", name)
    n      = int(np.asarray(data.counts).sum())
    data.points  = np.column_stack((np.linspace(0.0, 1.0, n), np.linspace(1.0, 0.0, n)))
    data.indices = np.arange(n)
    return data


def no_uvs(mesh, name="map1"):
    data = uv_data(mesh, "map1", name)
    data.points  = np.zeros((0, 2))
    data.indices = np.zeros(0, dtype=int)
    data.counts  = np.zeros(0, dtype=int)
    return data


def colours(mesh, rgba):
    return np.tile(rgba, (mesh.num_vertices, 1))


def tweaks(mesh):
    """The non-zero vertex tweaks (``pnts``) of `mesh` by logical index, read
    through the plug (a ``getAttr`` of a range would make the elements)."""
    plug = mesh.fn_set.findPlug("pnts", False)
    out  = {}
    for i in range(plug.numElements()):
        element = plug.elementByPhysicalIndex(i)
        value   = tuple(round(element.child(k).asDouble(), 6) + 0.0 for k in range(3))
        if any(value):
            out[element.logicalIndex()] = value
    return out


class _UndoCase(UndoWalk, MayaTestCase):
    """A new scene, the queue on / infinite / flushed (UndoWalk)."""

    TEST_START_NEW_SCENE = True

    def tearDown(self):
        try:
            cmds.undoInfo(state=True)
            cmds.flushUndo()
        finally:
            super().tearDown()

    def state(self, mesh):
        """The scene, the mesh's data and, on a mesh with history, its vertex tweaks
        (without history ``setPoints`` folds a tweak into the vertices, whose points
        are what counts: undo gives the points back, not the split)."""
        state = self.scene_state(meshes=[mesh.long_name])
        if mesh.fn_set.findPlug("inMesh", False).isDestination:
            state["tweaks"] = tweaks(mesh)
        return state

    def walk(self, mesh, edits, cycles=2, msg=""):
        """Makes each edit (one undo step each), then walks every undo and redo step
        ``cycles`` times: each gives the state recorded after that edit back, and
        reads through the held ``mesh`` answer after every step. Returns the states."""
        states = [self.state(mesh)]
        for edit in edits:
            edit()
            states.append(self.state(mesh))
        for i in range(len(states) - 1):
            self.assertNotEqual(states[i], states[i + 1], f"{msg} edit {i} changed nothing")
        for cycle in range(cycles):
            for i in range(len(states) - 1, 0, -1):
                self.undo_steps(1)
                self.assertEqual(self.state(mesh), states[i - 1], f"{msg} cycle {cycle} undo to {i - 1}")
                self.read_through(mesh)
            for i in range(1, len(states)):
                self.redo_steps(1)
                self.assertEqual(self.state(mesh), states[i], f"{msg} cycle {cycle} redo to {i}")
                self.read_through(mesh)
        return states

    @staticmethod
    def read_through(mesh):
        """Reads through a held Mesh and its ColorSets (a stale MFnMesh would crash)."""
        mesh.num_vertices
        for uv_set in mesh.uv_sets:
            mesh.get_uv_coords(uv_set)
            mesh.get_assigned_uvs(uv_set)
        for color_set in mesh.get_color_sets():
            color_set.data

    def assert_refused(self, mesh, call, error, regex=None):
        """``call`` raises ``error`` and leaves the scene and the undo queue as they were."""
        before = self.state(mesh)
        empty  = cmds.undoInfo(query=True, undoQueueEmpty=True)
        if regex is None:
            with self.assertRaises(error):
                call()
        else:
            with self.assertRaisesRegex(error, regex):
                call()
        self.assertEqual(self.state(mesh), before)
        self.assertEqual(cmds.undoInfo(query=True, undoQueueEmpty=True), empty)


# --- the rules ----------------------------------------------------------------------------


def _undo_bodies_calling_cmds(source):
    """``Class.method`` names of every doIt / undoIt / redoIt in the module `source`
    (and of the methods and module functions they reach) that name ``cmds`` or ``mel``."""
    tree      = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes   = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def methods(cls):
        found = {}
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                found.update(methods(classes[base.id]))
        found.update({n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)})
        return found

    bad = set()
    for cls in classes.values():
        members = methods(cls)
        if not {"doIt", "undoIt", "redoIt"} <= set(members):
            continue
        todo, seen = [(f"{cls.name}.{m}", members[m]) for m in ("doIt", "undoIt", "redoIt")], set()
        while todo:
            label, node = todo.pop()
            if label in seen:
                continue
            seen.add(label)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in ("cmds", "mel"):
                    bad.add(label)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    if sub.attr in members:
                        todo.append((f"{cls.name}.{sub.attr}", members[sub.attr]))
                elif isinstance(sub, ast.Name) and sub.id in functions:
                    todo.append((sub.id, functions[sub.id]))
    return sorted(bad)


class TestMeshEditRules(MayaTestCase):
    """The rules of rig's undoable edits (mesh.py's "rig's undoable mesh edits")."""

    def test_no_cmds_or_mel_in_any_undoable_body(self):
        for module in (mesh_module, skincluster_module):
            with self.subTest(module.__name__):
                self.assertEqual(_undo_bodies_calling_cmds(inspect.getsource(module)), [])

    def test_the_lint_sees_a_cmds_call_it_reaches(self):
        source = (
            "def helper():\n    cmds.setAttr('x', 1)\n\n"
            "class Base:\n    def redoIt(self):\n        mel.eval('x')\n\n"
            "class A(Base):\n    def doIt(self):\n        self._write()\n"
            "    def undoIt(self):\n        helper()\n    def _write(self):\n        pass\n"
        )
        self.assertEqual(_undo_bodies_calling_cmds(source), ["A.redoIt", "helper"])

    def test_a_mesh_function_set_is_new_on_every_access(self):
        cube = make_mesh()
        self.assertIsNot(cube.fn_set, cube.fn_set)
        self.assertEqual(cube.fn_set.numVertices, 8)

    def test_a_held_mesh_survives_native_set_edits(self):
        # an MFnMesh kept from before a polyUVSet / polyColorSet crashed Maya when used after it
        cube = make_mesh("history")
        cube.get_uv_coords()
        cmds.polyUVSet(cube.name, create=True, uvSet="uv2")
        cmds.polyColorSet(cube.name, create=True, colorSet="cs")
        cube.set_uv_data(uv_data(cube, "map1", "uv2"), "uv2")
        self.assertEqual(cube.fn_set.numUVs("uv2"), 14)
        self.assertEqual(cube.num_vertices, 8)


# --- set_points ---------------------------------------------------------------------------


class TestSetPointsUndo(_UndoCase):
    """``Mesh.set_points``: one undo step; a wrong point count raises before any edit."""

    def test_object_and_world_space(self):
        for kind in MESH_KINDS:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.xform("c", translation=(1, 2, 3), rotation=(10, 20, 30))
                cmds.flushUndo()
                target = points_of(cube, world=True) * 2.0
                self.walk(cube, [
                    lambda: cube.set_points(lifted(cube)),
                    lambda: cube.set_points(target, world_space=True),
                ], msg=kind)
                self.assertTrue(np.allclose(points_of(cube, world=True), target, atol=1e-5))

    def test_one_step(self):
        cube = make_mesh()
        cmds.flushUndo()
        cube.set_points(lifted(cube))
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_an_mpointarray_and_mesh_data_are_taken(self):
        cube   = make_mesh()
        points = OpenMaya.MPointArray(lifted(cube).tolist())
        data   = cube.serialize(world_space=False, include_uvs=False)
        data.points = np.asarray(data.points) * 3.0
        cmds.flushUndo()
        states = self.walk(cube, [lambda: cube.set_points(points), lambda: cube.set_points(data)])
        points[0] = OpenMaya.MPoint(9, 9, 9)  # the caller's array, changed afterwards
        self.undo_steps(2)
        self.redo_steps(2)
        self.assertEqual(self.state(cube), states[-1])

    def test_a_wrong_point_count_raises_before_any_edit(self):
        cube = make_mesh()
        cmds.flushUndo()
        for count in (0, 1, 7, 9):
            for undo in (True, False):
                with self.subTest(count=count, undo=undo):
                    cmds.undoInfo(state=undo)
                    try:
                        self.assert_refused(
                            cube, lambda: cube.set_points([(1.0, 1.0, 1.0)] * count), ValueError,
                            rf"^Mesh\.set_points: {count} points for the 8 vertices of cShape$",
                        )
                    finally:
                        cmds.undoInfo(state=True)

    def test_interleaved_with_cmds(self):
        cube = make_mesh()
        cmds.flushUndo()
        self.walk(cube, [
            lambda: move_vertices(cube),
            lambda: cube.set_points(lifted(cube)),
            lambda: cmds.move(0.0, 0.5, 0.0, "c.vtx[1]", relative=True),
            lambda: cube.set_points(points_of(cube) * 1.5),
        ])

    def test_undo_off_sets_and_queues_nothing(self):
        cube   = make_mesh()
        target = lifted(cube)
        cmds.flushUndo()
        cmds.undoInfo(state=False)
        try:
            cube.set_points(target)
        finally:
            cmds.undoInfo(state=True)
        self.assertTrue(np.allclose(points_of(cube), target))
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_new_scene_with_edits_queued(self):
        cube = make_mesh()
        cube.set_points(lifted(cube))
        cube.set_points(lifted(cube))
        self.undo_steps(1)  # one on the undo queue, one on the redo queue
        cmds.file(new=True, force=True)
        cube = make_mesh()
        cmds.flushUndo()
        self.walk(cube, [lambda: cube.set_points(lifted(cube))], cycles=1)


# --- UV sets ------------------------------------------------------------------------------


class TestSetUVDataUndo(_UndoCase):
    """``Mesh.set_uv_data``: one undo step, exact when the UV count changes or the
    old set was empty (history-free meshes)."""

    def test_uv_count_changes(self):
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube      = make_mesh(kind)
                fourteen  = uv_data(cube)
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.set_uv_data(per_face_vertex_uvs(cube), "map1"),  # 14 -> 24
                    lambda: cube.set_uv_data(fourteen, "map1"),                   # 24 -> 14
                    lambda: cube.set_uv_data(no_uvs(cube), "map1"),               # 14 -> 0
                ], msg=kind)
                self.assertEqual(cube.fn_set.numUVs("map1"), 0)

    def test_an_empty_old_set_comes_back_empty(self):
        cube = make_mesh()
        cube.add_uv_set("uv2")
        cmds.flushUndo()
        self.walk(cube, [
            lambda: cube.set_uv_data(uv_data(cube, "map1", "uv2", 0.5), "uv2"),
            lambda: cube.set_uv_data(per_face_vertex_uvs(cube, "uv2"), "uv2"),
        ])
        self.undo_steps(2)
        self.assertEqual(cube.fn_set.numUVs("uv2"), 0)
        self.assertEqual(set(cube.fn_set.getAssignedUVs("uv2")[0]), {0})

    def test_the_rename_path(self):
        cube = make_mesh()
        cube.add_uv_set("uv2")
        cmds.flushUndo()
        self.walk(cube, [lambda: cube.set_uv_data(uv_data(cube, "map1", "uv3", 0.5), "uv2")])
        self.assertEqual(list(cube.uv_sets), ["map1", "uv3"])

    def test_invalid_data_raises_before_any_edit(self):
        cube = make_mesh()
        cube.add_uv_set("uv2")
        cmds.flushUndo()
        cases = {}
        bad = uv_data(cube)
        bad.indices = np.array(bad.indices).copy()
        bad.indices[0] = 99
        cases["an index out of range"] = (bad, "map1", RuntimeError)
        bad = uv_data(cube)
        bad.counts = np.array(bad.counts)[:-1]
        cases["one count per face"] = (bad, "map1", RuntimeError)
        bad = uv_data(cube)
        bad.indices = np.array(bad.indices)[:-1]
        cases["counts adding up"] = (bad, "map1", RuntimeError)
        cases["a missing set"] = (uv_data(cube, "map1", "nope"), "nope", RuntimeError)
        cases["the new name exists"] = (uv_data(cube, "map1", "map1"), "uv2", RuntimeError)
        bad = uv_data(cube)
        bad.name = None
        cases["a name that is not a str"] = (bad, "map1", TypeError)
        for label, (data, uv_set, error) in cases.items():
            with self.subTest(label):
                self.assert_refused(cube, lambda: cube.set_uv_data(data, uv_set), error)

    def test_a_write_that_fails_half_way_puts_the_old_uvs_back(self):
        cube = make_mesh()
        cmds.flushUndo()
        bad = uv_data(cube, scale=0.5)
        bad.indices = np.array(bad.indices).copy()
        bad.indices[0] = 99
        original = mesh_module._check_uv_data
        mesh_module._check_uv_data = lambda *args: None
        try:
            self.assert_refused(cube, lambda: cube.set_uv_data(bad, "map1"), RuntimeError)
        finally:
            mesh_module._check_uv_data = original


class TestSetUVDataUndoWithHistory(_UndoCase):
    """``Mesh.set_uv_data`` on a mesh with history never clears the set (Maya's
    clearUVs there clears the CURRENT set through a polyMapDel node)."""

    def test_the_same_count_is_exact_and_adds_no_node(self):
        for kind in ("history", "skinned"):
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                states = self.walk(cube, [lambda: cube.set_uv_data(uv_data(cube, scale=0.5), "map1")], msg=kind)
                self.assertEqual(states[1]["nodes"], states[0]["nodes"])

    def test_more_uvs_undo_keeps_the_extra_ones_unassigned(self):
        cube = make_mesh("history")
        cmds.flushUndo()
        before = mesh_state(cube.long_name)
        nodes  = self.state(cube)["nodes"]
        cube.set_uv_data(per_face_vertex_uvs(cube), "map1")
        after = self.state(cube)
        self.assertEqual(cube.fn_set.numUVs("map1"), 24)
        self.assertEqual(after["nodes"], nodes)
        self.undo_steps(1)
        u, v, counts, ids = mesh_state(cube.long_name)["uvs"]["map1"]
        u0, v0, counts0, ids0 = before["uvs"]["map1"]
        self.assertEqual((counts, ids), (counts0, ids0))         # the old assignment
        self.assertEqual((u[: len(u0)], v[: len(v0)]), (u0, v0))  # the old UVs
        self.assertEqual(len(u), 24)                             # 10 extra, unassigned
        self.redo_steps(1)
        self.assertEqual(self.state(cube), after)

    def test_fewer_uvs_raise_before_any_edit(self):
        cube     = make_mesh("history")
        fourteen = uv_data(cube)
        cube.set_uv_data(per_face_vertex_uvs(cube), "map1")
        cmds.flushUndo()
        self.assert_refused(cube, lambda: cube.set_uv_data(fourteen, "map1"), RuntimeError, "history")


class TestUVSetUndo(_UndoCase):
    """``Mesh.add_uv_set`` / ``rename_uv_set`` / ``delete_uv_set``: one undo step each."""

    def test_add(self):
        for kind in MESH_KINDS:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                states = self.walk(cube, [lambda: cube.add_uv_set("uv2"), lambda: cube.add_uv_set("uv3")], msg=kind)
                self.assertEqual(list(cube.uv_sets), ["map1", "uv2", "uv3"])
                self.assertEqual(cube.current_uv_set, "map1")
                if kind in HISTORY_FREE:
                    self.assertEqual(states[2]["nodes"], states[0]["nodes"])  # no node

    def test_rename(self):
        for kind in MESH_KINDS:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                self.walk(cube, [lambda: cube.rename_uv_set("uvA"), lambda: cube.rename_uv_set("uvB", "uvA")], msg=kind)
                self.assertEqual(list(cube.uv_sets), ["uvB"])

    def test_delete_keeps_the_place_and_the_current_set(self):
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                for name, scale in (("a", 0.2), ("b", 0.4), ("c", 0.6)):
                    cube.add_uv_set(name)
                    cube.set_uv_data(uv_data(cube, "map1", name, scale), name)
                cmds.polyUVSet(cube.name, currentUVSet=True, uvSet="b")
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.delete_uv_set("a"),   # in the middle
                    lambda: cube.delete_uv_set(),      # the current one, b
                ], msg=kind)
                self.assertEqual(list(cube.uv_sets), ["map1", "c"])
                self.undo_steps(2)
                self.assertEqual(list(cube.uv_sets), ["map1", "a", "b", "c"])
                self.assertEqual(cube.current_uv_set, "b")

    def test_the_existing_errors(self):
        cube = make_mesh()
        cube.add_uv_set("uv2")
        cmds.flushUndo()
        self.assert_refused(cube, lambda: cube.add_uv_set("map1"), RuntimeError, r"^UV set map1 already exists\.$")
        self.assert_refused(cube, lambda: cube.rename_uv_set("x", "nope"), RuntimeError, "does not exist")
        self.assert_refused(cube, lambda: cube.rename_uv_set("uv2", "map1"), RuntimeError, "already exists")
        self.assert_refused(cube, lambda: cube.delete_uv_set("nope"), RuntimeError, "does not exist")
        self.assert_refused(cube, lambda: cube.delete_uv_set("map1"), RuntimeError, "default uv set")
        cube.rename_uv_set("uv2", "uv2")  # the same name: nothing to do, nothing queued
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


class TestUVSetEditsThenVertexEdits(_UndoCase):
    """The round-U review's blockers 24 / 25 (runs\\rU\\review_undo\\repro_sets_points.py):
    a UV set rename or delete after rig's UV data, then a vertex edit (rig's or the
    user's), undoes and redoes exactly (history-free meshes; with polyUVSet the undo
    left the set unassigned, or crashed Maya on a later API write)."""

    def test_set_uv_data_rename_path_then_vertex_edits(self):
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.set_uv_data(uv_data(cube, "map1", "other", 0.5)),
                    lambda: cube.set_points(lifted(cube)),
                    lambda: move_vertices(cube),
                ], msg=kind)

    def test_add_data_rename_then_vertex_edits(self):
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.add_uv_set("a"),
                    lambda: cube.set_uv_data(uv_data(cube, "map1", "a", 0.3), "a"),
                    lambda: cube.rename_uv_set("r", "a"),
                    lambda: cube.set_points(lifted(cube)),
                    lambda: move_vertices(cube),
                ], msg=kind)

    def test_add_data_delete_then_vertex_edits(self):
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.add_uv_set("a"),
                    lambda: cube.set_uv_data(uv_data(cube, "map1", "a", 0.3), "a"),
                    lambda: cube.delete_uv_set("a"),
                    lambda: cube.set_points(lifted(cube)),
                    lambda: move_vertices(cube),
                ], msg=kind)

    def test_with_history_nothing_crashes(self):
        # UVs written through the API into a set a history node made do not survive a
        # re-evaluation of the history (Maya): the walk is not exact there, but safe
        for kind in ("history", "skinned"):
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                cube.add_uv_set("a")
                cube.set_uv_data(uv_data(cube, "map1", "a", 0.3), "a")
                cube.rename_uv_set("r", "a")
                cube.delete_uv_set("r")
                cube.set_points(lifted(cube))
                move_vertices(cube)
                for _ in range(2):
                    self.assertEqual(self.undo_all(), 6)
                    self.read_through(cube)
                    self.assertEqual(list(cube.uv_sets), ["map1"])
                    self.assertEqual(self.redo_all(), 6)
                    self.read_through(cube)
                    self.assertEqual(list(cube.uv_sets), ["map1"])


# --- colour sets --------------------------------------------------------------------------


def partial_colors(mesh, name, rep=OpenMaya.MFnMesh.kRGBA):
    """A colour set `name` with colours on three face-vertices only (API)."""
    fn = mesh.fn_set
    fn.createColorSet(name, False, rep=rep)
    current = fn.currentColorSetName()
    fn.setCurrentColorSetName(name)
    fn.setFaceVertexColors([OpenMaya.MColor((0.9, 0.1, 0.2, 0.6))] * 3, [0, 2, 3], [0, 2, 3], rep=rep)
    if current:
        fn.setCurrentColorSetName(current)


class TestColorSetUndo(_UndoCase):
    """``Mesh.add_color_set``, ``ColorSet.data`` and ``ColorSet.delete``: one undo step
    each, exact on every kind of mesh (MESH_KINDS)."""

    def build(self, kind):
        """A new scene with a mesh of `kind` and two colour sets made before the
        test's edits: ``pre`` (RGB, clamped, every vertex) and ``part`` (RGBA, three
        face-vertices); ``pre`` is current. The queue is flushed."""
        cmds.file(new=True, force=True)
        mesh = make_mesh(kind)
        fn   = mesh.fn_set
        fn.createColorSet("pre", True, rep=OpenMaya.MFnMesh.kRGB)
        fn.setCurrentColorSetName("pre")
        fn.setVertexColors(
            [OpenMaya.MColor((i / 8.0, 0.25, 0.75)) for i in range(8)], list(range(8)), rep=OpenMaya.MFnMesh.kRGB
        )
        partial_colors(mesh, "part")
        cmds.flushUndo()
        return mesh

    def test_add(self):
        for kind in MESH_KINDS:
            with self.subTest(kind):
                mesh = self.build(kind)
                self.walk(mesh, [lambda: mesh.add_color_set("cs1", RGBA), lambda: mesh.add_color_set("cs2", ALPHA)], msg=kind)
                self.assertEqual([c.name for c in mesh.get_color_sets()], ["pre", "part", "cs1", "cs2"])

    def test_data(self):
        ramp = np.column_stack((np.linspace(0, 1, 8), np.linspace(1, 0, 8), np.full(8, 0.5), np.full(8, 0.25)))
        for kind in MESH_KINDS:
            with self.subTest(kind):
                mesh = self.build(kind)
                self.walk(mesh, [
                    lambda: setattr(ColorSet(mesh, "part"), "data", colours(mesh, [1.0, 0.0, 0.0, 1.0])),
                    lambda: setattr(ColorSet(mesh, "part"), "data", ramp),
                    lambda: setattr(ColorSet(mesh, "pre"), "data", ramp),   # the current set
                ], msg=kind)
                self.assertEqual(mesh.fn_set.currentColorSetName(), "pre")

    def test_delete_keeps_the_place_and_the_current_set(self):
        for kind in MESH_KINDS:
            with self.subTest(kind):
                mesh = self.build(kind)
                self.walk(mesh, [
                    lambda: ColorSet(mesh, "pre").delete(),   # the first and current set
                    lambda: mesh.add_color_set("cs1", RGB),
                    lambda: ColorSet(mesh, "part").delete(),
                ], msg=kind)
                self.assertEqual([c.name for c in mesh.get_color_sets()], ["cs1"])
                self.undo_steps(3)
                self.assertEqual([c.name for c in mesh.get_color_sets()], ["pre", "part"])
                self.assertEqual(mesh.fn_set.currentColorSetName(), "pre")

    def test_add_data_delete_then_vertex_edits(self):
        # the round-U review's blocker 25, colour half (with polyColorSet -delete the
        # undo lost the colours, then crashed Maya in the colour data's undo)
        for kind in MESH_KINDS:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                mesh = make_mesh(kind)
                cmds.flushUndo()
                self.walk(mesh, [
                    lambda: mesh.add_color_set("c", RGBA),
                    lambda: setattr(ColorSet(mesh, "c"), "data", colours(mesh, [0.1, 0.5, 0.9, 1.0])),
                    lambda: ColorSet(mesh, "c").delete(),
                    lambda: mesh.set_points(lifted(mesh)),
                    lambda: move_vertices(mesh),
                ], msg=kind)

    def test_every_edit_together_and_the_held_objects(self):
        blue = [0.0, 0.0, 1.0, 0.5]
        for kind in MESH_KINDS:
            with self.subTest(kind):
                mesh   = self.build(kind)
                before = self.state(mesh)
                cs     = mesh.add_color_set("cs1", RGBA)
                cs.data = colours(mesh, blue)
                ColorSet(mesh, "part").delete()
                cs2 = mesh.add_color_set("cs2", ALPHA)
                cs2.data = colours(mesh, blue)
                ColorSet(mesh, "pre").delete()
                after = self.state(mesh)
                for cycle in range(2):
                    self.assertEqual(self.undo_all(), 6)
                    self.assertEqual(self.state(mesh), before, f"{kind}: undo_all {cycle}")
                    self.assertEqual(self.redo_all(), 6)
                    self.assertEqual(self.state(mesh), after, f"{kind}: redo_all {cycle}")
                # the held Mesh and ColorSet still answer
                self.assertTrue(cs.is_valid)
                self.assertTrue(np.allclose(cs.data, blue))
                self.assertEqual([c.name for c in mesh.get_color_sets()], ["cs1", "cs2"])

    def test_the_existing_errors(self):
        mesh = self.build("free")
        self.assert_refused(mesh, lambda: mesh.add_color_set("pre", RGBA), ValueError, "has color set pre")
        self.assert_refused(mesh, lambda: mesh.add_color_set("x", "RGBA"), ValueError)
        self.assert_refused(mesh, lambda: setattr(ColorSet(mesh, "pre"), "data", np.zeros((3, 4))),
                            RuntimeError, "shape of data")
        self.assert_refused(mesh, lambda: setattr(ColorSet(mesh, "nope"), "data", np.zeros((8, 4))),
                            RuntimeError, "not a valid color set")
        ColorSet(mesh, "nope").delete()  # nothing to delete: nothing happens
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_undo_off_applies_and_queues_nothing(self):
        for kind in ("free", "history"):
            with self.subTest(kind):
                mesh = self.build(kind)
                cmds.undoInfo(state=False)
                try:
                    cs = mesh.add_color_set("cs1", RGBA)
                    cs.data = colours(mesh, [1.0, 0.0, 0.0, 1.0])
                    ColorSet(mesh, "part").delete()
                finally:
                    cmds.undoInfo(state=True)
                self.assertEqual([c.name for c in mesh.get_color_sets()], ["pre", "cs1"])
                self.assertTrue(np.allclose(cs.data, [1.0, 0.0, 0.0, 1.0]))
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

    def test_a_new_scene_with_colour_edits_queued(self):
        for kind in ("free", "history", "skinned"):
            with self.subTest(kind):
                mesh = self.build(kind)
                cs = mesh.add_color_set("cs1", RGBA)
                cs.data = colours(mesh, [1.0, 0.0, 0.0, 1.0])
                ColorSet(mesh, "part").delete()
                self.undo_steps(1)  # one on the redo queue
        mesh = self.build("history")
        self.walk(mesh, [lambda: mesh.add_color_set("cs1", RGBA)], cycles=1)


# --- the user's own set commands beside rig's set edits ---------------------------------------------


def set_elements(mesh, array):
    """{set name: [logical indices]} of the mesh's ``uvSet`` / ``colorSet`` elements
    (an emptied element reads under '')."""
    plug = mesh.fn_set.findPlug(array, False)
    out  = {}
    for i in range(plug.numElements()):
        element = plug.elementByPhysicalIndex(i)
        out.setdefault(element.child(0).asString(), []).append(element.logicalIndex())
    return out


def link_uv_set(mesh, uv_set):
    """A file texture linked to the UV set `uv_set` (what the UV Linking editor makes)."""
    texture = cmds.shadingNode("file", asTexture=True, name=f"tex_{uv_set}")
    index   = set_elements(mesh, "uvSet")[uv_set][0]
    cmds.uvLink(uvSet=f"{mesh.long_name}.uvSet[{index}].uvSetName", texture=texture)


class TestNativeSetEditsBesideRig(_UndoCase):
    """The user's polyUVSet / polyColorSet (the UV / Color Set Editors) beside rig's
    set edits on a history-free mesh (the round-U2 FIX review's blockers,
    runs\\rU2\\review_undo). Maya undoes its own set commands by element and by
    swapping the mesh's sets, but not those of the mesh's cached input (made by a
    vertex edit): rig's set edits sync that input first (an API write crashed
    Maya), and keep each set's uvSet[] / colorSet[] element."""

    def test_a_native_uv_set_delete_then_a_vertex_edit(self):
        # crashed Maya in the undo of set_uv_data (clearUVs), after a user's move or rig's set_points
        for vertex_edit in ("move", "set_points"):
            with self.subTest(vertex_edit):
                cmds.file(new=True, force=True)
                cube = make_mesh()
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cube.add_uv_set("a"),
                    lambda: cube.set_uv_data(uv_data(cube, "map1", "a", 0.3), "a"),
                    lambda: cmds.polyUVSet(cube.name, delete=True, uvSet="a"),
                    (lambda: move_vertices(cube)) if vertex_edit == "move" else (lambda: cube.set_points(lifted(cube))),
                ], msg=vertex_edit)

    def test_a_native_copy_rig_delete_then_a_vertex_edit(self):
        # crashed Maya in the undo of delete_uv_set (setUVs)
        cube = make_mesh()
        cmds.flushUndo()
        self.walk(cube, [
            lambda: cmds.polyUVSet(cube.name, copy=True, uvSet="map1", newUVSet="a"),
            lambda: cube.delete_uv_set("a"),
            lambda: move_vertices(cube),
        ])

    def test_a_native_colour_set_delete_then_a_vertex_edit(self):
        # crashed Maya in the undo of ColorSet.data (clearColors). After the vertex edit the
        # undo of Maya's own delete brings the colours back inexactly (Maya, not rig), so the
        # walk checks the ends: where it started, where it ended, both ways, twice
        for vertex_edit in ("move", "set_points"):
            with self.subTest(vertex_edit):
                cmds.file(new=True, force=True)
                mesh = make_mesh()
                cmds.flushUndo()
                before = self.state(mesh)
                mesh.add_color_set("cs", RGBA)
                ColorSet(mesh, "cs").data = colours(mesh, [0.1, 0.5, 0.9, 1.0])
                cmds.polyColorSet(mesh.name, delete=True, colorSet="cs")
                if vertex_edit == "move":
                    move_vertices(mesh)
                else:
                    mesh.set_points(lifted(mesh))
                after = self.state(mesh)
                for cycle in range(2):
                    self.assertEqual(self.undo_all(), 4)
                    self.read_through(mesh)
                    self.assertEqual(self.state(mesh), before, f"{vertex_edit} undo_all {cycle}")
                    self.assertEqual(self.redo_all(), 4)
                    self.read_through(mesh)
                    self.assertEqual(self.state(mesh), after, f"{vertex_edit} redo_all {cycle}")

    def test_rig_edits_after_the_undo_of_native_set_edits(self):
        # Maya alone leaves the sets 'a' and 'cs' out of the mesh's cached input: an API
        # write into them crashed Maya, and rename_uv_set did nothing
        cube = make_mesh()
        cmds.polyUVSet(cube.name, copy=True, uvSet="map1", newUVSet="a")
        cmds.polyColorSet(cube.name, create=True, colorSet="cs", representation="RGBA")
        cmds.polyUVSet(cube.name, delete=True, uvSet="a")
        cmds.polyColorSet(cube.name, delete=True, colorSet="cs")
        move_vertices(cube)
        self.undo_steps(3)
        self.assertEqual(list(cube.uv_sets), ["map1", "a"])
        self.assertEqual([c.name for c in cube.get_color_sets()], ["cs"])
        self.walk(cube, [
            lambda: cube.set_uv_data(uv_data(cube, "map1", "a", 0.5), "a"),
            lambda: setattr(ColorSet(cube, "cs"), "data", colours(cube, [0.2, 0.4, 0.6, 1.0])),
            lambda: cube.rename_uv_set("r", "a"),
        ])
        self.assertEqual(list(cube.uv_sets), ["map1", "r"])

    def test_a_native_create_rig_delete(self):
        # the redo of rig's delete deleted nothing, and the next undo made a set 'a1'
        for kind in HISTORY_FREE:
            with self.subTest(kind):
                cmds.file(new=True, force=True)
                cube = make_mesh(kind)
                cmds.flushUndo()
                self.walk(cube, [
                    lambda: cmds.polyUVSet(cube.name, create=True, uvSet="a"),
                    lambda: cube.delete_uv_set("a"),
                ], msg=kind)
                self.walk(cube, [
                    lambda: cmds.polyUVSet(cube.name, copy=True, uvSet="map1", newUVSet="b"),
                    lambda: cube.delete_uv_set("b"),
                ], msg=f"{kind} copy")
                self.walk(cube, [
                    lambda: cmds.polyColorSet(cube.name, create=True, colorSet="cs", representation="RGBA"),
                    lambda: ColorSet(cube, "cs").delete(),
                    lambda: cube.set_points(lifted(cube)),
                ], msg=f"{kind} colour")
                self.assertEqual(list(cube.uv_sets), ["map1"])

    def test_rig_add_then_a_native_delete(self):
        # rig's add, undone and redone, made the set in a new element, which the redo of
        # Maya's delete then left in place
        cube = make_mesh()
        cmds.flushUndo()
        self.walk(cube, [
            lambda: cube.add_uv_set("a"),
            lambda: cube.set_uv_data(uv_data(cube, "map1", "a", 0.3), "a"),
            lambda: cmds.polyUVSet(cube.name, delete=True, uvSet="a"),
            lambda: move_vertices(cube),
        ])


class TestSetElementsAndLinks(_UndoCase):
    """A set keeps its uvSet[] / colorSet[] element through rig's undo and redo, and
    the element its connections (a uvLink to a texture). The round-U2 FIX review's
    blocker: delete_uv_set's undo dropped the links of the set and of every later
    set, and each cycle moved the sets to new elements."""

    def build(self):
        cube = make_mesh()
        for name, scale in (("a", 0.2), ("b", 0.4)):
            cube.add_uv_set(name)
            cube.set_uv_data(uv_data(cube, "map1", name, scale), name)
            link_uv_set(cube, name)
        cmds.flushUndo()
        return cube

    def test_delete_keeps_the_elements_and_the_links(self):
        for deleted in ("a", "b"):  # a set in the middle, the last set
            with self.subTest(deleted):
                cmds.file(new=True, force=True)
                cube     = self.build()
                elements = set_elements(cube, "uvSet")
                self.assertEqual(elements, {"map1": [0], "a": [1], "b": [2]})
                self.walk(cube, [lambda: cube.delete_uv_set(deleted)], msg=deleted)  # connections included
                self.undo_steps(1)
                self.assertEqual(set_elements(cube, "uvSet"), elements)
                for name, chooser in (("a", "uvChooser1"), ("b", "uvChooser2")):
                    self.assertEqual(
                        cmds.listConnections(f"{cube.long_name}.uvSet[{elements[name][0]}].uvSetName", plugs=True),
                        [f"{chooser}.uvSets[0]"],
                    )

    def test_add_undo_and_redo_keep_the_element(self):
        cube = make_mesh()
        cmds.flushUndo()
        self.walk(cube, [lambda: cube.add_uv_set("a"), lambda: cube.add_color_set("cs", RGBA)])
        self.assertEqual(set_elements(cube, "uvSet"), {"map1": [0], "a": [1]})
        self.assertEqual(set_elements(cube, "colorSet"), {"cs": [0]})
        self.undo_steps(2)
        self.assertEqual(set_elements(cube, "uvSet"), {"map1": [0]})
        self.assertEqual(set_elements(cube, "colorSet"), {})

    def test_colour_delete_keeps_the_elements(self):
        mesh = make_mesh()
        for name in ("c0", "c1", "c2"):
            mesh.add_color_set(name, RGBA)
            ColorSet(mesh, name).data = colours(mesh, [0.1, 0.2, 0.3, 1.0])
        cmds.flushUndo()
        elements = set_elements(mesh, "colorSet")
        self.assertEqual(elements, {"c0": [0], "c1": [1], "c2": [2]})
        self.walk(mesh, [lambda: ColorSet(mesh, "c1").delete(), lambda: ColorSet(mesh, "c0").delete()])
        self.undo_steps(2)
        self.assertEqual(set_elements(mesh, "colorSet"), elements)
        self.assertEqual([c.name for c in mesh.get_color_sets()], ["c0", "c1", "c2"])


# --- several edits: create, chunks, errors, undo off --------------------------------------------


class TestMeshEditsTogether(_UndoCase):
    """rig's mesh edits with ``Mesh.create``, inside ``rig.undo_chunk``, with an error,
    and with undo off."""

    def edits(self, mesh):
        return [
            lambda: mesh.set_points(lifted(mesh)),
            lambda: mesh.add_uv_set("uv2"),
            lambda: mesh.set_uv_data(uv_data(mesh, "map1", "uv2", 0.5), "uv2"),
            lambda: mesh.set_uv_data(per_face_vertex_uvs(mesh), "map1"),
            lambda: mesh.rename_uv_set("uv3", "uv2"),
            lambda: mesh.add_color_set("cs1", RGBA),
            lambda: setattr(ColorSet(mesh, "cs1"), "data", colours(mesh, [1.0, 0.0, 0.0, 1.0])),
            lambda: mesh.delete_uv_set("uv3"),
            lambda: ColorSet(mesh, "cs1").delete(),
        ]

    def test_after_mesh_create_with_and_without_holes_and_uvs(self):
        source = cmds.polyCreateFacet(  # a face with one square hole, with its map1
            point=[(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4), (), (1, 0, 1), (3, 0, 1), (3, 0, 3), (1, 0, 3)],
            name="src",
            constructionHistory=False,
        )[0]
        holed_data, holed_uvs = Mesh(source).serialize(world_space=False)
        self.assertIsNotNone(holed_data.hole_faces)
        cube_data, cube_uvs   = make_mesh(name="src2").serialize()
        cmds.delete(source, "src2")
        cmds.flushUndo()
        for label, data, uvs in (
            ("cube with uvs", cube_data, cube_uvs),
            ("cube without uvs", cube_data, None),
            ("holed with uvs", holed_data, holed_uvs),
        ):
            with self.subTest(label):
                before = self.scene_state()
                mesh   = Mesh.create(data, uv_data=uvs, name="m")
                states = [self.scene_state(meshes=[mesh.long_name])]
                edits  = self.edits(mesh)
                if uvs is None:
                    edits = [e for i, e in enumerate(edits) if i not in (2, 3)]  # no UVs to copy
                for edit in edits:
                    edit()
                    states.append(self.scene_state(meshes=[mesh.long_name]))
                long_name = mesh.long_name
                for cycle in range(2):
                    for i in range(len(states) - 1, 0, -1):
                        self.undo_steps(1)
                        self.assertEqual(self.scene_state(meshes=[long_name]), states[i - 1], f"{label} {cycle} undo {i}")
                    self.undo_steps(1)  # the create
                    self.assertEqual(self.scene_state(), before, f"{label} {cycle} undo create")
                    self.redo_steps(1)
                    for i in range(1, len(states)):
                        self.assertEqual(self.scene_state(meshes=[long_name]), states[i - 1], f"{label} {cycle} redo {i}")
                        self.redo_steps(1)
                    self.assertEqual(self.scene_state(meshes=[long_name]), states[-1], f"{label} {cycle} redo all")
                    self.read_through(mesh)
                cmds.delete(long_name.split("|")[1])
                cmds.flushUndo()

    def test_inside_a_user_undo_chunk(self):
        mesh   = make_mesh()
        cmds.flushUndo()
        before = self.state(mesh)
        with rig.undo_chunk("rig.test.mesh"):
            cmds.setAttr("c.sx", 2.0)
            for edit in self.edits(mesh):
                edit()
            cmds.setAttr("c.sy", 3.0)
        after = self.state(mesh)
        self.assertEqual(self.undo_name(), "rig.test.mesh")
        for cycle in range(2):
            self.undo_steps(1)
            self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
            self.assertEqual(self.state(mesh), before, f"cycle {cycle} undo")
            self.redo_steps(1)
            self.assertEqual(self.state(mesh), after, f"cycle {cycle} redo")

    def test_an_error_inside_a_chunk(self):
        mesh   = make_mesh()
        cmds.flushUndo()
        before = self.state(mesh)
        with self.assertRaises(ValueError):
            with rig.undo_chunk("rig.test.error"):
                mesh.set_points(lifted(mesh))
                mesh.add_color_set("cs1", RGBA)
                mesh.set_points(np.zeros((3, 3)))  # refused: nothing more is edited
        after = self.state(mesh)
        self.assertEqual(self.undo_name(), "rig.test.error")
        self.undo_steps(1)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertEqual(self.state(mesh), before)
        self.redo_steps(1)
        self.assertEqual(self.state(mesh), after)

    def test_undo_off_applies_every_edit_and_queues_nothing(self):
        mesh = make_mesh()
        on   = make_mesh(name="d")
        cmds.flushUndo()
        for edit in self.edits(on):
            edit()
        expected = mesh_state(on.long_name)
        cmds.undoInfo(state=False)
        try:
            for edit in self.edits(mesh):
                edit()
        finally:
            cmds.undoInfo(state=True)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
        self.assertEqual(mesh_state(mesh.long_name), expected)


# --- skin weights beside mesh edits -------------------------------------------------------------


class TestSkinWeightsBesideMeshEdits(_UndoCase):
    """Skin weights and mesh edits on the same mesh, one step each."""

    def test_walk(self):
        from cgmath.geometry import SkinData

        mesh = make_mesh()
        cmds.select(clear=True)
        j1 = cmds.joint(name="j1", position=(0, -1, 0))
        cmds.select(clear=True)
        j2 = cmds.joint(name="j2", position=(0, 1, 0))
        cmds.select(clear=True)
        cmds.flushUndo()
        skins = ["cShape_skincluster"]

        def state():
            return self.scene_state(meshes=[mesh.long_name], skins=skins)

        states = [state()]
        for edit in (
            lambda: mesh.set_points(lifted(mesh)),
            lambda: SkinCluster.create(mesh.name, SkinData(influences=[j1, j2], weights=np.tile([0.75, 0.25], (8, 1)))),
            lambda: SkinCluster(skins[0]).set_weights(np.tile([0.25, 0.75], (8, 1))),
            lambda: mesh.add_color_set("cs1", RGBA),
        ):
            edit()
            states.append(state())
        for cycle in range(2):
            for i in range(len(states) - 1, 0, -1):
                self.undo_steps(1)
                self.assert_state_equal(state(), states[i - 1], f"cycle {cycle} undo to {i - 1}")
            for i in range(1, len(states)):
                self.redo_steps(1)
                self.assert_state_equal(state(), states[i], f"cycle {cycle} redo to {i}")
