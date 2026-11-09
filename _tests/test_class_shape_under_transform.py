"""A new shape under an existing transform (feat/shape-under-transform, commit 2; the user's
decision of 2026-10-29): ``transform=`` on ``Mesh.create``, ``NurbsCurve.create``,
``NurbsSurface.create`` and their class-form ``load``. Only the shape is made, under the target,
with ``world_space=`` (keyword-only, False by default) saying how the data's points go there:

* False: the data's points (and a mesh's normals) become the shape's own, as they are; the
  data's matrix is not used.
* True: they land where the data puts them in the world, through its matrix, taken into the
  target's space (the inverse of its world matrix; a mesh's locked normals through the inverse
  transpose).
  Without ``transform=`` it freezes the shape there (test_class_shape_decisions, commit 3, as for
  a singular target and the shape's namespace).

The shape rides the usual recorded ``createNode`` transform, then goes under the target (as
``parent -shape -relative``: a journaled API edit, the selection and the hilite list never
touched; under an instanced target, or with ``journal.API_EDITS`` off, the recorded command,
its selection change put back), the ride is deleted and the shape named ``<target>Shape`` (or
``name=``), inside the typed create: one undo step, only the shape a container member. The
target is read before anything is made (missing: NodeNotFoundError; a shape or a DG node:
NodeTypeError, a TypeError); a joint, a locked and a referenced target are taken; an instanced
one only with ``world_space=False``. An error once the shape is under the target deletes it.

Each test failed against the source files of 407a35e (negative control); none is a guard.
"""

import inspect
import os
import shutil
import tempfile

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig import container, NodeNotFoundError
from rig._internal import journal
from rig.nodetypes import Joint, Mesh, NurbsCurve, NurbsSurface, Transform
from rig.nodetypes.errors import NodeTypeError
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk, mesh_state, scene_nodes
from rig._tests.test_class_shape_space import _cylinder, _uneven

WORLD   = OpenMaya.MSpace.kWorld
OBJECT  = OpenMaya.MSpace.kObject
CLASSES = (Mesh, NurbsCurve, NurbsSurface)


# -- reads


def _xyz(points):
    """An MPointArray or an MFloatVectorArray, (N, 3)."""
    return np.array([(p.x, p.y, p.z) for p in points])


def _local(shape):
    return _xyz(shape.get_points(world_space=False))


def _world(shape):
    return _xyz(shape.get_points(world_space=True))


def _children(node):
    return cmds.listRelatives(node, children=True, fullPath=True) or []


def _rides():
    """The transforms a create rides with that are left in the scene (none, ever)."""
    return cmds.ls("polySurface*", "curve*", "surface*", type="transform")


def _data_points(data):
    """The points a create builds from `data`, (N, 3): a MeshData's, a B-spline's CVs (a periodic
    direction's wrap included, as Maya stores it)."""
    if hasattr(type(data), "cv"):
        return np.asarray(data.cv, dtype=float).reshape(-1, 3)
    return np.asarray(data.points, dtype=float).reshape(-1, 3)


def _shading(mesh):
    """``(hard edges, locked normals)`` of a mesh."""
    fn = mesh.get_fn_set()
    return (
        sum(not fn.isEdgeSmooth(e) for e in range(fn.numEdges)),
        sum(fn.isNormalLocked(n) for n in range(fn.numNormals)),
    )


def _normals(mesh, space=WORLD):
    """The normal of every face-vertex, face by face, in `space` (world by default)."""
    fn = mesh.get_fn_set()
    return np.array([
        list(fn.getFaceVertexNormal(f, v, space))
        for f in range(fn.numPolygons) for v in fn.getPolygonVertices(f)
    ])


def _surface_normals(surface, space=WORLD):
    """The normal of a surface at a few (u, v) inside its range, in `space` (world by default)."""
    fn = surface.get_fn_set()
    (u0, u1), (v0, v1) = fn.knotDomainInU, fn.knotDomainInV
    return np.array([
        list(fn.normal(u0 + (u1 - u0) * a, v0 + (v1 - v0) * b, space))
        for a, b in ((0.3, 0.2), (0.5, 0.5), (0.7, 0.8))
    ])


# -- the scene


def _sheared(name="ctl", kind="transform"):
    """A transform (or joint) moved, rotated and scaled under a group moved, rotated and unevenly
    scaled: its world matrix has a shear. Its long name."""
    grp = cmds.createNode("transform", name=f"{name}_grp")
    cmds.xform(grp, translation=(1, 1, 1), rotation=(40, 10, -30), scale=(2, 1, 0.5))
    node = cmds.createNode(kind, name=name, parent=grp)
    cmds.xform(node, translation=(-2, 3, 1), rotation=(15, -40, 70), scale=(1.5, 0.75, 2))
    return cmds.ls(node, long=True)[0]


def _cube():
    """A 2 x 2 x 2 cube with two UV sets, six edges soft (the rest hard) and a locked normal,
    moved, rotated and scaled: its Mesh."""
    t = cmds.polyCube(name="src", subdivisionsX=2, subdivisionsY=2, subdivisionsZ=2)[0]
    cmds.polySoftEdge(f"{t}.e[0:5]", angle=180)
    cmds.polyNormalPerVertex(f"{t}.vtx[0]", xyz=(0.0, 1.0, 0.0))
    cmds.polyUVSet(t, copy=True, uvSet="map1", newUVSet="uv2")
    cmds.polyEditUV(f"{t}.map[0:3]", u=0.5, v=0.25, uvSetName="uv2")
    cmds.delete(t, constructionHistory=True)
    cmds.xform(t, translation=(4, 5, 6), rotation=(10, 20, 30), scale=(2, 1, 0.5))
    return Mesh(t)


def _circle():
    """A periodic circle, moved and rotated: its NurbsCurve."""
    t = cmds.circle(name="crv", normal=(0, 1, 0), radius=2, constructionHistory=False)[0]
    cmds.xform(t, translation=(1, 2, 3), rotation=(30, 45, 60))
    return NurbsCurve(t)


def _ball():
    """A sphere (periodic in V), moved and scaled: its NurbsSurface."""
    t = cmds.sphere(name="ball", radius=1.5, constructionHistory=False)[0]
    cmds.xform(t, translation=(0, 3, -2), rotation=(0, 0, 20), scale=(1, 2, 1))
    return NurbsSurface(t)


def _source(cls):
    return {Mesh: _cube, NurbsCurve: _circle, NurbsSurface: _ball}[cls]()


def _args(shape, world_space=False):
    """What ``type(shape).create`` takes for `shape`'s data: ``(MeshData, UVList)`` or ``(data,)``."""
    data = shape.serialize(world_space=world_space)
    return data if isinstance(data, tuple) else (data,)


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.select(clear=True)

    def tearDown(self):
        cmds.namespace(set=":")
        super().tearDown()

    def folder(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        return folder

    def assert_nothing_made(self, error, pattern, call):
        """`call` raises `error` matching `pattern` with the scene as it was and no undo step."""
        before = scene_nodes()
        cmds.flushUndo()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(scene_nodes(), before)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))


class TestBothSpaces(_Case):
    """Under a moved, rotated, sheared target: False takes the data's points as the shape's own,
    True puts them where the data puts them in the world, from object- or world-space data."""

    def test_false_takes_the_points_as_they_are(self):
        target = _sheared()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args  = _args(_source(cls))
                shape = cls.create(*args, transform=target)
                self.assertEqual(shape.get_parent().get_long_name(), target)
                np.testing.assert_array_equal(_local(shape), _data_points(args[0]))
        np.testing.assert_allclose(cmds.getAttr(f"{target}.translate")[0], (-2, 3, 1), atol=1e-12)

    def test_true_lands_where_the_data_puts_them(self):
        target = _sheared()
        for cls in CLASSES:
            for space in (False, True):
                with self.subTest(cls=cls.__name__, data_in_world_space=space):
                    source = _source(cls)
                    shape  = cls.create(*_args(source, space), transform=target, world_space=True)
                    atol   = 1e-5 if cls is Mesh else 1e-9  # a mesh holds float32 points
                    np.testing.assert_allclose(_world(shape), _world(source), atol=atol)
                    cmds.delete(source.get_parent().get_long_name())

    def test_a_mesh_s_normals_follow_its_points(self):
        """Re-pinned ("lock only real locks", they were all set): a flat face's and a locked normal
        land where the source's are; a soft fan's is Maya's own for the points under the target."""
        target = _sheared()
        source = _cube()
        shape  = Mesh.create(*_args(source), transform=target, world_space=True)
        ids    = list(shape.get_fn_set().getNormalIds()[1])
        away   = np.abs(_normals(shape) - _normals(source)).max(axis=1) > 1e-5
        self.assertEqual(int(np.count_nonzero(away)), 6)
        self.assertTrue((np.bincount(ids)[ids] > 1)[away].all())  # a soft fan's normal has several face-vertices
        self.assertEqual(_shading(shape), _shading(source))

    def test_false_keeps_a_mesh_s_normals_as_its_own(self):
        """world_space=False: the data's normals are the shape's own, as its points are (in object
        space, not moved with the target's shear)."""
        target = _sheared()
        source = _cube()
        shape  = Mesh.create(*_args(source), transform=target)
        np.testing.assert_allclose(_normals(shape, OBJECT), _normals(source, OBJECT), atol=1e-5)
        self.assertEqual(_shading(shape), _shading(source))

    def test_points_and_older_data_are_world_positions(self):
        target = _sheared()
        points = [(0, 0, 0), (1, 2, 0), (2, 0, 1), (3, 1, 1), (4, 0, 0)]
        curve  = NurbsCurve.create(points, transform=target, world_space=True)
        np.testing.assert_allclose(_world(curve), points, atol=1e-9)
        curve = NurbsCurve.create(points, transform=target)
        np.testing.assert_allclose(_local(curve), points, atol=0)

    def test_an_object_s_transform(self):
        from cgmath.render.scene import Object

        target = _sheared()
        data   = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0]).serialize(include_uvs=False)
        obj    = Object(name="hero", mesh=data, translate=(5, 0, 0))
        np.testing.assert_allclose(_local(Mesh.create(obj, transform=target)), data.points, atol=0)
        placed = Mesh.create(obj, transform=target, world_space=True)
        np.testing.assert_allclose(_world(placed), data.points + (5, 0, 0), atol=1e-5)


class TestAJointTarget(_Case):
    def test_both_spaces_under_a_joint(self):
        jnt = cmds.createNode("joint", name="jnt")
        cmds.setAttr(f"{jnt}.jointOrient", 10, 20, 30)
        cmds.xform(jnt, translation=(1, 1, 1), rotation=(5, 0, -15), scale=(1, 2, 1))
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source = _source(cls)
                own    = cls.create(*_args(source), transform=Joint(jnt))
                placed = cls.create(*_args(source, True), transform="jnt", world_space=True)
                self.assertEqual([str(own.get_parent()), str(placed.get_parent())], ["jnt", "jnt"])
                np.testing.assert_array_equal(_local(own), _local(source))
                np.testing.assert_allclose(_world(placed), _world(source), atol=1e-5 if cls is Mesh else 1e-9)

    def test_the_shape_is_read_by_its_own_name(self):
        """A joint is no transform to Geometry's lookup (it reads nodeType 'transform'): a shape under
        it is reached by its name."""
        cmds.createNode("joint", name="jnt")
        shape = Mesh.create(*_args(_cube()), transform="jnt")
        self.assertEqual(Mesh("jntShape"), shape)
        with self.assertRaises(NodeTypeError):
            Mesh("jnt")


class TestLockedAndReferencedTargets(_Case):
    def test_a_locked_target(self):
        target = cmds.createNode("transform", name="ctl")
        cmds.lockNode(target, lock=True)
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args   = _args(_source(cls))
                before = scene_nodes()
                shape  = cls.create(*args, transform=target)
                self.assertEqual(str(shape.get_parent()), "ctl")
                self.assertEqual(cmds.lockNode(target, query=True, lock=True), [True])
                made = scene_nodes()
                self.undo_steps(1)
                self.assertEqual(scene_nodes(), before)
                self.redo_steps(1)
                self.assertEqual(scene_nodes(), made)

    def test_a_referenced_target(self):
        path = os.path.join(self.folder(), "ref_target.ma")
        cmds.xform(cmds.createNode("transform", name="refctl"), translation=(2, 0, -1), rotation=(0, 30, 0))
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="ref")
        source = _cube()
        cmds.flushUndo()
        shape = Mesh.create(*_args(source), transform="ref:refctl", world_space=True)
        # re-pinned (user decision 2026-10-30): the default name is in the target's namespace
        self.assertEqual(shape.get_long_name(), "|ref:refctl|ref:refctlShape")
        np.testing.assert_allclose(_world(shape), _world(source), atol=1e-5)
        node  = cmds.referenceQuery("ref:refctl", referenceNode=True)
        edits = cmds.referenceQuery(node, editStrings=True)
        self.assertTrue(any(e.startswith("parent -s -r ") and e.endswith('"|ref:refctl"') for e in edits), edits)
        uuid = cmds.ls(shape.get_long_name(), uuid=True)[0]
        self.undo_steps(1)
        self.assertEqual(_children("|ref:refctl"), [])
        self.assertEqual(cmds.referenceQuery(node, editStrings=True), [])
        self.redo_steps(1)
        self.assertEqual(cmds.ls(uuid, long=True), ["|ref:refctl|ref:refctlShape"])
        self.assertEqual(cmds.referenceQuery(node, editStrings=True), edits)


class TestAnInstancedTarget(_Case):
    def instanced(self):
        cmds.createNode("transform", name="g1")
        cmds.xform(cmds.createNode("transform", name="g2"), translation=(5, 0, 0))
        cmds.createNode("transform", name="ictl", parent="g1")
        cmds.parent("|g1|ictl", "g2", addObject=True, relative=True)
        return "|g1|ictl"

    def test_false_puts_the_shape_under_every_instance(self):
        target = self.instanced()
        source = _cube()
        shape  = Mesh.create(*_args(source), transform=target)
        paths  = cmds.ls(shape.get_long_name(), allPaths=True, long=True)
        self.assertEqual(sorted(paths), ["|g1|ictl|ictlShape", "|g2|ictl|ictlShape"])
        for path in paths:
            self.assertEqual(cmds.listSets(object=path), ["initialShadingGroup"])
        np.testing.assert_array_equal(_local(shape), _local(source))

    def test_the_shape_is_returned_under_the_instance_named(self):
        """The returned shape's path runs through the target as the call named it: its world
        queries read through that instance (|g2 is moved 5 in x)."""
        self.instanced()
        source = _cube()
        for target, offset in (("|g1|ictl", 0.0), ("|g2|ictl", 5.0)):
            with self.subTest(target=target):
                shape = Mesh.create(*_args(source), transform=target)
                self.assertEqual(shape.get_long_name().rpartition("|")[0], target)
                np.testing.assert_allclose(_world(shape), _local(source) + (offset, 0, 0), atol=1e-5)

    def test_true_is_refused_before_anything_is_made(self):
        target = self.instanced()
        args   = {cls: _args(_source(cls)) for cls in CLASSES}
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                self.assert_nothing_made(
                    ValueError,
                    rf"^{cls.__name__}\.create\(\): 'g1\|ictl' is instanced \(2 paths\) and the new shape goes "
                    r"under every instance, so world_space=True has no one place to put the points; "
                    r"world_space=False \(the default\) takes the data's points as the shape's own$",
                    lambda: cls.create(*args[cls], transform=target, world_space=True),
                )
        # a parent's instance makes the target instanced too
        cmds.createNode("transform", name="kid", parent=target)
        self.assert_nothing_made(ValueError, "is instanced",
                                 lambda: Mesh.create(*args[Mesh], transform="|g1|ictl|kid", world_space=True))


class TestTheTargetIsReadFirst(_Case):
    """A missing target, or a node that is no transform, is refused before anything is made: no
    node and no undo step; the class-form load reads it before the file."""

    def test_a_missing_target(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args = _args(_source(cls))
                self.assert_nothing_made(NodeNotFoundError, r"^no transform named 'nope'",
                                         lambda: cls.create(*args, transform="nope"))
                self.assert_nothing_made(NodeNotFoundError, r"^no transform named 'nope'",
                                         lambda: cls.load("no_such_file.npz", transform="nope"))

    def test_a_shape_or_a_dg_node(self):
        box  = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        args = {cls: _args(_source(cls)) for cls in CLASSES}
        cmds.createNode("multiplyDivide", name="md")
        for cls in CLASSES:
            for target, shown, kind in ((box, "boxShape", "mesh"), ("boxShape", "boxShape", "mesh"),
                                        ("md", "md", "multiplyDivide")):
                with self.subTest(cls=cls.__name__, target=str(target)):
                    self.assert_nothing_made(
                        TypeError,
                        # re-pinned (user decision 2026-10-30): any node derived from transform is one
                        rf"^{cls.__name__}\.create\(\): transform= is the transform \(or a node derived from one\) "
                        rf"the new shape goes under: '{shown}' is an? {kind}, not a transform",
                        lambda: cls.create(*args[cls], transform=target),
                    )
        with self.assertRaises(NodeTypeError):
            Mesh.create(*args[Mesh], transform="md")

    def test_anything_but_a_bool_for_the_space(self):
        target = cmds.createNode("transform", name="ctl")
        args   = _args(_cube())
        for value in (None, 1, "world"):
            with self.subTest(value=value):
                self.assert_nothing_made(
                    TypeError,
                    # re-pinned (user decision 2026-10-30): world_space= says this without transform= too
                    rf"^Mesh\.create\(\): world_space= takes True or False \(where the data's points land: in the "
                    rf"world, or as the shape's own\), not {value!r}$",
                    lambda: Mesh.create(*args, transform=target, world_space=value),
                )

    def test_anything_but_a_bool_for_the_space_of_a_load(self):
        """The class-form load's world_space= says what create's does: not the space of the file
        (True lands right from a file saved in either space)."""
        target = cmds.createNode("transform", name="ctl")
        folder = self.folder()
        for cls in CLASSES:
            path = _source(cls).save(os.path.join(folder, f"{cls.__name__}.npz"))
            for value in (None, 1):
                with self.subTest(cls=cls.__name__, value=value):
                    self.assert_nothing_made(
                        TypeError,
                        rf"^{cls.__name__}\.load: world_space= takes True or False \(where the data's points "
                        rf"land: in the world, or as the shape's own\), not {value!r}$",
                        lambda: cls.load(path, transform=target, world_space=value),
                    )


class TestALaterErrorLeavesNothing(_Case):
    def test_a_value_refused_once_the_shape_is_under_the_target(self):
        """The value is refused on the shape under the target, by its new name: it was made, moved
        and named before the error, and deleted after it."""
        target = cmds.createNode("transform", name="ctl")
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args   = _args(_source(cls))
                before = scene_nodes()
                with self.assertRaisesRegex(
                    ValueError, r"^Cannot inject sequence of size 2 into scalar attribute 'ctlShape\.visibility'"
                ):
                    cls.create(*args, transform=target, visibility=(1, 2))
                self.assertEqual(scene_nodes(), before)
                self.assertEqual((_children(target), _rides()), ([], []))

    def test_a_name_maya_refuses_once_the_shape_is_under_the_target(self):
        target = cmds.createNode("transform", name="ctl")
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args   = _args(_source(cls))
                before = scene_nodes()
                with self.assertRaisesRegex(RuntimeError, "no legal characters"):
                    cls.create(*args, transform=target, name="#")
                self.assertEqual(scene_nodes(), before)
                self.assertEqual((_children(target), _rides()), ([], []))


class TestOneUndoStep(_Case):
    def test_undo_takes_the_shape_redo_brings_back_the_same_one(self):
        target = _sheared()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args = _args(_source(cls))
                cmds.flushUndo()
                before = self.scene_state()
                shape  = cls.create(*args, transform=target, world_space=True)
                path   = shape.get_long_name()
                uuid   = cmds.ls(path, uuid=True)[0]
                after  = self.scene_state(meshes=[path] if cls is Mesh else [])
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.create")
                self.undo_steps(1)
                self.assert_state_equal(self.scene_state(), before)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.redo_steps(1)
                self.assert_state_equal(self.scene_state(meshes=[path] if cls is Mesh else []), after)
                self.assertEqual(cmds.ls(uuid, long=True), [path])
                self.assertEqual(shape.get_long_name(), path)


class TestTheSelectionIsKept(_Case):
    """The move under the target is a journaled API edit: the selection, the hilite list and
    SelectionChanged are left alone, on create and on the class-form load, through undo and redo.
    With journal.API_EDITS off (as without rig's undo plug-in) it is the recorded parent, whose
    selection change is put back."""

    def calls(self, target):
        """``[(label, call), ...]``: each class's create and class-form load under `target`."""
        folder = self.folder()
        calls  = []
        for cls in CLASSES:
            source = _source(cls)
            args   = _args(source)
            path   = source.save(os.path.join(folder, f"{cls.__name__}.npz"))
            calls.append((f"{cls.__name__}.create", lambda c=cls, a=args: c.create(*a, transform=target)))
            calls.append((f"{cls.__name__}.load", lambda c=cls, p=path: c.load(p, transform=target)))
        return calls

    def test_create_load_undo_and_redo(self):
        target = cmds.createNode("transform", name="ctl")
        cmds.createNode("transform", name="pickA")
        cmds.createNode("transform", name="pickB")
        calls = self.calls(target)
        for picked in (["pickB", "pickA"], []):
            for label, call in calls:
                with self.subTest(call=label, picked=picked):
                    cmds.select(clear=True)
                    for node in picked:
                        cmds.select(node, add=True)
                    call()
                    self.assertEqual(cmds.ls(orderedSelection=True), picked)
                    self.undo_steps(1)
                    self.assertEqual(cmds.ls(orderedSelection=True), picked)
                    self.redo_steps(1)
                    self.assertEqual(cmds.ls(orderedSelection=True), picked)

    def test_the_hilite_list_and_no_selection_change(self):
        """Components picked on a hilited mesh: the hilite list and the selection as they were after
        the call, its undo and its redo, and no SelectionChanged during the call (the recorded parent
        clears the hilite list and changes the selection three times)."""
        target = cmds.createNode("transform", name="ctl")
        box    = cmds.polyCube(name="box", constructionHistory=False)[0]
        calls  = self.calls(target)
        events = []
        listen = OpenMaya.MEventMessage.addEventCallback("SelectionChanged", lambda *_: events.append(1))
        self.addCleanup(OpenMaya.MMessage.removeCallback, listen)
        for label, call in calls:
            with self.subTest(call=label):
                cmds.select(clear=True)
                cmds.hilite(box, replace=True)
                cmds.select(f"{box}.vtx[1]", f"{box}.vtx[5]")
                before = (cmds.ls(hilite=True), cmds.ls(orderedSelection=True))
                del events[:]
                call()
                self.assertEqual(len(events), 0)
                self.assertEqual((cmds.ls(hilite=True), cmds.ls(orderedSelection=True)), before)
                self.undo_steps(1)
                self.assertEqual((cmds.ls(hilite=True), cmds.ls(orderedSelection=True)), before)
                self.redo_steps(1)
                self.assertEqual((cmds.ls(hilite=True), cmds.ls(orderedSelection=True)), before)

    def test_the_recorded_parent_puts_the_selection_back(self):
        target = cmds.createNode("transform", name="ctl")
        cmds.createNode("transform", name="picked")
        self.addCleanup(setattr, journal, "API_EDITS", journal.API_EDITS)
        journal.API_EDITS = False
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args = _args(_source(cls))
                cmds.select("picked", replace=True)
                cmds.flushUndo()
                before = scene_nodes()
                shape  = cls.create(*args, transform=target)
                self.assertEqual(cmds.ls(orderedSelection=True), ["picked"])
                self.assertEqual(shape.get_long_name(), "|ctl|ctlShape")
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.create")
                self.undo_steps(1)
                self.assertEqual(scene_nodes(), before)
                self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))
                self.assertEqual(cmds.ls(orderedSelection=True), ["picked"])


class TestTheShapeName(_Case):
    def test_after_the_target_and_numbered_by_maya(self):
        target = cmds.createNode("transform", name="ctl")
        made   = [cls.create(*_args(_source(cls)), transform=target) for cls in CLASSES]
        self.assertEqual([str(s) for s in made], ["ctlShape", "ctlShape1", "ctlShape2"])
        self.assertEqual(_children(target), ["|ctl|ctlShape", "|ctl|ctlShape1", "|ctl|ctlShape2"])

    def test_name_names_the_shape_as_given(self):
        target = cmds.createNode("transform", name="ctl")
        args   = _args(_cube())
        made   = [Mesh.create(*args, transform=target, name=name) for name in ("petal", "ctlShape7", "petal")]
        self.assertEqual([s.get_long_name() for s in made], ["|ctl|petal", "|ctl|ctlShape7", "|ctl|petal1"])

    def test_the_target_s_namespace_goes_with_the_default_name(self):
        """Re-pinned (user decision 2026-10-30): it was left out, the current namespace's instead."""
        cmds.namespace(add="char")
        target = cmds.createNode("transform", name="char:ctl")
        source = _cube()
        self.assertEqual(Mesh.create(*_args(source), transform=target).get_long_name(), "|char:ctl|char:ctlShape")
        cmds.namespace(set=":char")
        self.assertEqual(Mesh.create(*_args(source), transform=target).get_long_name(), "|char:ctl|char:ctlShape1")


class TestInAContainer(_Case):
    def test_only_the_shape_is_a_member(self):
        target = _sheared()
        args   = {cls: _args(_source(cls)) for cls in CLASSES}
        with container("box"):
            made = [cls.create(*args[cls], transform=target) for cls in CLASSES]
        self.assertEqual(
            sorted(cmds.container("box", query=True, nodeList=True)), sorted(str(s) for s in made)
        )
        self.assertIsNone(cmds.container(query=True, findContainer=[target]))
        self.assertEqual(_rides(), [])

    def test_an_explicit_name_takes_the_scope_prefix(self):
        target = cmds.createNode("transform", name="ctl")
        args   = _args(_cube())
        with container("leg"):
            with container("lower"):
                shape = Mesh.create(*args, transform=target, name="petal")
        self.assertEqual(shape.get_long_name(), "|ctl|lower_petal")
        self.assertEqual(cmds.container("leg", query=True, nodeList=True), ["lower_petal"])

    def test_container_false_registers_nothing(self):
        target = cmds.createNode("transform", name="ctl")
        args   = _args(_cube())
        with container("box"):
            Transform.create(name="other")
            Mesh.create(*args, transform=target, container=False)
        self.assertEqual(cmds.container("box", query=True, nodeList=True), ["other"])


class TestWhatTheRideGives(_Case):
    """Everything a create gives its shape today comes with it under the target."""

    def test_shading_uv_sets_hard_edges_and_locked_normals(self):
        target = _sheared()
        source = _cube()
        args   = _args(source)
        ride   = Mesh.create(*args)
        shape  = Mesh.create(*args, transform=target)
        self.assertEqual(mesh_state(shape.get_long_name()), mesh_state(ride.get_long_name()))
        self.assertEqual(shape.get_uv_sets(), ("map1", "uv2"))
        self.assertEqual(_shading(shape), _shading(source))
        self.assertEqual([str(e) for e in shape.get_shading_engines()], ["initialShadingGroup"])
        surface = NurbsSurface.create(*_args(_ball()), transform=target)
        self.assertEqual(cmds.listSets(object=surface.get_long_name()), ["initialShadingGroup"])

    def test_component_tags(self):
        target = cmds.createNode("transform", name="ctl")
        for cls, ids, token in ((Mesh, [0, 1, 2], "vtx[0:2]"), (NurbsCurve, [0, 1], "cv[0:1]"),
                                (NurbsSurface, np.array([[1, 0], [1, 1]]), "cv[1][0:1]")):
            with self.subTest(cls=cls.__name__):
                shape = cls.create(*_args(_source(cls)), transform=target)
                shape.add_tag("cap")
                shape.set_tag_contents("cap", ids)
                self.assertEqual(shape.get_tag_contents("cap"), [token])

    def test_a_holed_face(self):
        from cgmath.geometry import MeshData

        square = [(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)]
        data   = MeshData(
            indices      = np.arange(7),
            counts       = np.array([7]),
            points       = np.array(square + [(1, 0, 1), (3, 0, 1), (2, 0, 3)], float),
            hole_faces   = np.array([0]),
            hole_counts  = np.array([3]),
            hole_indices = np.array([4, 5, 6]),
        )
        target = _sheared()
        for space in (False, True):
            with self.subTest(world_space=space):
                shape = Mesh.create(data, transform=target, world_space=space)
                self.assertEqual(shape.get_fn_set().getHoles(), ((0, (4, 5, 6)),))
                np.testing.assert_allclose(_world(shape) if space else _local(shape), data.points, atol=1e-5)

    def test_a_curve_s_form_and_knots(self):
        target = _sheared()
        uneven = cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)],
                            knot=[0, 0, 0, 1, 5, 5, 5], name="uneven")
        for source in (_circle(), NurbsCurve(uneven)):
            with self.subTest(curve=str(source)):
                shape = NurbsCurve.create(source.serialize(), transform=target)
                for read in ("form", "degree", "numCVs"):
                    self.assertEqual(getattr(shape.get_fn_set(), read), getattr(source.get_fn_set(), read))
                self.assertEqual(list(shape.get_fn_set().knots()), list(source.get_fn_set().knots()))

    def test_a_surface_s_uneven_knots(self):
        target = _sheared()
        for source in (NurbsSurface(_uneven(cmds.createNode("transform", name="s"))),
                       NurbsSurface(_cylinder(cmds.createNode("transform", name="c")))):
            with self.subTest(surface=str(source)):
                shape = NurbsSurface.create(source.serialize(), transform=target, world_space=True)
                fn, own = shape.get_fn_set(), source.get_fn_set()
                for read in ("knotsInU", "knotsInV"):
                    self.assertEqual(list(getattr(fn, read)()), list(getattr(own, read)()))
                self.assertEqual((fn.formInU, fn.formInV), (own.formInU, own.formInV))
                np.testing.assert_allclose(_world(shape), _world(source), atol=1e-9)


class TestAMirroredTarget(_Case):
    def test_true_lands_and_locks_every_normal(self):
        """Pinned as documented, not fixed: under a target of negative determinant, world_space=True
        puts the points and the world normals where they were, and every normal is locked (Maya's
        own normals face the other way), as Mesh('x').load(path, world_space=True) does into a
        mirrored transform."""
        target = cmds.createNode("transform", name="mirror")
        cmds.xform(target, translation=(3, 0, 0), rotation=(0, 30, 0), scale=(-1, 1, 1))
        source = Mesh(cmds.polyCube(name="box", constructionHistory=False)[0])
        cmds.xform("box", translation=(1, 2, 3), rotation=(10, 20, 30))
        shape  = Mesh.create(*_args(source), transform=target, world_space=True)
        np.testing.assert_allclose(_world(shape), _world(source), atol=1e-5)
        np.testing.assert_allclose(_normals(shape), _normals(source), atol=1e-5)
        self.assertEqual(_shading(shape), (_shading(source)[0], shape.get_fn_set().numNormals))

    def test_true_turns_a_surface_inside_out(self):
        """Pinned as documented, not fixed: a surface's CVs land where they were, and its normals
        face the other way (inside out against the source), as NurbsSurface('x').load(path,
        world_space=True) does into a mirrored transform."""
        target = cmds.createNode("transform", name="mirror")
        cmds.xform(target, translation=(3, 0, 0), rotation=(0, 30, 0), scale=(-1, 1, 1))
        source = _ball()
        shape  = NurbsSurface.create(*_args(source, True), transform=target, world_space=True)
        np.testing.assert_allclose(_world(shape), _world(source), atol=1e-9)
        np.testing.assert_allclose(_surface_normals(shape), -_surface_normals(source), atol=1e-6)


class TestTheClassFormLoad(_Case):
    def test_both_spaces_from_files(self):
        folder = self.folder()
        target = _sheared()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source = _source(cls)
                files  = [os.path.join(folder, f"{cls.__name__}.npz")]
                placed = [os.path.join(folder, f"{cls.__name__}_world.npz")]
                if cls is Mesh:
                    files.append(os.path.join(folder, "uv.npz"))
                saved    = source.save(*files)
                in_world = source.save(*placed, world_space=True)
                own      = cls.load(*(saved if cls is Mesh else (saved,)), transform=target)
                self.assertEqual(self.undo_name(), f"rig.{cls.__name__}.load")
                self.assertEqual(own.get_parent().get_long_name(), target)
                np.testing.assert_array_equal(_local(own), _local(source))
                if cls is Mesh:
                    self.assertEqual(own.get_uv_sets(), ("map1", "uv2"))
                put  = cls.load(in_world, transform=target, world_space=True)
                path = put.get_long_name()
                np.testing.assert_allclose(_world(put), _world(source), atol=1e-5 if cls is Mesh else 1e-9)
                self.undo_steps(1)
                self.assertFalse(cmds.objExists(path))

    def test_the_signature(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                parameter = inspect.signature(cls.load).parameters["transform"]
                self.assertEqual((parameter.kind, parameter.default), (inspect.Parameter.KEYWORD_ONLY, None))


class TestTheKeywords(_Case):
    def test_create_takes_the_space_without_transform(self):
        """Re-pinned (user decision 2026-10-30): True was refused; it freezes the shape now
        (test_class_shape_decisions), and False stays the default: a new transform either way."""
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                args = _args(_source(cls))
                self.assertIsInstance(cls.create(*args, world_space=True), cls)
                self.assertIsInstance(cls.create(*args, world_space=False), cls)
                np.testing.assert_array_equal(_local(cls.create(*args, world_space=False)), _local(cls.create(*args)))

    def test_the_class_form_load_takes_it_without_transform(self):
        """Re-pinned (user decision 2026-10-30): it was refused whatever its value."""
        folder = self.folder()
        for cls, what in ((Mesh, "mesh"), (NurbsCurve, "curve"), (NurbsSurface, "surface")):
            path = _source(cls).save(os.path.join(folder, f"{what}.npz"))
            for value in (False, True):
                with self.subTest(cls=cls.__name__, world_space=value):
                    self.assertIsInstance(cls.load(path, world_space=value), cls)

    def test_the_create_signatures(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                parameters = inspect.signature(cls.create).parameters
                for name, default in (("transform", None), ("world_space", False)):
                    self.assertEqual(
                        (parameters[name].kind, parameters[name].default), (inspect.Parameter.KEYWORD_ONLY, default)
                    )

    def test_parent_names_transform(self):
        cmds.createNode("transform", name="ctl")
        args   = _args(_cube())
        ball   = _args(_ball())
        before = scene_nodes()
        for cls, data in ((Mesh, args), (NurbsCurve, ([(0, 0, 0), (1, 0, 0)], 1)), (NurbsSurface, ball)):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{cls.__name__}\.create\(\) takes no parent=: its command places the node; transform= "
                    r"makes the shape under an existing transform$",
                ):
                    cls.create(*data, parent="ctl")
                self.assertEqual(scene_nodes(), before)
