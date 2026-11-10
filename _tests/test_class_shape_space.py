"""One space rule for the shapes (feat/shape-under-transform, commit 1; the user's decision of
2026-10-29): local / object space by default everywhere, as Maya's API does, world space only
by ``world_space=True``, a keyword (a bool given by position is a TypeError). Part S, the
shapes; the transforms are in test_class_xform.

* Every query and every data method of Mesh, NurbsCurve and NurbsSurface reads object space
  by default: ``get_points``, ``get_closest_point``, ``get_uv_at_point``,
  ``get_vertex_normals``, ``get_vertices_above_plane`` / ``get_faces_above_plane``,
  ``serialize``, ``save``, and the update in place, ``Cls('x').load``, which no longer
  requires a curve's or a surface's space.
* A plug read follows the plug: an element of ``worldMesh`` / ``worldSpace`` is world space,
  every other plug object space (``outMesh``, ``local``, a deformer's output through its
  shape, a choice's output through the plug it routes); a consumer that is no shape (a
  skinCluster) serializes as it does.
* NURBS data is a cgmath shape, as a MeshData is (the user's decision of 2026-11-09, on
  cgmath r17-skeleton's model): it holds the shape's ``uuid`` and the ``parent_uuid`` of the
  transform its object-space points are relative to, no name and no matrix, and a surface its
  ``knots_u`` / ``knots_v`` (uneven knots kept as they are, nothing said; the user's
  cylinder.ma, testShape: V of degree 1 with knots 0, 0.554, 1.554, 2.108). create places its
  new transform where the data sits, the update in place puts the CVs there with
  ``world_space=True``, and a surface update refuses data that holds other knots.
* Where data sits (`TestPlacement`): the world matrix of the transform its ``parent_uuid``
  names, in ``hierarchy=`` when given (a ClipData at its frame), else in the scene by its UUID,
  else the identity (a file from another scene); an instanced scene transform is refused.
* rig's own callers that sample across meshes keep world space by keyword: transfer_from_meshes
  and a follicle's UV at a world position (the transfers' resampler: test_class_comptags).

Each test failed against the source files of 65eb6f4 (negative control), except the guards.
"""

import inspect
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig import Node
from rig.nodetypes import Follicle, Joint, Mesh, NurbsCurve, NurbsSurface, SkinCluster
from rig._tests._base import MayaTestCase

OBJECT   = OpenMaya.MSpace.kObject
WORLD    = OpenMaya.MSpace.kWorld

OPEN     = OpenMaya.MFnNurbsSurface.kOpen
PERIODIC = OpenMaya.MFnNurbsSurface.kPeriodic

# the user's cylinder.ma, testShape: periodic of degree 3 in U, degree 1 in V with these knots
CYLINDER_V_KNOTS = (0.0, 0.554097093776086, 1.554097093776086, 2.1081941875531793)


def _xyz(points):
    """An MPointArray or an MFloatVectorArray, (N, 3)."""
    return np.array([(p.x, p.y, p.z) for p in points])


def _local(shape):
    """The object-space points of `shape`, asked for by keyword (what any version reads)."""
    return _xyz(shape.get_points(world_space=False))


def _world_matrix(node):
    return np.array(cmds.getAttr(f"{node}.worldMatrix[0]")).reshape(4, 4)


def _uuid(node):
    """Maya's UUID of `node` (a rig node or a name), as the data's ``uuid`` / ``parent_uuid`` hold it."""
    return cmds.ls(str(node), uuid=True)[0]


def _surface(name, cvs, knots_u, knots_v, degrees, forms, parent):
    """A nurbsSurface made through the API with these knots (Maya's layout) under the transform
    `parent`, its shape named ``<name>Shape``: the shape's long name."""
    sel = OpenMaya.MSelectionList()
    sel.add(parent)
    shape = OpenMaya.MFnNurbsSurface().create(
        OpenMaya.MPointArray(cvs), OpenMaya.MDoubleArray(knots_u), OpenMaya.MDoubleArray(knots_v),
        degrees[0], degrees[1], forms[0], forms[1], False, sel.getDependNode(0),
    )
    return cmds.ls(cmds.rename(OpenMaya.MFnDagNode(shape).fullPathName(), f"{name}Shape"), long=True)[0]


def _cylinder(parent, name="cyl"):
    """The user's cylinder: a ring of 8 CVs (periodic, degree 3, Maya's default knots) at four
    heights (degree 1, the uneven knots of cylinder.ma). Its shape's long name."""
    ring = [(np.cos(2 * np.pi * i / 8), np.sin(2 * np.pi * i / 8)) for i in range(8)]
    cvs  = [(c, h, s) for c, s in ring + ring[:3] for h in (0.0, 1.0, 3.0, 4.0)]
    return _surface(name, cvs, range(-2, 11), CYLINDER_V_KNOTS, (3, 1), (PERIODIC, OPEN), parent)


def _uneven(parent, name="srf", knots_v=(0, 1, 3)):
    """5 x 3 CVs, degree 3 open in U with uneven knots, degree 1 in V (`knots_v`). Its shape."""
    cvs = [(u, v, 0.1 * ((u * v) % 3)) for u in range(5) for v in range(3)]
    return _surface(name, cvs, (0, 0, 0, 1, 5, 5, 5), knots_v, (3, 1), (OPEN, OPEN), parent)


def _scene():
    """Under ``grp`` (moved, rotated and unevenly scaled: a shear below it), each moved, rotated
    and scaled again: a cube, an open curve with uneven knots, a surface with uneven knots
    (degree 3 in U, 1 in V) and the user's cylinder. ``{label: shape long name}``."""
    grp = cmds.createNode("transform", name="grp")
    cmds.xform(grp, translation=(3, -2, 5), rotation=(30, 45, 60), scale=(2, 0.5, 1.5))
    cube  = cmds.polyCube(name="cube", constructionHistory=False)[0]
    curve = cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)], knot=[0, 0, 0, 1, 5, 5, 5],
                       name="crv")
    cmds.rename(cmds.listRelatives(curve, shapes=True)[0], "crvShape")
    cmds.parent(cube, curve, grp)
    srf = cmds.createNode("transform", name="srf", parent=grp)
    cyl = cmds.createNode("transform", name="cyl", parent=grp)
    for node, (t, r, s) in (
        ("|grp|cube", ((1, 2, 3), (10, 20, 30), (1.2, 0.8, 1.1))),
        ("|grp|crv", ((-1, 2, 0), (0, 90, 0), (1, 2, 1))),
        ("|grp|srf", ((0, 0, 2), (15, 0, 0), (1, 1, 3))),
        ("|grp|cyl", ((2, -1, 1), (5, 10, 15), (0.5, 1.5, 1))),
    ):
        cmds.xform(node, translation=t, rotation=r, scale=s)
    return {
        "cube":     "|grp|cube|cubeShape",
        "curve":    "|grp|crv|crvShape",
        "surface":  _uneven(srf),
        "cylinder": _cylinder(cyl),
    }


def _on_curve(fn, space=WORLD, samples=97):
    lo, hi = fn.knotDomain
    return np.array([list(fn.getPointAtParam(u, space))[:3] for u in np.linspace(lo, hi, samples)])


def _on_surface(fn, space=WORLD, samples=41):
    (u0, u1), (v0, v1) = fn.knotDomainInU, fn.knotDomainInV
    params = [(u, v) for u in np.linspace(u0, u1, samples) for v in np.linspace(v0, v1, samples)]
    return np.array([list(fn.getPointAtParam(u, v, space))[:3] for u, v in params])


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.dir = tempfile.mkdtemp(prefix="rig_shape_space_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def path(self, name):
        return os.path.join(self.dir, name)


# ---------------------------------------------------------------------------------------------
class TestTheShapeSpaceKeyword(_Case):
    """Every world_space= of the shapes is keyword-only and False by default."""

    def nodes(self):
        cube  = Mesh(cmds.polyCube(name="cube", constructionHistory=False)[0])
        crv   = NurbsCurve(cmds.circle(name="crv", constructionHistory=False)[0])
        plane = NurbsSurface(cmds.nurbsPlane(name="plane", constructionHistory=False)[0])
        return cube, crv, plane

    def test_every_world_space_is_a_keyword_defaulting_to_false(self):
        cube, crv, plane = self.nodes()
        for function in (
            Mesh.get_points, Mesh.set_points, Mesh.get_closest_point, Mesh.get_vertices_above_plane,
            Mesh.get_faces_above_plane, Mesh.get_uv_at_point, Mesh.get_vertex_normals, Mesh.serialize, Mesh.save,
            NurbsCurve.get_points, NurbsCurve.serialize, NurbsCurve.save,
            NurbsSurface.get_points, NurbsSurface.serialize, NurbsSurface.save,
            cube.load, crv.load, plane.load, Mesh.get_output_attr,
        ):
            with self.subTest(function.__qualname__):
                parameter = inspect.signature(function).parameters["world_space"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, False)
        # the parts a serialize writes: the same three flags on the three shapes, keyword-only, True
        for function in (Mesh.serialize, NurbsCurve.serialize, NurbsSurface.serialize):
            for flag in ("uvs", "normals", "colors"):
                with self.subTest(f"{function.__qualname__} {flag}"):
                    parameter = inspect.signature(function).parameters[flag]
                    self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertIs(parameter.default, True)

    def test_a_bool_by_position_is_a_type_error(self):
        cube, crv, plane = self.nodes()
        cmds.xform("cube", translation=(1, 2, 3))
        paths  = [node.save(self.path(f"{type(node).__name__}.npz")) for node in (cube, crv, plane)]
        points = cube.get_points(world_space=False)
        up     = np.array([0.0, 1.0, 0.0])
        nodes, before = cmds.ls(long=True), _xyz(cube.get_points(world_space=True))
        for label, call in (
            ("Mesh.get_points", lambda: cube.get_points(True)),
            ("Mesh.set_points", lambda: cube.set_points(points, True)),
            ("Mesh.get_closest_point", lambda: cube.get_closest_point((0, 0, 0), True)),
            ("Mesh.get_vertices_above_plane", lambda: cube.get_vertices_above_plane(np.zeros(3), up, None, True)),
            ("Mesh.get_faces_above_plane", lambda: cube.get_faces_above_plane(np.zeros(3), up, None, True)),
            ("Mesh.get_uv_at_point", lambda: cube.get_uv_at_point((0, 0, 0), None, True)),
            ("Mesh.get_vertex_normals", lambda: cube.get_vertex_normals(False, True)),
            ("Mesh.serialize", lambda: cube.serialize(True)),
            ("Mesh.serialize uvs", lambda: cube.serialize(True, False)),
            ("Mesh.save", lambda: cube.save(self.path("m.npz"), None, True)),
            ("Mesh('x').load", lambda: cube.load(paths[0], None, True)),
            ("NurbsCurve.get_points", lambda: crv.get_points(True)),
            ("NurbsCurve.serialize", lambda: crv.serialize(True)),
            ("NurbsCurve.save", lambda: crv.save(self.path("c.npz"), True)),
            ("NurbsCurve('x').load", lambda: crv.load(paths[1], True)),
            ("NurbsSurface.get_points", lambda: plane.get_points(True)),
            ("NurbsSurface.serialize", lambda: plane.serialize(True)),
            ("NurbsSurface.save", lambda: plane.save(self.path("s.npz"), True)),
            ("NurbsSurface('x').load", lambda: plane.load(paths[2], True)),
        ):
            with self.subTest(label):
                with self.assertRaises(TypeError):
                    call()
        self.assertEqual(cmds.ls(long=True), nodes)
        np.testing.assert_array_equal(_xyz(cube.get_points(world_space=True)), before)
        self.assertEqual(sorted(os.listdir(self.dir)), sorted(os.path.basename(p) for p in paths))

    def test_a_node_given_as_a_point_names_a_world_position_and_the_keyword(self):
        cube, _crv, _plane = self.nodes()
        cmds.spaceLocator(name="loc")
        with self.assertRaises(TypeError) as caught:
            cube.get_closest_point(Node("loc"))
        self.assertEqual(
            str(caught.exception),
            "Mesh.get_closest_point takes a point (an MPoint, or 3 numbers), not the node 'loc': pass a position: "
            "Node('loc').get_rotate_pivot(world_space=True), in world space (then world_space=True here too: object "
            "space is the default), or its translate's value, .t.get(), in its parent's space",
        )


# ---------------------------------------------------------------------------------------------
class TestTheQueriesReadObjectSpace(_Case):
    """A shape under a moved, rotated, unevenly scaled parent: its object space by default
    (Maya's kObject), world space with world_space=True (kWorld)."""

    def setUp(self):
        super().setUp()
        self.shapes = _scene()
        self.cube   = Mesh(self.shapes["cube"])

    def fn(self):
        return OpenMaya.MFnMesh(self.cube.get_mdagpath())

    def test_the_points(self):
        for space, kwargs in ((OBJECT, {}), (WORLD, {"world_space": True})):
            with self.subTest(space=space):
                np.testing.assert_array_equal(_xyz(self.cube.get_points(**kwargs)), _xyz(self.fn().getPoints(space)))
                for label in ("curve", "surface", "cylinder"):
                    node = Node(self.shapes[label])
                    np.testing.assert_array_equal(
                        _xyz(node.get_points(**kwargs)), _xyz(node.get_fn_set().cvPositions(space))
                    )
        # the two spaces are far apart under this parent
        self.assertGreater(np.abs(_xyz(self.cube.get_points()) - _xyz(self.cube.get_points(world_space=True))).max(), 1)

    def test_the_vertex_normals(self):
        for space, kwargs in ((OBJECT, {}), (WORLD, {"world_space": True})):
            with self.subTest(space=space):
                np.testing.assert_array_equal(
                    _xyz(self.cube.get_vertex_normals(**kwargs)), _xyz(self.fn().getVertexNormals(False, space))
                )

    def test_the_closest_point_and_the_uv_at_a_point(self):
        local   = OpenMaya.MPoint(0.7, 0.2, 0.1)  # off the cube's +X face, in its object space
        world   = local * self.cube.get_mdagpath().inclusiveMatrix()
        fn      = self.fn()
        closest = fn.getClosestPoint(local, OBJECT)[0]
        np.testing.assert_allclose(list(self.cube.get_closest_point(local)[0]), list(closest), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            list(self.cube.get_closest_point(world, world_space=True)[0]),
            list(closest * self.cube.get_mdagpath().inclusiveMatrix()), rtol=0, atol=1e-9,
        )
        # on the mesh: its UV
        uv = fn.getUVAtPoint(closest, OBJECT, "map1")
        self.assertEqual(tuple(self.cube.get_uv_at_point(closest, None)), tuple(uv))
        np.testing.assert_allclose(
            self.cube.get_uv_at_point(closest * self.cube.get_mdagpath().inclusiveMatrix(), None, world_space=True)[:2],
            uv[:2], rtol=0, atol=1e-6,
        )

    def test_the_vertices_and_faces_above_a_plane(self):
        origin, up = np.zeros(3), np.array([0.0, 1.0, 0.0])
        # the plane y = 0 of the cube's own space: its top half, its top face
        self.assertEqual(list(self.cube.get_vertices_above_plane(origin, up)), [2, 3, 4, 5])
        self.assertEqual(self.cube.get_faces_above_plane(origin, up), [1])
        # the world's plane y = 0: the cube's vertices where it puts them
        world    = _xyz(self.cube.get_points(world_space=True))
        expected = np.flatnonzero(world @ up > 0).tolist()
        self.assertNotEqual(expected, [2, 3, 4, 5])
        self.assertEqual(list(self.cube.get_vertices_above_plane(origin, up, world_space=True)), expected)

    def test_set_points_sets_object_space(self):
        """guard: set_points was object space by default already."""
        local = _local(self.cube) + [0, 1, 0]
        self.cube.set_points(local)
        np.testing.assert_allclose(_local(self.cube), local, rtol=0, atol=1e-6)
        world = _xyz(self.cube.get_points(world_space=True)) + [0, 1, 0]
        self.cube.set_points(world, world_space=True)
        np.testing.assert_allclose(_xyz(self.cube.get_points(world_space=True)), world, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------
class TestAMeshsDataIsObjectSpace(_Case):
    """Mesh.serialize / save by default: the object-space points and normals, and the UUID of the
    transform they are relative to; Mesh('x').load: the points as its own."""

    def setUp(self):
        super().setUp()
        self.cube = Mesh(_scene()["cube"])

    def test_serialize(self):
        fn = OpenMaya.MFnMesh(self.cube.get_mdagpath())
        data, _uvs = self.cube.serialize()
        np.testing.assert_array_equal(data.points, _xyz(fn.getPoints(OBJECT)))
        np.testing.assert_array_equal(np.asarray(data.normals), _xyz(fn.getNormals(OBJECT)))
        self.assertEqual((data.uuid, data.parent_uuid), (_uuid(self.cube), _uuid(self.cube.get_parent())))
        world = self.cube.serialize(world_space=True, uvs=False)
        np.testing.assert_array_equal(world.points, _xyz(fn.getPoints(WORLD)))
        self.assertEqual((world.uuid, world.parent_uuid), (_uuid(self.cube), None))

    def test_the_parent_is_read_through_the_shape_s_path(self):
        """No cmds.getAttr: a geometry plug's get() reads none (test_class_attribute)."""
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as get_attr:
            data = self.cube.serialize(world_space=False, uvs=False)
        get_attr.assert_not_called()
        self.assertEqual(data.parent_uuid, _uuid(self.cube.get_parent()))

    def test_the_default_save_and_create_put_it_back_where_it_was(self):
        made = Mesh.load(self.cube.save(self.path("cube.npz")), name="copy")
        np.testing.assert_array_equal(_local(made), _local(self.cube))
        np.testing.assert_allclose(_world_matrix(made.get_parent()), _world_matrix(self.cube.get_parent()),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            _xyz(made.get_points(world_space=True)), _xyz(self.cube.get_points(world_space=True)), rtol=0, atol=1e-9
        )

    def test_the_update_takes_the_points_as_its_own(self):
        path = self.cube.save(self.path("cube.npz"))
        copy = Mesh.load(path, name="copy")
        cmds.xform("copy", translation=(0, 10, 0), rotation=(0, 90, 0))
        cmds.xform("copy.vtx[0]", translation=(0, 3, 0), relative=True)
        copy.load(path)
        np.testing.assert_allclose(_local(copy), _local(self.cube), rtol=0, atol=1e-6)
        self.assertEqual(cmds.getAttr("copy.t")[0], (0.0, 10.0, 0.0))
        copy.load(path, world_space=True)
        np.testing.assert_allclose(_xyz(copy.get_points(world_space=True)),
                                   _xyz(self.cube.get_points(world_space=True)), rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------------------------
class TestPlacement(_Case):
    """Where data sits (the user's decision of 2026-11-09, on cgmath's shape model): the world
    matrix of the transform its ``parent_uuid`` names, found in ``hierarchy=`` when given (a
    ClipData at the frame it is on), else in the scene by its UUID, else the identity (a file from
    another scene); an instanced scene transform is refused."""

    def setUp(self):
        super().setUp()
        self.shapes = _scene()
        self.cube   = Mesh(self.shapes["cube"])
        self.curve  = NurbsCurve(self.shapes["curve"])

    def test_a_transform_the_scene_lacks_sits_at_identity(self):
        for label, node in (("cube", self.cube), ("curve", self.curve)):
            with self.subTest(label):
                data  = node.serialize(uvs=False)
                local = _local(node)
                cmds.delete(node.get_parent().get_long_name())
                for world_space in (False, True):
                    made = type(node).create(data, world_space=world_space)
                    np.testing.assert_array_equal(_local(made), local)
                    np.testing.assert_array_equal(_world_matrix(made.get_parent()), np.eye(4))

    def test_a_hierarchy_places_the_data(self):
        from cgmath.hierarchy import HierarchyData, TransformData

        for label, node in (("cube", self.cube), ("curve", self.curve)):
            with self.subTest(label):
                data  = node.serialize(uvs=False)
                under = TransformData("under", uuid=data.parent_uuid, translate=[10.0, 0.0, 0.0])
                rig   = HierarchyData([under])
                made  = type(node).create(data, hierarchy=rig)
                np.testing.assert_allclose(_world_matrix(made.get_parent()), under.world_matrix, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(_local(made), _local(node))
                frozen = type(node).create(data, world_space=True, hierarchy=rig)
                np.testing.assert_allclose(_local(frozen), _local(node) + (10.0, 0.0, 0.0), rtol=0, atol=1e-5)
                np.testing.assert_array_equal(_world_matrix(frozen.get_parent()), np.eye(4))
                # a hierarchy that lacks the transform: identity, cgmath's rule, not the scene's node
                other = HierarchyData([TransformData("other", translate=[5.0, 0.0, 0.0])])
                np.testing.assert_array_equal(_world_matrix(type(node).create(data, hierarchy=other).get_parent()),
                                              np.eye(4))

    def test_the_update_in_place_takes_a_hierarchy(self):
        from cgmath.hierarchy import HierarchyData, TransformData

        path  = self.cube.save(self.path("cube.npz"))
        data  = self.cube.serialize(uvs=False)
        under = TransformData("under", uuid=data.parent_uuid, translate=[0.0, 10.0, 0.0])
        copy  = Mesh.load(path, name="copy")
        copy.load(path, world_space=True, hierarchy=HierarchyData([under]))
        np.testing.assert_allclose(_xyz(copy.get_points(world_space=True)), _local(self.cube) + (0.0, 10.0, 0.0),
                                   rtol=0, atol=1e-5)

    def test_a_clip_places_at_the_frame_it_is_on(self):
        from cgmath.hierarchy import ClipData, HierarchyData, TransformData

        data  = self.curve.serialize()
        under = TransformData("under", uuid=data.parent_uuid)
        clip  = ClipData(HierarchyData([under]), frames=3, start_frame=0)
        clip.frames.translate = np.array([[(0.0, 0.0, 0.0)], [(0.0, 5.0, 0.0)], [(0.0, 9.0, 0.0)]])
        clip.frame            = 2
        made = NurbsCurve.create(data, hierarchy=clip)
        np.testing.assert_allclose(_world_matrix(made.get_parent())[3, :3], (0.0, 9.0, 0.0), rtol=0, atol=1e-12)

    def test_an_instanced_transform_is_refused(self):
        grp = cmds.group(self.cube.get_parent().get_long_name(), name="grp")
        cmds.instance(grp, name="grp_copy")
        data    = self.cube.serialize(uvs=False)
        refusal = r"^Mesh\.create\(\): the data sits under '.*', which is instanced \(2 paths\), so its points have no one place in the world; give hierarchy=, or serialize the shape in world space$"
        before  = cmds.ls()
        for world_space in (False, True):
            with self.subTest(world_space=world_space):
                with self.assertRaisesRegex(ValueError, refusal):
                    Mesh.create(data, world_space=world_space)
                self.assertEqual(cmds.ls(), before)
        target = cmds.createNode("transform", name="target")
        under  = Mesh.create(data, transform=target)   # no placement to read: the points as its own
        np.testing.assert_array_equal(_local(under), _local(self.cube))
        world = self.cube.serialize(world_space=True, uvs=False)
        np.testing.assert_array_equal(_world_matrix(Mesh.create(world).get_parent()), np.eye(4))


# ---------------------------------------------------------------------------------------------
class TestTheNurbsDataHoldsWhereItIs(_Case):
    """A curve's and a surface's data hold their uuid and the parent_uuid of the transform they sit
    under (and a surface its knots), as a MeshData does; create and load read them back."""

    def setUp(self):
        super().setUp()
        self.shapes = _scene()
        self.nurbs  = {label: Node(self.shapes[label]) for label in ("curve", "surface", "cylinder")}

    def test_serialize_writes_the_uuids(self):
        for label, node in self.nurbs.items():
            with self.subTest(label):
                data = node.serialize()
                self.assertEqual((data.uuid, data.parent_uuid), (_uuid(node), _uuid(node.get_parent())))
                self.assertFalse(hasattr(data, "name") or hasattr(data, "matrix"))
                np.testing.assert_array_equal(np.asarray(data.cv).reshape(-1, 3), _local(node))
                world = node.serialize(world_space=True)
                self.assertEqual((world.uuid, world.parent_uuid), (_uuid(node), None))
                np.testing.assert_allclose(np.asarray(world.cv).reshape(-1, 3), _xyz(node.get_points(world_space=True)),
                                           rtol=0, atol=1e-12)

    def test_uneven_knots_are_kept_and_nothing_is_said(self):
        parent = cmds.createNode("transform", name="grp")
        for shape in (_uneven(parent), _cylinder(parent)):
            with self.subTest(shape):
                with self.assertNoLogs("rig.nodetypes.nurbs", level="WARNING"):
                    data = NurbsSurface(shape).serialize()
                self.assertIsNotNone(data.knots_v)

    def test_a_surface_s_knots_as_maya_has_them(self):
        for label, knots_u, knots_v in (
            ("surface", (0, 0, 0, 1, 5, 5, 5), (0, 1, 3)),
            ("cylinder", None, CYLINDER_V_KNOTS),  # periodic U: Maya's default knots, none kept
        ):
            with self.subTest(label):
                data = self.nurbs[label].serialize()
                for given, kept in ((knots_u, data.knots_u), (knots_v, data.knots_v)):
                    if given is None:
                        self.assertIsNone(kept)
                    else:
                        np.testing.assert_array_equal(kept, np.asarray(given, dtype=float))

    def test_the_default_round_trip_comes_back_exactly(self):
        for label, node in self.nurbs.items():
            with self.subTest(label):
                source = node.get_fn_set()
                made   = type(node).load(node.save(self.path(f"{label}.npz")))
                got    = made.get_fn_set()
                # Maya's default name (the data holds none), at the source's world matrix (its
                # transform places it), its CVs as they were
                self.assertEqual(made.get_parent().get_long_name(),
                                 {"curve": "|curve1", "surface": "|surface1", "cylinder": "|surface2"}[label])
                np.testing.assert_allclose(_world_matrix(made.get_parent()), _world_matrix(node.get_parent()),
                                           rtol=0, atol=1e-12)
                np.testing.assert_array_equal(_local(made), _local(node))
                if label == "curve":
                    np.testing.assert_array_equal(list(got.knots()), list(source.knots()))
                    np.testing.assert_allclose(_on_curve(got), _on_curve(source), rtol=0, atol=1e-9)
                else:
                    np.testing.assert_array_equal(list(got.knotsInU()), list(source.knotsInU()))
                    np.testing.assert_array_equal(list(got.knotsInV()), list(source.knotsInV()))
                    np.testing.assert_allclose(_on_surface(got), _on_surface(source), rtol=0, atol=1e-9)

    def test_the_update_in_world_space_lands_where_the_data_sits(self):
        for label, node in self.nurbs.items():
            with self.subTest(label):
                path = node.save(self.path(f"{label}.npz"))
                copy = type(node).load(path, name=f"{label}_copy")
                cmds.xform(str(copy.get_parent()), translation=(0, 10, 0), rotation=(0, 90, 0))
                every = f"{copy}.cv[*]" if label == "curve" else f"{copy}.cv[*][*]"
                cmds.xform(every, translation=(0, 3, 0), relative=True)
                copy.load(path, world_space=True)
                np.testing.assert_allclose(_xyz(copy.get_points(world_space=True)),
                                           _xyz(node.get_points(world_space=True)), rtol=0, atol=1e-9)
                self.assertEqual(cmds.getAttr(f"{copy.get_parent()}.t")[0], (0.0, 10.0, 0.0))
                copy.load(path)
                np.testing.assert_allclose(_local(copy), _local(node), rtol=0, atol=1e-12)

    def test_a_surface_update_refuses_other_knots(self):
        source = self.nurbs["surface"]
        path   = source.save(self.path("srf.npz"))
        target = NurbsSurface(_uneven(cmds.createNode("transform", name="even"), "even", knots_v=(0, 1, 2)))
        before = _local(target)
        refusal = (
            r"holds another CV layout than 'evenShape' \(5 x 3 CVs of degrees 3 x 1 against its 5 x 3 of "
            r"degrees 3 x 1, or other forms or knots\)"
        )
        with self.assertRaisesRegex(ValueError, refusal):
            target.load(path)
        np.testing.assert_array_equal(_local(target), before)
        # re-pinned (user decision 2026-10-30, test_class_shape_decisions): data that holds no knots
        # (even ones, by Maya's default) on a surface uneven at degree 3 is refused too, the same words
        data         = source.serialize()
        data.knots_u = data.knots_v = None
        with self.assertRaisesRegex(ValueError, refusal):
            target.load(data)
        np.testing.assert_array_equal(_local(target), before)

    def test_evenly_spaced_knots_keep_their_range(self):
        """The counterpart of test_class_nurbs_create's 0..spans: knots that take another range
        than Maya's default ones are kept, the range with them."""
        plane = cmds.nurbsPlane(name="plane", constructionHistory=False, u=3, v=2)[0]
        made  = NurbsSurface.create(NurbsSurface(plane).serialize())
        self.assertEqual(tuple(made.get_fn_set().knotDomainInU), (0.0, 1.0))

    def test_the_cylinder_comes_back_exactly_and_says_nothing(self):
        cylinder = self.nurbs["cylinder"]
        with self.assertNoLogs("rig.nodetypes.nurbs", level="WARNING"):
            data = cylinder.serialize()
        made = NurbsSurface.create(data).get_fn_set()
        self.assertEqual(list(made.knotsInV()), list(CYLINDER_V_KNOTS))
        np.testing.assert_allclose(_on_surface(made), _on_surface(cylinder.get_fn_set()), rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
class TestAPlugReadFollowsThePlug(_Case):
    """Attribute.get() on a geometry plug: a worldMesh / worldSpace element in world space, every
    other plug in object space (with the transform's matrix, as serialize writes it)."""

    def setUp(self):
        super().setUp()
        self.shapes = _scene()

    def test_a_shape_s_own_plugs(self):
        cube, crv, srf = (Node(self.shapes[label]) for label in ("cube", "curve", "surface"))
        for label, plug, node, world in (
            ("worldMesh[0]", cube.worldMesh[0], cube, True),
            ("outMesh", cube.outMesh, cube, False),
            ("curve worldSpace[0]", crv.worldSpace[0], crv, True),
            ("curve local", crv.local, crv, False),
            ("surface worldSpace[0]", srf.worldSpace[0], srf, True),
            ("surface local", srf.local, srf, False),
        ):
            with self.subTest(label):
                data   = plug.get()
                data   = data[0] if isinstance(data, tuple) else data
                points = np.asarray(data.cv if hasattr(data, "cv") else data.points).reshape(-1, 3)
                np.testing.assert_allclose(points, _xyz(node.get_points(world_space=world)), rtol=0, atol=1e-12)
                self.assertEqual(data.parent_uuid, None if world else _uuid(node.get_parent()))

    def test_a_deformer_s_output_through_its_shape(self):
        cube = Mesh(self.shapes["cube"])
        skin = SkinCluster.create(cube.get_parent(), Joint.create(name="j1"))
        data, _uvs = skin.outputGeometry[0].get()
        np.testing.assert_allclose(data.points, _local(cube), rtol=0, atol=1e-6)
        self.assertEqual(data.parent_uuid, _uuid(cube.get_parent()))

    def test_a_choice_follows_the_plug_it_routes(self):
        cube, crv = Mesh(self.shapes["cube"]), NurbsCurve(self.shapes["curve"])
        choice = cmds.createNode("choice", name="pick")
        cmds.connectAttr(f"{crv}.worldSpace[0]", f"{choice}.input[0]")
        cmds.connectAttr(f"{cube}.outMesh", f"{choice}.input[1]")
        cmds.setAttr(f"{choice}.selector", 0)
        np.testing.assert_allclose(Node(choice).output.get().points, _xyz(crv.get_points(world_space=True)),
                                   rtol=0, atol=1e-12)
        cmds.setAttr(f"{choice}.selector", 1)
        np.testing.assert_allclose(Node(choice).output.get()[0].points, _local(cube), rtol=0, atol=1e-12)

    def test_a_consumer_that_is_no_shape_serializes_as_it_does(self):
        """guard: a skinCluster's serialize takes no space."""
        cube  = cmds.polyCube(name="body", constructionHistory=False)[0]
        skin  = SkinCluster.create(Node(cube), Joint.create(name="j1"))
        extra = cmds.createNode("polyCube", name="extra")
        cmds.connectAttr(f"{extra}.output", f"{skin}.input[1].inputGeometry")
        self.assertEqual(Node(extra).output.get(), skin.serialize())


# ---------------------------------------------------------------------------------------------
class TestTheCallersThatSampleAcrossMeshes(_Case):
    """guards: rig's own callers that measure across meshes keep world space, by keyword."""

    def test_transfer_from_meshes_samples_in_world_space(self):
        joint1 = Joint.create(name="j1")
        joint2 = Joint.create(name="j2", parent=joint1)
        joint2.t.set(0, 1, 0)
        # the source's vertices where they were in the world, its own space elsewhere
        source = Node(cmds.polyCube(name="source", height=2, constructionHistory=False)[0])
        cmds.xform(str(source), translation=(0, 0, 5))
        cmds.move(0, 0, -5, f"{source}.vtx[*]", relative=True, objectSpace=True)
        skin    = SkinCluster.create(source, [joint1, joint2])
        weights = np.random.default_rng(7).random((8, 2))
        skin.set_weights(weights / weights.sum(axis=1, keepdims=True))
        # the target's vertices where the source's are in the world, its own space elsewhere
        target = Node(cmds.polyCube(name="target", height=2, constructionHistory=False)[0])
        cmds.xform(str(target), translation=(10, 0, 0))
        cmds.move(-10, 0, 0, f"{target}.vtx[*]", relative=True, objectSpace=True)
        SkinCluster.transfer_from_meshes(str(source), [str(target)])
        copied = target.get_shape().get_deformers(node_type="skinCluster")[0]
        np.testing.assert_allclose(copied.get_weights(), skin.get_weights(), rtol=0, atol=1e-9)

    def test_a_follicle_lands_at_its_reference_in_the_world(self):
        plane = cmds.polyPlane(name="pl", subdivisionsX=4, subdivisionsY=4, constructionHistory=False)[0]
        cmds.xform(plane, translation=(5, 0, 0))
        locator = cmds.spaceLocator(name="loc")[0]
        cmds.xform(locator, translation=(5.2, 0, 0.1))
        follicle = Follicle.create_on_mesh(plane, locator, name="fol")
        landed   = cmds.xform(str(follicle.get_parent()), query=True, translation=True, worldSpace=True)
        np.testing.assert_allclose(landed, (5.2, 0, 0.1), rtol=0, atol=1e-6)
