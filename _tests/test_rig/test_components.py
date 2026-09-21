"""Tests for ``rig._internal.members`` -- Components, the Node fallbacks and
the left-hand-side normaliser."""

import time

import numpy as np
from maya import cmds
from rig import Node, Plug, PlugList
from rig._internal.generators import sequences
from rig._internal.members import (
    _MemberSpec,
    _Selection,
    Components,
    normalise,
)
from rig._internal.plug import ComponentPlug
from rig._internal.types import _is_components, _is_member_spec, _is_sequence
from rig._tests._base import MayaTestCase


def _cube(name="pCube1"):
    """A polyCube transform and the full path of its shape."""
    xform = cmds.polyCube(name=name, ch=False)[0]
    shape = cmds.listRelatives(xform, shapes=True, fullPath=True)[0]
    return xform, shape


def _two_shape_cube():
    """``pCube1`` holding BOTH ``pCube1Shape`` and ``pCube2Shape``."""
    xform, shape = _cube("pCube1")
    other, shape2 = _cube("pCube2")
    cmds.parent(shape2, xform, shape=True, relative=True)
    cmds.delete(other)
    shapes = cmds.listRelatives(xform, shapes=True, fullPath=True)
    return xform, shapes


def _sphere_surface():
    xform = cmds.sphere(name="nurbsSphere1")[0]
    return xform, cmds.listRelatives(xform, shapes=True, fullPath=True)[0]


def _circle():
    xform = cmds.circle(name="circle1")[0]
    return xform, cmds.listRelatives(xform, shapes=True, fullPath=True)[0]


def _lattice(target):
    """A 2 x 3 x 4 lattice on ``target``: the lattice transform and its shape."""
    lattice = cmds.lattice(target, divisions=(2, 3, 4))[1]
    return lattice, cmds.listRelatives(lattice, shapes=True, fullPath=True)[0]


class _Spec(_MemberSpec):
    """A kind that accepts every selection and implements no verb."""

    KIND    = "test"
    ACCEPTS = frozenset({"whole", "vtx", "e", "f", "cv", "pt", "uv"})


class _Strict(_MemberSpec):
    """A kind that accepts nothing: the ACCEPTS gate fires before any hook."""

    KIND = "strict"


# --------------------------------------------------------------------- #
#  Components construction
# --------------------------------------------------------------------- #


class TestComponentsConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_f_on_mesh_shape_is_whole_kind_handle(self):
        _, shape = _cube()
        faces    = Node(shape).f
        self.assertIsInstance(faces, Components)
        self.assertTrue(faces.is_all)
        self.assertEqual(faces.kind,     "f")
        self.assertEqual(faces.geometry, "mesh")
        self.assertEqual(faces.count,    6)
        self.assertEqual(faces.names,    [f"{shape}.f[*]"])
        np.testing.assert_array_equal(faces.indices, np.arange(6))
        self.assertEqual(str(faces.shape), "pCube1Shape")

    def test_e_on_transform_resolves_to_its_single_mesh_shape(self):
        xform, shape = _cube()
        edges        = Node(xform).e
        self.assertIsInstance(edges, Components)
        self.assertEqual(edges.kind,  "e")
        self.assertEqual(edges.count, 12)
        self.assertEqual(edges.names, [f"{shape}.e[*]"])

    def test_transform_with_two_mesh_shapes_raises_naming_them(self):
        xform, shapes = _two_shape_cube()
        with self.assertRaises(AttributeError) as ctx:
            _ = Node(xform).f
        for shape in shapes:
            self.assertIn(shape, str(ctx.exception))

    def test_real_attribute_wins_over_the_fallback(self):
        # ``f`` is the short name of ``form`` on a curve; ``face`` is a real
        # mesh plug sitting right next to ``f``.
        _, curve = _circle()
        _, shape = _cube()
        form = Node(curve).f
        self.assertIsInstance(form, Plug)
        self.assertTrue(str(form).endswith(".form"))
        face = Node(shape).face
        self.assertIsInstance(face, Plug)
        self.assertNotIsInstance(face, Components)

    def test_f_without_a_mesh_still_raises(self):
        joint    = cmds.createNode("joint", name="joint1")
        xform, _ = _circle()
        with self.assertRaises(AttributeError):
            _ = Node(joint).f
        with self.assertRaises(AttributeError):
            _ = Node(xform).e
        with self.assertRaises(AttributeError):
            _ = Node(xform).completely_made_up_attribute_xyz

    def test_explicit_carrier_is_a_sorted_set_and_builds_no_plugs(self):
        xform, shape = _cube()
        verts        = Components(Node(xform), "vtx", [4, 0, 2, 2])
        self.assertEqual(verts.kind, "vtx")
        self.assertFalse(verts.is_all)
        self.assertEqual(verts.count, 3)
        np.testing.assert_array_equal(verts.indices, [0, 2, 4])
        self.assertEqual(
            verts.names, [f"{shape}.vtx[0]", f"{shape}.vtx[2]", f"{shape}.vtx[4]"]
        )
        self.assertFalse(verts.indices.flags.writeable)

    def test_explicit_carrier_from_ndarray_and_negative_wrap(self):
        _, shape = _cube()
        verts    = Components(shape, "vtx", np.array([-1, 1]))
        np.testing.assert_array_equal(verts.indices, [1, 7])

    def test_explicit_carrier_whole_kind(self):
        _, shape = _cube()
        uvs      = Components(shape, "uv")
        self.assertTrue(uvs.is_all)
        self.assertEqual(uvs.count, 14)
        self.assertEqual(uvs.names, [f"{shape}.map[*]"])

    def test_explicit_carrier_rejections(self):
        xform, shape = _cube()
        joint        = cmds.createNode("joint", name="joint1")
        with self.assertRaises(TypeError):
            Components(shape)
        with self.assertRaises(TypeError):
            Components(shape, "bogus")
        with self.assertRaises(TypeError):
            Components(shape, "cv", [0])
        with self.assertRaises(TypeError):
            Components(joint, "vtx", [0])
        with self.assertRaises(TypeError):
            Components(shape, "vtx", [True])
        with self.assertRaises(TypeError):
            Components(shape, "vtx", [0.5])
        with self.assertRaises(TypeError):
            Components(shape, "vtx", [[0, 1]])
        with self.assertRaises(IndexError):
            Components(shape, "vtx", [0, 8])
        with self.assertRaises(IndexError):
            Components(shape, "f", [-7])

    def test_synonyms_canonicalise(self):
        _, shape = _cube()
        uvs      = Components(shape, "map", [1, 0])
        self.assertEqual(uvs.kind, "uv")
        self.assertEqual(uvs.names, [f"{shape}.map[0:1]"])
        self.assertEqual(Components(shape, "pnts", [3]).kind, "vtx")

    def test_string_constructor(self):
        xform, shape = _cube()
        faces        = Components("pCube1.f[0:3]")
        self.assertEqual(faces.kind, "f")
        self.assertEqual(str(faces.shape), "pCube1Shape")
        np.testing.assert_array_equal(faces.indices, [0, 1, 2, 3])
        self.assertTrue(Components("pCube1.f[*]").is_all)
        self.assertEqual(Components(f"{shape}.vtx[2]").names, [f"{shape}.vtx[2]"])
        self.assertEqual(Components("pCube1.map[1]").kind,    "uv")
        self.assertEqual(Components("pCube1.e[3:4]").names,   [f"{shape}.e[3:4]"])

    def test_string_constructor_rejections(self):
        _cube()
        with self.assertRaises(TypeError):
            Components("pCube1.tx")
        with self.assertRaises(TypeError):
            Components("pCube1.f[0:3]", "f")
        with self.assertRaises(TypeError):
            Components("pCube1.f[0:3]", indices=[0])
        with self.assertRaises(TypeError):
            Components("pCube1.cv[0]")
        with self.assertRaises(TypeError):
            Components("pCube1.f[0][1]")
        with self.assertRaises(IndexError):
            Components("pCube1.f[0:9]")

    def test_string_constructor_on_a_surface(self):
        xform, shape = _sphere_surface()
        cv = Components("nurbsSphere1.cv[1][2]")
        self.assertEqual(cv.kind, "cv")
        self.assertEqual(cv.geometry, "nurbsSurface")
        np.testing.assert_array_equal(cv.indices, [[1, 2]])
        self.assertEqual(cv.names, [f"{shape}.cv[1][2]"])
        row = Components("nurbsSphere1.cv[1][*]")
        self.assertEqual(row.count, 8)
        self.assertEqual(row.names, [f"{shape}.cv[1][0:7]"])
        rect = Components("nurbsSphere1.cv[1:2][2:3]")
        self.assertEqual(rect.names, [f"{shape}.cv[1][2:3]", f"{shape}.cv[2][2:3]"])

    def test_surface_whole_kind_resolves_distinct_cvs(self):
        # A default sphere is periodic in V: 77 controlPoints, 56 real CVs.
        xform, shape = _sphere_surface()
        cvs          = Components(xform, "cv")
        self.assertTrue(cvs.is_all)
        self.assertEqual(cvs.count,         56)
        self.assertEqual(cvs.indices.shape, (56, 2))
        self.assertEqual(cvs.names,         [f"{shape}.cv[*]"])

    def test_lattice_points_are_three_dimensional(self):
        xform, _ = _cube()
        lattice, shape = _lattice(xform)
        points = Components(lattice, "pt")
        self.assertTrue(points.is_all)
        self.assertEqual(points.count, 24)
        self.assertEqual(points.indices.shape, (24, 3))
        sparse = Components(shape, "pt", [[1, 2, 3], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(
            sparse.names, [f"{shape}.pt[0][0][0:1]", f"{shape}.pt[1][2][3]"]
        )
        with self.assertRaises(IndexError):
            Components(shape, "pt", [[2, 0, 0]])


# --------------------------------------------------------------------- #
#  Components indexing
# --------------------------------------------------------------------- #


class TestComponentsIndexing(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_int_keys(self):
        _, shape = _cube()
        faces    = Node(shape).f
        self.assertEqual(faces[2].names,           [f"{shape}.f[2]"])
        self.assertEqual(faces[-1].names,          [f"{shape}.f[5]"])
        self.assertEqual(faces[np.int64(3)].names, [f"{shape}.f[3]"])
        with self.assertRaises(IndexError):
            faces[6]
        with self.assertRaises(IndexError):
            faces[-7]

    def test_slices_clamp_and_the_full_slice_is_the_handle(self):
        _, shape = _cube()
        faces    = Node(shape).f
        self.assertEqual(faces[:3].names,   [f"{shape}.f[0:2]"])
        self.assertEqual(faces[:100].count, 6)
        self.assertEqual(faces[2:4].names,  [f"{shape}.f[2:3]"])
        self.assertEqual(
            faces[::2].names, [f"{shape}.f[0]", f"{shape}.f[2]", f"{shape}.f[4]"]
        )
        self.assertTrue(faces[:].is_all)
        self.assertTrue(faces[0:6:1].is_all)
        self.assertFalse(faces[0:6:2].is_all)
        empty = faces[6:]
        self.assertEqual(empty.count, 0)
        self.assertFalse(empty)
        self.assertEqual(empty.names, [])
        self.assertTrue(repr(empty).endswith('.f[]")'))

    def test_sequence_keys(self):
        _, shape = _cube()
        faces    = Node(shape).f
        self.assertEqual(
            faces[[0, 1, 2, 5]].names, [f"{shape}.f[0:2]", f"{shape}.f[5]"]
        )
        self.assertEqual(faces[(5, 0)].names, [f"{shape}.f[0]", f"{shape}.f[5]"])
        self.assertEqual(faces[np.array([4, -1])].names, [f"{shape}.f[4:5]"])
        self.assertEqual(faces[[]].count, 0)
        self.assertEqual(faces[np.array([], dtype=int)].count, 0)
        with self.assertRaises(IndexError):
            faces[[0, 6]]
        with self.assertRaises(IndexError):
            faces[[-7]]

    def test_bool_and_non_integer_keys_are_type_errors(self):
        _, shape = _cube()
        faces    = Node(shape).f
        with self.assertRaises(TypeError):
            faces[True]
        with self.assertRaises(TypeError):
            faces[[True, False]]
        with self.assertRaises(TypeError):
            faces[np.array([True] * 6)]
        with self.assertRaises(TypeError):
            faces[1.5]
        with self.assertRaises(TypeError):
            faces[[0.5]]
        with self.assertRaises(TypeError):
            faces["0"]
        with self.assertRaises(TypeError):
            faces[[[0, 1]]]

    def test_indexing_is_positional_on_the_sorted_selection(self):
        _, shape = _cube()
        picked   = Node(shape).f[[5, 2, 3]]
        np.testing.assert_array_equal(picked.indices, [2, 3, 5])
        self.assertEqual(picked[0].names,  [f"{shape}.f[2]"])
        self.assertEqual(picked[-1].names, [f"{shape}.f[5]"])
        self.assertEqual(picked[1:].names, [f"{shape}.f[3]", f"{shape}.f[5]"])
        with self.assertRaises(IndexError):
            picked[3]

    def test_surface_handle_indexes_the_grid_row_major(self):
        xform, shape = _sphere_surface()
        cvs          = Components(xform, "cv")
        np.testing.assert_array_equal(cvs[0].indices, [[0, 0]])
        np.testing.assert_array_equal(cvs[9].indices, [[1, 1]])
        self.assertEqual(cvs[:2].names, [f"{shape}.cv[0][0:1]"])
        self.assertEqual(cvs[[7, 8]].names, [f"{shape}.cv[0][7]", f"{shape}.cv[1][0]"])
        self.assertTrue(cvs[:].is_all)


# --------------------------------------------------------------------- #
#  Components operators, equality, display, PlugList behaviour
# --------------------------------------------------------------------- #


class TestComponentsOperators(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_rshift_none_reads_the_native_ids(self):
        xform, shape = _cube()
        faces        = Node(shape).f
        np.testing.assert_array_equal(faces >> None, np.arange(6))
        np.testing.assert_array_equal(faces[[0, 4]] >> None, [0, 4])
        self.assertIsInstance(faces[:3] >> None, np.ndarray)
        surface, _ = _sphere_surface()
        self.assertEqual((Components(surface, "cv")[:3] >> None).shape, (3, 2))

    def test_operator_rejections(self):
        _, shape = _cube()
        faces    = Node(shape).f[:3]
        with self.assertRaises(TypeError):
            faces >> "x"
        with self.assertRaises(TypeError):
            faces >> 5
        with self.assertRaises(NotImplementedError):
            faces >> _Spec("x")
        with self.assertRaises(NotImplementedError):
            faces << _Spec("x")
        with self.assertRaises(TypeError):
            faces << None
        with self.assertRaises(TypeError):
            faces << 5
        with self.assertRaises(TypeError):
            faces << "x"

    def test_no_length_no_iteration_no_arithmetic(self):
        _, shape = _cube()
        faces    = Node(shape).f
        with self.assertRaises(TypeError):
            len(faces)
        with self.assertRaises(TypeError):
            iter(faces)
        with self.assertRaises(TypeError):
            faces + 1
        with self.assertRaises(TypeError):
            -faces
        with self.assertRaises(TypeError):
            ~faces
        self.assertFalse(_is_sequence(faces))

    def test_equality_and_hashing(self):
        xform, shape = _cube()
        node         = Node(shape)
        self.assertEqual(node.f[:3], node.f[[2, 1, 0]])
        self.assertEqual(node.f, node.f[:])
        self.assertEqual(node.f, Components(xform, "f", range(6)))
        self.assertEqual(Components(xform, "f", range(6)), node.f)
        self.assertNotEqual(node.f, node.e)
        self.assertNotEqual(node.f[:3], node.f[:4])
        self.assertFalse(node.f == "pCube1Shape.f[*]")
        self.assertEqual(hash(node.f), hash(node.f[:2]))
        self.assertEqual(len({node.f, node.f[:], node.e}), 2)

    def test_bool_and_repr(self):
        _, shape = _cube()
        node     = Node(shape)
        self.assertTrue(node.f)
        self.assertFalse(node.f[6:])
        self.assertEqual(repr(node.f), f'Components("{shape}.f[*]")')
        self.assertEqual(
            repr(node.f[[0, 1, 2, 5]]), f'Components("{shape}.f[0:2] f[5]")'
        )
        self.assertEqual(
            repr(Components(shape, "uv", [3])), f'Components("{shape}.map[3]")'
        )
        sparse = Components(shape, "vtx", [0, 2, 4, 6])
        self.assertEqual(
            repr(sparse), f'Components("{shape}.vtx[0] vtx[2] vtx[4] vtx[6]")'
        )

    def test_repr_elides_long_selections(self):
        big    = cmds.polyPlane(name="plane1", sx=40, sy=40, ch=False)[0]
        sparse = Components(big, "vtx", np.arange(0, 1600, 2))
        text   = repr(sparse)
        self.assertIn("... (+", text)
        self.assertLess(len(text), 400)

    def test_survives_pluglist_and_broadcasts_as_a_scalar(self):
        xform, shape = _cube()
        node   = Node(xform)
        faces  = Node(shape).f[:3]
        listed = PlugList([faces])
        self.assertIs(listed[0], faces)
        mixed = PlugList([node, faces, Node(shape).vtx[0]])
        self.assertIs(mixed[1], faces)
        self.assertIn(faces, mixed)
        broadcast = PlugList([node, faces]).tx
        self.assertIsInstance(broadcast, PlugList)
        self.assertEqual(str(broadcast[0]), "pCube1.translateX")
        self.assertIs(broadcast[1], faces)
        self.assertEqual([str(x) for x in listed.translateX], [repr(faces)])
        with self.assertRaises(TypeError):
            PlugList(faces)
        pairs = list(sequences([1, 2], faces))
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(rhs is faces for _, rhs in pairs))

    def test_names_are_selectable(self):
        xform, shape = _cube()
        node         = Node(shape)
        cmds.select(node.f[[0, 1, 2, 5]].names)
        self.assertEqual(cmds.ls(sl=True), ["pCube1.f[0:2]", "pCube1.f[5]"])
        cmds.select(node.e.names)
        self.assertEqual(cmds.ls(sl=True), ["pCube1.e[0:11]"])
        cmds.select(Components(shape, "uv", [1, 2]).names)
        self.assertEqual(cmds.ls(sl=True), ["pCube1.map[1:2]"])
        surface, _ = _sphere_surface()
        cmds.select(Components(surface, "cv", [[1, 2], [1, 3], [2, 0]]).names)
        self.assertEqual(
            cmds.ls(sl=True), ["nurbsSphere1.cv[1][2:3]", "nurbsSphere1.cv[2][0]"]
        )
        lattice, lattice_shape = _lattice(xform)
        cmds.select(Components(lattice_shape, "pt", [[0, 0, 0], [0, 0, 1]]).names)
        self.assertEqual(cmds.ls(sl=True), ["ffd1Lattice.pt[0][0][0:1]"])

    def test_type_predicates(self):
        _, shape = _cube()
        faces    = Node(shape).f
        self.assertTrue(_is_components(faces))
        self.assertFalse(_is_components(Node(shape)))
        self.assertFalse(_is_components("pCube1.f[0]"))
        self.assertTrue(_is_member_spec(_Spec("x")))
        self.assertFalse(_is_member_spec(faces))
        self.assertFalse(_is_member_spec(None))


# --------------------------------------------------------------------- #
#  Node point-alias fallback through a transform
# --------------------------------------------------------------------- #


class TestNodeAliasFallback(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_vtx_on_a_transform_resolves_to_the_shape(self):
        xform, shape = _cube()
        handle       = Node(xform).vtx
        self.assertIsInstance(handle, Plug)
        self.assertEqual(str(handle), "pCube1Shape.controlPoints")
        self.assertEqual(str(Node(xform).vtx[3]), "pCube1Shape.controlPoints[3]")
        self.assertEqual(len(Node(xform).vtx[:4]), 4)
        self.assertEqual(str(Node(xform).map[0]), "pCube1Shape.uvpt[0]")

    def test_cv_on_a_surface_transform_is_a_component_plug(self):
        xform, _ = _sphere_surface()
        handle   = Node(xform).cv
        self.assertIsInstance(handle, ComponentPlug)
        self.assertEqual(str(Node(xform).cv[1, 2]), "nurbsSphere1Shape.cv[1][2]")
        curve, _ = _circle()
        self.assertEqual(str(Node(curve).cv[2]), "circle1Shape.controlPoints[2]")

    def test_alias_the_shape_lacks_still_raises(self):
        xform, _ = _cube()
        with self.assertRaises(AttributeError):
            _ = Node(xform).cv
        joint = cmds.createNode("joint", name="joint1")
        with self.assertRaises(AttributeError):
            _ = Node(joint).vtx

    def test_two_geometry_shapes_raise_naming_them(self):
        xform, shapes = _two_shape_cube()
        with self.assertRaises(AttributeError) as ctx:
            _ = Node(xform).vtx
        for shape in shapes:
            self.assertIn(shape, str(ctx.exception))


# --------------------------------------------------------------------- #
#  normalise()
# --------------------------------------------------------------------- #


class TestNormalise(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _one(self, lhs, want_shapes=True):
        result = normalise(lhs, want_shapes=want_shapes)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _Selection)
        return result[0]

    def test_transform_with_one_shape(self):
        xform, shape = _cube()
        node = Node(xform)
        sel  = self._one(node, want_shapes=True)
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("whole", shape, "mesh"))
        self.assertEqual(sel.tokens, ())
        self.assertIsNone(sel.indices)
        self.assertFalse(sel.is_all)
        self.assertEqual(sel.names, [shape])
        self.assertEqual(sel.source, (node,))
        sel = self._one(node, want_shapes=False)
        self.assertEqual(
            (sel.kind, sel.path, sel.node_type), ("whole", "|pCube1", "transform")
        )

    def test_transform_with_two_shapes(self):
        xform, shapes = _two_shape_cube()
        result        = normalise(Node(xform), want_shapes=True)
        self.assertEqual([s.path for s in result], shapes)
        self.assertEqual({s.kind for s in result}, {"whole"})
        sel = self._one(Node(xform), want_shapes=False)
        self.assertEqual(sel.path, "|pCube1")

    def test_mesh_shape_and_other_geometry_shapes(self):
        _, shape = _cube()
        sel      = self._one(Node(shape))
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("whole", shape, "mesh"))
        curve_xform, curve = _circle()
        self.assertEqual(self._one(Node(curve_xform)).path, curve)
        self.assertEqual(self._one(Node(curve)).node_type, "nurbsCurve")
        surface_xform, surface = _sphere_surface()
        self.assertEqual(self._one(Node(surface_xform)).path, surface)

    def test_joint_and_dg_node_are_whole(self):
        joint = cmds.createNode("joint", name="joint1")
        for want_shapes in (True, False):
            sel = self._one(Node(joint), want_shapes=want_shapes)
            self.assertEqual(
                (sel.kind, sel.path, sel.node_type), ("whole", "|joint1", "joint")
            )
        cmds.createNode("multiplyDivide", name="md1")
        sel = self._one(Node("md1"))
        self.assertEqual(
            (sel.kind, sel.path, sel.node_type), ("whole", "md1", "multiplyDivide")
        )

    def test_bare_vtx_handle_is_all_points(self):
        _, shape = _cube()
        handle = Node(shape).vtx
        sel    = self._one(handle)
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("vtx", shape, "mesh"))
        self.assertTrue(sel.is_all)
        self.assertEqual(sel.tokens, ("vtx[*]",))
        self.assertIsNone(sel.indices)
        self.assertFalse(sel.flat)
        self.assertEqual(sel.names, [f"{shape}.vtx[*]"])
        self.assertEqual(sel.source, (handle,))

    def test_vertex_element_plugs(self):
        _, shape = _cube()
        node = Node(shape)
        sel  = self._one(node.vtx[3])
        self.assertEqual(sel.tokens, ("vtx[3]",))
        np.testing.assert_array_equal(sel.indices, [3])
        sel = self._one(node.vtx[:8])
        self.assertEqual(sel.tokens, ("vtx[0:7]",))
        np.testing.assert_array_equal(sel.indices, np.arange(8))
        self.assertFalse(sel.is_all)
        sel = self._one(node.vtx[::3])
        self.assertEqual(sel.tokens, ("vtx[0]", "vtx[3]", "vtx[6]"))
        sel = self._one(node.vtx[[6, 1, 1]])
        self.assertEqual(sel.tokens, ("vtx[1]", "vtx[6]"))
        self.assertEqual(len(sel.source), 3)

    def test_surface_component_plugs_render_two_dimensional_tokens(self):
        _, shape = _sphere_surface()
        node = Node(shape)
        sel  = self._one(node.cv[1, 2])
        self.assertEqual((sel.kind, sel.node_type), ("cv", "nurbsSurface"))
        self.assertEqual(sel.tokens, ("cv[1][2]",))
        np.testing.assert_array_equal(sel.indices, [[1, 2]])
        self.assertFalse(sel.flat)
        sel = self._one(node.cv[1:3, 2:4])
        self.assertEqual(sel.tokens, ("cv[1][2:3]", "cv[2][2:3]"))
        self.assertEqual(sel.indices.shape, (4, 2))
        sel = self._one(node.cv)
        self.assertTrue(sel.is_all)
        self.assertEqual(sel.tokens, ("cv[*]",))

    def test_plain_surface_control_points_pass_through_flat(self):
        _, shape = _sphere_surface()
        node = Node(shape)
        sel  = self._one([node.controlPoints[3], node.controlPoints[4]])
        self.assertEqual(sel.kind, "cv")
        self.assertTrue(sel.flat)
        self.assertEqual(sel.tokens, ("controlPoints[3:4]",))
        np.testing.assert_array_equal(sel.indices, [3, 4])
        # Flat ids and native coordinates never merge into one token list.
        result = normalise([node.controlPoints[3], node.cv[0, 0]], want_shapes=True)
        self.assertEqual(
            [(s.kind, s.flat) for s in result], [("cv", True), ("cv", False)]
        )

    def test_curve_cvs_are_flat_native_ids(self):
        _, shape = _circle()
        sel      = self._one(Node(shape).cv[:3])
        self.assertEqual((sel.kind, sel.node_type), ("cv", "nurbsCurve"))
        self.assertEqual(sel.tokens, ("cv[0:2]",))
        self.assertFalse(sel.flat)

    def test_lattice_component_plugs(self):
        xform, _ = _cube()
        _, shape = _lattice(xform)
        node = Node(shape)
        sel  = self._one(node.pt[1, 2, 3])
        self.assertEqual((sel.kind, sel.node_type), ("pt", "lattice"))
        self.assertEqual(sel.tokens, ("pt[1][2][3]",))
        sel = self._one(node.pt[0, 0, :])
        self.assertEqual(sel.tokens, ("pt[0][0][0:3]",))
        sel = self._one(node.pt)
        self.assertEqual(sel.tokens, ("pt[*]",))
        sel = self._one(node.controlPoints[5])
        self.assertTrue(sel.flat)
        self.assertEqual(sel.tokens, ("controlPoints[5]",))

    def test_uv_plugs(self):
        _, shape = _cube()
        node = Node(shape)
        sel  = self._one(node.map[:4])
        self.assertEqual(sel.kind, "uv")
        self.assertEqual(sel.tokens, ("map[0:3]",))
        sel = self._one(node.map)
        self.assertTrue(sel.is_all)
        self.assertEqual(sel.tokens, ("map[*]",))

    def test_bare_faces_and_edges(self):
        xform, shape = _cube()
        node = Node(xform)
        sel  = self._one(node.f)
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("f", shape, "mesh"))
        self.assertTrue(sel.is_all)
        self.assertEqual(sel.tokens, ("f[*]",))
        sel = self._one(node.e)
        self.assertEqual(sel.tokens, ("e[*]",))
        self.assertEqual(sel.source, (node.e,))

    def test_face_selections(self):
        xform, shape = _cube()
        sel          = self._one(Node(xform).f[[0, 1, 2, 5]])
        self.assertEqual(sel.tokens, ("f[0:2]", "f[5]"))
        np.testing.assert_array_equal(sel.indices, [0, 1, 2, 5])
        self.assertFalse(sel.is_all)
        sel = self._one(Components(shape, "vtx", [7]))
        self.assertEqual(sel.tokens, ("vtx[7]",))

    def test_mixed_pluglist_keeps_every_kind_in_order(self):
        xform, shape = _cube()
        node   = Node(shape)
        lhs    = PlugList([node.vtx[0], Node(xform).f[0], Node(xform)])
        result = normalise(lhs, want_shapes=True)
        self.assertEqual([s.kind for s in result],   ["vtx", "f", "whole"])
        self.assertEqual({s.path for s in result},   {shape})
        self.assertEqual([s.tokens for s in result], [("vtx[0]",), ("f[0]",), ()])

    def test_equal_path_and_kind_merge(self):
        xform, shape = _cube()
        node  = Node(shape)
        faces = Node(xform).f
        sel   = self._one([faces[:2], faces[[4]], faces[1]])
        self.assertEqual(sel.tokens, ("f[0:1]", "f[4]"))
        self.assertEqual(len(sel.source), 3)
        sel = self._one([node.vtx[0], Components(xform, "vtx", [3]), node.vtx[0]])
        self.assertEqual(sel.tokens, ("vtx[0]", "vtx[3]"))
        sel = self._one([node.vtx[0], node.vtx])
        self.assertTrue(sel.is_all)
        self.assertEqual(sel.tokens, ("vtx[*]",))
        sel = self._one([Node(xform), Node(shape)])
        self.assertEqual(sel.kind, "whole")
        self.assertEqual(len(sel.source), 2)

    def test_whole_and_components_on_one_path_both_survive(self):
        _, shape = _cube()
        node   = Node(shape)
        result = normalise([node, node.vtx[0]], want_shapes=True)
        self.assertEqual([s.kind for s in result], ["whole", "vtx"])

    def test_nested_sequences_flatten(self):
        xform, shape = _cube()
        node   = Node(shape)
        lhs    = [[node.vtx[0]], (Node(xform).f[0], [node.map[1]])]
        result = normalise(lhs, want_shapes=True)
        self.assertEqual([s.kind for s in result], ["vtx", "f", "uv"])
        lhs    = PlugList([PlugList([node.vtx[1]]), node.vtx[2]])
        result = normalise(lhs, want_shapes=True)
        self.assertEqual(result[0].tokens, ("vtx[1:2]",))

    def test_multi_node_grouping(self):
        _, shape1 = _cube("pCube1")
        _, shape2 = _cube("pCube2")
        lhs    = [Node(shape1).vtx[0], Node(shape2).vtx[0], Node(shape1).vtx[1]]
        result = normalise(lhs, want_shapes=True)
        self.assertEqual(
            [(s.path, s.tokens) for s in result],
            [(shape1, ("vtx[0:1]",)), (shape2, ("vtx[0]",))],
        )

    def test_numbers_none_and_strings_raise_naming_the_element(self):
        xform, shape = _cube()
        node         = Node(xform)
        with self.assertRaises(TypeError) as ctx:
            normalise(PlugList([node, 5, None]), want_shapes=True)
        self.assertIn("element [1]", str(ctx.exception))
        self.assertIn("5", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            normalise([node, None], want_shapes=True)
        self.assertIn("element [1]", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            normalise([[node, [True]]], want_shapes=True)
        self.assertIn("element [0][1][0]", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            normalise("pCube1.f[0:3]", want_shapes=True)
        self.assertIn("Components('pCube1.f[0:3]')", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            normalise([node.f, f"{shape}.vtx[0]"], want_shapes=True)
        self.assertIn("element [1]", str(ctx.exception))
        with self.assertRaises(TypeError):
            normalise(5, want_shapes=True)
        with self.assertRaises(TypeError):
            normalise(None, want_shapes=True)

    def test_attribute_plugs_stand_for_their_node(self):
        xform, shape = _cube()
        node = Node(xform)
        sel  = self._one(node.tx, want_shapes=True)
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("whole", shape, "mesh"))
        self.assertEqual(sel.source, (node.tx,))
        sel = self._one(node.tx, want_shapes=False)
        self.assertEqual((sel.kind, sel.path), ("whole", "|pCube1"))
        # two plugs of one node are one node; a plug and its node merge
        sel = self._one([node.tx, node.ry], want_shapes=False)
        self.assertEqual(len(sel.source), 2)
        sel = self._one([node.t, node], want_shapes=False)
        self.assertEqual(sel.kind, "whole")
        result = normalise([Node(shape).vtx[0], node.tx], want_shapes=True)
        self.assertEqual([s.kind for s in result], ["vtx", "whole"])
        # a component's child and a non-component multi element are attributes
        self.assertEqual(self._one(Node(shape).vtx[0].xValue).kind, "whole")
        self.assertEqual(self._one(Node(shape).componentTags[0]).kind, "whole")
        cmds.createNode("multiplyDivide", name="md1")
        sel = self._one(Node("md1").input1X, want_shapes=True)
        self.assertEqual((sel.kind, sel.path, sel.node_type), ("whole", "md1", "multiplyDivide"))

    def test_empty_left_hand_side_raises(self):
        xform, _ = _cube()
        faces    = Node(xform).f
        for lhs in (PlugList([]), [], (), faces[6:], [faces[6:]], faces[[]]):
            with self.assertRaises(ValueError) as ctx:
                normalise(lhs, want_shapes=True)
            self.assertEqual(str(ctx.exception), "nothing to inject")

    def test_selection_indices_are_read_only(self):
        _, shape = _cube()
        sel      = self._one(Node(shape).vtx[:3])
        with self.assertRaises(ValueError):
            sel.indices[0] = 9


class TestNormalisePerformance(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_hundred_thousand_vertex_slice(self):
        plane = cmds.polyPlane(name="bigPlane", sx=316, sy=316, ch=False)[0]
        node  = Node(cmds.listRelatives(plane, shapes=True, fullPath=True)[0])
        self.assertEqual((node >> None).num_weight_points, 100489)
        sliced  = node.vtx[:]
        started = time.perf_counter()
        sel     = normalise(sliced, want_shapes=True)[0]
        elapsed = time.perf_counter() - started
        self.assertEqual(sel.tokens, ("vtx[0:100488]",))
        self.assertEqual(len(sel.indices), 100489)
        self.assertLess(elapsed, 0.3, f"normalise took {elapsed:.3f} s")

    def test_bare_handle_builds_zero_plugs(self):
        plane   = cmds.polyPlane(name="bigPlane", sx=316, sy=316, ch=False)[0]
        node    = Node(cmds.listRelatives(plane, shapes=True, fullPath=True)[0])
        started = time.perf_counter()
        handle  = node.vtx
        sel     = normalise(handle, want_shapes=True)[0]
        elapsed = time.perf_counter() - started
        self.assertIsInstance(handle, Plug)
        self.assertTrue(sel.is_all)
        self.assertIsNone(sel.indices)
        self.assertEqual(sel.tokens, ("vtx[*]",))
        self.assertLess(elapsed, 0.05, f"bare handle took {elapsed:.3f} s")
        started = time.perf_counter()
        faces   = node.f
        sel     = normalise(faces, want_shapes=True)[0]
        elapsed = time.perf_counter() - started
        self.assertTrue(faces.is_all)
        self.assertEqual(sel.tokens, ("f[*]",))
        self.assertLess(elapsed, 0.05, f"face handle took {elapsed:.3f} s")
        self.assertEqual(faces.count, 316 * 316)

    def test_explicit_carrier_at_scale(self):
        plane   = cmds.polyPlane(name="bigPlane", sx=316, sy=316, ch=False)[0]
        ids     = np.arange(0, 100489, 2)
        started = time.perf_counter()
        verts   = Components(plane, "vtx", ids)
        sel     = normalise(verts, want_shapes=True)[0]
        elapsed = time.perf_counter() - started
        self.assertEqual(verts.count, len(ids))
        self.assertEqual(len(sel.tokens), len(ids))
        self.assertLess(elapsed, 0.5, f"carrier took {elapsed:.3f} s")


# --------------------------------------------------------------------- #
#  Attribute fancy indexing (cube.vtx[ids])
# --------------------------------------------------------------------- #


class TestAttributeFancyIndexing(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_vtx_takes_a_sequence_of_ids(self):
        xform, shape = _cube()
        node   = Node(xform)
        picked = node.vtx[[0, 4, 7]]
        self.assertIsInstance(picked, PlugList)
        self.assertEqual(
            [str(x) for x in picked],
            [f"pCube1Shape.controlPoints[{i}]" for i in (0, 4, 7)],
        )
        picked = node.vtx[np.array([1, -1])]
        self.assertEqual(
            [str(x) for x in picked],
            ["pCube1Shape.controlPoints[1]", "pCube1Shape.controlPoints[7]"],
        )
        self.assertEqual(len(node.vtx[()]), 0)
        self.assertEqual(str(node.vtx[np.int64(2)]), "pCube1Shape.controlPoints[2]")

    def test_round_trip_through_components(self):
        xform, shape = _cube()
        node = Node(xform)
        ids  = node.f[[1, 3]] >> None
        sel  = normalise(node.vtx[ids], want_shapes=True)[0]
        self.assertEqual(sel.tokens, ("vtx[1]", "vtx[3]"))

    def test_rejections(self):
        xform, _ = _cube()
        node     = Node(xform)
        with self.assertRaises(TypeError):
            node.vtx[[True, False]]
        with self.assertRaises(TypeError):
            node.vtx[[0.5]]
        with self.assertRaises(TypeError):
            node.vtx[[[0, 1]]]
        with self.assertRaises(IndexError):
            node.vtx[[0, 8]]
        with self.assertRaises(IndexError):
            node.vtx[[-9]]

    def test_plain_multi_keeps_sparse_semantics(self):
        cmds.createNode("transform", name="ctrl")
        cmds.addAttr("ctrl", longName="weights", multi=True, attributeType="double")
        picked = Node("ctrl").weights[[2, 5]]
        self.assertEqual(
            [str(x) for x in picked], ["ctrl.weights[2]", "ctrl.weights[5]"]
        )


# --------------------------------------------------------------------- #
#  _MemberSpec placeholder contract
# --------------------------------------------------------------------- #


class TestMemberSpecContract(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_name_contract(self):
        with self.assertRaises(TypeError):
            _Spec("")
        with self.assertRaises(TypeError):
            _Spec(5)
        spec = _Spec("cap", at="pCube1")
        self.assertEqual(spec.name,  "cap")
        self.assertEqual(str(spec),  "cap")
        self.assertEqual(repr(spec), "_Spec('cap')")
        self.assertFalse(spec.removes)
        self.assertFalse(spec.purges)
        self.assertEqual(spec._options, {"at": "pCube1"})
        # an empty call is the purge: the same object as Spec(None)
        for purge in (_Spec(), _Spec(None)):
            self.assertTrue(purge.purges)
            self.assertIsNone(purge.name)
            self.assertEqual(repr(purge), "_Spec(None)")

    def test_negation_copies(self):
        spec    = _Spec("cap", at="pCube1")
        removal = -spec
        self.assertIsNot(removal, spec)
        self.assertIsInstance(removal, _Spec)
        self.assertTrue(removal.removes)
        self.assertFalse(spec.removes)
        self.assertEqual(removal.name, "cap")
        self.assertEqual(removal._options, spec._options)
        self.assertIsNot(removal._options, spec._options)
        self.assertEqual(repr(removal), "-_Spec('cap')")
        with self.assertRaises(TypeError):
            -removal
        with self.assertRaises(TypeError):
            -_Spec(None)
        with self.assertRaises(TypeError):
            ~spec
        with self.assertRaises(TypeError):
            ~_Spec(None)

    def test_verbs_reach_the_kind_hooks(self):
        # The base normalises and checks the LHS, then hands over to the
        # per-kind hooks; a kind that implements none of them says so.
        xform, _ = _cube()
        node     = Node(xform)
        with self.assertRaises(NotImplementedError):
            _Spec("cap").inject(node)
        with self.assertRaises(NotImplementedError):
            _Spec("cap").query(node)
        with self.assertRaises(NotImplementedError):
            _Spec.of(node)
        with self.assertRaises(TypeError):
            _Spec("cap").inject(None)
        with self.assertRaises(TypeError):
            (-_Spec("cap")).query(node)
        # the purge on '>>' enumerates through the same hook as ``of``
        with self.assertRaises(NotImplementedError):
            _Spec().query(node)
        # ACCEPTS is applied before the hooks.
        with self.assertRaisesRegex(TypeError, "does not take 'f' components"):
            _Strict("cap").inject(node.f)
        with self.assertRaisesRegex(TypeError, "does not take 'whole' components"):
            _Strict("cap").query(node)
        with self.assertRaisesRegex(TypeError, "does not take 'whole' components"):
            _Strict().query(node)
