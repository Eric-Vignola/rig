"""Vertex colours both ways (the user's decisions of 2026-11-09): ``Mesh.serialize`` writes
the colours of the mesh's first colour set into the data's ``colors`` / ``color_indices``
(cgmath r17-skeleton 72dfec4), and ``Mesh.create``, ``Mesh.load`` and ``Mesh('x').load``
bring them back as a colour set, the restore flag ``colors=`` beside ``normals=``.

* serialize: the first set (``getColorSetNames()[0]``), in one API call: the distinct RGBA
  colours and the one each face corner names; an unset corner is (0, 0, 0, 0), as Maya's own
  fbx exporter writes it; an RGB or Alpha set gives its channels as Maya holds them; a mesh
  with no set, or whose first set holds no colour, None. ``colors=False`` leaves them out,
  ``normals=False`` the normals, ``uvs=False`` the UVList; a flag that is not a bool is a
  TypeError before anything is read. ``save`` takes ``normals=`` / ``colors=`` the same way.
* create: ``colorSet1``, RGBA, unclamped, one colour a face corner, through the API inside
  the create's one undo step; colours held one a vertex go on every corner of their vertex;
  a holed face keeps each colour on its vertex; data without colours makes no set;
  ``colors=False`` reads none. Invalid colours are a RuntimeError before anything is made.
* Mesh('x').load: the data's colours into the mesh's first colour set, its representation
  kept, ``colorSet1`` made when it has none; ``colors=False``, or data without colours, leaves
  its sets alone; one undo step with the load.

The restore flag's pattern (the three doors, the TypeError, ``_RESTORE_FLAGS``) is pinned in
test_fu_normals_flag for both flags.
"""

import os
import re
from unittest import mock

import numpy as np
from maya import cmds
from maya.api import OpenMaya

from rig.nodetypes import Mesh
from rig.nodetypes import mesh as mesh_module
from rig.nodetypes.mesh import ColorSet
from rig._tests._undo import mesh_state
from rig._tests.test_class_mesh_create import _facet
from rig._tests.test_class_saveload import _Case

RGBA  = ColorSet.Representation.RGBA
RGB   = ColorSet.Representation.RGB
ALPHA = ColorSet.Representation.Alpha
UNSET = (0.0, 0.0, 0.0, 0.0)


def _shape(t):
    return cmds.listRelatives(t, shapes=True)[0]


def _fn(shape):
    sel = OpenMaya.MSelectionList()
    sel.add(shape)
    return OpenMaya.MFnMesh(sel.getDagPath(0))


def _corner_colors(shape, color_set):
    """Maya's colour of each face corner of the set `color_set` of `shape`, ``(C, 4)``; an
    unset corner (0, 0, 0, 0)."""
    held = _fn(shape).getFaceVertexColors(color_set, OpenMaya.MColor(UNSET))
    return np.array([tuple(c) for c in held], dtype=np.float64).reshape(-1, 4)


def _per_corner(data):
    """The data's colour of each face corner, ``(C, 4)``, through its form: the colour
    ``color_indices`` names, else its vertex's."""
    colors = np.asarray(data.colors, dtype=np.float64)
    ids    = np.asarray(data.indices if data.color_indices is None else data.color_indices)
    return colors[ids]


def _painted(name="src"):
    """A polyCube without history, its vertices red at alpha 0.5 and the corners of its face 0
    green (one set, ``colorSet1``: colours one a face corner, two distinct ones): the transform."""
    t = cmds.polyCube(name=name, constructionHistory=False)[0]
    cmds.polyColorPerVertex(t, rgb=(1.0, 0.0, 0.0), a=0.5, colorDisplayOption=False)
    cmds.polyColorPerVertex(f"{t}.f[0]", rgb=(0.0, 1.0, 0.0), a=1.0, colorDisplayOption=False)
    cmds.delete(t, constructionHistory=True)
    return t


def _cube(name="cube"):
    return cmds.polyCube(name=name, constructionHistory=False)[0]


# ---------------------------------------------------------------------------------------------
class TestSerializeColors(_Case):
    """``serialize``'s colours: the first colour set, as a MeshData holds colours."""

    def test_no_colour_set_gives_none(self):
        data = Mesh(_cube()).serialize(uvs=False)
        self.assertEqual((data.colors, data.color_indices), (None, None))

    def test_an_empty_first_set_gives_none(self):
        t = _cube()
        Mesh(t).add_color_set("empty", RGBA)
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual((data.colors, data.color_indices), (None, None))

    def test_the_corners_colours_compacted_to_the_distinct_ones(self):
        t       = _painted()
        corners = _corner_colors(_shape(t), "colorSet1")
        data    = Mesh(t).serialize(uvs=False)
        self.assertEqual(data.colors.dtype, np.float64)
        self.assertEqual(data.color_indices.dtype, np.int32)
        self.assertEqual((data.colors.shape, data.color_indices.shape), ((2, 4), (24,)))
        np.testing.assert_allclose(_per_corner(data), corners, rtol=0.0, atol=1e-6)
        self.assertEqual(len(np.unique(corners, axis=0)), 2)

    def test_a_colour_a_vertex_set_through_its_corners(self):
        # ColorSet.set_data writes one colour a vertex: each corner names its vertex's
        t    = _cube()
        ramp = np.column_stack((np.linspace(0, 1, 8), np.linspace(1, 0, 8), np.full(8, 0.5), np.full(8, 0.25)))
        Mesh(t).add_color_set("tint", RGBA).set_data(ramp)
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(data.colors.shape, (8, 4))
        np.testing.assert_allclose(_per_corner(data), ramp[np.asarray(data.indices)], rtol=0.0, atol=1e-6)

    def test_an_unset_corner_is_transparent_black(self):
        t = _cube()
        cmds.polyColorPerVertex(f"{t}.vtx[0:3]", rgb=(1.0, 0.0, 0.0), a=1.0, colorDisplayOption=False)
        cmds.delete(t, constructionHistory=True)
        data    = Mesh(t).serialize(uvs=False)
        corners = _per_corner(data)
        low     = np.asarray(data.indices) <= 3
        np.testing.assert_allclose(corners[low], np.tile([1.0, 0.0, 0.0, 1.0], (int(low.sum()), 1)), rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(corners[~low], np.tile(UNSET, (int((~low).sum()), 1)))
        # what Maya reports for such a corner (and its fbx exporter writes)
        self.assertEqual(cmds.polyColorPerVertex(f"{t}.vtx[4]", query=True, rgb=True), [0.0, 0.0, 0.0])
        self.assertEqual(cmds.polyColorPerVertex(f"{t}.vtx[4]", query=True, a=True), [0.0])

    def test_an_rgb_and_an_alpha_set_give_their_channels(self):
        t    = _cube()
        mesh = Mesh(t)
        mesh.add_color_set("rgb", RGB).set_data(np.tile([0.2, 0.4, 0.6, 0.9], (8, 1)))
        data = mesh.serialize(uvs=False)
        self.assertEqual(data.colors.shape, (1, 4))
        np.testing.assert_allclose(data.colors[0, :3], [0.2, 0.4, 0.6], rtol=0.0, atol=1e-6)
        self.assertEqual(float(data.colors[0, 3]), 1.0)  # an RGB set's alpha reads 1
        ColorSet(mesh, "rgb").delete()
        mesh.add_color_set("alpha", ALPHA).set_data(np.tile([0.2, 0.4, 0.6, 0.25], (8, 1)))
        data = mesh.serialize(uvs=False)
        self.assertEqual(data.colors.shape, (1, 4))
        self.assertAlmostEqual(float(data.colors[0, 3]), 0.25, places=6)

    def test_the_first_set_not_the_current_one(self):
        t    = _cube()
        mesh = Mesh(t)
        mesh.add_color_set("first", RGBA).set_data(np.tile([1.0, 0.0, 0.0, 1.0], (8, 1)))
        mesh.add_color_set("second", RGBA).set_data(np.tile([0.0, 0.0, 1.0, 1.0], (8, 1)))
        cmds.polyColorSet(_shape(t), currentColorSet=True, colorSet="second")
        self.assertEqual(_fn(_shape(t)).currentColorSetName(), "second")
        data = mesh.serialize(uvs=False)
        np.testing.assert_allclose(data.colors, [[1.0, 0.0, 0.0, 1.0]], rtol=0.0, atol=1e-6)
        self.assertEqual(_fn(_shape(t)).currentColorSetName(), "second")  # a read

    def test_a_holed_face_lists_its_hole_corners_too(self):
        t = _facet()
        cmds.polyColorPerVertex(f"{t}.vtx[0:2]", rgb=(1.0, 1.0, 0.0), a=1.0, colorDisplayOption=False)
        cmds.delete(t, constructionHistory=True)
        data = Mesh(t).serialize(uvs=False)
        self.assertEqual(data.color_indices.shape, (8,))
        np.testing.assert_allclose(_per_corner(data), _corner_colors(_shape(t), "colorSet1"), rtol=0.0, atol=1e-6)

    def test_world_space_keeps_the_colours(self):
        t = _painted()
        cmds.xform(t, translation=(1, 2, 3), rotation=(10, 20, 30))
        local, world = Mesh(t).serialize(uvs=False), Mesh(t).serialize(uvs=False, world_space=True)
        np.testing.assert_array_equal(_per_corner(world), _per_corner(local))

    def test_the_flags_leave_a_part_out(self):
        mesh = Mesh(_painted())
        data = mesh.serialize(uvs=False, colors=False)
        self.assertEqual((data.colors, data.color_indices), (None, None))
        self.assertIsNotNone(data.normals)
        data = mesh.serialize(uvs=False, normals=False)
        self.assertEqual((data.normals, data.normal_indices), (None, None))
        self.assertIsNotNone(data.colors)
        both = mesh.serialize()
        self.assertEqual((len(both), [uv.name for uv in both[1]]), (2, ["map1"]))
        with mock.patch.object(mesh_module, "_first_set_colors", wraps=mesh_module._first_set_colors) as read:
            mesh.serialize(uvs=False, colors=False)
        read.assert_not_called()  # colors=False reads none

    def test_a_flag_that_is_not_a_bool_is_a_type_error_before_anything_is_read(self):
        mesh = Mesh(_painted())
        for flag in ("uvs", "normals", "colors"):
            for value in (None, 0, 1, "yes", [True]):
                with self.subTest(flag=flag, value=value):
                    with mock.patch.object(mesh_module, "_first_set_colors", wraps=mesh_module._first_set_colors) as read:
                        with self.assertRaisesRegex(
                            TypeError,
                            f"^Mesh\\.serialize\\(\\): {flag}= is True \\(the default: the mesh's {flag} are "
                            f"serialized\\) or False \\(left out\\), not {re.escape(repr(value))}$",
                        ):
                            mesh.serialize(**{flag: value})
                    read.assert_not_called()
        # numpy's bool is a bool; save takes the same flags, checked before its paths
        self.assertIsNone(mesh.serialize(uvs=False, colors=np.bool_(False)).colors)
        with self.assertRaisesRegex(TypeError, "^Mesh\\.save: colors= is True \\(the default"):
            mesh.save(self.path("m.npz"), colors="no")
        self.assertFalse(os.path.exists(self.path("m.npz")))

    def test_save_and_the_plug_carry_them(self):
        from cgmath.geometry import MeshData

        mesh    = Mesh(_painted())
        corners = _per_corner(mesh.serialize(uvs=False))
        for ext in ("npz", "json"):
            with self.subTest(ext):
                back = MeshData.load(mesh.save(self.path(f"m.{ext}")))
                np.testing.assert_allclose(_per_corner(back), corners, rtol=0.0, atol=1e-6)
        self.assertIsNone(MeshData.load(mesh.save(self.path("plain.npz"), colors=False)).colors)
        self.assertIsNone(MeshData.load(mesh.save(self.path("flat.npz"), normals=False)).normals)
        data, _ = mesh.outMesh.get()
        np.testing.assert_allclose(_per_corner(data), corners, rtol=0.0, atol=1e-6)

    def test_the_transfers_read_no_colours(self):
        from cgmath.geometry.resample import ResampleMode

        src, dst = Mesh(_painted("src")), Mesh(_painted("dst"))
        with mock.patch.object(mesh_module, "_first_set_colors", wraps=mesh_module._first_set_colors) as read:
            mesh_module._resampler(src, dst)
            mesh_module._resampler(src, dst, ResampleMode.UV)
        read.assert_not_called()


# ---------------------------------------------------------------------------------------------
class TestRestoreColors(_Case):
    """``colors=``: the data's colours back as a colour set, on the three doors."""

    def source(self, t=None):
        """``(data, uvs, Maya's colour of each corner)`` of the transform `t` (a painted cube
        when None), then `t` deleted."""
        t       = t or _painted()
        corners = _corner_colors(_shape(t), "colorSet1") if _fn(_shape(t)).getColorSetNames() else None
        data, uvs = Mesh(t).serialize()
        cmds.delete(t)
        return data, list(uvs), corners

    def assert_colours(self, mesh, corners, name="colorSet1", rep=RGBA):
        shape = str(mesh)
        fn    = _fn(shape)
        self.assertEqual(list(fn.getColorSetNames()), [name])
        self.assertEqual(fn.getColorRepresentation(name), rep.value)
        self.assertFalse(fn.isColorClamped(name))
        np.testing.assert_allclose(_corner_colors(shape, name), corners, rtol=0.0, atol=1e-6)

    def test_create_gives_the_first_set_back_corner_for_corner(self):
        data, uvs, corners = self.source()
        self.ready()
        mesh = Mesh.create(data, uvs=uvs, name="m")
        self.assert_colours(mesh, corners)
        self.assertEqual(self.undo_name(), "rig.Mesh.create")
        self.assert_kept_selection()
        np.testing.assert_allclose(_per_corner(mesh.serialize(uvs=False)), _per_corner(data), rtol=0.0, atol=1e-6)
        # one undo step: the mesh and its set go, and come back
        self.undo_steps(1)
        self.assertFalse(cmds.objExists("m"))
        self.redo_steps(1)
        self.assert_colours(Mesh("mShape"), corners)

    def test_create_star_serialize(self):
        t       = _painted()
        corners = _corner_colors(_shape(t), "colorSet1")
        self.assert_colours(Mesh.create(*Mesh(t).serialize(), name="twin"), corners)

    def test_a_colour_a_vertex_goes_on_every_corner_of_its_vertex(self):
        data, uvs, _ = self.source(_cube("src"))
        ramp               = np.column_stack((np.linspace(0, 1, 8), np.linspace(1, 0, 8), np.full(8, 0.5), np.ones(8)))
        data.colors        = ramp
        data.color_indices = None
        mesh               = Mesh.create(data, uvs=uvs, name="m")
        self.assert_colours(mesh, ramp[np.asarray(data.indices)])
        self.assertEqual(_fn(str(mesh)).numColors("colorSet1"), 8)  # the pool as the data's: one a vertex

    def test_a_holed_face_keeps_each_colour_on_its_vertex(self):
        t = _facet()
        cmds.polyColorPerVertex(f"{t}.vtx[0:2]", rgb=(1.0, 1.0, 0.0), a=1.0, colorDisplayOption=False)
        cmds.polyColorPerVertex(f"{t}.vtx[5]", rgb=(0.0, 0.0, 1.0), a=1.0, colorDisplayOption=False)
        cmds.delete(t, constructionHistory=True)
        shape     = _shape(t)
        by_vertex = dict(zip(_fn(shape).getVertices()[1], map(tuple, _corner_colors(shape, "colorSet1"))))
        data, uvs = Mesh(t).serialize()
        cmds.delete(t)

        def placed(mesh):
            fn = _fn(str(mesh))
            return dict(zip(fn.getVertices()[1], map(tuple, _corner_colors(str(mesh), "colorSet1"))))

        self.assertEqual(placed(Mesh.create(data, uvs=uvs, name="m")), by_vertex)
        # the data's face listing its hole corners the other way round than its hole loop does:
        # each colour still lands on its vertex (as a UV does)
        face    = np.asarray(data.indices).copy()
        colour  = np.asarray(data.color_indices).copy()
        on_hole = np.isin(face, np.asarray(data.hole_indices))
        face[on_hole], colour[on_hole] = face[on_hole][::-1], colour[on_hole][::-1]
        rolled               = data.copy()
        rolled.indices       = face
        rolled.color_indices = colour
        self.assertEqual(placed(Mesh.create(rolled, name="o")), by_vertex)

    def test_data_without_colours_makes_no_set_and_false_reads_none(self):
        data, uvs, _ = self.source()
        plain               = data.copy()
        plain.colors        = None
        plain.color_indices = None
        self.assertEqual(Mesh.create(plain, uvs=uvs, name="plain").get_color_sets(), [])
        with mock.patch.object(mesh_module, "_data_colors", wraps=mesh_module._data_colors) as read, \
                mock.patch.object(mesh_module, "_restore_colors", wraps=mesh_module._restore_colors) as restore:
            mesh = Mesh.create(data, uvs=uvs, name="m", colors=False)
        read.assert_not_called()
        restore.assert_not_called()
        self.assertEqual(mesh.get_color_sets(), [])
        # the flag, not the mesh's own colors attribute (its internal colour array)
        self.assertTrue(cmds.attributeQuery("colors", type="mesh", exists=True))

    def test_invalid_colours_raise_with_nothing_made(self):
        data, uvs, _ = self.source()
        good_c, good_i = np.asarray(data.colors), np.asarray(data.color_indices)
        for label, colors, ids, text in (
            ("ids short", good_c, good_i[:-1], "23 color indices for 24 face-vertices (one each)"),
            ("id out of range", good_c, np.where(good_i == 0, 99, good_i), "a color index is not one of the 2 colors"),
            ("not RGBA", good_c[:, :3], good_i, "the colors are (2, 3), not (N, 4) RGBA"),
            ("nan", np.where(good_c == 0, np.nan, good_c), good_i, "the colors must be finite"),
            ("a colour a vertex, short", good_c, None, "2 colors for 8 vertices and no color_indices"),
        ):
            with self.subTest(label):
                # past cgmath's own checks (a MeshData built by other means)
                object.__setattr__(data, "colors", colors)
                object.__setattr__(data, "color_indices", ids)
                self.assert_refused(
                    RuntimeError, "^" + re.escape(f"MeshData: {text}"), lambda: Mesh.create(data, uvs=uvs, name="m")
                )

    def test_the_class_load_creates_with_them(self):
        data, uvs, corners = self.source()
        made = Mesh.create(data, uvs=uvs, name="body")
        path, uv_path = made.save(self.path("body.npz"), uvs=self.path("body_uv.npz"))
        self.ready()
        loaded = Mesh.load(path, uv_path, name="copy")
        self.assert_colours(loaded, corners)
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        self.assert_kept_selection()
        self.assertEqual(Mesh.load(path, uv_path, name="plain", colors=False).get_color_sets(), [])

    def test_the_update_writes_the_first_set(self):
        data, uvs, corners = self.source()
        made = Mesh.create(data, uvs=uvs, name="body")
        path, uv_path = made.save(self.path("body.npz"), uvs=self.path("body_uv.npz"))
        # a mesh without a colour set gets colorSet1
        target = Mesh(_cube("target"))
        self.ready()
        self.assertIs(target.load(path, uv_path), target)
        self.assert_colours(target, corners)
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        self.assert_kept_selection()
        # one with an RGB first set keeps it, and its second set
        rgb = Mesh(_cube("rgb"))
        rgb.add_color_set("tint", RGB).set_data(np.tile([0.2, 0.4, 0.6, 1.0], (8, 1)))
        rgb.add_color_set("second", RGBA)
        cmds.polyColorSet("rgbShape", currentColorSet=True, colorSet="second")
        rgb.load(path, uv_path)
        fn = _fn("rgbShape")
        self.assertEqual(list(fn.getColorSetNames()), ["tint", "second"])
        self.assertEqual(fn.getColorRepresentation("tint"), RGB.value)
        np.testing.assert_allclose(_corner_colors("rgbShape", "tint")[:, :3], corners[:, :3], rtol=0.0, atol=1e-6)
        self.assertEqual(fn.numColors("second"), 0)
        self.assertEqual(fn.currentColorSetName(), "second")
        # colors=False, and a file without colours: the mesh's own stay
        own = mesh_state("rgbShape")["colors"]
        rgb.load(path, uv_path, colors=False)
        self.assertEqual(mesh_state("rgbShape")["colors"], own)
        rgb.load(made.save(self.path("plain.npz"), colors=False), uv_path)
        self.assertEqual(mesh_state("rgbShape")["colors"], own)

    def test_undo_and_redo_of_the_update(self):
        data, uvs, _ = self.source()
        made = Mesh.create(data, uvs=uvs, name="body")
        path, uv_path = made.save(self.path("body.npz"), uvs=self.path("body_uv.npz"))
        for label, make in (
            ("no set", lambda t: None),
            ("an RGBA first set", lambda t: Mesh(t).add_color_set("c", RGBA).set_data(np.tile([0.0, 0.0, 1.0, 1.0], (8, 1)))),
        ):
            with self.subTest(label):
                t = _cube("t")
                make(t)
                target = Mesh(t)
                before = mesh_state(str(target))
                self.ready()
                target.load(path, uv_path)
                after = mesh_state(str(target))
                self.assertNotEqual(after["colors"], before["colors"])
                for _ in range(2):
                    self.undo_steps(1)
                    self.assertEqual(mesh_state(str(target)), before)
                    self.redo_steps(1)
                    self.assertEqual(mesh_state(str(target)), after)
                cmds.delete(t)
