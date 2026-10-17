"""Follow-up of the features run (the user's decision 1 of 2026-10-16): ``normals=``, a
restore flag.

``Mesh.create``, ``Mesh.load`` (the class form) and ``Mesh('x').load`` (the instance form)
take ``normals=True``: the data's normals bring back the hard / soft edges and the locked
normals, as before (the default). ``normals=False`` skips that restore, for speed: the data's
normals are not read, a new mesh is all smooth, an updated mesh keeps its own shading.

The pattern is named (``mesh._RESTORE_FLAGS``) so that a future vertex-colour flag follows
it: one keyword per part of the data a create or a load restores at a cost of its own, the
same on the three doors, True by default, a bool (anything else a TypeError before anything
is read or made).

Each test failed at fe09f88 (negative control: ``normals=False`` was taken as the mesh's
internal ``normals`` attribute), except the guards.
"""

import inspect
import re
from unittest import mock

import numpy as np
from maya import cmds

from rig.nodetypes import Mesh
from rig.nodetypes import _files, dg_node
from rig.nodetypes import mesh as mesh_module
from rig._tests._undo import mesh_state
from rig._tests.test_class_saveload import _Case, _hard_edges, _locked_rows


def _shaded(name="src"):
    """A polySphere (8 x 6), its first 8 edges hard and vertex 0's normal locked, without
    history: the transform."""
    t = cmds.polySphere(name=name, sx=8, sy=6, constructionHistory=False)[0]
    cmds.polySoftEdge(t, angle=180)
    cmds.polySoftEdge(f"{t}.e[0:7]", angle=0)
    cmds.polyNormalPerVertex(f"{t}.vtx[0]", xyz=(0.0, 1.0, 0.0))
    cmds.delete(t, constructionHistory=True)
    return t


class _NormalsCase(_Case):
    def source(self):
        """``(data, uvs, hard, locked)`` of a shaded sphere, then the sphere deleted."""
        t = _shaded()
        shape = cmds.listRelatives(t, shapes=True)[0]
        hard, locked = _hard_edges(shape), _locked_rows(shape)
        data, uvs = Mesh(t).serialize(world_space=False)
        cmds.delete(t)
        self.assertEqual(len(hard), 8)
        self.assertTrue(locked)
        return data, list(uvs), hard, locked

    def assert_untouched_shading(self, call):
        """`call()` reads none of the data's normals and restores no shading."""
        with mock.patch.object(mesh_module, "_shading_of", wraps=mesh_module._shading_of) as shading, \
                mock.patch.object(mesh_module, "_restore_shading", wraps=mesh_module._restore_shading) as restore, \
                mock.patch.object(mesh_module, "_data_normals", wraps=mesh_module._data_normals) as read:
            result = call()
        shading.assert_not_called()
        restore.assert_not_called()
        read.assert_not_called()
        return result


class TestTheFlagOnCreate(_NormalsCase):
    def test_the_default_restores_the_shading(self):
        """guard: normals=True is the default, and restores."""
        data, uvs, hard, locked = self.source()
        for label, kwargs in (("default", {}), ("True", {"normals": True})):
            with self.subTest(label):
                mesh = Mesh.create(data, uv_data=uvs, **kwargs)
                self.assertEqual(_hard_edges(str(mesh)), hard)
                self.assertEqual(_locked_rows(str(mesh)), locked)

    def test_false_skips_the_restore(self):
        data, uvs, _hard, _locked = self.source()
        self.ready()
        mesh = self.assert_untouched_shading(lambda: Mesh.create(data, uv_data=uvs, name="m", normals=False))
        self.assertEqual(_hard_edges(str(mesh)), set())
        self.assertEqual(_locked_rows(str(mesh)), set())
        # the same mesh as the data without normals gives, its points and UVs the data's
        plain = data.copy()
        plain.normals = None
        plain.normal_indices = None
        other = Mesh.create(plain, uv_data=uvs, name="plain")
        a, b = mesh_state(str(mesh)), mesh_state(str(other))
        for key in ("points", "uv_sets", "uvs"):
            self.assertEqual(a[key], b[key], key)
        # one undo step, as every create
        self.assertEqual(self.undo_name(), "rig.Mesh.create")

    def test_false_reads_no_normals(self):
        # normals that do not fit the faces are not read: nothing to refuse
        data, uvs, *_ = self.source()
        object.__setattr__(data, "normals", np.asarray(data.normals)[:, :2])
        with self.assertRaisesRegex(RuntimeError, re.escape("not (N, 3)")):
            Mesh.create(data, uv_data=uvs, name="bad")
        mesh = Mesh.create(data, uv_data=uvs, name="m", normals=False)
        self.assertEqual(_hard_edges(str(mesh)), set())

    def test_an_object(self):
        from cgmath.render.scene import Object

        data, uvs, hard, _ = self.source()
        obj = Object(name="hero", mesh=data, uv=uvs[0], translate=(5, 0, 0))
        smooth = self.assert_untouched_shading(lambda: Mesh.create(obj, normals=False))
        self.assertEqual(_hard_edges(str(smooth)), set())
        self.assertEqual(cmds.getAttr(f"{smooth.get_parent()}.t")[0], (5.0, 0.0, 0.0))
        self.assertEqual(_hard_edges(str(Mesh.create(obj))), hard)

    def test_not_the_mesh_attribute(self):
        # ``normals`` is also a mesh attribute (its internal normal array): the flag wins, and
        # the attribute is not set
        self.assertTrue(cmds.attributeQuery("normals", type="mesh", exists=True))
        data, uvs, *_ = self.source()
        with mock.patch.object(dg_node, "_attribute_keywords", wraps=dg_node._attribute_keywords) as attrs:
            Mesh.create(data, uv_data=uvs, name="m", normals=False)
        attrs.assert_not_called()


class TestTheFlagOnLoad(_NormalsCase):
    def saved(self):
        data, uvs, hard, locked = self.source()
        Mesh.create(data, uv_data=uvs, name="body")
        path, uv_path = Mesh("body").save(self.path("body.npz"), uvs=self.path("body_uv.npz"))
        return path, uv_path, hard, locked

    def test_the_class_form(self):
        path, uv_path, hard, locked = self.saved()
        self.ready()
        smooth = self.assert_untouched_shading(lambda: Mesh.load(path, uv_path, name="smooth", normals=False))
        self.assertEqual((_hard_edges(str(smooth)), _locked_rows(str(smooth))), (set(), set()))
        self.assertEqual(list(smooth.get_uv_sets()), ["map1"])
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        self.assert_kept_selection()
        shaded = Mesh.load(path, uv_path, name="shaded")
        self.assertEqual((_hard_edges(str(shaded)), _locked_rows(str(shaded))), (hard, locked))

    def test_the_instance_form_keeps_the_mesh_s_own_shading(self):
        path, uv_path, hard, locked = self.saved()
        copy = Mesh.load(path, uv_path, name="copy", normals=False)
        cmds.polySoftEdge("copy.e[20:23]", angle=0, constructionHistory=False)
        cmds.xform("copy.vtx[3]", translation=(0, 2, 0), relative=True)
        own = _hard_edges("copyShape"), _locked_rows("copyShape")
        self.assertEqual(len(own[0]), 4)
        before = mesh_state("copyShape")
        self.ready()
        self.assertIs(self.assert_untouched_shading(lambda: copy.load(path, uv_path, normals=False)), copy)
        # the points back from the file, the shading the mesh's own
        np.testing.assert_allclose(
            np.array(copy.get_points(world_space=False))[:, :3],
            np.array(Mesh("bodyShape").get_points(world_space=False))[:, :3], atol=1e-6,
        )
        self.assertEqual((_hard_edges("copyShape"), _locked_rows("copyShape")), own)
        self.assertEqual(self.undo_name(), "rig.Mesh.load")
        self.assert_kept_selection()
        self.undo_steps(1)
        self.assertEqual(mesh_state("copyShape"), before)
        self.assertEqual((_hard_edges("copyShape"), _locked_rows("copyShape")), own)
        # the default brings the file's shading back
        copy.load(path, uv_path)
        self.assertEqual((_hard_edges("copyShape"), _locked_rows("copyShape")), (hard, locked))


class TestTheFlagIsABool(_NormalsCase):
    def test_anything_else_is_a_type_error_before_anything_is_read(self):
        data, uvs, *_ = self.source()
        Mesh.create(data, uv_data=uvs, name="body")
        path = Mesh("body").save(self.path("body.npz"))
        for value in (None, 0, 1, "yes", [True]):
            for label, method, call in (
                ("create", "Mesh.create()", lambda: Mesh.create(data, uv_data=uvs, normals=value)),
                ("class load", "Mesh.load", lambda: Mesh.load(path, normals=value)),
                ("instance load", "Mesh('bodyShape').load", lambda: Mesh("body").load(path, normals=value)),
            ):
                with self.subTest(value=value, door=label):
                    before = mesh_state("bodyShape")
                    with mock.patch.object(_files, "read", wraps=_files.read) as read:
                        self.assert_refused(
                            TypeError,
                            f"^{re.escape(method)}: normals= is a restore flag, True \\(the default: the "
                            f"data's normals come back\\) or False \\(skipped\\), not {re.escape(repr(value))}$",
                            call,
                        )
                    read.assert_not_called()
                    self.assertEqual(mesh_state("bodyShape"), before)
        # numpy's bool is a bool
        Mesh.create(data, uv_data=uvs, name="np", normals=np.bool_(False))
        self.assertEqual(_hard_edges("npShape"), set())

    def test_the_instance_form_names_its_keywords(self):
        data, uvs, *_ = self.source()
        Mesh.create(data, uv_data=uvs, name="body")
        path = Mesh("body").save(self.path("body.npz"))
        self.assert_refused(
            TypeError,
            "^Mesh\\('bodyShape'\\).load updates the mesh in place: it takes the files, world_space= and "
            "normals= \\(got name=\\); Mesh.load\\(path, ...\\) makes a new mesh$",
            lambda: Mesh("body").load(path, name="x"),
        )


class TestThePattern(_NormalsCase):
    """The restore flags are one list, and every door takes each of them, True by default."""

    def test_every_door_takes_every_restore_flag(self):
        self.assertEqual(mesh_module._RESTORE_FLAGS, ("normals",))
        data, uvs, *_ = self.source()
        mesh = Mesh.create(data, uv_data=uvs, name="body")
        doors = {
            "Mesh.create": Mesh.create,
            "Mesh.load": Mesh.load,
            "Mesh('x').load": mesh.load,
            "Mesh._create": Mesh._create,
        }
        for flag in mesh_module._RESTORE_FLAGS:
            self.assertIn(flag, Mesh._CREATE_FLAGS)
            for label, door in doors.items():
                with self.subTest(flag=flag, door=label):
                    parameter = inspect.signature(door).parameters[flag]
                    self.assertIs(parameter.default, True)
                    if label != "Mesh._create":
                        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                        self.assertIn(f"{flag}=", inspect.getdoc(door).replace("``", ""))
