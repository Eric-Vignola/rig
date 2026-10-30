"""Follow-up of the features run (the user's decision 2 of 2026-10-16): the instance forms
``NurbsCurve('x').load(...)`` and ``NurbsSurface('x').load(...)`` REQUIRE ``world_space=``.

A BSplineData / BSplinePatchData holds no space: a file saved with ``world_space=False``
and loaded with the old default (True) put its object-space CVs in the world, moved by the
transform, silently. A load without ``world_space=``, or with anything but a bool, is now a
TypeError naming the argument, raised before the file is read (nothing read, nothing
changed, no undo step). ``Mesh('x').load`` keeps its default (a MeshData holds its matrix),
and the class forms (creating) still take no ``world_space=`` without ``transform=``.

Each test failed at fc692b9 (negative control: the default True moved the CVs), except the
guards.

Re-pinned for the user's decision of 2026-10-29 (one space rule: local / object space by
default everywhere, world space by ``world_space=True`` alone, by keyword): the updates in
place take ``world_space=False`` by default, as ``Mesh('x').load`` does, and save writes object
space by default, so a default save and a default load pair up (and a BSplineData holds its
matrix, as a MeshData does, on the cgmath that adds it). The required keyword is gone; its bool
check stays (anything but a bool is a TypeError, before the file is read): the tests pin the
default, the keyword, the check and each space's round trip.

Re-pinned for the user's decision of 2026-10-30: the class forms take ``world_space=`` without
``transform=`` too, where True freezes the new shape at its world positions; the bool check is
theirs as well.
"""

import inspect
import re
from unittest import mock

import numpy as np
from maya import cmds

from rig.nodetypes import Mesh, NurbsCurve, NurbsSurface
from rig.nodetypes import _files
from rig._tests.test_class_saveload import _Case, _cvs


class _SpaceCase(_Case):
    def curve(self):
        """A moved, rotated open curve: its shape."""
        t = cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name="crv")
        cmds.xform(t, translation=(5, 0, 0), rotation=(0, 90, 0))
        return cmds.listRelatives(t, shapes=True)[0]

    def surface(self):
        """A moved plane: its shape."""
        t = cmds.nurbsPlane(name="patch", constructionHistory=False, u=2, v=2)[0]
        cmds.xform(t, translation=(0, 3, 1), rotation=(30, 0, 0))
        return cmds.listRelatives(t, shapes=True)[0]

    def cases(self):
        """``[(class, shape, data class name)]``."""
        return [(NurbsCurve, self.curve(), "BSplineData"), (NurbsSurface, self.surface(), "BSplinePatchData")]

    def every_cv(self, cls, shape):
        return f"{shape}.cv[*]" if cls is NurbsCurve else f"{shape}.cv[*][*]"


class TestTheUpdateDefaultsToObjectSpace(_SpaceCase):
    """Re-pinned (2026-10-29, was TestWorldSpaceIsRequired): no longer required, object space by
    default, still a bool."""

    def test_no_world_space_is_object_space(self):
        """re-pinned: no world_space= loads in object space, the inverse of the default save."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                local, world = _cvs(shape, world_space=False), _cvs(shape)
                path = cls(shape).save(self.path(f"{shape}.npz"))
                cmds.xform(f"{shape}.cv[0]" if cls is NurbsCurve else f"{shape}.cv[0][0]",
                           translation=(0, 4, 0), relative=True)
                self.ready()
                self.assertIs(cls(shape).load(path).__class__, cls)
                np.testing.assert_allclose(_cvs(shape, world_space=False), local, atol=1e-9)
                np.testing.assert_allclose(_cvs(shape), world, atol=1e-9)
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), f"rig.{cls.__name__}.load")
                self.assert_kept_selection()

    def test_anything_but_a_bool_is_a_type_error(self):
        """guard: the check of 2026-10-16, kept with the default."""
        for cls, shape, _data in self.cases():
            path = cls(shape).save(self.path(f"{shape}.npz"))
            for value in (None, 0, 1, "world", [True]):
                with self.subTest(cls=cls.__name__, value=value):
                    with mock.patch.object(_files, "read", wraps=_files.read) as read:
                        self.assert_refused(
                            TypeError,
                            f"^{re.escape(cls.__name__)}\\('{shape}'\\).load: world_space= takes True or False "
                            f"\\(the space the file was saved in\\), not {re.escape(repr(value))}$",
                            lambda: cls(shape).load(path, world_space=value),
                        )
                    read.assert_not_called()

    def test_the_signature_defaults_to_object_space(self):
        """re-pinned: False by default, by keyword only; the docstring requires nothing."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                parameter = inspect.signature(cls(shape).load).parameters["world_space"]
                self.assertIs(parameter.default, False)
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertNotIn("REQUIRED", cls.load.__doc__)

    def test_create_keywords_are_refused_first(self):
        """guard: name= on the instance form is still its own TypeError."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"))
                self.assert_refused(TypeError, r"updates it in place: it takes the file and world_space= \(got name=\)",
                                    lambda: cls(shape).load(path, name="x"))


class TestEachSpaceLoads(_SpaceCase):
    def test_an_object_space_file_round_trips_with_false(self):
        """guard: an object-space file loads with world_space=False, one rig.<Cls>.load step."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                local = _cvs(shape, world_space=False)
                world = _cvs(shape)
                path = cls(shape).save(self.path(f"{shape}_local.npz"), world_space=False)
                cmds.xform(self.every_cv(cls, shape), translation=(0, 2, 0), relative=True)
                self.ready()
                self.assertIs(cls(shape).load(path, world_space=False).__class__, cls)
                np.testing.assert_allclose(_cvs(shape, world_space=False), local, atol=1e-9)
                np.testing.assert_allclose(_cvs(shape), world, atol=1e-9)
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), f"rig.{cls.__name__}.load")
                self.assert_kept_selection()

    def test_a_world_space_file_round_trips_with_true(self):
        """guard: save(world_space=True) with world_space=True (re-pinned: both by keyword)."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                world = _cvs(shape)
                path  = cls(shape).save(self.path(f"{shape}_world.npz"), world_space=True)
                cmds.xform(self.every_cv(cls, shape), translation=(1, 0, 0), relative=True)
                cls(shape).load(path, world_space=True)
                np.testing.assert_allclose(_cvs(shape), world, atol=1e-9)

    def test_the_data_object_too(self):
        """re-pinned: the data object loads in object space by default."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                local = _cvs(shape, world_space=False)
                data  = cls(shape).serialize(world_space=False)
                cmds.xform(self.every_cv(cls, shape), translation=(0, 0, 3), relative=True)
                cls(shape).load(data)
                np.testing.assert_allclose(_cvs(shape, world_space=False), local, atol=1e-9)
                cls(shape).load(data, world_space=np.bool_(False))


class TestTheOtherFormsKeepTheirRules(_SpaceCase):
    def test_the_mesh_has_the_same_default(self):
        """re-pinned: Mesh('x').load(path) is object space too, the inverse of the default save."""
        t = cmds.polyCube(name="body", constructionHistory=False)[0]
        cmds.xform(t, translation=(1, 2, 3))
        local     = np.array(Mesh(t).get_points())[:, :3]
        path      = Mesh(t).save(self.path("body.npz"))
        parameter = inspect.signature(Mesh(t).load).parameters["world_space"]
        self.assertIs(parameter.default, False)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        cmds.xform(f"{t}.vtx[*]", translation=(0, 5, 0), relative=True)
        Mesh(t).load(path)
        np.testing.assert_allclose(np.array(Mesh(t).get_points())[:, :3], local, atol=1e-6)

    def test_the_class_forms_take_a_bool(self):
        """Re-pinned (user decision 2026-10-30: world_space= without transform= freezes the shape
        at its world positions, test_class_shape_decisions): a bool, by keyword."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"))
                self.assert_refused(TypeError, rf"^{cls.__name__}\.load: world_space= takes True or False",
                                    lambda: cls.load(path, world_space=None))
                self.assertIsInstance(cls.load(path, world_space=True), cls)
                self.assertIsInstance(cls.load(path), cls)
