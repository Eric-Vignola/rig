"""Follow-up of the features run (the user's decision 2 of 2026-10-16): the instance forms
``NurbsCurve('x').load(...)`` and ``NurbsSurface('x').load(...)`` REQUIRE ``world_space=``.

A BSplineData / BSplinePatchData holds no space: a file saved with ``world_space=False``
and loaded with the old default (True) put its object-space CVs in the world, moved by the
transform, silently. A load without ``world_space=``, or with anything but a bool, is now a
TypeError naming the argument, raised before the file is read (nothing read, nothing
changed, no undo step). ``Mesh('x').load`` keeps its default (a MeshData holds its matrix),
and the class forms (creating) still take no ``world_space=``.

Each test failed at fc692b9 (negative control: the default True moved the CVs), except the
guards.
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


class TestWorldSpaceIsRequired(_SpaceCase):
    def test_no_world_space_is_a_type_error_before_the_file_is_read(self):
        for cls, shape, data in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"), world_space=False)
                cmds.xform(f"{shape}.cv[0]" if cls is NurbsCurve else f"{shape}.cv[0][0]",
                           translation=(0, 4, 0), relative=True)
                before = _cvs(shape)
                self.ready()
                with mock.patch.object(_files, "read", wraps=_files.read) as read:
                    self.assert_refused(
                        TypeError,
                        f"^{re.escape(cls.__name__)}\\('{shape}'\\).load needs world_space=: a {data} holds "
                        f"no space, so give the one the file was saved in \\(world_space=True for save\\(\\)'s "
                        f"default, world_space=False for save\\(world_space=False\\)\\); the other one moves the CVs$",
                        lambda: cls(shape).load(path),
                    )
                read.assert_not_called()
                np.testing.assert_array_equal(_cvs(shape), before)
                self.assertFalse(cmds.undoInfo(query=True, undoName=True))
                self.assert_kept_selection()

    def test_anything_but_a_bool_is_a_type_error(self):
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

    def test_the_signature_names_it_required(self):
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                parameter = inspect.signature(cls(shape).load).parameters["world_space"]
                self.assertIs(parameter.default, _files.REQUIRED)
                self.assertEqual(repr(parameter.default), "<required>")
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIn("REQUIRED", cls.load.__doc__)

    def test_create_keywords_are_refused_first(self):
        """guard: name= on the instance form is still its own TypeError."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"))
                self.assert_refused(TypeError, r"updates it in place: it takes the file and world_space= \(got name=\)",
                                    lambda: cls(shape).load(path, name="x"))


class TestEachSpaceLoads(_SpaceCase):
    def test_an_object_space_file_round_trips_with_false(self):
        """guard: the route the refusal names, one rig.<Cls>.load step."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                local = _cvs(shape, world_space=False)
                world = _cvs(shape)
                path = cls(shape).save(self.path(f"{shape}_local.npz"), world_space=False)
                cmds.xform(f"{shape}.cv[*]" if cls is NurbsCurve else f"{shape}.cv[*][*]",
                           translation=(0, 2, 0), relative=True)
                self.ready()
                self.assertIs(cls(shape).load(path, world_space=False).__class__, cls)
                np.testing.assert_allclose(_cvs(shape, world_space=False), local, atol=1e-9)
                np.testing.assert_allclose(_cvs(shape), world, atol=1e-9)
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), f"rig.{cls.__name__}.load")
                self.assert_kept_selection()

    def test_a_world_space_file_round_trips_with_true(self):
        """guard: save()'s default with world_space=True."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                world = _cvs(shape)
                path = cls(shape).save(self.path(f"{shape}_world.npz"))
                cmds.xform(f"{shape}.cv[*]" if cls is NurbsCurve else f"{shape}.cv[*][*]",
                           translation=(1, 0, 0), relative=True)
                cls(shape).load(path, world_space=True)
                np.testing.assert_allclose(_cvs(shape), world, atol=1e-9)

    def test_the_data_object_too(self):
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                data = cls(shape).serialize(world_space=False)
                self.assert_refused(TypeError, "needs world_space=", lambda: cls(shape).load(data))
                cls(shape).load(data, world_space=np.bool_(False))


class TestTheOtherFormsKeepTheirRules(_SpaceCase):
    def test_the_mesh_keeps_its_default(self):
        """guard: a MeshData holds its matrix; Mesh('x').load(path) is world space."""
        t = cmds.polyCube(name="body", constructionHistory=False)[0]
        cmds.xform(t, translation=(1, 2, 3))
        path = Mesh(t).save(self.path("body.npz"))
        parameter = inspect.signature(Mesh(t).load).parameters["world_space"]
        self.assertIs(parameter.default, True)
        Mesh(t).load(path)

    def test_the_class_forms_take_none(self):
        """guard: creating takes no world_space=."""
        for cls, shape, _data in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"))
                self.assert_refused(TypeError, rf"^{cls.__name__}.load\(...\) creates a new ",
                                    lambda: cls.load(path, world_space=True))
                self.assertIsInstance(cls.load(path), cls)
