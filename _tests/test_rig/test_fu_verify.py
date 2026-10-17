"""Follow-up of the features run, the verification of 878ba98 (the user's decisions of
2026-10-16): four findings.

* A lookup of a child below an element (``pma.find_attr('input3D[1].input3Dx')``) walked
  its parents three times: ``find_attr``'s own walk had said "below an element" already,
  then ``_hold_if_element`` walked up to the element again, then ``_top_path``. It now
  holds the attr with its ``_top_path`` directly (one walk more, the path's). And
  ``Node.__getattr__`` holds a plug as its attr is with the table read inline (a plug
  that is no element: one dict miss, no ``_hold_like`` call).
* A deformed curve's (or surface's) refusal named ``NurbsCurve('xOrig').load(path)``,
  which the required ``world_space=`` refuses: it names ``load(path, world_space=...)``.
* ``NurbsCurve('x').load(path, True)`` said ``NurbsCurve._load_into() takes 2 positional
  arguments but 3 were given``: it names ``world_space=`` by keyword.
* Decision 3 (speed): the hot statements (``node << Float(...) << 0.5 << lock``, a plug's
  arithmetic, every journaled statement) ran a ``from ... import`` per call (1.2-1.6 us
  each under Maya's import hook); they read the modules bound once.

Each test failed at 878ba98 (negative control), except the guards.
"""

import builtins
import re
from unittest import mock

import numpy as np
from maya import cmds

from rig import Float, Node, lock
from rig.nodetypes import Mesh, NurbsCurve, NurbsSurface, _base, _files, dg_node
from rig._tests._base import MayaTestCase
from rig._tests.test_class_saveload import _Case, _cvs


class TestAChildBelowAnElement(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_find_attr_holds_it_without_the_element_walk(self):
        cmds.createNode("plusMinusAverage", name="pma")
        with mock.patch.object(dg_node, "_hold_if_element", wraps=dg_node._hold_if_element) as walk:
            attr = Node("pma").find_attr("input3D[1].input3Dx")
        walk.assert_not_called()
        top, steps = _base._held_path(attr)
        self.assertEqual([is_element for is_element, _key in steps], [True, False])
        self.assertEqual(steps[0][1], 1)
        self.assertEqual(top, Node("pma").fn_set.attribute("input3D"))

    def test_the_held_path_finds_the_plug_again(self):
        """guard: the held attr's path rebinds to its own plug, as before."""
        cmds.createNode("plusMinusAverage", name="pma")
        for name in ("input3D[1].input3Dx", "input3D[2]", "input1D[0]"):
            with self.subTest(name):
                attr = Node("pma").find_attr(name)
                top, steps = _base._held_path(attr)
                cur = _base.OpenMaya.MPlug(Node("pma").mobject, top)
                for is_element, key in steps:
                    cur = cur.elementByLogicalIndex(key) if is_element else cur.child(key)
                self.assertEqual(cur.name(), attr.get_mplug().name())

    def test_getattr_holds_without_a_call(self):
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("transform", name="a")
        with mock.patch.object(dg_node, "_hold_like", wraps=dg_node._hold_like) as like:
            str(Node("a").translateX)
            node = Node("pma")
            held = node.find_attr("input3D[1]")
            plug = getattr(node, "input3D[1]")  # the cached attr's plug
        like.assert_not_called()
        self.assertIsNone(_base._held_path(Node("a").tx))
        self.assertEqual(_base._held_path(plug), _base._held_path(held))
        self.assertIsNotNone(_base._held_path(plug))


class _Deformed(_Case):
    def deformed_curve(self):
        crv = cmds.curve(name="crv", degree=1, point=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        cmds.rename(cmds.listRelatives(crv, shapes=True)[0], "crvShape")
        cmds.xform("crv", translation=(0, 0, 5))
        NurbsCurve("crvShape").save(self.path("crv.npz"))
        cmds.cluster("crv.cv[0]", name="clu")
        return "crvShape"

    def deformed_surface(self):
        cmds.nurbsPlane(name="patch", constructionHistory=False, u=2, v=2)
        cmds.xform("patch", translation=(0, 3, 0))
        NurbsSurface("patchShape").save(self.path("patch.npz"))
        cmds.cluster("patch.cv[0][0]", name="clu")
        return "patchShape"


class TestTheDeformedHint(_Deformed):
    def test_a_curve_and_a_surface_name_the_required_space(self):
        for cls, make, file in (
            (NurbsCurve, self.deformed_curve, "crv.npz"),
            (NurbsSurface, self.deformed_surface, "patch.npz"),
        ):
            with self.subTest(cls=cls.__name__):
                shape = make()
                self.ready()
                self.assert_refused(
                    ValueError,
                    rf"is deformed \(cluHandleCluster\).*: {cls.__name__}\('{shape}Orig'\)"
                    r"\.load\(path, world_space=\.\.\.\) updates the shape under them",
                    lambda: cls(shape).load(self.path(file), world_space=True),
                )
                # the route it names works: the shape under the cluster takes the file,
                # and the deformed one follows
                cmds.xform(f"{shape}Orig.cv[1]" if cls is NurbsCurve else f"{shape}Orig.cv[1][1]",
                           translation=(0, 7, 0), relative=True)
                moved = _cvs(f"{shape}Orig")
                cls(f"{shape}Orig").load(self.path(file), world_space=True)
                self.assertFalse(np.allclose(_cvs(f"{shape}Orig"), moved))
                np.testing.assert_allclose(_cvs(shape)[1], _cvs(f"{shape}Orig")[1], atol=1e-6)
                cmds.file(new=True, force=True)

    def test_the_mesh_keeps_its_hint(self):
        """guard: a MeshData holds its space; the mesh's hint names load(path)."""
        cmds.polyCube(name="body", ch=False)
        Mesh("bodyShape").save(self.path("body.npz"))
        cmds.cluster("body.vtx[0]", name="clu")
        self.assert_refused(
            ValueError, r"Mesh\('bodyShapeOrig'\)\.load\(path\) updates the shape under them",
            lambda: Mesh("bodyShape").load(self.path("body.npz")),
        )


class TestAPositionalSpace(_Case):
    def cases(self):
        crv = cmds.curve(point=[(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1)], name="crv")
        cmds.xform(crv, translation=(5, 0, 0))
        patch = cmds.nurbsPlane(name="patch", constructionHistory=False, u=2, v=2)[0]
        return [
            (NurbsCurve, cmds.listRelatives(crv, shapes=True)[0]),
            (NurbsSurface, cmds.listRelatives(patch, shapes=True)[0]),
        ]

    def test_by_position_is_a_type_error_naming_the_keyword(self):
        for cls, shape in self.cases():
            path = cls(shape).save(self.path(f"{shape}.npz"))
            data = cls(shape).serialize()
            before = _cvs(shape)
            self.ready()
            for args, shown in (((True,), "True"), ((False,), "False"), ((True, 1), "True, 1"),
                                ((data,), f"a {type(data).__name__}")):
                with self.subTest(cls=cls.__name__, args=shown):
                    with mock.patch.object(_files, "read", wraps=_files.read) as read:
                        self.assert_refused(
                            TypeError,
                            f"^{re.escape(cls.__name__)}\\('{shape}'\\)\\.load takes the file, then "
                            f"world_space= by keyword: load\\(path, world_space=\\.\\.\\.\\), not "
                            f"load\\(path, {re.escape(shown)}\\)$",
                            lambda: cls(shape).load(path, *args),
                        )
                    read.assert_not_called()
            np.testing.assert_array_equal(_cvs(shape), before)
            self.assertFalse(cmds.undoInfo(query=True, undoName=True))
            self.assert_kept_selection()

    def test_by_keyword_still_loads(self):
        """guard: the keyword form, and the signature, as before."""
        for cls, shape in self.cases():
            with self.subTest(cls=cls.__name__):
                path = cls(shape).save(self.path(f"{shape}.npz"))
                self.assertIs(cls(shape).load(path, world_space=True).__class__, cls)
                self.assert_refused(TypeError, "needs world_space=", lambda: cls(shape).load(path))


class TestNoImportPerStatement(MayaTestCase):
    """The hot statements run no ``from ... import`` once warm."""

    TEST_START_NEW_SCENE = True

    def count_imports(self, run):
        run()  # warm: every binding made
        seen = []
        real = builtins.__import__

        def counted(name, *args, **kwargs):
            if name.startswith("rig"):
                seen.append(name)
            return real(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", counted):
            run()
        return seen

    def test_add_attr_set_and_lock(self):
        names = iter(range(100))

        def run():
            node = Node(cmds.createNode("transform"))
            node << Float(f"w{next(names)}", k=False) << 0.5 << lock
        self.assertEqual(self.count_imports(run), [])

    def test_arithmetic_and_connects(self):
        a = Node(cmds.createNode("transform", name="a"))
        b = Node(cmds.createNode("transform", name="b"))

        def run():
            b.ty << (a.tx + b.tx) * 0.5
            b.rx << a.ry / 2.0
            b.rz << a.tz - 1.0
        self.assertEqual(self.count_imports(run), [])
