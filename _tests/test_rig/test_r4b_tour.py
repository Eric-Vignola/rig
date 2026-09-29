"""Round 4b, step NC9: one name per thing, and the construction rule end to end.

* ``TestNames``: over the public names of ``rig``, ``rig.shade``,
  ``rig.membership``, ``rig.spec`` and ``rig.nodetypes``, no name is bound both
  to a node class and to a declaration (an attribute or membership spec, a
  class or an instance), none of those names is a ``cgmath.geometry`` public
  name, and a name two modules export is one object (``Layer`` is
  ``DisplayLayer`` in both, a shader class is one class). The geometry data
  specs are ``MeshAttr``, ``NurbsCurveAttr`` and ``NurbsSurfaceAttr``, with no
  alias: ``rig.nodetypes.Mesh`` is the node class, ``rig.MeshAttr`` the
  attribute spec and ``cgmath.geometry.MeshData`` the data.
"""

import importlib
import inspect

import cgmath.geometry
from maya import cmds

import rig
from rig import MeshAttr, Node, NodeNotFoundError, NurbsCurveAttr, NurbsSurfaceAttr
from rig.nodetypes import Mesh, NurbsCurve, NurbsSurface
from rig.spec._base import _AttrSpec, _spec_from_attribute
from rig._internal.members import _MemberSpec
from rig._tests._base import MayaTestCase


_MODULES = ("rig", "rig.shade", "rig.membership", "rig.spec", "rig.nodetypes")


def _scene():
    return set(cmds.ls())


def _public(module):
    """A module's public names: its ``__all__``, else its names without a
    leading underscore."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    return names


def _is_node_class(obj):
    return inspect.isclass(obj) and issubclass(obj, Node)


def _is_declaration(obj):
    kinds = (_AttrSpec, _MemberSpec)
    return (inspect.isclass(obj) and issubclass(obj, kinds)) or isinstance(obj, kinds)


def _bindings():
    """``{name: [(module name, object), ...]}`` over the public names of the
    five modules."""
    bound = {}
    for module_name in _MODULES:
        module = importlib.import_module(module_name)
        for name in _public(module):
            bound.setdefault(name, []).append((module_name, getattr(module, name)))
    return bound


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assertRefused(self, error, pattern, call):
        """``call`` raises ``error`` matching ``pattern`` and writes nothing."""
        before = _scene()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(_scene() - before, set())
        self.assertEqual(before - _scene(), set())


class TestNames(_Case):
    """REC change 10 (CC-10): no public name is both a node class and a
    declaration, nor a ``cgmath.geometry`` name."""

    def test_no_name_is_both_a_node_class_and_a_declaration(self):
        both = {}
        for name, where in _bindings().items():
            nodes = [m for m, obj in where if _is_node_class(obj)]
            decls = [m for m, obj in where if _is_declaration(obj)]
            if nodes and decls:
                both[name] = {"node class": nodes, "declaration": decls}
        self.assertEqual(both, {})

    def test_a_name_two_modules_export_is_one_object(self):
        split = {
            name: [m for m, _ in where]
            for name, where in _bindings().items()
            if len({id(obj) for _, obj in where}) > 1
        }
        self.assertEqual(split, {})
        # the documented doubles
        self.assertIs(rig.Layer, rig.nodetypes.DisplayLayer)
        self.assertIs(rig.membership.Layer, rig.nodetypes.DisplayLayer)
        for name in ("Material", "Lambert", "Blinn", "Phong", "PhongE", "SurfaceShader",
                     "StandardSurface", "OpenPBRSurface"):
            with self.subTest(name=name):
                self.assertIs(getattr(rig.shade, name), getattr(rig.nodetypes, name))
        self.assertIs(rig.Tag, rig.membership.Tag)
        self.assertIs(rig.MeshAttr, rig.spec.MeshAttr)

    def test_no_node_class_or_declaration_takes_a_cgmath_geometry_name(self):
        geometry = {name for name in dir(cgmath.geometry) if not name.startswith("_")}
        self.assertIn("MeshData", geometry)
        taken = {
            name: [m for m, _ in where]
            for name, where in _bindings().items()
            if name in geometry and any(_is_node_class(obj) or _is_declaration(obj) for _, obj in where)
        }
        self.assertEqual(taken, {})
        # a cgmath.geometry class rig exports is that class
        for name, where in _bindings().items():
            data = getattr(cgmath.geometry, name, None)
            if inspect.isclass(data):
                for module_name, obj in where:
                    with self.subTest(name=name, module=module_name):
                        self.assertIs(obj, data)

    def test_the_data_specs_are_renamed_with_no_alias(self):
        for old, new in (("Mesh", "MeshAttr"), ("NurbsCurve", "NurbsCurveAttr"),
                         ("NurbsSurface", "NurbsSurfaceAttr")):
            with self.subTest(old=old):
                self.assertFalse(hasattr(rig, old))
                self.assertFalse(hasattr(rig.spec, old))
                self.assertNotIn(old, rig.__all__)
                self.assertNotIn(old, rig.spec.__all__)
                with self.assertRaises(ImportError):
                    exec(f"from rig import {old}", {})
                self.assertIn(new, rig.__all__)
                self.assertIn(new, rig.spec.__all__)
                self.assertIs(getattr(rig, new), getattr(rig.spec, new))
                self.assertTrue(issubclass(getattr(rig, new), _AttrSpec))
                self.assertTrue(_is_node_class(getattr(rig.nodetypes, old)))
        self.assertEqual(len(rig.spec.__all__), 23)

    def test_three_things_three_names(self):
        """``Mesh`` the shape node, ``MeshData`` its data, ``MeshAttr`` a
        ``mesh`` data attribute."""
        cmds.polyCube(name="cube", ch=False)
        shape = Mesh("cube")
        self.assertEqual(repr(shape), 'Mesh("cubeShape")')
        data = shape.serialize(include_uvs=False)
        self.assertIsInstance(data, cgmath.geometry.MeshData)
        copy = Mesh.create(data, name="copy")
        self.assertIsInstance(copy, Mesh)
        self.assertEqual(cmds.polyEvaluate(str(copy), vertex=True), 8)
        ctrl = Node.create("transform", name="ctrl")
        ctrl << MeshAttr("shapeIn")
        ctrl.shapeIn << shape.outMesh
        self.assertEqual(cmds.getAttr("ctrl.shapeIn", type=True), "mesh")
        self.assertEqual(cmds.listConnections("ctrl.shapeIn", source=True, plugs=True), ["cubeShape.outMesh"])

    def test_the_data_specs_add_their_data_types(self):
        ctrl = Node.create("transform", name="ctrl")
        ctrl << MeshAttr("meshIn") << NurbsCurveAttr("crvIn") << NurbsSurfaceAttr("srfIn")
        self.assertEqual(
            [cmds.getAttr(f"ctrl.{a}", type=True) for a in ("meshIn", "crvIn", "srfIn")],
            ["mesh", "nurbsCurve", "nurbsSurface"],
        )
        for spec, data_type in ((MeshAttr, "mesh"), (NurbsCurveAttr, "nurbsCurve"),
                                (NurbsSurfaceAttr, "nurbsSurface")):
            with self.subTest(spec=spec.__name__):
                self.assertEqual(spec("x", at="double").kargs,
                                 {"longName": "x", "dataType": data_type, "keyable": True})

    def test_a_clone_of_a_data_attribute_is_its_attr_spec(self):
        ctrl = Node.create("transform", name="ctrl")
        ctrl << MeshAttr("meshIn") << NurbsCurveAttr("crvIn") << NurbsSurfaceAttr("srfIn")
        other = Node.create("transform", name="other")
        for attr, spec, data_type in (("meshIn", MeshAttr, "mesh"), ("crvIn", NurbsCurveAttr, "nurbsCurve"),
                                      ("srfIn", NurbsSurfaceAttr, "nurbsSurface")):
            with self.subTest(attr=attr):
                self.assertIs(type(_spec_from_attribute(getattr(ctrl, attr), attr + "2", False)), spec)
                getattr(ctrl, attr) >> other
                self.assertEqual(cmds.getAttr(f"other.{attr}", type=True), data_type)

    def test_the_old_spelling_is_a_reference_and_writes_nothing(self):
        """``ctrl << Mesh("shapeIn")`` names a mesh node now: the strict
        reference raises before ``<<`` runs, so old declaration code is caught."""
        ctrl = Node.create("transform", name="ctrl")
        self.assertRefused(NodeNotFoundError, r"^no mesh named 'shapeIn'$", lambda: ctrl << Mesh("shapeIn"))
        self.assertRefused(NodeNotFoundError, r"^no nurbsCurve named 'c'$", lambda: ctrl << NurbsCurve("c"))
        self.assertRefused(NodeNotFoundError, r"^no nurbsSurface named 's'$", lambda: ctrl << NurbsSurface("s"))
        self.assertRefused(TypeError, r"^Mesh\('shapeIn', \.\.\.\)", lambda: Mesh("shapeIn", dataType="mesh"))
        self.assertFalse(cmds.attributeQuery("shapeIn", node="ctrl", exists=True))
