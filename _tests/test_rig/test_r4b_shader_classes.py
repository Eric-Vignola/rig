"""Round 4b, step NC5: surface shaders are node classes.

* ``TestShaderNodeClasses``: the cast rule over the live ``shader/surface``
  types (the seven exact classes, flat: a class never claims a Maya subtype;
  ``Material`` for every other surface type, one classification per type;
  a texture or utility stays ``DGNode``); the reference (``Lambert('b')`` on a
  blinn is a NodeTypeError naming ``Material('b')`` and the conversion;
  ``Material('b')`` is ``Blinn("b")``); ``find_all`` / ``exists`` /
  ``is_type`` agree.

Every refusal writes nothing (a zero ``cmds.ls()`` delta).
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig.nodetypes._base as _base
import rig.nodetypes.material_node as _material_node
from rig import container, Node, NodeTypeError, NodeNotFoundError
from rig.bridges import nodes as rn
from rig.nodetypes import (
    Blinn,
    DGNode,
    DisplayLayer,
    Lambert,
    Material,
    OpenPBRSurface,
    Phong,
    PhongE,
    ShadingEngine,
    StandardSurface,
    SurfaceShader,
    Transform,
)
from rig.nodetypes._base import get_custom_type
from rig._tests._base import MayaTestCase


_EXACT = {
    "lambert": Lambert,
    "blinn": Blinn,
    "phong": Phong,
    "phongE": PhongE,
    "surfaceShader": SurfaceShader,
    "standardSurface": StandardSurface,
    "openPBRSurface": OpenPBRSurface,
}

_CLASSES = (Material, *_EXACT.values())


def _scene():
    return set(cmds.ls())


def _shader(node_type, name):
    return cmds.shadingNode(node_type, asShader=True, name=name, skipSelect=True)


def _dsl1():
    return cmds.listConnections("defaultShaderList1.shaders", source=True, destination=False) or []


def _members(name):
    return sorted(cmds.container(name, query=True, nodeList=True) or [])


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assertRefused(self, error, pattern, call):
        """``call`` raises ``error`` matching ``pattern`` and writes nothing."""
        before = _scene()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(_scene() - before, set())
        self.assertEqual(before - _scene(), set())


class TestShaderNodeClasses(_Case):
    """The classes, the cast rule, the reference, find_all / exists / is_type."""

    def test_the_cast_table_over_the_live_surface_types(self):
        surface = cmds.listNodeTypes("shader/surface") or []
        self.assertLessEqual(set(_EXACT), set(surface))
        for node_type in surface:
            with self.subTest(node_type=node_type):
                name = cmds.createNode(node_type, name=f"n_{node_type}", skipSelect=True)
                node = Node(name)
                self.assertIs(type(node), _EXACT.get(node_type, Material))
                self.assertIsInstance(node, Material)
                self.assertIsInstance(node, DGNode)
        # the other shading nodes stay DGNode: a texture, a utility, the info
        for node_type in ("ramp", "multiplyDivide", "materialInfo"):
            with self.subTest(node_type=node_type):
                self.assertIs(type(Node(cmds.createNode(node_type, skipSelect=True))), DGNode)
        # Maya's defaults
        self.assertIs(type(Node("lambert1")), Lambert)
        self.assertIs(type(Node("standardSurface1")), StandardSurface)
        self.assertIs(type(Node("initialShadingGroup")), ShadingEngine)

    def test_the_classes_are_flat_and_exact(self):
        for cls in _EXACT.values():
            self.assertIs(cls.__base__, Material)
        _shader("blinn", "b")
        _shader("anisotropic", "ani")
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("transform", name="grp")
        # a blinn is no Lambert, although Maya derives it from lambert
        self.assertRefused(
            NodeTypeError,
            r"^'b' is a blinn, not a lambert; Node\('b'\) is Blinn\(\"b\"\); Material\('b'\) takes "
            r"any surface shader; Blinn\('b'\)\.astype\(Lambert\) converts it$",
            lambda: Lambert("b"),
        )
        self.assertRefused(
            NodeTypeError,
            r"Material\('ani'\)\.astype\(Lambert\) converts it$",
            lambda: Lambert("ani"),
        )
        # the generic reference takes any surface shader, most derived
        self.assertIs(type(Material("b")), Blinn)
        self.assertEqual(Material("b"), Node("b"))
        self.assertIs(type(Material("ani")), Material)
        held = Node("b")
        self.assertIs(Material(held), held)
        self.assertIs(Blinn(held), held)
        # not a shader: no conversion hint
        self.assertRefused(
            NodeTypeError,
            r"^'md' is a multiplyDivide, not a surface shader; Node\('md'\) is DGNode\(\"md\"\)$",
            lambda: Material("md"),
        )
        self.assertRefused(
            NodeTypeError,
            r"^'grp' is a transform, not a blinn; Node\('grp'\) is Transform\(\"grp\"\)$",
            lambda: Blinn("grp"),
        )
        self.assertRefused(NodeNotFoundError, r"^no blinn named 'nosuch'", lambda: Blinn("nosuch"))
        self.assertRefused(NodeNotFoundError, r"^no surface shader named 'x'", lambda: Material("x"))
        self.assertRefused(TypeError, "names no node", lambda: Blinn())
        self.assertRefused(TypeError, "takes no attributes", lambda: Blinn("b", color=(1, 0, 0)))
        # the classes' own constructor asserts the exact type (ValueError, the DAG precedent)
        with self.assertRaisesRegex(ValueError, "b is not a lambert"):
            Lambert._wrap("b")

    def test_find_all_exists_and_is_type_agree(self):
        for node_type in ("lambert", "blinn", "phong", "anisotropic", "rampShader", "surfaceShader"):
            _shader(node_type, f"x_{node_type}")
        cmds.createNode("ramp", name="tex")
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("transform", name="grp")
        names = sorted(cmds.ls())
        for cls in _CLASSES:
            listed = {str(node) for node in cls.find_all()}
            for node in cls.find_all():
                self.assertIsInstance(node, cls)
            for name in names:
                with self.subTest(cls=cls.__name__, name=name):
                    self.assertEqual(cls.exists(name), name in listed)
                    self.assertEqual(bool(cls.is_type(name)), name in listed)
        self.assertEqual(
            sorted(str(n) for n in Material.find_all()),
            sorted(["lambert1", "standardSurface1", *(f"x_{t}" for t in (
                "lambert", "blinn", "phong", "anisotropic", "rampShader", "surfaceShader"))]),
        )
        # C19: Maya's lambert subtypes, each typed by its own class
        loose = {str(n): type(n) for n in Lambert.find_all(exact_type=False)}
        self.assertEqual(
            loose,
            {"lambert1": Lambert, "x_lambert": Lambert, "x_blinn": Blinn, "x_phong": Phong,
             "x_anisotropic": Material},
        )
        self.assertFalse(isinstance(Node("x_blinn"), Lambert))
        # Node.find_all of a type
        self.assertEqual(Node.find_all("blinn"), [Blinn("x_blinn")])
        self.assertEqual(Node.find_all("anisotropic"), [Material("x_anisotropic")])

    def test_one_classification_per_type(self):
        for i in range(3):
            _shader("anisotropic", f"ani{i}")
            _shader("rampShader", f"rs{i}")
        _material_node._IS_SURFACE.clear()
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        with mock.patch.object(cmds, "getClassification", wraps=cmds.getClassification) as classify:
            for _ in range(2):
                for i in range(3):
                    self.assertIs(type(Node(f"ani{i}")), Material)
                    self.assertIs(type(Node(f"rs{i}")), Material)
            self.assertEqual(classify.call_count, 2)
            Node.create("anisotropic", name="made")
            Node.create("multiplyDivide", name="md1")
            Node.create("multiplyDivide", name="md2")
            classify.reset_mock()
            Node.create("multiplyDivide", name="md3")
            Material("ani0")
            self.assertEqual(classify.call_count, 0)

    def test_a_type_maya_does_not_know_is_not_remembered(self):
        _material_node._IS_SURFACE.clear()
        self.assertIsNone(_material_node._classify("rigNoSuchShaderType"))
        self.assertNotIn("rigNoSuchShaderType", _material_node._IS_SURFACE)
        self.assertIs(_material_node._classify("blinn"), Material)
        self.assertIsNone(_material_node._classify("multiplyDivide"))
        self.assertEqual(
            {k: _material_node._IS_SURFACE[k] for k in ("blinn", "multiplyDivide")},
            {"blinn": True, "multiplyDivide": False},
        )
