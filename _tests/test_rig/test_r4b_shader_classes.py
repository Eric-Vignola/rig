"""Round 4b, step NC5: surface shaders are node classes.

* ``TestShaderNodeClasses``: the cast rule over the live ``shader/surface``
  types (the seven exact classes, flat: a class never claims a Maya subtype;
  ``Material`` for every other surface type, one classification per type;
  a texture or utility stays ``DGNode``); the reference (``Lambert('b')`` on a
  blinn is a NodeTypeError naming ``Material('b')`` and the conversion;
  ``Material('b')`` is ``Blinn("b")``); ``find_all`` / ``exists`` /
  ``is_type`` agree; a user ``CUSTOM_NODE_TYPE`` subclass of ``Blinn``.
* ``TestShaderCreate``: ``Cls.create`` always builds a network (the shader in
  ``defaultShaderList1``, ``<shader>SG``, its materialInfo), never finds, never
  changes the selection, is one undo step, stays out of a ``with container()``
  scope (never prefixed) unless ``container=True``; its refusals write nothing;
  ``Material.create(type=...)`` is the door for any surface type; ``define``
  finds or makes the network; ``Node.create`` / ``Node.define`` of a surface
  type route there; ``rn.blinn()`` stays the bare shader.

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

    def test_a_custom_node_type_subclass(self):
        class _TaggedBlinn(Blinn):
            CUSTOM_NODE_TYPE = "r4bTaggedBlinn"

        def forget():
            _base._NODE_CLASS_DICT.pop("r4bTaggedBlinn", None)
            _base._CLASS_BY_TYPE.clear()
            _base._CASTABLE_TYPES.clear()

        self.addCleanup(forget)
        _shader("blinn", "plain")
        tagged = _TaggedBlinn.create(name="tagged", color=(0, 0, 1))
        self.assertIs(type(tagged), _TaggedBlinn)
        self.assertEqual(get_custom_type("tagged"), "r4bTaggedBlinn")
        self.assertEqual(cmds.nodeType("tagged"), "blinn")
        self.assertTrue(cmds.objExists("taggedSG"))
        self.assertIs(type(Node("tagged")), _TaggedBlinn)
        self.assertIs(type(Blinn("tagged")), _TaggedBlinn)
        self.assertIs(type(Material("tagged")), _TaggedBlinn)
        self.assertTrue(_TaggedBlinn.exists("tagged"))
        self.assertFalse(_TaggedBlinn.exists("plain"))
        self.assertFalse(_TaggedBlinn.exists("lambert1"))
        self.assertEqual(_TaggedBlinn.find_all(), [tagged])
        self.assertRefused(
            NodeTypeError, r"^'plain' is a blinn, not a r4bTaggedBlinn; Node\('plain'\) is "
            r"Blinn\(\"plain\"\); Material\('plain'\) takes any surface shader$",
            lambda: _TaggedBlinn("plain"),
        )


class TestShaderCreate(_Case):
    """create / define build the network; Node.create / Node.define route."""

    def tearDown(self):
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def test_create_builds_the_network(self):
        grp = cmds.createNode("transform", name="grp")
        cmds.select(grp)
        before = _scene()
        red = Blinn.create(name="red", color=(1, 0, 0))
        self.assertIs(type(red), Blinn)
        self.assertEqual(str(red), "red")
        info = cmds.listConnections("redSG.message", type="materialInfo")
        self.assertEqual(len(info), 1)
        self.assertEqual(_scene() - before, {"red", "redSG", info[0]})
        self.assertEqual(cmds.listConnections("redSG.surfaceShader", plugs=True), ["red.outColor"])
        self.assertIn("red", _dsl1())
        self.assertEqual(cmds.getAttr("red.color"), [(1.0, 0.0, 0.0)])
        self.assertEqual(ShadingEngine.for_material(red, create=False), ShadingEngine("redSG"))
        # the selection never changes
        self.assertEqual(cmds.ls(selection=True), ["grp"])
        # create never finds: a second network
        again = Blinn.create(name="red")
        self.assertEqual(str(again), "red1")
        self.assertEqual(cmds.listConnections("red1SG.surfaceShader"), ["red1"])
        # one undo step
        cmds.undo()
        self.assertFalse(cmds.objExists("red1"))
        self.assertFalse(cmds.objExists("red1SG"))
        self.assertTrue(cmds.objExists("red"))

    def test_create_names(self):
        self.assertEqual(str(Blinn.create()), "blinn")
        self.assertEqual(str(Blinn.create()), "blinn1")
        self.assertEqual(str(Lambert.create(n="skin")), "skin")
        self.assertEqual(str(Phong.create(name="shiny", skipSelect=True)), "shiny")
        cmds.namespace(add="look")
        self.assertEqual(Blinn.create(name="look:red").name, "look:red")
        self.assertTrue(cmds.objExists("look:redSG"))
        cmds.namespace(setNamespace="look")
        self.assertEqual(Blinn.create(name="blue").name, "look:blue")

    def test_create_takes_attributes(self):
        tex = cmds.createNode("ramp", name="tex")
        red = Lambert.create(name="red", color=Node(tex).outColor, diffuse=0.5)
        self.assertEqual(cmds.listConnections("red.color", plugs=True), ["tex.outColor"])
        self.assertAlmostEqual(cmds.getAttr("red.diffuse"), 0.5)
        self.assertIs(type(red), Lambert)
        self.assertRefused(
            AttributeError, r"Blinn\.create\(\): a blinn has no attribute 'colr'.*nothing was made",
            lambda: Blinn.create(name="x", colr=(1, 0, 0)),
        )

    def test_the_refusals_write_nothing(self):
        grp = cmds.createNode("transform", name="grp")
        cases = (
            (TypeError, r"not a surface shader \(drawdb/shader/operation", lambda: Material.create(type="multiplyDivide")),
            (TypeError, r"'ramp' is not a surface shader", lambda: Material.create(type="ramp")),
            (ValueError, r"'nope' is not a registered surface shader", lambda: Material.create(type="nope")),
            (TypeError, r"^Material\.create\(\) takes type=", lambda: Material.create(name="x")),
            (TypeError, r"^Material\.create\(\): type= is a node type name", lambda: Material.create(type=Blinn)),
            (TypeError, r"^Blinn\.create\(\) takes name= as a keyword \(got 'x'\)$", lambda: Blinn.create("x")),
            (TypeError, r"^Blinn\.create\(\) takes no parent=: a blinn is a DG node$", lambda: Blinn.create(parent=grp)),
            (TypeError, r"^Blinn\.create\(\) makes a blinn, not 'phong'", lambda: Blinn.create(type="phong")),
            (TypeError, r"create always makes a new node; Blinn\.define\('s'\) finds or makes it",
             lambda: Blinn.create(name="s", shared=True)),
            (TypeError, r"Node\.define\('anisotropic', 's'\) finds or makes it",
             lambda: Material.create(type="anisotropic", name="s", shared=True)),
            (TypeError, r"takes keyword arguments only", lambda: Node.create("anisotropic", "x")),
            (TypeError, r"^Blinn\.create\(\) takes name= as a keyword", lambda: Node.create("blinn", "x")),
        )
        for error, pattern, call in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(error, pattern, call)

    def test_the_generic_door_makes_any_surface_type(self):
        ani = Material.create(type="anisotropic", name="ani", roughness=0.6)
        self.assertIs(type(ani), Material)
        self.assertEqual(cmds.nodeType("ani"), "anisotropic")
        self.assertTrue(cmds.objExists("aniSG"))
        self.assertIn("ani", _dsl1())
        self.assertAlmostEqual(cmds.getAttr("ani.roughness"), 0.6)
        # a type with a class comes back typed
        self.assertIs(type(Material.create(type="blinn", name="b")), Blinn)
        self.assertIs(type(Material.create(type="blinn", name="b2")), Blinn)
        # Blinn's own type may be spelled
        self.assertIs(type(Blinn.create(type="blinn")), Blinn)

    def test_the_network_stays_out_of_a_scope_unless_asked(self):
        with container("box") as box:
            out   = Blinn.create(name="out")
            ani   = Node.create("anisotropic", name="ani")
            node  = Node.create("blinn", name="nb")
            asked = Lambert.create(name="asked", container=True)
            with container("inner"):
                flat  = Blinn.create(name="flat")
                flat2 = Material.create(type="rampShader", name="flat2", container=True)
        names = ("out", "ani", "nb", "asked", "flat", "flat2")
        self.assertEqual([str(n) for n in (out, ani, node, asked, flat, flat2)], list(names))
        for name in names:
            self.assertIn(name, _dsl1())
            self.assertTrue(cmds.objExists(f"{name}SG"))
        infos = {name: cmds.listConnections(f"{name}SG.message", type="materialInfo")[0] for name in ("asked", "flat2")}
        # only the networks asked for, whole (never prefixed); the geometry never
        self.assertEqual(
            _members(str(box)),
            sorted(["asked", "askedSG", infos["asked"], "flat2", "flat2SG", infos["flat2"]]),
        )
        self.assertFalse(cmds.ls("inner_*"))

    def test_define_finds_or_makes_the_network(self):
        red = Blinn.define("red", color=(1, 0, 0))
        self.assertIs(type(red), Blinn)
        self.assertTrue(cmds.objExists("redSG"))
        cmds.setAttr("red.color", 0, 1, 0, type="double3")
        before = _scene()
        self.assertEqual(Blinn.define("red", color=(1, 0, 0)), red)
        self.assertEqual(cmds.getAttr("red.color"), [(0.0, 1.0, 0.0)])
        self.assertEqual(Blinn.define("red", color=(0, 0, 1), update=True), red)
        self.assertEqual(cmds.getAttr("red.color"), [(0.0, 0.0, 1.0)])
        self.assertEqual(_scene(), before)
        # the other spellings of the same define
        self.assertEqual(Node.define("blinn", "red"), red)
        self.assertEqual(Material.define("red", type="blinn"), red)
        # another type: refused, naming the conversion
        self.assertRefused(
            NodeTypeError,
            r"^'red' is a blinn, not a phong; Node\('red'\) is Blinn\(\"red\"\); Material\('red'\) "
            r"takes any surface shader; Blinn\('red'\)\.astype\(Phong\) converts it$",
            lambda: Phong.define("red"),
        )
        self.assertRefused(
            NodeTypeError, r"^'red' is a blinn, not an anisotropic",
            lambda: Material.define("red", type="anisotropic"),
        )
        self.assertRefused(TypeError, r"^Material\.define\('x'\) takes type=", lambda: Material.define("x"))
        self.assertRefused(TypeError, r"takes no parent=", lambda: Blinn.define("x", parent="red"))
        self.assertRefused(AttributeError, r"a blinn has no attribute 'colr'", lambda: Blinn.define("red", colr=1))
        self.assertRefused(
            TypeError, r"not a surface shader", lambda: Material.define("x", type="multiplyDivide")
        )
        # a surface type without a class, through both doors
        ani = Material.define("ani", type="anisotropic", roughness=0.25)
        self.assertIs(type(ani), Material)
        self.assertTrue(cmds.objExists("aniSG"))
        before = _scene()
        self.assertEqual(Node.define("anisotropic", "ani"), ani)
        self.assertEqual(Material.define("ani", type="anisotropic"), ani)
        self.assertEqual(_scene(), before)
        # a found bare shader is returned as it is: no engine is built for it
        bare = _shader("blinn", "bare")
        before = _scene()
        self.assertEqual(str(Blinn.define(bare)), "bare")
        self.assertEqual(_scene(), before)
        # a define in a flattened scope is never prefixed
        with container("outer"):
            with container("inner"):
                keyed = Lambert.define("keyed")
        self.assertEqual(str(keyed), "keyed")

    def test_node_create_routes_a_surface_type(self):
        blinn = Node.create("blinn", name="nb", color=(1, 0, 0))
        self.assertIs(type(blinn), Blinn)
        self.assertTrue(cmds.objExists("nbSG"))
        ani = Node.create("anisotropic", name="na")
        self.assertIs(type(ani), Material)
        self.assertEqual(cmds.nodeType("na"), "anisotropic")
        self.assertTrue(cmds.objExists("naSG"))
        # rn.<shader>() is the bare shader, made by the scope's createNode
        with container("box") as box:
            bare = rn.blinn(name="bare")
        self.assertIs(type(bare), Blinn)
        self.assertFalse(cmds.objExists("bareSG"))
        self.assertNotIn("bare", _dsl1())
        self.assertEqual(_members(str(box)), ["bare"])
        # another unregistered type is unchanged: createNode, in the scope
        with container("box2") as box2:
            md = Node.create("multiplyDivide", name="md", operation="divide")
        self.assertIs(type(md), DGNode)
        self.assertEqual(_members(str(box2)), ["md"])
