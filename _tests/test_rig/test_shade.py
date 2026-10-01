"""Tests for ``rig.shade`` -- materials through the membership grammar.

Round 4b (NC7): a material is a node. ``Blinn("red")`` refers to a blinn
that exists (NodeNotFoundError before any write when it does not),
``Blinn.define("red", ...)`` finds it or makes its network and
``Blinn.create(name=...)`` always makes a new one; the shader node on the
right of ``<<`` assigns, ``-node`` removes, ``Material()`` / ``Blinn()`` is
the kind token, ``ShadingEngine("xSG")`` names exactly that engine and
``Default()`` is the ``initialShadingGroup`` node.

Every error test asserts a zero ``cmds.ls()`` delta: a refused spelling
writes nothing. Lists are never compared with ``assertEqual``.
"""

import os
import shutil
import tempfile
import time
from unittest import mock

import numpy as np
from maya import cmds
import rig.nodetypes as nodetypes
from rig import Components, container, List, Node, NodeNotFoundError, NodeTypeError, shade, Tag
from rig.bridges import nodes as rn
from rig.nodetypes import ShadingEngine
from rig.shade import (
    Blinn,
    Default,
    Lambert,
    Material,
    OpenPBRSurface,
    Phong,
    PhongE,
    StandardSurface,
    SurfaceShader,
)
from rig._tests._base import MayaTestCase


ISG = "initialShadingGroup"
DEFAULT = 'ShadingEngine("initialShadingGroup")'


def _shape(node):
    """Full path of the first shape under a transform ``Node`` / name."""
    return cmds.listRelatives(str(node), shapes=True, fullPath=True)[0]


def _members(engine):
    """Membership as ``cmds.sets`` prints it."""
    return cmds.sets(str(engine), query=True) or []


def _engines(shape):
    return sorted(set(cmds.listConnections(shape, type="shadingEngine") or []))


def _names(nodes):
    return [repr(node) for node in nodes]


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


# --------------------------------------------------------------------- #
#  Construction: tokens and references write nothing
# --------------------------------------------------------------------- #


class TestMaterialConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construction_makes_no_maya_calls(self):
        """Historical id (v2.0.0a2): pinned that a lazy material spec made zero
        Maya calls; it now pins that a reference and the tokens write nothing
        (round 4b NC7: a material is a node; Blinn("x") refers, never creates)."""
        red    = Blinn.define("red", color=(1, 0, 0), diffuse=0.5)
        before = set(cmds.ls())
        values = [Blinn("red"), Material("red"), Material(), Lambert(), Default(), -red, -Default()]
        with self.assertRaises(NodeNotFoundError):
            Blinn("nope")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(values[0], red)
        self.assertEqual(values[1], red)
        self.assertEqual(str(values[0]),  "red")
        self.assertEqual(repr(values[0]), 'Blinn("red")')
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        self.assertAlmostEqual(cmds.getAttr("red.diffuse"), 0.5)
        self.assertTrue(values[2].purges)
        self.assertEqual(repr(values[2]), "Material()")
        self.assertEqual(repr(values[3]), "Lambert()")
        self.assertEqual(repr(values[4]), DEFAULT)
        self.assertEqual(str(values[4]),  ISG)
        self.assertEqual(repr(values[5]), '-Blinn("red")')
        self.assertTrue(values[5].removes)
        self.assertEqual(repr(values[6]), f"-{DEFAULT}")
        for cls, node_type in (
            (Lambert, "lambert"), (Blinn, "blinn"), (Phong, "phong"),
            (PhongE, "phongE"), (SurfaceShader, "surfaceShader"),
            (StandardSurface, "standardSurface"), (OpenPBRSurface, "openPBRSurface"),
        ):
            self.assertIs(cls, getattr(nodetypes, cls.__name__))
            self.assertEqual(cls.NATIVE_NODE_TYPE, node_type)
            self.assertEqual(repr(cls()), f"{cls.__name__}()")
        self.assertIs(Material, nodetypes.Material)
        self.assertEqual(set(cmds.ls()), before)

    def test_an_empty_call_is_the_purge(self):
        # Material() / Blinn() are the kind token (Material(None) is refused);
        # Default() is initialShadingGroup's node
        for purge in (Material(), Blinn(), Lambert()):
            self.assertTrue(purge.purges)
            self.assertIsNone(purge.name)
        self.assertEqual(repr(Blinn()), "Blinn()")
        self.assertEqual(str(Default()), ISG)
        self.assertIsInstance(Default(), ShadingEngine)
        self.assertEqual(Default(), ShadingEngine(ISG))
        for call in (lambda: Material(None), lambda: Blinn(None), lambda: Lambert(None)):
            with self.assertRaisesRegex(TypeError, r"^None is not a material name; Material\(\) removes all$"):
                call()

    def test_rejected_constructions(self):
        red    = Blinn.define("red")
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Blinn("")
        with self.assertRaises((TypeError, ValueError)):
            Blinn(5)
        with self.assertRaisesRegex(TypeError, "takes no attributes"):
            Blinn("x", type="phong")
        with self.assertRaisesRegex(TypeError, r"makes a blinn, not 'phong'"):
            Blinn.create(name="x", type="phong")
        with self.assertRaisesRegex(TypeError, "takes no name"):
            Default("x")
        with self.assertRaisesRegex(TypeError, "takes no name"):
            Default(None)
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~red
        with self.assertRaises(TypeError):
            ~Material()
        with self.assertRaisesRegex(TypeError, "double negative"):
            -Blinn()
        with self.assertRaisesRegex(TypeError, "Material\\(\\) removes all"):
            -Blinn(None)
        with self.assertRaises(TypeError):
            -(-red)
        with self.assertRaisesRegex(NodeTypeError, r"Lambert\('lambert1'\)\.astype\(Phong\) converts it"):
            Phong(Node("lambert1"))
        self.assertEqual(set(cmds.ls()), before)
        # the same type twice is not a contradiction
        self.assertIs(type(Blinn.define("x", type="blinn")), Blinn)

    def test_negation_keeps_the_options(self):
        """Historical id (v2.0.0a2): pinned -spec keeping the spec's options
        (attrs, unique, update, container); it now pins that -node is the
        removal token of that node, which it reads at each use (round 4b NC7:
        the node is the whole state)."""
        red     = Blinn.define("red")
        removal = -red
        self.assertTrue(removal.removes)
        self.assertEqual(removal.name, "red")
        self.assertEqual(repr(removal), '-Blinn("red")')
        red.rename("scarlet")
        self.assertEqual(repr(removal), '-Blinn("scarlet")')
        engine = -red.engine
        self.assertTrue(engine.removes)
        self.assertEqual(repr(engine), '-ShadingEngine("scarletSG")')

    def test_methods_refuse_removal_and_purge_copies(self):
        """Historical id (v2.0.0a2): pinned the spec methods refusing a removal
        or purge copy; it now pins that a token is no material: its methods and
        plugs are the node's (AttributeError naming it), nothing written."""
        red    = Blinn.define("x")
        before = set(cmds.ls())
        for token, name in (
            (-red, "delete"), (-red, "rename"), (-red, "color"), (Material(), "delete"),
            (Material(), "rename"), (Blinn(), "engine"), (Material(), "node"),
        ):
            with self.subTest(token=repr(token), name=name):
                with self.assertRaisesRegex(AttributeError, "not a material"):
                    getattr(token, name)
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  Define / create: the network, kwargs, fresh networks
# --------------------------------------------------------------------- #


class TestMaterialCreate(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def test_create_wiring_and_selection(self):
        cmds.select(str(self.cube))
        selection = cmds.ls(selection=True)
        red       = Blinn.define("red", color=(1, 0, 0))
        result    = self.cube << red
        self.assertIs(result, self.cube)
        self.assertEqual(cmds.ls(selection=True), selection)
        self.assertEqual(cmds.nodeType("red"),    "blinn")
        self.assertEqual(cmds.nodeType("redSG"),  "shadingEngine")
        partition = cmds.listConnections("redSG.partition", plugs=True) or []
        self.assertTrue(any(x.startswith("renderPartition.sets") for x in partition))
        self.assertEqual(len(cmds.listConnections("redSG.message", type="materialInfo")), 1)
        self.assertTrue(cmds.listConnections("redSG.message", type="lightLinker"))
        self.assertEqual(
            cmds.listConnections("redSG.surfaceShader", source=True, destination=False),
            ["red"],
        )
        shaders = cmds.listConnections("defaultShaderList1.shaders", source=True, destination=False)
        self.assertEqual(shaders.count("red"), 1)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertEqual(_members(ISG), [])
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))

    def test_spec_state_is_unchanged_by_use(self):
        """Historical id (v2.0.0a2): pinned a spec's state unchanged by its use;
        it now pins the node's (round 4b NC7): the same name, class, uuid and
        network after three assignments."""
        red   = Blinn.define("red", color=(1, 0, 0))
        state = (red.name, type(red), repr(red), red.uuid, str(red.engine))
        self.cube       << red
        self.cube       << red
        self.cube.f[:2] << red
        self.assertEqual((red.name, type(red), repr(red), red.uuid, str(red.engine)), state)
        self.assertEqual(cmds.ls("red*", type="blinn"), ["red"])

    def test_found_material_is_a_plain_assignment(self):
        self.cube << Blinn.define("red", color=(1, 0, 0))
        other  = _cube("other")
        before = set(cmds.ls())
        result = other << Blinn.define("red", color=(0, 0, 1))
        self.assertIs(result, other)
        self.assertEqual(set(cmds.ls()), before)
        # set on a found material too: the script is the source of truth
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 0.0, 1.0))
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "otherShape"])
        # update=False leaves it as it is
        other << Blinn.define("red", color=(1, 0, 0), update=False)
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 0.0, 1.0))
        self.assertEqual(set(cmds.ls()), before)

    def test_plug_kwarg_connects_and_spec_kwarg_applies(self):
        from rig.spec import lock

        texture = rn.file(name="tex")
        self.cube << Blinn.define("skin", color=texture.outColor, diffuse=0.25)
        self.assertEqual(
            cmds.listConnections("skin.color", source=True, destination=False, plugs=True),
            ["tex.outColor"],
        )
        self.assertAlmostEqual(cmds.getAttr("skin.diffuse"), 0.25)
        # a spec kwarg applies to the attribute
        self.cube << Blinn.define("locked", diffuse=lock)
        self.assertTrue(cmds.getAttr("locked.diffuse", lock=True))

    def test_kwarg_typo_raises_before_any_write(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "no attribute 'colour'"):
            self.cube << Blinn.define("red", colour=(1, 0, 0))
        self.assertEqual(set(cmds.ls()), before)
        self.cube << Blinn.define("red")
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            self.cube << Blinn.define("red", colour=(1, 0, 0), update=True)
        with self.assertRaises(AttributeError):
            Blinn.create(name="red", colour=(1, 0, 0))
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_unique_builds_a_fresh_network_per_lshift(self):
        """Historical id (v2.0.0a2): pinned unique=True building a fresh network
        per '<<'; it now pins Lambert.create, which always makes a new network
        (round 4b NC7: the unique=True job is a verb)."""
        other = _cube("other")
        first = Lambert.create(name="plane", diffuse=0)
        self.cube << first
        second = Lambert.create(name="plane", diffuse=0)
        other << second
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(cmds.ls("plane*", type="lambert")), ["plane", "plane1"])
        self.assertEqual(sorted(cmds.ls("plane*", type="shadingEngine")), ["plane1SG", "planeSG"])
        self.assertEqual(str(second.engine), "plane1SG")
        self.assertEqual(cmds.getAttr("plane1.diffuse"), 0)
        self.assertEqual(_members("planeSG"), ["cubeShape"])
        self.assertEqual(_members("plane1SG"), ["otherShape"])
        # one list application is one network
        List([self.cube, other]) << Lambert.create(name="both")
        self.assertEqual(cmds.ls("both*", type="lambert"), ["both"])
        self.assertEqual(sorted(_members("bothSG")), ["cubeShape", "otherShape"])

    def test_namespace_idempotency(self):
        cmds.namespace(add="look")
        cmds.namespace(set="look")
        try:
            self.cube << Blinn.define("red")
            before = set(cmds.ls())
            self.cube << Blinn.define("red")
            self.assertEqual(set(cmds.ls()),           before)
            self.assertEqual(str(Blinn("red")),        "look:red")
            self.assertEqual(str(Blinn("red").engine), "look:redSG")
        finally:
            cmds.namespace(set=":")
        self.assertEqual(cmds.ls(type="blinn"), ["look:red"])
        self.assertEqual(str(Blinn("look:red")), "look:red")
        # from the root namespace the short name is a different material
        with self.assertRaisesRegex(NodeNotFoundError, "'look:red' exists"):
            Blinn("red")
        self.cube << Blinn.define("red")
        self.assertEqual(sorted(cmds.ls(type="blinn")), ["look:red", "red"])

    def test_wrap_an_existing_shader_a_bare_one_and_an_engine(self):
        """Historical id (v2.0.0a2): pinned the spec wrapping an existing shader,
        a bare one and an engine; it now pins the shader and engine nodes
        (round 4b NC7): lambert1 builds lambert1SG (ADD C12), a bare shader gets
        <shader>SG, an engine node is exactly that engine."""
        # any existing surface shader; a class asserts its exact type
        result = self.cube << Material("lambert1")
        self.assertIs(result, self.cube)
        self.assertEqual(_members("lambert1SG"), ["cubeShape"])
        self.assertEqual(str(Lambert("lambert1")), "lambert1")
        before = set(cmds.ls())
        with self.assertRaisesRegex(NodeTypeError, r"is a lambert, not a blinn.*Material\('lambert1'\)"):
            self.cube << Blinn("lambert1")
        self.assertEqual(set(cmds.ls()), before)
        # a bare shader: the adopt path builds its engine and the shader list link
        bare = rn.blinn(name="bare")
        self.assertIsNone(cmds.listConnections("bare", type="shadingEngine"))
        self.cube << Material("bare")
        self.assertEqual(_members("bareSG"), ["cubeShape"])
        self.assertEqual(str(Blinn("bare").engine), "bareSG")
        shaders = cmds.listConnections("defaultShaderList1.shaders", source=True, destination=False)
        self.assertEqual(shaders.count("bare"), 1)
        # an engine is no shader; its node names exactly that engine
        with self.assertRaisesRegex(NodeTypeError, "is a shadingEngine, not a surface shader"):
            Material("bareSG")
        self.assertEqual(ShadingEngine("bareSG").get_material(), bare)
        self.cube << Default()
        self.cube << ShadingEngine("bareSG")
        self.assertEqual(_members("bareSG"), ["cubeShape"])
        self.assertEqual(Default().get_material(), Node("standardSurface1"))
        self.assertEqual(Default(), Material("standardSurface1").engine)
        # an engine node is exactly that engine, a shaderless one too (round 4b
        # FIX); the particle engine is refused
        empty  = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="emptySG")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "particle"):
            self.cube << ShadingEngine("initialParticleSE")
        self.assertEqual(set(cmds.ls()), before)
        self.cube << ShadingEngine(empty)
        self.assertEqual(_members("emptySG"), ["cubeShape"])
        self.assertEqual(Material.of(self.cube), [ShadingEngine("emptySG")])
        self.cube << Default()
        self.assertEqual(bare, Material("bare"))

    def test_warns_when_maya_keeps_another_name(self):
        """Historical id (v2.0.0a2): pinned the warning when Maya kept another
        name than the spec's; it now pins that Blinn.define refuses it before
        any write (round 4b CC-4, the rename predicted: 'shared' is a Maya
        namespace), that Blinn.create takes Maya's name, and define's name
        checks before any call."""
        # 'shared' is a Maya namespace: shadingNode(name='shared') makes 'shared1'
        before = set(cmds.ls())
        with mock.patch.object(cmds, "warning") as warning:
            with self.assertRaisesRegex(ValueError, "'shared' is the name of a namespace; Maya would rename"):
                self.cube << Blinn.define("shared")
        warning.assert_not_called()
        self.assertEqual(set(cmds.ls()), before)
        with mock.patch.object(cmds, "warning") as warning:
            made = Blinn.create(name="shared")
        warning.assert_not_called()
        self.assertEqual(str(made), "shared1")
        # names Maya could not keep are refused before any call
        before = set(cmds.ls())
        for bad in ("my-mat", "bad name", "1st", "a.b", "", None.__class__, "|grp|mat"):
            with self.assertRaises((TypeError, ValueError)):
                Blinn.define(bad)
        self.assertEqual(set(cmds.ls()), before)
        for good in ("ok", "_x", "mat2"):
            self.assertEqual(str(Blinn.define(good)), good)

    def test_build_creates_without_a_target(self):
        """Historical id (v2.0.0a2): pinned spec.build(); it now pins
        Blinn.define making the network without any assignment, and finding it
        again (round 4b NC7)."""
        lib = Blinn.define("lib", color=(0, 1, 0))
        self.assertIsInstance(lib, Blinn)
        self.assertEqual(str(lib), "lib")
        self.assertEqual(_members("libSG"), [])
        self.assertEqual(cmds.getAttr("lib.color")[0], (0.0, 1.0, 0.0))
        self.assertEqual(_members(ISG), ["cubeShape"])
        # a second define finds it
        before = set(cmds.ls())
        self.assertEqual(Blinn.define("lib"), lib)
        self.assertEqual(set(cmds.ls()), before)
        with self.assertRaisesRegex(TypeError, "takes type="):
            Material.define("untyped")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(Default().get_material()), "standardSurface1")


# --------------------------------------------------------------------- #
#  The handle: a reference, then the node itself
# --------------------------------------------------------------------- #


class TestMaterialHandle(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube = _cube("cube")

    def test_handle_is_find_only(self):
        """Historical id (v2.0.0a2): pinned the spec as a find-only handle
        (node, engine and forwarded plugs raising until built); it now pins the
        reference (round 4b NC7): Blinn("red") raises before red exists and
        writes nothing, and once made the node is the handle, its plugs native."""
        before = set(cmds.ls())
        with self.assertRaisesRegex(NodeNotFoundError, "no blinn named 'red'"):
            Blinn("red")
        with self.assertRaises(ValueError):
            self.cube << Blinn("red")
        with self.assertRaises(ValueError):
            self.cube >> Blinn("red")
        self.assertEqual(set(cmds.ls()), before)
        red = Blinn.define("red")
        self.cube << red
        self.assertEqual(str(red), "red")
        self.assertEqual(str(red.engine), "redSG")
        plug = red.color << (0, 1, 0)
        self.assertEqual(str(plug), "red.color")
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 1.0, 0.0))
        red.color = (0, 0, 1)
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 0.0, 1.0))
        self.assertEqual(red.diffuse >> None, cmds.getAttr("red.diffuse"))
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            red.colour
        with self.assertRaises(AttributeError):
            red.colour = 1
        self.assertEqual(set(cmds.ls()), before)
        # a bare material has no engine yet
        rn.blinn(name="bare")
        with self.assertRaisesRegex(ValueError, "feeds no shading engine"):
            Blinn("bare").engine

    def test_ambiguous_names_refuse(self):
        cmds.polyCube(name="dup")
        cmds.rename(cmds.group(cmds.polyCube(name="dup")[0], name="grp") and "|grp|dup1", "dup")
        self.assertEqual(len(cmds.ls("dup")), 2)
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.cube << Material("dup")
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  Assign
# --------------------------------------------------------------------- #


class TestMaterialAssign(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)
        self.red   = Blinn.define("red")
        self.blue  = Blinn.define("blue")

    def test_faces_carve_and_return_the_components(self):
        self.cube << self.red
        faces  = self.cube.f[:3]
        result = faces << self.blue
        self.assertIs(result, faces)
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])
        self.assertEqual(_members("blueSG"), ["cube.f[0:2]"])
        self.cube.f[[4]] << self.blue
        self.assertEqual(_members("blueSG"), ["cube.f[0:2]", "cube.f[4]"])
        self.assertEqual(_members("redSG"), ["cube.f[3]", "cube.f[5]"])
        # a Components carrier and a string-built one work the same
        Components(self.cube, "f", [5]) << self.blue
        Components("cube.f[3]")         << self.blue
        self.assertEqual(_members("blueSG"), ["cube.f[0:5]"])
        self.assertEqual(_members("redSG"), [])

    def test_bare_f_handle_means_the_whole_object(self):
        result = self.cube.f << self.red
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertIsInstance(result, Components)
        self.cube.f[:] << self.blue
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        self.assertEqual(_members("redSG"), [])

    def test_faces_into_owner_is_noop(self):
        self.cube << self.red
        before = set(cmds.ls())
        result = self.cube.f[:2] << self.red
        self.assertIs(type(result), Components)
        self.assertEqual(set(cmds.ls()),          before)
        self.assertEqual(_members("redSG"),       ["cubeShape"])
        self.assertEqual(cmds.ls(type="groupId"), [])

    def test_node_lhs_assigns_its_own_shapes_never_the_subtree(self):
        grp    = Node(cmds.group(str(self.cube), name="grp"))
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"no shadeable shape.*cubeShape.*recurse"):
            grp << self.red
        with self.assertRaisesRegex(TypeError, "no shadeable shape"):
            Node.create("joint", name="joint1") << self.red
        self.assertEqual(set(cmds.ls()) - {"joint1"}, before)
        self.assertEqual(_engines(_shape(self.cube)), [ISG])
        # a transform with two mesh shapes assigns both
        other = _cube("other")
        cmds.parent(_shape(other), str(self.cube), shape=True, relative=True)
        cmds.delete(str(other))
        self.cube << self.red
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "otherShape"])
        # the shape itself works too
        Node(_shape(self.cube)) << self.blue
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        self.assertEqual(_members("redSG"), ["otherShape"])

    def test_surfaces_wear_materials_curves_and_lattices_do_not(self):
        srf = Node(cmds.sphere(name="srf")[0])
        self.assertIs(srf << self.red, srf)
        self.assertEqual(_members("redSG"), ["srfShape"])
        self.assertEqual(_names(Material.of(srf)), ['Blinn("red")'])
        self.assertIn(srf, self.red)
        crv    = Node(cmds.circle(name="crv")[0])
        lat    = Node(cmds.lattice(str(self.cube))[1])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "nurbsCurve, not a shadeable"):
            crv << self.red
        with self.assertRaisesRegex(TypeError, "lattice, not a shadeable"):
            lat << self.red
        with self.assertRaisesRegex(TypeError, r"without faces.*Material\.of"):
            srf >> self.red
        # a query answers by contents (round 4b FIX): a curve wears no material
        self.assertNotIn(crv, self.red)
        self.assertEqual(set(cmds.ls()), before)

    def test_a_plug_on_a_transform_with_a_control_curve_skips_it_too(self):
        """Historical id (v2.0.0a2): pinned an attribute plug of a transform with
        a control curve standing for the transform (the curve skipped); it now
        pins that plug refused on '<<', '>>' and of before any write (user
        decision Q4 option A: the node is the member), and the transform on the
        left skipping the curve ('in' reads the plug as the transform)."""
        crv = cmds.circle(name="crv", ch=False)[0]
        cmds.parent(_shape(crv), str(self.cube), shape=True, relative=True)
        cmds.delete(crv)
        before = set(cmds.ls())
        for call in (
            lambda: self.cube.tx << self.red,
            lambda: self.cube.t >> self.red,
            lambda: Material.of(self.cube.tx),
            lambda: self.cube.ty >> Material(),
            lambda: self.cube.sx << -self.red,
        ):
            with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                call()
        self.assertEqual(set(cmds.ls()), before)
        self.assertIs(self.cube << self.red, self.cube)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        np.testing.assert_array_equal(self.cube >> self.red, np.arange(6))
        self.assertEqual(_names(Material.of(self.cube)), ['Blinn("red")'])
        self.assertEqual(_names(self.cube >> Material()), ['Blinn("red")'])
        self.assertIn(self.cube.tx, self.red)
        self.cube << -self.red
        self.assertEqual(_members("redSG"), [])

    def test_transform_skips_its_control_curve_shape(self):
        crv = cmds.circle(name="crv", ch=False)[0]
        cmds.parent(_shape(crv), str(self.cube), shape=True, relative=True)
        cmds.delete(crv)
        self.assertEqual(
            sorted(cmds.listRelatives(str(self.cube), shapes=True)), ["crvShape", "cubeShape"]
        )
        result = self.cube << self.red
        self.assertIs(result, self.cube)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertEqual(_names(Material.of(self.cube)), ['Blinn("red")'])
        self.assertEqual([str(x) for x in shade.materials(self.cube)], ["red"])
        np.testing.assert_array_equal(self.cube >> self.red, np.arange(6))
        self.cube << -self.red
        self.assertEqual(_members("redSG"), [])
        # the curve shape named itself, and a transform with nothing
        # shadeable, keep raising
        lone   = Node(cmds.circle(name="lone", ch=False)[0])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "nurbsCurve, not a shadeable"):
            Node("|cube|crvShape") << self.red
        with self.assertRaisesRegex(TypeError, "nurbsCurve, not a shadeable"):
            lone << self.red
        with self.assertRaisesRegex(TypeError, "nurbsCurve, not a shadeable"):
            List([self.cube, lone]) << self.red
        # a query answers by contents (round 4b FIX): a curve wears none
        self.assertEqual(Material.of(lone), [])
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_engines(self.shape), [])

    def test_pluglist_lhs_is_one_material_one_engine_one_call(self):
        other = _cube("other")
        lhs   = List([self.cube, other.f[:2]])
        with mock.patch.object(cmds, "sets", wraps=cmds.sets) as sets:
            result = lhs << self.red
        self.assertIs(result, lhs)
        writes = [c for c in sets.call_args_list if c.kwargs.get("forceElement") == "redSG"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "other.f[0:1]"])
        self.assertEqual(sorted(cmds.ls(type="blinn")), ["blue", "red"])
        # the same node twice and its faces: the whole object wins
        List([self.cube, self.cube.f[:2]]) << self.blue
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        # a mixed list fails as a whole, nothing written
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "vertices"):
            List([other, self.cube.vtx[0]]) << self.red
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["other.f[0:1]"])

    def test_pluglist_broadcast_pairs_each_selection_with_its_spec(self):
        lhs    = List([self.cube.f[:2], self.cube.f[2:4]])
        result = lhs << [self.red, self.blue]
        self.assertIs(result, lhs)
        self.assertEqual(_members("redSG"), ["cube.f[0:1]"])
        self.assertEqual(_members("blueSG"), ["cube.f[2:3]"])
        answers = lhs >> [self.red, self.blue]
        np.testing.assert_array_equal(answers[0], [0, 1])
        np.testing.assert_array_equal(answers[1], [2, 3])

    def test_chain_returns_the_lhs_across_kinds(self):
        faces  = self.cube.f[:3]
        result = faces << Tag("lid") << self.red << self.blue
        self.assertIs(result, faces)
        np.testing.assert_array_equal(self.cube >> Tag("lid"), [0, 1, 2])
        self.assertEqual(_members("blueSG"), ["cube.f[0:2]"])
        self.assertEqual(_members("redSG"), [])
        self.assertIs(self.cube << self.red << Default(), self.cube)
        self.assertEqual(_members(ISG), ["cubeShape"])

    def test_an_attribute_plug_stands_for_its_node(self):
        """Historical id (v2.0.0a2): pinned an attribute plug standing for its
        node on '<<', '>>' and of; it now pins it refused there before any
        write (user decision Q4 option A: the node is the member), while 'in'
        and shade.materials still read a plug as its node and component plugs
        stay members."""
        before = set(cmds.ls())
        joint  = Node.create("joint", name="joint1")
        for label, call in (
            ("cube.tx << red",                  lambda: self.cube.tx << self.red),
            ("cube.t >> red",                   lambda: self.cube.t >> self.red),
            ("cube.rotate >> Blinn()",          lambda: self.cube.rotate >> Blinn()),
            ("Material.of(cube.visibility)",    lambda: Material.of(self.cube.visibility)),
            ("List([cube.tx, cube.ty]) << blue", lambda: List([self.cube.tx, self.cube.ty]) << self.blue),
            ("cube.sx << -blue",                lambda: self.cube.sx << -self.blue),
            ("cube.v << Default()",             lambda: self.cube.v << Default()),
            ("cube.v >> redSG",                 lambda: self.cube.v >> self.red.engine),
            ("shape.castsShadows << red",       lambda: Node(self.shape).castsShadows << self.red),
            ("joint.tx << red",                 lambda: joint.tx << self.red),
            ("List([cube.tx, cube.vtx[0]]) << blue", lambda: List([self.cube.tx, self.cube.vtx[0]]) << self.blue),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                    call()
        self.assertEqual(set(cmds.ls()) - {"joint1"}, before)
        self.assertEqual(_engines(self.shape), [ISG])
        # the node is the member; 'in' reads a plug as its node
        self.cube << self.red
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertIn(self.cube.tx, self.red)
        self.assertEqual([str(x) for x in shade.materials(self.cube.tx)], ["red"])
        # component plugs keep their meaning
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx[:3] << self.red
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cubeShape"])


# --------------------------------------------------------------------- #
#  Remove, purge, revert
# --------------------------------------------------------------------- #


class TestMaterialRemove(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)
        self.red   = Blinn.define("red")
        self.cube << self.red

    def test_faces_carve_a_whole_shape_engine(self):
        faces  = self.cube.f[:3]
        result = faces << -self.red
        self.assertIs(result, faces)
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])
        self.assertEqual(_engines(self.shape), ["redSG"])
        np.testing.assert_array_equal(self.cube >> self.red, [3, 4, 5])
        self.assertEqual(_names(Material.of(self.cube.f[0])), [])
        # faces in no engine: repair re-homes exactly those
        fixed = shade.repair(self.cube)
        self.assertEqual([str(x) for x in fixed], ["cubeShape"])
        self.assertEqual(_members(ISG),           ["cube.f[0:2]"])
        self.assertEqual(_members("redSG"),       ["cube.f[3:5]"])

    def test_faces_leave_a_per_face_membership(self):
        blue = Blinn.define("blue")
        self.cube.f[:3]     << blue
        self.cube.f[[0, 5]] << -blue
        self.assertEqual(_members("blueSG"), ["cube.f[1:2]"])
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])
        # removing non-members asserts a state that already holds
        before = set(cmds.ls())
        self.cube.f[:3] << -self.red
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])

    def test_whole_node_leaves_the_engine(self):
        result = self.cube << -self.red
        self.assertIs(result, self.cube)
        self.assertEqual(_members("redSG"), [])
        self.assertEqual(_engines(self.shape), [])
        self.cube.f[:3] << self.red
        self.cube       << -self.red
        self.assertEqual(_members("redSG"), [])
        self.cube.f << self.red
        self.cube.f << -self.red
        self.assertEqual(_members("redSG"), [])

    def test_missing_material_is_a_value_error(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no blinn named 'nope'"):
            self.cube.f[:2] << -Blinn("nope")
        with self.assertRaises(ValueError):
            self.cube << -Material("nope")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        # a bare material with no engine holds nothing: nothing to remove
        rn.blinn(name="bare")
        self.cube << -Blinn("bare")
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_purge_then_default_round_trip(self):
        self.cube.f[:3] << Blinn.define("blue")
        result = self.cube << Material()
        self.assertIs(result, self.cube)
        self.assertEqual(_engines(self.shape), [])
        self.assertEqual(_members("redSG"), [])
        self.assertEqual(_members("blueSG"), [])
        self.assertEqual(_names(Material.of(self.cube)), [])
        self.assertEqual(shade.bindings(self.cube), [])
        result = self.cube << Default()
        self.assertIs(result, self.cube)
        self.assertEqual(_members(ISG), ["cubeShape"])
        self.assertEqual(_names(Material.of(self.cube)), [DEFAULT])
        np.testing.assert_array_equal(self.cube.f[:6] >> Default(), np.arange(6))
        # any class's kind token purges; faces purge per face
        self.cube.f[:3] << self.red
        self.cube.f[:2] << Blinn()
        self.assertEqual(_members("redSG"), ["cube.f[2]"])
        self.assertEqual(_members(ISG), ["cube.f[3:5]"])
        self.cube << -Default()
        self.assertEqual(_engines(self.shape), ["redSG"])
        self.assertEqual(_members("redSG"), ["cube.f[2]"])

    def test_remainder_link_faces_are_recomputed_never_trusted(self):
        # Maya's own carve (a raw cmds.sets of faces into another engine)
        # leaves a compInstObjGroups link through which the owner claims
        # every face no other engine lists; once read while a face is
        # green, the link claims every face for good. rig computes the
        # remainder from the other engines' face groups instead.
        self.cube << Default()
        Blinn.define("a")
        link = f"{self.shape}.compInstObjGroups[0].compObjectGroups[0]"
        cmds.sets("cube.f[0:2]", edit=True, forceElement="aSG")
        self.assertEqual(cmds.listConnections(link), [ISG])
        cmds.sets("cube.f[3]", edit=True, remove=ISG)
        _members(ISG)
        cmds.sets("cube.f[3]", edit=True, forceElement=ISG)
        self.assertEqual(_members(ISG), ["cube.f[3:5]"])
        faces  = self.cube.f[[3]]
        result = faces << -Default()
        self.assertIs(result, faces)
        self.assertEqual(_members(ISG), ["cube.f[4:5]"])
        self.assertEqual(_members("aSG"), ["cube.f[0:2]"])
        self.assertEqual(_names(Material.of(self.cube.f[3])), [])
        self.assertIsNone(cmds.listConnections(link))
        np.testing.assert_array_equal(self.cube >> Default(), [4, 5])
        # the green face re-added to the OTHER engine before the removal:
        # nothing of that engine is stolen either
        self.cube << Default()
        cmds.sets("cube.f[0:2]", edit=True, forceElement="aSG")
        cmds.sets("cube.f[3]", edit=True, remove=ISG)
        _members(ISG)
        cmds.sets("cube.f[3]", edit=True, forceElement="aSG")
        self.cube.f[[3]] << -Default()
        self.assertEqual(_members(ISG), ["cube.f[4:5]"])
        self.assertEqual(_members("aSG"), ["cube.f[0:3]"])
        self.assertEqual(_names(Material.of(self.cube.f[3])), ['Blinn("a")'])

    def test_instanced_whole_object_faces_are_refused(self):
        # cmds.sets -remove of faces on an instanced path whose engine
        # holds the whole object there releases nothing: refused before
        # any write, on either path.
        self.cube << Default()
        inst   = Node(cmds.instance(str(self.cube))[0])
        before = set(cmds.ls())
        for lhs, spec in (
            (self.cube.f[:2], -Default()),
            (inst.f[:2],      -Default()),
            (inst.f[:2],      Material()),
        ):
            with self.assertRaisesRegex(TypeError, r"instance.*f\[keep\].*de-instance"):
                lhs << spec
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(sorted(_members(ISG)), ["cube1|cubeShape", "cube|cubeShape"])
        # the fix the error names: faces assigned explicitly are per
        # instance, and release
        inst.f[:2] << self.red
        self.assertEqual(_members("redSG"), ["cube1.f[0:1]"])
        self.assertEqual(sorted(_members(ISG)), ["cube1.f[2:5]", "cube|cubeShape"])
        inst.f[[2]] << -Default()
        self.assertEqual(sorted(_members(ISG)),          ["cube1.f[3:5]", "cube|cubeShape"])
        self.assertEqual(_names(Material.of(inst.f[2])), [])
        self.assertEqual(_names(Material.of(self.cube)), [DEFAULT])
        np.testing.assert_array_equal(inst >> Default(), [3, 4, 5])


# --------------------------------------------------------------------- #
#  Query and enumeration
# --------------------------------------------------------------------- #


class TestMaterialQuery(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def test_face_ids(self):
        got = self.cube >> Default()
        self.assertIsInstance(got, np.ndarray)
        self.assertNotIsInstance(got, Components)
        np.testing.assert_array_equal(got, np.arange(6))
        red = Blinn.define("red")
        self.cube.f[:3] << red
        np.testing.assert_array_equal(self.cube >> Blinn("red"),           [0, 1, 2])
        np.testing.assert_array_equal(self.cube >> Material("red"),        [0, 1, 2])
        np.testing.assert_array_equal(self.cube >> red.engine,             [0, 1, 2])
        np.testing.assert_array_equal(self.cube.f >> red,                  [0, 1, 2])
        np.testing.assert_array_equal(self.cube.f[1:5] >> red,             [1, 2])
        np.testing.assert_array_equal(self.cube.f[[5, 0]] >> red,          [0])
        self.assertEqual((self.cube.f[3:] >> red).shape, (0,))
        np.testing.assert_array_equal(self.cube >> Default(), [3, 4, 5])
        # an engine the shape is not in: empty, never None
        blue = Blinn.define("blue")
        _cube("other") << blue
        self.assertEqual((self.cube >> blue).shape, (0,))
        self.assertEqual((self.cube.f[:2] >> blue).shape, (0,))
        # a bare material: no engine, empty
        rn.blinn(name="bare")
        self.assertEqual((self.cube >> Blinn("bare")).shape, (0,))
        # the ids go back through the handle
        self.cube.f[self.cube >> red] << blue
        self.assertEqual(sorted(_members("blueSG")), ["cube.f[0:2]", "otherShape"])

    def test_query_refusals(self):
        self.cube.f[:3] << Blinn.define("red")
        other  = _cube("other")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no blinn named 'nope'"):
            self.cube >> Blinn("nope")
        with self.assertRaisesRegex(TypeError, r"is a blinn, not a lambert.*Material\('red'\)"):
            self.cube >> Lambert("red")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            List([self.cube, other]) >> Blinn("red")
        with self.assertRaisesRegex(TypeError, "Split them"):
            List([self.cube, self.cube.f[:2]]) >> Blinn("red")
        with self.assertRaises(TypeError):
            self.cube >> -Blinn("red")
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx[:2] >> Blinn("red")
        with self.assertRaises(TypeError):
            Node("lambert1") >> Blinn("red")
        self.assertEqual(set(cmds.ls()), before)
        # the same node twice is one node
        np.testing.assert_array_equal(List([self.cube, self.cube]) >> Blinn("red"), [0, 1, 2])

    def test_rshift_purge_enumerates_like_of(self):
        self.cube.f[:3] << Blinn.define("red")
        self.cube.f[3]  << Lambert.define("skin")
        for lhs in (
            self.cube, self.cube.f, self.cube.f[0], self.cube.f[[0, 3]],
            self.cube.f[4:],
        ):
            for cls in (Material, Blinn, Lambert):
                got = lhs >> cls()
                self.assertIsInstance(got, list)
                self.assertEqual(_names(got), _names(cls.of(lhs)))
        # re-pinned (round 4b NC6): an attribute plug on the left is refused
        # (the node is the member), on '>>' and of alike
        for cls in (Material, Blinn, Lambert):
            with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                self.cube.tx >> cls()
            with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                cls.of(self.cube.tx)
        self.assertEqual(
            _names(self.cube >> Material()), [DEFAULT, 'Blinn("red")', 'Lambert("skin")']
        )
        # re-pinned (round 4b NC7): Material(None) is refused; Material() is the token
        with self.assertRaisesRegex(TypeError, r"Material\(\) removes all"):
            self.cube >> Material(None)
        self.assertEqual(_names(self.cube >> Blinn()),          ['Blinn("red")'])
        self.assertEqual(_names(self.cube.f[2] >> Material()),  ['Blinn("red")'])
        self.assertEqual(_names(self.cube.f[4:] >> Material()), [DEFAULT])
        self.assertIs(type((self.cube >> Blinn())[0]), Blinn)
        # Default() is not a purge: it queries initialShadingGroup's faces
        np.testing.assert_array_equal(self.cube >> Default(), [4, 5])
        # re-injectable
        self.cube.f[5] << (self.cube >> Blinn())[0]
        np.testing.assert_array_equal(self.cube >> Blinn("red"), [0, 1, 2, 5])
        other  = _cube("other")
        before = set(cmds.ls())
        # an enumeration answers by contents (round 4b FIX): a vertex wears none
        self.assertEqual(self.cube.vtx[0] >> Material(), [])
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            List([self.cube, other]) >> Material()
        self.assertEqual(set(cmds.ls()), before)

    def test_of_types_by_the_live_node_type(self):
        """Historical id (v2.0.0a2): pinned Material.of typing its specs by the
        live node type; it now pins the live nodes it answers (round 4b NC7):
        each typed by its own class, Default() for initialShadingGroup."""
        self.assertEqual(_names(Material.of(self.cube)), [DEFAULT])
        self.assertEqual(Material.of(self.cube)[0], Default())
        self.cube.f[:3] << Blinn.define("red")
        self.cube.f[3]  << Material.define("ani", type="anisotropic")
        found = Material.of(self.cube)
        self.assertEqual(_names(found), [DEFAULT, 'Blinn("red")', 'Material("ani")'])
        self.assertIs(type(found[1]), Blinn)
        self.assertIs(type(found[2]), Material)
        self.assertEqual(_names(Material.of(self.cube.f[0])),      ['Blinn("red")'])
        self.assertEqual(_names(Material.of(self.cube.f[[0, 3]])), [])
        self.assertEqual(_names(Material.of(self.cube.f[4:])),     [DEFAULT])
        # the bare handle is the whole object
        self.assertEqual(_names(Material.of(self.cube.f)), _names(found))
        self.assertEqual(_names(Blinn.of(self.cube)), ['Blinn("red")'])
        self.assertEqual(_names(Lambert.of(self.cube)), [])
        self.assertIn(self.cube.f[4:], Default())
        self.assertNotIn(self.cube.f[0], Default())
        self.assertEqual(_names(StandardSurface.of(self.cube.f[5])), ['StandardSurface("standardSurface1")'])
        # re-injectable
        self.cube.f[5] << found[1]
        np.testing.assert_array_equal(self.cube >> Blinn("red"), [0, 1, 2, 5])
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Material.of(List([self.cube, _cube("other")]))
        # a query answers by contents (round 4b FIX): a vertex wears none
        self.assertEqual(Material.of(self.cube.vtx[0]), [])
        self.assertEqual(set(cmds.ls()) - {"other", "otherShape"}, before)

    def test_module_readers(self):
        self.cube.f[:3] << Blinn.define("red")
        self.cube.f[3]  << Blinn.define("blue")
        self.assertEqual(
            [str(x) for x in shade.materials(self.cube)],
            ["standardSurface1", "red", "blue"],
        )
        self.assertIsInstance(shade.materials(self.cube), List)
        self.assertEqual([str(x) for x in shade.materials(self.cube.f[:2])], ["red"])
        self.assertEqual([str(x) for x in shade.materials(self.cube.f[[0, 3]])], ["red", "blue"])
        found = shade.bindings(self.cube)
        self.assertEqual(
            sorted((str(m), f.indices.tolist()) for m, f in found),
            [("blue", [3]), ("red", [0, 1, 2]), ("standardSurface1", [4, 5])],
        )
        self.assertTrue(all(isinstance(f, Components) for _, f in found))
        self.assertEqual(
            [(str(m), f.indices.tolist()) for m, f in shade.bindings(self.cube.f[2:4])],
            [("red", [2]), ("blue", [3])],
        )
        # object level is the whole kind
        self.cube << Blinn("red")
        [(material, faces)] = shade.bindings(self.cube)
        self.assertEqual(material, Blinn("red"))
        self.assertTrue(faces.is_all)
        self.assertEqual(faces.names, [f"{self.shape}.f[*]"])
        self.assertEqual(shade.bindings(_cube("other").f[:2])[0][1].indices.tolist(), [0, 1])
        with self.assertRaisesRegex(TypeError, "per face"):
            shade.bindings(Node(cmds.sphere(name="srf")[0]))


# --------------------------------------------------------------------- #
#  Methods: delete, rename, repair, tidy
# --------------------------------------------------------------------- #


class TestMaterialMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def test_delete_leaves_members_green_and_repair_fixes(self):
        other = _cube("other")
        red   = Blinn.define("red")
        self.cube   << red
        other.f[:2] << red
        info = cmds.listConnections("redSG.message", type="materialInfo")
        self.assertIsNone(Material("red").delete())
        for name in ("red", "redSG", *info):
            self.assertFalse(cmds.objExists(name))
        self.assertEqual(_engines(self.shape), [])
        self.assertEqual(_names(Material.of(self.cube)), [])
        fixed = shade.repair()
        self.assertIsInstance(fixed, List)
        self.assertEqual(sorted(str(x) for x in fixed),  ["cubeShape", "otherShape"])
        self.assertEqual(_names(Material.of(self.cube)), [DEFAULT])
        self.assertEqual(sorted(_members(ISG)),          ["cubeShape", "other.f[0:5]"])
        # nothing left to repair
        self.assertEqual([str(x) for x in shade.repair()], [])

    def test_delete_refuses_default_and_referenced_nodes(self):
        self.cube << Default()
        before = set(cmds.ls())
        # a Maya default refuses with a TypeError, the material as the engine
        # does (round 4b FIX: the material's was a RuntimeError)
        with self.assertRaisesRegex(TypeError, "default node"):
            Material("lambert1").delete()
        with self.assertRaisesRegex(TypeError, "default node"):
            Material("standardSurface1").delete()
        with self.assertRaisesRegex(TypeError, "cannot be deleted"):
            Default().delete()
        with self.assertRaisesRegex(TypeError, "cannot be renamed"):
            Default().rename("x")
        with self.assertRaisesRegex(TypeError, "default node"):
            Material("lambert1").rename("x")
        with self.assertRaises(ValueError):
            Material("nope").delete()
        self.assertEqual(set(cmds.ls()), before)
        # a referenced material
        folder = tempfile.mkdtemp(prefix="rig_shade_ref_")
        path   = os.path.join(folder, "rig_shade_ref.ma")
        try:
            cmds.file(new=True, force=True)
            cmds.shadingNode("blinn", asShader=True, name="refmat")
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            before = set(cmds.ls())
            with self.assertRaisesRegex(RuntimeError, "referenced"):
                Material("ref:refmat").delete()
            with self.assertRaisesRegex(RuntimeError, "referenced"):
                Material("ref:refmat").rename("x")
            self.assertEqual(set(cmds.ls()), before)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_repair_judges_each_instance_path(self):
        inst = Node(cmds.instance(str(self.cube))[0])
        x    = Blinn.define("x")
        inst      << x
        self.cube << Material()
        self.assertEqual(_members(ISG), [])
        self.assertEqual([str(x) for x in shade.repair()], ["cube|cubeShape"])
        self.assertEqual(_members(ISG), ["cube|cubeShape"])
        self.assertEqual(_members("xSG"), ["cube1|cubeShape"])
        # the other way round: the scene walk reaches every instance path
        self.cube << x
        inst      << Material()
        self.assertEqual(_members("xSG"), ["cube|cubeShape"])
        self.assertEqual([str(x) for x in shade.repair()], ["cube1|cubeShape"])
        self.assertEqual(_members(ISG), ["cube1|cubeShape"])
        self.assertEqual(_members("xSG"), ["cube|cubeShape"])
        self.assertEqual([str(x) for x in shade.repair()], [])

    def test_rename_refuses_a_taken_engine_name(self):
        red = Blinn.define("red")
        self.cube << red
        cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="blueSG")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "'blueSG' already exists"):
            red.rename("blue")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.ls(type="blinn"), ["red"])
        self.assertEqual(str(red), "red")
        self.assertEqual(str(red.engine), "redSG")
        self.assertEqual(_members("redSG"), ["cubeShape"])
        # an engine off the convention is not renamed, so the name is free
        cmds.rename("redSG", "customSG")
        red.rename("blue")
        self.assertEqual(cmds.ls(type="blinn"), ["blue"])
        self.assertEqual(str(red.engine),       "customSG")
        self.assertEqual(cmds.ls("blueSG*"),    ["blueSG"])

    def test_rename_follows_the_engine_convention_and_the_spec(self):
        """Historical id (v2.0.0a2): pinned the spec following its rename; it
        now pins the node following it (round 4b NC7), with its engine on the
        <shader>SG convention."""
        red = Blinn.define("red")
        self.cube << red
        self.assertIsNone(red.rename("blue"))
        self.assertEqual(cmds.ls(type="blinn"), ["blue"])
        self.assertTrue(cmds.objExists("blueSG"))
        self.assertFalse(cmds.objExists("redSG"))
        self.assertEqual(str(red),           "blue")
        self.assertEqual(red,                Blinn("blue"))
        self.assertEqual(str(red.engine),    "blueSG")
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        # an engine off the convention keeps its name
        cmds.rename("blueSG", "customSG")
        Blinn("blue").rename("green")
        self.assertTrue(cmds.objExists("customSG"))
        self.assertEqual(str(Blinn("green").engine), "customSG")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "already exists"):
            Blinn("green").rename("lambert1")
        with self.assertRaises(TypeError):
            Blinn("green").rename("")
        self.assertEqual(set(cmds.ls()), before)

    def test_tidy_renormalises_and_sweeps_orphans(self):
        red, blue = Blinn.define("red"), Blinn.define("blue")
        self.cube << red
        counts = []
        for _ in range(5):
            self.cube.f[:3] << blue
            self.cube.f[:3] << red
            self.assertEqual(_members("redSG"), ["cube.f[0:5]"])
            self.assertIsNone(shade.tidy())
            self.assertEqual(_members("redSG"), ["cubeShape"])
            counts.append(len(cmds.ls(type="groupId")))
        self.assertEqual(counts, [0] * 5)
        self.assertEqual(cmds.ls(type="groupParts"), [])
        # a partial membership is left alone, with the groupId its face group
        # needs; a groupId nothing consumes goes
        self.cube.f[:3] << blue
        self.cube.f[:3] << -blue
        cmds.createNode("groupId", name="stray")
        self.assertEqual(len(cmds.ls(type="groupId")), 2)
        shade.tidy(self.cube)
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])
        self.assertEqual(len(cmds.ls(type="groupId")), 1)
        self.assertFalse(cmds.objExists("stray"))
        self.assertEqual(_names(Material.of(self.cube.f[0])), [])


# --------------------------------------------------------------------- #
#  Teaching errors: all before any write
# --------------------------------------------------------------------- #


class TestMaterialErrors(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)
        self.red   = Blinn.define("red")
        self.cube << self.red
        self.joint  = Node.create("joint", name="joint1")
        self.plain  = Node(cmds.sets(empty=True, name="plainSet"))
        self.before = set(cmds.ls())

    def assert_nothing_written(self):
        self.assertEqual(set(cmds.ls()), self.before)
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_node_rhs_keeps_todays_message(self):
        """Historical id (v2.0.0a2): pinned a node on the right of '<<'
        (lambert1, redSG, a str) keeping "Cannot inject Node into a bare Node";
        it now pins that a joint, a plain objectSet and a str keep that message,
        while a shader or engine node assigns (round 4b NC7; ADD C12: lambert1
        feeds no engine in Maya 2025, so it builds lambert1SG)."""
        with self.assertRaisesRegex(TypeError, "Cannot inject Node into a bare Node"):
            self.cube << self.joint
        with self.assertRaisesRegex(TypeError, "Cannot inject Node into a bare Node"):
            self.cube << self.plain
        with self.assertRaisesRegex(TypeError, "Cannot inject str into a bare Node"):
            self.cube << "red"
        self.assert_nothing_written()
        self.assertIs(self.cube << Node("lambert1"), self.cube)
        self.assertEqual(_members("lambert1SG"), ["cubeShape"])
        self.assertIs(self.cube << Node("redSG"), self.cube)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        # a member node on the left of a member node writes nothing (ADD C15)
        before = set(cmds.ls())
        for call in (
            lambda: Node("redSG") << self.red,
            lambda: self.red << Node("redSG"),
            lambda: self.plain << self.red,
        ):
            with self.assertRaisesRegex(TypeError, "not geometry"):
                call()
        self.assertEqual(set(cmds.ls()), before)

    def test_wrong_lhs(self):
        red = self.red
        with self.assertRaisesRegex(TypeError, "materials bind faces or whole objects"):
            self.cube.vtx[:3] << red
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx << red
        with self.assertRaisesRegex(TypeError, "edges"):
            self.cube.e[:2] << red
        with self.assertRaisesRegex(TypeError, "UVs"):
            self.cube.map[:2] << red
        with self.assertRaisesRegex(TypeError, "no shadeable shape"):
            self.joint << red
        with self.assertRaisesRegex(TypeError, "not geometry"):
            Node("lambert1") << red
        # re-pinned (round 4b NC6): an attribute plug is refused as a plug
        with self.assertRaisesRegex(TypeError, "'lambert1.color' is a plug"):
            Node("lambert1").color << red
        with self.assertRaisesRegex(TypeError, r"element \[1\]"):
            List([self.cube, 5, None]) << red
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.cube.t << [red, 1, 2]
        with self.assertRaisesRegex(ValueError, "nothing to inject"):
            self.cube.f[6:] << red
        with self.assertRaisesRegex(ValueError, "nothing to inject"):
            List([]) << red
        with self.assertRaisesRegex(TypeError, r"Components\("):
            red._member().inject([self.cube, f"{self.shape}.f[1]"])
        self.assert_nothing_written()

    def test_wrong_material(self):
        with self.assertRaisesRegex(TypeError, r"is a blinn, not a lambert.*Material\('red'\)"):
            self.cube << Lambert("red")
        with self.assertRaisesRegex(TypeError, "takes no attributes"):
            self.cube << Material("red", type="lambert")
        with self.assertRaisesRegex(ValueError, "no surface shader named 'nope'"):
            self.cube << Material("nope")
        with self.assertRaisesRegex(ValueError, "not a registered surface shader"):
            self.cube << Material.create(type="aiStandardSurface")
        with self.assertRaisesRegex(TypeError, "not a surface shader"):
            self.cube << Material.create(type="ramp")
        with self.assertRaisesRegex(TypeError, "not a surface shader"):
            self.cube << Material.create(type="displacementShader")
        with self.assertRaisesRegex(TypeError, "is a time, not a surface shader"):
            self.cube << Material("time1")
        with self.assertRaisesRegex(TypeError, "is a transform, not a surface shader"):
            self.cube << Material("cube")
        with self.assertRaises(TypeError):
            self.cube << Blinn("x", type="phong")
        with self.assertRaises(TypeError):
            self.cube << ~self.red
        with self.assertRaises(TypeError):
            self.cube << -Blinn(None)
        # no unknown node from an unregistered type
        self.assertEqual(cmds.ls(type="unknown"), [])
        self.assert_nothing_written()

    def test_a_material_feeding_several_engines_needs_a_named_one(self):
        cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="redX")
        cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="redY")
        cmds.connectAttr("red.outColor", "redX.surfaceShader")
        cmds.connectAttr("red.outColor", "redY.surfaceShader")
        cmds.delete("redSG")
        other       = _cube("other")
        self.before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "redX"):
            other << Blinn("red")
        with self.assertRaises(ValueError):
            Blinn("red").engine
        # a query reads every engine the shader feeds (round 4b FIX): no raise
        self.assertNotIn(other, Blinn("red"))
        self.assertEqual(set(cmds.ls()), self.before)
        # naming the engine picks it
        other << ShadingEngine("redY")
        self.assertEqual(_members("redY"), ["otherShape"])
        self.assertIn(other, ShadingEngine("redY"))


# --------------------------------------------------------------------- #
#  Container policy
# --------------------------------------------------------------------- #


class TestMaterialContainer(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def _owner(self, node):
        return cmds.container(query=True, findContainer=[str(node)])

    def test_a_new_network_stays_out_of_the_scope_by_default(self):
        with container("look"):
            self.cube << Blinn.define("red")
        self.assertIsNone(self._owner("red"))
        self.assertIsNone(self._owner("redSG"))
        for info in cmds.listConnections("redSG.message", type="materialInfo"):
            self.assertIsNone(self._owner(info))
        self.assertIsNone(cmds.container("look", query=True, nodeList=True))
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_container_true_enrols_the_network_geometry_never(self):
        with container("look"):
            self.cube << Blinn.define("red", container=True)
        members = cmds.container("look", query=True, nodeList=True)
        info    = cmds.listConnections("redSG.message", type="materialInfo")
        self.assertEqual(sorted(members), sorted(["red", "redSG", *info]))
        self.assertIsNone(self._owner(self.shape))
        self.assertIsNone(self._owner(self.cube))
        # no __rig__ tag: cleanup never sweeps a material
        self.assertFalse(cmds.attributeQuery("__rig__", node="red", exists=True))

    def test_no_flatten_prefix_in_a_nested_scope(self):
        with container("outer"):
            with container("inner"):
                self.cube << Blinn.define("red", container=True)
        self.assertTrue(cmds.objExists("red"))
        self.assertFalse(cmds.objExists("inner_red"))
        self.assertEqual(self._owner("red"), "outer")
        self.assertEqual(self._owner("redSG"), "outer")

    def test_found_nodes_are_never_moved(self):
        self.cube << Blinn.define("red")
        other = _cube("other")
        with container("look"):
            other       << Blinn.define("red", container=True)     # found: stays where it is
            other       << Blinn.define("free")                    # new, default: out
            other.f[:2] << Blinn.define("inside", container=True)
        self.assertIsNone(self._owner("red"))
        self.assertIsNone(self._owner("redSG"))
        self.assertIsNone(self._owner("free"))
        self.assertIsNone(self._owner("freeSG"))
        self.assertEqual(self._owner("inside"), "look")
        self.assertEqual(self._owner("insideSG"), "look")
        self.assertIsNone(self._owner(_shape(other)))

    def test_adopted_engine_follows_the_material(self):
        rn.blinn(name="bare", container=False)
        asset = cmds.container(name="asset")
        cmds.container(asset, edit=True, addNode=["bare"], force=True)
        rn.blinn(name="loose", container=False)
        with container("look"):
            self.cube       << Material("bare")
            self.cube.f[:2] << Material("loose")
        self.assertEqual(self._owner("bareSG"), "asset")
        info = cmds.listConnections("bareSG.message", type="materialInfo")[0]
        self.assertEqual(self._owner(info), "asset")
        self.assertIsNone(self._owner("looseSG"))
        self.assertIsNone(self._owner("loose"))
        self.assertEqual(cmds.container("look", query=True, nodeList=True), None)

    def test_deleting_an_enrolled_container_leaves_the_mesh_green_and_repair_fixes(self):
        with container("look"):
            self.cube << Blinn.define("red", container=True)
        cmds.delete("look")
        self.assertFalse(cmds.objExists("red"))
        self.assertEqual(_engines(self.shape), [])
        self.assertEqual([str(x) for x in shade.repair()], ["cubeShape"])
        self.assertEqual(_members(ISG), ["cubeShape"])


# --------------------------------------------------------------------- #
#  Cost: one shape, never the scene
# --------------------------------------------------------------------- #


class TestMaterialPerformance(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_reads_cost_one_shape_not_the_scene(self):
        # Membership is read from the shape's own plugs; enumerating the
        # engine instead makes every operation O(scene) (seconds with a
        # few thousand meshes on initialShadingGroup).
        for i in range(400):
            cmds.polyCube(name=f"c{i}", ch=False)
        cube  = Node("c0")
        start = time.perf_counter()
        found = Material.of(cube)
        ids   = cube >> Default()
        held  = cube in Default()
        cube.f[:2] << -Default()
        cube       << Material()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)
        self.assertEqual(_names(found), [DEFAULT])
        np.testing.assert_array_equal(ids, np.arange(6))
        self.assertTrue(held)
        self.assertEqual(_engines(_shape(cube)), [])
        self.assertEqual(len(_members(ISG)), 399)


# --------------------------------------------------------------------- #
#  Undo
# --------------------------------------------------------------------- #


class TestMaterialUndo(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_one_undo_reverts_a_whole_lshift(self):
        cmds.undoInfo(state=True, infinity=True)
        cube   = _cube("cube")
        other  = _cube("other")
        red    = Blinn.define("red", color=(1, 0, 0))
        before = set(cmds.ls())
        List([cube.f[:3], other]) << red
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        cmds.undo()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(sorted(_members(ISG)), ["cubeShape", "otherShape"])
        cmds.redo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        cube.f[:3] << -red
        self.assertEqual(_members("redSG"), ["otherShape"])
        cmds.undo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        cube << Material()
        self.assertEqual(_engines(_shape(cube)), [])
        cmds.undo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        red.delete()
        self.assertFalse(cmds.objExists("red"))
        cmds.undo()
        self.assertTrue(cmds.objExists("red"))
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])


# --------------------------------------------------------------------- #
#  Examples and exports
# --------------------------------------------------------------------- #


class TestShadeExamples(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_image_loop_builds_twice_with_distinct_materials(self):
        from rig.examples import image_loop

        frames = tempfile.mkdtemp(prefix="rig_frames_")
        for i in (1, 2, 3):
            with open(os.path.join(frames, f"frame.{i:04d}.png"), "wb"):
                pass
        first  = image_loop.create_plane(frames, name="run")
        second = image_loop.create_plane(frames, name="run")
        # re-pinned (round 4b NC7): the example returns the Lambert node itself
        self.assertIsInstance(first.material, Lambert)
        self.assertEqual(cmds.nodeType(str(first.material)), "lambert")
        self.assertNotEqual(first.material, second.material)
        for setup in (first, second):
            shape  = _shape(setup.transform)
            engine = _engines(shape)
            self.assertEqual(len(engine), 1)
            self.assertEqual(
                cmds.listConnections(f"{engine[0]}.surfaceShader", source=True, destination=False),
                [str(setup.material)],
            )
            self.assertEqual(cmds.getAttr(f"{setup.material}.diffuse"), 1)
            self.assertEqual(setup.shape.sequenceEnd >> None, 3)
            self.assertIn(setup.shape, setup.material)
        members = cmds.container("run_container", query=True, nodeList=True)
        self.assertIn("run", members)
        self.assertIn("runSG", members)

    def test_perspective_image_planes_builds(self):
        from rig.examples import perspective_image_planes

        setup = perspective_image_planes.create_setup("camera1", 2)
        self.assertEqual(len(setup.planes), 2)
        for i, shape in enumerate(setup.shapes):
            engine = _engines(str(shape))
            self.assertEqual(engine, [f"sticker_layer_{i}SG"])
            self.assertEqual(
                cmds.listConnections(f"{engine[0]}.surfaceShader", source=True, destination=False),
                [f"sticker_layer_{i}"],
            )
            self.assertIn(shape, Lambert(f"sticker_layer_{i}"))


class TestShadeExports(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_exports(self):
        import rig

        self.assertIs(rig.shade, shade)
        self.assertIn("shade", rig.__all__)
        self.assertEqual(
            sorted(shade.__all__),
            sorted([
                "Material", "Lambert", "Blinn", "Phong", "PhongE", "SurfaceShader",
                "StandardSurface", "OpenPBRSurface", "Default", "Conversion", "convert",
                "materials", "bindings", "repair", "tidy",
            ]),
        )
        # re-pinned (round 4b NC7): the shader classes are the node classes
        for name in ("Material", "Lambert", "Blinn", "Phong", "PhongE", "SurfaceShader",
                     "StandardSurface", "OpenPBRSurface"):
            self.assertIs(getattr(shade, name), getattr(nodetypes, name))
        for name in ("Material", "Blinn", "Lambert", "Default"):
            self.assertFalse(hasattr(rig, name))
        self.assertIn("Material", rig.__doc__)
