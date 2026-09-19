"""Tests for ``rig.shade`` -- materials through the membership grammar.

Every error test asserts a zero ``cmds.ls()`` delta: a refused spelling
writes nothing. PlugLists are never compared with ``assertEqual``.
"""

import os
import shutil
import tempfile
import time
from unittest import mock

import numpy as np
from maya import cmds
from rig import Components, container, Node, PlugList, shade, Tag
from rig.bridges import nodes as rn
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


def _shape(node):
    """Full path of the first shape under a transform ``Node`` / name."""
    return cmds.listRelatives(str(node), shapes=True, fullPath=True)[0]


def _members(engine):
    """Membership as ``cmds.sets`` prints it."""
    return cmds.sets(str(engine), query=True) or []


def _engines(shape):
    return sorted(set(cmds.listConnections(shape, type="shadingEngine") or []))


def _names(specs):
    return [repr(spec) for spec in specs]


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


# --------------------------------------------------------------------- #
#  Construction: lazy, validated, typed
# --------------------------------------------------------------------- #


class TestMaterialConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construction_makes_no_maya_calls(self):
        before = set(cmds.ls())
        specs  = [
            Blinn("red", color=(1, 0, 0), diffuse=0.5),
            Material("x", type="phong", unique=True, update=True, container=False),
            Material(None),
            Lambert(None),
            Default(),
            -Blinn("red"),
            Material(Node(ISG)),
        ]
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(specs[0]), "red")
        self.assertEqual(repr(specs[0]), "Blinn('red')")
        self.assertEqual(specs[0].type, "blinn")
        self.assertEqual(specs[0].attrs, {"color": (1, 0, 0), "diffuse": 0.5})
        self.assertEqual(specs[1].type, "phong")
        self.assertTrue(specs[2].purges)
        self.assertEqual(repr(specs[4]), "Default()")
        self.assertEqual(str(specs[4]), ISG)
        self.assertEqual(repr(specs[5]), "-Blinn('red')")
        self.assertTrue(specs[5].removes)
        self.assertEqual(str(specs[6]), ISG)
        for cls, node_type in (
            (Lambert, "lambert"), (Blinn, "blinn"), (Phong, "phong"),
            (PhongE, "phongE"), (SurfaceShader, "surfaceShader"),
            (StandardSurface, "standardSurface"), (OpenPBRSurface, "openPBRSurface"),
        ):
            self.assertEqual(cls("x").type, node_type)
            self.assertIs(shade._BY_TYPE[node_type], cls)
        self.assertIsNone(Material("x").type)

    def test_an_empty_call_is_the_purge(self):
        # Material() / Blinn() are the same spec as Material(None); Default()
        # keeps meaning initialShadingGroup (it has no name to leave out)
        for purge in (Material(), Blinn(), Lambert()):
            self.assertTrue(purge.purges)
            self.assertIsNone(purge.name)
        self.assertEqual(repr(Blinn()), "Blinn(None)")
        self.assertEqual(str(Default()), ISG)
        self.assertFalse(Default().purges)

    def test_rejected_constructions(self):
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Blinn("")
        with self.assertRaises(TypeError):
            Blinn(5)
        with self.assertRaisesRegex(TypeError, "contradictory"):
            Blinn("x", type="phong")
        with self.assertRaisesRegex(TypeError, "takes no name"):
            Default("x")
        with self.assertRaisesRegex(TypeError, "takes no name"):
            Default(None)
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~Blinn("x")
        with self.assertRaises(TypeError):
            ~Material(None)
        with self.assertRaisesRegex(TypeError, "double negative"):
            -Blinn(None)
        with self.assertRaises(TypeError):
            -(-Blinn("x"))
        with self.assertRaisesRegex(TypeError, "wrap it first"):
            Phong(Node("lambert1"))
        self.assertEqual(set(cmds.ls()), before)
        # the same type twice is not a contradiction
        self.assertEqual(Blinn("x", type="blinn").type, "blinn")

    def test_negation_keeps_the_options(self):
        spec    = Blinn("red", color=(1, 0, 0), unique=True, update=True, container=False)
        removal = -spec
        self.assertTrue(removal.removes)
        self.assertFalse(spec.removes)
        self.assertEqual(removal.attrs, spec.attrs)
        self.assertEqual(removal.type, "blinn")
        self.assertTrue(removal._unique)
        self.assertTrue(removal._update)
        self.assertIs(removal._container, False)

    def test_methods_refuse_removal_and_purge_copies(self):
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            (-Blinn("x")).delete()
        with self.assertRaises(TypeError):
            (-Blinn("x")).build()
        with self.assertRaises(TypeError):
            Material(None).delete()
        with self.assertRaises(TypeError):
            Material(None).rename("y")
        with self.assertRaises(TypeError):
            Blinn(None).build()
        with self.assertRaises(TypeError):
            Material(None).node
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  Create: find-or-create, kwargs, unique, wrapping
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
        red       = Blinn("red", color=(1, 0, 0))
        result    = self.cube << red
        self.assertIs(result, self.cube)
        self.assertEqual(cmds.ls(selection=True), selection)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertEqual(cmds.nodeType("redSG"), "shadingEngine")
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
        red   = Blinn("red", color=(1, 0, 0))
        state = (red.name, red.type, red.attrs, repr(red), red.removes, red._built)
        self.cube << red
        self.cube << red
        self.cube.f[:2] << red
        self.assertEqual(
            (red.name, red.type, red.attrs, repr(red), red.removes, red._built), state
        )
        self.assertEqual(cmds.ls("red*", type="blinn"), ["red"])

    def test_found_material_is_a_plain_assignment(self):
        self.cube << Blinn("red", color=(1, 0, 0))
        other = _cube("other")
        before = set(cmds.ls())
        result = other << Blinn("red", color=(0, 0, 1))
        self.assertIs(result, other)
        self.assertEqual(set(cmds.ls()), before)
        # kwargs skipped on a found material
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "otherShape"])
        # update=True re-asserts them
        other << Blinn("red", color=(0, 0, 1), update=True)
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 0.0, 1.0))
        self.assertEqual(set(cmds.ls()), before)

    def test_plug_kwarg_connects_and_spec_kwarg_applies(self):
        from rig.spec import lock

        texture = rn.file(name="tex")
        self.cube << Blinn("skin", color=texture.outColor, diffuse=0.25)
        self.assertEqual(
            cmds.listConnections("skin.color", source=True, destination=False, plugs=True),
            ["tex.outColor"],
        )
        self.assertAlmostEqual(cmds.getAttr("skin.diffuse"), 0.25)
        # a spec kwarg applies to the attribute
        self.cube << Blinn("locked", diffuse=lock)
        self.assertTrue(cmds.getAttr("locked.diffuse", lock=True))

    def test_kwarg_typo_raises_before_any_write(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "no attribute 'colour'"):
            self.cube << Blinn("red", colour=(1, 0, 0))
        self.assertEqual(set(cmds.ls()), before)
        self.cube << Blinn("red")
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            self.cube << Blinn("red", colour=(1, 0, 0), update=True)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_unique_builds_a_fresh_network_per_lshift(self):
        other = _cube("other")
        spec  = Lambert("plane", unique=True, diffuse=0)
        self.cube << spec
        first = str(spec.node)
        other << spec
        self.assertNotEqual(str(spec.node), first)
        self.assertEqual(sorted(cmds.ls("plane*", type="lambert")), ["plane", "plane1"])
        self.assertEqual(sorted(cmds.ls("plane*", type="shadingEngine")), ["plane1SG", "planeSG"])
        self.assertEqual(str(spec.engine), "plane1SG")
        self.assertEqual(cmds.getAttr("plane1.diffuse"), 0)
        # one list application is one network
        PlugList([self.cube, other]) << Lambert("both", unique=True)
        self.assertEqual(cmds.ls("both*", type="lambert"), ["both"])
        self.assertEqual(sorted(_members("bothSG")), ["cubeShape", "otherShape"])

    def test_namespace_idempotency(self):
        cmds.namespace(add="look")
        cmds.namespace(set="look")
        try:
            self.cube << Blinn("red")
            before = set(cmds.ls())
            self.cube << Blinn("red")
            self.assertEqual(set(cmds.ls()), before)
            self.assertEqual(str(Blinn("red").node), "look:red")
            self.assertEqual(str(Blinn("red").engine), "look:redSG")
        finally:
            cmds.namespace(set=":")
        self.assertEqual(cmds.ls(type="blinn"), ["look:red"])
        self.assertEqual(str(Blinn("look:red").node), "look:red")
        # from the root namespace the short name is a different material
        self.cube << Blinn("red")
        self.assertEqual(sorted(cmds.ls(type="blinn")), ["look:red", "red"])

    def test_wrap_an_existing_shader_a_bare_one_and_an_engine(self):
        # any existing surface shader, with its type asserted by a subclass
        result = self.cube << Material("lambert1")
        self.assertIs(result, self.cube)
        self.assertEqual(_members("lambert1SG"), ["cubeShape"])
        self.assertEqual(str(Lambert("lambert1").node), "lambert1")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"exists as a lambert.*Material\('lambert1'\)"):
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
        # an engine name or Node stands for its shader
        self.assertEqual(str(Material("bareSG").node), "bare")
        self.assertEqual(str(Material(Node("bareSG")).engine), "bareSG")
        self.assertEqual(str(Default().node), "standardSurface1")
        self.assertEqual(str(Default().engine), ISG)
        self.assertEqual(str(Material("standardSurface1").engine), ISG)
        # a shaderless engine names no material
        empty = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name="emptySG")
        with self.assertRaisesRegex(ValueError, "no surface shader"):
            self.cube << Material(empty)
        with self.assertRaisesRegex(TypeError, "particle"):
            self.cube << Material("initialParticleSE")
        self.assertEqual(bare, Material("bare").node)

    def test_warns_when_maya_keeps_another_name(self):
        # 'shared' is reserved by Maya: shadingNode(name='shared') makes 'shared1'
        with mock.patch.object(cmds, "warning") as warning:
            self.cube << Blinn("shared")
        self.assertFalse(cmds.objExists("shared"))
        self.assertTrue(cmds.objExists("shared1"))
        warning.assert_called_once()
        self.assertIn("shared1", warning.call_args[0][0])
        with mock.patch.object(cmds, "warning") as warning:
            self.cube << Blinn("shared", unique=True)
        warning.assert_not_called()
        # names Maya could not keep are refused before any call
        before = set(cmds.ls())
        for bad in ("my-mat", "bad name", "1st", "a.b", "", None.__class__):
            with self.assertRaises((TypeError, ValueError)):
                Blinn(bad)
        self.assertEqual(set(cmds.ls()), before)
        for good in ("ok", "ns:mat", "_x", "mat2", "|grp|mat"):
            self.assertEqual(str(Blinn(good)), good)

    def test_build_creates_without_a_target(self):
        spec = Blinn("lib", color=(0, 1, 0))
        node = spec.build()
        self.assertIsInstance(node, Node)
        self.assertEqual(str(node), "lib")
        self.assertEqual(_members("libSG"), [])
        self.assertEqual(cmds.getAttr("lib.color")[0], (0.0, 1.0, 0.0))
        self.assertEqual(_members(ISG), ["cubeShape"])
        # a second build finds it
        before = set(cmds.ls())
        self.assertEqual(spec.build(), node)
        self.assertEqual(set(cmds.ls()), before)
        with self.assertRaisesRegex(ValueError, "no type"):
            Material("untyped").build()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(Default().build()), "standardSurface1")


# --------------------------------------------------------------------- #
#  The handle: node, engine, attribute forwarding
# --------------------------------------------------------------------- #


class TestMaterialHandle(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube = _cube("cube")

    def test_handle_is_find_only(self):
        red    = Blinn("red")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no surface shader named 'red'"):
            red.node
        with self.assertRaises(ValueError):
            red.engine
        with self.assertRaises(ValueError):
            red.color
        with self.assertRaises(ValueError):
            red.color = (1, 0, 0)
        self.assertEqual(set(cmds.ls()), before)
        self.cube << red
        self.assertEqual(str(red.node), "red")
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
        with self.assertRaisesRegex(ValueError, "no shading engine"):
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
        self.red   = Blinn("red")
        self.blue  = Blinn("blue")

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
        Components("cube.f[3]") << self.blue
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
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.assertEqual(cmds.ls(type="groupId"), [])

    def test_node_lhs_assigns_its_own_shapes_never_the_subtree(self):
        grp = Node(cmds.group(str(self.cube), name="grp"))
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
        self.assertEqual(_names(Material.of(srf)), ["Blinn('red')"])
        crv = Node(cmds.circle(name="crv")[0])
        lat = Node(cmds.lattice(str(self.cube))[1])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "nurbsCurve, not a shadeable"):
            crv << self.red
        with self.assertRaisesRegex(TypeError, "lattice, not a shadeable"):
            lat << self.red
        with self.assertRaisesRegex(TypeError, r"without faces.*Material\.of"):
            srf >> self.red
        self.assertEqual(set(cmds.ls()), before)

    def test_a_plug_on_a_transform_with_a_control_curve_skips_it_too(self):
        crv = cmds.circle(name="crv", ch=False)[0]
        cmds.parent(_shape(crv), str(self.cube), shape=True, relative=True)
        cmds.delete(crv)
        plug   = self.cube.tx
        result = plug << self.red
        self.assertIs(result, plug)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        np.testing.assert_array_equal(self.cube.t >> self.red, np.arange(6))
        self.assertEqual(_names(Material.of(self.cube.tx)), ["Blinn('red')"])
        self.assertEqual(_names(self.cube.ty >> Material()), ["Blinn('red')"])
        self.cube.sx << -self.red
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
        self.assertEqual(_names(Material.of(self.cube)), ["Blinn('red')"])
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
            PlugList([self.cube, lone]) << self.red
        with self.assertRaises(TypeError):
            Material.of(lone)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_engines(self.shape), [])

    def test_pluglist_lhs_is_one_material_one_engine_one_call(self):
        other = _cube("other")
        lhs   = PlugList([self.cube, other.f[:2]])
        with mock.patch.object(cmds, "sets", wraps=cmds.sets) as sets:
            result = lhs << self.red
        self.assertIs(result, lhs)
        writes = [c for c in sets.call_args_list if c.kwargs.get("forceElement") == "redSG"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "other.f[0:1]"])
        self.assertEqual(cmds.ls(type="blinn"), ["red"])
        # the same node twice and its faces: the whole object wins
        PlugList([self.cube, self.cube.f[:2]]) << self.blue
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        # a mixed list fails as a whole, nothing written
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "vertices"):
            PlugList([other, self.cube.vtx[0]]) << Blinn("nope")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("redSG"), ["other.f[0:1]"])

    def test_pluglist_broadcast_pairs_each_selection_with_its_spec(self):
        lhs    = PlugList([self.cube.f[:2], self.cube.f[2:4]])
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
        plug   = self.cube.tx
        result = plug << self.red
        self.assertIs(result, plug)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        np.testing.assert_array_equal(self.cube.t >> self.red, np.arange(6))
        self.assertEqual(_names(self.cube.rotate >> Blinn()), ["Blinn('red')"])
        self.assertEqual(_names(Material.of(self.cube.visibility)), ["Blinn('red')"])
        self.assertEqual([str(x) for x in shade.materials(self.cube.tx)], ["red"])
        # two plugs of one node are one node
        PlugList([self.cube.tx, self.cube.ty]) << self.blue
        self.assertEqual(_members("blueSG"), ["cubeShape"])
        self.assertEqual(_members("redSG"), [])
        self.cube.sx << -self.blue
        self.assertEqual(_engines(self.shape), [])
        self.cube.v << Default()
        self.assertEqual(_members(ISG), ["cubeShape"])
        # the shape's own plugs stand for the shape
        Node(self.shape).castsShadows << self.red
        self.assertEqual(_members("redSG"), ["cubeShape"])
        # component plugs keep their meaning; a plug of a node with nothing
        # shadeable refuses as the node would
        joint = Node.create("joint", name="joint1")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx[:3] << self.red
        with self.assertRaisesRegex(TypeError, "no shadeable shape"):
            joint.tx << self.red
        with self.assertRaisesRegex(TypeError, "vertices"):
            PlugList([self.cube.tx, self.cube.vtx[0]]) << self.blue
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
        self.red   = Blinn("red")
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
        self.assertEqual(_members(ISG), ["cube.f[0:2]"])
        self.assertEqual(_members("redSG"), ["cube.f[3:5]"])

    def test_faces_leave_a_per_face_membership(self):
        self.cube.f[:3] << Blinn("blue")
        self.cube.f[[0, 5]] << -Blinn("blue")
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
        self.cube << -self.red
        self.assertEqual(_members("redSG"), [])
        self.cube.f << self.red
        self.cube.f << -self.red
        self.assertEqual(_members("redSG"), [])

    def test_missing_material_is_a_value_error(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no surface shader named 'nope'"):
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
        self.cube.f[:3] << Blinn("blue")
        result = self.cube << Material(None)
        self.assertIs(result, self.cube)
        self.assertEqual(_engines(self.shape), [])
        self.assertEqual(_members("redSG"), [])
        self.assertEqual(_members("blueSG"), [])
        self.assertEqual(_names(Material.of(self.cube)), [])
        self.assertEqual(shade.bindings(self.cube), [])
        result = self.cube << Default()
        self.assertIs(result, self.cube)
        self.assertEqual(_members(ISG), ["cubeShape"])
        self.assertEqual(_names(Material.of(self.cube)), ["Default()"])
        np.testing.assert_array_equal(self.cube.f[:6] >> Default(), np.arange(6))
        # any subclass purges; faces purge per face
        self.cube.f[:3] << self.red
        self.cube.f[:2] << Blinn(None)
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
        Blinn("a").build()
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
        self.assertEqual(_names(Material.of(self.cube.f[3])), ["Blinn('a')"])

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
            (inst.f[:2],      Material(None)),
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
        self.assertEqual(sorted(_members(ISG)), ["cube1.f[3:5]", "cube|cubeShape"])
        self.assertEqual(_names(Material.of(inst.f[2])), [])
        self.assertEqual(_names(Material.of(self.cube)), ["Default()"])
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
        self.cube.f[:3] << Blinn("red")
        np.testing.assert_array_equal(self.cube >> Blinn("red"), [0, 1, 2])
        np.testing.assert_array_equal(self.cube >> Material("red"), [0, 1, 2])
        np.testing.assert_array_equal(self.cube.f >> Blinn("red"), [0, 1, 2])
        np.testing.assert_array_equal(self.cube.f[1:5] >> Blinn("red"), [1, 2])
        np.testing.assert_array_equal(self.cube.f[[5, 0]] >> Blinn("red"), [0])
        self.assertEqual((self.cube.f[3:] >> Blinn("red")).shape, (0,))
        np.testing.assert_array_equal(self.cube >> Default(), [3, 4, 5])
        # an engine the shape is not in: empty, never None
        _cube("other") << Blinn("blue")
        self.assertEqual((self.cube >> Blinn("blue")).shape, (0,))
        self.assertEqual((self.cube.f[:2] >> Blinn("blue")).shape, (0,))
        # a bare material: no engine, empty
        rn.blinn(name="bare")
        self.assertEqual((self.cube >> Blinn("bare")).shape, (0,))
        # the ids go back through the handle
        self.cube.f[self.cube >> Blinn("red")] << Blinn("blue")
        self.assertEqual(sorted(_members("blueSG")), ["cube.f[0:2]", "otherShape"])

    def test_query_refusals(self):
        self.cube.f[:3] << Blinn("red")
        other  = _cube("other")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no surface shader named 'nope'"):
            self.cube >> Blinn("nope")
        with self.assertRaisesRegex(TypeError, r"exists as a blinn.*Material\('red'\)"):
            self.cube >> Lambert("red")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            PlugList([self.cube, other]) >> Blinn("red")
        with self.assertRaisesRegex(TypeError, "Split them"):
            PlugList([self.cube, self.cube.f[:2]]) >> Blinn("red")
        with self.assertRaises(TypeError):
            self.cube >> -Blinn("red")
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx[:2] >> Blinn("red")
        with self.assertRaises(TypeError):
            Node("lambert1") >> Blinn("red")
        self.assertEqual(set(cmds.ls()), before)
        # the same node twice is one node
        np.testing.assert_array_equal(PlugList([self.cube, self.cube]) >> Blinn("red"), [0, 1, 2])

    def test_rshift_purge_enumerates_like_of(self):
        self.cube.f[:3] << Blinn("red")
        self.cube.f[3]  << Lambert("skin")
        for lhs in (
            self.cube, self.cube.f, self.cube.f[0], self.cube.f[[0, 3]],
            self.cube.f[4:], self.cube.tx,
        ):
            for cls in (Material, Blinn, Lambert):
                got = lhs >> cls()
                self.assertIsInstance(got, list)
                self.assertEqual(_names(got), _names(cls.of(lhs)))
        self.assertEqual(
            _names(self.cube >> Material()), ["Default()", "Blinn('red')", "Lambert('skin')"]
        )
        self.assertEqual(_names(self.cube >> Material(None)), _names(self.cube >> Material()))
        self.assertEqual(_names(self.cube >> Blinn()), ["Blinn('red')"])
        self.assertEqual(_names(self.cube.f[2] >> Material()), ["Blinn('red')"])
        self.assertEqual(_names(self.cube.f[4:] >> Material()), ["Default()"])
        self.assertIs(type((self.cube >> Blinn())[0]), Blinn)
        # Default() is not a purge: it queries initialShadingGroup's faces
        np.testing.assert_array_equal(self.cube >> Default(), [4, 5])
        # re-injectable
        self.cube.f[5] << (self.cube >> Blinn())[0]
        np.testing.assert_array_equal(self.cube >> Blinn("red"), [0, 1, 2, 5])
        other  = _cube("other")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx[0] >> Material()
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            PlugList([self.cube, other]) >> Material()
        self.assertEqual(set(cmds.ls()), before)

    def test_of_types_by_the_live_node_type(self):
        self.assertEqual(_names(Material.of(self.cube)), ["Default()"])
        self.assertIsInstance(Material.of(self.cube)[0], Default)
        self.cube.f[:3] << Blinn("red")
        self.cube.f[3] << Material("ani", type="anisotropic")
        found = Material.of(self.cube)
        self.assertEqual(_names(found), ["Default()", "Blinn('red')", "Material('ani')"])
        self.assertIs(type(found[1]), Blinn)
        self.assertIs(type(found[2]), Material)
        self.assertEqual(_names(Material.of(self.cube.f[0])), ["Blinn('red')"])
        self.assertEqual(_names(Material.of(self.cube.f[[0, 3]])), [])
        self.assertEqual(_names(Material.of(self.cube.f[4:])), ["Default()"])
        # the bare handle is the whole object
        self.assertEqual(_names(Material.of(self.cube.f)), _names(found))
        self.assertEqual(_names(Blinn.of(self.cube)), ["Blinn('red')"])
        self.assertEqual(_names(Lambert.of(self.cube)), [])
        self.assertEqual(_names(Default.of(self.cube)), ["Default()"])
        self.assertEqual(_names(Default.of(self.cube.f[0])), [])
        self.assertEqual(_names(StandardSurface.of(self.cube.f[5])), ["StandardSurface('standardSurface1')"])
        # re-injectable
        self.cube.f[5] << found[1]
        np.testing.assert_array_equal(self.cube >> Blinn("red"), [0, 1, 2, 5])
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Material.of(PlugList([self.cube, _cube("other")]))
        with self.assertRaises(TypeError):
            Material.of(self.cube.vtx[0])
        self.assertEqual(set(cmds.ls()) - {"other", "otherShape"}, before)

    def test_module_readers(self):
        self.cube.f[:3] << Blinn("red")
        self.cube.f[3]  << Blinn("blue")
        self.assertEqual(
            [str(x) for x in shade.materials(self.cube)],
            ["standardSurface1", "red", "blue"],
        )
        self.assertIsInstance(shade.materials(self.cube), PlugList)
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
        self.assertEqual(str(material), "red")
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
        self.cube << Blinn("red")
        other.f[:2] << Blinn("red")
        info = cmds.listConnections("redSG.message", type="materialInfo")
        self.assertIsNone(Material("red").delete())
        for name in ("red", "redSG", *info):
            self.assertFalse(cmds.objExists(name))
        self.assertEqual(_engines(self.shape), [])
        self.assertEqual(_names(Material.of(self.cube)), [])
        fixed = shade.repair()
        self.assertIsInstance(fixed, PlugList)
        self.assertEqual(sorted(str(x) for x in fixed), ["cubeShape", "otherShape"])
        self.assertEqual(_names(Material.of(self.cube)), ["Default()"])
        self.assertEqual(sorted(_members(ISG)), ["cubeShape", "other.f[0:5]"])
        # nothing left to repair
        self.assertEqual([str(x) for x in shade.repair()], [])

    def test_delete_refuses_default_and_referenced_nodes(self):
        self.cube << Default()
        before = set(cmds.ls())
        with self.assertRaisesRegex(RuntimeError, "default node"):
            Material("lambert1").delete()
        with self.assertRaisesRegex(RuntimeError, "default node"):
            Material("standardSurface1").delete()
        with self.assertRaisesRegex(TypeError, "cannot be deleted"):
            Default().delete()
        with self.assertRaises(TypeError):
            Default().rename("x")
        with self.assertRaisesRegex(RuntimeError, "default node"):
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
        inst << Blinn("x")
        self.cube << Material(None)
        self.assertEqual(_members(ISG), [])
        self.assertEqual([str(x) for x in shade.repair()], ["cube|cubeShape"])
        self.assertEqual(_members(ISG), ["cube|cubeShape"])
        self.assertEqual(_members("xSG"), ["cube1|cubeShape"])
        # the other way round: the scene walk reaches every instance path
        self.cube << Blinn("x")
        inst << Material(None)
        self.assertEqual(_members("xSG"), ["cube|cubeShape"])
        self.assertEqual([str(x) for x in shade.repair()], ["cube1|cubeShape"])
        self.assertEqual(_members(ISG), ["cube1|cubeShape"])
        self.assertEqual(_members("xSG"), ["cube|cubeShape"])
        self.assertEqual([str(x) for x in shade.repair()], [])

    def test_rename_refuses_a_taken_engine_name(self):
        red = Blinn("red")
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
        self.assertEqual(str(red.engine), "customSG")
        self.assertEqual(cmds.ls("blueSG*"), ["blueSG"])

    def test_rename_follows_the_engine_convention_and_the_spec(self):
        red = Blinn("red")
        self.cube << red
        self.assertIsNone(red.rename("blue"))
        self.assertEqual(cmds.ls(type="blinn"), ["blue"])
        self.assertTrue(cmds.objExists("blueSG"))
        self.assertFalse(cmds.objExists("redSG"))
        self.assertEqual(str(red), "blue")
        self.assertEqual(str(red.node), "blue")
        self.assertEqual(str(red.engine), "blueSG")
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
        red, blue = Blinn("red"), Blinn("blue")
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
        self.cube << Blinn("red")
        self.joint = Node.create("joint", name="joint1")
        self.before = set(cmds.ls())

    def assert_nothing_written(self):
        self.assertEqual(set(cmds.ls()), self.before)
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_node_rhs_keeps_todays_message(self):
        with self.assertRaisesRegex(TypeError, "Cannot inject Node into a bare Node"):
            self.cube << Node("lambert1")
        with self.assertRaisesRegex(TypeError, "Cannot inject Node into a bare Node"):
            self.cube << Node("redSG")
        with self.assertRaisesRegex(TypeError, "Cannot inject str into a bare Node"):
            self.cube << "red"
        self.assert_nothing_written()

    def test_wrong_lhs(self):
        with self.assertRaisesRegex(TypeError, "materials bind faces or whole objects"):
            self.cube.vtx[:3] << Blinn("x")
        with self.assertRaisesRegex(TypeError, "vertices"):
            self.cube.vtx << Blinn("x")
        with self.assertRaisesRegex(TypeError, "edges"):
            self.cube.e[:2] << Blinn("x")
        with self.assertRaisesRegex(TypeError, "UVs"):
            self.cube.map[:2] << Blinn("x")
        with self.assertRaisesRegex(TypeError, "no shadeable shape"):
            self.joint << Blinn("x")
        with self.assertRaisesRegex(TypeError, "not geometry"):
            Node("lambert1") << Blinn("x")
        with self.assertRaisesRegex(TypeError, "not geometry"):
            Node("lambert1").color << Blinn("x")
        with self.assertRaisesRegex(TypeError, r"element \[1\]"):
            PlugList([self.cube, 5, None]) << Blinn("x")
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.cube.t << [Blinn("x"), 1, 2]
        with self.assertRaisesRegex(ValueError, "nothing to inject"):
            self.cube.f[6:] << Blinn("x")
        with self.assertRaises(ValueError):
            PlugList([]) << Blinn("x")
        with self.assertRaisesRegex(TypeError, r"Components\("):
            Blinn("x").inject([self.cube, f"{self.shape}.f[1]"])
        self.assert_nothing_written()

    def test_wrong_material(self):
        with self.assertRaisesRegex(TypeError, r"exists as a blinn.*Material\('red'\)"):
            self.cube << Lambert("red")
        with self.assertRaisesRegex(TypeError, "exists as a blinn"):
            self.cube << Material("red", type="lambert")
        with self.assertRaisesRegex(ValueError, "no type to build one"):
            self.cube << Material("nope")
        with self.assertRaisesRegex(ValueError, "not a registered surface shader"):
            self.cube << Material("x", type="aiStandardSurface")
        with self.assertRaisesRegex(TypeError, "not a surface shader"):
            self.cube << Material("x", type="ramp")
        with self.assertRaisesRegex(TypeError, "not a surface shader"):
            self.cube << Material("x", type="displacementShader")
        with self.assertRaisesRegex(TypeError, "not a surface shader or a shading engine"):
            self.cube << Material("time1")
        with self.assertRaisesRegex(TypeError, "not a surface shader or a shading engine"):
            self.cube << Material("cube")
        with self.assertRaises(TypeError):
            self.cube << Blinn("x", type="phong")
        with self.assertRaises(TypeError):
            self.cube << ~Blinn("x")
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
        other = _cube("other")
        self.before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "redX"):
            other << Blinn("red")
        with self.assertRaises(ValueError):
            Blinn("red").engine
        self.assertEqual(set(cmds.ls()), self.before)
        # naming the engine picks it
        other << Material("redY")
        self.assertEqual(_members("redY"), ["otherShape"])


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

    def test_new_network_joins_the_scope_geometry_does_not(self):
        with container("look"):
            self.cube << Blinn("red")
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
                self.cube << Blinn("red")
        self.assertTrue(cmds.objExists("red"))
        self.assertFalse(cmds.objExists("inner_red"))
        self.assertEqual(self._owner("red"), "outer")
        self.assertEqual(self._owner("redSG"), "outer")

    def test_found_nodes_are_never_moved_and_container_false_opts_out(self):
        self.cube << Blinn("red")
        other = _cube("other")
        with container("look"):
            other << Blinn("red")
            other << Blinn("free", container=False)
            other.f[:2] << Blinn("inside")
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
            self.cube << Material("bare")
            self.cube.f[:2] << Material("loose")
        self.assertEqual(self._owner("bareSG"), "asset")
        info = cmds.listConnections("bareSG.message", type="materialInfo")[0]
        self.assertEqual(self._owner(info), "asset")
        self.assertIsNone(self._owner("looseSG"))
        self.assertIsNone(self._owner("loose"))
        self.assertEqual(cmds.container("look", query=True, nodeList=True), None)

    def test_deleting_the_container_leaves_the_mesh_green_and_repair_fixes(self):
        with container("look"):
            self.cube << Blinn("red")
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
        cube.f[:2] << -Default()
        cube << Material(None)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)
        self.assertEqual(_names(found), ["Default()"])
        np.testing.assert_array_equal(ids, np.arange(6))
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
        before = set(cmds.ls())
        PlugList([cube.f[:3], other]) << Blinn("red", color=(1, 0, 0))
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        cmds.undo()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(sorted(_members(ISG)), ["cubeShape", "otherShape"])
        cmds.redo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        cube.f[:3] << -Blinn("red")
        self.assertEqual(_members("redSG"), ["otherShape"])
        cmds.undo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        cube << Material(None)
        self.assertEqual(_engines(_shape(cube)), [])
        cmds.undo()
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:2]", "otherShape"])
        Material("red").delete()
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
        self.assertIsInstance(first.material, Node)
        self.assertEqual(cmds.nodeType(str(first.material)), "lambert")
        self.assertNotEqual(str(first.material), str(second.material))
        for setup in (first, second):
            shape = _shape(setup.transform)
            engine = _engines(shape)
            self.assertEqual(len(engine), 1)
            self.assertEqual(
                cmds.listConnections(f"{engine[0]}.surfaceShader", source=True, destination=False),
                [str(setup.material)],
            )
            self.assertEqual(cmds.getAttr(f"{setup.material}.diffuse"), 1)
            self.assertEqual(setup.shape.sequenceEnd >> None, 3)
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
        for name in ("Material", "Blinn", "Lambert", "Default"):
            self.assertFalse(hasattr(rig, name))
        self.assertIn("Material", rig.__doc__)
