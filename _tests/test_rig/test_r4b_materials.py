"""Round 4b, step NC7: membership on nodes II -- materials are the membership
nodes, and a conversion returns the new node.

* ``TestMaterialNodes``: the recommendation's W3 / W4 / W6 rows (define once
  and assign many; queries and removals before a material exists never create
  it; the type mismatches); the tokens and their reprs (``Blinn()``,
  ``-red``, ``-ShadingEngine("redSG")``; ``Blinn(None)`` and
  ``ShadingEngine()`` refused, naming ``Material()``); ``Material.of`` /
  ``Blinn.of`` answer live nodes (flat classes; ``Default()`` for the default
  engine); ``rig.shade`` exports the node classes.
* ``TestNodeRhs``: a shader or engine node on the right of ``<<`` / ``>>``
  (the addendum's MN3b list): assign, remove, query, purge; one undo chunk
  per ``<<``; a dead, renamed or namespaced node; ``initialParticleSE`` and
  an engine fed by no shader refused; a shader over two engines (``<shader>SG``
  and the exact ``ShadingEngine('altSG')``); ``List([]) << red`` (ADD C14); a
  member node on the left (ADD C15); ``lambert1`` builds ``lambert1SG`` (ADD
  C12); other nodes keep today's message; a container scope.
* ``TestMaterialIn``: ``x in red`` / ``x in ShadingEngine('altSG')``: every
  face (or the whole object) of every shape, all members of a list, a plug
  standing for its node; the tokens and wrong kinds refused.
* ``TestConversionReturnsNode``: ``astype`` / ``shade.convert`` return the new
  node (a class or a type name; a name converts too); ``dry_run`` returns the
  report; ``strict``; the generic ``Material`` and non-shader targets refused;
  a reference never converts; a shader already of the type comes back as is.
* ``TestConvertedHandle``: the held old node (and its plugs) raise, naming the
  conversion; an undo revives it and kills the new one (plain message); a new
  scene or a file open clears the record; a chain of conversions; a
  namespaced node.

Every refusal asserts a zero ``cmds.ls()`` delta.
"""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds

import rig
import rig.nodetypes as nodetypes
import rig.nodetypes._base as _base
from rig import (
    AmbiguousNodeError,
    container,
    Layer,
    List,
    Node,
    NodeNotFoundError,
    NodeTypeError,
    shade,
)
from rig.bridges import nodes as rn
from rig.nodetypes import ObjectSet, ShadingEngine, Transform
from rig.shade import (
    Blinn,
    Conversion,
    Default,
    Lambert,
    Material,
    Phong,
    StandardSurface,
)
from rig._tests._base import MayaTestCase


ISG     = "initialShadingGroup"
DEFAULT = 'ShadingEngine("initialShadingGroup")'


def _scene():
    return set(cmds.ls())


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _members(engine):
    return cmds.sets(str(engine), query=True) or []


def _engines(shape):
    return sorted(set(cmds.listConnections(shape, type="shadingEngine") or []))


def _names(nodes):
    return [repr(node) for node in nodes]


def _engine(name):
    return cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=name)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)
        self.cube = _cube("cube")

    def assertRefused(self, error, pattern, call):
        """``call()`` raises ``error`` matching ``pattern`` and writes nothing."""
        before = _scene()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(_scene() - before, set())
        self.assertEqual(before - _scene(), set())


# --------------------------------------------------------------------- #
#  Materials are node classes (REC W3 / W4 / W6)
# --------------------------------------------------------------------- #


class TestMaterialNodes(_Case):
    def test_w3_define_once_and_assign_many(self):
        geos = [self.cube, _cube("b"), _cube("c")]
        red  = Blinn.define("red", color=(1, 0, 0))
        self.assertEqual(repr(red), 'Blinn("red")')
        for geo in geos:
            self.assertIs(geo << red, geo)
        self.assertEqual(sorted(_members("redSG")), ["bShape", "cShape", "cubeShape"])
        # a re-run finds it: 0 writes, the attributes are not re-applied
        before = _scene()
        self.assertEqual(Blinn.define("red", color=(0, 1, 0)), red)
        self.assertEqual(_scene(), before)
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        # the node is the handle: a plain plug
        tex = rn.file(name="tex")
        red.color << tex.outColor
        self.assertEqual(cmds.listConnections("red.color", source=True, plugs=True), ["tex.outColor"])
        # always new, into a scope when asked (the examples' spelling)
        with container("look"):
            m = Lambert.create(name="m", container=True, diffuse=1)
        plane = _cube("plane")
        self.assertIs(plane << m, plane)
        self.assertEqual(repr(m), 'Lambert("m")')
        self.assertEqual(sorted(cmds.container("look", query=True, nodeList=True)),
                         sorted(["m", "mSG", *cmds.listConnections("mSG.message", type="materialInfo")]))
        # a strict assignment: the reference
        self.assertIs(self.cube << Blinn("red"), self.cube)
        # a define of another type: NodeTypeError, 0 writes; astype converts
        cmds.shadingNode("phong", asShader=True, name="shiny")
        self.assertRefused(
            NodeTypeError, r"'shiny' is a phong, not a blinn.*Phong\('shiny'\)\.astype\(Blinn\) converts it",
            lambda: Blinn.define("shiny"),
        )
        self.assertEqual(repr(Phong("shiny").astype(Blinn)), 'Blinn("shiny")')

    def test_w4_queries_and_removals_before_existence_never_create(self):
        for call in (
            lambda: self.cube >> Blinn("red"),
            lambda: self.cube << -Blinn("red"),
            lambda: self.cube << Blinn("red"),
            lambda: self.cube in Blinn("red"),
            lambda: self.cube.f[:2] << -Material("red"),
        ):
            self.assertRefused(NodeNotFoundError, "no (blinn|surface shader) named 'red'", call)
        # still a ValueError, so old handlers catch it
        self.assertRefused(ValueError, "no blinn named 'red'", lambda: self.cube >> Blinn("red"))
        before = _scene()
        self.assertFalse(Blinn.exists("red"))
        self.assertFalse(Material.exists("cube"))
        self.assertFalse(Lambert.exists("initialShadingGroup"))
        self.assertEqual(_names(self.cube >> Material()), [DEFAULT])
        self.assertEqual(self.cube >> Blinn(), [])
        self.assertEqual(_scene(), before)
        self.assertIs(self.cube << Material(), self.cube)
        self.assertEqual(_engines("cubeShape"), [])
        self.assertEqual(self.cube >> Material(), [])
        self.assertRefused(TypeError, r"^None is not a material name; Material\(\) removes all$",
                           lambda: self.cube << Blinn(None))
        self.assertEqual(_scene(), before)

    def test_w6_type_mismatch(self):
        cmds.shadingNode("phong", asShader=True, name="shiny")
        cmds.shadingNode("blinn", asShader=True, name="b")
        self.assertRefused(
            NodeTypeError,
            r"^'shiny' is a phong, not a blinn; Node\('shiny'\) is Phong\(\"shiny\"\); Material\('shiny'\) "
            r"takes any surface shader; Phong\('shiny'\)\.astype\(Blinn\) converts it$",
            lambda: Blinn("shiny"),
        )
        self.assertRefused(NodeTypeError, "'b' is a blinn, not a lambert", lambda: Lambert("b"))
        self.assertEqual(repr(Material("b")), 'Blinn("b")')
        self.assertRefused(NodeTypeError, "'shiny' is a phong, not a blinn", lambda: Blinn.define("shiny"))
        self.assertRefused(NodeTypeError, "'b' is a blinn, not a transform", lambda: Transform("b"))
        self.assertRefused(NodeTypeError, "'b' is a blinn, not a shadingEngine", lambda: ShadingEngine("b"))
        self.assertRefused(NodeTypeError, "'initialShadingGroup' is a shadingEngine, not a surface shader",
                           lambda: Material(ISG))
        self.assertIs(type(ObjectSet(ISG)), ShadingEngine)
        self.assertEqual(ObjectSet(ISG), Default())

    def test_tokens_and_reprs(self):
        red    = Blinn.define("red")
        sg     = red.engine
        before = _scene()
        self.assertEqual(repr(Blinn()), "Blinn()")
        self.assertEqual(repr(Material()), "Material()")
        self.assertEqual(repr(-red), '-Blinn("red")')
        self.assertEqual(repr(-sg), '-ShadingEngine("redSG")')
        self.assertEqual(repr(-Default()), f"-{DEFAULT}")
        self.assertEqual(str(-red), "red")
        self.assertTrue((-red).removes and Material().purges)
        for call, pattern in (
            (lambda: Blinn(None), r"^None is not a material name; Material\(\) removes all$"),
            (lambda: Material(None), r"^None is not a material name; Material\(\) removes all$"),
            (lambda: ShadingEngine(), r"^ShadingEngine\(\) names no node: .*; Material\(\) takes the members "
                                      r"out of every shading engine$"),
            (lambda: ShadingEngine(None), r"^None is not a shadingEngine name; Material\(\) takes the members"),
            (lambda: -(-red), "cannot be negated again"),
            (lambda: -Material(), "double negative"),
            (lambda: -(-sg), "cannot be negated again"),
            (lambda: ~red, "unassigned"),
            (lambda: ~sg, "unassigned"),
            (lambda: ~Blinn(), "unassigned"),
            (lambda: Default("x"), "takes no name"),
            (lambda: Blinn("red", color=(1, 0, 0)), r"Blinn\.define\('red', \.\.\.\) finds or makes it"),
        ):
            with self.subTest(pattern):
                self.assertRefused(TypeError, pattern, call)
        for token in (-red, Blinn(), -sg):
            with self.assertRaisesRegex(AttributeError, "not a material"):
                token.color
        self.assertEqual(_scene(), before)

    def test_of_answers_live_nodes(self):
        red  = Blinn.define("red")
        skin = Lambert.define("skin")
        self.assertEqual(Material.of(self.cube), [Default()])
        self.cube.f[:2] << red
        self.cube.f[2]  << skin
        found = Material.of(self.cube)
        self.assertEqual(_names(found), [DEFAULT, 'Blinn("red")', 'Lambert("skin")'])
        self.assertEqual(found[1], red)
        self.assertEqual(hash(found[1]), hash(red))
        self.assertEqual(_names(Blinn.of(self.cube)), ['Blinn("red")'])
        self.assertEqual(_names(Lambert.of(self.cube)), ['Lambert("skin")'])   # flat: no blinn
        self.assertEqual(_names(StandardSurface.of(self.cube)), ['StandardSurface("standardSurface1")'])
        self.assertEqual(_names(self.cube >> Blinn()), _names(Blinn.of(self.cube)))
        # a shader over two engines holding faces is listed once
        _engine("altSG")
        cmds.connectAttr("red.outColor", "altSG.surfaceShader")
        self.cube.f[3] << ShadingEngine("altSG")
        self.assertEqual(_names(Blinn.of(self.cube)), ['Blinn("red")'])
        # an engine fed by no surface shader is listed by itself (it goes back through <<)
        _engine("rampSG")
        cmds.connectAttr(rn.ramp(name="ramp1").outColor, "rampSG.surfaceShader")
        cmds.sets("cube.f[4]", edit=True, forceElement="rampSG")
        self.assertIn('ShadingEngine("rampSG")', _names(Material.of(self.cube)))
        self.assertEqual(_names(Material.of(self.cube.f[4])), ['ShadingEngine("rampSG")'])
        self.assertEqual(Blinn.of(self.cube.f[4]), [])
        # every answer goes back through <<
        other = _cube("other")
        for node in Material.of(self.cube.f[0]):
            other << node
        self.assertEqual(sorted(_members("redSG")), ["cube.f[0:1]", "otherShape"])

    def test_the_shade_exports_are_the_node_classes(self):
        for name in ("Material", "Lambert", "Blinn", "Phong", "PhongE", "SurfaceShader",
                     "StandardSurface", "OpenPBRSurface"):
            self.assertIs(getattr(shade, name), getattr(nodetypes, name))
        self.assertIsInstance(Default(), ShadingEngine)
        self.assertEqual(str(Default()), ISG)
        self.assertIs(shade.Conversion, Conversion)
        self.assertFalse(hasattr(shade, "_BY_TYPE"))
        self.assertFalse(hasattr(rig, "Blinn"))


# --------------------------------------------------------------------- #
#  A node on the right of << / >> (ADD MN3b)
# --------------------------------------------------------------------- #


class TestNodeRhs(_Case):
    def setUp(self):
        super().setUp()
        self.red = Blinn.define("red")

    def test_assign_remove_query_purge(self):
        red = self.red
        self.assertIs(self.cube << red, self.cube)
        self.assertEqual(_members("redSG"), ["cubeShape"])
        np.testing.assert_array_equal(self.cube >> red, np.arange(6))
        faces = self.cube.f[:2]
        self.assertIs(faces << -red, faces)
        self.assertEqual(_members("redSG"), ["cube.f[2:5]"])
        np.testing.assert_array_equal(self.cube >> red, [2, 3, 4, 5])
        np.testing.assert_array_equal(self.cube >> red.engine, [2, 3, 4, 5])
        self.assertIs(self.cube << Blinn(), self.cube)
        self.assertEqual(_engines("cubeShape"), [])
        self.cube << Default()
        self.assertEqual(_members(ISG), ["cubeShape"])
        # a removal of what is not held is a no-op
        before = _scene()
        self.cube << -red
        self.assertEqual(_scene(), before)
        self.assertEqual(_members(ISG), ["cubeShape"])

    def test_one_undo_chunk_per_lshift(self):
        other = _cube("other")
        cmds.flushUndo()
        with mock.patch.object(cmds, "sets", wraps=cmds.sets) as sets:
            List([self.cube, other.f[:3]]) << self.red
        self.assertEqual(len([c for c in sets.call_args_list if c.kwargs.get("forceElement") == "redSG"]), 1)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        cmds.undo()
        self.assertEqual(_members("redSG"), [])
        self.assertEqual(sorted(_members(ISG)), ["cubeShape", "otherShape"])
        self.cube << self.red
        self.cube.f[:2] << -self.red
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        cmds.undo()
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.cube << Material()
        cmds.undo()
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_a_dead_node_raises_before_any_write(self):
        red = self.red
        self.cube << red
        cmds.delete("red")
        for call in (
            lambda: self.cube << red,
            lambda: self.cube << -red,
            lambda: self.cube >> red,
            lambda: self.cube in red,
            lambda: red.engine,
            lambda: red.astype(Phong),
        ):
            self.assertRefused(RuntimeError, "red already deleted!", call)
        cmds.undo()
        self.assertTrue(red.is_valid)
        self.assertIn(self.cube, red)
        other = _cube("other")
        other << red
        self.assertEqual(sorted(_members("redSG")), ["cubeShape", "otherShape"])
        # a new scene frees it: every use raises, naming the class
        token = -red
        cmds.file(new=True, force=True)
        cube = _cube("cube")
        for call in (lambda: cube << red, lambda: cube << token, lambda: cube in red):
            self.assertRefused(RuntimeError, r"Blinn node \(freed by a new scene", call)
        self.assertRefused(NodeNotFoundError, "no blinn named 'red'", lambda: Blinn("red"))

    def test_a_renamed_node_is_followed(self):
        red   = self.red
        token = -red
        cmds.rename("red", "crimson")
        self.assertEqual(repr(red), 'Blinn("crimson")')
        self.assertEqual(repr(token), '-Blinn("crimson")')
        self.cube << red
        self.assertEqual(_members("redSG"), ["cubeShape"])
        self.cube << token
        self.assertEqual(_members("redSG"), [])
        self.assertRefused(NodeNotFoundError, "no blinn named 'red'", lambda: Blinn("red"))
        # the name reused by a new node refers to the new node
        new = Blinn.define("red")
        self.assertNotEqual(new, red)
        self.cube << new
        self.assertEqual(_engines("cubeShape"), ["redSG1"])

    def test_a_namespaced_node(self):
        cmds.namespace(add="look")
        cmds.shadingNode("blinn", asShader=True, name="look:red")
        look = Blinn("look:red")
        self.cube << look
        self.assertEqual(_members("look:redSG"), ["cubeShape"])
        self.assertEqual(Material.of(self.cube), [look])
        cmds.namespace(setNamespace=":look")
        try:
            # :red and :look:red: the bare name is ambiguous, nothing written
            self.assertRefused(AmbiguousNodeError, "names :red and :look:red", lambda: self.cube << Blinn("red"))
            cmds.namespace(relativeNames=True)
            try:
                self.assertRefused(AmbiguousNodeError, "names :red and :look:red", lambda: self.cube >> Blinn("red"))
            finally:
                cmds.namespace(relativeNames=False)
            self.cube << Blinn("look:red")
            self.assertIn(self.cube, look)
            self.assertNotIn(self.cube, self.red)
        finally:
            cmds.namespace(setNamespace=":")
        self.cube << Blinn("red")
        self.assertEqual(_members("redSG"), ["cubeShape"])

    def test_the_particle_engine_and_a_shaderless_engine_are_refused(self):
        particle = ShadingEngine("initialParticleSE")
        empty    = ShadingEngine(_engine("emptySG"))
        for call in (
            lambda: self.cube << particle,
            lambda: self.cube << -particle,
            lambda: self.cube >> particle,
            lambda: self.cube in particle,
        ):
            self.assertRefused(TypeError, "is the particle engine, not a material", call)
        for call in (
            lambda: self.cube << empty,
            lambda: self.cube.f[:2] << -empty,
            lambda: self.cube >> empty,
            lambda: self.cube in empty,
        ):
            self.assertRefused(ValueError, "shading engine 'emptySG' has no surface shader", call)
        self.assertEqual(_members(ISG), ["cubeShape"])

    def test_a_shader_over_two_engines(self):
        red = self.red
        _engine("altSG")
        cmds.connectAttr("red.outColor", "altSG.surfaceShader")
        alt = ShadingEngine("altSG")
        # the shader goes to <shader>SG; the engine node is exactly that engine
        self.cube.f[:2] << red
        self.assertEqual(_members("redSG"), ["cube.f[0:1]"])
        self.cube.f[2:4] << alt
        self.assertEqual(_members("altSG"), ["cube.f[2:3]"])
        np.testing.assert_array_equal(self.cube >> alt, [2, 3])
        np.testing.assert_array_equal(self.cube >> red, [0, 1])
        self.assertIn(self.cube.f[2:4], alt)
        self.assertNotIn(self.cube.f[2:4], red)
        self.assertEqual(Material.of(self.cube.f[2]), [red])   # the shader (not a round trip: ADD C1)
        self.cube.f[2:4] << -alt
        self.assertEqual(_members("altSG"), [])
        # with no engine named <shader>SG the shader raises before any write
        _engine("otherSG")
        cmds.connectAttr("red.outColor", "otherSG.surfaceShader")
        cmds.delete("redSG")
        for call in (
            lambda: self.cube << red, lambda: self.cube >> red, lambda: self.cube in red,
            lambda: self.cube << -red,
        ):
            self.assertRefused(ValueError, "feeds 2 shading engines", call)
        self.cube << alt
        self.assertEqual(_members("altSG"), ["cubeShape"])

    def test_an_empty_list_is_a_value_error(self):
        for rhs in (self.red, -self.red, self.red.engine, Material()):
            self.assertRefused(ValueError, "nothing to inject", lambda rhs=rhs: List([]) << rhs)
        self.assertRefused(ValueError, "nothing to inject", lambda: List([]) >> self.red)

    def test_a_member_node_on_the_left_writes_nothing(self):
        sg     = self.red.engine
        layer  = Layer.define("L")
        wiring = sorted(cmds.listConnections("redSG", connections=True, plugs=True) or [])
        for call in (
            lambda: sg << self.red,
            lambda: layer << self.red,
            lambda: self.red << sg,
            lambda: self.red << -self.red,
            lambda: Default() << self.red,
            lambda: List([sg, self.cube]) << self.red,
        ):
            self.assertRefused(TypeError, "not geometry", call)
        self.assertEqual(sorted(cmds.listConnections("redSG", connections=True, plugs=True) or []), wiring)
        self.assertEqual(_members(ISG), ["cubeShape"])

    def test_lambert1_builds_lambert1sg(self):
        self.assertIsNone(cmds.listConnections("lambert1", type="shadingEngine"))
        self.cube << Lambert("lambert1")
        self.assertEqual(_members("lambert1SG"), ["cubeShape"])
        self.assertEqual(len(cmds.listConnections("lambert1SG.message", type="materialInfo")), 1)
        other  = _cube("other")
        before = _scene()
        other << Node("lambert1")
        self.assertEqual(_scene() - before, set())
        self.assertEqual(sorted(_members("lambert1SG")), ["cubeShape", "otherShape"])
        self.assertEqual(Lambert("lambert1").engine, ShadingEngine("lambert1SG"))
        self.assertNotEqual(Default(), ShadingEngine("lambert1SG"))

    def test_other_nodes_keep_todays_message(self):
        plain = Node(cmds.sets(empty=True, name="plainSet"))
        ramp  = rn.ramp(name="ramp1")
        grp   = Node.create("transform", name="grp")
        for rhs in (plain, ramp, grp):
            self.assertRefused(TypeError, "Cannot inject Node into a bare Node", lambda rhs=rhs: self.cube << rhs)
        # a plug >> a node still clones, except onto a membership node
        grp << rig.Float("w")
        clone = grp.w >> Node.create("transform", name="dst")
        self.assertEqual(str(clone), "dst.w")
        self.assertRefused(TypeError, "is a plug; membership takes the node", lambda: grp.w >> self.red)
        self.assertRefused(TypeError, "is a plug; membership takes the node", lambda: grp.w >> self.red.engine)

    def test_inside_a_container_scope(self):
        rn.blinn(name="bare", container=False)
        with container("look"):
            self.cube << Blinn("bare")
            inner = Blinn.define("inner")
            self.cube.f[:2] << inner
        self.assertIsNone(cmds.container("look", query=True, nodeList=True))
        self.assertEqual(str(inner), "inner")
        self.assertIsNone(cmds.container(query=True, findContainer=["bareSG"]))
        self.assertEqual(_members("innerSG"), ["cube.f[0:1]"])
        self.assertEqual(_members("bareSG"), ["cube.f[2:5]"])


# --------------------------------------------------------------------- #
#  'in': every face, or the whole object
# --------------------------------------------------------------------- #


class TestMaterialIn(_Case):
    def setUp(self):
        super().setUp()
        self.red   = Blinn.define("red")
        self.other = _cube("other")

    def test_every_face_or_the_whole_object(self):
        red = self.red
        before = _scene()
        self.assertNotIn(self.cube, red)
        self.assertIn(self.cube, Default())
        self.assertEqual(_scene(), before)
        self.cube << red
        self.assertIn(self.cube, red)
        self.assertIn(self.cube.f[:3], red)
        self.assertIn(self.cube.tx, red)          # a plug stands for its node
        self.assertIn(Node("cubeShape"), red)
        self.assertNotIn(self.cube, Default())
        self.cube.f[:2] << -red
        self.assertNotIn(self.cube, red)          # all members: not every face
        self.assertIn(self.cube.f[2:], red)
        self.assertNotIn(self.cube.f[1:3], red)
        self.assertTrue(self.cube.f[:2] not in red)
        # a list: every element
        self.other << red
        self.assertIn(List([self.cube.f[2:], self.other]), red)
        self.assertIn([self.cube.f[3], self.other.tx], red)
        self.assertNotIn(List([self.cube, self.other]), red)
        # a surface: the whole object
        srf = Node(cmds.sphere(name="srf", ch=False)[0])
        self.assertNotIn(srf, red)
        srf << red
        self.assertIn(srf, red)

    def test_the_engine_rule(self):
        _engine("altSG")
        cmds.connectAttr("red.outColor", "altSG.surfaceShader")
        self.cube << ShadingEngine("altSG")
        self.assertIn(self.cube, ShadingEngine("altSG"))
        self.assertNotIn(self.cube, self.red)      # red's engine is redSG
        self.cube << self.red
        self.assertIn(self.cube, self.red)
        self.assertNotIn(self.cube, ShadingEngine("altSG"))
        _engine("otherSG")
        cmds.connectAttr("red.outColor", "otherSG.surfaceShader")
        cmds.delete("redSG")
        self.assertRefused(ValueError, "feeds 2 shading engines", lambda: self.cube in self.red)

    def test_tokens_and_wrong_kinds_are_refused(self):
        crv = Node(cmds.circle(name="crv", ch=False)[0])
        for error, pattern, call in (
            (TypeError, r"'in' asks about one collection; -Blinn\(\"red\"\) is a removal", lambda: self.cube in -self.red),
            (TypeError, r"'in' asks about one collection; Material\(\) names every one", lambda: self.cube in Material()),
            (TypeError, "materials bind faces or whole objects", lambda: self.cube.vtx[:2] in self.red),
            (TypeError, "nurbsCurve, not a shadeable", lambda: crv in self.red),
            (TypeError, "not geometry", lambda: Node("lambert1") in self.red),
            (NodeNotFoundError, "no blinn named 'nope'", lambda: self.cube in Blinn("nope")),
            (ValueError, "nothing to inject", lambda: List([]) in self.red),
        ):
            with self.subTest(pattern):
                self.assertRefused(error, pattern, call)

    def test_a_shader_with_no_engine_holds_nothing(self):
        rn.blinn(name="bare")
        before = _scene()
        self.assertNotIn(self.cube, Blinn("bare"))
        self.assertEqual((self.cube >> Blinn("bare")).shape, (0,))
        self.assertEqual(_scene(), before)   # 'in' never builds the engine


# --------------------------------------------------------------------- #
#  Conversion returns the new node
# --------------------------------------------------------------------- #


class TestConversionReturnsNode(_Case):
    def setUp(self):
        super().setUp()
        self.red = Blinn.define("red", color=(1, 0, 0))
        self.cube << self.red

    def test_astype_and_convert_return_the_new_node(self):
        new = self.red.astype(Phong)
        self.assertIs(type(new), Phong)
        self.assertEqual(repr(new), 'Phong("red")')
        self.assertEqual(new, Node("red"))
        self.assertEqual(new, Phong("red"))
        self.assertIn(self.cube, new)
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        lam = shade.convert("red", "lambert")         # a name, a type name
        self.assertEqual(repr(lam), 'Lambert("red")')
        std = shade.convert(lam, StandardSurface)     # a node, a class
        self.assertEqual(repr(std), 'StandardSurface("red")')
        ramp = std.astype("rampShader")               # a type without a class
        self.assertIs(type(ramp), Material)
        self.assertEqual(cmds.nodeType("red"), "rampShader")
        self.assertEqual(Material.of(self.cube), [ramp])

    def test_dry_run_returns_the_report_and_writes_nothing(self):
        self.red.eccentricity << 0.6
        before = _scene()
        with mock.patch.object(cmds, "warning") as warn:
            report = self.red.astype(Phong, dry_run=True)
            same   = shade.convert("red", "phong", dry_run=True)
        self.assertEqual(warn.call_count, 0)
        self.assertEqual(_scene(), before)
        self.assertIsInstance(report, Conversion)
        self.assertEqual(report, same)
        self.assertEqual(report.parked, ("eccentricity",))
        self.assertEqual(str(report).splitlines()[0], "rig.shade: 'red' blinn -> phong parks:")
        self.assertTrue(self.red.is_valid)
        self.assertEqual(cmds.nodeType("red"), "blinn")
        # a shader already of the type: an empty report
        self.assertEqual(self.red.astype(Blinn, dry_run=True), Conversion("red", "blinn", "blinn"))
        with mock.patch.object(cmds, "warning") as warn:
            new = self.red.astype(Phong)
        self.assertEqual(warn.call_args[0][0], str(report))
        self.assertAlmostEqual(cmds.getAttr("red.__eccentricity__"), 0.6, places=5)
        self.assertIs(type(new), Phong)

    def test_strict_refuses_a_lossy_conversion(self):
        self.red.eccentricity << 0.6
        report = self.red.astype(Phong, dry_run=True)
        self.assertRefused(ValueError, "^rig.shade: 'red' blinn -> phong parks:",
                           lambda: self.red.astype(Phong, strict=True))
        with self.assertRaises(ValueError) as caught:
            shade.convert(self.red, "phong", strict=True)
        self.assertEqual(str(caught.exception), str(report))
        self.assertTrue(self.red.is_valid)
        self.red.eccentricity << 0.3
        self.assertIs(type(self.red.astype(Phong, strict=True)), Phong)

    def test_the_generic_material_and_other_targets_are_refused(self):
        wiring = sorted(cmds.listConnections("red", connections=True, plugs=True) or [])
        for error, pattern, target in (
            (TypeError, r"^Material names no node type", Material),
            (TypeError, r"^a conversion target is a shader class \(Phong\) or a node type name", Transform),
            (TypeError, r"^a conversion target", ""),
            (TypeError, r"^a conversion target", None),
            (TypeError, "not a surface shader", "multiplyDivide"),
            (ValueError, "not a registered surface shader", "noSuchType"),
        ):
            with self.subTest(target=target):
                self.assertRefused(error, pattern, lambda target=target: self.red.astype(target))
        self.assertRefused(TypeError, "membership token", lambda: shade.convert(-self.red, Phong))
        self.assertRefused(TypeError, "membership token", lambda: shade.convert(Blinn(), Phong))
        self.assertRefused(RuntimeError, "default node", lambda: shade.convert("lambert1", Phong))
        self.assertEqual(sorted(cmds.listConnections("red", connections=True, plugs=True) or []), wiring)
        self.assertTrue(self.red.is_valid)

    def test_a_reference_never_converts(self):
        for call in (lambda: Phong(self.red), lambda: Phong("red"), lambda: Phong(Node("red").color)):
            self.assertRefused(
                NodeTypeError, r"^'red' is a blinn, not a phong; .*; Blinn\('red'\)\.astype\(Phong\) converts it$",
                call,
            )
        self.assertEqual(cmds.nodeType("red"), "blinn")
        self.assertTrue(self.red.is_valid)

    def test_a_shader_of_the_type_is_returned_as_is(self):
        cmds.flushUndo()
        before = _scene()
        self.assertIs(self.red.astype(Blinn), self.red)
        self.assertIs(self.red.astype("blinn", color=(0, 1, 0)), self.red)
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "")
        self.assertIs(self.red.astype(Blinn, update=True, color=(0, 1, 0)), self.red)
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 1.0, 0.0))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        self.assertRefused(AttributeError, "no attribute 'nope'", lambda: self.red.astype(Blinn, nope=1))
        self.assertEqual(_scene(), before)


# --------------------------------------------------------------------- #
#  The held old node names the conversion
# --------------------------------------------------------------------- #


class TestConvertedHandle(_Case):
    CONVERTED = r"^'red' was converted to a phong; use the node astype\(\) returned \(red__rigold already deleted!\)$"

    def setUp(self):
        super().setUp()
        self.red = Blinn.define("red")
        self.cube << self.red

    def test_the_held_node_names_the_conversion(self):
        held  = self.red
        plug  = held.color
        token = -held
        new   = held.astype(Phong)
        for label, call in (
            ("str", lambda: str(held)), ("repr", lambda: repr(held)), ("name", lambda: held.name),
            ("plug", lambda: held.color.get()), ("get", lambda: plug.get()),
            ("set", lambda: plug << (1, 0, 0)), ("engine", lambda: held.engine),
            ("astype", lambda: held.astype(Lambert)), ("<<", lambda: self.cube << held),
            ("<< token", lambda: self.cube << token), ("in", lambda: self.cube in held),
            ("refer", lambda: Phong(held)), ("convert", lambda: shade.convert(held, "lambert")),
            ("delete", lambda: held.delete()),
        ):
            with self.subTest(label):
                self.assertRefused(RuntimeError, self.CONVERTED, call)
        self.assertFalse(held.is_valid)
        self.assertEqual(held == new, False)
        # a node deleted another way keeps the plain message
        other = Blinn.define("other")
        cmds.delete("other")
        self.assertRefused(RuntimeError, r"^other already deleted!$", lambda: str(other))
        self.assertIs(type(new), Phong)
        self.assertIn(self.cube, new)

    def test_an_undo_revives_the_old_node_and_kills_the_new(self):
        held = self.red
        new  = held.astype(Phong)
        cmds.undo()
        self.assertTrue(held.is_valid)
        self.assertEqual(held, Blinn("red"))
        self.assertIn(self.cube, held)
        self.assertRefused(RuntimeError, r"^red__rigconvert already deleted!$", lambda: str(new))
        cmds.redo()
        self.assertRefused(RuntimeError, self.CONVERTED, lambda: str(held))
        self.assertEqual(Node("red"), Phong("red"))
        self.assertIs(type(Node("red")), Phong)

    def test_a_new_scene_or_a_file_open_clears_the_record(self):
        held = self.red
        held.astype(Phong)
        self.assertEqual(_base._CONVERTED.get("red__rigold"), ("red", "phong"))
        cmds.file(new=True, force=True)
        self.assertEqual(_base._CONVERTED, {})
        self.assertRefused(RuntimeError, r"^Blinn node \(freed by a new scene", lambda: str(held))
        # a file open too
        folder = tempfile.mkdtemp(prefix="rig_nc7_open_")
        path   = os.path.join(folder, "nc7.ma")
        try:
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            Blinn.define("red").astype(Phong)
            self.assertIn("red__rigold", _base._CONVERTED)
            cmds.file(path, open=True, force=True)
            self.assertEqual(_base._CONVERTED, {})
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)
        # a node later deleted under the old aside name keeps the plain message
        cmds.createNode("transform", name="red__rigold")
        node = Node("red__rigold")
        cmds.delete("red__rigold")
        self.assertRefused(RuntimeError, r"^red__rigold already deleted!$", lambda: str(node))

    def test_a_chain_of_conversions(self):
        first  = self.red
        second = first.astype(Phong)
        third  = second.astype(Lambert)
        self.assertEqual(repr(third), 'Lambert("red")')
        for held in (first, second):
            self.assertRefused(RuntimeError, r"^'red' was converted to a lambert; use the node astype\(\) returned",
                               lambda held=held: str(held))
        self.assertIn(self.cube, third)
        cmds.undo()
        self.assertTrue(second.is_valid)
        self.assertRefused(RuntimeError, "already deleted", lambda: str(third))

    def test_a_namespaced_node(self):
        cmds.namespace(add="look")
        cmds.shadingNode("blinn", asShader=True, name="look:red")
        held = Blinn("look:red")
        new  = held.astype(Phong)
        self.assertEqual(repr(new), 'Phong("look:red")')
        self.assertRefused(RuntimeError, r"^'look:red' was converted to a phong; use the node astype\(\) returned",
                           lambda: str(held))
        # the root red is untouched
        self.assertTrue(self.red.is_valid)
        self.assertEqual(cmds.nodeType("red"), "blinn")
