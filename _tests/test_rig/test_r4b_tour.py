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

The tour: the recommendation's section 4, workflows W1 to W8, end to end with
the user's spellings (yes / no with ``in``, ids with ``>>``), each row
asserted with its ``cmds.ls()`` delta:

* ``TestTourW1``: an existing skeleton or geometry is referred to: a typo (the
  hint), a name two nodes have, the namespace rule, a referenced file define
  never builds into, the geometry hooks.
* ``TestTourW2``: ``build()`` twice makes no new transform; one base name
  under two parents is two keys; a nested name is not forked; create is always
  new and takes keywords; a container re-run and another module's node are
  refused by define and referred to instead.
* ``TestTourW3``: materials and layers are defined once and assigned to many;
  a re-run finds them and keeps their values; the node is the handle; create
  is always new; a strict assignment; faces and ids; a plug on the left.
* ``TestTourW4``: queries and removals before a material or layer exists
  raise at the reference and never create; ``exists``; the kind tokens;
  ``None`` refused.
* ``TestTourW5``: math in a container makes no lookup; ``define`` is the only
  one; a flattened scope prefixes the key.
* ``TestTourW6``: the type-mismatch rows.
* ``TestTourW7``: undo and redo, a new scene, held nodes, a recipe function,
  two undo steps for a define then an assignment, conversion returning the new
  node, a failed define deleting what it made.
* ``TestTourW8``: ``Tag`` unchanged; re-declaring a ``Float`` keeps its value
  and its wire.
"""

import importlib
import inspect
import os
import shutil
import tempfile
from unittest import mock

import cgmath.geometry
import numpy as np
from maya import cmds

import rig
import rig._internal.members as _members
from rig import (
    AmbiguousNodeError,
    container,
    Enum,
    Float,
    Layer,
    List,
    MeshAttr,
    Node,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
    NurbsCurveAttr,
    NurbsSurfaceAttr,
    shade,
    Tag,
)
from rig.bridges import nodes as rn
from rig.nodetypes import (
    DAGNode,
    DisplayLayer,
    Joint,
    Mesh,
    NurbsCurve,
    NurbsSurface,
    ObjectSet,
    ShadingEngine,
    Transform,
)
from rig.shade import Blinn, Default, Lambert, Material, Phong
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


def _transforms():
    return set(cmds.ls(type="transform", long=True))


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _engines(shape):
    return sorted(set(cmds.listConnections(shape, type="shadingEngine") or []))


def _layer_of(node):
    return cmds.listConnections(f"{node}.drawOverride", source=True, type="displayLayer")


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)

    def tearDown(self):
        cmds.namespace(relativeNames=False)
        cmds.namespace(setNamespace=":")
        super().tearDown()

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


# --------------------------------------------------------------------------- #
#  The tour: REC section 4, W1 to W8, with the user's spellings
# --------------------------------------------------------------------------- #


class TestTourW1(_Case):
    """W1: refer to an existing skeleton or geometry: a typo, a namespace, a
    referenced file. Naming never creates."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = tempfile.mkdtemp(prefix="rig_tour_")
        cls.path   = os.path.join(cls.folder, "char.ma").replace("\\", "/")
        cmds.file(new=True, force=True)
        cmds.createNode("joint", name="root")
        cmds.createNode("joint", name="spine_01", parent="root")
        cmds.createNode("transform", name="x")
        cmds.file(rename=cls.path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)

    @classmethod
    def tearDownClass(cls):
        cmds.file(new=True, force=True)
        shutil.rmtree(cls.folder, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        """The referenced character (``char:root|spine_01``, ``char:x``) and the
        scene's own nodes: a joint ``spine_01``, ``|g1|a`` and ``|g2|a``, ``x``,
        a cube ``body`` and an instanced cube ``box`` / ``box_inst``."""
        super().setUp()
        cmds.file(self.path, reference=True, namespace="char")
        cmds.createNode("joint", name="spine_01")
        for group in ("g1", "g2"):
            cmds.createNode("transform", name=group)
            cmds.createNode("transform", name="a", parent=group)
        cmds.createNode("transform", name="x")
        cmds.polyCube(name="body", ch=False)
        cmds.polyCube(name="box", ch=False)
        cmds.instance("box", name="box_inst")

    def assertFound(self, call):
        """``call()`` returns a node and writes nothing."""
        before = _scene()
        node   = call()
        self.assertEqual(_scene(), before)
        return node

    def test_a_typo_raises_with_a_hint_and_writes_nothing(self):
        self.assertEqual(repr(Joint("spine_01")), 'Joint("spine_01")')
        # the scene's own spine_01 is the hint, not the reference's char:spine_01
        self.assertRefused(NodeNotFoundError, r"^no joint named 'spnie_01' \(did you mean 'spine_01'\?\)$",
                           lambda: Joint("spnie_01"))
        self.assertRefused(NodeNotFoundError, r"^no node named 'spnie_01' \(did you mean 'spine_01'\?\)$",
                           lambda: Node("spnie_01"))
        self.assertRefused(NodeNotFoundError,
                           r"^no joint named 'char:spnie_01' \(did you mean 'char:spine_01'\?\)$",
                           lambda: Joint("char:spnie_01"))
        # one family, still caught by the handlers written for the old errors
        for old in (ValueError, TypeError, LookupError):
            with self.subTest(handler=old.__name__):
                self.assertRefused(old, "spnie_01", lambda: Joint("spnie_01"))

    def test_a_name_two_nodes_have_is_ambiguous(self):
        self.assertRefused(AmbiguousNodeError,
                           r"^'a' is ambiguous: it names 2 nodes: \|g[12]\|a, \|g[12]\|a; use a path$",
                           lambda: Transform("a"))
        self.assertEqual(Transform("g1|a").long_name, "|g1|a")
        self.assertEqual(Transform("|g2|a").long_name, "|g2|a")

    def test_one_lookup_rule_for_the_namespaces(self):
        self.assertRefused(NodeNotFoundError, r"^no joint named 'root' \('char:root' exists\)$",
                           lambda: Joint("root"))
        root = Joint("char:root")
        self.assertEqual(repr(root), 'Joint("char:root")')
        cmds.namespace(setNamespace=":char")
        for relative in (False, True):
            with self.subTest(relativeNames=relative):
                cmds.namespace(relativeNames=relative)
                self.assertEqual(Joint("root"), root)
                self.assertRefused(
                    AmbiguousNodeError,
                    r"^'x' is ambiguous: it names :x and :char:x; spell the namespace$",
                    lambda: Node("x"),
                )
                self.assertEqual(Node(":x").uuid, cmds.ls(":x", uuid=True)[0])
                self.assertEqual(Node(":char:x").uuid, cmds.ls(":char:x", uuid=True)[0])

    def test_define_never_builds_in_the_referenced_file(self):
        found = self.assertFound(lambda: Joint.define("char:root"))
        self.assertEqual(found, Joint("char:root"))
        self.assertTrue(cmds.referenceQuery(str(found), isNodeReferenced=True))
        in_ref = (r"the namespace 'char' belongs to the file reference charRN \(.*char\.ma\); "
                  r"define never makes a node there")
        self.assertRefused(ValueError, r"^Joint\.define\('char:spnie_01'\): " + in_ref,
                           lambda: Joint.define("char:spnie_01"))
        cmds.namespace(setNamespace=":char")
        self.assertRefused(ValueError, r"^Joint\.define\('newj'\): " + in_ref, lambda: Joint.define("newj"))
        cmds.namespace(setNamespace=":")
        self.assertRefused(ValueError,
                           r"^Joint\.define\('chr:root'\): there is no namespace 'chr'; define never creates one",
                           lambda: Joint.define("chr:root"))
        self.assertFalse(cmds.namespace(exists=":chr"))

    def test_the_geometry_hooks_and_an_instance(self):
        self.assertEqual(repr(Mesh("body")), 'Mesh("bodyShape")')
        self.assertEqual(Mesh("body"), Node("bodyShape"))
        # an instanced shape is one node: no ambiguity; a transform names its path
        self.assertEqual(repr(Mesh("boxShape")), 'Mesh("box|boxShape")')
        self.assertEqual(repr(Mesh("box_inst")), 'Mesh("box_inst|boxShape")')
        self.assertRefused(NodeTypeError, r"^'x' is a transform, not a mesh.*which has no mesh shape$",
                           lambda: Mesh("x"))


class TestTourW2(_Case):
    """W2: controls and groups under a parent, a re-run in the same scene, one
    base name twice on purpose."""

    @staticmethod
    def build():
        root = Transform.define("rig")
        ctls = []
        for side in "LR":
            arm = Transform.define(f"{side}_arm", parent=root)
            ctl = Transform.define("ctl", parent=arm, tx=1)
            ctl << Float("fk_ik", min=0, max=1)
            ctls.append(ctl)
        riders = [rn.joint(name="rider1") for _ in range(3)]
        return root, ctls, riders

    def test_a_rerun_makes_no_new_transform(self):
        root, ctls, riders = self.build()
        self.assertEqual([c.long_name for c in ctls], ["|rig|L_arm|ctl", "|rig|R_arm|ctl"])
        self.assertEqual([str(r) for r in riders], ["rider1", "rider2", "rider3"])
        cmds.setAttr("|rig|L_arm|ctl.tx", 3)
        cmds.createNode("transform", name="drv")
        cmds.connectAttr("drv.tx", "|rig|L_arm|ctl.fk_ik")
        before = _transforms()
        again, ctls_again, riders_again = self.build()
        # the defined anchors are found; only the always-new riders are new
        self.assertEqual((again, ctls_again), (root, ctls))
        self.assertEqual(_transforms() - before, {"|rider4", "|rider5", "|rider6"})
        self.assertEqual([str(r) for r in riders_again], ["rider4", "rider5", "rider6"])
        # found values are not re-applied; the re-declared fk_ik keeps its wire
        self.assertEqual(cmds.getAttr("|rig|L_arm|ctl.tx"), 3.0)
        self.assertEqual(cmds.listConnections("|rig|L_arm|ctl.fk_ik", source=True, plugs=True), ["drv.translateX"])
        self.assertEqual(cmds.attributeQuery("fk_ik", node="|rig|L_arm|ctl", range=True), [0.0, 1.0])

    def test_one_base_name_under_two_parents(self):
        self.build()
        self.assertEqual([cmds.getAttr(f"|rig|{s}_arm|ctl.tx") for s in "LR"], [1.0, 1.0])
        self.assertRefused(
            AmbiguousNodeError,
            r"^'ctl' is ambiguous: it names 2 nodes: \|rig\|[LR]_arm\|ctl, \|rig\|[LR]_arm\|ctl; use a path$",
            lambda: Transform("ctl"),
        )
        self.assertRefused(
            AmbiguousNodeError,
            r"^Transform\.define\('ctl'\): 'ctl' already names \|rig\|[LR]_arm\|ctl and \|rig\|[LR]_arm\|ctl; "
            r"a transform at the key \|ctl would be one more: pass parent= to key one of them$",
            lambda: Transform.define("ctl"),
        )
        before = _scene()
        left = Transform.define("ctl", parent="|rig|L_arm", tx=5, update=True)
        self.assertEqual(_scene(), before)
        self.assertEqual((left.long_name, cmds.getAttr("|rig|L_arm|ctl.tx")), ("|rig|L_arm|ctl", 5.0))
        self.assertEqual(Transform("L_arm|ctl"), left)

    def test_a_nested_name_is_not_forked(self):
        cmds.createNode("transform", name="char_grp")
        cmds.createNode("joint", name="root", parent="char_grp")
        self.assertRefused(
            AmbiguousNodeError,
            r"^Joint\.define\('root'\): 'root' exists at \|char_grp\|root; Joint\('root'\) refers to it, "
            r"and Joint\.define\('root', parent='char_grp'\) keys it there$",
            lambda: Joint.define("root"),
        )
        before = _scene()
        self.assertEqual(Joint.define("root", parent="char_grp"), Joint("root"))
        self.assertEqual(_scene(), before)

    def test_create_is_always_new_and_takes_keywords(self):
        root = Transform.define("rig")
        made = [Transform.create(name="tmp", parent=root) for _ in range(2)]
        self.assertEqual([m.long_name for m in made], ["|rig|tmp", "|rig|tmp1"])
        self.assertRefused(TypeError, r"^Transform\.create\(\) takes name= and parent= as keywords \(got 'tmp'\)$",
                           lambda: Transform.create("tmp"))
        self.assertRefused(NodeNotFoundError, r"^no DAG node named 'nosuch'$",
                           lambda: Transform.define("x", parent="nosuch"))
        self.assertEqual([str(rn.joint(name="rider1")) for _ in range(2)], ["rider1", "rider2"])

    def test_a_container_rerun_is_refused(self):
        def build():
            with container("arm") as arm:
                Transform.define("arm_root")
            return arm

        first = build()
        with container("arm") as second:
            self.assertEqual(str(second), "arm1")
            self.assertRefused(
                ValueError,
                r"^'arm_root' belongs to container 'arm' from an earlier run; this scope is 'arm1'\. "
                r"Delete 'arm' to rebuild it, or build in a new scene\.$",
                lambda: Transform.define("arm_root"),
            )
        self.assertEqual(cmds.container(str(first), query=True, nodeList=True), ["arm_root"])
        self.assertIsNone(cmds.container(str(second), query=True, nodeList=True))

    def test_another_modules_node_is_referred_to(self):
        with container("arm"):
            root = Transform.define("rig_root")
        with container("leg") as leg:
            self.assertRefused(
                ValueError,
                r"^'rig_root' belongs to container 'arm'; define inside 'leg' only finds nodes this build "
                r"scope owns\. Refer to it with Transform\('rig_root'\), or define it outside the containers\.$",
                lambda: Transform.define("rig_root"),
            )
            self.assertEqual(Transform("rig_root"), root)
        self.assertIsNone(cmds.container(str(leg), query=True, nodeList=True))


class TestTourW3(_Case):
    """W3: materials and layers are defined once, with attributes, and assigned
    to many; a re-run finds them."""

    def setUp(self):
        super().setUp()
        self.geos = [_cube(f"geo{i}") for i in range(3)]

    def test_define_once_and_assign_many(self):
        before = _scene()
        red    = Blinn.define("red", color=(1, 0, 0))
        self.assertEqual(repr(red), 'Blinn("red")')
        made = _scene() - before
        self.assertEqual(len(made), 3)
        self.assertEqual({cmds.nodeType(n) for n in made}, {"blinn", "shadingEngine", "materialInfo"})
        self.assertEqual({"red", "redSG"} - made, set())
        before = _scene()
        for geo in self.geos:
            self.assertIs(geo << red, geo)
        self.assertEqual(_scene(), before)
        self.assertEqual([geo in red for geo in self.geos], [True, True, True])
        np.testing.assert_array_equal(self.geos[0] >> red, np.arange(6))
        self.assertEqual(Material.of(self.geos[1]), [red])
        self.assertEqual(self.geos[2] >> Material(), [red])
        self.assertEqual(sorted(cmds.sets("redSG", query=True)), ["geo0Shape", "geo1Shape", "geo2Shape"])

    def test_a_rerun_finds_the_network_and_keeps_its_values(self):
        red = Blinn.define("red", color=(1, 0, 0))
        for geo in self.geos:
            geo << red
        before = _scene()
        self.assertEqual(Blinn.define("red", color=(0, 1, 0)), red)
        for geo in self.geos:
            geo << Blinn.define("red", color=(0, 1, 0))
        self.assertEqual(_scene(), before)
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        self.assertEqual(sorted(cmds.sets("redSG", query=True)), ["geo0Shape", "geo1Shape", "geo2Shape"])
        self.assertEqual(Blinn.define("red", color=(0, 1, 0), update=True), red)
        self.assertEqual(cmds.getAttr("red.color")[0], (0.0, 1.0, 0.0))
        self.assertEqual(_scene(), before)

    def test_a_layer_is_defined_and_takes_a_list(self):
        before = _scene()
        proxy  = Layer.define("proxy", displayType=2)
        self.assertEqual((repr(proxy), _scene() - before), ('DisplayLayer("proxy")', {"proxy"}))
        self.assertIs(Layer, DisplayLayer)
        self.assertEqual(List(self.geos) << proxy, List(self.geos))
        self.assertEqual([geo in proxy for geo in self.geos], [True, True, True])
        self.assertTrue(self.geos[0].tx in proxy)     # a plug stands for its node
        self.assertEqual(self.geos[1] >> Layer(), proxy)
        self.assertEqual(DisplayLayer.of(self.geos[2]), [proxy])
        # a re-run finds it and keeps its values
        self.assertEqual(Layer.define("proxy", displayType=0), proxy)
        self.assertEqual(cmds.getAttr("proxy.displayType"), 2)
        self.assertEqual(_scene() - before, {"proxy"})

    def test_the_node_is_the_handle(self):
        red = Blinn.define("red")
        tex = rn.checker(name="tex")
        red.color << tex.outColor
        self.assertEqual(cmds.listConnections("red.color", source=True, plugs=True), ["tex.outColor"])
        self.assertEqual(red, Node("red"))
        self.assertEqual(red, Blinn("red"))
        self.assertEqual(repr(red.engine), 'ShadingEngine("redSG")')

    def test_create_is_always_new(self):
        made = [Lambert.create(name="m", diffuse=1) for _ in range(2)]
        self.assertEqual([repr(m) for m in made], ['Lambert("m")', 'Lambert("m1")'])
        self.assertEqual([repr(m.engine) for m in made], ['ShadingEngine("mSG")', 'ShadingEngine("m1SG")'])
        plane = _cube("plane")
        self.assertIs(plane << made[1], plane)
        self.assertEqual((plane in made[1], plane in made[0]), (True, False))
        self.assertEqual(cmds.getAttr("m1.diffuse"), 1.0)

    def test_a_strict_assignment_and_another_type(self):
        cube = self.geos[0]
        self.assertRefused(NodeNotFoundError, r"^no blinn named 'rde'$", lambda: cube << Blinn("rde"))
        self.assertEqual(_engines("geo0Shape"), ["initialShadingGroup"])
        cmds.shadingNode("phong", asShader=True, name="shiny")
        self.assertRefused(
            NodeTypeError,
            r"^'shiny' is a phong, not a blinn; Node\('shiny'\) is Phong\(\"shiny\"\); Material\('shiny'\) "
            r"takes any surface shader; Phong\('shiny'\)\.astype\(Blinn\) converts it$",
            lambda: Blinn.define("shiny"),
        )
        shiny = Phong("shiny").astype(Blinn)
        self.assertEqual(repr(shiny), 'Blinn("shiny")')
        self.assertIs(cube << shiny, cube)
        self.assertTrue(cube in Blinn("shiny"))

    def test_faces_all_members_and_ids(self):
        cube = self.geos[0]
        blue = Blinn.define("blue")
        cube.f[:3] << blue
        self.assertEqual((cube.f[:3] in blue, cube.f[:4] in blue, cube in blue), (True, False, False))
        np.testing.assert_array_equal(cube >> blue, [0, 1, 2])
        np.testing.assert_array_equal(cube.f >> blue, [0, 1, 2])
        cube.f[1] << -blue
        np.testing.assert_array_equal(cube >> blue, [0, 2])
        cube << Default()
        self.assertEqual((cube >> blue).tolist(), [])

    def test_a_plug_on_the_left_is_refused(self):
        cube, other = self.geos[0], self.geos[1]
        red   = Blinn.define("red")
        proxy = Layer.define("proxy")
        self.assertRefused(TypeError,
                           r"^'geo0\.translateX' is a plug; membership takes the node: geo0 << Blinn\(\"red\"\)",
                           lambda: cube.tx << red)
        self.assertRefused(TypeError,
                           r"^'geo0\.translate' is a plug; membership takes the node: ask with geo0\.translate in",
                           lambda: cube.t >> red)
        self.assertRefused(TypeError, r"^'geo0\.translateX' is a plug", lambda: cube.tx << -proxy)
        self.assertRefused(TypeError, r"^'geo0\.translateX' is a plug", lambda: cube.tx << Material())
        # in: a plug stands for its node
        cube << red
        self.assertEqual((cube.tx in red, cube.tx in proxy), (True, False))
        # plug >> node still clones the attribute
        cube << Float("w")
        cube.w >> other
        self.assertEqual(cmds.attributeQuery("w", node="geo1", attributeType=True), "double")


class TestTourW4(_Case):
    """W4: a query or a removal before the material or layer exists raises at
    the reference and never creates it."""

    def setUp(self):
        super().setUp()
        self.cube = _cube("cube")

    def test_queries_and_removals_raise_at_the_reference(self):
        cube = self.cube
        for label, call in (
            ("cube >> Blinn('red')", lambda: cube >> Blinn("red")),
            ("cube << -Blinn('red')", lambda: cube << -Blinn("red")),
            ("cube << Blinn('red')", lambda: cube << Blinn("red")),
            ("cube in Blinn('red')", lambda: cube in Blinn("red")),
            ("cube.f[:2] in Material('red')", lambda: cube.f[:2] in Material("red")),
            ("cube >> Layer('L')", lambda: cube >> Layer("L")),
            ("cube << -Layer('L')", lambda: cube << -Layer("L")),
            ("cube in Layer('L')", lambda: cube in Layer("L")),
            ("cube.tx in Layer('L')", lambda: cube.tx in Layer("L")),
            ("List([cube]) << Layer('L')", lambda: List([cube]) << Layer("L")),
        ):
            with self.subTest(row=label):
                self.assertRefused(NodeNotFoundError,
                                   r"^no (blinn|surface shader|displayLayer) named '(red|L)'$", call)
                self.assertRefused(ValueError, "named", call)   # old handlers catch it
        self.assertEqual((cmds.objExists("red"), cmds.objExists("L")), (False, False))
        self.assertEqual(_engines("cubeShape"), ["initialShadingGroup"])

    def test_exists_is_would_the_reference_succeed(self):
        before = _scene()
        self.assertEqual((Blinn.exists("red"), Layer.exists("cube"), Material.exists("cube")),
                         (False, False, False))
        self.assertTrue(Transform.exists("cube"))
        self.assertEqual(_scene(), before)
        Blinn.define("red")
        self.assertEqual(
            (Blinn.exists("red"), Material.exists("red"), Lambert.exists("red"), Layer.exists("red")),
            (True, True, False, False),
        )

    def test_the_kind_tokens_never_create(self):
        cube   = self.cube
        before = _scene()
        self.assertEqual(cube >> Material(), [Default()])
        self.assertEqual(Material.of(cube), [ShadingEngine("initialShadingGroup")])
        self.assertIsNone(cube >> Layer())
        self.assertFalse(cube in Tag("ghost"))
        self.assertEqual((cube.vtx[:3] >> Tag("ghost")).tolist(), [])
        self.assertIs(cube << Material(), cube)                 # green: in no engine
        self.assertEqual((_engines("cubeShape"), cube >> Material()), ([], []))
        self.assertIs(cube << Default(), cube)
        self.assertEqual(_engines("cubeShape"), ["initialShadingGroup"])
        self.assertIs(cube << Layer(), cube)
        self.assertEqual(_scene(), before)

    def test_none_is_refused(self):
        cube = self.cube
        for call, pattern in (
            (lambda: cube << Blinn(None), r"^None is not a material name; Material\(\) removes all$"),
            (lambda: cube << Material(None), r"^None is not a material name; Material\(\) removes all$"),
            (lambda: cube << Layer(None), r"^None is not a layer name; Layer\(\) is defaultLayer"),
            (lambda: cube.vtx[:5] << Tag(None), r"^None is not a tag name; Tag\(\) means every tag$"),
        ):
            with self.subTest(pattern=pattern):
                self.assertRefused(TypeError, pattern, call)
        self.assertEqual(_engines("cubeShape"), ["initialShadingGroup"])


def _count_lookups(run):
    """``(run(), counts)``: the name lookups made while ``run`` runs
    (``cmds.ls``, ``cmds.objExists``, ``cmds.namespaceInfo``, the membership
    ``_find_node``), as the NC0 pins count them."""
    names   = ("ls", "objExists", "namespaceInfo")
    patches = [mock.patch.object(cmds, name, wraps=getattr(cmds, name)) for name in names]
    patches.append(mock.patch.object(_members, "_find_node", wraps=_members._find_node))
    mocks = [patch.start() for patch in patches]
    try:
        result = run()
    finally:
        for patch in patches:
            patch.stop()
    return result, dict(zip(names + ("_find_node",), (m.call_count for m in mocks)))


_NO_LOOKUP = {"ls": 0, "objExists": 0, "namespaceInfo": 0, "_find_node": 0}


class TestTourW5(_Case):
    """W5: math inside ``with container()`` makes new nodes and looks no name
    up; ``define`` is the only lookup, because the author asked for one."""

    def _inputs(self):
        a   = Node.create("transform", name="a")
        b   = Node.create("transform", name="b")
        ctl = Node.create("transform", name="ctl")
        ctl << Float("w", dv=0.5)
        b.t << (2, 4, 6)
        return a, b, ctl

    def _block(self):
        """The W5 block, each statement counted on its own."""
        a, b, ctl = self._inputs()
        counts = {}
        with container("lerp") as lerp:
            _, counts["math"]   = _count_lookups(lambda: ctl.t << (b.t - a.t) * ctl.w + a.t)
            mds, counts["rn"]   = _count_lookups(lambda: [rn.multiplyDivide(name="mul1") for _ in range(2)])
            k, counts["create"] = _count_lookups(lambda: Transform.create(name="k"))
            anchor, counts["define"] = _count_lookups(lambda: Transform.define("anchor"))
        return lerp, mds, k, anchor, counts

    def test_define_is_the_only_lookup(self):
        self._block()           # the per-process type caches warm up (see test_r4b_pins)
        self.new_scene()
        lerp, mds, k, anchor, counts = self._block()
        self.assertEqual({key: counts[key] for key in ("math", "rn", "create")},
                         {"math": _NO_LOOKUP, "rn": _NO_LOOKUP, "create": _NO_LOOKUP})
        self.assertGreater(counts["define"]["ls"], 0)
        self.assertEqual(cmds.getAttr("ctl.t")[0], (1.0, 2.0, 3.0))
        members = cmds.container(str(lerp), query=True, nodeList=True)
        self.assertTrue({"k", "anchor", *map(str, mds)} <= set(members), members)
        self.assertEqual(len({str(m) for m in mds}), 2)          # always new
        # outside the scope the anchor is found: its container is gone from the stack
        self.assertEqual(Transform.define("anchor"), anchor)

    def test_a_flattened_scope_prefixes_names_and_the_key(self):
        a, b, ctl = self._inputs()
        with container("rig") as rig_scope:
            with container("lerp"):
                ctl.t << (b.t - a.t) * ctl.w + a.t
                made   = [Transform.create(name="k") for _ in range(2)]
                anchor = Transform.define("anchor")
                self.assertEqual(Transform.define("anchor"), anchor)
        self.assertEqual([str(m) for m in made], ["lerp_k", "lerp_k1"])
        self.assertEqual(str(anchor), "lerp_anchor")
        members = cmds.container(str(rig_scope), query=True, nodeList=True)
        self.assertTrue(members and all(m.startswith("lerp_") for m in members), members)
        # outside the scope 'anchor' is another key
        self.assertEqual(Transform.define("anchor").long_name, "|anchor")


class TestTourW6(_Case):
    """W6: the type-mismatch rows. A subtype is found most derived; another
    type is a NodeTypeError that writes nothing."""

    def setUp(self):
        super().setUp()
        cmds.createNode("joint", name="j1")
        cmds.createNode("transform", name="grp")
        cmds.shadingNode("blinn", asShader=True, name="red")
        cmds.shadingNode("phong", asShader=True, name="shiny")
        cmds.shadingNode("blinn", asShader=True, name="b")

    def test_a_subtype_is_the_most_derived_node(self):
        before = _scene()
        j1 = Transform("j1")
        self.assertIs(type(j1), Joint)
        self.assertEqual((j1 == Node("j1"), Node("j1") == j1, hash(j1) == hash(Node("j1"))), (True, True, True))
        self.assertEqual(Transform.define("j1"), j1)
        self.assertIs(type(Transform.define("j1")), Joint)
        self.assertIs(type(ObjectSet("initialShadingGroup")), ShadingEngine)
        self.assertEqual(repr(Material("b")), 'Blinn("b")')
        self.assertEqual(_scene(), before)

    def test_dag_mismatches(self):
        self.assertRefused(NodeTypeError,
                           r"^'grp' is a transform, not a joint; Node\('grp'\) is Transform\(\"grp\"\)$",
                           lambda: Joint("grp"))
        self.assertRefused(NodeTypeError,
                           r"^'red' is a blinn, not a transform; Node\('red'\) is Blinn\(\"red\"\)$",
                           lambda: Transform("red"))
        self.assertRefused(NodeTypeError, r"^'grp' is a transform, not a joint", lambda: Joint.define("grp"))
        self.assertTrue(issubclass(NodeTypeError, (TypeError, ValueError)))

    def test_dg_mismatches(self):
        self.assertRefused(NodeTypeError,
                           r"^'red' is a blinn, not a displayLayer; Node\('red'\) is Blinn\(\"red\"\)$",
                           lambda: DisplayLayer("red"))
        self.assertRefused(NodeTypeError,
                           r"^'grp' is a transform, not a shadingEngine; Node\('grp'\) is Transform\(\"grp\"\)$",
                           lambda: ShadingEngine("grp"))
        self.assertRefused(NodeTypeError, r"^'j1' is a joint, not a displayLayer", lambda: Layer.define("j1"))

    def test_the_shader_classes_are_siblings(self):
        self.assertRefused(
            NodeTypeError,
            r"^'shiny' is a phong, not a blinn; Node\('shiny'\) is Phong\(\"shiny\"\); Material\('shiny'\) "
            r"takes any surface shader; Phong\('shiny'\)\.astype\(Blinn\) converts it$",
            lambda: Blinn("shiny"),
        )
        self.assertRefused(NodeTypeError,
                           r"^'b' is a blinn, not a lambert; .*Blinn\('b'\)\.astype\(Lambert\) converts it$",
                           lambda: Lambert("b"))
        self.assertRefused(NodeTypeError, r"^'shiny' is a phong, not a blinn", lambda: Blinn.define("shiny"))
        # a class call never converts: astype does
        self.assertRefused(NodeTypeError, r"Blinn\('red'\)\.astype\(Phong\) converts it$",
                           lambda: Phong(Node("red")))
        self.assertEqual(cmds.nodeType("red"), "blinn")


class TestTourW7(_Case):
    """W7: undo, a new scene, held nodes, conversion."""

    def test_undo_and_redo_a_create(self):
        held = Transform.create(name="h")
        cmds.undo()
        with self.assertRaisesRegex(RuntimeError, r"^h already deleted!$"):
            str(held)
        self.assertRefused(NodeNotFoundError, r"^no transform named 'h'$", lambda: Transform("h"))
        cmds.redo()
        self.assertEqual((str(held), Transform("h")), ("h", held))

    def test_a_new_scene_kills_every_held_node(self):
        red   = Blinn.define("red", color=(1, 0, 0))
        layer = Layer.define("L")
        cube  = _cube("cube")
        cmds.file(new=True, force=True)
        for held, cls in ((red, "Blinn"), (layer, "DisplayLayer"), (cube, "Transform")):
            with self.subTest(held=cls):
                with self.assertRaisesRegex(RuntimeError,
                                            rf"^{cls} node \(freed by a new scene, .*\) already deleted!$"):
                    str(held)
        other = _cube("other")
        self.assertRefused(RuntimeError, "freed by a new scene", lambda: other << red)
        self.assertEqual(_engines("otherShape"), ["initialShadingGroup"])

    def test_a_recipe_is_a_function(self):
        def red_look():
            return Blinn.define("red", color=(1, 0, 0))

        first = red_look()
        self.assertEqual(red_look(), first)
        cmds.file(new=True, force=True)
        again = red_look()
        self.assertEqual((repr(again), cmds.getAttr("red.color")[0]), ('Blinn("red")', (1.0, 0.0, 0.0)))

    def test_define_then_assign_is_two_undo_steps(self):
        cube = _cube("cube")
        cmds.flushUndo()
        self.assertIs(cube << Blinn.define("red", color=(1, 0, 0)), cube)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.material")
        cmds.undo()
        self.assertEqual((cmds.objExists("red"), _engines("cubeShape")), (True, ["initialShadingGroup"]))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.define")
        cmds.undo()
        self.assertFalse(cmds.objExists("red"))
        cmds.redo()
        cmds.redo()
        self.assertTrue(cube in Blinn("red"))

    def test_astype_returns_the_new_node(self):
        cube = _cube("cube")
        red  = Blinn.define("red", color=(1, 0, 0))
        cube << red
        phong = red.astype(Phong)
        self.assertEqual(repr(phong), 'Phong("red")')
        with self.assertRaisesRegex(RuntimeError,
                                    r"^'red' was converted to a phong; use the node astype\(\) returned"):
            str(red)
        self.assertTrue(cube in phong)
        self.assertEqual(cmds.getAttr("red.color")[0], (1.0, 0.0, 0.0))
        back = shade.convert(phong, Blinn)
        self.assertEqual((repr(back), cube in back), ('Blinn("red")', True))
        self.assertEqual(back, Blinn("red"))

    def test_a_failed_define_deletes_what_it_made(self):
        make = DAGNode._create.__func__

        def renaming(cls, **kwargs):
            return cmds.rename(make(cls, **kwargs), "moved")

        for queue in (True, False):
            with self.subTest(undo_queue=queue):
                cmds.undoInfo(state=queue)
                try:
                    with mock.patch.object(Transform, "_create", classmethod(renaming)):
                        self.assertRefused(NodeTypeError, r"define deleted what the call made \(\|moved\)$",
                                           lambda: Transform.define("x"))
                finally:
                    cmds.undoInfo(state=True)


class TestTourW8(_Case):
    """W8: ``Tag`` stays a declaration, like ``Float``; re-declaring an
    attribute keeps its value and its wire."""

    def setUp(self):
        super().setUp()
        self.cube = _cube("cube")

    def test_the_tag_idioms_are_unchanged(self):
        cube   = self.cube
        before = _scene()
        cube.f[:3] << Tag("cap")
        cube.f[:3] << Tag("cap")                 # applying it twice is applying it once
        self.assertEqual((cube >> Tag("cap")).tolist(), [0, 1, 2])
        cube.f[0] << -Tag("cap")
        self.assertEqual((cube >> Tag("cap")).tolist(), [1, 2])
        self.assertEqual((cube.f >> Tag("cap")).tolist(), [1, 2])
        self.assertEqual((cube.f[1:3] in Tag("cap"), cube.f[:3] in Tag("cap"), cube in Tag("cap")),
                         (True, False, True))
        self.assertIn("Tag('cap')", [repr(tag) for tag in cube >> Tag()])
        cube.vtx[:5] << Tag()                    # vertices out of every tag: a face tag keeps its faces
        self.assertEqual((cube >> Tag("cap")).tolist(), [1, 2])
        self.assertEqual(_scene(), before)       # a tag lives on the shape

    def test_a_missing_tag_and_none(self):
        cube = self.cube
        self.assertFalse(cube in Tag("ghost"))
        self.assertTrue(cube not in Tag("ghost"))
        self.assertEqual((cube.vtx[:3] >> Tag("ghost")).tolist(), [])
        self.assertRefused(TypeError, r"^None is not a tag name; Tag\(\) means every tag$",
                           lambda: cube.vtx[:2] << Tag(None))

    def test_the_user_decision_example(self):
        node = Node.create("transform", name="n")
        drv  = Node.create("transform", name="drv")
        drv.tx << 7
        node << Float("test", dv=5)
        node.test << drv.tx
        before = _scene()
        node << Float("test")
        self.assertEqual(cmds.getAttr("n.test"), 7.0)
        self.assertEqual(cmds.listConnections("n.test", source=True, plugs=True), ["drv.translateX"])
        self.assertEqual(cmds.addAttr("n.test", query=True, defaultValue=True), 5.0)
        node << Float("test", max=10)
        self.assertEqual((cmds.getAttr("n.test"), cmds.attributeQuery("test", node="n", max=True)),
                         (7.0, [10.0]))
        self.assertEqual(cmds.listConnections("n.test", source=True, plugs=True), ["drv.translateX"])
        self.assertEqual(_scene(), before)

    def test_a_redeclaration_keeps_the_wire(self):
        ctl = Node.create("transform", name="ctl")
        src = rn.multiplyDivide(name="src")
        ctl << Float("w", min=0)
        ctl.w << src.outputX
        ctl << Float("w", max=5)
        self.assertEqual(cmds.listConnections("ctl.w", source=True, plugs=True), ["src.outputX"])
        self.assertEqual(cmds.attributeQuery("w", node="ctl", range=True), [0.0, 5.0])

    def test_another_kind_raises_and_overwrite_replaces(self):
        ctl = Node.create("transform", name="ctl")
        src = rn.multiplyDivide(name="src")
        ctl << Float("w", max=5)
        ctl.w << src.outputX
        self.assertRefused(TypeError, r"^'ctl\.w' exists as a double, not an enum; overwrite=True replaces it$",
                           lambda: ctl << Enum("w", en="a:b"))
        self.assertEqual(cmds.listConnections("ctl.w", source=True, plugs=True), ["src.outputX"])
        ctl << Float("w", overwrite=True)
        self.assertIsNone(cmds.listConnections("ctl.w", source=True, plugs=True))
        self.assertFalse(cmds.attributeQuery("w", node="ctl", maxExists=True))
