"""Round 4b, step NC4: ``Cls.define`` finds the node at its key, or makes it there.

* ``TestDefine``: the key is the node ``create(name=..., parent=...)`` would make
  (the world without ``parent=``; the current namespace for a bare name), so a
  define twice is one node, ``L_arm|ctl`` and ``R_arm|ctl`` are two keys and a
  re-run makes nothing; a found node keeps its values unless ``update=True``, is
  of the class or a subclass (``Transform.define("j1")`` is ``Joint("j1")``),
  else NodeTypeError; attribute names are checked on the type before any lookup;
  the name is a key (no ``|``, parts Maya keeps as written), ``parent=`` a
  reference; a DG node holding the name is refused (Maya would rename the new
  node); the registries define like any class; the classes built from data refuse
  define, naming their creator; a made node is one undo step.
* ``TestNodeDefine``: ``Node.define(type, name)`` runs a registered class's define,
  and keys, guards and makes a type no class is registered for, checking the exact
  type on a hit.
* ``TestDefineElsewhere``: without ``parent=`` a name the reference already gives
  another node (nested, ``|char_grp|root``: REC C2; in the other namespace the
  lookup rule reads, ``:x`` while ``char`` is current: C3) is refused, not forked.
* ``TestDefineReferenceNamespace``: define never makes a node in a file reference's
  namespace, loaded or unloaded, current or spelled (C4), never creates a
  namespace, and finds a node the reference holds.
* ``TestDefineCleanup``: a made node that is not the key after all is deleted,
  exactly what the call made, with the undo queue on or off; the user's previous
  undo step is never undone (CC-4).
* ``TestDefineInContainer``: the flattened scope's prefix is in the key; a node a
  container off the scope's stack owns is refused (a re-run in the same scene; a
  node from another module's container); an enclosing container's node is found;
  a registry is checked only with ``container=True``; a found node is never added
  to the scope.

Every refusal writes nothing (a zero ``cmds.ls()`` delta).
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds

from rig import (
    AmbiguousNodeError,
    container,
    Float,
    Node,
    NodeNotFoundError,
    NodeTypeError,
    set_options,
    undo_chunk,
)
from rig.nodetypes import (
    BlendShape,
    Choice,
    DAGNode,
    DGNode,
    DisplayLayer,
    Follicle,
    Geometry,
    Joint,
    Mesh,
    NurbsCurve,
    NurbsSurface,
    ObjectSet,
    Reference,
    ShadingEngine,
    SkinCluster,
    Transform,
)
from rig._internal.container import Container
from rig._tests._base import MayaTestCase


def _scene():
    return set(cmds.ls())


def _transforms():
    return set(cmds.ls(type="transform", long=True))


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

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


class TestDefine(_Case):
    """REC W2 / W6: the key, the found branch, the checks before any write."""

    def test_twice_is_one_node(self):
        first  = Transform.define("rig")
        second = Transform.define("rig")
        self.assertEqual(first, second)
        self.assertEqual(first.long_name, "|rig")
        self.assertEqual(cmds.ls("rig*", type="transform"), ["rig"])
        self.assertIs(type(first), Transform)
        # the reference finds what define made
        self.assertEqual(Transform("rig"), first)

    def test_one_base_name_under_two_parents_is_two_keys(self):
        root = Transform.define("rig")
        ctls = []
        for side in "LR":
            arm = Transform.define(f"{side}_arm", parent=root)
            ctls.append(Transform.define("ctl", parent=arm, tx=1))
        self.assertEqual([c.long_name for c in ctls], ["|rig|L_arm|ctl", "|rig|R_arm|ctl"])
        self.assertEqual([cmds.getAttr(f"{c.long_name}.tx") for c in ctls], [1.0, 1.0])
        # found by path, by node object and by a partial path
        self.assertEqual(Transform.define("ctl", parent="|rig|L_arm"), ctls[0])
        self.assertEqual(Transform.define("ctl", parent=Node("R_arm")), ctls[1])

    def test_a_rerun_makes_no_new_transform(self):
        def build():
            root = Transform.define("rig")
            for side in "LR":
                arm = Transform.define(f"{side}_arm", parent=root)
                ctl = Transform.define("ctl", parent=arm, tx=1)
                ctl << Float("fk_ik", min=0, max=1)
            return root

        build()
        cmds.setAttr("|rig|L_arm|ctl.tx", 3)  # an artist's edit
        made = _transforms()
        build()
        self.assertEqual(_transforms(), made)
        # the build's values are set again: the script is the source of truth
        self.assertEqual(cmds.getAttr("|rig|L_arm|ctl.tx"), 1.0)

    def test_update_is_on_by_default(self):
        # the script is the source of truth (the user's decision, 2026-10-01):
        # every define, and a same-type conversion, sets the values it is given
        import inspect

        from rig import shade
        from rig.nodetypes.material_node import Material

        for call in (DGNode.define, Node.define, Material.define, Material.astype, shade.convert):
            with self.subTest(call=call.__qualname__):
                self.assertIs(inspect.signature(call).parameters["update"].default, True)

    def test_update_applies_the_values_in_one_undo_step(self):
        ctl = Transform.define("ctl", tx=1, ty=2)
        cmds.setAttr("ctl.tx", 9)
        self.assertEqual(Transform.define("ctl", tx=5, update=False), ctl)  # update=False: left as it is
        self.assertEqual(cmds.getAttr("ctl.tx"), 9.0)
        self.assertEqual(Transform.define("ctl", tx=5, ty=6), ctl)            # the default: set
        self.assertEqual((cmds.getAttr("ctl.tx"), cmds.getAttr("ctl.ty")), (5.0, 6.0))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.define")
        cmds.undo()
        self.assertEqual((cmds.getAttr("ctl.tx"), cmds.getAttr("ctl.ty")), (9.0, 2.0))

    def test_a_made_node_is_one_undo_step(self):
        ctl = Transform.define("ctl", tx=1, rotateOrder="zxy")
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.define")
        cmds.undo()
        self.assertFalse(cmds.objExists("ctl"))
        cmds.redo()
        self.assertTrue(ctl.is_valid)
        self.assertEqual((cmds.getAttr("ctl.tx"), cmds.getAttr("ctl.rotateOrder")), (1.0, 2))

    def test_an_attribute_typo_raises_on_a_hit_and_a_miss(self):
        Transform.define("rig")
        pattern = r"^Transform\.define\(\): a transform has no attribute 'tyop', and no create flag"
        self.assertRefused(AttributeError, pattern, lambda: Transform.define("rig", tyop=1))
        self.assertRefused(AttributeError, pattern, lambda: Transform.define("new", tyop=1))
        self.assertRefused(TypeError, "field", lambda: Transform.define("new", rotateOrder="nope"))

    def test_a_subtype_is_found_most_derived(self):
        cmds.createNode("joint", name="j1")
        found = Transform.define("j1", tx=4)
        self.assertIs(type(found), Joint)
        self.assertEqual(found, Node("j1"))
        self.assertEqual(cmds.getAttr("j1.tx"), 4.0)  # found, and set
        # an engine is an objectSet
        self.assertIs(type(ObjectSet.define("initialShadingGroup")), ShadingEngine)

    def test_another_type_is_a_node_type_error(self):
        cmds.createNode("transform", name="grp")
        cmds.sets(empty=True, name="plain")
        cases = (
            (lambda: Joint.define("grp"), r"^'grp' is a transform, not a joint; Node\('grp'\) is Transform\(\"grp\"\)$"),
            (lambda: DisplayLayer.define("grp"), r"^'grp' is a transform, not a displayLayer"),
            (lambda: ShadingEngine.define("plain"), r"^'plain' is an objectSet, not a shadingEngine"),
            (lambda: Choice.define("grp"), r"^'grp' is a transform, not a choice"),
        )
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(NodeTypeError, pattern, call)

    def test_the_parent_is_a_reference_resolved_before_any_write(self):
        for g in ("g1", "g2"):
            cmds.createNode("transform", name=g)
            cmds.createNode("transform", name="a", parent=g)
        cmds.createNode("multiplyDivide", name="md")
        self.assertRefused(NodeNotFoundError, r"^no DAG node named 'nosuch'", lambda: Transform.define("x", parent="nosuch"))
        self.assertRefused(AmbiguousNodeError, r"use a path", lambda: Transform.define("x", parent="a"))
        self.assertRefused(NodeTypeError, r"not a DAG node", lambda: Transform.define("x", parent="md"))
        dead = Node("g2")
        cmds.delete("g2")
        self.assertRefused(RuntimeError, "deleted", lambda: Transform.define("x", parent=dead))
        self.assertEqual(Transform.define("x", parent="g1|a").long_name, "|g1|a|x")

    def test_the_name_is_a_key(self):
        cmds.createNode("transform", name="rig")
        cases = (
            (TypeError, r"^Transform\.define\('rig\|x'\): a define's name is a key, not a path: "
                        r"Transform\.define\('x', parent='rig'\) keys it under 'rig'$",
             lambda: Transform.define("rig|x")),
            (TypeError, r"Transform\.define\('x'\) keys it at the world", lambda: Transform.define("|x")),
            (TypeError, r"^Choice\.define\('a\|b'\): a choice is a DG node", lambda: Choice.define("a|b")),
            (ValueError, r"^Transform\.define\('1bad'\): '1bad' is not a name Maya keeps as written",
             lambda: Transform.define("1bad")),
            (ValueError, "not a name Maya keeps", lambda: Transform.define("a-b")),
            (ValueError, "not a name Maya keeps", lambda: Transform.define("r*")),
            (ValueError, "not a name Maya keeps", lambda: Transform.define("")),
            (ValueError, "not a name Maya keeps", lambda: Transform.define("char:")),
            (TypeError, r"^Transform\.define\(name\) takes the node's name, a str", lambda: Transform.define(None)),
            (TypeError, "a str", lambda: Transform.define(Node("rig"))),
            (TypeError, r"takes its name first and parent= \(got n=\)", lambda: Transform.define("x", n="y")),
            (TypeError, r"\(got p=\)", lambda: Transform.define("x", p="rig")),
            (TypeError, r"^Choice\.define\(\) takes no parent=: a choice is a DG node$",
             lambda: Choice.define("c", parent="rig")),
        )
        for error, pattern, call in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(error, pattern, call)

    def test_a_name_maya_would_change_is_refused(self):
        """CC-4: the renames are predicted before anything is made."""
        cmds.createNode("multiplyDivide", name="knob")
        cmds.createNode("transform", name="grp")
        for g in ("g1", "g2"):
            cmds.createNode("transform", name=g)
            cmds.createNode("transform", name="a", parent=g)
        cases = (
            (lambda: Transform.define("knob"),
             r"^Transform\.define\('knob'\): 'knob' is taken by a multiplyDivide \(knob\); "
             r"Maya would rename a new transform$"),
            (lambda: Transform.define("knob", parent="grp"), r"'knob' is taken by a multiplyDivide"),
            (lambda: Joint.define("knob", parent="grp"), r"Maya would rename a new joint"),
            # a DG name is unique among every node's short names
            (lambda: Choice.define("a"), r"^Choice\.define\('a'\): 'a' is taken by 2 nodes \(\|g\d\|a, \|g\d\|a\)"),
        )
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(NodeTypeError, pattern, call)
        # a DAG node of the same short name under another parent is no clash
        self.assertEqual(Transform.define("a", parent="grp").long_name, "|grp|a")

    def test_in_the_current_namespace(self):
        cmds.namespace(add="char")
        cmds.namespace(setNamespace=":char")
        made = Transform.define("x")
        self.assertEqual(str(made), "char:x")
        self.assertEqual(Transform.define("x"), made)
        self.assertEqual(Transform.define("char:x"), made)  # a spelled namespace is read from the root
        layer = DisplayLayer.define("L")
        self.assertEqual(str(layer), "char:L")
        # ':' is the root namespace
        self.assertEqual(Transform.define(":top_x").long_name, "|top_x")
        cmds.namespace(setNamespace=":")
        self.assertEqual(Transform.define("char:x"), made)
        self.assertEqual(str(Joint.define("char:j")), "char:j")

    def test_relative_names_do_not_change_the_key(self):
        cmds.namespace(add="char")
        grp = Transform.create(name="grp")
        cmds.namespace(setNamespace=":char")
        cmds.namespace(relativeNames=True)
        made = Transform.define("x")
        under = Transform.define("y", parent=grp)
        self.assertEqual(Transform.define("x"), made)
        self.assertEqual(Transform.define("y", parent=grp), under)
        cmds.namespace(relativeNames=False)
        cmds.namespace(setNamespace=":")
        self.assertEqual(made.long_name, "|char:x")
        self.assertEqual(under.long_name, "|grp|char:y")

    def test_the_registries_define(self):
        cmds.createNode("transform", name="sel")
        cmds.select("sel")
        layer = DisplayLayer.define("L", displayType=2)
        self.assertEqual((str(layer), cmds.getAttr("L.displayType")), ("L", 2))
        self.assertIsNone(cmds.editDisplayLayerMembers("L", query=True))  # the selection is never taken
        self.assertEqual(DisplayLayer.define("L", displayType=0), layer)
        self.assertEqual(cmds.getAttr("L.displayType"), 0)  # found, and set
        objset = ObjectSet.define("S")
        self.assertEqual((type(objset), ObjectSet.define("S")), (ObjectSet, objset))
        engine = ShadingEngine.define("redSG")
        self.assertIs(type(engine), ShadingEngine)
        self.assertTrue(cmds.listConnections("redSG.partition", plugs=True))
        self.assertEqual(ShadingEngine.define("redSG"), engine)
        self.assertEqual(cmds.ls(type="shadingEngine"), ["initialParticleSE", "initialShadingGroup", "redSG"])

    def test_the_classes_built_from_data_refuse_define(self):
        cube = cmds.polyCube(name="cube", ch=False)[0]
        cmds.circle(name="circle", ch=False)
        for cls, creator in (
            (Mesh, r"Mesh\.create\(mesh_data"),
            (NurbsCurve, r"NurbsCurve\.create\(points"),
            (NurbsSurface, r"rc\.nurbsPlane"),
            (SkinCluster, r"SkinCluster\.create\(geom, influences\)"),
            (BlendShape, r"BlendShape\.create\(\*targets, base\)"),
            (Reference, r"Reference\.create\(file_path, namespace\)"),
            (Follicle, r"Follicle\.create_on_mesh"),
            (Container, r"with container\('x'\)"),
        ):
            with self.subTest(cls=cls.__name__):
                pattern = rf"^{cls.__name__}\.define\(\) is refused: .*{creator}"
                self.assertRefused(TypeError, pattern, lambda: cls.define("thing"))
        # on a hit too: define is not how these are found (Mesh('x') refers)
        self.assertRefused(TypeError, "Mesh.define", lambda: Mesh.define(cube))
        self.assertRefused(TypeError, "Mesh.define", lambda: Mesh.define("cubeShape"))
        self.assertRefused(TypeError, "NurbsCurve.define", lambda: NurbsCurve.define("circleShape"))
        # the other doors' messages do not name a define these classes refuse
        with self.assertRaisesRegex(
            TypeError, r"^Mesh\(\) names no node: Mesh\('x'\) refers to x; Mesh\.create\(name='x'\) makes one$"
        ):
            Mesh()
        with self.assertRaisesRegex(
            TypeError,
            r"^SkinCluster\.create\(shared=\.\.\.\): create always makes a new node; "
            r"SkinCluster\('x'\) refers to an existing one$",
        ):
            SkinCluster.create(shared=True)
        # the base classes of an abstract type make no node
        for cls in (DGNode, DAGNode, Geometry):
            with self.subTest(cls=cls.__name__):
                self.assertRefused(TypeError, r"abstract Maya type; Node\.define\('<type>', 'x'\)",
                                   lambda: cls.define("cube"))

    def test_undone_or_renamed_away_is_made_again(self):
        """E1: define reads the scene on every call: a node undone, or renamed
        off its key, is no longer there, so define makes a new one at the key."""
        first = Transform.define("rig")
        cmds.undo()
        self.assertFalse(first.is_valid)
        again = Transform.define("rig")
        self.assertEqual(again.long_name, "|rig")
        cmds.rename("rig", "rig_old")
        made = Transform.define("rig")
        self.assertEqual((made.long_name, again.long_name), ("|rig", "|rig_old"))
        self.assertEqual(Transform.define("rig"), made)

    def test_a_node_deleted_is_made_again(self):
        first = Transform.define("rig")
        cmds.delete("rig")
        second = Transform.define("rig")
        self.assertFalse(first.is_valid)
        self.assertEqual(second.long_name, "|rig")
        cmds.file(new=True, force=True)
        with self.assertRaisesRegex(RuntimeError, "deleted"):
            second.tx << 1
        self.assertEqual(Transform.define("rig").long_name, "|rig")


class TestNodeDefine(_Case):
    """``Node.define(type, name, ...)``: the untyped door."""

    def test_a_registered_type_runs_its_class_define(self):
        root = Node.define("transform", "rig")
        self.assertIs(type(root), Transform)
        self.assertEqual(Node.define("transform", "rig"), root)
        joint = Node.define("joint", "j", parent=root, tx=1)
        self.assertEqual((type(joint), joint.long_name, cmds.getAttr("|rig|j.tx")), (Joint, "|rig|j", 1.0))
        self.assertIs(type(Node.define("displayLayer", "L")), DisplayLayer)
        self.assertRefused(TypeError, r"^SkinCluster\.define\(\) is refused", lambda: Node.define("skinCluster", "skin"))
        self.assertRefused(NodeTypeError, r"not a joint", lambda: Node.define("joint", "rig"))

    def test_an_unregistered_type(self):
        made = Node.define("multiplyDivide", "md", operation="divide", input1X=3)
        self.assertEqual((cmds.nodeType("md"), cmds.getAttr("md.operation"), cmds.getAttr("md.input1X")),
                         ("multiplyDivide", 2, 3.0))
        self.assertEqual(Node.define("multiplyDivide", "md", operation="multiply", update=False), made)
        self.assertEqual(cmds.getAttr("md.operation"), 2)  # update=False: left as it is
        Node.define("multiplyDivide", "md", operation="multiply")  # the default: set
        self.assertEqual(cmds.getAttr("md.operation"), 1)
        cmds.createNode("transform", name="rig")
        cases = (
            (NodeTypeError, r"^'md' is a multiplyDivide, not a plusMinusAverage",
             lambda: Node.define("plusMinusAverage", "md")),
            (NodeTypeError, r"^'rig' is a transform, not a multiplyDivide",
             lambda: Node.define("multiplyDivide", "rig")),
            (AttributeError, r"^Node\.define\('multiplyDivide', \.\.\.\): a multiplyDivide has no attribute 'tyop'",
             lambda: Node.define("multiplyDivide", "new", tyop=1)),
            (ValueError, r"^'noSuchType' is not a Maya node type$", lambda: Node.define("noSuchType", "x")),
            (TypeError, r"^Node\.define\('locator', 'loc'\): a locator is a shape",
             lambda: Node.define("locator", "loc")),
            (TypeError, r"^Node\.define\('container', 'box'\) is refused: a container is made by its scope",
             lambda: Node.define("container", "box")),
            (TypeError, r"takes no parent=: a multiplyDivide is a DG node",
             lambda: Node.define("multiplyDivide", "m2", parent="rig")),
        )
        for error, pattern, call in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(error, pattern, call)

    def test_an_unregistered_dag_type_takes_a_parent(self):
        rig = Transform.create(name="rig")
        made = Node.define("aimConstraint", "aim", parent=rig)
        self.assertEqual((cmds.nodeType(str(made)), made.long_name), ("aimConstraint", "|rig|aim"))
        self.assertEqual(Node.define("aimConstraint", "aim", parent="rig"), made)
        self.assertRefused(AmbiguousNodeError, r"'aim' exists at \|rig\|aim", lambda: Node.define("aimConstraint", "aim"))

    def test_an_unregistered_type_in_a_flattened_scope(self):
        with container("outer") as outer:
            with container("inner"):
                made  = Node.define("multiplyDivide", "m")
                again = Node.define("multiplyDivide", "m")
        self.assertEqual((str(made), again), ("inner_m", made))
        self.assertEqual(cmds.container(str(outer), query=True, nodeList=True), ["inner_m"])


class TestDefineElsewhere(_Case):
    """CC-1 / CC-2: without ``parent=`` a name the reference already gives another
    node is refused, not forked."""

    def test_c2_a_nested_name_is_refused_without_parent(self):
        cmds.createNode("transform", name="char_grp")
        root = Joint(cmds.createNode("joint", name="root", parent="char_grp"))
        self.assertRefused(
            AmbiguousNodeError,
            r"^Joint\.define\('root'\): 'root' exists at \|char_grp\|root; Joint\('root'\) refers to it, "
            r"and Joint\.define\('root', parent='char_grp'\) keys it there$",
            lambda: Joint.define("root"),
        )
        self.assertEqual(Joint("root"), root)  # still one node
        self.assertEqual(Joint.define("root", parent="char_grp"), root)
        cmds.createNode("transform", name="other")
        made = Joint.define("root", parent="other")  # an explicit key
        self.assertEqual(made.long_name, "|other|root")

    def test_another_type_nested_names_node(self):
        """The hint names no define of that key: a transform there would make
        ``Joint.define('t', parent='grp')`` raise (FIX: review safety D7)."""
        cmds.createNode("transform", name="grp")
        cmds.createNode("transform", name="t", parent="grp")
        self.assertRefused(
            AmbiguousNodeError,
            r"^Joint\.define\('t'\): 't' exists at \|grp\|t; Node\('t'\) refers to it$",
            lambda: Joint.define("t"),
        )

    def test_two_nested_nodes_are_ambiguous(self):
        for side in "LR":
            Transform.define("ctl", parent=Transform.define(f"{side}_arm"))
        self.assertRefused(
            AmbiguousNodeError,
            r"^Transform\.define\('ctl'\): 'ctl' already names \|\w_arm\|ctl and \|\w_arm\|ctl; "
            r"a transform at the key \|ctl would be one more: pass parent= to key one of them$",
            lambda: Transform.define("ctl"),
        )

    def test_c3_the_other_namespace_the_lookup_rule_reads(self):
        cmds.createNode("transform", name="x")
        cmds.namespace(add="char")
        cmds.namespace(setNamespace=":char")
        for relative in (False, True):
            cmds.namespace(relativeNames=relative)
            with self.subTest(relativeNames=relative):
                self.assertRefused(
                    AmbiguousNodeError,
                    r"^Transform\.define\('x'\): 'x' exists as :x; this define's key is \|char:x: "
                    r"Transform\(':x'\) refers to it, Transform\.define\('char:x'\) makes \|char:x$",
                    lambda: Transform.define("x"),
                )
        cmds.namespace(relativeNames=False)
        made = Transform.define("char:x")  # spelled: its own key
        self.assertEqual(made.long_name, "|char:x")
        self.assertEqual(Transform.define("x"), made)  # found at the key now

    def test_a_dg_key_in_the_other_namespace(self):
        cmds.createDisplayLayer(name="L", empty=True)
        cmds.namespace(add="char")
        cmds.namespace(setNamespace=":char")
        self.assertRefused(
            AmbiguousNodeError,
            r"^DisplayLayer\.define\('L'\): 'L' exists as :L; this define's key is char:L: "
            r"DisplayLayer\(':L'\) refers to it",
            lambda: DisplayLayer.define("L"),
        )

    def test_from_the_root_a_namespaced_node_is_not_in_the_way(self):
        """The lookup rule reads a bare name at the root namespace only while the
        root is current: ``Transform('x')`` does not see ``char:x``, so a root
        ``x`` is no second node under that name."""
        cmds.namespace(add="char")
        cmds.createNode("transform", name="char:x")
        self.assertEqual(Transform.define("x").long_name, "|x")


class TestDefineReferenceNamespace(_Case):
    """CC-3 / REC C4: define never makes a node in a file reference's namespace."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = tempfile.mkdtemp(prefix="rig_define_")
        cls.path   = os.path.join(cls.folder, "one_joint.ma").replace("\\", "/")
        cmds.file(new=True, force=True)
        cmds.createNode("joint", name="spine")
        cmds.file(rename=cls.path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)

    @classmethod
    def tearDownClass(cls):
        cmds.file(new=True, force=True)
        shutil.rmtree(cls.folder, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        cmds.file(self.path, reference=True, namespace="char")
        self.ref_node = cmds.referenceQuery(self.path, referenceNode=True)

    def test_the_current_reference_namespace_is_refused(self):
        cmds.namespace(setNamespace=":char")
        pattern = r"^Joint\.define\('newj'\): the namespace 'char' belongs to the file reference charRN \(.*one_joint\.ma\)"
        self.assertRefused(ValueError, pattern, lambda: Joint.define("newj"))
        self.assertRefused(ValueError, "belongs to the file reference", lambda: DisplayLayer.define("L"))
        self.assertRefused(ValueError, "belongs to the file reference", lambda: Node.define("multiplyDivide", "m"))

    def test_a_spelled_reference_namespace_is_refused(self):
        self.assertRefused(ValueError, r"^Joint\.define\('char:spnie'\): the namespace 'char' belongs",
                           lambda: Joint.define("char:spnie"))

    def test_a_node_the_reference_holds_is_found(self):
        """``Joint.define('char:spine')`` works when the node exists: the guard
        runs only when define would make a node."""
        before = _scene()
        found  = Joint.define("char:spine", tx=3)
        self.assertEqual(found, Joint("char:spine"))
        self.assertTrue(cmds.referenceQuery(str(found), isNodeReferenced=True))
        cmds.namespace(setNamespace=":char")
        self.assertEqual(Joint.define("spine"), found)
        self.assertEqual(_scene(), before)
        # the values given are set, as on any found node: a reference edit
        self.assertEqual(cmds.getAttr("char:spine.tx"), 3.0)

    def test_an_unknown_namespace_is_refused_and_not_created(self):
        self.assertRefused(ValueError, r"^Joint\.define\('chr:root'\): there is no namespace 'chr'; define never creates one",
                           lambda: Joint.define("chr:root"))
        self.assertFalse(cmds.namespace(exists=":chr"))

    def test_an_unloaded_reference_keeps_its_namespace(self):
        cmds.file(unloadReference=self.ref_node)
        self.assertTrue(cmds.namespace(exists=":char"))
        self.assertRefused(ValueError, "belongs to the file reference", lambda: Joint.define("char:newj"))
        cmds.namespace(setNamespace=":char")
        self.assertRefused(ValueError, "belongs to the file reference", lambda: Joint.define("newj"))

    def test_a_namespace_inside_the_reference_namespace_is_refused(self):
        cmds.namespace(add="sub", parent=":char")
        self.assertRefused(ValueError, r"the namespace 'char:sub' belongs to the file reference charRN",
                           lambda: Joint.define("char:sub:j"))


class TestDefineCleanup(_Case):
    """CC-4: a made node that is not the key is deleted, never undone."""

    def _renaming(self):
        make = DAGNode._create.__func__

        def renaming(cls, **kwargs):
            return cmds.rename(make(cls, **kwargs), "moved")

        return mock.patch.object(Transform, "_create", classmethod(renaming))

    def test_with_the_undo_queue_on(self):
        with undo_chunk("user_step"):
            cmds.createNode("transform", name="user_node")
        before = _scene()
        with self._renaming():
            self.assertRefused(
                NodeTypeError,
                r"^Transform\.define\('x'\): Maya made '\|moved', not the key '\|x'; define deleted "
                r"what the call made \(\|moved\)$",
                lambda: Transform.define("x", tx=1),
            )
        self.assertEqual(_scene(), before)
        # the user's step was not undone: one undo reverts define's own step
        # (made and deleted: nothing), the next one the user's
        cmds.undo()
        self.assertEqual(_scene(), before)
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "user_step")
        cmds.undo()
        self.assertFalse(cmds.objExists("user_node"))

    def test_with_the_undo_queue_off(self):
        cmds.undoInfo(state=False)
        try:
            with self._renaming():
                self.assertRefused(NodeTypeError, "define deleted what the call made", lambda: Transform.define("x"))
        finally:
            cmds.undoInfo(state=True)

    def test_in_a_scope(self):
        """The scope's container keeps its members, and the hyperLayout Maya
        gives a container with its first member (deleting it would delete the
        container)."""
        with container("box") as box:
            Transform.define("first")
            with self._renaming():
                self.assertRefused(NodeTypeError, r"define deleted what the call made \(\|moved\)$",
                                   lambda: Transform.define("x"))
        self.assertEqual(cmds.container(str(box), query=True, nodeList=True), ["first"])
        with container("empty") as empty:
            before = _scene()
            with self._renaming():
                with self.assertRaisesRegex(NodeTypeError, r"define deleted what the call made \(\|moved\)$"):
                    Transform.define("x")
            layout = cmds.listConnections(f"{empty}.hyperLayout") or []
            self.assertEqual(_scene() - before, set(layout))
        self.assertTrue(cmds.objExists(str(empty)))
        self.assertIsNone(cmds.container(str(empty), query=True, nodeList=True))


class TestDefineInContainer(_Case):
    """CC-11 and the flattened scope's key."""

    def test_a_flattened_scope_prefixes_the_key(self):
        with container("arm") as arm:
            with container("inner"):
                made  = Transform.define("k", tx=1)
                again = Transform.define("k")
        self.assertEqual((str(made), again), ("inner_k", made))
        self.assertEqual(cmds.container(str(arm), query=True, nodeList=True), ["inner_k"])
        # outside the scope 'k' is another key
        self.assertEqual(Transform.define("k").long_name, "|k")

    def test_a_rerun_in_a_new_container_is_refused(self):
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
                r"Delete 'arm' and 'arm1' to rebuild it, or build in a new scene\.$",
                lambda: Transform.define("arm_root"),
            )
        self.assertEqual(cmds.container(str(first), query=True, nodeList=True), ["arm_root"])
        self.assertIsNone(cmds.container(str(second), query=True, nodeList=True))

    def test_a_node_from_another_container_is_refused(self):
        with container("arm"):
            root = Transform.define("rig_root")
        with container("leg") as leg:
            self.assertRefused(
                ValueError,
                r"^'rig_root' belongs to container 'arm'; define inside 'leg' only finds nodes this "
                r"build scope owns\. Refer to it with Transform\('rig_root'\), or define it outside "
                r"the containers\.$",
                lambda: Transform.define("rig_root"),
            )
            self.assertEqual(Transform("rig_root"), root)
        self.assertIsNone(cmds.container(str(leg), query=True, nodeList=True))
        # outside any scope it is found
        self.assertEqual(Transform.define("rig_root"), root)

    def test_a_node_an_enclosing_container_owns_is_found(self):
        with container("rig") as rig:
            root = Transform.define("root")
            with container("arm", preserve=True) as arm:
                self.assertEqual(Transform.define("root"), root)
                Transform.define("arm_ctl")
        self.assertEqual(cmds.container(str(rig), query=True, nodeList=True), ["root", "arm"])
        self.assertEqual(cmds.container(str(arm), query=True, nodeList=True), ["arm_ctl"])

    def test_registries_are_checked_only_with_container_true(self):
        with container("arm") as arm:
            layer = DisplayLayer.define("L", container=True)
        self.assertEqual(cmds.container(str(arm), query=True, nodeList=True), ["L"])
        with container("leg"):
            self.assertEqual(DisplayLayer.define("L"), layer)
            self.assertRefused(ValueError, r"^'L' belongs to container 'arm'",
                               lambda: DisplayLayer.define("L", container=True))

    def test_found_nodes_are_never_enrolled(self):
        free = Transform.define("free")
        with container("box") as box:
            self.assertEqual(Transform.define("free", tx=2, update=True), free)
        self.assertIsNone(cmds.container(str(box), query=True, nodeList=True))
        self.assertEqual(cmds.getAttr("free.tx"), 2.0)
        self.assertIsNone(cmds.container(query=True, findContainer=["free"]))

    def test_without_real_containers_nothing_is_checked(self):
        with container("arm"):
            Transform.define("arm_root")
        set_options(create_containers=False)
        try:
            with container("again"):
                self.assertEqual(str(Transform.define("arm_root")), "arm_root")
        finally:
            set_options(create_containers=True)
