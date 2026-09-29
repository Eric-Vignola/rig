"""Round 4b, step NC2: ``Cls(x)`` refers, the canonical typed node, type-checked.

* ``TestReference``: a node class other than ``Node`` is the strict typed
  reference. ``Cls(x)`` takes what ``Node(x)`` takes, by the same lookup rule,
  and returns the node ``Node(x)`` gives when it is an instance of ``Cls``: the
  most derived class (``Transform("j1")`` is a ``Joint``, equal to
  ``Node("j1")`` both ways; D9 amended), and ``Cls(x) is x`` for a node object
  of the class. A node of another type raises NodeTypeError, the DG classes
  included (``DisplayLayer("red")`` on a blinn); a subtype passes
  (``ObjectSet("initialShadingGroup")`` is the ``ShadingEngine``). The hooks:
  a geometry class takes a transform's first non-intermediate shape of its
  type, the unregistered ``Container`` takes a ``container`` node, and a user
  wrapper class with no node type of its own wraps what its ``is_type``
  accepts. No argument, ``None``, a second argument or a keyword raises
  TypeError. Nothing here writes the scene.
* ``TestInternalSites``: every ``X(...)`` call of a node class in the library
  (and ``cls(...)`` / ``type(self)(...)`` inside a node class) is on an
  allowlist with its reason: a site whose node's exact class is known uses
  ``X._wrap(x)`` (today's constructor), the others take user input and keep
  the reference. A new site fails until it is listed. The cast core and the
  converted sites never run the reference (round 5 A2 builds on it).
* ``TestExists``: ``Cls.exists(x)`` is True exactly when ``Cls(x)`` returns a
  node; False for a missing name, another type, a deleted node object and no
  name (``None``); an ambiguous name and a pattern raise.
"""

import ast
import collections
import os
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import (
    AmbiguousNodeError,
    Container,
    Node,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
)
from rig.nodetypes import (
    _base,
    Choice,
    DAGNode,
    DGNode,
    DisplayLayer,
    Geometry,
    Joint,
    Mesh,
    NurbsCurve,
    ObjectSet,
    Reference,
    ShadingEngine,
    SkinCluster,
    Transform,
)
from rig._tests._base import MayaTestCase


def _build_scene():
    """``j1`` and ``spine_01`` (joints), ``grp`` (an empty transform), ``red``
    (a blinn), ``body`` (a cube, ``bodyShape``), ``crv`` (a curve), ``|g1|a``
    and ``|g2|a``, ``box`` (a container), ``L`` (a display layer), ``s1`` (an
    object set). The current namespace is the root."""
    cmds.createNode("joint", name="j1")
    cmds.createNode("joint", name="spine_01")
    cmds.createNode("transform", name="grp")
    cmds.shadingNode("blinn", asShader=True, name="red")
    cmds.polyCube(name="body", ch=False)
    cmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], name="crv")
    cmds.rename(cmds.listRelatives("crv", shapes=True)[0], "crvShape")
    for group in ("g1", "g2"):
        cmds.createNode("transform", name=group)
        cmds.createNode("transform", name="a", parent=group)
    cmds.createNode("container", name="box")
    cmds.createDisplayLayer(name="L", empty=True)
    cmds.sets(name="s1", empty=True)
    cmds.select(clear=True)


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


class TestReference(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _build_scene()

    def tearDown(self):
        cmds.namespace(set=":")
        super().tearDown()

    def assertRefused(self, error, pattern, call):
        """``call`` raises ``error`` matching ``pattern`` and writes nothing."""
        before = set(cmds.ls())
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(set(cmds.ls()), before)

    # -- the most derived class

    def test_a_subtype_comes_back_most_derived(self):
        joint = Transform("j1")
        self.assertIs(type(joint), Joint)
        self.assertEqual(repr(joint), 'Joint("j1")')
        # equal to Node(x) both ways, one hash
        self.assertEqual(joint, Node("j1"))
        self.assertEqual(Node("j1"), joint)
        self.assertEqual(hash(joint), hash(Node("j1")))
        self.assertEqual(len({joint: 1, Node("j1"): 2}), 1)
        for cls in (DGNode, DAGNode, Transform, Joint):
            with self.subTest(cls=cls.__name__):
                self.assertIs(type(cls("j1")), Joint)

    def test_the_class_of_the_node_node_x_gives(self):
        for name in ("j1", "grp", "red", "bodyShape", "crv", "L", "s1", "initialShadingGroup"):
            with self.subTest(name=name):
                self.assertIs(type(DGNode(name)), type(Node(name)))
                self.assertEqual(DGNode(name), Node(name))

    def test_a_node_of_the_class_is_itself(self):
        grp, joint, mesh = Node("grp"), Node("j1"), Node("bodyShape")
        for cls, node in (
            (DGNode, grp), (DAGNode, grp), (Transform, grp),
            (Transform, joint), (Joint, joint), (Mesh, mesh), (Geometry, mesh),
        ):
            with self.subTest(cls=cls.__name__, node=str(node)):
                self.assertIs(cls(node), node)

    def test_the_inputs_node_takes(self):
        grp = Node("grp")
        sel = OpenMaya.MSelectionList()
        sel.add("j1")
        cases = (
            ("a dotted name", lambda: Transform("grp.tx"), grp),
            ("a plug", lambda: Transform(grp.tx), grp),
            ("a typed attribute", lambda: Transform(grp.find_attr("tx")), grp),
            ("an MObject", lambda: Transform(_mobject("j1")), Node("j1")),
            ("an MDagPath", lambda: DAGNode(sel.getDagPath(0)), Node("j1")),
            ("an MPlug", lambda: Transform(grp.tx.plug), grp),
            ("a uuid", lambda: Transform(cmds.ls("grp", uuid=True)[0]), grp),
            ("a path", lambda: Transform("|g1|a"), Node("|g1|a")),
            # the attribute of a dotted name is not checked (C11)
            ("a dotted name of no attribute", lambda: Transform("grp.nosuch"), grp),
            ("a component plug", lambda: Mesh(Node("body").vtx[0]), Node("bodyShape")),
        )
        for label, call, expected in cases:
            with self.subTest(input=label):
                node = call()
                self.assertIs(type(node), type(expected))
                self.assertEqual(node, expected)
        self.assertIs(Transform(grp.tx), grp)
        self.assertRefused(
            NodeTypeError, "'bodyShape' is a mesh, not a transform",
            lambda: Transform(Node("body").vtx[0]),
        )

    # -- another type

    def test_another_type_raises_node_type_error(self):
        before = set(cmds.ls())
        with self.assertRaises(NodeTypeError) as caught:
            Joint("grp")
        self.assertEqual(set(cmds.ls()), before)
        error = caught.exception
        self.assertEqual(
            str(error), "'grp' is a transform, not a joint; Node('grp') is Transform(\"grp\")"
        )
        self.assertEqual((error.name, error.label, error.node_type), ("grp", "joint", "transform"))
        # one of the family: a TypeError and a ValueError, never a lookup error
        self.assertIsInstance(error, TypeError)
        self.assertIsInstance(error, ValueError)
        self.assertNotIsInstance(error, NodeLookupError)

    def test_the_dg_classes_check_the_type(self):
        cases = (
            (lambda: DisplayLayer("red"), "'red' is a blinn, not a displayLayer"),
            (lambda: DisplayLayer("grp"), "'grp' is a transform, not a displayLayer"),
            (lambda: ShadingEngine("grp"), "'grp' is a transform, not a shadingEngine"),
            (lambda: ShadingEngine("red"), "'red' is a blinn, not a shadingEngine"),
            (lambda: ShadingEngine("s1"), "'s1' is an objectSet, not a shadingEngine"),
            (lambda: ObjectSet("grp"), "'grp' is a transform, not an objectSet"),
            (lambda: Choice("red"), "'red' is a blinn, not a choice"),
            (lambda: Reference("grp"), "'grp' is a transform, not a reference"),
            (lambda: SkinCluster("L"), "'L' is a displayLayer, not a skinCluster"),
            (lambda: Transform("red"), "'red' is a blinn, not a transform"),
            (lambda: DAGNode("red"), "'red' is a blinn, not a DAG node"),
        )
        for call, text in cases:
            with self.subTest(text=text):
                self.assertRefused(NodeTypeError, text, call)

    def test_an_engine_is_an_object_set(self):
        engine = ObjectSet("initialShadingGroup")
        self.assertIs(type(engine), ShadingEngine)
        self.assertEqual(engine, Node("initialShadingGroup"))
        self.assertEqual(Node("initialShadingGroup"), engine)
        self.assertIs(type(ObjectSet("s1")), ObjectSet)
        self.assertIs(type(DGNode("L")), DisplayLayer)

    def test_the_old_exact_class_wrapper_is_gone(self):
        # a Transform object of a joint, a DGNode object of a blinn used to
        # come back unequal to Node(x); now each class call is Node(x)
        for cls, name in ((Transform, "j1"), (DGNode, "bodyShape"), (DAGNode, "crvShape")):
            with self.subTest(cls=cls.__name__, name=name):
                self.assertIs(type(cls(name)), type(Node(name)))
                self.assertEqual(cls(name), Node(name))
        # the exact-class constructor is private, for sites that know the type
        self.assertIs(type(Transform._wrap("j1")), Transform)
        self.assertFalse(hasattr(Node("grp"), "_wrap"))

    def test_equality_with_a_non_node_is_not_implemented(self):
        grp = Node("grp")
        self.assertIs(grp.__eq__("grp"), NotImplemented)
        self.assertIs(grp.__eq__(1), NotImplemented)
        before = set(cmds.ls())
        self.assertFalse(grp == "grp")
        self.assertTrue(grp != "grp")
        # a plug answers for itself: never equal, and no node is built
        self.assertFalse(grp == grp.tx)
        self.assertTrue(grp != grp.tx)
        self.assertIn(grp, [grp.tx, grp])
        self.assertEqual(set(cmds.ls()), before)
        # an object that answers for nodes is asked
        class _AnyNode:
            def __eq__(self, other):
                return isinstance(other, DGNode)

        self.assertTrue(grp == _AnyNode())
        # between nodes: the class and the name, as before
        self.assertIs(grp.__eq__(Node("j1")), False)
        self.assertEqual(Container("box"), Node("box"))
        self.assertFalse(Container("box") == "box")

    # -- the hooks

    def test_a_geometry_class_takes_a_transform_for_its_shape(self):
        self.assertEqual(repr(Mesh("body")), 'Mesh("bodyShape")')
        self.assertEqual(repr(Mesh(Node("body"))), 'Mesh("bodyShape")')
        self.assertEqual(repr(Geometry("body")), 'Mesh("bodyShape")')
        self.assertEqual(repr(NurbsCurve("crv")), 'NurbsCurve("crvShape")')
        self.assertEqual(repr(Geometry("crv")), 'NurbsCurve("crvShape")')
        # the first NON-intermediate shape of the class's type
        orig = cmds.createNode("mesh", name="bodyOrig", parent="body")
        cmds.setAttr(f"{orig}.intermediateObject", True)
        cmds.reorder(orig, front=True)
        self.assertEqual(repr(Mesh("body")), 'Mesh("bodyShape")')
        # an instanced shape: the path through the transform named
        cmds.instance("body", name="body_inst")
        self.assertEqual(Mesh("body_inst").name, "body_inst|bodyShape")
        # refused: no shape of the type, a shape of another type, a joint
        cases = (
            (lambda: Mesh("grp"), "'grp' is a transform, not a mesh; Node\\('grp'\\) is "
                                  "Transform\\(\"grp\"\\), which has no mesh shape"),
            (lambda: NurbsCurve("body"), "which has no nurbsCurve shape"),
            (lambda: Mesh("crvShape"), "'crvShape' is a nurbsCurve, not a mesh"),
            (lambda: Mesh("j1"), "'j1' is a joint, not a mesh"),
            (lambda: Mesh("red"), "'red' is a blinn, not a mesh"),
        )
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(NodeTypeError, pattern, call)

    def test_container(self):
        box = Container("box")
        self.assertIs(type(box), Container)
        self.assertEqual(repr(box), 'Container("box")')
        self.assertIs(Container(box), box)
        # a cast of a container is a plain node, equal both ways
        self.assertEqual(box, Node("box"))
        self.assertEqual(Node("box"), box)
        self.assertEqual(Container(Node("box")), box)
        self.assertRefused(
            NodeTypeError, "'grp' is a transform, not a container", lambda: Container("grp")
        )
        self.assertRefused(NodeNotFoundError, "no container named 'nosuch'", lambda: Container("nosuch"))

    def test_a_custom_node_type_class(self):
        class _RefCtl(Transform):
            CUSTOM_NODE_TYPE = "r4bRefCtl"

        ctl = _RefCtl.create(name="ctl")
        self.assertIs(type(_RefCtl("ctl")), _RefCtl)
        # the most derived class through its base class too
        self.assertIs(type(Transform("ctl")), _RefCtl)
        self.assertEqual(Transform("ctl"), ctl)
        # a plain transform is not one
        self.assertRefused(
            NodeTypeError, "'grp' is a transform, not a r4bRefCtl", lambda: _RefCtl("grp")
        )

    def test_an_unregistered_user_class_wraps_what_its_is_type_accepts(self):
        class _Wrapper(Transform):
            pass

        node = _Wrapper("grp")
        self.assertIs(type(node), _Wrapper)
        self.assertEqual(node.name, "grp")
        self.assertIs(_Wrapper(node), node)
        # its constructor's rule: a joint passes Transform's is_type
        self.assertIs(type(_Wrapper("j1")), _Wrapper)
        self.assertRefused(NodeTypeError, "'red' is a blinn, not a transform", lambda: _Wrapper("red"))
        self.assertRefused(NodeNotFoundError, "no transform named 'nosuch'", lambda: _Wrapper("nosuch"))

    # -- what a class call refuses

    def test_no_name_none_and_attributes_are_type_errors(self):
        cases = (
            (lambda: Transform(), TypeError,
             r"^Transform\(\) names no node: Transform\('x'\) refers to x; "
             r"Transform\.define\('x'\) finds or makes it; Transform\.create\(name='x'\) makes one$"),
            (lambda: Transform(None), TypeError, r"^None is not a transform name$"),
            # re-pinned (round 4b NC6): a membership class's None names its kind token
            (lambda: DisplayLayer(None), TypeError,
             r"^None is not a layer name; Layer\(\) is defaultLayer \(it removes from every layer\)$"),
            (lambda: ObjectSet(None), TypeError, r"^None is not an objectSet name$"),
            (lambda: Container(None), TypeError, r"^None is not a container name$"),
            (lambda: Transform("grp", tx=1), TypeError,
             r"^Transform\('grp', \.\.\.\) refers to an existing transform and takes no "
             r"attributes; Transform\.define\('grp', \.\.\.\) finds or makes it, "
             r"Transform\.create\(name='grp', \.\.\.\) makes a new one$"),
            (lambda: Transform("grp", "x"), TypeError, r"refers to an existing transform"),
            (lambda: Transform(name="grp"), TypeError, r"^Transform\('grp', \.\.\.\) refers"),
            (lambda: Joint(name="new_joint"), TypeError, r"Joint\.create\(name='new_joint'"),
            (lambda: DisplayLayer("L", displayType=2), TypeError, r"existing displayLayer"),
        )
        for call, error, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertRefused(error, pattern, call)
        # a refusal is not a lookup error
        with self.assertRaises(TypeError) as caught:
            Transform(None)
        self.assertNotIsInstance(caught.exception, (NodeLookupError, NodeTypeError))

    def test_a_miss_names_the_class(self):
        self.assertRefused(
            NodeNotFoundError,
            r"^no joint named 'spnie_01' \(did you mean 'spine_01'\?\)$",
            lambda: Joint("spnie_01"),
        )
        self.assertRefused(NodeNotFoundError, r"^no mesh named 'nosuch'", lambda: Mesh("nosuch"))
        self.assertRefused(
            AmbiguousNodeError, "'a' is ambiguous: it names 2 nodes", lambda: Transform("a")
        )
        self.assertRefused(NodeLookupError, "'gr\\*' is a pattern, not a transform name",
                           lambda: Transform("gr*"))

    def test_one_lookup_rule_in_a_namespace(self):
        cmds.createNode("transform", name="x")
        cmds.namespace(add="char")
        cmds.createNode("transform", name="char:x")
        cmds.createNode("joint", name="char:root")
        cmds.namespace(set="char")
        for relative in (False, True):
            cmds.namespace(relativeNames=relative)
            try:
                with self.subTest(relativeNames=relative):
                    self.assertEqual(Joint("root"), Node(":char:root"))
                    # the typed hit used to return :x here (Maya's rule)
                    self.assertRefused(
                        AmbiguousNodeError, "names :x and :char:x; spell the namespace",
                        lambda: Transform("x"),
                    )
                    self.assertEqual(Transform(":x").uuid, cmds.ls(":x", uuid=True)[0])
                    self.assertRefused(NodeLookupError, "is a pattern", lambda: Transform("x*"))
            finally:
                cmds.namespace(relativeNames=False)

    def test_a_held_node_across_undo_and_a_new_scene(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        held = Transform.create(name="held")
        cmds.undo()
        self.assertRefused(NodeNotFoundError, "no transform named 'held'", lambda: Transform("held"))
        cmds.redo()
        self.assertEqual(Transform("held"), held)
        self.assertIs(Transform(held), held)
        cmds.file(new=True, force=True)
        self.assertRefused(NodeNotFoundError, "no transform named 'held'", lambda: Transform("held"))
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            Joint(held)
        # the name reused by a new node refers to the new node
        cmds.createNode("joint", name="held")
        self.assertIs(type(Transform("held")), Joint)
        self.assertTrue(Transform("held").is_valid)

    def test_a_renamed_node(self):
        grp = Node("grp")
        cmds.rename("grp", "moved")
        self.assertRefused(NodeNotFoundError, "no transform named 'grp'", lambda: Transform("grp"))
        self.assertIs(Transform(grp), grp)
        self.assertEqual(Transform("moved"), grp)

    def test_inside_a_container_scope_it_only_reads(self):
        from rig import container

        with container("arm") as arm:
            before = set(cmds.ls())
            self.assertEqual(Transform("grp"), Node("grp"))
            self.assertIs(type(ObjectSet("initialShadingGroup")), ShadingEngine)
            self.assertEqual(Joint("j1"), Node("j1"))
            with self.assertRaises(NodeTypeError):
                Joint("grp")
            self.assertEqual(set(cmds.ls()), before)
        self.assertNotIn("grp", cmds.container(str(arm), query=True, nodeList=True) or [])


# The library's node-class calls, (file, enclosing def, callee) -> (count, why).
# A site whose node's exact class is known calls X._wrap(x) (today's constructor)
# and is not listed; these take user input (or answer for it) and keep the
# reference, which type-checks and returns the most derived class.
_ALLOWED_SITES = {
    ("nodetypes/dg_node.py", "DGNode.exists", "cls"): (
        1, "the reference itself: exists answers whether cls(name) returns a node"),
    ("nodetypes/dg_node.py", "_define", "DAGNode"): (
        1, "user input: define's parent=, resolved by the reference rule before any write"),
    ("nodetypes/dag_node.py", "_parent_path", "DAGNode"): (
        1, "user input: a create's parent= (DAGNode._create, container.createNode, rn), "
           "resolved by the reference rule define reads parent= with (FIX)"),
    ("nodetypes/follicle.py", "Follicle.create_on_mesh", "Transform"): (
        1, "user input: the mesh transform to attach to"),
    ("nodetypes/joint.py", "Joint.match_hierarchy", "Joint"): (
        1, "user input: the target joint, by name"),
    ("nodetypes/mesh.py", "Mesh.transfer_component_tags", "Mesh"): (
        1, "user input: the other mesh (a transform stands for its shape)"),
    ("nodetypes/mesh.py", "Mesh.transfer_maps", "Mesh"): (
        1, "user input: the other mesh (a transform stands for its shape)"),
    ("nodetypes/skincluster.py", "flatten_vertices", "Mesh"): (
        2, "user input: the names of the vertex strings given"),
    ("nodetypes/skincluster.py", "SkinCluster._sanitize_influences", "Joint"): (
        1, "user input: an influence name, after Joint.exists"),
    ("nodetypes/skincluster.py", "SkinCluster.connect_bind_pre_matrices", "Joint"): (
        1, "user input: the driver the caller's search function names"),
    ("nodetypes/skincluster.py", "SkinCluster.set_influence_objects", "Joint"): (
        1, "user input: the influences given"),
    ("nodetypes/skincluster.py", "SkinCluster.get_mesh", "Mesh"): (
        1, "the deformed geometry may be another type: the reference type-checks it"),
    ("nodetypes/skincluster.py", "SkinCluster.transfer_to_mesh", "Mesh"): (
        1, "user input: the other mesh"),
    ("nodetypes/skincluster.py", "SkinCluster.copy_skincluster", "Mesh"): (
        2, "user input: the source meshes"),
    # round 4b NC7: the rig.shade specs retired; rig.shade.Material is the node class
    ("shade.py", "convert", "Material"): (
        1, "user input: the shader to convert, a node or a name"),
}


def _node_class_names():
    """The names of the library's node classes (every class under ``Node``
    defined in a ``rig`` module outside the tests), ``Node`` excluded."""
    names, todo = set(), [Node]
    while todo:
        cls = todo.pop()
        for sub in cls.__subclasses__():
            todo.append(sub)
            module = sub.__module__
            if module.startswith("rig.") and not module.startswith("rig._tests"):
                names.add(sub.__name__)
    return names


def _library_sites(root, class_names):
    """``(file, enclosing def, callee) -> count`` of the node-class calls in
    ``root`` (not ``_tests``, not ``examples``)."""
    sites = collections.Counter()
    for folder, subs, files in os.walk(root):
        subs[:] = [s for s in subs if s not in ("_tests", "examples", "__pycache__")]
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(folder, file)
            rel  = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())

            def visit(node, classes, defs):
                if isinstance(node, ast.ClassDef):
                    classes, defs = classes + [node.name], defs + [node.name]
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs = defs + [node.name]
                elif isinstance(node, ast.Call):
                    func, callee = node.func, None
                    in_node_class = bool(classes) and classes[-1] in class_names
                    if isinstance(func, ast.Name):
                        if func.id in class_names or (func.id == "cls" and in_node_class):
                            callee = func.id
                    elif (
                        isinstance(func, ast.Attribute)
                        and func.attr in class_names
                        and isinstance(func.value, ast.Name)
                        and func.value.id in ("nodetypes", "rig")
                    ):
                        callee = func.attr
                    elif (
                        isinstance(func, ast.Call)
                        and isinstance(func.func, ast.Name)
                        and func.func.id == "type"
                        and in_node_class
                    ):
                        callee = "type(...)"
                    if callee:
                        sites[(rel, ".".join(defs), callee)] += 1
                for child in ast.iter_child_nodes(node):
                    visit(child, classes, defs)

            visit(tree, [], [])
    return sites


class TestInternalSites(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _build_scene()

    def test_the_cast_never_runs_the_reference(self):
        calls = []
        original = _base._refer

        def counting(cls, args, kwargs):
            calls.append(cls.__name__)
            return original(cls, args, kwargs)

        with mock.patch.object(_base, "_refer", counting):
            for name in ("j1", "grp", "red", "bodyShape", "L", "s1", "box", "initialShadingGroup"):
                Node(name)
                _base._cast(name)
                _base._cast(_mobject(name))
            Node.wrap(cmds.ls(type="transform"))
            Transform.create(name="made")
            Joint.find_all()
            ShadingEngine.for_material("red")
            DisplayLayer.for_node("body")
            self.assertEqual(calls, [])
            Transform("grp")
            self.assertEqual(calls, ["Transform"])

    def test_every_node_class_call_is_listed(self):
        class_names = _node_class_names()
        # the classes the pin knows (a new node class joins by itself)
        self.assertLessEqual(
            {"DGNode", "DAGNode", "Transform", "Joint", "Mesh", "ShadingEngine",
             "DisplayLayer", "ObjectSet", "Container", "SkinCluster"},
            class_names,
        )
        sites   = _library_sites(os.path.dirname(rig.__file__), class_names)
        allowed = {key: count for key, (count, _why) in _ALLOWED_SITES.items()}
        new     = {key: n for key, n in sites.items() if allowed.get(key) != n}
        stale   = {key: n for key, n in allowed.items() if sites.get(key) != n}
        self.assertEqual(
            new, {},
            "a node-class call Cls(x) the pin does not know: use Cls._wrap(x) when "
            "the node's exact class is known (just made, read by type, type-checked), "
            "else list the site in _ALLOWED_SITES with its reason",
        )
        self.assertEqual(stale, {}, "an _ALLOWED_SITES entry no longer matches the code")

    def test_the_scan_finds_a_site(self):
        # the scanner itself: a cls(...) in a node class, a typed call, a
        # _wrap (not a site) and a call in a plain class
        import tempfile

        source = (
            "class Transform:\n"
            "    @classmethod\n"
            "    def f(cls, x):\n"
            "        cls._wrap(x)\n"
            "        return cls(x), Joint(x), type(self)(x)\n"
            "class Spec:\n"
            "    def g(cls):\n"
            "        return cls()\n"
        )
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "m.py"), "w", encoding="utf-8") as handle:
                handle.write(source)
            sites = _library_sites(root, {"Transform", "Joint"})
        self.assertEqual(
            dict(sites),
            {
                ("m.py", "Transform.f", "cls"): 1,
                ("m.py", "Transform.f", "Joint"): 1,
                ("m.py", "Transform.f", "type(...)"): 1,
            },
        )


class TestExists(MayaTestCase):
    """``Cls.exists(x)`` is True exactly when ``Cls(x)`` returns a node (C12)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        _build_scene()

    def tearDown(self):
        cmds.namespace(set=":")
        super().tearDown()

    def test_the_c12_rows(self):
        before = set(cmds.ls())
        # a transform is not a layer, a blinn is not an engine (both True before)
        self.assertFalse(DisplayLayer.exists("body"))
        self.assertFalse(ShadingEngine.exists("red"))
        # ambiguous raises (a TypeError before): False would let
        # `if not exists: create` add a third 'a'
        with self.assertRaisesRegex(AmbiguousNodeError, "'a' is ambiguous"):
            Transform.exists("a")
        # a pattern is a search, not a name (True before)
        with self.assertRaises(NodeLookupError) as caught:
            DGNode.exists("red*")
        self.assertNotIsInstance(caught.exception, (NodeNotFoundError, AmbiguousNodeError))
        self.assertEqual(set(cmds.ls()), before)

    def test_answers_whether_the_reference_returns_a_node(self):
        rows = (
            (Transform, "j1", True),  # a joint is a transform (False before)
            (Joint, "j1", True),
            (Joint, "grp", False),
            (Transform, "nosuch", False),
            (DGNode, "", False),
            (DGNode, "red", True),
            (ObjectSet, "initialShadingGroup", True),
            (ShadingEngine, "s1", False),
            (DisplayLayer, "L", True),
            (Mesh, "body", True),  # a transform with a mesh shape
            (Mesh, "grp", False),
            (NurbsCurve, "crv", True),
            (Container, "box", True),
            (Container, "grp", False),
            (Transform, "|g1|a", True),
            (Transform, "grp.tx", True),
        )
        for cls, name, expected in rows:
            with self.subTest(cls=cls.__name__, name=name):
                self.assertIs(cls.exists(name), expected)
                if expected:
                    self.assertIsInstance(cls(name), cls)

    def test_node_objects_and_none(self):
        grp = Node("grp")
        self.assertTrue(Transform.exists(grp))
        self.assertFalse(Joint.exists(grp))
        self.assertTrue(Transform.exists(grp.tx))
        cmds.delete("grp")
        self.assertFalse(Transform.exists(grp))
        # no name names no node (Transform(None) raises TypeError)
        self.assertIs(Transform.exists(None), False)
        self.assertIs(Transform.exists(""), False)

    def test_user_classes(self):
        class _ExCtl(Transform):
            CUSTOM_NODE_TYPE = "r4bExistsCtl"

        class _ExWrapper(Transform):
            pass

        _ExCtl.create(name="ctl")
        self.assertTrue(_ExCtl.exists("ctl"))
        self.assertFalse(_ExCtl.exists("grp"))
        self.assertTrue(Transform.exists("ctl"))
        self.assertTrue(_ExWrapper.exists("grp"))
        self.assertTrue(_ExWrapper.exists("j1"))
        self.assertFalse(_ExWrapper.exists("red"))

    def test_a_held_node_across_undo_rename_and_a_new_scene(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.flushUndo()
        held = Transform.create(name="held")
        cmds.undo()
        self.assertFalse(Transform.exists(held))
        self.assertFalse(Transform.exists("held"))
        cmds.redo()
        self.assertTrue(Transform.exists(held))
        cmds.rename("held", "moved")
        self.assertTrue(Transform.exists(held))
        self.assertFalse(Transform.exists("held"))
        self.assertTrue(Transform.exists("moved"))
        cmds.file(new=True, force=True)
        self.assertFalse(Transform.exists(held))
        self.assertFalse(Transform.exists("moved"))

    def test_in_a_namespace_it_follows_the_lookup_rule(self):
        cmds.createNode("transform", name="x")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        cmds.namespace(set="char")
        self.assertTrue(Joint.exists("root"))
        cmds.createNode("transform", name=":char:x")
        with self.assertRaisesRegex(AmbiguousNodeError, "spell the namespace"):
            Transform.exists("x")
        cmds.namespace(set=":")
        self.assertFalse(Joint.exists("root"))
        self.assertTrue(Transform.exists("x"))
