"""Round 4b follow-up F3: the typed reference ``Cls(x)`` and ``Node(x)`` build
a node of an already-cast type without its type check.

The cast builds a class that keeps the DGNode or DAGNode constructor, type
check, name property and API 1.0 cache, for a (typeName, typeId) it already
built once, without running ``is_type`` again (``_construct_checked_type``):
for a name as it did for an MObject, and from the MSelectionList the cast just
resolved the name in. Such a class's ``is_type`` only depends on the node type
and the custom type attr, which the type key already decided.

* ``TestSameNodes``: every class (and ``Node``) x every name of a mixed scene
  x three namespace modes (the root, ``char`` current, ``char`` current with
  ``namespace -relativeNames`` on): the node object a warm cast gives (its
  whole state: class, names, ``__dict__`` keys in order, fn set, path, API 1.0
  handle, attr cache) or the error it raises is what the full constructor
  gives, with the cast caches cleared before each call; walked forwards and
  backwards (another node of a type cast first); every MObject and DAG path too.
* ``TestTheShortcut``: a warm name cast of a transform, joint, camera or DG node
  runs no type query; the first cast of a type by name runs the constructor
  and marks the type (an MObject cast of it then skips the check too); the
  shortcut takes the cast's own selection for a name; a class with its own
  constructor or type check still runs them for a name.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node
from rig.nodetypes import _base
from rig.nodetypes import DAGNode, DGNode, Joint, Transform
from rig._tests._base import MayaTestCase


def _mobject(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _clear_caches():
    _base._CLASS_BY_TYPE.clear()
    _base._CASTABLE_TYPES.clear()


def _forget(*node_types):
    for node_type in node_types:
        _base._NODE_CLASS_DICT.pop(node_type, None)
    _clear_caches()


def _state(node):
    """Everything a node object holds, in the order its constructor set it."""
    d   = vars(node)
    out = [type(node).__name__, repr(node), node.name, node.long_name, list(d)]
    out.append(type(d["_fn_set"]).__name__)
    out.append((d["_fn_set1"].name(), d["_objhandle1"].isValid()))
    out.append(d["_mdagpath"].fullPathName() if "_mdagpath" in d else None)
    out.append(dict(d["_attr_dict"]))
    out.append(_mobject(node.long_name) == d["_mobject"])
    return out


def _outcome(call):
    """The state of the node ``call()`` returns, or the type and message of its error."""
    try:
        return ("returns", _state(call()))
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ("raises", type(error).__name__, str(error))


class TestSameNodes(MayaTestCase):
    TEST_START_NEW_SCENE = True

    NAMES = (
        "j1", "spine", "grp", "red", "ani", "body", "bodyShape", "|body|bodyShape",
        "body_inst", "body_inst|bodyShape", "crv", "curveShape1", "a", "|g1|a", "g1|a", "box",
        "L", "s1", "md", "net", "locShape", "emptyShape", "initialShadingGroup", "lambert1",
        "time1", "persp", "perspShape", "x", "root", ":root", "char:root", ":char:root",
        "char:x", ":char:x", "sub:x", ":sub:x", "ctl", "aliased", "held", "nosuch", "", "|",
        "1bad", "grp.tx", "red*", "0123456789abcdef0123456789abcdef",
    )

    def setUp(self):
        super().setUp()

        class _F3Ctl(Transform):
            CUSTOM_NODE_TYPE = "r4bfReferenceCtl"

        class _F3Wrapper(Transform):
            pass

        class _F3DG(DGNode):
            pass

        self.addCleanup(_forget, "r4bfReferenceCtl")
        self.addCleanup(cmds.namespace, setNamespace=":")
        self.addCleanup(cmds.namespace, relativeNames=False)
        self.extra_classes = [_F3Ctl, _F3Wrapper, _F3DG]
        cmds.createNode("joint", name="j1")
        cmds.createNode("joint", name="spine")
        cmds.createNode("transform", name="grp")
        cmds.shadingNode("blinn", asShader=True, name="red")
        cmds.shadingNode("anisotropic", asShader=True, name="ani")
        cmds.polyCube(name="body", ch=False)
        cmds.instance("body", name="body_inst")
        cmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], name="crv")
        for group in ("g1", "g2"):
            cmds.createNode("transform", name=group)
            cmds.createNode("transform", name="a", parent=group)
        cmds.createNode("container", name="box")
        cmds.createDisplayLayer(name="L", empty=True)
        cmds.sets(name="s1", empty=True)
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("network", name="net")
        cmds.createNode("locator", name="locShape")
        cmds.createNode("mesh", name="emptyShape")  # its MFnMesh refuses it
        cmds.createNode("transform", name="x")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        cmds.createNode("transform", name="char:x")
        cmds.namespace(add="sub")
        cmds.createNode("transform", name="sub:x")
        _F3Ctl.create(name="ctl")
        cmds.createNode("transform", name="aliased")
        cmds.aliasAttr(_base.CUSTOM_TYPE_ATTR, "aliased.tx")
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="held")
        cmds.undo()
        cmds.select(clear=True)

    def classes(self):
        found = list(dict.fromkeys(_base._NODE_CLASS_DICT.values()))
        return [Node] + found + [DAGNode, DGNode] + self.extra_classes

    def modes(self):
        for namespace, relative in ((":", False), ("char", False), ("char", True)):
            cmds.namespace(setNamespace=":")
            cmds.namespace(relativeNames=relative)
            cmds.namespace(setNamespace=namespace)
            yield namespace, relative
        cmds.namespace(setNamespace=":")
        cmds.namespace(relativeNames=False)

    def assert_warm_is_cold(self, calls):
        """Each call's outcome with the cast caches cleared before it (the full
        constructor) equals its outcome in a warm walk, forwards and backwards."""
        cold = {}
        for label, call in calls:
            _clear_caches()
            cold[label] = _outcome(call)
        self.assertTrue(any(o[0] == "returns" for o in cold.values()))
        for order in ("forwards", "backwards"):
            _clear_caches()
            walk = calls if order == "forwards" else calls[::-1]
            wrong = [(label, cold[label], got) for label, call in walk
                     for got in [_outcome(call)] if got != cold[label]]
            self.assertEqual(wrong, [], order)

    def test_every_class_and_name_gives_the_constructors_node(self):
        names = list(self.NAMES) + cmds.ls(["grp", "j1", "md", "char:root"], uuid=True)
        before = set(cmds.ls())
        for namespace, relative in self.modes():
            with self.subTest(namespace=namespace, relative=relative):
                self.assert_warm_is_cold([
                    ((cls.__name__, name), lambda cls=cls, name=name: cls(name))
                    for cls in self.classes() for name in names
                ])
        self.assertEqual(set(cmds.ls()), before)

    def test_every_mobject_and_dag_path_gives_the_constructors_node(self):
        inputs = []
        for long_name in cmds.ls(long=True):
            mobj = _mobject(long_name)
            inputs.append(("mobj " + long_name, mobj))
            if mobj.hasFn(OpenMaya.MFn.kDagNode):
                inputs += [("path " + p.fullPathName(), p)
                           for p in OpenMaya.MDagPath.getAllPathsTo(mobj)]
        for namespace, relative in self.modes():
            with self.subTest(namespace=namespace, relative=relative):
                self.assert_warm_is_cold([
                    ((cls.__name__, label), lambda cls=cls, obj=obj: cls(obj))
                    for cls in (Node, Transform, DAGNode, DGNode) for label, obj in inputs
                ])


class TestTheShortcut(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        registered = dict(_base._NODE_CLASS_DICT)

        def restore():
            _base._NODE_CLASS_DICT.clear()
            _base._NODE_CLASS_DICT.update(registered)
            _clear_caches()

        self.addCleanup(restore)
        _clear_caches()

    @staticmethod
    def type_queries(call):
        """What ``call()`` returns and how many ``nodeType`` / ``objectType`` /
        ``attributeQuery`` commands it ran."""
        probes = [mock.patch.object(cmds, name, wraps=getattr(cmds, name))
                  for name in ("nodeType", "objectType", "attributeQuery")]
        mocks = [probe.start() for probe in probes]
        try:
            result = call()
        finally:
            for probe in probes:
                probe.stop()
        return result, sum(m.call_count for m in mocks)

    def test_a_warm_name_cast_runs_no_type_query(self):
        cmds.createNode("transform", name="t1")
        cmds.createNode("transform", name="t2", parent="t1")
        cmds.createNode("joint", name="j1")
        cmds.createNode("joint", name="j2")
        cmds.createNode("camera", name="cam1Shape", parent="t1")  # a DAG node no class claims
        cmds.createNode("camera", name="cam2Shape", parent="t2")
        cmds.createNode("multiplyDivide", name="md1")
        cmds.createNode("multiplyDivide", name="md2")
        for first in ("t1", "j1", "cam1Shape", "md1"):
            Node(first)
        uuid = cmds.ls("t2", uuid=True)[0]
        for call, cls in (
            (lambda: Node("t2"), Transform),
            (lambda: Transform("t2"), Transform),
            (lambda: Transform("|t1|t2"), Transform),
            (lambda: Transform("t2.tx"), Transform),
            (lambda: Node(uuid), Transform),
            (lambda: Joint("j2"), Joint),
            (lambda: Transform("j2"), Joint),
            (lambda: Node("cam2Shape"), DAGNode),
            (lambda: DAGNode("cam2Shape"), DAGNode),
            (lambda: Node("md2"), DGNode),
            (lambda: DGNode("md2"), DGNode),
        ):
            with self.subTest(cls=cls.__name__):
                node, queries = self.type_queries(call)
                self.assertIs(type(node), cls)
                self.assertEqual(queries, 0)

    def test_the_first_name_cast_of_a_type_runs_the_constructor_and_marks_the_type(self):
        cmds.createNode("transform", name="t1")
        cmds.createNode("transform", name="t2")
        node, queries = self.type_queries(lambda: Transform("t1"))
        self.assertIs(type(node), Transform)
        self.assertGreater(queries, 0)
        self.assertIn(("transform", OpenMaya.MFnDependencyNode(_mobject("t1")).typeId.id()),
                      _base._CASTABLE_TYPES)
        # an MObject cast of the type then skips the check too
        node, queries = self.type_queries(lambda: Node(_mobject("t2")))
        self.assertIs(type(node), Transform)
        self.assertEqual(queries, 0)

    def test_a_name_is_built_from_the_casts_own_selection(self):
        cmds.createNode("transform", name="t1")
        cmds.createNode("transform", name="t2")
        Node("t1")
        checked = _base._construct_checked_type
        with mock.patch.object(_base, "_construct_checked_type", wraps=checked) as spy:
            by_name = Transform("t2")
            by_mobject = Node(_mobject("t2"))
        (name_args, _), (mobject_args, _) = spy.call_args_list
        self.assertEqual(name_args[:2], (Transform, "t2"))
        sel = name_args[2]
        self.assertIsInstance(sel, OpenMaya.MSelectionList)
        self.assertEqual(list(sel.getSelectionStrings()), ["t2"])
        self.assertEqual(mobject_args, (Transform, "t2", None))
        self.assertEqual(_state(by_name), _state(by_mobject))

    def test_a_class_with_its_own_constructor_or_type_check_runs_them_for_a_name(self):
        class _Network(DGNode):
            NATIVE_NODE_TYPE = "network"
            inits            = 0

            def __init__(self, node):
                type(self).inits += 1
                super().__init__(node)

        class _Condition(DGNode):
            NATIVE_NODE_TYPE = "condition"
            checks           = 0

            @classmethod
            def is_type(cls, node_name, failfast=False, **kwargs):
                cls.checks += 1
                return super().is_type(node_name, failfast=failfast, **kwargs)

        for node_type, cls, counter in (
            ("network", _Network, "inits"),
            ("condition", _Condition, "checks"),
        ):
            with self.subTest(node_type=node_type):
                nodes = [cmds.createNode(node_type) for _ in range(2)]
                for name in nodes + nodes:
                    self.assertIs(type(Node(name)), cls)
                self.assertIs(type(cls(nodes[0])), cls)
                self.assertEqual(getattr(cls, counter), 5)
