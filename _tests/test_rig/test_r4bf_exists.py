"""Round 4b follow-up F2: ``Cls.exists`` without building a node or raising.

``Cls.exists(x)`` answers whether ``Cls(x)`` returns a node. It now takes the
reference's own lookup in its probe mode (``_node_from_str(..., probe=True)``):
a miss gives None instead of a NodeNotFoundError, and a node whose class the
cast's type key decides is answered from that class (``_cast(x, build=False)``)
instead of being built.

* ``TestSameAnswers``: every class x every name of a mixed scene (joints,
  transforms, DAG duplicates, an instance, shapes, shaders, engines, sets, a
  layer, a container, a custom-typed node and class, an unregistered wrapper,
  namespaced and referenced nodes, paths, attributes, uuids, patterns, ``''``,
  a name undone away) at the root namespace, with ``char`` current, and with
  ``namespace -relativeNames`` on: the answer, or the error, of the r4b
  implementation (``try: cls(x)``, the oracle). Node objects and ``None`` too.
* ``TestTheProbe``: the class ``_cast(x, build=False)`` gives is the class
  ``_cast(x)`` builds; a plain name MSelectionList resolves to no node makes
  the cast fail (the probe's None); a node the lookup finds after the probe's
  None gives what the reference gives.
* ``TestNothingBuiltNothingRaised``: a hit of the class or a subclass builds no
  node, a miss makes no NodeNotFoundError and runs no cast by name, another
  type makes no NodeTypeError.
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node
from rig.nodetypes import _base
from rig.nodetypes import (
    DAGNode,
    DGNode,
    DisplayLayer,
    Joint,
    Mesh,
    NurbsCurve,
    ObjectSet,
    ShadingEngine,
    Transform,
)
from rig.nodetypes.errors import (
    AmbiguousNodeError,
    NodeLookupError,
    NodeNotFoundError,
    NodeTypeError,
)
from rig._tests._base import MayaTestCase


def _oracle(cls, name):
    """``Cls.exists`` as round 4b wrote it: the reference, caught."""
    if name is None:
        return False
    try:
        node = cls(name)
    except (NodeNotFoundError, NodeTypeError):
        return False
    return node.is_valid


def _outcome(call):
    """The answer of ``call()``, or the type and message of its error."""
    try:
        return ("returns", call())
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ("raises", type(error).__name__, str(error))


def _forget(*node_types):
    for node_type in node_types:
        _base._NODE_CLASS_DICT.pop(node_type, None)
    _base._CLASS_BY_TYPE.clear()
    _base._CASTABLE_TYPES.clear()


class _Scene(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        class _FCtl(Transform):
            CUSTOM_NODE_TYPE = "r4bfExistsCtl"

        class _FWrapper(Transform):
            pass

        self.addCleanup(_forget, "r4bfExistsCtl")
        self.ctl_class, self.wrapper_class = _FCtl, _FWrapper
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        self.addCleanup(cmds.file, new=True, force=True)
        self.addCleanup(cmds.namespace, setNamespace=":")
        self.addCleanup(cmds.namespace, relativeNames=False)
        # a referenced file
        cmds.createNode("transform", name="rtr")
        cmds.createNode("joint", name="rj", parent="rtr")
        path = os.path.join(folder, "r4bf_exists_ref.ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)

        cmds.createNode("joint", name="j1")
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
        cmds.createNode("transform", name="x")
        cmds.namespace(add="char")
        cmds.createNode("joint", name="char:root")
        cmds.createNode("transform", name="char:x")
        cmds.namespace(add="sub")
        cmds.createNode("transform", name="sub:x")
        _FCtl.create(name="ctl")
        cmds.file(path, reference=True, namespace="REF1")
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="held")
        cmds.undo()
        cmds.select(clear=True)


class TestSameAnswers(_Scene):
    NAMES = (
        "j1", "grp", "red", "ani", "body", "bodyShape", "|body|bodyShape", "body_inst",
        "crv", "curveShape1", "a", "|g1|a", "g1|a", "|g1|a|", "box", "L", "s1",
        "initialShadingGroup", "lambert1", "time1", "persp", "x", "root", ":root",
        "char:root", ":char:root", "char:x", ":char:x", "sub:x", ":sub:x", "char:sub:x",
        "char:", "nosuch:x", "ctl", "REF1:rtr", "REF1:rj", "rtr", "held", "nosuch", "",
        "|", ":", "1bad", "bad name", "grp.tx", "j1.rx", "red*", "gr[p]", "r?d",
        "0123456789abcdef0123456789abcdef", "6D9E4F2B-1111-2222-3333-444455556666",
    )

    def classes(self):
        found = list(dict.fromkeys(_base._NODE_CLASS_DICT.values()))
        return found + [DGNode, DAGNode, self.wrapper_class]

    def test_every_class_and_name_answers_as_the_reference(self):
        names = list(self.NAMES) + cmds.ls(["grp", "j1", "char:root", "REF1:rj"], uuid=True)
        before = set(cmds.ls())
        for namespace, relative in ((":", False), ("char", False), ("char", True)):
            cmds.namespace(setNamespace=":")
            cmds.namespace(relativeNames=relative)
            cmds.namespace(setNamespace=namespace)
            for cls in self.classes():
                for name in names:
                    with self.subTest(namespace=namespace, relative=relative, cls=cls.__name__,
                                      name=name):
                        self.assertEqual(
                            _outcome(lambda: cls.exists(name)),
                            _outcome(lambda: _oracle(cls, name)),
                        )
        cmds.namespace(setNamespace=":")
        cmds.namespace(relativeNames=False)
        self.assertEqual(set(cmds.ls()), before)

    def test_the_rows_nc2_pinned(self):
        grp = Node("grp")
        cases = (
            (Transform, "nosuch", False),  # a missing name
            (DisplayLayer, "grp", False),  # another type
            (Transform, "j1", True),  # a subtype
            (Transform, None, False),
            (Transform, "", False),
            (Transform, grp, True),  # a node object
            (Joint, grp, False),
            (Transform, grp.tx, True),
            (Transform, cmds.ls("grp", uuid=True)[0], True),  # a uuid
            (ObjectSet, "initialShadingGroup", True),
            (ShadingEngine, "s1", False),
            (Mesh, "body", True),  # a transform with a mesh shape
            (Mesh, "grp", False),
            (NurbsCurve, "crv", True),
            (self.ctl_class, "ctl", True),
            (self.ctl_class, "grp", False),
            (self.wrapper_class, "grp", True),
            (self.wrapper_class, "red", False),
        )
        for cls, name, expected in cases:
            with self.subTest(cls=cls.__name__, name=name):
                self.assertIs(cls.exists(name), expected)
        with self.assertRaisesRegex(AmbiguousNodeError, "'a' is ambiguous"):
            Transform.exists("a")
        with self.assertRaises(NodeLookupError) as caught:
            DGNode.exists("red*")
        self.assertNotIsInstance(caught.exception, (NodeNotFoundError, AmbiguousNodeError))
        cmds.delete("grp")
        self.assertIs(Transform.exists(grp), False)  # a deleted node object

    def test_namespaced_names_while_a_namespace_is_current(self):
        for relative in (False, True):
            with self.subTest(relativeNames=relative):
                cmds.namespace(relativeNames=relative)
                cmds.namespace(setNamespace="char")
                self.assertIs(Joint.exists("root"), True)
                self.assertIs(Transform.exists("sub:x"), True)
                self.assertIs(Transform.exists("char:x"), True)
                with self.assertRaisesRegex(AmbiguousNodeError, "spell the namespace"):
                    Transform.exists("x")
                cmds.namespace(setNamespace=":")
                self.assertIs(Joint.exists("root"), False)
                self.assertIs(Transform.exists("x"), True)
                cmds.namespace(relativeNames=False)


class TestTheProbe(_Scene):
    def test_the_probed_class_is_the_class_the_cast_builds(self):
        checked = 0
        for name in cmds.ls() + cmds.ls(long=True) + cmds.ls(uuid=True):
            probed = _outcome(lambda: _base._cast(name, build=False))
            built  = _outcome(lambda: _base._cast(name))
            with self.subTest(name=name):
                if probed[0] == "returns" and isinstance(probed[1], type):
                    checked += 1
                    self.assertEqual(built[0], "returns")
                    self.assertIs(type(built[1]), probed[1])
                elif probed[0] == "returns":
                    # a custom type (or an alias of its name): built by name
                    self.assertIs(type(built[1]), type(probed[1]))
                else:
                    self.assertEqual(probed, built)
        self.assertGreater(checked, 50)

    def test_a_plain_name_no_node_has_fails_the_cast(self):
        for name in ("nosuch", ":nosuch", "|nosuch", "char:nosuch", "nosuch:x", "|g1|b", "held",
                     "1bad", "|", ":", "char:", "char"):
            with self.subTest(name=name):
                self.assertTrue(_base._PLAIN_NODE_NAME.match(name))
                with self.assertRaises(RuntimeError):
                    OpenMaya.MSelectionList().add(name)
                self.assertIsNone(_base._cast(name, build=False))
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    _base._cast(name)

    def test_a_node_the_lookup_finds_after_the_probe_gives_the_references_answer(self):
        # the probe's None followed by a lookup that finds the node (a case
        # Maya does not make): the answer is the reference's cast
        real = _base._cast

        def unresolved(obj, build=True):
            return None if (obj == "grp" and not build) else real(obj, build)

        with mock.patch.object(_base, "_cast", unresolved):
            self.assertIs(Transform.exists("grp"), True)
            self.assertIs(Joint.exists("grp"), False)


class TestNothingBuiltNothingRaised(_Scene):
    def count(self, target, attribute):
        calls = []
        original = getattr(target, attribute)

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        patcher = mock.patch.object(target, attribute, spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_a_hit_builds_no_node(self):
        # bind the cast's constructor parts first (they compare __init__ by identity)
        _base._construct_checked_type(Transform, "grp")
        dg_built  = self.count(DGNode, "__init__")
        dag_built = self.count(DAGNode, "__init__")
        refers    = self.count(_base, "_refer")
        for cls, name in ((Transform, "grp"), (Transform, "j1"), (Joint, "j1"), (DGNode, "red"),
                          (DGNode, "grp"), (DisplayLayer, "L"), (Transform, "|g1|a")):
            with self.subTest(cls=cls.__name__, name=name):
                self.assertIs(cls.exists(name), True)
        self.assertEqual((dg_built, dag_built, refers), ([], [], []))

    def test_a_miss_raises_no_lookup_error_and_casts_nothing_by_name(self):
        made    = self.count(NodeNotFoundError, "__init__")
        by_name = self.count(_base, "_cast_by_name")
        for name in ("nosuch", "char:nosuch", "|g1|nosuch", "held", "1bad"):
            with self.subTest(name=name):
                self.assertIs(Transform.exists(name), False)
        self.assertEqual((made, by_name), ([], []))

    def test_another_type_makes_no_type_error(self):
        made = self.count(NodeTypeError, "__init__")
        for cls, name in ((Joint, "grp"), (DisplayLayer, "red"), (Mesh, "grp"),
                          (ShadingEngine, "s1")):
            with self.subTest(cls=cls.__name__, name=name):
                self.assertIs(cls.exists(name), False)
        self.assertEqual(made, [])
