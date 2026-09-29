"""Round 4b, step NC0: pins of what the construction rule must not move.

Written on the unmodified library (the tree after B1), before any 4b change:

* ``TestMathPathLookups``: the math and creation paths make no name lookup.
  Any later step that adds one here (``cmds.ls``, ``cmds.objExists``,
  ``cmds.namespaceInfo`` or the membership ``_find_node``) fails this test.
* ``TestAlwaysNew``: ``Node.create``, ``rn.*``, ``rc.*``, ``Cls.create``,
  ``with container(...)`` and ``container.createNode`` always make a new
  node: two calls, two nodes, the second uniquified by Maya.
* ``TestTagIdiomsStay``: the component-tag spellings 4b keeps, with their
  results (``components >> Tag('x')`` included: the ids form).
* ``TestPlugCloneStays``: ``plug >> node`` clones the attribute, ``plug >>
  'name'`` clones it on the same node (``'node.name'`` onto that node),
  ``plug >> container_node`` publishes (and the scope form passes through in
  a flattened scope).

The rows 4b flips (error types, the typed reference, the plug-left membership
forms, ``Spec(None)``, ``shared=``, the attribute re-declaration ...) are not
pinned here; the rows probe records them.
"""

from unittest import mock

import numpy as np
from maya import cmds

import rig._internal.members as _members
import rig.membership as _membership
import rig.shade as _shade
from rig import Container, Node, Plug, Tag, container
from rig.bridges import commands as rc
from rig.bridges import nodes as rn
from rig.nodetypes import Transform
from rig.spec import Float
from rig._tests._base import MayaTestCase


# The lookup counts of every scenario below, measured on the unmodified tree
# (runs/r4b/NC0/probe_pins.txt): warm, each scenario makes 0 calls of each.
# Cold, the first cast of a node type in the process runs
# ``cmds.ls(name, dag=True)`` once (the type cache of ``_native_node_class``)
# and the first ``rn.<type>`` lists the node types once (``cmds.ls(nt=True)``).
# Those are per-process type caches, not name lookups, and other tests clear or
# fill them, so each test runs its scenario once in a scene it then discards
# before it counts.
_WARM_LOOKUPS = {"ls": 0, "objExists": 0, "namespaceInfo": 0, "_find_node": 0}

_COUNTED_CMDS = ("ls", "objExists", "namespaceInfo")


def _count_lookups(run):
    """``(run(), counts)``: the calls of each counted name lookup while ``run``
    runs. ``_find_node`` is patched in every module that imports it (round 4b
    NC6: ``rig.membership`` no longer does, its layers being nodes)."""
    find_node = mock.MagicMock(wraps=_members._find_node)
    patches = [mock.patch.object(cmds, name, wraps=getattr(cmds, name)) for name in _COUNTED_CMDS]
    patches += [
        mock.patch.object(module, "_find_node", find_node)
        for module in (_members, _membership, _shade)
        if hasattr(module, "_find_node")
    ]
    mocks = [patch.start() for patch in patches]
    try:
        result = run()
    finally:
        for patch in patches:
            patch.stop()
    counts = {name: m.call_count for name, m in zip(_COUNTED_CMDS, mocks)}
    counts["_find_node"] = find_node.call_count
    return result, counts


def _members_of(name):
    return sorted(cmds.container(name, query=True, nodeList=True) or [])


class TestMathPathLookups(MayaTestCase):
    """The math and creation hot paths make no name lookup (0 ``cmds.ls`` /
    ``objExists`` / ``namespaceInfo`` / ``_find_node`` calls). Any later step
    adding a lookup here fails this test."""

    TEST_START_NEW_SCENE = True

    def _warm_then_count(self, build):
        """Run ``build()()`` once and discard its scene (the per-process type
        caches warm up), then count the lookups of ``build()()`` in a new
        scene."""
        build()()
        self.new_scene()
        return _count_lookups(build())

    def test_scoped_math_block(self):
        def build():
            a   = Node.create("transform", name="a")
            b   = Node.create("transform", name="b")
            out = Node.create("transform", name="out")
            out << Float("w", dv=0.5)
            a.t << (1, 2, 3)
            b.t << (3, 4, 5)

            def run():
                with container("pin"):
                    out.t << (b.t - a.t) * out.w + a.t

            return run

        _, counts = self._warm_then_count(build)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual(len(_members_of("pin")), 3)
        np.testing.assert_array_almost_equal(cmds.getAttr("out.t")[0], (2, 3, 4))

    def test_rn_factory_outside_and_inside_a_scope(self):
        def build_outside():
            return lambda: [rn.multiplyDivide(name="mul1") for _ in range(3)]

        made, counts = self._warm_then_count(build_outside)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual([str(n) for n in made], ["mul1", "mul2", "mul3"])

        def build_inside():
            def run():
                with container("pin"):
                    return [rn.multiplyDivide(name="mul1") for _ in range(3)]

            return run

        made, counts = self._warm_then_count(build_inside)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual(_members_of("pin"), sorted(str(n) for n in made))

    def test_node_create_outside_and_inside_a_scope(self):
        def make():
            return [Node.create("multiplyDivide", name="m"), Node.create("transform", name="t")]

        made, counts = self._warm_then_count(lambda: make)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual([str(n) for n in made], ["m", "t"])

        def build_inside():
            def run():
                with container("pin"):
                    return make()

            return run

        made, counts = self._warm_then_count(build_inside)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual(_members_of("pin"), sorted(str(n) for n in made))

    def test_rc_command_outside_and_inside_a_scope(self):
        made, counts = self._warm_then_count(lambda: lambda: rc.polyCube(name="c"))
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual([str(n) for n in made], ["c", "polyCube1"])

        def build_inside():
            def run():
                with container("pin"):
                    return rc.polyCube(name="c")

            return run

        made, counts = self._warm_then_count(build_inside)
        self.assertEqual(counts, _WARM_LOOKUPS)
        self.assertEqual([str(n) for n in made], ["c", "polyCube1"])
        self.assertTrue(set(str(n) for n in made) <= set(_members_of("pin")))


class TestAlwaysNew(MayaTestCase):
    """Two calls, two nodes: the creators never find a node by its name."""

    TEST_START_NEW_SCENE = True

    def _two(self, make):
        """The names of two nodes ``make()`` returns, checked distinct."""
        first, second = make(), make()
        self.assertNotEqual(first.uuid, second.uuid)
        return str(first), str(second)

    def test_node_create(self):
        self.assertEqual(self._two(lambda: Node.create("transform", name="x")), ("x", "x1"))
        self.assertEqual(type(Node("x1")), Transform)

    def test_rn_factory(self):
        self.assertEqual(self._two(lambda: rn.transform(name="r")), ("r", "r1"))

    def test_rc_command(self):
        made = [rc.polyCube(name="c") for _ in range(2)]
        self.assertEqual([[str(n) for n in m] for m in made], [["c", "polyCube1"], ["c1", "polyCube2"]])
        self.assertNotEqual(made[0][0].uuid, made[1][0].uuid)

    def test_typed_create(self):
        self.assertEqual(self._two(lambda: Transform.create(name="t")), ("t", "t1"))
        self.assertEqual(type(Node("t1")), Transform)

    def test_container_scope(self):
        scopes = []
        for _ in range(2):
            with container("a") as ctn:
                scopes.append(ctn)
        self.assertEqual([type(c) for c in scopes], [Container, Container])
        self.assertEqual([str(c) for c in scopes], ["a", "a1"])

    def test_flattened_scope_prefixes_then_uniquifies(self):
        with container("outer"):
            with container("inner") as inner:
                self.assertIsNone(inner)  # nested scopes flatten by default
                made = [container.createNode("multiplyDivide", name="mul1") for _ in range(2)]
        self.assertEqual([str(n) for n in made], ["inner_mul1", "inner_mul2"])
        self.assertEqual(_members_of("outer"), ["inner_mul1", "inner_mul2"])

    def test_inside_a_scope_too(self):
        with container("s"):
            made = {
                "Node.create":      self._two(lambda: Node.create("transform", name="x")),
                "rn":               self._two(lambda: rn.transform(name="r")),
                "Transform.create": self._two(lambda: Transform.create(name="t")),
                "container.createNode": self._two(
                    lambda: container.createNode("multiplyDivide", name="m")
                ),
            }
        self.assertEqual(
            made,
            {
                "Node.create":          ("x", "x1"),
                "rn":                   ("r", "r1"),
                "Transform.create":     ("t", "t1"),
                "container.createNode": ("m", "m1"),
            },
        )
        self.assertEqual(_members_of("s"), ["m", "m1", "r", "r1", "t", "t1", "x", "x1"])


class TestTagIdiomsStay(MayaTestCase):
    """The component-tag spellings round 4b keeps, with their results."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = Node(cmds.polyCube(name="cube", ch=False)[0])
        self.shape = Node(cmds.listRelatives("cube", shapes=True, fullPath=True)[0])

    def _names(self, tags):
        self.assertIsInstance(tags, list)
        for tag in tags:
            self.assertIsInstance(tag, Tag)
        return sorted(str(tag) for tag in tags)

    def test_face_members_add_idempotently_and_remove(self):
        faces = self.cube.f[:3]
        self.assertIs(faces << Tag("cap"), faces)
        self.assertEqual(self.shape.get_component_tag_contents("cap"), ["f[0:2]"])
        # adding again asserts a state that already holds: nothing written
        before = set(cmds.ls())
        again  = self.cube.f[:3]
        self.assertIs(again << Tag("cap"), again)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(self.shape.get_component_tag_contents("cap"), ["f[0:2]"])
        face = self.cube.f[0]
        self.assertIs(face << -Tag("cap"), face)
        ids = self.cube >> Tag("cap")
        self.assertIsInstance(ids, np.ndarray)
        np.testing.assert_array_equal(ids, [1, 2])
        # the baked tags of the cube are untouched
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])

    def test_node_on_the_left_creates_and_deletes_the_tag(self):
        self.assertIs(self.cube << Tag("empty"), self.cube)
        self.assertIn("empty", self.shape.component_tags)
        self.assertEqual((self.cube >> Tag("empty")).shape, (0,))
        self.assertIs(self.cube << -Tag("empty"), self.cube)
        self.assertNotIn("empty", self.shape.component_tags)
        with self.assertRaisesRegex(ValueError, "no component tag 'empty'"):
            self.cube >> Tag("empty")

    def test_no_name_purges_and_enumerates(self):
        self.cube.vtx[:8] << Tag("pts")
        self.assertEqual(self._names(self.cube.vtx[3] >> Tag()), ["pts"])
        lhs = self.cube.vtx[:5]
        self.assertIs(lhs << Tag(), lhs)
        np.testing.assert_array_equal(self.cube >> Tag("pts"), [5, 6, 7])
        self.assertEqual(self._names(self.cube.vtx[3] >> Tag()), [])
        self.assertEqual(self._names(self.cube.vtx[6] >> Tag()), ["pts"])
        # a vertex purge leaves the face tags alone
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])

    def test_components_on_the_left_read_their_ids(self):
        # ``components >> Tag('x')``: the native ids of the left-hand side that
        # are in the tag (the ids form 4b keeps; yes/no moves to ``in``)
        self.cube.vtx[2:6] << Tag("pts")
        self.cube.f[:3] << Tag("cap")
        for label, lhs, tag, expected in (
            ("vtx[4:8]", self.cube.vtx[4:8], "pts", [4, 5]),
            ("vtx",      self.cube.vtx,      "pts", [2, 3, 4, 5]),
            ("vtx[3]",   self.cube.vtx[3],   "pts", [3]),
            ("vtx[0]",   self.cube.vtx[0],   "pts", []),
            ("f[1:]",    self.cube.f[1:],    "cap", [1, 2]),
            ("f",        self.cube.f,        "cap", [0, 1, 2]),
        ):
            with self.subTest(lhs=label):
                ids = lhs >> Tag(tag)
                self.assertIsInstance(ids, np.ndarray)
                np.testing.assert_array_equal(ids, expected)
        # re-pinned (round 4b NC6, user decision 2026-09-28: a tag is untyped
        # for queries): vertices against a face tag are none of its members, an
        # empty id array (it was a TypeError at NC0)
        before = set(cmds.ls())
        ids    = self.cube.vtx[:2] >> Tag("cap")
        self.assertIsInstance(ids, np.ndarray)
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(set(cmds.ls()), before)

    def test_expression_sugar_writes_the_name(self):
        cluster = Node(cmds.cluster("cube")[0])
        expr    = cluster.input[0].componentTagExpression
        self.cube.vtx[:4] << Tag("lid")
        with mock.patch.object(cmds, "warning") as warning:
            self.assertIs(expr << Tag("lid"), expr)
        warning.assert_not_called()
        self.assertEqual(expr >> None, "lid")


class TestPlugCloneStays(MayaTestCase):
    """``plug >> node`` clones the attribute; ``plug >> 'name'`` clones it on
    the same node and ``plug >> 'node.name'`` onto that node; ``plug >>
    container_node`` publishes."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.src = Node.create("transform", name="src")
        Node.create("transform", name="other_transform")
        self.src << Float("w", min=0, max=1) << 0.25

    def test_plug_to_a_node_clones_the_attribute(self):
        before = set(cmds.ls())
        clone  = self.src.w >> Node("other_transform")
        self.assertIsInstance(clone, Plug)
        self.assertEqual(str(clone), "other_transform.w")
        self.assertEqual(cmds.attributeQuery("w", node="other_transform", attributeType=True), "double")
        self.assertIsNone(cmds.listConnections("other_transform.w"))
        self.assertIsNone(cmds.listConnections("src.w"))
        self.assertEqual(set(cmds.ls()), before)

    def test_plug_to_a_name_clones_on_the_same_node(self):
        before = set(cmds.ls())
        clone  = self.src.w >> "w2"
        self.assertIsInstance(clone, Plug)
        self.assertEqual(str(clone), "src.w2")
        self.assertEqual(cmds.attributeQuery("w2", node="src", attributeType=True), "double")
        self.assertAlmostEqual(cmds.getAttr("src.w2"), 0.25)
        self.assertIsNone(cmds.listConnections("src.w2"))
        self.assertEqual(set(cmds.ls()), before)

    def test_plug_to_a_node_dot_name_clones_onto_that_node(self):
        before = set(cmds.ls())
        clone  = self.src.w >> "other_transform.w3"
        self.assertIsInstance(clone, Plug)
        self.assertEqual(str(clone), "other_transform.w3")
        self.assertEqual(cmds.attributeQuery("w3", node="other_transform", attributeType=True), "double")
        self.assertAlmostEqual(cmds.getAttr("other_transform.w3"), 0.25)
        self.assertIsNone(cmds.listConnections("other_transform.w3"))
        self.assertEqual(set(cmds.ls()), before)

    def test_plug_to_a_container_node_publishes(self):
        with container("outer") as ctn:
            node = Node.create("transform", name="cube1")
            node << Float("blend", dv=0.5)
            published = node.blend >> ctn
        self.assertIsInstance(published, Plug)
        self.assertEqual(str(published), "cube1.blend")
        self.assertEqual(cmds.container("outer", query=True, publishName=True), ["blend"])
        self.assertEqual(cmds.container("outer", query=True, bindAttr=True), ["cube1.blend", "blend"])

    def test_plug_to_the_scope_passes_through_in_a_flattened_scope(self):
        with container("outer"):
            with container("inner"):
                node = Node.create("transform", name="cube2")
                node << Float("blend")
                result = node.blend >> container
        self.assertEqual(str(result), "inner_cube2.blend")
        self.assertIsNone(cmds.container("outer", query=True, publishName=True))
