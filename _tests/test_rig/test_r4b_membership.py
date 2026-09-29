"""Round 4b, step NC6: membership on nodes I (the machinery, layers, Tag).

* ``TestLayerNode``: ``Layer is DisplayLayer``; a layer node on the right of
  ``<<`` assigns (a ``List`` in one ``editDisplayLayerMembers``), ``-layer``
  removes, ``Layer()`` purges and enumerates, ``in`` answers yes or no,
  ``Layer.of`` lists the layer; a missing layer is the strict reference's
  NodeNotFoundError before any write; defaultLayer reads as no layer; the
  node's ``delete`` / ``rename`` / ``clear`` refuse defaultLayer and a
  referenced layer; each ``<<`` is one undo step; a held layer across a
  delete, an undo, a rename and a new scene; namespaces and scopes.
* ``TestTokens``: the reprs of the kind and removal tokens, the double
  negatives, ``~``, ``Cls(None)`` refused, the tokens touch no layer command,
  the one membership slot (a bound tuple).
* ``TestIn``: ``in`` / ``not in`` with all-members semantics for layers and
  tags (node on the left, components, lists, a plug standing for its node, a
  missing tag False); the tokens are refused.
* ``TestPlugLeftRefused``: an attribute plug on the left of a membership
  ``<<`` / ``>>`` / ``of`` is refused before any write, for every kind (a
  string plug is never written; a mixed ``List`` is refused before any element
  is written or cloned); component plugs stay members; the deformer
  expression sugar runs first; ``plug >> node`` still clones for other nodes.
* ``TestTagRules``: ``Tag(None)`` refused; the user decision of 2026-09-28
  (a tag is untyped for queries: components of another kind are not in it,
  ``>>`` answers an empty id array, removing them is a no-op; adding them to a
  non-empty tag is refused, naming why); queries follow what Maya reads of a
  tag whose stored contents mix kinds.

Every refusal asserts a zero ``cmds.ls()`` delta.
"""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds

import rig
import rig._internal.types as _types
from rig import (
    AmbiguousNodeError,
    container,
    Layer,
    List,
    Node,
    NodeNotFoundError,
    Tag,
)
from rig.nodetypes import DisplayLayer, ShadingEngine
from rig.shade import Blinn, Material
from rig.spec import Float, String
from rig._internal.members import _MemberSpec
from rig._tests._base import MayaTestCase


PLUG = "is a plug; membership takes the node"


def _scene():
    return set(cmds.ls())


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _layer(node):
    """The layer a node's own ``drawOverride`` reads, or None."""
    layers = cmds.listConnections(
        f"{node}.drawOverride", source=True, destination=False, type="displayLayer"
    )
    return layers[0] if layers else None


def _members(layer):
    return sorted(
        cmds.editDisplayLayerMembers(str(layer), query=True, fullNames=True, noRecurse=True) or []
    )


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def assertRefused(self, error, pattern, call):
        """``call()`` raises ``error`` matching ``pattern`` and writes nothing."""
        before = _scene()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(_scene(), before)


# --------------------------------------------------------------------- #
#  Layers are the display layer nodes
# --------------------------------------------------------------------- #


class TestLayerNode(_Case):
    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.other = _cube("other")
        self.L     = Layer.define("L")

    def test_layer_is_displaylayer(self):
        self.assertIs(rig.Layer, DisplayLayer)
        self.assertIs(rig.membership.Layer, DisplayLayer)
        self.assertIsInstance(self.L, DisplayLayer)
        self.assertEqual(Layer("L"), self.L)
        self.assertEqual(Layer.create(name="L2").name, "L2")
        self.assertEqual(Layer.define("L3", displayType=2).displayType >> None, 2)

    def test_assign_remove_purge_and_the_answers(self):
        before = _scene()
        self.assertIs(self.cube << self.L, self.cube)
        self.assertEqual(_scene(), before)
        self.assertEqual(_layer("|cube"), "L")
        self.assertTrue(self.cube in self.L)
        self.assertEqual(self.cube >> Layer(), self.L)
        self.assertIsInstance(self.cube >> Layer(), DisplayLayer)
        self.assertEqual(Layer.of(self.cube), [self.L])
        self.assertEqual(DisplayLayer.of(self.cube), [self.L])
        self.assertIsNone(self.other >> Layer())
        self.assertEqual(Layer.of(self.other), [])
        self.assertIs(self.cube << -self.L, self.cube)
        self.assertIsNone(_layer("|cube"))
        self.cube << self.L
        self.assertIs(self.cube << Layer(), self.cube)
        self.assertIsNone(_layer("|cube"))
        self.assertEqual(_scene(), before)

    def test_a_list_is_one_edit_call(self):
        lhs = List([self.cube, self.other])
        with mock.patch.object(cmds, "editDisplayLayerMembers", wraps=cmds.editDisplayLayerMembers) as edit:
            self.assertIs(lhs << self.L, lhs)
        writes = [c for c in edit.call_args_list if not c.kwargs.get("query")]
        self.assertEqual(len(writes), 1)
        self.assertEqual(_members("L"), ["|cube", "|other"])
        self.assertTrue(lhs in self.L)
        with mock.patch.object(cmds, "editDisplayLayerMembers", wraps=cmds.editDisplayLayerMembers) as edit:
            lhs << -self.L
        self.assertEqual(len([c for c in edit.call_args_list if not c.kwargs.get("query")]), 1)
        self.assertEqual(_members("L"), [])

    def test_a_missing_layer_is_the_references_error(self):
        for label, call in (
            ("Layer('ghost')",          lambda: Layer("ghost")),
            ("cube << Layer('ghost')",  lambda: self.cube << Layer("ghost")),
            ("cube << -Layer('ghost')", lambda: self.cube << -Layer("ghost")),
            ("cube in Layer('ghost')",  lambda: self.cube in Layer("ghost")),
        ):
            with self.subTest(label):
                self.assertRefused(NodeNotFoundError, r"^no displayLayer named 'ghost'", call)
        self.assertFalse(Layer.exists("ghost"))
        self.assertFalse(cmds.objExists("ghost"))

    def test_default_layer_reads_as_no_layer(self):
        default = Layer("defaultLayer")
        self.assertTrue(self.cube in default)
        self.cube << self.L
        self.assertFalse(self.cube in default)
        self.assertIs(self.cube << default, self.cube)
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(self.cube >> Layer())
        self.assertRefused(TypeError, "contradictory", lambda: self.cube << -default)

    def test_node_guards(self):
        default = Layer("defaultLayer")
        for verb, call in (
            ("deleted", default.delete),
            ("renamed", lambda: default.rename("z")),
            ("cleared", default.clear),
        ):
            with self.subTest(verb):
                self.assertRefused(TypeError, rf"^'defaultLayer' cannot be {verb}: it is the layer of no layer$", call)
        self.other << Layer.define("taken")
        for error, pattern, call in (
            (ValueError, r"^'taken' already exists$", lambda: self.L.rename("taken")),
            (ValueError, r"^'cube' already exists$", lambda: self.L.rename("cube")),
            (ValueError, "not a display layer name Maya keeps", lambda: self.L.rename("1bad")),
            (ValueError, "not a display layer name Maya keeps", lambda: self.L.rename("a|b")),
            (TypeError, "non-empty str", lambda: self.L.rename("")),
            (TypeError, "non-empty str", lambda: self.L.rename(None)),
        ):
            with self.subTest(pattern):
                self.assertRefused(error, pattern, call)
        self.assertIsNone(self.L.rename("L"))
        self.cube << self.L
        self.L.rename("M")
        self.assertEqual(str(self.L), "M")
        self.assertTrue(self.cube in self.L)

    def test_a_referenced_layer_refuses_the_network_verbs(self):
        folder = tempfile.mkdtemp(prefix="rig_layer_")
        try:
            path = os.path.join(folder, "layer.ma").replace("\\", "/")
            cmds.file(new=True, force=True)
            cube = cmds.polyCube(name="rcube", ch=False)[0]
            cmds.createDisplayLayer(cube, name="rl", noRecurse=True)
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            layer = Layer("ref:rl")
            self.assertTrue(Node("ref:rcube") in layer)
            for verb, call in (
                ("deleted", layer.delete),
                ("renamed", lambda: layer.rename("mine")),
                ("cleared", layer.clear),
            ):
                with self.subTest(verb):
                    self.assertRefused(RuntimeError, rf"^'ref:rl' is referenced and cannot be {verb}", call)
            self.assertTrue(Node("ref:rcube") in layer)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_one_undo_per_lshift(self):
        cmds.undoInfo(state=True, infinity=True)
        for step, check in (
            (lambda: self.cube << self.L, lambda: self.assertEqual(_layer("|cube"), "L")),
            (lambda: List([self.cube, self.other]) << self.L, lambda: self.assertEqual(_members("L"), ["|cube", "|other"])),
        ):
            before = (_layer("|cube"), _layer("|other"))
            step()
            check()
            cmds.undo()
            self.assertEqual((_layer("|cube"), _layer("|other")), before)
        List([self.cube, self.other]) << self.L
        for step in (
            lambda: self.cube << -self.L,
            lambda: self.cube << Layer(),
            lambda: self.L.clear(),
        ):
            step()
            self.assertIsNone(_layer("|cube"))
            cmds.undo()
            self.assertEqual(_layer("|cube"), "L")
        self.L.delete()
        self.assertFalse(cmds.objExists("L"))
        cmds.undo()
        self.assertEqual(_members("L"), ["|cube", "|other"])
        self.L.rename("M")
        cmds.undo()
        self.assertEqual(str(self.L), "L")

    def test_a_held_layer_across_delete_undo_rename_and_a_new_scene(self):
        cmds.undoInfo(state=True, infinity=True)
        held = self.L
        cmds.rename("L", "renamed")
        self.assertEqual(repr(-held), '-DisplayLayer("renamed")')
        self.cube << held
        self.assertEqual(_layer("|cube"), "renamed")
        cmds.delete("renamed")
        for label, call in (
            ("cube << held", lambda: self.cube << held),
            ("cube << -held", lambda: self.cube << -held),
            ("cube in held", lambda: self.cube in held),
        ):
            with self.subTest(label):
                self.assertRefused(RuntimeError, "already deleted", call)
        cmds.undo()
        self.assertTrue(self.cube in held)
        self.cube << -held
        self.assertIsNone(_layer("|cube"))
        # a layer made under the old name is another node
        cmds.delete("renamed")
        fresh = Layer.define("renamed")
        self.assertRefused(RuntimeError, "already deleted", lambda: self.cube << held)
        self.cube << fresh
        self.assertEqual(_layer("|cube"), "renamed")
        cmds.file(new=True, force=True)
        cube = _cube("cube")
        self.assertRefused(RuntimeError, "already deleted", lambda: cube << fresh)
        self.assertRefused(NodeNotFoundError, "no displayLayer named 'renamed'", lambda: Layer("renamed"))

    def test_namespaces(self):
        cmds.namespace(add="char")
        cmds.createDisplayLayer(name="char:L", empty=True)
        cmds.namespace(set=":char")
        try:
            for relative in (False, True):
                cmds.namespace(relativeNames=relative)
                with self.subTest(relative_names=relative):
                    self.assertRefused(AmbiguousNodeError, "spell the namespace", lambda: self.cube << Layer("L"))
                    self.cube << Layer(":char:L")
                    self.assertTrue(self.cube in Layer(":char:L"))
                    self.assertFalse(self.cube in Layer(":L"))
                    self.assertEqual(self.cube >> Layer(), Layer(":char:L"))
                    self.cube << Layer()
                    self.assertIsNone(self.cube >> Layer())
        finally:
            cmds.namespace(relativeNames=False)
            cmds.namespace(set=":")

    def test_scopes_never_enrol_a_layer(self):
        with container("real") as real:
            self.cube << self.L
            inner = Layer.define("inner")
            self.other << inner
        self.assertIsNone(cmds.container(query=True, findContainer=["L"]))
        self.assertIsNone(cmds.container(query=True, findContainer=["inner"]))
        self.assertEqual(_layer("|cube"), "L")
        self.assertEqual(_layer("|other"), "inner")
        self.assertTrue(cmds.objExists(str(real)))


# --------------------------------------------------------------------- #
#  The kind and removal tokens
# --------------------------------------------------------------------- #


class TestTokens(_Case):
    def setUp(self):
        super().setUp()
        self.L = DisplayLayer.define("L")

    def test_reprs(self):
        self.assertEqual(repr(DisplayLayer()), "DisplayLayer()")
        self.assertEqual(repr(DisplayLayer()), "DisplayLayer()")
        self.assertEqual(repr(-self.L), '-DisplayLayer("L")')
        self.assertEqual(repr(Tag()), "Tag()")
        self.assertEqual(str(Tag()), "Tag()")
        self.assertEqual(repr(-Tag("cap")), "-Tag('cap')")
        self.assertTrue(DisplayLayer().purges)
        self.assertFalse((-self.L).purges)
        self.assertTrue((-self.L).removes)

    def test_double_negatives_and_invert(self):
        for pattern, call in (
            (r'^--DisplayLayer\("L"\): a removal cannot be negated again$', lambda: -(-self.L)),
            (r"^-DisplayLayer\(\) is a double negative", lambda: -DisplayLayer()),
            (r'^~DisplayLayer\("L"\) is unassigned', lambda: ~self.L),
            (r"^~DisplayLayer\(\) is unassigned", lambda: ~DisplayLayer()),
            (r"^-Tag\(\) is a double negative", lambda: -Tag()),
            (r"^--Tag\('cap'\)", lambda: -(-Tag("cap"))),
            (r"^~Tag\(\.\.\.\) is unassigned", lambda: ~Tag()),
        ):
            with self.subTest(pattern):
                self.assertRefused(TypeError, pattern, call)

    def test_none_is_refused(self):
        self.assertRefused(
            TypeError,
            r"^None is not a layer name; Layer\(\) is defaultLayer \(it removes from every layer\)$",
            lambda: DisplayLayer(None),
        )

    def test_tokens_touch_no_layer_command(self):
        refused = AssertionError("a token called a layer command")
        before  = _scene()
        with mock.patch.object(cmds, "createDisplayLayer", side_effect=refused), \
             mock.patch.object(cmds, "editDisplayLayerMembers", side_effect=refused), \
             mock.patch.object(cmds, "listConnections", side_effect=refused), \
             mock.patch.object(cmds, "componentTag", side_effect=refused):
            tokens = (DisplayLayer(), -self.L, Tag(), -Tag("cap"), self.L._member())
        self.assertEqual(_scene(), before)
        self.assertEqual(len(tokens), 5)

    def test_tokens_are_no_layers(self):
        for token in (DisplayLayer(), -self.L):
            for name in ("delete", "rename", "clear", "visibility", "is_default"):
                with self.subTest(token=repr(token), name=name):
                    with self.assertRaisesRegex(AttributeError, "not a layer"):
                        getattr(token, name)

    def test_the_membership_slot_is_one_bound_tuple(self):
        # re-pinned (round 4b NC7): the shader classes and ShadingEngine joined
        # the tuple; a material is a node (Blinn("red") refers to one that exists)
        self.assertEqual(_types._MEMBERSHIP, (_MemberSpec, DisplayLayer, Material, ShadingEngine))
        cube = _cube("cube")
        cube << Float("w")
        red = Blinn.create(name="red")
        for value in (
            Tag("cap"), Tag(), -Tag("cap"), self.L, DisplayLayer(), -self.L, red, -red, Blinn(),
            Material(), red.engine, -red.engine,
        ):
            with self.subTest(repr(value)):
                self.assertTrue(_types._is_membership(value))
        for value in (cube, cube.w, Float("x"), None, "L", 1, [self.L]):
            with self.subTest(repr(value)):
                self.assertFalse(_types._is_membership(value))
        self.assertIs(Tag("cap")._member().__class__, Tag)
        self.assertEqual(self.L._member()._name, "L")


# --------------------------------------------------------------------- #
#  'in' / 'not in': yes or no, all members
# --------------------------------------------------------------------- #


class TestIn(_Case):
    def setUp(self):
        super().setUp()
        self.sph   = Node(cmds.polySphere(name="sph", ch=False)[0])
        self.cube  = _cube("cube")
        self.L     = DisplayLayer.define("L")
        self.sph.vtx[:8] << Tag("cap")
        self.sph.f[:3]   << Tag("lid")
        self.sph         << Tag("empty")

    def test_layers(self):
        before = _scene()
        self.sph << self.L
        self.assertTrue(self.sph in self.L)
        self.assertFalse(self.sph not in self.L)
        self.assertTrue(self.sph.tx in self.L)
        self.assertTrue(List([self.sph, self.sph.ry]) in self.L)
        self.assertFalse(self.cube in self.L)
        self.assertTrue(self.cube not in self.L)
        self.assertFalse(List([self.sph, self.cube]) in self.L)
        self.assertFalse([self.sph, self.cube.tx] in self.L)
        self.assertEqual(_scene(), before)
        self.assertRefused(TypeError, "layers hold objects", lambda: self.sph.f[0] in self.L)
        self.assertRefused(TypeError, "layers hold DAG objects", lambda: Node("lambert1") in self.L)

    def test_tag_node_on_the_left(self):
        self.assertTrue(self.sph in Tag("cap"))
        self.assertTrue(self.sph in Tag("empty"))
        self.assertTrue(self.sph.tx in Tag("lid"))
        self.assertFalse(self.cube in Tag("cap"))
        self.assertFalse(List([self.sph, self.cube]) in Tag("cap"))
        joint = Node.create("joint", name="j")
        self.assertRefused(TypeError, "not geometry", lambda: joint in Tag("cap"))
        self.assertRefused(TypeError, "not geometry", lambda: joint.tx in Tag("cap"))

    def test_tag_components_are_all_members(self):
        self.assertTrue(self.sph.vtx[:8] in Tag("cap"))
        self.assertTrue(self.sph.vtx[3] in Tag("cap"))
        self.assertFalse(self.sph.vtx[:9] in Tag("cap"))
        self.assertTrue(self.sph.vtx[:9] not in Tag("cap"))
        self.assertTrue(List([self.sph.vtx[0], self.sph.vtx[7]]) in Tag("cap"))
        self.assertTrue(self.sph.f[:3] in Tag("lid"))
        self.assertFalse(self.sph.vtx[0] in Tag("empty"))
        self.sph.vtx << Tag("all")
        self.assertTrue(self.sph.vtx in Tag("all"))
        self.assertFalse(self.sph.vtx in Tag("cap"))
        # a node and its components: every one holds
        self.assertTrue(List([self.sph, self.sph.vtx[1]]) in Tag("cap"))

    def test_a_missing_tag_is_false(self):
        shape  = Node(cmds.listRelatives("sph", shapes=True, fullPath=True)[0])
        tags   = shape.component_tags
        before = _scene()
        self.assertFalse(self.sph in Tag("ghost"))
        self.assertFalse(self.sph.vtx[0] in Tag("ghost"))
        self.assertTrue(self.sph.vtx[0] not in Tag("ghost"))
        self.assertFalse(self.sph.tx in Tag("ghost"))
        self.assertFalse(List([self.sph.vtx[0], self.sph.f[0]]) in Tag("ghost"))
        self.assertEqual(_scene(), before)
        self.assertEqual(shape.component_tags, tags)

    def test_other_kinds_are_false(self):
        before = _scene()
        self.assertFalse(self.sph.f[:2] in Tag("cap"))
        self.assertTrue(self.sph.f[:2] not in Tag("cap"))
        self.assertFalse(self.sph.e[:2] in Tag("cap"))
        self.assertFalse(self.sph.vtx[0] in Tag("lid"))
        # a list mixing kinds: all members, so False while one kind is not in
        self.assertFalse(List([self.sph.vtx[0], self.sph.f[0]]) in Tag("cap"))
        self.assertFalse([self.sph.vtx[0], self.sph.f[0]] in Tag("lid"))
        self.assertEqual(_scene(), before)

    def test_tokens_are_refused(self):
        for pattern, call in (
            (r"^'in' asks about one collection; Tag\(\) names every one of its kind: x >> Tag\(\) enumerates",
             lambda: self.sph in Tag()),
            (r"^'in' asks about one collection; -Tag\('cap'\) is a removal: ask with x in Tag\('cap'\)$",
             lambda: self.sph.vtx[0] in -Tag("cap")),
            (r"^'in' asks about one collection; DisplayLayer\(\) names every one of its kind: x >> DisplayLayer\(\)",
             lambda: self.sph in DisplayLayer()),
            (r'^\'in\' asks about one collection; -DisplayLayer\("L"\) is a removal: ask with x in DisplayLayer\("L"\)$',
             lambda: self.sph in -self.L),
            ("at= places a tag", lambda: self.sph.vtx[0] in Tag("cap", at=self.sph)),
        ):
            with self.subTest(pattern):
                self.assertRefused(TypeError, pattern, call)


# --------------------------------------------------------------------- #
#  A plug on the left of '<<' / '>>' / of is refused
# --------------------------------------------------------------------- #


class TestPlugLeftRefused(_Case):
    def setUp(self):
        super().setUp()
        self.cube = _cube("cube")
        self.L    = DisplayLayer.define("L")
        self.cube.vtx[:4] << Tag("cap")
        self.cube << Float("w") << String("note")

    def _state(self):
        return (_scene(), _layer("|cube"), self.cube >> Tag("cap"), cmds.getAttr("cube.note"))

    def assertUnchanged(self, before):
        after = self._state()
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1], before[1])
        np.testing.assert_array_equal(after[2], before[2])
        self.assertEqual(after[3], before[3])

    def test_every_kind_and_verb(self):
        # re-pinned (round 4b NC7): a material is a node, made before the
        # refusals; the refusals leave cube where it was (initialShadingGroup)
        red = Blinn.create(name="red")
        for spec in (self.L, -self.L, DisplayLayer(), Tag("cap"), -Tag("cap"), Tag(), red, -red, Material()):
            for plug in (self.cube.tx, self.cube.t, self.cube.w, self.cube.note, self.cube.visibility):
                before = self._state()
                with self.subTest(spec=repr(spec), plug=str(plug), verb="<<"):
                    with self.assertRaisesRegex(TypeError, f"^'{plug}' {PLUG}"):
                        plug << spec
                if not getattr(spec, "removes", False):
                    with self.subTest(spec=repr(spec), plug=str(plug), verb=">>"):
                        with self.assertRaisesRegex(TypeError, f"^'{plug}' {PLUG}"):
                            plug >> spec
                self.assertUnchanged(before)
        for kind in (DisplayLayer, Tag, Material, Blinn):
            with self.subTest(of=kind.__name__):
                self.assertRefused(TypeError, f"^'cube.translateX' {PLUG}", lambda: kind.of(self.cube.tx))
        self.assertEqual(cmds.listConnections("cubeShape", type="shadingEngine"), ["initialShadingGroup"])
        self.assertIsNone(cmds.sets("redSG", query=True))

    def test_the_messages(self):
        for pattern, call in (
            (r'^\'cube.translateX\' is a plug; membership takes the node: cube << DisplayLayer\("L"\) '
             r"\(to connect, name a plug: other.attr\)$", lambda: self.cube.tx << self.L),
            (r'^\'cube.translate\' is a plug; membership takes the node: ask with cube.translate in '
             r'DisplayLayer\("L"\) \(a plug stands for its node there\)$', lambda: self.cube.t >> self.L),
            (r"^'cube.translate' is a plug; membership takes the node: ask with cube.translate in "
             r"Tag\('cap'\) \(a plug stands for its node there\), or cube >> Tag\('cap'\) for ids$",
             lambda: self.cube.t >> Tag("cap")),
            (r"^'cube.translate' is a plug; membership takes the node: cube >> DisplayLayer\(\) enumerates$",
             lambda: self.cube.t >> DisplayLayer()),
            (r"^'cube.translateX' is a plug; membership takes the node: Layer.of\(cube\)$",
             lambda: DisplayLayer.of(self.cube.tx)),
            (r"^'cube.translateX' is a plug; membership takes the node: Tag.of\(cube\)$",
             lambda: Tag.of(self.cube.tx)),
        ):
            with self.subTest(pattern):
                self.assertRefused(TypeError, pattern, call)

    def test_a_string_plug_is_refused_not_written(self):
        before = self._state()
        with self.assertRaisesRegex(TypeError, f"^'cube.note' {PLUG}: cube << DisplayLayer"):
            self.cube.note << self.L
        self.assertUnchanged(before)
        self.assertIsNone(cmds.getAttr("cube.note"))
        self.assertFalse(self.cube in self.L)

    def test_a_mixed_list_is_refused_before_any_write_or_clone(self):
        before = self._state()
        with self.assertRaisesRegex(TypeError, f"^'cube.translateX' {PLUG}"):
            List([self.cube.tx, self.cube]) << self.L
        with self.assertRaisesRegex(TypeError, f"^'cube.translateX' {PLUG}"):
            List([self.cube, self.cube.tx]) << self.L
        with self.assertRaisesRegex(TypeError, f"^'cube.w' {PLUG}"):
            List([self.cube.w, self.cube]) >> self.L
        with self.assertRaisesRegex(TypeError, f"^'cube.w' {PLUG}"):
            List([self.cube.w, self.cube]) >> Tag("cap")
        with self.assertRaisesRegex(TypeError, f"^'cube.w' {PLUG}"):
            List([self.cube.w, self.cube.vtx[0]]) << -Tag("cap")
        self.assertUnchanged(before)
        self.assertFalse(cmds.attributeQuery("w", node="L", exists=True))

    def test_component_plugs_are_members(self):
        lhs = self.cube.vtx[4:6]
        self.assertIs(lhs << Tag("cap"), lhs)
        np.testing.assert_array_equal(self.cube >> Tag("cap"), np.arange(6))
        np.testing.assert_array_equal(self.cube.vtx[5] >> Tag("cap"), [5])
        self.assertTrue(self.cube.vtx[5] in Tag("cap"))
        self.cube.vtx << Tag("every")
        self.assertEqual(len(self.cube >> Tag("every")), 8)
        np.testing.assert_array_equal(self.cube.vtx >> Tag("cap"), np.arange(6))
        self.assertRefused(TypeError, "layers hold objects", lambda: self.cube.vtx[0] << self.L)
        srf = Node(cmds.sphere(name="srf", ch=False)[0])
        srf.cv[1, 2] << Tag("rim")
        np.testing.assert_array_equal(srf >> Tag("rim"), [[1, 2]])
        self.assertTrue(srf.cv[1, 2] in Tag("rim"))

    def test_the_expression_sugar_runs_first(self):
        cluster = Node(cmds.cluster("cube")[0])
        expr    = cluster.input[0].componentTagExpression
        with mock.patch.object(cmds, "warning") as warning:
            self.assertIs(expr << Tag("cap"), expr)
        warning.assert_not_called()
        self.assertEqual(expr >> None, "cap")
        self.assertRefused(TypeError, PLUG, lambda: expr >> Tag("cap"))
        self.assertRefused(TypeError, PLUG, lambda: cluster.envelope << Tag("cap"))
        self.assertRefused(TypeError, PLUG, lambda: expr << self.L)

    def test_plug_rshift_other_nodes_still_clones(self):
        other = Node.create("transform", name="other")
        clone = self.cube.w >> other
        self.assertEqual(str(clone), "other.w")
        self.assertRefused(TypeError, PLUG, lambda: self.cube.w >> self.L)
        self.assertFalse(cmds.attributeQuery("w", node="L", exists=True))


# --------------------------------------------------------------------- #
#  Tag: None refused; untyped for queries (user decision 2026-09-28)
# --------------------------------------------------------------------- #


class TestTagRules(_Case):
    def setUp(self):
        super().setUp()
        self.sph   = Node(cmds.polySphere(name="sph", ch=False)[0])
        self.shape = Node(cmds.listRelatives("sph", shapes=True, fullPath=True)[0])
        self.sph.vtx[:8] << Tag("vtxTag")
        self.sph.f[:3]   << Tag("faceTag")

    def test_tag_none_is_refused(self):
        for pattern, call in (
            (r"^None is not a tag name; Tag\(\) means every tag$", lambda: Tag(None)),
            (r"^None is not a tag name", lambda: Tag(None, force=True)),
            (r"^None is not a tag name", lambda: Tag(name=None)),
            (r"^None is not a tag name", lambda: self.sph.vtx[:2] << Tag(None)),
            (r"^None is not a tag name", lambda: self.sph >> Tag(None)),
        ):
            with self.subTest(pattern):
                self.assertRefused(TypeError, pattern, call)
        # the no-argument call is the kind token: purge on '<<', enumerate on '>>'
        self.assertEqual(sorted(str(t) for t in self.sph >> Tag()), ["faceTag", "vtxTag"])
        lhs = self.sph.vtx[:2]
        self.assertIs(lhs << Tag(), lhs)
        np.testing.assert_array_equal(self.sph >> Tag("vtxTag"), np.arange(2, 8))

    def test_queries_answer_by_contents(self):
        before = _scene()
        self.assertFalse(self.sph.f[:3] in Tag("vtxTag"))
        self.assertTrue(self.sph.f[:3] not in Tag("vtxTag"))
        self.assertFalse(self.sph.vtx[:2] in Tag("faceTag"))
        for lhs, tag, same_kind_empty in (
            (self.sph.f[:3], "vtxTag", self.sph.f[10:12] >> Tag("faceTag")),
            (self.sph.f,     "vtxTag", self.sph.f[10:12] >> Tag("faceTag")),
            (self.sph.e[:4], "faceTag", self.sph.e[:4] >> Tag("ghost")),
            (self.sph.vtx[:2], "faceTag", self.sph.vtx[20:22] >> Tag("vtxTag")),
        ):
            with self.subTest(lhs=repr(lhs), tag=tag):
                ids = lhs >> Tag(tag)
                self.assertIsInstance(ids, np.ndarray)
                self.assertEqual(ids.shape, (0,))
                self.assertEqual(ids.shape, same_kind_empty.shape)
        self.assertEqual((self.sph.f[:3] >> Tag("vtxTag")).dtype, (self.sph.f[10:12] >> Tag("faceTag")).dtype)
        # Tag.of already skips other kinds
        self.assertEqual([str(t) for t in Tag.of(self.sph.f[0])], ["faceTag"])
        self.assertEqual([str(t) for t in Tag.of(self.sph.vtx[0])], ["vtxTag"])
        self.assertEqual(_scene(), before)

    def test_a_missing_tag_answers_no_ids_for_components(self):
        srf     = Node(cmds.sphere(name="srf", ch=False)[0])
        cube    = cmds.polyCube(name="lc", ch=False)[0]
        lattice = Node(cmds.lattice(cube, divisions=(2, 3, 4))[1])
        before  = _scene()
        # the shape of a normal empty answer: (0,), or coordinates on a surface
        # (0, 2) and a lattice (0, 3)
        ids = self.sph.vtx[:3] >> Tag("ghost")
        self.assertIsInstance(ids, np.ndarray)
        self.assertEqual(ids.shape, (0,))
        self.assertEqual((srf.cv[1:3, 2:4] >> Tag("ghost")).shape, (0, 2))
        self.assertEqual((srf.cv >> Tag("ghost")).shape, (0, 2))
        self.assertEqual((lattice.pt[0, 0, :] >> Tag("ghost")).shape, (0, 3))
        self.assertEqual((lattice.pt >> Tag("ghost")).shape, (0, 3))
        self.assertEqual(_scene(), before)
        # the node on the left asks for the tag's contents: a missing tag is an error
        self.assertRefused(ValueError, "no component tag 'ghost' on sphShape", lambda: self.sph >> Tag("ghost"))

    def test_removing_another_kind_is_a_no_op(self):
        before = _scene()
        with mock.patch.object(cmds, "componentTag", wraps=cmds.componentTag) as tag_cmd:
            for lhs in (self.sph.f[:2], self.sph.e[:3], self.sph.f):
                with self.subTest(repr(lhs)):
                    self.assertIs(lhs << -Tag("vtxTag"), lhs)
            vtx = self.sph.vtx[:2]
            self.assertIs(vtx << -Tag("faceTag"), vtx)
        self.assertFalse([c for c in tag_cmd.call_args_list if not c.kwargs.get("queryEdit")])
        self.assertEqual(_scene(), before)
        np.testing.assert_array_equal(self.sph >> Tag("vtxTag"), np.arange(8))
        np.testing.assert_array_equal(self.sph >> Tag("faceTag"), [0, 1, 2])

    def test_adding_another_kind_to_a_non_empty_tag_is_refused(self):
        pattern = (
            r"^'vtxTag' on sphShape is a vertex tag, and Maya reads a tag as one component type, "
            r"the first one stored: it would store \|sph\|sphShape\.f\[0:1\] and never read them, "
            r"so nothing was written\. Tag\('vtxTag'\)\.set\(<faces>\) replaces its contents$"
        )
        self.assertRefused(TypeError, pattern, lambda: self.sph.f[:2] << Tag("vtxTag"))
        self.assertRefused(TypeError, "is a face tag, and Maya reads a tag as one component type",
                           lambda: self.sph.vtx[:2] << Tag("faceTag"))
        np.testing.assert_array_equal(self.sph >> Tag("vtxTag"), np.arange(8))
        # an EMPTY tag takes any kind
        self.sph << Tag("empty")
        self.sph.e[:2] << Tag("empty")
        np.testing.assert_array_equal(self.sph >> Tag("empty"), [0, 1])
        self.assertTrue(self.sph.e[:2] in Tag("empty"))
        # set() replaces, and may flip the kind
        Tag("vtxTag").set(self.sph.f[:2])
        self.assertTrue(self.sph.f[:2] in Tag("vtxTag"))
        self.assertFalse(self.sph.vtx[0] in Tag("vtxTag"))

    def test_queries_follow_what_maya_reads_of_a_mixed_tag(self):
        # the componentTagContents plug stores a mixed list; Maya reads a tag
        # as the kind of its FIRST stored entry and ignores the rest
        shape = str(self.shape)
        rows  = (("faceFirst", ["f[0]", "vtx[12:13]"]), ("vtxFirst", ["vtx[12:13]", "f[0]"]))
        for i, (name, tokens) in enumerate(rows, start=10):
            cmds.setAttr(f"{shape}.componentTags[{i}].componentTagName", name, type="string")
            cmds.setAttr(f"{shape}.componentTags[{i}].componentTagContents", len(tokens), *tokens, type="componentList")
        self.assertTrue(self.sph.f[0] in Tag("faceFirst"))
        self.assertFalse(self.sph.vtx[12:14] in Tag("faceFirst"))
        np.testing.assert_array_equal(self.sph.vtx[12:14] >> Tag("faceFirst"), [])
        self.assertTrue(self.sph.vtx[12:14] in Tag("vtxFirst"))
        self.assertFalse(self.sph.f[0] in Tag("vtxFirst"))
        np.testing.assert_array_equal(self.sph.f[:2] >> Tag("vtxFirst"), [])
        self.assertIn("faceFirst", [str(t) for t in Tag.of(self.sph.f[0])])
        self.assertNotIn("vtxFirst", [str(t) for t in Tag.of(self.sph.f[0])])
        # adding the ignored kind is refused; removing it is a no-op
        before = _scene()
        self.assertRefused(TypeError, "Maya reads a tag as one component type", lambda: self.sph.vtx[20] << Tag("faceFirst"))
        vtx = self.sph.vtx[12]
        self.assertIs(vtx << -Tag("faceFirst"), vtx)
        self.assertEqual(_scene(), before)
        self.assertEqual(
            cmds.getAttr(f"{shape}.componentTags[10].componentTagContents"), ["f[0]", "vtx[12:13]"]
        )
