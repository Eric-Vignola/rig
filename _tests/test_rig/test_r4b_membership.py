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
from rig.nodetypes import DisplayLayer
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
        self.assertEqual(_types._MEMBERSHIP, (_MemberSpec, DisplayLayer))
        cube = _cube("cube")
        cube << Float("w")
        for value in (Tag("cap"), Tag(), -Tag("cap"), self.L, DisplayLayer(), -self.L, Blinn("red")):
            with self.subTest(repr(value)):
                self.assertTrue(_types._is_membership(value))
        for value in (cube, cube.w, Float("x"), None, "L", 1, [self.L]):
            with self.subTest(repr(value)):
                self.assertFalse(_types._is_membership(value))
        self.assertIs(Tag("cap")._member().__class__, Tag)
        self.assertEqual(self.L._member()._name, "L")
