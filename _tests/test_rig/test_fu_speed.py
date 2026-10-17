"""Follow-up of the features run (the user's decision 3 of 2026-10-16): the review fix's
element hold made every compound child access cost a walk.

``DGNode.find_attr`` called ``_hold_if_element`` for every plug it built, and for a child
plug (``tx``, ``ry`` ...) that walked the plug up with ``_top_path`` (a list of steps, its
attribute MObjects) only to find it below no element: about 1.5-2 us per access,
micro_plug_get_and_str +15 % over v2.0.0a5. Now:

* ``find_attr`` reads "an element or below one" from the walk it already made for the
  unresolved multi child (``_multi_child_walk``), and calls ``_hold_if_element`` for such a
  plug only: a child below no element costs no second walk.
* ``_hold_if_element`` answers a child below no element through its parents' ``isElement``
  / ``isChild`` alone (no path built); ``_top_path`` answers a top plug (an array root)
  without its walk.
* ``_plug_lock_refusal`` reads the owner's fn set for the node lock (no new
  MFnDependencyNode per lock).

The answers are the same: an exhaustive check over every attribute, element and child of a
few node types compares the new walks with ``_top_path``. The "no walk" tests failed at
750d051 (negative control); the exactness and lock tests are guards.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node, Plug
from rig._internal.plug import InjectionError
from rig.nodetypes import _base, dg_node
from rig.nodetypes._base import Attribute
from rig._tests._base import MayaTestCase


def _reference_element(mplug):
    """The review fix's answer (9d34992): `_top_path`'s walk, element first."""
    cur, element = mplug, False
    while True:
        if cur.isElement:
            cur, element = cur.array(), True
        elif cur.isChild:
            cur = cur.parent()
        else:
            break
    return element


def _plugs():
    """Every attribute of a few node types, the first two elements of each array and their
    children, and the children's elements (nested multis): MPlugs."""
    cmds.createNode("multiplyDivide", name="md")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.polyCube(name="cube", constructionHistory=False)
    cmds.select(clear=True)
    cmds.joint(name="j1")
    cmds.select(clear=True)
    cmds.polyCube(name="skinned", constructionHistory=False)
    cmds.skinCluster("j1", "skinned", name="skin")
    cmds.polyCube(name="tgt", constructionHistory=False)
    cmds.blendShape("tgt", "skinned", name="bls")
    cmds.createNode("transform", name="xf")
    cmds.addAttr("xf", longName="cmp", attributeType="compound", numberOfChildren=2, multi=True)
    cmds.addAttr("xf", longName="cx", attributeType="double", parent="cmp")
    cmds.addAttr("xf", longName="inner", attributeType="double", multi=True, parent="cmp")
    found = []
    for name in ("md", "pma", "cubeShape", "skin", "bls", "xf", "j1"):
        fn = OpenMaya.MFnDependencyNode(Node(name).get_mobject())
        for i in range(fn.attributeCount()):
            attribute = fn.attribute(i)
            if not OpenMaya.MFnAttribute(attribute).parent.isNull():
                continue  # a child: reached from its parent's plug below
            stack = [fn.findPlug(attribute, False)]
            while stack:
                plug = stack.pop()
                found.append(plug)
                if plug.isArray:
                    stack.extend(plug.elementByLogicalIndex(k) for k in (0, 3))
                elif plug.isCompound:
                    stack.extend(plug.child(k) for k in range(plug.numChildren()))
    return found


class TestTheSameAnswers(MayaTestCase):
    """guards: the new walks answer as `_top_path` does, for every plug."""

    TEST_START_NEW_SCENE = True

    def test_every_plug(self):
        plugs = _plugs()
        self.assertGreater(len(plugs), 1500)
        kinds = {"top": 0, "child": 0, "element": 0, "below an element": 0}
        for plug in plugs:
            want = _reference_element(plug)
            (_top, steps), element = _base._top_path(plug)
            self.assertIs(element, want, plug.name())
            if not want:
                self.assertEqual(steps, (), plug.name())
            if plug.isElement or plug.isChild:
                self.assertEqual(dg_node._multi_child_walk(plug), (False, want), plug.name())
                kinds["element" if plug.isElement else "below an element" if want else "child"] += 1
            else:
                kinds["top"] += 1
            attr = _base._new_attr(Attribute, plug, OpenMaya.MObjectHandle(plug.node()))
            with mock.patch.object(_base, "_hold_path", wraps=_base._hold_path) as hold:
                _base._hold_if_element(attr)
            self.assertEqual(hold.call_count, int(want), plug.name())
        self.assertTrue(all(kinds.values()), kinds)

    def test_an_unresolved_child(self):
        # a bare child of a multi: weightList[-1].weights, refused as before
        cmds.select(clear=True)
        cmds.joint(name="j1")
        cmds.polyCube(name="geo", constructionHistory=False)
        cmds.skinCluster("j1", "geo", name="skin")
        plug = OpenMaya.MFnDependencyNode(Node("skin").get_mobject()).findPlug("weights", False)
        self.assertEqual(dg_node._multi_child_walk(plug), (True, True))
        self.assertIs(dg_node._is_unresolved_multi_child(plug), True)
        with self.assertRaisesRegex(AttributeError, "a child of an unindexed multi"):
            Node("skin").find_attr("weights")

    def test_found_elements_are_still_held(self):
        cmds.createNode("plusMinusAverage", name="pma")
        for name in ("input3D[0]", "input3D[1].input3Dx", "input1D[2]"):
            with self.subTest(name):
                attr = Node("pma").find_attr(name)
                self.assertIsNotNone(_base._held_path(attr), name)
        for name in ("output1D", "output3Dx", "input3D"):
            with self.subTest(name):
                self.assertIsNone(_base._held_path(Node("pma").find_attr(name)), name)


class TestNoWalkForAChild(MayaTestCase):
    """A compound child below no element is found without a second walk."""

    TEST_START_NEW_SCENE = True

    def test_find_attr_of_a_child(self):
        cmds.createNode("transform", name="a")
        # find_attr calls its module's binding of `_top_path` (dg_node imports it)
        with mock.patch.object(dg_node, "_top_path", wraps=_base._top_path) as top, \
                mock.patch.object(dg_node, "_hold_if_element", wraps=dg_node._hold_if_element) as hold:
            for name in ("translateX", "ry", "sz", "rotatePivotX"):
                str(getattr(Node("a"), name))
                Node("a").find_attr(name)
        top.assert_not_called()
        hold.assert_not_called()

    def test_hold_if_element_of_a_child(self):
        cmds.createNode("transform", name="a")
        with mock.patch.object(_base, "_top_path", wraps=_base._top_path) as top:
            for name in ("a.tx", "a.rotateY"):
                attr = Attribute(name)
                plug = Plug(name)
                self.assertIsNone(_base._held_path(attr))
                self.assertIsNone(_base._held_path(plug))
        top.assert_not_called()

    def test_an_element_is_walked_once_for_its_path(self):
        cmds.createNode("plusMinusAverage", name="pma")
        with mock.patch.object(dg_node, "_top_path", wraps=_base._top_path) as top:
            attr = Node("pma").find_attr("input3D[1].input3Dx")
        self.assertEqual(top.call_count, 1)
        top_attr, steps = _base._held_path(attr)
        self.assertEqual([is_element for is_element, _key in steps], [True, False])


class TestTheLockRefusal(MayaTestCase):
    """guards: the node lock is read from the owner's fn set, or the plug's node's."""

    TEST_START_NEW_SCENE = True

    def test_a_locked_node_refuses_with_and_without_an_owner(self):
        cmds.createNode("transform", name="a")
        cmds.lockNode("a", lock=True)
        for label, attr in (("owned", Node("a").tx), ("no owner", Attribute("a.ty"))):
            with self.subTest(label):
                self.assertIs(attr.__dict__.get("_node") is not None, label == "owned")
                with self.assertRaisesRegex(InjectionError, "is on the locked node 'a'"):
                    attr.lock()
                self.assertFalse(cmds.getAttr(str(attr), lock=True))
        cmds.lockNode("a", lock=False)
        Node("a").tx.lock()
        Attribute("a.ty").lock()
        self.assertEqual([cmds.getAttr("a.tx", lock=True), cmds.getAttr("a.ty", lock=True)], [True, True])

    def test_a_shape_and_its_transform(self):
        # the shape locked, its transform not: each plug reads its own node's lock
        cmds.polyCube(name="cube", constructionHistory=False)
        cmds.lockNode("cubeShape", lock=True)
        for plug in (Node("cubeShape").castsShadows, Attribute("cubeShape.castsShadows")):
            with self.assertRaisesRegex(InjectionError, "is on the locked node 'cubeShape'"):
                plug.lock()
        Node("cube").tx.lock()
        self.assertTrue(cmds.getAttr("cube.tx", lock=True))
        cmds.lockNode("cubeShape", lock=False)
