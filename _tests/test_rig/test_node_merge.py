"""Tests for the merge of the DSL ``Node`` wrapper into the ``rig.nodetypes``
hierarchy. Each class names the step of the merge it belongs to:

* S0: Plug property setters, ``bool(plug)``, ``find_attr`` filters and caching,
  ``rename_attr`` with a Plug.
"""

from unittest import mock

from maya import cmds
from rig import Node, Plug
from rig.nodetypes import PyNode
from rig._tests._base import MayaTestCase


class TestPlugQuickWins(MayaTestCase):
    """S0: the Plug and ``find_attr`` fixes that do not depend on the merge."""

    TEST_START_NEW_SCENE = True

    def test_property_setters_work_on_a_plug(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="knob", at="double", keyable=True)
        plug = Node(net).knob
        plug.alias = "dial"
        self.assertEqual(Plug("net.knob").alias, "dial")
        plug.alias = None
        self.assertEqual(Plug("net.knob").alias, "knob")
        plug.is_keyable = False
        self.assertFalse(cmds.getAttr("net.knob", keyable=True))
        plug.is_channel_box = True
        self.assertTrue(cmds.getAttr("net.knob", channelBox=True))
        plug.default_value = 2.5
        self.assertEqual(cmds.addAttr("net.knob", query=True, defaultValue=True), 2.5)
        plug.is_locked = True
        self.assertTrue(cmds.getAttr("net.knob", lock=True))
        # none of them became a Python attribute of the plug
        for name in ("alias", "is_keyable", "is_channel_box", "default_value", "is_locked"):
            self.assertNotIn(name, vars(plug))

    def test_str_method_names_stay_sugar(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="center", at="double")
        cmds.addAttr(net, ln="other", at="double")
        plug = Plug("net.other")
        plug.center = 5
        self.assertEqual(cmds.getAttr("net.center"), 5.0)
        self.assertNotIn("center", vars(plug))

    def test_bool_makes_no_container_query(self):
        cmds.createNode("transform", name="a")
        plugs = (Node("a").tx, Plug("a.t"), Node("a").t[0])
        with mock.patch.object(cmds, "container", wraps=cmds.container) as probe:
            self.assertEqual([bool(plug) for plug in plugs], [True] * 3)
        self.assertEqual(probe.call_count, 0)
        # a comparison result still reports whether its operands match
        self.assertTrue(Node("a").tx == Plug("a.tx"))
        self.assertFalse(Node("a").tx == Node("a").ty)

    def test_find_attr_filters_a_cache_hit(self):
        cmds.createNode("transform", name="a")
        dg     = PyNode("a")
        cached = dg.find_attr("tx")
        self.assertIsNone(dg.find_attr("tx", data_type="string"))
        self.assertIsNone(dg.find_attr("translateX", category="noSuchCategory"))
        self.assertIs(dg.find_attr("tx", data_type="doubleLinear"), cached)
        self.assertIs(dg.find_attr("translateX"), cached)

    def test_readded_extension_attr_through_the_same_wrapper(self):
        cmds.createNode("transform", name="a")
        dg = PyNode("a")
        try:
            cmds.addExtension(nodeType="transform", longName="mergeExt", at="double")
            self.assertEqual(dg.find_attr("mergeExt").data_type, "double")
            cmds.deleteExtension(
                nodeType="transform", attribute="mergeExt", forceDelete=True
            )
            cmds.addExtension(nodeType="transform", longName="mergeExt", dataType="string")
            attr = dg.find_attr("mergeExt")
            self.assertEqual(str(attr), "a.mergeExt")
            self.assertEqual(attr.data_type, "string")
            self.assertNotIn("mergeExt", dg._attr_dict)
            # normal attrs are still cached
            dg.find_attr("tx")
            self.assertIn("translateX", dg._attr_dict)
        finally:
            if cmds.attributeQuery("mergeExt", type="transform", exists=True):
                cmds.deleteExtension(
                    nodeType="transform", attribute="mergeExt", forceDelete=True
                )

    def test_rename_attr_with_a_plug_creates_no_node(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="foo", at="double")
        dg     = PyNode(net)
        before = sorted(cmds.ls())
        result = dg.rename_attr(Node(net).foo, "bar")
        self.assertEqual(sorted(cmds.ls()), before)
        self.assertTrue(cmds.attributeQuery("bar", node=net, exists=True))
        self.assertEqual(str(result), "net.bar")
