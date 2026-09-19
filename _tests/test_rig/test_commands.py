"""Tests for ``rig.bridges.commands`` (PEP 562 lazy maya.cmds wrappers) and
``Node.wrap()`` (explicit converter)."""

from maya import cmds
from rig import Node, PlugList
from rig.bridges import commands as rc
from rig._tests._base import MayaTestCase


class TestCommandsWrappers(MayaTestCase):
    """v4.C: PEP 562 lazy command wrappers around ``maya.cmds``."""

    TEST_START_NEW_SCENE = True

    def test_create_node_returns_node(self):
        n = rc.createNode("transform", name="cube1")
        self.assertIsInstance(n, Node)
        self.assertEqual(str(n), "cube1")

    def test_ls_returns_pluglist_of_nodes(self):
        rc.createNode("transform", name="a")
        rc.createNode("transform", name="b")
        cmds.select("a", "b")
        result = rc.ls(sl=True)
        self.assertIsInstance(result, PlugList)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, Node)

    def test_node_arg_coerced_to_str(self):
        # Maya commands take string node names. The wrapper must coerce
        # Node objects back to strings before the cmds call.
        child  = rc.createNode("transform", name="child")
        parent = rc.createNode("transform", name="parent1")
        # rc.parent(child_node, parent_node) -- both Nodes, not strings
        rc.parent(child, parent)
        # Verify the parenting actually happened.
        parents = cmds.listRelatives("child", parent=True) or []
        self.assertEqual(parents, ["parent1"])

    def test_list_of_nodes_arg_coerced(self):
        a      = rc.createNode("transform", name="a")
        b      = rc.createNode("transform", name="b")
        parent = rc.createNode("transform", name="p")
        rc.parent([a, b], parent)
        for child in ("a", "b"):
            self.assertEqual(cmds.listRelatives(child, parent=True) or [], ["p"])

    def test_get_attr_value_passes_through_unchanged(self):
        # rc.getAttr returning a numeric value should NOT try to wrap it
        # as a Node (it's a value, not a node name).
        n = rc.createNode("transform", name="cube1")
        n.tx << 5.0
        val = rc.getAttr("cube1.tx")
        self.assertIsInstance(val, float)
        self.assertAlmostEqual(val, 5.0)

    def test_lazy_wrapper_caching(self):
        # First access builds the wrapper; subsequent accesses return cached.
        from rig.bridges.commands import _WRAPPER_CACHE

        # Use a command unlikely to be touched by other tests.
        _ = rc.about
        self.assertIn("about", _WRAPPER_CACHE)
        cached = _WRAPPER_CACHE["about"]
        self.assertIs(rc.about, cached)

    def test_dir_lists_maya_cmds_callables(self):
        # Tab-completion via __dir__ should expose all wrappable commands.
        attrs = dir(rc)
        self.assertIn("ls",         attrs)
        self.assertIn("createNode", attrs)
        self.assertIn("parent",     attrs)
        # Should not include underscore-prefixed names.
        self.assertFalse(any(a.startswith("_") for a in attrs))

    def test_unknown_command_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = rc.thisCommandDefinitelyDoesNotExist

    def test_container_optout_kwarg_consumed(self):
        # ``container=False`` should be removed from kwargs before reaching
        # the cmds call (Maya wouldn't recognize it).
        n = rc.createNode("transform", name="standalone", container=False)
        self.assertIsInstance(n, Node)
        # No exception means container=False was popped before mc.createNode.

    def test_bool_result_passed_through(self):
        # rc.objExists returns bool, must NOT be wrapped as Node.
        rc.createNode("transform", name="exists1")
        result = rc.objExists("exists1")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)


class TestCommandsContainerScope(MayaTestCase):
    """Only the nodes a call CREATES join the active container: a query,
    a parent or a rename never moves a node in; creation is tracked through
    Maya's node-added message, not read off the result."""

    TEST_START_NEW_SCENE = True

    def _members(self, name):
        return sorted(cmds.container(name, query=True, nodeList=True) or [])

    def test_created_nodes_join_queries_and_edits_never_capture(self):
        from rig import container

        ctrl = rc.createNode("transform", name="ctrl")          # outside any scope
        mesh = rc.polyCube(name="mesh")[0]
        with container("build"):
            driven = rc.createNode("transform", name="driven")
            rc.ls("ctrl")                                        # a query
            rc.listRelatives(mesh, s=True)                       # a query
            rc.getAttr("ctrl.t")                                 # a value
            rc.parent(ctrl, driven)                              # an edit: returns the child
            rc.rename(mesh, "renamed")                           # an edit: returns the node
        self.assertEqual(self._members("build"), ["driven"])
        self.assertIsNone(cmds.container(query=True, findContainer=["driven|ctrl"]))
        self.assertIsNone(cmds.container(query=True, findContainer=["renamed"]))
        self.assertIsNone(cmds.container(query=True, findContainer=["renamedShape"]))

    def test_creation_is_tracked_not_read_off_the_result(self):
        from rig import container

        with container("build"):
            rc.polyCube(name="box")                              # returns transform + polyCube; the shape joins too
        self.assertEqual(self._members("build"), ["box", "boxShape", "polyCube1"])

    def test_container_false_keeps_created_nodes_out_and_is_harmless_on_a_query(self):
        from rig import container

        rc.createNode("transform", name="ctrl")
        with container("build"):
            rc.createNode("transform", name="driven")
            found = rc.ls("ctrl", container=False)
            rc.createNode("transform", name="loose", container=False)
        self.assertEqual([str(x) for x in found], ["ctrl"])
        self.assertEqual(self._members("build"), ["driven"])

    def test_no_scope_means_no_tracking(self):
        n = rc.createNode("transform", name="free")
        self.assertIsNone(cmds.container(query=True, findContainer=[str(n)]))


class TestNodeWrap(MayaTestCase):
    """v4.C: explicit ``Node.wrap()`` for direct ``maya.cmds`` results."""

    TEST_START_NEW_SCENE = True

    def test_wrap_string_returns_node(self):
        cmds.createNode("transform", name="foo")
        result = Node.wrap("foo")
        self.assertIsInstance(result, Node)
        self.assertEqual(str(result), "foo")

    def test_wrap_list_returns_pluglist_of_nodes(self):
        cmds.createNode("transform", name="a")
        cmds.createNode("transform", name="b")
        result = Node.wrap(["a", "b"])
        self.assertIsInstance(result, PlugList)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, Node)

    def test_wrap_none_returns_none(self):
        self.assertIsNone(Node.wrap(None))

    def test_wrap_numeric_passes_through(self):
        self.assertEqual(Node.wrap(5.0), 5.0)
        self.assertEqual(Node.wrap(42), 42)

    def test_wrap_invalid_string_passes_through(self):
        # If the string isn't a valid node name, it should pass through
        # unchanged (so wrap(getAttr(..., asString=True)) doesn't break).
        result = Node.wrap("not_a_node_xyz_12345")
        self.assertEqual(result, "not_a_node_xyz_12345")