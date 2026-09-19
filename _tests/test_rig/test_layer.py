"""Tests for ``rig.membership.Layer`` -- display layers through the membership
grammar.

Every error test asserts a zero ``cmds.ls()`` delta: a refused spelling
writes nothing. PlugLists are never compared with ``assertEqual``.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Components, container, Layer, Node, PlugList, Tag
from rig.maya.nodetypes import DisplayLayer
from rig._tests._base import MayaTestCase


DEFAULT = "defaultLayer"


def _cube(name):
    return Node(cmds.polyCube(name=name, ch=False)[0])


def _shape(node):
    """Full path of the first shape under a transform ``Node`` / name."""
    return cmds.listRelatives(str(node), shapes=True, fullPath=True)[0]


def _members(layer):
    """The explicit members of a layer, as full paths."""
    return cmds.editDisplayLayerMembers(
        str(layer), query=True, fullNames=True, noRecurse=True
    ) or []


def _layer(node):
    """The layer a node's own ``drawOverride`` reads, or None."""
    layers = cmds.listConnections(
        f"{node}.drawOverride", source=True, destination=False, type="displayLayer"
    )
    return layers[0] if layers else None


def _dag(path):
    selection = OpenMaya.MSelectionList()
    selection.add(path)
    return selection.getDagPath(0)


def _names(specs):
    return [repr(spec) for spec in specs]


# --------------------------------------------------------------------- #
#  Construction: lazy, validated
# --------------------------------------------------------------------- #


class TestLayerConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construction_makes_no_maya_calls(self):
        before = set(cmds.ls())
        specs  = [
            Layer("x"),
            Layer("ref", displayType=2, visibility=False, update=True),
            Layer(),
            Layer(None),
            -Layer("x"),
            Layer(DEFAULT),
        ]
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(specs[0]), "x")
        self.assertEqual(repr(specs[0]), "Layer('x')")
        self.assertEqual(specs[0].attrs, {})
        self.assertFalse(specs[0]._update)
        self.assertEqual(specs[1].attrs, {"displayType": 2, "visibility": False})
        self.assertTrue(specs[1]._update)
        # an empty call is the purge: the same spec as Layer(None)
        for purge in (specs[2], specs[3]):
            self.assertTrue(purge.purges)
            self.assertIsNone(purge.name)
            self.assertEqual(repr(purge), "Layer(None)")
        self.assertTrue(specs[4].removes)
        self.assertEqual(repr(specs[4]), "-Layer('x')")
        self.assertEqual(str(specs[5]), DEFAULT)
        self.assertEqual(Layer("x"), Layer("x"))
        self.assertNotEqual(Layer("x"), -Layer("x"))
        self.assertNotEqual(Layer("x"), Layer("y"))
        self.assertEqual(len({Layer("x"), Layer("x"), Layer("y")}), 2)

    def test_rejected_constructions(self):
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Layer("")
        with self.assertRaises(TypeError):
            Layer(5)
        for bad in ("bad name", "1x", "a-b", "a.b", "|grp|x"):
            with self.assertRaises(ValueError):
                Layer(bad)
        with self.assertRaisesRegex(TypeError, "members belong on the left"):
            Layer("x", [0, 1, 2])
        with self.assertRaises(TypeError):
            Layer("x", [])
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~Layer("x")
        with self.assertRaises(TypeError):
            ~Layer()
        with self.assertRaisesRegex(TypeError, "double negative"):
            -Layer()
        with self.assertRaises(TypeError):
            -(-Layer("x"))
        self.assertEqual(set(cmds.ls()), before)
        for good in ("ok", "ns:layer", "_x", "layer2"):
            self.assertEqual(str(Layer(good)), good)

    def test_methods_refuse_removal_and_purge_copies(self):
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            (-Layer("x")).delete()
        with self.assertRaises(TypeError):
            (-Layer("x")).rename("y")
        with self.assertRaises(TypeError):
            Layer().delete()
        with self.assertRaises(TypeError):
            Layer().clear()
        with self.assertRaises(TypeError):
            Layer().rename("y")
        with self.assertRaisesRegex(TypeError, "names no layer"):
            Layer().node
        with self.assertRaises(TypeError):
            Layer().visibility
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  Add: find-or-create, exclusive, the node itself
# --------------------------------------------------------------------- #


class TestLayerAdd(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def test_find_or_create_and_return_the_lhs(self):
        cmds.select(str(self.cube))
        selection = cmds.ls(selection=True)
        before    = set(cmds.ls())
        result    = self.cube << Layer("x")
        self.assertIs(result, self.cube)
        self.assertEqual(set(cmds.ls()) - before, {"x"})
        self.assertEqual(cmds.nodeType("x"), "displayLayer")
        self.assertEqual(cmds.ls(selection=True), selection)
        self.assertEqual(
            cmds.editDisplayLayerGlobals(query=True, currentDisplayLayer=True), DEFAULT
        )
        self.assertEqual(_members("x"), ["|cube"])
        self.assertEqual(_layer("|cube"), "x")
        # the transform, never its shape
        self.assertIsNone(_layer(self.shape))
        # present: an already-true assertion, nothing new
        before = set(cmds.ls())
        self.cube << Layer("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])
        # a second node joins the found layer
        other = _cube("other")
        other << Layer("x")
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])

    def test_membership_is_exclusive(self):
        self.cube << Layer("x")
        result = self.cube << Layer("y")
        self.assertIs(result, self.cube)
        self.assertEqual(_members("x"), [])
        self.assertEqual(_members("y"), ["|cube"])
        self.assertEqual(_layer("|cube"), "y")
        # a chain ends in the last layer
        self.assertIs(self.cube << Layer("a") << Layer("b"), self.cube)
        self.assertEqual(_layer("|cube"), "b")
        self.assertEqual(_members("a"), [])
        # chains across kinds return the left-hand side
        self.assertIs(self.cube << Tag("tt") << Layer("c"), self.cube)
        self.assertEqual(_layer("|cube"), "c")

    def test_kwargs_are_the_layers_attributes_on_create(self):
        self.cube << Layer("ref", displayType=2, visibility=False)
        self.assertEqual(cmds.getAttr("ref.displayType"), 2)
        self.assertFalse(cmds.getAttr("ref.visibility"))
        # skipped on a found layer
        other  = _cube("other")
        before = set(cmds.ls())
        other << Layer("ref", displayType=0, visibility=True)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.getAttr("ref.displayType"), 2)
        self.assertFalse(cmds.getAttr("ref.visibility"))
        # update=True re-asserts them
        other << Layer("ref", displayType=1, visibility=True, update=True)
        self.assertEqual(cmds.getAttr("ref.displayType"), 1)
        self.assertTrue(cmds.getAttr("ref.visibility"))
        self.assertEqual(set(cmds.ls()), before)

    def test_kwarg_typo_raises_before_any_write(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "no attribute 'visibilty'"):
            self.cube << Layer("x", visibilty=False)
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))
        self.cube << Layer("x")
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            self.cube << Layer("x", visibilty=False, update=True)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])

    def test_a_parented_group_adds_the_node_never_the_subtree(self):
        child = _cube("child")
        grp   = Node(cmds.group(str(child), name="grp"))
        grp << Layer("x", visibility=False)
        self.assertEqual(_members("x"), ["|grp"])
        self.assertEqual(_layer("|grp"), "x")
        self.assertIsNone(_layer("|grp|child"))
        self.assertIsNone(_layer("|grp|child|childShape"))
        self.assertEqual(Layer.of(child), [])
        self.assertIsNone(child >> Layer())
        self.assertFalse(child >> Layer("x"))
        # yet the child draws with the parent's override, through the DAG
        self.assertFalse(_dag("|grp|child|childShape").isVisible())
        Layer("x").visibility << True
        self.assertTrue(_dag("|grp|child|childShape").isVisible())
        Layer("x").displayType = 1
        self.assertTrue(_dag("|grp|child").isTemplated())
        # the child in a layer of its own is explicit there
        child << Layer("y")
        self.assertEqual(_members("y"), ["|grp|child"])
        self.assertEqual(_members("x"), ["|grp"])

    def test_shapes_joints_and_dg_nodes(self):
        # a shape named itself is the member; a joint works; a DG node refuses
        shape = Node(self.shape)
        self.assertIs(shape << Layer("x"), shape)
        self.assertEqual(_members("x"), [self.shape])
        self.assertIsNone(_layer("|cube"))
        joint = Node.create("joint", name="joint1")
        joint << Layer("x")
        self.assertEqual(sorted(_members("x")), [self.shape, "|joint1"])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            Node("lambert1") << Layer("x")
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            Node("lambert1") >> Layer("x")
        with self.assertRaises(TypeError):
            Layer.of(Node("time1"))
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            PlugList([self.cube, Node("lambert1")]) << Layer("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_pluglist_lhs_is_one_call(self):
        other = _cube("other")
        lhs   = PlugList([self.cube, other])
        with mock.patch.object(
            cmds, "editDisplayLayerMembers", wraps=cmds.editDisplayLayerMembers
        ) as edit:
            result = lhs << Layer("x")
        self.assertIs(result, lhs)
        writes = [c for c in edit.call_args_list if not c.kwargs.get("query")]
        self.assertEqual(len(writes), 1)
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        # a mixed list fails as a whole, nothing written
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold objects"):
            PlugList([other, self.cube.f[0]]) << Layer("y")
        with self.assertRaisesRegex(TypeError, r"element \[1\]"):
            PlugList([self.cube, 5, None]) << Layer("y")
        with self.assertRaises(ValueError):
            PlugList([]) << Layer("y")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_layer("|cube"), "x")

    def test_pluglist_broadcast_pairs_each_node_with_its_spec(self):
        other = _cube("other")
        lhs   = PlugList([self.cube, other])
        self.assertIs(lhs << [Layer("a"), Layer("b")], lhs)
        self.assertEqual(_layer("|cube"), "a")
        self.assertEqual(_layer("|other"), "b")
        answers = lhs >> [Layer("a"), Layer("b")]
        self.assertEqual(list(answers), [True, True])
        self.assertEqual([repr(x) for x in lhs >> [Layer(), Layer()]], ["Layer('a')", "Layer('b')"])

    def test_an_attribute_plug_stands_for_its_node(self):
        plug   = self.cube.tx
        result = plug << Layer("x")
        self.assertIs(result, plug)
        self.assertEqual(_members("x"), ["|cube"])
        self.assertTrue(self.cube.t >> Layer("x"))
        self.assertEqual(repr(self.cube.rotate >> Layer()), "Layer('x')")
        self.assertEqual(_names(Layer.of(self.cube.visibility)), ["Layer('x')"])
        PlugList([self.cube.tx, self.cube.ty]) << Layer("y")
        self.assertEqual(_layer("|cube"), "y")
        self.cube.sx << -Layer("y")
        self.assertIsNone(_layer("|cube"))
        self.cube.tx << Layer("y")
        self.cube.ry << Layer()
        self.assertIsNone(self.cube.ty >> Layer())
        # the shape's own plug stands for the shape
        Node(self.shape).castsShadows << Layer("x")
        self.assertEqual(_members("x"), [self.shape])
        # component plugs keep their meaning
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold objects"):
            self.cube.vtx[0] << Layer("x")
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.cube.t << [Layer("x"), 1, 2]
        self.assertEqual(set(cmds.ls()), before)

    def test_a_new_layer_never_joins_the_container(self):
        with container("rigctn"):
            self.cube << Layer("x")
            Node.create("transform", name="inside")
        self.assertIsNone(cmds.container(query=True, findContainer=["x"]))
        self.assertEqual(cmds.container("rigctn", query=True, nodeList=True), ["inside"])
        self.assertIsNone(cmds.container(query=True, findContainer=["cube"]))

    def test_namespace_idempotency(self):
        cmds.namespace(add="look")
        cmds.namespace(set="look")
        try:
            self.cube << Layer("x")
            self.assertEqual(sorted(cmds.ls(type="displayLayer")), [DEFAULT, "look:x"])
            before = set(cmds.ls())
            self.cube << Layer("x")
            self.assertEqual(set(cmds.ls()), before)
            self.assertEqual(str(Layer("x").node), "look:x")
            self.assertTrue(self.cube >> Layer("x"))
            self.assertEqual(repr(self.cube >> Layer()), "Layer('look:x')")
        finally:
            cmds.namespace(set=":")
        self.assertEqual(str(Layer("look:x").node), "look:x")
        # from the root namespace the short name is a different layer
        self.cube << Layer("x")
        self.assertEqual(sorted(cmds.ls(type="displayLayer")), [DEFAULT, "look:x", "x"])
        self.assertEqual(_members("look:x"), [])

    def test_a_non_layer_node_of_that_name_refuses(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "exists and is a transform"):
            self.cube << Layer("cube")
        with self.assertRaisesRegex(TypeError, "exists and is a lambert"):
            self.cube >> Layer("lambert1")
        with self.assertRaises(TypeError):
            Layer("cube").node
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_duplicates_keep_their_layer(self):
        self.cube << Layer("x")
        dup = Node(cmds.duplicate(str(self.cube))[0])
        self.assertTrue(dup >> Layer("x"))
        self.assertEqual(sorted(_members("x")), ["|cube", "|cube1"])


# --------------------------------------------------------------------- #
#  Remove, purge and defaultLayer
# --------------------------------------------------------------------- #


class TestLayerRemove(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.other = _cube("other")
        self.cube  << Layer("x")
        self.other << Layer("y")

    def test_removal_lands_in_default_layer(self):
        result = self.cube << -Layer("x")
        self.assertIs(result, self.cube)
        self.assertIsNone(_layer("|cube"))
        self.assertEqual(_members("x"), [])
        self.assertTrue(cmds.objExists("x"))
        self.assertEqual(sorted(_members(DEFAULT)), ["|cube", "|cube|cubeShape", "|other|otherShape"])
        # removing a non-member asserts a state that already holds
        before = set(cmds.ls())
        self.cube << -Layer("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_removal_from_another_layer_is_a_no_op(self):
        before = set(cmds.ls())
        result = self.other << -Layer("x")
        self.assertIs(result, self.other)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_layer("|other"), "y")
        self.assertEqual(_members("y"), ["|other"])
        # a list: only the members of x leave
        PlugList([self.cube, self.other]) << -Layer("x")
        self.assertIsNone(_layer("|cube"))
        self.assertEqual(_layer("|other"), "y")

    def test_purge_lands_in_default_layer(self):
        lhs    = PlugList([self.cube, self.other])
        result = lhs << Layer()
        self.assertIs(result, lhs)
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertEqual(_members("x"), [])
        self.assertEqual(_members("y"), [])
        # already in defaultLayer: nothing changes
        before = set(cmds.ls())
        self.cube << Layer(None)
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_default_layer_wraps_and_reads_as_no_layer(self):
        self.assertFalse(self.cube >> Layer(DEFAULT))
        result = self.cube << Layer(DEFAULT)
        self.assertIs(result, self.cube)
        self.assertIsNone(_layer("|cube"))
        self.assertTrue(self.cube >> Layer(DEFAULT))
        self.assertIsNone(self.cube >> Layer())
        self.assertEqual(Layer.of(self.cube), [])
        self.assertEqual(str(Layer(DEFAULT).node), DEFAULT)
        self.assertTrue(Layer(DEFAULT).visibility >> None)
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "contradictory"):
            self.cube << -Layer(DEFAULT)
        with self.assertRaisesRegex(TypeError, "cannot be deleted"):
            Layer(DEFAULT).delete()
        with self.assertRaisesRegex(TypeError, "cannot be renamed"):
            Layer(DEFAULT).rename("z")
        with self.assertRaisesRegex(TypeError, "cannot be cleared"):
            Layer(DEFAULT).clear()
        self.assertEqual(set(cmds.ls()), before)
        self.assertTrue(cmds.objExists(DEFAULT))

    def test_missing_layer_is_a_value_error(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no display layer named 'nope'"):
            self.cube << -Layer("nope")
        with self.assertRaises(ValueError):
            self.cube >> Layer("nope")
        with self.assertRaises(ValueError):
            Layer("nope").node
        with self.assertRaises(ValueError):
            Layer("nope").delete()
        with self.assertRaises(ValueError):
            Layer("nope").rename("still")
        with self.assertRaises(ValueError):
            Layer("nope").clear()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_layer("|cube"), "x")


# --------------------------------------------------------------------- #
#  Query and enumeration
# --------------------------------------------------------------------- #


class TestLayerQuery(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.other = _cube("other")
        self.cube << Layer("x")

    def test_query_is_a_bool(self):
        self.assertIs(self.cube >> Layer("x"), True)
        self.assertIs(self.other >> Layer("x"), False)
        self.assertIs(self.cube >> Layer("y") if cmds.objExists("y") else False, False)
        self.other << Layer("y")
        self.assertIs(self.cube >> Layer("y"), False)
        self.assertIs(self.other >> Layer("y"), True)
        # the shape is not in the layer its transform is in
        self.assertIs(Node(_shape(self.cube)) >> Layer("x"), False)
        # the same node twice is one node
        self.assertIs(PlugList([self.cube, self.cube]) >> Layer("x"), True)

    def test_purge_enumerates_the_one_layer_or_none(self):
        found = self.cube >> Layer()
        self.assertIsInstance(found, Layer)
        self.assertEqual(found, Layer("x"))
        self.assertEqual(str(found), "x")
        self.assertEqual(found.attrs, {})
        self.assertEqual(str(found.node), "x")
        self.assertIsNone(self.other >> Layer())
        self.assertEqual(_names(Layer.of(self.cube)), ["Layer('x')"])
        self.assertEqual(Layer.of(self.other), [])
        # re-injectable
        self.other << found
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        self.assertEqual(repr(self.other >> Layer(None)), "Layer('x')")

    def test_query_refusals(self):
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            self.cube >> -Layer("x")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            PlugList([self.cube, self.other]) >> Layer("x")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            PlugList([self.cube, self.other]) >> Layer()
        with self.assertRaises(TypeError):
            Layer.of(PlugList([self.cube, self.other]))
        for lhs in (self.cube.f, self.cube.f[0], self.cube.vtx, self.cube.vtx[:2],
                    self.cube.e[0], Components(self.cube, "vtx", [0])):
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                lhs >> Layer("x")
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                lhs >> Layer()
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                Layer.of(lhs)
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                lhs << Layer("x")
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                lhs << -Layer("x")
            with self.assertRaisesRegex(TypeError, "layers hold objects"):
                lhs << Layer()
        with self.assertRaises(ValueError):
            self.cube.f[6:] << Layer("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])
        self.assertIsNone(_layer(_shape(self.cube)))


# --------------------------------------------------------------------- #
#  The handle and the methods
# --------------------------------------------------------------------- #


class TestLayerMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.other = _cube("other")

    def test_handle_is_find_only(self):
        bg     = Layer("bg")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no display layer named 'bg'"):
            bg.node
        with self.assertRaises(ValueError):
            bg.visibility
        with self.assertRaises(ValueError):
            bg.visibility = False
        self.assertEqual(set(cmds.ls()), before)
        self.cube << bg
        node = bg.node
        self.assertIsInstance(node, Node)
        self.assertEqual(str(node), "bg")
        self.assertIsInstance(node._dg_node, DisplayLayer)
        plug = bg.visibility << False
        self.assertEqual(str(plug), "bg.visibility")
        self.assertFalse(cmds.getAttr("bg.visibility"))
        bg.visibility = True
        self.assertTrue(cmds.getAttr("bg.visibility"))
        bg.displayType << 2
        self.assertEqual(bg.displayType >> None, 2)
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            bg.visibilty
        with self.assertRaises(AttributeError):
            bg.visibilty = 1
        self.assertEqual(set(cmds.ls()), before)
        # the spec's state is unchanged by use
        self.assertEqual((bg.name, bg.attrs, repr(bg), bg.removes), ("bg", {}, "Layer('bg')", False))

    def test_delete_sends_members_to_default_layer(self):
        PlugList([self.cube, self.other]) << Layer("x")
        self.assertIsNone(Layer("x").delete())
        self.assertFalse(cmds.objExists("x"))
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertIsNone(self.cube >> Layer())
        self.assertEqual(cmds.ls(type="displayLayer"), [DEFAULT])

    def test_rename_follows_the_spec(self):
        bg = Layer("bg")
        self.cube << bg
        self.assertIsNone(bg.rename("back"))
        self.assertEqual(sorted(cmds.ls(type="displayLayer")), ["back", DEFAULT])
        self.assertEqual(str(bg), "back")
        self.assertEqual(str(bg.node), "back")
        self.assertTrue(self.cube >> bg)
        self.assertEqual(_layer("|cube"), "back")
        self.assertEqual(repr(self.cube >> Layer()), "Layer('back')")
        self.other << Layer("taken")
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "already exists"):
            bg.rename("taken")
        with self.assertRaisesRegex(ValueError, "already exists"):
            bg.rename("cube")
        with self.assertRaises(ValueError):
            bg.rename("1bad")
        with self.assertRaises(TypeError):
            bg.rename("")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(str(bg), "back")

    def test_clear_keeps_the_layer(self):
        PlugList([self.cube, self.other]) << Layer("x", visibility=False)
        self.assertIsNone(Layer("x").clear())
        self.assertTrue(cmds.objExists("x"))
        self.assertEqual(_members("x"), [])
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertFalse(cmds.getAttr("x.visibility"))
        # an empty layer clears to itself
        Layer("x").clear()
        self.assertEqual(_members("x"), [])


# --------------------------------------------------------------------- #
#  Undo
# --------------------------------------------------------------------- #


class TestLayerUndo(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_one_undo_reverts_a_whole_lshift(self):
        cmds.undoInfo(state=True, infinity=True)
        cube   = _cube("cube")
        other  = _cube("other")
        before = set(cmds.ls())
        PlugList([cube, other]) << Layer("x", displayType=2)
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        cmds.undo()
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        cmds.redo()
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        self.assertEqual(cmds.getAttr("x.displayType"), 2)
        cube << Layer("y")
        self.assertEqual(_layer("|cube"), "y")
        cmds.undo()
        self.assertEqual(_layer("|cube"), "x")
        self.assertFalse(cmds.objExists("y"))
        cube << -Layer("x")
        self.assertIsNone(_layer("|cube"))
        cmds.undo()
        self.assertEqual(_layer("|cube"), "x")
        cube << Layer()
        cmds.undo()
        self.assertEqual(_layer("|cube"), "x")
        Layer("x").delete()
        self.assertFalse(cmds.objExists("x"))
        cmds.undo()
        self.assertTrue(cmds.objExists("x"))
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        Layer("x").rename("z")
        cmds.undo()
        self.assertTrue(cmds.objExists("x"))
        self.assertEqual(_layer("|cube"), "x")


# --------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------- #


class TestLayerExports(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_top_level_names(self):
        import rig
        from rig import membership

        self.assertIs(rig.Layer, membership.Layer)
        self.assertIn("Layer", rig.__all__)
        self.assertIn("Layer", membership.__all__)
        self.assertIn("Layer", rig.__doc__)
        self.assertEqual(Layer.KIND, "layer")
        self.assertTrue(Layer.EXCLUSIVE)
        self.assertEqual(Layer.ACCEPTS, frozenset({"whole"}))
