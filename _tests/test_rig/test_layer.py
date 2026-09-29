"""Tests for ``rig.Layer`` -- display layers through the membership grammar.

Round 4b NC6 re-spelled every test (the ids are kept): ``Layer`` is the
display layer node class itself (``rig.Layer is DisplayLayer``). ``Layer('x')``
refers to a layer that exists and never creates it; ``Layer.define('x', ...)``
finds or makes it (its keywords are the layer's attributes, set when it is
made); ``cube << layer`` assigns, ``cube << -layer`` removes, ``Layer()`` is
the kind token (defaultLayer on ``<<``, the enumeration on ``>>``);
``cube in layer`` asks yes or no (a layer has no ids: ``cube >> layer``
raises); an attribute plug on the left of ``<<`` / ``>>`` / ``Layer.of`` is
refused (the node is the member) and stands for its node in ``in``.

Every error test asserts a zero ``cmds.ls()`` delta: a refused spelling
writes nothing. Lists are never compared with ``assertEqual``.
"""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Components, container, Layer, List, Node, NodeNotFoundError, NodeTypeError, Tag
from rig.nodetypes import DisplayLayer
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


def _names(layers):
    return [repr(layer) for layer in layers]


# --------------------------------------------------------------------- #
#  Construction: tokens and references write nothing
# --------------------------------------------------------------------- #


class TestLayerConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construction_makes_no_maya_calls(self):
        """Historical id (v2.0.0a2): pinned that building a Layer spec made zero
        Maya calls; it now pins that the kind and removal tokens make no layer
        call and that a reference (Layer('x'): Layer is DisplayLayer, the lazy
        spec retired in round 4b NC6) writes nothing."""
        layer  = Layer.define("x")
        before = set(cmds.ls())
        refused = AssertionError("a token called a layer command")
        with mock.patch.object(cmds, "createDisplayLayer", side_effect=refused), \
             mock.patch.object(cmds, "editDisplayLayerMembers", side_effect=refused), \
             mock.patch.object(cmds, "listConnections", side_effect=refused):
            tokens = [Layer(), -layer, -Layer("x"), Layer("x"), Layer(DEFAULT)]
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(repr(tokens[0]), "DisplayLayer()")
        self.assertEqual(repr(tokens[1]), '-DisplayLayer("x")')
        self.assertEqual(repr(tokens[2]), '-DisplayLayer("x")')
        self.assertTrue(tokens[0].purges)
        self.assertIsNone(tokens[0].name)
        self.assertTrue(tokens[1].removes)
        self.assertEqual(tokens[1].name, "x")
        # a reference is the node itself
        self.assertIsInstance(tokens[3], DisplayLayer)
        self.assertEqual(tokens[3], layer)
        self.assertEqual(repr(tokens[3]), 'DisplayLayer("x")')
        self.assertEqual(str(tokens[4]), DEFAULT)
        Layer.define("y")
        self.assertEqual(len({Layer("x"), Layer("x"), Layer("y")}), 2)
        self.assertNotEqual(Layer("x"), Layer("y"))

    def test_rejected_constructions(self):
        cmds.namespace(add="ns")
        Layer.define("x")
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            Layer("")
        with self.assertRaises((TypeError, ValueError)):
            Layer(5)
        # a missing name is a NodeNotFoundError (a ValueError) at the reference
        for bad in ("bad name", "1x", "a-b", "a.b", "|grp|x", "nope"):
            with self.assertRaises(ValueError):
                Layer(bad)
        # define checks the name it would make
        for bad in ("bad name", "1x", "a-b", "|grp|x"):
            with self.assertRaises((TypeError, ValueError)):
                Layer.define(bad)
        with self.assertRaisesRegex(TypeError, r"^None is not a layer name; Layer\(\) is defaultLayer"):
            Layer(None)
        with self.assertRaisesRegex(TypeError, "refers to an existing displayLayer and takes no attributes"):
            Layer("x", [0, 1, 2])
        with self.assertRaisesRegex(TypeError, "refers to an existing displayLayer"):
            Layer("x", displayType=2)
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~Layer("x")
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~Layer()
        with self.assertRaisesRegex(TypeError, "double negative"):
            -Layer()
        with self.assertRaisesRegex(TypeError, "cannot be negated again"):
            -(-Layer("x"))
        self.assertEqual(set(cmds.ls()), before)
        for good in ("ok", "ns:layer", "_x", "layer2"):
            Layer.define(good)
            self.assertEqual(str(Layer(good)), good)

    def test_methods_refuse_removal_and_purge_copies(self):
        # re-pinned (round 4b NC6): the tokens are no layer; the methods and
        # plugs are the layer node's (an AttributeError naming it)
        layer  = Layer.define("x")
        before = set(cmds.ls())
        for token in (-layer, Layer()):
            for name in ("delete", "rename", "clear", "visibility", "node"):
                with self.subTest(token=repr(token), name=name):
                    with self.assertRaisesRegex(AttributeError, "token.*not a layer"):
                        getattr(token, name)
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  Add: define finds or makes, exclusive, the node itself
# --------------------------------------------------------------------- #


class TestLayerAdd(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.shape = _shape(self.cube)

    def test_find_or_create_and_return_the_lhs(self):
        """Historical id (v2.0.0a2): pinned cube << Layer('x') finding or
        creating the layer; it now pins Layer.define('x') finding or making it
        (never the current layer, the selection untouched) and cube << layer
        returning the left-hand side."""
        cmds.select(str(self.cube))
        selection = cmds.ls(selection=True)
        before    = set(cmds.ls())
        layer     = Layer.define("x")
        self.assertEqual(set(cmds.ls()) - before, {"x"})
        result    = self.cube << layer
        self.assertIs(result, self.cube)
        self.assertEqual(set(cmds.ls()) - before, {"x"})
        self.assertEqual(cmds.nodeType("x"),      "displayLayer")
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
        self.assertEqual(Layer.define("x"), layer)
        self.cube << Layer.define("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])
        # a second node joins the found layer
        other = _cube("other")
        other << Layer("x")
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])

    def test_membership_is_exclusive(self):
        self.cube << Layer.define("x")
        result = self.cube << Layer.define("y")
        self.assertIs(result, self.cube)
        self.assertEqual(_members("x"),   [])
        self.assertEqual(_members("y"),   ["|cube"])
        self.assertEqual(_layer("|cube"), "y")
        # a chain ends in the last layer
        self.assertIs(self.cube << Layer.define("a") << Layer.define("b"), self.cube)
        self.assertEqual(_layer("|cube"), "b")
        self.assertEqual(_members("a"), [])
        # chains across kinds return the left-hand side
        self.assertIs(self.cube << Tag("tt") << Layer.define("c"), self.cube)
        self.assertEqual(_layer("|cube"), "c")

    def test_kwargs_are_the_layers_attributes_on_create(self):
        # re-spelled (round 4b NC6): the attributes go to Layer.define
        self.cube << Layer.define("ref", displayType=2, visibility=False)
        self.assertEqual(cmds.getAttr("ref.displayType"), 2)
        self.assertFalse(cmds.getAttr("ref.visibility"))
        # skipped on a found layer
        other  = _cube("other")
        before = set(cmds.ls())
        other << Layer.define("ref", displayType=0, visibility=True)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.getAttr("ref.displayType"), 2)
        self.assertFalse(cmds.getAttr("ref.visibility"))
        # update=True re-asserts them
        other << Layer.define("ref", displayType=1, visibility=True, update=True)
        self.assertEqual(cmds.getAttr("ref.displayType"), 1)
        self.assertTrue(cmds.getAttr("ref.visibility"))
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(sorted(_members("ref")), ["|cube", "|other"])

    def test_kwarg_typo_raises_before_any_write(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(AttributeError, "visibilty"):
            self.cube << Layer.define("x", visibilty=False)
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))
        self.cube << Layer.define("x")
        before = set(cmds.ls())
        with self.assertRaises(AttributeError):
            Layer.define("x", visibilty=False, update=True)
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])

    def test_a_parented_group_adds_the_node_never_the_subtree(self):
        child = _cube("child")
        grp   = Node(cmds.group(str(child), name="grp"))
        grp << Layer.define("x", visibility=False)
        self.assertEqual(_members("x"), ["|grp"])
        self.assertEqual(_layer("|grp"), "x")
        self.assertIsNone(_layer("|grp|child"))
        self.assertIsNone(_layer("|grp|child|childShape"))
        self.assertEqual(Layer.of(child), [])
        self.assertIsNone(child >> Layer())
        self.assertFalse(child in Layer("x"))
        self.assertTrue(grp in Layer("x"))
        # yet the child draws with the parent's override, through the DAG
        self.assertFalse(_dag("|grp|child|childShape").isVisible())
        Layer("x").visibility << True
        self.assertTrue(_dag("|grp|child|childShape").isVisible())
        Layer("x").displayType = 1
        self.assertTrue(_dag("|grp|child").isTemplated())
        # the child in a layer of its own is explicit there
        child << Layer.define("y")
        self.assertEqual(_members("y"), ["|grp|child"])
        self.assertEqual(_members("x"), ["|grp"])

    def test_shapes_joints_and_dg_nodes(self):
        # a shape named itself is the member; a joint works; a DG node refuses
        layer = Layer.define("x")
        shape = Node(self.shape)
        self.assertIs(shape << layer, shape)
        self.assertEqual(_members("x"), [self.shape])
        self.assertIsNone(_layer("|cube"))
        joint = Node.create("joint", name="joint1")
        joint << layer
        self.assertEqual(sorted(_members("x")), [self.shape, "|joint1"])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            Node("lambert1") << layer
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            Node("lambert1") in layer
        with self.assertRaises(TypeError):
            Layer.of(Node("time1"))
        with self.assertRaisesRegex(TypeError, "layers hold DAG objects"):
            List([self.cube, Node("lambert1")]) << layer
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_pluglist_lhs_is_one_call(self):
        other = _cube("other")
        lhs   = List([self.cube, other])
        x     = Layer.define("x")
        y     = Layer.define("y")
        with mock.patch.object(
            cmds, "editDisplayLayerMembers", wraps=cmds.editDisplayLayerMembers
        ) as edit:
            result = lhs << x
        self.assertIs(result, lhs)
        writes = [c for c in edit.call_args_list if not c.kwargs.get("query")]
        self.assertEqual(len(writes), 1)
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        # a mixed list fails as a whole, nothing written
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold objects"):
            List([other, self.cube.f[0]]) << y
        with self.assertRaisesRegex(TypeError, r"element \[1\]"):
            List([self.cube, 5, None]) << y
        with self.assertRaises(ValueError):
            List([]) << y
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_layer("|cube"), "x")

    def test_pluglist_broadcast_pairs_each_node_with_its_spec(self):
        # re-spelled (round 4b NC6): layer nodes pair with the nodes; the
        # yes / no answer is 'in' (a layer has no ids for '>>')
        other = _cube("other")
        lhs   = List([self.cube, other])
        a, b  = Layer.define("a"), Layer.define("b")
        self.assertIs(lhs << [a, b], lhs)
        self.assertEqual(_layer("|cube"), "a")
        self.assertEqual(_layer("|other"), "b")
        self.assertEqual([self.cube in a, other in b], [True, True])
        self.assertEqual(
            [repr(x) for x in lhs >> [Layer(), Layer()]], ['DisplayLayer("a")', 'DisplayLayer("b")']
        )
        with self.assertRaisesRegex(TypeError, "has no ids"):
            lhs >> [a, b]

    def test_an_attribute_plug_stands_for_its_node(self):
        """Historical id (v2.0.0a2): pinned an attribute plug standing for its
        node on '<<', '>>' and Layer.of; it now pins that it does so for 'in',
        and that '<<', '>>' and Layer.of refuse it before any write (user
        decision Q4 option A: the node is the member)."""
        x, y   = Layer.define("x"), Layer.define("y")
        before = set(cmds.ls())
        for label, call in (
            ("cube.tx << x",                 lambda: self.cube.tx << x),
            ("cube.t >> x",                  lambda: self.cube.t >> x),
            ("cube.rotate >> Layer()",       lambda: self.cube.rotate >> Layer()),
            ("Layer.of(cube.visibility)",    lambda: Layer.of(self.cube.visibility)),
            ("List([cube.tx, cube.ty]) << y", lambda: List([self.cube.tx, self.cube.ty]) << y),
            ("List([cube.tx, cube]) << y",   lambda: List([self.cube.tx, self.cube]) << y),
            ("cube.sx << -y",                lambda: self.cube.sx << -y),
            ("cube.ry << Layer()",           lambda: self.cube.ry << Layer()),
            ("shape.castsShadows << x",      lambda: Node(self.shape).castsShadows << x),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                    call()
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))
        self.assertEqual(_members("x"), [])
        # the node is the member; in 'in' a plug stands for its node
        self.cube << x
        self.assertTrue(self.cube.tx in x)
        self.assertTrue(self.cube.t in x)
        self.assertFalse(self.cube.tx in y)
        self.assertTrue(List([self.cube.tx, self.cube]) in x)
        # component plugs keep their meaning
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "layers hold objects"):
            self.cube.vtx[0] << x
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.cube.t << [x, 1, 2]
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])

    def test_a_new_layer_never_joins_the_container(self):
        with container("rigctn"):
            self.cube << Layer.define("x")
            Node.create("transform", name="inside")
        self.assertIsNone(cmds.container(query=True, findContainer=["x"]))
        self.assertEqual(cmds.container("rigctn", query=True, nodeList=True), ["inside"])
        self.assertIsNone(cmds.container(query=True, findContainer=["cube"]))

    def test_namespace_idempotency(self):
        cmds.namespace(add="look")
        cmds.namespace(set="look")
        try:
            self.cube << Layer.define("x")
            self.assertEqual(sorted(cmds.ls(type="displayLayer")), [DEFAULT, "look:x"])
            before = set(cmds.ls())
            self.cube << Layer.define("x")
            self.assertEqual(set(cmds.ls()), before)
            self.assertEqual(str(Layer("x")), "look:x")
            self.assertTrue(self.cube in Layer("x"))
            self.assertEqual(repr(self.cube >> Layer()), 'DisplayLayer("look:x")')
        finally:
            cmds.namespace(set=":")
        self.assertEqual(str(Layer("look:x")), "look:x")
        # from the root namespace the short name is a different layer
        self.cube << Layer.define("x")
        self.assertEqual(sorted(cmds.ls(type="displayLayer")), [DEFAULT, "look:x", "x"])
        self.assertEqual(_members("look:x"), [])

    def test_a_non_layer_node_of_that_name_refuses(self):
        # re-pinned (round 4b NC6): the reference's NodeTypeError (a TypeError)
        before = set(cmds.ls())
        with self.assertRaisesRegex(NodeTypeError, "'cube' is a transform, not a displayLayer"):
            self.cube << Layer("cube")
        with self.assertRaisesRegex(NodeTypeError, "'lambert1' is a lambert, not a displayLayer"):
            self.cube in Layer("lambert1")
        with self.assertRaisesRegex(NodeTypeError, "'cube' is a transform"):
            Layer.define("cube")
        self.assertFalse(Layer.exists("cube"))
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))

    def test_duplicates_keep_their_layer(self):
        self.cube << Layer.define("x")
        dup = Node(cmds.duplicate(str(self.cube))[0])
        self.assertTrue(dup in Layer("x"))
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
        self.cube  << Layer.define("x")
        self.other << Layer.define("y")

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
        self.assertEqual(set(cmds.ls()),   before)
        self.assertEqual(_layer("|other"), "y")
        self.assertEqual(_members("y"),    ["|other"])
        # a list: only the members of x leave
        List([self.cube, self.other]) << -Layer("x")
        self.assertIsNone(_layer("|cube"))
        self.assertEqual(_layer("|other"), "y")

    def test_purge_lands_in_default_layer(self):
        lhs    = List([self.cube, self.other])
        result = lhs << Layer()
        self.assertIs(result, lhs)
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertEqual(_members("x"), [])
        self.assertEqual(_members("y"), [])
        # already in defaultLayer: nothing changes
        before = set(cmds.ls())
        self.cube << Layer()
        self.assertEqual(set(cmds.ls()), before)
        self.assertIsNone(_layer("|cube"))
        # re-pinned (round 4b NC6): Layer(None) is refused, not the purge
        with self.assertRaisesRegex(TypeError, "None is not a layer name"):
            self.cube << Layer(None)

    def test_default_layer_wraps_and_reads_as_no_layer(self):
        self.assertFalse(self.cube in Layer(DEFAULT))
        result = self.cube << Layer(DEFAULT)
        self.assertIs(result, self.cube)
        self.assertIsNone(_layer("|cube"))
        self.assertTrue(self.cube in Layer(DEFAULT))
        self.assertIsNone(self.cube >> Layer())
        self.assertEqual(Layer.of(self.cube), [])
        self.assertEqual(str(Layer(DEFAULT)), DEFAULT)
        self.assertTrue(Layer(DEFAULT).is_default)
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
        # re-pinned (round 4b NC6): the strict reference raises NodeNotFoundError
        # (a ValueError) before any verb runs, so nothing is ever created
        before = set(cmds.ls())
        for label, call in (
            ("Layer('nope')",          lambda: Layer("nope")),
            ("cube << -Layer('nope')", lambda: self.cube << -Layer("nope")),
            ("cube << Layer('nope')",  lambda: self.cube << Layer("nope")),
            ("cube in Layer('nope')",  lambda: self.cube in Layer("nope")),
            ("Layer('nope').delete()", lambda: Layer("nope").delete()),
            ("Layer('nope').rename()", lambda: Layer("nope").rename("still")),
            ("Layer('nope').clear()",  lambda: Layer("nope").clear()),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, "no displayLayer named 'nope'"):
                    call()
        with self.assertRaises(NodeNotFoundError):
            Layer("nope")
        self.assertFalse(Layer.exists("nope"))
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
        self.cube << Layer.define("x")

    def test_query_is_a_bool(self):
        """Historical id (v2.0.0a2): pinned node >> Layer('x') answering a
        bool; it now pins 'in' / 'not in' answering it (user decision Q4 option
        A, all-members) and '>>' with a named layer refused (a layer holds
        whole objects and has no ids)."""
        self.assertIs(self.cube in Layer("x"), True)
        self.assertIs(self.other in Layer("x"), False)
        self.assertIs(self.other not in Layer("x"), True)
        self.assertFalse(Layer.exists("y"))
        self.other << Layer.define("y")
        self.assertIs(self.cube in Layer("y"), False)
        self.assertIs(self.other in Layer("y"), True)
        # the shape is not in the layer its transform is in
        self.assertIs(Node(_shape(self.cube)) in Layer("x"), False)
        # the same node twice is one node; a list is all-members
        self.assertIs(List([self.cube, self.cube]) in Layer("x"), True)
        self.assertIs(List([self.cube, self.other]) in Layer("x"), False)
        self.assertIs([self.cube, self.cube.tx] in Layer("x"), True)
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"has no ids; ask with cube in DisplayLayer\("):
            self.cube >> Layer("x")
        self.assertEqual(set(cmds.ls()), before)

    def test_purge_enumerates_the_one_layer_or_none(self):
        # re-pinned (round 4b NC6): the answers are the layer nodes
        found = self.cube >> Layer()
        self.assertIsInstance(found, DisplayLayer)
        self.assertEqual(found,      Layer("x"))
        self.assertEqual(str(found), "x")
        self.assertEqual(repr(found), 'DisplayLayer("x")')
        self.assertIsNone(self.other >> Layer())
        self.assertEqual(_names(Layer.of(self.cube)), ['DisplayLayer("x")'])
        self.assertIsInstance(Layer.of(self.cube)[0], DisplayLayer)
        self.assertEqual(Layer.of(self.other), [])
        # re-injectable
        self.other << found
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        self.assertEqual(repr(self.other >> Layer()), 'DisplayLayer("x")')

    def test_query_refusals(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "is a removal"):
            self.cube >> -Layer("x")
        with self.assertRaisesRegex(TypeError, "has no ids"):
            List([self.cube, self.other]) >> Layer("x")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            List([self.cube, self.other]) >> Layer()
        with self.assertRaises(TypeError):
            Layer.of(List([self.cube, self.other]))
        for lhs in (self.cube.f, self.cube.f[0], self.cube.vtx, self.cube.vtx[:2],
                    self.cube.e[0], Components(self.cube, "vtx", [0])):
            for label, call in (
                (">> Layer('x')",  lambda: lhs >> Layer("x")),
                (">> Layer()",     lambda: lhs >> Layer()),
                ("Layer.of",       lambda: Layer.of(lhs)),
                ("in Layer('x')",  lambda: lhs in Layer("x")),
                ("<< Layer('x')",  lambda: lhs << Layer("x")),
                ("<< -Layer('x')", lambda: lhs << -Layer("x")),
                ("<< Layer()",     lambda: lhs << Layer()),
            ):
                with self.subTest(lhs=repr(lhs), call=label):
                    with self.assertRaisesRegex(TypeError, "layers hold objects"):
                        call()
        with self.assertRaises(ValueError):
            self.cube.f[6:] << Layer("x")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_members("x"), ["|cube"])
        self.assertIsNone(_layer(_shape(self.cube)))


# --------------------------------------------------------------------- #
#  The node and its methods
# --------------------------------------------------------------------- #


class TestLayerMethods(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = _cube("cube")
        self.other = _cube("other")

    def test_handle_is_find_only(self):
        """Historical id (v2.0.0a2): pinned the Layer spec as a find-only handle
        (ValueError until the layer existed, then forwarding to it); it now pins
        the reference as find-only: Layer('bg') raises before the layer exists
        and writes nothing, and once defined it is the node itself (its plugs
        are native)."""
        before = set(cmds.ls())
        with self.assertRaisesRegex(NodeNotFoundError, "no displayLayer named 'bg'"):
            Layer("bg")
        with self.assertRaises(ValueError):
            Layer("bg").visibility
        self.assertEqual(set(cmds.ls()), before)
        bg = Layer.define("bg")
        self.cube << bg
        node = Layer("bg")
        self.assertIsInstance(node, Node)
        self.assertIsInstance(node, DisplayLayer)
        self.assertEqual(node, bg)
        self.assertEqual(str(node), "bg")
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
        self.assertEqual(repr(bg), 'DisplayLayer("bg")')

    def test_delete_sends_members_to_default_layer(self):
        List([self.cube, self.other]) << Layer.define("x")
        self.assertIsNone(Layer("x").delete())
        self.assertFalse(cmds.objExists("x"))
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertIsNone(self.cube >> Layer())
        self.assertEqual(cmds.ls(type="displayLayer"), [DEFAULT])

    def test_rename_follows_the_spec(self):
        """Historical id (v2.0.0a2): pinned the spec following its layer's
        rename; it now pins the layer node following it (the object held is
        the node), with the spec's refusals before any write."""
        bg = Layer.define("bg")
        self.cube << bg
        self.assertIsNone(bg.rename("back"))
        self.assertEqual(sorted(cmds.ls(type="displayLayer")), ["back", DEFAULT])
        self.assertEqual(str(bg), "back")
        self.assertEqual(bg, Layer("back"))
        self.assertTrue(self.cube in bg)
        self.assertEqual(_layer("|cube"), "back")
        self.assertEqual(repr(self.cube >> Layer()), 'DisplayLayer("back")')
        self.other << Layer.define("taken")
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
        List([self.cube, self.other]) << Layer.define("x", visibility=False)
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
        # re-spelled (round 4b NC6): define is its own undo step (rig.define),
        # each '<<' one more (rig.layer)
        cmds.undoInfo(state=True, infinity=True)
        cube   = _cube("cube")
        other  = _cube("other")
        before = set(cmds.ls())
        x      = Layer.define("x", displayType=2)
        List([cube, other]) << x
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        cmds.undo()
        self.assertIsNone(_layer("|cube"))
        self.assertIsNone(_layer("|other"))
        self.assertEqual(cmds.getAttr("x.displayType"), 2)
        cmds.undo()
        self.assertEqual(set(cmds.ls()), before)
        cmds.redo()
        cmds.redo()
        self.assertEqual(sorted(_members("x")), ["|cube", "|other"])
        self.assertEqual(cmds.getAttr("x.displayType"), 2)
        y = Layer.define("y")
        cube << y
        self.assertEqual(_layer("|cube"), "y")
        cmds.undo()
        self.assertEqual(_layer("|cube"), "x")
        self.assertTrue(cmds.objExists("y"))
        cmds.undo()
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
        """Historical id (v2.0.0a2): pinned rig.Layer as membership.Layer, the
        spec with its KIND / EXCLUSIVE / ACCEPTS; it now pins that rig.Layer,
        membership.Layer and rig.nodetypes.DisplayLayer are one class (Layer
        is DisplayLayer, round 4b NC6)."""
        import rig
        from rig import membership, nodetypes

        self.assertIs(rig.Layer, membership.Layer)
        self.assertIs(rig.Layer, DisplayLayer)
        self.assertIs(nodetypes.DisplayLayer, Layer)
        self.assertIn("Layer", rig.__all__)
        self.assertIn("Layer", membership.__all__)
        self.assertIn("Layer", rig.__doc__)
        self.assertTrue(DisplayLayer._MEMBER_KIND)
