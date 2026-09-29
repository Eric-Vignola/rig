"""Tests for ``rig.membership.Tag`` -- component tags through the membership
grammar, and the ``<<`` / ``>>`` dispatch that carries them.

Every error test asserts a zero ``cmds.ls()`` delta: a refused spelling
writes nothing. Lists are never compared with ``assertEqual``.
"""

from unittest import mock

import numpy as np
from maya import cmds
from rig import Components, List, Node, Plug, Tag
from rig.spec import String
from rig._tests._base import MayaTestCase


def _shape(node):
    """Full path of the first shape under a transform ``Node`` / name."""
    return cmds.listRelatives(str(node), shapes=True, fullPath=True)[0]


def _geo(node):
    return Node(_shape(node))


def _tags(node):
    return _geo(node).component_tags


def _contents(node, name):
    return _geo(node).get_component_tag_contents(name)


def _entries(node, name):
    """``(node, editable, final)`` of every history entry for ``name``."""
    return [
        (e["node"], e["editable"], e["final"])
        for e in _geo(node).get_component_tag_history() or []
        if e["key"] == name
    ]


def _names(specs):
    return sorted(str(spec) for spec in specs)


CUBE_TAGS = ["back", "bottom", "front", "left", "right", "top"]


# --------------------------------------------------------------------- #
#  Construction
# --------------------------------------------------------------------- #


class TestTagConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_no_name_is_the_purge(self):
        # re-pinned (round 4b NC6, user decision Q4: Spec(None) is refused): the
        # empty call is the purge, the kind token Tag() (repr "Tag()"); Tag(None)
        # is a TypeError, so a failed lookup never means every tag
        before = set(cmds.ls())
        purge  = Tag()
        self.assertTrue(purge.purges)
        self.assertIsNone(purge.name)
        self.assertEqual(repr(purge), "Tag()")
        with self.assertRaisesRegex(TypeError, r"^None is not a tag name; Tag\(\) means every tag$"):
            Tag(None)
        with self.assertRaises(TypeError):
            Tag(None, force=True)
        self.assertEqual(set(cmds.ls()), before)

    def test_name_validation(self):
        for bad in ("t", "q", "1bad", "bad name", "bad-name", "a.b"):
            with self.assertRaises(ValueError):
                Tag(bad)
        with self.assertRaises(TypeError):
            Tag("")
        with self.assertRaises(TypeError):
            Tag(5)
        for good in ("ok", "ns:tag", "_x", "top2", "a_b"):
            self.assertEqual(str(Tag(good)), good)
        self.assertEqual(repr(Tag("cap")), "Tag('cap')")

    def test_members_never_ride_on_the_spec(self):
        with self.assertRaisesRegex(TypeError, r"Tag\('x'\)\.set\("):
            Tag("foo", [0, 1, 2])
        with self.assertRaises(TypeError):
            Tag("foo", [0, 1, 2], None)
        with self.assertRaises(TypeError):
            Tag("foo", [])

    def test_negation_and_the_rejected_operators(self):
        spec    = Tag("cap", at="pCube1", force=True)
        removal = -spec
        self.assertTrue(removal.removes)
        self.assertFalse(spec.removes)
        self.assertEqual(removal._options, {"at": "pCube1", "force": True})
        self.assertEqual(repr(removal), "-Tag('cap')")
        with self.assertRaises(TypeError):
            -removal
        with self.assertRaisesRegex(TypeError, "double negative"):
            -Tag()
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~spec
        with self.assertRaisesRegex(TypeError, "unassigned"):
            ~Tag()

    def test_methods_refuse_removal_and_purge_copies(self):
        sph    = Node(cmds.polySphere(name="sph")[0])
        before = set(cmds.ls())
        with self.assertRaises(TypeError):
            (-Tag("cap")).set(sph.vtx[:2])
        with self.assertRaisesRegex(TypeError, "a method names one tag"):
            Tag().clear(sph)
        with self.assertRaisesRegex(TypeError, "a method names one tag"):
            Tag().delete(sph)
        with self.assertRaises(TypeError):
            (-Tag("cap")).rename(sph, "lid")
        self.assertEqual(set(cmds.ls()), before)


# --------------------------------------------------------------------- #
#  A polySphere: no procedural tags, the injection node is the shape
# --------------------------------------------------------------------- #


class TestTagOnSphere(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.sph   = Node(cmds.polySphere(name="sph")[0])
        self.shape = _shape(self.sph)

    def test_node_on_the_left_is_the_tag_itself(self):
        result = self.sph << Tag("cap")
        self.assertIs(result, self.sph)
        self.assertEqual(_tags(self.sph), ["cap"])
        self.assertEqual(_contents(self.sph, "cap"), [])
        empty = self.sph >> Tag("cap")
        self.assertIsInstance(empty, np.ndarray)
        self.assertEqual(empty.shape, (0,))
        # present: an already-true assertion, nothing written
        before = set(cmds.ls())
        self.sph << Tag("cap")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), ["cap"])

    def test_members_create_with_components_and_return_the_lhs(self):
        lhs    = self.sph.vtx[:5]
        result = lhs << Tag("cap")
        self.assertIs(result, lhs)
        self.assertEqual(_contents(self.sph, "cap"), ["vtx[0:4]"])
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1, 2, 3, 4])
        self.assertEqual(_entries(self.sph, "cap"), [("sphShape", True, True)])

    def test_first_population_of_an_empty_tag_replaces(self):
        # Maya: ``modify add`` on an EMPTY tag returns False and changes
        # nothing (pinned on a raw tag) -- rig's first population must go
        # through ``modify replace``.
        cmds.componentTag(self.shape, create=True, newTagName="raw", injectionLocation=self.shape)
        self.assertFalse(
            cmds.componentTag(
                f"{self.shape}.vtx[0:2]", modify="add", tagName="raw",
                injectionLocation=self.shape,
            )
        )
        self.assertEqual(_contents(self.sph, "raw"), [])
        self.sph         << Tag("cap")
        self.sph.vtx[:5] << Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), np.arange(5))

    def test_add_is_a_union(self):
        self.sph.vtx[:5]  << Tag("cap")
        self.sph.vtx[3:8] << Tag("cap")
        self.assertEqual(_contents(self.sph, "cap"), ["vtx[0:7]"])
        self.sph.vtx[[20, 10]] << Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1, 2, 3, 4, 5, 6, 7, 10, 20])

    def test_faces_into_a_vertex_tag_refuses_with_nothing_written(self):
        self.sph.vtx[:5] << Tag("cap")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"vertex tag.*Tag\('cap'\)\.set\("):
            self.sph.f[:2] << Tag("cap")
        with self.assertRaisesRegex(TypeError, "vertex tag"):
            self.sph.e[:2] << Tag("cap")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_contents(self.sph, "cap"), ["vtx[0:4]"])

    def test_face_and_edge_tags_read_back_in_their_own_category(self):
        faces  = self.sph.f[:3]
        result = faces << Tag("lid")
        self.assertIs(result, faces)
        self.assertEqual(_contents(self.sph, "lid"), ["f[0:2]"])
        # face ids, never the vertex cast of geometryAttrInfo -pointIndices
        np.testing.assert_array_equal(self.sph >> Tag("lid"), [0, 1, 2])
        self.sph.e[[0, 4]] << Tag("ed")
        self.assertEqual(_contents(self.sph, "ed"), ["e[0]", "e[4]"])
        np.testing.assert_array_equal(self.sph >> Tag("ed"), [0, 4])
        self.assertEqual(_geo(self.sph).get_component_tag_category("ed"), "e")

    def test_bare_handles_mean_every_component(self):
        self.sph.vtx << Tag("allv")
        self.assertEqual(len(self.sph >> Tag("allv")), 382)
        self.sph.f << Tag("allf")
        self.assertEqual(len(self.sph >> Tag("allf")), 400)
        self.sph.e << Tag("alle")
        self.assertEqual(len(self.sph >> Tag("alle")), 780)

    def test_remove_members(self):
        self.sph.vtx[:8] << Tag("cap")
        lhs    = self.sph.vtx[:2]
        result = lhs << -Tag("cap")
        self.assertIs(result, lhs)
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [2, 3, 4, 5, 6, 7])
        # removing non-members asserts a state that already holds
        self.sph.vtx[:2] << -Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [2, 3, 4, 5, 6, 7])
        # re-pinned (round 4b NC6, user decision 2026-09-28: a tag is untyped for
        # queries): faces are not in a vertex tag, so removing them is a no-op
        # with nothing written (it used to be a TypeError)
        before = set(cmds.ls())
        with mock.patch.object(cmds, "componentTag", wraps=cmds.componentTag) as tag_cmd:
            faces = self.sph.f[:1]
            self.assertIs(faces << -Tag("cap"), faces)
        self.assertFalse([c for c in tag_cmd.call_args_list if "modify" in c.kwargs])
        self.assertEqual(set(cmds.ls()), before)
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [2, 3, 4, 5, 6, 7])

    def test_remove_all_keeps_the_tag(self):
        self.sph.vtx[:8] << Tag("cap")
        self.sph.vtx     << -Tag("cap")
        self.assertEqual(_tags(self.sph), ["cap"])
        self.assertEqual((self.sph >> Tag("cap")).shape, (0,))
        # an empty tag takes any category on its first population
        self.sph.f[:2] << Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1])
        self.assertEqual(_geo(self.sph).get_component_tag_category("cap"), "f")
        # removing from an empty tag is a no-op whatever the category
        self.sph.f << -Tag("cap")
        self.assertEqual((self.sph >> Tag("cap")).shape, (0,))
        self.sph.vtx[:3] << -Tag("cap")
        self.sph.e[:3]   << -Tag("cap")
        self.assertEqual(_tags(self.sph), ["cap"])

    def test_node_on_the_left_of_a_removal_deletes_the_tag(self):
        self.sph.vtx[:8] << Tag("cap")
        result = self.sph << -Tag("cap")
        self.assertIs(result, self.sph)
        self.assertEqual(_tags(self.sph), [])
        # the deleted tag holds nothing: no ids, not in it (round 4b follow-up
        # F4; it was a ValueError)
        self.assertEqual((self.sph >> Tag("cap")).shape, (0,))
        self.assertFalse(self.sph in Tag("cap"))

    def test_missing_tag_is_a_value_error_never_an_empty_answer(self):
        """Historical id (v2.0.0a2): pinned every spelling on a missing tag as a
        ValueError; it now pins them but the queries: the ids of a missing tag
        are an empty array, with components (user decision 2026-09-28, the
        lead's default for '>>', consistent with 'in' answering False) and with
        the node on the left (round 4b follow-up F4, the same rule)."""
        before = set(cmds.ls())
        with self.assertRaisesRegex(ValueError, "no component tag 'nope'"):
            self.sph.vtx[:2] << -Tag("nope")
        with self.assertRaises(ValueError):
            self.sph << -Tag("nope")
        ids = self.sph >> Tag("nope")
        self.assertIsInstance(ids, np.ndarray)
        self.assertEqual(ids.shape, (0,))
        ids = self.sph.vtx[:2] >> Tag("nope")
        self.assertIsInstance(ids, np.ndarray)
        self.assertEqual(ids.shape, (0,))
        self.assertFalse(self.sph.vtx[:2] in Tag("nope"))
        self.assertFalse(self.sph in Tag("nope"))
        with self.assertRaises(ValueError):
            Tag("nope").clear(self.sph)
        with self.assertRaises(ValueError):
            Tag("nope").delete(self.sph)
        with self.assertRaises(ValueError):
            Tag("nope").rename(self.sph, "still")
        self.assertEqual(set(cmds.ls()), before)

    def test_purge_components_from_every_tag_of_their_category(self):
        self.sph.vtx[:4]  << Tag("aa")
        self.sph.vtx[2:6] << Tag("bb")
        self.sph.f[:2]    << Tag("ff")
        lhs    = self.sph.vtx[:3]
        result = lhs << Tag()
        self.assertIs(result, lhs)
        np.testing.assert_array_equal(self.sph >> Tag("aa"), [3])
        np.testing.assert_array_equal(self.sph >> Tag("bb"), [3, 4, 5])
        np.testing.assert_array_equal(self.sph >> Tag("ff"), [0, 1])
        self.assertEqual(_tags(self.sph), ["aa", "bb", "ff"])

    def test_purge_node_deletes_every_editable_tag(self):
        self.sph.vtx[:4] << Tag("aa")
        self.sph.f[:2]   << Tag("ff")
        self.sph         << Tag("empty")
        result = self.sph << Tag()
        self.assertIs(result, self.sph)
        self.assertEqual(_tags(self.sph), [])

    def test_query_a_slice_returns_the_ids_that_are_in_the_tag(self):
        self.sph.vtx[:8] << Tag("cap")
        got = self.sph.vtx[4:12] >> Tag("cap")
        self.assertIsInstance(got, np.ndarray)
        self.assertNotIsInstance(got, Components)
        np.testing.assert_array_equal(got, [4, 5, 6, 7])
        self.assertEqual((self.sph.vtx[8:] >> Tag("cap")).shape, (0,))
        np.testing.assert_array_equal(self.sph.vtx[[7, 0, 9]] >> Tag("cap"), [0, 7])
        np.testing.assert_array_equal(self.sph.vtx >> Tag("cap"), np.arange(8))
        np.testing.assert_array_equal(Components(self.sph, "vtx", [1, 30]) >> Tag("cap"), [1])
        # re-pinned (round 4b NC6, user decision 2026-09-28: a tag is untyped for
        # queries): faces are not in a vertex tag, so their ids are an empty
        # array (it used to be a TypeError), and 'in' answers False
        for faces in (self.sph.f, self.sph.f[:2]):
            ids = faces >> Tag("cap")
            self.assertIsInstance(ids, np.ndarray)
            self.assertEqual(ids.shape, (0,))
            self.assertFalse(faces in Tag("cap"))
        self.assertTrue(self.sph.vtx[:8] in Tag("cap"))
        self.assertFalse(self.sph.vtx[:9] in Tag("cap"))
        # the ids go back through the handle
        self.sph.vtx[self.sph >> Tag("cap")] << Tag("copy")
        np.testing.assert_array_equal(self.sph >> Tag("copy"), np.arange(8))

    def test_query_refuses_a_removal_and_enumerates_on_a_purge(self):
        self.sph.vtx[:8] << Tag("cap")
        self.sph.f[:3]   << Tag("lid")
        with self.assertRaises(TypeError):
            self.sph >> -Tag("cap")
        # '>> Tag()' is Tag.of: the tags holding the left-hand side
        for lhs in (
            self.sph, self.sph.vtx[:2], self.sph.vtx[3], self.sph.vtx[[3, 9]],
            self.sph.f[:2], self.sph.e[0], self.sph.vtx,
        ):
            got = lhs >> Tag()
            self.assertIsInstance(got, list)
            self.assertEqual(_names(got), _names(Tag.of(lhs)))
        self.assertEqual(_names(self.sph >> Tag()), ["cap", "lid"])
        # re-pinned (round 4b NC6): an attribute plug on the left is refused
        # (the node is the member), and Tag(None) is no longer the kind token
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
            self.sph.tx >> Tag()
        with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
            Tag.of(self.sph.tx)
        with self.assertRaisesRegex(TypeError, "None is not a tag name"):
            self.sph >> Tag(None)
        self.assertEqual(set(cmds.ls()), before)
        found = self.sph.vtx[3] >> Tag()
        self.assertEqual(_names(found), ["cap"])
        self.assertIsInstance(found[0], Tag)
        self.assertEqual(_names(self.sph.vtx[[3, 9]] >> Tag()), [])
        self.assertEqual(_names(self.sph.f[:2] >> Tag()), ["lid"])
        # re-injectable
        self.sph.vtx[9] << found[0]
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1, 2, 3, 4, 5, 6, 7, 9])
        # the same node twice is one node
        np.testing.assert_array_equal(
            List([self.sph, self.sph]) >> Tag("cap"), [0, 1, 2, 3, 4, 5, 6, 7, 9]
        )
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            List([self.sph, Node(cmds.polySphere(name="other")[0])]) >> Tag()
        self.assertEqual(set(cmds.ls()) - {"other", "otherShape", "polySphere2"}, before)

    def test_an_attribute_plug_stands_for_its_node(self):
        """Historical id (v2.0.0a2): pinned an attribute plug standing for its
        node on '<<', '>>' and Tag.of; it now pins that it stands for its node
        in 'in' only, and is refused on '<<', '>>' and Tag.of before any write
        (user decision Q4 option A: the node is the member), while component
        plugs stay members."""
        plug   = self.sph.tx
        before = set(cmds.ls())
        for label, call in (
            ("sph.tx << Tag('cap')",          lambda: plug << Tag("cap")),
            ("sph.ty << -Tag('cap')",         lambda: self.sph.ty << -Tag("cap")),
            ("sph.t >> Tag('cap')",           lambda: self.sph.t >> Tag("cap")),
            ("sph.rotate >> Tag()",           lambda: self.sph.rotate >> Tag()),
            ("Tag.of(sph.visibility)",        lambda: Tag.of(self.sph.visibility)),
            ("Tag.of(shape.componentTags[0])", lambda: Tag.of(Node(self.shape).componentTags[0])),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(TypeError, "is a plug; membership takes the node"):
                    call()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), [])
        # the node is the member: the tag itself, on the shape
        self.assertIs(self.sph << Tag("cap"), self.sph)
        self.assertEqual(_entries(self.sph, "cap"), [("sphShape", True, True)])
        self.sph.vtx[:3] << Tag("cap")
        # 'in': a plug stands for its node
        self.assertTrue(self.sph.tx in Tag("cap"))
        self.assertTrue(self.sph.t in Tag("cap"))
        self.assertTrue(List([self.sph.tx, self.sph.vtx[1]]) in Tag("cap"))
        self.assertFalse(self.sph.tx in Tag("nope"))
        self.sph << -Tag("cap")
        self.assertEqual(_tags(self.sph), [])
        # component plugs keep their own meaning
        self.sph.vtx[[1, 2]] << Tag("verts")
        np.testing.assert_array_equal(self.sph >> Tag("verts"), [1, 2])
        # a plug of a node without geometry: refused as a plug; in 'in' it
        # stands for the node, which is no geometry
        joint  = Node(cmds.createNode("joint", name="joint1"))
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "'joint1.translateX' is a plug"):
            joint.tx << Tag("xx")
        with self.assertRaisesRegex(TypeError, "'joint1.translateX' is a plug"):
            joint.tx >> Tag()
        # a query answers by contents (round 4b FIX): a joint has no tag
        self.assertNotIn(joint.tx, Tag("xx"))
        with self.assertRaisesRegex(TypeError, "is a plug"):
            self.sph.tx >> Tag("nope")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), ["verts"])

    def test_of(self):
        self.sph.vtx[:8] << Tag("cap")
        self.sph.f[:3]   << Tag("lid")
        self.assertEqual(_names(Tag.of(self.sph)), ["cap", "lid"])
        found = Tag.of(self.sph.vtx[3])
        self.assertEqual(_names(found), ["cap"])
        self.assertIsInstance(found[0], Tag)
        self.assertEqual(_names(Tag.of(self.sph.vtx[[3, 9]])), [])
        self.assertEqual(_names(Tag.of(self.sph.vtx[20])),     [])
        self.assertEqual(_names(Tag.of(self.sph.f[:2])),       ["lid"])
        self.assertEqual(_names(Tag.of(self.sph.e[0])),        [])
        # re-injectable
        self.sph.vtx[9] << found[0]
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1, 2, 3, 4, 5, 6, 7, 9])

    def test_set_replaces_and_may_flip_the_category(self):
        self.sph.vtx[:8] << Tag("cap")
        self.assertIsNone(Tag("cap").set(self.sph.f[:2]))
        self.assertEqual(_contents(self.sph, "cap"), ["f[0:1]"])
        self.assertEqual(_geo(self.sph).get_component_tag_category("cap"), "f")
        Tag("cap").set(self.sph.vtx[[1, 3]])
        self.assertEqual(_contents(self.sph, "cap"), ["vtx[1]", "vtx[3]"])
        Tag("fresh").set(self.sph.e[:2])
        self.assertEqual(_contents(self.sph, "fresh"), ["e[0:1]"])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "takes members"):
            Tag("cap").set(self.sph)
        self.assertEqual(set(cmds.ls()), before)

    def test_clear_rename_delete_methods(self):
        self.sph.vtx[:8] << Tag("cap")
        self.sph.vtx[:2] << Tag("other")
        self.assertIsNone(Tag("cap").clear(self.sph))
        self.assertEqual(_tags(self.sph), ["cap", "other"])
        self.assertEqual((self.sph >> Tag("cap")).shape, (0,))
        Tag("cap").rename(self.sph, "crown")
        self.assertEqual(_tags(self.sph), ["crown", "other"])
        before = set(cmds.ls())
        with self.assertRaises(ValueError):
            Tag("crown").rename(self.sph, "1bad")
        with self.assertRaisesRegex(ValueError, "already exists"):
            Tag("crown").rename(self.sph, "other")
        with self.assertRaisesRegex(TypeError, "takes the node"):
            Tag("crown").clear(self.sph.vtx[:2])
        with self.assertRaises(TypeError):
            Tag("crown").delete(self.sph.f[:2])
        self.assertEqual(set(cmds.ls()), before)
        Tag("crown").delete(self.sph)
        self.assertEqual(_tags(self.sph), ["other"])

    def test_of_applies_the_same_kind_gate_as_the_enumeration(self):
        """Pins (round 4b FIX): the enumeration and ``of`` answer by contents
        alike: UVs, which no tag holds, are in none; ``<<`` still refuses them."""
        before = set(cmds.ls())
        self.assertEqual(Tag.of(self.sph.map[0]), [])
        self.assertEqual(Tag.of(self.sph.map), [])
        self.assertEqual(self.sph.map[0] >> Tag(), [])
        with self.assertRaisesRegex(TypeError, "UVs cannot be tagged"):
            self.sph.map[0] << Tag("uvs")
        self.assertEqual(set(cmds.ls()), before)

    def test_uvs_non_geometry_and_channel_fanout_refuse(self):
        joint  = Node(cmds.createNode("joint", name="joint1"))
        empty  = Node.create("transform", name="empty")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "UVs cannot be tagged"):
            self.sph.map[:4] << Tag("uvs")
        with self.assertRaisesRegex(TypeError, "UVs cannot be tagged"):
            Components(self.shape, "uv", [0]) << Tag("uvs")
        with self.assertRaisesRegex(TypeError, "not geometry"):
            joint << Tag("xx")
        with self.assertRaisesRegex(TypeError, "not geometry"):
            empty << Tag("xx")
        with self.assertRaisesRegex(TypeError, "not geometry"):
            joint >> Tag("xx")
        # a collection spec has no per-channel meaning
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.sph.t << [Tag("xx"), 1, 2]
        with self.assertRaisesRegex(TypeError, "cannot be fanned"):
            self.sph.t << [Tag(), 1, 2]
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), [])

    def test_mixed_categories_and_node_plus_components_write_nothing(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "one category"):
            List([self.sph.vtx[0], self.sph.f[0]]) << Tag("mix")
        with self.assertRaisesRegex(TypeError, "Split them"):
            List([self.sph, self.sph.vtx[0]]) << Tag("mix")
        with self.assertRaises(TypeError):
            [self.sph.vtx[0], self.sph.f[0]] >> Tag("mix")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), [])

    def test_foreign_elements_refuse_naming_the_element(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"element \[1\]"):
            List([self.sph, 5, None]) << Tag("xx")
        # a raw component string reaches the normaliser only outside a
        # List (whose constructor lifts it to a Plug)
        with self.assertRaisesRegex(TypeError, r"Components\("):
            Tag("xx").inject([self.sph.vtx[0], f"{self.shape}.vtx[1]"])
        with self.assertRaises(ValueError):
            List([]) << Tag("xx")
        with self.assertRaises(ValueError):
            self.sph.f[400:] << Tag("xx")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), [])

    def test_a_string_plug_that_is_not_an_expression_stands_for_its_node(self):
        """Historical id (v2.0.0a2): pinned a string plug that is no deformer
        expression standing for its node (the tag made on it, the string never
        written); it now pins it refused before any write, the string never
        written either (user decision Q4 option A: the node is the member)."""
        # only a deformer's componentTagExpression receives the name (decided
        # by attribute name); any other string plug is an attribute plug
        self.sph << String("label")
        plug   = self.sph.label
        node   = Node.create("transform", name="labelled")
        node << String("label")
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "'sph.label' is a plug; membership takes the node: sph << Tag"):
            plug << Tag("viaLabel")
        with self.assertRaisesRegex(TypeError, "'labelled.label' is a plug"):
            node.label << Tag("xx")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), [])
        self.assertIsNone(cmds.getAttr("sph.label"))
        self.assertIsNone(cmds.getAttr("labelled.label"))

    def test_chain_returns_the_lhs(self):
        faces  = self.sph.f[:3]
        result = faces << Tag("aa") << Tag("bb")
        self.assertIs(result, faces)
        np.testing.assert_array_equal(self.sph >> Tag("aa"), [0, 1, 2])
        np.testing.assert_array_equal(self.sph >> Tag("bb"), [0, 1, 2])
        # '<< None' is never a clear: the first '<<' ran, the second refuses
        with self.assertRaisesRegex(TypeError, "Components << None"):
            faces << Tag("cc") << None
        np.testing.assert_array_equal(self.sph >> Tag("cc"), [0, 1, 2])

    def test_pluglist_broadcast_pairs_each_selection_with_its_spec(self):
        lhs    = List([self.sph.f[:2], self.sph.f[2:4]])
        result = lhs << [Tag("aa"), Tag("bb")]
        self.assertIs(result, lhs)
        np.testing.assert_array_equal(self.sph >> Tag("aa"), [0, 1])
        np.testing.assert_array_equal(self.sph >> Tag("bb"), [2, 3])
        answers = lhs >> [Tag("aa"), Tag("bb")]
        self.assertIsInstance(answers, List)
        np.testing.assert_array_equal(answers[0], [0, 1])
        np.testing.assert_array_equal(answers[1], [2, 3])
        # one spec for the whole list: one union
        List([self.sph.f[:2], self.sph.f[4:6]]) << Tag("cc")
        np.testing.assert_array_equal(self.sph >> Tag("cc"), [0, 1, 4, 5])

    def test_multi_node_pluglist_one_command_per_node_one_category_per_node(self):
        cube = Node(cmds.polySphere(name="other")[0])
        lhs  = List([self.sph.vtx[:2], cube.f[:2], self.sph.vtx[5]])
        lhs << Tag("mixed")
        np.testing.assert_array_equal(self.sph >> Tag("mixed"), [0, 1, 5])
        np.testing.assert_array_equal(cube >> Tag("mixed"), [0, 1])
        self.assertEqual(_geo(self.sph).get_component_tag_category("mixed"), "v")
        self.assertEqual(_geo(cube).get_component_tag_category("mixed"), "f")
        with self.assertRaisesRegex(TypeError, "one node at a time"):
            List([self.sph, cube]) >> Tag("mixed")
        with self.assertRaises(TypeError):
            Tag.of(List([self.sph, cube]))

    def test_one_undo_reverts_a_whole_lshift(self):
        cmds.undoInfo(state=True, infinity=True)
        cube = Node(cmds.polySphere(name="other")[0])
        List([self.sph.vtx[:3], cube.vtx[:3]]) << Tag("both")
        np.testing.assert_array_equal(self.sph >> Tag("both"), [0, 1, 2])
        np.testing.assert_array_equal(cube >> Tag("both"), [0, 1, 2])
        cmds.undo()
        self.assertEqual(_tags(self.sph), [])
        self.assertEqual(_tags(cube), [])
        cmds.redo()
        np.testing.assert_array_equal(self.sph >> Tag("both"), [0, 1, 2])
        np.testing.assert_array_equal(cube >> Tag("both"), [0, 1, 2])
        self.sph.vtx[3:6] << Tag("both")
        cmds.undo()
        np.testing.assert_array_equal(self.sph >> Tag("both"), [0, 1, 2])
        self.sph << Tag()
        self.assertEqual(_tags(self.sph), [])
        cmds.undo()
        np.testing.assert_array_equal(self.sph >> Tag("both"), [0, 1, 2])


# --------------------------------------------------------------------- #
#  A clustered mesh: tags live on the Orig; deformers read them by name
# --------------------------------------------------------------------- #


class TestTagOnClusteredMesh(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.sph     = Node(cmds.polySphere(name="sph", ch=False)[0])
        self.shape   = _shape(self.sph)
        self.cluster = Node(cmds.cluster("sph")[0])
        self.orig    = cmds.deformableShape(self.shape, tagInjectionNode=True)[0]
        self.expr    = self.cluster.input[0].componentTagExpression

    def test_tags_land_on_the_orig(self):
        self.assertNotEqual(self.orig, "sphShape")
        self.sph.vtx[:4] << Tag("cap")
        self.assertEqual(_entries(self.sph, "cap"), [(self.orig, True, True)])
        self.assertIsNone(cmds.getAttr(f"{self.shape}.componentTags", multiIndices=True))
        self.assertEqual(cmds.getAttr(f"{self.orig}.componentTags", multiIndices=True), [0])
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [0, 1, 2, 3])
        self.sph.vtx[4:6] << Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), np.arange(6))
        self.sph.vtx[0] << -Tag("cap")
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [1, 2, 3, 4, 5])
        self.assertEqual(_names(Tag.of(self.sph.vtx[1])), ["cap"])
        # at= pointing at another node than the editable home refuses
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "already lives"):
            self.sph.vtx[:2] << Tag("cap", at=self.shape)
        self.assertEqual(set(cmds.ls()), before)
        np.testing.assert_array_equal(self.sph >> Tag("cap"), [1, 2, 3, 4, 5])
        self.sph << -Tag("cap")
        self.assertEqual(_tags(self.sph), [])
        self.assertIsNone(cmds.getAttr(f"{self.orig}.componentTags", multiIndices=True))

    def test_expression_sugar_writes_the_name(self):
        self.sph.vtx[:4] << Tag("cap")
        with mock.patch.object(cmds, "warning") as warning:
            result = self.expr << Tag("cap")
        self.assertIs(result, self.expr)
        self.assertEqual(self.expr >> None, "cap")
        warning.assert_not_called()
        # the deformer reads it: only tagged vertices move
        self.cluster.input[0].componentTagExpression >> None
        handle = cmds.listConnections(f"{self.cluster}.matrix", s=True, d=False)[0]
        cmds.setAttr(f"{handle}.ty", 5)
        moved = cmds.pointPosition(f"{self.shape}.vtx[0]", world=True)[1]
        still = cmds.pointPosition(f"{self.shape}.vtx[100]", world=True)[1]
        self.assertGreater(moved, 3)
        self.assertLess(still, 3)
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "'!cap'"):
            self.expr << -Tag("cap")
        with self.assertRaisesRegex(TypeError, "has no name to write"):
            self.expr << Tag()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(self.expr >> None, "cap")
        # the sugar is decided by attribute name, before the attribute-plug
        # refusal (re-pinned, round 4b NC6: any other attribute plug is refused,
        # where it used to stand for its node, a DG node refused as not geometry)
        with self.assertRaisesRegex(TypeError, "'cluster1.envelope' is a plug"):
            self.cluster.envelope << Tag("cap")
        self.assertEqual(_tags(self.sph), ["cap"])
        # expressions stay plain strings
        self.expr << "cap + lid"
        self.assertEqual(self.expr >> None, "cap + lid")

    def test_expression_sugar_warns_when_the_tag_is_downstream_of_the_deformer(self):
        cmds.polySmooth("sph")
        self.assertEqual(
            cmds.deformableShape(self.shape, tagInjectionNode=True)[0], "sphShape"
        )
        self.sph.vtx[:5] << Tag("late")
        self.assertEqual(_entries(self.sph, "late"), [("sphShape", True, True)])
        with mock.patch.object(cmds, "warning") as warning:
            self.expr << Tag("late")
        self.assertEqual(self.expr >> None, "late")
        warning.assert_called_once()
        self.assertIn("'late'", warning.call_args[0][0])
        self.assertIn("cluster1.input[0]", warning.call_args[0][0])

    def test_at_upstream_of_a_topology_change_verifies_against_the_holder(self):
        cmds.polySmooth("sph")                       # 382 Orig points, 1562 on the shape
        self.sph.vtx << Tag("up", at=self.orig)      # every point of the ORIG, not the LHS
        self.assertEqual(Node(self.orig).get_component_tag_indices("up").size, 382)
        self.sph.vtx[:5] << Tag("up2", at=self.orig)
        np.testing.assert_array_equal(
            Node(self.orig).get_component_tag_indices("up2"), np.arange(5)
        )
        with self.assertRaisesRegex(TypeError, "at= places a tag"):
            self.sph >> Tag("up", at=self.orig)

    def test_node_lhs_refuses_a_procedural_name(self):
        cube   = Node(cmds.polyCube(name="hc", ch=True)[0])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            cube << Tag("top")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(cmds.geometryAttrInfo("hcShape.outMesh", componentTagCategory=True, componentTagExpression="top"), "f")

    def test_delete_refuses_while_referenced_unless_forced(self):
        self.sph.vtx[:4] << Tag("cap")
        self.expr        << Tag("cap")
        before = set(cmds.ls())
        with self.assertRaisesRegex(
            RuntimeError, r"cluster1\.input\[0\]\.componentTagExpression.*force=True"
        ):
            Tag("cap").delete(self.sph)
        with self.assertRaises(RuntimeError):
            self.sph << -Tag("cap")
        with self.assertRaises(RuntimeError):
            self.sph << Tag()
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), ["cap"])
        # an unreferenced tag still goes without force
        self.sph.vtx[:2] << Tag("free")
        self.sph         << -Tag("free")
        self.assertEqual(_tags(self.sph), ["cap"])
        Tag("cap").delete(self.sph, force=True)
        self.assertEqual(_tags(self.sph), [])
        self.sph.vtx[:4] << Tag("cap")
        self.sph         << -Tag("cap", force=True)
        self.assertEqual(_tags(self.sph), [])
        self.sph.vtx[:4] << Tag("cap")
        self.sph         << Tag(force=True)
        self.assertEqual(_tags(self.sph), [])

    def test_rename_rewrites_exact_references_with_force(self):
        self.sph.vtx[:4] << Tag("cap")
        self.expr        << "cap + lid"
        before = set(cmds.ls())
        with self.assertRaisesRegex(RuntimeError, "force=True"):
            Tag("cap").rename(self.sph, "crown")
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_tags(self.sph), ["cap"])
        Tag("cap").rename(self.sph, "crown", force=True)
        self.assertEqual(_tags(self.sph), ["crown"])
        self.assertEqual(self.expr >> None, "crown + lid")
        np.testing.assert_array_equal(self.sph >> Tag("crown"), [0, 1, 2, 3])
        self.expr << "!crown"
        Tag("crown", force=True).rename(self.sph, "cap")
        self.assertEqual(self.expr >> None, "!cap")
        # a glob reference refuses even with force, nothing renamed
        self.expr << "ca* - lid"
        with self.assertRaisesRegex(RuntimeError, "glob"):
            Tag("cap").rename(self.sph, "crown", force=True)
        self.assertEqual(_tags(self.sph), ["cap"])
        self.assertEqual(self.expr >> None, "ca* - lid")

    def test_a_reference_to_a_same_named_tag_on_another_mesh_does_not_block(self):
        other   = Node(cmds.polySphere(name="other", ch=False)[0])
        cluster = Node(cmds.cluster("other")[0])
        other.vtx[:2]                           << Tag("cap")
        cluster.input[0].componentTagExpression << Tag("cap")
        self.sph.vtx[:4]                        << Tag("cap")
        self.sph                                << -Tag("cap")
        self.assertEqual(_tags(self.sph), [])
        self.assertEqual(_tags(other), ["cap"])


# --------------------------------------------------------------------- #
#  A polyCube WITH history: six procedural face tags owned by polyCube1
# --------------------------------------------------------------------- #


class TestTagOnHistoryCube(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = Node(cmds.polyCube(name="hc")[0])
        self.shape = _shape(self.cube)

    def test_procedural_name_refuses_and_suggests_at(self):
        before = set(cmds.ls())
        with self.assertRaisesRegex(
            TypeError, r"PROCEDURAL.*polyCube1.*Tag\('top', at='hcShape'\)"
        ):
            self.cube.f[:3] << Tag("top")
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            self.cube.f[1] << -Tag("top")
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            self.cube << -Tag("top")
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            Tag("top").clear(self.cube)
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            Tag("top").rename(self.cube, "cap")
        with self.assertRaisesRegex(TypeError, "PROCEDURAL"):
            Tag("top").set(self.cube.f[:2])
        self.assertEqual(set(cmds.ls()), before)
        self.assertEqual(_contents(self.cube, "top"), ["f[1]"])
        self.assertIsNone(cmds.getAttr(f"{self.shape}.componentTags", multiIndices=True))

    def test_at_shadows_and_later_edits_find_the_shadow(self):
        faces  = self.cube.f[:3]
        result = faces << Tag("top", at=self.cube)
        self.assertIs(result, faces)
        np.testing.assert_array_equal(self.cube >> Tag("top"), [0, 1, 2])
        self.assertEqual(
            _entries(self.cube, "top"),
            [("polyCube1", False, False), ("hcShape", True, True)],
        )
        # later edits pass injectionLocation=hcShape without at=
        self.cube.f[3] << Tag("top")
        np.testing.assert_array_equal(self.cube >> Tag("top"), [0, 1, 2, 3])
        self.cube.f[0] << -Tag("top")
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1, 2, 3])
        # face 3 is polyCube's own 'bottom' as well as the shadow
        self.assertEqual(_names(Tag.of(self.cube.f[3])), ["bottom", "top"])
        Tag("top").set(self.cube.vtx[:2])
        self.assertEqual(_contents(self.cube, "top"), ["vtx[0:1]"])
        # deleting the shadow uncovers the procedural tag again
        self.cube << -Tag("top")
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])
        self.assertEqual(_entries(self.cube, "top"), [("polyCube1", False, True)])
        # the node on the left with at= creates an empty shadow
        self.cube << Tag("front", at=self.shape)
        self.assertEqual((self.cube >> Tag("front")).shape, (0,))

    def test_at_must_be_the_shape_or_upstream_of_it(self):
        other  = Node(cmds.polySphere(name="other")[0])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "at="):
            self.cube.f[:2] << Tag("cap", at=other)
        with self.assertRaisesRegex(TypeError, "at="):
            self.cube.f[:2] << Tag("cap", at=Node.create("transform", name="grp"))
        self.assertEqual(set(cmds.ls()) - {"grp"}, before)
        self.assertEqual(_tags(self.cube), CUBE_TAGS)
        # at= naming the node an editable tag already lives on edits it there
        self.cube.f[:2] << Tag("cap")
        self.cube.f[3]  << Tag("cap", at=self.shape)
        np.testing.assert_array_equal(self.cube >> Tag("cap"), [0, 1, 3])
        self.assertEqual(_entries(self.cube, "cap"), [("hcShape", True, True)])

    def test_purge_lists_procedural_names_in_one_warning(self):
        self.cube.f[:2] << Tag("cap")
        with mock.patch.object(cmds, "warning") as warning:
            self.cube.f[:2] << Tag()
        self.assertEqual((self.cube >> Tag("cap")).shape, (0,))
        warning.assert_called_once()
        message = warning.call_args[0][0]
        for name in CUBE_TAGS:
            self.assertIn(name, message)
        self.assertIn("polyCube1", message)
        with mock.patch.object(cmds, "warning") as warning:
            self.cube << Tag()
        self.assertEqual(_tags(self.cube), CUBE_TAGS)
        warning.assert_called_once()
        # a vertex purge does not mention the face tags
        self.cube.vtx[:2] << Tag("verts")
        with mock.patch.object(cmds, "warning") as warning:
            self.cube.vtx[:2] << Tag()
        warning.assert_not_called()
        self.assertEqual((self.cube >> Tag("verts")).shape, (0,))

    def test_query_and_of_see_procedural_tags(self):
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])
        self.assertEqual(_names(Tag.of(self.cube)), CUBE_TAGS)
        self.assertEqual(_names(Tag.of(self.cube.f[1])), ["top"])
        np.testing.assert_array_equal(self.cube.f[[1, 2]] >> Tag("top"), [1])


# --------------------------------------------------------------------- #
#  A polyCube ch=False: the six tags are baked into the shape's own multi
# --------------------------------------------------------------------- #


class TestTagOnBakedCube(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.cube  = Node(cmds.polyCube(name="bc", ch=False)[0])
        self.shape = _shape(self.cube)

    def test_baked_tags_edit_through_the_plug_path(self):
        # the command refuses them (pinned), the plug path edits them
        self.assertFalse(
            cmds.componentTag(
                f"{self.shape}.f[4]", modify="add", tagName="top",
                injectionLocation=self.shape,
            )
        )
        faces  = self.cube.f[:3]
        result = faces << Tag("top")
        self.assertIs(result, faces)
        np.testing.assert_array_equal(self.cube >> Tag("top"), [0, 1, 2])
        self.cube.f[0] << -Tag("top")
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1, 2])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, "face tag"):
            self.cube.vtx[:2] << Tag("back")
        self.assertEqual(set(cmds.ls()), before)
        Tag("front").set(self.cube.vtx[:2])
        self.assertEqual(_contents(self.cube, "front"), ["vtx[0:1]"])
        self.assertEqual(_geo(self.cube).get_component_tag_category("front"), "v")
        Tag("left").clear(self.cube)
        self.assertEqual((self.cube >> Tag("left")).shape, (0,))
        Tag("right").rename(self.cube, "east")
        self.assertIn("east", _tags(self.cube))
        self.assertNotIn("right", _tags(self.cube))
        self.cube << -Tag("top")
        self.assertNotIn("top", _tags(self.cube))
        self.cube.f << -Tag("bottom")
        self.assertEqual((self.cube >> Tag("bottom")).shape, (0,))
        self.cube.vtx[:2] << Tag()
        self.assertEqual((self.cube >> Tag("front")).shape, (0,))
        self.cube << Tag()
        self.assertEqual(_tags(self.cube), [])

    def test_baked_edits_undo_as_one_step(self):
        cmds.undoInfo(state=True, infinity=True)
        List([self.cube.f[:3], self.cube.f[[5]]]) << Tag("top")
        np.testing.assert_array_equal(self.cube >> Tag("top"), [0, 1, 2, 5])
        cmds.undo()
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])
        self.cube << -Tag("top")
        cmds.undo()
        self.assertEqual(_tags(self.cube), CUBE_TAGS)

    def test_purge_ignores_the_dead_entry_a_deleted_baked_tag_leaves(self):
        self.cube << -Tag("top")
        self.assertNotIn("top", _tags(self.cube))
        with mock.patch.object(cmds, "warning") as warning:
            self.cube << Tag()
        warning.assert_not_called()
        self.assertEqual(_tags(self.cube), [])

    def test_query_and_of_on_a_baked_cube(self):
        np.testing.assert_array_equal(self.cube >> Tag("top"), [1])
        self.assertEqual(_names(Tag.of(self.cube.f[1])), ["top"])
        self.assertEqual(_names(Tag.of(self.cube)),      CUBE_TAGS)
        self.assertEqual(_entries(self.cube, "top"),     [("bcShape", False, False), ("bcShape", True, True)])


# --------------------------------------------------------------------- #
#  Curves, periodic surfaces and lattices
# --------------------------------------------------------------------- #


class TestTagOnOtherGeometry(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_nurbs_curve(self):
        crv = Node(cmds.curve(name="crv", d=1, p=[(i, 0, 0) for i in range(6)]))
        lhs = crv.cv[:3]
        self.assertIs(lhs << Tag("root"), lhs)
        self.assertEqual(_contents(crv, "root"), ["cv[0:2]"])
        np.testing.assert_array_equal(crv >> Tag("root"), [0, 1, 2])
        np.testing.assert_array_equal(crv.cv[1:5] >> Tag("root"), [1, 2])
        self.assertEqual(_names(Tag.of(crv.cv[0])), ["root"])
        crv.cv[0] << -Tag("root")
        np.testing.assert_array_equal(crv >> Tag("root"), [1, 2])
        crv.cv << Tag("all")
        self.assertEqual(len(crv >> Tag("all")), 6)
        Tag("root").rename(crv, "base")
        self.assertEqual(_tags(crv), ["base", "all"])

    def test_periodic_surface_round_trips_two_dimensional_ids(self):
        srf = Node(cmds.sphere(name="srf")[0])
        lhs = srf.cv[1:3, 2:4]
        self.assertIs(lhs << Tag("rim"), lhs)
        got = srf >> Tag("rim")
        self.assertEqual(got.shape, (4, 2))
        np.testing.assert_array_equal(got, [[1, 2], [1, 3], [2, 2], [2, 3]])
        np.testing.assert_array_equal(srf.cv[1, 2] >> Tag("rim"), [[1, 2]])
        self.assertEqual((srf.cv[3, 0] >> Tag("rim")).shape, (0, 2))
        np.testing.assert_array_equal(srf.cv[1:2, :] >> Tag("rim"), [[1, 2], [1, 3]])
        self.assertEqual(_names(Tag.of(srf.cv[1, 2])), ["rim"])
        srf.cv[2, 2] << -Tag("rim")
        np.testing.assert_array_equal(srf >> Tag("rim"), [[1, 2], [1, 3], [2, 3]])
        # the flat controlPoints spelling writes (Maya normalises it) but
        # never indexes a query
        srf.controlPoints[[0, 1]] << Tag("flat")
        np.testing.assert_array_equal(srf >> Tag("flat"), [[0, 0], [0, 1]])
        before = set(cmds.ls())
        with self.assertRaisesRegex(TypeError, r"cv\[u, v\]"):
            srf.controlPoints[3] >> Tag("rim")
        self.assertEqual(set(cmds.ls()), before)
        # the bare handle is every DISTINCT cv: 56 on a periodic sphere
        srf.cv << Tag("all")
        self.assertEqual((srf >> Tag("all")).shape, (56, 2))
        self.assertEqual(_names(Tag.of(srf.cv[1, 2])), ["all", "rim"])
        Tag("rim").set(srf.cv[0, :2])
        np.testing.assert_array_equal(srf >> Tag("rim"), [[0, 0], [0, 1]])

    def test_lattice_round_trips_three_dimensional_ids(self):
        cube    = cmds.polyCube(name="lc", ch=False)[0]
        lattice = Node(cmds.lattice(cube, divisions=(2, 3, 4))[1])
        lhs     = lattice.pt[0, 0, :]
        self.assertIs(lhs << Tag("corner"), lhs)
        got = lattice >> Tag("corner")
        self.assertEqual(got.shape, (4, 3))
        np.testing.assert_array_equal(got, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
        np.testing.assert_array_equal(lattice.pt[0, 0, 1] >> Tag("corner"), [[0, 0, 1]])
        self.assertEqual((lattice.pt[1, 2, 3] >> Tag("corner")).shape, (0, 3))
        lattice.pt[0, 0, 0] << -Tag("corner")
        self.assertEqual((lattice >> Tag("corner")).shape, (3, 3))
        lattice.pt << Tag("allpt")
        self.assertEqual((lattice >> Tag("allpt")).shape, (24, 3))
        self.assertEqual(_names(Tag.of(lattice.pt[0, 0, 1])), ["allpt", "corner"])
        Components(lattice, "pt", [[1, 2, 3]]) << Tag("corner")
        self.assertEqual((lattice >> Tag("corner")).shape, (4, 3))


# --------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------- #


class TestTagExports(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_top_level_names(self):
        import rig
        from rig import membership

        self.assertIs(rig.Tag,        membership.Tag)
        self.assertIs(rig.Layer,      membership.Layer)
        self.assertIs(rig.Components, membership.Components)
        self.assertIs(rig.membership, membership)
        for name in ("Tag", "Layer", "Components", "membership"):
            self.assertIn(name, rig.__all__)
        self.assertEqual(sorted(membership.__all__), ["Components", "Layer", "Tag"])
        self.assertIsInstance(Tag("cap"), Tag)
        self.assertIsInstance(Node,       type)
        self.assertIsInstance(Plug,       type)
