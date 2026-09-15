import unittest

import numpy as np
from maya import cmds
from rig.maya.nodetypes import PyNode
from rig._tests._base import initialize_standalone, MayaTestCase

# start standalone so that cmds can be imported
initialize_standalone()

maya_major = int(cmds.about(majorVersion=True))
maya_minor = int(cmds.about(minorVersion=True))


@unittest.skipIf(
    (maya_major < 2022 or (maya_major == 2022 and maya_minor < 5)),
    "Skip maya versions without component tags or containt bugs.",
)
class TestComponentTags(MayaTestCase):
    """
    Test component tags functionalities (in Geometry class)
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()

        # make a sphere with some comp tags
        self.mesh        = PyNode(cmds.polySphere(ch=False)[0]).get_children(type="mesh")[0]
        self.tags        = ["tag1", "tag2"]
        self.ids         = [0, 1]
        self.cats        = ["v", "f"]
        self.contents    = [["vtx[2:3]"], ["f[4]", "f[6]"]]
        self.content_ids = [[2, 3], [4, 6]]
        for tag, ids, cat in zip(self.tags, self.content_ids, self.cats):
            self.mesh.add_component_tag(tag)
            self.mesh.set_component_tag_contents(tag, ids, category=cat)

    def test_query(self):
        self.assertEqual(self.mesh.component_tags, self.tags)
        for each in self.tags:
            self.assertTrue(self.mesh.has_component_tag(each))
        self.assertFalse(self.mesh.has_component_tag("aaa"))
        for i, (tag, cat) in enumerate(zip(self.tags, self.cats)):
            self.assertEqual(self.mesh.get_component_tag_category(tag), cat)
            self.assertEqual(self.mesh.get_component_tag_category(i), cat)
        for i, (tag, cont) in enumerate(zip(self.tags, self.contents)):
            self.assertEqual(self.mesh.get_component_tag_contents(tag), cont)
            self.assertEqual(self.mesh.get_component_tag_contents(i), cont)
        for tag, i in zip(self.tags, self.ids):
            self.assertEqual(self.mesh.get_component_tag_index(tag), i)
        self.assertEqual(self.mesh.get_component_tag_index("aaa"), -1)
        self.assertFalse(self.mesh.is_component_tag_empty(self.tags[0]))

    def test_edit(self):
        self.mesh.set_component_tag_contents(1, [3, 5, 6], category="e")
        self.assertEqual(
            self.mesh.get_component_tag_contents(self.tags[1]), ["e[3]", "e[5:6]"]
        )

        self.mesh.set_component_tag_contents(1, ["f[2]", "f[3]", "f[4]"])
        self.assertEqual(self.mesh.get_component_tag_contents(self.tags[1]), ["f[2:4]"])

        self.mesh.set_component_tag_contents(1, ["a.vtx[1]"])
        self.assertEqual(self.mesh.get_component_tag_contents(self.tags[1]), ["vtx[1]"])

        # Maya rejects single-character component tag names (and names starting
        # with a digit) with an opaque "Maya command error", so use two characters.
        self.mesh.rename_component_tag(self.tags[0], "ta")
        self.assertFalse(self.mesh.has_component_tag(self.tags[0]))
        self.assertTrue(self.mesh.has_component_tag("ta"))

        self.mesh.rename_component_tag(1, "tb")
        self.assertFalse(self.mesh.has_component_tag(self.tags[1]))
        self.assertTrue(self.mesh.has_component_tag("tb"))

        self.mesh.remove_component_tag("ta")
        self.assertFalse(self.mesh.has_component_tag("ta"))
        self.assertTrue(self.mesh.has_component_tag("tb"))
        self.mesh.remove_component_tag(1)
        self.assertFalse(self.mesh.has_component_tag("tb"))

    def test_serialize(self):
        org_data = self.mesh.get_component_tag_data(self.tags[0])
        self.assertTrue(np.allclose(org_data.indices, self.content_ids[0]))

        new_target = "new_target"
        i          = self.mesh.add_component_tag(new_target)
        self.assertTrue(self.mesh.is_component_tag_empty(2))
        self.assertEqual(self.mesh.component_tags, self.tags + [new_target])
        self.assertEqual(self.mesh.get_component_tag_index(new_target), 2)

        data = self.mesh.get_component_tag_data(new_target)
        self.assertEqual(data.indices.size, 0)

        self.mesh.set_component_tag_contents(i, org_data)
        data = self.mesh.get_component_tag_data(i)
        self.assertEqual(data, org_data)
        self.assertEqual(self.mesh.get_component_tag_contents(i), self.contents[0])

        self.assertEqual(len(self.mesh.serialize_component_tags()), 3)
        self.assertEqual(
            len(self.mesh.serialize_component_tags(match_name="new_.*")), 1
        )