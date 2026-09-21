import unittest

import numpy as np
from maya import cmds
from rig.maya.nodetypes import PyNode
from rig.maya.nodetypes.deformer import tag_references
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


@unittest.skipIf(
    (maya_major < 2022 or (maya_major == 2022 and maya_minor < 5)),
    "Skip maya versions without component tags or containt bugs.",
)
class TestComponentTagLocations(MayaTestCase):
    """
    Test component tags where Maya really stores them: on the injection node
    of a deformed mesh, on curves and surfaces, and on polyCubes with
    procedural (history) or baked (ch=False) tags; plus deformer references.
    """

    TEST_START_NEW_SCENE = True

    CUBE_TAGS = ["back", "bottom", "front", "left", "right", "top"]

    def test_injection_node_undeformed(self):
        mesh = PyNode(cmds.polySphere(ch=False)[0]).get_children(type="mesh")[0]
        self.assertEqual(mesh.injection_node, mesh)

    def test_clustered_mesh(self):
        xform = cmds.polySphere(ch=False)[0]
        mesh  = PyNode(xform).get_children(type="mesh")[0]
        cmds.cluster(xform)
        orig = cmds.deformableShape(mesh.name, tagInjectionNode=True)[0]
        self.assertNotEqual(orig, mesh.name)
        self.assertEqual(mesh.injection_node.name, orig)

        # the editor writes where Maya's own command writes: the Orig
        i = mesh.add_component_tag("rtag")
        self.assertEqual(cmds.getAttr(f"{orig}.componentTags", multiIndices=True), [i])
        self.assertIsNone(cmds.getAttr(f"{mesh.name}.componentTags", multiIndices=True))
        mesh.set_component_tag_contents("rtag", [1, 2])
        self.assertEqual(mesh.get_component_tag_contents("rtag"), ["vtx[1:2]"])

        # and the editors see what Maya's own command wrote there
        cmds.componentTag(f"{mesh.name}.vtx[5]", create=True, newTagName="mtag")
        self.assertTrue(mesh.has_component_tag("mtag"))
        self.assertNotEqual(mesh.get_component_tag_index("mtag"), -1)
        self.assertEqual(mesh.component_tags, ["rtag", "mtag"])
        mesh.set_component_tag_contents("mtag", [7, 8, 9])
        self.assertEqual(mesh.get_component_tag_contents("mtag"), ["vtx[7:9]"])
        self.assertTrue(np.array_equal(mesh.get_component_tag_indices("mtag"), [7, 8, 9]))
        mesh.rename_component_tag("mtag", "ntag")
        self.assertTrue(mesh.has_component_tag("ntag"))
        mesh.remove_component_tag("ntag")
        self.assertFalse(mesh.has_component_tag("ntag"))
        self.assertEqual(mesh.component_tags, ["rtag"])

    def test_nurbs_curve(self):
        crv   = cmds.curve(point=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])
        curve = PyNode(crv).get_children(type="nurbsCurve")[0]
        curve.add_component_tag("root")
        curve.set_component_tag_contents("root", [0, 1, 2])
        # the native token: a 'vtx[...]' tag on a curve resolves to nothing
        self.assertEqual(curve.get_component_tag_contents("root"), ["cv[0:2]"])
        self.assertEqual(curve.get_component_tag_category("root"), "v")
        self.assertTrue(np.array_equal(curve.get_component_tag_indices("root"), [0, 1, 2]))

        data = curve.get_component_tag_data("root")
        self.assertTrue(np.array_equal(data.indices, [0, 1, 2]))
        self.assertEqual(data.component_type, "v")

        curve.set_component_tag_contents("root", [])
        self.assertTrue(curve.is_component_tag_empty("root"))
        self.assertEqual(curve.get_component_tag_indices("root").shape, (0,))
        curve.set_component_tag_contents("root", data)
        self.assertEqual(curve.get_component_tag_contents("root"), ["cv[0:2]"])

    def test_periodic_nurbs_surface(self):
        surface = PyNode(cmds.sphere()[0]).get_children(type="nurbsSurface")[0]
        # the whole read path raised AttributeError on a nurbsSurface
        self.assertFalse(surface.has_component_tag("rim"))
        surface.add_component_tag("rim")
        self.assertTrue(surface.has_component_tag("rim"))
        self.assertTrue(surface.is_component_tag_empty("rim"))
        self.assertEqual(surface.get_component_tag_indices("rim").shape, (0, 2))

        # 2-D coordinates round trip in the cv[u][v] form
        coords = np.array([[1, 0], [1, 1], [2, 5]])
        surface.set_component_tag_contents("rim", coords)
        self.assertEqual(
            surface.get_component_tag_contents("rim"), ["cv[1][0:1]", "cv[2][5]"]
        )
        self.assertTrue(np.array_equal(surface.get_component_tag_indices("rim"), coords))
        data = surface.get_component_tag_data("rim")
        self.assertTrue(np.array_equal(data.indices, coords))
        self.assertEqual(data.component_type, "v")

        surface.set_component_tag_contents("rim", ["cv[0][0:2]"])
        self.assertTrue(
            np.array_equal(
                surface.get_component_tag_indices("rim"), [[0, 0], [0, 1], [0, 2]]
            )
        )
        surface.set_component_tag_contents("rim", data)
        self.assertTrue(np.array_equal(surface.get_component_tag_indices("rim"), coords))
        self.assertEqual(surface.serialize_component_tags(), [data])

    def test_polycube_with_history_is_procedural(self):
        mesh = PyNode(cmds.polyCube()[0]).get_children(type="mesh")[0]
        self.assertTrue(mesh.has_component_tag("top"))
        self.assertEqual(mesh.get_component_tag_index("top"), -1)
        self.assertEqual(mesh.component_tags, self.CUBE_TAGS)
        self.assertEqual(mesh.get_component_tag_contents("top"), ["f[1]"])
        self.assertTrue(np.array_equal(mesh.get_component_tag_indices("top"), [1]))
        self.assertEqual(mesh.get_component_tag_data("top").component_type, "f")
        self.assertEqual(len(mesh.serialize_component_tags()), 6)

        # procedural tags refuse every edit loudly and stay intact
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            mesh.set_component_tag_contents("top", [0, 1], category="f")
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            mesh.remove_component_tag("top")
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            mesh.rename_component_tag("top", "cap")
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            mesh.add_component_tag("top")
        self.assertEqual(mesh.get_component_tag_contents("top"), ["f[1]"])
        self.assertIsNone(cmds.getAttr(f"{mesh.name}.componentTags", multiIndices=True))

        # a new name is editable next to them
        mesh.add_component_tag("cap")
        mesh.set_component_tag_contents("cap", [0, 1], category="f")
        self.assertEqual(mesh.component_tags, ["cap"] + self.CUBE_TAGS)
        self.assertEqual(mesh.get_component_tag_contents("cap"), ["f[0:1]"])

    def test_polycube_baked_tags_are_editable(self):
        mesh = PyNode(cmds.polyCube(ch=False)[0]).get_children(type="mesh")[0]
        self.assertEqual(mesh.component_tags, self.CUBE_TAGS)
        self.assertEqual(mesh.get_component_tag_index("top"), 5)
        self.assertTrue(mesh.has_component_tag(5))

        # baked tags live in the shape's own multi: the plug path edits them
        mesh.set_component_tag_contents("top", [0, 1], category="f")
        self.assertEqual(mesh.get_component_tag_contents("top"), ["f[0:1]"])
        mesh.rename_component_tag("top", "cap")
        self.assertFalse(mesh.has_component_tag("top"))
        self.assertTrue(np.array_equal(mesh.get_component_tag_indices("cap"), [0, 1]))
        mesh.remove_component_tag("cap")
        self.assertFalse(mesh.has_component_tag("cap"))
        self.assertEqual(mesh.component_tags, self.CUBE_TAGS[:-1])

    def test_tag_references(self):
        cluster = cmds.cluster(cmds.polySphere(ch=False)[0])[0]
        expr    = f"{cluster}.input[0].componentTagExpression"
        # the bare star is not a reference to any name
        self.assertEqual(tag_references("cap"), [])

        cmds.setAttr(expr, "cap + lid", type="string")
        self.assertEqual(tag_references("cap"),    [(cluster, 0)])
        self.assertEqual(tag_references("lid"),    [(cluster, 0)])
        self.assertEqual(tag_references("ca"),     [])
        self.assertEqual(tag_references("caps"),   [])
        self.assertEqual(tag_references("ns:cap"), [])

        cmds.setAttr(expr, "!cap", type="string")
        self.assertEqual(tag_references("cap"), [(cluster, 0)])
        cmds.setAttr(expr, "ns:cap", type="string")
        self.assertEqual(tag_references("cap"), [])
        self.assertEqual(tag_references("ns:cap"), [(cluster, 0)])

        # a glob that could match counts; the bare star still does not
        cmds.setAttr(expr, "ca* - *", type="string")
        self.assertEqual(tag_references("cap"), [(cluster, 0)])
        self.assertEqual(tag_references("lid"), [])
    def test_transfer_component_tags_lands_on_the_target_injection_node(self):
        try:
            from cgmath.geometry.resample import MeshDataResampler  # noqa: F401
        except ImportError:
            self.skipTest("cgmath.geometry.resample unavailable")
        src = PyNode(cmds.polySphere(ch=False, name="srcS")[0]).get_children(type="mesh")[0]
        dst = PyNode(cmds.polySphere(ch=False, name="dstS")[0]).get_children(type="mesh")[0]
        src.add_component_tag("cap")
        src.set_component_tag_contents("cap", [0, 1, 2, 3], "v")
        cmds.cluster(dst.name)                       # dst now injects on its Orig
        self.assertNotEqual(dst.injection_node.name, dst.name)
        src.transfer_component_tags("cap", dst)
        # written where Maya reads it, so the visible shape resolves it
        self.assertTrue(dst.has_component_tag("cap"))
        self.assertGreaterEqual(dst.get_component_tag_index("cap"), 0)
        self.assertGreater(dst.get_component_tag_indices("cap").size, 0)
        # a colliding procedural name on a history polyCube refuses instead of no-op'ing
        cube = PyNode(cmds.polyCube(ch=True, name="hcube")[0]).get_children(type="mesh")[0]
        src.add_component_tag("top")
        src.set_component_tag_contents("top", [0, 1], "v")
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            src.transfer_component_tags("top", cube)
