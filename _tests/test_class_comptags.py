import re
import unittest

import numpy as np
from maya import cmds
from rig.nodetypes import Node
from rig.nodetypes.deformer import tag_references
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
        self.mesh        = Node(cmds.polySphere(ch=False)[0]).get_children(type="mesh")[0]
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
class TestComponentTagNames(MayaTestCase):
    """``add_component_tag`` refuses, before anything is made, the names Maya
    refuses with a bare "Maya command error", and only those (runs/r5/SKIN)."""

    TEST_START_NEW_SCENE = True

    REFUSED = ["b", "h", "x", "B", "_", "1", "", "  ", " a ", "1a", "a-b", "a..b", "a.:b", "a.", ":a", "a\tb", "a*", "f[0]"]
    TAKEN   = {"ab": "ab", "_1": "_1", "a.b": "a.b", "a:b": "a:b", "a.1": "a.1", " cd\t": "cd", "a b": "a_b", "c \td": "c_d"}

    def setUp(self):
        super().setUp()
        self.mesh = Node(cmds.polyCube(ch=False)[0]).get_shape()

    def tag_multi(self):
        return cmds.getAttr(f"{self.mesh}.componentTags", multiIndices=True)

    def test_add_component_tag_refuses_a_name_maya_refuses_before_anything_is_made(self):
        before = self.tag_multi()
        for name in self.REFUSED:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "not a component tag name: Maya takes .* at least two characters"):
                    self.mesh.add_component_tag(name)
                self.assertEqual(self.tag_multi(), before)

    def test_add_component_tag_takes_the_names_maya_takes(self):
        for name, stored in self.TAKEN.items():
            with self.subTest(name=name):
                i = self.mesh.add_component_tag(name)
                self.assertEqual(self.mesh.get_component_tag_name(i), stored)
                self.assertIn(stored, self.mesh.component_tags)

    def test_add_component_tag_finds_a_tag_by_the_name_maya_stores(self):
        """' ab' is stored 'ab': it was looked up as ' ab', so a second 'ab' was
        made, and force=True left the first one (runs/r5/SKIN/review)."""
        self.mesh.add_component_tag("ab")
        for name in (" ab", "ab\t"):
            with self.subTest(name=name):
                before = self.tag_multi()
                with self.assertRaisesRegex(RuntimeError, "Component tag 'ab' already exists"):
                    self.mesh.add_component_tag(name)
                self.assertEqual(self.tag_multi(), before)

                i = self.mesh.add_component_tag(name, force=True)
                self.assertEqual(self.mesh.get_component_tag_name(i), "ab")
                self.assertEqual(self.mesh.component_tags.count("ab"), 1)

    def test_rename_component_tag_refuses_a_name_maya_refuses(self):
        """It failed with a bare "Maya command error" (runs/r5/SKIN/review)."""
        i = self.mesh.add_component_tag("ab")
        for name in ("b", "1a", "a-b"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"{name!r} is not a component tag name: Maya takes"):
                    self.mesh.rename_component_tag("ab", name)
                self.assertEqual(self.mesh.get_component_tag_name(i), "ab")
        self.mesh.rename_component_tag("ab", "c d")
        self.assertEqual(self.mesh.get_component_tag_name(i), "c_d")

    def test_a_name_that_is_not_a_str_is_refused_before_anything_else(self):
        """add_component_tag(5) said "Component tag '5' already exists": the int
        was looked up as the index of the cube's sixth tag, and force=True removed
        that tag (runs/r5/REBIND)."""
        names  = list(self.mesh.component_tags)
        before = self.tag_multi()
        for key in (5, 0, 1.5, None, b"ab", ["ab"]):
            for force in (False, True):
                with self.subTest(key=key, force=force):
                    with self.assertRaisesRegex(TypeError, f"^{re.escape(repr(key))} is not a component tag name: .*a str"):
                        self.mesh.add_component_tag(key, force=force)
                    self.assertEqual(self.tag_multi(), before)
                    self.assertEqual(self.mesh.component_tags, names)
            with self.subTest(key=key, rename=True):
                with self.assertRaisesRegex(TypeError, "is not a component tag name: .*a str"):
                    self.mesh.rename_component_tag("top", key)
                self.assertEqual(self.mesh.component_tags, names)

    def test_the_name_rule_is_mayas(self):
        """Every name of one to three characters from this alphabet: rig refuses
        exactly what Maya's setAttr refuses."""
        from itertools import product
        from rig.nodetypes.geometry import _check_component_tag_name

        plug  = f"{self.mesh}.componentTags[100].componentTagName"
        wrong = []
        for size in (1, 2, 3):
            for name in map("".join, product("aB_1.: -\t\n*", repeat=size)):
                try:
                    cmds.setAttr(plug, name, type="string")
                    maya_takes = True
                except RuntimeError:
                    maya_takes = False
                try:
                    _check_component_tag_name(name)
                    rig_takes = True
                except ValueError:
                    rig_takes = False
                if maya_takes != rig_takes:
                    wrong.append((name, maya_takes))
        self.assertEqual(wrong, [])


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
        mesh = Node(cmds.polySphere(ch=False)[0]).get_children(type="mesh")[0]
        self.assertEqual(mesh.injection_node, mesh)

    def test_clustered_mesh(self):
        xform = cmds.polySphere(ch=False)[0]
        mesh  = Node(xform).get_children(type="mesh")[0]
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
        curve = Node(crv).get_children(type="nurbsCurve")[0]
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
        surface = Node(cmds.sphere()[0]).get_children(type="nurbsSurface")[0]
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
        mesh = Node(cmds.polyCube()[0]).get_children(type="mesh")[0]
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
        mesh = Node(cmds.polyCube(ch=False)[0]).get_children(type="mesh")[0]
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
        src = Node(cmds.polySphere(ch=False, name="srcS")[0]).get_children(type="mesh")[0]
        dst = Node(cmds.polySphere(ch=False, name="dstS")[0]).get_children(type="mesh")[0]
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
        cube = Node(cmds.polyCube(ch=True, name="hcube")[0]).get_children(type="mesh")[0]
        src.add_component_tag("top")
        src.set_component_tag_contents("top", [0, 1], "v")
        with self.assertRaisesRegex(RuntimeError, "procedural"):
            src.transfer_component_tags("top", cube)

    def test_transfer_component_tags_carries_the_tag_contents(self):
        """The tag's own vertices, not every vertex (cgmath's resample_map
        normalizes the one-influence data to the whole mesh)."""
        try:
            from cgmath.geometry.resample import MeshDataResampler  # noqa: F401
        except ImportError:
            self.skipTest("cgmath.geometry.resample unavailable")
        src = Node(cmds.polyPlane(ch=False, name="srcP", sx=4, sy=4)[0]).get_children(type="mesh")[0]
        dst = Node(cmds.polyPlane(ch=False, name="dstP", sx=4, sy=4)[0]).get_children(type="mesh")[0]
        src.add_component_tag("xfer_pts")
        src.set_component_tag_contents("xfer_pts", [0, 1, 2, 7], "v")
        src.transfer_component_tags("xfer_pts", dst)
        self.assertEqual(dst.get_component_tag_indices("xfer_pts").tolist(), [0, 1, 2, 7])

    def test_transfer_component_tags_face_and_edge_tags_are_not_the_whole_mesh(self):
        """A face or edge tag comes back holding its own components, not every face
        or edge (runs/r5/XFER/review/probe_maps.py; exactly, since TAGS)."""
        try:
            from cgmath.geometry.resample import MeshDataResampler  # noqa: F401
        except ImportError:
            self.skipTest("cgmath.geometry.resample unavailable")
        src = Node(cmds.polyPlane(ch=False, name="srcP", sx=4, sy=4)[0]).get_children(type="mesh")[0]
        dst = Node(cmds.polyPlane(ch=False, name="dstP", sx=4, sy=4)[0]).get_children(type="mesh")[0]
        for kind, indices, count in (("f", [5], 16), ("e", [7, 20], 40)):
            with self.subTest(kind):
                tag = f"xfer_{kind}"
                src.add_component_tag(tag)
                src.set_component_tag_contents(tag, indices, kind)
                src.transfer_component_tags(tag, dst)
                got = set(dst.get_component_tag_indices(tag).tolist())
                self.assertLessEqual(set(indices), got)
                self.assertLess(len(got), count, sorted(got))

    # --- where a transferred tag lands (TAGS: the user's rule of 2026-10-15)

    def _planes(self, src_divisions=4, dst_divisions=4, **dst_kwargs):
        try:
            from cgmath.geometry.resample import MeshDataResampler  # noqa: F401
        except ImportError:
            self.skipTest("cgmath.geometry.resample unavailable")
        src = Node(cmds.polyPlane(ch=False, name="srcP", sx=src_divisions, sy=src_divisions)[0]).get_shape()
        dst = Node(cmds.polyPlane(ch=False, name="dstP", sx=dst_divisions, sy=dst_divisions, **dst_kwargs)[0]).get_shape()
        return src, dst

    @staticmethod
    def _edge_ends(mesh):
        """Maya's own edge table: {edge id: (vertex, vertex)}."""
        from maya.api import OpenMaya
        fn = OpenMaya.MFnMesh(mesh.mdagpath)
        return {e: tuple(sorted(fn.getEdgeVertices(e))) for e in range(fn.numEdges)}

    def _transfer(self, src, dst, tag, ids, kind, **kwargs):
        if not src.has_component_tag(tag):
            src.add_component_tag(tag)
        src.set_component_tag_contents(tag, ids, kind)
        src.transfer_component_tags(tag, dst, **kwargs)
        return dst.get_component_tag_indices(tag).tolist()

    def test_transfer_component_tags_comes_back_exactly_onto_the_same_topology(self):
        """Every face, edge and vertex alone, and sets of them, come back as they
        were, by SPATIAL and by UV (the face and edge tags used to grow a ring, and
        the edges came back in cgmath's numbering, not Maya's)."""
        from cgmath.geometry.resample import ResampleMode

        try:
            from cgmath.geometry.resample import MeshDataResampler  # noqa: F401
        except ImportError:
            self.skipTest("cgmath.geometry.resample unavailable")
        for make in ("sphere", "torus"):
            src = Node(getattr(cmds, "poly" + make.title())(ch=False, name="s" + make)[0]).get_shape()
            dst = Node(getattr(cmds, "poly" + make.title())(ch=False, name="d" + make)[0]).get_shape()
            counts = {"f": cmds.polyEvaluate(src.name, face=True), "e": len(self._edge_ends(src)), "v": src.num_vertices}
            for mode in (ResampleMode.SPATIAL, ResampleMode.UV):
                for kind, count in counts.items():
                    with self.subTest(make, mode=mode.name, kind=kind):
                        for i in range(0, count, 7):
                            self.assertEqual(self._transfer(src, dst, "t" + kind, [i], kind, mode=mode), [i])
                        # several faces / vertices at once (edges: see the plane row below)
                        if kind != "e":
                            ids = list(range(0, count, 3))
                            self.assertEqual(self._transfer(src, dst, "t" + kind, ids, kind, mode=mode), ids)

        # an edge loop: a whole row of the plane, no rung between its vertices
        src, dst = self._planes()
        ends = self._edge_ends(src)
        row = sorted(e for e, (a, b) in ends.items() if (a, b) in {(5 + c, 6 + c) for c in range(4)})
        self.assertEqual(len(row), 4)
        self.assertEqual(self._transfer(src, dst, "row", row, "e"), row)

    def test_transfer_component_tags_by_uv_lands_where_the_uvs_are(self):
        """A face tag by UV follows the UVs, wherever the target is in space; by
        SPATIAL onto a far-away target it lands on the source face closest to each
        target face centre."""
        from cgmath.geometry.resample import ResampleMode

        src, dst = self._planes()
        cmds.move(30, 0, 0, dst.get_parent().name)
        self.assertEqual(self._transfer(src, dst, "uvf", [5, 6], "f", mode=ResampleMode.UV), [5, 6])
        # far along +x, every target face centre is closest to the source's +x column
        self.assertEqual(self._transfer(src, dst, "spf", [3, 7, 11, 15], "f"), list(range(16)))
        self.assertEqual(self._transfer(src, dst, "spf", [5, 6], "f"), [])

    def test_transfer_component_tags_faces_land_by_their_centres_onto_a_finer_mesh(self):
        """Onto a plane of twice the divisions, a face tag is the faces whose
        centres are on the tagged faces: four per face, not a ring more."""
        from cgmath.geometry.resample import ResampleMode
        from maya.api import OpenMaya

        src, dst = self._planes(4, 8)
        fn = OpenMaya.MFnMesh(dst.mdagpath)
        pts = fn.getPoints(OpenMaya.MSpace.kWorld)

        def centre(f):
            ids = fn.getPolygonVertices(f)
            return (sum(pts[i].x for i in ids) / len(ids), sum(pts[i].z for i in ids) / len(ids))

        tagged = [5, 6, 9]
        for mode in (ResampleMode.SPATIAL, ResampleMode.UV):
            with self.subTest(mode=mode.name):
                got = self._transfer(src, dst, "fine_f", tagged, "f", mode=mode)
                self.assertEqual(len(got), 4 * len(tagged), got)
                # each landed face's centre is inside one tagged source face
                for f in got:
                    p = OpenMaya.MPoint(centre(f)[0], 0, centre(f)[1])
                    closest = OpenMaya.MFnMesh(src.mdagpath).getClosestPoint(p, OpenMaya.MSpace.kWorld)[1]
                    self.assertIn(closest, tagged, f)

    @staticmethod
    def _create(name, points, faces):
        """A mesh of `faces` on `points`, its UVs the points' (x, z) / 4."""
        from maya.api import OpenMaya

        fn     = OpenMaya.MFnMesh()
        counts = [len(f) for f in faces]
        ids    = [i for f in faces for i in f]
        node   = fn.create([OpenMaya.MPoint(*p) for p in points], counts, ids)
        fn.setUVs([p[0] / 4.0 for p in points], [p[2] / 4.0 for p in points])
        fn.assignUVs(counts, ids)
        return Node(cmds.rename(OpenMaya.MFnDependencyNode(node).name(), name)).get_shape()

    def test_transfer_component_tags_concave_quad_comes_back_exactly(self):
        """A flat dart quad and the triangle under its corner mean, in either face
        order: the dart lands by its first triangle's centre, so each face comes
        back alone (by its corner mean the dart landed on the triangle when the
        triangle came first). Lifting the dart's reflex corner off the plane keeps
        its corner mean, the middle of the bilinear patch cgmath samples."""
        from cgmath.geometry.resample import ResampleMode

        for lift in (0.0, 0.5):
            points = [(0, 0, 0), (2, lift, 2), (4, 0, 0), (2, 0, 3)]
            for order in ([[0, 3, 2, 1], [0, 1, 2]], [[0, 1, 2], [0, 3, 2, 1]]):
                src = self._create("dart_src", points, order)
                dst = self._create("dart_dst", points, order)
                for mode in (ResampleMode.SPATIAL, ResampleMode.UV):
                    for f in (0, 1):
                        with self.subTest(lift=lift, order=order, mode=mode.name, face=f):
                            self.assertEqual(self._transfer(src, dst, "dart", [f], "f", mode=mode), [f])
                cmds.delete(src.get_parent().name, dst.get_parent().name)

    def test_transfer_component_tags_bent_concave_quads_keep_their_centre(self):
        """A sphere with jittered points (seeded): faces 806, 807, 835 and 837 sit
        by concave quads that are not flat, whose first triangle's centre is off
        the bilinear patch cgmath samples. They came back wrong when every concave
        face used its triangle; by their corner mean, the patch's middle, they
        come back alone."""
        from maya.api import OpenMaya

        shapes = []
        for name in ("bent_src", "bent_dst"):
            shape = Node(cmds.polySphere(ch=False, name=name, sx=30, sy=30)[0]).get_shape()
            fn    = OpenMaya.MFnMesh(shape.mdagpath)
            pts   = np.array([[p.x, p.y, p.z] for p in fn.getPoints()])
            pts  += np.random.default_rng(3).normal(scale=0.03, size=pts.shape)
            fn.setPoints([OpenMaya.MPoint(*p) for p in pts])
            shapes.append(shape)
        src, dst = shapes
        for f in [806, 807, 835, 837] + list(range(0, 900, 37)):
            with self.subTest(face=f):
                self.assertEqual(self._transfer(src, dst, "bent", [f], "f"), [f])

    def test_transfer_component_tags_holed_face_lands_by_a_point_on_it(self):
        """A 3x3 face with a 1x1 hole in the middle, under a 3x3 plane: its corner
        mean is the hole's centre, so it used to be in by the face under the hole
        and out by the ring of faces it covers."""
        from maya.api import OpenMaya

        src = Node(cmds.polyPlane(ch=False, name="ring_src", sx=3, sy=3, w=3, h=3)[0]).get_shape()
        dst = Node(cmds.polyPlane(ch=False, name="ring_dst", sx=1, sy=1, w=3, h=3)[0]).get_shape()
        hole = [OpenMaya.MPoint(x, 0, z) for x, z in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
        OpenMaya.MFnMesh(dst.mdagpath).addHoles(0, hole, [4])
        self.assertEqual(len(OpenMaya.MFnMesh(dst.mdagpath).getHoles()), 1)
        self.assertEqual(self._transfer(src, dst, "ring", [0, 1, 2, 3, 5, 6, 7, 8], "f"), [0])
        self.assertEqual(self._transfer(src, dst, "ring", [4], "f"), [])

    def test_transfer_component_tags_edge_joining_two_tagged_loops_is_in(self):
        """An edge is in when both its ends are: tagging two parallel rows of a
        plane also tags the rungs between them (the user's accepted case), and
        nothing else."""
        src, dst = self._planes()
        ends = self._edge_ends(src)
        rows = {(r * 5 + c, r * 5 + c + 1) for r in (1, 2) for c in range(4)}
        tagged = sorted(e for e, pair in ends.items() if pair in rows)
        verts = {v for e in tagged for v in ends[e]}
        expected = sorted(e for e, (a, b) in self._edge_ends(dst).items() if a in verts and b in verts)
        self.assertEqual(len(tagged), 8)
        self.assertEqual(len(expected), 8 + 5)  # the two rows and their five rungs
        self.assertEqual(self._transfer(src, dst, "rows", tagged, "e"), expected)
