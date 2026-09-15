"""Tests for ``rig._node`` -- the Node wrapper + lift()."""

from maya import cmds
from rig import lift, Node, Plug
from rig.spec import Float
from rig._tests._base import MayaTestCase


class TestNodeConstruction(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_construct_from_string(self):
        cmds.createNode("transform", name="cube1")
        node = Node("cube1")
        self.assertEqual(str(node), "cube1")

    def test_construct_from_existing_rignode(self):
        node1 = Node.create("transform", name="cube1")
        node2 = Node(node1)
        self.assertEqual(str(node1), str(node2))

    def test_construct_from_attribute_string_strips_attr(self):
        # v3.O: ``Node("cube1.translateX")`` strips the attribute and
        # returns a Node for the owning DG node -- symmetric with the
        # original DSL's ``node._`` idiom for going from a plug back to
        # the bare node.
        cmds.createNode("transform", name="cube1")
        node = Node("cube1.translateX")
        self.assertIsInstance(node, Node)
        self.assertEqual(str(node), "cube1")

    def test_construct_from_translate_compound_string(self):
        # Same behavior with a compound attribute name.
        cmds.createNode("transform", name="cube1")
        node = Node("cube1.translate")
        self.assertEqual(str(node), "cube1")

    def test_construct_from_indexed_component_string(self):
        # v3.O: indexed component strings are also accepted by the strip.
        cube  = cmds.polyCube(name="poly1")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(f"{shape}.vtx[0]")
        self.assertEqual(str(node), shape)

    def test_construct_from_plug_strips_to_node(self):
        # v3.O: ``Node(plug)`` returns the owning bare Node.
        cmds.createNode("transform", name="cube1")
        plug = Plug("cube1.translateX")
        node = Node(plug)
        self.assertIsInstance(node, Node)
        self.assertEqual(str(node), "cube1")

    def test_create_classmethod(self):
        node = Node.create("transform", name="cube1")
        self.assertIsInstance(node, Node)
        self.assertTrue(cmds.objExists(str(node)))


class TestNodeAttributeAccess(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_native_attr_returns_plug(self):
        node = Node.create("transform", name="cube1")
        plug = node.translateX
        self.assertIsInstance(plug, Plug)

    def test_dynamic_attr_after_addAttr(self):
        node = Node.create("transform", name="cube1")
        node << Float("custom")
        plug = node.custom
        self.assertIsInstance(plug, Plug)

    def test_method_passthrough(self):
        # DGNode methods (like list_attr) still work.
        node  = Node.create("transform", name="cube1")
        attrs = node.list_attr()
        self.assertGreater(len(attrs), 0)


class TestNodeAssignmentSugar(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_setattr_sugar(self):
        node    = Node.create("transform", name="cube1")
        node.tx = 4.2
        self.assertAlmostEqual(cmds.getAttr("cube1.tx"), 4.2)

    def test_setattr_underscore_bypass(self):
        # Internal state (_*) should bypass the inject sugar.
        node = Node.create("transform", name="cube1")
        # Verify by reading back internal state directly.
        node._dg_node  # accessing existing _dg_node should not raise
        self.assertIsNotNone(node._dg_node)


class TestNodeInjectSpec(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_inject_spec_returns_plug(self):
        node   = Node.create("transform", name="cube1")
        result = node << Float("blend")
        self.assertIsInstance(result, Plug)
        self.assertEqual(result.alias, "blend")

    def test_inject_unsupported_value_raises(self):
        node = Node.create("transform", name="cube1")
        # Bare scalar is not a spec and not a matrix -- should raise.
        with self.assertRaises(TypeError):
            node << 5


class TestNodeInjectMatrix(MayaTestCase):
    """``node << matrix`` applies the matrix to the transform's channels.

    Plug-typed matrix sources go through ``_matrix_to_transform`` shorthand
    (live decomposeMatrix). Static numpy / nested-list sources go through
    ``_try_matrix_source_routing`` (Python-side decomposition + setAttr).
    """

    TEST_START_NEW_SCENE = True

    def test_node_lshift_matrix_plug_inserts_decompose(self):
        """Regression: ``node1 << node2.matrix`` must decompose, not raise."""
        a = Node.create("transform", name="a_xform")
        b = Node.create("transform", name="b_xform")
        b << a.matrix
        # b.t / b.r / b.s should each have a decomposeMatrix in the
        # incoming chain.
        for channel in ("t", "r", "s"):
            connections = (
                cmds.listConnections(
                    f"b_xform.{channel}", source=True, destination=False
                )
                or []
            )
            types = [cmds.nodeType(c) for c in connections]
            self.assertIn(
                "decomposeMatrix",
                types,
                msg=f"b_xform.{channel} missing decomposeMatrix in driver chain",
            )

    def test_node_lshift_static_4x4_decomposes(self):
        import numpy as np

        cube     = Node.create("transform", name="cube_xform")
        m        = np.eye(4)
        m[3, :3] = [7.0, 8.0, 9.0]
        cube << m
        self.assertAlmostEqual(cmds.getAttr("cube_xform.tx"), 7.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube_xform.ty"), 8.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube_xform.tz"), 9.0, places=4)

    def test_node_lshift_static_3x3_preserves_translation(self):
        import numpy as np

        cube = Node.create("transform", name="cube_xform")
        cube.t << [10, 20, 30]
        cube   << np.eye(3) * 2.0
        # Translation preserved (3x3 has no translation row).
        self.assertAlmostEqual(cmds.getAttr("cube_xform.tx"), 10.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube_xform.ty"), 20.0, places=4)
        self.assertAlmostEqual(cmds.getAttr("cube_xform.tz"), 30.0, places=4)
        # Scale set.
        self.assertAlmostEqual(cmds.getAttr("cube_xform.sx"), 2.0, places=4)


class TestNodeIdentity(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_str_repr_hash_eq(self):
        node = Node.create("transform", name="cube1")
        self.assertEqual(str(node), "cube1")
        self.assertIn("cube1", repr(node))
        # Two Nodes wrapping the same DGNode are equal.
        node2 = Node("cube1")
        self.assertEqual(node, node2)
        self.assertEqual(hash(node), hash(node2))

    def test_fspath(self):
        node = Node.create("transform", name="cube1")
        self.assertEqual(node.__fspath__(), "cube1")


class TestNodeRshift(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_rshift_none_returns_dgnode(self):
        # `Node >> None` returns the underlying typed DGNode instance
        # (e.g. Transform), kicking the user out of the rig DSL into
        # the rig.maya.nodetypes typed-node world.
        from rig.maya.nodetypes.dg_node import DGNode

        node   = Node.create("transform", name="cube1")
        result = node >> None
        self.assertIsInstance(result, DGNode)
        # Should NOT be a Node wrapper -- should be the typed DGNode subclass.
        self.assertNotIsInstance(result, Node)
        self.assertEqual(str(result), "cube1")

    def test_rshift_other_raises(self):
        node = Node.create("transform", name="cube1")
        # Non-None RHS still raises (only `>> None` is supported on Nodes).
        with self.assertRaises(TypeError):
            _ = node >> 5


class TestLift(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_lift_string_with_dot_returns_plug(self):
        cmds.createNode("transform", name="cube1")
        result = lift("cube1.translateX")
        self.assertIsInstance(result, Plug)

    def test_lift_string_no_dot_returns_rignode(self):
        cmds.createNode("transform", name="cube1")
        result = lift("cube1")
        self.assertIsInstance(result, Node)

    def test_lift_passes_through_plug(self):
        node   = Node.create("transform", name="cube1")
        plug   = node.translateX
        result = lift(plug)
        self.assertIs(result, plug)

    def test_lift_passes_through_rignode(self):
        node   = Node.create("transform", name="cube1")
        result = lift(node)
        self.assertIs(result, node)

    def test_lift_rejects_unsupported(self):
        with self.assertRaises(TypeError):
            lift(42)


class TestNodeComponentAliasResolution(MayaTestCase):
    """Regression: ``DGNode.find_attr`` resolves Maya component aliases
    (``vtx``, ``cv``, ``pt``, ``uv``, ``map``) to their underlying plug names
    (``controlPoints``, ``uvpt``).

    These aliases are recognised by ``cmds`` and ``MSelectionList``
    parsing but NOT by ``MFnDependencyNode.findPlug``, so the rig DSL
    needs the fallback alias map in ``dg_node._COMPONENT_ALIASES``.
    """

    TEST_START_NEW_SCENE = True

    def test_mesh_vtx_resolves_to_controlpoints(self):
        cube  = cmds.polyCube(name="cube_for_vtx")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        plug  = node.vtx
        # Should resolve to controlPoints, not raise.
        self.assertIsInstance(plug, Plug)
        self.assertIn("controlPoints", str(plug))

    def test_mesh_vtx_indexed(self):
        cube  = cmds.polyCube(name="cube_for_vtx_idx")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        plug  = node.vtx[0]
        self.assertIsInstance(plug, Plug)
        self.assertIn("controlPoints[0]", str(plug))

    def test_curve_cv_resolves_to_controlpoints(self):
        curve = cmds.curve(name="curve_for_cv", p=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        shape = cmds.listRelatives(curve, shapes=True)[0]
        node  = Node(shape)
        plug  = node.cv[0]
        self.assertIsInstance(plug, Plug)
        self.assertIn("controlPoints[0]", str(plug))

    def test_lattice_pt_resolves_to_controlpoints(self):
        cmds.polyCube(name="cube_for_lattice")
        lattice_result = cmds.lattice(
            "cube_for_lattice"
        )  # returns [ffd, lattice, base]
        lattice_shape = cmds.listRelatives(lattice_result[1], shapes=True)[0]
        node          = Node(lattice_shape)
        plug          = node.pt
        self.assertIsInstance(plug, Plug)
        self.assertIn("controlPoints", str(plug))

    def test_mesh_map_resolves_to_uvpt(self):
        cube  = cmds.polyCube(name="cube_for_map")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        plug  = node.map
        self.assertIsInstance(plug, Plug)
        self.assertIn("uvpt", str(plug))

    def test_mesh_uv_resolves_to_uvpt(self):
        # ``uv`` is the alternative spelling of ``map`` and must land on the
        # same canonical plug.
        cube  = cmds.polyCube(name="cube_for_uv")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        plug  = node.uv
        self.assertIsInstance(plug, Plug)
        self.assertIn("uvpt", str(plug))

    # -- cross-path consistency: node.<alias> must match Plug("mesh.<alias>")
    #    so the DSL resolves a component to the SAME canonical plug whether you
    #    go through Node attribute access (find_attr) or Plug construction
    #    (_maybe_translate_component). Maya 2026 changed cmds.listAttr's
    #    reported name (vtx/pt -> pnts, map -> uvSet.uvSetPoints), which had
    #    silently diverged the two paths.

    def test_vtx_resolution_matches_plug_construction(self):
        cube  = cmds.polyCube(name="cube_consistency_vtx")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        self.assertEqual(str(node.vtx[0]), str(Plug(f"{shape}.vtx[0]")))

    def test_map_resolution_matches_plug_construction(self):
        cube  = cmds.polyCube(name="cube_consistency_map")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        self.assertEqual(str(node.map[0]), str(Plug(f"{shape}.map[0]")))

    def test_pnts_resolution_matches_plug_construction(self):
        # ``pnts`` is a *real* mesh attr, so ``findPlug`` resolves it in
        # find_attr's PRIMARY branch and bypasses the alias-normalization
        # fallback. As a registered component alias it must still match the
        # Plug path (-> stable ``controlPoints``), not the version-specific
        # ``pnts`` tweak attr it literally names.
        cube  = cmds.polyCube(name="cube_consistency_pnts")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        self.assertEqual(str(node.pnts[0]), str(Plug(f"{shape}.pnts[0]")))
        self.assertIn("controlPoints[0]", str(node.pnts[0]))

    def test_pt_resolution_matches_plug_construction(self):
        # ``pt`` is the short name of mesh ``pnts``; ``findPlug`` likewise
        # resolves it in the primary branch. Must canonicalize the same way.
        cube  = cmds.polyCube(name="cube_consistency_pt")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        self.assertEqual(str(node.pt[0]), str(Plug(f"{shape}.pt[0]")))
        self.assertIn("controlPoints[0]", str(node.pt[0]))

    def test_canonicalize_falls_back_to_original_when_findplug_raises(self):
        # Coverage: the defensive ``except RuntimeError: return plug`` in
        # ``_canonicalize_component_alias_plug``. ``pnts`` resolves via the
        # primary findPlug branch, then re-resolves to ``controlPoints``; that
        # second lookup succeeds on any real mesh, so force it to fail by
        # mapping the alias to a non-existent canonical and assert the original
        # plug is returned unchanged (not raised).
        from unittest import mock

        from rig.maya.nodetypes import dg_node as dg_node_mod

        cube  = cmds.polyCube(name="cube_canon_fallback")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        with mock.patch.dict(
            dg_node_mod._CANONICAL_COMPONENT_PLUG, {"pnts": "noSuchCanonicalAttr"}
        ):
            plug = node.pnts
        self.assertIsInstance(plug, Plug)
        self.assertIn("pnts", str(plug))
        self.assertNotIn("controlPoints", str(plug))

    # -- functional liveness + path equivalence: the resolved canonical must
    #    be the attribute that actually DRIVES the component, not a
    #    same-named-but-dead one. On Maya 2026 cmds.listAttr reports
    #    uvSet.uvSetPoints for map[N], but that array is independent of the
    #    live UVs; uvpt is what cmds.polyEditUV reflects. These assert that
    #    setting via node.<alias>[N] (a) actually changes the live geometry
    #    and (b) produces the SAME result as the Plug(...) construction path.
    #    (The exact set arithmetic -- controlPoints applies an absolute vertex
    #    position rather than a raw tweak -- is pre-existing DSL behavior and
    #    deliberately not asserted here; we only pin liveness + equivalence.)

    def test_node_vtx_set_matches_plug_path_and_is_live(self):
        a      = cmds.polyCube(name="cube_set_node")[0]
        ashape = cmds.listRelatives(a, shapes=True)[0]
        b      = cmds.polyCube(name="cube_set_plug")[0]
        bshape = cmds.listRelatives(b, shapes=True)[0]
        base   = cmds.pointPosition(f"{ashape}.vtx[0]", local=True)
        Node(ashape).vtx[0]      << (1.0, 2.0, 3.0)
        Plug(f"{bshape}.vtx[0]") << (1.0, 2.0, 3.0)
        pa = cmds.pointPosition(f"{ashape}.vtx[0]", local=True)
        pb = cmds.pointPosition(f"{bshape}.vtx[0]", local=True)
        for x, y in zip(pa, pb):
            self.assertAlmostEqual(x, y, places=4)  # path equivalence
        # Live: the vertex actually moved (would be unchanged for a dead attr).
        self.assertNotAlmostEqual(pa[0], base[0], places=4)

    def test_node_map_set_matches_plug_path_and_is_live(self):
        a      = cmds.polyCube(name="cube_uv_node")[0]
        ashape = cmds.listRelatives(a, shapes=True)[0]
        b      = cmds.polyCube(name="cube_uv_plug")[0]
        bshape = cmds.listRelatives(b, shapes=True)[0]
        orig   = cmds.polyEditUV(f"{ashape}.map[0]", query=True)
        Node(ashape).map[0]      << (0.25, 0.75)
        Plug(f"{bshape}.map[0]") << (0.25, 0.75)
        ua = cmds.polyEditUV(f"{ashape}.map[0]", query=True)
        ub = cmds.polyEditUV(f"{bshape}.map[0]", query=True)
        for x, y in zip(ua, ub):
            self.assertAlmostEqual(x, y, places=4)  # path equivalence
        # Live: the UV actually changed (would stay `orig` if alias resolved
        # to the independent/dead uvSetPoints array instead of uvpt).
        self.assertNotAlmostEqual(ua[1], orig[1], places=4)

    def test_unknown_alias_still_raises(self):
        cube  = cmds.polyCube(name="cube_for_unknown")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        node  = Node(shape)
        # `e` (edges) is a component type but NOT in the alias map and
        # has no underlying plug -- must still raise AttributeError.
        with self.assertRaises(AttributeError):
            _ = node.completely_made_up_attribute_xyz


class TestNodeMultiDimComponentIndexing(MayaTestCase):
    """numpy-style N-D indexing for multi-dimensional geometry components.

    NURBS-surface ``cv`` (2-D) and lattice ``pt`` (3-D) support numpy-style
    per-axis indexing: ``cv[u, v]`` is an element; a bare/partial index or any
    slice yields a ``PlugList`` (row / column / grid / range); the resulting
    plugs *display* as ``cv[u][v]`` / ``pt[s][t][u]`` while resolving to the
    real flat ``controlPoints[k]`` storage. 1-D components (curve ``cv``, mesh
    ``vtx``/``pnts``, mesh ``uv``/``map``) are unchanged: a single index is the
    element. Flat access remains exclusively through ``controlPoints``.
    """

    TEST_START_NEW_SCENE = True

    # -- fixtures -- #

    def _plane(self, name):
        # default nurbsPlane: open 4x4 CV grid (U=V=4, no periodicity).
        srf = cmds.nurbsPlane(name=name)[0]
        return cmds.listRelatives(srf, shapes=True)[0]

    def _lattice(self, geo, divisions):
        cmds.polyCube(name=geo)
        ffd = cmds.lattice(geo, divisions=divisions)  # [ffd, lattice, base]
        return cmds.listRelatives(ffd[1], shapes=True)[0]

    # -- display / identity (component-named, not controlPoints) -- #

    def test_surface_element_displays_component_name(self):
        shape = self._plane("np_disp")
        node  = Node(shape)
        self.assertEqual(str(node.cv[0, 0]), f"{shape}.cv[0][0]")
        self.assertEqual(str(node.cv[1, 2]), f"{shape}.cv[1][2]")

    def test_surface_element_chained_index_displays_component_name(self):
        # cv[i][j] still works: cv[i] is the row, [j] picks the element.
        shape = self._plane("np_chain")
        self.assertEqual(str(Node(shape).cv[1][2]), f"{shape}.cv[1][2]")

    def test_lattice_element_displays_component_name(self):
        shape = self._lattice("lat_disp", (3, 4, 5))
        self.assertEqual(str(Node(shape).pt[1, 2, 0]), f"{shape}.pt[1][2][0]")

    def test_bare_handle_is_controlpoints_multi(self):
        from rig._internal.plug import ComponentPlug

        shape = self._plane("np_bare")
        bare  = Node(shape).cv
        self.assertIsInstance(bare, ComponentPlug)
        self.assertEqual(str(bare), f"{shape}.controlPoints")

    def test_element_equals_flat_controlpoint_plug(self):
        # equals() is identity-by-storage: a component element (displays
        # cv[u][v]) and a flat controlPoints Plug to the SAME underlying MPlug
        # are .equals()-equal, symmetric, and consistent with __hash__.
        shape = self._plane("np_equals")
        node  = Node(shape)
        elem  = node.cv[1, 2]  # -> controlPoints[6] on an open 4x4 plane
        flat  = Plug(f"{shape}.controlPoints[6]")
        self.assertTrue(elem.equals(flat))
        self.assertTrue(flat.equals(elem))        # symmetric
        self.assertEqual(hash(elem), hash(flat))  # hash/equals consistent
        # a different element is NOT equal
        self.assertFalse(elem.equals(node.cv[1, 3]))

    # -- resolution correctness: same storage as the Plug string path -- #

    def test_surface_element_resolves_to_same_controlpoint_as_plug_path(self):
        shape = self._plane("np_resolve")
        node  = Node(shape)
        for u, v in [(0, 0), (1, 2), (3, 3)]:
            self.assertEqual(
                node.cv[u, v].name,
                Plug(f"{shape}.cv[{u}][{v}]").name,
                msg=f"cv[{u},{v}] storage diverged from Plug path",
            )
            self.assertIn("controlPoints", node.cv[u, v].name)

    def test_lattice_element_resolves_to_same_controlpoint_as_plug_path(self):
        shape = self._lattice("lat_resolve", (3, 4, 5))
        node  = Node(shape)
        for s, t, u in [(0, 0, 0), (1, 2, 0), (2, 3, 4)]:
            self.assertEqual(
                node.pt[s, t, u].name,
                Plug(f"{shape}.pt[{s}][{t}][{u}]").name,
                msg=f"pt[{s},{t},{u}] storage diverged from Plug path",
            )

    # -- numpy shape semantics (plane: open 4x4) -- #

    def test_all_int_tuple_returns_single_plug(self):
        from rig._internal.list import PlugList

        shape = self._plane("np_single")
        elem  = Node(shape).cv[0, 0]
        self.assertIsInstance(elem, Plug)
        self.assertNotIsInstance(elem, PlugList)

    def test_bare_single_index_returns_row(self):
        from rig._internal.list import PlugList

        shape = self._plane("np_row")
        row   = Node(shape).cv[0]
        self.assertIsInstance(row, PlugList)
        self.assertEqual(
            [str(p) for p in row], [f"{shape}.cv[0][{v}]" for v in range(4)]
        )

    def test_row_via_explicit_slice_matches_bare(self):
        shape = self._plane("np_row2")
        node  = Node(shape)
        self.assertEqual([str(p) for p in node.cv[0, :]], [str(p) for p in node.cv[0]])

    def test_column_slice_returns_column(self):
        from rig._internal.list import PlugList

        shape = self._plane("np_col")
        col   = Node(shape).cv[:, 1]
        self.assertIsInstance(col, PlugList)
        self.assertEqual(
            [str(p) for p in col], [f"{shape}.cv[{u}][1]" for u in range(4)]
        )

    def test_full_slice_returns_row_major_grid(self):
        shape = self._plane("np_grid")
        node  = Node(shape)
        grid  = node.cv[:]
        self.assertEqual(len(grid),     16)
        self.assertEqual(str(grid[0]),  f"{shape}.cv[0][0]")
        self.assertEqual(str(grid[-1]), f"{shape}.cv[3][3]")
        # cv[:] is identical to cv[:, :]
        self.assertEqual([str(p) for p in grid], [str(p) for p in node.cv[:, :]])

    def test_range_slice(self):
        shape = self._plane("np_range")
        rng   = Node(shape).cv[1:3, 2]
        self.assertEqual(
            [str(p) for p in rng], [f"{shape}.cv[1][2]", f"{shape}.cv[2][2]"]
        )

    def test_negative_index(self):
        shape = self._plane("np_neg")
        self.assertEqual(str(Node(shape).cv[-1, 0]), f"{shape}.cv[3][0]")

    def test_step_slice(self):
        shape   = self._plane("np_step")
        stepped = Node(shape).cv[::2, 0]
        self.assertEqual(
            [str(p) for p in stepped], [f"{shape}.cv[0][0]", f"{shape}.cv[2][0]"]
        )

    # -- periodic surface: full grid uses the DISTINCT CV count -- #

    def test_periodic_sphere_full_grid_uses_distinct_cv_count(self):
        # A nurbsSphere is periodic in V (numCVsInV=11 but only 8 distinct);
        # cv[:] must enumerate the 7x8=56 distinct CVs, not 7x11.
        sph   = cmds.sphere(name="sph_periodic")[0]
        shape = cmds.listRelatives(sph, shapes=True)[0]
        self.assertEqual(len(Node(shape).cv[:]), 56)

    # -- lattice 3-D shapes -- #

    def test_lattice_single_element(self):
        from rig._internal.list import PlugList

        shape = self._lattice("lat_elem", (3, 4, 5))
        elem  = Node(shape).pt[1, 2, 0]
        self.assertIsInstance(elem, Plug)
        self.assertNotIsInstance(elem, PlugList)

    def test_lattice_line(self):
        shape = self._lattice("lat_line", (3, 4, 5))
        line  = Node(shape).pt[1, 2]  # all u at s=1,t=2 -> 5
        self.assertEqual(
            [str(p) for p in line], [f"{shape}.pt[1][2][{u}]" for u in range(5)]
        )

    def test_lattice_slab(self):
        shape = self._lattice("lat_slab", (3, 4, 5))
        slab  = Node(shape).pt[0]  # all (t,u) at s=0 -> 4*5=20
        self.assertEqual(len(slab), 20)
        self.assertEqual(str(slab[0]), f"{shape}.pt[0][0][0]")

    def test_lattice_plane(self):
        shape = self._lattice("lat_plane", (3, 4, 5))
        plane = Node(shape).pt[:, :, 0]  # all (s,t) at u=0 -> 3*4=12
        self.assertEqual(len(plane), 12)
        self.assertEqual(str(plane[0]), f"{shape}.pt[0][0][0]")

    # -- operations are live and match the Plug path -- #

    def test_surface_element_set_is_live_and_matches_plug(self):
        a    = self._plane("np_set_a")
        b    = self._plane("np_set_b")
        base = cmds.pointPosition(f"{a}.cv[1][2]", local=True)
        Node(a).cv[1, 2]      << (1.0, 2.0, 3.0)
        Plug(f"{b}.cv[1][2]") << (1.0, 2.0, 3.0)
        pa = cmds.pointPosition(f"{a}.cv[1][2]", local=True)
        pb = cmds.pointPosition(f"{b}.cv[1][2]", local=True)
        for x, y in zip(pa, pb):
            self.assertAlmostEqual(x, y, places=4)  # path equivalence
        self.assertNotAlmostEqual(pa[0], base[0], places=4)  # live

    # -- rename safety: a held handle resolves via the LIVE node -- #

    def test_surface_handle_survives_shape_rename(self):
        # A handle bound BEFORE a rename must still resolve/set elements
        # afterwards -- resolution derives the node name from the live MPlug,
        # not a cached string (MObjectHandle-based identity, per DSL contract).
        shape   = self._plane("np_rename_pre")
        handle  = Node(shape).cv  # bind before the rename
        renamed = cmds.rename(shape, "np_rename_post")
        # resolution path (_axis_sizes bounds-check + _element) uses new name
        elem = handle[1, 2]
        self.assertEqual(str(elem), f"{renamed}.cv[1][2]")
        self.assertEqual(elem.name, Plug(f"{renamed}.cv[1][2]").name)
        # and a write through the held handle drives the renamed shape
        base = cmds.pointPosition(f"{renamed}.cv[1][2]", local=True)
        handle[1, 2] << (4.0, 5.0, 6.0)
        after = cmds.pointPosition(f"{renamed}.cv[1][2]", local=True)
        self.assertNotAlmostEqual(after[0], base[0], places=4)

    def test_lattice_handle_survives_shape_rename(self):
        # Same rename safety on the lattice branch of _axis_sizes (sDivisions).
        shape   = self._lattice("lat_rename_pre", (3, 4, 5))
        handle  = Node(shape).pt  # bind before the rename
        renamed = cmds.rename(shape, "lat_rename_post")
        slab    = handle[0]       # exercises _axis_sizes lattice branch + _element
        self.assertEqual(len(slab), 20)  # 4 * 5 at s=0
        self.assertEqual(str(slab[0]), f"{renamed}.pt[0][0][0]")

    # -- lattice element write is live and matches the Plug path -- #

    def test_lattice_element_set_is_live_and_matches_plug(self):
        a    = self._lattice("lat_set_a", (3, 4, 5))
        b    = self._lattice("lat_set_b", (3, 4, 5))
        base = cmds.getAttr(f"{a}.pt[1][2][0]")[0]
        Node(a).pt[1, 2, 0]      << (5.0, 5.0, 5.0)
        Plug(f"{b}.pt[1][2][0]") << (5.0, 5.0, 5.0)
        pa = cmds.getAttr(f"{a}.pt[1][2][0]")[0]
        pb = cmds.getAttr(f"{b}.pt[1][2][0]")[0]
        for x, y in zip(pa, pb):
            self.assertAlmostEqual(x, y, places=4)  # path equivalence
        self.assertNotAlmostEqual(pa[0], base[0], places=4)  # live

    # -- writes through a slice PlugList (broadcast / per-element / connect) -- #

    def test_column_asymmetric_per_element_set(self):
        # cv[:, 0] is a PlugList of 4 elements; a 4-item list assigns one tuple
        # per element (not a broadcast of the last value).
        shape = self._plane("np_col_asym")
        Node(shape).cv[:, 0] << [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
        xs = [
            cmds.pointPosition(f"{shape}.cv[{u}][0]", local=True)[0] for u in range(4)
        ]
        for got, want in zip(xs, [1.0, 2.0, 3.0, 4.0]):
            self.assertAlmostEqual(got, want, places=4)

    def test_column_scalar_broadcast_set(self):
        # A bare scalar broadcasts to every element of the column.
        shape = self._plane("np_col_bcast")
        Node(shape).cv[:, 3] << 9.0
        for u in range(4):
            x = cmds.pointPosition(f"{shape}.cv[{u}][3]", local=True)[0]
            self.assertAlmostEqual(x, 9.0, places=4)

    def test_column_vectorized_connect(self):
        # cv[:, 2] << PlugList(sources) connects one source per element.
        from rig._internal.list import PlugList

        shape = self._plane("np_col_conn")
        locs  = [cmds.spaceLocator(name=f"np_loc{u}")[0] for u in range(4)]
        Node(shape).cv[:, 2] << PlugList([Node(loc).translate for loc in locs])
        for u, loc in enumerate(locs):
            conns = cmds.listConnections(
                f"{shape}.cv[{u}][2]", s=True, d=False, plugs=True
            )
            self.assertEqual(conns, [f"{loc}.translate"])

    def test_element_connect_drives_controlpoint(self):
        # Connecting a driver INTO a resolved element lands on the underlying
        # controlPoints[k] (distinct code path from a literal set).
        shape = self._plane("np_elem_conn")
        loc   = cmds.spaceLocator(name="np_elem_loc")[0]
        Node(shape).cv[1, 2] << Node(loc).translate
        via_cv = cmds.listConnections(f"{shape}.cv[1][2]", s=True, d=False, plugs=True)
        self.assertEqual(via_cv, [f"{loc}.translate"])
        via_cp = cmds.listConnections(
            f"{shape}.controlPoints[6]", s=True, d=False, plugs=True
        )
        self.assertEqual(via_cp, [f"{loc}.translate"])

    # -- numpy-aware reads (>> None) stack per element -- #

    def test_row_get_returns_stacked_array(self):
        import numpy as np

        shape = self._plane("np_row_get")
        val   = Node(shape).cv[0] >> None
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (4, 3))

    def test_grid_column_and_element_get_shapes(self):
        shape = self._plane("np_grid_get")
        node  = Node(shape)
        self.assertEqual((node.cv[:] >> None).shape,    (16, 3))
        self.assertEqual((node.cv[:, 1] >> None).shape, (4, 3))
        self.assertEqual((node.cv[0, 0] >> None).shape, (3,))

    # -- periodic in BOTH axes: torus collapses U and V independently -- #

    def test_periodic_torus_both_axes_collapse(self):
        # A torus is periodic in U AND V (numU=7 degU=3 -> 4, numV=11 degV=3
        # -> 8). cv[:] enumerates 4*8=32; cv[:, 0] pins the U-axis subtraction.
        tor   = cmds.torus(name="tor_periodic")[0]
        shape = cmds.listRelatives(tor, shapes=True)[0]
        node  = Node(shape)
        self.assertEqual(len(node.cv[:]), 32)
        self.assertEqual(len(node.cv[:, 0]), 4)

    # -- slice edge cases (length-1 / empty / mixed sub-grid) -- #

    def test_length_one_slice_is_pluglist_not_element(self):
        # A length-1 slice still yields a PlugList -- it does NOT collapse to a
        # bare element (the "element iff every axis is an explicit int" rule).
        from rig._internal.list import PlugList

        shape = self._plane("np_len1")
        sel   = Node(shape).cv[0:1, 0]
        self.assertIsInstance(sel, PlugList)
        self.assertEqual(len(sel), 1)

    def test_empty_slice_returns_empty_pluglist(self):
        from rig._internal.list import PlugList

        shape = self._plane("np_empty")
        sel   = Node(shape).cv[5:5, 0]
        self.assertIsInstance(sel, PlugList)
        self.assertEqual(len(sel), 0)

    def test_mixed_subgrid_range_by_full_slice(self):
        shape = self._plane("np_mixed")
        sub   = Node(shape).cv[1:3, :]
        self.assertEqual(
            [str(p) for p in sub],
            [f"{shape}.cv[{u}][{v}]" for u in (1, 2) for v in range(4)],
        )

    # -- 1-D components unchanged (single index = element) -- #

    def test_curve_cv_1d_is_element_not_wrapped(self):
        curve = cmds.curve(name="curve_1d", p=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        shape = cmds.listRelatives(curve, shapes=True)[0]
        plug  = Node(shape).cv[1]
        self.assertEqual(str(plug), str(Plug(f"{shape}.cv[1]")))
        self.assertIn("controlPoints[1]", str(plug))

    def test_mesh_vtx_1d_is_element(self):
        cube  = cmds.polyCube(name="cube_1d_vtx")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        self.assertIn("controlPoints[2]", str(Node(shape).vtx[2]))

    def test_mesh_uv_1d_is_element(self):
        # UVs are 1-D (single linear index), not multi-dim.
        cube  = cmds.polyCube(name="cube_1d_uv")[0]
        shape = cmds.listRelatives(cube, shapes=True)[0]
        self.assertIn("uvpt[0]", str(Node(shape).map[0]))

    # -- flat controlPoints unchanged; a bare index is NOT a flat element -- #

    def test_controlpoints_flat_access_unchanged(self):
        shape = self._plane("np_cp")
        self.assertEqual(str(Node(shape).controlPoints[0]), f"{shape}.controlPoints[0]")

    def test_bare_single_index_is_not_flat_controlpoint(self):
        from rig._internal.list import PlugList

        shape = self._plane("np_notflat")
        # cv[2] is row 2 (a PlugList), NOT the flat controlPoints[2].
        self.assertIsInstance(Node(shape).cv[2], PlugList)

    # -- errors -- #

    def test_too_many_indices_raises(self):
        shape = self._plane("np_toomany")
        with self.assertRaises(IndexError):
            _ = Node(shape).cv[0, 0, 0]

    def test_out_of_range_index_raises(self):
        shape = self._plane("np_oor")
        node  = Node(shape)
        with self.assertRaises(IndexError):
            _ = node.cv[99, 0]
        with self.assertRaises(IndexError):
            _ = node.cv[0, 99]
        # too-negative index takes the spec+size normalization path
        with self.assertRaises(IndexError):
            _ = node.cv[-99, 0]

    def test_invalid_index_type_raises(self):
        shape = self._plane("np_badidx")
        with self.assertRaises(TypeError):
            _ = Node(shape).cv[0, "x"]

    def test_bool_index_raises_type_error(self):
        # ``bool`` subclasses ``int`` but is not a valid component index;
        # it must raise a clear TypeError, not be treated as 0/1.
        shape = self._plane("np_boolidx")
        with self.assertRaises(TypeError):
            _ = Node(shape).cv[True, 0]

    def test_resolved_element_indexes_as_plain_plug(self):
        # A resolved element is a normal Plug, not a component handle: further
        # indexing routes to Plug.__getitem__ (it does NOT re-enter numpy
        # axis resolution). Like any component-leaf plug -- e.g.
        # ``Plug("shape.controlPoints[0]")[0]`` -- positional indexing of the
        # leaf raises, since controlPoints[k] is not itself a multi.
        shape = self._plane("np_child")
        elem  = Node(shape).cv[0, 0]
        with self.assertRaises(RuntimeError):
            _ = elem[0]

    # -- defensive gates (coverage) -- #

    def test_maybe_component_plug_returns_none_when_node_probe_raises(self):
        # Defensive: if probing the owning node (name / nodeType) raises, the
        # upgrade is skipped (returns None) so the caller keeps its plain Plug.
        # Tested directly with a fake attr -- patching the shared ``cmds``
        # module would break unrelated call sites (e.g. Node construction).
        from unittest import mock

        from rig._internal.plug import _maybe_component_plug

        fake_attr = mock.Mock()
        type(fake_attr).node = mock.PropertyMock(side_effect=AttributeError("no node"))
        self.assertIsNone(_maybe_component_plug("cv", fake_attr))

    def test_component_plug_skipped_for_non_alias_attr(self):
        # ``controlPoints`` is the real attr, not the registered multi-dim
        # alias (``cv``) for this node type -- it stays a plain Plug.
        from rig._internal.plug import ComponentPlug

        shape = self._plane("np_nonalias")
        plug  = Node(shape).controlPoints
        self.assertNotIsInstance(plug, ComponentPlug)
        self.assertIn("controlPoints", str(plug))


class TestNodeUnresolvedMultiChild(MayaTestCase):
    """Regression: a bare child-of-multi name must raise AttributeError, not
    resolve to an unusable ``<multi>[-1].<child>`` plug.

    blendShape ``weights`` is the child of the ``weightList`` multi; Maya's
    ``findPlug("weights")`` returns ``weightList[-1].weights`` (unresolved
    parent index). Indexing it returned the wrong plug and slicing it aborted
    Maya with ``TDEbadMultiIndex``. The intended target-weight accessor is the
    top-level ``weight`` multi, which stays unaffected.
    """

    TEST_START_NEW_SCENE = True

    def _blendshape(self):
        base = cmds.polyCube(name="base")[0]
        t1   = cmds.polyCube(name="t1")[0]
        t2   = cmds.polyCube(name="t2")[0]
        bs   = cmds.blendShape(t1, t2, base, name="bs1")[0]
        return Node(bs)

    def test_weights_child_of_multi_raises(self):
        bs = self._blendshape()
        with self.assertRaises(AttributeError):
            _ = bs.weights

    def test_weights_slice_does_not_abort(self):
        # Before the fix ``bs.weights[:]`` aborted Maya (TDEbadMultiIndex);
        # attribute access now raises before any slicing occurs.
        bs = self._blendshape()
        with self.assertRaises(AttributeError):
            _ = bs.weights[:]

    def test_weight_multi_accessor_unaffected(self):
        from rig import PlugList

        bs     = self._blendshape()
        sliced = bs.weight[:]
        self.assertIsInstance(sliced, PlugList)
        self.assertEqual(len(sliced), 2)
        self.assertIsInstance(bs.weight[0], Plug)

    def test_find_attr_quiet_returns_none(self):
        bs = self._blendshape()
        self.assertIsNone((bs >> None).find_attr("weights", quiet=True))


class TestNodeBlendShapeAlias(MayaTestCase):
    """Node attribute access must resolve blendShape target-weight aliases.

    A blendShape aliases each ``weight[i]`` element to its target name (Maya's
    ``aliasAttr``). ``findPlug`` doesn't resolve those aliases, so ``bs.<alias>``
    falls back to ``MSelectionList``, which returns the concrete ``weight[i]``
    element whose ``name()`` reports the alias.
    """

    TEST_START_NEW_SCENE = True

    def _blendshape(self):
        base = cmds.polyCube(name="base")[0]
        t1   = cmds.polyCube(name="t1")[0]
        t2   = cmds.polyCube(name="t2")[0]
        bs   = cmds.blendShape(t1, t2, base, name="bs1")[0]
        return Node(bs)

    def _alias_map(self, bs):
        # cmds.aliasAttr returns [aliasName, plugName, ...] e.g.
        # ['happy', 'weight[0]', 'sad', 'weight[1]']; map alias -> element index.
        pairs = cmds.aliasAttr(str(bs), query=True) or []
        return {
            alias: int(plug.split("[")[1].rstrip("]"))
            for alias, plug in zip(pairs[0::2], pairs[1::2])
        }

    def test_alias_direct_access(self):
        bs      = self._blendshape()
        aliases = self._alias_map(bs)
        self.assertTrue(aliases)
        for alias in aliases:
            plug = getattr(bs, alias)
            self.assertIsInstance(plug, Plug)
            self.assertIn(alias, str(plug))

    def test_alias_matches_weight_element(self):
        bs = self._blendshape()
        for alias, idx in self._alias_map(bs).items():
            self.assertEqual(str(getattr(bs, alias)), str(bs.weight[idx]))

    def test_weight_element_reports_alias(self):
        bs = self._blendshape()
        for alias, idx in self._alias_map(bs).items():
            self.assertIn(alias, str(bs.weight[idx]))

    def test_unknown_attr_still_raises(self):
        bs = self._blendshape()
        with self.assertRaises(AttributeError):
            _ = bs.definitely_not_an_attr

    def test_unknown_attr_quiet_returns_none(self):
        bs = self._blendshape()
        self.assertIsNone((bs >> None).find_attr("definitely_not_an_attr", quiet=True))