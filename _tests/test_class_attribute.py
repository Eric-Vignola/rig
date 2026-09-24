import logging

import maya.cmds as cmds
from maya.api import OpenMaya
from rig.nodetypes import PyNode
from rig._tests._base import MayaTestCase

LOGGER = logging.getLogger(__name__)


TYPE_DICT = {
    "bool":         {"val": True},
    "long":         {"val": 3},
    "short":        {"val": 3},
    "byte":         {"val": 3},
    "char":         {"val": 3},
    "enum":         {"val": 3, "kwargs": {"enumName": "a:b:c:d"}},
    "float":        {"val": 3.5},
    "double":       {"val": 3.5},
    "doubleAngle":  {"val": 3.5},
    "doubleLinear": {"val": 3.5},
    "string":       {"is_dt": True, "val": "abc"},
    "stringArray": {
        "is_dt":   True,
        "val":     ["abc", "def"],
        "set_val": "cast_with_length",
    },
    "message":   {},
    "time":      {"val": 3},
    "matrix":    {"is_dt": True, "val": list(range(16)), "set_val": "cast"},
    "fltMatrix": {"val": list(range(16)), "set_val": "cast", "return_type": "matrix"},
    # TODO how to test those?
    # "reflectanceRGB": {"is_dt": True},
    # "reflectance": {},
    # "spectrumRGB": {"is_dt": True},
    # "spectrum": {},
    # TODO how to test float2, float3, etc as at?
    "float2":      {"is_dt": True, "val": [1.2, 2.3], "set_val": "cast"},
    "float3":      {"is_dt": True, "val": [1.2, 2.3, 3.4], "set_val": "cast"},
    "double2":     {"is_dt": True, "val": [1.2, 2.3], "set_val": "cast"},
    "double3":     {"is_dt": True, "val": [1.2, 2.3, 3.4], "set_val": "cast"},
    "long2":       {"is_dt": True, "val": [1, 2], "set_val": "cast"},
    "long3":       {"is_dt": True, "val": [1, 2, 3], "set_val": "cast"},
    "short2":      {"is_dt": True, "val": [1, 2], "set_val": "cast"},
    "short3":      {"is_dt": True, "val": [1, 2, 3], "set_val": "cast"},
    "doubleArray": {"is_dt": True, "val": [1.2, 2.3, 3.4, 4.5]},
    "floatArray":  {"is_dt": True, "val": [1.2, 2.3, 3.4, 4.5]},
    "Int32Array":  {"is_dt": True, "val": [1, 2, 3, 4]},
    "vectorArray": {
        "is_dt":   True,
        "val":     [[1, 2, 1], [2, 3, 1], [3, 4, 1]],
        "set_val": "cast_with_length",
    },
    "pointArray": {
        "is_dt":   True,
        "val":     [[1, 1, 1, 1], [2, 2, 2, 1], [3, 3, 3, 1]],
        "set_val": "cast_with_length",
    },
}


class TestAttributeClass(MayaTestCase):
    """
    Attribute class unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.node = PyNode.create("transform", name="test_xform")

    def test_creation_and_properties(self):
        for typ, data in TYPE_DICT.items():
            is_dt = "is_dt" in data
            # assemble creation kwargs
            kwargs = {"dt": typ} if is_dt else {"at": typ}
            if "kwargs" in data:
                kwargs.update(data["kwargs"])

            # add attribute
            attr_name = f"test_{typ}_attr"
            attr      = self.node.add_attr(f"test_{typ}_attr", **kwargs)

            # property check
            self.assertTrue(attr.fn_set.hasObj(attr.mobject))
            self.assertEqual(
                attr.attribute_type, "typed" if data.get("is_dt", False) else typ
            )
            self.assertEqual(attr.data_type, data.get("return_type", typ))
            self.assertEqual(attr.name,      attr_name)
            self.assertEqual(attr.full_name, f"{self.node}.{attr_name}")
            self.assertTrue(attr.is_dynamic)
            self.assertFalse(attr.is_multi)

            # value check
            val = data.get("val")
            if val:
                # set val
                if data.get("set_val") == "cast_with_length":
                    attr.set(len(val), *val)
                elif data.get("set_val") == "cast":
                    attr.set(*val)
                else:
                    attr.set(val)

                # get value
                new_val = attr.get()
                if isinstance(val, (list, tuple)):
                    self.assertTrue(self.assert_list_equal(new_val, val))
                else:
                    self.assertEqual(new_val, val)

            # test category
            self.assertEqual(attr.get_categories(), [])
            self.assertFalse(attr.has_category("aaa"))
            categories = ["cat1", "cat2"]
            attr.add_category(categories)
            self.assertEqual(attr.get_categories(), categories)
            attr.add_category(["aaa", "cat2"])
            self.assertEqual(attr.get_categories(), categories + ["aaa"])

        # test on a native attr
        attr = self.node.tx
        self.assertFalse(attr.is_dynamic)
        self.assertEqual(attr.node, self.node)

        self.assertEqual(attr.alias, attr.name)
        alias      = f"{typ}_alias"
        attr.alias = f"{typ}_alias"
        self.assertEqual(attr.alias, f"{typ}_alias")
        self.assertEqual(self.node.find_alias(alias), attr)
        attr.alias = None
        self.assertEqual(attr.alias, attr.name)
        with self.assertRaises(RuntimeError):
            self.node.find_alias(alias)

        self.assertFalse(attr.is_locked)
        attr.is_locked = True
        self.assertTrue(attr.is_locked)

        self.assertTrue(attr.is_keyable)
        attr.is_keyable = False
        self.assertFalse(attr.is_keyable)

        self.assertEqual(attr.default_value, 0)
        with self.assertRaises(RuntimeError):
            attr.default_value = 1

    def test_compound_attr(self):
        num_c     = 3
        attr_name = "test_compound"
        attr = self.node.add_attr(
            attr_name, attributeType="compound", numberOfChildren=num_c
        )
        for i in range(num_c):
            self.node.add_attr(f"c{i}", attributeType="message", parent=attr_name)
        attr   = self.node.find_attr(attr_name)

        c_attr = self.node.c0
        self.assertEqual(c_attr.get_parent(), attr)
        self.assertEqual(attr.c0,             c_attr)
        self.assertEqual(attr.num_children,   num_c)

    def test_array_attr(self):
        attr_name = "test_array"
        attr      = self.node.add_attr(attr_name, attributeType="double", multi=True)
        self.assertEqual(attr.default_value, 0)
        self.assertTrue(attr.is_multi)
        attr[1].set(1)
        attr[4].set(4)
        self.assert_list_equal(attr.get_logical_indices(), [1, 4])
        self.assertEqual(attr.num_elements, 2)

    def test_typed_attr(self):
        attr_name = "test_typed"
        attr      = self.node.add_attr(attr_name, dataType="doubleArray")
        attr.set((0, 1, 0, 0, 4, 0))
        ids, vals = attr.filter_array_values(0.0)
        self.assertEqual(ids, [1, 4])
        self.assertEqual(vals, [1.0, 4.0])

    def test_counted_array_attr_from_list(self):
        # a single list is expanded to the (count, *items) form of cmds.setAttr()
        for typ in ("stringArray", "vectorArray", "pointArray"):
            val  = TYPE_DICT[typ]["val"]
            attr = self.node.add_attr(f"test_{typ}_list", dataType=typ)
            attr.set(val)
            self.assertTrue(self.assert_list_equal(attr.get(), val))

            # the explicit form still works
            attr.set(len(val), *val)
            self.assertTrue(self.assert_list_equal(attr.get(), val))

    def test_connections(self):
        attr1 = self.node.tx
        attr2 = self.node.ty
        for attr in (attr1, attr2):
            self.assertFalse(attr.is_connected)
            self.assertTrue(attr.is_free_to_change)

        attr1 >> attr2
        self.assertTrue(attr1.is_connected)
        self.assertTrue(attr1.is_free_to_change)
        self.assertTrue(attr2.is_connected)
        self.assertFalse(attr2.is_free_to_change)

        self.assertEqual(attr1.get_connected_attrs(src=False, dst=True), [attr2])
        self.assertEqual(attr2.get_connected_attrs(src=True, dst=False), [attr1])

        attr1 // attr2
        for attr in (attr1, attr2):
            self.assertFalse(attr.is_connected)
            self.assertTrue(attr.is_free_to_change)

        attr1 >> attr2
        attr3 = self.node.tz
        with self.assertRaises(RuntimeError):
            attr3.connect(attr2)
        attr3.connect(attr2, force=True)
        self.assertFalse(attr1.is_connected)
        self.assertTrue(attr2.is_connected)
        self.assertTrue(attr3.is_connected)

        attr3.break_connections()
        for attr in (attr1, attr2, attr3):
            self.assertFalse(attr.is_connected)

    def test_set_attrs(self):
        xform = PyNode(cmds.polyCube(ch=False)[0])
        data = {
            "tx": 1,
            "ty": 2,
            "tz": 3,
        }
        xform.set_attrs(**data)
        self.assertEqual(xform.t.get()[0], (1, 2, 3))

        # gracefully skp missing attrs
        data = {"translate": [10, 20, 30], "jointOrient": [100, 200, 300]}
        xform.set_attrs(skip_missing=True, **data)
        self.assertEqual(xform.t.get()[0], (10, 20, 30))

    def test_slicing(self):
        camera = PyNode("persp")
        pma    = PyNode.create("plusMinusAverage")
        camera.t >> pma.input3D[0]
        camera.t >> pma.input3D[1]
        camera.t >> pma.input3D[2]
        camera.t >> pma.input3D[3]
        self.assertEqual(len(pma.input3D[:]), 4)

    # TODO: create a Component class and move tests there
    def test_component_types(self):
        obj = cmds.polyCube()[0]
        obj = PyNode(obj)
        self.assertEqual(obj.t._component_type, "unknown")
        self.assertEqual(obj.tx._component_type, "unknown")

        obj = PyNode(cmds.listHistory(obj)[0])
        self.assertEqual(obj.pnts._component_type,    "kMeshVertComponent")
        self.assertEqual(obj.pnts[0]._component_type, "kMeshVertComponent")
        self.assertEqual(len(obj.pnts[:]),            8)  # 8
        self.assertEqual(
            obj.pnts[:], [item.node.pnts[i] for i, item in enumerate(obj.pnts[:])]
        )

        # test nurbs curve, surf and lattice (all use "cp")
        counts   = [4, 56, 20]  # expected component counts
        att_type = ["kCurveCVComponent", "kSurfaceCVComponent", "kLatticeComponent"]
        curve = cmds.curve(
            per=True,
            p=[
                (0, 0, 0),
                (3, 5, 6),
                (5, 6, 7),
                (9, 9, 9),
                (0, 0, 0),
                (3, 5, 6),
                (5, 6, 7),
            ],
            k=[-2, -1, 0, 1, 2, 3, 4, 5, 6],
        )
        nurbs   = cmds.sphere()[0]
        lattice = cmds.lattice(nurbs)[1]

        for i, obj in enumerate([curve, nurbs, lattice]):
            obj = PyNode(cmds.listHistory(obj)[0])
            self.assertEqual(obj.cp._component_type,    att_type[i])
            self.assertEqual(obj.cp[0]._component_type, att_type[i])
            self.assertEqual(len(obj.cp[:]),            counts[i])
            self.assertEqual(
                obj.cp[:], [item.node.cp[i] for i, item in enumerate(obj.cp[:])]
            )


class TestAttributeGetGeometry(MayaTestCase):
    """
    Tests for `Attribute.get()`'s typed-geometry dispatch.

    Covers the four-tier resolution introduced for typed geometry data plugs
    (mesh, nurbsCurve, nurbsSurface):
      1. Numeric / string / matrix / compound values still pass through
         `cmds.getAttr()` exactly (connection state irrelevant).
      2. Plugs whose owning node is a SHAPE matching the geometry type return
         the shape itself (e.g. ``mesh.outMesh`` -> ``Mesh`` node).
      3. Plugs with a downstream consumer return the consumer node
         (e.g. ``skinCluster.outputGeometry[0]`` -> ``Mesh`` of the bound shape).
      4. Plugs with no downstream consumer return the matching OpenMaya fn set
         wrapping the plug's computed data (``MFnMesh`` / ``MFnNurbsCurve``).
      5. Plugs with no input feeding them return ``None``.

    Also verifies that:
      - ``cmds.getAttr`` is bypassed entirely for known geometry plugs (so the
        ``# Error: The data is not a numeric or string value...`` message is
        not emitted).
      - The geometry-typed-attr detection is cached per Attribute instance.
    """

    TEST_START_NEW_SCENE = True

    def _make_skinned_cube(self):
        """Returns (cube_xform, cube_shape, joint, skinCluster) names."""
        cube_xform = cmds.polyCube(name="pCube1", constructionHistory=True)[0]
        cube_shape = cmds.listRelatives(cube_xform, shapes=True)[0]
        joint      = cmds.joint(name="joint1")
        cmds.select([joint, cube_xform])
        skin = cmds.skinCluster(joint, cube_xform, name="skinCluster1")[0]
        return cube_xform, cube_shape, joint, skin

    def _make_skinned_curve(self):
        """Returns (curve_xform, curve_shape, joint, skinCluster) names."""
        crv = cmds.curve(
            name="curve1", p=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], degree=3
        )
        crv_shape = cmds.listRelatives(crv, shapes=True)[0]
        joint     = cmds.joint(name="curve_joint")
        cmds.select([joint, crv])
        skin = cmds.skinCluster(joint, crv, name="skinCluster_crv")[0]
        return crv, crv_shape, joint, skin

    # -- 1. Numeric / passthrough behavior is preserved

    def test_get_returns_number_for_free_numeric_attr(self):
        loc = cmds.spaceLocator(name="loc_free")[0]
        cmds.setAttr(f"{loc}.translateX", 1.5)
        self.assertEqual(PyNode(loc).translateX.get(), 1.5)

    def test_get_returns_computed_number_for_connected_numeric_attr(self):
        loc = cmds.spaceLocator(name="loc_conn")[0]
        md  = cmds.createNode("multiplyDivide", name="md1")
        cmds.setAttr(f"{md}.input1X", 3.5)
        cmds.setAttr(f"{md}.input2X", 2.0)
        cmds.connectAttr(f"{md}.outputX", f"{loc}.translateX", force=True)

        # connected attr should still return the computed number, not a node
        self.assertEqual(PyNode(loc).translateX.get(), 7.0)

    def test_get_returns_list_for_vector_attr(self):
        loc = cmds.spaceLocator(name="loc_vec")[0]
        cmds.setAttr(f"{loc}.translate", 1.0, 2.0, 3.0, type="double3")
        self.assertEqual(PyNode(loc).translate.get(), [(1.0, 2.0, 3.0)])

    def test_get_returns_list_for_matrix_attr(self):
        cube_xform, _, _, _ = self._make_skinned_cube()
        val = PyNode(cube_xform).worldMatrix[0].get()
        self.assertEqual(len(val), 16)
        self.assertEqual(val[0],   1.0)  # identity diagonal
        self.assertEqual(val[5],   1.0)
        self.assertEqual(val[10],  1.0)
        self.assertEqual(val[15],  1.0)

    # -- 2. Shape-owner returns the shape itself

    def test_get_returns_mesh_for_mesh_outMesh(self):
        _, cube_shape, _, _ = self._make_skinned_cube()
        result = PyNode(cube_shape).outMesh.get()
        # Should resolve to the owning shape (returned as PyNode().serialize()),
        # NOT MFnMesh wrapping data.
        self.assertNotIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result, PyNode(cube_shape).serialize())

    def test_get_returns_mesh_for_mesh_inMesh(self):
        _, cube_shape, _, _ = self._make_skinned_cube()
        result = PyNode(cube_shape).inMesh.get()
        self.assertNotIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result, PyNode(cube_shape).serialize())

    def test_get_returns_mesh_for_mesh_worldMesh(self):
        _, cube_shape, _, _ = self._make_skinned_cube()
        result = PyNode(cube_shape).worldMesh[0].get()
        self.assertNotIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result, PyNode(cube_shape).serialize())

    def test_get_returns_nurbscurve_for_curve_local(self):
        _, crv_shape, _, _ = self._make_skinned_curve()
        result = PyNode(crv_shape).local.get()
        self.assertNotIsInstance(result, OpenMaya.MFnNurbsCurve)
        self.assertEqual(result, PyNode(crv_shape).serialize())

    # -- 3. Output plug with downstream consumer returns the consumer node

    def test_get_follows_skincluster_output_to_mesh(self):
        _, cube_shape, _, skin = self._make_skinned_cube()
        result = PyNode(skin).outputGeometry[0].get()
        self.assertNotIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result, PyNode(cube_shape).serialize())

    def test_get_follows_skincluster_output_to_nurbscurve(self):
        _, crv_shape, _, skin = self._make_skinned_curve()
        result = PyNode(skin).outputGeometry[0].get()
        self.assertNotIsInstance(result, OpenMaya.MFnNurbsCurve)
        self.assertEqual(result, PyNode(crv_shape).serialize())

    # -- 4. Disconnected output returns the wrapping fn set with valid data

    def test_get_returns_mfnmesh_for_disconnected_skincluster_output(self):
        _, cube_shape, _, skin = self._make_skinned_cube()
        cmds.disconnectAttr(f"{skin}.outputGeometry[0]", f"{cube_shape}.inMesh")

        result = PyNode(skin).outputGeometry[0].get()
        self.assertIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result.numVertices, 8)
        self.assertEqual(result.numPolygons, 6)
        # Confirm the data is real (not uninitialized) -- vertex 0 of a unit
        # poly cube is at (-0.5, -0.5, 0.5).
        pts = result.getPoints()
        self.assertAlmostEqual(pts[0].x, -0.5, places=5)
        self.assertAlmostEqual(pts[0].y, -0.5, places=5)
        self.assertAlmostEqual(pts[0].z, 0.5,  places=5)

    def test_get_returns_mfnnurbscurve_for_disconnected_skincluster_output(self):
        _, crv_shape, _, skin = self._make_skinned_curve()
        cmds.disconnectAttr(f"{skin}.outputGeometry[0]", f"{crv_shape}.create")

        result = PyNode(skin).outputGeometry[0].get()
        self.assertIsInstance(result, OpenMaya.MFnNurbsCurve)
        self.assertEqual(result.numCVs, 4)

    # -- 5. Empty / orphan plug returns None

    def test_get_returns_none_for_bare_skincluster_output(self):
        bare   = cmds.createNode("skinCluster", name="skinCluster_bare")
        result = PyNode(bare).outputGeometry[0].get()
        self.assertIsNone(result)

    # -- behavioral guarantees

    def test_get_skips_cmds_getAttr_for_geometry_typed_attrs(self):
        """For geometry plugs, get() must never call cmds.getAttr without
        type=True. The value-fetch form is what triggers Maya's
        '# Error: The data is not a numeric or string value' emission.
        """
        _, _, _, skin = self._make_skinned_cube()
        attr = PyNode(skin).outputGeometry[0]

        original_getAttr = cmds.getAttr
        calls            = []

        def spy_getAttr(*args, **kwargs):
            calls.append((args, kwargs))
            return original_getAttr(*args, **kwargs)

        cmds.getAttr = spy_getAttr
        try:
            attr.get()
        finally:
            cmds.getAttr = original_getAttr

        # All cmds.getAttr calls during a geometry get() must use type=True
        # (the cache check). The bare-value form must never be called.
        self.assertTrue(
            all(kwargs.get("type") for _, kwargs in calls),
            f"cmds.getAttr was called without type=True: {calls!r}",
        )

    def test_get_caches_geometry_typed_attr_check(self):
        """Second get() on the same Attribute instance must not re-call
        cmds.getAttr(type=True) for the geometry-attr detection."""
        _, _, _, skin = self._make_skinned_cube()
        attr = PyNode(skin).outputGeometry[0]

        # warm the cache
        attr.get()

        original_getAttr = cmds.getAttr
        calls            = []

        def spy_getAttr(*args, **kwargs):
            calls.append((args, kwargs))
            return original_getAttr(*args, **kwargs)

        cmds.getAttr = spy_getAttr
        try:
            attr.get()  # second call -- cache should short-circuit type check
        finally:
            cmds.getAttr = original_getAttr

        # No cmds.getAttr should be needed at all for the second call:
        # geometry-typed cache hits, and the dispatch goes straight to the
        # OpenMaya-only resolution path.
        self.assertEqual(calls, [], f"Cache failed to short-circuit: {calls!r}")

    def test_get_raises_for_message_attr(self):
        """Message attrs preserve cmds.getAttr semantics -- they raise."""
        node = cmds.createNode("transform", name="msg_node")
        with self.assertRaises(RuntimeError):
            PyNode(node).message.get()

    # -- 6. Geometry-routing nodes (e.g. choice) -- step 2.5 of the dispatch

    def _make_choice_node(self):
        """Build a choice node with input[0]=curve, input[1]=mesh.

        Returns (choice_node_name, curve_shape_name, mesh_shape_name).
        """
        crv = cmds.curve(
            name   = "choice_curve",
            p      = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            degree = 3,
        )
        crv_shape  = cmds.listRelatives(crv, shapes=True)[0]
        cube       = cmds.polyCube(name="choice_cube", constructionHistory=False)[0]
        mesh_shape = cmds.listRelatives(cube, shapes=True)[0]
        choice     = cmds.createNode("choice", name="choice1")
        cmds.connectAttr(f"{crv_shape}.worldSpace[0]", f"{choice}.input[0]")
        cmds.connectAttr(f"{mesh_shape}.outMesh", f"{choice}.input[1]")
        return choice, crv_shape, mesh_shape

    def test_get_choice_with_selector_0_returns_nurbscurve(self):
        """choice.output with selector=0 -> NurbsCurve of the connected curve shape."""
        choice, crv_shape, _ = self._make_choice_node()
        cmds.setAttr(f"{choice}.selector", 0)
        result = PyNode(choice).output.get()
        self.assertNotIsInstance(result, OpenMaya.MFnNurbsCurve)
        self.assertEqual(result, PyNode(crv_shape).serialize())

    def test_get_choice_with_selector_1_returns_mesh(self):
        """choice.output with selector=1 -> Mesh of the connected mesh shape."""
        choice, _, mesh_shape = self._make_choice_node()
        cmds.setAttr(f"{choice}.selector", 1)
        result = PyNode(choice).output.get()
        self.assertNotIsInstance(result, OpenMaya.MFnMesh)
        self.assertEqual(result, PyNode(mesh_shape).serialize())

    def test_get_choice_changes_with_selector(self):
        """Flipping the selector swaps the returned node type."""
        choice, crv_shape, mesh_shape = self._make_choice_node()
        choice_node = PyNode(choice)

        choice_node.selector.set(0)
        first = choice_node.output.get()
        self.assertEqual(first, PyNode(crv_shape).serialize())

        choice_node.selector.set(1)
        second = choice_node.output.get()
        self.assertEqual(second, PyNode(mesh_shape).serialize())

    def test_get_choice_destination_takes_priority_over_routing(self):
        """When choice.output is connected downstream, step 2 (downstream
        consumer) wins over step 2.5 (upstream routing trace).

        Note: source mesh and consumer mesh have IDENTICAL data after the
        connection (consumer.inMesh is fed by the choice.output that comes from
        the source). MeshData.__eq__ is value-based, so we can't distinguish
        the two paths via data equality alone -- instead we check that the
        identity reference (the shape name) embedded in the repr corresponds
        to the consumer.
        """
        choice, _, mesh_shape = self._make_choice_node()
        cmds.setAttr(f"{choice}.selector", 1)

        # Hook up choice.output to an empty mesh's inMesh -- that empty mesh
        # becomes the downstream consumer.
        consumer_xform = cmds.createNode("transform", name="consumer_xform")
        consumer_shape = cmds.createNode(
            "mesh", name="consumer_shape", parent=consumer_xform
        )
        cmds.connectAttr(f"{choice}.output", f"{consumer_shape}.inMesh")

        result = PyNode(choice).output.get()
        # Result data must equal the consumer's serialized form.
        self.assertEqual(result, PyNode(consumer_shape).serialize())
        # Identity check via the shape name in the repr -- the dispatch must
        # have returned the consumer node, not the upstream source.
        self.assertIn(consumer_shape, repr(result))
        self.assertNotIn(mesh_shape, repr(result))

    def test_get_choice_with_no_input_connected_returns_mfn_or_none(self):
        """A choice node with the selected input slot empty falls through past
        step 2.5 and either wraps the (uninitialized) data in MFn or returns
        None -- the routing tracer must not raise."""
        choice = cmds.createNode("choice", name="choice_empty")
        # selector defaults to 0; input[0] is not connected
        # Should not raise; returns None or MFn -- both are acceptable for this
        # degenerate scene.
        result = PyNode(choice).output.get()
        self.assertTrue(
            result is None
            or isinstance(result, (OpenMaya.MFnMesh, OpenMaya.MFnNurbsCurve)),
            f"Unexpected return for empty choice: {result!r} ({type(result).__name__})",
        )

    def test_geometry_routing_tracers_contains_choice(self):
        """The public dispatch table must register the choice tracer so users
        can extend it for their own routing nodes."""
        from rig.nodetypes._base import GEOMETRY_ROUTING_TRACERS

        self.assertIn("choice", GEOMETRY_ROUTING_TRACERS)
        self.assertTrue(callable(GEOMETRY_ROUTING_TRACERS["choice"]))

    # -- 7. Polymorphic choice output (geometry + non-geometry types)

    def _make_polymorphic_choice_node(self):
        """Build a choice with input[0]=curve, input[1]=mesh, input[2]=matrix.

        Returns (choice_node, curve_shape, mesh_shape, transform_with_matrix).
        """
        crv = cmds.curve(
            name   = "poly_curve",
            p      = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            degree = 3,
        )
        crv_shape  = cmds.listRelatives(crv, shapes=True)[0]
        cube       = cmds.polyCube(name="poly_cube", constructionHistory=False)[0]
        mesh_shape = cmds.listRelatives(cube, shapes=True)[0]
        xform      = cmds.createNode("transform", name="poly_xform")
        cmds.setAttr(f"{xform}.translateX", 1.5)

        choice = cmds.createNode("choice", name="poly_choice")
        cmds.connectAttr(f"{crv_shape}.worldSpace[0]", f"{choice}.input[0]")
        cmds.connectAttr(f"{mesh_shape}.outMesh",      f"{choice}.input[1]")
        cmds.connectAttr(f"{xform}.matrix",            f"{choice}.input[2]")
        return choice, crv_shape, mesh_shape, xform

    def test_get_choice_with_matrix_input_returns_matrix_list(self):
        """choice.output with selector pointing at a matrix input should return
        the 16-float matrix list, NOT None and NOT a node wrapper."""
        choice, _, _, _ = self._make_polymorphic_choice_node()
        cmds.setAttr(f"{choice}.selector", 2)
        result = PyNode(choice).output.get()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 16)

    def test_get_choice_recovers_after_geometry_then_matrix(self):
        """Regression: previously the boolean cache `_geometry_attr_cache` was
        set to True on the first geometry get(), causing all subsequent get()
        calls (including matrix selector) to take the geometry path and return
        None. With polymorphic-aware caching, flipping the selector recovers
        the correct value type each time.
        """
        choice, crv_shape, _, _ = self._make_polymorphic_choice_node()
        choice_node = PyNode(choice)

        # First call: geometry -- sets the (potentially poisonous) cache state
        choice_node.selector.set(0)
        first = choice_node.output.get()
        self.assertEqual(first, PyNode(crv_shape).serialize())

        # Second call on same Attribute instance: matrix -- must NOT inherit the
        # geometry-True cache. Should return the matrix value.
        choice_node.selector.set(2)
        second = choice_node.output.get()
        self.assertIsInstance(second, list)
        self.assertEqual(len(second), 16)

    def test_get_choice_recovers_after_matrix_then_geometry(self):
        """Symmetric regression: starting with a matrix selector then flipping
        to geometry must also work (cache must not lock in 'False')."""
        choice, _, mesh_shape, _ = self._make_polymorphic_choice_node()
        choice_node = PyNode(choice)

        choice_node.selector.set(2)
        first = choice_node.output.get()
        self.assertIsInstance(first, list)
        self.assertEqual(len(first), 16)

        choice_node.selector.set(1)
        second = choice_node.output.get()
        self.assertEqual(second, PyNode(mesh_shape).serialize())

    def test_get_choice_does_not_cache_geometry_check(self):
        """For polymorphic-output nodes (choice), `cmds.getAttr type=True` must
        be re-called on every get() -- the cache short-circuit only applies to
        non-polymorphic owners."""
        choice, _, _, _ = self._make_polymorphic_choice_node()
        attr = PyNode(choice).output

        # First get() to warm any internal state
        cmds.setAttr(f"{choice}.selector", 0)
        attr.get()

        # Now spy on cmds.getAttr for the second call
        original_getAttr = cmds.getAttr
        calls            = []

        def spy_getAttr(*args, **kwargs):
            calls.append((args, kwargs))
            return original_getAttr(*args, **kwargs)

        cmds.getAttr = spy_getAttr
        try:
            cmds.setAttr(f"{choice}.selector", 2)
            attr.get()
        finally:
            cmds.getAttr = original_getAttr

        # type=True must have been called (no cache short-circuit) so the
        # dispatch sees the new 'matrix' type.
        type_calls = [c for c in calls if c[1].get("type")]
        self.assertGreaterEqual(
            len(type_calls),
            1,
            f"Polymorphic choice should re-check type=True per get(): {calls!r}",
        )

    def test_polymorphic_output_node_types_contains_choice(self):
        """The public set of polymorphic-output types must include `choice`."""
        from rig.nodetypes._base import POLYMORPHIC_OUTPUT_NODE_TYPES

        self.assertIn("choice", POLYMORPHIC_OUTPUT_NODE_TYPES)


class TestPolymorphicChoiceCharacterization(MayaTestCase):
    """Characterization tests for polymorphic ``choice`` node behavior.

    These tests document the CURRENT observable behavior of ``choice.output``
    plug evaluation, including:

      - Raw Maya API behavior (``cmds.getAttr``, ``MPlug.asMObject``,
        ``MPlug.source``, ``MPlug.destinations``)
      - ``Attribute.get()`` dispatch under polymorphic typing across input
        types (geometry, scalar, compound, matrix, string, empty)
      - Cache invariants (``_geometry_attr_cache``, ``_owner_is_polymorphic``)
      - Selector edge cases (out-of-range, empty slot, minimum value)
      - Dispatch ordering between shape-owner / destinations / routing /
        MFn-wrap paths

    Purpose: serve as a regression safety net before any future refactor of
    the polymorphic dispatch logic. If a test fails after a change, the
    behavior contract has changed -- review carefully before updating the test.
    """

    TEST_START_NEW_SCENE = True

    # -------- helpers --------

    @staticmethod
    def _get_plug(node, attr):
        from maya.api import OpenMaya

        sel = OpenMaya.MSelectionList()
        sel.add(f"{node}.{attr}")
        return sel.getPlug(0)

    def _make_choice_with_inputs(self, inputs_by_index=None):
        """Build a choice node with the given inputs.

        ``inputs_by_index`` is a dict mapping logical index -> source plug
        name string. Returns the choice node name.
        """
        choice = cmds.createNode("choice", name="ch_char")
        for idx, source_attr in (inputs_by_index or {}).items():
            cmds.connectAttr(source_attr, f"{choice}.input[{idx}]")
        return choice

    def _make_curve_shape(self, name="curve_char"):
        crv = cmds.curve(
            name   = name,
            p      = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            degree = 3,
        )
        return cmds.listRelatives(crv, shapes=True)[0]

    def _make_mesh_shape(self, name="mesh_char"):
        cube = cmds.polyCube(name=name, constructionHistory=False)[0]
        return cmds.listRelatives(cube, shapes=True)[0]

    # ============================================================
    # SECTION A -- Raw Maya API behavior on the choice.output plug
    # ============================================================

    def test_chr_choice_output_source_is_always_null(self):
        """`choice.output.source()` is null regardless of inputs/selector.

        Choice computes its output dynamically; there is no direct DG
        connection on the output plug itself.
        """
        loc = cmds.spaceLocator(name="loc_char")[0]
        choice = self._make_choice_with_inputs(
            {0: f"{loc}.translate", 1: f"{loc}.matrix"}
        )
        out_plug = self._get_plug(choice, "output")

        for sel in (0, 1):
            cmds.setAttr(f"{choice}.selector", sel)
            self.assertTrue(
                out_plug.source().isNull,
                f"source() unexpectedly non-null for selector={sel}",
            )

    def test_chr_choice_output_destinations_empty_when_unconnected(self):
        """`choice.output.destinations()` is empty before downstream connect."""
        loc    = cmds.spaceLocator(name="loc_char")[0]
        choice = self._make_choice_with_inputs({0: f"{loc}.translate"})
        # MPlug.destinations() returns an MPlugArray, which has no built-in
        # equality with []; convert to list before comparing.
        self.assertEqual(list(self._get_plug(choice, "output").destinations()), [])

    def test_chr_choice_output_destinations_lists_consumer_plug(self):
        """`destinations()` returns the consumer plug after connect."""
        loc1   = cmds.spaceLocator(name="loc_in_char")[0]
        loc2   = cmds.spaceLocator(name="loc_out_char")[0]
        choice = self._make_choice_with_inputs({0: f"{loc1}.translate"})
        cmds.connectAttr(f"{choice}.output", f"{loc2}.translate")

        dests      = self._get_plug(choice, "output").destinations()
        dest_names = [p.name() for p in dests]
        self.assertEqual(dest_names, [f"{loc2}.translate"])

    def test_chr_empty_choice_output_type_is_Tdata(self):
        """A bare choice (no inputs) reports type 'Tdata' via cmds.getAttr.

        Notable: 'Tdata' is NOT in ``GEOMETRY_DATA_TYPES``, so `Attribute.get()`
        falls through to the cmds.getAttr passthrough path.
        """
        choice = self._make_choice_with_inputs()
        self.assertEqual(cmds.getAttr(f"{choice}.output", type=True), "Tdata")

    def test_chr_empty_choice_output_value_is_None(self):
        """A bare choice (no inputs) returns None from cmds.getAttr."""
        choice = self._make_choice_with_inputs()
        self.assertIsNone(cmds.getAttr(f"{choice}.output"))

    def test_chr_empty_choice_output_asMObject_raises(self):
        """A bare choice (no inputs) raises RuntimeError from MPlug.asMObject.

        This is the documented failure mode that ``_get_geometry_value()``
        guards against with try/except RuntimeError.
        """
        choice = self._make_choice_with_inputs()
        with self.assertRaises(RuntimeError):
            self._get_plug(choice, "output").asMObject()

    def test_chr_choice_output_type_changes_per_selector(self):
        """``cmds.getAttr type=True`` reflects the CURRENTLY selected input.

        This is the entire reason why ``_geometry_attr_cache`` cannot be
        memoized for polymorphic owners.
        """
        crv_shape  = self._make_curve_shape()
        mesh_shape = self._make_mesh_shape()
        loc        = cmds.spaceLocator(name="loc_mtx_char")[0]
        choice = self._make_choice_with_inputs(
            {
                0: f"{crv_shape}.worldSpace[0]",
                1: f"{mesh_shape}.outMesh",
                2: f"{loc}.matrix",
            }
        )
        observed = []
        for sel in (0, 1, 2):
            cmds.setAttr(f"{choice}.selector", sel)
            observed.append(cmds.getAttr(f"{choice}.output", type=True))
        self.assertEqual(observed, ["nurbsCurve", "mesh", "matrix"])

    def test_chr_selector_out_of_range_returns_None_with_no_prior_eval(self):
        """Selector larger than the highest connected input index returns None
        from cmds.getAttr when no prior evaluation has populated the cache.

        Note: in some Maya states (after a prior valid evaluation) Maya retains
        the LAST evaluated value as a cached fallback. The behavior is
        order-dependent and effectively undefined for callers -- exercise care
        when seeing a non-None result for an out-of-range selector.
        """
        loc = cmds.spaceLocator(name="loc_clamp")[0]
        choice = self._make_choice_with_inputs(
            {0: f"{loc}.translateX", 1: f"{loc}.translateY"}
        )
        # Fresh choice + selector past last index, no prior evaluation
        cmds.setAttr(f"{choice}.selector", 99)
        self.assertIsNone(cmds.getAttr(f"{choice}.output"))

    def test_chr_selector_minimum_is_zero_negative_rejected(self):
        """The choice node defines selector with a hard minimum of 0;
        ``cmds.setAttr`` raises RuntimeError for negative values.
        """
        choice = self._make_choice_with_inputs()
        with self.assertRaises(RuntimeError):
            cmds.setAttr(f"{choice}.selector", -1)

    def test_chr_selector_at_unconnected_slot_reports_Tdata(self):
        """Selector pointing at an input slot with no source connection
        behaves the same as an empty choice: type='Tdata', value=None.
        """
        loc = cmds.spaceLocator(name="loc_sparse")[0]
        # Connect only input[5] -- leave [0..4] unconnected.
        choice = self._make_choice_with_inputs({5: f"{loc}.translateX"})
        cmds.setAttr(f"{choice}.selector", 2)  # empty slot
        self.assertEqual(cmds.getAttr(f"{choice}.output", type=True), "Tdata")
        self.assertIsNone(cmds.getAttr(f"{choice}.output"))

    # ============================================================
    # SECTION B -- Attribute.get() dispatch for non-geometry inputs
    # ============================================================

    def test_chr_get_returns_double_value_for_double_input(self):
        """choice with a double input -> cmds.getAttr passthrough -> float."""
        loc = cmds.spaceLocator(name="loc_double")[0]
        cmds.setAttr(f"{loc}.translateX", 2.5)
        choice = self._make_choice_with_inputs({0: f"{loc}.translateX"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        self.assertIsInstance(result, float)
        self.assertEqual(result, 2.5)

    def test_chr_get_returns_long_input_as_float(self):
        """Maya reports long inputs through choice as 'typed' and returns the
        value as a float (not an int).
        """
        src = cmds.createNode("transform", name="src_long")
        cmds.addAttr(src, longName="num", attributeType="long", defaultValue=42)
        choice = self._make_choice_with_inputs({0: f"{src}.num"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        self.assertIsInstance(result, float)
        self.assertEqual(result, 42.0)

    def test_chr_get_returns_bool_input_as_float(self):
        """Maya reports bool inputs through choice as 'typed' and returns 0.0/1.0
        (not Python True/False).
        """
        src = cmds.createNode("transform", name="src_bool")
        cmds.addAttr(src, longName="flag", attributeType="bool", defaultValue=True)
        choice = self._make_choice_with_inputs({0: f"{src}.flag"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        self.assertEqual(result, 1.0)

    def test_chr_get_returns_string_for_string_input(self):
        """choice with a string input returns the string value verbatim."""
        src = cmds.createNode("transform", name="src_str")
        cmds.addAttr(src, longName="label", dataType="string")
        cmds.setAttr(f"{src}.label", "hello", type="string")
        choice = self._make_choice_with_inputs({0: f"{src}.label"})
        cmds.setAttr(f"{choice}.selector", 0)

        self.assertEqual(PyNode(choice).output.get(), "hello")

    def test_chr_get_returns_compound_for_vector_input(self):
        """choice with a compound (translate) input returns the cmds.getAttr
        list-of-tuples form.
        """
        loc = cmds.spaceLocator(name="loc_vec_char")[0]
        cmds.setAttr(f"{loc}.translate", 1.0, 2.0, 3.0, type="double3")
        choice = self._make_choice_with_inputs({0: f"{loc}.translate"})
        cmds.setAttr(f"{choice}.selector", 0)

        self.assertEqual(PyNode(choice).output.get(), [(1.0, 2.0, 3.0)])

    def test_chr_get_returns_matrix_list_for_matrix_input(self):
        """choice with a matrix input returns a 16-float list (NOT None)."""
        loc    = cmds.spaceLocator(name="loc_mtx_char")[0]
        choice = self._make_choice_with_inputs({0: f"{loc}.matrix"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 16)

    def test_chr_get_returns_None_for_empty_choice(self):
        """choice with NO inputs returns None from get()."""
        choice = self._make_choice_with_inputs()
        self.assertIsNone(PyNode(choice).output.get())

    def test_chr_get_returns_None_for_unconnected_selector_slot(self):
        """Selector pointing at empty slot returns None from get()."""
        loc    = cmds.spaceLocator(name="loc_sparse2")[0]
        choice = self._make_choice_with_inputs({5: f"{loc}.translateX"})
        cmds.setAttr(f"{choice}.selector", 2)
        self.assertIsNone(PyNode(choice).output.get())

    # ============================================================
    # SECTION C -- Geometry routing dispatch (the user-reported case)
    # ============================================================

    def test_chr_get_returns_serialized_curve_for_curve_input(self):
        """selector points at a NURBS-curve input -> result equals the source
        curve shape's serialized form.
        """
        crv_shape = self._make_curve_shape("curve_chr")
        choice    = self._make_choice_with_inputs({0: f"{crv_shape}.worldSpace[0]"})
        cmds.setAttr(f"{choice}.selector", 0)
        self.assertEqual(PyNode(choice).output.get(), PyNode(crv_shape).serialize())

    def test_chr_get_returns_serialized_mesh_for_mesh_input(self):
        """selector points at a mesh input -> result equals the source mesh
        shape's serialized form.
        """
        mesh_shape = self._make_mesh_shape("mesh_chr")
        choice     = self._make_choice_with_inputs({0: f"{mesh_shape}.outMesh"})
        cmds.setAttr(f"{choice}.selector", 0)
        self.assertEqual(PyNode(choice).output.get(), PyNode(mesh_shape).serialize())

    def test_chr_get_returns_serialized_surface_for_nurbs_surface_input(self):
        """selector points at a NURBS surface input -> result equals the source
        surface shape's serialized form.
        """
        sphere     = cmds.sphere(name="sphere_chr")[0]
        surf_shape = cmds.listRelatives(sphere, shapes=True)[0]
        choice     = self._make_choice_with_inputs({0: f"{surf_shape}.worldSpace[0]"})
        cmds.setAttr(f"{choice}.selector", 0)
        self.assertEqual(PyNode(choice).output.get(), PyNode(surf_shape).serialize())

    # ============================================================
    # SECTION D -- Cache invariants
    # ============================================================

    def test_chr_polymorphic_owner_cache_starts_None(self):
        """`_polymorphic_owner_cache` is None on a fresh Attribute (lazy
        evaluation).
        """
        loc    = cmds.spaceLocator(name="loc_cache_a")[0]
        choice = self._make_choice_with_inputs({0: f"{loc}.translateX"})
        attr   = PyNode(choice).output
        self.assertIsNone(attr._polymorphic_owner_cache)

    def test_chr_polymorphic_owner_cache_set_after_first_check(self):
        """Touching `_owner_is_polymorphic` populates the cache to True for
        a choice node.
        """
        loc    = cmds.spaceLocator(name="loc_cache_b")[0]
        choice = self._make_choice_with_inputs({0: f"{loc}.translateX"})
        attr   = PyNode(choice).output
        _      = attr._owner_is_polymorphic
        self.assertEqual(attr._polymorphic_owner_cache, True)

    def test_chr_polymorphic_owner_cache_False_for_non_polymorphic(self):
        """Non-polymorphic owners (e.g. a plain mesh) cache False."""
        mesh_shape = self._make_mesh_shape("mesh_cache")
        attr       = PyNode(mesh_shape).outMesh
        _          = attr._owner_is_polymorphic
        self.assertEqual(attr._polymorphic_owner_cache, False)

    def test_chr_geometry_attr_cache_never_set_for_polymorphic(self):
        """For a polymorphic owner the bool cache is never populated, even
        after several get() calls.
        """
        crv_shape = self._make_curve_shape("crv_cache")
        choice    = self._make_choice_with_inputs({0: f"{crv_shape}.worldSpace[0]"})
        attr      = PyNode(choice).output
        for _ in range(3):
            cmds.setAttr(f"{choice}.selector", 0)
            attr.get()
        self.assertIsNone(attr._geometry_attr_cache)

    def test_chr_geometry_attr_cache_set_True_for_non_polymorphic_geometry(self):
        """For a non-polymorphic GEOMETRY owner, the bool cache populates True
        after the first get().
        """
        cube_x = cmds.polyCube(name="cube_cache", constructionHistory=True)[0]
        joint  = cmds.joint(name="j_cache")
        cmds.select([joint, cube_x])
        skin = cmds.skinCluster(joint, cube_x, name="skin_cache")[0]
        attr = PyNode(skin).outputGeometry[0]
        attr.get()
        self.assertEqual(attr._geometry_attr_cache, True)

    def test_chr_geometry_attr_cache_set_False_for_non_polymorphic_numeric(self):
        """For a non-polymorphic NUMERIC owner, the bool cache populates False
        after the first get().
        """
        loc  = cmds.spaceLocator(name="loc_numcache")[0]
        attr = PyNode(loc).translateX
        attr.get()
        self.assertEqual(attr._geometry_attr_cache, False)

    def test_chr_polymorphic_owner_cache_persists_across_selector_changes(self):
        """The polymorphic-owner classification does not change when the
        selector changes -- it's a property of the owning node type.
        """
        crv_shape = self._make_curve_shape("crv_pers")
        loc       = cmds.spaceLocator(name="loc_pers")[0]
        choice = self._make_choice_with_inputs(
            {0: f"{crv_shape}.worldSpace[0]", 1: f"{loc}.matrix"}
        )
        attr = PyNode(choice).output
        for sel in (0, 1, 0, 1, 0):
            cmds.setAttr(f"{choice}.selector", sel)
            attr.get()
            self.assertEqual(attr._polymorphic_owner_cache, True)

    # ============================================================
    # SECTION E -- Dispatch ordering between paths
    # ============================================================

    def test_chr_dispatch_order_destinations_wins_over_routing_for_geometry(self):
        """When a polymorphic owner's geometry output BOTH has a downstream
        consumer AND a routable upstream input, the downstream consumer is
        returned (step 2 wins over step 2.5).
        """
        mesh_shape = self._make_mesh_shape("mesh_order")
        choice     = self._make_choice_with_inputs({0: f"{mesh_shape}.outMesh"})
        cmds.setAttr(f"{choice}.selector", 0)

        consumer_xform = cmds.createNode("transform", name="consumer_xform_o")
        consumer_shape = cmds.createNode(
            "mesh", name="consumer_shape_o", parent=consumer_xform
        )
        cmds.connectAttr(f"{choice}.output", f"{consumer_shape}.inMesh")

        result = PyNode(choice).output.get()
        # The dispatch must have returned the CONSUMER (identity check via
        # the shape name in the repr). Data equality alone can't distinguish
        # since both have identical mesh data.
        self.assertIn(consumer_shape, repr(result))
        self.assertNotIn(mesh_shape, repr(result))

    def test_chr_routing_returns_serialized_shape_when_upstream_is_shape(self):
        """When choice.output is unconnected downstream and the selected
        input's source is a SHAPE matching the data type, the routing tracer
        returns that shape's serialized form.
        """
        mesh_shape = self._make_mesh_shape("mesh_route_shape")
        choice     = self._make_choice_with_inputs({0: f"{mesh_shape}.outMesh"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        self.assertEqual(result, PyNode(mesh_shape).serialize())

    def test_chr_routing_returns_None_when_upstream_is_not_shape(self):
        """When the selected input's source is NOT a shape (e.g. another
        deformer's output), the routing tracer falls through. Without a
        downstream consumer the data is wrapped in MFn for matched data
        types or returns None for an unmatched mismatch.
        """
        # Build a skinCluster->choice chain. The choice's input source is the
        # skinCluster (NOT a shape).
        cube_x = cmds.polyCube(name="cube_route", constructionHistory=True)[0]
        cube_s = cmds.listRelatives(cube_x, shapes=True)[0]
        joint  = cmds.joint(name="j_route")
        cmds.select([joint, cube_x])
        skin = cmds.skinCluster(joint, cube_x, name="skin_route")[0]
        cmds.disconnectAttr(f"{skin}.outputGeometry[0]", f"{cube_s}.inMesh")

        choice = self._make_choice_with_inputs({0: f"{skin}.outputGeometry[0]"})
        cmds.setAttr(f"{choice}.selector", 0)

        result = PyNode(choice).output.get()
        # Source of input[0] is skinCluster (not a shape) -> routing returns
        # nothing -> step 3 fires and wraps data in MFnMesh.
        self.assertIsInstance(result, OpenMaya.MFnMesh)

    def test_chr_routing_falls_through_when_chained_choice_source(self):
        """For chained choices (choice -> choice), the inner choice's output is
        not a shape, so the outer routing tracer falls through to MFn-wrap.
        """
        mesh_shape = self._make_mesh_shape("mesh_chain_chr")
        ch1        = self._make_choice_with_inputs({0: f"{mesh_shape}.outMesh"})
        # Build the second choice manually with input[0] from ch1.output
        ch2 = cmds.createNode("choice", name="ch2_chain_chr")
        cmds.connectAttr(f"{ch1}.output", f"{ch2}.input[0]")
        cmds.setAttr(f"{ch1}.selector", 0)
        cmds.setAttr(f"{ch2}.selector", 0)

        result = PyNode(ch2).output.get()
        # ch2.input[0].source().node() is ch1 (not a shape) -> routing falls
        # through; ch2.output.asMObject() returns valid kMeshData -> MFnMesh.
        self.assertIsInstance(result, OpenMaya.MFnMesh)