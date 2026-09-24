from maya import cmds
from rig.nodetypes import PyNode, Transform
from rig._tests._base import MayaTestCase


class TestBaseNodes(MayaTestCase):
    """
    DGNode & DAGNode unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.node_name = "test_xform"

        # make 3 transform nodes, 2 with the same name
        self.parent = PyNode.create("transform", name=self.node_name)
        self.child  = PyNode.create("transform", parent=self.parent)
        self.child.rename(self.node_name)

    def test_uuid(self):
        # checks if a PyNode object can be created from a uuid string
        node_from_uid = PyNode(self.parent.uuid)
        self.assertEqual(self.parent, node_from_uid)

    def test_properties(self):
        # casting node name
        self.assertEqual(PyNode("|" + self.node_name), self.parent)
        self.assertEqual(PyNode(self.parent.mobject),  self.parent)
        self.assertEqual(PyNode(self.parent.mdagpath), self.parent)

        # name
        self.assertEqual(self.child.short_name, self.parent.short_name, self.node_name)
        self.assertNotEqual(self.child.name,      self.parent.name)
        self.assertNotEqual(self.child.long_name, self.parent.long_name)
        self.assertNotEqual(self.child,           self.parent)

        # type and search
        joint = cmds.createNode("joint")
        self.assertFalse(self.child.is_type(joint))

        all_xforms = PyNode.find_all("transform")
        self.assertFalse(PyNode(joint) in all_xforms)
        for each in (self.parent, self.child):
            self.assertTrue(each in all_xforms)
        all_xforms = PyNode.find_all("transform", exact_type=False)
        for each in (PyNode(joint), self.parent, self.child):
            self.assertTrue(each in all_xforms)

    def test_delete(self):
        self.parent.delete()
        self.assertFalse(cmds.objExists(self.node_name))
        self.assertFalse(self.parent.is_valid)
        self.assertFalse(self.child.is_valid)

    def test_attr(self):
        attr_name     = "test_attr"
        new_attr_name = "aaa"

        # rename attr
        attr = self.child.add_attr(attr_name, dataType="doubleArray")
        self.assertEqual(attr.name, attr_name)
        self.child.rename_attr(attr_name, new_attr_name)
        self.assertEqual(attr.name, new_attr_name)
        self.assertEqual(self.child.list_attr(userDefined=True), [attr])

        # delete attr
        self.assertTrue(self.child.delete_attr(new_attr_name))
        self.assertEqual(self.child.list_attr(userDefined=True), [])
        self.assertFalse(self.child.has_attr(new_attr_name))

    def test_duplicate(self):
        dup = self.parent.duplicate(returnRootsOnly=True)
        self.assertEqual(len(dup), 1)
        self.assertTrue(dup[0].short_name.startswith(self.parent.short_name))
        self.assertEqual(dup[0].get_children()[0].short_name, self.child.short_name)

        dup = self.parent.duplicate(renameChildren=True)
        self.assertEqual(len(dup), 2)
        self.assertTrue(dup[0].short_name.startswith(self.parent.short_name))
        self.assertNotEqual(dup[1].short_name, self.child.short_name)

    def test_hier(self):
        child2 = PyNode.create("transform")
        child2.set_parent(self.child)

        self.assertEqual(self.parent.get_children(), [self.child])
        self.assertEqual(set(self.parent.get_children(ad=True)), {self.child, child2})
        self.assertEqual(self.child.get_parent(), self.parent)
        self.assertEqual(child2.get_parent(), self.child)

        self.assertEqual(child2.get_parent(1), self.parent)
        self.assertEqual(child2.get_parents(), [self.child, self.parent])

    def test_custom_node(self):
        custom_type_name = "testNodeType"

        class TestNode(Transform):
            CUSTOM_NODE_TYPE = custom_type_name

        c_node_name = "test_custom_node"
        c_node      = TestNode.create(name=c_node_name)
        c_node2     = PyNode.create(custom_type_name)

        self.assertEqual(set(PyNode.find_all(custom_type_name)), {c_node, c_node2})

        self.assertFalse(Transform.is_type(c_node_name))
        self.assertTrue(Transform.is_type(c_node_name, exact_type=False))
        self.assertTrue(TestNode.is_type(c_node_name))
        self.assertTrue(TestNode.is_type(c_node_name, exact_type=False))
        self.assertFalse(TestNode.is_type(self.child.name))
        self.assertFalse(TestNode.is_type(self.child.name, exact_type=False))

    def test_connections(self):
        joint = PyNode.create("joint")
        self.parent.tx >> self.child.ty
        self.child.ty  >> joint.tz

        self.assertEqual(
            joint.find_connected_nodes(depth=0, source=True, destination=False),
            [self.child],
        )

        for i in range(1, 4, 1):
            nodes = joint.find_connected_nodes(depth=i, source=True, destination=False)
            self.assertEqual(nodes, [self.child, self.parent])

        nodes = self.parent.find_connected_nodes(
            depth=0, source=False, destination=True
        )
        self.assertEqual(nodes, [self.child])

        for i in range(1, 4, 1):
            nodes = self.parent.find_connected_nodes(
                depth=i, source=False, destination=True
            )
            self.assertEqual(nodes, [self.child, joint])

        nodes = self.parent.find_connected_nodes(
            depth=4, source=False, destination=True, node_type="joint"
        )
        self.assertEqual(nodes, [joint])

    def test_exists(self):
        self.assertFalse(Transform.exists("abc"))
        cmds.createNode("objectSet", name="abc")
        self.assertFalse(Transform.exists("abc"))
        cmds.delete("abc")
        cmds.createNode("transform", name="abc")
        self.assertTrue(Transform.exists("abc"))
        cmds.delete("abc")
        cmds.createNode("joint", name="abc")
        self.assertFalse(Transform.exists("abc"))


class TestListAttrCompoundChildren(MayaTestCase):
    """``DGNode.list_attr()`` must not raise on multi-of-compound child
    names like ``publishedNodeInfo.publishedNode``.

    Regression: pre-fix, ``cmds.listAttr`` would return
    ``"publishedNodeInfo.publishedNode"`` (the compound child name) and
    ``find_attr("publishedNodeInfo.publishedNode")`` would call
    ``MSelectionList.add(node.publishedNodeInfo.publishedNode)`` which
    Maya rejects because ``publishedNodeInfo`` is a multi attribute and
    requires an element index. The fix passes ``quiet=True`` and skips
    the unresolvable child names.
    """

    def setUp(self):
        super().setUp()
        # transform nodes have publishedNodeInfo (multi-of-compound) which
        # triggers the bug.
        self._node_name = cmds.createNode("transform", name="cube_listattr")
        self._node      = PyNode(self._node_name)

    def tearDown(self):
        super().tearDown()
        if cmds.objExists(self._node_name):
            cmds.delete(self._node_name)

    def test_list_attr_does_not_raise_on_compound_children(self):
        # Pre-fix: would raise ``AttributeError: Attribute not found:
        # cube_listattr.publishedNodeInfo.publishedNode``.
        # Post-fix: returns a non-empty list of resolvable Attribute
        # objects, silently skipping the unresolvable compound children.
        attrs = self._node.list_attr()
        self.assertGreater(len(attrs), 0)

    def test_list_attr_skips_unresolvable_children(self):
        # Verify no None values leak into the returned list.
        attrs = self._node.list_attr()
        self.assertTrue(all(a is not None for a in attrs))


class TestComponentAliasResolution(MayaTestCase):
    """``DGNode.find_attr`` must respect node type when resolving Maya
    component aliases (``vtx``, ``cv``, ``pt``, ``pnts``, ``map``, ``uv``).

    Regression: pre-fix, the static ``_COMPONENT_ALIASES`` fallback would
    resolve any alias in the dict to its canonical plug name (e.g.
    ``cv -> controlPoints``) regardless of node type -- so ``mesh.cv``
    (a nurbsCurve / nurbsSurface alias) silently succeeded instead of
    failing. The fix uses ``cmds.listAttr(f"{node}.{alias}[0]")`` which
    Maya itself rejects when the alias is not valid for the node type.

    Per Maya docs:
    * ``vtx`` is the mesh vertex alias.
    * ``cv`` is the nurbsCurve / nurbsSurface CV alias.
    * ``pt`` is the lattice point alias (also recognised on mesh as a
      synonym for ``pnts``).
    * ``pnts`` is the mesh-specific points attribute.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cube_xform, _ = cmds.polyCube(name="alias_cube")
        self._mesh_shape = cmds.listRelatives(cube_xform, shapes=True)[0]

        curve_xform = cmds.curve(
            p=[(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0)], name="alias_curve"
        )
        self._curve_shape   = cmds.listRelatives(curve_xform, shapes=True)[0]

        lat                 = cmds.lattice(cube_xform)
        self._lattice_shape = cmds.listRelatives(lat[1], shapes=True)[0]

    def _fresh(self, shape):
        # always use a fresh PyNode so we don't read a cached _attr_dict entry
        return PyNode(shape)

    # --- mesh: should accept mesh aliases, reject curve aliases

    def test_mesh_accepts_vtx(self):
        attr = self._fresh(self._mesh_shape).find_attr("vtx")
        self.assertIsNotNone(attr)

    def test_mesh_accepts_pnts(self):
        attr = self._fresh(self._mesh_shape).find_attr("pnts")
        self.assertIsNotNone(attr)

    def test_mesh_accepts_pt(self):
        attr = self._fresh(self._mesh_shape).find_attr("pt")
        self.assertIsNotNone(attr)

    def test_mesh_rejects_cv_quiet(self):
        # cv is a nurbs alias, not a mesh alias
        attr = self._fresh(self._mesh_shape).find_attr("cv", quiet=True)
        self.assertIsNone(attr)

    def test_mesh_rejects_cv_loud(self):
        with self.assertRaises(AttributeError):
            self._fresh(self._mesh_shape).find_attr("cv", quiet=False)

    # --- nurbsCurve: should accept curve aliases, reject mesh / lattice aliases

    def test_curve_accepts_cv(self):
        attr = self._fresh(self._curve_shape).find_attr("cv")
        self.assertIsNotNone(attr)

    def test_curve_rejects_vtx_quiet(self):
        attr = self._fresh(self._curve_shape).find_attr("vtx", quiet=True)
        self.assertIsNone(attr)

    def test_curve_rejects_vtx_loud(self):
        with self.assertRaises(AttributeError):
            self._fresh(self._curve_shape).find_attr("vtx", quiet=False)

    def test_curve_rejects_pnts_quiet(self):
        # pnts is a mesh-specific attribute
        attr = self._fresh(self._curve_shape).find_attr("pnts", quiet=True)
        self.assertIsNone(attr)

    # --- lattice: should accept pt, reject mesh / curve aliases

    def test_lattice_accepts_pt(self):
        attr = self._fresh(self._lattice_shape).find_attr("pt")
        self.assertIsNotNone(attr)

    def test_lattice_rejects_vtx_quiet(self):
        attr = self._fresh(self._lattice_shape).find_attr("vtx", quiet=True)
        self.assertIsNone(attr)

    def test_lattice_rejects_cv_quiet(self):
        attr = self._fresh(self._lattice_shape).find_attr("cv", quiet=True)
        self.assertIsNone(attr)

    def test_lattice_rejects_pnts_quiet(self):
        attr = self._fresh(self._lattice_shape).find_attr("pnts", quiet=True)
        self.assertIsNone(attr)

    # --- controlPoints / cp are inherited and must continue to work everywhere

    def test_mesh_accepts_controlPoints(self):
        self.assertIsNotNone(self._fresh(self._mesh_shape).find_attr("controlPoints"))

    def test_mesh_accepts_cp(self):
        self.assertIsNotNone(self._fresh(self._mesh_shape).find_attr("cp"))

    def test_curve_accepts_controlPoints(self):
        self.assertIsNotNone(self._fresh(self._curve_shape).find_attr("controlPoints"))

    def test_curve_accepts_cp(self):
        self.assertIsNotNone(self._fresh(self._curve_shape).find_attr("cp"))

    def test_lattice_accepts_controlPoints(self):
        self.assertIsNotNone(
            self._fresh(self._lattice_shape).find_attr("controlPoints")
        )

    def test_lattice_accepts_cp(self):
        self.assertIsNotNone(self._fresh(self._lattice_shape).find_attr("cp"))

    # --- regression guard: __getattr__ access path must also reject

    def test_mesh_dot_cv_via_getattr_raises(self):
        mesh = self._fresh(self._mesh_shape)
        # __getattr__ calls find_attr(..., quiet=False) -> AttributeError on miss
        with self.assertRaises(AttributeError):
            mesh.cv