from maya import cmds
from rig import Node
from rig.maya.nodetypes import Mesh, ObjectSet, PyNode, ShadingEngine
from rig._tests._base import MayaTestCase


def _cube(name: str) -> tuple[str, str]:
    """Makes a history-free cube and returns (transform, full shape path)."""
    xform = cmds.polyCube(ch=False, name=name)[0]
    return xform, cmds.listRelatives(xform, shapes=True, fullPath=True)[0]


def _material(name: str) -> str:
    return cmds.shadingNode("lambert", asShader=True, name=name)


def _engine(name: str, material: str) -> ShadingEngine:
    engine = ShadingEngine.create(name=name)
    engine.set_material(material)
    return engine


def _raw_members(engine) -> list[str]:
    """Membership as cmds.sets() prints it (shape name or transform.f[a:b])."""
    return cmds.sets(str(engine), query=True) or []


def _faces(engine) -> list[tuple[str, list[int] | None]]:
    """(full shape path, face ids | None) per member, sorted by path."""
    return sorted(
        (
            (node.long_name, None if ids is None else ids.tolist())
            for node, ids in engine.get_face_members()
        ),
        key=lambda x: x[0],
    )


class TestShadingEngine(MayaTestCase):
    """
    Shading engine node type unit tests.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.xform, self.shape = _cube("cube")
        self.mat_a = _material("matA")
        self.mat_b = _material("matB")
        self.sg_a  = _engine("matASG", self.mat_a)
        self.sg_b  = _engine("matBSG", self.mat_b)

    # --- creation and registration

    def test_create_wiring(self):
        sg = ShadingEngine.create(name="wiredSG")
        self.assertIs(type(sg), ShadingEngine)
        self.assertEqual(sg.name, "wiredSG")
        self.assertEqual(sg.node_type, "shadingEngine")
        partition = cmds.listConnections(f"{sg}.partition", plugs=True) or []
        self.assertTrue(any(x.startswith("renderPartition.sets") for x in partition))
        self.assertEqual(len(sg.get_material_info()), 1)
        self.assertTrue(cmds.listConnections(f"{sg}.message", type="lightLinker"))
        self.assertIsNone(sg.get_material())

        # a created engine accepts members
        sg.set_material(self.mat_a)
        self.assertEqual(sg.get_material(), PyNode(self.mat_a))
        sg.assign([self.shape])
        self.assertEqual(_faces(sg), [(self.shape, None)])

        # a createNode'd one is dead
        bare = cmds.createNode("shadingEngine", name="bareSG")
        cmds.connectAttr(f"{self.mat_a}.outColor", f"{bare}.surfaceShader")
        with self.assertRaisesRegex(RuntimeError, "Source node will not allow"):
            cmds.sets(self.shape, edit=True, forceElement=bare)

    def test_registration(self):
        default = PyNode(ShadingEngine.DEFAULT)
        self.assertIs(type(default), ShadingEngine)
        self.assertIs(type(Node(ShadingEngine.DEFAULT) >> None), ShadingEngine)
        self.assertIs(type(PyNode.create("shadingEngine", name="viaFactorySG")), ShadingEngine)

        # registration makes ObjectSet(sg) and PyNode(sg) different objects
        name = self.sg_a.name
        self.assertEqual(ShadingEngine(name), PyNode(name))
        self.assertNotEqual(ShadingEngine(name), ObjectSet(name))
        self.assertNotEqual(PyNode(name), ObjectSet(name))

        # a fresh cube sits in the default engine
        self.assertEqual(Mesh(self.shape).get_shading_engines(), [default])
        self.assertEqual(Mesh(self.shape).get_shading_engines()[0], PyNode(ShadingEngine.DEFAULT))

    # --- for_material

    def test_for_material_none_creates(self):
        mat = cmds.createNode("lambert", name="lonely")
        self.assertIsNone(ShadingEngine.for_material(mat, create=False))
        self.assertFalse(cmds.objExists("lonelySG"))

        sg = ShadingEngine.for_material(mat)
        self.assertIs(type(sg), ShadingEngine)
        self.assertEqual(sg.name, "lonelySG")
        self.assertEqual(sg.get_material(), PyNode(mat))
        shaders = cmds.listConnections("defaultShaderList1.shaders", source=True, destination=False)
        self.assertEqual(shaders.count(mat), 1)

        # one engine -> that engine, and the link is not doubled
        self.assertEqual(ShadingEngine.for_material(mat), sg)
        self.assertEqual(ShadingEngine.for_material(PyNode(mat), create=False), sg)
        shaders = cmds.listConnections("defaultShaderList1.shaders", source=True, destination=False)
        self.assertEqual(shaders.count(mat), 1)

    def test_for_material_defaults(self):
        # lambert1 feeds no engine; standardSurface1 feeds both default engines
        # and initialParticleSE is dropped
        self.assertIsNone(ShadingEngine.for_material("lambert1", create=False))
        self.assertEqual(
            ShadingEngine.for_material("standardSurface1", create=False),
            PyNode(ShadingEngine.DEFAULT),
        )
        with self.assertRaises(ValueError):
            ShadingEngine.for_material("noSuchMaterial")

    def test_for_material_many(self):
        mat = _material("multi")
        _engine("multiX", mat)
        _engine("multiY", mat)
        with self.assertRaisesRegex(ValueError, "multiX"):
            ShadingEngine.for_material(mat)
        named = _engine("multiSG", mat)
        self.assertEqual(ShadingEngine.for_material(mat), named)

    # --- get_face_members

    def test_face_members_childless(self):
        self.sg_a.assign([self.shape])
        self.assertEqual(_faces(self.sg_a), [(self.shape, None)])
        self.assertEqual(_faces(PyNode(ShadingEngine.DEFAULT)), [])

        self.sg_b.assign([f"{self.xform}.f[0:2]"])
        self.assertEqual(_faces(self.sg_b), [(self.shape, [0, 1, 2])])
        self.assertEqual(_faces(self.sg_a), [(self.shape, [3, 4, 5])])
        # the inherited reader drops the face entries
        self.assertEqual(self.sg_b.get_members(), [])

    def test_face_members_parented(self):
        par = cmds.group(empty=True, name="par")
        kid, _ = _cube("kid")
        cmds.parent(kid, par)
        kid_shape = "|par|kid|kidShape"
        self.sg_a.assign([kid_shape])
        self.assertEqual(_faces(self.sg_a), [(kid_shape, None)])
        self.sg_b.assign(["kid.f[1]"])
        self.assertEqual(_faces(self.sg_b), [(kid_shape, [1])])
        self.assertEqual(_faces(self.sg_a), [(kid_shape, [0, 2, 3, 4, 5])])

    def test_face_members_two_shapes(self):
        other, other_shape = _cube("other")
        cmds.parent(other_shape, self.xform, shape=True, relative=True)
        cmds.delete(other)
        self.sg_a.assign([self.xform])
        self.assertEqual(
            _faces(self.sg_a), [("|cube|cubeShape", None), ("|cube|otherShape", None)]
        )

    def test_face_members_instance(self):
        inst       = cmds.instance(self.xform)[0]
        inst_shape = f"|{inst}|cubeShape"
        self.sg_a.assign([inst])
        self.assertEqual(_faces(self.sg_a), [(inst_shape, None)])
        self.assertEqual(_faces(PyNode(ShadingEngine.DEFAULT)), [(self.shape, None)])

        self.sg_b.assign([f"{inst}.f[0:1]"])
        self.assertEqual(_faces(self.sg_b), [(inst_shape, [0, 1])])
        self.assertEqual(_faces(self.sg_a), [(inst_shape, [2, 3, 4, 5])])
        self.assertEqual(_faces(PyNode(ShadingEngine.DEFAULT)), [(self.shape, None)])

    # --- assign

    def test_assign_object_level(self):
        self.sg_a.assign([self.shape])
        self.assertEqual(_raw_members(self.sg_a), ["cubeShape"])
        self.assertEqual(_raw_members(PyNode(ShadingEngine.DEFAULT)), [])
        self.sg_a.assign([self.shape])
        self.assertEqual(_raw_members(self.sg_a), ["cubeShape"])
        self.sg_b.assign([self.xform])
        self.assertEqual(_raw_members(self.sg_b), ["cubeShape"])
        self.assertEqual(_raw_members(self.sg_a), [])

    def test_assign_per_face_carve(self):
        self.sg_a.assign([self.shape])
        self.sg_b.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
        self.assertEqual(_faces(self.sg_a), [(self.shape, [2, 3, 4, 5])])
        self.assertEqual(_faces(self.sg_b), [(self.shape, [0, 1])])
        self.assertEqual(_raw_members(self.sg_b), ["cube.f[0:1]"])

    def test_faces_into_owner_is_noop(self):
        self.sg_a.assign([self.shape])
        self.sg_a.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
        self.assertEqual(_raw_members(self.sg_a), ["cubeShape"])
        self.assertEqual(cmds.ls(type="groupId"), [])

    def test_all_faces_renormalise(self):
        self.sg_a.assign([self.shape])
        self.sg_b.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
        self.assertEqual(len(cmds.ls(type="groupId")), 3)
        self.sg_a.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
        self.assertEqual(_raw_members(self.sg_a), ["cubeShape"])
        self.assertEqual(_raw_members(self.sg_b), [])
        self.assertEqual(cmds.ls(type="groupId"), [])

        # Maya alone never collapses the per-face form; the reader still does
        self.sg_b.assign([f"{self.xform}.f[0:1]"])
        self.sg_a.assign([f"{self.xform}.f[0:1]"], normalise=False)
        self.assertEqual(_raw_members(self.sg_a), ["cube.f[0:5]"])
        self.assertEqual(_faces(self.sg_a), [(self.shape, None)])

    def test_all_faces_renormalise_with_history(self):
        xform = cmds.polyCube(ch=True, name="hist")[0]
        shape = cmds.listRelatives(xform, shapes=True, fullPath=True)[0]
        self.sg_a.assign([shape])
        self.sg_b.assign([f"{xform}.f[0:1]"], touched=[shape], normalise=True)
        self.assertEqual(len(cmds.ls(type="groupParts")), 2)
        self.sg_a.assign([f"{xform}.f[0:1]"], touched=[shape], normalise=True)
        self.assertEqual(_raw_members(self.sg_a), ["histShape"])
        self.assertEqual(cmds.ls(type="groupId"), [])
        self.assertEqual(cmds.ls(type="groupParts"), [])
        self.assertTrue(cmds.objExists("polyCube1"))

    def test_orphan_group_ids_constant(self):
        self.sg_a.assign([self.shape])
        counts = []
        for _ in range(5):
            self.sg_b.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
            self.sg_a.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
            counts.append(len(cmds.ls(type="groupId")))
        self.assertEqual(counts, [0] * 5)
        self.assertEqual(_raw_members(self.sg_a), ["cubeShape"])

    def test_instanced_not_renormalised(self):
        inst       = cmds.instance(self.xform)[0]
        inst_shape = f"|{inst}|cubeShape"
        self.sg_a.assign([inst])
        self.sg_b.assign([f"{inst}.f[0:1]"], touched=[inst_shape], normalise=True)
        self.sg_a.assign([f"{inst}.f[0:1]"], touched=[inst_shape], normalise=True)
        self.assertEqual(_raw_members(self.sg_a), [f"{inst}.f[0:5]"])
        self.assertEqual(_raw_members(self.sg_b), [])
        self.assertEqual(_faces(self.sg_a), [(inst_shape, None)])
        self.assertEqual(_faces(PyNode(ShadingEngine.DEFAULT)), [(self.shape, None)])

    def test_assign_touched_must_be_unique(self):
        with self.assertRaises(ValueError):
            self.sg_a.assign([self.shape], touched=["noSuchShape"])

    # --- ObjectSet.get_or_create

    def test_get_or_create_guard(self):
        plain  = cmds.sets(empty=True, name="plain")
        before = cmds.ls()
        with self.assertRaisesRegex(TypeError, "transform"):
            ObjectSet.get_or_create(self.xform)
        with self.assertRaisesRegex(TypeError, "objectSet"):
            ShadingEngine.get_or_create(plain)
        self.assertEqual(cmds.ls(), before)

        self.assertEqual(ObjectSet.get_or_create(plain), ObjectSet(plain))
        self.assertEqual(ShadingEngine.get_or_create(self.sg_a.name), self.sg_a)
        created = ShadingEngine.get_or_create("freshSG")
        self.assertIs(type(created), ShadingEngine)
        self.assertTrue(cmds.listConnections(f"{created}.partition", plugs=True))

    def test_get_or_create_namespace(self):
        cmds.namespace(add="ns")
        cmds.namespace(set="ns")
        try:
            first  = ObjectSet.get_or_create("inNs")
            second = ObjectSet.get_or_create("inNs")
        finally:
            cmds.namespace(set=":")
        self.assertEqual(first.name, "ns:inNs")
        self.assertEqual(first, second)
        self.assertEqual(cmds.ls("ns:*", type="objectSet"), ["ns:inNs"])

    # --- Mesh readers

    def test_mesh_get_materials_skips_shaderless_engine(self):
        bare = ShadingEngine.create(name="noMatSG")
        bare.assign([self.shape])
        mesh = Mesh(self.shape)
        self.assertEqual(mesh.get_materials(), [])
        self.assertEqual(mesh.get_shading_engines(), [])

        self.sg_a.assign([f"{self.xform}.f[0:1]"], touched=[self.shape], normalise=True)
        self.assertEqual(mesh.get_materials(), [PyNode(self.mat_a)])
        self.assertEqual(mesh.get_shading_engines(), [self.sg_a])
        self.assertIs(type(mesh.get_shading_engines()[0]), ShadingEngine)
