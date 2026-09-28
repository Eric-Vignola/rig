"""Round 4a, review fixes (step FIX).

* A scope registers only the nodes a call made for itself: a deformer's
  ``...ShapeOrig`` under the user's mesh, a history shape and a skin's shared
  ``bindPose`` stay out, so deleting the container leaves the mesh at rest and
  a later skin of the same joints keeps its bind pose.
"""

from maya import cmds
from maya.api import OpenMaya

from rig import container, Node
from rig._internal.container import ContainerOptions
from rig.bridges import commands as rc
from rig.nodetypes import SkinCluster
from rig._tests._base import MayaTestCase


def _members(ctn):
    """The nodes of a Maya container, as it lists them."""
    return sorted(cmds.container(str(ctn), query=True, nodeList=True) or [])


def _heights(shape):
    """The y of the first four points of mesh `shape`."""
    sel = OpenMaya.MSelectionList()
    sel.add(shape)
    return [round(p.y, 3) for p in OpenMaya.MFnMesh(sel.getDagPath(0)).getPoints()][:4]


class _SceneCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self._options = {
            key: getattr(ContainerOptions, key)
            for key in ("skip_selection", "create_containers", "flatten_containers")
        }
        cmds.select(clear=True)

    def tearDown(self):
        for key, value in self._options.items():
            setattr(ContainerOptions, key, value)
        cmds.namespace(set=":")
        self.assertFalse(container.is_active)
        super().tearDown()


class TestScopeRegistersItsOwnNodes(_SceneCase):
    def _body(self, name="body"):
        cube  = cmds.polyCube(name=name, ch=False)[0]
        shape = cmds.listRelatives(cube, shapes=True, fullPath=True)[0]
        return cube, shape

    def test_deleting_the_container_of_a_skin_keeps_the_rest_shape(self):
        cube, shape = self._body()
        cmds.createNode("joint", name="j1")
        rest = _heights(shape)
        with container("rigA") as rig_a:
            skin = SkinCluster.create(cube, "j1")
        self.assertEqual(_members(rig_a), [str(skin)])
        cmds.setAttr("j1.ty", 5)
        self.assertNotEqual(_heights(shape), rest)
        cmds.delete(str(rig_a))
        cmds.setAttr("j1.ty", 0)
        self.assertFalse(cmds.ls(type="skinCluster"))
        self.assertEqual(_heights(shape), rest)

    def test_a_later_skin_keeps_the_shared_bind_pose(self):
        first, _ = self._body("a")
        second, _ = self._body("b")
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            SkinCluster.create(first, "j1")
        outside = SkinCluster.create(second, "j1")
        pose = cmds.listConnections(f"{outside}.bindPose")
        self.assertTrue(pose)
        cmds.delete(str(rig_a))
        self.assertEqual(cmds.listConnections(f"{outside}.bindPose"), pose)

    def test_the_bridges_and_node_create(self):
        target, _ = self._body("tgt")
        base, _ = self._body("base")
        cluster_base, _ = self._body("cbase")
        skinned, _ = self._body("skinned")
        extruded, _ = self._body("extruded")
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            blend = Node.create("blendShape", target, base)
            handle = rc.cluster(cluster_base)
            skin = rc.skinCluster(skinned, "j1")
            extrude = rc.polyExtrudeFacet(f"{extruded}.f[0]")
        members = _members(rig_a)
        self.assertIn(str(blend), members)
        self.assertIn(str(skin[0]), members)
        self.assertIn(str(extrude[0]), members)
        # the nodes the calls made for the user's meshes stay out
        for name in ("baseShapeOrig", "cbaseShapeOrig", "skinnedShapeOrig"):
            self.assertTrue(cmds.objExists(name), name)
            self.assertNotIn(name, members)
        self.assertFalse([m for m in members if cmds.nodeType(m) == "dagPose"])
        history = [
            m for m in members
            if cmds.nodeType(m) == "mesh" and cmds.getAttr(f"{m}.intermediateObject")
        ]
        self.assertEqual(history, [])
        # the cluster and its handle transform and shape are the call's own
        self.assertEqual([str(x) for x in handle], ["cluster1", "cluster1Handle"])
        for name in ("cluster1", "cluster1Handle", "cluster1HandleShape"):
            self.assertIn(name, members)

    def test_a_node_the_call_returns_is_its_own(self):
        cmds.createNode("joint", name="j1")
        with container("rigA") as rig_a:
            pose = rc.dagPose("j1", save=True, name="pose")
        self.assertEqual(cmds.nodeType(str(pose)), "dagPose")
        self.assertIn(str(pose), _members(rig_a))
