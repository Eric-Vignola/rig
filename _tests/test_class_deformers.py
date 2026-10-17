"""DAGNode.get_deformers: the deformers findRelatedDeformer finds, nearest the shape
first, read through the API without an undo step (TAGS, the user's decision of
2026-10-15; the case-by-case proof is runs/r5/TAGS/tools/prove_deformers.py)."""

import os
import tempfile

from maya import cmds, mel
from rig.nodetypes import Node
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _chain(name):
    """A cube deformed by skinCluster, blendShape (front of chain), cluster,
    deltaMush, twist and ffd, applied in that order; the chain nearest the shape
    first."""
    cmds.polyCube(name=name)
    cmds.select(clear=True)
    cmds.joint(name=name + "_j")
    cmds.skinCluster(name + "_j", name, name=name + "_sk")
    cmds.polyCube(name=name + "_tgt")
    cmds.blendShape(name + "_tgt", name, frontOfChain=True, name=name + "_bs")
    cmds.cluster(name, name=name + "_clu")
    cmds.deltaMush(name, name=name + "_dm")
    cmds.nonLinear(name, type="twist", name=name + "_tw")
    cmds.lattice(name, name=name + "_ffd")
    return [name + s for s in ("_ffd", "_tw", "_dm", "_clu", "_sk", "_bs")]


class TestGetDeformers(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    @staticmethod
    def names(node, **kwargs):
        return [x.get_name() for x in Node(node).get_deformers(**kwargs)]

    def test_get_deformers_leaves_no_undo_step(self):
        """findRelatedDeformer is a recorded command: each call was an undo step,
        and after an undo it cleared the redo queue."""
        _chain("c")
        cmds.flushUndo()
        for node in ("c", "cShape", "cShapeOrig"):
            Node(node).get_deformers()
            Node(node).get_deformers(node_type="cluster", first_only=True)
        self.assertTrue(cmds.undoInfo(query=True, undoQueueEmpty=True))

        cmds.createNode("transform", name="redo_me")
        cmds.undo()
        Node("cShape").get_deformers()
        self.assertFalse(cmds.undoInfo(query=True, redoQueueEmpty=True))
        cmds.redo()
        self.assertTrue(cmds.objExists("redo_me"))

    def test_get_deformers_lists_the_chain_nearest_the_shape_first(self):
        """findDeformers lists a chain in no set order (it changes between two
        identical chains); get_deformers lists it as the history does."""
        for k in range(3):
            chain = _chain(f"c{k}")
            with self.subTest(k):
                self.assertEqual(self.names(f"c{k}Shape"), chain)
                self.assertEqual(self.names(f"c{k}"), chain)
                self.assertEqual(Node(f"c{k}").get_deformers(first_only=True).get_name(), chain[0])
                self.assertEqual(self.names(f"c{k}Shape", node_type=["cluster", "ffd"]), [chain[0], chain[3]])

    def test_get_deformers_finds_what_findRelatedDeformer_finds(self):
        """The same deformers, case by case, in the chain's order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "deformed.ma").replace("\\", "/")
            cmds.polySphere(name="rs")
            cmds.cluster("rs", name="rs_clu")
            cmds.nonLinear("rs", type="bend", name="rs_bend")
            cmds.file(rename=path)
            cmds.file(force=True, type="mayaAscii", save=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="R")

            cmds.polyCube(name="none")
            cmds.polyCube(name="sk")
            cmds.select(clear=True)
            cmds.joint(name="sk_j")
            cmds.skinCluster("sk_j", "sk", name="sk_skin")
            cmds.polyCube(name="cl")
            cmds.cluster("cl", name="cl_clu")
            cmds.polyCube(name="lat")
            cmds.lattice("lat", name="lat_ffd")
            cmds.polyCube(name="dm")
            cmds.deltaMush("dm", name="dm_mush")
            cmds.polyPlane(name="wr")
            cmds.curve(name="wcrv", p=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
            cmds.wire("wr", w="wcrv", name="wr_wire")
            cmds.polyCube(name="tn")
            cmds.tension("tn", name="tn_tension")
            cmds.polyCube(name="nl")
            cmds.nonLinear("nl", type="bend", name="nl_bend")
            cmds.polyCube(name="bstgt")
            cmds.cluster("bstgt", name="bstgt_clu")
            cmds.polyCube(name="bs")
            cmds.blendShape("bstgt", "bs", name="bs_bs")
            chain = _chain("c")
            cmds.polyCube(name="inst")
            cmds.cluster("inst", name="inst_clu")
            cmds.instance("inst", name="inst2")
            # a cluster's envelope driven by another mesh's cluster
            cmds.polyCube(name="ea")
            cmds.cluster("ea", name="ea_clu")
            cmds.polyCube(name="eb")
            cmds.cluster("eb", name="eb_clu")
            cmds.connectAttr("ea_clu.envelope", "eb_clu.envelope")
            # a blendShape target kept as an intermediate shape brings its deformers
            cmds.polyCube(name="base")
            cmds.polyCube(name="iotgt")
            cmds.cluster("iotgt", name="iotgt_clu")
            cmds.blendShape("iotgt", "base", name="base_bs")
            cmds.parent("iotgtShape", "base", shape=True, relative=True)
            cmds.setAttr("|base|iotgtShape.intermediateObject", 1)
            cmds.delete("iotgt")

            cases = {
                "none": [], "noneShape": [],
                "sk": ["sk_skin"], "skShape": ["sk_skin"], "skShapeOrig": [],
                "clShape": ["cl_clu"], "latShape": ["lat_ffd"], "lat_ffdLatticeShape": [],
                "dmShape": ["dm_mush"], "wrShape": ["wr_wire"], "wcrv": [],
                "tnShape": ["tn_tension"], "nlShape": ["nl_bend"],
                "bsShape": ["bs_bs"], "bstgtShape": ["bstgt_clu"],
                "c": chain, "cShape": chain,
                "inst|instShape": ["inst_clu"], "inst2|instShape": ["inst_clu"], "inst2": ["inst_clu"],
                "ebShape": ["eb_clu"], "|base|baseShape": ["base_bs", "iotgt_clu"],
                "R:rs": ["R:rs_bend", "R:rs_clu"], "R:rsShape": ["R:rs_bend", "R:rs_clu"],
            }
            for name, expected in cases.items():
                with self.subTest(name):
                    found = list(mel.eval(f'findRelatedDeformer "{name}"') or [])
                    self.assertEqual(sorted(self.names(name)), sorted(found))
                    self.assertEqual(self.names(name), expected)
            cmds.file(new=True, force=True)

    def test_get_deformers_of_a_transform_reads_its_visible_shape(self):
        """On a transform, the first non-intermediate shape's deformers (the rule of
        Transform.get_shape); findDeformers took the first shape, intermediate or not."""
        cmds.polyCube(name="t")
        cmds.cluster("t", name="t_clu")
        cmds.reorder("tShapeOrig", front=True)
        self.assertEqual(cmds.listRelatives("t", shapes=True), ["tShapeOrig", "tShape"])
        self.assertEqual(self.names("t"), ["t_clu"])
        self.assertEqual(self.names("tShape"), ["t_clu"])
