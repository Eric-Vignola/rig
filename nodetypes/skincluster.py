"""
Skincluster class
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim
from rig.nodetypes._base import PyNode
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.deformer import Deformer
from rig.nodetypes.dg_node import get_short_name
from rig.nodetypes.joint import Joint
from rig.nodetypes.mesh import Mesh
from rig.nodetypes.plugins import load_plugin

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

INF_TYPE = Union[str, DAGNode, List[Union[str, DAGNode]]]


def flatten_vertices(seq):
    """
    convert a sequence of strings (eg: object.component[5], .component[10:200])
    to a dict of obj:flattened indices
    """
    r       = re.compile(r"([0-9a-zA-Z._|:]+)\[(\d+)(?::(\d+))?\]")
    results = {}

    for item in sorted(seq):
        m = r.match(item)

        # if there's no match, a shape is given with no components,
        # use all vertices as indices
        if m is None:
            name = Mesh(item)
            rng  = list(range(10))
        else:
            name, comp = m.group(1).split(".")
            if comp != "vtx":
                raise RuntimeError(f"[{comp} is not a supported component]")
            name = Mesh(name)
            start, end = int(m.group(2)), m.group(3)
            rng = range(start, int(end) + 1) if end else (start,)

        if name in results:
            results[name].extend(rng)
        else:
            results[name] = list(rng)

    # make sure results are unique
    for name, indices in results.items():
        results[name] = np.unique(indices)

    return results


class SkinCluster(Deformer):
    """
    SkinCluster class
    """

    NATIVE_NODE_TYPE = "skinCluster"
    FN_SET           = OpenMayaAnim.MFnSkinCluster

    # --- creation

    @classmethod
    def _create(cls, geom: str, influences: INF_TYPE | SkinData, **kwargs) -> str:
        """[Internal] Creates a skin cluster node and return its name.

        Args:
            geom: A geometry to create the skincluster on.
            influences: A list of influence object names, or a SkinData object.
                If a SkinData is given, its influences are used to create the
                skincluster and its weights are applied immediately after.
            kwargs: kwargs supported by cmds.skinCluster().

        Returns:
            Name of the created skincluster node.
        """
        from cgmath.geometry import SkinData

        # if a SkinData is given, create from its influences and apply its
        # weights once the node exists
        skin_data = influences if isinstance(influences, SkinData) else None
        if skin_data is not None:
            influences = skin_data.influences

        influences = cls._sanitize_influences(influences)

        # find the target geometry
        geoms_found = cmds.ls(geom)
        if not geoms_found:
            raise RuntimeError(f"Geometry {geom} not found.")
        geom = geoms_found[0]

        # delete current skincluster if exists
        skin = cmds.ls(cmds.listHistory(geom), type="skinCluster")
        if skin:
            cmds.delete(skin)

        # set some default kwargs
        for ln, sn, dv in (
            ("toSelectedBones", "tsb", True),
            ("name", "n", get_short_name(geom) + "_skincluster"),
        ):
            kwargs[ln] = kwargs.get(ln, kwargs.get(sn, dv))
            if sn in kwargs:
                kwargs.pop(sn)

        # make the skincluster
        name = cmds.skinCluster(influences, geom, **kwargs)[0]

        # weights columns already align with skin_data.influences (creation
        # preserves order), so set them directly without serialize/conform
        if skin_data is not None:
            SetSkinWeightsCommand(cls(name), skin_data.weights)

        return name

    @classmethod
    def _sanitize_influences(cls, infs: INF_TYPE) -> list[Joint]:
        infs = [infs] if not isinstance(infs, (list, tuple, set)) else infs

        out     = []
        missing = []
        multi   = []
        for inf in infs:
            if isinstance(inf, Joint):
                out.append(inf)
            else:
                inf = str(inf)
                if not Joint.exists(inf):
                    missing.append(inf)
                elif len(cmds.ls(inf, long=True, type="joint")) > 1:
                    missing.append(inf)
                else:
                    out.append(Joint(inf))

        if missing:
            raise RuntimeError(f"Missing joints found: {missing}")
        if multi:
            raise RuntimeError(f"Multiple joints found with the same name: {multi}")

        return out

    def connect_bind_pre_matrices(self, search_func: Callable) -> None:
        """For each influence joint, finds a driver joint and connect its
        worldInverseMatrix to the corresponding bindPreMatrix of this skincluster.

        Args:
            A function that takes a joint name and returns a driver joint name.
        """
        for i, inf in enumerate(self.get_influence_objects()):
            driver = search_func(inf.name)
            if not Joint.exists(driver):
                raise RuntimeError(f"Driver joint not found: {driver}.")
            Joint(driver).worldInverseMatrix[0] >> self.bindPreMatrix[i]

    def get_influence_objects(self) -> list[DAGNode]:
        """Returns a list of influence objects."""
        return [PyNode(x) for x in self.fn_set.influenceObjects()]

    def add_influence_objects(self, infs: INF_TYPE) -> None:
        """Adds influence objects to this skincluster.

        Args:
            infs: A list of influence object names.
        """
        infs = self._sanitize_influences(infs)
        # Adding a SINGLE influence makes Maya select it, discarding whatever
        # the caller had picked; two or more leaves the selection alone. The
        # last entry goes back last because cmds.select sorts a batch, and
        # pickers that track selection order read the last one.
        ordered = cmds.ls(orderedSelection=True, long=True) or []
        try:
            cmds.skinCluster(self.name, edit=True, addInfluence=infs, wt=0.0)
        finally:
            if ordered:
                cmds.select(ordered[:-1], replace=True)
                cmds.select(ordered[-1], add=True)
            else:
                cmds.select(clear=True)

    def remove_influence_objects(self, infs: INF_TYPE) -> None:
        """Removes influence objects from this skincluster.

        Args:
            infs: A list of influence object names.
        """
        cmds.skinCluster(self.name, edit=True, removeInfluence=infs)

    def set_influence_objects(self, infs: INF_TYPE) -> None:
        """Sets the influence objects of this skincluster.

        Args:
            infs: A list of influence object names.
        """
        cur  = self.get_influence_objects()
        infs = [infs] if not isinstance(infs, (list, tuple)) else infs
        infs = [Joint(x) for x in infs]
        if cur == infs:
            return

        # a hacky way to avoid empty influence error
        tmp_joint = Joint.create()
        try:
            self.add_influence_objects(tmp_joint)
            self.remove_influence_objects(cur)
            self.add_influence_objects(infs)
            self.remove_influence_objects(tmp_joint)
        finally:
            tmp_joint.delete()

    # --- weights

    def get_weights(self) -> np.ndarray:
        """Returns a 2D numpy array of weight values (per component per influence)."""
        geom  = self.get_geometries()[0]
        comps = geom.get_component_mobject()
        weights, inf_count = self.fn_set.getWeights(geom.mdagpath, comps)
        weights = np.array(weights)
        return np.reshape(weights, (geom.num_weight_points, inf_count))

    def set_weights(
        self,
        data:     np.ndarray | SkinData,
        additive: bool                    = False,
        indices:  None       | np.ndarray = None,
    ) -> None:
        """Sets the weights of this skincluster.
        If weights is a SkinData object, update influence objects to match the data.

        Args:
            data: A SkinData object or a 2D array of weight values
                (per component per influence)
            additive: If True, adds the skin data influences onto the existing influences.
                Otherwise, replaces the existing influences.
                Ignored if `data` is not a SkinData object.
            indices: A 1D array of vertex indices to set the weights. If None, set all weights.
        """
        from cgmath.geometry import SkinData

        if not isinstance(data, SkinData):
            SetSkinWeightsCommand(self, data, indices=indices)
            return

        # make sure incoming data has long inf names,
        # to ensure reliable influence conformation
        org_infs            = self._sanitize_influences(data.influences)
        new_data            = data.copy()
        new_data.influences = [x.long_name for x in org_infs]

        # conform received SkinData so influences align to self
        cur_data = self.serialize(full_path=True)
        SkinData.conform(cur_data, new_data)
        new_infs = self._sanitize_influences(cur_data.influences)

        # add any missing joints to self, maintaining order
        cur_infs = set(self.get_influence_objects())
        to_add   = [x for x in new_infs if x not in cur_infs]
        if to_add:
            self.add_influence_objects(to_add)

        # apply weights
        SetSkinWeightsCommand(self, new_data.weights, indices=indices)

        # remove excessive influences that might be introduced by conforming
        if not additive:
            to_remove = set(new_infs) - set(org_infs)
            if to_remove:
                self.remove_influence_objects(to_remove)

    @staticmethod
    def rebind(geo) -> None:
        shape = cmds.listRelatives(geo, shapes=True, pa=True)[0]
        skins = cmds.ls(cmds.listHistory(shape), type="skinCluster")
        if not skins:
            return

        for skin in skins:
            joints = cmds.skinCluster(skin, query=True, influence=True)
            cmds.setAttr(skin + ".envelope", 0)
            cmds.skinCluster(skin, edit=True, unbindKeepHistory=True)

            # delete bindPose
            dagPose = cmds.dagPose(geo, query=True, bindPose=True)
            if dagPose:
                cmds.delete(dagPose)
            dagPose = cmds.listConnections(skin + ".bindPose", d=False, type="dagPose")
            if dagPose:
                cmds.delete(dagPose)

            # rebind
            cmds.skinCluster(joints, shape, toSelectedBones=True)
            cmds.setAttr(skin + ".envelope", 1)

    def get_mesh(self) -> Mesh:
        """Returns the mesh this skincluster is attached to."""
        return Mesh(self.get_geometries()[0])

    # --- data transfer

    def transfer_to_mesh(
        self,
        other: Mesh | str,
        **kwargs,
    ) -> SkinCluster:
        """Transfer skin weights to another mesh.

        Args:
            other: A mesh to transfer skin weights to.
            kwargs: Transfer kwargs supported in MeshDataResampler.resample_skin_weights().

        Returns:
            The new skincluster node on the other mesh.
        """
        from cgmath.geometry.resample import MeshDataResampler

        this  = self.get_mesh()
        other = Mesh(other)

        mesh_data_a, uv_data_a = this.serialize(include_uvs=True)
        mesh_data_b, uv_data_b = other.serialize(include_uvs=True)

        resampler = MeshDataResampler(
            mesh_data_a, mesh_data_b, uv_data_a[0], uv_data_b[0]
        )
        skin_data_b = resampler.resample_skin_weights(self.serialize(), **kwargs)
        return other.apply_skin_data(skin_data_b)

    @staticmethod
    def copy_skincluster(
        source_meshes: list[str] | str, target_vertices: list[str] = None
    ) -> None:
        """Many to One skincluster transfer"""
        from cgmath.geometry import SkinData

        # process source_meshes
        if isinstance(source_meshes, str):
            source_meshes = [Mesh(source_meshes)]
        else:
            source_meshes = [Mesh(x) for x in source_meshes]

        # process source skinclusters
        source_skins = []
        for mesh in source_meshes:
            skin = mesh.get_deformers(node_type="skinCluster")
            # use the last found skinCluster if any
            if skin:
                source_skins.append(skin[-1])
            else:
                raise RuntimeError(f"No skinCluster found on {mesh}")

        # serialize + combined the source data
        source_mesh_data = sum([x.serialize()[0] for x in source_meshes])
        source_skin_data = sum([x.serialize() for x in source_skins])
        source_skin_data.remove_unused()

        # process each target
        target_data = flatten_vertices(target_vertices)
        for target_mesh, indices in target_data.items():
            # serialize target_mesh
            target_mesh_data = target_mesh.serialize()[0]

            # make a copy of the source_skin_data before it gets conformed
            skin_data = source_skin_data.copy()

            # get the target skin
            target_skin = target_mesh.get_deformers(node_type="skinCluster")

            # if no skinCluster present, make a default one
            if not target_skin:
                default     = skin_data.influences[:1]
                target_skin = SkinCluster.create(target_mesh, default)
            else:
                target_skin = target_skin[-1]

            target_skin_data = target_skin.serialize()

            # conform the SkinData before sampling
            SkinData.conform(target_skin_data, skin_data)

            # spatially sample the target points
            samples                           = source_mesh_data.sample(target_mesh_data.points[indices])
            target_skin_data.weights[indices] = samples(skin_data.weights)

            # set the weights
            target_skin.set_weights(target_skin_data.weights)

    # --- serialization

    def serialize(self, full_path: bool = False, optimize: bool = False) -> SkinData:
        """Serialize this skincluster and optimize the data.

        Args:
            full_path: If True, return full path names for influences.
                Otherwise, return short names.
            optimize: If True, optimize the serialized data by pruning tiny weights
                and removed unused influences.

        Returns:
            A SkinData object.
        """
        from cgmath.geometry import SkinData

        infs = [
            x.fullPathName() if full_path else x.partialPathName()
            for x in self.fn_set.influenceObjects()
        ]
        data = SkinData(influences=infs, weights=self.get_weights())

        if optimize:
            data.prune(min_value=0.0001)
            data.remove_unused()
        return data

    def normalize_weights(self) -> None:
        """Normalize the skin weights."""
        skin_data = self.serialize()
        skin_data.normalize()
        self.set_weights(skin_data)

    def remove_unused_influences(self, tolerance: float, normalize: bool) -> None:
        """Remove unused influences, including influences with small weights within the given tolerance."""
        skin_data = self.serialize()
        skin_data.remove_unused(tolerance, normalize)
        self.set_weights(skin_data)

    def set_max_influences(self, count: int) -> None:
        """Set the maximum number of influences per vertex."""
        skin_data = self.serialize()
        skin_data.set_max_influences(count)
        self.set_weights(skin_data)

    def prune_small_weights(self, threshold: float) -> None:
        """Set weights below a certain threshold to zero."""
        skin_data = self.serialize()
        skin_data.prune(threshold)
        self.set_weights(skin_data)

    def inpaint(
        self,
        indices:        list[int],
        neighbors:      np.ndarray,
        iterations:     int,
        max_influences: int,
    ) -> None:
        """Inpaint smooth skin weights of selected vertices, given their indices and quad-based neighbors."""
        skin_data = self.serialize()
        data      = skin_data.copy()
        data.inpaint(
            neighbors     = neighbors,
            indices       = indices,
            iterations    = iterations,
            contributions = 0.5,
            receptions    = 1.0,
            tolerance     = -1,
        )

        # set weights if solution is valid
        if data.valid:
            data.set_max_influences(max_influences)
            self.set_weights(data)
        else:
            raise RuntimeError(
                f"Inpainting failed. Solution needs more than {iterations} iterations."
            )


class SetSkinWeightsCommand:
    """
    A set skin weights command with undo/redo support.
    """

    def __init__(
        self,
        skincluster: SkinCluster,
        weights:     np.ndarray,
        indices:     None | np.ndarray = None,
    ) -> None:
        super().__init__()

        # gather info from this skincluster
        self._fn_set = skincluster.fn_set
        inf_count    = len(self._fn_set.influenceObjects())
        inf_ids      = OpenMaya.MIntArray(range(inf_count))
        geom         = skincluster.get_geometries()[0]
        geom_path    = geom.mdagpath

        # if no indices are given, set all weights
        comps = geom.get_component_mobject(indices=indices)

        # if weights is 2D, reshape it to 1D
        if weights.ndim == 2:
            weights = weights.flatten()

        # internal vars
        self._weights     = OpenMaya.MDoubleArray(weights)
        self._old_weights = None
        self._args        = [geom_path, comps, inf_ids]

        with load_plugin("undoable_api_command"):
            cmds.runUndoableAPICommand(self)

    def doIt(self) -> None:
        """Sets the requested weights and store the old weights for undo()."""
        args              = self._args + [self._weights, True, True]
        self._old_weights = self._fn_set.setWeights(*args)

    def undoIt(self) -> None:
        """Sets the weights back to the old weights, prior to callign doIt()."""
        args = self._args + [self._old_weights, True, False]
        self._fn_set.setWeights(*args)

    def redoIt(self) -> None:
        """Sets the requested weights, do NOT store old weights."""
        args = self._args + [self._weights, True, False]
        self._fn_set.setWeights(*args)