"""
Blendshape class
"""

from __future__ import annotations

import logging
import re

import numpy as np
from maya import cmds
from maya.api import OpenMaya
from numpy.typing import ArrayLike
from rig.maya.nodetypes._base import Attribute
from rig.maya.nodetypes.deformer import Deformer
from rig.maya.nodetypes.geometry import iter_component_ranges
from rig.maya.nodetypes.transform import Transform

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class BlendShape(Deformer):
    """
    Blenshape class
    """

    # the maya native node type string
    NATIVE_NODE_TYPE = "blendShape"

    @classmethod
    def _create(cls, *args, **kwargs) -> str:
        """[Internal] Creates a blendshape node and returns the mesh name.

        Args:
            args: args supported by cmds.blendShape(). A MorphData or MorphList
                can be given in place of a target geometry, in which case a
                target is built for each morph and its offsets are applied
                immediately after the node is created.
            kwargs: kwargs supported by cmds.blendShape()
        """
        from cgmath.geometry import MorphData, MorphList

        # split the morph data out of the geometries, it is applied once the
        # node exists
        morphs = MorphList()
        geoms  = []
        for arg in args:
            if isinstance(arg, MorphData):
                morphs.append(arg)
            elif isinstance(arg, MorphList):
                morphs.extend(arg)
            else:
                geoms.append(arg)

        # default frontOfChain to True
        kwargs["frontOfChain"] = kwargs.get("frontOfChain", kwargs.get("foc", True))
        if "foc" in kwargs:
            kwargs.pop("foc")

        name = cmds.blendShape(*geoms, **kwargs)[0]

        if morphs:
            cls(name).set_targets_data(morphs)

        return name

    # --- attr helpers

    def target_data_attrs(self, index: int) -> tuple[Attribute, Attribute, Attribute]:
        """Returns 2 attribute objects pointing to the 2 data attrs of a given target:

        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputGeomTarget
        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputPointsTarget
        .inputTarget[0].inputTargetGroup[x].inputTargetItem[6000].inputComponentsTarget
        """
        group_attr = self.inputTarget[0].inputTargetGroup
        group_attr = group_attr[index]
        item_attr  = group_attr.inputTargetItem[6000]
        return (
            item_attr.inputGeomTarget,
            item_attr.inputPointsTarget,
            item_attr.inputComponentsTarget,
        )

    # --- target operations

    @property
    def num_targets(self) -> int:
        """Returns the number of targets"""
        return self.weight.num_elements

    def get_targets(self) -> list[str]:
        """Returns a list of targets in this node."""
        return cmds.listAttr(self.name, string="weight", multi=True)

    def get_target_indices(self) -> list[int]:
        """Returns a list of target index in this node."""
        return self.weight.get_logical_indices()

    def get_target_index(self, target_name: str, failfast: bool = False) -> int:
        """Returns the logical index of a target, or -1 if not found.

        Args:
            target_name: A target name to work with.
            failfast: If True, raise error if target not found.
        """
        target_attr = self.find_alias(target_name, quiet=True)
        if target_attr:
            return target_attr.logical_index()
        if failfast:
            raise RuntimeError(f"Target {self.name}.{target_name} not found.")
        return -1

    def get_target_name(self, target_id: int) -> str | None:
        """Returns the name a target, or None if not found.

        Args:
            target_id: A target index to work with.
        """
        return self.get_target_weight_attr(target_id).alias

    def set_target_name(self, target: int | str, name: str) -> None:
        """Sets the name of a target.

        Args:
            target: A target name or index.
            name: New name.
        """
        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target
        self.get_target_weight_attr(target_id).alias = name

    def get_target_weight_attr(self, target: str | int) -> Attribute:
        """Returns the weight attribute a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
        """
        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target
        return self.weight[target_id]

    def get_target_weight(self, target: str | int) -> float:
        """Returns the weight of a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
        """
        return self.get_target_weight_attr(target).get()

    def set_target_weight(self, target: str | int, val: float) -> None:
        """Sets the weight of a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
            val: Weight value.
        """
        return self.get_target_weight_attr(target).set(val)

    def add_empty_target(self, target_name: str) -> int:
        """Adds a new empty target.

        Args:
            target_name: The new target to add.

        Raises:
            RuntimeError: If target already exists.

        Returns:
            The target index.
        """
        if not self.fn_set.findAlias(target_name).isNull():
            raise RuntimeError(f"Target {self.name}.{target_name} already exists.")

        weight_attr            = self.weight
        index                  = weight_attr.get_next_available_index()
        weight_attr            = weight_attr[index]
        weight_attr.alias      = target_name
        weight_attr.is_keyable = True
        weight_attr.set(0)
        return index

    def rebuild_target(
        self,
        target: int | str | MorphData,
        name:   str | None            = None,
    ) -> Transform:
        """Rebuils a geometry from data stored in a given target."""
        from cgmath.geometry import MorphData

        target_id = -1
        if isinstance(target, str):
            target_id = self.get_target_index(target, failfast=True)
        elif isinstance(target, int):
            target_id = target

        # duplicates the base mesh
        geom        = self.get_geometries()[0].get_parent()
        target_mesh = geom.duplicate_geometry(name=name)

        # compute offsetted points
        org_mesh_data = self.get_original_geometries()[0].serialize(
            world_space=False, include_uvs=False
        )
        points      = org_mesh_data.points
        target_data = self.get_target_data(target_id, original_mesh_data=org_mesh_data)
        points[target_data.indices] += target_data.offsets

        # apply points
        shape = target_mesh.get_children(shapes=True)[0]
        shape.set_points(points, world_space=False)

        return target_mesh

    # --- serialization

    def get_target_data(
        self,
        target:             str        | int,
        tolerance:          float      | None = None,
        neighbors:          np.ndarray | None = None,
        original_mesh_data: MeshData   | None = None,
    ) -> MorphData:
        """Serialize a target.
        Using index is faster than name.

        Args:
            target: A target name or index.
            tolerance: Tolerance for removing tiny offsets.
                If None use MorphData's default tolerance.
            neighbors: a numpy array of vertex neighboring data.
                used for pruning offsets.
            original_mesh_data: the original(pre-deformation) mesh data of the
                source mesh this blendshape is applied to, used for computing target
                data from a live mesh. If None, serialize a fresh mesh data (slower), and
                if no original mesh data is provided, serialize directly from the blendshape node.

        Returns:
            A `MorphData` object.
        """
        from cgmath.geometry import MeshData, MorphData

        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=True)
        else:
            target_id = target

        # get weight alias/target name and weight value
        weight_attr = self.get_target_weight_attr(target)
        target_name = weight_attr.alias
        # get target attributes
        geom_attr, offset_attr, comp_attr = self.target_data_attrs(target_id)

        # if a live target mesh is connected, compute target data from mesh data
        # (the blendshape offsets attr is empty when a live mesh is connected)
        target_mesh = geom_attr.list_connections(
            source=True, destination=False, plugs=False
        )
        target_mesh = geom_attr.get_connected_attrs(src=True, dst=False)
        if target_mesh:
            kwargs           = {"include_uvs": False, "world_space": False}
            target_mesh      = target_mesh[0].node
            target_mesh_data = target_mesh.serialize(**kwargs)
            if not original_mesh_data:
                original_mesh_data = self.get_original_geometries()
                if original_mesh_data:
                    original_mesh_data = original_mesh_data[0].serialize(**kwargs)

            if original_mesh_data:
                target_data = MorphData.from_mesh_data(
                    original_mesh_data, target_mesh_data, target_name=target_name
                )
                target_data.prune_offsets(tolerance=tolerance, neighbors=neighbors)
                return target_data

        data = MorphData(
            name    = target_name,
            indices = np.empty((0,), dtype=int),
            offsets = np.empty((0, 3), dtype=float),
        )

        comp_data = comp_attr.get_data_fn_set()
        if comp_data:
            offsets = offset_attr.get()
            if offsets:
                data.offsets = np.array(offsets)[:, :-1]
                indices      = []
                for i in range(comp_data.length()):
                    comp = OpenMaya.MFnSingleIndexedComponent(comp_data.get(i))
                    indices.extend(comp.getElements())
                data.indices = np.array(indices)

                # remove offsets below the given tolerance
                data.prune_offsets(tolerance=tolerance, neighbors=neighbors)

                # sort numerically
                # maya allows unsorted indices, but sorted indices is more optimal
                data.sort()

        return data

    def set_target_data(
        self,
        target:    str       | int,
        data:      MorphData | tuple[ArrayLike, ArrayLike],
        tolerance: float     | None                        = None,
        force:     bool                                    = True,
    ) -> None:
        """Sets the offset data of a given target.
        Using index is faster than name.

        Args:
            target: A target name or index to set.
            data: A MorphData object or (indices, offsets).
            tolerance: Tolerance for removing tiny offsets.
                If None use MorphData's default tolerance.
            force: If True, a new target will be added if no target exists.
        """
        from cgmath.geometry import MorphData

        if not isinstance(target, int):
            target_id = self.get_target_index(target, failfast=(not force))

            # if the shape doesn't exist, add a new entry
            if target_id < 0 and force:
                target_id = self.add_empty_target(target)

        else:
            target_id = target

        # remove tiny offsets first
        if not isinstance(data, MorphData):
            data = MorphData(
                name="name", indices=np.array(data[0]), offsets=np.array(data[1])
            )
        if tolerance is None:
            data.prune_offsets()
        else:
            data.prune_offsets(tolerance=tolerance)

        _, offset_attr, comp_attr = self.target_data_attrs(target_id)
        offset_attr.set(len(data.offsets), *data.offsets.tolist(), type="pointArray")

        # convert indices to range strings
        #
        # this reduces the number of indices in the component list attr. it makes
        # all related operations much more efficient, also smaller file size and memory
        # footprint.
        comps = list(iter_component_ranges("vtx", data.indices))
        comp_attr.set(len(comps), *comps, type="componentList")

    def serialize(self, match_name: list[str] | str | None = None) -> MorphList:
        """Serialize this blendshape.

        Args:
            match_name: One or more of target names or regex to match.
                If None, use all targets.

        Returns:
            A list of `MorphData` objects.
        """
        from cgmath.geometry import MorphList

        names       = [match_name] if isinstance(match_name, str) else match_name
        regex       = re.compile(r"|".join([rf"^{x}$" for x in names])) if names else None
        weight_attr = self.weight
        org_mesh    = self.get_original_geometries()

        # if there is an original mesh, use it to compute target data
        if org_mesh:
            org_mesh = org_mesh[0].serialize(include_uvs=False)

        target_data_list = []
        for i in self.get_target_indices():
            if regex:
                target_name = weight_attr[i].alias
                if not regex.match(target_name):
                    continue

            target_data_list.append(
                self.get_target_data(i, original_mesh_data=org_mesh)
            )

        return MorphList(target_data_list)

    def set_targets_data(
        self,
        data:      MorphList,
        tolerance: float | None = None,
        force:     bool         = True,
    ) -> None:
        """Sets the offset data of a given target.
        Using index is faster than name.

        Args:
            target: A target name or index to set.
            data: A MorphList object.
            tolerance: Tolerance for removing tiny offsets.
                If None use MorphData's default tolerance.
            force: If True, a new target will be added if no target exists.
        """
        from cgmath.geometry import MorphList

        for target in data:
            self.set_target_data(
                target=target.name, data=target, tolerance=tolerance, force=force
            )