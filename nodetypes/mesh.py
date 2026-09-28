"""
Mesh node class
"""

from __future__ import annotations

import contextlib
import enum
import re
from collections import namedtuple
from numbers import Number
from typing import Generator, List

import numpy as np
from maya import cmds, mel
from maya.api import OpenMaya
from rig._internal.undo import _undo_chunk
from rig.nodetypes._base import Attribute
from rig.nodetypes._base import _cast
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.geometry import Geometry
from rig.nodetypes.object_set import ObjectSet
from rig.nodetypes.shading_engine import ShadingEngine
from rig.nodetypes.plugins import _run_undoable
from scipy.spatial import cKDTree


MAP_DATATYPE         = "doubleArray"
MAP_DEFAULT_VALUE    = 0.0
MAP_DEFAULT_CATEGORY = "PaintableMap"  # all maps will have this category assigned


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2


class Mesh(Geometry):
    """
    Mesh node class
    """

    NATIVE_NODE_TYPE = "mesh"
    FN_SET           = OpenMaya.MFnMesh
    POINT_COMP_TYPE  = "vtx"

    # built from its mesh data (``Node.create("mesh")`` with none raises, naming it)
    _CREATE_TAKES_INPUTS = "mesh_data"

    @property
    def fn_set(self) -> OpenMaya.MFnMesh:
        """A new ``MFnMesh`` of this mesh, on every access (about 3 us).

        An MFnMesh kept across an edit of the mesh made some other way (a
        ``cmds.polyUVSet`` / ``polyColorSet`` create, a UV or colour set made
        or deleted through another MFnMesh, a colour edit that adds a node to
        its history, or the undo / redo of one) can point at freed geometry and
        crash Maya when it is used again (round U,
        runs/rU/U3/clearuvs_probe.py, stale_probe.py; runs/rU2/S3/probe). Keep
        one for a single call at most.
        """
        return OpenMaya.MFnMesh(self.mdagpath)

    # --- creation

    @classmethod
    def _create(
        cls,
        mesh_data: MeshData,
        name:      str | None          = None,
        uv_data:   list[UVData] | None = None,
        **kwargs,
    ) -> str:
        """[Internal] Builds the mesh of `mesh_data` with the UV sets of `uv_data`
        (a list, or None) and returns the transform's name. `Mesh.create` wraps
        it in its undo chunk; see there."""
        return _create_mesh(mesh_data, name, uv_data)

    @classmethod
    def create(
        cls,
        mesh_data: MeshData,
        uv_data:   UVData   | UVList | list[UVData] | None = None,
        name:      str      | None                         = None,
        container: bool     | None                         = None,
    ) -> "Mesh":
        """Creates a mesh object from a mesh data object.

        Args:
            mesh_data: A MeshData object.
            uv_data: One or more UVData objects. The first one fills the default
                set (``map1``, renamed to its name when that differs); each
                other one becomes a new set of its name, in order.
            name: The name of the mesh to create (a trailing ``Shape<digits>``
                is dropped: ``"mShape2"`` names the transform ``m2`` and the
                shape ``mShape2``). None: ``mesh_data.name``, else
                ``polySurface<N>``.
            container: Inside ``with container()``, whether the transform and
                the shape are registered with the scope (None: yes), as for
                every typed create (see `DGNode.create`).

        The whole call is ONE undo step named ``rig.Mesh.create``: one
        ``cmds.undo()`` removes the transform and the shape (with the UV sets,
        the ``initialShadingGroup`` membership, the matrix and, in a scope, the
        registration), and ``cmds.redo()`` brings back the same node (its UUID,
        and any Mesh object held on it, stay valid). The mesh rides a recorded
        ``createNode`` transform: the shape is made under it through the API,
        and its UVs are written through the API before any recorded command
        touches the mesh. It never changes the selection and leaves no
        construction history.

        A face with holes is built as the triangulation of its outer loop and
        hole loops, whose internal edges one ``polyDelEdge`` (no construction
        history, no Orig shape) removes again. Its face-vertex lists may start
        at another vertex of each loop than ``mesh_data``'s (Maya's merge
        order); each UV stays on its vertex. Invalid mesh data raises
        RuntimeError before anything is made; an error after the transform is
        made deletes it again before it propagates.
        """
        from cgmath.geometry import UVData

        if uv_data:
            if isinstance(uv_data, UVData):
                uv_data = [uv_data]
        else:
            uv_data = None
        with _undo_chunk("rig.Mesh.create"):
            return super().create(mesh_data, uv_data=uv_data, name=name, container=container)

    # --- mesh geometry data methods

    @property
    def num_vertices(self) -> int:
        """Returns the number of vertices."""
        return self.fn_set.numVertices

    @property
    def num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        return self.num_vertices

    @property
    def num_polygons(self) -> int:
        """Returns the number of faces."""
        return self.fn_set.numPolygons

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the vertex points of the mesh.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the mesh's vertices.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.getPoints(space)

    def set_points(
        self, points: OpenMaya.MPointArray, world_space: bool = False
    ) -> None:
        """Set the vertex points of the mesh: one undo step.

        Args:
            points: One point per vertex: an MPointArray, a point sequence (list
                or numpy array) or a MeshData (its points). Copied: changing it
                afterwards changes nothing.
            world_space: If True, the points are in world space, otherwise in
                object space.

        Raises:
            ValueError: before any edit, when there is not one point per vertex.
        """
        from cgmath.geometry import MeshData

        # if given MeshData
        if isinstance(points, MeshData):
            points = points.points

        # a point sequence (list, numpy array) or an MPointArray, copied: the undo
        # queue keeps the points of this call (the API converts a list about twice
        # as fast as an ndarray)
        if isinstance(points, np.ndarray):
            points = points.tolist()
        points = OpenMaya.MPointArray(points)

        count = self.num_vertices
        if len(points) != count:
            raise ValueError(
                f"Mesh.set_points: {len(points)} points for the {count} vertices of {self.name}"
            )
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        _MeshSetPointsCommand(self, points, space)

    def get_closest_point(
        self,
        point:       OpenMaya.MPoint | list[float] | np.ndarray,
        world_space: bool                                       = True,
    ) -> OpenMaya.MPoint:
        """Returns the closest mesh surface point to a given point in space."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        if not isinstance(point, OpenMaya.MPoint):
            point = OpenMaya.MPoint(point)
        return self.fn_set.getClosestPoint(point, space)

    def get_vertices_above_plane(
        self,
        point:       np.ndarray,
        normal:      np.ndarray,
        offset:      float | None = None,
        world_space: bool         = True,
    ) -> list[int]:
        """Returns list of vertex numbers that are above the provided point and normal."""
        _point = point.copy()
        if offset:
            _point += offset * normal

        # Get a direction towards the plane position foreach point in the mesh.
        points   = np.array(self.get_points(world_space=world_space))
        points   = np.delete(points, 3, axis=1)
        to_plane = points - _point
        to_plane = to_plane / np.linalg.norm(to_plane)

        # numpy for dot products and filtering out indices above plane.
        dot_products = np.dot(to_plane, normal)
        return np.where(dot_products > 0)[0]

    def get_faces_above_plane(
        self,
        point:       np.ndarray,
        normal:      np.ndarray,
        offset:      float | None = None,
        world_space: bool         = True,
    ) -> list[int]:
        """Returns list of face numbers that are above the provided point and normal."""
        vert_indices = self.get_vertices_above_plane(
            point, normal, offset=offset, world_space=world_space
        )

        # Convert indices to True/False array.
        is_vert_above_plane = np.isin(np.arange(self.num_vertices), vert_indices)

        # Loop over each face and check if all vertices are above the plane.
        face_indices = []
        fn = self.fn_set
        for face_idx in range(fn.numPolygons):
            if all(
                (
                    is_vert_above_plane[i]
                    for i in fn.getPolygonVertices(face_idx)
                )
            ):
                face_indices.append(face_idx)

        return face_indices

    def get_uv_at_point(
        self,
        point:       OpenMaya.MPoint | list[float] | np.ndarray,
        uv_set:      str             | None,
        world_space: bool                                       = True,
    ) -> OpenMaya.MPoint:
        """Returns the UV coordinates of the mesh at a given point."""
        space  = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        uv_set = uv_set or self.current_uv_set
        if not isinstance(point, OpenMaya.MPoint):
            point = OpenMaya.MPoint(point)
        return self.fn_set.getUVAtPoint(point, space, uv_set)

    def get_vertex_normals(
        self, angle_weighted: bool = False, world_space: bool = True
    ) -> OpenMaya.MFloatVectorArray:
        """Returns the vertex normals of the mesh."""
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.getVertexNormals(angle_weighted, space)

    def get_per_face_vertex_normals(self):
        """
        For a given mesh, return a list of per face vertex normals.
        Return:
            OpenMaya.MFloatVectorArray:  list of normals
        """

        normals = []
        mSel    = OpenMaya.MSelectionList()
        mSel.add(self.name)
        dag_path  = mSel.getDagPath(0)

        mesh_iter = OpenMaya.MItMeshFaceVertex(dag_path)
        while not mesh_iter.isDone():
            normal = mesh_iter.getNormal(OpenMaya.MSpace.kWorld)
            normals.append(normal)

            mesh_iter.next()

        return normals

    # --- materials

    def get_materials(
        self, as_pairs: bool = False
    ) -> list[DAGNode] | list[tuple[ShadingEngine, DAGNode]]:
        """Returns a list of materials associated with this geometry.

        Args:
            as_pairs: If True, returns a list of (shading_engine, material) pairs.
                Otherwise return a list of materials.
        """
        nodes = []
        shading_engines = set(
            cmds.listConnections(self.name, d=True, s=False, type="shadingEngine") or []
        )
        if not shading_engines:
            return nodes

        for sg in shading_engines:
            sg = ShadingEngine(sg)
            # skip empty shading engines
            if not sg.get_members(as_components=True):
                continue

            # skip shading engines with no material
            mat = cmds.listConnections(
                f"{sg}.surfaceShader", source=True, destination=False, plugs=False
            )
            if not mat:
                continue
            mat = _cast(mat[0])
            if not as_pairs:
                nodes.append(mat)
            else:
                nodes.append((sg, mat))

        return nodes

    def get_shading_engines(self) -> list[ShadingEngine]:
        """Returns a list of shading engines associated with this geometry.

        Args:
            as_pairs: If True, returns a list of (shading_engine, material) pairs.
                Otherwise return a list of shading engines.
        """
        return [x for x, _ in self.get_materials(as_pairs=True)]

    def get_material_bindings(
        self, verbose: bool = False
    ) -> ObjectSet | list[tuple[DAGNode, np.ndarray]]:
        """Get the binding relationship between this geometry and the
        associated materials.

        Returns:
            If the entire geom is bind to a single material, returns the material
            node. Otherwise return a list of (material, face_indices) pairs.

            if verbose is True the output will always be (material, face_indices) pairs.
        """
        bindings          = {}
        face_count        = self.num_polygons
        whole_mesh_mat    = None  # used to record the material assigned to the entire mesh
        assigned_face_ids = set()
        for sg, mat in self.get_materials(as_pairs=True):
            indices = set()
            for s, comp in sg.get_members(as_components=True):
                if s == self and comp:
                    comp = OpenMaya.MFnSingleIndexedComponent(comp)
                    indices.update(comp.getElements())

            if not indices or len(indices) == face_count:
                whole_mesh_mat = mat
            else:
                assigned_face_ids.update(indices)
                bindings.setdefault(mat, set())
                bindings[mat].update(indices)

        # only a single material covers the entire mesh
        if not bindings and whole_mesh_mat and not verbose:
            return whole_mesh_mat

        # use the single material to cover missing faces, if found
        if len(assigned_face_ids) != face_count and whole_mesh_mat:
            missing_face_ids = set(range(face_count)) - assigned_face_ids
            bindings.setdefault(whole_mesh_mat, set())
            bindings[whole_mesh_mat].update(missing_face_ids)

        return [(x, np.array(sorted(bindings[x]))) for x in sorted(bindings.keys())]

    # --- UV methods

    @property
    def uv_sets(self) -> list[str]:
        """Returns a list of UV set names."""
        return self.fn_set.getUVSetNames()

    @property
    def current_uv_set(self) -> str:
        """Returns the current uv set."""
        return self.fn_set.currentUVSetName()

    def add_uv_set(self, uv_set: str) -> None:
        """Adds an empty UV set: one undo step.

        Args:
            uv_set: A uv set name to add.

        Raises:
            RuntimeError: The mesh already has a UV set of that name.
        """
        if uv_set in self.uv_sets:
            raise RuntimeError(f"UV set {uv_set} already exists.")
        _MeshAddUVSetCommand(self, uv_set)

    def delete_uv_set(self, uv_set: str | None = None) -> None:
        """Deletes a given uv set: one undo step, whose undo brings the set back
        with its UVs (current again if it was).

        Args:
            uv_set: A uv set name to delete. If None, use current uv set.

        Raises:
            RuntimeError: The mesh has no UV set of that name, or it is the
                mesh's default (first) set, which Maya never deletes.
        """
        uv_set = uv_set or self.current_uv_set
        names  = self.uv_sets
        if uv_set not in names:
            raise RuntimeError(f"UV set {uv_set} does not exist.")
        if uv_set == names[0]:
            raise RuntimeError("The default uv set cannot be deleted.")
        _MeshDeleteUVSetCommand(self, uv_set)

    def delete_map(self, map_name: str | None = None) -> None:
        """Deletes a given map.

        Args:
            map_name: A map name to delete. If None, use current map.
        """
        map_name = f"{self.name}.{map_name}"
        if cmds.objExists(map_name):
            cmds.deleteAttr(map_name)

    def rename_uv_set(self, new_name: str, uv_set: str | None = None) -> None:
        """Renames the current uv set: one undo step.

        Args:
            new_name: New uv set name.
            uv_set: A uv set name to rename. If None, use current uv set.

        Raises:
            RuntimeError: The mesh has no UV set `uv_set`, or has one named
                `new_name` already.
        """
        uv_set = uv_set or self.current_uv_set
        if uv_set != new_name:
            names = self.uv_sets
            if uv_set not in names:
                raise RuntimeError(f"UV set {uv_set} does not exist.")
            if new_name in names:
                raise RuntimeError(f"UV set {new_name} already exists.")
            _MeshRenameUVSetCommand(self, uv_set, new_name)

    def get_uv_coords(self, uv_set: str | None = None) -> np.ndarray:
        """Returns an array of uv coordinates.

        Args:
            uv_set: A uv set name to query. If None, use current uv set.
        """
        uv_set = uv_set or self.current_uv_set
        u_vals, v_vals = self.fn_set.getUVs(uv_set)
        uv_coords       = np.empty((len(u_vals), 2))
        uv_coords[:, 0] = np.array(u_vals)
        uv_coords[:, 1] = np.array(v_vals)
        return uv_coords

    def get_assigned_uvs(
        self, uv_set: str | None = None
    ) -> tuple[OpenMaya.MIntArray, OpenMaya.MIntArray]:
        """Returns a tuple of (uv_counts, uv_ids)

        * uv_counts: The container for the uv counts for each polygon in the mesh
        * uv_ids: The container for the uv indices mapped to each polygon-vertex

        Args:
            uv_set: A uv set name to query. If None, use current uv set.
        """
        uv_set = uv_set or self.current_uv_set
        return self.fn_set.getAssignedUVs(uv_set)

    def set_uv_data(self, uv_data: UVData, uv_set: str | None = None) -> None:
        """Sets the data of an uv set.

        The set's UVs and their assignment are replaced by `uv_data`'s (the UV
        count may change; data without UVs empties the set), then the set is
        renamed ``uv_data.name`` when that differs from `uv_set`. One undo
        step, whose undo puts back the old name, UVs and assignment.

        On a mesh with history (construction history or a deformer) the set is
        never cleared (Maya's ``clearUVs`` there clears the current set): data
        with fewer UVs than the set has raises, and the undo of data with more
        puts back the old UVs and assignment but keeps the extra UVs in the
        set, unassigned.

        Args:
            uv_data: A UVData object.
            uv_set: A uv set name to operate on. If None, use current uv set.

        Raises:
            RuntimeError: before any edit, when `uv_set` does not exist, another
                set is named ``uv_data.name`` already, the data does not fit the
                mesh (not one UV count per face, counts not adding up to the
                indices, an index that is not one of the UVs), or it has fewer
                UVs than the set of a mesh with history.
            TypeError: ``uv_data.name`` is not a non-empty str.
        """
        uv_set = uv_set or self.current_uv_set
        _check_uv_data(OpenMaya.MFnMesh(self.mdagpath), uv_set, uv_data)
        _MeshSetUVDataCommand(self, uv_set, uv_data)

    # --- deformation

    def apply_skin_data(self, skin_data: SkinData):
        """Apply skin data to this mesh."""
        from cgmath.geometry import SkinData

        # avoid circular import
        from rig.nodetypes import SkinCluster

        # reuse an existing skincluster, or create one directly from the data
        skin = self.get_deformers(node_type="skinCluster")
        if skin:
            skin = skin[0]
            skin.set_weights(skin_data)
        else:
            skin = SkinCluster.create(self, skin_data)
        return skin

    def bake_deformation(self, max_influences: int | None = None):
        """Bakes the deformation to a skincluster on this mesh.

        Args:
            max_influences: The maximum number of influences to bake.

        Returns:
            The baked skincluster.
        """
        # set to bind pose
        cmds.select(self.name, replace=True)
        mel.eval("gotoBindPose;")

        # bake delta mush, use a high max inf count to maintain fidelity
        cmds.bakeDeformer(
            srcSkeletonName = "root_joint",
            srcMeshName     = self.name,
            dstSkeletonName = "root_joint",
            dstMeshName     = self.name,
            maxInfluences   = 8,
        )

        # set max inf
        if max_influences is not None:
            skin = self.get_deformers(node_type="skinCluster")[0]
            skin.set_max_influences(4)

        return skin

    # --- serialization

    def serialize_uv(self, uv_set: str | None = None) -> UVData:
        """Serialize an UV set.

        Args:
            uv_set: A uv set name to query. If None, use current uv set.

        Returns:
            A UVData object.
        """
        from cgmath.geometry import UVData

        uv_set = uv_set or self.current_uv_set
        points = self.get_uv_coords(uv_set)
        uv_counts, uv_ids = self.get_assigned_uvs(uv_set)
        return UVData(
            name    = uv_set,
            indices = np.asarray(uv_ids),
            counts  = np.asarray(uv_counts),
            points  = np.asarray(points),
        )

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> MeshData | tuple[MeshData, UVList]:
        """Serialize this mesh.

        Args:
            world_space: If True, stores the worldSpace matrix
            include_uvs: If True, serialize all uv sets. Otherwise skip them.

        Returns:
            A MeshData object or (MeshData, [UVData]).
        """
        from cgmath.geometry import MeshData, UVList

        fn     = self.fn_set
        points = self.get_points(world_space=world_space)
        vert_counts, vert_ids = fn.getVertices()

        space      = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        normals    = fn.getNormals(space)
        normal_ids = []
        for face_id in range(fn.numPolygons):
            normal_ids.extend(fn.getFaceNormalIds(face_id))

        mesh_data = MeshData(
            indices        = np.asarray(vert_ids),
            counts         = np.asarray(vert_counts),
            points         = np.asarray(points)[:, :-1],
            normals        = np.asarray(normals),
            normal_indices = np.asarray(normal_ids),
            name           = self.clean_name,
        )

        # extract hole data from faces with interior boundaries
        holes = fn.getHoles()
        if holes:
            hole_faces, hole_verts = zip(*holes)
            mesh_data.hole_faces  = np.asarray(hole_faces)
            mesh_data.hole_counts = np.asarray([len(hv) for hv in hole_verts])
            mesh_data.hole_indices = np.concatenate(
                [np.asarray(hv) for hv in hole_verts]
            )

        # record desired matrix
        if world_space:
            mesh_data.matrix = np.eye(4)
        else:
            mesh_data.matrix = np.array(self.get_parent().find_attr("wm").get()).reshape(4, 4)

        if include_uvs:
            uv_list = UVList([self.serialize_uv(x) for x in self.uv_sets])
            if mesh_data.hole_faces is not None:
                _propagate_holes_to_uvs(mesh_data, uv_list)
            return mesh_data, uv_list
        return mesh_data

    # --- paintable map methods

    @property
    def paintable_maps(self) -> list[str]:
        """Returns a list of paintable maps."""
        return [x.name for x in self.get_map_attrs()]

    def get_map_data(self, map_name: Attribute | str, sparse: bool = True) -> MapData:
        """Get the map data from a given map attr.

        Args:
            map_name: The map attr name or plug to get the data from.

        Returns:
            A  MapData object
        """
        from cgmath.geometry import MapData

        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        # cap filtered ids & values using vert count,
        # since Maya doesn't auto-clear stale map values...
        ids, values = map_attr.filter_array_values(
            MAP_DEFAULT_VALUE, sparse=sparse, cap_count=self.num_vertices
        )
        # remove default category as all maps has it
        categories = set(map_attr.get_categories())
        categories.remove(MAP_DEFAULT_CATEGORY)

        return MapData(
            name          = map_attr.name,
            categories    = sorted(categories),
            indices       = ids,
            values        = values,
            default_value = MAP_DEFAULT_VALUE,
        )

    def serialize_maps(
        self,
        match_name: list[str] | str | None       = None,
        category:   str       | list[str] | None = None,
        sparse:     bool                         = True,
    ) -> list[MapData]:
        """Serialize maps into a dict.

        Args:
            match_name: One or more map name or regex.
                If None, serialize all maps.
            category: One or more categories to filter maps by.
                If None use MAP_DEFAULT_CATEGORY

        Returns:
            A list of MapData objects.
        """
        from cgmath.geometry import MapData

        data_list = []
        for map_attr in self.get_map_attrs(match_name=match_name, category=category):
            data_list.append(self.get_map_data(map_attr, sparse=sparse))
        return data_list

    def add_map(
        self,
        map_name:  str,
        paintable: bool                           = False,
        category:  str         | list[str] | None = None,
        values:    list[float] | float | None     = None,
        force:     bool                           = False,
    ) -> Attribute:
        """Adds a map attribute.

        Args:
            map_name: Name of the map attribute.
            paintable: Whether to make the map paintable.
            category: One or more categories to assign to this map.
                MAP_DEFAULT_CATEGORY is always assigned.
            values: Values to initialize the map with. Can be a list or a single value.
            force: If True, remove existing map if found.

        Returns:
            The map attribute created.
        """
        if self.has_attr(map_name):
            if force:
                self.delete_attr(map_name)
            else:
                raise RuntimeError("{} map already exists.".format(map_name))

        # ensure MAP_DEFAULT_CATEGORY is always included
        category = category or MAP_DEFAULT_CATEGORY
        if isinstance(category, str):
            category = [category]
        if MAP_DEFAULT_CATEGORY not in category:
            category.append(MAP_DEFAULT_CATEGORY)

        map_attr = self.add_attr(
            map_name,
            storable = True,
            dataType = MAP_DATATYPE,
            category = category,
        )

        self.set_map_values(map_attr, values)
        if paintable:
            self.set_attr_paintable(map_name)
        return map_attr

    def get_map_values(self, map_name: Attribute | str) -> list[float]:
        """Returns the values of a given map."""
        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        values     = map_attr.get()
        vert_count = self.num_vertices

        # cap values using vert count
        # since Maya doesn't auto-clear stale map values...
        if values and len(values) > vert_count:
            return values[:vert_count]

        # return defaults
        if not values:
            return [MAP_DEFAULT_VALUE] * vert_count

        return values

    def set_map_values(
        self, map_attr: Attribute | str, values: list[float] | float | MapData | None
    ) -> None:
        """Sets the values of a given map.

        Args:
            map_name: The map attr object or name to set the value to.
            values: value array, or a single value, or None (used MAP_DEFAULT_VALUE).
        """
        from cgmath.geometry import MapData

        attr: Attribute = self.find_map_attr(map_attr, failfast=True)
        if isinstance(values, MapData):
            vals = values.to_dense_array(self.num_vertices)
        else:
            vals = self.sanitize_map_values(values)
        attr.set(vals)

    def duplicate_map(
        self,
        map_name:     Attribute | str,
        new_map_name: str,
        paintable:    bool            = False,
    ) -> Attribute:
        """Duplicates an existing map attribute.

        Args:
            map_name: A map to duplicate.
            new_map_name: New name for the duplicated map.
            paintable: Whether to make the map paintable.

        Returns:
            The new map attribute.
        """
        if not self.has_attr(map_name):
            raise RuntimeError("{} map not found.".format(map))
        if self.has_attr(new_map_name):
            raise RuntimeError("{} map already exists.".format(new_map_name))
        values = self.get_map_values(map_name)
        return self.add_map(new_map_name, paintable=paintable, values=values)

    def find_map_attr(
        self, map_name: Attribute | str, failfast: bool = False
    ) -> Attribute | None:
        """Finds a map attr by name.

        Raises:
            ValueError: If `failfast` = True and no map is found.
        """
        attr = self.find_attr(map_name)
        if (
            attr
            and attr.data_type == MAP_DATATYPE
            and attr.has_category(MAP_DEFAULT_CATEGORY)
        ):
            return attr
        if failfast:
            raise ValueError(f"{map_name} is not a map attribute.")

    def get_map_attrs(
        self,
        match_name: list[str] | str | None = None,
        category:   list[str] | str | None = None,
    ) -> list[Attribute]:
        """Returns a list of map attributes.

        Args:
            match_name: One or more map name or regex.
                If None, serialize all maps.
            category: One or more category to filter maps by.
                MAP_DEFAULT_CATEGORY is always used.
        """
        names = [match_name] if isinstance(match_name, str) else match_name
        regex = re.compile(r"|".join([rf"^{x}$" for x in names])) if names else None

        category = category or MAP_DEFAULT_CATEGORY
        return [
            x
            for x in self.list_attr(userDefined=True)
            if x.data_type == MAP_DATATYPE
            and (not category or x.has_category(category))
            and (not regex or regex.match(x.name))
        ]

    def mirror_map(
        self,
        map_name:       Attribute | str,
        mirror_axis:    Axis             = Axis.X,
        mirror_inverse: bool      | None = False,
    ) -> None:
        """Mirrors this map.

        TODO code is pretty hacky at the moment, needs to be refactored for speed improvements
        Replicate closer to Maya's copySkinWeights mirrormode options
        """
        cur_map_vals = self.get_map_values(map_name)
        vert_points  = self.get_points(world_space=False)
        vert_count   = len(vert_points)

        np_mesh_points_array = np.empty((vert_count, 3), dtype=float)

        for i in range(vert_count):
            np_mesh_points_array[i, 0] = vert_points[i].x
            np_mesh_points_array[i, 1] = vert_points[i].y
            np_mesh_points_array[i, 2] = vert_points[i].z

        mesh_idx_and_points_to_mirror = []

        invert_axis_val = -1.0 if mirror_inverse else 1.0
        for idx, vert_pos in enumerate(np_mesh_points_array):
            mirror_value = vert_pos[mirror_axis.value] * invert_axis_val
            if mirror_value > 0.0:
                mirror_pos                    = np.copy(vert_pos)
                mirror_pos[mirror_axis.value] = mirror_pos[mirror_axis.value] * -1
                mesh_idx_and_points_to_mirror.append([idx, mirror_pos])

        kdtree = cKDTree(np_mesh_points_array)
        # # find mirror index
        for vert_idx, mirror_vert_pos in mesh_idx_and_points_to_mirror:
            dst, idx = kdtree.query(mirror_vert_pos)
            cur_map_vals[idx] = cur_map_vals[vert_idx]

        self.set_map_values(map_name, cur_map_vals)

    def merge_maps(self, map_source: str, map_target: str):
        raise NotImplementedError

    def merge_maps_list(
        self, map_merge_from_list: list[str], map_merge_to: str | None = None
    ):
        raise NotImplementedError

    def sanitize_map_values(self, values: list[float] | float | None) -> list[float]:
        """Returns a sanitized value list that matches vert count of this mesh.

        Args:
            values: Values to sanitize.
                Value list: resize it to match vert count by pruning extras
                    or padding with 0s.
                Single value: create a list of mesh vert size of that value.

        Returns:
            sanitized values.
        """
        values = MAP_DEFAULT_VALUE if values is None else values

        # a single value
        vert_count = self.num_vertices
        if isinstance(values, Number):
            return [float(values)] * vert_count

        val_array = np.array(values, dtype=float)
        count     = val_array.size
        if count == vert_count:
            return val_array.tolist()
        elif count > vert_count:
            return val_array[:vert_count].tolist()
        else:
            val_array = np.append(
                val_array, np.repeat(MAP_DEFAULT_VALUE, vert_count - count)
            )
            return val_array.tolist()

    def paint_map(self, map_name: Attribute | str) -> None:
        """Enable painting mode on the given map."""
        map_attr: Attribute = self.find_map_attr(map_name, failfast=True)
        self.set_attr_paintable(map_name)
        paint_attr = f"{self.NATIVE_NODE_TYPE}.{self.long_name}.{map_attr.name}"
        cmds.select(map_attr.node, replace=True)
        mel.eval(
            f'artSetToolAndSelectAttr("artAttrCtx", "{paint_attr}");artAttrInitPaintableAttr;'
        )

    # --- Color Set methods

    def get_color_sets(self) -> List[ColorSet]:
        """Returns a list of the colors sets for this object"""
        return [
            ColorSet(mesh=self, name=name) for name in self.fn_set.getColorSetNames()
        ]

    def add_color_set(
        self, name: str, representation: ColorSet.Representation
    ) -> ColorSet:
        """Create a color set with the given name and representation (not
        clamped): one undo step. The first colour set of a mesh becomes its
        current one.

        Raises:
            ValueError: `representation` is not a ``ColorSet.Representation``
                (or its value), or the mesh already has a colour set `name`.
        """
        # cast or throw ValueError
        representation = ColorSet.Representation(representation)

        if name in self.fn_set.getColorSetNames():
            raise ValueError(f"{self.name} has color set {name}!")
        _MeshAddColorSetCommand(self, name, representation.value)
        return ColorSet(mesh=self, name=name)

    # --- data transfer

    def transfer_component_tags(
        self, tags: str | list[str], other: Mesh | str, **kwargs
    ) -> None:
        """Transfer component tags from this mesh to the other mesh.

        Args:
            tags: One or more component tags to transfer.
            other: The other mesh to transfer the component tag to.
            kwargs: Transfer kwargs supported in MeshDataResampler.resample_map().
        """
        from cgmath.geometry.resample import MeshDataResampler

        other = Mesh(other)

        mesh_data_a, uv_data_a = self.serialize(include_uvs=True)
        mesh_data_b, uv_data_b = other.serialize(include_uvs=True)
        resampler = MeshDataResampler(
            mesh_data_a, mesh_data_b, uv_data_a[0], uv_data_b[0]
        )

        for t in tags if isinstance(tags, (list, tuple)) else [tags]:
            if not other.has_component_tag(t):
                other.add_component_tag(t)

            data = self.get_component_tag_data(t)

            new_data = []
            if data.indices.size > 0:
                new_data = resampler.resample_map(data, **kwargs)

            other.set_component_tag_contents(t, new_data)

    def transfer_maps(self, maps: str | list[str], other: Mesh | str, **kwargs) -> None:
        """Transfer a mesh map from this mesh to the other mesh.

        Args:
            maps: One or more map names to transfer.
            other: The other mesh to transfer the map to.
            kwargs: Transfer kwargs supported in MeshDataResampler.resample_map().
        """
        from cgmath.geometry.resample import MeshDataResampler

        other = Mesh(other)
        mesh_data_a, uv_data_a = self.serialize(include_uvs=True)
        mesh_data_b, uv_data_b = other.serialize(include_uvs=True)
        resampler = MeshDataResampler(
            mesh_data_a, mesh_data_b, uv_data_a[0], uv_data_b[0]
        )

        for m in maps if isinstance(maps, (list, tuple)) else [maps]:
            data     = self.get_map_data(m)
            new_data = resampler.resample_map(data, **kwargs)
            other.add_map(m, new_data, force=True)
            other.set_map_values(m, new_data)


def _triangulate_holed_face(
    points:             np.ndarray,
    face_verts:         np.ndarray,
    hole_loops:         list[list[int]],
    out_counts:         list[int],
    out_indices:        list[int],
    out_internal_edges: set[tuple[int, int]],
) -> None:
    """CDT-triangulate a face with holes and collect internal edges.

    Appends triangle counts/indices to *out_counts*/*out_indices* and
    adds non-boundary edges (as global ``(min, max)`` vertex pairs) to
    *out_internal_edges* for later deletion.
    """
    from cgmath.geometry._cdt import get_boundary_edges, triangulate_ngon

    hole_vert_set = set()
    for hvs in hole_loops:
        hole_vert_set.update(hvs)
    outer = [int(v) for v in face_verts if v not in hole_vert_set]

    local_tris = triangulate_ngon(points, outer, hole_loops)

    # local->global index mapping
    face_indices = list(outer)
    for hb in hole_loops:
        face_indices.extend(hb)

    # boundary edges that must NOT be deleted
    boundary = get_boundary_edges(outer)
    for hb in hole_loops:
        boundary |= get_boundary_edges(hb)

    for tri in local_tris:
        ga, gb, gc = face_indices[tri[0]], face_indices[tri[1]], face_indices[tri[2]]
        out_counts.append(3)
        out_indices.extend([ga, gb, gc])
        for e in [(ga, gb), (gb, gc), (gc, ga)]:
            ek = (min(e), max(e))
            if ek not in boundary:
                out_internal_edges.add(ek)


def _delete_internal_edges(
    shape: OpenMaya.MObject, internal_edges: set[tuple[int, int]]
) -> None:
    """Deletes the triangulation edges of the holed faces (``(min, max)``
    vertex pairs) from the mesh `shape`, restoring each N-gon with its holes.

    One ``polyDelEdge`` of every edge, ``cleanVertices=False`` (no vertex
    goes) and ``constructionHistory=False`` (no ``polyDelEdge`` node, no Orig
    shape). Each edge is found from one of its vertices (a lookup per edge,
    not a scan of the mesh). The shape's UVs must be written before this call:
    it is a recorded command that rewrites the mesh, so API data written after
    it would be lost on redo (the ride rule), while data written before it
    comes back with every undo / redo, each UV on its vertex.
    """
    fn     = OpenMaya.MFnMesh(shape)
    it     = OpenMaya.MItMeshVertex(shape)
    found  = set()
    for a, b in internal_edges:
        a, b = int(a), int(b)
        it.setIndex(a)
        for eid in it.getConnectedEdges():
            v0, v1 = fn.getEdgeVertices(eid)
            if v0 == b or v1 == b:
                found.add(eid)
                break
    if found:
        path = OpenMaya.MFnDagNode(shape).fullPathName()
        cmds.polyDelEdge(
            [f"{path}.e[{eid}]" for eid in sorted(found)],
            cleanVertices=False,
            constructionHistory=False,
        )


def _mesh_topology(
    mesh_data: MeshData,
) -> tuple[object, object, set[tuple[int, int]], dict[int, list[int]]]:
    """Checks `mesh_data` and returns what `MFnMesh.create` builds from it:
    ``(counts, indices, internal_edges, hole_triangles)``, the counts and
    indices as lists (the API converts a list 2-3x faster than an ndarray).

    Without holes: the data's own counts / indices, no edges and no
    triangles. With holes, every holed face is CDT-triangulated in place
    (`_triangulate_holed_face`), ``internal_edges`` holds the triangulation
    edges `_delete_internal_edges` removes again, and ``hole_triangles`` maps
    each holed face to the flat vertex ids of its triangles.

    Raises RuntimeError, before anything is made, when the face counts do not
    add up to the indices, an index is not a point, or the hole arrays do not
    match each other, the faces or the points.
    """
    counts   = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    indices  = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    n_points = len(mesh_data.points)
    if int(counts.sum()) != len(indices):
        raise RuntimeError(
            f"MeshData: the face counts add up to {int(counts.sum())} face-vertices, "
            f"but there are {len(indices)} indices"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= n_points):
        raise RuntimeError(
            f"MeshData: a face index is not one of the {n_points} points "
            f"(indices range {int(indices.min())}..{int(indices.max())})"
        )

    has_holes = mesh_data.hole_faces is not None and len(mesh_data.hole_faces) > 0
    if not has_holes:
        return counts.tolist(), indices.tolist(), set(), {}

    if mesh_data.hole_counts is None or mesh_data.hole_indices is None:
        raise RuntimeError("MeshData: hole_faces is set without hole_counts / hole_indices")
    hole_faces   = np.asarray(mesh_data.hole_faces, dtype=np.int64).ravel()
    hole_counts  = np.asarray(mesh_data.hole_counts, dtype=np.int64).ravel()
    hole_indices = np.asarray(mesh_data.hole_indices, dtype=np.int64).ravel()
    if len(hole_faces) != len(hole_counts) or int(hole_counts.sum()) != len(hole_indices):
        raise RuntimeError(
            f"MeshData: {len(hole_faces)} hole faces, {len(hole_counts)} hole counts "
            f"adding up to {int(hole_counts.sum())}, and {len(hole_indices)} hole indices "
            "do not match"
        )
    if int(hole_faces.min()) < 0 or int(hole_faces.max()) >= len(counts):
        raise RuntimeError(f"MeshData: a hole face is not one of the {len(counts)} faces")
    if len(hole_indices) and (
        int(hole_indices.min()) < 0 or int(hole_indices.max()) >= n_points
    ):
        raise RuntimeError(f"MeshData: a hole index is not one of the {n_points} points")

    hole_map       = _build_hole_map(mesh_data)
    new_counts     = []
    new_indices    = []
    internal_edges = set()
    hole_triangles = {}
    offset         = 0
    for fi in range(len(counts)):
        c          = int(counts[fi])
        face_verts = mesh_data.indices[offset : offset + c]
        if fi not in hole_map:
            new_counts.append(c)
            new_indices.extend(int(v) for v in face_verts)
        else:
            start = len(new_indices)
            _triangulate_holed_face(
                mesh_data.points,
                face_verts,
                hole_map[fi],
                new_counts,
                new_indices,
                internal_edges,
            )
            hole_triangles[fi] = new_indices[start:]
        offset += c
    return new_counts, new_indices, internal_edges, hole_triangles


# a UV set's data in the shape `_write_uv_set` reads (a UVData has these fields)
_UVArrays = namedtuple("_UVArrays", "name points counts indices")


def _triangulated_uvs(
    mesh_data: MeshData, uv_data: UVData, hole_triangles: dict[int, list[int]]
) -> _UVArrays:
    """`uv_data`'s assignment for the mesh `_mesh_topology` builds: each holed
    face's UVs are handed to its triangles by vertex (every other face keeps
    its own), so each UV stays on its vertex through `_delete_internal_edges`.

    Raises RuntimeError when the UV counts do not match the faces (one count
    per face, 0 or the face's vertex count on a holed face) or the UV indices.
    """
    mesh_counts = np.asarray(mesh_data.counts, dtype=np.int64).ravel()
    mesh_ids    = np.asarray(mesh_data.indices, dtype=np.int64).ravel()
    uv_counts   = np.asarray(uv_data.counts, dtype=np.int64).ravel()
    uv_ids      = np.asarray(uv_data.indices, dtype=np.int64).ravel()
    if len(uv_counts) != len(mesh_counts) or int(uv_counts.sum()) != len(uv_ids):
        raise RuntimeError(
            f"UVData {uv_data.name!r}: {len(uv_counts)} face counts adding up to "
            f"{int(uv_counts.sum())} for {len(mesh_counts)} faces and {len(uv_ids)} indices"
        )
    mesh_starts = np.concatenate(([0], np.cumsum(mesh_counts)))
    uv_starts   = np.concatenate(([0], np.cumsum(uv_counts)))
    counts, ids = [], []
    previous    = 0
    for face in sorted(hole_triangles):
        # the faces before this one keep their assignment
        counts.append(uv_counts[previous:face])
        ids.append(uv_ids[uv_starts[previous] : uv_starts[face]])
        triangles = hole_triangles[face]
        n_uvs     = int(uv_counts[face])
        if n_uvs == 0:
            counts.append(np.zeros(len(triangles) // 3, dtype=np.int64))
        elif n_uvs == int(mesh_counts[face]):
            by_vertex = dict(
                zip(
                    mesh_ids[mesh_starts[face] : mesh_starts[face + 1]].tolist(),
                    uv_ids[uv_starts[face] : uv_starts[face + 1]].tolist(),
                )
            )
            try:
                ids.append(np.array([by_vertex[v] for v in triangles], dtype=np.int64))
            except KeyError as exc:
                raise RuntimeError(
                    f"UVData {uv_data.name!r}: holed face {face} has no UV for vertex {exc}"
                ) from None
            counts.append(np.full(len(triangles) // 3, 3, dtype=np.int64))
        else:
            raise RuntimeError(
                f"UVData {uv_data.name!r}: holed face {face} has {n_uvs} UVs for "
                f"{int(mesh_counts[face])} face-vertices"
            )
        previous = face + 1
    counts.append(uv_counts[previous:])
    ids.append(uv_ids[uv_starts[previous] :])
    return _UVArrays(uv_data.name, uv_data.points, np.concatenate(counts), np.concatenate(ids))


def _write_uv_set(
    fn: OpenMaya.MFnMesh, uv_set: str, uv_data: UVData, clear: bool = True
) -> None:
    """Replaces the UVs of the set `uv_set` of the mesh `fn` with `uv_data`'s
    (``points`` as ``(n, 2)``, ``counts``, ``indices``; a UVData or a
    `_UVArrays`). ``clearUVs`` first, so a different UV count or an empty
    set is written exactly (``clear=False`` skips it for a set known to be
    empty, a new one: about 1 ms on 10k faces); nothing more when the data
    holds no UVs. Pure API (no cmds): safe in a journal item's undo / redo."""
    if clear:
        fn.clearUVs(uv_set)
    points  = np.asarray(uv_data.points)
    indices = np.asarray(uv_data.indices)
    if len(points) or len(indices):
        fn.setUVs(points[:, 0].tolist(), points[:, 1].tolist(), uv_set)
        fn.assignUVs(np.asarray(uv_data.counts).tolist(), indices.tolist(), uv_set)


def _mesh_create_name(name: str | None) -> str | None:
    """The name `Mesh.create` gives the transform: `name` without a trailing
    ``Shape<digits>`` (``"mShape2"`` is ``"m2"``, whose shape Maya names
    ``mShape2``); None when there is no name."""
    if name and re.search("Shape[0-9]*$", name):
        name = "".join(name.rpartition("Shape")[::2])
    return name or None


# the flat 4x4 identity: a MeshData.matrix that needs no xform
_IDENTITY_16 = np.eye(4).ravel()


def _create_mesh(
    mesh_data: MeshData, name: str | None, uv_data: list[UVData] | None
) -> str:
    """`Mesh._create`'s body: the ride. Returns the transform's name.

    In this order (`Mesh.create` holds the one undo chunk around it):

    1. check the data, triangulate the holed faces and hand their UVs to the
       triangles (`_mesh_topology`, `_triangulated_uvs`): nothing is made yet;
    2. ``cmds.createNode("transform", name="polySurface#", skipSelect=True)``,
       the recorded node the mesh rides with (today's default name);
    3. ``MFnMesh.create(..., parent=<transform>)``: the shape, under it;
    4. the UV sets through the API (`_write_uv_set`): the first into the
       current set (renamed to its name when that differs), each other one
       into a new set of its name;
    5. the holed faces' internal edges (`_delete_internal_edges`), after the
       UVs;
    6. ``cmds.sets`` into ``initialShadingGroup``, ``cmds.rename`` of the
       transform (Maya renames the shape after it, as it always did) and
       ``cmds.xform`` of the data's matrix (skipped for the identity, which
       the new transform already has).

    Any error after step 2 deletes the transform before it propagates.
    """
    counts, indices, internal_edges, hole_triangles = _mesh_topology(mesh_data)
    uv_sets = list(uv_data or ())
    if hole_triangles:
        uv_sets = [_triangulated_uvs(mesh_data, uv, hole_triangles) for uv in uv_sets]
    name = _mesh_create_name(name or mesh_data.name)

    xform  = cmds.createNode("transform", name="polySurface#", skipSelect=True)
    sel    = OpenMaya.MSelectionList()
    sel.add(xform)
    parent = sel.getDependNode(0)
    handle = OpenMaya.MObjectHandle(parent)
    try:
        shape = Mesh.FN_SET().create(
            OpenMaya.MPointArray(np.asarray(mesh_data.points).tolist()),
            counts,
            indices,
            parent=parent,
        )
        fn = OpenMaya.MFnMesh(shape)
        for i, uv in enumerate(uv_sets):
            if i == 0:
                # the default set (new, so empty)
                current = fn.currentUVSetName()
                _write_uv_set(fn, current, uv, clear=False)
                if uv.name != current:
                    fn.renameUVSet(current, uv.name)
            else:
                if uv.name in fn.getUVSetNames():
                    raise RuntimeError(f"UV set {uv.name} already exists.")
                fn.createUVSet(uv.name)
                _write_uv_set(fn, uv.name, uv, clear=False)
        if internal_edges:
            _delete_internal_edges(shape, internal_edges)
        cmds.sets(xform, forceElement="initialShadingGroup")
        if name:
            # the shape follows the transform (polySurfaceShape<N> -> <name>Shape)
            xform = cmds.rename(xform, name)
        matrix = mesh_data.matrix.ravel()
        if not np.array_equal(matrix, _IDENTITY_16):
            # a new transform is at rest already (world-space data has no matrix)
            cmds.xform(xform, matrix=matrix)
    except BaseException:
        if handle.isValid():
            cmds.delete(OpenMaya.MFnDagNode(parent).fullPathName())
        raise
    return xform


def _build_hole_map(mesh_data: MeshData) -> dict[int, list[list[int]]]:
    """Builds a mapping of face_index -> list of hole vertex index lists."""
    from cgmath.geometry import MeshData

    hole_map: dict[int, list[list[int]]] = {}
    offset = 0
    for i in range(len(mesh_data.hole_faces)):
        face_idx = int(mesh_data.hole_faces[i])
        count    = int(mesh_data.hole_counts[i])
        verts    = list(mesh_data.hole_indices[offset : offset + count])
        hole_map.setdefault(face_idx, []).append(verts)
        offset += count
    return hole_map


def _propagate_holes_to_uvs(mesh_data: MeshData, uv_list: UVList) -> None:
    """Copy hole metadata from MeshData to each UVData, remapping vertex indices.

    MeshData.indices and UVData.indices are parallel (same face-vertex order),
    so for each mesh hole vertex we read the UV vertex at the same position.
    """
    from cgmath.geometry import MeshData, UVList

    face_starts = np.concatenate([[0], np.cumsum(mesh_data.counts[:-1])])

    for uv in uv_list:
        uv.hole_faces   = mesh_data.hole_faces.copy()
        uv.hole_counts  = mesh_data.hole_counts.copy()

        uv_hole_indices = []
        offset          = 0
        for i in range(len(mesh_data.hole_faces)):
            face_idx = int(mesh_data.hole_faces[i])
            count    = int(mesh_data.hole_counts[i])
            mesh_hole_verts = {
                int(v) for v in mesh_data.hole_indices[offset : offset + count]
            }

            start      = int(face_starts[face_idx])
            face_count = int(mesh_data.counts[face_idx])
            for j in range(face_count):
                if int(mesh_data.indices[start + j]) in mesh_hole_verts:
                    uv_hole_indices.append(int(uv.indices[start + j]))

            offset += count

        uv.hole_indices = np.array(uv_hole_indices, dtype=int)


# --- rig's undoable mesh edits ------------------------------------------------------------
#
# Each class below is one edit rig puts on Maya's undo queue through its plug-in
# command (`_run_undoable`, one undo step): the caller checks the edit first, the
# constructor keeps what the edit needs and runs it, doIt makes the change (and keeps
# what undoIt needs), undoIt puts the mesh back and redoIt makes the change again.
#
# * API only: no cmds or mel in doIt, undoIt or redoIt. redo must replay the API
#   change (a cmds call there would be recorded, and flush the redo queue), and a
#   cmds call in doIt would be an undo step of its own.
# * A new MFnMesh for every call, from a copy of the mesh's MDagPath: an MFnMesh
#   kept across an edit of the mesh's sets made some other way (a polyUVSet /
#   polyColorSet, or the undo / redo of one) can point at freed geometry and crash
#   Maya when it is used again.
# * UV and colour SETS are made, renamed and deleted here too, never with
#   polyUVSet / polyColorSet: Maya undoes those by swapping the mesh's set data,
#   which, beside rig's API data edits and a later vertex edit, brings back a
#   broken set (an API write into it crashes Maya) or wrong data on redo.
# * The user's own polyUVSet / polyColorSet can still leave such a set (Maya does
#   it without rig): every set edit first puts the mesh's sets into its cached
#   input (`_sync_sets`).
# * A set keeps its element of the mesh's uvSet[] / colorSet[] array, and the
#   element its connections (a uvLink): an undone set creation frees the element it
#   made, a delete's undo makes the sets again in their elements and reconnects
#   them (`_free_elements`, `_relink`). Maya's own set commands undo by element.
# * A mesh with history (its inMesh connected: construction history or a deformer)
#   gets a node for each API set or colour edit: the edit passes an MDGModifier,
#   and its undo / redo are the modifier's.


def _has_history(fn: OpenMaya.MFnMesh) -> bool:
    """True when the mesh's ``inMesh`` is connected (construction history, a
    deformer)."""
    return fn.findPlug("inMesh", False).isDestination


def _put_points(fn: OpenMaya.MFnMesh, points: OpenMaya.MPointArray, space: int) -> None:
    fn.setPoints(points, space)
    fn.updateSurface()


def _modified(modifier: OpenMaya.MDGModifier | None) -> dict:
    """The ``modifier`` keyword of an MFnMesh call: none without one (the API
    takes no None)."""
    return {} if modifier is None else {"modifier": modifier}


def _sync_sets(fn: OpenMaya.MFnMesh) -> bool:
    """Puts the UV and colour sets of the history-free mesh `fn` into its cached
    input mesh when the two differ; True when it did (`fn` is then stale).

    A history-free mesh keeps a copy of its input (``cachedInMesh``) from its
    first vertex tweak on, and API set edits go through that copy. The undo /
    redo of Maya's polyUVSet / polyColorSet swaps the mesh's sets but not the
    copy's: after a native set edit, a vertex edit and the undo of both, the
    mesh has a set its copy has not, and any API write into that set crashes
    Maya (Maya's own polyUVSet -delete fails there). The copy takes the mesh's
    data with its own points (the tweaks stay added to them)."""
    if fn.findPlug("inMesh", False).isDestination:
        return False
    plug = fn.findPlug("cachedInMesh", False)
    try:
        cached = OpenMaya.MFnMesh(plug.asMObject())
    except RuntimeError:  # no copy (never tweaked)
        return False
    if (
        cached.getUVSetNames() == fn.getUVSetNames()
        and cached.getColorSetNames() == fn.getColorSetNames()
    ):
        return False
    if cached.numVertices != fn.numVertices:
        raise RuntimeError(
            f"{fn.fullPathName()}: Maya left the mesh out of step with its cached input "
            "(the undo of a native mesh edit after a vertex edit); flush the undo queue "
            "(cmds.flushUndo()) before editing its sets"
        )
    data = OpenMaya.MFnMeshData().create()
    OpenMaya.MFnMesh().copy(fn.findPlug("outMesh", False).asMObject(), data)
    OpenMaya.MFnMesh(data).setPoints(cached.getPoints())
    plug.setMObject(data)
    return True


def _set_element(fn: OpenMaya.MFnMesh, array: str, name: str) -> OpenMaya.MPlug | None:
    """The element of the mesh's ``uvSet`` / ``colorSet`` `array` that holds the
    set `name` (its first child is the set's name), or None."""
    plug = fn.findPlug(array, False)
    for i in range(plug.numElements()):
        element = plug.elementByPhysicalIndex(i)
        if element.child(0).asString() == name:
            return element
    return None


def _links(fn: OpenMaya.MFnMesh, array: str, name: str) -> tuple:
    """What `_relink` makes again for the set `name`: ``(element index,
    [(child, [destination plugs])])``."""
    element  = _set_element(fn, array, name)
    children = [(c, list(element.child(c).destinations())) for c in range(element.numChildren())]
    return element.logicalIndex(), children


def _free_elements(fn: OpenMaya.MFnMesh, array: str, indices) -> None:
    """Removes the EMPTY elements `indices` of `array` (an API delete empties
    the set's element and keeps it), so that the API's next sets take them
    back (it takes the first free element)."""
    plug     = fn.findPlug(array, False)
    modifier = OpenMaya.MDGModifier()
    for index in indices:
        element = plug.elementByLogicalIndex(index)
        if not element.child(0).asString():
            modifier.removeMultiInstance(element, True)
    modifier.doIt()


def _relink(fn: OpenMaya.MFnMesh, array: str, name: str, links: tuple) -> None:
    """Connects the element of the set `name` to the destinations `links` holds
    (`_links`) that nothing drives now."""
    element  = _set_element(fn, array, name)
    modifier = OpenMaya.MDGModifier()
    for child, destinations in links[1]:
        for destination in destinations:
            if not destination.isDestination:
                modifier.connect(element.child(child), destination)
    modifier.doIt()


class _MeshEdit:
    """[Internal] An undoable edit of one mesh: a copy of its MDagPath, and
    `fn`, a new MFnMesh of it on every call, its sets synced (`_sync_sets`)."""

    def __init__(self, mesh: Mesh) -> None:
        self._path = OpenMaya.MDagPath(mesh.mdagpath)

    def fn(self) -> OpenMaya.MFnMesh:
        fn = OpenMaya.MFnMesh(self._path)
        return OpenMaya.MFnMesh(self._path) if _sync_sets(fn) else fn


class _MeshSetPointsCommand(_MeshEdit):
    """`Mesh.set_points`: every point of the mesh, in one space. It edits no
    set, so it skips `_sync_sets` (setPoints keeps the sets either way)."""

    def __init__(self, mesh: Mesh, points: OpenMaya.MPointArray, space: int) -> None:
        super().__init__(mesh)
        self._points = points
        self._space  = space
        self._old    = None
        _run_undoable(self)

    def doIt(self) -> None:
        fn        = OpenMaya.MFnMesh(self._path)
        self._old = fn.getPoints(self._space)
        _put_points(fn, self._points, self._space)

    def undoIt(self) -> None:
        _put_points(OpenMaya.MFnMesh(self._path), self._old, self._space)

    def redoIt(self) -> None:
        _put_points(OpenMaya.MFnMesh(self._path), self._points, self._space)


class _MeshSetsEdit(_MeshEdit):
    """[Internal] An edit of the mesh's UV or colour sets. `_capture` keeps
    what the undo needs, then `_apply` makes the edit. Without history, redo
    is `_apply` again and undo is `_revert`. With history (``_modifier`` is
    set before `_capture`), `_apply` gets an MDGModifier, which records the
    nodes the API adds, and undo / redo are the modifier's. The subclass sets
    its fields, then calls this constructor, which runs it."""

    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh)
        self._modifier = None
        _run_undoable(self)

    def doIt(self) -> None:
        fn = self.fn()
        if _has_history(fn):
            self._modifier = OpenMaya.MDGModifier()
        self._capture(fn)
        self._apply(fn, self._modifier)

    def undoIt(self) -> None:
        fn = self.fn()  # first: a mesh deleted since raises here, not in the modifier
        if self._modifier is None:
            self._revert(fn)
        else:
            self._modifier.undoIt()

    def redoIt(self) -> None:
        fn = self.fn()
        if self._modifier is None:
            self._apply(fn, None)
        else:
            self._modifier.doIt()

    def _capture(self, fn: OpenMaya.MFnMesh) -> None:
        pass

    def _apply(self, fn: OpenMaya.MFnMesh, modifier: OpenMaya.MDGModifier | None) -> None:
        raise NotImplementedError

    def _revert(self, fn: OpenMaya.MFnMesh) -> None:
        raise NotImplementedError


# --- UV sets


def _uv_lists(uv_data: UVData) -> tuple[list, list, list, list]:
    """`uv_data`'s UVs as the lists the API takes: ``(u, v, counts, ids)``."""
    points = np.asarray(uv_data.points, dtype=float).reshape(-1, 2)
    return (
        points[:, 0].tolist(),
        points[:, 1].tolist(),
        np.asarray(uv_data.counts, dtype=np.int64).ravel().tolist(),
        np.asarray(uv_data.indices, dtype=np.int64).ravel().tolist(),
    )


def _uvs_of(fn: OpenMaya.MFnMesh, uv_set: str) -> tuple:
    """The UVs of the set `uv_set`: ``(u, v, counts, ids)``."""
    return (*fn.getUVs(uv_set), *fn.getAssignedUVs(uv_set))


def _put_uvs(fn: OpenMaya.MFnMesh, uv_set: str, uvs: tuple, history: bool) -> None:
    """Replaces the UVs of the set `uv_set` with `uvs` (``(u, v, counts, ids)``).

    Fewer UVs than the set has: without history the set is cleared first
    (``setUVs`` cannot shrink it); with history it is not (Maya's ``clearUVs``
    there adds a ``polyMapDel`` node that clears the CURRENT set), so the extra
    UVs stay in the set, unassigned (`Mesh.set_uv_data` refuses to write fewer
    UVs there)."""
    u, v, counts, ids = uvs
    now = fn.numUVs(uv_set)
    if len(u) < now:
        if history:
            old_u, old_v = fn.getUVs(uv_set)
            u = list(u) + list(old_u)[len(u):]
            v = list(v) + list(old_v)[len(v):]
        else:
            fn.clearUVs(uv_set)
    if len(u):
        fn.setUVs(u, v, uv_set)
        fn.assignUVs(counts, ids, uv_set)


class _MeshAddUVSetCommand(_MeshSetsEdit):
    """`Mesh.add_uv_set`: an empty UV set. Undo frees its element, so a redo
    makes it in the same one."""

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name = name
        super().__init__(mesh)

    def _apply(self, fn, modifier):
        fn.createUVSet(self._name, **_modified(modifier))

    def _revert(self, fn):
        index = _set_element(fn, "uvSet", self._name).logicalIndex()
        fn.deleteUVSet(self._name)
        _free_elements(fn, "uvSet", [index])


class _MeshRenameUVSetCommand(_MeshEdit):
    """`Mesh.rename_uv_set`. A rename adds no node, with or without history,
    so it takes no modifier (whose undo would re-evaluate the history and drop
    the UVs written into the set through the API)."""

    def __init__(self, mesh: Mesh, name: str, new_name: str) -> None:
        super().__init__(mesh)
        self._name     = name
        self._new_name = new_name
        _run_undoable(self)

    def doIt(self) -> None:
        self.fn().renameUVSet(self._name, self._new_name)

    def undoIt(self) -> None:
        self.fn().renameUVSet(self._new_name, self._name)

    def redoIt(self) -> None:
        self.doIt()


class _MeshDeleteUVSetCommand(_MeshSetsEdit):
    """`Mesh.delete_uv_set`. Undo makes the set again with its UVs, in its
    place (the API adds a set last, so the sets after it are made again after
    it), each in its uvSet[] element with its connections, and the current set
    is current again. With history the modifier's undo brings the set back from
    the history, without the UVs written into it through the API: they are
    written again."""

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name    = name
        self._uvs     = None
        self._links   = None
        self._later   = ()
        self._current = ""
        super().__init__(mesh)

    def _capture(self, fn):
        names         = list(fn.getUVSetNames())
        self._uvs     = _uvs_of(fn, self._name)
        self._links   = _links(fn, "uvSet", self._name)
        self._later   = names[names.index(self._name) + 1 :]
        self._current = fn.currentUVSetName()

    def _apply(self, fn, modifier):
        fn.deleteUVSet(self._name, **_modified(modifier))

    def undoIt(self) -> None:
        super().undoIt()
        if self._modifier is not None:
            _put_uvs(self.fn(), self._name, self._uvs, True)

    def _revert(self, fn):
        later = [(name, _uvs_of(fn, name), _links(fn, "uvSet", name)) for name in self._later]
        for name, _, _ in later:
            fn.deleteUVSet(name)
        sets = [(self._name, self._uvs, self._links), *later]
        _free_elements(fn, "uvSet", [links[0] for _, _, links in sets])
        for name, uvs, links in sets:
            fn.createUVSet(name)
            _put_uvs(fn, name, uvs, False)
            _relink(fn, "uvSet", name, links)
        if fn.currentUVSetName() != self._current:
            fn.setCurrentUVSetName(self._current)


class _MeshSetUVDataCommand(_MeshEdit):
    """`Mesh.set_uv_data`: the UVs of one set, and its rename to the data's
    name. The UVs are data, written through the API with or without history
    (see `_put_uvs`)."""

    def __init__(self, mesh: Mesh, uv_set: str, uv_data: UVData) -> None:
        super().__init__(mesh)
        self._uv_set  = uv_set
        self._name    = uv_data.name
        self._new     = _uv_lists(uv_data)
        self._old     = None
        self._history = False
        _run_undoable(self)

    def doIt(self) -> None:
        fn            = self.fn()
        self._history = _has_history(fn)
        self._old     = _uvs_of(fn, self._uv_set)
        try:
            _put_uvs(fn, self._uv_set, self._new, self._history)
        except BaseException:
            _put_uvs(fn, self._uv_set, self._old, self._history)  # nothing is queued
            raise
        self._rename(fn, self._uv_set, self._name)

    def undoIt(self) -> None:
        fn = self.fn()
        self._rename(fn, self._name, self._uv_set)
        _put_uvs(fn, self._uv_set, self._old, self._history)

    def redoIt(self) -> None:
        fn = self.fn()
        _put_uvs(fn, self._uv_set, self._new, self._history)
        self._rename(fn, self._uv_set, self._name)

    @staticmethod
    def _rename(fn: OpenMaya.MFnMesh, name: str, new_name: str) -> None:
        if name != new_name:
            fn.renameUVSet(name, new_name)


def _check_uv_data(fn: OpenMaya.MFnMesh, uv_set: str, uv_data: UVData) -> None:
    """Raises, before any edit, when `Mesh.set_uv_data` cannot write `uv_data`
    into the set `uv_set` of the mesh `fn` and rename it ``uv_data.name``."""
    names = fn.getUVSetNames()
    if uv_set not in names:
        raise RuntimeError(f"UV set {uv_set} does not exist.")
    name = uv_data.name
    if not isinstance(name, str) or not name:
        raise TypeError(f"UVData.name must be a non-empty str, not {name!r}")
    if name != uv_set and name in names:
        raise RuntimeError(f"UV set {name} already exists.")
    n_uvs   = len(np.asarray(uv_data.points, dtype=float).reshape(-1, 2))
    counts  = np.asarray(uv_data.counts, dtype=np.int64).ravel()
    indices = np.asarray(uv_data.indices, dtype=np.int64).ravel()
    if n_uvs < fn.numUVs(uv_set) and _has_history(fn):
        raise RuntimeError(
            f"UVData {name!r}: {n_uvs} UVs for the {fn.numUVs(uv_set)} of UV set {uv_set}: "
            "rig cannot remove UVs from a mesh with history (Maya's clearUVs there clears "
            "the current UV set); delete its history first"
        )
    if not n_uvs and not len(indices):
        return  # no UVs: the set is emptied
    if len(counts) != fn.numPolygons:
        raise RuntimeError(f"UVData {name!r}: {len(counts)} face counts for {fn.numPolygons} faces")
    if int(counts.sum()) != len(indices):
        raise RuntimeError(
            f"UVData {name!r}: the face counts add up to {int(counts.sum())}, "
            f"but there are {len(indices)} indices"
        )
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= n_uvs):
        raise RuntimeError(
            f"UVData {name!r}: an index is not one of the {n_uvs} UVs "
            f"(indices range {int(indices.min())}..{int(indices.max())})"
        )


# --- colour sets


def _color_ids(fn: OpenMaya.MFnMesh, color_set: str) -> list[int]:
    """The colour index of every face-vertex of `color_set`, face by face (-1:
    no colour)."""
    ids = []
    it  = OpenMaya.MItMeshPolygon(fn.object())
    while not it.isDone():
        try:
            ids.extend(it.getColorIndices(color_set))
        except RuntimeError:  # a face without colours
            ids.extend([-1] * it.polygonVertexCount())
        it.next()
    return ids


def _capture_colors(fn: OpenMaya.MFnMesh, color_set: str) -> tuple:
    """The colours of `color_set` as `_put_colors` puts them back: ``(pool,
    ids)``; nothing to read for a set without colours."""
    if not fn.numColors(color_set):
        return (), ()
    return fn.getColors(color_set), _color_ids(fn, color_set)


def _put_colors(fn: OpenMaya.MFnMesh, color_set: str, colors: tuple, rep: int) -> None:
    """Replaces the colours of `color_set` with `colors` (`_capture_colors`):
    ``clearColors``, then the pool and the per-face-vertex indices."""
    pool, ids = colors
    fn.clearColors(color_set)
    if len(pool):
        fn.setColors(pool, color_set, rep=rep)
        fn.assignColors(ids, color_set)


def _write_vertex_colors(
    fn:        OpenMaya.MFnMesh,
    color_set: str,
    colors:    OpenMaya.MColorArray,
    rep:       int,
    modifier:  OpenMaya.MDGModifier | None,
) -> None:
    """``setVertexColors`` of every vertex into `color_set`, made the current
    set for the write (the current set is put back after it)."""
    current = fn.currentColorSetName()
    if current != color_set:
        fn.setCurrentColorSetName(color_set)
    try:
        fn.setVertexColors(colors, range(fn.numVertices), rep=rep, **_modified(modifier))
    finally:
        if current and current != color_set:
            fn.setCurrentColorSetName(current)


class _MeshAddColorSetCommand(_MeshSetsEdit):
    """`Mesh.add_color_set`: an empty, unclamped colour set. Undo frees its
    element, so a redo makes it in the same one."""

    def __init__(self, mesh: Mesh, name: str, rep: int) -> None:
        self._name = name
        self._rep  = rep
        super().__init__(mesh)

    def _apply(self, fn, modifier):
        fn.createColorSet(self._name, False, rep=self._rep, **_modified(modifier))

    def _revert(self, fn):
        index = _set_element(fn, "colorSet", self._name).logicalIndex()
        fn.deleteColorSet(self._name)
        _free_elements(fn, "colorSet", [index])


class _ColorSetDataCommand(_MeshSetsEdit):
    """`ColorSet.data`: one colour per vertex. Undo puts the set's colours back
    face-vertex by face-vertex; the current colour set stays current."""

    def __init__(self, mesh: Mesh, name: str, rep: int, colors: OpenMaya.MColorArray) -> None:
        self._name    = name
        self._rep     = rep
        self._colors  = colors
        self._old     = None
        self._current = ""
        super().__init__(mesh)

    def undoIt(self) -> None:
        super().undoIt()
        self._keep_current()

    def redoIt(self) -> None:
        super().redoIt()
        self._keep_current()

    def _keep_current(self) -> None:
        # with history, the modifier's undo / redo re-evaluates the mesh
        fn = self.fn()
        if self._current and fn.currentColorSetName() != self._current:
            fn.setCurrentColorSetName(self._current)

    def _capture(self, fn):
        self._current = fn.currentColorSetName()
        if self._modifier is None:
            self._old = _capture_colors(fn, self._name)

    def _apply(self, fn, modifier):
        _write_vertex_colors(fn, self._name, self._colors, self._rep, modifier)

    def _revert(self, fn):
        _put_colors(fn, self._name, self._old, self._rep)


def _color_set_of(fn: OpenMaya.MFnMesh, color_set: str) -> tuple:
    """What `_make_color_set` makes `color_set` again from: ``(representation,
    clamped, colours)``."""
    return (
        fn.getColorRepresentation(color_set),
        fn.isColorClamped(color_set),
        _capture_colors(fn, color_set),
    )


def _make_color_set(fn: OpenMaya.MFnMesh, color_set: str, saved: tuple) -> None:
    """Makes the colour set `color_set` again from `saved` (`_color_set_of`)."""
    rep, clamped, colors = saved
    fn.createColorSet(color_set, clamped, rep=rep)
    _put_colors(fn, color_set, colors, rep)


class _ColorSetDeleteCommand(_MeshSetsEdit):
    """`ColorSet.delete`. Undo makes the set again with its representation,
    clamping and colours, in its place (the API adds a set last, so the sets
    after it are made again after it), each in its colorSet[] element with its
    connections, and the current set is current again."""

    def __init__(self, mesh: Mesh, name: str) -> None:
        self._name    = name
        self._saved   = None
        self._links   = None
        self._later   = ()
        self._current = ""
        super().__init__(mesh)

    def _capture(self, fn):
        if self._modifier is None:
            names         = list(fn.getColorSetNames())
            self._saved   = _color_set_of(fn, self._name)
            self._links   = _links(fn, "colorSet", self._name)
            self._later   = names[names.index(self._name) + 1 :]
            self._current = fn.currentColorSetName()

    def _apply(self, fn, modifier):
        fn.deleteColorSet(self._name, **_modified(modifier))

    def _revert(self, fn):
        later = [(name, _color_set_of(fn, name), _links(fn, "colorSet", name)) for name in self._later]
        for name, _, _ in later:
            fn.deleteColorSet(name)
        sets = [(self._name, self._saved, self._links), *later]
        _free_elements(fn, "colorSet", [links[0] for _, _, links in sets])
        for name, saved, links in sets:
            _make_color_set(fn, name, saved)
            _relink(fn, "colorSet", name, links)
        if self._current and fn.currentColorSetName() != self._current:
            fn.setCurrentColorSetName(self._current)


class ColorSet:
    """Represents a color set of a mesh"""

    class Representation(enum.Enum):
        """
        Enum for color set representations
        """

        Alpha = OpenMaya.MFnMesh.kAlpha
        RGB   = OpenMaya.MFnMesh.kRGB
        RGBA  = OpenMaya.MFnMesh.kRGBA

    def __init__(self, mesh: Mesh, name: str):
        self._mesh = mesh
        self._name = name

    def __repr__(self):
        return f"ColorSet<{self.mesh.name}.{self.name}>)"

    @property
    def name(self) -> str:
        """The name of the color set"""
        return self._name

    @property
    def mesh(self) -> Mesh:
        """The mesh this color set belongs too"""
        return self._mesh

    @property
    def is_valid(self) -> bool:
        """Does the underlying colorset still exist?"""
        return self.name in self.mesh.fn_set.getColorSetNames()

    @property
    def representation(self) -> ColorSet.Representation:
        """The representation of the color set (ie the data type Alpha|RGB|RGBA)"""
        if not self.is_valid:
            raise RuntimeError(
                f"{self.name} is not a valid color set for {self.mesh.name}!"
            )
        return self.Representation(self.mesh.fn_set.getColorRepresentation(self.name))

    @property
    def data(self) -> np.array:
        """the data of the color set expressed as a numpy array."""
        with self.as_current_color_set():
            return np.array(self.mesh.fn_set.getVertexColors())

    @data.setter
    def data(self, data):
        """set the colorset data: one colour per vertex, a ``(V, 4)`` array. One
        undo step, whose undo puts back the set's colours as they were,
        face-vertex by face-vertex. The current colour set is unchanged."""
        mesh = self.mesh
        if data.shape[0] != mesh.num_vertices:
            raise RuntimeError(f"shape of data is not ({mesh.num_vertices}, 4): !")
        rep = self.representation.value  # raises for a colour set that is gone
        _ColorSetDataCommand(mesh, self.name, rep, OpenMaya.MColorArray(data))

    @contextlib.contextmanager
    def as_current_color_set(self) -> Generator[None, None, None]:
        """Context manager that sets the current color set to this one"""
        if not self.is_valid:
            raise RuntimeError(
                f"{self.name} is not a valid color set for {self.mesh.name}!"
            )

        try:
            current_colorset = self.mesh.fn_set.currentColorSetName()
            self.mesh.fn_set.setCurrentColorSetName(self.name)
            yield
        finally:
            if current_colorset:
                self.mesh.fn_set.setCurrentColorSetName(current_colorset)

    def delete(self):
        """Delete this color set (nothing happens when it no longer exists): one
        undo step, whose undo brings the set back with its representation,
        clamping and colours (current again if it was)."""
        if self.is_valid:
            _ColorSetDeleteCommand(self.mesh, self.name)