"""
Mesh node class
"""

from __future__ import annotations

import contextlib
import enum
import re
from numbers import Number
from typing import Generator, List

import numpy as np
from maya import cmds, mel
from maya.api import OpenMaya
from rig.maya.nodetypes._base import Attribute
from rig.maya.nodetypes.dag_node import DAGNode, PyNode
from rig.maya.nodetypes.geometry import Geometry
from rig.maya.nodetypes.object_set import ObjectSet
from rig.maya.nodetypes.shading_engine import ShadingEngine
from rig.maya.plugins import load_plugin
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

    # --- creation

    @classmethod
    def _create(cls, mesh_data: MeshData, name: str | None = None, **kwargs) -> str:
        """[Internal] Creates a mesh from a mesh data object and returns the mesh name.

        Args:
            mesh_data: A MeshData object.
        """
        from cgmath.geometry import MeshData

        cmd = _MeshCreateCommand(mesh_data, name=name)
        with load_plugin("undoable_api_command"):
            return cmds.runUndoableAPICommand(cmd)[0]

    @classmethod
    def create(
        cls,
        mesh_data: MeshData,
        uv_data:   UVData   | UVList | list[UVData] | None = None,
        name:      str      | None                         = None,
    ) -> "Mesh":
        """Creates a mesh object from a mesh data object.

        Args:
            mesh_data: A MeshData object.
            uv_data: One or more UVData objects.
            name: The name of the mesh to create
        """
        from cgmath.geometry import MeshData, UVData, UVList

        mesh = super().create(mesh_data, name=name)
        if uv_data:
            if isinstance(uv_data, UVData):
                uv_data = [uv_data]
            for i, each in enumerate(uv_data):
                # reuse the default uv set
                if i == 0:
                    mesh.set_uv_data(each, i)
                else:
                    mesh.add_uv_set(each.name)
                    mesh.set_uv_data(each, each.name)
        return mesh

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
        """Set the vertex points of the mesh."""
        from cgmath.geometry import MeshData

        # if given MeshData
        if isinstance(points, MeshData):
            points = OpenMaya.MPointArray(points.points)

        # else assume we're given a point sequence (list, or numpy array)
        elif not isinstance(points, OpenMaya.MPointArray):
            points = OpenMaya.MPointArray(points)

        _MeshSetPointsCommand(self, points, world_space)

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
        for face_idx in range(self.num_polygons):
            if all(
                (
                    is_vert_above_plane[i]
                    for i in self.fn_set.getPolygonVertices(face_idx)
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
            mat = PyNode(mat[0])
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
        """Adds an emptyUV set.

        Args:
            uv_set: A uv set name to add.
        """
        if uv_set in self.uv_sets:
            raise RuntimeError(f"UV set {uv_set} already exists.")
        _MeshAddUVSetCommand(self, uv_set)

    def delete_uv_set(self, uv_set: str | None = None) -> None:
        """Deletes a given uv set.

        Args:
            uv_set: A uv set name to delete. If None, use current uv set.
        """
        uv_set = uv_set or self.current_uv_set
        cmds.polyUVSet(self.name, delete=True, uvSet=uv_set)

    def delete_map(self, map_name: str | None = None) -> None:
        """Deletes a given map.

        Args:
            map_name: A map name to delete. If None, use current map.
        """
        map_name = f"{self.name}.{map_name}"
        if cmds.objExists(map_name):
            cmds.deleteAttr(map_name)

    def rename_uv_set(self, new_name: str, uv_set: str | None = None) -> None:
        """Renames the current uv set.

        Args:
            new_name: New uv set name.
            uv_set: A uv set name to rename. If None, use current uv set.
        """
        uv_set = uv_set or self.current_uv_set
        if uv_set != new_name:
            cmds.polyUVSet(self.name, rename=True, uvSet=uv_set, newUVSet=new_name)

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

        Args:
            uv_data: A UVData object.
            uv_set: A uv set name to operate on. If None, use current uv set.
        """
        from cgmath.geometry import UVData

        uv_set = uv_set or self.current_uv_set
        _MeshSetUVDataCommand(self, uv_set, uv_data)

    # --- deformation

    def apply_skin_data(self, skin_data: SkinData):
        """Apply skin data to this mesh."""
        from cgmath.geometry import SkinData

        # avoid circular import
        from rig.maya.nodetypes import SkinCluster

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

        points = self.get_points(world_space=world_space)
        vert_counts, vert_ids = self.fn_set.getVertices()

        space      = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        normals    = self.fn_set.getNormals(space)
        normal_ids = []
        for face_id in range(self.fn_set.numPolygons):
            normal_ids.extend(self.fn_set.getFaceNormalIds(face_id))

        mesh_data = MeshData(
            indices        = np.asarray(vert_ids),
            counts         = np.asarray(vert_counts),
            points         = np.asarray(points)[:, :-1],
            normals        = np.asarray(normals),
            normal_indices = np.asarray(normal_ids),
            name           = self.clean_name,
        )

        # extract hole data from faces with interior boundaries
        holes = self.fn_set.getHoles()
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
            mesh_data.matrix = np.array(self.get_parent().wm.get()).reshape(4, 4)

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
        """Create a color set with the given name and representation"""
        # cast or throw ValueError
        representation = ColorSet.Representation(representation)

        if name in self.fn_set.getColorSetNames():
            raise ValueError(f"{self.name} has color set {name}!")
        created_name = self.fn_set.createColorSet(name, False, rep=representation.value)
        assert created_name == name
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


def _delete_internal_edges(xform: str, internal_edges: set[tuple[int, int]]) -> None:
    """Delete triangulation edges from a mesh to restore N-gon + hole topology."""
    shape = cmds.listRelatives(xform, shapes=True, fullPath=True)[0]
    sel   = OpenMaya.MSelectionList()
    sel.add(shape)
    fn = OpenMaya.MFnMesh(sel.getDagPath(0))

    edge_ids = []
    for eid in range(fn.numEdges):
        v0, v1 = fn.getEdgeVertices(eid)
        if (min(v0, v1), max(v0, v1)) in internal_edges:
            edge_ids.append(eid)

    if edge_ids:
        edges = [f"{shape}.e[{eid}]" for eid in edge_ids]
        cmds.polyDelEdge(edges, cleanVertices=False)


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


class _MeshCreateCommand:
    def __init__(self, mesh_data: MeshData, name: str | None = None) -> None:
        from cgmath.geometry import MeshData

        super().__init__()
        self._mesh_data  = mesh_data
        self._name       = name
        self._mesh_xform = None

    def doIt(self) -> None:
        mesh_data = self._mesh_data
        points    = mesh_data.points

        has_holes = mesh_data.hole_faces is not None and len(mesh_data.hole_faces) > 0

        if not has_holes:
            mobject = Mesh.FN_SET().create(
                OpenMaya.MPointArray(points),
                mesh_data.counts,
                mesh_data.indices,
            )
            self._mesh_xform = OpenMaya.MFnDagNode(mobject).partialPathName()
        else:
            self._mesh_xform = self._create_with_holes()

        cmds.sets(self._mesh_xform, forceElement="initialShadingGroup")

        # if a name is not specified, attempt to use the one stored in mesh_data
        self._name = self._name or mesh_data.name

        if self._name:
            # if name ends with "Shape#", remove it
            if bool(re.search("Shape[0-9]*$", self._name)):
                self._name = "".join(self._name.rpartition("Shape")[::2])
            # apply name to the transform and let maya rename the shape
            self._mesh_xform = cmds.rename(self._mesh_xform, self._name)

        # transform the parent by the stored matrix
        cmds.xform(self._mesh_xform, matrix=mesh_data.matrix.ravel())

        return self._mesh_xform

    def _create_with_holes(self) -> str:
        """Creates a mesh with hole-aware face creation.

        CDT-triangulates holed faces, creates the entire mesh via a single
        MFnMesh.create() call, then deletes the internal triangulation
        edges so Maya reconstructs the original N-gon + hole topology.
        """
        mesh_data = self._mesh_data
        points    = mesh_data.points
        counts    = mesh_data.counts
        indices   = mesh_data.indices
        hole_map  = _build_hole_map(mesh_data)

        new_counts     = []
        new_indices    = []
        internal_edges = set()

        idx_offset = 0
        for fi in range(len(counts)):
            c          = int(counts[fi])
            face_verts = indices[idx_offset : idx_offset + c]

            if fi not in hole_map:
                new_counts.append(c)
                new_indices.extend(int(v) for v in face_verts)
            else:
                _triangulate_holed_face(
                    points,
                    face_verts,
                    hole_map[fi],
                    new_counts,
                    new_indices,
                    internal_edges,
                )

            idx_offset += c

        # single MFnMesh.create() call for the entire mesh
        mobject = Mesh.FN_SET().create(
            OpenMaya.MPointArray(points),
            new_counts,
            new_indices,
        )
        xform = OpenMaya.MFnDagNode(mobject).partialPathName()

        # delete internal triangulation edges to restore hole topology
        if internal_edges:
            _delete_internal_edges(xform, internal_edges)

        return xform

    def undoIt(self) -> None:
        if self._mesh_xform and cmds.objExists(self._mesh_xform):
            cmds.delete(self._mesh_xform)

    def redoIt(self) -> None:
        self.doIt()


class _MeshSetPointsCommand:
    def __init__(
        self, mesh: Mesh, points: OpenMaya.MPointArray, world_space: bool
    ) -> None:
        super().__init__()
        self._fn_set     = mesh.fn_set
        self._points     = points
        self._space      = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        self._old_points = None

        with load_plugin("undoable_api_command"):
            cmds.runUndoableAPICommand(self)

    def doIt(self) -> None:
        self._old_points = self._fn_set.getPoints(self._space)
        self.redoIt()

    def undoIt(self) -> None:
        self._fn_set.setPoints(self._old_points, self._space)
        self._fn_set.updateSurface()

    def redoIt(self) -> None:
        self._fn_set.setPoints(self._points, self._space)
        self._fn_set.updateSurface()


class _MeshAddUVSetCommand:
    def __init__(self, mesh: Mesh, uv_set: str) -> None:
        super().__init__()
        self._fn_set = mesh.fn_set
        self._uv_set = uv_set

        with load_plugin("undoable_api_command"):
            cmds.runUndoableAPICommand(self)

    def doIt(self) -> None:
        self._fn_set.createUVSet(self._uv_set)

    def undoIt(self) -> None:
        self._fn_set.deleteUVSet(self._uv_set)

    def redoIt(self) -> None:
        self.doIt()


class _MeshSetUVDataCommand:
    def __init__(self, mesh: Mesh, uv_set: str, uv_data: UVData) -> None:
        from cgmath.geometry import UVData

        super().__init__()
        self._fn_set  = mesh.fn_set
        self._uv_set  = uv_set
        self._u_vals  = uv_data.points[:, 0]
        self._v_vals  = uv_data.points[:, 1]
        self._ids     = uv_data.indices
        self._counts  = uv_data.counts
        self._uv_name = uv_data.name

        self._old_u_vals = None
        self._old_v_vals = None
        self._old_ids    = None
        self._old_counts = None

        with load_plugin("undoable_api_command"):
            cmds.runUndoableAPICommand(self)

    def doIt(self) -> None:
        self._old_u_vals, self._old_v_vals = self._fn_set.getUVs(self._uv_set)
        self._old_counts, self._old_ids = self._fn_set.getAssignedUVs(self._uv_set)
        self.redoIt()

    def undoIt(self) -> None:
        self._fn_set.setUVs(self._old_u_vals, self._old_v_vals, self._uv_name)
        self._fn_set.assignUVs(self._old_counts, self._old_ids, self._uv_name)
        if self._uv_name != self._uv_set:
            self._fn_set.renameUVSet(self._uv_name, self._uv_set)

    def redoIt(self) -> None:
        self._fn_set.setUVs(self._u_vals, self._v_vals, self._uv_set)
        self._fn_set.assignUVs(self._counts, self._ids, self._uv_set)
        if self._uv_name != self._uv_set:
            self._fn_set.renameUVSet(self._uv_set, self._uv_name)


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
        """set the colorset data"""
        if data.shape[0] != self.mesh.num_vertices:
            raise RuntimeError(f"shape of data is not ({self.mesh.num_vertices}, 4): !")
        with self.as_current_color_set():
            self.mesh.fn_set.setVertexColors(
                data, range(self.mesh.num_vertices), rep=self.representation.value
            )

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
        """Delete this color set"""
        if self.is_valid:
            self.mesh.fn_set.deleteColorSet(self.name)