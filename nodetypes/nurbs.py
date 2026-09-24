"""
Nurbs node class for NurbsCurve and NurbsSurface
"""

from __future__ import annotations

import numpy as np
from maya.api import OpenMaya
from rig.nodetypes.geometry import Geometry


class NurbsCurve(Geometry):
    """
    nurbsCurve node class
    """

    NATIVE_NODE_TYPE = "nurbsCurve"
    FN_SET           = OpenMaya.MFnNurbsCurve
    POINT_COMP_TYPE  = "cv"

    @property
    def num_cvs(self) -> int:
        """Returns the number of CVs."""
        return self.fn_set.numCVs

    @property
    def num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count = self.fn_set.numCVs
        if self.fn_set.form == OpenMaya.MFnNurbsCurve.kPeriodic:
            cv_count -= self.fn_set.degree
        return cv_count

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsCurve.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.cvPositions(space)

    # --- serialization

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> BSplineData:
        """Serialize this nurbsCurve to BSplineData.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            BSplineData object.
        """
        from cgmath.geometry import BSplineData

        points = self.get_points(world_space=world_space)
        degree = self.fn_set.degree
        closed = self.fn_set.form != OpenMaya.MFnNurbsCurve.kOpen

        points = np.array(points)[:, :3]

        return BSplineData(points=points, degree=degree, periodic=closed)


class NurbsSurface(Geometry):
    """
    nurbsSurface node class
    """

    NATIVE_NODE_TYPE = "nurbsSurface"
    FN_SET           = OpenMaya.MFnNurbsSurface
    POINT_COMP_TYPE  = "cv"

    @property
    def num_cvs(self) -> int:
        """Returns the number of cvs."""
        return self.fn_set.numCVsInU * self.fn_set.numCVsInV

    @property
    def num_weight_points(self) -> int:
        """Returns the number of points in this geometry that can be skin weighted."""
        cv_count_u = self.fn_set.numCVsInU
        if self.fn_set.formInU == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_u -= self.fn_set.degreeInU
        cv_count_v = self.fn_set.numCVsInV
        if self.fn_set.formInV == OpenMaya.MFnNurbsSurface.kPeriodic:
            cv_count_v -= self.fn_set.degreeInV
        return cv_count_u * cv_count_v

    def get_points(self, world_space: bool = True) -> OpenMaya.MPointArray:
        """Get the cv points of the nurbsSurface.

        Args:
            world_space: If True, query points in world space, otherwise object space.

        Returns:
            A MPointArray of the cvs.
        """
        space = OpenMaya.MSpace.kWorld if world_space else OpenMaya.MSpace.kObject
        return self.fn_set.cvPositions(space)

    # --- serialization

    def serialize(
        self, world_space: bool = True, include_uvs: bool = True
    ) -> BSplinePatchData:
        """Serialize this nurbsSurface to BSplinePatchData.

        Args:
            world_space: If True, query points in world space, otherwise object space.
            include_uvs: Unused; kept for signature parity with `Mesh.serialize`
                and `NurbsCurve.serialize`. BSplinePatchData does not store UVs.

        Returns:
            A BSplinePatchData object.
        """
        from cgmath.geometry import BSplinePatchData

        points = self.get_points(world_space=world_space)

        num_cvs_u  = self.fn_set.numCVsInU
        num_cvs_v  = self.fn_set.numCVsInV
        degree_u   = self.fn_set.degreeInU
        degree_v   = self.fn_set.degreeInV
        periodic_u = self.fn_set.formInU != OpenMaya.MFnNurbsSurface.kOpen
        periodic_v = self.fn_set.formInV != OpenMaya.MFnNurbsSurface.kOpen

        # Maya's cvPositions() returns a flat MPointArray of length
        # numCVsInU * numCVsInV ordered with V varying fastest -- i.e.
        # flat[u * numCVsInV + v] == CV at grid position (u, v). A C-order
        # reshape into (numCVsInU, numCVsInV, 3) gives the expected
        # `grid[u, v]` layout. The trailing homogeneous-w column from the
        # MPointArray is dropped via `[:, :3]`.
        grid = np.array(points)[:, :3].reshape(num_cvs_u, num_cvs_v, 3)

        # For periodic directions Maya exposes wrapped CVs (the first
        # `degree` rows/cols repeated at the end). BSplinePatchData expects
        # only the unique CVs and re-applies the wrap internally via its
        # geometry index, so trim them here for periodic axes.
        if periodic_u:
            grid = grid[: num_cvs_u - degree_u, :, :]
        if periodic_v:
            grid = grid[:, : num_cvs_v - degree_v, :]

        return BSplinePatchData(
            points     = grid,
            degree_u   = degree_u,
            degree_v   = degree_v,
            periodic_u = periodic_u,
            periodic_v = periodic_v,
        )