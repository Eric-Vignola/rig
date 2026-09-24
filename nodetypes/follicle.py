"""
follicle node class

TODO: write unit tests.
"""

from __future__ import annotations

import numpy as np
from maya import cmds
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.transform import Transform


class Follicle(DAGNode):
    """
    Follicle node class
    """

    NATIVE_NODE_TYPE = "follicle"

    @classmethod
    def create_on_mesh(
        cls,
        mesh:         Transform | str,
        ref_object:   list[str] | str | Transform,
        name:         str       | None            = None,
        uv_set:       str       | None            = None,
        fudge:        bool                        = True,
        fudge_factor: float                       = 0.00001,
    ):
        """
        Creates a follicle node and attaches it to a given mesh.
        (refactored from follicle_constraint() in  arvr/libraries/maya/scripts/OSS_node_utils.py)

        Args:
            mesh: A mesh transform node to attach the follicle to.
            ref_object: Reference object(s) for determine the follicle location.
                Can be a transform node or a list of vertices.
            name: Optional name for the follicle node.
            uv_set: A uv set to attach the follicle to.
            fudge: TODO
        """
        name  = name or "follicle"
        xform = Transform(mesh)
        mesh  = xform.get_shape()

        # get the desired follicle location from the reference object
        # (verts or a transform)
        if isinstance(ref_object, (list, tuple)):
            verts  = cmds.polyListComponentConversion(ref_object, tv=True)
            verts  = cmds.ls(verts, flatten=True)
            points = (cmds.xform(x, q=True, t=True, ws=True) for x in verts)
            t      = np.mean(np.array(points), axis=0)
        else:
            t = cmds.xform(ref_object, rp=True, q=True, ws=True)

        # create the follicle node and make connections
        follicle  = cls(cmds.createNode("follicle", name=name + "Shape"))
        fTranform = follicle.get_parent()
        follicle.outTranslate >> fTranform.translate
        follicle.outRotate    >> fTranform.rotate
        mesh.worldMatrix      >> follicle.inputWorldMatrix
        mesh.outMesh          >> follicle.inputMesh
        if uv_set:
            follicle.mapSetName.set(uv_set)

        # move follicle to the desired location
        uv = mesh.get_uv_at_point(t, uv_set=uv_set)
        follicle.set_uv_values(uv)

        # Oh fun, if closestPointOnMesh finds a UV edge (and edge isn't sharing another uv shell)
        # that u,v has a 50/50 chance of actually registering with follicle.  It might not find any mesh
        # So.... let's test that slightly move the uv in 4 directions to see if we can get something.
        #
        def is_at_origin(transform):
            position = cmds.xform(transform, q=True, t=True, ws=True)
            return position[0] == 0 and position[1] == 0 and position[2] == 0

        if fudge:
            if is_at_origin(fTranform):
                follicle.parameterU.set(uv[0] + fudge_factor)
            if is_at_origin(fTranform):
                follicle.set_uv_values(uv)
                follicle.parameterU.set(uv[0] - fudge_factor)
            if is_at_origin(fTranform):
                follicle.set_uv_values(uv)
                follicle.parameterV.set(uv[1] + fudge_factor)
            if is_at_origin(fTranform):
                follicle.set_uv_values(uv)
                follicle.parameterV.set(uv[1] - fudge_factor)
            if is_at_origin(fTranform):
                raise Exception(
                    f"Can't attach follicle {follicle.name} to meaningful uv at {uv}"
                )

        return follicle

    def constrain(
        self,
        constrainee:     str       | Transform,
        vertices:        list[str] | None      = None,
        maintain_offset: bool                  = True,
        translation:     bool                  = True,
        rotation:        bool                  = True,
    ) -> None:
        """Constrains a transform node to this follicle."""
        parent = self.get_parent()

        if maintain_offset is False:
            outPos = cmds.getAttr(f"{self.name}.outTranslate")
            cmds.setAttr(f"{constrainee}.translate", *outPos[0])
        # cmds.setAttr(follicle+".simulationMethod", 0)
        if translation:
            cmds.pointConstraint(parent, constrainee, mo=True)
        if rotation:
            cmds.orientConstraint(parent, constrainee, mo=True)

    def set_uv_values(self, uv: list[float]) -> None:
        """Sets the uv values of this follicle."""
        self.parameterU.set(uv[0])
        self.parameterV.set(uv[1])