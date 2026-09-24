"""
Move all node types into the same namespace
"""

from rig.nodetypes._base import Attribute, PyNode
from rig.nodetypes.blendshape import BlendShape
from rig.nodetypes.choice import Choice
from rig.nodetypes.dag_node import DAGNode
from rig.nodetypes.dg_node import DGNode
from rig.nodetypes.display_layer import DisplayLayer
from rig.nodetypes.follicle import Follicle
from rig.nodetypes.geometry import Geometry
from rig.nodetypes.joint import Joint
from rig.nodetypes.mesh import Axis, Mesh
from rig.nodetypes.nurbs import NurbsCurve, NurbsSurface
from rig.nodetypes.object_set import ObjectSet
from rig.nodetypes.reference import Reference
from rig.nodetypes.shading_engine import ShadingEngine
from rig.nodetypes.skincluster import SkinCluster
from rig.nodetypes.transform import Transform