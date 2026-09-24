"""
Move all node types into the same namespace
"""

from rig.maya.nodetypes._base import Attribute, PyNode
from rig.maya.nodetypes.blendshape import BlendShape
from rig.maya.nodetypes.choice import Choice
from rig.maya.nodetypes.dag_node import DAGNode
from rig.maya.nodetypes.dg_node import DGNode
from rig.maya.nodetypes.display_layer import DisplayLayer
from rig.maya.nodetypes.follicle import Follicle
from rig.maya.nodetypes.geometry import Geometry
from rig.maya.nodetypes.joint import Joint
from rig.maya.nodetypes.mesh import Axis, Mesh
from rig.maya.nodetypes.nurbs import NurbsCurve, NurbsSurface
from rig.maya.nodetypes.object_set import ObjectSet
from rig.maya.nodetypes.reference import Reference
from rig.maya.nodetypes.shading_engine import ShadingEngine
from rig.maya.nodetypes.skincluster import SkinCluster
from rig.maya.nodetypes.transform import Transform