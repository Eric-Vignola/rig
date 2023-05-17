"""
BSD 3-Clause License:
Copyright (c)  2023, Eric Vignola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:


1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of copyright holders nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import maya.cmds     as mc
import rig.commands  as rc
import rig.nodes     as rn
import rig.functions as rf
import numbers

from rig.attributes import Float, Vector, Enum, lock, hide
from rig import Node, List, container, condition
from rig import matrix
from rig import interpolate
from rig import constant
from rig._language import _is_sequence



def make_curve(cv, degree=3, periodic=False, name='railCurve1', k=None):
    """
    Builds an open or closed (periodic) curve with given desired control vertices

    You can implement your own knot vector (k) for fun curve crazyness
    """
    count = len(cv)
    if periodic:

        # compute closed knot vector
        if not k:
            k = list(range(0-degree+1,count+degree))

        # make the curve
        curve = rc.curve(per=periodic, d=degree, p=cv+cv[:degree], k=k)


    # compute open knot vector
    else:
        if not k:
            k = [min(max(0,x-degree+1), count-degree) for x in range(count+degree-1)]  

        # make the curve 
        curve = rc.curve(per=periodic, d=degree, p=cv, k=k)


    # rename the transform and shape
    curve = mc.rename(curve, name)
    
    # make the curve nice and smooth
    mc.displaySmoothness(curve, pointsWire=32)
    
    # return the curve and it's shape
    return (Node(curve), rc.listRelatives(curve, type='nurbsCurve')[-1])



def create_rail(position_controls,
                u,
                orient_controls=None,
                scale_controls=None,
                rail_name='rail1',
                rider_name='rider1',
                degree=3, 
                periodic=False,
                aim_axis=1,
                up_axis=0,
                invert_aim=False,
                invert_up=False,
                control_up=None,
                invert_up_control=False,
                debug=True):


    """
    Ye olde rail rig
    """
    
    def _order_controls(controls, position_controls):
        if not controls is None:
            if not _is_sequence(controls):
                controls = [controls]
                
            # if more then 1 control, make sure they're all position_controls
            # and and then sequentially order them to match the position_controls order
            if len(controls) > 1:
                if not all([str(x) in position_controls for x in controls]):
                    raise Exception('When more than 1, controls must be members of position_controls.')
    
                controls = [x for x in position_controls if str(x) in list([str(y) for y in controls])]
        
            return List(controls)
        
        return None
    
               
                 

    # ---------------- BEGIN! ---------------- #
    with container('railNode1'):    
        
        # do some input cleanup
        if control_up is None:
            control_up = up_axis
            

        position_controls = List(position_controls)
        orient_controls   = _order_controls(orient_controls, position_controls)
        scale_controls    = _order_controls(scale_controls,  position_controls)
        

        # Create the spread of u values the riders will be spread on
        # if u is a numeric value, it represents a count of equidistent rail riders.
        if isinstance(u, numbers.Real):
            if u > 1:
                u = [x/(u-1.) for x in range(u)]
    
            elif u == 1:
                u = [0.]
                
                
        # this must be a list, make sure numbers are converted
        # to floats, and strings are converted to Nodes
        else:
            u = list(u)
            for i, v in enumerate(u):
                if isinstance(v, str):
                    u[i] = Node(u[i])
                else:
                    u[i] = float(u[i])
                
                
            
        
        # get curve control positions
        pos = [mc.xform(x, q=True, ws=True, t=True) for x in position_controls]
        
        # if periodic, roll the positions so u=0 matches first controller
        if periodic and degree > 1:
            if degree==2:
                raise Exception('periodic rail spine does not support degree 2 curves')
            
            # shift the verts so knots can register with the control points
            shift = {3:2, 5:3, 7:4}[degree]
            pos_ = pos[shift:]
            pos_ += pos[:shift]
            pos = pos_
            
            

        rail,  rail_shape = make_curve(pos, 
                                       degree=degree, 
                                       periodic=periodic, 
                                       name=rail_name)

        # hide SRT channels
        rail.s << hide
        rail.r << hide
        rail.t << hide

   
        # Build the proxy shape (with horrible hack to parent the shape under main rail transform)
        proxy, proxy_shape = make_curve(pos, 
                                        degree=degree, 
                                        periodic=periodic, 
                                        name='{}Proxy'.format(rail))
    
        proxy_shape.v << False # hide the shape 
        mc.parent(proxy_shape, rail, r=True, s=True) # horrible hack to duplicate a shape
        mc.delete(proxy)                             # ugh i'm so sorry, Maya's fault 
    

        # Connected the controls to the rail's cv's
        if not periodic or (periodic and degree == 1):
            rail_shape.cv[:] << position_controls.wm * rail.wim
        
        else:
            # shift the cv order so the proper control points move the proper knots
            shift = {3:1, 5:2, 7:3}[degree]
            cvs   = list(rail_shape.cv[:])
            cvs_  = cvs[shift:]
            cvs_ += cvs[:shift]
            cvs   = List(cvs_)         
            
            cvs << position_controls.wm * rail.wim
            
            
        # now that the curve is properly constructed, if periodic append
        # the first control to the end of the lise for proper cumsum 
        if periodic:
            position_controls.append(position_controls[0])
    
    
        # Create curve info nodes to get the original lengths
        current_length = (rn.curveInfo().inputCurve << rail_shape.worldSpace[0] ).arcLength
        default_length = (rn.curveInfo().inputCurve << proxy_shape.worldSpace[0]).arcLength
    
    
        # Precompute offset data which will be applied to all riders along the rail.
        stretch_ratio  = default_length / current_length
        stretch_delta  = (current_length - default_length) / current_length
        
    
        # Add useful debug attributes and plug in the proper values
        rail << Float('defaultLength', k=False) << default_length
        rail << Float('currentLength', k=False) << current_length
        rail << Float('stretchRatio',  k=False) << stretch_ratio
        rail << Float('stretchDelta',  k=False) << stretch_delta
    
    
        # Add common attributes to the rail to dictate rider's orientation and offset
        rail << Enum('aimAxis', en='X:Y:Z:', dv=aim_axis)
        rail << Enum('upAxis',  en='X:Y:Z:', dv=up_axis) 
        rail << Enum('invertAim')
        rail << Enum('invertUp')
    
        rail << Float('pivot',          min=0, max=1)
        rail << Float('stretch',  dv=1, min=0, max=1)
        rail << Float('Scale',    dv=1)
        rail << Float('shift')
        
        
        rail << Enum('scaleProjection',  en='Frozen:Infinite:Clamped')
        rail << Enum('rotateProjection', en='Frozen:Infinite:Clamped')
        
  
        
        # ---- Prep orient/scale vector modulation ---- #
        orient_vectors = None
        orient_weights = None
        scale_vectors  = None
        scale_weights  = None
        
        if orient_controls or scale_controls:
            
            # record ratios between position_controls
            deltas = rf.dist(position_controls.wm[:-1], position_controls.wm[1:])
            if len(position_controls) > 2:
                cumsum = rf.cumsum(deltas)
                cumsum = cumsum/cumsum[-1]
                cumsum = [constant(0)] + list(cumsum)               
            else:
                cumsum = [constant(0),constant(1)]
                
            
            # plug the cumulative sum's u values back to control for debug
            if debug:
                position_controls << Float('u') << cumsum << lock

            # plug compression, for future feature?
            #rail << Float('uCompression', multi=True) << hide
            #rail.uCompression[:len(deltas)] << deltas/(deltas >> None)            

            
            # orientation
            if not orient_controls is None:
                orient_controls = List(orient_controls)
                orient_controls << Enum('upAxis',  en='X:Y:Z:', dv=up_axis)
                orient_controls << Enum('invertUp')
                
                orient_vectors = rf.choice([[1,0,0],
                                            [0,1,0],
                                            [0,0,1]], 
                                           selector=orient_controls.upAxis)              
    
                orient_vectors = orient_vectors * condition(orient_controls.invertUp, -1, 1)
                orient_vectors = rf.unit(matrix.multiply(orient_vectors, orient_controls.wm, local=True))
                                
                
                if len(orient_controls) > 1:
                    orient_weights = List([cumsum[position_controls.index(x)] for x in orient_controls])
                    
                    if periodic:
                        # if this is a complete loop, append beginning to end
                        if position_controls.index(orient_controls[0]) == 0:
                            orient_weights.append(constant(1))
                            orient_vectors.append(orient_vectors[0])
                          
                        # add buffers to beginning and end  
                        else:
                            orient_weights = List([0-(1-orient_weights[-1])] + list(orient_weights))
                            orient_vectors = List([orient_vectors[-1]]  + list(orient_vectors))
                                                    
                            orient_weights.append(1-(0-orient_weights[1]))
                            orient_vectors.append(orient_vectors[1])
                      
                            
                    # orient weights frozen state
                    orient_weights = condition(rail.rotateProjection==0, 
                                               orient_weights >> None,
                                               orient_weights)                      
                            
                        
            # scale
            if not scale_controls is None:
                scale_controls = List(scale_controls)
                scale_vectors  = matrix.decompose(scale_controls.wm).outputScale
                
                if len(scale_controls) > 1:
                    scale_weights  = List([cumsum[position_controls.index(x)] for x in scale_controls])

                    if periodic:
                        # if this is a complete loop, append beginning to end
                        if position_controls.index(scale_controls[0]) == 0:
                            scale_weights.append(constant(1))   
                            scale_vectors.append(scale_vectors[0])
                          
                        # add buffers to beginning and end    
                        else:
                            scale_weights = List([0-(1-scale_weights[-1])] + list(scale_weights))
                            scale_vectors = List([scale_vectors[-1]]  + list(scale_vectors))
                                                    
                            scale_weights.append(1-(0-scale_weights[1]))
                            scale_vectors.append(scale_vectors[1])
                            
                            
                    # scale weights frozen state
                    scale_weights = condition(rail.scaleProjection==0, 
                                              scale_weights >> None,
                                              scale_weights)                    
                            


    
        # If the curve is open, we'll add tangents at both ends to position riders
        # beyond the curve's domain when the curve is squashed.
        if not periodic:
            rail << Enum('translateProjection', en='Clamped:Infinite', dv=1)
            rail << Float('uTangentStart', dv=0.001, min=0, max=1) << hide # reasonable near 0 value
            rail << Float('uTangentEnd',   dv=0.999, min=0, max=1) << hide # reasonable near 1 value
    
            tangent_aim = rf.choice([[1,0,0],
                                     [0,1,0],
                                     [0,0,1]],
                                    selector=rail.aimAxis)
            

            # tangent0 will be located at the start of the rail
            # at a reasonable near 0 value to allow collapsed start points.
            tangent0 = rn.motionPath()
            tangent0.fractionMode  << 1 # turn on percentage mode
            tangent0.uValue        << rail.uTangentStart
            tangent0.geometryPath  << rail_shape.worldSpace[0]
    
            tangent0.frontAxis     << rail.aimAxis
            tangent0.upAxis        << rail.upAxis
            tangent0.inverseFront  << rail.invertAim
            tangent0.inverseUp     << rail.invertUp
            
            edge0    = tangent0.allCoordinates
            tangent0 = rf.unit(tangent_aim * tangent0.orientMatrix) * current_length * rail.translateProjection

    
            # tangent1 will be located at the end of the rail
            # at a reasonable near 1 value to allow collapsed start points.    
            tangent1 = rn.motionPath()
            tangent1.fractionMode  << 1 # turn on percentage mode
            tangent1.uValue        << rail.uTangentEnd
            tangent1.geometryPath  << rail_shape.worldSpace[0]
    
            tangent1.frontAxis     << rail.aimAxis
            tangent1.upAxis        << rail.upAxis
            tangent1.inverseFront  << rail.invertAim
            tangent1.inverseUp     << rail.invertUp   
                  
            edge1    = tangent1.allCoordinates
            tangent1 = rf.unit(tangent_aim * tangent1.orientMatrix) * current_length * rail.translateProjection
    
            

        # ---- ADD RAIL RIDERS ---- #
        
        for u_default in u:
            
            # create a rider
            rider = rn.joint(name=rider_name, container=False) # make a joint but not in the container
            
            # by parenting the rider under the rail, rider's name will be 
            # automatically renamed and indexed by the rc.parent operation.
            # because rc is used, the command remains aware of the node's
            # name change... with mc, we'd have to capture the return and recast
            # as a node, ex: rider = Node(mc.parent(rider, rail))
            rc.parent(rider, rail)
            
            if debug:
                debug_cube = rc.polyCube()[0]
                rc.parent(debug_cube, rider)

            
            # add it's assigned u value
            rider << Float('uDefault') << u_default
            u_default = rider.uDefault
    
            rider_path = rn.motionPath()
            rider_path.fractionMode  << 1 # turn on percentage mode
            rider_path.geometryPath  << rail_shape.worldSpace[0]
    
            rider_path.frontAxis     << rail.aimAxis
            rider_path.upAxis        << rail.upAxis
            rider_path.inverseFront  << rail.invertAim
            rider_path.inverseUp     << rail.invertUp   
            
            
            # modulate the u translation
            u_translate = (u_default*stretch_ratio + stretch_delta*rail.pivot) * rf.rev(rail.stretch)
            u_translate = u_translate + u_default*rail.stretch
            u_translate = rail.shift + rail.pivot + (u_translate - rail.pivot) * rail.Scale
            
            if periodic:
                u_translate = u_translate % 1               
            
            
            # plug u translation into the motion path
            rider_path.uValue << u_translate  
            
            
            # create a final matrix to move the rider
            rider_matrix = rn.composeMatrix()
            
            
            # -------- PLUG SCALE -------- #
            if scale_controls:
                if len(scale_controls) == 1:
                    rider_matrix.inputScale << scale_vectors
                    
                else:
                    frozen   = condition(rail.scaleProjection==0, 
                                         u_default,
                                         u_translate)                    
                    
                    clamped  = condition(rail.scaleProjection==2,
                                         rf.clamp(frozen,
                                                  scale_weights[0],
                                                  scale_weights[-1]),
                                         frozen)     
                                        
                    
                    rider_matrix.inputScale << interpolate.sequence(clamped, 
                                                                    scale_weights, 
                                                                    scale_vectors,
                                                                    method=interpolate.elerp)
            
            
            # -------- PLUG ROTATION -------- #
            if orient_controls:
                rider_matrix.inputRotate << rider_path.rotate
                
                if len(orient_controls) == 1:
                    rider_path.worldUpVector << orient_vectors
                    
                else:
                    
                    frozen   = condition(rail.rotateProjection==0, 
                                         u_default,
                                         u_translate)                    
                
                    clamped  = condition(rail.rotateProjection==2,
                                         rf.clamp(frozen,
                                                  orient_weights[0],
                                                  orient_weights[-1]),
                                         frozen)                      
                    

                    rider_path.worldUpVector << interpolate.sequence(clamped, 
                                                                     orient_weights, 
                                                                     orient_vectors, 
                                                                     method=interpolate.slerp)
                    
                

            # -------- PLUG TRANSLATION -------- #
            local_position = rider_path.allCoordinates
            
            # if not periodic, set the projections
            if not periodic:
                local_position = condition(u_translate < 0, (tangent0 *  u_translate)    + edge0, rider_path.allCoordinates)
                local_position = condition(u_translate > 1, (tangent1 * (u_translate-1)) + edge1, local_position)
                           

            rider_matrix.inputTranslate << local_position
            rider_matrix = rider_matrix.outputMatrix * rail.wim
        
            rider << rider_matrix # plug srt+shear




#import rig
#import rig.commands as rc
#import numpy as np
#position_controls = rc.ls(sl=True)
#u = np.linspace(0,1,20)
#degree = 3
#periodic = False

#create_rail(position_controls,
            #u,
            #scale_controls=position_controls,
            #orient_controls=position_controls,
            #degree=degree, 
            #periodic=periodic,
            #aim_axis=1, up_axis=0)
            
