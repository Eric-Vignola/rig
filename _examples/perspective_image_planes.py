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

import maya.cmds as mc
import rig.commands  as rc
import rig.nodes     as rn
import rig.functions as rf

from rig.attributes import Int, Float, String, Enum, lock, hide
from rig import Node, List, container, condition
from rig import trigonometry as trig

from collections import namedtuple


def get_scale(aperture, focal_length, distance):
    """
    Computes the scale of an image plane relative to the camera
    """
    focal_length = focal_length * 0.0393700787 # conv mm to inches
    aov = trig.atan(aperture/(2*focal_length)) # angle of view

    x = trig.sin(aov)
    y = (1-(x**2))**0.5

    return x*2 * distance/y



def create_setup(camera,                    # name of the camera to create
                 count,                     # number of planes to create
                 name    = 'STICKER_LAYER', # plane names 
                 default = None,            # any default image
                 parent  = None,            # parent (if different from camera)
                 offset  = 10):             # offset in X between planes
    
    
    
    # prep a named tuple container to return everything created in a nice package
    output = namedtuple('output', 'camera planes')
    
    # put all image entries directly on the camera for convenience
    camera, camera_shape = rc.camera(name=camera, container=False)
    output.camera = camera
    output.planes = List()
    output.shapes = List()
    
    camera.filmFit                << 3     # set camera to overscan
    camera.overscan               << 1.1   # so there's always a box visible
    camera.displayResolution      << False # hide resolution gate
    camera.displayFilmGate        << True  # show film gate
    camera.displayGateMask        << True  # show film mask
    camera.displayGateMaskOpacity << 1     # set mask opacity to opaque
    
    
    #--- create the image plane setup ---#
    horizonal_list = List()
    vertical_list  = List()
    
    with container('{}_container'.format(name.lower())) as node_container:
        
        # add the camera shape 
        node_container.add(camera_shape)
        
        for i in range(count):
            
            # append a number to the end of the name
            name_suffix = '{}_{}'.format(name,i)
            
            
            # make an image plane
            transform, plane = rc.polyPlane(sx=1, 
                                            sy=1, 
                                            ax=[0,0,1], 
                                            n='mesh_{}'.format(name_suffix.upper()),
                                            container=False)
    
            
            # add plane constructor to container and
            # parent the transform to something useful
            container.add(plane)
            
            if parent:
                mc.parent(transform, parent)
            else:
                mc.parent(transform, camera)
                
                
            # offset the plane
            transform.tz << (i+1) * -offset
            
            
            # lock and hide transform channels
            transform.tx << lock << hide
            transform.ty << lock << hide        
            transform.s  << lock << hide
            transform.r  << lock << hide
            transform.v  << lock << hide 
            
            
            # create a shading network for the poly plane
            material_name = name_suffix.lower()
            material = rc.shadingNode('lambert', 
                                      asShader=True, 
                                      name=material_name)
            material.diffuse      << 0
            material.ambientColor << 1
            
            shading_engine = rc.sets(name="{}SG".format(material),
                                     empty=True,
                                     renderable=True,
                                     noSurfaceShader=True)
            
            shading_engine.surfaceShader << material.outColor
            
            
            # assign shader to plane
            Node('defaultShaderList1.s') << material.msg # hypershade backwards compatibility
            shape = rc.listRelatives(transform, type='mesh')[0]
            rc.sets(shape, e=True, forceElement=shading_engine)
            
            
            # create a file node and plug it into shader
            texture = rn.file()
            material.color        << texture.outColor
            material.ambientColor << texture.outColor
            material.diffuse      << 1

            shape << String('image') << default
            shape << Enum('sequenceType', en='Static:Sequence:Looping:Ping Pong')

            shape << Int('sequenceStart', min=0)
            shape << Int('sequenceEnd'  , min=0, dv=100)
            shape << Int('sequenceOffset')
            
            
            # setup the animated background
            current  = rf.frame() - shape.sequenceStart + shape.sequenceOffset
            duration = shape.sequenceEnd - shape.sequenceStart
            static   = rf.clamp(current, shape.sequenceStart, shape.sequenceEnd)
            looping  = current % (duration+1)
            
            # ping pong!
            length   = 2*duration
            state    = rf.abs(current-shape.sequenceStart) % length
            pingpong = condition(state > duration, length-state, state+shape.sequenceStart)
            
            
            texture.fileTextureName   << shape.image
            texture.useFrameExtension << (shape.sequenceType > 0)
            texture.frameOffset       << lock # don't use this!
            #texture.frameExtension    << condition(shape.sequenceType==2, looping, static)
            texture.frameExtension    << rf.choice([0, static, looping, pingpong], selector=shape.sequenceType)
            
            # remap transparency
            remap = rn.colorCorrect()
            remap.inColor         << texture.outTransparency
            shape                 << Float('alpha',    dv=0, min=0, max=1)
            shape                 << Float('opacity',  dv=1, min=0, max=1)
            remap.colGain         << rf.rev(rf.rev(shape.alpha) * shape.opacity)
            remap.colOffset       << rf.rev(shape.opacity)
            material.transparency << remap.outColor
            
            
    
            # build the math node relationships
            # inputs:
            # - distance to camera
            # - camera's aperture (if not ortho)
            # - image's size
            
            # make a proxy texture file node to retrieve image size
            texture_ = rn.file()
            texture_.fileTextureName << shape.image            
            
            # if texture is not set, set image aspect ratio to 1
            distance   = rf.dist(transform.wm, camera.wm)
            width      = condition(texture_.outSizeX > 0, texture_.outSizeX, 1)
            height     = condition(texture_.outSizeY > 0, texture_.outSizeY, 1)
            horizontal = condition(height > width,  width/height, 1)
            vertical   = condition(width  > height, height/width, 1)
        
            X = get_scale(horizontal, camera_shape.fl, distance)
            Y = get_scale(vertical,   camera_shape.fl, distance)
        
            X  = condition(camera_shape.orthographic, (X/Y) * camera_shape.ow, X)
            Y  = condition(camera_shape.orthographic, camera_shape.ow, Y)
            
            X_ = condition(camera_shape.orthographic, camera_shape.ow, X)
            Y_ = condition(camera_shape.orthographic, (Y/X) * camera_shape.ow, Y)    

            plane.width  << condition(X<Y, X, X_)
            plane.height << condition(X<Y, Y, Y_)
        
        
            # build list of values to be plugged in the camera's aspect ratio
            h = condition(vertical < horizontal, 1, X/Y)
            v = condition(horizontal < vertical, 1, Y/X)        
            
            horizonal_list.append(h)
            vertical_list.append(v)
            
            
            # add the created plane to the output list
            output.planes.append(transform)
            output.shapes.append(shape)
    
        
    
        # plug the system back into the camera's aspect ratio
        if count > 1:  
            camera_shape.hfa << rf.max(horizonal_list)
            camera_shape.vfa << rf.max(vertical_list)
        else:
            camera_shape.hfa << horizonal_list
            camera_shape.vfa << vertical_list        
            
        return output
        
        
    

"""
#--- EXAMPLE WORKFLOW ---#

# create the camera setup with 5 image planes
setup = create_setup('camera1', 5)



# populate image planes with texture files
# also use the $HOME env variable to demonstrate environment variable support
import glob
images = [x.replace('\\','/') for x in glob.glob(os.path.expandvars(r'Z:\fbavatars\_common\library\maya\scripts\python\3rd-party\rig2\_examples\images\*.jpg'))]
images = [x.replace(os.path.expandvars('$HOME'),'$HOME') for x in images]

setup.shapes.image[:] << images


# activate all alpha channels but the last one
setup.planes.alpha[:-1] << 1


# have fun!

"""
