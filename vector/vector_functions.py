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

from .._language import container, memoize, _is_sequence, _get_compound
from .._language import constant, container, condition
from ..functions import unit, cross, min, max, choice


X = [1,0,0]
Y = [0,1,0]
Z = [0,0,1]


@memoize
def toMatrix(aim_vector, up_vector=Y, aim_axis=X, up_axis=Y, position=None):
    """ 
    toMatrix(aim_vector, up_vector=Y, aim_axis=X, up_axis=Y, position=None)

        Constructs an orthogonal matrix from an aim and up vector.
        Aim and Up axes can be specified as 0:x 1:y 2:z
        An optional position component can be piped in.
        This acts similarly to an aim constraint.

        Examples
        --------
        >>> toMatrix(pCube1.t, pCube2.t)
    """
    
    # If aimMatrix exists
    if 'aimMatrix' in mc.ls(nt=True):
        
        node = container.createNode('aimMatrix', n='vectorToMatrix1')
        node.primaryMode           << 1 # aim
        node.secondaryMode         << 2 # align
        
        node.primaryTargetVector   << aim_vector
        node.secondaryTargetVector << up_vector
        
        node.primaryInputAxis      << aim_axis
        node.secondaryInputAxis    << up_axis

        return node.outputMatrix
    
    
    
    # construct an aimMatrix setup
    # in this mode, the aim/up axis must be absolute (X:0, Y:1, Z:2)
    if _is_sequence(aim_axis):
        aim_axis = aim_axis.index(max([abs(x) for x in aim_axis]))
        
    if _is_sequence(up_axis):
        up_axis = up_axis.index(max([abs(x) for x in up_axis]))
    
    
    with container('vectorToMatrix1'):

        V0 = unit(aim_vector)
        V2 = cross(aim_vector, up_vector, normalize=True)
        V1 = cross(V2, aim_vector, normalize=True)

        # Figure out which axis we're trying to extrapolate
        ii = aim_axis
        jj = up_axis        
        kk = ( min(ii,jj) - max(ii,jj) + min(ii,jj) ) % 3

        # catch the axis flip condition
        #flip = constant(0)
        flip = (((ii == 0) & (jj == 2)) | 
                ((ii == 1) & (jj == 0)) |
                ((ii == 2) & (jj == 1)))

        V2 = condition(flip, -V2, V2)


        # catch the remapped ii/jj/kk axes condition
        # 0,1 == 0,1,2 ---> 0,1,2 # IDENTICAL
        # 0,2 == 0,2,1 ---> 0,2,1 # IDENTICAL
        # 1,0 == 1,0,2 ---> 1,0,2 # IDENTICAL
        # 1,2 == 1,2,0 ---> 2,0,1 # !!! REMAP !!!
        # 2,0 == 2,0,1 ---> 1,2,0 # !!! REMAP !!!
        # 2,1 == 2,1,0 ---> 2,1,0 # IDENTICAL
        test0 = ((ii == 1) & (jj == 2))
        test1 = ((ii == 2) & (jj == 0))

        ii = condition(test0, 2, ii)
        jj = condition(test0, 0, jj)
        kk = condition(test0, 1, kk)

        ii = condition(test1, 1, ii)
        jj = condition(test1, 2, jj)
        kk = condition(test1, 0, kk)


        # plug into a a 4x4 matrix
        choice_x = constant(choice(V0,V1,V2, selector=ii))
        choice_y = constant(choice(V0,V1,V2, selector=jj))
        choice_z = constant(choice(V0,V1,V2, selector=kk))


        return matrix(x=choice_x,
                      y=choice_y,
                      z=choice_z,
                      position=position)


@memoize
def toEuler(aim_vector, up_vector, aim_axis=X, up_axis=Y, rotate_order=0):
    """ 
    toEuler(aim_vector, up_vector, aim_axis=X, up_axis=Y, rotate_order=0)

        Constructs an orthogonal matrix from an aim and up vector,
        and outputs an Euler angle with a given rotate_order.
        Aim and Up axes can be specified as 0:x 1:y 2:z


        Examples
        --------
        >>> toEuler(pCube1.t, pCube2.t, rotate_order=pCube3.ro)
        """

    with container('vectorToEuler1'):
        aim_vector = container.publish_input(aim_vector, 'aimVector')
        up_vector  = container.publish_input(up_vector,  'upVector')
        aim_axis   = container.publish_input(aim_axis,   'aimAxis')
        up_axis    = container.publish_input(up_vector,  'upAxis')        
        
        node = toMatrix(aim_vector, up_vector, aim_axis=aim_axis, up_axis=up_axis)
        node = matrixDecompose(node, rotate_order=rotate_order)

        output = node.outputRotate
        return container.publish_output(output, 'output')


@memoize
def toQuaternion(aim_vector, up_vector, aim_axis=X, up_axis=Y):
    """ 
    toQuaternion(aim_vector, up_vector, aim_axis=X, up_axis=Y)

        Constructs an orthogonal matrix from an aim and up vector,
        and outputs a quaternion.
        Aim and Up axes can be specified as 0:x 1:y 2:z


        Examples
        --------
        >>> toQuaternion(pCube1.t, pCube2.t, rotate_order=pCube3.ro)
        """

    with container('vectorToQuat1'):
        aim_vector = container.publish_input(aim_vector, 'aimVector')
        up_vector  = container.publish_input(up_vector,  'upVector')
        aim_axis   = container.publish_input(aim_axis,   'aimAxis')
        up_axis    = container.publish_input(up_vector,  'upAxis')
        
        node = toMatrix(aim_vector, up_vector, aim_axis=aim_axis, up_axis=up_axis)
        node = matrixDecompose(node)

        output = node.outputQuat
        
        return container.publish_output(output, 'output')


@memoize
def determinant(x, y, z):
    """ 
    determinant(x, y ,z)

        Returns the determinent of a 3x3 square matrix given
        by 3 vectors representing a matrix X,Y,Z axes.

        Examples
        --------
        >>> determinant(pCube1.t, pCube2.t, pCube3.t)
        """

    with container('vectorDeterminant1'):
        x = container.publish_input(x, 'vector1')
        y = container.publish_input(y, 'vector2')
        z = container.publish_input(z, 'vector3')
        
        X = _get_compound(x)
        Y = _get_compound(y)
        Z = _get_compound(z)

        o = X[0]*(Y[1]*Z[2] - Y[2]*Z[1]) - X[1]*(Y[0]*Z[2] - Y[2]*Z[0]) + X[2]*(Y[0]*Z[1] - Y[1]*Z[0])

        return container.publish_output(o, 'determinant')
