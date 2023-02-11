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


from .._language import memoize, vectorize
from .._language import container, constant, _get_compound
from .._language import _quaternion_to_euler, _quaternion_add
from .._language import _quaternion_subtract, _quaternion_multiply

from ..matrix import compose
from ..functions import mag
from ..trigonometry import atan2



# ------------------------------- QUATERNIONS -------------------------------- #
@vectorize
@memoize
def add(quat1, quat2):
    """ 
    add(<quat1>, <quat2>)

        Returns the sum of added quaternions.

        Examples
        --------
        >>> add(pCube1.rq, pCube2.rq)
    """
    #node = container.createNode('quatAdd')
    #node.input1Quat << quat1
    #node.input2Quat << quat2

    #return node.outputQuat
    return _quaternion_add(quat1, quat2)


@vectorize
@memoize
def multiply(quat1, quat2):
    """ 
    multiply(<quat1>, <quat2>)

        Returns the product of multiplied quaternions.

        Examples
        --------
        >>> multiply(pCube1.rq, pCube2.rq)
    """
    #node = container.createNode('quatProd')
    #node.input1Quat << quat1
    #node.input2Quat << quat2

    #return node.outputQuat
    return _quaternion_multiply(quat1, quat2)


@vectorize
@memoize
def subtract(quat1, quat2):
    """ 
    subtract(<quat1>, <quat2>)

        Returns the sum of subtracted quaternions.

        Examples
        --------
        >>> subtract(pCube1.rq, pCube2.rq)
    """
    #node = container.createNode('quatSub')
    #node.input1Quat << quat1
    #node.input2Quat << quat2

    #return node.outputQuat
    return _quaternion_subtract(quat1, quat2)


@vectorize
@memoize
def negate(quat):
    """ 
    negate(<quat>)

        Negates a quaternion.

        Examples
        --------
        >>> negate(pCube1.rq)
    """
    node = container.createNode('quatNegate')
    node.inputQuat << quat

    return node.outputQuat


@vectorize
@memoize
def normalize(quat):
    """ 
    normalize(<quat>)

        Normalizes a quaternion.

        Examples
        --------
        >>> normalize(pCube1.rq)
    """
    node = container.createNode('quatNormalize')
    node.inputQuat << quat

    return node.outputQuat


@vectorize
@memoize
def invert(quat):
    """ 
    invert(<quat>)

        Inverts a quaternion.

        Examples
        --------
        >>> invert(pCube1.rq)
    """
    node = container.createNode('quatInvert')
    node.inputQuat << quat

    return node.outputQuat


@vectorize
@memoize
def conjugate(quat):
    """ 
    conjugate(<quat>)

        Conjugates a quaternion.

        Examples
        --------
        >>> conjugate(pCube1.rq)
    """
    node = container.createNode('quatConjugate')
    node.inputQuat << quat

    return node.outputQuat


@vectorize
@memoize
def angle(quat1, quat2):
    """ 
    angle(<quat0>, <quat1>)

        Returns the angle in radians of the shortest arc between two quaternions

        Examples
        --------
        >>> angle(pCube1.rq, pCube2.rq)
    """
    with container('quatAngle1'):
        quat1 = container.publish_input(quat1, 'input1Quat')
        quat2 = container.publish_input(quat2, 'input2Quat')
        
        qd = multiply(invert(quat1), quat2)
        output = 2 * atan2(mag(toVector(qd)), qd.outputQuatW)
        
        return container.publish_input(output, 'angle')


@vectorize
@memoize
def toEuler(quat, rotate_order=None):
    """ 
    toEuler(<quat>)

        Turns a quaternion into a euler angle.

        Examples
        --------
        >>> toEuler(pCube1.rq, rotate_order=pCube1.ro)
    """
    #node = container.createNode('quatToEuler')
    #node.inputQuat << quat
    #node.inputRotateOrder << rotate_order

    #return node.outputRotate
    return _quaternion_to_euler(quat, rotate_order=rotate_order)


@vectorize
@memoize
def toMatrix(quat, rotate_order=None):
    """ 
    toMatrix(<quat>)

        Turns a quaternion into a rotation matrix.

        Examples
        --------
        >>> toMatrix(pCube1.rq)
    """
    return compose(rotate=quat)



@vectorize
@memoize
def toVector(quat):
    """ 
    toVector(<quat>)

        Extracts the vector component of a quaternion.

        Examples
        --------
        >>> toVector(pCube1.rq)
    """
    attrs = _get_compound(quat)
    return constant(attrs[:-1])
    

@vectorize
@memoize
def interpolate(quat0, quat1, weight=0.5):
    """
    interpolate(<input>, <input>, ...)

        Slerps between two quaternions with optional weight values.
        (default = 0.5)

        Examples
        --------
        >>> quatSlerp(pCube1.wm, pCube2.wm)
        >>> quatSlerp(pCube1.wm, pCube2.wm, pCube1.weight)

    """
    node = container.createNode('quatSlerp')
    node.input1Quat << quat0
    node.input2Quat << quat1
    node.inputT << weight

    return node.outputQuat



