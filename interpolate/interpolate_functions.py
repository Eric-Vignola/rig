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

import numbers

from .._language import container, memoize, vectorize
from .._language import condition, container, constant

from ..functions import rev, choice, mag, searchSortedLeft
from ..trigonometry import radians, sin
from ..matrix import interpolate as matrix_interpolate
from ..matrix import decompose, compose
from ..quaternion import interpolate as quat_interpolate


# ------------------------------ INTERPOLATORS ------------------------------- #
@vectorize
@memoize
def lerp(input1, input2, weight=0.5):
    """ 
    lerp(<input1>, <input2>, weight=<weight>)

        Linear interpolation between two inputs and a weight value.

        Examples
        --------
        >>> lerp(pCube1.tx, pCube2.tx, weight=pCube3.weight)
    """
    if all([isinstance(x, numbers.Real) for x in [input1, input2, weight]]):
        return (input2 - input1) * weight + input1
    
    with container('lerp1'):
        input1 = container.publish_input(input1, 'input1')
        input2 = container.publish_input(input2, 'input2')
        weight = container.publish_input(weight, 'weight')
        
        output = (input2 - input1) * weight + input1
        
        return container.publish_output(output, 'output')


@vectorize
@memoize
def elerp(input1, input2, weight=0.5):
    """ 
    elerp(<input1>, <input2>, weight=<weight>)

        Exponent linear interpolation between two inputs and a weight value.

        Examples
        --------
        >>> elerp(pCube1.tx, pCube2.tx, weight=pCube3.weight)
    """
    if all([isinstance(x, numbers.Real) for x in [input1, input2, weight]]):
        return input1 ** rev(weight) * input2**weight 

    with container('elerp1'):
        input1 = container.publish_input(input1, 'input1')
        input2 = container.publish_input(input2, 'input2')
        weight = container.publish_input(weight, 'weight')
        
        output = input1 ** rev(weight) * input2**weight
        
        fuck = container.publish_output(output, 'output')
        return fuck


@vectorize
@memoize   
def slerp(input1, input2, weight=0.5):
    """ 
    slerp(<input1>, <input2>, weight=<weight>)

        Spherical linear interpolation between two inputs and a weight value.

        Examples
        --------
        >>> slerp(pCube1.tx, pCube2.tx, weight=pCube3.weight)
    """

    with container('slerp1'):
        input1 = container.publish_input(input1, 'input1')
        input2 = container.publish_input(input2, 'input2')
        weight = container.publish_input(weight, 'weight')
        
        
        node = container.createNode('angleBetween')        
        node.vector1 << input1
        node.vector2 << input2
        angle = radians(node.angle)

        # quiet div by zero
        sine = sin(angle)
        test = (sine==0)
        #_div = (input1 * sin(rev(weight) * angle) + input2 * sin(weight * angle)) / sine
        #_div.operation = condition(sine==0, 0, 2)
        _div = (input1 * sin(rev(weight) * angle) + input2 * sin(weight * angle)) / 1
        _div.operation = condition(test, 0, 2) 
        _div.input2 << sine

        output = condition(test, input1, _div)
        
        return container.publish_output(output, 'output')
    

@vectorize
@memoize
def transform(matrix0, matrix1, weight=0.5):
    """ 
    transform(<matrix0>, <matrix1>, weight=<weight>)

        Special case transform interpolation.

        Examples
        --------
        >>> transform(pCube1.wm, pCube2.wm, weight=pCube3.weight)
    """    
    with container('interpTransform1'):
        
        matrix0 = container.publish_input(matrix0, 'matrix0')
        matrix1 = container.publish_input(matrix1, 'matrix1')
        weight  = container.publish_input(weight, 'weight')        
        
        M0 = decompose(matrix0)
        M1 = decompose(matrix1)
        
        translate = lerp(M0.outputTranslate, M1.outputTranslate, weight=weight)
        rotate    = quat_interpolate(M0.outputQuat, M1.outputQuat, weight=weight)
        scale     = elerp(M0.outputScale, M1.outputScale, weight=weight)
        
        output = compose(scale=scale,
                         rotate=rotate,
                         translate=translate)
    
        return container.publish_output(output, 'output')
    
    

@memoize
def sequence(x, xp, yp, method=lerp):
    """ 
    sequence(<x>, <xp>, <yp>, method=lerp)

        Interpolates a sequence of values
        
        x:      The x-coordinates at which to evaluate the interpolated values.
        xp:     Monotonically increasing x-coordinates of the data points.
        yp:     The y-coordinates of the data points, same length as xp.
        method: Interpolation methods: lerp (default), slerp, elerp, transform
                Can also be rig.matrix.interpolate or rig.quaternion.interpolate.
                Interpolation format needs to be (value0, value1, weight)

        Examples
        --------
        >>> interp(pCube1.tx, 
                   [pCube2.tx, pCube3.tx, pCube4.tx], 
                   [pCube2.sy, pCube3.sy, pCube4.sy], 
                   method='elerp')
    """    
    
    with container('sequence1'):
        x  = container.publish_input(x, 'x')
        xp = container.publish_input(xp, 'xp')
        yp = container.publish_input(yp, 'yp')          
    
        # get the boundaries
        i  = searchSortedLeft(xp[:-1], x, return_index=True)
        x0 = choice(xp[:-1], selector=i)
        x1 = choice(xp[1:],  selector=i)
        y0 = choice(yp[:-1], selector=i)
        y1 = choice(yp[1:],  selector=i)
        
        # compute weight deltas
        weight  = (x-x0)/(x1-x0)
    
    
        # Return interpolation according to desired method
        output = method(y0, y1, weight=weight)
        
        return container.publish_output(output, 'output')
    
