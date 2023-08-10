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
from .._language import condition, container, _constant
from .._language import _get_compound,  _is_compound
from .._generators import sequences

from ..functions import rev, choice, mag, searchsorted, clamp
from ..trigonometry import radians, sin
from ..matrix import decompose, compose
from ..quaternion import interpolate as quat_interpolate


import maya.cmds as mc
MAYA_VERSION = int(mc.about(version=True))



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
    
    
    # new acos node type in Maya 2024
    if MAYA_VERSION >= 2024:
        if not _is_compound(input1) and not _is_compound(input2) and not _is_compound(weight):
            node = container.createNode('lerp')
            node.input1 << input1
            node.input2 << input2
            node.weight << weight
            return node.output
        
        else:
            with container('lerp1'):
                input1 = container.publish_input(input1, 'input1')
                input2 = container.publish_input(input2, 'input2')
                weight = container.publish_input(weight, 'weight')
                
                input_plugs1 = _get_compound(input1)
                input_plugs2 = _get_compound(input2)
                input_weight = _get_compound(weight)
                output_plugs = []
                
                for p1,p2,w in sequences(input_plugs1, input_plugs2, input_weight):
                    node = container.createNode('lerp')
                    node.input1 << p1
                    node.input2 << p2
                    node.weight << w
                    output_plugs.append(node.output)
                
                count  = len(output_plugs)
                output = _constant([0]*count, name='output_plug1')
                output << output_plugs
                
                return container.publish_output(output, 'output')      


    # default to old method    
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
        _div.operation << condition(test, 0, 2) 
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
        i  = searchsorted(xp[:-1], x, return_index=True, side='left')
        x0 = choice(xp[:-1], selector=i)
        x1 = choice(xp[1:],  selector=i)
        y0 = choice(yp[:-1], selector=i)
        y1 = choice(yp[1:],  selector=i)
        
        # compute weight deltas
        weight  = (x-x0)/(x1-x0)
    
    
        # Return interpolation according to desired method
        output = method(y0, y1, weight=weight)
        
        return container.publish_output(output, 'output')
    


    
@vectorize
@memoize    
def smoothstep(input1, input2, weight=0.5, normalize=False):
    
    """ 
    smoothstep(<token>, <min_value>, <max_value>)

        Perform a smoothstep interpolation between two values
        https://en.wikipedia.org/wiki/Smoothstep

        Examples
        --------
        >>> smoothstep(pCube1.tx)
    """
    if all([isinstance(x, numbers.Real) for x in [weight, input1, input2]]):
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        x = x * x * (3.0 - 2.0 * x)
        
        if not normalize:
            return x
        
        return  x * (input2 - input1) + input1


    # new ceil node type in Maya 2024
    if MAYA_VERSION >= 2024:
        if not any([_is_compound(weight), _is_compound(input1), _is_compound(input2)]):
            if not normalize:
                node = container.createNode('smoothStep', name='smoothStep1')
                node.input     << weight
                node.leftEdge  << input1 
                node.rightEdge << input2 
                return node.output
            
            else:
                with container('smoothstep1'):
                    input1 = _get_compound(container.publish_input(input1, 'min'))
                    input2 = _get_compound(container.publish_input(input2, 'max'))
                    weight = _get_compound(container.publish_input(weight, 'weight'))
                    
                    node = container.createNode('smoothStep', name='smoothStep1')
                    node.input     << weight
                    node.leftEdge  << input1 
                    node.rightEdge << input2
                    
                    output = node.output * (input2 - input1) + input1
                    
                    return container.publish_output(output, 'output')                   
        
        else:
            with container('smoothstep1'):
                input1 = _get_compound(container.publish_input(input1, 'min'))
                input2 = _get_compound(container.publish_input(input2, 'max'))
                weight = _get_compound(container.publish_input(weight, 'weight'))
            
                output_plugs = []
                for plugs in sequences(weight, input1, input2):
                    node = container.createNode('smoothStep', name='smoothStep1')
                    node.input     << plugs[0]
                    node.leftEdge  << plugs[1]
                    node.rightEdge << plugs[2]
                    output_plugs.append(node.output)                    
                
                count  = len(output_plugs)
                output = _constant([0]*count, name='output_plug1')
                output << output_plugs
                
                if not normalize:
                    output = output * (input2 - input1) + input1
                    
                return container.publish_output(output, 'output')

        
        
    # old fashioned way
    with container('smoothstep1'):
        weight     = container.publish_input(weight,     'input')
        input1 = container.publish_input(input1, 'min')
        input2 = container.publish_input(input2, 'max')
        
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        output = x * x * (3.0 - 2.0 * x)
        
        if not normalize:
            output = output * (input2 - input1) + input1
            
        return container.publish_output(output, 'output')       



@vectorize
@memoize    
def smootherstep(input1, input2, weight=0.5, normalize=False):
    
    """ 
    smootherstep(<token>, <min_value>, <max_value>)

        Perform a smootherstep interpolation between two values
        https://en.wikipedia.org/wiki/Smoothstep

        Examples
        --------
        >>> smootherstep(pCube1.tx)
    """
    if all([isinstance(x, numbers.Real) for x in [weight, input1, input2]]):
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        x = x * x * x * (x * (x * 6 - 15) + 10)
        
        if not normalize:
            return x

        return  x * (input2 - input1) + input1
    
    
    with container('smootherstep1'):
        input1 = container.publish_input(input1, 'min')
        input2 = container.publish_input(input2, 'max')
        weight = container.publish_input(weight, 'weight')
        
        x = clamp((weight - input1) / (input2 - input1), 0.0, 1.0)
        output = x * x * x * (x * (x * 6 - 15) + 10)
        
        if not normalize:
            output = output * (input2 - input1) + input1
            
        return container.publish_output(output, 'output')    
    
    
    
    
    
    
#@vectorize
#@memoize
#def sigmoid(token, min_val=0, max_val=1):
    #"""
    #sigmoid(<token>, <min_value>, <max_value>)

        #Clamps values between a min and a max using a sigmoid function
        #https://en.wikipedia.org/wiki/Smoothstep
        #https://en.wikipedia.org/wiki/Sigmoid_function

        #Examples
        #--------
        #>>> smoothclamp(pCube1.tx)
    
    #"""
    
    #with container('exponentRange1'):
        #token   = container.publish_input(token, 'input')
        #min_val = container.publish_input(min_val, 'min')
        #max_val = container.publish_input(max_val, 'max')
        
        #delta  = token - min_val
        #spread = max_val - min_val
        #test   = delta/spread
        #ratio  = 1 - exp(-1 * abs(test))
        
        #output = min_val + (ratio * spread * sign(test))
        #return container.publish_output(output, 'output')
        
          

#def sigmoid(x, mi, mx):
    #x = (x - mi) / (mx - mi)
    #x = (1+200**(-x+0.5))**-1
    #return x * (mx - mi) + mi

