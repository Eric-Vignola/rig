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

import math
import numbers

from .._language import memoize, vectorize
from .._language import container, condition, _get_compound
from .._language import constant, _plus_minus_average, _multiply_divide
from ..functions import rev, abs, min, max

# ------------------------------- TRIGONOMETRY ------------------------------- #

@vectorize
@memoize 
def degrees(token):
    """
    degrees(<input>)

        Converts incomming values from radians to degrees.
        (obj in radians * 57.29577951)

        Examples
        --------
        >>> degrees(radians(pCube1.rx)) # returns a network which converts rotationX to radians and back to degrees.
        >>> degrees(radians(pCube1.r))  # returns a network which converts [rx, ry, rz] to radians and back to degrees.
    """
    if isinstance(token, numbers.Real):
        return math.degrees(token)

    return _multiply_divide(token, (180./math.pi), operation=1, name='degrees1')

@vectorize
@memoize   
def radians(token):
    """ 
    radians(<input>)

        Converts incomming values from degrees to radians.
        (input in degrees * 0.017453292)

        Examples
        --------
        >>> radians(pCube1.rx) # returns a network which converts rotationX to radians.
        >>> radians(pCube1.r)  # returns a network which converts [rx, ry, rz] to radians.
    """
    if isinstance(token, numbers.Real):
        return math.radians(token)

    return _multiply_divide(token, (math.pi/180.), operation=1, name='radians1')

@vectorize
@memoize
def sin(token):
    """
    sin(<input>)

        Creates a sine function (in radians).

        Examples
        --------
        >>> sin(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
        >>> sin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
    """
    if isinstance(token, numbers.Real):
        return math.sin(token)
    

    with container('sin1'):
        token = container.publish_input(token, 'input')
        
        results = []
        for target in _get_compound(token):
            node = container.createNode('eulerToQuat')
            node.inputRotateX << target*(360./math.pi)
            results.append(node.outputQuatX)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')
        

@vectorize
@memoize
def sind(token):
    """ 
    sind(<input>)

        Creates a sine function (in degrees).

        Examples
        --------
        >>> sind(pCube1.tx) # returns a network which passes pCube1's translateX into a sine function.
        >>> sind(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a sine functions.
    """  
    if isinstance(token, numbers.Real):
        return math.sin(math.radians(token))


    with container('sind1'):
        token = container.publish_input(token, 'input')
        
        results = []
        for target in _get_compound(token):
            node = container.createNode('eulerToQuat')
            node.inputRotateX << target*2
            results.append(node.outputQuatX)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')

@vectorize
@memoize
def cos(token):
    """ 
    cos(<input>)

        Creates a cosine function (in radians).

        Examples
        --------
        >>> cos(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
        >>> cos(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
    """   
    if isinstance(token, numbers.Real):
        return math.cos(token)


    with container('cos1'):
        token = container.publish_input(token, 'input')
        
        results = []
        for target in _get_compound(token):
            node = container.createNode('eulerToQuat')
            node.inputRotateX << target*(360./math.pi)
            results.append(node.outputQuatW)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')

@vectorize
@memoize
def cosd(token):
    """ 
    cosd(<input>)

        Creates a cosine function (in degrees).

        Examples
        --------
        >>> cosd(pCube1.tx) # returns a network which passes pCube1's translateX into a cosine function.
        >>> cosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a cosine functions.
    """   
    if isinstance(token, numbers.Real):
        return math.cos(math.radians(token))


    with container('cosd1'):
        token = container.publish_input(token, 'input')
        
        results = []
        for target in _get_compound(token):
            node = container.createNode('eulerToQuat')
            node.inputRotateX << target*2
            results.append(node.outputQuatW)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
            
        return container.publish_output(output, 'output')
    

@vectorize
@memoize 
def tan(token):
    """ 
    tan(<input>)

        Returns a tan function (in radians).

        Examples
        --------
        >>> tan(pCube1.tx) # returns a network which passes pCube1's translateX into a tan approximation function.
        >>> tan(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan approximation functions.
    """  
    if isinstance(token, numbers.Real):
        return math.tan(token)

    with container('tan1'):
        token = container.publish_input(token, 'input')
        
        _sin = sin(token)
        _cos = cos(token)
        _div = _sin/_cos

        # quiet div by zero and return infinity
        _div.operation = condition(_cos==0, 0, 2)

        output = condition(_cos==0, float('inf'), _div)
        return container.publish_output(output, 'output')


@vectorize
@memoize   
def tand(token):
    """ 
    tan(<input>)

        Returns a tan function (in degrees).

        Examples
        --------
        >>> tan(pCube1.tx) # returns a network which passes pCube1's translateX into a tan function.
        >>> tan(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into a tan functions.
    """
    if isinstance(token, numbers.Real):
        return math.tan(math.radians(token))

    with container('tand1'):
        token = container.publish_input(token, 'input')
        
        _sind = sind(token)
        _cosd = cosd(token)
        _div = _sind/_cosd

        # quiet div by zero and return infinity
        _div.operation = condition(_cosd==0, 0, 2)

        output = condition(_cosd==0, float('inf'), _div)
        return container.publish_output(output, 'output')


@vectorize
@memoize
def asind(token):
    """ 
    asin(<input>)

        Calculates an arc sine function (in degrees).

        Examples
        --------
        >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine function.
        >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine  functions.
    """ 
    if isinstance(token, numbers.Real):
        return math.degrees(math.asin(token))
    

    with container('asind1'):
        token = container.publish_input(token, 'input')
        
        results = []

        for target in _get_compound(token):
            node = container.createNode('angleBetween')
            node.vector1 << 0
            node.vector2 << 0

            adj = rev(target*target)**0.5 
            node.vector1X << adj
            node.vector1Y << target
            node.vector2X << condition(abs(target)==1, 1, adj)

            result = condition(target < 0, -node.angle, node.angle)

            results.append(result)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')

@vectorize
@memoize    
def asin(token):
    """ 
    asin(<input>)

        Calculates an arc sine function (in radians).

        Examples
        --------
        >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc sine approximation function.
        >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc sine approximation functions.
    """
    if isinstance(token, numbers.Real):
        return math.asin(token)

    with container('asin1'):
        token = container.publish_input(token, 'input')
        output = radians(asind(token))
        
        return container.publish_output(output, 'output')


@vectorize
@memoize    
def acosd(token):
    """ 
    acosd(<input>)

        Calculates an arc cosine function (in degrees).

        Examples
        --------
        >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine function.
        >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine functions.
    """
    if isinstance(token, numbers.Real):
        return math.degrees(math.acos(token))


    with container('acosd1'):
        token = container.publish_input(token, 'input')
        
        results = []

        for target in _get_compound(token):
            node = container.createNode('angleBetween')
            node.vector1 << 0
            node.vector2 << 0

            adj = rev(target*target)**0.5 
            node.vector1X << target
            node.vector1Y << adj
            node.vector2X << condition(abs(target)==1, 1, adj)

            result = condition(target < 0, -node.angle, node.angle)

            results.append(result)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')


@vectorize
@memoize
def acos(token):
    """ 
    acos(<input>)

        Calculates an arc cosine function (in radians).

        Examples
        --------
        >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc cosine approximation function.
        >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc cosine approximation functions.
    """
    if isinstance(token, numbers.Real):
        return math.acos(token)

    with container('acos1'):
        token = container.publish_input(token, 'input')
        output = radians(acosd(token))
        
        return container.publish_output(output, 'output')
    

@vectorize
@memoize
def atand(token):
    """ 
    atand(<input>)

        Calculates an arc tan function (in degrees).

        Examples
        --------
        >>> acosd(pCube1.tx) # returns a network which passes pCube1's translateX into an arc tan function.
        >>> acosd(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc tan functions.
    """
    if isinstance(token, numbers.Real):
        return math.degrees(math.atan(token))


    with container('atand1'):
        token = container.publish_input(token, 'input')
        
        results = []

        for target in _get_compound(token):
            node = container.createNode('angleBetween')
            node.vector1 << [1,0,0]
            node.vector2 << [1,0,0]

            node.vector1Y << target            
            result = condition(target < 0, -node.angle, node.angle)

            results.append(result)

        if len(results) > 1:
            output = constant(results)
        else:
            output = results[0]
        
        return container.publish_output(output, 'output')

@vectorize
@memoize  
def atan(token):
    """ 
    atan(<input>)

        Calculates an arc tan function (in radians).

        Examples
        --------
        >>> atan(pCube1.tx)
        >>> atan(pCube1.t)
    """

    if isinstance(token, numbers.Real):
        return math.atan(token)

    with container('atan1'):
        token  = container.publish_input(token, 'input')
        output = radians(atand(token))
        return container.publish_output(output, 'output')

@vectorize
@memoize    
def atan2(y,x):
    if all([isinstance(x, numbers.Real) for x in [y, x]]):
        return math.atan2(y, x)

    with container('atan2'):
        y = container.publish_input(y, 'y')
        x = container.publish_input(x, 'x')
        
        div = y/x
        div.operation << condition(x==0, 0, 2) # quiet div by zero
        div = atan(div)
        
        out = condition( (x>0)           ,  div,           0)
        out = condition(((x<0)  & (y>=0)),  div+math.pi, out)
        out = condition(((x<0)  & (y<0)) ,  div-math.pi, out)
        out = condition(((x==0) & (y>0)) ,  math.pi/2,   out)
        out = condition(((x==0) & (y<0)) , -math.pi/2,   out)
        
        return container.publish_output(out, 'output')
        

@vectorize      
@memoize  
def atan2d(y, x):
    """ 
    atan2d(<y>, <x>)

        Approximates the principal value of the arc tangent of y/x (in degrees).

        Examples
        --------
        >>> asin(pCube1.tx) # returns a network which passes pCube1's translateX into an arc tan approximation function.
        >>> asin(pCube1.t)  # returns a network which passes pCube1's [tx, ty, tz] into an arc tan approximation functions.
    """
    with container('atan2d_1'):
        y = container.publish_input(y, 'y')
        x = container.publish_input(x, 'x')
        
        out = atan2(radians(y), radians(x))
        
        return container.publish_output(out, 'output')