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

try:
    # python 2.7
    import __builtin__ as builtins
except:
    # python 3
    import builtins
    
import maya.cmds as mc
import numbers
import math


from ._language import container, memoize, condition, List, Node, vectorize
from ._language import _is_sequence, _is_node, _is_compound, _is_matrix
from ._language import _plus_minus_average, _multiply_divide,  _constant


def _sequence_to_node(obj):
    """
    patch to catch situation where a function is given a
    sequence that needs to be used inside an expression.
    """
    if _is_sequence(obj):
        return _constant(obj)

    return obj



# TODO: should this be memoized?
@memoize
def frame():
    """ 
    frame()

        Creates an animCurveTL node with a linear curve with tangents 
        set to infinity, so the curve data maps 1:1 with timeline

        Examples
        --------
        >>> frame()
    """    

    node = container.createNode('animCurveTL', name='frame1')
    mc.setKeyframe(str(node), t=0, v=0.)
    mc.setKeyframe(str(node), t=1, v=1.)
    mc.keyTangent(str(node), e=True, itt='linear', ott='linear')
    node.preInfinity << 4
    node.postInfinity << 4

    return node.o



# ----------------------------- COMMON COMMANDS ------------------------------ #
@vectorize
@memoize
def clamp(token, min_value, max_value):
    """ 
    clamp(<token>, <min_value>, <max_value>)

        Clamps values between a min and a max.

        Examples
        --------
        >>> clamp(pCube1.ty, 0, pCube2.ty) # clamps pCube1.ty value between 0 and pCube2.ty
        >>> clamp(pCube1.t, -1, 1) # clamps [tx, ty, tz] of pCube1 between -1 and 1
    """
    with container('clamp1'):
        token     = container.publish_input(token,     'input')
        min_value = container.publish_input(min_value, 'min')
        max_value = container.publish_input(max_value, 'max')
        
        less_than = condition(token < min_value, min_value, token)
        result    = condition(token > max_value, max_value, less_than)
        
        return container.publish_output(result, 'output')
    

@vectorize
@memoize
def abs(token):
    """ 
    abs(<token>)

        Outputs the absolute value an input.

        Examples
        --------
        >>> abs(pCube1.t)
        >>> abs(pCube1.tx)
    """
    if isinstance(token, numbers.Real):
        return builtins.abs(token)

    with container('abs1'):
        token  = container.publish_input(token, 'input')
        result = condition(token < 0, -token, token)
        return container.publish_output(result, 'output')

@vectorize
@memoize
def int(token):
    """ 
    int(<token>)

        Returns a proper int(float) operation.

        Examples
        --------
        >>> int(pCube1.t)
        >>> int(pCube1.tx)
    """
    if isinstance(token, numbers.Real):
        return builtins.int(token)    

    with container('int1'):
        token  = container.publish_input(token, 'input')
        result = _constant(condition(token > 0, token-0.4999999, token+0.4999999) , dtype='long')
        return container.publish_output(result, 'output')


@vectorize
@memoize
def round(token, digits):
    """ 
    round(<token>, <digits>)

        Rounds a riven value by n digits

        Examples
        --------
        >>> round(pCube1.t,1)

    """
    if all([isinstance(x, numbers.Real) for x in [token, digits]]):
        return builtins.round(token, digits)

    with container('round1'):
        token  = container.publish_input(token, 'input')
        scale  = 10 ** -digits
        result = _constant(token/scale, dtype='long') * scale
        return container.publish_output(result, 'output')


@vectorize
@memoize
def floor(token):
    """ 
    floor(<token>)

        Returns the floor value of the input.

        Examples
        --------
        >>> floor(pCube1.t)
        >>> floor(pCube1.tx)
    """
    if isinstance(token, numbers.Real):
        return math.floor(token) 

    with container('floor1'):
        token  = container.publish_input(token, 'input')
        result = _constant(token - 0.4999999, dtype='long')
        return container.publish_output(result, 'output')


@vectorize
@memoize    
def ceil(token):
    """ 
    ceil(<token>)

        Returns the ceil value of the input.

        Examples
        --------
        >>> ceil(pCube1.t)
        >>> ceil(pCube1.tx)
    """
    if isinstance(token, numbers.Real):
        return math.floor(token) 

    with container('ceil1'):
        token  = container.publish_input(token, 'input')
        result = _constant(token + 0.4999999, dtype='long')
        return container.publish_output(result, 'output')



@memoize
def sum(tokens):
    """
    sum([token, token, ...])

        Single node operation to sum all items in the list.

        Examples
        --------
        >>> sum(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """  
    if not any([_is_node(x) for x in tokens]):
        return builtins.sum(tokens)

    return _plus_minus_average(*tokens, operation=1, name='sum1')




@memoize
def avg(tokens):
    """
    avg([token, token, ...])

        Single node operation to avg all items in the list.

        Examples
        --------
        >>> avg(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """   
    if all([isinstance(x, numbers.Real) for x in tokens]):
        return builtins.sum(tokens)/float(len(tokens))

    return _plus_minus_average(*tokens, operation=3, name='avg1') 



@memoize
def max(tokens):
    """ 
    max([token, token, ...])

        Returns the highest value in the list of inputs.

        Examples
        --------
        >>> max(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """ 
    if not any([_is_node(x) for x in tokens]):
        return builtins.max(tokens)    


    if len(tokens) < 2:
        raise Exception('max() requires minimum 2 inputs, given: {}'.format(tokens))

    with container('max1'):
        tokens = container.publish_input(tokens, 'input')
        
        output = tokens[0]
        for obj in tokens[1:]:
            output = condition(output > obj, output, obj)    

        return container.publish_output(output, 'output')  


@memoize    
def min(tokens):
    """ 
    min([token, token, ...])

        Returns the highest value in the list of inputs.

        Examples
        --------
        >>> min(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """ 
    if not any([_is_node(x) for x in tokens]):
        return builtins.min(tokens)  

    if len(tokens) < 2:
        raise Exception('min() requires minimum 2 inputs, given: {}'.format(tokens))

    with container('min1'):
        tokens = container.publish_input(tokens, 'input')
        
        output = tokens[0]
        for obj in tokens[1:]:
            output = condition(output < obj, output, obj)    

        return container.publish_output(output, 'output')      


@vectorize
@memoize
def exp(token):
    """ 
    exp(<token>)

        Return e raised to the power x, 
        where e is the base of natural logarithms (2.718281...)

        Examples
        --------
        >>> exp(pCube1.tx)
    """

    if isinstance(token, numbers.Real):
        return math.exp(token) 

    return _multiply_divide(math.e, token, operation=3, name='exp1')


@vectorize
@memoize
def sign(token):
    """ 
    sign(<token>)

        Returns -1 for values < 0. +1 for values >= 0.

        Examples
        --------
        >>> sign(pCube1.t)
        >>> sign(pCube1.tx)
    """
    if isinstance(token, numbers.Real):
        return bool(token >= 0) - bool(token < 0)

    with container('sign1'):
        token  = container.publish_input(token, 'input')
        output = condition(token<0, -1, 1)
        return container.publish_output(output, 'output')  


@vectorize
@memoize
def sqrt(token):
    """ 
    sqrt(<token>)

        Returns the square root of a value. (same as doing x ** 0.5)

        Examples
        --------
        >>> sqrt(pCube1.t)
        >>> sqrt(pCube1.tx)

    """

    if isinstance(token, numbers.Real):
        return token ** 0.5

    return _multiply_divide(token, 0.5, operation=3, name='sqrt1')


@vectorize
@memoize
def pow(base, exponent):
    """ 
    sqrt(<base>, <exponent>)

        Returns the the power of base to the exponent. (same as doing x ** y)

        Examples
        --------
        >>> pow(pCube1.t, 5)
        >>> pow(pCube1.tx, pCube2.ty)

    """

    if all([isinstance(x, numbers.Real) for x in [base, exponent]]):
        return base ** exponent
    
    return _multiply_divide(base, exponent, operation=3, name='pow1')


@vectorize
@memoize    
def mag(token):
    """ 
    mag(<input>)

        Returns the magnitude of an input.

        Examples
        --------
        >>> mag(pCube1.t)  # Computes the magnitude of [tx, ty, tz].
        >>> mag(pCube1.wm) # Computes the magnitude of the 4x4 matrix's position component.
    """
    if _is_sequence(token):
        if all([isinstance(x, numbers.Real) for x in token]):
            return builtins.sum([x**2 for x in token])**0.5


    node = container.createNode('distanceBetween', name='mag1')
    
    if _is_matrix(token):
        node.inMatrix1 << token
    else:
        node.point1 << token
        
    return node.distance



@memoize
def choice(tokens, selector=None):
    """ 
    choice([token, token, token, ...], selector=None)

        Examples
        --------
        >>> choice([pCube2.wm, pCube3.wm], selector=pCube1.someEnum)
        >>> choice([None, pCube2.wm, pCube3.wm]) # leaves selector unplugged.
    
    """
    node = container.createNode('choice')

    if not selector is None:
        node.selector << selector

    for obj in tokens:
        obj = _sequence_to_node(obj) # in case given [x, Node().attr, z]
        node.input << obj

    return node.output


@vectorize
@memoize
def rev(token):
    """ 
    rev(<token>, selector=None)

        Reverses a value (1-x)

        Examples
        --------
        >>> rev(pCube1.v)
    """
    if isinstance(token, numbers.Real):
        return 1.0 - token

    node = container.createNode('reverse')  

    if _is_compound(token):
        node.input << token
        return node.output

    else:
        node.inputX << token
        return node.outputX



# -------------------------- ALGEBRA MATH FUNCTIONS -------------------------- #
@vectorize
@memoize
def dot(input1, input2, normalize=False):
    """ 
    dot(<input1>, <input2>, normalize=<True/False/input3>)

            Uses a vectorProduct to do a dot product between two vector inputs.

            Examples
            --------
            >>> dot(pCube1.t, pCube2.t)
    """
    #sum( [a[i][0]*b[i] for i in range(len(b))] ) 
    # TODO: TEST FOR MATRICES AND DO MATRIX DOT PRODUCT?
    node = container.createNode('vectorProduct', name='dot1')
    node.operation << 1
    node.normalizeOutput << normalize

    node.input1 << input1
    node.input2 << input2

    return node.output


@vectorize
@memoize
def cross(input1, input2, normalize=False):
    """ 
    dot(<input1>, <input2>, normalize=<True/False/input3>)

            Uses a vectorProduct to do a cross product between two vector inputs.

            Examples
            --------
            >>> dot(pCube1.t, pCube2.t)
    """
    #c = [a[1]*b[2] - a[2]*b[1],
         #a[2]*b[0] - a[0]*b[2],
         #a[0]*b[1] - a[1]*b[0]]

    node = container.createNode('vectorProduct', name='cross1')
    node.operation << 2
    node.normalizeOutput << normalize

    node.input1 << input1
    node.input2 << input2

    return node.output


@vectorize
@memoize
def unit(token):

    """ 
    unit(<input>)

            Creates a network that yields a unit vector.

            Examples
            --------
            >>> unit(pCube1.t)
    """
    with container('unit1'):
        token  = container.publish_input(token, 'input')
        
        _mag = mag(token)
        _div = token/_mag
        _div.operation << condition(_mag==0, 0, 2) # quiet div by zero

        return container.publish_output(_div, 'output')


@vectorize
@memoize
def dist(input1, input2):
    """ 
    dist(<input1>, <input2>)

        Creates a distanceBetween node to find distance between points or matrices.

        Examples
        --------
        >>> dist(pCube1.t,  pCube2.t)
        >>> dist(pCube1.wm, pCube2.wm)
        >>> dist(pCube1.t,  pCube2.wm)
    """
    node = container.createNode('distanceBetween', name='dist1')
    
    # process input1
    try:
        if _is_matrix(input1):
            node.inMatrix1 << input1
        else:
            node.point1 << input1

    # in case input1 is a list        
    except:
        node.point1 << input1

    # process input2
    try:
        if _is_matrix(input2):
            node.inMatrix2 << input2
        else:
            node.point2 << input2 

    # in case input2 is a list     
    except:
        node.point2 << input2     


    return node.distance



@vectorize
@memoize
def exponentRange(token, min_val=0, max_val=1):
    
    with container('exponentRange1'):
        token   = container.publish_input(token, 'input')
        min_val = container.publish_input(min_val, 'min')
        max_val = container.publish_input(max_val, 'max')
        
        delta  = token - min_val
        spread = max_val - min_val
        test   = delta/spread
        ratio  = 1 - exp(-1 * abs(test))
        
        output = min_val + (ratio * spread * sign(test))
        return container.publish_output(output, 'output')

    
    

@memoize   
def searchSortedLeft(tokens, query, return_index=False):
    with container('searchSortedLeft1'):
        tokens = container.publish_input(tokens, 'input')
        query  = container.publish_input(query,  'query')
        
        if not return_index:
            result = tokens[0]
            for t in tokens[1:]:
                result = condition(t <= query, t, result)
                
        else:
            result = 0
            i = 1
            for t in tokens[1:]:
                result = condition(t <= query, i, result)
                i+=1
                
        return container.publish_output(result, 'output')



@memoize
def searchSortedRight(tokens, query, return_index=False):
    
    with container('searchSortedRight1'):
        tokens = container.publish_input(tokens, 'input')
        query  = container.publish_input(query,  'query')
        
        if not return_index:
            result = tokens[-1]
            for t in tokens[:-1][::-1]:
                result = condition(t >= query, t, result)
                
        else:
            result = len(tokens)-1
            i = len(tokens)-2
            for t in tokens[:-1][::-1]:
                result = condition(t >= query, i, result)
                i-=1

        return container.publish_output(result, 'output')
    
    
@memoize   
def diff(tokens):
    
    with container('diff1'):
        tokens = container.publish_input(tokens, 'input')
    
        results = List()
        for a,b in zip(tokens[:-1], tokens[1:]):
            results.append(b-a)
            
        return container.publish_output(results, 'output')
    

@memoize  
def cumsum(tokens):
    with container('cumsum1'):
        tokens = container.publish_input(tokens, 'input')
        
        results = List([(0+tokens[0])])
        for obj in tokens[1:]:
            results.append(obj + results[-1])
            
        return container.publish_output(results, 'output')


