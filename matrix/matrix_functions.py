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


from .._generators import sequences

from .._language import container, memoize, vectorize
from .._language import _get_compound, _is_node, _is_matrix
from .._language import _decompose_matrix, _compose_matrix, _matrix_multiply
from .._language import _matrix_add, _matrix_inverse

from ..functions import rev

import maya.cmds as mc
MAYA_VERSION = int(mc.about(version=True))



# ---------------------------------- MATRIX ---------------------------------- #'
@vectorize
@memoize
def decompose(token, rotate_order=None):
    """ 
    decompose(<input>, <rotate_order=None>)

        Extracts the position component of a matrix by default.
        Other components can be extracted via variable attribute override.
        Available:
            - .outputTranslate << default
            - .outputRotate
            - .outputScale
            - .outputShear
            - .outputQuat
            - .inputrotate_order


        Examples
        --------
        >>> decompose(pCube1.wm)
    """

    #node = container.createNode('decomposeMatrix')
    #node.inputMatrix << token
    #node.inputRotateOrder << rotate_order
    #return node.outputTranslate
    return _decompose_matrix(token, rotate_order=rotate_order)

@vectorize
@memoize
def inverse(token):
    """ 
    inverse(<input>)

        Returns the inverse matrix.

        Examples
        --------
        >>> inverse(pCube1.wm)
    """
    #node = container.createNode('inverseMatrix')
    #node.inputMatrix << token
    #return node.outputMatrix
    return _matrix_inverse(token)

@vectorize
@memoize
def transpose(token):
    """ 
    transpose(<input>)

        Returns the transposed matrix.

        Examples
        --------
        >>> transpose(pCube1.wm)
    """

    node = container.createNode('transposeMatrix')
    node.inputMatrix << token
    return node.outputMatrix  

@vectorize
@memoize
def toQuaternion(token):
    """ 
    toQuaternion(<input>)

        Converts a matrix into a quaternion.

        Examples
        --------
        >>> toQuaternion(pCube1.wm)
    """
    return decompose(token).outputQuat


@vectorize
@memoize
def toEuler(token, rotate_order=None):
    """ 
    toEuler(<input>)

        Converts a matrix into euler angles.

        Examples
        --------
        >>> toQuaternion(pCube1.wm)
    """
    return decompose(token, rotate_order=rotate_order).outputRotate


@vectorize
@memoize
def compose(scale=None, rotate=None, translate=None, shear=None, rotate_order=None):
    """ 
    compose(<translate>, <rotate/quaternion>, <scale>, <shear> <rotate_order>)

        Constructs a matrix from a list of up to 5 inputs.

        Examples
        --------
        >>> compose(translate=pCube1.t, eulerToQuat(pCube1.r), scale=pCube1.s) # inputQuaternion will be used
        >>> compose(position=pCube3.t) # identity matrix with just a position
    """

    #node = container.createNode('composeMatrix')

    ## plug translate
    #if not translate is None:
        #node.inputTranslate << translate

    ## plug scale
    #if not scale is None:
        #node.inputScale << scale

    ## plug shear 
    #if not shear is None:
        #node.inputShear << shear  

    ## plug rotate
    #if not rotate is None:

        ## is this a quaternion?
        #quat_test = _get_compound(rotate)
        #if len(quat_test) == 4:
            #node.useEulerRotation << 0
            #node.inputQuat << rotate

        ## this is euler angle
        #else:
            #node.inputRotate << rotate
            #node.inputrotate_order << rotate_order

    #return node.outputMatrix
    return _compose_matrix(scale=scale, rotate=rotate, translate=translate, shear=shear, rotate_order=rotate_order)


@vectorize
@memoize
def multiply(*tokens, **kargs):
    """ 
    multiply(<input>, <input>, ..., local=False)

        Multiplies 2 or more matrices together.
        If the first token is a compound and the second a matrix,
        do a pointMatrix multiplication. If "local" kwyword argument
        will plug into the node's vectorMultiply attribute

        Examples
        --------
        >>> pCube1.wm * pCube2.wm
        >>> pCube1.t * pCube2.wm
        >>> multiply(pCube1.wm, pCube2.wm, pCube3.wm)
        >>> multiply(pCube1.t, pCube2.wm)
        >>> multiply([1,0,0], pCube2.wm, local=True)
    """

    #local = ([kargs.pop(x) for x in kargs.keys() if x in ['local']] or [None] )[-1]
    #if kargs:      
        #raise Exception('Unsupported keyword args: {}'.format(kargs.keys()))

    ## are we doing point matrix mult?
    #if len(tokens) == 2:
        #count = 0
        #matrix_index = 0
        #vector_index = 0

        #for i, obj in enumerate(tokens):
            
            #if _is_matrix(obj):
                #matrix_index = i
                #count +=1
            #else:
                #vector_index = i
            

        #if count == 1:
            #node = container.createNode('pointMatrixMult')
            #node.inMatrix << tokens[matrix_index]
            #node.inPoint << tokens[vector_index]
            #node.vectorMultiply << local

            #return node.output


    ## do a straight matrix sum operation
    #node = container.createNode('multMatrix')
    #for obj in tokens:
        #node.matrixIn << obj
    #return node.matrixSum

    return _matrix_multiply(*tokens, **kargs)

@vectorize
@memoize
def add(*tokens, **kargs):
    """ 
    add(<input>, <input>, ..., weights=[w0, w1, ...])

        Adds matrices together with optional weights.

        Examples
        --------
        >>> pCube1.wm + pCube2.wm
        >>> add([pCube1.wm, pCube2.wm, pCube3.wm, ...])
        >>> add([pCube1.wm, pCube2.wm], weights=[pCube1.ty, rev(pCube1.ty)])

    """
    #weights = ([kargs.pop(x) for x in kargs.keys() if x in ['weights']] or [None] )[-1]
    #if kargs:      
        #raise Exception('Unsupported keyword args: {}'%kargs.keys()) 


    #if weights is None:
        #node = container.createNode('addMatrix')
        #for obj in tokens:
            #node.matrixIn << obj 

    #else:

        #if isinstance(weights, numbers.Real) or _is_node(weights):
            #weights = [weights]

        #node  = container.createNode('wtAddMatrix')
        #index = 0
        #for obj, w in sequences(tokens, weights):
            #node.wtMatrix[index].matrixIn << obj   
            #node.wtMatrix[index].weightIn << w        

            #index+=1

    #return node.matrixSum 
    return _matrix_add(*tokens, **kargs)


@vectorize
@memoize
def interpolate(input1, input2, weight=0.5):

    with container('matrixInterpolate1'):
        node  = container.createNode('wtAddMatrix')
        node.wtMatrix[0].matrixIn << input1
        node.wtMatrix[0].weightIn << rev(weight)

        node.wtMatrix[1].matrixIn << input2
        node.wtMatrix[1].weightIn << weight

        return node.matrixSum

@vectorize
@memoize
def matrix(x=None, y=None, z=None, position=None):
    """ 
    matrix(x=None, y=None, z=None, position=None)

        Constructs a matrix from up to 4 vectors (X,Y,Z,position)
        Without any inputs this is simply an idntity matrix

        Examples
        --------
        >>> matrix(x=pCube1.t, y=pCube2.t, z=pCube3.t)
        >>> matrix(pCube1.t, pCube2.t, pCube3.t, pCube4.t)
    """


    node = container.createNode('fourByFourMatrix')
    if not x is None:
        for plug, comp in sequences([node.in00, node.in01, node.in02], _get_compound(x)):
            plug << comp

    if not y is None:
        for plug, comp in sequences([node.in10, node.in11, node.in12], _get_compound(y)):
            plug << comp 

    if not z is None:
        for plug, comp in sequences([node.in20, node.in21, node.in22], _get_compound(z)):
            plug << comp    

    if not position is None:
        for plug, comp in sequences([node.in30, node.in31, node.in32], _get_compound(position)):
            plug << comp    


    return node.output


@vectorize
@memoize
def determinant(token):
    """ 
    determinant(x, y ,z)
        Returns the determinent of a rotation components of a 4x4 matrix

        Examples
        --------
        >>> determinant(pCube1.wm)
        """

    # new determinant node type in Maya 2024
    if MAYA_VERSION >= 2024:
        node = container.createNode('determinant')
        node.input << token
        return node.output
    
    
    # default to old method
    with container('matrixDeterminant1'):
        token = container.publish_input(token, 'input')
        
        X = multiply([1,0,0], token, local=True)
        Y = multiply([0,1,0], token, local=True)
        Z = multiply([0,0,1], token, local=True)
        
        X = _get_compound(X)
        Y = _get_compound(Y)
        Z = _get_compound(Z)

        output = X[0]*(Y[1]*Z[2] - Y[2]*Z[1]) - X[1]*(Y[0]*Z[2] - Y[2]*Z[0]) + X[2]*(Y[0]*Z[1] - Y[1]*Z[0])

        return container.publish_output(output, 'output')
