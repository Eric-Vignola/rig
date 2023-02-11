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


from .._language import memoize, vectorize, container
from .._language import _euler_to_quaternion, _quaternion_to_euler
from ..matrix import compose, decompose


@vectorize
@memoize
def toMatrix(token, rotate_order=None):
    """ 
    toMatrix(<input>, rotate_order=<None>)

        Turns an euler angle into a transform matrix.

        Examples
        --------
        >>> toMatrix(pCube1.r, rotate_order=pCube1.ro)
    """
    return compose(rotate=token, rotate_order=rotate_order)

@vectorize
@memoize
def toQuaternion(token, rotate_order=None):
    """ 
    toQuaternion(<input>, rotate_order=<None>)

        Turns an euler angle into a quaternion.

        Examples
        --------
        >>> toQuaternion(pCube1.r, rotate_order=pCube1.ro)
    """
    #node = container.createNode('eulerToQuat')
    #node.inputRotate << token
    #node.inputRotateOrder << rotate_order
    #return node.outputQuat
    return _euler_to_quaternion(token, rotate_order=rotate_order)


@vectorize
@memoize
def toEuler(token, rotate_order0=None, rotate_order1=None):
    """ 
    toEuler(<input>, rotate_order0=<None>, rotate_order1=<None>)

        Converts an euler angle into a different rotate order.

        Examples
        --------
        >>> toEuler(pCube1.r, rotate_order0=0, rotate_order1=3)
    """ 
    if rotate_order0 is None or rotate_order1 is None:
        raise Exception('must specify a two rotate orders.')
    
    node = toQuaternion(token, rotate_order=rotate_order0)
    return _quaternion_to_euler(node, rotate_order=rotate_order1)