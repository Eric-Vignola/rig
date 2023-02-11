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

import unittest

try:
    import maya.standalone
    import maya.cmds as mc
    import maya.api.OpenMaya as om
    maya.standalone.initialize()

except:
    pass

from ..commands import polyCube
from ..interpolate import lerp, slerp, elerp
from ..interpolate import transform, sequence



class TestInterpolate(unittest.TestCase):

    def setUp(self):
        pass


    def testElerp(self):
        mc.file(new=True, f=True)
        obj1 = polyCube()[0]
        obj2 = polyCube()[0]
        obj3 = polyCube()[0]
        
        obj1.s << 10 # same as [10,10,10]
        obj1.t << [-5, 0, 0]
        obj2.t << [ 5, 0, 0]
        
        # setup the elerp interpolator
        interp = elerp(obj1.s, obj2.s)
        interp.weight << 0.25
        obj3.s << interp
        
        # test value
        expected = 10 ** (0.75) * 1**0.25
        obtained = obj3.sx >> None

        self.assertEqual(abs(expected-obtained) < 0.000001, True)

