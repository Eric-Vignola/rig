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


from ..trigonometry import sin, cos, tan, asin, acos, atan
from .._language import constant

import random


class TestNode(unittest.TestCase):

    def setUp(self):
        pass      


    def testTrig(self):
        
        def test_trig(func, count=500):
    
            #mc.file(new=True, f=True)
            
            # create a constant and pass it into the desired function
            const = constant(0)
            test  = func(const)
            random_val = 0
    
            for i in range(count):
                while random_val in [0, -1.0, 1.0]:
                    random_val = 1-random.random()*2
    
                # set the constant
                const << random_val
    
                # do the test
                test_val = test >> None
                math_val = func(random_val)
                error = abs(math_val-test_val)
    
                # assert if error < 0.00001 
                self.assertEqual(error < 0.00001, True)
                
                
        
        test_trig(sin)
        test_trig(cos)
        test_trig(tan) 
        test_trig(asin)
        test_trig(acos)
        test_trig(atan)





