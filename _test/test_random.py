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


from .._language import Node
from ..random import random, randint, uniform



class TestRandom(unittest.TestCase):

    def setUp(self):
        pass


    def testRandom(self):

        mc.file(new=True, f=True)
        mc.currentTime(0)

        # decare a node with a random plug
        node = Node('persp')
        node.tx << random()
        
        # verify the value changes
        default = node.tx >> None
        mc.currentTime(1)
        updated = node.tx >> None
        self.assertEqual(updated != default, True)


    def testTrigger(self):

        # check trigger mechanism
        mc.file(new=True, f=True)
        mc.currentTime(0)
        
        # decare a node with a random plug
        node = Node('persp')
        node.ty << random(trigger=node.tx)
        default = node.ty >> None
        
        # trigger did not change
        mc.currentTime(1)
        updated1 = node.ty >> None
        self.assertEqual(updated1 == default, True)
        
        # trigger updated
        node.tx << 5
        updated2 = node.ty >> None
        self.assertEqual(updated2 != default, True)
        
        
        
        
    def testRandint(self):

        mc.file(new=True, f=True)
        mc.currentTime(0)

        # decare a node with a randint plug
        node = Node('persp')
        node.tx << randint(0, 99999)

        # verify the value changes
        default = node.tx >> None
        mc.currentTime(1)
    
        updated = node.tx >> None
        self.assertEqual(updated != default, True)
        
        
        # verify changed is an integer
        test0 = default % 1 == 0 # init value is int
        test1 = default != updated
        self.assertEqual(test0 and test1, True)
        
        
        
    def testUniform(self):

        mc.file(new=True, f=True)
        mc.currentTime(0)

        # decare a node with a uniform plug
        node = Node('persp')
        node.tx << uniform(0, 99999)

        # verify the value changes
        default = node.tx >> None
        mc.currentTime(1)
    
        updated = node.tx >> None
        self.assertEqual(updated != default, True)
        
        
        # verify changed is a float
        test0 = default % 1 > 0 # init value is int
        test1 = default != updated
        self.assertEqual(test0 and test1, True)           