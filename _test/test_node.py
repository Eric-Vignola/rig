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


from .._language import Node, List, condition, vectorize
from ..attributes import Float, Vector



class TestNode(unittest.TestCase):

    def setUp(self):
        pass


    def testStr(self):

        mc.file(new=True, f=True)


        # --- plusMinusAverage --- #
        node = Node(mc.createNode('plusMinusAverage'))

        # test picking a random index
        self.assertEqual(str(node.input3D[3]), 'plusMinusAverage1.input3D[3]')        

        # test default index when no index given
        self.assertEqual(str(node.input3D), 'plusMinusAverage1.input3D[0]')

        # test slicing when nothing's been plugged or set
        self.assertEqual(list(node.input3D[:]), [])

        # plug a node
        mc.connectAttr('persp.t', str(node.input3D))

        # retest slicing
        self.assertEqual([str(x) for x in node.input3D[:]], ['plusMinusAverage1.input3D[0]'])

        # retest default index
        self.assertEqual(str(node.input3D), 'plusMinusAverage1.input3D[1]')

        # test forcing a range
        self.assertEqual([str(x) for x in node.input3D[12:14]], ['plusMinusAverage1.input3D[12]', 'plusMinusAverage1.input3D[13]'])

        # get compound attrs
        self.assertEqual(node.input3D[3].__data__.compound, ['input3Dx', 'input3Dy', 'input3Dz'])




        # --- transform --- #

        # test variable names
        node = Node('persp')
        self.assertEqual(str(node),    'persp')
        self.assertEqual(str(node.t),  'persp.translate')
        self.assertEqual(str(node.tx), 'persp.translateX')

        # get compound attrs
        self.assertEqual(node.tx.__data__.compound, None)
        self.assertEqual(node.t.__data__.compound,  ['translateX', 'translateY', 'translateZ'])


        ## slice compound
        #self.assertEqual([str(x) for x in node.t[:]],  ['persp.translate.translateX', 'persp.translate.translateY', 'persp.translate.translateZ'])
        #self.assertEqual([str(x) for x in node.t[:2]], ['persp.translate.translateX', 'persp.translate.translateY'])
        #self.assertEqual(str(node.t[2]), 'persp.translate.translateZ')


        # decare a node with an attribute
        node = Node('persp.t')
        self.assertEqual(str(node), 'persp.translate')


        # reset the attribute name 
        self.assertEqual(str(node._), 'persp')




        # --- control points --- #
        curve = mc.curve(d=1, p=[[0,0,0],[0,1,0],[0,2,0],[0,3,0]], k=[0, 1, 2, 3])
        node = Node(curve)
        self.assertEqual(str(node),              'curve1')
        self.assertEqual(str(node.cv),           'curveShape1.controlPoints[0]')
        self.assertEqual(str(node.cv[0]),        'curveShape1.controlPoints[0]')
        self.assertEqual(str(node.cv.zValue),    'curveShape1.controlPoints[0].zValue')
        self.assertEqual(str(node.cv[0].zValue), 'curveShape1.controlPoints[0].zValue')

        self.assertEqual(node.cv.zValue.__data__.compound,      None)
        self.assertEqual(node.cv.__data__.compound,             ['xValue', 'yValue', 'zValue'])             

        self.assertEqual(node.cv.zValue.__data__.point, False)
        self.assertEqual(node.cv.__data__.point,        True)           


        # --- slicing --- #
        self.assertEqual([str(x) for x in node.cv[:]],    ['curveShape1.controlPoints[0]',
                                                           'curveShape1.controlPoints[1]',
                                                           'curveShape1.controlPoints[2]',
                                                           'curveShape1.controlPoints[3]'])

        self.assertEqual([str(x) for x in node.cv[1:]],   ['curveShape1.controlPoints[1]',
                                                           'curveShape1.controlPoints[2]',
                                                           'curveShape1.controlPoints[3]']) 

        self.assertEqual([str(x) for x in node.cv[:-1]],  ['curveShape1.controlPoints[0]',
                                                           'curveShape1.controlPoints[1]',
                                                           'curveShape1.controlPoints[2]'])

        self.assertEqual([str(x) for x in node.cv[::-1]], ['curveShape1.controlPoints[3]',
                                                           'curveShape1.controlPoints[2]',
                                                           'curveShape1.controlPoints[1]',
                                                           'curveShape1.controlPoints[0]'])          

        self.assertEqual([str(x) for x in node.cv[::2]],  ['curveShape1.controlPoints[0]',
                                                           'curveShape1.controlPoints[2]'])         




    def testInject(self):
        # --- simple setAttr --- #  
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]     
        obj2 = Node(mc.polyCube()[0]).tx << 5

        self.assertEqual(mc.getAttr(str(obj1.t)), [(1.0, 2.0, 3.0)])
        self.assertEqual(mc.getAttr(str(obj2.tx)), 5.0)


        # --- simple connections --- #
        # one to one
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]     
        obj2 = Node(mc.polyCube()[0]).tx << 5        
        obj1.tx << obj2.tx
        self.assertEqual(mc.getAttr(str(obj1.tx)), 5.0)


        # one to compound
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]     
        obj2 = Node(mc.polyCube()[0]).tx << 5        
        obj1.t << obj2.tx
        self.assertEqual(mc.getAttr(str(obj1.t)), [(5.0, 5.0, 5.0)])


        # compound to one with inline assignment
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]   
        obj2 = Node(mc.polyCube()[0]).tx << 5    

        obj2.tx << obj1.t
        self.assertEqual(mc.getAttr(str(obj2.tx)), 1.0)


        # compound to compound with inline assignment
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]      
        obj2 = Node(mc.polyCube()[0]).tx << 5   

        obj2.t << obj1.t
        self.assertEqual(mc.getAttr(str(obj2.t)), [(1.0, 2.0, 3.0)])      


        # connect to a multi attr and automatically append the index
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  
        obj2 = Node(mc.polyCube()[0]).tx << 5
        add  = Node(mc.createNode('plusMinusAverage'))

        add.input3D << obj1.t
        add.input3D << [10, 20, 30]
        obj2.t << add.output3D

        self.assertEqual(mc.getAttr(str(obj2.t)), [(11.0, 22.0, 33.0)]) 


        # connect to a choice node and automatically append the index
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  
        obj2 = Node(mc.polyCube()[0]).t << [10, 20, 30]
        obj3 = Node(mc.polyCube()[0])
        choice = Node(mc.createNode('choice'))

        choice.input << obj1.t
        choice.input << obj2.t
        choice.selector << 1
        obj3.t << choice.output

        self.assertEqual(mc.getAttr(str(obj3.t)), [(10.0, 20.0, 30.0)]) 

        
        # Test disconnect attr
        connections = bool(mc.listConnections(str(obj3.t), s=True, d=False, p=True))
        self.assertEqual(connections, True)
        
        obj3.t << None
        connections = bool(mc.listConnections(str(obj3.t), s=True, d=False, p=True))
        self.assertEqual(connections, False)        
        

    def testArithmetic(self):

        # --- node to node addition + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  
        obj2 = Node(mc.polyCube()[0]).t << [10, 20, 30]

        # do it multiple times to test @memoize
        for i in range(5):
            result = obj1.t + obj2.t
            self.assertEqual(mc.getAttr(str(result)), [(11.0, 22.0, 33.0)])

        self.assertEqual(len(mc.ls(type='plusMinusAverage')), 1)



        # --- node to float addition + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  

        # do it multiple times to test @memoize
        for i in range(5):
            result = obj1.t + 10.0
            self.assertEqual(mc.getAttr(str(result)), [(11.0, 12.0, 13.0)])

        self.assertEqual(len(mc.ls(type='plusMinusAverage')), 1)



        # --- float to node addition + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  

        # do it multiple times to test @memoize
        for i in range(5):
            result = 10 + obj1.t
            self.assertEqual(mc.getAttr(str(result)), [(11.0, 12.0, 13.0)])

        # do it multiple times to test @memoize
        for i in range(5):
            result = 10.0 + obj1.t
            self.assertEqual(mc.getAttr(str(result)), [(11.0, 12.0, 13.0)])            

        self.assertEqual(len(mc.ls(type='plusMinusAverage')), 1)



        # --- node to float multiplication + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [1, 2, 3]  

        # do it multiple times to test @memoize
        for i in range(5):
            result = obj1.t * 10.0
            self.assertEqual(mc.getAttr(str(result)), [(10.0, 20.0, 30.0)])

        # do it multiple times to test @memoize
        for i in range(5):
            result = 10 * obj1.t
            self.assertEqual(mc.getAttr(str(result)), [(10.0, 20.0, 30.0)])        

        self.assertEqual(len(mc.ls(type='multiplyDivide')), 2)



        # --- node to float division + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [10, 20, 30]

        # do it multiple times to test @memoize
        for i in range(5):
            result = obj1.t / 10.0
            self.assertEqual(mc.getAttr(str(result)), [(1.0, 2.0, 3.0)])

        # do it multiple times to test @memoize
        for i in range(5):
            result = 10 / obj1.t
            self.assertEqual([round(x, 3) for x in mc.getAttr(str(result))[0]], [1.0, 0.5, round(10/30., 3)])        

        self.assertEqual(len(mc.ls(type='multiplyDivide')), 2)



        # --- floor div + memoization --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [10, 20, 30]  

        # do it multiple times to test @memoize
        for i in range(5):
            result = obj1.t // 2
            self.assertEqual(mc.getAttr(str(result)), [(5, 10, 15)])

        # do it multiple times to test @memoize
        for i in range(5):
            result = [100, 200, 300] // obj1.t
            self.assertEqual(mc.getAttr(str(result)), [(10, 10, 10)])        

        self.assertEqual(len(mc.ls(type='multiplyDivide')), 2)


        # --- modulo --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0]).t << [10, 20, 30]  

        result = obj1.t % 9
        self.assertEqual(mc.getAttr(str(result)), [(1, 2, 3)])


        # --- point matrix mult --- #
        mc.file(new=True, f=True)

        obj1 = Node(mc.polyCube()[0])
        obj1.r << [62, -52.212, 14]

        correct =  [round(x, 3) for x in (om.MVector(1,0,0) * om.MMatrix(mc.getAttr('pCube1.wm')))]

        # do it multiple times to test @memoize
        for i in range(5):
            result = (obj1.wm * [1,0,0])
            test = [round(x, 3) for x in mc.getAttr(str(result))[0]]

            self.assertEqual(test, correct)

        # memoize test
        self.assertEqual(len(mc.ls(type='pointMatrixMult')), 1)




    def testShorthand(self):

        # --- full transform --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0])
        obj2 = Node(mc.polyCube()[0])

        obj1.s << [0.5, 0.1, 0.7]
        obj1.r << [45.0, -23.2, 11.0]
        obj1.t << [23.0, 10.0, 14.0]


        # do it multiple times to test @memoize
        for i in range(5):
            obj2 << obj1.wm

            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.s))[0]], [0.5, 0.1, 0.7])
            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.r))[0]], [45.0, -23.2, 11.0])
            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.t))[0]], [23.0, 10.0, 14.0])

        self.assertEqual(len(mc.ls(type='decomposeMatrix')), 1)




        # --- partial transform --- #
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0])
        obj2 = Node(mc.polyCube()[0])

        obj1.s << [0.5, 0.1, 0.7]
        obj1.r << [45.0, -23.2, 11.0]
        obj1.t << [23.0, 10.0, 14.0]

        # do it multiple times to test @memoize
        for i in range(5):
            obj2.ry << obj1.wm  # only plug ry

            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.s))[0]], [1., 1., 1.])
            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.r))[0]], [0., -23.2, 0.])
            self.assertEqual([round(x, 3) for x in mc.getAttr(str(obj2.t))[0]], [0., 0., 0.])

        self.assertEqual(len(mc.ls(type='decomposeMatrix')), 1)





    def testList(self):
        mc.file(new=True, f=True)
        list0 = List([1,2,3,4])
        list1 = List([5,6,7,8])
        list0 + list1



    def testArguments(self):
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0])
        obj2 = Node(mc.polyCube()[0])

        obj1 << Float('test1')
        obj1 << Vector('test2')
        obj1 << Float('test3', multi=True)
        obj1 << Vector('test4', multi=True)

        obj1.test1 >> obj2
        obj1.test2 >> obj2
        obj1.test3 >> obj2
        obj1.test4 >> obj2
        
        
    def testCondition(self):
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0])   
        obj2 = Node(mc.polyCube()[0])
        
        obj1.tx << condition(obj2.tx > 0, 5, 10)
        self.assertEqual(mc.getAttr(str(obj1.tx)), 10.) 
        obj2.tx << 1
        self.assertEqual(mc.getAttr(str(obj1.tx)), 5.) 
        
        
        mc.file(new=True, f=True)
        obj1 = Node(mc.polyCube()[0])   
        obj2 = Node(mc.polyCube()[0])        
        obj1.t << condition(obj2.t > [[1,2,3]], [[1,2,3]], [[1,2,3]])
        
        
        
        
        
        
        





