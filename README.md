# Rig
A Python-based language to reduce the generation of complex node
networks into a simpler human readable form.

![](https://github.com/Eric-Vignola/rig/blob/main/_examples/ye_olde_lerp.gif)

## About
In 3D applications built on node.attribute foundations (ex: Maya, Houdini)
the core act of "rigging" can be reduced into a few major axes:

When we rig, we:
* Create nodes (ex: math nodes, shapes, deformers, etc...)
* Set attribute values. (ex: set a boolean switch to a math node)
* Connect nodes to one another into a complex network.

With Rig, these operations can be reduced into a human readable Python script,
and simplifies the act of building a complex network of nodes.

<details>
<p>
   <summary>For example...</summary>
   
In Maya: 
   * 1- Create two transforms.
   * 2- Elevate the translate attribute of the first transform to the 2nd power. 
   * 3- Plug the result into the second transform's translate attribute.

   ```python
   import maya.cmds as mc

   obj1 = mc.createNode('transform')       # create first transform (returns str())
   obj2 = mc.createNode('transform')       # create first transform (returns str())
   power = mc.createNode('multiplyDivide') # create multiplyDivide node (returns str())
   mc.setAttr(f'{power}.operation', 3)     # sets multiplyDivide node's operation to "power"
   mc.setAttr(f'{power}.input2', 2, 2, 2)  # sets multiplyDivide node's input2 to 2 (x ** 2)
   mc.connectAttr(f'{obj1}.t',f'{power}.input1') # plug 1st transform's .t attribute into input1
   mc.connectAttr(f'{power}.output',f'{obj2}.t') # plug output to 2nd transform's .t attribute

   ```

   With Rig, becomes this:

   ```python
   import rig.commands as rc
   obj1 = rc.createNode('transform')
   obj2 = rc.createNode('transform')
   obj2.t << obj1.t ** 2 # 5 lines of code into one

   ```

</p>
</details>


setAttr/connectAttr calls are done via the __lshift__ (<<) operator. This is referred to as "injection". Everything else follows standard Python lexical structure.

In addition, Rig supports vectorized operations on asymmetric lists.
This adds to the language's simplicity and lets users code in stacked sequences.

Most rig functions are memoized to keep track of repeated operations to minimize waste.

Finally, Rig has a built in translator to logically connect attribute of
different types.

This implementation is still a work in progress, was written for Autodesk Maya,
but could be ported to any application built with a Python interpreter.

## Examples

The *_examples* folder contains scripts you can run directly from your script editor to get a feel for how the language works. And if you're just curious, scratch the surface by expanding the sections below!


<details>
<p>
   <summary>connect/disconnect/set attributes on nodes</summary>

   ```python
   from rig import Node

   # Manually create two cubes and create python Node objects to manipulate them
   obj1 = Node('pCube1')
   obj2 = Node('pCube2')

   # Connect pCube2.t to pCube1.t
   obj2.t << obj1.t

   # Disconnect any incomming connection to pCube2.t
   obj2.t << None

   # setAttr on pCube2.t to 1,2,3
   obj2.t << [1,2,3]

   ```
</p>
</details>


<details>
<p>
   <summary>working with lists</summary>

   ```python
   from rig import Node, List

   # Manually create four cubes and create a python List object to manipulate them
   node_list = List(['pCube1','pCube2','pCube3','pCube4'])
   node = Node('pCube5')

   # Set translation of all elements of node_list to [0,0,0]
   node_list.t << [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

   # or simply
   node_list.t << 0
   
   # Connect pCube5.t to all elements of node_list.t
   node_list.t << node.t

   # Disconnect any incomming connections to node_list
   node_list.t << None

   # Connect pCube1 and pCube2 to pCube3 and pCube4
   node_list[2:].t << node_list[:2].t

   # Add two lists in parallel
   new_node_list = List(['pCube6','pCube7','pCube8','pCube9'])
   added = new_node_list.t + node_list.t
   print(added) # List([Node(add1.output3D), Node(add2.output3D), Node(add3.output3D), Node(add4.output3D)])


   ```
</p>
</details>

<details>
<p>
   <summary>injecting new attributes</summary>

   ```python
   import rig.commands as rc  # these are maya.cmds which output as rig node instances
   from rig import Node, List
   from rig.attributes import Float, Vector, Enum, lock

   # create a cube via rig.commands (wrapped maya.cmds)
   obj1 = rc.polyCube()[0]
   
   # create pCube1.awesomeFloat as a float attribute, set it to 5 and finally lock it
   obj1 << Float('awesomeFloat') << 5 << lock

   # add a Vector to a List
   node_list = List(['pCube1','pCube2','pCube3','pCube4'])
   node_list << Vector('awesomeVector')

   # add an Enum to the list, set default value to be 'green'
   node_list << Enum('color', en=['red','green','blue'], dv=1)

   # add another enum to the first two elements of the list.
   # not specifying an enum value will default to 'False:True'
   node_list[:2] << Enum('switch')

   ```
</p>
</details>

<details>
<p>
   <summary>simple math operations</summary>

   ```python
   
   from rig import Node
   from rig.attributes import Float

   # create 3 cubes
   obj1 = rc.polyCube()[0]
   obj2 = rc.polyCube()[0]
   obj3 = rc.polyCube()[0]

   # add pCube1.tx to pCube2.tx
   add = obj1.tx + obj2.tx
   print(add) # Node('add1.output1D') where add1 is a plusMinusAverage node

   # divide that by 4
   divided = add / 4
   print(divided) # Node('mult1.output') where mult1 is a multiplyDivide node

   # to the power of 2
   power = divided ** 2
   print(power) # Node('pow1.output') where pow1 is a multiplyDivide node

   # add a 'weight' attribute to pCube3 and do a simple lerp operation
   # between pCube1.t and pCube2.t driven by pCube3.weight
   obj3 << Float('weight', min=0, max=1)
   obj3.t << (obj2.t - obj1.t) * obj3.weight + obj1.t

   ```
</p>
</details>

<details>
<p>
   <summary>working with commands, functions and nodes</summary>

   ```python
   from rig import Node
   import rig.commands as rc  # these are maya.cmds which output as rig node instances
   import rig.functions as rf # common python functions that can handle connections 
   import rig.nodes as rn     # createNode wrappers for all defined maya node types
                              # non createNode keyword arguments will be used for injection.

   # get all the cameras transforms wrapped in a List instance
   cameras = rc.listRelatives(rc.ls(type='camera'), p=True) # List([Node(front), Node(persp), Node(side), Node(top)])

   # use rf.max() similar to max()
   rf.max([1,5,4,2]) # returns 5, just like max would

   # use rf.max() with nodes
   rf.max(cameras.tx) # returns a container who's output will be the highest .tx attribute value

   # create a network node called test 
   node = rn.network(n='test') # Node('test')

   # create a multiplyDivide node and set it's operation attribute to 'power'
   node = rn.multiplyDivide(operation=3)
   
   ```
</p>
</details>

<details>
<p>
   <summary>use shorthand connections </summary>

   ```python
   from rig import Node

   obj1 = rc.polyCube()[0]
   obj2 = rc.polyCube()[0]
   obj3 = rc.polyCube()[0]

   # decompose pCube1's world matrix and plug it directly in pCube2.t
   obj2.t << obj1.wm

   # perform a point/matrix operation using a constant
   obj2.t << [10,0,0] * obj1.wm

   # perform a point/matrix operation using pCube3.t
   obj2.t << obj3.t * obj1.wm

   ```
</p>
</details>

<details>
<p>
   <summary>components, multi attributes, aliases </summary>

   ```python
   import rig.commands as rc
   from rig.attributes import Float

   """
   you can interface with components and multi attributes like list objects
   """
   # create a polySphere
   obj = rc.polySphere()[0]
   print(obj.vtx)      # not specifying an index will return the first unconnected component
   print(obj.vtx[0])   # prints the first component
   print(obj.vtx[:])   # prints all components
   print(obj.vtx[::2]) # prints every even component

   # move every other vertex to 0,0,0 using injection
   obj.vtx[::2] << [0,0,0]

   # ----------------------------------------------------------------- #

   # add a multi attr to the object
   obj << Float('weight', m=True)
   print(obj.weight) # not specifying an index will return the first unset index

   # set the first 4 indices
   obj.weight[:4] << [0,2,4,6]
   print(obj.weight) # prints the first unset index (4)

   # ----------------------------------------------------------------- #

   # create 3 dummy shapes and a target to receive blendShapes
   happy   = rc.polyCube(name='happy')[0]
   sad     = rc.polyCube(name='sad')[0]
   neutral = rc.polyCube(name='neutral')[0]
   target  = rc.polyCube(name='target')[0]

   # create the blendShape
   morph = rc.blendShape([happy,sad,neutral,target], n='morph')[0] # Node(morph)

   # list all the morph aliases
   print(morph.weight[:]) # [Node(morph.happy), Node(morph.sad), Node(morph.neutral)]

   # set happy to 1
   morph.happy << 1

   # reset all the shapes to 0
   morph.weight[:] << 0

   ```
</p>
</details>

## Requirements
Autodesk Maya (for this implementation).

## Author
* **Eric Vignola** (eric.vignola@gmail.com)

## License
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
