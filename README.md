# Rig
A Python-based language to reduce the generation of complex node
networks into a simpler human readable form.

## About
In 3D applications built on node.attribute foundations (ex: Maya, Houdini)
the core act of "rigging" can be reduced into a few major axes:

When we rig, we:
* Create nodes (ex: math nodes, shapes, deformers, etc...)
* Set attribute values. (ex: set a boolean switch to a math node)
* Connect nodes to one another into a complex network.

With Rig, these operations can be reduced into a human readable Python script,
which simplifies the act of building a complex network of nodes.

setAttr/connectAttr are done via the __lshift__ operator (<<), everything else
follows standard Python lexical structure.

In addition, Rig supports vectorized operations on asymmetric lists.
This adds to the language's simplicity and lets users code in stacked sequences.

Most rig functions are memoized to keep track of repeated operations to minimize waste.

Finally, Rig has a built in translator to logically connect attribute of
different types.

This implementation is still a work in progress, was written for Autodesk Maya,
but could be ported to any application built with a Python interpreter.

## Examples
<details>
<p>
   <summary>connect/disconnect/set attributes nodes</summary>

   ```python
   from rig import Node

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
   <summary>Working with lists</summary>

   ```python
   from rig import Node, List

   node_list = List(['pCube1','pCube2','pCube3','pCube4'])
   node = Node('pCube5')

   # Set all elements of node_list to [0,0,0]
   node_list.t << [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
   # or
   node_list.t << 0
   
   # Connect pCube5.t to all elements of node_list.t
   node_list.t << node.t

   # Disconnect any incomming connections to node_list
   node_list.t << None

   # Connect pCube1 and pCube2 to pCube3 and pCube4
   node_list[2:].t << node_list[:2].t

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
