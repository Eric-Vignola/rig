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

import maya.cmds as mc
import inspect

from ._language import container as _container # because of clash with existing node type
from ._language import vectorize as _vectorize # because of clash with existing node type
from ._language import _is_sequence, _is_basestring, _is_node, _is_list
from ._language import Node, List



# -------------------------- MAYA COMMAND WRAPPERS --------------------------- #
"""    
Wraps every callable command into a function that will return the output as a List()
example: ls(sl=True) # returns: List([Node(pCube1), Node(pCube2),...])
"""
for name, _ in inspect.getmembers(mc, callable):
    if not name in dir(__builtins__):
        
        code = '''def {0}(*args, **kargs):
        
            # check for container flag
            add_to_container = True
            if 'container' in kargs:
                add_to_container = kargs.pop('container')
            
            
            # convert Node arguments to str to resolve any
            # name changes to the nodes 
            # (ex when parenting and name no longer unique)
            args  = list(args)
            kargs = dict(kargs)
            
            for i in range(len(args)):
                if _is_node(args[i]):
                    args[i] = str(args[i])
                    
                elif _is_list(args[i]):
                    args[i] = [str(x) for x in args[i] if _is_node(x)]
                    
                
            for key in kargs:
                if _is_node(kargs[key]):
                    kargs[key] = str(kargs[key])
                    
                elif _is_list(kargs[key]):
                    kargs[key] = [str(x) for x in kargs[key] if _is_node(x)]
                
            
               
            # run the command and wrap the output within the language
            result = mc.{0}(*args, **kargs)
            
            
            # add any created node to the container
            if add_to_container:
                _container.add(result)
                
                # add any resulting shapes not listed in the result output
                try:
                    _container.add(mc.listRelatives(result, ad=True))
                except:
                    pass
            
            
            # If sequence given, try to return as List
            if _is_sequence(result):
                # this command returns a list of nodes
                try:
                    return List(result)
                except:
                    return result
                    
            
              
            # this is a numeric or string result
            if _is_basestring(result):
                try:
                    return Node(result)
                except:
                    return result
            
            
            # this is something else, return as is
            return result
            
        '''

        exec(code.format(name))




## -------------------- EXPERIMENTAL VECTORIZED FUNCTIONS --------------------- #


## --- cluster --- #
#cluster = _vectorize(cluster)


## --- connectAttr --- #
#connectAttr = _vectorize(mc.connectAttr, favor_index=1)


## --- disconnectAttr --- #
#@_vectorize
#def disconnectAttr(*args, **kargs):
    #"""
    #if a single argument is given, assume user has given us
    #destinations and wants to disconnect all incoming connections.
    #"""
    
    ## normal use case
    #if len(args) > 1:
        #mc.disconnectAttr(*args, **kargs)
        
    ## disconnect incoming connection
    #else:
        #connections = mc.listConnections(args[0], s=True, d=False, p=True)
        #if connections:
            #disconnectAttr(connections[0], args[0], **kargs)




## --- xform --- #
#@_vectorize
#def xform(*args, **kargs):
    #"""
    #Does the xform command and ignores the presence of an attribute
    #"""
    #target = str(args[0]).split('.')[0]
    
    #return mc.xform(target, **kargs)
    

## --- getAttr --- #
## mc.getAttr already operates as vectorized 
## (ex: mc.getAttr(['pCube1.t', 'pCube2.t']))
#getAttr = mc.getAttr 


## --- setAttr --- #
#@_vectorize
#def setAttr(*args, **kargs):
    #target = args[0]
    #values = args[1:]

    #if values:
        #try:
            #return mc.setAttr(target, *values[0], **kargs)
        #except:
            #return mc.setAttr(target, values[0], **kargs)
    #else:
        #return mc.setAttr(target, **kargs)












