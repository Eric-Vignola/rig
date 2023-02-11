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
import collections

from ._language import container as _container # because of clash with existing conainer node type
from ._language import _is_sequence
from ._language import Node



# ---------------------------- MAYA NODE WRAPPERS ---------------------------- #
"""
Creates a function named after every registeted nodetype whose kargs 
will be used to setAttr on the newly created node and wrapped in a Node() object
"""

for name in mc.ls(nt=True):
    code = '''def {0}(**kargs):
        
        # parse out createNode's keyword arguments, set defaults if missing
        keyword_arguments = dict(kargs)
        name              = ([kargs.pop(x) for x in keyword_arguments if x in ['name',        'n']] or [None])[-1]
        parent            = ([kargs.pop(x) for x in keyword_arguments if x in ['parent',      'p']] or [None] )[-1]
        shared            = ([kargs.pop(x) for x in keyword_arguments if x in ['shared',      's']] or [False])[-1]
        skipSelect        = ([kargs.pop(x) for x in keyword_arguments if x in ['skipSelect', 'ss']] or [False])[-1]
        add_to_container  = ([kargs.pop(x) for x in keyword_arguments if x in ['container'       ]] or [True])[-1]
  
        # create the node
        if not name is None:
            node = _container.createNode('{0}', 
                                         name=name, 
                                         parent=parent, 
                                         shared=shared, 
                                         skipSelect=skipSelect,
                                         container=add_to_container)
                          
        else:
            node = _container.createNode('{0}', 
                                          parent=parent, 
                                          shared=shared, 
                                          skipSelect=skipSelect,
                                          container=add_to_container)
        
        
        # any further keyword arguments are considered setAttrs 
        for k in kargs:
            if _is_sequence(kargs[k]):
                mc.setAttr('%s.%s'%(node, k), *kargs[k])
            else:
                mc.setAttr('%s.%s'%(node, k), kargs[k])
    
        return Node(node)
        
    '''
    
    exec(code.format(name))

    