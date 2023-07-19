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


# WIP UTILITIES TO EXPAND AND COLLAPSE CONTAINERS WITHIN THE SCOPE OF THIS PACKAGE
# eg:
#     expand_container('cumsum1')
#     collapse_container('cumsum1')


import maya.cmds as mc
import maya.mel as mel
import pickle, codecs
from ._language import _pickle_to_name


def _get_node_editor():
    return mel.eval('getCurrentNodeEditor()')

def _get_container_data(node):
    data = mc.getAttr(f'{node}.notes')
    data = pickle.loads(codecs.decode(data.encode(), "base64"))
    data['plugs'] = _pickle_to_name(data['plugs'], uid=False)
    data['nodes'] = _pickle_to_name(data['nodes'], uid=False)
    
    return data
    
    
def expand_container(node, update=True):
    # expands a container's contents
    data = _get_container_data(node)
    mc.container(node, edit=True, removeNode=data['nodes'])
    
    if update:
        editor = _get_node_editor()
        
        if editor:
            mc.nodeEditor(editor, e=True, frameAll=True, addNode=data['nodes'])
            mc.nodeEditor(editor, e=True, frameAll=True, removeNode=node)
            
        
    return data['nodes']


def collapse_container(node, update=True):
    # collapses the container back to its components
    data = _get_container_data(node)

    # restore the scope
    mc.container(node, edit=True, addNode=data['nodes'], force=True)
    
    # restore the published inputs and outputs
    for i in range(len(data['plugs'])):
        mc.container('cumsum1', 
                     edit=True, 
                     publishAndBind=[data['plugs'][i], data['names'][i]])


    if update:
        editor = _get_node_editor()
        
        if editor:        
            mc.nodeEditor(editor, e=True, frameAll=True, removeNode=data['nodes'])
            mc.nodeEditor(editor, e=True, frameAll=True, addNode=node)