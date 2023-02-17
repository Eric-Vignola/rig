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
import numbers, re
import maya.api.OpenMaya as om

def _is_list(obj):
    """ tests if given input is a List class """
    try:
        return getattr(obj.__class__, '__CLASS_TYPE__', None) == 'List'
    except Exception:
        return False

def _is_node(obj):
    """ tests if given input is a Node class """
    try:
        return getattr(obj.__class__, '__CLASS_TYPE__', None) == 'Node'
    except Exception:
        return False
    

def _is_compound(obj):
    """ tests object has compound attrs """
    try:
        return bool(obj.__data__.compound) or obj.__data__.any
    except:
        return False


def _is_basestring(obj):
    """ tests for basestring """
    try:
        return isinstance(obj, basestring) # python 2.7
    except:
        return isinstance(obj, str) # python 3
    
    
def _is_sequence(obj):
    """ tests if given input is a sequence """
    
    if _is_basestring(obj):
        return False
    
    try:
        len(obj)
        if isinstance(obj, dict):
            return False
    except Exception:
        return False
    
    return True


def _is_array(obj):
    """ tests object has array attrs """
    try:
        return obj.__data__.array
    except:
        return False
    

def _get_compound(obj):
    """ returns compound component (or sequence) """
    try:
        return ['{}.{}'.format(obj, att) for att in obj.__data__.compound]
    except:
        if _is_sequence(obj):
            return obj
        return [obj]


def _get_attr_type(node_attr):
    '''
    Return a node's attribute type by parsing its "addAttr" command
    '''

    tracker = om.MSelectionList() 
    tracker.add(node_attr)
    plug = tracker.getPlug(0)
    attr = plug.attribute()  
    
    if attr.hasFn(om.MFn.kAttribute):
        
        # look for at or dt flags
        attr_fn = om.MFnAttribute(attr)
        kargs = attr_fn.getAddAttrCmd(longFlags=False).split()        
        if '-at' in kargs:
            return re.findall('"([^"]*)"', kargs[kargs.index('-at')+1])[0]
            
        elif '-dt' in kargs:
            return re.findall('"([^"]*)"', kargs[kargs.index('-dt')+1])[0]



    raise Exception(f'cannot derive attr type from {node_attr}')


# ------------------------------- ATTRIBUTES --------------------------------- #

class _Attribute(object):
    
    __CLASS_TYPE__ = 'Attribute'
    
    
    def __init__(self, name=None, **kargs):

        self.kargs = dict(kargs)
        
        if not name is None:
            self.kargs['keyable']  = ([self.kargs.pop(x) for x in kargs if x in ['keyable', 'k']]   or [True])[-1]
            self.kargs['longName'] = ([self.kargs.pop(x) for x in kargs if x in ['longName', 'ln']] or [name])[-1]
            
            self.size              = ([self.kargs.pop(x) for x in kargs if x in ['size']]      or [None])[-1]
            self.overwrite         = ([self.kargs.pop(x) for x in kargs if x in ['overwrite']] or [True])[-1]
            self.compound          = None
            self.compoundType      = None
            self.notes             = None
        
     
    def apply(self, node):
        kargs = dict(self.kargs)

        name  = ([kargs[x] for x in kargs if x in ['longName', 'ln']]     or [None])[-1]
        multi = ([kargs[x] for x in kargs if x in ['multi', 'm']]         or [False])[-1]
        dv    = ([kargs[x] for x in kargs if x in ['defaultValue', 'dv']] or [0])[-1]
        
        
        # if no name given, we are doing setAttr edit on the given node
        # ex lock, hide, etc...
        if not name:
            
            # hack: if the only args are keyable and channelBox, and the given
            # object is a compound attr, list the compound and apply the values
            if len(kargs) == 2:
                if 'keyable' in kargs and 'channelBox' in kargs:
                    for item in _get_compound(node):
                        mc.setAttr(item, **kargs)

            mc.setAttr(str(node), **kargs)    
            return node


        # ignore any given attribute
        node_string = str(node._)
        
        
        # quietly delete the attribute if it already exists
        if mc.attributeQuery(name, node=node_string, exists=True):
            if self.overwrite:
                try:
                    mc.setAttr('%s.%s'%(node_string, name), lock=False)
                    mc.deleteAttr('%s.%s'%(node_string, name))
                except:
                    pass
          
            # we're not overwriting, return node as is (next index if multi)
            else:
                return node('%s.%s'%(node_string, name))

            
        # if this is a compound attr
        if self.compound:
            count = len(self.compound)
            
            kargs = dict(self.kargs)
            if 'at' in kargs or 'attributeType' in kargs:
                kargs['at'] = next(kargs[x] for x in kargs if x in ['attributeType', 'at'])
                kargs['at'] = '%s%s'%(kargs['at'], count)
                kargs.pop('attributeType', None)
            
            elif 'dt' in kargs or 'dataType' in kargs:
                kargs['dt'] = next(kargs[x] for x in kargs if x in ['dataType', 'dt'])
                kargs['dt'] = '%s%s'%(kargs['at'], count)
                kargs.pop('dataType', None)
                
                
            # remove default value and create parent compound
            kargs.pop('defaultValue', None)
            kargs.pop('dv', None)
            mc.addAttr(node_string, **kargs)


            for i in range(count):
                # add children
                kargs = dict(self.kargs)
                kargs['parent'] = kargs['longName']
                kargs.pop('p', None)
                kargs.pop('multi', None)
                kargs.pop('m', None)
                
                if self.compoundType:
                    kargs['attributeType'] = self.compoundType
                
                if 'dv' in self.kargs:
                    kargs['dv'] = self.kargs['dv'][i]

                elif 'defaultValue' in self.kargs:
                    kargs['defaultValue'] = self.kargs['defaultValue'][i]  

                    
                kargs['longName'] = '%s%s'%(self.kargs['longName'], self.compound[i])
                mc.addAttr(node_string, **kargs)        
            
                            
        else: 
            mc.addAttr(node_string, **self.kargs)
            
            
            
        # For multi attribute, preset size if specified
        if multi and not self.size is None:
            
            for i in range(self.size):
                node_attr = node('%s.%s'%(node_string, name))

                try:
                    node_attr << dv # floats, ints, vectors
                except:
                    try:
                        mc.setAttr(str(node_attr), '', type='string')
                    except:
                        try:
                            mc.setAttr(str(node_attr), 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, type='matrix')
                        except:                    
                            pass
            
        
        # immediately set string attr if this is a note
        node = node('%s.%s'%(node_string, name))
                    
        if not self.notes is None:
            node << self.notes
            
        return node
            


class Int(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Int, self).__init__(name, **kargs) # python 2.7

        self.kargs['attributeType'] = 'long'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)   
           

class Angle(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Angle, self).__init__(name, **kargs) # python 2.7
    
        self.kargs['attributeType'] = 'doubleAngle'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)
        

class Float(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Float, self).__init__(name, **kargs) # python 2.7
    
        self.kargs['attributeType'] = 'double'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)

        
class Bool(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Bool, self).__init__(name, **kargs) # python 2.7
            
        self.kargs['attributeType'] = 'bool'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)         
        
        
class String(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(String, self).__init__(name, **kargs) # python 2.7

        self.kargs['dataType'] = 'string'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None)
    
    
class Mesh(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Mesh, self).__init__(name, **kargs) # python 2.7

        self.kargs['dataType'] = 'mesh'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None) 
        
        
class Time(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Time, self).__init__(name, **kargs) # python 2.7

        self.kargs['dataType'] = 'time'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None) 
        
        
class NurbsCurve(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(NurbsCurve, self).__init__(name, **kargs) # python 2.7
            
        self.kargs['dataType'] = 'nurbsCurve'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None)         
        

class NurbsSurface(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(NurbsSurface, self).__init__(name, **kargs) # python 2.7

        self.kargs['dataType'] = 'nurbsSurface'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None)            
        
        
class Vector(Float):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3    
        except:
            super(Vector, self).__init__(name, **kargs) # python 2.7
        
        self.compound = ['X', 'Y', 'Z']
        
        
class Quat(Float):
    """
    Class to add a quaternion type attribute to a node
    """
    
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3    
        except:
            super(Quat, self).__init__(name, **kargs) # python 2.7
        
        self.compound = ['X', 'Y', 'Z', 'W']
        
        

class Color(Float):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3    
        except:
            super(Color, self).__init__(name, **kargs) # python 2.7
        
        self.compound = ['R', 'G', 'B']
        
        
        
class Euler(Float):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3    
        except:
            super(Euler, self).__init__(name, **kargs) # python 2.7
        
        self.compound = ['X', 'Y', 'Z']
        self.compoundType = 'doubleAngle'


class Matrix(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Matrix, self).__init__(name, **kargs) # python 2.7
    
        self.kargs['attributeType'] = 'matrix'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)



class Enum(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Enum, self).__init__(name, **kargs) # python 2.7

        self.kargs['attributeType'] = 'enum'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)
        
        self.kargs['en'] = ([self.kargs.pop(x) for x in kargs if x in ['enumName', 'en']] or ['False:True:'])[-1]
        
        # allow kargs to be given in the form of a sequence
        if isinstance(self.kargs['en'], (list, set, tuple)):
            self.kargs['en'] = (':'.join(self.kargs['en'])) + ':'
        
        
        
class Note(_Attribute):
    def __init__(self, text=None):
        try:
            super().__init__(self, 'notes') # python 3
        except:
            super(Note, self).__init__('notes') # python 2.7

        self.kargs['dataType'] = 'string'
        self.kargs.pop('dt', None)
        self.kargs.pop('attributeType', None)
        self.kargs.pop('at', None)
        
        self.kargs['keyable']  = False
        self.kargs['writable'] = False
        self.kargs.pop('k', None)
        self.kargs.pop('w', None)
        
        if not text is None:
            self.notes = text
            
            
class Message(_Attribute):
    def __init__(self, name, **kargs):
        try:
            super().__init__(self, name, **kargs) # python 3
        except:
            super(Message, self).__init__(name, **kargs) # python 2.7
    
        self.kargs['attributeType'] = 'message'
        self.kargs.pop('at', None)
        self.kargs.pop('dataType', None)
        self.kargs.pop('dt', None)
            


# -------------------------------- UTILITIES --------------------------------- #
lock   = _Attribute(lock=True)
unlock = _Attribute(lock=False)
hide   = _Attribute(keyable=False, channelBox=False)
unhide = _Attribute(keyable=True,  channelBox=False)




def _clone_attribute(src_node_attr,
                     dst_node,
                     attr_name=None,
                     multi=None,
                     connect=False):
    """
    !!!! HACK EXPERIMENT TO CLONE AND CONNECT ATTRS !!!!
    Attribute cloning interface
    If src_node_attr is a sequence, attribute cloning will be derived from
    the first entry.
    """
    # is this a number, string, or sequence
    if isinstance(src_node_attr, bool):
        dst_node << Bool(attr_name, multi=bool(multi))
        
    elif isinstance(src_node_attr, numbers.Real):
        dst_node << Float(attr_name, multi=bool(multi))
        
    elif not _is_node(src_node_attr) and isinstance(src_node_attr, str):
        dst_node << String(attr_name, multi=bool(multi))
    
    elif not _is_list(src_node_attr) and _is_sequence(src_node_attr) and not any(_is_node(x) for x in src_node_attr):
        count = len(src_node_attr)
        
        if count == 3:
            dst_node << Vector(attr_name, multi=bool(multi))
        elif count == 4:
            dst_node << Quat(attr_name, multi=bool(multi))
        elif count == 16:
            dst_node << Matrix(attr_name, multi=bool(multi))
        else:
            raise Exception('Unsupported sequence for attribute cloning')

    
    # source is a Node    
    else:
        # sequence is assumed to be all same type
        if not _is_sequence(src_node_attr):
            obj = str(src_node_attr)
            node, attr = obj.split('.')[0], ('.'.join(obj.split('.')[1:])).split('[')[0]  
            compound   = _is_compound(src_node_attr)
            
            if compound:
                compound_attrs = _get_compound(src_node_attr)        
            
            # override multi
            if multi is None:
                multi = _is_array(src_node_attr)
                
        else:
            # find the first node in the sequence and use it
            # to establish the attribute type
            obj = next(obj for obj in src_node_attr if _is_node(obj))
            node, attr = obj.split('.')[0], ('.'.join(obj.split('.')[1:])).split('[')[0]  
            compound   = _is_compound(src_node_attr[0])
            
            if compound:
                compound_attrs = _get_compound(src_node_attr[0])        
            
            # override multi
            multi = True       
        
        
        
        # override attribute name
        if not attr_name:
            attr_name = attr.split('.')[-1]
        
        if compound:
            
            # Euler or Vector
            if len(compound_attrs) == 3:
                attr_type = _get_attr_type(compound_attrs[0])
                
                if attr_type == 'doubleAngle':
                    dst_node << Euler(attr_name, multi=multi)
                else:
                    dst_node << Vector(attr_name, multi=multi)
            
            
            elif len(compound_attrs) == 4:
                dst_node << Quat(attr_name, multi=multi)
                
            else:
                raise Exception(f'Unsupported compound attribute: {src_node_attr}')
                       
        
        else:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if obj.endswith(']'):
                obj = '['.join(obj.split('[')[:-1]) + '[0]'
    
            attr_type = _get_attr_type(obj)
    
            if attr_type == 'enum':
                enums = mc.attributeQuery(attr, node=node, listEnum=True)[0].split(':')
                dst_node << Enum(attr_name, en=enums)
            
            else:
                MAPPING = {'long':         Int,
                           'int':          Int,
                           'matrix':       Matrix,
                           'doubleAngle':  Angle,
                           'doubleLinear': Float,
                           'double':       Float,
                           'float':        Float,
                           'double3':      Vector, 
                           'bool':         Bool,
                           'string':       String,
                           'time':         Time, 
                           'mesh':         Mesh, 
                           'nurbsCurve':   NurbsCurve, 
                           'nurbsSurface': NurbsSurface,
                           'message':      Message}  
    
                dst_node << MAPPING[attr_type](attr_name, multi=multi)
    
    
    
    # prepare for return
    dst_node = dst_node('%s.%s'%(dst_node._, attr_name))
    

    # populate values
    if multi:
        if connect:
            for i, item in enumerate(src_node_attr[:]):
                dst_node[i] << item
        else:
            try:
                for i, item in enumerate(src_node_attr[:]):
                    try:
                        value = mc.getAttr(str(item))[0]
                    except:
                        value = mc.getAttr(str(item))                
                
                    dst_node[i] << value
                    
            except:
                pass
            
            
        return dst_node[:] # return List()
            
            
    else:
        if connect:
            dst_node << src_node_attr
        else:
            
            try:
                try:
                    value = mc.getAttr(str(src_node_attr))[0]
                except:
                    value = mc.getAttr(str(src_node_attr))
    
                dst_node << value
                
            except:
                pass
            
        return dst_node # return Node()

