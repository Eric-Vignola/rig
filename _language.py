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
import maya.api.OpenMaya as om
import re
import pickle
import numbers

from functools import wraps

from ._generators import sequences, arguments
from .attributes import _clone_attribute


# Container Options
CREATE_CONTAINER = True
USE_SHORTHAND    = True
SKIP_SELECTION   = True
PUBLISH_PLUGS    = True



class DebugPrint():
    def __init__(self, message, execute=True):
        self.message = message
        self.execute = execute
        
    def __enter__(self):
        if self.execute:
            print (self.message)
        
    def __exit__(self, *args):
        if self.execute:
            print ('')


# -------------------------- MEMOIZATION FUNCTIONS --------------------------- #

def _name_to_pickle(node_attr):
    """ used to create a unique memo key """
    if isinstance(node_attr, (list, set, tuple)):
        return [_name_to_pickle(x) for x in node_attr]

    split    = str(node_attr).split('.')
    split[0] = mc.ls(split[0], uid=True)[0]
    return '.'.join(split)


def _pickle_to_name(uuid_attr):
    """ convets a memo key back to a Node """
    if isinstance(uuid_attr, (list, set, tuple)):
        return [_pickle_to_name(x) for x in uuid_attr]

    split = uuid_attr.split('.')
    node  = mc.ls(split[0], uid=True)

    if node:
        split[0] = node[0]
        node_attr = '.'.join(split)
        return Node(node_attr)

    else:
        raise Exception('Node {} no longer in scene.'.format(uuid_attr))


def memoize(func):
    """ memoizes a function's return according to both args ans kargs """
    
    # empty namespace to hold data
    class namespace():
        pass    
    
    def _deep_float(*args):
        # When memoizing, we don't need to see the difference
        # between ints and floats when making keys
        args_  = []
        for x in args:
            if x is None or _is_node(x) or isinstance(x, str):
                args_.append(x)
    
            elif isinstance(x, numbers.Real):
                args_.append(float(x))
    
            elif _is_sequence(x):
                args_.append(_deep_float(*x))
    
        return args_ 

    
    
    @wraps(func)
    def wrapper(*args, **kargs):
        
        # TODO: clean the cache?
        
        
        # force all ints to floats to generate a common key
        args_  = _deep_float(*args)
        kargs_ = dict(zip(kargs.keys(), _deep_float(*kargs.values())))
        key    = hash(pickle.dumps((args_, kargs_, container.containers[:1])))

        if key in cache:
            result = cache[key]
            if result.uuids:
                if len(mc.ls(result.uuids)) == len(result.uuids):
                    return result.data
                else:
                    cache.pop(key, None)


        # result not cashed, memoize the result
        result = namespace()
        result.data  = func(*args, **kargs)
        result.uuids = []
        
        if not result.data is None:
            
            if _is_node(result.data):
                result.uuids = [str(result.data).split('.')[0]]
                
            elif _is_sequence(result.data):
                for item in result.data:
                    if _is_node(item):
                        result.uuids.append(str(item).split('.')[0])

            # convert to uuids
            result.uuids = mc.ls(result.uuids, uuid=True)


        cache[key] = result
        return result.data

    cache = {}
    return wrapper




def vectorize(func, favor_index=None):
    """ 
    Decorator that passes assymetric arguments to a function.
    
    if favor_index is not None, vectorization will stop once the
    first arg count hits a specified index
    """
    
    @wraps(func)
    def wrapper(*args, **kargs):
    
        #depth = lambda L: int(_is_sequence(L) and max(map(depth, L))+1)
        #depth = lambda L: int((_is_sequence(L) and max(map(depth, L))+1)\
            #or (_is_node(L) and L.__data__.compound is not None))        
        
        results = []

        # are any args or kargs given?
        if args or kargs:
    
            # trigger vectorization only if a List is given
            valid_args  = True#args and any([_is_list(x) for x in args])
            valid_kargs = True#kargs and any([_is_list(x) for x in kargs.values()])

            if valid_args or valid_kargs:
                max_count = None
                if not favor_index is None:
                    max_count = len(args[favor_index])
    
                count   = 0
                for args_, kargs_ in arguments(*args, **kargs):
                    res = func(*args_, **kargs_)
                    if res:
                        results.append(res)
    
                    count += 1
                    if not favor_index is None and count == max_count:
                        break
    
            # run the function without vectorization
            else:
                return func(*args, **kargs)
    
    
        # run the function on its own
        else:  
            return func()
    
    
        # if only one item result return single output or List
        if results:
            if len(results) > 1:
                try: 
                    return List(results)
                except:
                    return results
    
            return results[0]
        
    return wrapper







# -------------------------------- UTILITIES --------------------------------- #
def _getPlugType(attr):
    try:  
        if attr.hasFn(om.MFn.kAttribute):
            attr_fn = om.MFnAttribute(attr)
            kargs = attr_fn.getAddAttrCmd(longFlags=False).split()
            
            if '-at' in kargs:
                return re.findall('"([^"]*)"', kargs[kargs.index('-at')+1])[0]
                
            elif '-dt' in kargs:
                return re.findall('"([^"]*)"', kargs[kargs.index('-dt')+1])[0]
    except:
        pass
        
    return None


def _plugIsMatrix(attr):
    #attr = plug.attribute()
    if attr.hasFn(om.MFn.kTypedAttribute):
        attr_fn = om.MFnTypedAttribute(attr)
        return attr_fn.attrType() == 5
    
    return False


def _plugIsAny(attr):
    #attr = plug.attribute()
    if attr.hasFn(om.MFn.kTypedAttribute):
        attr_fn = om.MFnTypedAttribute(attr)
        return attr_fn.attrType() == 24
    
    return False

def _plugIsCompound(attr):
    #attr = plug.attribute()
    if attr.hasFn(om.MFn.kCompoundAttribute):
        attr_fn = om.MFnCompoundAttribute(attr)
        return attr_fn.numChildren()
    
    return None



def _getAttrTypeFromPlug(plug):
    """
    Returns the string type name of the given attr. Raises RuntimeError
    if the node does not have an attribute of the given name.

    **attr_name**	str name of attr

    **RETURNS**		str attribute type

    >>> node.getAttrType("myAttr")
    >>> #RESULTS: "double"

    """        

    if self.hasAttr(attr_name):

        plug = self._fn_set.findPlug(attr_name, False)
        attr = plug.attribute()

        # numeric
        if attr.hasFn(om.MFn.kNumericAttribute):
            attr_fn = om.MFnNumericAttribute(attr)
            return self.ATTR_NUM_TYPE_STR_MAP[attr_fn.numericType()]
        
        # unit
        elif attr.hasFn(om.MFn.kUnitAttribute):
            attr_fn = om.MFnUnitAttribute(attr)
            return self.ATTR_UNIT_TYPE_STR_MAP[attr_fn.unitType()]
        
        # typed
        elif attr.hasFn(om.MFn.kTypedAttribute):
            attr_fn = om.MFnTypedAttribute(attr)
            return self.ATTR_TYPED_TYPE_STR_MAP[attr_fn.attrType()]       
        
        # enum
        elif attr.hasFn(om.MFn.kEnumAttribute):
            #attr_fn = om.MFnEnumAttribute(attr)
            return self.ATTR_ENUM_STR
        
        # message
        if attr.hasFn(om.MFn.kMessageAttribute):
            #attr_fn = om.MFnMessageAttribute(attr)
            return self.ATTR_MESSAGE_STR
        
        else:
            raise RuntimeError(f"{self}.{attr_name} currently unsupported by getAttrType().")
        
        
        
    else:
        raise RuntimeError(f"{self} does not have an attribute named: {attr_name}")




def _disconnect_attr(*args, **kargs):
    """
    if a single argument is given, assume user has given us
    destinations and wants to disconnect all incoming connections.
    """

    try:
        # normal use case: disconnect src to dst
        if len(args) > 1:
            mc.disconnectAttr(*args, **kargs)

        # find incomming connections and disconnect
        else:
            connections = mc.listConnections(args[0], s=True, d=False, p=True)
            if connections:
                _disconnect_attr(connections[0], args[0], **kargs)

    except:
        pass



def _is_basestring(obj):
    """ tests for basestring """
    try:
        return isinstance(obj, basestring) # python 2.7
    except:
        return isinstance(obj, str) # python 3

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
    
def _is_attribute(obj):
    """ tests if given input is an Attribute class """
    try:
        return getattr(obj.__class__, '__CLASS_TYPE__', None) == 'Attribute'
    except Exception:
        return False

def _is_compound(obj):
    """ tests object has compound attrs """
    try:
        
        # special case for choice nodes, which must be tested using their input plug
        if obj.__data__.choice:
            try:
                choice_node = str(obj).split('.')[0]
                return bool(Node(mc.listConnections(f'{choice_node}.input[0]', p=True, s=True, d=False)[0]).__data__.compound)
            except:
                pass
        
        return bool(obj.__data__.compound)
    except:
        pass
        
    return False
    
    
#def _is_compound(obj):
    #""" tests object has compound attrs """
    #try:
        #return bool(obj.__data__.compound) or obj.__data__.any
    #except:
        #return False

    
def _is_array(obj):
    """ tests object has array attrs """
    try:
        return obj.__data__.array
    except:
        return False
            

def _get_attr_type(obj):
    try:
        return mc.getAttr(str(obj), type=True)
    except:
        return None


def _get_compound(obj):
    """ returns compound component (or sequence) """
    try:
        #return [Node('{}.{}'.format(obj, att)) for att in obj.__data__.compound]
        return List([f'{obj}.{att}' for att in obj.__data__.compound])
    except:
        if _is_sequence(obj):
            return obj
        
        #return [obj]
        return List([obj])
                    

    
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



def _is_matrix(obj):
    try:
        return obj.__data__.type == 'matrix'
    except:
        return False
    
    
def _is_quaternion(obj):
    try:
        if _is_compound(obj):
            return len(obj.__data__.compound) == 4
    except:
        return False
    
    
def _is_vector(obj):
    try:
        if _is_compound(obj):
            return len(obj.__data__.compound) == 3
    except:
        return False   

def _is_transform(obj):
    try:
        return obj.__data__.transform == True
    except:
        return False

def _is_control_point(obj):
    try:
        return obj.__data__.point == True
    except:
        return False    

    
@memoize    
def _plus_minus_average(*args, operation=1, name='add1'):
    node = container.createNode('plusMinusAverage', name=name)
    node.operation << operation

    if any([_is_compound(x) for x in args]):
        for obj in args:
            node.input3D << obj
        return node.output3D

    else:
        for obj in args:
            node.input1D << obj 
        return node.output1D


@memoize
def _multiply_divide(input1, input2, operation=1, name='mult1'):
    node = container.createNode('multiplyDivide', name=name)
    node.operation << operation

    if any([_is_compound(input1), _is_compound(input2)]):
        node.input1 << input1
        node.input2 << input2
        return node.output
    
    else:
        node.input1X << input1
        node.input2X << input2
        return node.outputX   



@memoize
def _decompose_matrix(token, rotate_order=None):
    node = container.createNode('decomposeMatrix')
    node.inputMatrix << token
    node.inputRotateOrder << rotate_order

    return node.outputTranslate


@memoize
def _compose_matrix(scale=None, rotate=None, translate=None, shear=None, rotate_order=None):
    node = container.createNode('composeMatrix')

    # plug translate
    if not translate is None:
        node.inputTranslate << translate

    # plug scale
    if not scale is None:
        node.inputScale << scale

    # plug shear 
    if not shear is None:
        node.inputShear << shear

    # plug rotate
    if not rotate is None:

        # is this a quaternion?
        quat_test = _get_compound(rotate)
        if len(quat_test) == 4:
            node.useEulerRotation << 0
            node.inputQuat << rotate

        # this is euler angle
        else:
            node.inputRotate << rotate
            node.inputRotateOrder << rotate_order
        

    return node.outputMatrix

@memoize
def _quaternion_to_euler(quat, rotate_order=None):
    node = container.createNode('quatToEuler')
    node.inputQuat << quat
    node.inputRotateOrder << rotate_order

    return node.outputRotate

@memoize
def _euler_to_quaternion(token, rotate_order=None):
    node = container.createNode('eulerToQuat')
    node.inputRotate << token
    node.inputRotateOrder << rotate_order
    
    return node.outputQuat

@memoize
def _matrix_multiply(*tokens, **kargs):
    local = ([kargs.pop(x) for x in list(kargs.keys()) if x in ['local']] or [None] )[-1]
    if kargs:      
        raise Exception('Unsupported keyword args: {}'.format(kargs.keys()))

    # are we doing point matrix mult?
    if len(tokens) == 2:
        count = 0
        matrix_index = 0
        vector_index = 0

        for i, obj in enumerate(tokens):
            
            if _is_matrix(obj):
                matrix_index = i
                count +=1
            else:
                vector_index = i
            
                
        if count == 1:
            node = container.createNode('pointMatrixMult')
            node.inMatrix << tokens[matrix_index]
            node.inPoint  << tokens[vector_index]
            node.vectorMultiply << local

            return node.output


    # do a straight matrix sum operation
    node = container.createNode('multMatrix')
    for obj in tokens:
        node.matrixIn << obj
    return node.matrixSum


@memoize
def _matrix_add(*tokens, **kargs):
    weights = ([kargs.pop(x) for x in list(kargs.keys()) if x in ['weights']] or [None] )[-1]
    if kargs:      
        raise Exception('Unsupported keyword args: {}'%kargs.keys()) 


    if weights is None:
        node = container.createNode('addMatrix')
        for obj in tokens:
            node.matrixIn << obj 

    else:

        if isinstance(weights, numbers.Real) or _is_node(weights):
            weights = [weights]

        node  = container.createNode('wtAddMatrix')
        index = 0
        for obj, w in sequences(tokens, weights):
            node.wtMatrix[index].matrixIn << obj   
            node.wtMatrix[index].weightIn << w        
            index+=1

    return node.matrixSum 

@memoize
def _matrix_inverse(token):
    node = container.createNode('inverseMatrix')
    node.inputMatrix << token
    return node.outputMatrix    

@memoize
def _quaternion_add(quat1, quat2):
    node = container.createNode('quatAdd')
    node.input1Quat << quat1
    node.input2Quat << quat2

    return node.outputQuat

@memoize
def _quaternion_multiply(quat1, quat2):
    node = container.createNode('quatProd')
    node.input1Quat << quat1
    node.input2Quat << quat2

    return node.outputQuat

@memoize
def _quaternion_subtract(quat1, quat2):
    node = container.createNode('quatSub')
    node.input1Quat << quat1
    node.input2Quat << quat2

    return node.outputQuat



@memoize
def _condition_op(input0, op, input1):
    """
    Defines the basic condition operators and returns a simple True/False output.
    This can then be used in condition (cond) function that operates like an "if" statement
    such as cond(<condition_op>, <if_true>, <if_false>)
    
    ex: cond(Node('pCube1').t > 5, Node('pCube2').t, Node('pCube3').t)
    
    """

    # make sure condition operator is valid
    OPERATORS = {'==' : 0, 
                 '!=' : 1, 
                 '>'  : 2, 
                 '>=' : 3, 
                 '<'  : 4, 
                 '<=' : 5}

    NAMES = {'==' : 'equal1', 
             '!=' : 'not_equal1', 
             '>'  : 'greater1', 
             '>=' : 'greater_or_equal1',
             '<'  : 'lesser1',
             '<=' : 'lesser_or_equal1'}

    if op not in OPERATORS:
        raise Exception('Unsupported condition operator. given: {}'.format(op))        


    # if neither input0 or input1 are compound attrs, just do a straight up setup
    if not _is_compound(input0) and not _is_compound(input1):
        node = container.createNode('condition', name=NAMES[op]) 

        node.firstTerm    << input0 # set or connect first term
        node.secondTerm   << input1 # set or connect second term
        node.operation    << OPERATORS[op] # set condition

        node.colorIfTrue  << 1 # True
        node.colorIfFalse << 0 # False

        return node.outColorR


    # build a setup and pipe the output to a vector
    else:

        with container(NAMES[op]):

            # create pass through plugs
            compound_input0 = _get_compound(input0)
            compound_input1 = _get_compound(input1)
            count0 = len(compound_input0)
            count1 = len(compound_input1) 
            count  = max(count0, count1)
            output_plug  = _constant([0]*count, name='output_plug1')
            output_plugs = _get_compound(output_plug)

            # publish the plugs
            index = 0

            for input0, input1 in sequences(compound_input0, compound_input1):
                node = container.createNode('condition', name=NAMES[op], ss=True)
                
                node.firstTerm    << input0 # set or connect first term
                node.secondTerm   << input1 # set or connect second term
                node.operation    << OPERATORS[op] # set condition

                node.colorIfTrue  << 1
                node.colorIfFalse << 0

                output_plugs[index] << node.outColorR

                index += 1


        return output_plug


@vectorize
@memoize
def condition(condition_op, if_true, if_false):

    """ 
    cond(<condition_op>, <token if true>, <token if false>)

        Creates condition node to solve "if" statements.

        Examples
        --------
        >>> cond(pCube1.t > pCube2.t, 0, pCube3.t)
        >>> cond(pCube1.rx < 45, pCube1.rx, 45)
    """    

    if isinstance(condition_op, numbers.Real):
        if condition_op:
            return if_true
        else:
            return if_false


    # if condition_op is a compound attrs, just do a straight up setup
    if not _is_compound(condition_op):
        node = container.createNode('condition', name='condition1') 

        node.firstTerm    << condition_op  # set or connect first term
        node.secondTerm   << 1 # set or connect second term
        node.operation    << 0 # set condition

        node.colorIfTrue  << if_true  # True
        node.colorIfFalse << if_false # False

        if _is_compound(if_true) or _is_compound(if_false):
            return node.outColor
        return node.outColorR


    # build a setup and pipe the output to a vector
    else:

        with container('condition1'):

            # create pass through plugs
            compound_op     = _get_compound(condition_op)
            compound_input0 = _get_compound(if_true)
            compound_input1 = _get_compound(if_false)

            # if the operator is all numbers
            if all([isinstance(x, numbers.Real) for x in compound_op]):
                result = []
                for op, if_true, if_false in sequences(compound_op, compound_input0, compound_input1):
                    if op:
                        result.append(is_true)
                    else:
                        result.append(is_false)

                return result


            # else, build a tree and publish the plugs
            index = 0
            count0 = len(compound_input0)
            count1 = len(compound_input1) 
            count2 = len(compound_op) 
            count  = max(count0, count1, count2)
            
            # this will create a neat output plug when publishing nodes
            if container.create_container and count == 3:
                output_plug  = container.createNode('plusMinusAverage', name='output_plug1').output3D
                output_plugs = _get_compound(output_plug)
                input_plugs  = _get_compound(output_plug.input3D)
                
            elif container.create_container and count == 1:
                output_plug  = container.createNode('plusMinusAverage', name='output_plug1').output1D
                output_plugs = _get_compound(output_plug)
                input_plugs  = _get_compound(output_plug.input1D)
                
            elif container.create_container and count == 2:
                output_plug  = container.createNode('plusMinusAverage', name='output_plug1').output2D
                output_plugs = _get_compound(output_plug)
                input_plugs  = _get_compound(output_plug.input2D)
            
            # use a constant
            else:
                output_plug  = _constant([0]*count, name='output_plug1')
                output_plugs = _get_compound(output_plug)
                input_plugs  = output_plugs
                
            
            
            for op, if_true, if_false in sequences(compound_op, compound_input0, compound_input1):

                if isinstance(op, numbers.Real):
                    if op:
                        input_plugs[index] << if_true
                    else:
                        input_plugs[index] << if_false

                else:
                    
                    node = container.createNode('condition', name='condition1', ss=True)
                    
                    node.firstTerm    << op # set or connect first term
                    node.secondTerm   << 1 # set or connect second term
                    node.operation    << 0 # set condition
                    
                    node.colorIfTrue  << if_true
                    node.colorIfFalse << if_false

                    input_plugs[index] << node.outColorR
                    
                index += 1


        return output_plug    





# -------------------------- INIT container MANAGER -------------------------- #

class Container():
    """ 
    A class that tracks the container depth and tracks created nodes and connections.
    """
    
    def __init__(self):
        self.containers       = [] # keep track of container depth
        self.published        = None
        self.create_container = CREATE_CONTAINER
        self.skip_selection   = SKIP_SELECTION
        self.use_shorthand    = USE_SHORTHAND
        self.publish_plugs    = PUBLISH_PLUGS
        self.debug            = True
      
    
    def __call__(self, name):
        """ Sets the desired container name which will be created
            when the create_node method is invoked
        """
        if self.create_container:
            new_container = None
            if not self.containers:
                new_container = mc.createNode('container', name=name)
                
            self.containers.append(new_container)    
                
        return self
    
    
    def __enter__(self):
        return self
    
    
    def __exit__(self, exception_type, exception_val, trace):
        self.containers = self.containers[:-1]
        if not self.containers:
            self.published = None
            
    
    def _cleanup_unit_conversion(self, plugs):

        # include any conversion nodes into the leaf container
        if plugs and self.containers:
            unit_conversion_nodes = mc.listConnections(plugs, s=True, d=False, type='unitConversion') or None

            if unit_conversion_nodes:
                self.add(unit_conversion_nodes)

    
    
    
    def shorthand(self, src, dst):

        # --- GRACEFULLY HANDLES SHORTHAND NODE CONNECTIONS --- #

        def _matrix_to_transform(src, dst):

            attr = '.'.join(str(dst).split('.')[1:])
            node = _decompose_matrix(src, rotate_order=dst.ro)
            if not attr:
                self.inject(node.outputScale,     dst.s)
                self.inject(node.outputRotate,    dst.r)
                self.inject(node.outputTranslate, dst.t)
                self.inject(node.outputShear,     dst.shear)              

            else:
                if attr in ['scale', 'scaleX', 'scaleY', 'scaleZ']:
                    self.inject(node.outputScale, dst)

                elif attr in ['rotate', 'rotateX', 'rotateY', 'rotateZ']:
                    self.inject(node.outputRotate, dst) 

                elif attr in ['translate', 'translateX', 'translateY', 'translateZ']:
                    self.inject(node.outputTranslate, dst) 

                elif attr in ['shear', 'shearXY', 'shearXZ', 'shearYZ']:
                    self.inject(node.outputShear, dst)

                else:
                    return False

            return True

        def _matrix_to_quaternion(src, dst):
            node = _decompose_matrix(src)
            self.inject(node.outputQuat, dst)                    
            return True

        def _matrix_to_vector(src, dst):
            node = _decompose_matrix(src, rotate_order=dst.ro)
            self.inject(node, dst)
            return True
        
        def _matrix_to_point(src, dst):
            transform = Node(mc.listRelatives(str(dst).split('.')[0], p=True)[0])
            node = _decompose_matrix(src, rotate_order=transform.ro)
            self.inject(node, dst)
            return True        
        
        def _quaternion_to_transform(src, dst):
            attr = '.'.join(str(dst).split('.')[1:])

            if not attr:
                node =_quaternion_to_euler(src, rotate_order=dst.ro)
                self.inject(node, dst.r)
                return True

            elif attr in ['rotate', 'rotateX', 'rotateY', 'rotateZ']:                       
                node = _quaternion_to_euler(src, rotate_order=dst.ro)
                self.inject(node, dst)
                return True

            return False

        def _quaternion_to_vector(src, dst):
            attrs = _get_compound(quat)
            node = _constant(attrs[:-1])
            self.inject(node, dst)
            return True

        def _quaternion_to_matrix(src, dst):
            node = _compose_matrix(rotate=src)
            self.inject(node, dst)  
            return True

        def _vector_to_transform(src, dst):
            attr = '.'.join(str(dst).split('.')[1:])
            if not attr:
                self.inject(src, dst.t)
            else:
                self.inject(src, dst)

            return True

        def _vector_to_quaternion(src, dst):
            node = _euler_to_quaternion(src)
            self.inject(node, dst)
            return True

        def _vector_to_matrix(src, dst):
            node = _compose_matrix(translate=src)
            self.inject(node, dst)
            return True

        def _transform_to_quaternion(src, dst):
            attr = '.'.join(str(src).split('.')[1:])

            if not attr:
                node = _decompose_matrix(src.matrix)
                self.inject(node.outputQuat, dst)
                return True

            elif attr in ['rotate', 'rotateX', 'rotateY', 'rotateZ']:
                node = _euler_to_quaternion(src)
                self.inject(node, dst)
                return True

            return False


        def _transform_to_vector(src, dst):
            attr = '.'.join(str(src).split('.')[1:])

            if not attr:
                self.inject(src.t, dst)
            else:
                self.inject(src, dst)

            return True

        def _transform_to_matrix(src, dst):
            
            attr = '.'.join(str(src).split('.')[1:])
            if not attr:
                self.inject(src.matrix, dst)
                return True

            elif attr in ['scale', 'scaleX', 'scaleY', 'scaleZ']:
                node = _compose_matrix(scale=src)
                self.inject(node, dst)
                return True

            elif attr in ['rotate', 'rotateX', 'rotateY', 'rotateZ']:
                node = _compose_matrix(rotate=src, rotate_order=src.ro)
                self.inject(node, dst)
                return True

            elif attr in ['translate', 'translateX', 'translateY', 'translateZ']:
                node = _compose_matrix(translate=src)
                self.inject(node, dst)
                return True

            elif attr in ['shear', 'shearXY', 'shearXZ', 'shearYZ']:
                node = _compose_matrix(shear=src.shear)
                self.inject(node, dst)
                return True


            return False



        if not self.use_shorthand:
            return False

        if _is_matrix(src):
            if _is_transform(dst):
                return _matrix_to_transform(src, dst)
            if _is_quaternion(dst):
                return _matrix_to_quaternion(src, dst)
            if _is_control_point(dst):
                return _matrix_to_point(src, dst)
            if _is_vector(dst):
                return _matrix_to_vector(src, dst)


        if _is_quaternion(src):
            if _is_transform(dst):
                return _quaternion_to_transform(src, dst)
            if _is_vector(dst):
                return _quaternion_to_vector(src, dst)
            if _is_matrix(dst):
                return _quaternion_to_matrix(src, dst)        

            
        if _is_transform(src):
            if _is_quaternion(dst):
                return _transform_to_quaternion(src, dst)
            if _is_vector(dst) or _is_control_point(dst):
                return _transform_to_vector(src, dst)
            if _is_matrix(dst):
                return _transform_to_matrix(src, dst)    

        if _is_vector(src):
            if _is_transform(dst):
                return _vector_to_transform(src, dst)
            if _is_quaternion(dst):
                return _vector_to_quaternion(src, dst)
            if _is_matrix(dst):
                return _vector_to_matrix(src, dst)   


        return False    
    
    
    
    
    def inject(self, src, dst, force=True):
        """ 
        sets attr or connects [src] to [dst]
        [dst] is always presumed to be a Node with an attr
 
    
    
        # --- CONNECTION LOGIC FOR COMPOUND ATTRIBUTES TYPES ---#

        Connects an attribute and adds unitConversion nodes to the leaf container
        
        Connection logic
        - one to one    x   --> x
        - comp to comp  xyz --> xyz 
        - one to many   x   --> (x)yz
                        x   --> x(y)z
                        x   --> xy(z)
        - many to many  (x) y  z  --> (x) y  z 
                         x (y) z  -->  x (y) z 
                         x  y (z) -->  x  y (z)
        - many to one
            - WITH connect_index = True and destination is a compound leaf attr (ex: pCube1.tx)
              (x) y  z  --> x
               x (y) z  --> y
               x  y (z) --> z
              
            - WITH connect_index = False OR destination is a non compound leaf attr
              (x) y z --> x
              (x) y z --> y
              (x) y z --> z
        """     
    
        def _set_or_connect(src, dst, force=True):
            """ 
            sets attr or connects [src] to [dst]
            [dst] is always presumed to be a Node with an attr
            """

            _disconnect_attr(str(dst))
            
            # is src a node?
            if _is_node(src):
                if '.' in src:
                    dst = str(dst)
                    mc.connectAttr(str(src), dst, force=force)
                    self._cleanup_unit_conversion(dst)
                else:
                    raise Exception("No attribute specified for {}.".format(src))
            
            # is src a number?
            elif isinstance(src, numbers.Real):
                try:
                    mc.setAttr(str(dst), src)
                except:
                    for x, y in sequences([src], _get_compound(dst)):
                        try:
                            mc.setAttr(str(y), x)
                        except:
                            pass
        
            # is data a string?
            elif _is_basestring(src):
                mc.setAttr(str(dst), src, type='string')
        
        
            # is data a sequence?
            elif _is_sequence(src):
                
                # are we setting a matrix?
                if _get_attr_type(dst) == 'matrix':
                    mc.setAttr(str(dst), *src, type='matrix')
                
                # try to set a vector, or quaternion
                else:
                    dst = _get_compound(dst)
                    if len(dst) < len(src):
                        src = src[:len(dst)]  
                    
                    for x, y in sequences(src, dst):
                        if not x is None:
                            _set_or_connect(x, y, force=force)

                    
            # if we're given None, skip and do nothing
            elif src is None:
                pass
        
            else:
                raise Exception("Don't know how to handle {} for set/plug purposes.".format(dst))
            
    
        # Disconnect Attr
        if src is None:
            _disconnect_attr(str(dst))        

        # if destination is typed, trust the user knows what they want
        elif dst.__data__.type == 'typed':
            #print (f'compound to typed {src} ---> {dst}')
            _set_or_connect(src, dst, force=force)
            
        #elif dst.__data__._is_array and _is_sequence(src):
            #pass
        
        else:

            compound_src = _is_compound(src)
            compound_dst = _is_compound(dst)            
                
            # compound to compound / attr to attr
            if (compound_src and compound_dst) or (not compound_src and not compound_dst):
                #print (f'compound to compound {src} ---> {dst}')
                _set_or_connect(src, dst, force=force)
                
            elif compound_src and not compound_dst and dst.__data__.index:
                compound_src = _get_compound(src)
                index = min(dst.__data__.index, len(compound_src)-1)
                #print (f'compound to index {compound_src[index]} ---> {dst}')
                _set_or_connect(compound_src[index], dst)
            
            else:
                compound_src = _get_compound(src)
                compound_dst = _get_compound(dst)
                    
                if len(compound_dst) < len(compound_src):
                    compound_src = compound_src[:len(compound_dst)]        
            
                for src, dst in sequences(compound_src, compound_dst):
                    #print (f'index to index {src} ---> {dst}')
                    _set_or_connect(src, dst, force=force)    
                
            

    
    def createNode(self, *args, **kargs):
        """ Creates a new node and adds it to the leaf container
        """
        # check for container flag
        add_to_container = True
        if 'container' in kargs:
            add_to_container = kargs.pop('container')
            
        # check for skipSelect
        if not 'ss' in kargs and not 'skipSelect' in kargs:
            kargs['skipSelect'] = self.skip_selection
          
        # create the node  
        node = mc.createNode(*args, **kargs)

        # add node to root container
        if add_to_container:
            self.add(node)

        # return as node type
        return Node(node)
    
    
      
    def add(self, nodes):
        """ Adds given node to the leaf level container stack
        """
        if self.containers:
            try:
                mc.container(self.containers[0], edit=True, addNode=nodes, force=True)
            except:
                pass
    
    
    def get_members(self):
        """
        Returns the members of the current leaf level container
        """
        if self.containers:
            return mc.container(self.containers[0], q=True, nodeList=True)
        
        
    def publish_input(self, node_attr, publish_name):
        """ Basic interface to bind an input attribute to the top container
        """
        return self.publish(node_attr, publish_name)
    

    def publish_output(self, node_attr, publish_name):
        """ Basic interface to bind an input attribute to the top container
        """
        return self.publish(node_attr, publish_name, lock=True)    
    
    
    
    def publish(self, node_attr, publish_name, lock=False):
        """ Adds attributes to the container. Giving up on container publishing
            mechanism which does not handle arrays properly.
        """
        def _lock(token):
            mc.setAttr(str(token), l=True)
            
            if _is_compound(token):
                for att in _get_compound(token):
                    mc.setAttr(str(att), l=True)            
        
          
        # only publish at the top level container
        if self.create_container and self.publish_plugs and len(self.containers) == 1:
            
            # create the published node if none exists
            if not self.published:
                self.published = container.createNode('network', name=f'published_{self.containers[0]}')
                        
            
            # publish the plugs
            result = _clone_attribute(node_attr, 
                                      self.published, 
                                      attr_name=publish_name,
                                      connect=True)
            
            if _is_list(result):
                for i, item in enumerate(result):
                    mc.container(self.containers[0], 
                                 edit=True, 
                                 publishAndBind=[item, f'{publish_name}{i}'])
                    
                    if lock:
                        _lock(item)
                        
            else:
                mc.container(self.containers[0], 
                             edit=True, 
                             publishAndBind=[result, publish_name])
                
                if lock:
                    _lock(result)                

            return result


        # return what was given with no publishing bypass
        return node_attr    
    
    
    
    

              
            

# init the Container context manager
container = Container()


# init the options function
def options(create_container = CREATE_CONTAINER, 
            skip_selection   = SKIP_SELECTION, 
            use_shorthand    = USE_SHORTHAND, 
            publish_plugs    = PUBLISH_PLUGS):
    
    """
    globally sets options, ex: create containers, debug prints, etc...
    """
    # turn container creation on or off
    container.create_container = create_container
    
    # enable shorthand to do common high level conenctions (ex: directly do node2.t << node1.wm )
    container.use_shorthand   = use_shorthand
    
    # skip node selection on creation (basically mc.createNode's ss argument)
    container.skip_selection  = skip_selection
    
    # handles plug publishing mechanism
    container.publish_plugs   = publish_plugs





def _constant(values, name='constant1', dtype='double'):

    # input can be a node.attr, a list or a number    
    DTYPE = {'double':float, 'float':float, 'long':int, 'int':int}

    if not dtype in DTYPE:
        raise Exception('{} is an undupported dtype.'.format(dtype))

    
    # create a blank network node
    node = container.createNode('network', name=name)

    
    # create a single value if count == 1
    values = _get_compound(values)
    count  = len(values)
    if count == 1:
        mc.addAttr(str(node), ln='value', at=dtype, k=True)
        node.value << values[0]

    # create a compound attr
    else:
        values_ = []
        attrs   = list('XYZW')
        mc.addAttr(str(node), ln='value', at='{}{}'.format(dtype,count), k=True)

        for i in range(count):
            mc.addAttr(str(node), ln='value{}'.format(attrs[i]), at=dtype, p='value', k=True)        

            # set the value
            if isinstance(values[i], numbers.Real):
                values_.append(DTYPE[dtype](values[i]))
            else:
                values_.append(values[i])

        node.value << values_


    # return node.value
    return node.value



@memoize
def constant(values, name='constant1', dtype='double'):
    """ memoized version of _constant
    """ 
    return _constant(values=values, name=name, dtype=dtype)




    
class List(om.MObject):
    """
    A list container whose __getattr__ calls will affect the list of Node objects
    Non-Node's will be skipped over
    
    example: list_of_things = List(['pCube1', 3.1415, 'pCube2', 'pCube3'])
             list_of_things.tx    ---> ['pCube1.tx', 3.1415, 'pCube2.tx', 'pCube3.tx']
             list_of_things[1:].r ---> [3.1415, 'pCube2.r', 'pCube3.r']
    """

    __CLASS_TYPE__ = 'List'

    def __init__(self, items=[]):
        self.values = []
        self.__initialize__(items)
        
    def __nonzero__(self):
        return len(self.values) > 0    

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.values))

    def __str__(self):
        return repr(self.values)

    def __call__(self, items):
        self.__initialize__(items)

    def __initialize__(self, items):
        self.values = []
        if items:
            if _is_sequence(items):
                if _is_list(items):
                    self.values = list(items.values)
                else:
                    for x in items:
                        if isinstance(x, numbers.Real):
                            self.values.append(x)
                        
                        elif ':' in str(x).split('.')[-1]:
                            for y in mc.ls(str(x), fl=True):
                                self.values.append(Node(y))
                        else:
                            self.values.append(Node(x))
                                      
            else:
                raise Exception('{} must be a sequence.'.format(items))
            
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return List(self.values[key])
        elif _is_sequence(key):
            return List([self.values[k] for k in key])

        if isinstance(self.values[key], numbers.Real):
            return self.values[key]
        
        return Node(self.values[key])

    def __getattr__(self, name):
        return List([x.__getattr__(name) if _is_node(x) else x for x in self.values])

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __reversed__(self):
        return reversed(self.values)
    
    def __contains__(self, item):
        return item in [x.__str__() for x in self.values]
    
    def __reduce__(self):
        """ To allow @memoize of function args and kargs """
        return (_pickle_to_name, (_name_to_pickle(self.values),))

    def __reduce_ex__(self, protocol):
        """ To allow @memoize of function args and kargs """
        return self.__reduce__()     
    
    
    
    
    # ---------------------------- LIST-like METHODS ----------------------------- #
            
    def append(self, item):
        if isinstance(item, numbers.Real) or item is None:
            self.values.append(item)
        else:
            self.values.append(Node(item))    
        
    def insert(self, ii, item):
        if isinstance(item, numbers.Real) or item is None:
            self.values.insert(ii, item)
        else:
            self.values.insert(ii, Node(item))
            
    def extend(self, item):
        if _is_list(item):
            self.values.extend(item.values)

        elif _is_sequence(item):
            for x in item:
                self.append(x)
        else:
            raise Exception('extend must be given a list, given: {}'.format(item))
        
    def count(self, query):
        count = 0
        for item in self.values:
            if _is_node(item):
                if str(item) == query:
                    count += 1
            elif item == query:
                count += 1
                
        return count
            
    def index(self, query):
        query = str(query)
        for i, item in enumerate(self.values):
            if str(item) == query:
                return i
            
        raise ValueError('{} is not in List()'.format(query))
    
    
    
    # --------------------------- EXPERIMENTAL METHODS --------------------------- #
    def __rshift__(self, other):
        results = []
        for x, y in sequences(self.values, other):
            results.append(x >> y)
            
        return results    
    
    # ---------------------------- ASSIGNMENT METHODS ---------------------------- #
    def __lshift__(self, other):
        if self.values:
            
            # test for attribute injection
            if _is_attribute(other):
                return List([other.apply(x) for x in self.values])
            
            i = 0
            for x, y in sequences(self.values, other):
                        
                # stop if other outnumbers self 
                i += 1
                if i > len(self.values):
                    break
                
                if _is_node(x):
                    x << y
                
        return self
               
                
    

    # --------------------------- ARITHMETIC OPERATORS --------------------------- #
    def __add__(self, other):
        # x + y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x + y)
        return result
    
    def __radd__(self, other):
        # y + x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y + x)
        return result
    
    def __sub__(self, other):
        # x - y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x - y)
        return result
    
    def __rsub__(self, other):
        # y - x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y - x)  
        return result
    
    def __mul__(self, other):
        # x * y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x * y)    
        return result
    
    def __rmul__(self, other):
        # y * x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y * x)  
        return result
    
    def __truediv__(self, other):
        # x / y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x / y)
        return result
    
    def __rtruediv__(self, other):
        # y / x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y / x)   
        return result
    
    def __pow__(self, other):
        # x ** y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x ** y)     
        return result
    
    def __rpow__(self, other):
        # y ** x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y ** x)       
        return result
       
    def __floordiv__(self, other):
        # x // y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x // y) 
        return result
    
    def __rfloordiv__(self, other):
        # y // x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y // x)  
        return result
    
    def __mod__(self, other):
        # x % y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x % y)
        return result
    
    def __rmod__(self, other):
        # x % y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y % x)  
        return result
    
    def __and__(self, other):
        # x & y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x & y)    
        return result
    
    def __rand__(self, other):
        # y & x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y & x)   
        return result
    
    def __or__(self, other):
        # x | y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x | y)   
        return result
    
    def __ror__(self, other):
        # y | x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y | x)  
        return result
    
    def __xor__(self, other):
        # x ^ y
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x ^ y)  
        return result
    
    def __rxor__(self, other):
        # y ^ x
        result = List()
        for x, y in sequences(self.values, other):
            result.append(y ^ x) 
        return result     
    
    def __neg__(self):
        # -x --> -1 * x
        result = List()
        for obj in self.values:
            result.append(-obj)
        return result
    
    def __invert__(self):
        # ~x --> (1 - x)
        result = List()
        for obj in self.values:
            result.append(~obj)   
        return result
    
    
    # --------------------------- COMPARISON OPERATOR ---------------------------- #
    def __eq__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x == y)  
        return result
    
    def __ne__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x != y)   
        return result
    
    def __ge__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x >= y)   
        return result
    
    def __le__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x <= y)  
        return result
    
    def __gt__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x > y) 
        return result
    
    def __lt__(self, other):
        result = List()
        for x, y in sequences(self.values, other):
            result.append(x < y) 
        return result


    # --------------------- PYTHON 2.7 COMPATIBILITY METHODS --------------------- #
    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __div__(self, other):
        return self.__truediv__(other)
    
    def __rdiv__(self, other):
        return self.__rtruediv__(other)





def _resolve_attribute(token, alias=True):
    """
    Resolves any attributes to long form and appends any missing index.
    """
    token = str(token)

    # if we have an attribute
    if '.' in token:
        
        # init the tracker to resolve aliases
        attribute_tracker = om.MSelectionList()
        if alias:
            try:
                attribute_tracker.add(token)        
                token = attribute_tracker.getSelectionStrings()[0]
            except:
                pass
        
        
        # strip out any given index value
        indexed = token.endswith(']')
        index = None
        if indexed:
            index = int(re.findall('\[(.*?)\]',token)[-1])
            token = '['.join(token.split('[')[:-1])        
        
        
        # is this a controil point?
        is_point = False
        try:
            try:
                test0 = not bool(mc.nodeType(f'{token}', inherited=True)) # will fail controlPoints, etc...
            except:
                test0 = True

            test1 = 'controlPoint' in mc.nodeType(f'{token}[0]', inherited=True) 
            is_point = test0 and test1
        except:
            pass


        # fix control point (.cv, .vtx, etc..) to .controlPoints
        if is_point:
            token = re.sub('\..*','.controlPoints', token)             
        
        
        # reset tracker
        attribute_tracker.clear()
        attribute_tracker.add(token)
        plug = attribute_tracker.getPlug(0)
        token = attribute_tracker.getSelectionStrings()[0] # will return the long attribute name

        # if no index was given see if we need to extrapolate one
        if not indexed:

            if plug.isArray:
                index = 0     
                try:
                    # if control point find the first unconnected point
                    if is_point:
                        if plug.numConnectedElements():
                            for i in range(plug.numElements()-1, -1, -1):
                                if not plug.elementByLogicalIndex(i).isConnected():
                                    index = i
                                    break
            
            
                    # numeric array, find the next unset index
                    else:
                        try:
                            index = plug.elementByPhysicalIndex(plug.numElements()-1).logicalIndex()+1
                        except:
                            index = 0
                except:
                    pass  
        
    
        # append extrapolated index and record the attribute
        if not index is None:
            token = f'{token}[{index}]'


    return token   


              
class Node(str):

    __CLASS_TYPE__ = 'Node'
    

    def __init__(self, token):

        # empty namespace to hold data
        class namespace():
            pass
    
        """
        sets the object's internal data structure
        """
        tracker = om.MSelectionList() 
        self.__data__ = namespace()
        self.__data__.node      = None  # stores proper MFn used to query data
        self.__data__.attribute = None  # stores a str()  of the attribute
        self.__data__.compound  = None  # stores a list() of the compound children names
        self.__data__.type      = None  # stores a str()  of the attribute type
        self.__data__.index     = None  # stores the index of a compound child
        self.__data__.any       = False # True if plug is any (ex: choice node)          
        self.__data__.transform = False # True if node is a dag transform
        self.__data__.choice    = False # True if node is type choice
        self.__data__.point     = False # True if attr is a control point (.vtx, .cv, etc...)
        self.__data__.array     = False # True if attr is array attr (.input[0], .input[1], etc...)
        
            
        # Resolve any missing index and attribute long names
        token = _resolve_attribute(token)

        # do we have a plug?
        if '.' in token:
            self.__data__.array = token.endswith(']')
            self.__data__.point = '.controlPoints[' in token and self.__data__.array

            # get some plug info
            if not self.__data__.array:
                tracker.add(token)
            else:
                last_index = '['.join(token.split('[')[:-1])
                tracker.add(last_index)
                
                
            plug = tracker.getPlug(0)
            attr = plug.attribute()            
            self.__data__.type     = _getPlugType(attr)
            self.__data__.any      = _plugIsAny(attr)
            self.__data__.compound = _plugIsCompound(attr)
            

            # store the long attribute
            self.__data__.attribute = '.'.join(token.split('.')[1:])
                    
                    
            # set the compound attrs, or compound child index
            if not self.__data__.compound is None:
                self.__data__.compound = [x.split('.')[-1] for x in mc.listAttr(plug.name())[1:]]
            else:
                try:
                    self.__data__.index = mc.listAttr(plug.parent().name())[1:].index('.'.join(plug.name().split('.')[1:]))            
                except:
                    pass
             
             
        # finally, check if token is a transform
        inherited = mc.nodeType(token, inherited=True)
        self.__data__.transform = 'transform' in inherited
        self.__data__.choice = 'choice' in inherited
        
        # store the proper function to track the node's name
        obj = om.MGlobal.getSelectionListByName(token).getDependNode(0)
    
        if tuple(filter(obj.hasFn, (om.MFn.kTransform, om.MFn.kPluginLocatorNode, om.MFn.kShape, om.MFn.kWorld))):
            self.__data__.node = om.MFnDagNode(obj)
        else:
            self.__data__.node = om.MFnDependencyNode(obj)
            
            
        
            
        
      
    def __new__(cls, token):
        """
        Resolves missing index upon creation.
        TODO: THIS IS JUST FOR REAL TIME MAYA DEBUG SO
              PROPER INDEX SHOWS UP WHEN HIGHLIGHT-EXECUTE
        """
        token = _resolve_attribute(token)
        return super().__new__(cls, token)


       
    def __call__(self, token):
        return Node(token)
    

    def __str__(self):
        node = self.__data__.node
        attr = self.__data__.attribute
        
        if node.hasUniqueName():
            node = node.name()
        else:
            node = node.fullPathName()        
                
        if attr:
            return f'{node}.{attr}'
        
        return node
    
    
    def __hash__(self):
        # returns a hash built from the uuid
        split = str(self).split('.')
        split[0] = mc.ls(split[0], uid=True)[0]
        return hash('.'.join(split))
        
        

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, str(self))


    def __getattr__(self, attr):
        node = str(self)

        # if name is _ or ____, clear attribute
        if attr and not attr.strip('_'):
            return Node(node.split('.')[0])        

        # append name to attribute stack 
        try:
            return Node(f'{node}.{attr}')
        except:
            return Node(f"{node.split('.')[0]}.{attr}")
        



    def __getitem__(self, obj):
        
        node = str(self)

        if node.endswith(']'):
            node = '['.join(node.split('[')[:-1])


        # if we're not slicing, return as index
        if not isinstance(obj, slice) and not _is_sequence(obj): 
            token = '{}[{}]'.format(node, obj)
            return Node(token)


        # if we are slicing/multi indexing, return List depending on node type
        if self.__data__.point:
            elements = mc.ls('{}[*]'.format(node), fl=True)
            
            # hack to fix closed curves who will add additional control points
            elements = list(dict.fromkeys(mc.ls(elements)))

        else:
            try:
                name = '.'.join(node.split('.')[:-1])
                elements = [f'{name}.{x}' for x in mc.listAttr('{}[*]'.format(node))]
                
            except:
                elements = []
                
                
            # find the max index, if user asks for a higher index on a multiattr,
            # give the user what they ask.
            index = None
            try:
                # using this instead of testing size via getAttr(size=True) to get
                # the index of the last available plug
                for attr in mc.listAttr('{}[*]'.format(node))[::-1]:
                    
                    # ignore compounds
                    if '[' in attr and not '.' in attr:
                        index = int(attr.split('[')[1][:-1])+1
                        break
            except:
                index = 0            
            
            
            if not index is None:
                if isinstance(obj, slice):
                    stop = index if (obj.stop is None or obj.stop < 0) else obj.stop
                else:
                    stop = index if max(obj) < 0 else max(obj) + 1
                    
                elements = ['{}[{}]'.format(node, i) for i in range(stop)]


        # return the sliced list
        if isinstance(obj, slice):
            return List(elements[slice(obj.start, obj.stop, obj.step)])
        else:
            return List([elements[x] for x in obj])
    


    def __nonzero__(self):
        """ Returns false if the node is no longer in the scene """
        return mc.objExists(str(self))       

    def __contains__(self, item):
        """ 'Cube' in Node('pCube1.tx') ---> True """ 
        return item in str(self)

    def __reduce__(self):
        """ To allow @memoize of function args and kargs """
        return (_pickle_to_name, (_name_to_pickle(str(self)),))

    def __reduce_ex__(self, protocol):
        """ To allow @memoize of function args and kargs """
        return self.__reduce__()
    
    
    
    # ---------------------- ASSIGNMENT/CONNECTION OPERATOR ---------------------- #
    #def __setattr__(self, name, obj):
        #"""
        #Handles set/connectAttr calls
        #ex: Node('pCube1').t = Node('pCube2').t
            #Node('pCube1').t = [1,2,3]
        #"""
        #return self.__getattr__(name).__lshift__(obj)
    
    
    def __rshift__(self, other):
        return mc.getAttr(str(self))    
    
    
    def __lshift__(self, obj):
        """
        Handles set/connectAttr calls
        ex: Node('pCube1').t << Node('pCube2').tself
            Node('pCube1').t << [1,2,3]
        """
        # test for attribute injection
        if _is_attribute(obj):
            return obj.apply(self)
    
        # proceed to regular data injection logic
        else:
                
            # try injection via shorthand (ex: pCube1.t << pCube2.wm)
            if not container.shorthand(obj, self):
                
                # standard injection
                container.inject(obj, self)
            
            return self
    
    
    # --------------------------- ARITHMETIC OPERATORS --------------------------- #
    @memoize
    def __add__(self, other):
        
        # Are we doing matrix addition
        if _is_matrix(self):
            return _matrix_add(self, other)

        # Or Quaternion?
        elif _is_quaternion(self):
            return _quaternion_add(self, other)
        
        # Or a point Matrix multiplication?

        # x + y
        return _plus_minus_average(self, other, operation=1, name='add1')


    @memoize
    def __radd__(self, other):
        
        # Are we doing matrix addition
        if _is_matrix(self):
            return _matrix_add(other, self)
        
        # Or Quaternion?
        elif _is_quaternion(self):
            return _quaternion_add(other, self)
        
        # y + x
        return _plus_minus_average(other, self, operation=1, name='add1')
    
    
    
    @memoize
    def __sub__(self, other):
        # Are we doing matrix subtraction
        if _is_matrix(self):
            return _matrix_multiply(self, _matrix_inverse(other))
        
        # Or Quaternion?
        elif _is_quaternion(self):
            return _quaternion_subtract(self, other)        
        
        # x - y
        return _plus_minus_average(self, other, operation=2, name='sub1')
    
    
    @memoize
    def __rsub__(self, other):
        # Are we doing matrix subtraction
        if _is_matrix(self):
            return _matrix_multiply(other, _matrix_inverse(self))
        
        # Or Quaternion?
        elif _is_quaternion(self):
            return _quaternion_subtract(other, self)   
        
        # y - x
        return _plus_minus_average(other, self, operation=2, name='sub1')        

    @memoize
    def __mul__(self, other):
        
        # mult matrix
        if _is_matrix(self) or _is_matrix(other):
            return _matrix_multiply(self, other)
        
        # Or Quaternion?
        elif _is_quaternion(self) or _is_quaternion(other):
            return _quaternion_multiply(self, other)
        
        # x * y
        return _multiply_divide(self, other, operation=1, name='mul1')

    @memoize
    def __rmul__(self, other):
        # Are we doing matrix multiplication
        if _is_matrix(self) or _is_matrix(other):
            return _matrix_multiply(other, self)
        
        # Or Quaternion?
        elif _is_quaternion(self) or _is_quaternion(other):
            return _quaternion_multiply(other, self)
        
        # y * x
        return _multiply_divide(other, self, operation=1, name='mul1')    
    
    @memoize
    def __truediv__(self, other):
        # x / y
        return _multiply_divide(self, other, operation=2, name='div1')
    
    @memoize
    def __rtruediv__(self, other):
        # y / x
        return _multiply_divide(other, self, operation=2, name='div1')
        
    @memoize
    def __pow__(self, other):
        # x ** y
        return _multiply_divide(self, other, operation=3, name='pow1')

    @memoize
    def __rpow__(self, other):
        # x ** y
        return _multiply_divide(other, self, operation=3, name='pow1')
    
    
    @memoize     
    def __floordiv__(self, other):
        # x // y
        with container('floordiv1'):
            return _constant(self/other - 0.4999999, dtype='long')

    @memoize     
    def __rfloordiv__(self, other):
        # y // x
        with container('floordiv1'):
            return _constant(other/self - 0.4999999, dtype='long')    

    @memoize
    def __mod__(self, other):
        # x % y
        with container('modulo1'):
            return self - _constant(self/other - 0.4999999, dtype='long') * other

    @memoize
    def __rmod__(self, other):
        # x % y
        with container('modulo1'):
            return self - _constant(other/self - 0.4999999, dtype='long') * other    

    @memoize    
    def __and__(self, other):
        # x & y
        with container('logical_and1'):
            return ((self!=0) + (other!=0)) == 2

    @memoize    
    def __rand__(self, other):
        # y & x
        with container('logical_and1'):
            return ((other!=0) + (self!=0)) == 2    

    @memoize 
    def __or__(self, other):
        # x | y
        with container('logical_or1'):
            return ((self!=0) + (other != 0)) > 0

    @memoize 
    def __ror__(self, other):
        # y | x
        with container('logical_or1'):
            return ((other!=0) + (self != 0)) > 0

    @memoize
    def __xor__(self, other):
        # x ^ y
        with container('logical_xor1'):
            return ((self!=0) + (other != 0)) == 1

    @memoize
    def __rxor__(self, other):
        # y ^ x
        with container('logical_xor1'):
            return ((other!=0) + (self != 0)) == 1        

    @memoize    
    def __neg__(self):
        # -x --> -1 * x
        return _multiply_divide(self, -1, operation=1, name='negate1')

    @memoize
    def __invert__(self):
        # ~x --> (1 - x)
        return _plus_minus_average(1, self, operation=2, name='invert1')
    
    
    # --------------------------- COMPARISON OPERATOR ---------------------------- #
    @memoize
    def __eq__(self, other):
        return _condition_op(self, '==', other)
    @memoize
    def __ne__(self, other):
        return _condition_op(self, '!=', other)
    @memoize
    def __ge__(self, other):
        return _condition_op(self, '>=', other)
    @memoize
    def __le__(self, other):
        return _condition_op(self, '<=', other)
    @memoize
    def __gt__(self, other):
        return _condition_op(self, '>',  other)
    @memoize
    def __lt__(self, other):
        return _condition_op(self, '<',  other)
    
    
    
    # --------------------- PYTHON 2.7 COMPATIBILITY METHODS --------------------- #

    @memoize
    def __div__(self, other):
        # python 2.7
        return self.__truediv__(other)
    
    @memoize
    def __rdiv__(self, other):
        # python 2.7
        return self.__rtruediv__(other)
    