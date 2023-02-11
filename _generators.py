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


# ------------------------- ASYMMETRICAL GENERATORS -------------------------- #
def _is_basestring(obj):
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




def _yield(obj, index):
    """
    Yields the "index" of a list, tuple, set or dict
    For dicts we yield the key at the given index.
    
    Sets and Dicts are not ordered in python 2.7
    so results are given in their hashed order.
    """
    # set
    if isinstance(obj, (set,)):
        for i, elem in enumerate(obj):
            if i == index:
                return elem    
    
    # list, tuple, nd.array sequence
    elif _is_sequence(obj):#isinstance(obj, (list, tuple,)):
        return list(obj)[index]
    
    # dict, operate on the keys
    elif isinstance(obj, (dict,)):
        for i, key in enumerate(obj):
            if i == index:
                return key
            
    # just return object as is      
    return obj
    


def sequences(*args):
    """
    Asymmetric generator for list, tuples, sets and dict keys.
    
    Yields up to the maximum count capping each smaller items to their last entry.
    
    ex:
    for x in sequences(5, 'hey', ['wow','cool'], [['this is neat'], ['right'], ['heck yeah!']]):
        print (x)
        
    # [5, 'hey', 'wow', ['this is neat']]
    # [5, 'hey', 'cool', ['right']]
    # [5, 'hey', 'cool', ['heck yeah!']]

    """
    # count size of each sublist
    counts = [len(x) if (isinstance(x, (set, tuple, list, dict)) or _is_sequence(x)) else 1 for x in args]

    # get the max sublist size, this will determine
    # capping index per sublist to yield
    max_size = max(counts)

    # for each sublist, yield an element capped at the highest count
    for i in range(max_size):
        result = []
        for j, c in enumerate(counts):
            index = min(i, c-1)
            result.append( _yield(args[j], index) )

        yield result

        
  
def dictionaries(**kargs):
    """
    Asymmetric dict generator.
    
    Yields up to the maximum count capping each smaller items to their last entry.
    
    ex:
    for x in generators.dictionaries(neat=[5,6,7], great=3, woot=[[1,2,3],[4,5,6,7,8]]):
        print (x)   
        
    # {'great': 3, 'woot': [1, 2, 3], 'neat': 5}
    # {'great': 3, 'woot': [4, 5, 6, 7, 8], 'neat': 6}
    # {'great': 3, 'woot': [4, 5, 6, 7, 8], 'neat': 7}
    
    """    
    
    keys = list(kargs.keys())
    vals = list(kargs.values())

    # count size of each sublist
    counts = [len(x) if isinstance(x, (set, tuple, list, dict)) else 1 for x in vals]
       
    # get the max sublist size, this will determine
    # capping index per sublist to yield
    max_size = max(counts)
            
    # for each sublist, yield an element capped at the highest count
    for i in range(max_size):
        result = {}
        for j, c in enumerate(counts):
            index = min(i, c-1)
            result[keys[j]] = _yield(vals[j], index)

        yield result
    
    
def arguments(*args, **kargs):
    """
    Asymmetric argument generator which accepts
    objects of type list, tuple, set and dict of unequal sizes.
    
    Yields up to the maximum count capping each smaller items to their last entry.
        
    Sets and dict are not ordered in python 2.7
    so results may get weird.
    """
    keys = None
    vals = None
    counts    = []
    kw_counts = []
    max_count = 0
    
    if args:
        counts = [len(x) if (_is_sequence(x) or isinstance(x, (set, tuple, list, dict))) else 1 for x in args]
        max_count = max(counts)
    
    if kargs:
        keys = list(kargs.keys())
        vals = list(kargs.values())
        kw_counts = [len(x) if (_is_sequence(x) or isinstance(x, (set, tuple, list, dict))) else 1 for x in vals]
        max_count = max(kw_counts+[max_count])
        
    for i in range(max_count):
        result = []
        for j, c in enumerate(counts):
            index = min(i, c-1)
            result.append( _yield(args[j], index) )
        
        kw_result = {}
        for j, c in enumerate(kw_counts):
            index = min(i, c-1)
            kw_result[keys[j]] = _yield(vals[j], index)
            
        yield result, kw_result
