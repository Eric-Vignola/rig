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



from .._language import _is_compound
from .._language import container, constant, condition, _get_compound, vectorize
from ..functions import frame

from random import randint as _randint

            
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#                                                                      #
# Warning: Nodes generated with the `random` module will trigger       #
#          Maya's cycle graph warning. The `random` module depends     #
#          on a self feeding cycle to generate random numbers via      #
#          the linear congruential generator algorithm.                #
#                                                                      #
#          https://en.wikipedia.org/wiki/Linear_congruential_generator #
#                                                                      #
#          This warning is moot and this cycle will have no ill effect #
#          on your scene. (Other than a nag at scene load.)            #
#                                                                      #
#          (Use 'cycleCheck -e off' to disable this warning.)          #
#                                                                      #
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#




# ------------------------- RANDOM NUBER GENERATORS -------------------------- #

# !!! do not memoize !!!
@vectorize
def random(trigger=None, seed=None):
    """ 
    random(trigger=None, seed=None)

        Creates a pseudo random function.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.

        Examples
        --------
        >>> obj1.ty << random()
        >>> obj1.ty << random(obj2.tz) # update when obj2.tz changes
        >>> obj1.ty << random(0)       # will expose a trigger variable
    """     
        
    with container('random1'):
        if seed is None:
            seed = _randint(0,123456789)
        
        if trigger is None:
            trigger = frame()
        else:
            trigger = container.publish_input(trigger, 'trigger')
        
            if _is_compound(trigger):
                trigger = sum(_get_compound(trigger))


        # ZX81 recipe
        # https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        # https://www.ams.org/publicoutreach/feature-column/fcarc-random        
        m = 2**16  # modulus
        a = 75     # multiplier
        c = 74     # increment

        init  = constant([seed]*3, dtype='long')               # this initiates the seed and receives the feedback loop
        reset = condition(init.valueX == 0, seed, init.valueX) # this catches the scene load reset condition and re-injects the seed when input is 0

        iteration = (a * reset + c) % m                                          # this is the normal iteration to the next random number
        update    = constant([iteration, 0, trigger], name='CYCLE_SAFE_RANDOM_GENERATOR1') # takes the iteration and packages with a frame update
        init << update.value                                                     # update compound attr is plugged back in the init node to trigger a recompute
        
        output = update.valueX / m # this is the final output / modulus to give us a 0,0-1.0 range

        
        return container.publish_output(output, 'output')




@vectorize
def random3D(trigger=None, seed=None):
    """ 
    random3D()

        Creates a pseudo random 3D function.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.

        Examples
        --------
        >>> pCube1.t = noise3D()
    """

    with container('random3D1'):
        if seed is None:
            seed = [_randint(0,123456789),
                    _randint(0,123456789),
                    _randint(0,123456789)]
            
        if not trigger is None:
            trigger = container.publish_input(trigger, 'trigger')
        
            if _is_compound(trigger):
                trigger = sum(_get_compound(trigger))        
        
        output = constant([random(trigger=trigger, seed=seed[0]),
                           random(trigger=trigger, seed=seed[1]),
                           random(trigger=trigger, seed=seed[2])])
        
        return container.publish_output(output, 'output')
        

@vectorize
def uniform(start, end, trigger=None, seed=None):
    """ 
    uniform()
        Creates a pseudo random function mapped between a float range.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.


        Examples
        --------
        >>> pCube1.tx = uniform(0,5)
    """

    with container('uniform1'):
        if not trigger is None:
            trigger = container.publish_input(trigger, 'trigger')        
        
        start  = container.publish_input(start, 'start')
        end    = container.publish_input(end,   'end')
        output = (end-start) * random(trigger=trigger, seed=seed) + start
        
        return container.publish_output(output, 'output')


@vectorize
def uniform3D(start, end, trigger=None, seed=None):
    """ 
    uniform()

        Creates a pseudo 3D random function mapped between a float range.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.

    """

    with container('uniform3D1'):
        if not trigger is None:
            trigger = container.publish_input(trigger, 'trigger')
        
            if _is_compound(trigger):
                trigger = sum(_get_compound(trigger))          
        
        start  = container.publish_input(start, 'start')
        end    = container.publish_input(end,   'end')        
        output = (end-start) * random3D(trigger=trigger, seed=seed) + start
        
        return container.publish_output(output, 'output')


@vectorize
def randint(start, end, trigger=None, seed=None):
    """ 
    randint()

        Creates a pseudo random function mapped between an integer range.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.


        Examples
        --------
        >>> pCube1.tx = uniform(0,5)
    """

    with container('randint1'):
        if not trigger is None:
            trigger = container.publish_input(trigger, 'trigger')
            
        start  = container.publish_input(start, 'start')
        end    = container.publish_input(end,   'end')           
        output = constant(uniform(start, end, trigger=trigger, seed=seed), dtype='long')

        return container.publish_output(output, 'output')



@vectorize
def randint3D(start, end, trigger=None, seed=None):
    """ 
    randint3D()

        Creates a 3D pseudo random function mapped between an integer range.
        If no update trigger specified, node will use a frame change.
        If seed is None, a random seed will be chosen.


    """
        
    with container('randint3D'):
        if not trigger is None:
            trigger = container.publish_input(trigger, 'trigger')
        
            if _is_compound(trigger):
                trigger = sum(_get_compound(trigger))
                
        start  = container.publish_input(start, 'start')
        end    = container.publish_input(end,   'end')          
        output = constant(uniform3D(start, end, trigger=trigger, seed=seed), dtype='long')
        
        return container.publish_output(output, 'output')
