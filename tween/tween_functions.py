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


from .._language import container, memoize, vectorize
from .._language import condition
from ..functions import clamp, sqrt, pow

from ..trigonometry import sind, cosd



# --------------------------------- TWEENERS --------------------------------- #
# https://kodi.wiki/?title=Tweeners	


@vectorize
@memoize
def inLinear(t):
    """
    Simple linear tweening, no easing.
    easing in: linear
    """
    with container('inLinear1'):
        t = container.publish_input(t, 'input')
        o = clamp(t,0,1)
        return container.publish_output(o, 'output')
    
@vectorize
@memoize
def outLinear(t):
    """
    Simple linear tweening, no easing.
    easing out: linear
    """
    with container('outLinear1'):
        t = container.publish_input(t, 'input')
        o = inLinear(1-t)
        return container.publish_output(o, 'output')

@vectorize
@memoize
def inQuad(t):
    """
    Easing equation function for a quadratic (t^2)
    easing in: accelerating from zero velocity.
    """
    with container('inQuad1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow(t,2)
        return container.publish_output(o, 'output')
    
@vectorize
@memoize
def outQuad(t):
    """
    Easing equation function for a quadratic (t^2)
    easing out: decelerating to zero velocity.
    """
    with container('outQuad1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -t * (t-2)
        return container.publish_output(o, 'output')
    
@vectorize
@memoize
def inOutQuad(t):
    """
    Easing equation function for a quadratic (t^2)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutQuad1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inQuad(t*2) * 0.5
        greater = outQuad((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')

@vectorize
@memoize
def outInQuad(t):
    """
    Easing equation function for a quadratic (t^2)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInQuad1'):
        t = container.publish_input(t, 'input')
        o = inOutQuad(1-t)
        return container.publish_output(o, 'output')

@vectorize
@memoize
def inCubic(t):
    """
    Easing equation function for a cubic (t^3)
    easing in: accelerating from zero velocity.
    """
    with container('inCubic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow(t,3)
        return container.publish_output(o, 'output')
    
@vectorize
@memoize
def outCubic(t):
    """
    Easing equation function for a cubic (t^3)
    easing out: decelerating from zero velocity.
    """
    with container('outCubic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow((t-1), 3) + 1
        return container.publish_output(o, 'output')
    
@vectorize  
@memoize
def inOutCubic(t):
    """
    Easing equation function for a quartic (t^4)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutCubic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inQuart(t*2) * 0.5
        greater = outQuart((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        return container.publish_output(o, 'output')

@vectorize
@memoize
def outInCubic(t):
    """
    Easing equation function for a cubic (t^3)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInCubic1'):
        t = container.publish_input(t, 'input')
        o = inOutCubic(1-t)
        return container.publish_output(o, 'output')

@vectorize    
@memoize
def inQuart(t):
    """
    Easing equation function for a quartic (t^4)
    easing in: accelerating from zero velocity.
    """
    with container('inQuart1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow(t,4)
        return container.publish_output(o, 'output')


@vectorize    
@memoize
def outQuart(t):
    """
    Easing equation function for a quartic (t^4)
    easing out: decelerating from zero velocity.
    """
    with container('outQuart1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -1 * (pow((t-1), 4) - 1)
        return container.publish_output(o, 'output')
    
@vectorize   
@memoize
def inOutQuart(t):
    """
    Easing equation function for a quartic (t^4)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutQuart1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inQuart(t*2) * 0.5
        greater = outQuart((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')


@vectorize
@memoize
def outInQuart(t):
    """
    Easing equation function for a quartic (t^4)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInQuart1'):
        t = container.publish_input(t, 'input')
        o = inOutQuart(1-t)
        return container.publish_output(o, 'output')
    
   
@vectorize 
@memoize
def inQuint(t):
    """
    Easing equation function for a quintic (t^5)
    easing in: accelerating from zero velocity.
    """
    with container('inQuint1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow(t,5)
        return container.publish_output(o, 'output')

@vectorize    
@memoize
def outQuint(t):
    """
    Easing equation function for a quintic (t^5)
    easing out: decelerating from zero velocity.
    """
    with container('outQuint1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow((t-1), 5) + 1
        return container.publish_output(o, 'output')
 
@vectorize   
@memoize
def inOutQuint(t):
    """
    Easing equation function for a quintic (t^5)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutQuint1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inQuint(t*2) * 0.5
        greater = outQuint((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')
    

@vectorize
@memoize
def outInQuint(t):
    """
    Easing equation function for a quintic (t^5)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInQuint1'):
        t = container.publish_input(t, 'input')
        o = inOutQuint(1-t)
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def inSine(t):
    """
    Easing equation function for a sinusoidal (sin(t))
    easing in: accelerating from zero velocity.
    """
    with container('inSine1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -1 * cosd(t*90) + 1
        
        return container.publish_output(o, 'output')
  
 
@vectorize  
@memoize
def outSine(t):
    """
    Easing equation function for a sinusoidal (sin(t))
    easing out: decelerating from zero velocity.
    """
    with container('outSine1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = sind(t*90)
        return container.publish_output(o, 'output')
 

@vectorize   
@memoize
def inOutSine(t):
    """
    Easing equation function for a sinusoidal (sin(t))
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutSine1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -0.5 * (cosd(180*t) - 1)
        return container.publish_output(o, 'output')
 

@vectorize   
@memoize
def outInSine(t):
    """
    Easing equation function for a sinusoidal (sin(t))
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('inOutSine1'):
        t = container.publish_input(t, 'input')
        o = inOutSine(1-t)
        return container.publish_output(o, 'output')
 
 
@vectorize   
@memoize
def inExpo(t):
    """
    Easing equation function for an exponential (2^t)
    easing in: accelerating from zero velocity.
    """
    with container('inExpo1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = pow(2,10*(t-1))
        return container.publish_output(o, 'output')
   
   
@vectorize    
@memoize
def outExpo(t):
    """
    Easing equation function for an exponential (2^t)
    easing out: decelerating from zero velocity.
    """
    with container('outExpo1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -1 * pow(2,(-10*t)) + 1
        return container.publish_output(o, 'output')


@vectorize
@memoize
def inOutExpo(t):
    """
    Easing equation function for a exponential (2^t)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutExpo1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inQuint(t*2) * 0.5
        greater = outQuint((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')


@vectorize
@memoize
def outInExpo(t):
    """
    Easing equation function for a exponential (2^t)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInExpo1'):
        t = container.publish_input(t, 'input')
        o = inOutExpo(1-t)
        return container.publish_output(o, 'output')


@vectorize
@memoize
def inCirc(t):
    """
    Easing equation function for a circular (sqrt(1-t^2))
    easing in: accelerating from zero velocity.
    """
    with container('inCirc1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = -1 * sqrt(1-(t*t)) + 1
        return container.publish_output(o, 'output')
  
  
@vectorize    
@memoize
def outCirc(t):
    """
    Easing equation function for a circular (sqrt(1-t^2))
    easing out: decelerating from zero velocity.
    """
    with container('outCirc1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        o = sqrt(1 - pow((t-1), 2))
        return container.publish_output(o, 'output')
    
    
@vectorize    
@memoize
def inOutCirc(t):
    """
    Easing equation function for a circular (sqrt(1-t^2))
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutCirc1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inCirc(t*2) * 0.5
        greater = outCirc((t-0.5) * 2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')


@vectorize
@memoize
def outInCirc(t):
    """
    Easing equation function for a circular (sqrt(1-t^2))
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInCirc1'):
        t = container.publish_input(t, 'input')
        o = inOutCirc(1-t)
        return container.publish_output(o, 'output')
    
    
@vectorize
@memoize
def inElastic(t):
    """
    Easing equation function for an elastic (exponentially decaying sine wave)
    easing in: accelerating from zero velocity.
    """
    with container('inElastic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        p = 0.3
        a = 1
        s = (p/360)*90
        t = t - 1
        o = -1 * pow(2,10*t) * sind((t-s) * 360 / p)
        return container.publish_output(o, 'output')
 

@vectorize   
@memoize
def outElastic(t):
    """
    Easing equation function for an elastic (exponentially decaying sine wave)
    easing out: decelerating from zero velocity.
    """
    with container('outElastic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        p = 0.3
        a = 1
        s = (p/360)*90
        o = pow(2,(-10*t)) * sind((t-s) * 360 / p) + 1
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def inOutElastic(t):
    """
    Easing equation function for an elastic (exponentially decaying sine wave)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutElastic1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        lesser  = inElastic(t*2) * 0.5
        greater = outElastic((t-0.5) * 2 ) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def outInElastic(t):
    """
    Easing equation function for an elastic (exponentially decaying sine wave)
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInElastic1'):
        t = container.publish_input(t, 'input')
        o = inOutElastic(1-t)
        return container.publish_output(o, 'output')
    

@vectorize
@memoize
def inBack(t):
    """
    Easing equation function for a back (overshooting cubic easing: (s+1)*t^3 - s*t^2)
    easing in: accelerating from zero velocity.
    """
    with container('inBack1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        s = 1.70158
        o = t*t*((s+1)*t - s)
        return container.publish_output(o, 'output')


@vectorize
@memoize
def outBack(t):
    """
    Easing equation function for a back (overshooting cubic easing: (s+1)*t^3 - s*t^2) 
    easing out: decelerating from zero velocity.
    """
    with container('outBack1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        s = 1.70158
        t = t - 1
        o = t*t*((s+1)*t + s) + 1
        return container.publish_output(o, 'output')


@vectorize
@memoize
def inOutBack(t):
    """
    Easing equation function for a back (overshooting cubic easing: (s+1)*t^3 - s*t^2) 
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutBack1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        lesser = inBack(t*2) * 0.5
        greater = outBack((t-0.5)*2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def outInBack(t):
    """
    Easing equation function for a back (overshooting cubic easing: (s+1)*t^3 - s*t^2) 
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('outInBack1'):
        t = container.publish_input(t, 'input')
        o = inOutBack(1-t)
        return container.publish_output(o, 'output')
   
   
@vectorize 
@memoize
def outBounce(t):
    """
    Easing equation function for a bounce (exponentially decaying parabolic bounce) 
    easing out: decelerating from zero velocity.
    """
    with container('outBounce1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        
        b1 = 7.5625*t*t
        b2 = t - (1.5/2.75)
        b2 = 7.5625*b2*b2+0.75
        b3 = t - (2.25/2.75)
        b3 = 7.5625*b3*b3+0.9375
        b4 = t - (2.625/2.75)
        b4 = 7.5625*b4*b4+0.984375      
        
        b3 =   condition(t < (2.5/2.75), b3, b4)
        b2 =   condition(t < (2/2.75),   b2, b3)
        o  = condition(t < (1/2.75),   b1, b2)
        
        return container.publish_output(o, 'output')


@vectorize
@memoize
def inBounce(t):
    """
    Easing equation function for a bounce (exponentially decaying parabolic bounce)
    easing in: accelerating from zero velocity.
    """
    with container('inBounce1'):
        t = container.publish_input(t, 'input')
        o = 1 - outBounce(1-t)
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def inOutBounce(t):
    """
    Easing equation function for a bounce (exponentially decaying parabolic bounce)
    easing in/out: acceleration until halfway, then deceleration.
    """
    with container('inOutBounce1'):
        t = container.publish_input(t, 'input')
        t = clamp(t,0,1)
        lesser  = inBounce(t*2) * 0.5
        greater = outBounce((t-0.5)*2) * 0.5 + 0.5
        o = condition(t < 0.5, lesser, greater)
        
        return container.publish_output(o, 'output')
  
  
@vectorize  
@memoize
def outInBounce(t):
    """
    Easing equation function for a bounce (exponentially decaying parabolic bounce) 
    easing out/in: deceleration until halfway, then acceleration.
    """
    with container('outInBounce1'):
        t = container.publish_input(t, 'input')
        o = inOutBounce(1-t)
        return container.publish_output(o, 'output')
    

