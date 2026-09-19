# `rig.spec` — cheatsheet

Copy-paste recipes for every spec class, every kwarg the base class
understands, every modifier, and the four operators a spec takes part in.
The blocks run top to bottom as one script and share a namespace; the
nodes built in **Setup** are reused until a section says otherwise.

Concepts, conventions and the verified behaviour live in [`README.md`](README.md).

## Contents

| # | Section | Covers |
|---|---|---|
| — | [Setup](#setup) | Maya standalone, a clean scene, two transforms |
| 1 | [Import surface](#1-import-surface) | the 23 names, the `rig.*` re-exports, the collisions |
| 2 | [Anatomy of a spec](#2-anatomy-of-a-spec) | `.kargs`, laziness, reuse, the return value |
| 3 | [Numeric — `Float` `Int` `Bool` `Angle` `Time`](#3-numeric--float-int-bool-angle-time) | |
| 4 | [Typed — `String` `Matrix` `Message` `Mesh` `NurbsCurve` `NurbsSurface`](#4-typed--string-matrix-message-mesh-nurbscurve-nurbssurface) | |
| 5 | [`Enum`](#5-enum) | list or colon string, default, set by index |
| 6 | [Compounds — `Vector` `Color` `Euler` `Quat`](#6-compounds--vector-color-euler-quat) | children, per-child defaults, ranges |
| 7 | [The kwargs](#7-the-kwargs) | `min` `max` `dv` `keyable` `k` `hidden` `sn` `nn` `multi` `size` `overwrite` |
| 8 | [Modifiers — `lock` `unlock` `hide` `unhide` `skip`](#8-modifiers--lock-unlock-hide-unhide-skip) | on plugs, on compounds, inside a fan-out |
| 9 | [`destroy`](#9-destroy) | both forms, variadic, `strict` `silent` `verbose`, refusals |
| 10 | [`overwrite=`](#10-overwrite) | replace vs keep |
| 11 | [`node >> spec` declares an output](#11-node--spec-declares-an-output) | `writable=False` |
| 12 | [`plug >> Node` and `plug >> "name"` clone](#12-plug--node-and-plug--name-clone) | what travels, what does not |
| 13 | [Multis and pre-sizing](#13-multis-and-pre-sizing) | `multi=True`, `size=`, slices, `append` |
| 14 | [`PlugList` fan-out](#14-pluglist-fan-out) | one spec, many nodes |
| 15 | [`Note`](#15-note) | the `notes` attribute |
| 16 | [Leading underscores](#16-leading-underscores) | `__parked__` |

---

## Setup

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)

import numpy as np

from rig import Node, PlugList
from rig.spec import (
    Angle, Bool, Float, Int, Time,
    Color, Euler, Quat, Vector,
    Matrix, Mesh, Message, NurbsCurve, NurbsSurface, String,
    Enum,
    Note, destroy, hide, lock, skip, unhide, unlock,
)

ctrl = Node.create("transform", name="ctrl")
drv  = Node.create("transform", name="drv")


def at(plug):
    """The attributeType of a plug, e.g. 'double', 'long', 'double3'."""
    node, attr = plug.split(".")
    return cmds.attributeQuery(attr, node=node, attributeType=True)


def dt(plug):
    """The data type cmds.getAttr reports, e.g. 'string', 'mesh'."""
    return cmds.getAttr(plug, type=True)


print(repr(ctrl), cmds.ls(type="container"))   # Node("ctrl") [] -- no container outside a `with container():`
```

---

## 1. Import surface

Twenty-three names. `rig.spec` is the home and every one is re-exported
at the top of the package, as the same object.

```python
import rig
from rig.spec import (
    Angle, Bool, Float, Int, Time,                    # numeric
    Color, Euler, Quat, Vector,                       # compound
    Matrix, Mesh, Message, NurbsCurve, NurbsSurface, String,   # typed
    Enum,
    Note, destroy, hide, lock, skip, unhide, unlock,  # modifiers
)

print(rig.spec.__all__ == [n for n in rig.spec.__all__ if getattr(rig, n) is getattr(rig.spec, n)])   # True
print(rig.Float is Float, rig.lock is lock)                                                            # True True
```

Three collisions to keep in mind — the capitalised name is always the
attribute spec:

```python
from rig.maya.nodetypes import Mesh as MeshNode

print(rig.Mesh is MeshNode)          # False -- rig.Mesh is the `mesh` data attribute; MeshNode is the shape
print(rig.Matrix, rig.matrix)        # <class 'rig.spec.typed.Matrix'> <module 'rig.matrix' ...>
print(rig.Color.__mro__[1].__name__) # Float -- an RGB double3, not a colour value
```

---

## 2. Anatomy of a spec

The positional argument is the long name; the kwargs are `addAttr` flags
plus three the spec keeps (`multi`, `size`, `overwrite`); `keyable=True`
is filled in. Nothing touches Maya until `<<`.

```python
spec = Float("weight", min=0, max=1, dv=0.5)
print(spec.kargs)                                                # {'min': 0, 'max': 1, 'dv': 0.5, 'keyable': True, 'longName': 'weight', 'attributeType': 'double'}
print(spec.size, spec.overwrite, spec.compound, spec.notes)      # None True None None
print(cmds.attributeQuery("weight", node="ctrl", exists=True))   # False
```

Injection adds the attribute and returns the **new plug**, so a value and
then a modifier chain on. The spec itself is reusable.

```python
plug = ctrl << spec << 0.25 << lock
print(repr(plug), plug >> None, cmds.getAttr("ctrl.weight", lock=True))   # Plug("ctrl.weight") 0.25 True
print(cmds.attributeQuery("weight", node="ctrl", min=True), cmds.attributeQuery("weight", node="ctrl", max=True))   # [0.0] [1.0]

drv << spec
print(cmds.getAttr("drv.weight"))                                          # 0.5 -- the default; nothing leaked from ctrl
```

A plug on the left means its node — `ctrl.tx << Float("y")` adds `y` to
`ctrl`:

```python
print(repr(ctrl.tx << Float("y")))   # Plug("ctrl.y")
```

---

## 3. Numeric — `Float` `Int` `Bool` `Angle` `Time`

| Spec | `attributeType` | Value comes back as |
|---|---|---|
| `Float` | `double` | `float` |
| `Int` | `long` | `int` |
| `Bool` | `bool` | `bool` |
| `Angle` | `doubleAngle` | `float`, degrees |
| `Time` | *(broken — see below)* | |

```python
ctrl << Float("blend") << 0.75
ctrl << Int("count", min=0, max=10, dv=5)
ctrl << Bool("flag", dv=True)
ctrl << Angle("twist") << 90

print(at("ctrl.blend"), ctrl.blend >> None)                         # double 0.75
print(at("ctrl.count"), ctrl.count >> None, cmds.attributeQuery("count", node="ctrl", max=True))   # long 5 [10.0]
print(at("ctrl.flag"),  ctrl.flag >> None)                          # bool True
print(at("ctrl.twist"), ctrl.twist >> None)                         # doubleAngle 90.0
```

`Time` asks `addAttr` for `dataType="time"`, which Maya does not have
(`time` is an attribute type), so it raises. The plain `cmds` call is the
workaround; the plug behaves normally afterwards.

```python
print(Time("when").kargs)                                           # {'keyable': True, 'longName': 'when', 'dataType': 'time'}
try:
    ctrl << Time("when")
except RuntimeError as e:
    print(e)                                                        # Type specified for new attribute's data type is unknown.

cmds.addAttr("ctrl", ln="when", at="time")
ctrl.when << 24
print(at("ctrl.when"), ctrl.when >> None)                           # time 24.0
```

---

## 4. Typed — `String` `Matrix` `Message` `Mesh` `NurbsCurve` `NurbsSurface`

`Matrix` and `Message` are attribute types; the other four are data
types, which `attributeQuery` reports as `typed` and `getAttr` by name.

```python
ctrl << String("label") << "hello"
print(dt("ctrl.label"), ctrl.label >> None)                         # string hello

ctrl << Matrix("xf")
print(at("ctrl.xf"), (ctrl.xf >> None).shape)                       # matrix (4, 4)
ctrl.xf << drv.worldMatrix                                          # a plug connects
print(ctrl.xf.get_inputs())                                         # PlugList([Plug("drv.worldMatrix")])
ctrl.xf << None
ctrl.xf << np.eye(4) * 2                                            # an array sets
print((ctrl.xf >> None)[0, 0])                                      # 2.0

ctrl << Message("driver")
ctrl.driver << drv.message
print(at("ctrl.driver"), ctrl.driver.get_inputs())                  # message PlugList([Plug("drv.message")])
```

The geometry data types are wires for shapes:

```python
cmds.polyCube(name="cube")
ctrl << Mesh("shapeIn")
ctrl.shapeIn << Node("cubeShape").outMesh
print(dt("ctrl.shapeIn"), ctrl.shapeIn.get_inputs())                # mesh PlugList([Plug("cubeShape.outMesh")])

ctrl << NurbsCurve("crvIn")
ctrl << NurbsSurface("srfIn")
print(dt("ctrl.crvIn"), dt("ctrl.srfIn"), at("ctrl.crvIn"))         # nurbsCurve nurbsSurface typed
```

---

## 5. `Enum`

`en=` (or `enumName=`) takes a list or a colon-delimited string; the
default is a two-state `False:True`. Values are indices; `plug.enums`
lists the names.

```python
ctrl << Enum("mode", en=["off", "on", "auto"]) << 2
print(cmds.attributeQuery("mode", node="ctrl", listEnum=True), ctrl.mode >> None)   # ['off:on:auto'] 2
print(ctrl.mode.enums, cmds.getAttr("ctrl.mode", asString=True))                    # ['off', 'on', 'auto'] auto

ctrl << Enum("side", en="L:R:")                                     # trailing colon optional
ctrl << Enum("toggle")
ctrl << Enum("axis", en=["x", "y", "z"], dv=2)
print(cmds.attributeQuery("side", node="ctrl", listEnum=True), cmds.attributeQuery("toggle", node="ctrl", listEnum=True), ctrl.axis >> None)   # ['L:R'] ['False:True'] 2

try:
    ctrl.mode << "on"
except Exception as e:
    print(type(e).__name__)                                         # InjectionError -- set by index, not by name
```

---

## 6. Compounds — `Vector` `Color` `Euler` `Quat`

All four subclass `Float`; the compound decides the child suffixes and,
for `Euler`, the child type.

| Spec | Parent | Children |
|---|---|---|
| `Vector` | `double3` | `X Y Z`, `double` |
| `Color` | `double3` | `R G B`, `double` |
| `Euler` | `double3` | `X Y Z`, `doubleAngle` |
| `Quat` | `double4` | `X Y Z W`, `double` |

```python
ctrl << Vector("aim")
ctrl << Color("tint")
ctrl << Euler("rot")
ctrl << Quat("q")

print(at("ctrl.aim"), cmds.attributeQuery("aim", node="ctrl", listChildren=True), at("ctrl.aimX"))    # double3 ['aimX', 'aimY', 'aimZ'] double
print(cmds.attributeQuery("tint", node="ctrl", listChildren=True))                                    # ['tintR', 'tintG', 'tintB']
print(at("ctrl.rot"), at("ctrl.rotX"))                                                                # double3 doubleAngle
print(at("ctrl.q"), ctrl.q.num_children)                                                              # double4 4
```

A compound takes a sequence, a plug of the same width, or a per-channel
list; it reads back as a NumPy array.

```python
ctrl.aim << [1, 2, 3]
print(ctrl.aim >> None, ctrl.aimX >> None)      # [1. 2. 3.] 1.0
ctrl.rot << [90, 0, 0]
print(ctrl.rot >> None)                         # [90.  0.  0.]
ctrl.q << [0, 0, 0, 1]
print(ctrl.q >> None)                           # [0. 0. 0. 1.]
ctrl.tint << drv.t
print(ctrl.tint.get_inputs())                   # PlugList([Plug("drv.translate")])
```

Per-child defaults are `defaultValue=[...]`; `min` / `max` land on the
children.

```python
ctrl << Vector("up", defaultValue=[0, 1, 0])
ctrl << Color("base", defaultValue=[0.5, 0.5, 1.0], min=0, max=1)
print(ctrl.up >> None, ctrl.base >> None)                                                                          # [0. 1. 0.] [0.5 0.5 1. ]
print(cmds.attributeQuery("baseR", node="ctrl", min=True), cmds.attributeQuery("baseR", node="ctrl", max=True))   # [0.0] [1.0]
```

`dv=[...]` on a compound is the one spelling that fails — and it leaves a
half-built parent that blocks the name (README, *Real behaviour, verified*). A scalar
`dv` on a compound is ignored.

```python
try:
    ctrl << Vector("broken", dv=[1, 0, 0])
except TypeError as e:
    print(str(e)[:33])                          # Invalid arguments for flag 'dv'.
ctrl << Vector("flat", dv=5)
print(ctrl.flat >> None)                        # [0. 0. 0.]
```

---

## 7. The kwargs

| Kwarg | Alias | Meaning | Handled by |
|---|---|---|---|
| `min`, `max` | | range; on a compound, the children's | `addAttr` |
| `dv` | `defaultValue` | default; a sequence per child needs the `defaultValue` spelling | `addAttr` |
| `keyable` | `k` | default `True`; `False` also drops it from the channel box | `addAttr` |
| `hidden` | `h` | hidden from the UI | `addAttr` |
| `sn`, `nn` | `shortName`, `niceName` | | `addAttr` |
| `en` | `enumName` | `Enum` names, list or colon string | `Enum` |
| `multi` | `m` | array attribute | the spec |
| `size` | | pre-create indices `0 .. size-1` on a multi | the spec |
| `overwrite` | | default `True`: replace an existing name; `False`: keep it | the spec |

Anything else (`writable`, `readable`, `storable`, `softMin`, ...) is an
`addAttr` flag and goes straight through.

```python
ctrl << Float("quiet", k=False)
ctrl << Float("secret", hidden=True)
ctrl << Float("blendIn", sn="bi", nn="Blend In", softMinValue=0.0, softMaxValue=1.0)

print(cmds.getAttr("ctrl.quiet", keyable=True), cmds.getAttr("ctrl.quiet", channelBox=True))     # False False
print(cmds.attributeQuery("secret", node="ctrl", hidden=True))                                    # True
print(cmds.attributeQuery("blendIn", node="ctrl", shortName=True), cmds.attributeQuery("blendIn", node="ctrl", niceName=True))   # bi Blend In
print(cmds.attributeQuery("blendIn", node="ctrl", softMax=True))                                  # [1.0]
```

A long name that matches a built-in **short** name is refused by Maya
(`c` is `center` on a transform), and the spec's existence check cannot
see it coming:

```python
try:
    ctrl << Float("c")
except RuntimeError as e:
    print(e)                                    # Found no valid items to add the attribute to.
```

---

## 8. Modifiers — `lock` `unlock` `hide` `unhide` `skip`

Specs with no name: their kwargs are `setAttr` flags, they edit the plug
on the left and return it.

```python
print(lock.kargs, unlock.kargs, hide.kargs, unhide.kargs, skip.kargs)   # {'lock': True} {'lock': False} {'keyable': False, 'channelBox': False} {'keyable': True, 'channelBox': False} {}
print(repr(skip), repr(destroy))                                       # <skip> <destroy>
```

```python
ctrl.tx << 5 << lock
print(cmds.getAttr("ctrl.tx", lock=True))       # True
try:
    ctrl.tx << 6
except Exception as e:
    print(type(e).__name__, str(e)[:57])        # InjectionError Cannot set/connect 'ctrl.translateX': locked attribute(s)
ctrl.tx << unlock << 6
print(ctrl.tx >> None)                          # 6.0

ctrl.ty << hide
print(cmds.getAttr("ctrl.ty", keyable=True), cmds.getAttr("ctrl.ty", channelBox=True))   # False False
ctrl.ty << unhide
print(cmds.getAttr("ctrl.ty", keyable=True), cmds.getAttr("ctrl.ty", channelBox=True))   # True False
```

On a compound, `hide` / `unhide` fan out to the children and `lock` sits
on the parent (the children inherit it):

```python
ctrl.t << hide
print([cmds.getAttr(f"ctrl.t{c}", keyable=True) for c in "xyz"])        # [False, False, False]
ctrl.t << unhide
ctrl.r << lock
print(cmds.getAttr("ctrl.r", lock=True), cmds.getAttr("ctrl.rx", lock=True))   # True True
ctrl.r << unlock
```

Inside a per-channel list a modifier applies to its channel, `skip`
leaves the channel alone, and `None` disconnects it:

```python
ctrl.s << [lock, skip, hide]
print(cmds.getAttr("ctrl.sx", lock=True), cmds.getAttr("ctrl.sz", keyable=True))   # True False
ctrl.s << [unlock, skip, unhide]

ctrl.tx << drv.tx
ctrl.tz << drv.tz
ctrl.t << [skip, 4.0, skip]                     # write Y, keep the X and Z drivers
print(ctrl.tx.get_inputs(), ctrl.t >> None)     # PlugList([Plug("drv.translateX")]) [0. 4. 0.]
ctrl.t << [None, 4.0, None]                     # write Y, DISCONNECT X and Z
print(ctrl.tx.get_inputs(), ctrl.tz.get_inputs())   # PlugList([]) PlugList([])
```

---

## 9. `destroy`

One symbol, two spellings: `plug << destroy` for a plug you hold,
`node << destroy("a", "b", ...)` for names. Both follow `cmds.deleteAttr`:
connections are auto-severed (one INFO line on the `rig._internal.plug`
logger), the node form returns the node.

```python
ctrl << Float("foo") << 1
ctrl.foo << destroy
print(cmds.attributeQuery("foo", node="ctrl", exists=True))          # False

ctrl << Float("bar"); ctrl << Float("baz")
print(repr(ctrl << destroy("bar", "baz")), cmds.attributeQuery("baz", node="ctrl", exists=True))   # Node("ctrl") False
```

Keyword flags:

| Flag | Effect |
|---|---|
| `strict=True` | refuse (`RuntimeError`) when the attribute has any connection; nothing is deleted |
| `silent=True` | a missing name is a no-op, so a cleanup can run twice |
| `verbose=True` | log every severed connection by name instead of a count |

```python
ctrl << Float("scratch")
drv.ty << ctrl.scratch
try:
    ctrl << destroy("scratch", strict=True)
except RuntimeError as e:
    print(str(e).splitlines()[0])               # Cannot destroy 'ctrl.scratch': has 0 incoming + 1 outgoing connection(s):
print(cmds.attributeQuery("scratch", node="ctrl", exists=True))          # True -- untouched
ctrl << destroy("scratch", verbose=True)            # logs: destroy ctrl.scratch: disconnecting ctrl.scratch -> drv.translateY
print(drv.ty.get_inputs())                      # PlugList([])

ctrl << destroy("nope", silent=True)            # no-op
try:
    ctrl << destroy("nope")
except AttributeError as e:
    print(str(e)[:48])                          # Cannot destroy nonexistent attribute 'ctrl.nope'
```

A compound parent cascades; a child alone, a locked attribute and a
static attribute are refused by Maya. The wrong spelling is a `TypeError`
that names the right one.

```python
ctrl << Vector("gone")
ctrl << destroy("gone")
print(cmds.attributeQuery("goneX", node="ctrl", exists=True))        # False

for bad in (lambda: ctrl.aimX << destroy,
            lambda: ctrl << destroy("translate"),
            lambda: ctrl << destroy("weight")):                        # weight is locked from section 2
    try:
        bad()
    except RuntimeError as e:
        print(str(e).splitlines()[0][:48])
# Cannot delete child 'ctrl.aimX' of compound attr
# Cannot delete static attribute 'ctrl.translate'
# 'ctrl.weight' is locked and may not be removed.

for bad in (lambda: ctrl << destroy, lambda: ctrl.aim << destroy("aim"),
            lambda: destroy(), lambda: destroy("x", strict=True, silent=True)):
    try:
        bad()
    except (TypeError, ValueError) as e:
        print(type(e).__name__, str(e)[:40])
# TypeError 'node << destroy' is ambiguous (no attr
# TypeError destroy(...) is for Nodes, not Plugs. To
# ValueError destroy() requires at least one attr nam
# ValueError destroy(...): strict=True and silent=Tru
```

---

## 10. `overwrite=`

Default `True`: an existing attribute of that name is unlocked, deleted
(connections and value with it) and rebuilt from the spec. `False`: the
existing attribute stays and its plug is returned, whatever its type.

```python
ctrl << Float("mix", min=0, max=1) << 0.75
drv.tz << ctrl.mix
ctrl << Float("mix", min=-5, max=5)
print(ctrl.mix >> None, cmds.attributeQuery("mix", node="ctrl", min=True), drv.tz.get_inputs())   # 0.0 [-5.0] PlugList([])

ctrl.mix << 0.3 << lock
ctrl << Float("mix", dv=2.0)
print(ctrl.mix >> None, cmds.getAttr("ctrl.mix", lock=True))                                     # 2.0 False

ctrl.mix << 0.3
kept = ctrl << Float("mix", min=0, max=1, overwrite=False)
print(kept >> None, cmds.attributeQuery("mix", node="ctrl", minExists=True))                    # 0.3 False
print(repr(ctrl << String("mix", overwrite=False)), at("ctrl.mix"))                              # Plug("ctrl.mix") double -- no type check
```

---

## 11. `node >> spec` declares an output

The mirror of `<<`: the same spec, applied with `writable=False` on a
copy (the caller's spec is untouched). Nothing can connect into it;
`setAttr` still can, and anything can read it or connect from it.

```python
out = ctrl >> Float("result")
print(repr(out), cmds.attributeQuery("result", node="ctrl", writable=True))   # Plug("ctrl.result") False
out << 5
drv.sx << out
print(out >> None, drv.sx.get_inputs())         # 5.0 PlugList([Plug("ctrl.result")])
try:
    out << drv.tx
except Exception as e:
    print(type(e).__name__)                     # InjectionError

m = Matrix("outMatrix")
ctrl >> m
print("writable" in m.kargs)                    # False -- the spec was copied, not stamped
ctrl >> Vector("outVec")
print(cmds.attributeQuery("outVecX", node="ctrl", writable=True))   # False -- children too
```

`>> None` and anything else keep their old meaning:

```python
print(type(ctrl >> None).__name__)              # Transform -- the typed DGNode
try:
    ctrl >> 42
except TypeError as e:
    print(str(e)[:23])                          # '>>' on a Node supports
```

---

## 12. `plug >> Node` and `plug >> "name"` clone

`plug >> node` reads the plug's type and adds the equivalent attribute
under the same name. What travels: type, compound shape, enum names,
multi-ness (populated indices and their values). What does not: `min`,
`max`, `dv`, the value, connections.

```python
cmds.file(new=True, force=True)
src = Node.create("transform", name="src")
dst = Node.create("transform", name="dst")
src << Float("mix", min=0, max=1) << 0.4
src << Enum("side", en=["L", "R"]) << 1
src << Euler("rot")
src << Float("w", multi=True)
src.w[0] << 1; src.w[2] << 3

c = src.mix >> dst
print(repr(c), c >> None, cmds.attributeQuery("mix", node="dst", minExists=True))   # Plug("dst.mix") 0.0 False
src.side >> dst
src.rot >> dst
print(cmds.attributeQuery("side", node="dst", listEnum=True), at("dst.rotX"))       # ['L:R'] doubleAngle
src.w >> dst
print(cmds.getAttr("dst.w", multiIndices=True), cmds.getAttr("dst.w[2]"))           # [0, 2] 3.0
```

The named forms — `plug >> "name"` on the same node, `plug >> "node.name"`
on another — are the same clone under a chosen name, plus the current
value, and they refuse a taken name. `plug >> Node` does not: it
replaces a same-named dynamic attribute.

```python
print(repr(src.mix >> "mix2"), cmds.getAttr("src.mix2"))            # Plug("src.mix2") 0.4
print(repr(src.mix >> "dst.mixCopy"), cmds.getAttr("dst.mixCopy"))  # Plug("dst.mixCopy") 0.4
try:
    src.mix >> "dst.mix"
except TypeError as e:
    print(str(e)[:50])                                              # 'dst' already has an attribute 'mix': '>>' clones,

dst.mix << 0.9
src.mix >> dst
print(cmds.getAttr("dst.mix"))                                      # 0.0 -- replaced, value reset
```

Built-ins clone too, and `Color` comes back as a `Vector`:

```python
print(repr(src.rotateOrder >> "roCopy"), cmds.attributeQuery("roCopy", node="src", listEnum=True))   # Plug("src.roCopy") ['xyz:yzx:zxy:xzy:yxz:zyx']
print(repr(src.t >> "tCopy"), at("src.tCopy"))                                                       # Plug("src.tCopy") double3
src << Color("tint") << [0.1, 0.2, 0.3]
src.tint >> dst
print(cmds.attributeQuery("tint", node="dst", listChildren=True))                                    # ['tintX', 'tintY', 'tintZ']
```

---

## 13. Multis and pre-sizing

`multi=True` (or `m=True`) makes an array. Maya creates indices only when
they are written, so `size=n` writes the default into `0 .. n-1` right
away; without it a slice is empty until something lands.

```python
cmds.file(new=True, force=True)
ctrl = Node.create("transform", name="ctrl")

ctrl << Float("weights", multi=True, size=3)
print(ctrl.weights.is_multi, cmds.getAttr("ctrl.weights", multiIndices=True))   # True [0, 1, 2]
ctrl.weights[1] << 0.5
print(ctrl.weights[:] >> None)                                                  # [0.  0.5 0. ]
ctrl.weights[:] << [1, 2, 3]
ctrl.weights.append(9.0)
print(cmds.getAttr("ctrl.weights", multiIndices=True), ctrl.weights[3] >> None)   # [0, 1, 2, 3] 9.0

ctrl << Float("lazy", multi=True)
print(cmds.getAttr("ctrl.lazy", multiIndices=True))                             # None
ctrl.lazy[4] << 1.0
print(cmds.getAttr("ctrl.lazy", multiIndices=True))                             # [4]
```

Any spec takes `multi`; the pre-size default is `dv`, else `0`, which a
`Matrix` turns into identity and a `String` into `'0'`.

```python
ctrl << Float("dvm", multi=True, size=2, dv=7.0)
ctrl << Vector("offsets", multi=True, size=2)
ctrl << Matrix("mats", multi=True, size=2)
ctrl << String("names", multi=True, size=2)
print(ctrl.dvm[:] >> None, cmds.getAttr("ctrl.offsets", multiIndices=True))   # [7. 7.] [0, 1]
print((ctrl.mats[:] >> None).shape, cmds.getAttr("ctrl.names[0]"))            # (2, 4, 4) 0
```

---

## 14. `PlugList` fan-out

A spec injected into a `PlugList` is applied to every element's node
(a plug element means its node); the result is a `PlugList` of the new
plugs, so values, modifiers and `>> None` broadcast next.

```python
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")
nodes = PlugList([a, b])

gains = nodes << Float("gain", min=0, max=2) << [0.5, 1.5] << lock
print(repr(gains), gains >> None)                                     # PlugList([Plug("a.gain"), Plug("b.gain")]) [0.5 1.5]
print([cmds.getAttr(f"{n}.gain", lock=True) for n in ("a", "b")])    # [True, True]

aims = nodes << Vector("aim") << [[1, 0, 0], [0, 1, 0]]
print(aims >> None)                                                   # [[1. 0. 0.]
                                                                      #  [0. 1. 0.]]
print(repr(nodes >> Float("out")), cmds.attributeQuery("out", node="b", writable=True))   # PlugList([Plug("a.out"), Plug("b.out")]) False
print(repr(PlugList([a.tx, b.ty]) << Float("viaPlug")))               # PlugList([Plug("a.viaPlug"), Plug("b.viaPlug")])
```

Modifiers pair element-wise, `destroy` returns the nodes, and a list of
specs pairs but returns the **left-hand side**, not the new plugs:

```python
nodes.tx << [lock, hide]
print(cmds.getAttr("a.tx", lock=True), cmds.getAttr("b.tx", keyable=True))   # True False
nodes.tx << [unlock, unhide]

print(repr(nodes << [Float("fa"), Float("fb")]))                     # PlugList([Node("a"), Node("b")])
print(cmds.attributeQuery("fa", node="a", exists=True), cmds.attributeQuery("fa", node="b", exists=True))   # True False

gains << unlock
print(repr(nodes << destroy("gain")), cmds.attributeQuery("gain", node="a", exists=True))   # PlugList([Node("a"), Node("b")]) False
try:
    PlugList([a, 3.5]) << Float("x")                                 # a string that names a node would lift to it
except TypeError as e:
    print(str(e)[:36])                                               # element [1] (3.5) is not a Plug or a
```

---

## 15. `Note`

Adds Maya's `notes` string attribute — hidden, non-keyable,
`writable=False` — and sets the text when given one. A second `Note`
replaces the first.

```python
n = ctrl << Note("Built by the spine module.")
print(repr(n), cmds.getAttr("ctrl.notes"))                                                  # Plug("ctrl.notes") Built by the spine module.
print(cmds.getAttr("ctrl.notes", keyable=True), cmds.attributeQuery("notes", node="ctrl", hidden=True))   # False True
ctrl << Note("second")
print(cmds.getAttr("ctrl.notes"))                                                           # second
print(repr(a << Note()), cmds.getAttr("a.notes"))                                           # Plug("a.notes") None
```

---

## 16. Leading underscores

A `_`-prefixed name is an ordinary attribute once the node has it; only
names the node does **not** have keep raising `AttributeError`, which is
what keeps Python's own probes cheap.

```python
parked = ctrl << Float("__parked__") << 3.5
print(repr(parked), ctrl.__parked__ >> None)        # Plug("ctrl.__parked__") 3.5
ctrl.__parked__ = 4.0                               # assignment is `<<` sugar, here too
print(cmds.getAttr("ctrl.__parked__"))              # 4.0
print(repr(ctrl.tx >> "__tx__"), ctrl.__tx__ >> None)   # Plug("ctrl.__tx__") 0.0 -- parking spelled with a clone
try:
    ctrl._nope
except AttributeError as e:
    print(e)                                        # _nope
```

---

## Where to go next

| Read | For |
|---|---|
| [`README.md`](README.md) | the concepts, the conventions, the verified behaviour |
| [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) | the whole DSL: operators, `PlugList` broadcasting, containers, membership |
| [`../maya/README.md`](../maya/README.md) | `Attribute` and `DGNode`, the layer every spec lands on |
