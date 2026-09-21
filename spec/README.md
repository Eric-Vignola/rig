# `rig.spec` — attribute specs

The attribute half of [`rig`](../README.md). A spec is a small, lazy
description of a Maya attribute — its name, type, range, default and any
other `cmds.addAttr` flag — that does nothing until it meets `<<`. Inject
it into a node and the attribute exists; the spec hands back the new plug
so the value goes next.

Every block on this page runs top to bottom as one script, under `mayapy`
or inside Maya. This is the setup:

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)
```

```python
from rig import Node
from rig.spec import Float, Vector, Enum, lock, hide

ctrl   = Node.create("transform", name="ctrl")

weight = ctrl << Float("weight", min=0, max=1) << 0.5 << lock
print(repr(weight), weight >> None, cmds.getAttr("ctrl.weight", lock=True))   # Plug("ctrl.weight") 0.5 True

ctrl << Vector("aim")
ctrl << Enum("mode", en=["off", "on", "auto"]) << 2
ctrl << Float("blend", multi=True, size=3)
print(cmds.listAttr("ctrl", userDefined=True))
# ['weight', 'aim', 'aimX', 'aimY', 'aimZ', 'mode', 'blend']
```

Four `addAttr` calls, three `setAttr` calls, one lock — and nothing
happened until each spec reached `<<`.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste an example of every spec class, kwarg and modifier | [`CHEATSHEET.md`](CHEATSHEET.md) — runnable top to bottom |
| The operator table, the conventions, the rest of the DSL | [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) |
| The typed node layer a spec lands on (`Attribute`, `DGNode`) | [`../maya/README.md`](../maya/README.md) |

---

## What's inside

| File | Contents |
|---|---|
| `__init__.py` | re-exports the 23 public names below |
| `_base.py` | `_AttrSpec` — the base class; `apply()`, the compound builder, multi pre-sizing, and `_clone_attribute` behind `plug >> Node` |
| `numeric.py` | `Float`, `Int`, `Bool`, `Angle`, `Time` |
| `compound.py` | `Vector`, `Quat`, `Color`, `Euler` — 3- and 4-channel compounds |
| `typed.py` | `String`, `Matrix`, `Mesh`, `NurbsCurve`, `NurbsSurface`, `Message` |
| `enum_attr.py` | `Enum` (the file avoids shadowing the stdlib `enum`) |
| `modifiers.py` | `lock`, `unlock`, `hide`, `unhide`, `skip`, `destroy`, `Note` |

The public surface is what `__init__.py` re-exports, and every one of
those names is re-exported again at the top of the package, so both
spellings are the same object:

```python
import rig
from rig.spec import (
    Angle, Bool, Float, Int, Time,                    # numeric
    Color, Euler, Quat, Vector,                       # compound
    Matrix, Mesh, Message, NurbsCurve, NurbsSurface, String,   # typed
    Enum,
    Note, destroy, hide, lock, skip, unhide, unlock,  # modifiers
)
print(rig.Float is Float, rig.lock is lock, rig.destroy is destroy)  # True True True
print(len(rig.spec.__all__))                                         # 23
```

### Names that collide on purpose

The top-level re-export puts a few spec names next to things that sound
alike. The capitalised name is always the attribute spec.

| Name | Is | Not to be confused with |
|---|---|---|
| `rig.Color` | an RGB `double3` attribute spec (children `R`, `G`, `B`) | a colour value, or a vertex colour set |
| `rig.Mesh` | a `mesh` data attribute spec | `rig.maya.nodetypes.Mesh`, the typed shape node |
| `rig.Matrix` | a `matrix` attribute spec | `rig.matrix`, the matrix function library |
| `rig.Vector`, `rig.Euler`, `rig.Quat` | compound attribute specs | `rig.vector`, `rig.euler`, `rig.quaternion`, the function libraries |

```python
from rig.maya.nodetypes import Mesh as MeshNode
print(rig.Mesh is MeshNode, rig.Matrix is rig.matrix)   # False False
```

---

## Concepts

### A spec is a bag of `addAttr` kwargs

The positional argument is the long name. Every keyword is an `addAttr`
flag, short or long spelling, with three exceptions the spec keeps for
itself: `multi` / `m`, `size` and `overwrite`. `keyable=True` is filled
in for you.

```python
spec = Float("weight", min=0, max=1, dv=0.5)
print(spec.kargs)                                               # {'min': 0, 'max': 1, 'dv': 0.5, 'keyable': True, 'longName': 'weight', 'attributeType': 'double'}
print(spec.size, spec.overwrite, spec.compound)                 # None True None
print(cmds.attributeQuery("weight", node="ctrl", exists=True))  # True -- from the taste above, not from this spec
```

Nothing has been created. The same spec can be injected into as many
nodes as you like; `>>` copies it before stamping anything on it, so a
spec is never mutated by use.

```python
a = Node.create("transform", name="a")
b = Node.create("transform", name="b")
a << spec
b << spec
print(cmds.getAttr("a.weight"), cmds.getAttr("b.weight"))   # 0.5 0.5
```

### `<<` adds the attribute and returns the new plug

`<<` normally returns its left-hand side, which is what makes
`obj.t << [1, 2, 3] << lock` chain. An attribute spec is the one
exception: `node << Float("x")` returns the **new plug**, because the
next thing you inject is its value, and after the value, a modifier.

```python
plug = ctrl << Float("gain") << 2.0 << hide
print(repr(plug), plug >> None, cmds.getAttr("ctrl.gain", keyable=True))   # Plug("ctrl.gain") 2.0 False
```

A plug on the left works too and means its node: `ctrl.tx << Float("y")`
adds `y` to `ctrl`.

### `overwrite=` decides what an existing name means

By default a spec **replaces** an attribute that already has its name:
the old one is unlocked, deleted (severing its connections) and rebuilt
from the spec, so the value goes back to the default. `overwrite=False`
keeps the existing attribute and returns its plug, without checking that
the types agree.

```python
ctrl << Float("blend", min=0, max=1) << 0.75
ctrl << Float("blend", min=-5, max=5)
print(cmds.getAttr("ctrl.blend"), cmds.attributeQuery("blend", node="ctrl", min=True))   # 0.0 [-5.0]

ctrl.blend << 0.3
same = ctrl << Float("blend", min=0, max=1, overwrite=False)
print(same >> None, cmds.attributeQuery("blend", node="ctrl", min=True))                 # 0.3 [-5.0]
```

### Modifiers are specs with no name

`lock`, `unlock`, `hide`, `unhide` are plain `_AttrSpec` instances whose
kwargs are `setAttr` flags instead of `addAttr` flags; injected into a
plug they edit it and return it. `skip` and `destroy` are sentinels.

| Modifier | Does | On a compound |
|---|---|---|
| `lock` / `unlock` | `setAttr(lock=True / False)` | the parent; children inherit the lock |
| `hide` | `keyable=False, channelBox=False` | fans out to every child |
| `unhide` | `keyable=True, channelBox=False` | fans out to every child |
| `skip` | nothing — leave the target as it is | |
| `destroy` | delete the plug (`plug << destroy`) or named attrs (`node << destroy("a", "b")`) | a parent cascades to its children; a child alone is refused by Maya |

`skip` earns its keep inside a per-channel fan-out, where `None` means
*disconnect*:

```python
drv = Node.create("transform", name="drv")
ctrl.tx << drv.tx
ctrl.tz << drv.tz
ctrl.t  << [skip, 4.0, skip]                     # write Y; X and Z keep their drivers
print(ctrl.tx.get_inputs(), ctrl.t >> None)     # PlugList([Plug("drv.translateX")]) [0. 4. 0.]
ctrl.t << [None, 4.0, None]                     # write Y; X and Z are DISCONNECTED
print(ctrl.tx.get_inputs())                     # PlugList([])
```

`destroy` follows `cmds.deleteAttr`: connections are auto-severed (one
INFO line on the `rig._internal.plug` logger, one per connection with
`verbose=True`), `strict=True` refuses to delete a connected attribute,
`silent=True` makes a missing name a no-op. A locked attribute and a
static one (`translate`) are refused by Maya either way.

```python
from rig.spec import destroy

ctrl   << Float("scratch")
drv.ty << ctrl.scratch
try:
    ctrl << destroy("scratch", strict=True)
except RuntimeError as e:
    print(str(e).splitlines()[0])               # Cannot destroy 'ctrl.scratch': has 0 incoming + 1 outgoing connection(s):
ctrl.scratch << destroy                             # the plug form; auto-disconnects drv.ty
print(cmds.attributeQuery("scratch", node="ctrl", exists=True), drv.ty.get_inputs())   # False PlugList([])
```

### `node >> Float("x")` declares an output

The operator points the way data flows: `<<` adds an input,
`>>` adds an output, the same spec with `writable=False` stamped on a
copy. Nothing can connect *into* it; anything can read it or connect
*from* it; and, Maya's rule, `setAttr` still works, so it is a place for
results and constants rather than a lock.

```python
result = ctrl >> Float("result")
print(repr(result), cmds.attributeQuery("result", node="ctrl", writable=True))   # Plug("ctrl.result") False
result << 5
drv.sx << result
print(drv.sx.get_inputs())                      # PlugList([Plug("ctrl.result")])
try:
    result << drv.tx
except Exception as e:
    print(type(e).__name__)                     # InjectionError
```

`node >> Float("x")` declares; `node >> Tag("x")` (a collection spec)
queries. The right-hand family decides, as the
[root README](../README.md) puts it.

### `plug >> Node` clones the attribute

`>>` with a node on the right reads the plug's type and adds an
equivalent attribute to that node under the same name. Type, compound
shape, enum names and multi-ness travel; **min, max, default, value and
connections do not**. A multi's populated indices are mirrored, values
included. `plug >> "name"` and `plug >> "other.name"` are the same clone
under a chosen name plus the current value, and they refuse a name the
target already has.

```python
src = Node.create("transform", name="src")
dst = Node.create("transform", name="dst")
src << Float("mix", min=0, max=1) << 0.4
src << Enum("side", en=["L", "R"]) << 1

print(repr(src.mix >> dst), cmds.getAttr("dst.mix"), cmds.attributeQuery("mix", node="dst", minExists=True))  # Plug("dst.mix") 0.0 False
print(repr(src.mix >> "mix2"), cmds.getAttr("src.mix2"))                                                      # Plug("src.mix2") 0.4
src.side >> dst
print(cmds.attributeQuery("side", node="dst", listEnum=True))                                                  # ['L:R']
try:
    src.mix >> "dst.mix"
except TypeError as e:
    print(str(e)[:50])                                                                                         # 'dst' already has an attribute 'mix': '>>' clones,
```

### Multis and pre-sizing

`multi=True` makes an array attribute. Maya arrays have no indices until
something is written to them, so `size=n` writes the default into
`[0 .. n-1]` right away and slicing works immediately.

```python
ctrl << Float("weights", multi=True, size=3)
print(cmds.getAttr("ctrl.weights", multiIndices=True), ctrl.weights[:] >> None)   # [0, 1, 2] [0. 0. 0.]
ctrl << Vector("offsets", multi=True)
print(cmds.getAttr("ctrl.offsets", multiIndices=True))                            # None -- no size, no indices yet
```

### `PlugList` fans a spec out

A spec injected into a `PlugList` is applied to every element's node,
and the result is a `PlugList` of the new plugs, so the value and the
modifiers broadcast next.

```python
from rig import PlugList

nodes = PlugList([a, b])
gains = nodes << Float("gain", min=0, max=2) << [0.5, 1.5] << lock
print(repr(gains), gains >> None)   # PlugList([Plug("a.gain"), Plug("b.gain")]) [0.5 1.5]
print(repr(nodes >> Float("out")))  # PlugList([Plug("a.out"), Plug("b.out")])
```

### Leading underscores are fine

`node << Float("__parked__")` adds it and `node.__parked__` reads it:
the node only refuses `_`-prefixed names it does not have, so Python's
own probes (`__deepcopy__`) stay cheap and a private attribute is a
normal attribute.

```python
parked = ctrl << Float("__parked__") << 3.5
print(repr(parked), ctrl.__parked__ >> None)   # Plug("ctrl.__parked__") 3.5
```

---

## Conventions

- **The positional argument is the long name.** `sn=` / `nn=` and every
  other `addAttr` flag pass straight through. `keyable=True` is the
  default; `k=False` makes a hidden-from-channel-box attribute, `hidden=True`
  hides it from the UI entirely.
- **`min` / `max` / `dv` (or `defaultValue`)** are `addAttr`'s own. On a
  compound they land on the children; a per-child default is a sequence
  passed as **`defaultValue=[...]`** (see *Real behaviour, verified* for
  `dv=[...]`).
- **Compound children are name + suffix**: `Vector("aim")` is `aim` plus
  `aimX aimY aimZ` (`double3`), `Color` is `R G B`, `Quat` is `X Y Z W`
  (`double4`), `Euler` is `X Y Z` with each child a `doubleAngle`.
- **Angles are degrees** on the node, as everywhere in Maya's UI units:
  `ctrl.twist << 90` then `ctrl.twist >> None` is `90.0`.
- **Enums are set by index.** `en=` takes a list or a colon string;
  the value is the position, and a string is an `InjectionError`.
  `plug.enums` lists the names.
- **Modifiers return the plug they edited**, so they can sit anywhere in
  a chain and inside a per-channel list: `ctrl.s << [lock, skip, hide]`.
- **Every spec name is also at `rig.*`**; `rig.spec.*` is the home.
- **`destroy` is two spellings of one thing**: `plug << destroy` for a
  plug you hold, `node << destroy("a", "b", ...)` for names. `node <<
  destroy` (no names) and `plug << destroy("a")` are `TypeError`s that
  name the other form.

---

## Real behaviour, verified

Verified on Maya 2025; not bugs to work around blindly.

- **`Time` cannot be injected.** It asks `addAttr` for
  `dataType="time"`, and `time` is an *attribute* type, so injection
  raises `RuntimeError: Type specified for new attribute's data type is
  unknown`. Cloning a time plug fails the same way. Until that is fixed,
  `cmds.addAttr(node, ln="when", at="time")` and then `node.when` works
  like any other plug.
- **A per-child default is `defaultValue=[...]`, not `dv=[...]`.**
  `Vector("aim", dv=[1, 0, 0])` forwards the list to every child next to
  its scalar default and `addAttr` raises `TypeError` — *after* creating
  the parent. That half-built parent is invisible to `attributeQuery`,
  `listAttr`, `deleteAttr` and `destroy`, yet blocks the name on that
  node for the rest of the session. Use `defaultValue=[1, 0, 0]`. A
  scalar `dv=5` on a compound is silently ignored.
- **A `>> Node` clone is type-only.** `min`, `max`, `dv`, the value and
  the keyable state are not carried (only the named forms copy the
  value). And where `plug >> "name"` refuses an existing name, `plug >>
  Node` **replaces** a same-named dynamic attribute on the target
  (`overwrite=True` in the spec it builds), resetting its value.
  Cloning a `PlugList` of same-named plugs onto one node therefore keeps
  only the last.
- **`Color` clones back as a `Vector`.** The clone sees three `double`
  children and builds `X Y Z`, so `src.tint >> dst` gives `dst.tintX`,
  not `dst.tintR`.
- **A name that matches a built-in short name fails inside `addAttr`.**
  `Float("c")` on a transform (`c` is `center`; `tmp` is `template`) is a
  `RuntimeError: Found no valid items to add the attribute to`. The spec's own
  existence check looks at long names only, so `overwrite` never
  triggers.
- **A compound child cannot be destroyed alone**: `ctrl.aimX << destroy`
  is Maya's `Cannot delete child 'ctrl.aimX' of compound attribute 'aim'`.
  Destroy the parent.
- **Pre-sizing writes the default into every index**, which for a
  `String` multi is the string `'0'`.
- **`nodes << [Float("a"), Float("b")]`** (a list of specs on a
  `PlugList`) pairs each element with its own spec but returns the
  **nodes**, not the new plugs; only a single spec broadcast returns
  plugs.
- **`unhide` leaves `channelBox=False`.** It sets `keyable=True`, which
  is what shows the attribute again; a non-keyable-but-displayed state
  needs `plug.set(channelBox=True)` by hand.
- **`overwrite=True` is destructive by design**: value reset, outgoing
  and incoming connections severed, a lock removed first. Reach for
  `overwrite=False` in code that may run twice.
