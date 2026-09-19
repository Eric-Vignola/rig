# `rig.bridges` — cheatsheet

One runnable block per behaviour of `rig.bridges.commands` (`rc`) and
`rig.bridges.nodes` (`rn`), plus `Node.wrap`. The blocks run top to bottom as
one script and share a namespace; the imports in **Setup** are reused
everywhere below. Any section that needs a clean scene resets it first.

Concepts and the file map live in [`README.md`](README.md).

## Contents

| # | Section | Covers |
|---|---|---|
| — | [Setup](#setup) | mayapy / Maya init, the three imports |
| 1 | [Import surface](#1-import-surface) | lazy wrappers, caching, `dir()`, unknown names |
| 2 | [Commands return `Node` and `PlugList`](#2-commands-return-node-and-pluglist) | `createNode`, `polyCube`, `ls`, `listRelatives`, empty results, broadcasting |
| 3 | [Values pass through](#3-values-pass-through) | `getAttr`, `objExists`, `nodeType`, `xform`, `listAttr`, the two wrap traps |
| 4 | [Arguments go in as strings](#4-arguments-go-in-as-strings) | `Node` / list-of-`Node` / `Plug` arguments, in args and kwargs |
| 5 | [The `_NO_COERCE` list](#5-the-_no_coerce-list) | nine commands whose arguments reach Maya untouched |
| 6 | [Every result joins the active container](#6-every-result-joins-the-active-container) | the query gotcha, `container=False`, what refuses |
| 7 | [`Node.wrap` for manual use](#7-nodewrap-for-manual-use) | converting raw `maya.cmds` results |
| 8 | [Factories: the create kwargs](#8-factories-the-create-kwargs) | `name` / `n`, `parent` / `p`, `shared` / `s`, `skipSelect` / `ss` |
| 9 | [Attribute kwargs go through `<<`](#9-attribute-kwargs-go-through-) | scalars, compounds, multis, plugs, matrices, specs come after |
| 10 | [Keyword collisions](#10-keyword-collisions) | `and_`, `or_`, `not_` |
| 11 | [Keyword-only signature](#11-keyword-only-signature) | `rn.blinn("x")` vs `rn.blinn(name="x")` |
| 12 | [Factories and the container scope](#12-factories-and-the-container-scope) | `container=False`, nested-scope name prefix |
| 13 | [Plugin node types and `_refresh_node_types`](#13-plugin-node-types-and-_refresh_node_types) | the cached type set, `rc.createNode` and `unknown` nodes |

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

from rig import Node, PlugList, container
from rig.bridges import commands as rc
from rig.bridges import nodes as rn
```

---

## 1. Import surface

`rig.bridges/__init__.py` is empty; import the two submodules by name. Every
`rc.<name>` is built on first access and cached, and keeps the real
`maya.cmds` function reachable as `__wrapped__`.

```python
print("ls" in dir(rc), "createNode" in dir(rc))               # True True
print("transform" in dir(rn), "plusMinusAverage" in dir(rn))  # True True
print(rc.ls is rc.ls, rc.ls.__wrapped__ is cmds.ls)           # True True -- built once, cached
print(rc.ls.__qualname__, rn.transform.__qualname__)          # rig.bridges.commands.ls rig.bridges.nodes.transform
```

A name that is not a `maya.cmds` callable, or not a registered node type, is
an `AttributeError` that says which:

```python
for probe in (lambda: rc.noSuchCommand, lambda: rn.noSuchNodeType):
    try:
        probe()
    except AttributeError as err:
        print(err)
# 'rig.bridges.commands' has no command 'noSuchCommand' (no such attribute in maya.cmds)
# 'rig.bridges.nodes' has no node type 'noSuchNodeType' (no such Maya nodetype registered). If you just loaded a plugin, call ``rig.bridges.nodes._refresh_node_types()``.
```

---

## 2. Commands return `Node` and `PlugList`

A string result is a `Node`, a list result is a `PlugList`; `str(node)` is
the real Maya name.

```python
cmds.file(new=True, force=True)
cube = rc.createNode("transform", name="cube")
print(repr(cube), str(cube))                     # Node("cube") cube
print(rc.polyCube(name="box"))                   # PlugList([Node("box"), Node("polyCube1")])
cmds.select("cube", "box")
sel = rc.ls(sl=True)
print(sel, type(sel).__name__)                   # PlugList([Node("cube"), Node("box")]) PlugList
print(rc.listRelatives("box", s=True))           # PlugList([Node("boxShape")])
```

Empty stays empty, and `None` stays `None` — whatever the command itself does:

```python
print(rc.ls("nothing_*"), rc.listRelatives("cube", p=True))   # PlugList([]) None
```

A `PlugList` broadcasts, so a query result takes `<<` directly:

```python
sel.ty << 2
print(sel.ty, cmds.getAttr("box.ty"))            # PlugList([Plug("cube.translateY"), Plug("box.translateY")]) 2.0
```

---

## 3. Values pass through

Booleans, numbers and strings that are not node names come back unchanged. A
list of values is still a `PlugList`; a list of non-node strings is a plain
`list`.

```python
cube.tx << 5
print(rc.getAttr("cube.tx"), rc.objExists("cube"), rc.nodeType("cube"))   # 5.0 True transform
print(rc.getAttr("cube.t"))                      # PlugList([(5.0, 2.0, 0.0)])
print(rc.xform("cube", q=True, ws=True, t=True)) # PlugList([5.0, 2.0, 0.0])
print(rc.listAttr("cube", k=True)[:2])           # ['visibility', 'translateX'] -- not node names: a plain list
print(rc.delete("box"))                          # None
```

Trap one: a string *value* that happens to name a node is wrapped as that
node.

```python
cmds.addAttr("cube", ln="label", dt="string")
cmds.setAttr("cube.label", "persp", type="string")
print(repr(rc.getAttr("cube.label")))            # Node("persp") -- the value names a node, so it is one
cmds.setAttr("cube.label", "hello", type="string")
print(repr(rc.getAttr("cube.label")))            # 'hello'
```

Trap two: plug strings collapse to their node (`Node("a.tx")` strips the
attribute), so `plugs=True` is lost. Ask the DSL, or raw `cmds`, for plugs.

```python
other = rc.createNode("transform", name="other")
other.tx << cube.tx
print(rc.listConnections("other.tx", p=True))    # PlugList([Node("cube")]) -- the attribute is gone
print(cmds.listConnections("other.tx", p=True))  # ['cube.translateX']
print(other.tx.get_inputs())                     # PlugList([Plug("cube.translateX")])
```

---

## 4. Arguments go in as strings

A `Node` becomes its name, a list or tuple holding any `Node` becomes a list
of names, in positional and keyword arguments alike. A `Plug` is already a
`str`.

```python
cmds.file(new=True, force=True)
root = rc.createNode("transform", name="root")
a    = rc.createNode("transform", name="a")
b    = rc.createNode("transform", name="b", parent=root)      # a Node in a kwarg
print(rc.parent(a, root))                                     # PlugList([Node("a")])
print(sorted(str(x) for x in rc.listRelatives(root, c=True))) # ['a', 'b']
print(rc.parent([a, b], world=True))                          # PlugList([Node("a"), Node("b")]) -- a list of Nodes
print(rc.getAttr(a.tx))                                       # 0.0 -- a Plug argument
```

---

## 5. The `_NO_COERCE` list

Nine commands carry callback strings, expressions or code to evaluate, so
their arguments reach Maya exactly as written. Pass names to those.

```python
from rig.bridges.commands import _NO_COERCE

print(sorted(_NO_COERCE))
# ['error', 'evalDeferred', 'expression', 'redo', 'scriptJob', 'scriptNode', 'undo', 'undoInfo', 'warning']
expr = rc.expression(s="a.ty = a.tx * 2;", o=str(a))         # str(node), not node
print(repr(expr))                                             # Node("expression1") -- results still wrap
a.tx << 3
print(cmds.getAttr("a.ty"))                                   # 6.0
```

In Maya 2025 `maya.cmds` stringifies a `Node` on its own, so `o=a` would also
have worked; the bridge does not rely on that anywhere else.

---

## 6. Every result joins the active container

Inside `with container(...)`, whatever a wrapper returns is added to the
container — created or merely looked up. A query captures the queried node.

```python
cmds.file(new=True, force=True)
ctrl = rc.createNode("transform", name="ctrl")           # made OUTSIDE any scope
mesh = rc.polyCube(name="mesh")[0]
with container("build") as ctn:
    driven = rc.createNode("transform", name="driven")   # created here: joins, as expected
    found  = rc.ls("ctrl")                               # a query -- and ctrl joins too
    shape  = rc.listRelatives(mesh, s=True)              # so does meshShape
    value  = rc.getAttr("ctrl.t")                        # a value: nothing to add, nothing happens
print(repr(ctn), cmds.container("build", q=True, nodeList=True))   # Container("build") ['driven', 'ctrl', 'meshShape']
```

Commands that return the nodes they touched capture those too:

```python
cmds.file(new=True, force=True)
kid = rc.createNode("transform", name="kid")
with container("build"):
    root = rn.transform(name="root")
    rc.parent(kid, root)                                 # returns the child, so the child joins
print(cmds.container("build", q=True, nodeList=True))    # ['root', 'kid']
```

`container=False` opts out. It is popped before the call reaches Maya, on any
command, whether or not the command creates anything:

```python
cmds.file(new=True, force=True)
ctrl = rc.createNode("transform", name="ctrl")
with container("build"):
    driven = rc.createNode("transform", name="driven")
    found  = rc.ls("ctrl", container=False)
    loose  = rc.createNode("transform", name="loose", container=False)
print(cmds.container("build", q=True, nodeList=True))    # ['driven']
```

Maya's default nodes refuse membership with a warning per node and stay out:

```python
with container("scope"):
    cams = rc.ls(type="camera")                          # Warning: Skipping 'perspShape'. Node cannot be added to assets. (x4)
print(cams, cmds.container("scope", q=True, nodeList=True))
# PlugList([Node("frontShape"), Node("perspShape"), Node("sideShape"), Node("topShape")]) None
```

---

## 7. `Node.wrap` for manual use

When you call `maya.cmds` yourself, `Node.wrap` converts the result the same
way the wrappers do — and never touches the container scope.

```python
cmds.file(new=True, force=True)
cmds.createNode("transform", name="foo")
print(repr(Node.wrap("foo")), Node.wrap(["foo", "persp"]))       # Node("foo") PlugList([Node("foo"), Node("persp")])
print(Node.wrap(None), Node.wrap(5.0), Node.wrap("not_a_node"))  # None 5.0 not_a_node
print(repr(Node.wrap("foo.tx")))                                 # Node("foo") -- attribute stripped, as in section 3
with container("build"):
    wrapped = Node.wrap(cmds.createNode("transform", name="wrapped"))
    created = Node.create("transform", name="created")
print(cmds.container("build", q=True, nodeList=True))            # ['created'] -- wrap never joins a scope
```

---

## 8. Factories: the create kwargs

`rn.<nodeType>(...)` is `cmds.createNode` followed by attribute injection.
Four kwargs, with their Maya short forms, go to `createNode`; `container` is
the scope opt-out; everything else is an attribute (section 9).

```python
cmds.file(new=True, force=True)
root  = rn.transform(name="root")
child = rn.transform(n="child", p=root)                  # short forms, a Node as parent
print(repr(root), repr(child), cmds.listRelatives("child", parent=True))   # Node("root") Node("child") ['root']
print(str(rn.transform(name="root")))                    # root1 -- a clash at the same DAG level: Maya uniquifies, str(node) is the real name
print(str(rn.transform(name="child")))                   # |child -- no clash with |root|child, so no rename: str(node) is the shortest unique path
print(rn.transform.__doc__.splitlines()[0])              # Create a Maya ``transform`` node and apply attribute kwargs.
```

Factories skip selection by default (the rig option `skip_selection`);
`rc.createNode` selects, as Maya does.

```python
print(cmds.ls(sl=True))                                  # []
picked = rn.transform(name="picked", skipSelect=False)   # or ss=False
print(cmds.ls(sl=True))                                  # ['picked']
rc.createNode("transform", name="viaCmds")
print(cmds.ls(sl=True))                                  # ['viaCmds']
```

`shared=True` is Maya's shared-node flag. A second create of the same shared
name makes `cmds.createNode` return `None`, which the factory cannot wrap:

```python
shared = rn.transform(name="shared1", shared=True)       # or s=True
try:
    rn.transform(name="shared1", shared=True)
except ValueError:
    print("ValueError", cmds.ls("shared1*"))             # ValueError ['shared1']
print(rc.createNode("transform", name="shared1", shared=True))   # None -- the command path just returns it
```

---

## 9. Attribute kwargs go through `<<`

Every remaining kwarg is `getattr(node, key) << value`, in order, so the
whole injection grammar is available at creation time.

```python
cmds.file(new=True, force=True)
pma = rn.plusMinusAverage(operation=2, input1D=[10, 3, 1])   # a scalar, then a multi fan-out
print(pma.output1D >> None)                                   # 6.0
src = rn.transform(name="src", translate=[1, 2, 3], ry=45)   # a compound, and a short name
print(cmds.getAttr("src.t"), cmds.getAttr("src.ry"))          # [(1.0, 2.0, 3.0)] 45.0
dst = rn.transform(name="dst", tx=src.tx)                     # a Plug value connects
print(cmds.listConnections("dst.tx", p=True))                 # ['src.translateX']
```

A matrix value takes the matrix shorthand: a literal is decomposed and set, a
matrix plug gets a `decomposeMatrix` wired in.

```python
import numpy as np

m = np.eye(4)
m[3, :3] = [9, 0, 0]
lit = rn.transform(name="lit", matrix=m.tolist())
print(cmds.getAttr("lit.t"), cmds.listConnections("lit.t"))  # [(9.0, 0.0, 0.0)] None
live = rn.transform(name="live", matrix=src.wm)
print(cmds.listConnections("live.t"))                         # ['decomposeMatrix1']
```

A kwarg names an attribute that must already exist. Specs are added after
creation, with `<<` on the node:

```python
from rig.spec import Float

try:
    rn.network(name="params", weight=Float("weight"))
except AttributeError as err:
    print(err)                                                # Attribute not found: params.weight
params = rn.network(name="params2")
params << Float("weight") << 0.5
print(params.weight >> None)                                  # 0.5
```

The failed call has already created `params` by the time the kwarg fails.

---

## 10. Keyword collisions

Node types named after Python keywords take a trailing underscore. Maya's
`and` / `or` / `not` logic nodes exist from Maya 2024.

```python
gate = rn.and_(name="gate")
print(cmds.nodeType(str(gate)), cmds.nodeType(str(rn.or_(name="either"))), cmds.nodeType(str(rn.not_(name="flip"))))   # and or not
print("and_" in dir(rn), "and" in dir(rn), rn.and_.__name__)   # True False and
```

---

## 11. Keyword-only signature

A factory takes no positional arguments; the node type is the attribute name
you reached it by.

```python
try:
    rn.blinn("shiny")
except TypeError as err:
    print(err)                                    # rig.bridges.nodes.blinn() takes 0 positional arguments but 1 was given
shiny = rn.blinn(name="shiny", color=[1, 0, 0])
print(repr(shiny), cmds.getAttr("shiny.color"))   # Node("shiny") [(1.0, 0.0, 0.0)]
```

---

## 12. Factories and the container scope

A factory joins the active scope like any created node; `container=False`
keeps it out. Nested scopes flatten by default and prefix `name=` with the
scope name — `with container` behaviour, visible here because `name` goes
through it.

```python
cmds.file(new=True, force=True)
with container("net"):
    inside  = rn.transform(name="inside")
    outside = rn.transform(name="outside", container=False)
    with container("sub"):
        nested = rn.transform(name="nested")
print(cmds.container("net", q=True, nodeList=True), repr(nested))   # ['inside', 'sub_nested'] Node("sub_nested")
```

---

## 13. Plugin node types and `_refresh_node_types`

The type set behind `rn` is `cmds.ls(nodeTypes=True)`, read once at the first
`rn.<name>` lookup and cached. A plugin loaded after that is invisible until
you refresh. Types Maya registers up front are in the set even while their
plugin is unloaded and the factory loads the plugin on demand — in Maya 2025
that covers the `matrixNodes` types but not the `quatNodes` ones.

```python
try:
    rn.floatMath
    print("registered")                            # lookdevKit was already loaded at the first lookup
except AttributeError:
    print("AttributeError")                        # the fresh mayapy case
cmds.loadPlugin("lookdevKit", quiet=True)
try:
    rn.floatMath
    print("registered")
except AttributeError:
    print("AttributeError")                        # still -- the cached set is stale
rn._refresh_node_types()                           # re-query Maya, drop the factory cache
fm = rn.floatMath(operation=2, floatA=3, floatB=4)
print(repr(fm), fm.outFloat >> None)               # Node("floatMath1") 12.0
```

`rc.createNode` goes straight to `maya.cmds` and loads nothing. Given a type
Maya does not know, Maya makes an `unknown` node and warns; the factory path
(`rn.<type>`, `Node.create`) loads `matrixNodes` / `quatNodes` for you.

```python
u = rc.createNode("noSuchNodeType")               # Warning: Unrecognized node type 'noSuchNodeType'; preserving node information during this session.
print(repr(u), cmds.nodeType(str(u)))             # Node("unknown1") unknown
q = Node.create("quatSlerp")
print(repr(q), cmds.pluginInfo("quatNodes", q=True, loaded=True))   # Node("quatSlerp1") True
```

A command a plugin adds to `maya.cmds` needs no refresh: `rc.<name>` looks it
up on first call.

---

## Where to go next

| Read | For |
|---|---|
| [`README.md`](README.md) | concepts, the result and argument tables, the four ways to make a node |
| [`../README.md`](../README.md) | the language these results plug into |
| [`../CHEATSHEET.md`](../CHEATSHEET.md) | every operator and top-level name of the DSL, runnable |
