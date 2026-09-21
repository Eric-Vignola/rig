# `rig.bridges` — `maya.cmds` and node types, in DSL form

The two things you call most in Maya, handed back as DSL objects. `commands`
wraps every `maya.cmds` function so node names come back as `Node` /
`PlugList`; `nodes` gives every registered node type a factory that creates
the node and sets its attributes in one call. Both are built lazily, one name
at a time, on first touch.

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
from rig.bridges import commands as rc
from rig.bridges import nodes as rn

cube, history = rc.polyCube(name="cube")                  # every node name comes back as a Node
ctrl = rn.transform(name="ctrl", translate=[0, 5, 0])     # createNode + attribute setup, one call
cube.t << ctrl.t * 2
print(repr(cube), repr(ctrl), cmds.getAttr("cube.t"))     # Node("cube") Node("ctrl") [(0.0, 10.0, 0.0)]
```

`rc` and `rn` are the conventional aliases; the rest of the `rig` docs use
them too.

---

## Where to go next

| You want to... | Read |
|---|---|
| Copy-paste a runnable example of every behaviour on this page | [`CHEATSHEET.md`](CHEATSHEET.md) |
| The language the results plug into (`<<`, `>>`, containers, specs) | [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) |
| The per-command flags | `maya.cmds` documentation — `rc.<name>` takes exactly what `cmds.<name>` takes, plus `container=` |

---

## What's inside

| File | Contents |
|---|---|
| `__init__.py` | empty — import the submodules by name: `from rig.bridges import commands, nodes` |
| `commands.py` | the PEP 562 `__getattr__` that builds one wrapper per `maya.cmds` name, `_coerce` (arguments in), `_wrap_result` (results out), `_NO_COERCE`, `_WRAPPER_CACHE`, `__dir__` |
| `nodes.py` | the `__getattr__` that builds one factory per node type, `_CREATE_KWARGS`, `_resolve_alias` (`and_` → `and`), `_refresh_node_types`, `_NODE_TYPES`, `__dir__` |

Neither module defines a public function of its own. Every name you call is a
Maya name: `rc.ls` is `maya.cmds.ls`, `rn.transform` is the `transform` node
type. `Node.wrap`, the manual converter, lives on `rig.Node`.

---

## Concepts

### Nothing exists until you touch it

`rc.ls` does not exist when the module imports. The first `rc.ls` builds a
wrapper around `maya.cmds.ls`, caches it, and every later access returns that
same function (`rc.ls is rc.ls`). The wrapper keeps the original reachable as
`rc.ls.__wrapped__` and carries a `__qualname__` of `rig.bridges.commands.ls`,
so tracebacks and `inspect` read naturally.

`rn.transform` works the same way, against the set of node types Maya reports
through `cmds.ls(nodeTypes=True)`. That set is read once, at the first
`rn.<name>` lookup, and cached; see *plugin node types* below for what that
means.

A name that is not on `maya.cmds`, or not a registered node type, raises
`AttributeError` with a message that says so. `dir(rc)` lists every callable
on `maya.cmds`; `dir(rn)` lists every registered node type, with keyword
collisions already spelled `and_` / `or_` / `not_`.

### What comes back

Every wrapper runs the real command, then converts the result:

| `maya.cmds` returned | you get |
|---|---|
| a `str` that names a node | `Node` |
| a `str` that names a plug (`"a.translateX"`) | `Node("a")` — the attribute is stripped |
| a `str` that is not a node (`"transform"`, `"hello"`) | the `str`, unchanged |
| a `list` of node names | `PlugList` of `Node` |
| a `list` of values (`getAttr` of a compound, `xform -q`) | `PlugList` of those values |
| a `list` of strings that are not nodes (`listAttr`) | a plain `list` |
| `bool`, a number, `None` | unchanged |

So `rc.createNode("transform")` is a `Node`, `rc.polyCube()` is
`PlugList([Node("pCube1"), Node("polyCube1")])`, `rc.ls(sl=True)` with nothing
selected is `PlugList([])`, `rc.listRelatives(x, p=True)` on a root node is
`None`, and `rc.getAttr("x.tx")` is a `float`. A `PlugList` broadcasts, so
`rc.ls(sl=True).ty << 2` is a one-liner.

Two consequences are easy to trip on. A string *value* that happens to name
an existing node is wrapped as that node (`rc.getAttr("x.label")` holding
`"persp"` comes back as `Node("persp")`). And plug strings collapse to their
node, so `rc.listConnections(x, plugs=True)` loses the attribute — use
`plug.get_inputs()` / `plug.get_outputs()` on the DSL side, or raw
`cmds.listConnections`, when you need the plug.

### What goes in

Maya commands take node names; the DSL holds `Node` objects. Before the call,
every positional and keyword argument is converted: a `Node` becomes its name,
a list or tuple that contains any `Node` becomes a list of names. A `Plug` is
already a `str` subclass and passes through as is. So `rc.parent(child, root)`,
`rc.parent([a, b], root)`, `rc.createNode("transform", parent=root)` and
`rc.getAttr(node.tx)` all work with DSL objects in hand.

Nine commands are exempt. Their arguments reach Maya exactly as written,
because they carry callback strings, expressions or code to evaluate:

| `_NO_COERCE` |
|---|
| `evalDeferred`, `scriptJob`, `scriptNode`, `expression`, `undo`, `redo`, `undoInfo`, `warning`, `error` |

Pass names (`str(node)`) to those. In Maya 2025 `maya.cmds` stringifies a
`Node` on its own, so `rc.expression(o=node)` happens to work anyway; the
bridge converts explicitly rather than lean on that.

### Only the nodes a call creates join the active container

Inside a `with container("name"):` block, **the nodes a wrapper creates are
added to that container** — and nothing else. The wrapper watches Maya's
node-added message for the duration of the call, so it knows what the
command made as opposed to what it merely returned. A query, a `parent` or a
`rename` returns nodes that already existed and moves none of them:

```python
from rig import container

cmds.file(new=True, force=True)
ctrl = rc.createNode("transform", name="ctrl")           # made OUTSIDE any scope
with container("build"):
    driven = rn.transform(name="driven")
    rc.ls("ctrl")            # a query: looks, never captures
    rc.parent(ctrl, driven)  # an edit: moves ctrl in the DAG, not into the scope
print(cmds.container("build", q=True, nodeList=True))    # ['driven']
```

Creation is tracked exactly, not by what comes back: `rc.polyCube()` returns
the transform and the `polyCube` node, and the mesh shape it also made joins
with them. Value results are harmless: a `float`, a `bool` or a list of
tuples has nothing to add. Maya's default cameras refuse membership with a
warning per node (`Skipping 'perspShape'. Node cannot be added to assets.`)
and stay out.

The opt-out is `container=False`: the created nodes stay where Maya put
them. The wrapper pops the flag before Maya sees it, on every command —
`rc.createNode("transform", container=False)` — and the factories take it
too: `rn.transform(name="loose", container=False)`. On a query it is
accepted and changes nothing, and outside any `with container` block there
is nothing to join.

### A factory is `createNode` plus `<<`

`rn.<nodeType>(**kwargs)` is keyword-only: `rn.blinn("x")` is a `TypeError`,
`rn.blinn(name="x")` is the call. The kwargs split in two.

| kwarg | short form | goes to |
|---|---|---|
| `name` | `n` | `cmds.createNode` |
| `parent` | `p` | `cmds.createNode` |
| `shared` | `s` | `cmds.createNode` |
| `skipSelect` | `ss` | `cmds.createNode` — default is the rig option `skip_selection`, `True` as shipped |
| `container` | | the scope opt-out above |

Everything else is applied **in order** as `getattr(node, key) << value`, so
the whole injection grammar is available at creation: a number sets, a list
sets a compound, a list on a multi fans out (`input1D=[1, 2, 3]`), a `Plug`
connects, a literal matrix is decomposed and set, a matrix `Plug` gets a
`decomposeMatrix` wired in. Short attribute names work (`tx=5`). The
attribute has to exist already — an attribute spec (`Float("weight")`) is not
a valid kwarg value; add it after creation with `node << Float("weight")`.

Node types whose name is a Python keyword take a trailing underscore:
`rn.and_`, `rn.or_`, `rn.not_` create Maya's `and` / `or` / `not` logic nodes
(Maya 2024+). `dir(rn)` lists the underscore forms.

### Plugin node types

The type set behind `rn` is cached at the first lookup. Types that Maya
registers up front are always there, and the factory path loads their plugin
on demand — in Maya 2025 the `matrixNodes` types (`decomposeMatrix`, ...)
are registered even while that plugin is unloaded. The `quatNodes` types are
not, and neither is anything from a plugin loaded *after* that first lookup
(`lookdevKit`, a studio plugin): those are invisible until you ask for a
refresh.

```python
cmds.loadPlugin("lookdevKit", quiet=True)
rn._refresh_node_types()                                  # re-query Maya, drop the factory cache
fm = rn.floatMath(operation=2, floatA=3, floatB=4)
print(fm.outFloat >> None)                                # 12.0
```

The `AttributeError` for an unknown type names `_refresh_node_types()` in its
message. `rc` needs no refresh: a command a plugin adds to `maya.cmds` is
found on first call.

### Four ways to make a node

| Call | joins the active scope | selects the new node | loads `matrixNodes` / `quatNodes` on demand | attribute kwargs |
|---|---|---|---|---|
| `rc.createNode("transform", name="x")` | yes | yes — Maya's default | no — an unregistered type becomes an `unknown` node, with a warning | no |
| `rn.transform(name="x", tx=5)` | yes | no (`skip_selection`) | yes, once the type is in the cached set — `decomposeMatrix` always is, `quatSlerp` only after `quatNodes` loads and a refresh | yes, through `<<` |
| `Node.create("transform", name="x")` | yes | no | yes, with no registry check in the way | no |
| `Node.wrap(cmds.createNode("transform", name="x"))` | no | yes | no | no |

`Node.wrap` is the manual converter for when you call `maya.cmds` yourself:
`str` → `Node`, list → `PlugList`, `None` and numbers pass through, a string
that is not a node passes through. It never touches the container scope.

---

## Conventions

- **Maya's names, Maya's flags.** `rc.<name>` takes exactly what
  `cmds.<name>` takes; `rn.<type>` is exactly the Maya type name. The only
  kwarg the bridge consumes on a command is `container`; a factory also
  consumes `name` / `n`, `parent` / `p`, `shared` / `s`, `skipSelect` / `ss`.
- **Factories are keyword-only.**
- **`str(node)` is the real name.** Maya uniquifies a clash at the same DAG
  level, so `rn.transform(name="root")` twice gives `root` and `root1`; a
  `child` under `root` and a `child` at world level both keep their name and
  `str(node)` is the shortest unique path, `|child`.
- **Results are the DSL's own types** — `Node`, `Plug` (a `str` subclass),
  `PlugList` — so anything a wrapper returns takes `<<`, `>>` and the math
  operators directly.
- **Keyword collisions** use PEP 8's trailing underscore: `and_`, `or_`,
  `not_`, and any future type whose name is a Python keyword.

---

## Real behaviour, verified

Not bugs to work around blindly — how it actually behaves under Maya 2025.

- A string result that names an existing node is wrapped as that node, even
  when it is a value: `rc.getAttr("x.label")` holding `"persp"` returns
  `Node("persp")`; holding `"hello"` it returns `'hello'`.
- Plug strings collapse to nodes. `rc.listConnections("b.tx", p=True)` is
  `PlugList([Node("a")])`, the same as without `p=True`. Use
  `b.tx.get_inputs()` (`PlugList([Plug("a.translateX")])`) or
  `cmds.listConnections`.
- A list of values comes back as a `PlugList`: `rc.getAttr("x.t")` is
  `PlugList([(1.0, 2.0, 3.0)])`, `rc.xform("x", q=True, t=True)` is
  `PlugList([1.0, 2.0, 3.0])`. A list of non-node strings (`rc.listAttr`)
  stays a plain `list`.
- `rc.createNode` goes straight to `maya.cmds` and loads no plugin.
  `rc.createNode("quatSlerp")` with `quatNodes` unloaded gives an `unknown`
  node and a Maya warning; `Node.create("quatSlerp")` loads the plugin
  first. `rn.quatSlerp` is an `AttributeError` until the plugin is loaded
  *and* `_refresh_node_types()` has run, because Maya does not list the
  `quatNodes` types while the plugin is unloaded. It does list the
  `matrixNodes` types, so `rn.decomposeMatrix()` works from a cold start.
- `shared=True` on a name that already exists: `cmds.createNode` returns
  `None`, so `rc.createNode(..., shared=True)` returns `None` and
  `rn.transform(..., shared=True)` raises `ValueError` while wrapping it.
- A spec object as a factory kwarg fails with `AttributeError: Attribute not
  found: <node>.<attr>` — after the node has been created. Add specs after.
- Inside a flattened nested scope, `name=` is prefixed with the scope name:
  `with container("net"): with container("sub"): rn.transform(name="x")` is
  `sub_x`. That is `with container` behaviour, not the factory's.
- `rc.createNode` leaves the new node selected; `rn.<type>` and `Node.create`
  do not (`rig.set_options(skip_selection=False)` or `skipSelect=False` to
  change that).
- `rn.and_` / `rn.or_` / `rn.not_` need Maya 2024 or later, where those node
  types exist. On older Maya they are an `AttributeError` like any other
  unregistered type.
