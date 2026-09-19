# `rig.examples` — four rigs, built end to end

Four scripts that each build something complete with the DSL: two rail
spines (a tutorial one and the full one), a camera sticker rig and a
looping image plane. Each is a plain module with one build function; the
`__main__` block at the bottom is its demo. The module docstrings say what
each script does and which features it leans on — read those first, then
run one.

Every `python` block on this page runs top to bottom as one script, inside
Maya or under `mayapy`. The first block is the setup.

```python
from maya import standalone
try:
    standalone.initialize()          # running from mayapy; inside Maya this raises and is skipped
except Exception:
    pass
from maya import cmds
cmds.file(new=True, force=True)
```

A spine in six lines: four locators become the CVs of a curve, twelve
joints ride it.

```python
from rig import Node, PlugList
from rig.examples import rail_spine_simple

controls = PlugList()
for i in range(4):
    controls.append(Node(cmds.spaceLocator()[0]))
    controls[i].ty << i * 5                                 # stacked along Y

rail   = rail_spine_simple.create_simple_rail(controls, riders=12)
riders = cmds.listRelatives(str(rail), type="joint")
print(rail, len(riders), riders[0], riders[-1])            # rail1 12 rider1 rider12
print(Node(riders[-1]).t.get())                             # [ 0. 15.  0.]
```

---

## Where to go next

| You want to... | Read |
|---|---|
| The language these scripts are written in | [`../README.md`](../README.md) · [`../CHEATSHEET.md`](../CHEATSHEET.md) |
| The tutorial spine, one feature at a time | [`rail_spine_simple.py`](rail_spine_simple.py) — about 100 lines of rig code |
| The full spine: closed loops, any odd degree, projection modes | [`rail_spine.py`](rail_spine.py) |
| Image planes that hold their aspect and their size in frame | [`perspective_image_planes.py`](perspective_image_planes.py) |
| One plane that loops a numbered image sequence | [`image_loop.py`](image_loop.py) |
| The docstring of any of them, from a Maya prompt | `help(rail_spine_simple)` after the import |

---

## What's inside

| File | Builds | Entry point |
|---|---|---|
| `rail_spine_simple.py` | a cubic curve driven by N controls, M rider joints along it, four knobs | `create_simple_rail(position_controls, riders=10, ...)` |
| `rail_spine.py` | the same rail with closed loops, degree 1/3/5/7, per-control orient and scale, three projection modes | `create_rail(position_controls, u, ...)` |
| `perspective_image_planes.py` | a camera plus N image planes under it, each shaped to its image and sized by its distance | `create_setup(camera, count, ...)` |
| `image_loop.py` | one plane facing +Z that loops a numbered image sequence forever | `create_plane(image_dir, name="image_loop", target_size=10.0)` |
| `images/` | `dog1.jpg` … `dog5.jpg` — a fixture both image scripts can eat | |
| `ye_olde_lerp.gif` | the animation the root README embeds | |
| `__init__.py` | empty; it makes `rig.examples` importable | |

Every builder returns rig objects, never strings: a `Node` for the rail
transform, a `namedtuple` of `Node`s and `PlugList`s for the image rigs.

### Which language features each one showcases

| Feature | `rail_spine_simple` | `rail_spine` | `perspective_image_planes` | `image_loop` |
|---|---|---|---|---|
| `with container("name"):` — a scoped utility graph | yes | yes | yes | yes |
| `container.add(node)` — enrol a node made outside the scope | | | yes | yes |
| `rc.*` — `maya.cmds` returning `Node` / `PlugList` (`rc.curve`, `rc.polyPlane`, `rc.parent`) | yes | yes | yes | yes |
| `rn.*` — node factories with attribute kwargs (`rn.motionPath()`, `rn.file()`, `rn.colorCorrect()`) | yes | yes | yes | yes |
| `node << Float("x", min=0, max=1) << value << lock` spec injection | `Float` | `Float`, `Enum` | `Float`, `Int`, `Enum`, `String` | `Float`, `Int`, `String` |
| `plug << lock`, `plug << hide` | `hide` | both | both | `lock` |
| `PlugList(controls).wm * rail.wim` — broadcast matrix multiply | yes | yes | | |
| `rail_shape.cv[:] << matrices` — matrix-to-CV shorthand | yes | yes | | |
| `rider << matrix_plug` — matrix-to-transform shorthand (t / r / s + shear) | yes | yes | | |
| `plug.get()` — the `getAttr` idiom | yes | yes | | demo |
| `matrix.axis`, `matrix.multiply(local=True)`, `matrix.decompose` | axis, multiply | multiply, decompose | | |
| `normalize`, `dist` — cross-type dispatch verbs | `normalize` | both | `dist` | |
| `lerp`, `slerp`, `elerp`, `interpolate.sequence(method=...)` | `lerp`, `slerp` | `slerp`, `elerp`, `sequence` | | |
| `condition(a > b, x, y)` — comparisons build `condition` nodes | `rf.condition` | yes | yes | yes |
| `rf.choice`, `rf.clamp`, `rf.cumsum`, `rf.rev`, `rf.max`, `rf.abs` | | choice, clamp, cumsum, rev | choice, clamp, rev, max, abs | rev |
| `rf.frame()` and `%` — scene time and modulo | | `%` | both | both |
| `trigonometry.atand`, `trigonometry.sind` — degree variants, no conversion nodes | | | yes | |
| `constant(0)` — a literal that must be a plug | | yes | yes | |
| `Lambert(name, unique=True, ...)`; `shape << material` | | | yes | yes |
| `set_options(create_containers=...)` | | demo | | |
| `container=False` on a factory, then `rc.parent` | yes | yes | yes | `container=False` only |
| `Lambert(..., container=True)` — a per-plane look opts into the container | | | yes | yes |

`rc` is `rig.bridges.commands`, `rn` is `rig.bridges.nodes`, `rf` is
`rig.functions`.

---

## Running one from the Script Editor

Each script is importable with `rig`'s parent folder on `sys.path`. Import
the module and call its build function; the arguments are plain Maya names,
`Node`s or a `PlugList` of either.

<!-- notest -->
```python
from rig import Node, PlugList
from rig.bridges import commands as rc
from rig.examples import rail_spine_simple, rail_spine, perspective_image_planes, image_loop

controls = rc.ls(selection=True)                                   # a PlugList of the selected controls
rail  = rail_spine_simple.create_simple_rail(controls, riders=10)
rail  = rail_spine.create_rail(controls, 20, orient_controls=controls, scale_controls=controls)
setup = perspective_image_planes.create_setup("camera1", 5)        # .camera .planes .shapes
loop  = image_loop.create_plane("D:/frames/run_cycle")             # .transform .shape .material .texture
```

The `__main__` block at the bottom of each file is a demo: open the file
in the Script Editor and run it, and `__name__` is `"__main__"` there, so
the demo builds. The two rail demos need nothing; the two image demos
look for pictures at paths on the author's machine and only print a
message when they find none — point them at the `images/` folder instead
(the blocks below do). Under `mayapy` the demos are not the way in:
initialise `maya.standalone` first, as the setup block at the top of this
page does, then import and call.

`debug=True` is the default on both rail builders and parents a
`polyCube` under every rider so you can see them; pass `debug=False` for
a production build.

---

## `rail_spine_simple.py` — the tutorial

A stripped-down sibling of `rail_spine` that walks through the DSL one
feature at a time, in eight numbered steps you can read in one sitting:

1. `rc.curve(d=3, p=cv_positions)` from the controls' world positions;
   the rail transform's `t` / `r` / `s` are hidden — it is only a parent.
2. `rail_shape.cv[:] << position_controls.wm * rail.wim` — every CV is
   driven live. A `PlugList` of `worldMatrix` plugs times the rail's
   `worldInverseMatrix` lands the controls in rail space, and injecting a
   matrix into a CV extracts its translation for you. No `decomposeMatrix`
   by hand.
3. A `curveInfo` reads the live arc length; `current_length.get()`
   snapshots the build-time length as a Python float, and
   `default_length / current_length` builds the stretch ratio plug.
4. `matrix.axis(controls[0].wm, up_axis)` and `normalize(...)` give the
   first and last control's up vector.
5. Two helper `motionPath` nodes sample the curve just inside `u = 0` and
   `u = 1`; `matrix.multiply(aim_vec, mp.orientMatrix, local=True)` turns
   the aim axis through that frame into a world-space end tangent.
6. Per rider: a joint (`container=False`, then `rc.parent`), a `uDefault`
   attribute, and the u maths — `lerp(anchored, uDefault, rail.stretch)`
   between glide and locked, then `Scale` around `pivot`, then `shift`.
7. A `motionPath` per rider with `worldUpVector << slerp(first_up, last_up, rider.uDefault)`
   — the twist is frozen on the rider's *default* u so it does not drift
   as the spine deforms.
8. `rf.condition(u < 0, tangent_start * u + edge_start, ...)` extrapolates
   along the end tangents outside `[0, 1]`; `composeMatrix` assembles
   translate + rotate; `rider << rider_matrix.outputMatrix * rail.wim`
   decomposes it into the joint's channels.

The knobs land on the rail transform:

| Knob | Default | Does |
|---|---|---|
| `pivot` | 0 | the u value that stays put under `stretch` and `Scale` |
| `stretch` | 0 | 0 = riders glide with the curve's length, 1 = locked to their default u |
| `Scale` | 1 | spread (> 1) or squeeze (< 1) the riders around `pivot` |
| `shift` | 0 | slide every rider along the curve |

Aim and up axis are baked in at build time (`aim_axis=1`, `up_axis=0`:
aim Y, up X); the full script makes them live enums.

```python
print(cmds.listAttr(str(rail), userDefined=True))         # ['pivot', 'stretch', 'Scale', 'shift']
print(rail.pivot.get(), rail.stretch.get(), rail.Scale.get(), rail.shift.get())   # 0.0 0.0 1.0 0.0

mid = Node(riders[6])
print([round(v, 3) for v in mid.t.get()])                 # [0.0, 8.182, 0.0]

controls[1].tx << 10                                       # bend the spine: the riders glide with it
print([round(v, 3) for v in mid.t.get()])                 # [4.29, 6.12, 0.0]

rail.shift << 0.5                                          # push the last rider past u = 1 ...
print([round(v, 3) for v in Node(riders[-1]).t.get()])    # [-0.028, 20.908, 0.0]  -- on the end tangent
```

The last rider is past the end of the curve and keeps going along its
tangent instead of piling up at the tip: that is the infinite projection
this version bakes in.

---

## `rail_spine.py` — the full rail

Everything the tutorial omits on purpose, from the original rig
`rail_spine.py`:

- **Closed loops.** `periodic=True` rolls the CV order so `u = 0` sits on
  the first control and wraps `u_translate % 1`. Degrees 1, 3, 5 and 7;
  a periodic degree-2 rail raises.
- **A live default length.** A second, hidden curve shape (`railProxy`)
  is built from the same positions and parented under the rail; a
  `curveInfo` on each gives `defaultLength`, `currentLength`,
  `stretchRatio` and `stretchDelta` as attributes you can read in the
  channel box.
- **Live axes.** `aimAxis` / `upAxis` enums and `invertAim` / `invertUp`
  toggles on the rail drive every `motionPath`.
- **Per-control orient and scale.** `orient_controls` get an `upAxis` enum
  and an `invertUp` toggle each; `rf.choice([[1,0,0],[0,1,0],[0,0,1]], selector=...)`
  picks the vector, `matrix.multiply(..., local=True)` rotates it, and
  `interpolate.sequence(u, weights, vectors, method=slerp)` blends along
  the curve. `scale_controls` do the same through `matrix.decompose(...).outputScale`
  and `method=elerp`. The weights are the cumulative arc-length ratios
  between controls (`rf.cumsum` of `dist(wm[:-1], wm[1:])`), exposed as a
  locked `u` on each control when `debug=True`.
- **Projection modes.** `rotateProjection` and `scaleProjection` are
  `Frozen:Infinite:Clamped` enums: sample at the rider's default u, at its
  live u, or at its live u clamped to the first and last control.
  `translateProjection` (`Clamped:Infinite`, open rails only) is the
  end-tangent extrapolation, with `uTangentStart` / `uTangentEnd` hidden.
- **`u` as a count or a list.** `u=20` spreads twenty riders evenly; a
  list gives each rider its own u, and a string in that list names a plug.

`create_rail(position_controls, u, orient_controls=None, scale_controls=None, rail_name="rail1", rider_name="rider1", degree=3, periodic=False, aim_axis=1, up_axis=0, invert_aim=False, invert_up=False, control_up=None, invert_up_control=False, debug=True)`
returns the rail `Node`. `orient_controls` and `scale_controls` must be
members of `position_controls` when there is more than one of them.

```python
cmds.file(new=True, force=True)
from rig import set_options
from rig.examples import rail_spine

set_options(create_containers=True)                        # False shows the raw graph instead of railNode1

controls = PlugList()
for i in range(5):
    controls.append(Node(cmds.spaceLocator()[0]))
    controls[i].ty << i * 5

rail = rail_spine.create_rail(
    controls, 20,                                          # twenty riders, evenly spread
    orient_controls = controls,
    scale_controls  = controls,
    degree          = 3,
    periodic        = False,
    aim_axis        = 1,
    up_axis         = 0,
)
riders = cmds.listRelatives(str(rail), type="joint")
print(len(riders), cmds.ls(type="container"))               # 20 ['railNode1']
print(cmds.listAttr(str(rail), userDefined=True))
# ['defaultLength', 'currentLength', 'stretchRatio', 'stretchDelta', 'aimAxis', 'upAxis',
#  'invertAim', 'invertUp', 'pivot', 'stretch', 'Scale', 'shift', 'scaleProjection',
#  'rotateProjection', 'translateProjection', 'uTangentStart', 'uTangentEnd']
print(cmds.listAttr(str(controls[0]), userDefined=True))    # ['u', 'upAxis', 'invertUp']
print(controls.u.get())                                     # [0.   0.25 0.5  0.75 1.  ]
print(rail.defaultLength.get(), rail.stretchRatio.get())    # 20.0 1.0
```

The demo at the bottom of the file then does this:

```python
rail.stretch         << 0                                  # glide
rail.scaleProjection << 2                                  # Clamped
controls[1].tx << 10
controls[1].s  << [5, 0.1, 5]                              # a fat, flat second control

print(round(rail.stretchRatio.get(), 4))                   # 0.8016  -- the curve got longer
mid = Node(riders[5])
print([round(v, 3) for v in mid.t.get()])                  # [4.436, 2.805, 0.0]
print([round(v, 3) for v in mid.s.get()])                  # [2.671, 0.245, 2.671]  -- elerp toward control 1
```

Twenty riders with orient and scale is about 1,600 nodes in `railNode1`
on Maya 2025 and takes a handful of seconds to build under `mayapy`; the
tutorial spine is a couple of seconds. The container holds the maths; the
joints sit under the rail, outside it.

---

## `perspective_image_planes.py` — stickers on a camera

`create_setup(camera, count, name="STICKER_LAYER", default=None, parent=None, offset=10)`
builds a camera and `count` poly planes parented under it, one every
`offset` units down `-Z`. Each plane:

- takes its **aspect** from the image (`outSizeX` / `outSizeY` of a probe
  `file` node — a second one, so the size does not depend on the animated
  texture) and its **size** from its distance to the camera:
  `get_scale(aperture, focal_length, distance)` turns the camera's angle
  of view, `trig.atand(aperture / (2 * fl))`, into a width at that
  distance. The degree variants `atand` / `sind` feed Maya's angle plugs
  directly; the radian ones would insert conversion nodes.
- gets a `Lambert` of its own, full-bright (`diffuse=0, ambientColor=1`
  at creation, then `color` and `ambientColor` wired to the texture), with
  a `colorCorrect` remapping the texture's transparency through two dials.
- carries its controls **on the shape**, so selecting the mesh shows them
  in the channel box:

| Attribute | Type | Does |
|---|---|---|
| `image` | `String` | the file path; `useFrameExtension` follows `sequenceType > 0` |
| `sequenceType` | `Enum` | `Static:Sequence:Looping:Ping Pong` — `rf.choice` picks the frame maths |
| `sequenceStart`, `sequenceEnd`, `sequenceOffset` | `Int` | the frame range, relative to `rf.frame()` |
| `alpha` | `Float` 0..1 | how much of the image's transparency to honour |
| `opacity` | `Float` 0..1 | a global fade |

`tx`, `ty`, `r`, `s` and `v` on each plane are locked and hidden — only
`tz` is yours. When `count > 1`, `rf.max` over the planes' aspect ratios
drives the camera's `hfa` / `vfa`; an orthographic camera is handled by
`condition(camera_shape.orthographic, ...)`.

It returns `(camera, planes, shapes)`: the camera transform `Node`, a
`PlugList` of plane transforms and a `PlugList` of plane shapes. Because
`shapes.image` is a `PlugList` of plugs, `shapes.image[:] << list_of_paths`
fans a list of files across the planes in one line.

```python
cmds.file(new=True, force=True)
import glob
import os

import rig.examples
from rig.examples import perspective_image_planes

images_dir = os.path.join(os.path.dirname(rig.examples.__file__), "images")
setup      = perspective_image_planes.create_setup("camera1", 5)

print(setup.camera, setup.planes[0], setup.shapes[0])       # camera2 mesh_STICKER_LAYER_0 mesh_STICKER_LAYER_0Shape
print(setup.planes.tz.get())                                 # [-10. -20. -30. -40. -50.]
print(cmds.listAttr(str(setup.shapes[0]), userDefined=True))
# ['image', 'sequenceType', 'sequenceStart', 'sequenceEnd', 'sequenceOffset', 'alpha', 'opacity']

setup.shapes.image[:] << sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
print([os.path.basename(p) for p in setup.shapes.image.get()])
# ['dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg', 'dog5.jpg']

planes = cmds.ls(type="polyPlane")
print([(round(cmds.getAttr(p + ".width"), 2), round(cmds.getAttr(p + ".height"), 2)) for p in planes])
# [(7.26, 4.84), (10.89, 14.51), (21.77, 16.31), (29.03, 29.01), (36.29, 25.28)]

from rig.shade import Material
print(setup.shapes[0] >> Material(), cmds.ls(type="container"))   # [Lambert('sticker_layer_0')] ['sticker_layer_container']
```

Five images of five different aspects, five planes at five distances, all
filling the same frame.

---

## `image_loop.py` — one plane, one sequence, forever

`create_plane(image_dir, name="image_loop", target_size=10.0)` scans a
folder for image files (`png jpg jpeg tif tiff exr tga`, either case),
reads the trailing digits before the extension as the frame number
(`frame.0001.png` is frame 1), and builds one plane facing `+Z` that plays
the range on a loop. A folder with no numbers in its filenames is a single
still.

The loop is three lines of plug arithmetic:

```
cycle  = sequenceEnd - sequenceStart + 1
looped = (frame() + sequenceOffset - sequenceStart) % cycle + sequenceStart
texture.frameExtension << looped
```

The plane's largest side is `target_size`; the other side follows the
image's aspect, read from a probe `file` node so a sequence with uneven
frame sizes does not wobble. `sequenceStart` / `sequenceEnd` /
`sequenceOffset`, `alpha` and `opacity` live on the shape, exactly as in
the camera rig, plus a `sequenceName` string that feeds the texture. It
returns `(transform, shape, material, texture)`.

The `images/` folder next to the script happens to be a five-frame
sequence: `dog1.jpg` … `dog5.jpg`.

```python
cmds.file(new=True, force=True)
from rig.examples import image_loop

loop = image_loop.create_plane(images_dir, name="dogs")
print(loop)
# Output(transform=Node("mesh_dogs"), shape=Node("mesh_dogsShape"), material=Node("dogs"), texture=Node("file1"))
print(loop.shape.sequenceStart.get(), loop.shape.sequenceEnd.get())   # 1 5

plane = cmds.ls(type="polyPlane")[0]
print(round(cmds.getAttr(plane + ".width"), 3), round(cmds.getAttr(plane + ".height"), 3))   # 10.0 6.667

frames = []
for t in (1, 5, 6, 7):
    cmds.currentTime(t)
    frames.append(loop.texture.frameExtension.get())
print(frames)                                                          # [1, 5, 1, 2]  -- wraps after frame 5

print(loop.shape >> Material(), cmds.ls(type="container"))             # [Lambert('dogs')] ['dogs_container']
```

---

## The shading boilerplate, before and after

Both image scripts used to build their material by hand: a shader, a
shading engine, the `surfaceShader` wire, the `defaultShaderList1`
registration, then the assignment. A dozen lines, and the shading
engine's `materialInfo` never joined the container.

<!-- notest -->
```python
material = rc.shadingNode("lambert", asShader=True, name=name)
material.diffuse      << 0
material.ambientColor << 1

sg = rc.sets(
    name            = "{}SG".format(name),
    empty           = True,
    renderable      = True,
    noSurfaceShader = True,
)
sg.surfaceShader             << material.outColor
Plug("defaultShaderList1.s") << material.msg
rc.sets(shape, e=True, forceElement=sg)
```

Now, with `rig.shade`:

<!-- notest -->
```python
texture  = rn.file()
material = Lambert(
    name,
    unique       = True,
    container    = True,
    diffuse      = 1,
    color        = texture.outColor,
    ambientColor = texture.outColor,
)
shape << material
material.transparency << remap.outColor      # the spec is the handle for the material's plugs
```

`Lambert(name, ...)` is a lazy handle — it makes no Maya call until it
meets `<<`. Then the shader, `<name>SG`, its `materialInfo` and the
`defaultShaderList1` link are built in one go, the kwargs are applied as
attribute injections (a value sets `diffuse`, a plug connects the texture
into `color` and `ambientColor`), and the shape is moved into the engine
with one `cmds.sets(forceElement=...)`. A material is a scene-level asset
and stays out of the active container by default; `container=True` opts
this per-plane look in, so deleting the plane's container takes its look
with it. The spec keeps working as the material afterwards:
`material.transparency << ...` reaches the lambert's plug. `unique=True` keeps the old behaviour on a
rebuild: a second `create_plane(name="run")` gets its own `run1` and
`run1SG` instead of re-using `run`. `m.node` is the material `Node` the
rest of the script wires textures into; `m.engine` is the shading engine.

---

## Real behaviour, verified

Verified on Maya 2025; not bugs to work around blindly.

- `create_setup("camera1", n)` on a fresh scene hands back the transform
  `camera2`, with the shape named `camera1Shape1`. That is
  `cmds.camera(name="camera1")`: the shape gets your name plus `Shape` and
  a counter, the transform gets your name with its trailing digits
  stripped and Maya's own counter appended (`name="cam7"` gives `cam1` and
  `cam7Shape1`). Use `setup.camera`, not the string you passed in.
- The riders of both rails are created with `container=False` and then
  `rc.parent`ed under the rail. Maya renames a node when it is reparented,
  which would confuse the container's membership tracking; the maths sits
  inside `railNode1` / `simpleRail1`, the joints and their debug cubes sit
  outside, and `rc.parent` keeps the `Node` pointing at the renamed joint.
- `rail_spine_simple` snapshots the default arc length with `.get()` at
  build time, so editing the controls after the build changes the stretch
  ratio; `rail_spine` keeps a hidden proxy curve shape under the rail so
  the default stays live.
- `periodic=True` with `degree=2` raises in `rail_spine`; 1, 3, 5 and 7
  are the supported degrees, and the CV roll that puts `u = 0` on the
  first control is keyed on the degree.
- Both image scripts load the `lookdevKit` plugin for `colorCorrect` and
  then call `rn._refresh_node_types()`: `rig.bridges.nodes` caches Maya's
  node-type list on first use, so a plugin loaded afterwards is invisible
  to `rn.colorCorrect()` until the cache is refreshed. Private name, but
  it is what the scripts do.
- `create_setup` runs without any image on disk — until `image` is set
  the planes are square, sized by the angle of view alone (`7.26` at ten
  units from a 35 mm camera), because an `outSizeX` / `outSizeY` of `0`
  is clamped to `1` by a `condition`. `create_plane` raises `ValueError`
  on a folder with no image files.
- The `__main__` demos of the two image scripts point at
  `/Users/ericvignola/...`; on any other machine `perspective_image_planes`
  builds and prints "No images found", `image_loop` raises. The `images/`
  folder beside them is a valid input for both.
- `PlugList` reads come back as NumPy arrays (`controls.ty.get()` is
  `[ 0.  5. 10. 15.]`), and so does a compound on a single `Node`
  (`rider.t.get()` is `[ 0. 15.  0.]`).
- `set_options(create_containers=False)` before a build leaves the whole
  graph loose in the scene — handy for reading it in the Node Editor, and
  what the `rail_spine` demo's comment invites you to try.
