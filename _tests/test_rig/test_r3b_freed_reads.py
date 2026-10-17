"""Regression tests: no rig entry point reads a freed node's API objects.

A file open (or a new scene, a reference unload) frees the nodes of the old
scene; the MPlugs, MObjects and fn sets a held plug or node object keeps then
point at freed memory, and only its MObjectHandle may be read. Every entry
point checks that handle (`_ensure_owner_alive`, `DGNode.ensure_valid`) before
its first API read. `Plug.get` read the MPlug (``isArray``, then the generic
slot test's ``attribute()``) before its check, which crashed Maya (2000 held
plugs: 48 crashes in 48 runs, runs/features/r3b); a membership left-hand side
(``plug in Tag("x")``, ``Tag.of(plug)``, ``[plug] in Tag("x")``) read the MPlug
the same way, and the public raw getter ``get_mplug()`` handed it out
(``Plug(old.get_mplug())`` after a reopen: 24 crashes in 24 runs).

A freed read only crashes when Maya reused the memory, so these tests make it
certain instead: once the scene is freed, every API object a held plug or node
keeps (but its handles) is swapped for a tripwire that fails the test when it
is read, and every public operation must raise ``"... already deleted!"`` (or
answer without reading it: a node's str and repr, a plug's repr, user decision
2026-10-16, and the List operations, which skip a freed element).

`TestNoFreedReadWithArguments` extends the walk to the methods with arguments
and to rig's module functions with the freed object in every argument
position (the r3b verifier's audit, runs/features/r3b/verify/tools/audit_args.py),
in its own mayapy, so a crash fails the test instead of the run.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import Layer, List, Node, Plug, Tag, skip
from rig.nodetypes._base import Attribute
from rig.nodetypes._base import Node as _NodeBase
from rig._tests._base import MayaTestCase
from rig._tests.test_rig import test_r3b_unowned_plugs as r3b


_API_TYPES = (
    OpenMaya.MPlug, OpenMaya.MObject, OpenMaya.MFnBase, OpenMaya.MDagPath,
    OpenMaya.MSelectionList, OpenMaya.MPlugArray, OpenMaya.MObjectArray,
)


class _Read(BaseException):
    """A tripwire was read (a BaseException: an ``except Exception`` in the
    code under test cannot hide it)."""


class _Tripwire:
    """Stands for a freed API object: any read records where it came from."""

    __slots__ = ("_what", "_reads")

    def __init__(self, what, reads):
        object.__setattr__(self, "_what", what)
        object.__setattr__(self, "_reads", reads)

    def _hit(self, how):
        what  = object.__getattribute__(self, "_what")
        stack = "".join(traceback.format_stack(limit=6)[:-2])
        object.__getattribute__(self, "_reads").append(f"{what} {how}\n{stack}")
        raise _Read(f"{what} {how}")

    def __getattr__(self, name):
        object.__getattribute__(self, "_hit")("." + name)

    def __bool__(self):
        object.__getattribute__(self, "_hit")("bool()")

    def __eq__(self, other):
        object.__getattribute__(self, "_hit")("==")

    def __ne__(self, other):
        object.__getattribute__(self, "_hit")("!=")

    def __hash__(self):
        object.__getattribute__(self, "_hit")("hash()")

    def __iter__(self):
        object.__getattribute__(self, "_hit")("iter()")

    def __len__(self):
        object.__getattribute__(self, "_hit")("len()")


def _wire(obj, label, reads, seen):
    """Swap every API object `obj` keeps (and the plugs and nodes it keeps
    keep) for a tripwire; its MObjectHandles stay (they are safe to read)."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, list):
        for item in obj:
            _wire(item, label, reads, seen)
    state = getattr(obj, "__dict__", None)
    if state is None:
        return
    for key, value in list(state.items()):
        if isinstance(value, OpenMaya.MObjectHandle):
            continue
        if isinstance(value, _API_TYPES):
            state[key] = _Tripwire(f"{label}: {key}", reads)
        elif isinstance(value, (Attribute, _NodeBase, list)):
            _wire(value, label, reads, seen)
        elif isinstance(value, dict):
            for k, v in list(value.items()):
                if isinstance(v, _API_TYPES):
                    value[k] = _Tripwire(f"{label}: {key}[{k!r}]", reads)
                elif isinstance(v, (Attribute, _NodeBase)):
                    _wire(v, label, reads, seen)


def _held():
    """The plugs and nodes a rigger can hold across a file open: every plug
    with no owner of round 3b, owned ones, multi (array) roots (`Plug.get`
    read their MPlug first), and node objects."""
    held = {label: attr for label, (attr, _node) in r3b._unowned("").items()}
    held.update({
        "Plug(str) multi":     Plug("pma.input1D"),
        "Plug(MPlug) multi":   Plug(r3b._mplug("pma.input1D")),
        "Attribute(str) multi": Attribute("pma.input1D"),
        "Plug worldMatrix":    Plug("held.worldMatrix"),
        "owned tx":            Node("held").tx,
        "owned multi":         Node("pma").input1D,
        "owned element":       Node("pma").input1D[0],
        "owned worldMesh":     Node("pcShape").worldMesh,
        "owned vtx":           Node("pcShape").vtx[1],
        "owned vtx slice":     Node("pcShape").vtx[0:3],
        "owned cv":            Node("npShape").cv,
        "find_attr multi":     Node("pma").find_attr("input1D"),
        "get_inputs()[0] owned": Node("held").tx.get_inputs()[0],
    })
    nodes = {
        "node held":    Node("held"),
        "node pma":     Node("pma"),
        "node pc":      Node("pc"),
        "node pcShape": Node("pcShape"),
    }
    return held, nodes


# the operations of a held plug that must raise "... already deleted!"
_PLUG_OPS = {
    "str":          str,
    "get":          lambda p: p.get(),
    ">> None":      lambda p: p >> None,
    "get_mplug":    lambda p: p.get_mplug(),
    "Plug(get_mplug())": lambda p: Plug(p.get_mplug()),
    "get_full_name": lambda p: p.get_full_name(),
    "get_data_type": lambda p: p.get_data_type(),
    "get_data_fn_set": lambda p: p.get_data_fn_set(),
    "Node(plug)":   Node,
    "in Tag":       lambda p: p in Tag("probe"),
    "in Layer":     lambda p: p in Layer("probeLayer"),
    "Tag.of":       lambda p: Tag.of(p),
    "Layer.of":     lambda p: Layer.of(p),
    "[p] in Tag":   lambda p: [p] in Tag("probe"),
}
_DSL_OPS = {
    ">> None":      lambda p: p >> None,
    "<< 1":         lambda p: p << 1,
    "<< Tag":       lambda p: p << Tag("probe"),
    ">> Tag":       lambda p: p >> Tag("probe"),
    "+ 1":          lambda p: p + 1,
}
_NODE_OPS = {
    "get_name":     lambda n: n.get_name(),
    "get_namespace": lambda n: n.get_namespace(),
    "get_fn_set":   lambda n: n.get_fn_set(),
    "in Tag":       lambda n: n in Tag("probe"),
    "Tag.of":       lambda n: Tag.of(n),
    "[n] in Tag":   lambda n: [n] in Tag("probe"),
}
# the operations that answer, without reading the freed node (user decision
# 2026-10-16): a node's str and repr are its marked last known name, a plug's
# repr the name it was built with, marked; a List skips a freed element
_ANSWERS = {
    "repr plug":    repr,
    "List.get":     lambda x: List([x]).get(),
    "List << Tag":  lambda x: List([x]) << Tag("probe"),
    "List << [Tag]": lambda x: List([x]) << [Tag("probe")],
    "List >> Tag":  lambda x: List([x]) >> Tag("probe"),
    "List << 1":    lambda x: List([x]) << 1,
    "List + 1":     lambda x: List([x]) + 1,
    "List.lock":    lambda x: List([x]).lock(),
    "x in List":    lambda x: x in List([x]),
}


class TestNoFreedReadAcrossAFileOpen(MayaTestCase):
    """Held across a file open of the same names (and a new scene), no plug or
    node reads an API object of its freed node."""

    TEST_START_NEW_SCENE = True

    def _freed(self, free):
        folder = tempfile.mkdtemp(prefix="rig_freed_reads_")
        self.addCleanup(shutil.rmtree, folder, True)
        self.addCleanup(cmds.file, new=True, force=True)
        path = os.path.join(folder, "held.ma").replace(os.sep, "/")
        r3b._build()
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        held, nodes = _held()
        if free == "open":
            cmds.file(path, open=True, force=True)  # new nodes take the names
        else:
            cmds.file(new=True, force=True)
        cmds.createDisplayLayer(name="probeLayer", empty=True)  # `Layer("probeLayer")`
        reads = []
        seen  = set()
        for label, obj in {**held, **nodes}.items():
            _wire(obj, label, reads, seen)
        return held, nodes, reads

    def _check(self, label, name, op, obj, reads, answers=False):
        with self.subTest(held=label, op=name):
            before = len(reads)
            try:
                op(obj)
            except _Read:
                pass
            except RuntimeError as error:
                if not answers:
                    self.assertRegex(str(error), r"already deleted!$")
            except Exception as error:  # noqa: BLE001
                if not answers:
                    self.fail(f"{name} of a freed {label} raised {type(error).__name__}: {error}")
            else:
                if not answers:
                    self.fail(f"{name} of a freed {label} returned")
            self.assertEqual(reads[before:], [], f"{name} read a freed API object")

    def _assert_no_freed_read(self, free):
        held, nodes, reads = self._freed(free)
        for label, attr in held.items():
            answers = dict(_ANSWERS)
            if isinstance(attr, List):
                # a List of freed plugs (``vtx[0:3]``): its operations skip
                # them, its str and repr show them marked
                ops = {}
                answers.update({"str": str, "List attr": lambda x: x.get_mplug})
            else:
                ops = dict(_PLUG_OPS)
                if isinstance(attr, Plug):
                    ops.update(_DSL_OPS)
                else:
                    # a bare Attribute is refused by the membership verbs as a
                    # str (no read)
                    for name in ("in Tag", "in Layer", "Tag.of", "Layer.of", "[p] in Tag"):
                        answers[name] = ops.pop(name)
                if vars(attr).get("_node") is not None:
                    # an owned plug hands its node object back
                    ops.pop("Node(plug)")
            for name, op in ops.items():
                self._check(label, name, op, attr, reads)
            for name, op in answers.items():
                self._check(label, name, op, attr, reads, answers=True)
        for label, node in nodes.items():
            for name, op in _NODE_OPS.items():
                self._check(label, name, op, node, reads)
            for name, op in _ANSWERS.items():
                if name != "repr plug":
                    self._check(label, name, op, node, reads, answers=True)
            # str and repr answer: the marked last known name (user decision
            # 2026-10-16), read with no API object
            with self.subTest(held=label, op="str / repr"):
                before = len(reads)
                short  = label.split()[-1]
                self.assertEqual(str(node), f"{short} (deleted)")
                self.assertEqual(repr(node), f'{type(node).__name__}("{short}", deleted)')
                self.assertEqual(reads[before:], [])

    def test_across_a_file_open_of_the_same_names(self):
        self._assert_no_freed_read("open")

    def test_across_a_new_scene(self):
        self._assert_no_freed_read("new")

    def test_no_public_member_reads_a_freed_api_object(self):
        # every public method taking no argument, of every held plug and node
        # class: it may raise or answer (a str method, a class constant), but
        # never read the freed node (a new method that reads its MPlug, fn set
        # or MObject before its check fails here)
        held, nodes, reads = self._freed("open")
        for label, obj in {**held, **nodes}.items():
            cls = type(obj)
            for name in sorted(dir(cls)):
                if name.startswith("_") or name == "delete" or hasattr(str, name):
                    continue
                with self.subTest(held=label, member=name):
                    before = len(reads)
                    try:
                        value = getattr(obj, name)
                        if callable(value) and not isinstance(value, (Attribute, _NodeBase)):
                            result = value()
                            if hasattr(result, "__next__"):
                                list(result)
                    except _Read:
                        pass
                    except Exception:  # noqa: BLE001 -- raising is fine
                        pass
                    self.assertEqual(reads[before:], [], f"{name} read a freed API object")

    def test_the_tripwire_sees_the_read_it_guards_against(self):
        # the probe itself: a raw read of a held MPlug once freed is caught
        held, _nodes, reads = self._freed("open")
        with self.assertRaises(_Read):
            vars(held["Plug(str) multi"])["_mplug"].isArray
        self.assertEqual(len(reads), 1)
        self.assertIn("Plug(str) multi: _mplug .isArray", reads[0])

    def test_a_freed_list_element_is_skipped_unread(self):
        # a List holding a freed node and plug answers for the rest, with
        # rig.skip in their slots (user decision 2026-10-16), reading nothing
        held, nodes, reads = self._freed("open")
        live  = Node(cmds.createNode("transform", name="live"))
        items = List([nodes["node held"], live])
        plugs = List([held["owned tx"], live.tx])
        self.assertEqual(items.tx, List([skip, live.tx]))
        plugs << 2.0
        self.assertEqual(cmds.getAttr("live.tx"), 2.0)
        self.assertEqual(plugs.get(), [skip, 2.0])
        self.assertEqual(reads, [])


# The audit: run in its own mayapy (a crash fails the test, not the run). The
# held objects of `_held` plus a choice multi, Lists and a memo result are freed
# by a file open of the same names and wired; then every method is called with
# arguments, the binary operators with a live operand on either side, and every
# public function of rig's modules with the freed object in each position. Any
# tripwire read is a freed-memory read.
_AUDIT = r"""
import collections, inspect, json, os, shutil, sys, tempfile
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    import rig
    from rig import List, Node, Plug, Tag
    from rig._tests.test_rig import test_r3b_freed_reads as frt
    from rig._tests.test_rig import test_r3b_unowned_plugs as r3b

    folder = tempfile.mkdtemp(prefix="rig_freed_audit_")
    path = os.path.join(folder, "held.ma").replace(os.sep, "/")
    cmds.file(new=True, force=True)
    r3b._build()
    cmds.createNode("choice", name="ch")
    cmds.connectAttr("src.ty", "ch.input[0]")
    cmds.file(rename=path)
    cmds.file(save=True, type="mayaAscii", force=True)
    held, nodes = frt._held()
    held["choice input"] = Node("ch").input
    held["List plugs"] = List([Node("held").tx, Plug("held.ty")])
    held["List nodes"] = List([Node("held"), Node("src")])
    held["memo add"] = Node("held").tx + Node("src").ty
    nodes["node np"] = Node("npShape")
    nodes["node ch"] = Node("ch")
    cmds.file(path, open=True, force=True)
    cmds.createDisplayLayer(name="probeLayer", empty=True)
    cmds.createNode("transform", name="live")
    cmds.polyCube(name="livecube", constructionHistory=False)
    live_plug = Node("live").tx
    live_node = Node("live")
    reads = []
    seen = set()
    for label, obj in {**held, **nodes}.items():
        frt._wire(obj, label, reads, seen)

    ARGS = [("tx",), ("translateX",), (1,), (0.5,), ("x",), (True,)]
    findings = collections.OrderedDict()
    counts = collections.Counter()

    def call(where, fn, *a):
        before = len(reads)
        try:
            r = fn(*a)
            if hasattr(r, "__next__"):
                list(r)
            counts["ret"] += 1
        except frt._Read:
            counts["read"] += 1
        except Exception as e:  # noqa: BLE001
            counts["deleted" if "already deleted" in str(e) else "exc"] += 1
        new = reads[before:]
        if new:
            findings.setdefault(where, new[0][:1500])
        del reads[before:]

    MODULES = ["functions", "vector", "matrix", "trigonometry", "interpolate", "euler",
               "quaternion", "random", "tween", "utils", "membership", "shade"]
    mods = {m: __import__("rig." + m, fromlist=["x"]) for m in MODULES}
    for label, obj in {**held, **nodes}.items():
        for name in sorted(dir(type(obj))):
            if name.startswith("_") or name == "delete":
                continue
            try:
                member = inspect.getattr_static(obj, name)
            except AttributeError:
                continue
            if isinstance(member, property):
                continue
            for a in ARGS + [(live_plug,), (live_node,), (obj,)]:
                call(f"{label} .{name}{tuple(type(x).__name__ for x in a)}",
                     lambda n=name, a=a: getattr(obj, n)(*a))
        for opname, op in {
            "live << x": lambda: live_plug << obj, "x >> live": lambda: obj >> live_plug,
            "live + x": lambda: live_plug + obj, "x + live": lambda: obj + live_plug,
            "live == x": lambda: live_plug == obj, "live.equals(x)": lambda: live_plug.equals(obj),
            "x in List([live])": lambda: obj in List([live_plug]),
            "List([live,x]) << 1": lambda: List([live_plug, obj]) << 1,
            "List([live,x]).get": lambda: List([live_plug, obj]).get(),
            "List([live,x]) + 1": lambda: List([live_plug, obj]) + 1,
            "List([live,x]).tx": lambda: List([live_node, obj]).tx,
            "live_node << x": lambda: live_node << obj,
            "x << live_node": lambda: obj << live_node,
            "x >> live_node": lambda: obj >> live_node,
            "live.set(x)": lambda: live_plug.set(obj),
            "live << [x]": lambda: live_plug << [obj],
            "List([x]) >> live_node": lambda: List([obj]) >> live_node,
            "List([live_node, x]).lock()": lambda: List([live_node, obj]).lock(),
            "Plug(x)": lambda: Plug(obj), "Node(x)": lambda: Node(obj),
            "cube.vtx in x": lambda: Node("livecubeShape").vtx[:2] in obj,
            "x in Tag": lambda: obj in Tag("probe"),
            "dict key": lambda: {obj: 1}[obj],
            "sorted": lambda: sorted([obj, live_plug], key=str),
            "get_closest_point(x)": lambda: Node("livecubeShape").get_closest_point(obj),
        }.items():
            call(f"{label} {opname}", op)
        for mname, mod in mods.items():
            for fname in sorted(n for n in dir(mod) if not n.startswith("_")):
                fn = getattr(mod, fname)
                if not callable(fn) or inspect.isclass(fn) or inspect.ismodule(fn):
                    continue
                if not str(getattr(fn, "__module__", "") or "").startswith("rig"):
                    continue
                for a in ((obj,), (obj, live_plug), (live_plug, obj), (obj, 0.5), (obj, obj),
                          (live_plug, live_plug, obj)):
                    call(f"{label} {mname}.{fname}{tuple('X' if x is obj else type(x).__name__ for x in a)}",
                         fn, *a)
    cmds.file(new=True, force=True)
    shutil.rmtree(folder, ignore_errors=True)
    print("RESULT " + json.dumps({"file": rig.__file__, "counts": dict(counts),
                                  "findings": dict(list(findings.items())[:20])}))
    sys.stdout.flush()
finally:
    try:
        maya.standalone.uninitialize()
    except BaseException:
        pass
    sys.stdout.flush()
    os._exit(0)
"""


class TestNoFreedReadWithArguments(MayaTestCase):
    """The r3b verifier's audit as a deterministic test (see `_AUDIT`)."""

    def test_methods_with_arguments_and_module_functions_read_nothing_freed(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
        env["MAYA_DISABLE_CER"] = "1"
        cwd = tempfile.mkdtemp(prefix="rig_freed_audit_cwd_")
        try:
            proc = subprocess.run([sys.executable, "-c", _AUDIT], cwd=cwd, env=env, timeout=1200,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        out = proc.stdout.decode("utf-8", "replace")
        err = proc.stderr.decode("utf-8", "replace")
        lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
        result = json.loads(lines[0][len("RESULT "):])
        here = os.path.normcase(os.path.dirname(rig.__file__))
        self.assertEqual(os.path.normcase(os.path.dirname(result["file"])), here)
        self.assertEqual(result["findings"], {})
        self.assertEqual(result["counts"].get("read", 0), 0)
        # the walk ran: tens of thousands of calls, most of them refused
        self.assertGreater(sum(result["counts"].values()), 50000)
        self.assertGreater(result["counts"].get("deleted", 0), 30000)
