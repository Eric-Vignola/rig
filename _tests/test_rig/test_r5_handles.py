"""Round 5, step A1 (NW6, F17): the validity handles are API 2.0, and a plug
holds the handle of its node in one slot.

A node object's handle (``_objhandle``) is an API 2.0 ``MObjectHandle`` built
from the MObject its constructor resolved, while the node is alive (the
constructors, the cast's replay, a copy). A plug holds the handle of its node
as ``_handle``: its owner's ``_objhandle`` object itself, or, with no owner,
the one it took when it was built from a name or an MPlug (the MPlug's node).
A dynamic attr's plug holds the handle of its attribute as ``_attrh``.

* ``TestNoApi1Left``: the tripwire (no package line names the API 1.0 handle,
  fn set or plug slots, the retired node-handle lookup and table, or imports
  ``maya.OpenMaya``), and no node, plug or memo entry holds an API 1.0 object;
  the handles are not GC-tracked.
* ``TestTheOutcomesOf45d2c13``: for a DG node, a DAG node, owned plugs, a plug
  with no owner, dynamic attr plugs and memo entries, every outcome (the value,
  or the error type and message) across a delete, its undo and redo, a flush, a
  new scene, a file open, a reference unload and a same-name re-create after a
  flush is the one 45d2c13 gave (``_ORACLE_45D2C13``, written by
  ``runs\\r5\\A1\\gen_oracle.py`` running ``_outcomes`` on rig_r5base).
* ``TestOneHandlePerPlug`` (F17): an owned plug, its children, elements and
  parent, its copies, the attrs a node finds, a spec's plug and a component
  element hold their owner's ``_objhandle`` itself; a plug with no owner holds
  the one it took, and its owner's once it casts its node.
* ``TestOwnerlessHandles``: a plug built from an MPlug holds a valid handle of
  its own node on the names that took a unique-name round trip before (a
  non-unique short name, an instance, namespaces, a reference, an underworld
  curve, a DAG node beside a DG node of its name; at the root, in another
  namespace and with ``namespace -relativeNames``); 45d2c13 gave every one a
  handle of its node too (``runs\\r5\\A1\\probe_attr_handles_base.txt``). A
  null plug holds none.
* ``TestAttributeHandles``: the attribute-handle rule is 45d2c13's: a plug
  rebuilt from the MPlug of a dynamic attribute deleted to the undo queue holds
  its handle when rig had found that attribute before, else none, whether or
  not the name was added again; the memo makes no attribute check for an
  attribute its name no longer finds.
* ``TestSerials``: a plug with no owner of a node deleted to the undo queue
  hashes with the serial its node object had (the serial table keeps the
  node's MObject: an API 2.0 handle gives none for a deleted node), and two
  deleted nodes keep two serials.
* ``TestCollectHandles``: the memo's ``_collect_handles`` gives the handles the
  by-name route gives (of the same nodes, with the same answers) for owned DG
  and DAG plugs, an instanced one, a component element, plugs with no owner, a
  deleted and a freed node's plugs, a freed dynamic attribute, node objects and
  a node class with its own ``name``; an owned DG plug is answered without a
  name lookup.
"""

import gc
import os
import re
import shutil
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import Node, Plug, functions
from rig._internal import memoize as memoize_module
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute
from rig.nodetypes.dg_node import DGNode
from rig.spec import Float
from rig._tests._base import MayaTestCase


def _mplug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _outcome(call):
    """``["returns", value]`` or ``["raises", error type name, message]``."""
    try:
        value = call()
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ["raises", type(error).__name__, str(error)]
    return ["returns", value]


# --------------------------------------------------------------------------- #
#  The outcomes table (`_outcomes` runs on 45d2c13 too: public API only)
# --------------------------------------------------------------------------- #

_FREES = ("delete", "undo", "redo", "flush", "new", "open", "unload", "recreate")


def _build_targets(prefix=""):
    """``md`` (a multiplyDivide, input1X 3, a dynamic double ``dynf`` 0.25) and
    ``xf`` (a transform, tx 5), named with `prefix` ('' or a namespace)."""
    md = cmds.createNode("multiplyDivide", name=f"{prefix}md")
    cmds.setAttr(f"{md}.input1X", 3)
    cmds.addAttr(md, longName="dynf", attributeType="double")
    cmds.setAttr(f"{md}.dynf", 0.25)
    xf = cmds.createNode("transform", name=f"{prefix}xf")
    cmds.setAttr(f"{xf}.tx", 5)


def _hold(prefix=""):
    """The objects held across a free, built (and used) before it."""
    held = {}
    dg  = held["dg"]  = Node(f"{prefix}md")
    dag = held["dag"] = Node(f"{prefix}xf")
    held["owned"]     = dg.input1X
    held["owned_dag"] = dag.tx
    held["loose"]     = Plug(f"{prefix}md.input1Y")  # no owner: never asked for its node
    held["dyn"]       = dg.dynf
    held["loose_dyn"] = Plug(f"{prefix}md.dynf")
    held["arg"]       = dg.outputX
    held["h_owned"]     = hash(held["owned"])
    held["h_owned_dag"] = hash(held["owned_dag"])
    held["first"]     = functions.choice([held["arg"], 1.0], 0)
    held["first_dyn"] = functions.choice([held["dyn"], 2.0], 0)
    return held


def _ops(held):
    """The operations whose outcomes are compared, in the order they run."""
    dg, dag = held["dg"], held["dag"]
    return (
        ("dg.name", lambda: dg.name),
        ("dg.is_valid", lambda: dg.is_valid),
        ("dg.ensure_valid", lambda: dg.ensure_valid()),
        ("dg.attr", lambda: str(dg.input2X)),
        ("dag.name", lambda: dag.name),
        ("dag.is_valid", lambda: dag.is_valid),
        ("dag.attr", lambda: dag.tx.get()),
        ("owned.str", lambda: str(held["owned"])),
        ("owned.get", lambda: held["owned"].get()),
        ("owned.set", lambda: (held["owned"] << 4.0) is held["owned"]),
        ("owned.hash", lambda: hash(held["owned"]) == held["h_owned"]),
        ("owned_dag.get", lambda: held["owned_dag"].get()),
        ("owned_dag.hash", lambda: hash(held["owned_dag"]) == held["h_owned_dag"]),
        ("loose.str", lambda: str(held["loose"])),
        ("loose.get", lambda: held["loose"].get()),
        ("loose.node", lambda: held["loose"].node.name),
        ("dyn.str", lambda: str(held["dyn"])),
        ("dyn.get", lambda: held["dyn"].get()),
        ("dyn.set", lambda: (held["dyn"] << 0.5) is held["dyn"]),
        ("loose_dyn.get", lambda: held["loose_dyn"].get()),
        ("memo.again", lambda: functions.choice([held["arg"], 1.0], 0) is held["first"]),
        ("memo_dyn.again", lambda: functions.choice([held["dyn"], 2.0], 0) is held["first_dyn"]),
        ("memo.result", lambda: str(held["first"])),
    )


def _churn():
    """Nodes made after a free, so freed memory is reused."""
    for _ in range(100):
        cmds.createNode("multiplyDivide")


def _outcomes():
    """{free: {op: outcome}}: each free in a fresh scene, with an infinite undo
    queue (restored after)."""
    table  = {}
    folder = tempfile.mkdtemp(prefix="rig_r5_handles_")
    try:
        for free in _FREES:
            cmds.file(new=True, force=True)
            cmds.undoInfo(state=True, infinity=True)
            cmds.flushUndo()
            prefix = ""
            if free == "unload":
                _build_targets()
                path = os.path.join(folder, "targets.ma").replace(os.sep, "/")
                cmds.file(rename=path)
                cmds.file(save=True, type="mayaAscii", force=True)
                cmds.file(new=True, force=True)
                cmds.file(path, reference=True, namespace="ref")
                prefix = "ref:"
            else:
                _build_targets()
            if free == "open":
                path = os.path.join(folder, "scene.ma").replace(os.sep, "/")
                cmds.file(rename=path)
                cmds.file(save=True, type="mayaAscii", force=True)
            held = _hold(prefix)
            if free in ("delete", "undo", "redo", "flush", "recreate"):
                cmds.delete(f"{prefix}md", f"{prefix}xf")
            if free in ("undo", "redo"):
                cmds.undo()
            if free == "redo":
                cmds.redo()
            if free in ("flush", "recreate"):
                cmds.flushUndo()
                _churn()
            if free == "recreate":
                _build_targets()
            if free == "new":
                cmds.file(new=True, force=True)
                _churn()
            if free == "open":
                cmds.file(path, open=True, force=True)
                _churn()
            if free == "unload":
                cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
                _churn()
            table[free] = {label: _outcome(op) for label, op in _ops(held)}
    finally:
        cmds.file(new=True, force=True)
        cmds.undoInfo(state=True, infinity=True)
        shutil.rmtree(folder, ignore_errors=True)
    return table


# --------------------------------------------------------------------------- #
#  The tests
# --------------------------------------------------------------------------- #

# a line of the package that names an API 1.0 handle, its fn set, the retired
# plug slots, the retired node-handle lookup and table, or imports API 1.0
_TRIPWIRE = re.compile(
    r"OpenMaya1|_objhandle1|_fn_set1|_cache_api1|_handle1|_attr1|_NODE_HANDLES"
    r"|\b_node_handle\b|_prune_node_handles"
    r"|from maya import [^\n#]*\bOpenMaya\b|import maya\.OpenMaya\b"
)


class TestNoApi1Left(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_the_tripwire(self):
        root = os.path.dirname(rig.__file__)
        hits = []
        for base, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in ("_tests", "__pycache__"))
            for name in sorted(f for f in files if f.endswith(".py")):
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if _TRIPWIRE.search(line):
                            hits.append(f"{os.path.relpath(path, root)}:{number}: {line.strip()}")
        self.assertEqual(hits, [])

    def test_no_api1_object_is_held(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dynf", attributeType="double")
        cmds.createNode("transform", name="xf")
        cmds.sphere(name="ball", constructionHistory=False)
        md, xf = Node("md"), Node("xf")

        @memoize_module.memoize
        def memo(plug):
            return plug * 2

        memo(md.dynf)
        plugs = {
            "owned": md.input1X,
            "owned dag": xf.tx,
            "child": xf.t.tx,
            "element": xf.worldMatrix[0],
            "no owner": Plug("md.input1Y"),
            "dynamic": md.dynf,
            "typed": md.find_attr("input2X"),
            "component": Node("ballShape").cv[1, 2],
        }
        objects = {"dg": vars(md), "dag": vars(xf)}
        objects.update({label: vars(plug) for label, plug in plugs.items()})
        entries = list(memo._cache.values())
        self.assertEqual(len(entries), 1)
        objects["memo handles"] = dict(enumerate(entries[0].handles))
        checks = [h for h in entries[0].handles if isinstance(h, memoize_module._AttrCheck)]
        self.assertEqual(len(checks), 1)
        objects["memo attr check"] = {
            slot: getattr(checks[0], slot) for slot in memoize_module._AttrCheck.__slots__
        }
        for label, state in objects.items():
            for key, value in state.items():
                with self.subTest(object=label, key=key):
                    self.assertNotEqual(type(value).__module__, "maya.OpenMaya")
        for label, (value, key) in {
            "dg": (md, "_objhandle"), "dag": (xf, "_objhandle"),
            "plug": (plugs["no owner"], "_handle"), "attr": (plugs["dynamic"], "_attrh"),
        }.items():
            with self.subTest(handle=label):
                handle = vars(value)[key]
                self.assertIsInstance(handle, OpenMaya.MObjectHandle)
                self.assertFalse(gc.is_tracked(handle))


class TestTheOutcomesOf45d2c13(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_every_outcome_is_45d2c13s(self):
        table = _outcomes()
        self.assertEqual(sorted(table), sorted(_ORACLE_45D2C13))
        for free, expected in _ORACLE_45D2C13.items():
            self.assertEqual(sorted(table[free]), sorted(expected), free)
            for op, outcome in expected.items():
                with self.subTest(free=free, op=op):
                    self.assertEqual(table[free][op], outcome)


class TestOneHandlePerPlug(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_owned_plugs_hold_their_owners_handle(self):
        cmds.createNode("transform", name="t")
        cmds.aliasAttr("hoist", "t.tz")
        cmds.createNode("multiplyDivide", name="md")
        cmds.sphere(name="ball", constructionHistory=False)
        t, md, ball = Node("t"), Node("md"), Node("ballShape")
        owned = {
            "plug": (t, t.tx),
            "compound": (t, t.t),
            "child by name": (t, t.t.tx),
            "child [0]": (t, t.t[0]),
            "child(1)": (t, t.t.child(1)),
            "element": (t, t.worldMatrix[0]),
            "element of a slice": (t, t.worldMatrix[0:1][0]),
            "Plug(plug)": (t, Plug(t.tx)),
            "find_attr": (t, t.find_attr("tx")),
            "typed child": (t, t.find_attr("t").child(0)),
            "typed parent": (t, t.find_attr("tx").get_parent()),
            "spec": (t, t << Float("k")),
            "dg plug": (md, md.input1X),
            "dg compound child": (md, md.input1.input1Y),
            "component element": (ball, ball.cv[1, 2]),
        }
        for label, (node, plug) in owned.items():
            with self.subTest(plug=label):
                self.assertIs(vars(plug)["_node"], node)
                self.assertIs(vars(plug)["_handle"], vars(node)["_objhandle"])
        # a node's alias lookup: no owner, the node's handle
        alias = t.find_alias("hoist")
        self.assertIsNone(vars(alias)["_node"])
        self.assertIs(vars(alias)["_handle"], vars(t)["_objhandle"])

    def test_a_plug_takes_its_owners_handle_when_it_casts_its_node(self):
        cmds.createNode("transform", name="t")
        plug = Plug("t.tx")
        own  = vars(plug)["_handle"]
        self.assertIsNone(vars(plug)["_node"])
        self.assertIsInstance(own, OpenMaya.MObjectHandle)
        self.assertTrue(own.isValid())
        node = plug.node
        self.assertIs(vars(plug)["_handle"], vars(node)["_objhandle"])
        self.assertEqual(vars(plug)["_handle"].hashCode(), own.hashCode())
        # the plugs derived from it after hold that handle too
        parent = Plug("t.t")
        parent.node
        child  = parent.translateX
        self.assertIs(vars(child)["_node"], vars(parent)["_node"])
        self.assertIs(vars(child)["_handle"], vars(parent.node)["_objhandle"])

    def test_the_handles_answer_alike(self):
        # the owner's handle and a plug's own handle of one node answer the
        # same across a delete, its undo and a new scene
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="t")
        owned, loose = Node("t").tx, Plug("t.ty")
        handles = (vars(owned)["_handle"], vars(loose)["_handle"])
        self.assertIsNot(handles[0], handles[1])

        def answers():
            return [(h.isValid(), h.isAlive(), h.hashCode()) for h in handles]

        self.assertEqual(answers()[0], answers()[1])
        cmds.delete("t")
        self.assertEqual(answers()[0], answers()[1])
        self.assertEqual(answers()[0][:2], (False, True))
        cmds.undo()
        self.assertEqual(answers()[0][:2], (True, True))
        cmds.file(new=True, force=True)
        self.assertEqual([a[:2] for a in answers()], [(False, False), (False, False)])


class TestOwnerlessHandles(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _scene(self, folder):
        ref = os.path.join(folder, "ref.ma").replace(os.sep, "/")
        cmds.createNode("multiplyDivide", name="refmd")
        cmds.file(rename=ref)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        for group in ("g1", "g2"):
            cmds.createNode("transform", name=group)
            cmds.createNode("transform", name="dup", parent=group)
        cmds.createNode("transform", name="inst_a")
        cmds.createNode("locator", name="instShape", parent="inst_a")
        cmds.createNode("transform", name="inst_b")
        cmds.parent("|inst_a|instShape", "inst_b", addObject=True, shape=True)
        cmds.namespace(add="ns")
        cmds.namespace(add="deep", parent="ns")
        cmds.createNode("multiplyDivide", name="ns:md")
        cmds.createNode("multiplyDivide", name="ns:deep:md")
        cmds.createNode("multiplyDivide", name="md")
        surface = cmds.nurbsPlane(name="plane", constructionHistory=False)[0]
        cmds.curveOnSurface(surface, uv=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.1)], degree=1)
        cmds.file(ref, reference=True, namespace="R")
        cmds.createNode("multiplyDivide", name="same")
        beside = cmds.createNode("transform", name="same")  # Maya names it same1
        names = [
            "|g1|dup.tx", "|g2|dup.tx", "|inst_a|instShape.visibility",
            "|inst_b|instShape.visibility", "ns:md.input1X", "ns:deep:md.input1X",
            "md.input1X", "R:refmd.input1X", f"|{beside}.tx",
        ]
        names += [f"{curve}.visibility" for curve in cmds.ls(type="nurbsCurve", long=True)]
        return names

    def test_every_hard_name_takes_its_nodes_handle(self):
        folder = tempfile.mkdtemp(prefix="rig_r5_ownerless_")
        try:
            names = self._scene(folder)
            self.assertEqual(len(names), 10)  # one underworld curve
            for name in names:
                mplug = _mplug(name)
                code  = OpenMaya.MObjectHandle(mplug.node()).hashCode()
                for mode in ("root", "ns current", "ns current, relativeNames"):
                    try:
                        if mode != "root":
                            cmds.namespace(setNamespace=":ns")
                        if mode.endswith("relativeNames"):
                            cmds.namespace(relativeNames=True)
                        for build in (Plug, Attribute, _base._cast):
                            with self.subTest(name=name, mode=mode, build=build.__name__):
                                plug   = build(mplug)
                                handle = vars(plug)["_handle"]
                                self.assertIsNone(vars(plug)["_node"])
                                self.assertIsInstance(handle, OpenMaya.MObjectHandle)
                                self.assertTrue(handle.isValid())
                                self.assertEqual(handle.hashCode(), code)
                    finally:
                        cmds.namespace(relativeNames=False)
                        cmds.namespace(setNamespace=":")
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_null_plug_holds_none(self):
        self.assertIsNone(_base._mplug_handle(OpenMaya.MPlug()))
        self.assertIsNone(vars(Attribute(OpenMaya.MPlug()))["_handle"])
        self.assertIsNone(vars(Attribute(OpenMaya.MPlug()))["_attrh"])


class TestAttributeHandles(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_a_plug_rebuilt_from_a_deleted_attributes_mplug(self):
        # 45d2c13's answers (runs\r5\A1\probe_attr_handles_base.txt, part 3):
        # the handle when rig found the attribute before, else none
        for readd in (False, True):
            for seen in (True, False):
                with self.subTest(readd=readd, seen=seen):
                    cmds.file(new=True, force=True)
                    cmds.undoInfo(state=True, infinity=True)
                    cmds.flushUndo()
                    cmds.createNode("multiplyDivide", name="hm")
                    cmds.addAttr("hm", longName="dynf", attributeType="double")
                    raw = _mplug("hm.dynf")
                    if seen:
                        self.assertIsNotNone(vars(Node("hm").dynf)["_attrh"])
                    cmds.deleteAttr("hm.dynf")
                    if readd:
                        cmds.addAttr("hm", longName="dynf", attributeType="double")
                    for build in (Attribute, Plug):
                        handle = vars(build(raw))["_attrh"]
                        if seen:
                            self.assertIsInstance(handle, OpenMaya.MObjectHandle)
                            self.assertTrue(handle.isAlive())
                            code = OpenMaya.MObjectHandle(raw.attribute()).hashCode()
                            self.assertEqual(handle.hashCode(), code)
                        else:
                            self.assertIsNone(handle)

    def test_no_memo_check_for_an_attribute_its_name_no_longer_finds(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("multiplyDivide", name="hm")
        cmds.addAttr("hm", longName="dynf", attributeType="double")
        plug  = Node("hm").dynf
        check = memoize_module._attr_check(plug)
        self.assertTrue(check.isAlive() and check.isValid())
        cmds.deleteAttr("hm.dynf")
        self.assertFalse(check.isAlive() and check.isValid())
        self.assertIsNone(memoize_module._attr_check(plug))  # 45d2c13's: none
        cmds.undo()
        again = memoize_module._attr_check(plug)
        self.assertTrue(again.isAlive() and again.isValid())
        self.assertTrue(check.isAlive() and check.isValid())


class TestSerials(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_an_ownerless_plug_of_a_deleted_node_keeps_its_nodes_serial(self):
        cmds.undoInfo(state=True, infinity=True)
        hashes, loose = {}, {}
        for name in ("a", "b"):
            node = Node(cmds.createNode("transform", name=name))
            hashes[name] = hash(node.tx)
            loose[name]  = Plug(f"{name}.tx")  # no owner, not hashed yet
        cmds.delete("a", "b")
        for name in ("a", "b"):
            with self.subTest(name=name):
                self.assertIsNone(vars(loose[name])["_node"])
                self.assertFalse(vars(loose[name])["_handle"].isValid())
                self.assertEqual(hash(loose[name]), hashes[name])
        self.assertNotEqual(hash(loose["a"]), hash(loose["b"]))
        cmds.undo()
        for name in ("a", "b"):
            self.assertEqual(hash(Plug(f"{name}.tx")), hashes[name])
            self.assertEqual(hash(Node(name).tx), hashes[name])
        for entries in _base._NODE_SERIALS.values():
            for handle, mobject, serial in entries:
                self.assertIsInstance(handle, OpenMaya.MObjectHandle)
                self.assertIsInstance(mobject, OpenMaya.MObject)
                self.assertIsInstance(serial, int)


def _by_name(obj):
    """The handles the by-name route gives `obj`: named (raising: none), the
    node its name names, resolved by name."""
    out = []
    try:
        name = obj.full_name.split(".", 1)[0] if isinstance(obj, Attribute) else str(obj)
        sel  = OpenMaya.MSelectionList()
        sel.add(name)
        out.append(OpenMaya.MObjectHandle(sel.getDependNode(0)))
    except Exception:  # noqa: BLE001 - naming raised: nothing, as the memo does
        pass
    return out


def _answers(handles):
    return [(h.hashCode(), h.isAlive(), h.isValid()) for h in handles]


class TestCollectHandles(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        _base._CLASS_BY_TYPE.clear()
        _base._CASTABLE_TYPES.clear()
        super().tearDown()

    def _objects(self, renamed):
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dynf", attributeType="double")
        cmds.createNode("transform", name="T1")
        cmds.createNode("locator", name="S", parent="T1")
        cmds.createNode("transform", name="T2")
        cmds.parent("|T1|S", "T2", addObject=True, shape=True)
        cmds.sphere(name="ball", constructionHistory=False)
        md, t2s = Node("md"), Node("|T2|S")
        return {
            "owned dg": md.input1X,
            "owned dg dynamic": md.dynf,
            "owned dag": Node("T1").tx,
            "instanced": t2s.visibility,
            "component element": Node("ballShape").cv[1, 2],
            "no owner": Plug("md.input1Y"),
            "typed": md.find_attr("input2X"),
            "dg node": md,
            "dag node": t2s,
            "renamed class": renamed._wrap("md"),
            "renamed class plug": renamed._wrap("md").input1Z,
        }

    def _assert_by_name(self, objects, state):
        for label, obj in objects.items():
            with self.subTest(state=state, object=label):
                out = []
                memoize_module._collect_handles(obj, out)
                self.assertEqual(_answers(out), _answers(_by_name(obj)))

    def test_the_handles_are_the_by_name_routes(self):
        class _Renamed(DGNode):
            """A DG node class with a `name` of its own: named by name."""

            @property
            def name(self):
                return self.fn_set.name()

        cmds.undoInfo(state=True, infinity=True)
        self._assert_by_name(self._objects(_Renamed), "live")
        cmds.file(new=True, force=True)
        deleted = self._objects(_Renamed)
        cmds.delete("md", "T1", "T2", "ball")
        self._assert_by_name(deleted, "deleted")
        cmds.file(new=True, force=True)
        freed = self._objects(_Renamed)
        cmds.file(new=True, force=True)
        for _ in range(50):
            cmds.createNode("multiplyDivide")
        self._assert_by_name(freed, "freed")
        cmds.file(new=True, force=True)
        attr = self._objects(_Renamed)["owned dg dynamic"]
        cmds.deleteAttr("md.dynf")
        cmds.flushUndo()
        self._assert_by_name({"freed dynamic attr": attr}, "freed attribute")

    def test_an_owned_dg_plug_makes_no_name_lookup(self):
        cmds.createNode("multiplyDivide", name="md")
        md   = Node("md")
        plug = md.input1X
        out  = []
        with mock.patch.object(OpenMaya, "MSelectionList", wraps=OpenMaya.MSelectionList) as spy:
            memoize_module._collect_handles(plug, out)
            memoize_module._collect_handles(md, out)
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], vars(md)["_objhandle"])
        self.assertIs(out[1], vars(md)["_objhandle"])


# the text of a freed node's error, after the node's class or name
_FREED = " node (freed by a new scene, a file open or a reference unload) already deleted!"

# The outcomes `_outcomes` gave on 45d2c13 (rig_r5base), written by
# runs\r5\A1\gen_oracle.py: {free: {op: ["returns", value] or ["raises", type, message]}}
_ORACLE_45D2C13 = {
    'delete': {
        'dag.attr': ['raises', 'RuntimeError', 'xf already deleted!'],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'xf already deleted!'],
        'dg.attr': ['raises', 'RuntimeError', 'md already deleted!'],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'md already deleted!'],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.set': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.node': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'memo.again': ['raises', 'RuntimeError', 'md already deleted!'],
        'memo.result': ['raises', 'RuntimeError', 'choice1 already deleted!'],
        'memo_dyn.again': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned_dag.get': ['raises', 'RuntimeError', 'xf already deleted!'],
        'owned_dag.hash': ['returns', True],
    },
    'undo': {
        'dag.attr': ['returns', 5.0],
        'dag.is_valid': ['returns', True],
        'dag.name': ['returns', 'xf'],
        'dg.attr': ['returns', 'md.input2X'],
        'dg.ensure_valid': ['returns', None],
        'dg.is_valid': ['returns', True],
        'dg.name': ['returns', 'md'],
        'dyn.get': ['returns', 0.25],
        'dyn.set': ['returns', True],
        'dyn.str': ['returns', 'md.dynf'],
        'loose.get': ['returns', 0.0],
        'loose.node': ['returns', 'md'],
        'loose.str': ['returns', 'md.input1Y'],
        'loose_dyn.get': ['returns', 0.5],
        'memo.again': ['returns', True],
        'memo.result': ['returns', 'choice1.output'],
        'memo_dyn.again': ['returns', True],
        'owned.get': ['returns', 3.0],
        'owned.hash': ['returns', True],
        'owned.set': ['returns', True],
        'owned.str': ['returns', 'md.input1X'],
        'owned_dag.get': ['returns', 5.0],
        'owned_dag.hash': ['returns', True],
    },
    'redo': {
        'dag.attr': ['raises', 'RuntimeError', 'xf already deleted!'],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'xf already deleted!'],
        'dg.attr': ['raises', 'RuntimeError', 'md already deleted!'],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'md already deleted!'],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.set': ['raises', 'RuntimeError', 'md already deleted!'],
        'dyn.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.node': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'memo.again': ['raises', 'RuntimeError', 'md already deleted!'],
        'memo.result': ['raises', 'RuntimeError', 'choice1 already deleted!'],
        'memo_dyn.again': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.get': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned.str': ['raises', 'RuntimeError', 'md already deleted!'],
        'owned_dag.get': ['raises', 'RuntimeError', 'xf already deleted!'],
        'owned_dag.hash': ['returns', True],
    },
    'flush': {
        'dag.attr': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dg.attr': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'loose.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.node': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.str': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'memo.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'memo.result': ['raises', 'RuntimeError', 'Choice' + _FREED],
        'memo_dyn.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned_dag.get': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'owned_dag.hash': ['returns', True],
    },
    'new': {
        'dag.attr': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dg.attr': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'loose.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.node': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.str': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'memo.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'memo.result': ['raises', 'RuntimeError', 'Choice' + _FREED],
        'memo_dyn.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned_dag.get': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'owned_dag.hash': ['returns', True],
    },
    'open': {
        'dag.attr': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dg.attr': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'loose.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.node': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.str': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'memo.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'memo.result': ['raises', 'RuntimeError', 'Choice' + _FREED],
        'memo_dyn.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned_dag.get': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'owned_dag.hash': ['returns', True],
    },
    'unload': {
        'dag.attr': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dg.attr': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'loose.get': ['raises', 'RuntimeError', 'ref:md' + _FREED],
        'loose.node': ['raises', 'RuntimeError', 'ref:md' + _FREED],
        'loose.str': ['raises', 'RuntimeError', 'ref:md' + _FREED],
        'loose_dyn.get': ['raises', 'RuntimeError', 'ref:md' + _FREED],
        'memo.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'memo.result': ['returns', 'choice1.output'],
        'memo_dyn.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned_dag.get': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'owned_dag.hash': ['returns', True],
    },
    'recreate': {
        'dag.attr': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dag.is_valid': ['returns', False],
        'dag.name': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'dg.attr': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.ensure_valid': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dg.is_valid': ['returns', False],
        'dg.name': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'dyn.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'loose.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.node': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose.str': ['raises', 'RuntimeError', 'md' + _FREED],
        'loose_dyn.get': ['raises', 'RuntimeError', 'md' + _FREED],
        'memo.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'memo.result': ['raises', 'RuntimeError', 'Choice' + _FREED],
        'memo_dyn.again': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.get': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.hash': ['returns', True],
        'owned.set': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned.str': ['raises', 'RuntimeError', 'DGNode' + _FREED],
        'owned_dag.get': ['raises', 'RuntimeError', 'Transform' + _FREED],
        'owned_dag.hash': ['returns', True],
    },
}
