"""Regression tests for round 3b: a plug with no owner checks its node.

A plug built from a string or a bare MPlug, never asked for its node, has no
owner handle for the round-3 guards to read. Its node was only reachable through
its MPlug, so once a new scene, a file open or a reference unload freed that node,
naming the plug read freed memory: a Maya crash, another node's name, or a wrong
value (``Plug("held.ty").get()`` returned 1000.1 after a new scene). A node
deleted to the undo queue was cast by name, so the plug read the new node that
took the name. Such a plug now takes an API 1.0 handle of its node when it is
built (its children, elements and parent share it), and raises the round-3
``"... already deleted!"`` instead.
"""

import os
import shutil
import tempfile
from unittest import mock

from maya import cmds, OpenMaya as om1
from maya.api import OpenMaya
from rig import lift, List, Node, Plug
from rig.nodetypes import PyNode, _base
from rig.nodetypes._base import Attribute
from rig._internal import plug as plug_module
from rig._internal.plug import ComponentPlug
from rig.spec import Float
from rig._tests._base import MayaTestCase


_FREED = (
    r"^[\w:|]+ node \(freed by a new scene, a file open or a reference unload\) "
    r"already deleted!$"
)


def _hash_code(name):
    """The API 1.0 MObjectHandle hashCode of the node `name`."""
    sel = om1.MSelectionList()
    sel.add(name)
    mobject = om1.MObject()
    sel.getDependNode(0, mobject)
    return om1.MObjectHandle(mobject).hashCode()


def _mplug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _build():
    """``held`` (a transform, ty 3, tz aliased ``hoist``), ``src`` driving
    held.tx, the cube ``pc``, the nurbs plane ``np`` and a ``pma`` with two
    input1D elements."""
    cmds.createNode("transform", name="held")
    cmds.setAttr("held.ty", 3)
    cmds.aliasAttr("hoist", "held.tz")
    cmds.createNode("transform", name="src")
    cmds.connectAttr("src.tx", "held.tx")
    cmds.polyCube(name="pc", constructionHistory=False)
    cmds.nurbsPlane(name="np", patchesU=3, patchesV=3, constructionHistory=False)
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.setAttr("pma.input1D[0]", 1)
    cmds.setAttr("pma.input1D[1]", 2)


def _unowned(prefix):
    """Every way to get a plug with no owner, keyed by a label, each with the
    name of the node it is on. Each derived plug has a parent of its own: an
    index into a compound, or a slice of a multi, names the parent (reading its
    component type), which casts its node first, so the result has an owner."""
    held = f"{prefix}held"
    pma  = f"{prefix}pma"
    cv   = Node(f"{prefix}npShape").cv.plug
    return {
        "Plug(str)":                  (Plug(f"{held}.ty"), held),
        "Plug(MPlug)":                (Plug(_mplug(f"{held}.ty")), held),
        "Plug(component str)":        (Plug(f"{prefix}pcShape.vtx[1]"), f"{prefix}pcShape"),
        "Attribute(str)":             (Attribute(f"{held}.ty"), held),
        "Attribute(MPlug)":           (Attribute(_mplug(f"{held}.ty")), held),
        "PyNode(str)":                (PyNode(f"{held}.ty"), held),
        "PyNode(MPlug)":              (PyNode(_mplug(f"{held}.ty")), held),
        "lift(str)":                  (lift(f"{held}.ty"), held),
        "List(str)[0]":               (List([f"{held}.ty"])[0], held),
        "Plug child by name":         (Plug(f"{held}.t").translateY, held),
        "Plug child(1)":              (Plug(f"{held}.t").child(1), held),
        "Plug compound [1]":          (Plug(f"{held}.t")[1], held),
        "Plug compound [1:2][0]":     (Plug(f"{held}.t")[1:2][0], held),
        "Plug element [0]":           (Plug(f"{pma}.input1D")[0], pma),
        "Plug element [0:1][0]":      (Plug(f"{pma}.input1D")[0:1][0], pma),
        "Plug(MPlug) child":          (Plug(_mplug(f"{held}.t")).child(2), held),
        "Plug element child":         (Plug(f"{pma}.input3D")[1].input3Dx, pma),
        "typed child by name":        (Attribute(f"{held}.t").translateY, held),
        "typed child(1)":             (Attribute(f"{held}.t").child(1), held),
        "typed element [0]":          (Attribute(f"{pma}.input1D")[0], pma),
        "typed element physical 0":   (Attribute(f"{pma}.input1D").element_by_physical_index(0), pma),
        "typed get_parent":           (Attribute(f"{held}.tx").get_parent(), held),
        "spec apply":                 (Node(held) << Float("added"), held),
        "find_alias":                 (PyNode(held).find_alias("hoist"), held),
        "attr of the shape":          (Node(f"{prefix}pc").worldMesh, f"{prefix}pcShape"),
        "get_inputs()[0]":            (Plug(f"{held}.tx").get_inputs()[0], f"{prefix}src"),
        "get_connected_attrs()[0]":   (Attribute(f"{held}.tx").get_connected_attrs(src=True, dst=False)[0], f"{prefix}src"),
        "ComponentPlug(MPlug)":       (ComponentPlug(cv, f"{prefix}npShape", "cv", 2, None), f"{prefix}npShape"),
    }


def _guard_handle(attr):
    """The API 1.0 handle `_ensure_owner_alive` reads for `attr`: its owner's, or
    the one of its node it took (None if it has neither)."""
    owner = vars(attr)["_node"]
    if owner is None:
        return vars(attr)["_handle1"]
    # re-pinned (round 4a M4, C8): the owner is the node object itself
    return owner._objhandle1


# the paths above whose plug has an owner: the parent cast its node when it was
# named (see `_unowned`), and (re-pinned, round 4a M10, spec S5) a spec's plug,
# owned by the node object it was applied to; every other one has none
_OWNED_AFTER_NAMING = {
    "Plug compound [1]", "Plug compound [1:2][0]", "Plug element [0:1][0]", "spec apply",
}


class TestUnownedPlugsTakeANodeHandle(MayaTestCase):
    """Each construction path of a plug with no owner takes the API 1.0 handle
    of its node; a plug with an owner takes none (its owner's is read)."""

    TEST_START_NEW_SCENE = True

    def test_every_path_takes_the_handle_of_its_node(self):
        _build()
        for label, (attr, node) in _unowned("").items():
            with self.subTest(path=label):
                owned = label in _OWNED_AFTER_NAMING
                self.assertIs(vars(attr)["_node"] is not None, owned)
                handle = _guard_handle(attr)
                self.assertIsInstance(handle, om1.MObjectHandle)
                self.assertEqual(handle.hashCode(), _hash_code(node))
                self.assertTrue(handle.isValid())

    def test_the_handle_does_not_change_the_plug(self):
        # the plug is named, read and hashed as before; asked for its node, it
        # casts it as before (and the owner then guards it)
        _build()
        plug = Plug("held.ty")
        self.assertEqual(str(plug), "held.translateY")
        self.assertEqual(plug.get(), 3.0)
        self.assertEqual(hash(plug), hash(Node("held").ty))
        # re-pinned (round 4a M4, C8): the cast is the typed node, typed repr
        self.assertEqual(repr(plug.node), 'Transform("held")')
        component = Plug("pcShape.vtx[1]")
        self.assertEqual(str.__str__(component), "pcShape.vtx[1]")
        self.assertEqual(str(component), "pcShape.controlPoints[1]")

    def test_owned_paths_look_up_no_handle(self):
        # the lookup (a name parse, about 7 us) is only for a plug with no owner:
        # the plugs of a node object, their children and elements, the attrs a
        # node finds and a spec applied to a node never make it
        _build()
        node  = Node("held")
        typed = PyNode("held")
        cmds.sphere(name="sphere", constructionHistory=False)
        surface = Node("sphereShape")
        with mock.patch.object(_base, "_node_handle", wraps=_base._node_handle) as spy:
            plugs = [
                node.tx, node.t, node.t[0], node.t[0:2], node.t.translateX,
                node.t.child(1), node.worldMatrix[0], node.worldMatrix[0:1],
                typed.find_attr("tx"), typed.find_attr("t").child(0),
                typed.find_attr("t").translateY, typed.find_attr("worldMatrix")[0],
                typed.find_attr("tx").get_parent(), Plug(node.tx),
                node << Float("weight"), surface.cv, surface.cv[1, 2],
                surface.cv[1], typed.find_alias("hoist"),
            ]
            str(List(plugs))
        self.assertEqual(spy.call_count, 0)
        self.assertIs(vars(node.tx)["_node"], node)
        self.assertIsNone(vars(node.tx)["_handle1"])

    def test_the_state_of_every_constructor_is_the_same(self):
        _build()
        mplug = _mplug("held.ty")
        reference = vars(Attribute(mplug))
        for label, attr in (
            ("_new_attr", _base._new_attr(Attribute, mplug)),
            ("Plug(MPlug)", Plug(mplug)),
            ("_named_plug", plug_module._named_plug("held.ty", Node("held"))),
            ("_named_plug without node", plug_module._named_plug("held.ty")),
        ):
            with self.subTest(constructor=label):
                self.assertEqual(tuple(vars(attr)), tuple(reference))
                self.assertEqual(attr.plug, mplug)

    def test_a_spec_hands_the_handle_of_its_node(self):
        # re-pinned (round 4a M10, spec S5): a spec's plug is owned by the node
        # object it was applied to (for a Plug target, the one the plug holds),
        # so it is checked through that node's handle and takes none of its own
        # (it took the node's API 1.0 handle as `_handle1`)
        _build()
        node = Node("held")
        for target in (node, node.tx):
            with self.subTest(target=type(target).__name__):
                plug = target << Float("fresh", overwrite=True)
                self.assertIsInstance(plug, Plug)
                self.assertEqual(str(plug), "held.fresh")
                self.assertIs(vars(plug)["_node"], node)
                self.assertIsNone(vars(plug)["_handle1"])
                self.assertEqual(_guard_handle(plug).hashCode(), _hash_code("held"))
        kept = node << Float("kept")
        self.assertIs(vars(kept)["_node"], node)
        self.assertIsNone(vars(kept)["_handle1"])
        typed = PyNode("held")
        self.assertIs(vars(typed.find_alias("hoist"))["_handle1"], typed._objhandle1)

    def test_a_name_that_resolves_to_no_node_takes_no_handle(self):
        # nothing to check: as before, the plug is not checked
        self.assertIsNone(_base._node_handle("nothing.tx"))
        self.assertIsNone(_base._mplug_handle(OpenMaya.MPlug()))


class TestUnownedPlugsHeldAcrossAFreeRaise(MayaTestCase):
    """A plug with no owner held across a new scene, a file open that reuses its
    names, or a reference unload raises "... already deleted!" (it crashed Maya,
    named another node, or read a wrong value), hashes, and is not a probe's
    attribute."""

    TEST_START_NEW_SCENE = True

    def _assert_freed(self, held):
        ops = {
            "str":        str,
            "repr":       repr,
            "full_name":  lambda p: p.full_name,
            "get":        lambda p: p.get(),
            "node":       lambda p: p.node,
            "Node(plug)": Node,
            "Plug(plug)": Plug,
            "==":         lambda p: p == p,
            "data_type":  lambda p: p.data_type,
            "is_locked":  lambda p: p.is_locked,
            "set":        lambda p: p.set(1),
        }
        dsl_ops = {
            "plus 1":     lambda p: p + 1,
            "<< 1":       lambda p: p << 1,
            ">> None":    lambda p: p >> None,
            "get_inputs": lambda p: p.get_inputs(),
            "child":      lambda p: p.child(0),
        }
        # a typed attr is lifted into the DSL (a Plug is returned as is)
        typed_ops = {"lift": lift, "List": lambda p: List([p])}
        for label, (attr, _node) in held.items():
            dsl = isinstance(attr, Plug)
            checked = {**ops, **(dsl_ops if dsl else typed_ops)}
            if label in _OWNED_AFTER_NAMING:
                # a plug with an owner returns the node object it holds, as in
                # round 3 (it reads nothing of the node)
                self.assertIsNotNone(checked.pop("node")(attr))
                self.assertIsInstance(checked.pop("Node(plug)")(attr), Node)
            for name, op in checked.items():
                with self.subTest(path=label, op=name):
                    with self.assertRaisesRegex(RuntimeError, _FREED):
                        op(attr)
            with self.subTest(path=label, op="hash"):
                self.assertIsInstance(hash(attr), int)
            if dsl:
                # a Python probe finds nothing (a typed attr raises, as an
                # owned one does)
                with self.subTest(path=label, op="hasattr"):
                    self.assertFalse(hasattr(attr, "__array__"))

    def test_across_a_new_scene(self):
        _build()
        held = _unowned("")
        cmds.file(new=True, force=True)
        for _ in range(200):
            cmds.createNode("multiplyDivide")
        self._assert_freed(held)

    def test_across_a_file_open_of_the_same_names(self):
        folder = tempfile.mkdtemp(prefix="rig_r3b_open_")
        path   = os.path.join(folder, "held.ma").replace(os.sep, "/")
        try:
            _build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            held = _unowned("")
            cmds.file(path, open=True, force=True)
            self.assertEqual(cmds.getAttr("held.ty"), 3.0)  # a new node took the name
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_across_a_reference_unload(self):
        folder = tempfile.mkdtemp(prefix="rig_r3b_ref_")
        path   = os.path.join(folder, "held_ref.ma").replace(os.sep, "/")
        try:
            _build()
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            held = _unowned("ref:")
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
            self._assert_freed(held)
        finally:
            cmds.file(new=True, force=True)
            shutil.rmtree(folder, ignore_errors=True)

    def test_the_freed_error_names_the_node_it_was_built_with(self):
        _build()
        plug, typed, derived = Plug("held.ty"), Attribute("held.t"), Plug("held.t").translateY
        cmds.file(new=True, force=True)
        freed = "(freed by a new scene, a file open or a reference unload) already deleted!"
        for label, attr in (("Plug", plug), ("typed", typed), ("child", derived)):
            with self.subTest(plug=label):
                with self.assertRaises(RuntimeError) as ctx:
                    str(attr)
                self.assertEqual(str(ctx.exception), f"held node {freed}")

    def test_a_plug_asked_for_its_node_raises_through_its_owner(self):
        # once cast, the owner guards the plug, with the round-3 message
        _build()
        plug = Plug("held.ty")
        self.assertEqual(plug.get(), 3.0)
        cmds.file(new=True, force=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Transform node \(freed by a new scene, a file open or a reference "
            r"unload\) already deleted!$",
        ):
            plug.get()

    def test_a_freed_hash_is_the_owned_plugs(self):
        # a plug of a freed node that was never hashed hashes by its node's
        # handle and its str buffer, owner or not
        _build()
        unowned = Plug("held.translateY")
        owned   = Node("held").ty
        self.assertEqual(str.__str__(unowned), str.__str__(owned))
        cmds.file(new=True, force=True)
        self.assertEqual(hash(unowned), hash(owned))


class TestUnownedPlugsOfADeletedNodeRaise(MayaTestCase):
    """A plug with no owner whose node was deleted to the undo queue raises the
    deleted node's "already deleted!" (it read the new node that took the name),
    keeps its hash, and reads its node again once an undo brings it back."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)

    def test_a_new_node_of_the_name_is_not_read(self):
        for node_type, attr in (("multiplyDivide", "input1X"), ("transform", "ty")):
            with self.subTest(node_type=node_type):
                name = cmds.createNode(node_type, name=f"gone_{node_type}")
                cmds.setAttr(f"{name}.{attr}", 3)
                plugs = {
                    "Plug(str)":        Plug(f"{name}.{attr}"),
                    "Plug(MPlug)":      Plug(_mplug(f"{name}.{attr}")),
                    "Attribute(str)":   Attribute(f"{name}.{attr}"),
                    "PyNode(MPlug)":    PyNode(_mplug(f"{name}.{attr}")),
                }
                hashes = {label: hash(plug) for label, plug in plugs.items()}
                for plug in plugs.values():
                    vars(plug).pop("_plug_hash", None)
                    vars(plug)["_node"] = None  # hashing cast the node: drop it
                cmds.delete(name)
                cmds.createNode(node_type, name=name)
                cmds.setAttr(f"{name}.{attr}", 7)
                for label, plug in plugs.items():
                    with self.subTest(plug=label):
                        for op in (str, lambda p: p.get(), lambda p: p.node):
                            with self.assertRaises(RuntimeError) as ctx:
                                op(plug)
                            self.assertEqual(str(ctx.exception), f"{name} already deleted!")
                        self.assertEqual(hash(plug), hashes[label])
                cmds.delete(name)
                cmds.undo()
                cmds.undo()
                cmds.undo()
                cmds.undo()  # the first node is back
                for label, plug in plugs.items():
                    with self.subTest(plug=label, after="undo"):
                        self.assertEqual(plug.get(), 3.0)
                        self.assertEqual(hash(plug), hashes[label])
