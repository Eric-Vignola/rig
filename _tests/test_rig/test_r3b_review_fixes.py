"""Regression tests for the round-3b review: the reads that skipped the owner
check, and the DSL functions that did not check their plug operands.

``Attribute.name``, ``mobject``, ``fn_set`` and ``is_typed`` read the MPlug
without the round-3 owner check, so after a new scene, a file open or a
reference unload a plug of a dynamic attr (whose attribute is freed with its
node) returned another node's attribute name or crashed Maya; ``attribute_type``,
``enums``, ``default_value`` and the category methods read ``name`` first. The
public ``@operands`` functions (``rig.normalize``, ``rig.dist``, ``rig.inverse``,
``rig.to_euler``, ``rig.vector.length``...) classified a freed plug operand
before anything checked it, and crashed Maya. Both now raise the round-3
``"... already deleted!"``, owner or not.
"""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from maya import cmds, OpenMaya as om1
from maya.api import OpenMaya
import rig
from rig import Node, Plug, PlugList
from rig.bridges import commands as rc
from rig.nodetypes import PyNode, _base
from rig.nodetypes._base import Attribute
from rig._internal.operands import operands
from rig._tests._base import MayaTestCase


_FREED = (
    r"^[\w:|]+ node \(freed by a new scene, a file open or a reference unload\) "
    r"already deleted!$"
)


def _mplug(name):
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _hash_code(name):
    """The API 1.0 MObjectHandle hashCode of the node `name`."""
    sel = om1.MSelectionList()
    sel.add(name)
    mobject = om1.MObject()
    sel.getDependNode(0, mobject)
    return om1.MObjectHandle(mobject).hashCode()


def _build():
    """``held``, a transform with dynamic attrs of each kind: a double (dynf), an
    enum (dyne), a double multi with elements 0 and 3 (dynm), a double array
    (dyna), a double3 (dynv), a matrix (dynmat) and a double4 (dynq)."""
    cmds.createNode("transform", name="held")
    cmds.addAttr("held", longName="dynf", attributeType="double", defaultValue=0.5)
    cmds.addAttr("held", longName="dyne", attributeType="enum", enumName="aa:bb:cc")
    cmds.addAttr("held", longName="dynm", attributeType="double", multi=True)
    cmds.addAttr("held", longName="dyna", dataType="doubleArray")
    cmds.addAttr("held", longName="dynv", attributeType="double3")
    for axis in "XYZ":
        cmds.addAttr("held", longName=f"dynv{axis}", attributeType="double", parent="dynv")
    cmds.addAttr("held", longName="dynmat", attributeType="matrix")
    cmds.addAttr("held", longName="dynq", attributeType="double4")
    for axis in "XYZW":
        cmds.addAttr("held", longName=f"dynq{axis}", attributeType="double", parent="dynq")
    cmds.setAttr("held.dynm[0]", 1.0)
    cmds.setAttr("held.dynm[3]", 4.0)
    cmds.setAttr("held.dyna", [1.0, 0.0, 2.0], type="doubleArray")


class _HeldAcrossAFree(MayaTestCase):
    """Builds ``held``, hands its plugs to ``make(prefix)`` and frees them: a new
    scene, a file open that reuses the names, or a reference unload. Then new
    nodes with dynamic attrs take the freed memory."""

    TEST_START_NEW_SCENE = True

    def _held_across(self, free, make):
        folder = tempfile.mkdtemp(prefix="rig_r3b_fix_")
        self.addCleanup(shutil.rmtree, folder, True)
        self.addCleanup(cmds.file, new=True, force=True)
        path = os.path.join(folder, "held.ma").replace(os.sep, "/")
        _build()
        prefix = ""
        if free != "new":
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
        if free == "unload":
            cmds.file(new=True, force=True)
            cmds.file(path, reference=True, namespace="ref")
            prefix = "ref:"
        held = make(prefix)
        if free == "new":
            cmds.file(new=True, force=True)
        elif free == "open":
            cmds.file(path, open=True, force=True)
            self.assertTrue(cmds.objExists("held.dynf"))  # a new node took the name
        else:
            cmds.file(unloadReference=cmds.referenceQuery(path, referenceNode=True))
        for i in range(200):
            node = cmds.createNode("transform", name=f"fill{i}")
            cmds.addAttr(node, longName="zz", attributeType="double")
            cmds.addAttr(node, longName="zzm", attributeType="double", multi=True)
        return held


def _attr_plugs(prefix):
    """Plugs of held's dynamic attrs, with an owner and without, keyed by label.
    ``owned, cached`` has read its attribute and fn set before the free."""
    node   = Node(f"{prefix}held")
    cached = node.dynf
    cached.mobject
    cached.fn_set
    return {
        "owned":          node.dyne,
        "owned, cached":  cached,
        "typed owned":    PyNode(f"{prefix}held").find_attr("dyna"),
        "Plug(str)":      Plug(f"{prefix}held.dyne"),
        "Attribute(str)": Attribute(f"{prefix}held.dynf"),
        "Plug(MPlug)":    Plug(_mplug(f"{prefix}held.dynf")),
        "typed element":  Attribute(f"{prefix}held.dynm")[3],
        "Plug child":     Plug(f"{prefix}held.dynv").dynvX,
        "typed child":    Attribute(f"{prefix}held.dynv").child(0),
        "typed array":    Attribute(f"{prefix}held.dyna"),
    }


_ATTR_OPS = {
    "name":                lambda a: a.name,
    "alias":               lambda a: a.alias,
    "mobject":             lambda a: a.mobject,
    "fn_set":              lambda a: a.fn_set,
    "is_typed":            lambda a: a.is_typed,
    "attribute_type":      lambda a: a.attribute_type,
    "enums":               lambda a: a.enums,
    "default_value":       lambda a: a.default_value,
    "get_categories":      lambda a: a.get_categories(),
    "has_category":        lambda a: a.has_category("rig"),
    "filter_array_values": lambda a: a.filter_array_values(0.0),
}


class TestAttributeReadsHeldAcrossAFreeRaise(_HeldAcrossAFree):
    """``name``, ``alias``, ``mobject``, ``fn_set``, ``is_typed`` and the reads
    built on them raise "already deleted!" for a plug of a freed node, owner or
    not (they returned another node's attribute name, or crashed Maya)."""

    def _assert_raise(self, held):
        for label, attr in held.items():
            for op_name, op in _ATTR_OPS.items():
                with self.subTest(plug=label, op=op_name):
                    with self.assertRaisesRegex(RuntimeError, _FREED):
                        op(attr)

    def test_across_a_new_scene(self):
        self._assert_raise(self._held_across("new", _attr_plugs))

    def test_across_a_file_open_of_the_same_names(self):
        self._assert_raise(self._held_across("open", _attr_plugs))

    def test_across_a_reference_unload(self):
        self._assert_raise(self._held_across("unload", _attr_plugs))

    def test_live_reads_are_unchanged(self):
        _build()
        cmds.undoInfo(state=True, infinity=True)
        for label, enum in (
            ("owned", Node("held").dyne),
            ("Plug(str)", Plug("held.dyne")),
            ("Plug(MPlug)", Plug(_mplug("held.dyne"))),
        ):
            with self.subTest(plug=label):
                self.assertEqual(enum.name, "dyne")
                self.assertEqual(enum.alias, "dyne")
                self.assertEqual(enum.mobject.apiType(), OpenMaya.MFn.kEnumAttribute)
                self.assertIsInstance(enum.fn_set, OpenMaya.MFnEnumAttribute)
                self.assertFalse(enum.is_typed)
                self.assertEqual(enum.attribute_type, "enum")
                self.assertEqual(enum.enums, ["aa", "bb", "cc"])
        array = Attribute("held.dyna")
        self.assertTrue(array.is_typed)
        self.assertEqual(array.filter_array_values(0.0), ([0, 2], [1.0, 2.0]))
        self.assertEqual(Attribute("held.dynm")[3].name, "dynm[3]")
        self.assertEqual(Plug("held.dynv").dynvX.name, "dynvX")
        self.assertEqual(Node("held").dynf.default_value, 0.5)
        # a node deleted to the undo queue is still alive: its attrs read on
        dynf = Node("held").dynf
        cmds.delete("held")
        self.assertEqual(dynf.name, "dynf")
        self.assertEqual(dynf.mobject.apiType(), OpenMaya.MFn.kNumericAttribute)


def _operand_plugs(prefix):
    """(double, double3, matrix, double4) plugs of held, per way of getting one."""
    names = ("dynf", "dynv", "dynmat", "dynq")
    node  = Node(f"{prefix}held")
    return {
        "owned":          tuple(getattr(node, name) for name in names),
        "Plug(str)":      tuple(Plug(f"{prefix}held.{name}") for name in names),
        "Attribute(str)": tuple(Attribute(f"{prefix}held.{name}") for name in names),
        "Plug(MPlug)":    tuple(Plug(_mplug(f"{prefix}held.{name}")) for name in names),
    }


def _functions(f, v, m, q):
    return {
        "vector.length":      lambda: rig.vector.length(v),
        "normalize":          lambda: rig.normalize(v),
        "dist":               lambda: rig.dist(v, (0.0, 0.0, 0.0)),
        "inverse":            lambda: rig.inverse(m),
        "to_euler":           lambda: rig.to_euler(q),
        "matrix.decompose":   lambda: rig.matrix.decompose(m),
        "vector.dot":         lambda: rig.vector.dot(v, (1.0, 0.0, 0.0)),
        "functions.clamp":    lambda: rig.functions.clamp(f, 0.0, 1.0),
        "lerp":               lambda: rig.lerp(f, 1.0, 0.5),
        "condition":          lambda: rig.condition(f, 1.0, 2.0),
        "functions.choice":   lambda: rig.functions.choice([f, 2.0], 0),
        "functions.sum list": lambda: rig.functions.sum([1.0, [f]]),
        "keyword operand":    lambda: rig.lerp(1.0, 2.0, weight=f),
        "config operand":     lambda: rig.euler.to_matrix(
            Node("live").rotate, rotate_order=f
        ),
    }


class TestDSLFunctionsCheckTheirFreedOperands(_HeldAcrossAFree):
    """A public DSL function raises a freed plug operand's "already deleted!"
    before it classifies it or builds anything (``rig.normalize``, ``rig.dist``,
    ``rig.inverse``, ``rig.to_euler`` and ``rig.vector.length`` crashed Maya)."""

    def _assert_raise(self, held):
        cmds.createNode("transform", name="live")
        for label, plugs in held.items():
            for fn_name, call in _functions(*plugs).items():
                with self.subTest(plug=label, function=fn_name):
                    before = len(cmds.ls())
                    with self.assertRaisesRegex(RuntimeError, _FREED):
                        call()
                    self.assertEqual(len(cmds.ls()), before)

    def test_across_a_new_scene(self):
        self._assert_raise(self._held_across("new", _operand_plugs))

    def test_across_a_file_open_of_the_same_names(self):
        self._assert_raise(self._held_across("open", _operand_plugs))

    def test_across_a_reference_unload(self):
        self._assert_raise(self._held_across("unload", _operand_plugs))

    def test_every_operand_container_is_walked(self):
        # the check walks lists, tuples, PlugLists and object arrays, nested, as
        # the plain str check does; the function never runs
        ran = []

        @operands(config=("mode",))
        def probe(a, b=None, mode=None):
            ran.append(True)

        held = self._held_across("new", lambda prefix: Plug(f"{prefix}held.dynf"))
        for label, args, kwargs in (
            ("positional", (held,), {}),
            ("list", ([1.0, held],), {}),
            ("tuple", ((held, 1.0),), {}),
            ("nested", ([[1.0, (held,)]],), {}),
            ("PlugList", (PlugList([held]),), {}),
            ("object array", (np.array([1.0, held], dtype=object),), {}),
            ("keyword", (1.0,), {"b": [held]}),
            ("config", (1.0,), {"mode": held}),
            ("config positional", (1.0, None, held), {}),
        ):
            with self.subTest(operand=label):
                with self.assertRaisesRegex(RuntimeError, _FREED):
                    probe(*args, **kwargs)
        self.assertEqual(ran, [])
        # a plain str before the freed plug raises its TypeError first, as before
        with self.assertRaises(TypeError):
            probe(["cube.ty", held])


class TestNodeHandlesAreLookedUpOncePerNode(MayaTestCase):
    """A plug with no owner takes the API 1.0 handle of its node from a table
    after the first plug of that node (a name parse each cost about 12 us), and
    the table never hands a freed node's handle to a later node."""

    TEST_START_NEW_SCENE = True

    def _handle(self, attr):
        return vars(attr)["_handle1"]

    def test_the_second_plug_of_a_node_makes_no_lookup(self):
        cmds.createNode("transform", name="held")
        first = Plug("held.tx")
        with mock.patch.object(_base, "_node_handle", wraps=_base._node_handle) as spy:
            plugs = [
                Plug("held.ty"), Attribute("held.tz"), PyNode("held.rx"),
                Plug(_mplug("held.ry")), Attribute(_mplug("held.rz")),
                PyNode(_mplug("held.sx")), PlugList(["held.sy"])[0],
            ]
        self.assertEqual(spy.call_count, 0)
        for plug in plugs:
            self.assertIs(self._handle(plug), self._handle(first))

    def test_connection_results_take_their_nodes_handles(self):
        cmds.createNode("transform", name="src")
        for i in range(5):
            cmds.createNode("transform", name=f"dst{i}")
            cmds.connectAttr("src.tx", f"dst{i}.tx")
        for _ in range(2):  # looked up, then found in the table
            outputs = Node("src").tx.get_outputs()
            self.assertEqual(
                sorted(str(p) for p in outputs), [f"dst{i}.translateX" for i in range(5)]
            )
            for plug in outputs:
                node = str(plug).split(".")[0]
                self.assertEqual(self._handle(plug).hashCode(), _hash_code(node))
                self.assertTrue(self._handle(plug).isValid())

    def test_the_handle_is_the_plugs_node_of_a_shared_name(self):
        for group in ("ga", "gb"):
            cmds.createNode("transform", name=group)
            cmds.createNode("transform", name="dup", parent=group)
        for path in ("|ga|dup", "|gb|dup"):
            for plug in (Plug(f"{path}.tx"), Plug(_mplug(f"{path}.tx"))):
                with self.subTest(path=path, plug=str.__str__(plug)):
                    self.assertEqual(self._handle(plug).hashCode(), _hash_code(path))

    def test_a_freed_nodes_handle_is_never_handed_on(self):
        cmds.createNode("transform", name="held")
        held = Plug("held.tx")
        self.assertTrue(_base._NODE_HANDLES)
        cmds.file(new=True, force=True)
        fresh = []
        for i in range(300):
            name = cmds.createNode("transform", name=f"n{i}")
            fresh.append((name, Plug(f"{name}.tx")))
        for name, plug in fresh:
            handle = self._handle(plug)
            self.assertTrue(handle.isValid())
            self.assertEqual(handle.hashCode(), _hash_code(name))
            self.assertEqual(om1.MFnDependencyNode(handle.objectRef()).name(), name)
        with self.assertRaisesRegex(RuntimeError, _FREED):
            str(held)

    def test_the_table_drops_freed_nodes(self):
        cmds.createNode("transform", name="held")
        Plug("held.tx")
        cmds.file(new=True, force=True)
        _base._prune_node_handles()
        for entries in _base._NODE_HANDLES.values():
            for _mobject, handle in entries:
                self.assertTrue(handle.isAlive())


class TestAPlugOfADeletedNodesMPlugRaises(MayaTestCase):
    """A plug built from the MPlug of a node already deleted to the undo queue can
    take no handle of it (a deleted node is not found by name), so it raises the
    deleted node's "already deleted!" when it is built. It read, and wrote, the
    node that took the name, or whatever reused the memory once the deleted node
    was freed."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)

    def test_it_raises_when_it_is_built(self):
        for retaken in (False, True):
            with self.subTest(retaken=retaken):
                cmds.file(new=True, force=True)
                cmds.createNode("transform", name="held")
                cmds.addAttr("held", longName="dynf", attributeType="double")
                cmds.setAttr("held.dynf", 7.0)
                mplug = Node("held").dynf.plug  # an MPlug the caller keeps
                known = _mplug("held.ty")
                Plug(known)  # the node is in the handle table too
                cmds.delete("held")
                if retaken:
                    cmds.createNode("transform", name="held")
                    cmds.addAttr("held", longName="dynf", attributeType="double")
                    cmds.setAttr("held.dynf", 99.0)
                for label, build in (
                    ("Plug(MPlug)", lambda: Plug(mplug)),
                    ("Attribute(MPlug)", lambda: Attribute(mplug)),
                    ("PyNode(MPlug)", lambda: PyNode(mplug)),
                    ("Plug(MPlug) of a known node", lambda: Plug(known)),
                ):
                    with self.subTest(retaken=retaken, plug=label):
                        with self.assertRaises(RuntimeError) as ctx:
                            build()
                        self.assertEqual(str(ctx.exception), "held already deleted!")
                if retaken:
                    self.assertEqual(cmds.getAttr("held.dynf"), 99.0)
                    for _ in range(3):  # the new node's setAttr, addAttr, create
                        cmds.undo()
                cmds.undo()  # the first node is back: its MPlug builds a plug
                self.assertEqual(Plug(mplug).get(), 7.0)
                self.assertEqual(str(Attribute(mplug)), "held.dynf")

    def test_a_null_mplug_takes_no_handle(self):
        self.assertIsNone(_base._mplug_handle(OpenMaya.MPlug()))


class TestTheCommandsBridgeChecksItsPlugs(_HeldAcrossAFree):
    """``rig.bridges.commands`` hands a plug's str buffer to cmds. Once its node
    was deleted or freed, the buffer names the node that took the name, so
    ``rc.getAttr(plug)`` read and ``rc.setAttr(plug, 7)`` wrote that node while
    ``plug.get()`` raised. The bridge now raises the plug's "already deleted!"."""

    def _plugs(self, prefix):
        resolved = Plug(f"{prefix}held.dynf")
        str(resolved)  # cast its node: it has an owner now
        return {
            "owned":      Node(f"{prefix}held").dynf,
            "resolved":   resolved,
            "unresolved": Plug(f"{prefix}held.dynf"),
            "typed":      Attribute(f"{prefix}held.dynf"),
        }

    def _assert_raise(self, held, pattern):
        exists = cmds.objExists("held.dynf")
        if exists:
            cmds.setAttr("held.dynf", 42.0)
        for label, plug in held.items():
            for op_name, op in (
                ("getAttr", lambda p: rc.getAttr(p)),
                ("setAttr", lambda p: rc.setAttr(p, 7.0)),
                ("list", lambda p: rc.listConnections([p])),
                ("keyword", lambda p: rc.getAttr(p, type=True)),
            ):
                with self.subTest(plug=label, op=op_name):
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        op(plug)
        if exists:
            self.assertEqual(cmds.getAttr("held.dynf"), 42.0)

    def test_across_a_new_scene(self):
        self._assert_raise(self._held_across("new", self._plugs), _FREED)

    def test_across_a_file_open_of_the_same_names(self):
        self._assert_raise(self._held_across("open", self._plugs), _FREED)

    def test_across_a_reference_unload(self):
        self._assert_raise(self._held_across("unload", self._plugs), _FREED)

    def test_across_a_delete_with_the_name_taken(self):
        cmds.undoInfo(state=True, infinity=True)
        _build()
        held = self._plugs("")
        cmds.delete("held")
        cmds.createNode("transform", name="held")
        cmds.addAttr("held", longName="dynf", attributeType="double")
        self._assert_raise(held, r"^held already deleted!$")

    def test_live_plugs_pass_through_as_before(self):
        _build()
        cmds.createNode("transform", name="other")
        rc.setAttr(Plug("held.dynf"), 3.0)
        self.assertEqual(rc.getAttr(Node("held").dynf), 3.0)
        rc.connectAttr(Node("held").dynf, Plug("other.tx"))
        self.assertEqual(cmds.listConnections("other.tx", plugs=True), ["held.dynf"])
        cube = cmds.polyCube(name="pc", constructionHistory=False)[0]
        rc.select([Node(cube).vtx[1], Node("other")])
        self.assertEqual(cmds.ls(selection=True), ["pc.vtx[1]", "other"])


class TestAPlugKeepsItsHashAcrossADelete(MayaTestCase):
    """A per-instance array read without an index (``worldMatrix``,
    ``instObjGroups``) hashed without the index of its instance while its node
    was deleted to the undo queue, and with it once the undo brought the node
    back, so a dict or set key made in between was lost. Owner or not, it now
    hashes the same across the delete and the undo."""

    TEST_START_NEW_SCENE = True

    def test_a_key_made_while_deleted_is_found_after_the_undo(self):
        cmds.undoInfo(state=True, infinity=True)
        cmds.polyCube(name="held", constructionHistory=False)
        makers = {
            "owned worldMatrix":           lambda: Node("held").worldMatrix,
            "owned shape instObjGroups":   lambda: Node("heldShape").instObjGroups,
            "Plug(MPlug) worldMatrix":     lambda: Plug(Node("held").worldMatrix.plug),
            "Plug(MPlug) instObjGroups":   lambda: Plug(_mplug("heldShape.instObjGroups")),
            "typed worldMatrix":           lambda: PyNode("held").find_attr("worldMatrix"),
            "Plug(str) worldMatrix":       lambda: Plug("held.worldMatrix"),
            "owned ty":                    lambda: Node("held").ty,
            "Plug(MPlug) ty":              lambda: Plug(_mplug("held.ty")),
        }
        for label, make in makers.items():
            with self.subTest(plug=label):
                plug = make()
                live = hash(plug)
                cmds.delete("held")
                keys = {plug}
                deleted = hash(plug)
                cmds.undo()
                self.assertEqual(hash(plug), deleted)
                self.assertEqual(hash(plug), live)
                self.assertIn(plug, keys)


class TestTheDeletedMessagesShareOneText(MayaTestCase):
    """A node object, a plug with an owner and a plug with none raise the
    round-3 texts, built by one helper (`_deleted_error`) so they cannot drift:
    ``"<name> already deleted!"`` for a deleted node, and ``"<class> node (freed
    by a new scene, a file open or a reference unload) already deleted!"`` for a
    freed one, a plug with no owner naming its node by the name it was built
    with (the class it would be cast to was never known)."""

    TEST_START_NEW_SCENE = True

    def test_the_texts(self):
        freed = "(freed by a new scene, a file open or a reference unload) already deleted!"
        self.assertEqual(str(_base._deleted_error("held")), "held already deleted!")
        self.assertEqual(
            str(_base._deleted_error("Transform", freed=True)), f"Transform node {freed}"
        )
        cmds.undoInfo(state=True, infinity=True)
        cmds.createNode("transform", name="held")
        node, owned, unowned = PyNode("held"), Node("held").tx, Plug("held.ty")
        cmds.delete("held")
        for label, op in (
            ("node", node.ensure_valid), ("owned", lambda: str(owned)),
            ("unowned", lambda: str(unowned)),
        ):
            with self.subTest(target=label, free="delete"):
                with self.assertRaises(RuntimeError) as ctx:
                    op()
                self.assertEqual(str(ctx.exception), "held already deleted!")
        cmds.undo()
        cmds.file(new=True, force=True)
        for label, op, name in (
            ("node", node.ensure_valid, "Transform"),
            ("owned", lambda: str(owned), "Transform"),
            ("unowned", lambda: str(unowned), "held"),
        ):
            with self.subTest(target=label, free="new scene"):
                with self.assertRaises(RuntimeError) as ctx:
                    op()
                self.assertEqual(str(ctx.exception), f"{name} node {freed}")
