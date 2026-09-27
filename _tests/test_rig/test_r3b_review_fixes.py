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

import numpy as np
from maya import cmds
from maya.api import OpenMaya
import rig
from rig import Node, Plug, PlugList
from rig.nodetypes import PyNode
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
