"""A plug a connection query gave, read after its connection is gone (2026-11-12).

Maya keeps a *networked* MPlug inside the node for a connected plug, and its connection
queries hand that one out: ``connectedTo``, ``source``, ``destinations``, and
``elementByLogicalIndex`` of a connected element. rig kept it. Maya frees it when the
connection goes (a disconnect, a delete of the attribute, an undo), and the next read of the
held plug was an access violation in ``MPlug::node``, which no ``except`` sees: 40 of 56
cases at 2149ac8, one mayapy each, through ``get_inputs`` / ``get_outputs``,
``get_connected_attrs``, ``iter_connected_attrs`` and ``node.attr[i]`` (the other 16, an
element across an undo, were saved by the Undo callback of ``test_fs_fix_held_elements``).
Inside ``maya.exe`` it took the whole suite down, in
``test_r3b_review_fixes.TestAPlugOfAFreedDynamicAttrRaises``.

rig now keeps the non-networked twin of such an MPlug (``rig.nodetypes._base._unnetworked``):
it names the same plug and is not Maya's to free.

* `TestUnnetworked`: the Maya fact, and the twin.
* `TestHeldAcrossTheEndOfItsConnection`: a plug of each kind, got each way, read after a
  disconnect, a delete of its attribute, and the undo of each; it reads as a plug made by
  name reads.
* `TestTheOtherStores`: the other places an MPlug is kept (``_plug_at``, the element rebinds
  after a disconnect and after an undo, a mesh's set links).

Every test asserts that the MPlug it holds is not networked BEFORE the connection goes: on
the old tree that assert fails, and the read that would take Maya down is never made.
"""

from maya import cmds
from maya.api import OpenMaya

from rig import Node, Plug
from rig._internal import plug as plug_module
from rig.nodetypes import Mesh, _base
from rig.nodetypes import mesh as mesh_module
from rig.nodetypes._base import Attribute
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _mplug(name):
    """The MPlug `name` finds: not a networked one."""
    sel = OpenMaya.MSelectionList()
    sel.add(name)
    return sel.getPlug(0)


def _build():
    """``src`` and ``held``, two transforms with the dynamic attributes the cases wire: on
    ``src`` a double (out), a double multi (srcm) and a compound multi (srcc: srccA,
    srccB); on ``held`` a double multi (dynm) and a double3 (dynv)."""
    cmds.createNode("transform", name="held")
    cmds.createNode("transform", name="src")
    cmds.addAttr("src", longName="out", attributeType="double")
    cmds.addAttr("src", longName="srcm", attributeType="double", multi=True)
    cmds.addAttr("src", longName="srcc", attributeType="compound", numberOfChildren=2, multi=True)
    cmds.addAttr("src",  longName="srccA", attributeType="double", parent="srcc")
    cmds.addAttr("src",  longName="srccB", attributeType="double", parent="srcc")
    cmds.addAttr("held", longName="dynm",  attributeType="double", multi=True)
    cmds.addAttr("held", longName="dynv", attributeType="double3")
    for axis in "XYZ":
        cmds.addAttr("held", longName="dynv" + axis, attributeType="double", parent="dynv")
    cmds.setAttr("src.out",           2.5)
    cmds.setAttr("src.srcm[2]",       3.5)
    cmds.setAttr("src.srcc[1].srccB", 4.5)


def _fresh(wire):
    """A new scene with `_build`'s nodes and the connection `wire`."""
    cmds.file(new=True, force=True)
    _build()
    cmds.connectAttr(*wire)


# case: (the connection, the attribute a delete removes, the far end the plug is asked of,
#        whether the held plug is the source)
_CASES = {
    "a source":                   (("src.out", "held.tx"), "src.out", "held.tx", True),
    "a source element":           (("src.srcm[2]", "held.tx"), "src.srcm", "held.tx", True),
    "a child of a source element": (("src.srcc[1].srccB", "held.tx"), "src.srcc", "held.tx", True),
    "a destination element":      (("src.tx", "held.dynm[3]"), "held.dynm", "src.tx", False),
    "a destination child":        (("src.tx", "held.dynvY"), "held.dynv", "src.tx", False),
}

# the plug of each case, by attribute access from its node (an element, or below one)
_BY_INDEX = {
    "a source element":            lambda: Node("src").srcm[2],
    "a child of a source element": lambda: Node("src").srcc[1].srccB,
    "a destination element":       lambda: Node("held").dynm[3],
}


def _queried(far, source):
    """The MPlug Maya's connection query gives for the other end of `far`."""
    return _mplug(far).connectedTo(source, not source)[0]


def _far(far):
    node, name = far.split(".")
    return getattr(Node(node), name)


# how the held plug is got: (case, far, source) -> plug, None when the way has no such plug
_VIAS = {
    "get_inputs / get_outputs": lambda case, far, source: (
        _far(far).get_inputs() if source else _far(far).get_outputs()
    )[0],
    "get_connected_attrs":      lambda case, far, source: _far(far).get_connected_attrs(src=source, dst=not source)[0],
    "iter_connected_attrs":     lambda case, far, source: next(
        _far(far).iter_connected_attrs(src=source, dst=not source)
    ),
    "Plug(MPlug)":              lambda case, far, source: Plug(_queried(far, source)),
    "Attribute(MPlug)":         lambda case, far, source: Attribute(_queried(far, source)),
    "node.attr[i]":             lambda case, far, source: _BY_INDEX[case]() if case in _BY_INDEX else None,
}

_OPS = {
    "name":         _base._name_of,
    "get_name":     lambda p: p.get_name(),
    "get":          lambda p: p.get(),
    "get_node":     lambda p: str(p.get_node()),
    "is_connected": lambda p: p.is_connected(),
    "is_locked":    lambda p: p.is_locked(),
    "mobject":      lambda p: p.get_mobject().isNull(),
    "data_type":    lambda p: p.get_data_type(),
}


def _outcome(op, plug):
    """What `op` does with `plug`: its value, or the exception it raises."""
    try:
        return op(plug)
    except Exception as err:  # noqa: BLE001 -- compared with the outcome of a plug made by name
        return type(err).__name__, str(err)


class TestUnnetworked(MayaTestCase):
    """`_base._unnetworked`: the twin of an MPlug names the same plug and is not networked."""

    TEST_START_NEW_SCENE = True

    def test_a_connection_query_gives_a_networked_mplug(self):
        """The Maya fact (Maya 2025): the MPlug a name finds is not networked, the one a
        connection query gives is."""
        for case, (wire, _attr, far, source) in _CASES.items():
            with self.subTest(case):
                _fresh(wire)
                self.assertFalse(_mplug(wire[0]).isNetworked)
                self.assertFalse(_mplug(wire[1]).isNetworked)
                self.assertTrue(_queried(far, source).isNetworked)

    def test_the_twin_names_the_same_plug(self):
        for case, (wire, _attr, far, source) in _CASES.items():
            with self.subTest(case):
                _fresh(wire)
                networked = _queried(far, source)
                twin      = _base._unnetworked(networked)
                name      = networked.name()
                self.assertFalse(twin.isNetworked)
                self.assertEqual(twin, networked)
                self.assertEqual(twin.name(), name)
                self.assertEqual(twin.node(), networked.node())
                self.assertEqual(twin.attribute(), networked.attribute())
                self.assertEqual((twin.isElement, twin.isChild), (networked.isElement, networked.isChild))
                self.assertEqual(twin.isConnected, True)
                del networked
                cmds.disconnectAttr(*wire)
                self.assertEqual(twin.name(), name)
                self.assertEqual(twin.isConnected, False)

    def test_the_twin_of_a_plug_that_is_not_networked(self):
        _build()
        for name in ("src.out", "src.srcm[2]", "src.srcc[1].srccB", "held.dynm[3]", "held.dynvY", "held.tx"):
            with self.subTest(name):
                found = _mplug(name)
                twin  = _base._unnetworked(found)
                self.assertFalse(twin.isNetworked)
                self.assertEqual(twin, found)
                self.assertEqual(twin.name(), found.name())


class TestHeldAcrossTheEndOfItsConnection(UndoWalk, MayaTestCase):
    """A plug rig handed out while it was connected, read after the connection is gone, and
    after the undo that brings it back: it reads as a plug made by name does."""

    TEST_START_NEW_SCENE = True

    def _held_across(self, action, undo):
        for case, (wire, attr, far, source) in _CASES.items():
            name = wire[0] if source else wire[1]
            for via, get in _VIAS.items():
                with self.subTest(case=case, via=via):
                    _fresh(wire)
                    plug = get(case, far, source)
                    if plug is None:
                        continue
                    by_name = Plug(name)
                    # first: on the old tree this fails, and no freed MPlug is read below
                    self.assertFalse(vars(plug)["_mplug"].isNetworked)
                    self.assertFalse(vars(by_name)["_mplug"].isNetworked)
                    self.assertEqual(_base._name_of(plug), name)
                    self.assertTrue(plug.is_connected())
                    if action == "disconnect":
                        cmds.disconnectAttr(*wire)
                    else:
                        cmds.deleteAttr(attr)
                    self.assertFalse(cmds.objExists(name) and cmds.isConnected(*wire))
                    if undo:
                        cmds.undo()
                        self.assertTrue(cmds.isConnected(*wire))
                    self.assertEqual(_base._name_of(plug), name)
                    self.assertEqual(plug.is_connected(), undo)
                    self.assertFalse(vars(plug)["_mplug"].isNetworked)
                    for label, op in _OPS.items():
                        self.assertEqual(_outcome(op, plug), _outcome(op, by_name), label)

    def test_across_a_disconnect(self):
        self._held_across("disconnect", undo=False)

    def test_across_a_delete_of_its_attribute(self):
        # the delete stays in the undo queue, so the attribute lives on: what the suite's run
        # in maya.exe left of test_r3b_review_fixes' "full undo queue" case
        self._held_across("delete", undo=False)

    def test_across_a_disconnect_and_its_undo(self):
        self._held_across("disconnect", undo=True)

    def test_across_a_delete_and_its_undo(self):
        self._held_across("delete", undo=True)

    def test_what_a_held_plug_reads(self):
        """The values, spelled out for one case of each action."""
        _build()
        cmds.connectAttr("src.out", "held.tx")
        plug = Node("held").tx.get_inputs()[0]
        self.assertFalse(vars(plug)["_mplug"].isNetworked)
        cmds.disconnectAttr("src.out", "held.tx")
        self.assertEqual(
            [_outcome(op, plug) for op in _OPS.values()],
            ["src.out", "out", 2.5, "src", False, False, False, "double"],
        )
        cmds.undo()
        self.assertTrue(plug.is_connected())
        cmds.deleteAttr("src.out")
        self.assertEqual(_base._name_of(plug), "src.out")
        self.assertEqual(str(plug.get_node()), "src")
        self.assertFalse(plug.is_connected())
        with self.assertRaises(ValueError):
            plug.get()
        cmds.undo()
        self.assertEqual(plug.get(), 2.5)
        self.assertTrue(plug.is_connected())


class TestTheOtherStores(UndoWalk, MayaTestCase):
    """The other places rig keeps an MPlug: none keeps a networked one."""

    TEST_START_NEW_SCENE = True

    def test_plug_at(self):
        for case, (wire, _attr, far, source) in _CASES.items():
            with self.subTest(case):
                _fresh(wire)
                networked = _queried(far, source)
                again     = plug_module._plug_at(plug_module._plug_path(networked))
                same      = again == networked, again.name() == networked.name()
                del networked
                self.assertFalse(again.isNetworked)
                self.assertEqual(same, (True, True))

    def test_an_element_found_again_after_a_disconnect(self):
        # the element drives something else, so it is still connected when rig finds its
        # MPlug again (`_rebind_element`)
        _build()
        cmds.connectAttr("src.tx", "held.dynm[3]")
        cmds.connectAttr("held.dynm[3]", "src.ty")
        element = Attribute("held.dynm[3]")
        Attribute("src.tx").disconnect(element)
        self.assertFalse(cmds.isConnected("src.tx", "held.dynm[3]"))
        self.assertFalse(vars(element)["_mplug"].isNetworked)
        cmds.disconnectAttr("held.dynm[3]", "src.ty")
        self.assertEqual(_base._name_of(element), "held.dynm[3]")
        self.assertFalse(element.is_connected())

    def test_an_element_found_again_after_an_undo(self):
        # held while free, connected since: the Undo callback finds its MPlug again
        # (`_rebind_held_elements`) while it is connected
        _build()
        element = Node("held").dynm[3]
        cmds.connectAttr("src.tx", "held.dynm[3]")
        cmds.setAttr("src.ty", 1.0)
        cmds.undo()
        self.assertTrue(cmds.isConnected("src.tx", "held.dynm[3]"))
        self.assertFalse(vars(element)["_mplug"].isNetworked)
        cmds.disconnectAttr("src.tx", "held.dynm[3]")
        self.assertEqual(_base._name_of(element), "held.dynm[3]")
        self.assertFalse(element.is_connected())

    def test_the_links_of_a_uv_set(self):
        # kept across the delete of the set and its undo (`Mesh.delete_uv_set`)
        cube = Mesh(cmds.polyCube(name="c", constructionHistory=False)[0])
        cube.add_uv_set("a")
        texture = cmds.shadingNode("file", asTexture=True, name="tex")
        cmds.uvLink(uvSet=f"{cube.get_long_name()}.uvSet[1].uvSetName", texture=texture)
        _index, children = mesh_module._links(cube.get_fn_set(), "uvSet", "a")
        found = [each for _child, destinations in children for each in destinations]
        self.assertEqual([each.name() for each in found], ["uvChooser1.uvSets[0]"])
        self.assertEqual([each.isNetworked for each in found], [False])
        cube.delete_uv_set("a")
        self.assertEqual([each.name() for each in found], ["uvChooser1.uvSets[0]"])
        cmds.undo()
        self.assertEqual(
            cmds.listConnections(f"{cube.get_long_name()}.uvSet[1].uvSetName", plugs=True),
            ["uvChooser1.uvSets[0]"],
        )
