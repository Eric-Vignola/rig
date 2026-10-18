"""Round 5, step A6: small items.

* CR-12: the Choice node class's data type hook reads the selector's value from
  its plug (``MPlug.asInt``) where it called ``selector.get()``, whose
  ``getAttr -type`` probe (a long is never geometry) and ``getAttr`` read the
  same int: ``MPlug.asInt`` evaluates the plug at the current time as getAttr
  does, connected or not. The node is valid when ``data_type`` calls the hook
  (it named the attr first); a direct call on a deleted or freed node raises
  the "already deleted!" ``get()`` raised (the hook checks the node), so its
  plug is never read.
* Lazy child caches: the caches an attr keeps of its child attributes
  (``Attribute.child``, a child by name) are made on first use instead of with
  every attr, two dicts fewer per plug (about 50,000 per rail_spine build).
  They are read through ``__dict__``, so an absent one never reaches
  ``__getattr__``; a name lookup fills both, over earlier ``child(i)`` entries,
  as before.

Tests:

* ``TestTheChoiceHook``: over selector states (unconnected, in and out of
  range; driven by a long, a double with fractions and halves, an angle through
  a unitConversion, time1, an anim curve between and beyond its keys, an
  expression; locked; muted) the hook's answer and ``output.data_type`` equal
  the ``get()`` route's (ee03336's hook, copied here), and the plug's int is
  ``get()``'s; the hook makes no ``getAttr`` call on the selector (ee03336: a
  type and a value query); a direct call on a deleted and on a freed node raises
  what the ``get()`` route raises, and the undo answers again.
* ``TestLazyChildCaches``: no constructor makes the caches; ``child`` makes the
  index cache and a name lookup both, each attr its own; a name lookup fills
  the index cache over an earlier ``child(i)`` entry; a cached child is
  returned again; a non-compound, a missing name and an unhashable index raise
  as before and make no name cache; the copy ``_full_name_buffer`` makes of a
  fresh attr (an instanced owner) holds no cache and is the attr kept; a
  ``copy.copy`` of a plug made before any child lookup makes its own caches.
* ``TestTheChildLookupsOfEe03336``: child lookups (by name, by index, twice,
  mixed; missing and private names, out of range, negative and unhashable
  indices) on DG, DAG, instanced, dynamic, element, array, message and generic
  plugs, through a node's ``Plug``, ``find_attr``'s Attribute and owner-less
  plugs, and on held plugs across a delete, its undo and a new scene, give
  what ee03336 gave (``r5_small_oracle_ee03336.json`` beside this file, written
  by ``runs\\r5\\A6\\gen_oracle.py`` running ``_child_outcomes`` on the snapshot
  of ee03336): the result (type, str buffer, name, owner, handles, MPlug name)
  or the error with its whole chain, and which results are the same object.
"""

import copy
import json
import os
import re
from unittest import mock

from maya import cmds

from rig import Node, Plug
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute, _name_of
from rig.nodetypes.choice import Choice
from rig.nodetypes.dg_node import DGNode
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk

NAMES = "_Attribute__child_name_dict"
IDS   = "_Attribute__child_id_dict"


def _chain(error):
    """[type name, message, has a cause, suppresses its context] of `error` and of
    each exception it was raised from (`__cause__`) or while handling
    (`__context__`), the whole chain Python prints."""
    out  = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        out.append(
            [type(error).__name__, str(error), error.__cause__ is not None, error.__suppress_context__]
        )
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return out


def _outcome(func):
    """["returns", value] or ["raises", the error's chain]."""
    try:
        return ["returns", func()]
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ["raises", _chain(error)]


# --------------------------------------------------------------------------- #
#  CR-12: the Choice hook
# --------------------------------------------------------------------------- #


def _old_hook(self, attr):
    """Choice's hook as ee03336 wrote it: the selector through ``get()``."""
    if attr.get_name() == "output":
        sel_id  = self.find_attr("selector").get()
        in_attr = self.find_attr(f"input[{sel_id}]", quiet=True)
        if in_attr is not None:
            src_attr = _base._connected_attrs(in_attr, src=True, dst=False, first_only=True)
            # (re-pinned, user decision 8A, 2026-10-17: a plug has no truth value;
            # the hook tests `is not None`, as here)
            if src_attr is not None:
                return src_attr.get_data_type()
    elif attr.get_name().startswith("input"):
        src_attr = _base._connected_attrs(attr, src=True, dst=False, first_only=True)
        if src_attr is not None:
            return src_attr.get_data_type()
    return DGNode._attr_data_type_fallback(self, attr)


def _choice_scene():
    """A choice ``ch`` whose inputs 0-2 are a float3, a doubleLinear and a matrix
    (getAttr's type of the output is "typed" for input 1, so data_type runs the
    hook there)."""
    cmds.file(new=True, force=True)
    cmds.createNode("choice", name="ch")
    cmds.createNode("multiplyDivide", name="srcA")
    cmds.createNode("transform", name="srcB")
    cmds.createNode("composeMatrix", name="srcM")
    cmds.connectAttr("srcA.output", "ch.input[0]")
    cmds.connectAttr("srcB.tx", "ch.input[1]")
    cmds.connectAttr("srcM.outputMatrix", "ch.input[2]")


def _selector_states():
    """Yields a label for each selector state, in a scene `_choice_scene` made."""
    for value in (0, 1, 2, 3, 7):
        _choice_scene()
        cmds.setAttr("ch.selector", value)
        yield f"unconnected {value}"
    for kind, values in (
        ("long", (1, -1, 9)),
        ("double", (0.4, 0.5, 1.49, 1.5, -0.5, -0.6, 2.5)),
    ):
        for value in values:
            _choice_scene()
            cmds.createNode("network", name="drv")
            cmds.addAttr("drv", longName="drive", attributeType=kind)
            cmds.connectAttr("drv.drive", "ch.selector")
            cmds.setAttr("drv.drive", value)
            yield f"a {kind} driver at {value}"
    for value in (1.0, 57.0):
        _choice_scene()
        cmds.createNode("transform", name="drv")
        cmds.connectAttr("drv.rx", "ch.selector")  # a unitConversion
        cmds.setAttr("drv.rx", value)
        yield f"an angle driver at {value}"
    _choice_scene()
    cmds.connectAttr("time1.outTime", "ch.selector")
    for frame in (0, 1, 2.5):
        cmds.currentTime(frame)
        yield f"time1 at frame {frame}"
    _choice_scene()
    cmds.setKeyframe("ch.selector", time=1, value=0, inTangentType="linear", outTangentType="linear")
    cmds.setKeyframe("ch.selector", time=11, value=2, inTangentType="linear", outTangentType="linear")
    for frame in (-5, 3.3, 4, 8.7, 20):
        cmds.currentTime(frame)
        yield f"an anim curve at frame {frame}"
    _choice_scene()
    cmds.expression(string="ch.selector = frame / 2;", name="ex")
    for frame in (1, 3, 5.5):
        cmds.currentTime(frame)
        yield f"an expression at frame {frame}"
    _choice_scene()
    cmds.setAttr("ch.selector", 1)
    cmds.setAttr("ch.selector", lock=True)
    yield "locked at 1"
    _choice_scene()
    cmds.setKeyframe("ch.selector", time=1, value=2)
    cmds.setKeyframe("ch.selector", time=5, value=0)
    cmds.currentTime(3)
    cmds.mute("ch.selector")
    cmds.currentTime(5)
    yield "muted at frame 3, read at frame 5"
    cmds.currentTime(1)


class TestTheChoiceHook(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        cmds.currentTime(1)
        super().tearDown()

    def test_the_hook_answers_as_the_get_route(self):
        count = 0
        for label in _selector_states():
            with self.subTest(state=label):
                count += 1
                node     = Node("ch")
                selector = node.find_attr("selector")
                output   = node.find_attr("output")
                value    = selector.get_mplug().asInt()
                self.assertEqual(value, selector.get())
                self.assertIs(type(value), type(selector.get()))
                self.assertEqual(
                    _outcome(lambda: Choice._attr_data_type_fallback(node, output)),
                    _outcome(lambda: _old_hook(node, output)),
                )
                # through data_type, from a fresh plug, against the old hook in
                # its place
                new = _outcome(lambda: Node("ch").output.get_data_type())
                with mock.patch.object(Choice, "_attr_data_type_fallback", _old_hook):
                    old = _outcome(lambda: Node("ch").output.get_data_type())
                self.assertEqual(new, old)
        self.assertEqual(count, 30)

    def test_the_states_reach_every_input(self):
        # the table selects each input, one with no source and one past the
        # inputs, from a plain value, a driver, an anim curve and an expression
        answers = set()
        for _ in _selector_states():
            node = Node("ch")
            answers.add(_old_hook(node, node.find_attr("output")))
        self.assertEqual(answers, {"float3", "doubleLinear", "matrix", "Tdata"})

    def test_the_hook_reads_the_selector_without_getattr(self):
        _choice_scene()
        cmds.setAttr("ch.selector", 1)
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            typ = Node("ch").output.get_data_type()
        self.assertEqual(typ, "doubleLinear")
        # only data_type's own query (ee03336: then a type and a value query of
        # ch.selector, get()'s); its answer, "typed", is what runs the hook
        self.assertEqual(
            [(call.args, call.kwargs) for call in probe.call_args_list],
            [(("ch.output",), {"type": True})],
        )
        self.assertEqual(cmds.getAttr("ch.output", type=True), "typed")
        node = Node("ch")
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            node._attr_data_type_fallback(node.find_attr("output"))
        self.assertEqual(probe.call_args_list, [])

    def test_a_deleted_or_freed_choice_raises_as_get_does(self):
        _choice_scene()
        cmds.setAttr("ch.selector", 1)
        node   = Node("ch")
        output = node.find_attr("output")
        # the hook's selector attr is cached on the node, as after a data_type
        self.assertEqual(node._attr_data_type_fallback(output), "doubleLinear")
        cmds.delete("ch")
        new = _outcome(lambda: Choice._attr_data_type_fallback(node, output))
        self.assertEqual(new, _outcome(lambda: _old_hook(node, output)))
        self.assertEqual(new, ["raises", [["RuntimeError", "ch already deleted!", False, False]]])
        cmds.undo()
        self.assertEqual(node._attr_data_type_fallback(output), "doubleLinear")
        # freed: a new scene (the attr handed in is a live choice's, so its name
        # passes and the hook reaches the freed node's selector)
        cmds.file(new=True, force=True)
        live = Node(cmds.createNode("choice")).find_attr("output")
        new  = _outcome(lambda: Choice._attr_data_type_fallback(node, live))
        self.assertEqual(new, _outcome(lambda: _old_hook(node, live)))
        freed = "Choice node (freed by a new scene, a file open or a reference unload)"
        self.assertEqual(new, ["raises", [["RuntimeError", f"{freed} already deleted!", False, False]]])


# --------------------------------------------------------------------------- #
#  Lazy child caches
# --------------------------------------------------------------------------- #


def _instanced_scene():
    """A transform S under T1, instanced under T2."""
    top = cmds.createNode("transform", name="T1")
    cmds.createNode("transform", name="S", parent=top)
    cmds.createNode("transform", name="T2")
    cmds.parent("T1|S", "T2", add=True, relative=True)


class TestLazyChildCaches(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def _scene(self):
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dynv", attributeType="double3")
        for axis in "XYZ":
            cmds.addAttr("md", longName="dynv" + axis, attributeType="double", parent="dynv")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.setAttr("pma.input3D[1].input3Dx", 1.0)
        cmds.sphere(name="sph", constructionHistory=False)
        _instanced_scene()

    def test_no_constructor_makes_them(self):
        self._scene()
        node  = Node("md")
        mplug = node.find_attr("input1").get_mplug()
        made = {
            "Attribute(MPlug)":   lambda: Attribute(mplug),
            "Attribute(name)":    lambda: Attribute("md.input1"),
            "Plug(MPlug)":        lambda: Plug(mplug),
            "Plug(name)":         lambda: Plug("md.input1"),
            "Plug(Attribute)":    lambda: Plug(node.find_attr("input1")),
            "_new_attr":          lambda: _base._new_attr(Plug, mplug),
            "node.attr":          lambda: Node("md").input1,
            "node.attr again":    lambda: node.input1,
            "find_attr":          lambda: Node("md").find_attr("input1"),
            "a child by name":    lambda: node.input1.input1X,
            "a child by index":   lambda: node.input1[0],
            "child(0)":           lambda: node.input1.child(0),
            "Attribute.child(0)": lambda: Attribute("md.input1").child(0),
            "an element":         lambda: Node("pma").input3D[1],
            "an element's child": lambda: Node("pma").input3D[1].input3Dx,
            "get_parent()":       lambda: node.find_attr("input1X").get_parent(),
            "a dynamic attr":     lambda: Node("md").dynv,
            "a component plug":   lambda: Node("sph").cv,
            "its element":        lambda: Node("sph").cv[1, 2],
            "an instanced plug":  lambda: Node("|T2|S").t,
            "its child":          lambda: Node("|T2|S").t.translateX,
        }
        for label, make in made.items():
            with self.subTest(made=label):
                state = vars(make())
                self.assertNotIn(NAMES, state)
                self.assertNotIn(IDS, state)
        state = _base._attr_state(mplug, None)
        self.assertNotIn(NAMES, state)
        self.assertNotIn(IDS, state)

    def test_each_attr_makes_its_own_on_first_use(self):
        self._scene()
        for cls in (Attribute, Plug):
            with self.subTest(cls=cls.__name__):
                first, second = cls("md.input1"), cls("md.input1")
                first.child(0)
                self.assertIn(IDS, vars(first))
                self.assertNotIn(NAMES, vars(first))
                getattr(first, "input1Y")
                getattr(second, "input1Y")
                for cache in (NAMES, IDS):
                    self.assertIsNot(vars(first)[cache], vars(second)[cache])
                self.assertIsNot(vars(first)[NAMES], vars(first)[IDS])
                # the name lookup filled both, every child
                self.assertEqual(sorted(vars(first)[NAMES]), ["input1X", "input1Y", "input1Z"])
                self.assertEqual(sorted(vars(first)[IDS]), [0, 1, 2])
                for cache in (NAMES, IDS):
                    for child in vars(first)[cache].values():
                        self.assertIs(type(child), Attribute)

    def test_a_name_lookup_fills_the_index_cache_as_before(self):
        self._scene()
        attr  = Attribute("md.input1")
        early = attr.child(0)
        self.assertIs(attr.child(0), early)
        by_name = attr.input1X
        # the name lookup built every child again and filled the index cache
        # over the earlier entry (as ee03336 did)
        self.assertIsNot(by_name, early)
        self.assertIs(attr.child(0), by_name)
        self.assertIs(attr.input1X, by_name)
        self.assertEqual(str(by_name), "md.input1X")
        self.assertEqual(str(early), "md.input1X")
        # a plug's lookups are new plugs each time, of the cached child's MPlug
        plug = Plug("md.input1")
        self.assertIsNot(plug.input1X, plug.input1X)
        self.assertIs(plug.input1X.get_mplug(), vars(plug)[NAMES]["input1X"].get_mplug())

    def test_errors_and_non_compounds_as_before(self):
        self._scene()
        scalar = Attribute("md.input1X")
        # re-pinned (user decision 2026-10-16, the property ban): AttributeError,
        # so hasattr answers (it was TypeError 'Plug is not a compound parent')
        with self.assertRaisesRegex(
            AttributeError, r"^md\.input1X is not a compound: it has no child 'input1Y'\.$"
        ):
            scalar.input1Y
        self.assertFalse(hasattr(scalar, "input1Y"))
        self.assertNotIn(NAMES, vars(scalar))
        self.assertNotIn(IDS, vars(scalar))
        plug = Plug("md.input1X")
        with self.assertRaisesRegex(
            AttributeError, r"^'md\.input1X' has no child or sibling attribute 'nope'\.$"
        ):
            plug.nope
        self.assertNotIn(NAMES, vars(plug))
        compound = Attribute("md.input1")
        with self.assertRaisesRegex(AttributeError, r"^md\.input1\.nope not found\.$"):
            compound.nope
        self.assertEqual(sorted(vars(compound)[NAMES]), ["input1X", "input1Y", "input1Z"])
        fresh = Attribute("md.input1")
        with self.assertRaisesRegex(TypeError, r"^unhashable type: 'list'$"):
            fresh.child([0])
        self.assertNotIn(NAMES, vars(fresh))

    def test_a_full_name_buffer_copy_holds_no_cache(self):
        self._scene()
        held  = Node("|T2|S")
        calls = []
        real  = _base._full_name_buffer

        def spy(attr):
            before = (NAMES in vars(attr), IDS in vars(attr))
            copied = real(attr)
            calls.append(before + (copied is attr, NAMES in vars(copied), IDS in vars(copied)))
            return copied

        with mock.patch.object(_base, "_full_name_buffer", spy):
            plug  = held.t
            child = plug.translateX
        # the plug and each child it cached were named through T2's path, the
        # children by a copy of a fresh attr; no attr held a cache, nor a copy
        self.assertTrue(calls)
        for before_names, before_ids, same, after_names, after_ids in calls:
            self.assertEqual((before_names, before_ids, after_names, after_ids), (False,) * 4)
        self.assertGreaterEqual(sum(1 for call in calls if not call[2]), 4)
        self.assertEqual(str.__str__(plug), "T2|S.translate")
        self.assertEqual(str.__str__(child), "T2|S.translateX")
        kept = vars(plug)[NAMES]
        self.assertEqual(
            {name: str.__str__(each) for name, each in kept.items()},
            {axis: f"T2|S.{axis}" for axis in ("translateX", "translateY", "translateZ")},
        )
        for each in kept.values():
            self.assertNotIn(NAMES, vars(each))
            self.assertNotIn(IDS, vars(each))

    def test_a_copy_of_a_fresh_plug_makes_its_own(self):
        # ee03336's copy shared the original's (then empty) caches; both are made
        # on first use now, so each makes its own. The lookups answer alike
        self._scene()
        plug = Node("md").input1
        dup  = copy.copy(plug)
        self.assertNotIn(NAMES, vars(dup))
        self.assertEqual(str(dup.input1X), str(plug.input1X))
        self.assertEqual(dup.input1X.get_mplug(), plug.input1X.get_mplug())
        self.assertIsNot(vars(dup)[NAMES], vars(plug)[NAMES])
        # a copy made once the caches exist shares them, as before
        again = copy.copy(plug)
        self.assertIs(vars(again)[NAMES], vars(plug)[NAMES])


# --------------------------------------------------------------------------- #
#  The child lookups, against ee03336 (`_child_outcomes` runs there too: public
#  API, and the plug state both sides keep)
# --------------------------------------------------------------------------- #

# (node, path, its first and last child's long names) of the plugs looked up
_CHILD_PLUGS = (
    ("md", "input1", ("input1X", "input1Z")),
    ("md", "input1X", ()),
    ("md", "dynv", ("dynvX", "dynvZ")),
    ("md", "message", ()),
    ("pma", "input3D", ("input3Dx", "input3Dz")),
    ("pma", "input3D[1]", ("input3Dx", "input3Dz")),
    ("pma", "input1D", ()),
    ("ch", "input", ()),
    ("ch", "output", ()),
    ("|T1|S", "t", ("translateX", "translateZ")),
    ("|T2|S", "t", ("translateX", "translateZ")),
    ("|T2|S", "worldMatrix", ()),
    ("T1", "rotatePivot", ("rotatePivotX", "rotatePivotZ")),
    ("cubeShape", "pnts[0]", ("pntx", "pntz")),
)
# the ways a plug is looked up: a node's Plug, find_attr's Attribute, and
# owner-less ones of the name
_HOWS = ("node.attr", "find_attr", "Plug", "Attribute")


def _child_scene():
    """The scene every plug's lookups start from (fixed names)."""
    cmds.file(new=True, force=True)
    cmds.createNode("multiplyDivide", name="md")
    cmds.addAttr("md", longName="dynv", attributeType="double3")
    for axis in "XYZ":
        cmds.addAttr("md", longName="dynv" + axis, attributeType="double", parent="dynv")
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.setAttr("pma.input3D[1].input3Dx", 1.0)
    cmds.createNode("choice", name="ch")
    cmds.connectAttr("md.output", "ch.input[0]")
    _instanced_scene()
    cmds.polyCube(name="cube", constructionHistory=False)


def _plug_of(how, node_name, path):
    """(a fresh plug of ``node_name.path`` looked up `how`, the node object it came
    from or None)."""
    if how == "Plug":
        return Plug(f"{node_name}.{path}"), None
    if how == "Attribute":
        return Attribute(f"{node_name}.{path}"), None
    node = Node(node_name)
    if how == "find_attr":
        return node.find_attr(path), node
    root, _, index = path.partition("[")
    plug = getattr(node, root)
    if index:
        plug = plug[int(index[:-1])]
    return plug, node


def _owner_label(value, node):
    owner = vars(value).get("_node")
    if owner is not None and owner is node:
        return "the node"
    return None if owner is None else type(owner).__name__


def _handle_label(value, node):
    handle = vars(value).get("_handle")
    if handle is None:
        return None
    if node is not None and handle is vars(node).get("_objhandle"):
        return "the node's"
    return "its own"


def _result(value, node):
    """What a lookup returned: for a plug its type, str buffer, name (or the
    error naming it raises), owner, node handle, attribute handle and MPlug
    name (read only once naming it passed: a freed node's MPlug points at freed
    memory); a bool as it is; a tuple item by item."""
    if isinstance(value, tuple):
        return [_result(each, node) for each in value]
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return [type(value).__name__, None if value is None else repr(value)]
    state = vars(value)
    mplug = None
    try:
        # rig's private name reader: it raises "... already deleted!" for a plug
        # of a deleted or freed node, as str() did at ee03336 (user decision 6C,
        # 2026-10-17: str() is the marked name now), so the MPlug is never read then
        name  = _name_of(value)
        mplug = state.get("_mplug")
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        name = ["raises", _chain(error)]
    return [
        type(value).__name__,
        str.__str__(value),
        name,
        _owner_label(value, node),
        _handle_label(value, node),
        state.get("_attrh") is not None,
        None if mplug is None else mplug.name(),
    ]


def _call(func):
    """The outcome of ``func()``, which returns (a lookup's result, the node object
    its plug came from or None): ["returns", `_result`] or ["raises", the
    error's chain]."""
    try:
        value, node = func()
        return ["returns", _result(value, node)]
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ["raises", _chain(error)]


def _looked_up(how, node_name, path, op):
    """(``op`` of a fresh plug of ``node_name.path`` looked up `how`, its node)."""
    plug, node = _plug_of(how, node_name, path)
    return op(plug), node


def _ops(names):
    """{label: op} of the lookups made on a plug whose first and last child's long
    names are `names` (a non-compound: a name it does not have)."""
    first, last = names or ("input1X", "input1Z")

    def twice(lookup):
        def op(plug):
            one = lookup(plug)
            return one, lookup(plug) is one
        return op

    def child_name_child(plug):
        early   = plug.child(0)
        by_name = getattr(plug, first)
        late    = plug.child(0)
        return early, by_name, late, early is by_name, late is by_name, late is early

    def name_child(plug):
        by_name = getattr(plug, first)
        return by_name, plug.child(0) is by_name, getattr(plug, last) is plug.child(2)

    return {
        "by name":          lambda plug: getattr(plug, first),
        "by name twice":    twice(lambda plug: getattr(plug, first)),
        "last by name":     lambda plug: getattr(plug, last),
        "missing name":     lambda plug: plug.nope,
        "private name":     lambda plug: getattr(plug, "_nope"),
        "child(0)":         lambda plug: plug.child(0),
        "child(0) twice":   twice(lambda plug: plug.child(0)),
        "child(2)":         lambda plug: plug.child(2),
        "child(99)":        lambda plug: plug.child(99),
        "child(-1)":        lambda plug: plug.child(-1),
        "child([0])":       lambda plug: plug.child([0]),
        "child, name, child": child_name_child,
        "name, child":      name_child,
    }


# held plugs looked up again across a delete (the undo queue on), its undo and a
# new scene, after a first lookup by name and by index
_HELD = (("md", "input1", ("input1X", "input1Z")), ("|T2|S", "t", ("translateX", "translateZ")))
_HELD_OPS = ("by name", "last by name", "child(0)", "child(2)", "missing name", "private name")


def _held_outcomes(table):
    """Adds to `table` the outcomes of `_HELD_OPS` on the plugs of `_HELD`, held
    (looked up each of `_HOWS` ways, then a child by name and by index) across a
    delete, its undo and a new scene."""
    for node_name, path, names in _HELD:
        _child_scene()
        held = {how: _plug_of(how, node_name, path) for how in _HOWS}
        ops  = _ops(names)
        for plug, node in held.values():
            getattr(plug, names[0])
            plug.child(2)
        for state, change in (
            ("deleted", lambda: cmds.delete(node_name)),
            ("undone", cmds.undo),
            ("freed", lambda: cmds.file(new=True, force=True)),
        ):
            change()
            for how, (plug, node) in held.items():
                for label in _HELD_OPS:
                    key = f"{node_name}.{path} {how} {state}: {label}"
                    table[key] = _call(lambda: (ops[label](plug), node))


def _child_outcomes():
    """{"<node>.<path> <how>: <op>": outcome} of every op of `_ops` on a fresh plug
    of each of `_CHILD_PLUGS` looked up each of `_HOWS` ways, and of the held
    plugs of `_HELD` across a delete, its undo and a new scene."""
    queue = (
        cmds.undoInfo(query=True, state=True),
        cmds.undoInfo(query=True, infinity=True),
        cmds.undoInfo(query=True, length=True),
    )
    cmds.undoInfo(state=True, infinity=True)
    table = {}
    try:
        _child_scene()
        cmds.flushUndo()
        for node_name, path, names in _CHILD_PLUGS:
            for how in _HOWS:
                for label, op in _ops(names).items():
                    table[f"{node_name}.{path} {how}: {label}"] = _call(
                        lambda: _looked_up(how, node_name, path, op)
                    )
        _held_outcomes(table)
    finally:
        cmds.file(new=True, force=True)
        cmds.flushUndo()
        state, infinity, length = queue
        if infinity:
            cmds.undoInfo(infinity=True)
        else:
            cmds.undoInfo(infinity=False, length=length)
        cmds.undoInfo(state=state)
    return table


# ee03336's outcomes (runs\r5\A6\gen_oracle.py on its snapshot)
_ORACLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "r5_small_oracle_ee03336.json"
)


def _non_compound_repinned(expected, got):
    """`expected` (an ee03336 outcome) with its non-compound lookup error re-pinned
    (user decision 2026-10-16, the property ban): a bare Attribute's name lookup
    on a non-compound raises AttributeError, so ``hasattr`` answers, where it
    raised TypeError 'Plug is not a compound parent'; the message is `got`'s
    when it says so."""
    if not (expected[0] == "raises" and got[0] == "raises" and len(expected[1]) == len(got[1])):
        return expected
    chain = []
    for old, new in zip(expected[1], got[1]):
        if (
            old[:2] == ["TypeError", "Plug is not a compound parent"]
            and new[0] == "AttributeError"
            and re.search(r" is not a compound: it has no child '\w+'\.$", new[1])
        ):
            old = ["AttributeError", new[1], *old[2:]]
        chain.append(old)
    return ["raises", chain]


def _underscore_probe_repinned(case, expected, got):
    """`expected` (an ee03336 outcome) re-pinned for a '_' name looked up on a
    plug of a deleted or freed node (user decision 2026-10-16, must-fix: a '_'
    probe raises AttributeError, never RuntimeError, so ``hasattr`` and numpy's
    probes answer; fix round 2026-10-16): ee03336's "already deleted!" becomes
    `got`'s AttributeError. Any other case is `expected` as it is."""
    if (
        case.endswith(": private name")
        and (" deleted:" in case or " freed:" in case)
        and expected[0] == "raises" and got[0] == "raises"
        and expected[1][0][0] == "RuntimeError" and "already deleted!" in expected[1][0][1]
        and got[1][0][0] == "AttributeError"
    ):
        return got
    return expected


class TestTheChildLookupsOfEe03336(MayaTestCase):
    """Every child lookup of `_child_outcomes` gives what ee03336 gave."""

    def test_every_child_lookup_as_ee03336(self):
        with open(_ORACLE_PATH) as fh:
            oracle = json.load(fh)
        table = json.loads(json.dumps(_child_outcomes()))
        self.assertEqual(sorted(table), sorted(oracle))
        for case in sorted(oracle):
            with self.subTest(case=case):
                # re-pinned (user decision 2026-10-16, the property ban): the
                # non-compound lookup error (`_non_compound_repinned`)
                expected = _non_compound_repinned(oracle[case], table[case])
                # re-pinned (user decision 2026-10-16, must-fix: a '_' probe on a
                # deleted or freed node's plug is an AttributeError)
                expected = _underscore_probe_repinned(case, expected, table[case])
                self.assertEqual(table[case], expected)

    def test_the_table_reaches_every_branch(self):
        # the oracle holds children, cache hits, the errors of each kind, and
        # held plugs naming a deleted and a freed node
        with open(_ORACLE_PATH) as fh:
            oracle = json.load(fh)
        returned = [case for case, outcome in oracle.items() if outcome[0] == "returns"]
        errors   = {outcome[1][0][0] for outcome in oracle.values() if outcome[0] == "raises"}
        texts    = [outcome[1][0][1] for outcome in oracle.values() if outcome[0] == "raises"]
        self.assertGreater(len(returned), 200)
        self.assertTrue({"AttributeError", "TypeError", "RuntimeError"} <= errors, errors)
        self.assertTrue(any(text.endswith(" not found.") for text in texts))
        self.assertTrue(any("unhashable type" in text for text in texts))
        self.assertTrue(any("not a compound parent" in text for text in texts))
        names = json.dumps(oracle)
        self.assertIn("already deleted!", names)
        # a cached child comes back the same object, a plug's lookups never do
        self.assertEqual(oracle["md.input1 find_attr: by name twice"][1][1], True)
        self.assertEqual(oracle["md.input1 node.attr: by name twice"][1][1], False)
