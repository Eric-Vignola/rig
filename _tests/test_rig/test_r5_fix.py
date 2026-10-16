"""Round 5, Phase A fix step: what the two reviews of A1-A6 found.

* H4 (A4) shared one data type query per operand of a ``<<``. A choice whose
  selected input is empty cannot compute its output: Maya leaves the output
  dirty, every query of it runs the compute again, and each run flips what a
  target connected to it afterwards reads (``xf.v << ch.output`` then reads
  False after one query, True after two). So a fact that found no type
  ("Tdata", "typed") on a plug whose query may run a compute
  (``shorthand._query_may_compute``: a plug some attribute of its node affects,
  or a destination, itself or above it), or whose query raised, is queried
  again where it was before; one that found a type, or a plug that holds stored
  data only (a new choice slot), is reused.
* ``find_attr`` answers a cached attr named by a plain str with no filter
  without calling its lookup (A5 had moved the cache hit into it).
* ``memoize._collect_handles`` follows the naming rule of ``_attribute_key``
  (``_base._dg_named_owner``, and ``_base._dg_named_node`` for a node object),
  so a class that names its nodes through its own attribute lookup is collected
  by name, as its key is.
* The scene callbacks' prune drops the freed nodes' serials
  (``_base._NODE_SERIALS``), which kept every scene's until its size limit.
  (It dropped the freed attributes' handles of ``_base._ATTR_HANDLES`` too,
  until round 5 AF retired that table: lead decision 2026-10-02, option c.)

Tests:

* ``TestAnEmptyChoiceOutput``: ``<<`` from the output of a choice with an
  empty selected input, into a bool, a compound, a translate, dynamic bool and
  double attrs, an auto-indexed multi and a new choice slot, and through
  ``functions.choice`` with a selector out of range, leaves the values
  45d2c13 left (and queries the output as often); the facts' reuse rule and
  ``_query_may_compute`` over plug shapes; Maya's flip itself.
* ``TestFindAttrHit``: a hit returns the cached attr without the lookup; every
  other form (a filter, an int, a Plug, an Attribute, a miss) takes the lookup
  and answers as it does; a node with no attr cache raises as the lookup does.
* ``TestOneNamingRule``: memoize keeps no rule of its own; the node test is the
  owner test over classes, patches and instance attributes; a plug of a class
  that answers ``name`` through its own ``__getattribute__`` is collected, and
  keyed, by the name it gives.
* ``TestFreedTablesArePruned``: a new scene drops the old scene's node serials,
  a prune keeps the live ones (the same objects), and a plug built again takes
  its own handle of its attribute.
"""

from collections import Counter
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

from rig import Node, Plug, functions
from rig._internal import memoize as memoize_module
from rig._internal import shorthand as shorthand_module
from rig.nodetypes import _base
from rig.nodetypes._base import Attribute
from rig.nodetypes.dg_node import DGNode
from rig._tests._base import MayaTestCase


def _type_queries(call):
    """The plugs ``call()`` queries with ``cmds.getAttr(..., type=True)``, counted."""
    with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
        call()
    return Counter(str(c.args[0]) for c in probe.call_args_list if c.kwargs.get("type"))


def _empty_choice():
    """A choice with no input (its output cannot compute) and a transform."""
    cmds.createNode("choice", name="ch")
    cmds.createNode("transform", name="xf")
    cmds.createNode("transform", name="xf2")
    cmds.setAttr("xf.s", 2, 2, 2)
    cmds.setAttr("xf.tx", 7)


class TestAnEmptyChoiceOutput(MayaTestCase):
    """H4: a fact that found no type on a plug whose query may compute is not
    reused, so a failing compute runs as often as at 45d2c13."""

    TEST_START_NEW_SCENE = True

    def test_maya_flips_the_value_with_each_query(self):
        # the mechanism: a type query of the output runs its failing compute,
        # and the number of runs decides what a target connected later reads
        reads = []
        for queries in (0, 1, 2, 3):
            cmds.file(new=True, force=True)
            _empty_choice()
            for _ in range(queries):
                self.assertEqual(cmds.getAttr("ch.output", type=True), "Tdata")
            cmds.connectAttr("ch.output", "xf.visibility", force=True)
            reads.append(cmds.getAttr("xf.visibility"))
        self.assertEqual(reads, [True, False, True, False])

    def test_a_target_reads_what_45d2c13_left(self):
        # (label, op, the output's type queries during the op at 45d2c13, the
        # plugs read after and what 45d2c13 read)
        cases = (
            ("a bool", lambda: Node("xf").visibility << Node("ch").output, 2,
             {"xf.visibility": True}),
            ("a compound", lambda: Node("xf").s << Node("ch").output, 2,
             {"xf.s": [(2.0, 2.0, 2.0)]}),
            ("a translate", lambda: Node("xf").tx << Node("ch").output, 2,
             {"xf.tx": 7.0}),
            ("an auto-indexed multi", lambda: Node("pma").input1D << Node("ch").output, 2,
             {"pma.input1D[0]": 0.0}),
            # re-pinned (features step 4, Typed): into a generic slot no shorthand runs,
            # so the output's type is not queried (45d2c13: 2; an even count either way,
            # so a later target reads the same)
            ("a new choice slot", lambda: Node("ch2").input[0] << Node("ch").output, 0, {}),
        )
        for label, op, queries, reads in cases:
            with self.subTest(case=label):
                cmds.file(new=True, force=True)
                _empty_choice()
                cmds.createNode("plusMinusAverage", name="pma")
                cmds.createNode("choice", name="ch2")
                self.assertEqual(_type_queries(op)["ch.output"], queries)
                for plug, value in reads.items():
                    self.assertEqual(cmds.getAttr(plug), value)
                # another target connected to the output afterwards
                cmds.connectAttr("ch.output", "xf2.visibility", force=True)
                self.assertIs(cmds.getAttr("xf2.visibility"), True)

    def test_dynamic_attrs_and_a_second_target(self):
        _empty_choice()
        cmds.addAttr("xf", longName="flag", attributeType="bool", defaultValue=True)
        cmds.addAttr("xf", longName="amount", attributeType="double", defaultValue=5.0)
        Node("xf").flag << Node("ch").output
        self.assertIs(cmds.getAttr("xf.flag"), True)       # the old head: False
        Node("xf").amount << Node("ch").output
        self.assertEqual(cmds.getAttr("xf.amount"), 5.0)   # the old head: 0.0
        self.assertIs(cmds.getAttr("xf.flag"), True)
        Node("xf").ty << Node("ch").output
        self.assertEqual(cmds.getAttr("xf.ty"), 0.0)
        self.assertEqual(cmds.getAttr("xf.amount"), 5.0)

    def test_functions_choice_with_its_selector_out_of_range(self):
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("transform", name="xf")
        cmds.setAttr("md.input1X", 3.0)
        cmds.setAttr("md.input1Y", 4.0)
        out = functions.choice([Node("md").outputX, Node("md").outputY], 5)
        name = str(out).split(".")[0]
        self.assertEqual(_type_queries(lambda: Node("xf").visibility << out)[str(out)], 2)
        self.assertIs(cmds.getAttr("xf.visibility"), True)  # the old head: False
        Node("xf").tx << out
        self.assertEqual(cmds.getAttr("xf.tx"), 0.0)
        self.assertIs(cmds.getAttr("xf.visibility"), True)
        self.assertEqual(cmds.getAttr(f"{name}.selector"), 5)

    def test_the_facts_reuse_rule(self):
        _empty_choice()
        cmds.createNode("choice", name="fed")
        cmds.createNode("choice", name="typed")
        cmds.createNode("multiplyDivide", name="md")
        cmds.connectAttr("ch.output", "fed.input[0]")
        cmds.connectAttr("md.output", "typed.input[0]")
        held = Node("md").outputY
        str(held)
        for label, plug, data_type, reusable in (
            ("an output that cannot compute", Node("ch").output, "Tdata", False),
            ("a new slot (stored data only)", Node("ch").input[3], "Tdata", True),
            ("a slot fed by an output that cannot compute", Node("fed").input[0], "Tdata", False),
            ("an output its selected input types", Node("typed").output, "float3", True),
            ("a static type", Node("md").input1X, "float", True),
        ):
            with self.subTest(case=label):
                facts = shorthand_module._Facts(plug)
                self.assertEqual(facts._get_data_type(), data_type)
                self.assertIs(facts._reusable, reusable)
        cmds.delete("md")
        facts = shorthand_module._Facts(held)
        self.assertIsNone(facts._get_data_type())  # it raised
        self.assertIs(facts._reusable, False)

    def test_which_queries_may_compute(self):
        cmds.createNode("choice", name="ch")
        cmds.createNode("multiplyDivide", name="md")
        cmds.createNode("plusMinusAverage", name="pma")
        cmds.createNode("transform", name="xf")
        cmds.addAttr("xf", longName="dynFree", attributeType="double")
        cmds.addAttr("xf", longName="dynFed", attributeType="double")
        cmds.connectAttr("md.outputX", "xf.dynFed")
        cmds.connectAttr("md.output", "pma.input3D[0]")
        cmds.setAttr("pma.input3D[1].input3Dx", 1.0)
        for label, plug, may in (
            ("a choice output (its inputs affect it)", Node("ch").output, True),
            ("an output", Node("md").outputX, True),
            ("a new choice slot", Node("ch").input[3], False),
            ("a multi root of an input", Node("ch").input, False),
            ("an unconnected input", Node("md").input1X, False),
            ("an unconnected dynamic attr", Node("xf").dynFree, False),
            ("a connected dynamic attr", Node("xf").dynFed, True),
            ("a child of a connected element", Node("pma").input3D[0].input3Dx, True),
            ("a child of an unconnected element", Node("pma").input3D[1].input3Dx, False),
        ):
            with self.subTest(case=label):
                self.assertIs(shorthand_module._query_may_compute(plug), may)

        class _NoPlug:
            @property
            def plug(self):
                raise RuntimeError("no plug")

        self.assertIs(shorthand_module._query_may_compute(_NoPlug()), True)


def _outcome(call):
    """("ok", type, str buffer) of what ``call()`` returns (None as is), or
    ("raises", type, message)."""
    try:
        answer = call()
    except Exception as error:  # noqa: BLE001 - the error is the thing compared
        return ("raises", type(error), str(error))
    if answer is None:
        return ("ok", None, None)
    return ("ok", type(answer), str.__str__(answer))


class TestFindAttrHit(MayaTestCase):
    """A5 moved find_attr's cache hit into its lookup; the hit is read first again."""

    TEST_START_NEW_SCENE = True

    def _lookups(self, call):
        with mock.patch.object(
            DGNode, "_find_attr_fresh", autospec=True, side_effect=DGNode._find_attr_fresh
        ) as lookup:
            answer = _outcome(call)
        return answer, lookup.call_count

    def test_a_cached_name_is_answered_without_the_lookup(self):
        node = Node(cmds.createNode("multiplyDivide", name="md"))
        with mock.patch.object(
            DGNode, "_find_attr_fresh", autospec=True, side_effect=DGNode._find_attr_fresh
        ) as lookup:
            first = node.find_attr("input1X")
            self.assertEqual(lookup.call_count, 1)  # a miss: the lookup builds it
            again = node.find_attr("input1X")
            self.assertEqual(lookup.call_count, 1)
        self.assertIs(again, first)
        self.assertIs(again, vars(node)["_attr_dict"]["input1X"])
        self.assertIs(again.node, node)

    def test_every_other_form_takes_the_lookup(self):
        node = Node(cmds.createNode("multiplyDivide", name="md"))
        node.find_attr("input1X")
        plug = node.input1X
        attr = node.find_attr("input1Y")
        for label, args, kwargs in (
            ("a data type filter", ("input1X",), {"data_type": "float"}),
            ("a category filter", ("input1X",), {"category": "no such category"}),
            ("an int", (0,), {}),
            ("a Plug", (plug,), {}),
            ("an Attribute", (attr,), {}),
            ("a miss", ("no_such_attr",), {"quiet": True}),
            ("a miss that raises", ("no_such_attr",), {}),
        ):
            with self.subTest(case=label):
                answer, calls = self._lookups(lambda: node.find_attr(*args, **kwargs))
                self.assertEqual(calls, 1)
                direct = _outcome(lambda: DGNode._find_attr_fresh(node, *args, **kwargs)[0])
                self.assertEqual(answer, direct)

    def test_a_node_with_no_attr_cache_raises_as_the_lookup_does(self):
        node  = Node(cmds.createNode("multiplyDivide", name="md"))
        cache = vars(node).pop("_attr_dict")
        try:
            hit    = _outcome(lambda: node.find_attr("input1X"))
            lookup = _outcome(lambda: DGNode._find_attr_fresh(node, "input1X"))
        finally:
            vars(node)["_attr_dict"] = cache
        self.assertEqual(hit[0], "raises")
        self.assertEqual(hit, lookup)


class TestOneNamingRule(MayaTestCase):
    """memoize's handle collection and its keys name a plug by one rule."""

    TEST_START_NEW_SCENE = True

    def test_memoize_keeps_no_rule_of_its_own(self):
        self.assertFalse(hasattr(memoize_module, "_dg_named"))
        self.assertFalse(hasattr(memoize_module, "_DG_NAMING"))
        self.assertIs(memoize_module._dg_named_owner, _base._dg_named_owner)
        self.assertIs(memoize_module._dg_named_node, _base._dg_named_node)

    def test_the_node_test_is_the_owner_test(self):
        dg  = Node(cmds.createNode("multiplyDivide", name="md"))
        dag = Node(cmds.createNode("transform", name="xf"))

        def agree():
            for node, plug in ((dg, dg.input1X), (dag, dag.tx)):
                owner = _base._dg_named_owner(plug)
                self.assertIs(owner, node if _base._dg_named_node(node) else None)

        self.assertIs(_base._dg_named_node(dg), True)
        self.assertIs(_base._dg_named_node(dag), False)
        agree()
        for owner, name, value in (
            (DGNode, "ensure_valid", lambda node: None),
            (DGNode, "fn_set", property(lambda node: node._fn_set)),
            (DGNode, "name", property(lambda node: node._fn_set.name())),
            (type(dg), "__getattribute__", lambda node, name: object.__getattribute__(node, name)),
        ):
            with self.subTest(patched=f"{owner.__name__}.{name}"):
                with mock.patch.object(owner, name, value):
                    self.assertIs(_base._dg_named_node(dg), False)
                    agree()
                self.assertIs(_base._dg_named_node(dg), True)
        vars(dg)["ensure_valid"] = dg.ensure_valid
        try:
            self.assertIs(_base._dg_named_node(dg), False)
            agree()
        finally:
            del vars(dg)["ensure_valid"]

    def test_a_class_naming_through_its_own_lookup_is_collected_by_name(self):
        held  = Node(cmds.createNode("multiplyDivide", name="held"))
        other = Node(cmds.createNode("multiplyDivide", name="other"))
        plug  = held.input1X
        str(plug)

        def renamed(node, name):
            if name == "name":
                return "other"
            return object.__getattribute__(node, name)

        def handles(obj):
            out = []
            memoize_module._collect_handles(obj, out)
            return [h.hashCode() for h in out]

        self.assertEqual(handles(plug), [held._objhandle.hashCode()])
        self.assertEqual(handles(held), [held._objhandle.hashCode()])
        with mock.patch.object(type(held), "__getattribute__", renamed):
            # the name gives the other node, and so do the handles, as the key
            self.assertEqual(plug.full_name, "other.input1X")
            self.assertEqual(handles(plug), [other._objhandle.hashCode()])
            self.assertEqual(handles(held), [other._objhandle.hashCode()])
            self.assertEqual(
                memoize_module._attribute_key(plug)[0],
                (other._fn_set.uuid().asString(), other._objhandle.hashCode()),
            )


class TestFreedTablesArePruned(MayaTestCase):
    """The scene callbacks' prune drops the freed serials of `_base`'s table.

    re-pinned (round 5 AF, lead decision 2026-10-02, option c): the attribute
    handle table these tests checked too (``_ATTR_HANDLES``) is gone: every plug
    takes the handle of its own attribute when it is built."""

    TEST_START_NEW_SCENE = True

    def _fill(self):
        cmds.createNode("multiplyDivide", name="md")
        cmds.addAttr("md", longName="dynx", attributeType="double")
        plug = Node("md").dynx
        hash(Plug("md.input1X"))
        handle = vars(plug)["_attrh"]
        self.assertIsNotNone(handle)
        return plug, handle

    def _dead(self):
        return len([e for entries in _base._NODE_SERIALS.values() for e in entries if not e[0].isAlive()])

    def test_a_new_scene_drops_the_freed_entries(self):
        self.assertTrue(
            memoize_module._SCENE_CALLBACKS_READY or memoize_module._ensure_scene_callbacks()
        )
        plug, handle = self._fill()
        self.assertTrue(_base._NODE_SERIALS)
        cmds.file(new=True, force=True)
        self.assertEqual(self._dead(), 0)
        self.assertFalse(handle.isAlive())  # the plug's own handle tells it

    def test_a_prune_keeps_the_live_entries(self):
        plug, handle = self._fill()
        serials = {c: [(id(e[0]), e[2]) for e in v] for c, v in _base._NODE_SERIALS.items()
                   if all(e[0].isAlive() for e in v)}
        self.assertTrue(serials)
        memoize_module._prune_after_scene_change()
        for c, entries in serials.items():
            self.assertEqual([(id(e[0]), e[2]) for e in _base._NODE_SERIALS[c]], entries)
        # a plug built again from the attribute's MPlug takes its own handle of it
        sel = OpenMaya.MSelectionList()
        sel.add("md.dynx")
        again = vars(Attribute(sel.getPlug(0)))["_attrh"]
        self.assertIsNot(again, handle)
        self.assertEqual(again.hashCode(), handle.hashCode())
        self.assertTrue(again.isAlive())
