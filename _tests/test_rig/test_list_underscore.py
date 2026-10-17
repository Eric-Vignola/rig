"""A ``List`` broadcasts a Maya attr whose name starts with ``_`` (LU, 2026-10-16).

The bug (G:\\My Drive\\rig_list_underscore_names.md): ``List.__getattr__`` raised
AttributeError for every ``_`` name, and ``List.__setattr__`` stored every ``_`` name as Python
state on the list, so ``nodes.__tag__`` raised and ``nodes.__tag__ = 5`` changed nothing in
Maya, though ``Node`` and ``Plug`` resolve such a name (``persp.__is_camera__``).

* `TestRead`: ``nodes.__tag__`` is the List of every element's plug: nodes, plugs (a sibling
  attr, a compound child), a mix, element plugs, a ``multi[:]`` slice.
* `TestWrite`: ``nodes.__tag__ = 5`` sets every element (one undo step, both edit routes) and
  leaves ``vars(nodes)`` as it was; the list's own state stays Python state.
* `TestNotEveryElement`: a name the first element lacks is AttributeError; a name only some
  elements have raises naming the first that lacks it, in any order, and sets nothing; an
  assignment of a name no element has raises, as on a node (the '_' rule, user decision
  2026-10-16); a List subclass keeps the private state its class body declares.
* `TestProbes`: Python's probes (``__deepcopy__``, ``copy.copy``, an empty List, numpy, a
  Python member of the node such as ``_fn_set``) get what they got before, and a probe on a
  long list costs one lookup.
* `TestDeleted`: an element deleted to the undo queue or freed by a new scene is skipped in
  either place, rig.skip in its slot, with one warning naming it (re-pinned, user decision
  2026-10-16 "Deleted nodes in a List": it raised ``already deleted!``).

Negative control: on 818c24f (runs\\features\\listunderscore\\negctl3.json) 15 of the 23 tests
fail; the eight guards (the probes, a Python member of an element, the list's own state, a
name the first element lacks, a child only one plug has, the public names, an assignment no
element has, a subclass's declared state) pass there too. On 317d9eb (negctl2.json, the first
LU commit) the five LU FIX tests fail: an assignment the first element lacks, a deleted first
element (both), and the two probe costs.
"""

import copy
from unittest import mock

import numpy as np
from maya import cmds

from rig import Float, InjectionError, List, Node
from rig._internal import journal
from rig._internal import list as list_module
from rig._internal.plug import Plug
from rig.nodetypes.dg_node import DGNode
from rig._tests._base import MayaTestCase
from rig._tests._undo import UndoWalk


def _tagged(*names, value=1.0):
    """Transforms ``names``, each with the doubles ``__tag__`` and ``tag`` set to ``value``."""
    for name in names:
        cmds.createNode("transform", name=name)
        Node(name) << Float("__tag__") << value
        Node(name) << Float("tag") << value
    return [Node(name) for name in names]


def _pair(name):
    """A double2 ``__pair__`` (``__pairX__`` / ``__pairY__``) and a compound multi ``arr``
    with the double child ``__w__`` (elements 0 and 1) on ``name``."""
    cmds.addAttr(name, ln="__pair__", at="double2")
    cmds.addAttr(name, ln="__pairX__", at="double", p="__pair__")
    cmds.addAttr(name, ln="__pairY__", at="double", p="__pair__")
    cmds.addAttr(name, ln="arr", at="compound", nc=1, m=True)
    cmds.addAttr(name, ln="__w__", at="double", p="arr")
    cmds.setAttr(name + ".arr[0].__w__", 0)
    cmds.setAttr(name + ".arr[1].__w__", 0)


def _names(items):
    return [str(x) for x in items]


class _Case(UndoWalk, MayaTestCase):
    TEST_START_NEW_SCENE = True


class TestRead(_Case):
    def test_nodes(self):
        a, b = _tagged("a", "b")
        found = List([a, b]).__tag__
        self.assertIs(type(found), List)
        self.assertEqual(_names(found), ["a.__tag__", "b.__tag__"])
        self.assertEqual(list(found >> None), [1.0, 1.0])
        # as the str key does, and as each node does
        self.assertEqual(_names(List([a, b])["__tag__"]), _names(found))
        self.assertEqual(str(a.__tag__), "a.__tag__")

    def test_the_camera_flag_of_the_report(self):
        cmds.addAttr("persp", ln="__is_camera__", at="bool")
        cmds.addAttr("top", ln="__is_camera__", at="bool")
        cams = List([Node("persp"), Node("top")])
        self.assertEqual(_names(cams.__is_camera__), ["persp.__is_camera__", "top.__is_camera__"])
        cams.__is_camera__ = True
        self.assertTrue(cmds.getAttr("persp.__is_camera__"))
        self.assertTrue(cmds.getAttr("top.__is_camera__"))

    def test_plugs_reach_an_attr_of_their_node(self):
        a, b = _tagged("a", "b")
        self.assertEqual(_names(List([a.t, b.t]).__tag__), ["a.__tag__", "b.__tag__"])

    def test_plugs_reach_a_compound_child(self):
        a, b = _tagged("a", "b")
        _pair("a")
        _pair("b")
        found = List([a.__pair__, b.__pair__]).__pairX__
        self.assertEqual(_names(found), ["a.__pairX__", "b.__pairX__"])
        self.assertEqual(str(a.__pair__.__pairX__), "a.__pairX__")  # as a plug does

    def test_nodes_and_plugs_mixed(self):
        a, b = _tagged("a", "b")
        self.assertEqual(_names(List([a, b.t]).__tag__), ["a.__tag__", "b.__tag__"])

    def test_element_plugs(self):
        a, b = _tagged("a", "b")
        _pair("a")
        _pair("b")
        found = List([a.arr[0], a.arr[1], b.arr[0]]).__w__
        self.assertEqual(_names(found), ["a.arr[0].__w__", "a.arr[1].__w__", "b.arr[0].__w__"])
        self.assertEqual(_names(a.arr[:].__w__), ["a.arr[0].__w__", "a.arr[1].__w__"])
        a.arr[:].__w__ = 3
        self.assertEqual(cmds.getAttr("a.arr[1].__w__"), 3)

    def test_public_names_are_unchanged(self):
        a, b = _tagged("a", "b")
        nodes = List([a, b])
        self.assertEqual(_names(nodes.tag), ["a.tag", "b.tag"])
        self.assertEqual(type(nodes.delete).__name__, "_Methods")
        with self.assertRaises(TypeError):
            nodes.delete()


class TestWrite(_Case):
    def test_sets_every_element_as_the_public_name_does(self):
        # one undo step on the API route; on the cmds route (API_EDITS False) a List
        # broadcast takes one step per element for a public name too (818c24f), so the
        # '_' name is held to the public name's count there
        steps = {}
        for api in (True, False):
            for attr in ("tag", "__tag__"):
                with self.subTest(api=api, attr=attr):
                    self.new_scene()
                    self.undo_queue_restore()
                    self.undo_queue_on()
                    with mock.patch.object(journal, "API_EDITS", api):
                        a, b = _tagged("a", "b")
                        nodes = List([a, b])
                        before = dict(vars(nodes))
                        cmds.flushUndo()
                        setattr(nodes, attr, 5)
                        self.assertEqual(cmds.getAttr(f"a.{attr}"), 5)
                        self.assertEqual(cmds.getAttr(f"b.{attr}"), 5)
                        self.assertEqual(vars(nodes), before)
                        steps[api, attr] = self.undo_all()
                        self.assertEqual(cmds.getAttr(f"a.{attr}"), 1)
                        self.assertEqual(cmds.getAttr(f"b.{attr}"), 1)
                        self.redo_all()
                        self.assertEqual(cmds.getAttr(f"b.{attr}"), 5)
        self.assertEqual(steps[True, "__tag__"], 1)
        self.assertEqual(steps[False, "__tag__"], steps[False, "tag"])

    def test_pairs_a_sequence_and_connects_a_plug(self):
        a, b, src = _tagged("a", "b", "src")
        nodes = List([a, b])
        nodes.__tag__ = [7, 8]
        self.assertEqual((cmds.getAttr("a.__tag__"), cmds.getAttr("b.__tag__")), (7, 8))
        nodes.__tag__ = src.tag
        self.assertEqual(cmds.listConnections("b.__tag__", source=True, plugs=True), ["src.tag"])

    def test_as_the_public_name_does_when_an_element_refuses(self):
        # a refusal mid-statement: the same outcome as the public name, and undo still works
        results = {}
        for attr in ("tag", "__tag__"):
            with self.subTest(attr=attr):
                self.new_scene()
                self.undo_queue_restore()
                self.undo_queue_on()
                a, b = _tagged("a", "b")
                cmds.setAttr(f"b.{attr}", lock=True)
                cmds.flushUndo()
                with self.assertRaises(InjectionError):
                    setattr(List([a, b]), attr, 5)
                results[attr] = (cmds.getAttr(f"a.{attr}"), cmds.getAttr(f"b.{attr}"))
                cmds.setAttr("a.tx", 2)
                self.undo_steps(1)
                self.assertEqual(cmds.getAttr("a.tx"), 0)
        self.assertEqual(results["__tag__"], results["tag"])

    def test_the_lists_own_state_stays_python_state(self):
        a, b = _tagged("a", "b")
        nodes = List([a, b])
        nodes._parent_multi = 3
        self.assertEqual(vars(nodes)["_parent_multi"], 3)
        # a name already stored on the list, though every element has that attr
        vars(nodes)["__tag__"] = "mine"
        nodes.__tag__ = 9
        self.assertEqual(vars(nodes)["__tag__"], 9)
        self.assertEqual(cmds.getAttr("a.__tag__"), 1)
        # a name the class declares
        nodes.__doc__ = "doc"
        self.assertEqual(vars(nodes)["__doc__"], "doc")
        # _Methods' own name, on a plain List too
        cmds.addAttr("a", ln="_method", at="double")
        cmds.addAttr("b", ln="_method", at="double")
        nodes._method = "x"
        self.assertEqual(vars(nodes)["_method"], "x")
        self.assertEqual(cmds.getAttr("a._method"), 0)
        methods = nodes.delete
        self.assertEqual(methods._method, "delete")


class TestNotEveryElement(_Case):
    def test_the_first_lacks_it(self):
        a, = _tagged("a")
        d = Node.create("transform", name="d")
        for items in ([d, a], [3.0, a], [None, a]):
            with self.subTest(items=items):
                with self.assertRaises(AttributeError) as caught:
                    List(items).__tag__
                self.assertEqual(str(caught.exception), "__tag__")  # as before
                self.assertEqual(getattr(List(items), "__tag__", "raised"), "raised")

    def test_an_assignment_the_first_lacks_raises_when_another_has_it(self):
        # in any order: [d, a] raises as [a, d] does, and sets nothing (LU FIX)
        a, = _tagged("a")
        d = Node.create("transform", name="d")
        for items in ([d, a], [None, a], [3.0, a], [d, d.t, a.t]):
            with self.subTest(items=_names(items)):
                nodes = List(items)
                state = dict(vars(nodes))
                with self.assertRaisesRegex(
                    AttributeError, r"element \[0\] .*'__tag__', which element \[\d\] has"
                ):
                    nodes.__tag__ = 7
                self.assertEqual(vars(nodes), state)
                self.assertEqual(cmds.getAttr("a.__tag__"), 1)  # no partial broadcast
                with self.assertRaises(AttributeError):  # a read still raises
                    nodes.__tag__

    def test_an_assignment_no_element_has_is_python_state_as_on_a_node(self):
        # re-pinned (user decision 2026-10-16, the '_' rule B): a '_' name that is
        # no List state, no class member and no element's Maya attr raises, as on
        # a node (the typo the pending question was about), and stores nothing
        a, = _tagged("a")
        d = Node.create("transform", name="d")
        nodes = List([d, a])
        with self.assertRaisesRegex(AttributeError, r"no element has '__tga__'"):
            nodes.__tga__ = 5
        self.assertNotIn("__tga__", vars(nodes))
        with self.assertRaises(AttributeError):
            nodes.__tga__
        with self.assertRaisesRegex(AttributeError, r"declares no '__tga__'"):
            d.__tga__ = 5
        self.assertNotIn("__tga__", vars(d))

    def test_a_subclass_keeps_the_private_state_its_class_declares(self):
        # a '_' name a List subclass writes in its __init__ is Python state when
        # its class body declares it, as on a node class (undeclared, it is a
        # broadcast when every element has a Maya attr of that name: documented)
        class Sub(List):
            _extra = None

            def __init__(self, items, extra):
                super().__init__(items)
                self._extra = extra

        a, b = _tagged("a", "b")
        cmds.addAttr("a", ln="_extra", at="double")
        cmds.addAttr("b", ln="_extra", at="double")
        nodes = Sub([a, b], 7)
        self.assertEqual(vars(nodes)["_extra"], 7)
        self.assertEqual(cmds.getAttr("a._extra"), 0)
        self.assertEqual(copy.copy(nodes)._extra, 7)

    def test_only_the_first_has_it(self):
        a, b = _tagged("a", "b")
        d = Node.create("transform", name="d")
        for items in ([a, d], [a, b, d], [a, 3.0], [a.t, d.t]):
            with self.subTest(items=_names(items)):
                with self.assertRaisesRegex(AttributeError, r"element \[\d\] .*'__tag__'"):
                    List(items).__tag__
                state = dict(vars(List(items)))
                nodes = List(items)
                with self.assertRaisesRegex(AttributeError, r"element \[\d\]"):
                    nodes.__tag__ = 5
                self.assertEqual(vars(nodes), state)
                self.assertEqual(cmds.getAttr("a.__tag__"), 1)  # nothing set
                self.assertEqual(cmds.getAttr("b.__tag__"), 1)

    def test_a_child_only_some_plugs_have(self):
        a, b = _tagged("a", "b")
        _pair("a")
        with self.assertRaises(AttributeError):
            List([a.__pair__, b.t]).__pairX__


class TestProbes(_Case):
    def test_protocol_probes(self):
        a, b = _tagged("a", "b")
        nodes = List([a, b])
        self.assertIsNone(getattr(nodes, "__deepcopy__", None))
        self.assertFalse(hasattr(nodes, "__array__"))
        self.assertFalse(hasattr(nodes, "__fspath__"))
        self.assertEqual(_names(copy.copy(nodes)), ["a", "b"])
        self.assertEqual(getattr(List([]), "__tag__", "raised"), "raised")
        self.assertEqual(list(copy.deepcopy(List([1, 2]))), [1, 2])
        self.assertEqual(list(np.array(List([1.0, 2.0]))), [1.0, 2.0])
        with self.assertRaises(AttributeError):
            List([]).__tag__
        empty = List([])
        # re-pinned (user decision 2026-10-16, the '_' rule B): no element has it,
        # so an assignment raises and stores nothing
        with self.assertRaisesRegex(AttributeError, r"no element has '__tag__'"):
            empty.__tag__ = 3
        self.assertNotIn("__tag__", vars(empty))

    def test_never_a_python_member_of_an_element(self):
        a, b = _tagged("a", "b")
        nodes = List([a, b])
        for name in ("_fn_set", "_attr_dict", "_mobject", "_objhandle"):
            with self.subTest(name=name):
                self.assertEqual(getattr(nodes, name, "raised"), "raised")
        self.assertEqual(getattr(List([a.t]), "_mplug", "raised"), "raised")

    def test_a_probe_costs_one_lookup(self):
        # one element read, by its node's hasAttribute (the node's own '_' check,
        # without the raise of DGNode.__getattr__)
        nodes = List(Node.create("transform", name=f"n{i}") for i in range(50))
        reads = []
        calls = []
        read  = list_module._private_plug
        real  = DGNode.__getattr__

        def reading(x, name):
            reads.append(name)
            return read(x, name)

        def counting(self, name):
            calls.append(name)
            return real(self, name)

        with mock.patch.object(list_module, "_private_plug", reading), \
                mock.patch.object(DGNode, "__getattr__", counting):
            self.assertIsNone(getattr(nodes, "__deepcopy__", None))
        self.assertEqual(reads, ["__deepcopy__"])
        self.assertEqual(calls, [])

    def test_a_probe_on_plugs_never_runs_the_plug_lookup(self):
        # a name the node lacks is refused by its hasAttribute, as on a node: no
        # child-name table, no node cast (LU FIX); one it has still resolves
        a, b = _tagged("a", "b")
        _pair("a")
        _pair("b")
        plugs = List([a.t, b.t, a.__pair__])
        calls = []
        real  = Plug.__getattr__

        def counting(self, name):
            calls.append(name)
            return real(self, name)

        with mock.patch.object(Plug, "__getattr__", counting):
            self.assertIsNone(getattr(plugs, "__deepcopy__", None))
            self.assertEqual(_names(np.asarray(List([a.t]), dtype=object)), ["a.translate"])
            self.assertEqual(_names(plugs.__tag__), ["a.__tag__", "b.__tag__", "a.__tag__"])
            self.assertEqual(_names(List([a.__pair__, b.__pair__]).__pairY__),
                             ["a.__pairY__", "b.__pairY__"])
        self.assertNotIn("__deepcopy__", calls)
        self.assertNotIn("__array__", calls)
        self.assertIn("__tag__", calls)


class TestDeleted(_Case):
    # re-pinned (robustness run, user decision 2026-10-16 "Deleted nodes in a List"):
    # a deleted or freed element is skipped wherever it is, rig.skip in its slot and one
    # warning naming it; the other elements broadcast (it raised "already deleted!")

    def _assert_skipped(self, items, gone):
        """``List(items).__tag__`` reads and sets the live elements only; `gone` is the
        index of the deleted one."""
        from rig import skip

        with mock.patch.object(cmds, "warning") as warn:
            got = List(items).__tag__
        self.assertEqual(warn.call_count, 1)
        self.assertRegex(warn.call_args[0][0], rf"^List element \[{gone}\] \S+ was deleted; skipped$")
        self.assertIs(got[gone], skip)
        self.assertEqual([str(x) for x in got if x is not skip], ["a.__tag__"])
        self.assertIsNone(getattr(List(items), "__deepcopy__", None))
        nodes = List(items)
        with mock.patch.object(cmds, "warning"):
            nodes.__tag__ = 5
        self.assertNotIn("__tag__", vars(nodes))
        self.assertEqual(cmds.getAttr("a.__tag__"), 5)
        with mock.patch.object(cmds, "warning"):
            nodes.__tag__ = 1
        self.assertEqual(cmds.getAttr("a.__tag__"), 1)

    def test_deleted_to_the_undo_queue(self):
        from rig import skip

        a, b = _tagged("a", "b")
        bt = b.t
        cmds.delete("b")
        for items in ([b, a], [bt, a.t]):
            with self.subTest(first=str(items[1])):
                self._assert_skipped(items, 0)
        for items in ([a, b], [a.t, bt]):
            with self.subTest(second=str(items[0])):
                self._assert_skipped(items, 1)
        # as the public name does
        with mock.patch.object(cmds, "warning"):
            tags = List([a, b]).tag
        self.assertEqual([str(tags[0]), tags[1]], ["a.tag", skip])
        while not cmds.objExists("b"):  # the sets above, then the delete
            cmds.undo()
        self.assertEqual(_names(List([a, b]).__tag__), ["a.__tag__", "b.__tag__"])

    def test_freed_by_a_new_scene(self):
        a, b = _tagged("a", "b")
        bt = b.t
        self.new_scene()
        a2, = _tagged("a")
        for items in ([b, a2], [bt, a2.t]):
            with self.subTest(first=type(items[0]).__name__):
                self._assert_skipped(items, 0)
        for items in ([a2, b], [a2.t, bt]):
            with self.subTest(second=type(items[1]).__name__):
                self._assert_skipped(items, 1)
