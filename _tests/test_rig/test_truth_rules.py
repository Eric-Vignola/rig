"""Python ``if`` / ``else`` and DSL comparisons (user decisions 7A-11A, 2026-10-17).

Python asks a truth value once and runs one branch; a DSL comparison is only known when Maya
evaluates it. So only the result of ``==`` / ``!=`` between two plugs has a truth value (are they
the same Maya plug?); everything else raises a TypeError naming the way to an answer. rig never
rewrites user code (no folding): the Script Editor and a sourced script behave the same.

* `TestAComparisonWithAValue` (7A): ``plug == value`` / ``plug != value`` in ``if``, ``not``,
  ``and`` / ``or``, ``assert``, ``while`` and a plain-list ``in`` raises, naming ``.get()``,
  ``rig.condition(...)``, ``.equals()`` and, for a Maya attr named like a method, the method.
* `TestTwoPlugsKeepTheirIdentity` (7A): a comparison of two plugs answers "the same plug?", so a
  plain list's ``in`` / ``index`` / ``remove``, a dict and a set work as before.
* `TestAPlugHasNoTruthValue` (8A): a plain plug, a typed Attribute (a `find_attr` result) and
  every other DSL result raise ("test for a missing plug with `is None`").
* `TestAListOfComparisons` (9A): a comparison broadcast is a List with no truth value; ``&``,
  ``|``, ``^``, ``~``, slices, index lists and copies keep it one; the message claims no Maya
  evaluation; the List rule's guard runs first.
* `TestTheNodeStays` (10A): a refused comparison leaves its node, as an ordering always did.
* `TestPlugEqualsNone` (11A): ``plug == None`` / ``!= None`` raise at once and build nothing.
* `TestRigTestsWithIsNone`: rig's own code tests a plug with ``is None`` (a lint over its modules).
* `TestRawMayaNeverAsksATruth` (in its own mayapy): Maya's boolean flags, OpenMaya's bool
  arguments and Python callbacks given these objects raise cleanly; the session keeps working.

Negative control: every test fails on 5ff3d86 (runs/features/final_impl/S2_truth).
"""

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from maya import cmds

import rig
from rig import List, Node, Plug, condition, force_nodes
from rig import functions as rf
from rig._internal import list as list_module
from rig._tests._base import MayaTestCase

_VALUE = r"the output of a comparison with a value"
_PLAIN = r"^a plug has no truth value \('%s'\): read its value with \.get\(\), test for a missing plug with `is None`"
_TESTS = (
    r"^the truth value of a List of comparisons is ambiguous, as a numpy array's is: use all\(\) "
    r"or any\(\) on the values \(all\(ctls\.tx\.get\(\) > 0\)\), or pick per element with rig\.condition\(\.\.\.\)$"
)


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)
        self.a, self.b, self.c = (Node.create("transform", name=n) for n in "abc")
        self.ctl = Node.create("transform", name="ctl")
        cmds.addAttr("ctl", longName="is_locked", attributeType="bool")  # value False
        self.ctls = List([self.a, self.b])
        self.a.tx << 5


class TestAComparisonWithAValue(_Case):
    def test_every_truth_context_raises(self):
        a = self.a
        for label, idiom in (
            ("if", lambda: "then" if a.tx == 99 else "else"),
            ("if !=", lambda: "then" if a.tx != 5 else "else"),
            ("not", lambda: not (a.tx == 5)),
            ("and", lambda: (a.tx == 5) and 1),
            ("or", lambda: (a.tx == 5) or 1),
            ("bool", lambda: bool(a.v == True)),  # noqa: E712 (the idiom under test)
            ("in a plain list", lambda: a.tx in [1, 2]),
            ("assert", lambda: exec("assert a.tx == 99", {"a": a})),
            ("while", lambda: exec("while a.tx == 5:\n    break", {"a": a})),
        ):
            with self.subTest(idiom=label):
                with self.assertRaisesRegex(TypeError, _VALUE):
                    idiom()

    def test_the_message_names_the_ways_to_an_answer(self):
        a = self.a
        with self.assertRaises(TypeError) as caught:
            bool(a.tx != 0)
        self.assertEqual(
            str(caught.exception),
            "the truth value of 'not_equal1.outColorR', the output of a comparison with a value (!= 0), "
            "is only known when Maya evaluates it: read the values first (a.translateX.get() != 0), "
            "pick with rig.condition(...), or compare plugs by identity with .equals()",
        )
        with self.assertRaisesRegex(TypeError, r"\(== \[0, 0, 0\]\).*\(all\(a\.translate\.get\(\) == \[0, 0, 0\]\)\)"):
            bool(a.t == [0, 0, 0])

    def test_a_maya_attr_named_like_a_method_names_the_method(self):
        ctl = self.ctl
        with self.assertRaises(TypeError) as caught:
            if ctl.is_locked == True:  # noqa: E712 (the user's case)
                pass
        self.assertEqual(
            str(caught.exception),
            "the truth value of 'equal1.output', the output of a comparison with a value (== True), "
            "is only known when Maya evaluates it: read the values first (ctl.is_locked.get() == True), "
            "pick with rig.condition(...), or compare plugs by identity with .equals(); 'ctl.is_locked' "
            "is the Maya attribute 'is_locked', the method is Transform.is_locked()",
        )
        # what answers: the value, and the method
        self.assertIs(bool(ctl.is_locked.get() == True), False)  # noqa: E712
        self.assertIs(ctl.is_locked(), False)

    def test_the_result_is_still_the_network(self):
        a, b = self.a, self.b
        test = a.tx == 5
        self.assertIsInstance(test, Plug)
        b.ty << condition(test, 1, 2)
        self.assertEqual(cmds.getAttr("b.ty"), 1.0)
        a.tx << 3
        self.assertEqual(cmds.getAttr("b.ty"), 2.0)
        b.tz << condition(a.tx != 3, 1, 2)
        self.assertEqual(cmds.getAttr("b.tz"), 2.0)


class TestTwoPlugsKeepTheirIdentity(_Case):
    def test_identity_truth(self):
        a, b, c = self.a, self.b, self.c
        self.assertIs(bool(a.tx == b.tx), False)
        self.assertIs(bool(a.tx != b.tx), True)
        self.assertIs(a.tx == a.translateX, True)   # one plug: folded, no node
        self.assertIs(a.tx != Node("a").tx, False)
        self.assertIs(a.tx in [b.tx, c.tx, a.tx], True)
        self.assertEqual([b.tx, c.tx, a.tx].index(a.tx), 2)
        held = [b.tx, Plug("a.tx")]
        held.remove(a.tx)
        self.assertEqual(len(held), 1)
        self.assertEqual({a.tx: "x"}[Node("a").translateX], "x")
        self.assertIn(Plug("a.tx"), {b.tx, a.tx})
        with force_nodes():
            same = a.tx == Node("a").tx
        self.assertIsInstance(same, Plug)
        self.assertIs(bool(same), True)
        self.assertIs(a.tx.equals(b.tx), False)


class TestAPlugHasNoTruthValue(_Case):
    def test_a_plain_plug(self):
        a = self.a
        for label, idiom in (
            ("if", lambda: "then" if a.ty else "else"),
            ("or", lambda: getattr(a, "tx", None) or 0),
            ("any", lambda: any([a.tx])),
            ("filter", lambda: list(filter(None, [a.tx]))),
            ("not", lambda: not a.tx),
        ):
            with self.subTest(idiom=label):
                with self.assertRaisesRegex(TypeError, _PLAIN % r"a\.translate[XY]"):
                    idiom()

    def test_a_find_attr_result_one_rule(self):
        a = self.a
        with self.assertRaisesRegex(TypeError, _PLAIN % r"a\.translateX"):
            bool(a.find_attr("tx"))
        with self.assertRaisesRegex(TypeError, _PLAIN % r"a\.speed"):
            cmds.addAttr("a", longName="speed")
            if a.find_attr("speed", quiet=True):
                pass
        self.assertIsNone(a.find_attr("nope", quiet=True))
        self.assertIsNotNone(a.find_attr("speed", quiet=True))

    def test_a_maya_attr_named_like_a_method_names_the_method(self):
        with self.assertRaises(TypeError) as caught:
            if self.ctl.is_locked:
                pass
        self.assertEqual(
            str(caught.exception),
            "a plug has no truth value ('ctl.is_locked'): read its value with .get(), test for a missing "
            "plug with `is None`; 'ctl.is_locked' is the Maya attribute 'is_locked', the method is "
            "Transform.is_locked()",
        )

    def test_every_dsl_result(self):
        a, b = self.a, self.b
        for label, make, name in (
            ("rf.equal", lambda: rf.equal(a.tx, 99), r"equal1\.output"),
            ("&", lambda: a.tx & 0, r"logical_and1\.output"),
            ("condition", lambda: condition(a.tx > 0, 1, 2), r"condition1\.outColorR"),
            ("+", lambda: a.tx + b.tx, r".+"),
        ):
            with self.subTest(result=label):
                result = make()
                with self.assertRaisesRegex(TypeError, _PLAIN % name):
                    bool(result)

    def test_the_matrix_shorthand_reads_the_rotate_order_with_is_none(self):
        # rig's own `or` on a plug (shorthand: the destination's rotate order) is an `is None` test
        a, b = self.a, self.b
        b.rotateOrder << 2
        b.t << (1, 2, 3)
        a << b.worldMatrix
        self.assertEqual(cmds.getAttr("a.ty"), 2.0)


class TestAListOfComparisons(_Case):
    def test_each_comparison_and_logical_operator(self):
        ctls, a = self.ctls, self.a
        for label, make in (
            ("> value", lambda: ctls.tx > 0),
            ("== value", lambda: ctls.tx == 99),
            ("!= List", lambda: ctls.tx != ctls.ty),
            ("List == List of nodes", lambda: List([a]) == List([a])),
            ("values", lambda: List([1.0, 2.0]) == [5.0, 6.0]),
            ("&", lambda: (ctls.tx > 0) & (ctls.ty > 0)),
            ("plain & tests", lambda: ctls.tx & (ctls.ty > 0)),
            ("|", lambda: (ctls.tx > 0) | (ctls.ty > 0)),
            ("^", lambda: (ctls.tx > 0) ^ (ctls.ty > 0)),
            ("~", lambda: ~(ctls.tx > 0)),
            ("slice", lambda: (ctls.tx > 0)[:1]),
            ("index list", lambda: (ctls.tx > 0)[[0]]),
            ("copy", lambda: (ctls.tx > 0).copy()),
            ("copy.copy", lambda: copy.copy(ctls.tx > 0)),
        ):
            with self.subTest(make=label):
                tests = make()
                self.assertIs(type(tests), list_module._Tests)
                with self.assertRaisesRegex(TypeError, _TESTS):
                    bool(tests)

    def test_python_idioms_on_it(self):
        ctls = self.ctls
        self.assertNotIn("Maya", _TESTS)
        with self.assertRaisesRegex(TypeError, _TESTS):
            if ctls.tx > 0:
                pass
        with self.assertRaisesRegex(TypeError, _TESTS):
            [1, ctls].index(ctls)  # 1 == ctls is a List of comparisons
        # the values answer, and so does a List of Python bools
        self.assertIs(all(ctls.tx.get() > 0), False)
        self.assertIs(any(ctls.tx.get() > 0), True)
        self.assertIs(all(List([1.0, 2.0]) == [1.0, 2.0]), True)
        # len, indexing and iteration answer; arithmetic and List(...) give a plain List
        tests = ctls.tx > 0
        self.assertEqual(len(tests), 2)
        self.assertIsInstance(tests[0], Plug)
        self.assertEqual(len(list(tests)), 2)
        self.assertIs(type(tests + 1), List)
        self.assertIs(type(List(tests)), List)

    def test_the_list_rule_first(self):
        tests = self.ctls.tx > 0
        cmds.delete("b")
        with self.assertRaisesRegex(RuntimeError, r"this List can no longer be used; build a new List$"):
            bool(tests)


class TestTheNodeStays(_Case):
    def test_a_refused_comparison_leaves_its_node(self):
        a = self.a
        with self.assertRaisesRegex(TypeError, _VALUE):
            if a.tx == 5:
                pass
        self.assertEqual(cmds.ls(type="equal"), ["equal1"])
        # undone as any comparison is (its createNode, then its connects: the
        # README's undo model), with nothing deleted while raising
        cmds.undo()
        cmds.undo()
        self.assertEqual(cmds.ls(type="equal"), [])
        self.assertEqual(cmds.getAttr("a.tx"), 5.0)

    def test_a_held_result_and_a_user_container_survive(self):
        a, b = self.a, self.b
        from rig import container
        with container("conditionRig"):
            with self.assertRaises(TypeError):
                if a.t == [0, 0, 0]:
                    pass
        self.assertTrue(cmds.objExists("conditionRig"))
        held = a.tx == 5
        with self.assertRaises(TypeError):
            if a.tx == 5:
                pass
        b.ty << condition(held, 1, 2)
        self.assertEqual(cmds.getAttr("b.ty"), 1.0)


class TestPlugEqualsNone(_Case):
    def test_raises_at_once_and_builds_nothing(self):
        a, ctls = self.a, self.ctls
        before = set(cmds.ls())
        for label, idiom, test in (
            ("== None", lambda: a.tx == None, "is None"),  # noqa: E711 (the idiom under test)
            ("None ==", lambda: None == a.tx, "is None"),  # noqa: E711
            ("!= None", lambda: a.tx != None, "is not None"),  # noqa: E711
            ("None in", lambda: None in [a.tx], "is None"),
            ("List == None", lambda: ctls.tx == None, "is None"),  # noqa: E711
        ):
            with self.subTest(idiom=label):
                with self.assertRaisesRegex(TypeError, r"^a\.translateX [!=]= None: a plug is never None: test it with `%s`$" % test):
                    idiom()
        self.assertEqual(set(cmds.ls()), before)
        self.assertIs(a.tx is None, False)
        # undo still works: the last statement before is undone
        cmds.undo()
        self.assertEqual(cmds.getAttr("a.tx"), 0.0)


# rig's modules test a plug with `is None` (8A): no truth test of what these give
_GIVES_A_PLUG = {
    "find_attr", "find_alias", "_find_attr", "_raw_find_attr", "_find_attr_fresh", "_safe_attr",
    "_connected_attrs", "rename_attr",
}


def _plug_truth_tests(path):
    """(line, text) of each truth test, in the module at `path`, of a call that gives a plug
    (`_GIVES_A_PLUG`) or of a name assigned from one in the same function."""
    src  = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out  = []

    def gives(call):
        f = call.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name == "_connected_attrs":
            return any(k.arg == "first_only" for k in call.keywords)
        return name in _GIVES_A_PLUG

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and gives(node.value):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr.endswith("_attr") for t in node.targets
            ) and isinstance(node.value, (ast.Call, ast.Subscript)):
                names.update(ast.unparse(t) for t in node.targets if isinstance(t, ast.Attribute))

        def check(e):
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
                e = e.operand
            text = ast.unparse(e)
            if (isinstance(e, ast.Call) and gives(e)) or text in names:
                out.append((e.lineno, text))

        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                check(node.test)
            elif isinstance(node, ast.BoolOp):
                for value in node.values:
                    check(value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                check(node)
    return out


class TestRigTestsWithIsNone(MayaTestCase):
    def test_no_truth_test_of_a_plug_in_rig(self):
        root  = os.path.dirname(os.path.abspath(rig.__file__))
        found = []
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("_tests", "__pycache__")]
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(folder, name)
                    found += [
                        f"{os.path.relpath(path, root)}:{line} {text}" for line, text in _plug_truth_tests(path)
                    ]
        self.assertEqual(found, [])


def _in_own_mayapy(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
    env["MAYA_DISABLE_CER"] = "1"
    cwd = tempfile.mkdtemp(prefix="rig_truth_")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, timeout=900,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")


_RAW = r"""
import json, os, sys
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    import maya.api.OpenMaya as om
    import rig
    from rig import List, Node
    out = {"file": rig.__file__}
    def outcome(call):
        try:
            call()
            return "returned"
        except Exception as error:  # noqa: BLE001
            return "%s: %s" % (type(error).__name__, str(error).splitlines()[0][:200])
    cmds.undoInfo(state=True, infinity=True)
    a = Node.create("transform", name="a"); b = Node.create("transform", name="b")
    ctl = Node.create("transform", name="ctl"); cmds.addAttr("ctl", ln="is_locked", at="bool")
    objects = {
        "plug": lambda: a.tx, "attr": lambda: a.find_attr("tx"), "method plug": lambda: ctl.is_locked,
        "== value": lambda: a.tx == 5, "ordering": lambda: a.tx > 0,
        "List of comparisons": lambda: List([a, b]).tx > 0,
    }
    fn = om.MFnDependencyNode(om.MSelectionList().add("a").getDependNode(0))
    calls = {
        "ls -selection": lambda x: cmds.ls(selection=x),
        "select -clear": lambda x: cmds.select(clear=x),
        "xform -query": lambda x: cmds.xform("a", query=x, translation=True),
        "parent -world": lambda x: cmds.parent("a", world=x),
        "setAttr -lock": lambda x: cmds.setAttr("a.tz", lock=x),
        "MPlug.isLocked": lambda x: setattr(fn.findPlug("tz", False), "isLocked", x),
        "findPlug": lambda x: fn.findPlug("tx", x),
        "node-added callback": lambda x: (om.MDGMessage.addNodeAddedCallback(lambda *_: bool(x), "dependNode"),
                                          cmds.createNode("network")),
    }
    for kind, make in objects.items():
        for label, call in calls.items():
            out[kind + " / " + label] = outcome(lambda: call(make()))
    # the session still works
    cmds.createNode("transform", name="after")
    Node("after").ty << 2
    cmds.undo(); cmds.redo()
    out["after"] = cmds.getAttr("after.ty")
    print("RESULT " + json.dumps(out))
    sys.stdout.flush()
finally:
    try:
        maya.standalone.uninitialize()
    except BaseException:
        pass
    sys.stdout.flush()
    os._exit(0)
"""


class TestRawMayaNeverAsksATruth(MayaTestCase):
    def test_boolean_flags_api_arguments_and_callbacks(self):
        code, out, err = _in_own_mayapy(_RAW)
        lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
        result = json.loads(lines[0][len("RESULT "):])
        self.assertEqual(os.path.normcase(result.pop("file")), os.path.normcase(rig.__file__))
        self.assertEqual(result.pop("after"), 2.0)
        self.assertEqual(len(result), 48)
        for case, outcome in sorted(result.items()):
            with self.subTest(case=case):
                if case.endswith("callback"):
                    # a callback's TypeError is reported by Maya; the call itself returns
                    self.assertEqual(outcome, "returned")
                else:
                    # Maya and OpenMaya refuse the type (boolean expected) before any truth test
                    self.assertRegex(outcome, r"^TypeError: ")


class TestTheUndoStateIsPutBack(MayaTestCase):
    """S2_finish: a case that turns undo on puts back the state it found. These
    cases once left undo off, and with undo off Maya frees a deleted dynamic
    attribute, which failed test_types' readded-attr test when it ran after them."""

    def test_every_undo_case_puts_the_undo_state_back(self):
        from rig._tests.test_rig import test_list_rule, test_robustness

        cases = (
            TestTwoPlugsKeepTheirIdentity("test_identity_truth"),
            test_list_rule.TestTheSceneEpoch(_first_test(test_list_rule.TestTheSceneEpoch)),
            test_robustness.TestDeletedNodeName(_first_test(test_robustness.TestDeletedNodeName)),
        )
        found = cmds.undoInfo(query=True, state=True)
        self.addCleanup(cmds.undoInfo, state=found)
        for case in cases:
            for state in (True, False):
                with self.subTest(case=case.id(), state=state):
                    cmds.undoInfo(state=state)
                    result = unittest.TestResult()
                    case.run(result)
                    self.assertEqual(result.failures + result.errors, [])
                    self.assertIs(cmds.undoInfo(query=True, state=True), state)


def _first_test(case_class):
    return unittest.defaultTestLoader.getTestCaseNames(case_class)[0]
