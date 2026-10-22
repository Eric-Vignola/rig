"""The List rule (user decisions 1A-4A and 6C, 2026-10-17): a List holds what is in the scene.

* `TestEveryRigUseRaises`: while an element is deleted (to the undo queue) or freed, every rig
  operation on the List raises ONE plain RuntimeError naming the first such element (4A); its
  construction from such an element raises with "a List holds what is in the scene".
* `TestThePythonProtocolAnswers`: Python's len(), integer indexing and iteration never refuse it
  (Maya's normal and String[...] flag readers go through them; a refusal crashes Maya), and
  print / repr / str / an f-string of it never raise (3A).
* `TestStateBased`: an undo that brings the element back makes the List usable again; a redo
  raises again; flushUndo and a new scene give "was deleted and freed" (1A).
* `TestCopies`: List(lst), lst[:], lst.copy(), copy.copy, pickle raise (2A); list(lst) answers.
* `TestNestedAndKinds`: a nested List names the path ([1][0]); plugs, Components, tokens, bound
  methods and bare predicates count through their node; a List of values is never invalid.
* `TestNumpy`: np.array(lst) is a checked use; numpy 2's ``copy=False`` is a ValueError.
* `TestKeywordArguments`: rig's wrappers check a List given as a keyword argument / flag value
  (``rc.select(add=lst)``, ``skin.remove_influence_objects(lst)``, ``cmds_edit``).
* `TestThePrivateNameReader`: str() of a deleted / freed node, plug, Components and token is the
  marked name for everyone; rig reads names through `_name_of`, which raises "already deleted!";
  no caller is inspected anywhere (``_called_from_rig`` / ``_user_view`` are gone).
* `TestRawMayaNeverCrashes` (each in its own mayapy): raw maya.cmds given an invalid List as a
  positional argument, as the value of the String[...] / Int[...] flags and of boolean flags, a
  plain list holding a deleted node (Maya's TypeError, nothing deleted), the removal window
  (a user's node-removed callback using the List), a pending Python error inside Maya's parser,
  and rig reloaded or re-imported with Lists of the old copy.
* `TestTheSceneEpoch`: the epoch and the doomed handles live on ``sys`` (reload-safe); every
  notification moves the epoch first; a handle that cannot be recorded makes no List trusted
  until the next undo / redo / scene event.

Negative control: every test fails on f545846 (the skip rule): runs/features/final_impl/S1_list.
"""

import copy
import importlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
from maya import cmds

import rig
from rig import Layer, List, Node, Plug, Tag
from rig._internal import journal
from rig._internal import list as list_module
from rig.bridges import commands as rc
from rig.nodetypes import _base
from rig.nodetypes._base import _name_of
from rig.nodetypes import Blinn
from rig.spec import Float
from rig._tests._base import MayaTestCase

_USED = "this List can no longer be used; build a new List"


def _used(index, what="Transform", name="c1", kind="was deleted", tail=_USED):
    """The regex of the List error naming element `index`."""
    return rf'^List element {index} {what}\("{name}", deleted\) {kind}: {tail}$'


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)
        self.c0, self.c1, self.c2 = (Node.create("transform", name=f"c{i}") for i in range(3))
        self.ctls = List([self.c0, self.c1, self.c2])
        self.ctls.tx << [1, 2, 3]


class TestEveryRigUseRaises(_Case):
    def test_each_use_raises_one_runtime_error(self):
        ctls, c0 = self.ctls, self.c0
        cmds.delete("c1")
        layer = Layer.define("L")
        uses = {
            "broadcast": lambda: ctls.tx, "item name": lambda: ctls["tx"], "assign": lambda: setattr(ctls, "tx", 1),
            "<<": lambda: ctls.tx << 1, ">>": lambda: ctls >> None, "get": lambda: ctls.get(),
            "lock": lambda: ctls.lock(), "unlock": lambda: ctls.unlock(), "<< spec": lambda: ctls << Float("k"),
            "<< layer": lambda: ctls << layer, "in layer": lambda: ctls in layer, ">> Layer()": lambda: ctls >> Layer(),
            "in": lambda: c0 in ctls, "index": lambda: ctls.index(c0), "count": lambda: ctls.count(c0),
            "remove": lambda: ctls.remove(c0), "+": lambda: ctls + 1, "radd": lambda: 1 + ctls, "==": lambda: ctls == ctls,
            "neg": lambda: -ctls, "truth": lambda: bool(ctls), "hash": lambda: hash(ctls),
            "append": lambda: ctls.append(c0), "extend": lambda: ctls.extend([c0]), "insert": lambda: ctls.insert(0, c0),
            "+=": lambda: ctls.__iadd__([c0]), "setitem": lambda: ctls.__setitem__(0, c0), "slice set": lambda: ctls.__setitem__(slice(0, 1), [c0]),
            "del": lambda: ctls.__delitem__(0), "pop": lambda: ctls.pop(), "sort": lambda: ctls.sort(key=id),
            "reverse": lambda: ctls.reverse(), "clear": lambda: ctls.clear(), "*=": lambda: ctls.__imul__(2),
            "plug << List": lambda: c0.ty << ctls.tx, "plug + List": lambda: c0.ty + ctls,
            "condition": lambda: rig.condition(ctls, 1, 2), "AnimCurve.of": lambda: rig.AnimCurve.of(ctls),
            "delete(List)": lambda: c0.delete(ctls), "find_all": lambda: rig.nodetypes.Transform.find_all(ctls),
            "rc.select": lambda: rc.select(ctls), "Tag": lambda: ctls << Tag("cap"),
        }
        for label, use in uses.items():
            with self.subTest(use=label):
                with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")) as ctx:
                    use()
                self.assertIs(type(ctx.exception), RuntimeError)  # 4A: a plain RuntimeError
        # nothing was changed or built
        self.assertEqual(len(ctls), 3)
        self.assertEqual([cmds.getAttr(f"c{i}.tx") for i in (0, 2)], [1, 3])
        self.assertEqual(cmds.ls(type="condition"), [])

    def test_a_list_holds_what_is_in_the_scene(self):
        c0, c1 = self.c0, self.c1
        plug = c1.tx
        cmds.delete("c1")
        tail = "a List holds what is in the scene"
        for label, build, index in (
            ("List", lambda: List([c0, c1]), r"\[1\]"),
            ("append", lambda: List([c0]).append(c1), r"\[1\]"),
            ("extend", lambda: List([c0]).extend([c1]), r"\[1\]"),
            ("insert", lambda: List([c0]).insert(0, c1), r"\[0\]"),
            ("+=", lambda: List([c0]).__iadd__([c1]), r"\[1\]"),
            ("setitem", lambda: List([c0]).__setitem__(0, c1), r"\[0\]"),
        ):
            with self.subTest(build=label):
                with self.assertRaisesRegex(RuntimeError, _used(index, tail=tail)):
                    build()
        with self.assertRaisesRegex(RuntimeError, _used(r"\[0\]", "Plug", "c1.translateX",
                                                        "belongs to a node that was deleted", tail)):
            List([plug])


class TestThePythonProtocolAnswers(_Case):
    def test_len_indexing_and_iteration(self):
        ctls = self.ctls
        cmds.delete("c1")
        self.assertEqual(len(ctls), 3)
        self.assertIs(ctls[1], list.__getitem__(ctls, 1))
        self.assertIs(ctls[np.int64(2)], self.c2)
        self.assertEqual([x for x in ctls], list(ctls))
        self.assertEqual(len(tuple(ctls)), 3)
        self.assertEqual(len(list(reversed(ctls))), 3)
        a, b, c = ctls
        self.assertIs(b, list.__getitem__(ctls, 1))

    def test_print_repr_str_never_raise(self):
        ctls = self.ctls
        cmds.delete("c1")
        text = 'List([Transform("c0"), Transform("c1", deleted), Transform("c2")])'
        self.assertEqual((repr(ctls), str(ctls), f"{ctls}"), (text, text, text))
        self.assertEqual(str(self.c1), "c1 (deleted)")
        self.assertEqual(f"{self.c1}", "c1 (deleted)")
        self.assertEqual(repr(self.c1), 'Transform("c1", deleted)')


class TestStateBased(_Case):
    def test_undo_redo_flush(self):
        ctls = self.ctls
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
            ctls.tx.get()
        cmds.undo()
        self.assertEqual(list(ctls.tx.get()), [1, 2, 3])
        cmds.redo()
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
            ctls.tx.get()
        cmds.flushUndo()
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]", kind="was deleted and freed")):
            ctls.tx.get()

    def test_new_scene_frees(self):
        ctls = self.ctls
        self.new_scene()
        with self.assertRaisesRegex(RuntimeError, _used(r"\[0\]", name="c0", kind="was deleted and freed")):
            ctls << 1
        self.assertEqual(len(ctls), 3)

    def test_a_valid_list_costs_one_comparison(self):
        # the epoch: a List found valid keeps it; a removal anywhere moves it once
        ctls = self.ctls
        bool(ctls)
        self.assertEqual(ctls._epoch, journal.EPOCH[0])
        with mock.patch.object(list_module, "_first_gone", wraps=list_module._first_gone) as scan:
            for _ in range(5):
                ctls.tx
            self.assertEqual(scan.call_count, 0)
            cmds.delete(cmds.createNode("transform"))
            ctls.tx
            ctls.tx
            self.assertEqual(scan.call_count, 1)


class TestCopies(_Case):
    def test_a_copy_is_a_use(self):
        ctls = self.ctls
        cmds.delete("c1")
        for label, call in (("List(lst)", lambda: List(ctls)), ("slice", lambda: ctls[:]),
                            ("list of indices", lambda: ctls[[0, 2]]), ("copy()", lambda: ctls.copy()),
                            ("copy.copy", lambda: copy.copy(ctls)), ("deepcopy", lambda: copy.deepcopy(ctls)),
                            ("pickle", lambda: pickle.dumps(ctls))):
            with self.subTest(copy=label):
                with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
                    call()
        self.assertEqual(len(list(ctls)), 3)  # Python's own list() iterates


class TestNestedAndKinds(_Case):
    def test_nested_path(self):
        nested = List([List([self.c0]), List([self.c2, self.c1])])
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]\[1\]")):
            nested.get()
        with self.assertRaisesRegex(RuntimeError, _used(r"\[0\]\[1\]", tail="a List holds what is in the scene")):
            List([nested[1]])

    def test_elements_count_through_their_node(self):
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        red = Blinn.define("red")
        layer = Layer.define("lay")
        kinds = {
            "plug": (List([cube.tx]), 'Plug\\("cube.translateX", deleted\\) belongs to a node that was deleted'),
            "faces": (List([cube.f[0]]), 'Components\\("cubeShape.f\\[0\\]", deleted\\) belongs to a node that was deleted'),
            "bound method": (List([cube]).get_name, "<bound method .* belongs to a node that was deleted"),
            "bare predicate": (List([cube]).is_locked, "<predicate .* belongs to a node that was deleted"),
        }
        tokens = List([-red, -layer])
        cmds.delete("cube", "red", "lay")
        for label, (lst, text) in kinds.items():
            with self.subTest(kind=label):
                with self.assertRaisesRegex(RuntimeError, rf"^List element \[0\] {text}: {_USED}$"):
                    bool(lst)
        with self.assertRaisesRegex(RuntimeError, r'^List element \[0\] -Blinn\("red", deleted\) belongs'):
            bool(tokens)
        values = List([0.0, (1, 2, 3), None, 2])
        self.assertTrue(values)  # a List of values is never invalid
        self.assertEqual(len(values), 4)


class TestNumpy(_Case):
    def test_an_array_is_a_use(self):
        ctls = self.ctls
        self.assertEqual(np.array(ctls.tx.get()).tolist(), [1, 2, 3])
        self.assertEqual(np.asarray(List([1.0, 2.0])).tolist(), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "without a copy"):
            List([1.0]).__array__(copy=False)
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
            np.array(ctls)
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
            np.asarray(ctls)


class TestKeywordArguments(_Case):
    def test_wrappers_check_keyword_lists(self):
        ctls = self.ctls
        cmds.sets(name="st", empty=True)
        cmds.delete("c1")
        for label, call in (
            ("rc.select(add=)", lambda: rc.select("c0", add=ctls)),
            ("rc.sets(forceElement=)", lambda: rc.sets(ctls, forceElement="st")),
            ("rc.container(addNode=)", lambda: rc.container(addNode=ctls)),
            ("cmds_edit flag", lambda: journal.cmds_edit.select("c0", deselect=ctls)),
            ("plain list holding a List", lambda: rc.select([ctls])),
            ("find_all(kw)", lambda: rig.nodetypes.Transform.find_all(ctls, long=True)),
        ):
            with self.subTest(call=label):
                with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
                    call()
        self.assertEqual(cmds.ls(type="container"), [])

    def test_skin_influences(self):
        # the review's case: remove_influence_objects(L) was a silent no-op
        base = cmds.polyCube(name="base", ch=False)[0]
        cmds.select(clear=True)
        j1 = cmds.joint(name="j1")
        cmds.select(clear=True)
        j2 = cmds.joint(name="j2")
        skin = Node(cmds.skinCluster(j1, j2, base, toSelectedBones=True)[0])
        joints = List([Node(j1), Node(j2)])
        cmds.select(clear=True)
        j3 = cmds.joint(name="j3")
        held = List([Node(j3), Node(j2)])
        cmds.delete("j3")
        with self.assertRaisesRegex(RuntimeError, _used(r"\[0\]", "Joint", "j3")):
            skin.remove_influence_objects(held)
        with self.assertRaisesRegex(RuntimeError, _used(r"\[0\]", "Joint", "j3")):
            skin.add_influence_objects(held)
        self.assertEqual(sorted(str(x) for x in skin.get_influence_objects()), ["j1", "j2"])
        self.assertEqual(len(joints), 2)


class TestThePrivateNameReader(_Case):
    def test_str_is_marked_for_everyone(self):
        cube = Node(cmds.polyCube(name="cube", ch=False)[0])
        faces = cube.f[:2]
        red = Blinn.define("red")
        token = -red
        plug = self.c1.tx
        cmds.delete("c1", "cube", "red")
        marked = {
            "node": (self.c1, "c1 (deleted)"),
            "plug": (plug, "c1.translateX (deleted)"),
            "faces": (faces, 'Components("cubeShape.f[0:1]", deleted)'),
            "token": (token, "red (deleted)"),
        }
        for label, (obj, text) in marked.items():
            with self.subTest(kind=label):
                self.assertEqual(str(obj), text)
                self.assertEqual(f"{obj}", text)
                with self.assertRaisesRegex(RuntimeError, "already deleted!$"):
                    _name_of(obj)
        cmds.undo()
        self.assertEqual((_name_of(self.c1), _name_of(plug), _name_of("x"), _name_of(3)),
                         ("c1", "c1.translateX", "x", "3"))
        self.assertEqual(_name_of(token), "red")

    def test_freed(self):
        node = self.c1
        plug = node.tx
        self.new_scene()
        self.assertEqual((str(node), str(plug)), ("c1 (deleted)", "c1.translateX (deleted)"))
        with self.assertRaisesRegex(RuntimeError, r"^Transform node \(freed by a new scene"):
            _name_of(node)

    def test_no_caller_is_inspected(self):
        # the marked name is the same from rig's own code: no frame is read
        for name in ("_called_from_rig", "_user_view", "_USER_VIEW", "_RIG_FRAMES"):
            self.assertFalse(hasattr(_base, name), name)
        self.assertFalse(hasattr(journal, "forward"))
        cmds.delete("c1")
        # str() called from a frame of rig's own package reads the marked name too
        # (the robustness run's _called_from_rig raised there)
        rig_frame = {"__name__": "rig._internal.list", "node": self.c1}
        exec(compile("text = str(node)", "<rig frame>", "exec"), rig_frame)
        self.assertEqual(rig_frame["text"], "c1 (deleted)")
        exec(compile("text = repr(lst)", "<rig frame>", "exec"), dict(rig_frame, lst=self.ctls))
        # rig's own code names a deleted node through the reader: its error, not a marked name
        with self.assertRaisesRegex(RuntimeError, r"^c1 already deleted!$"):
            self.c1.rename("other")

    def test_raw_cmds_given_the_marked_name(self):
        c1, c2 = self.c1, self.c2
        cmds.delete("c1")
        # a plain list: Maya reads the marked name and raises, deleting nothing (= 5B)
        with self.assertRaisesRegex(TypeError, "is invalid"):
            cmds.delete([c1, c2])
        self.assertTrue(cmds.objExists("c2"))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            cmds.select([c2, c1])
        with self.assertRaisesRegex(RuntimeError, r"^c1 already deleted!$"):
            rc.delete([c1, c2])
        self.assertTrue(cmds.objExists("c2"))


class TestTheSceneEpoch(_Case):
    def test_kept_on_sys(self):
        self.assertIs(journal.EPOCH, sys.__dict__["_rig_scene_epoch"])
        self.assertIs(journal.DOOMED, sys.__dict__["_rig_scene_doomed"])
        self.assertIs(list_module._EPOCH, journal.EPOCH)

    def test_the_epoch_moves_first(self):
        # a notification that fails after its first line still moved the epoch
        before = journal.EPOCH[0]
        with mock.patch.object(journal, "_S", None):  # every later line of the callbacks fails
            for callback, args in ((journal._discard, ()), (journal._undone_or_redone, ()),
                                   (journal._settle, ()), (journal._node_removed, (None,))):
                try:
                    callback(*args)
                except Exception:  # noqa: BLE001
                    pass
        self.assertGreaterEqual(journal.EPOCH[0], before + 4)
        del journal.DOOMED[:]

    def test_an_unrecorded_handle_trusts_no_list(self):
        ctls = self.ctls
        journal._node_removed(object())  # MObjectHandle(object()) raises: _UNTRUSTED
        self.assertIs(journal.DOOMED[-1], journal._UNTRUSTED)
        self.assertFalse(list_module._settled())
        bool(ctls)
        self.assertNotEqual(ctls._epoch, journal.EPOCH[0])  # checked, not kept
        journal._undone_or_redone()  # the next undo / redo / scene event
        self.assertTrue(list_module._settled())
        bool(ctls)
        self.assertEqual(ctls._epoch, journal.EPOCH[0])

    def test_the_removal_window(self):
        # a user's node-removed callback uses the List while c1 is being removed:
        # no epoch is kept then, so the List raises once c1 is gone
        from maya.api import OpenMaya as om

        ctls = self.ctls
        seen = []
        cid = om.MDGMessage.addNodeRemovedCallback(lambda *_: seen.append(bool(ctls)), "dependNode")
        try:
            cmds.delete("c1")
        finally:
            om.MMessage.removeCallback(cid)
        self.assertIn(True, seen)
        with self.assertRaisesRegex(RuntimeError, _used(r"\[1\]")):
            ctls.tx.get()


# --------------------------------------------------------------------- #
#  Raw Maya, each case in its own mayapy (a crash fails the test, not the run)
# --------------------------------------------------------------------- #


def _in_own_mayapy(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
    env["MAYA_DISABLE_CER"] = "1"
    cwd = tempfile.mkdtemp(prefix="rig_list_rule_")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, timeout=900,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return (proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace"))


_HEAD = r"""
import importlib, json, os, sys
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    import rig
    from rig import List, Node
    out = {"file": rig.__file__}
    def outcome(call):
        try:
            value = call()
            return "returned " + type(value).__name__
        except Exception as error:  # noqa: BLE001
            return "%s: %s" % (type(error).__name__, str(error).splitlines()[0][:200])
    cmds.undoInfo(state=True, infinity=True)
    cmds.createNode("transform", name="live")
    cmds.container(name="cont1")
    def held():
        name = cmds.createNode("transform", name="gone")
        g = Node(name)
        lst = List([g, Node("live")]); bool(lst)
        cmds.delete(name)
        return lst, g
"""
_TAIL = r"""
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

_RAW = _HEAD + r"""
    flags = {
        "container -addNode": lambda x: cmds.container("cont1", edit=True, addNode=x),
        "container -removeNode": lambda x: cmds.container("cont1", edit=True, removeNode=x),
        "container create -addNode": lambda x: cmds.container(addNode=x),
        "container -findContainer": lambda x: cmds.container(query=True, findContainer=x),
        "commandEcho -filter": lambda x: cmds.commandEcho(filter=x),
        "commandEcho -addFilter": lambda x: cmds.commandEcho(addFilter=x),
        "bakeDeformer -influences": lambda x: cmds.bakeDeformer(influences=x),
        "select": lambda x: cmds.select(x), "delete": lambda x: cmds.delete(x), "ls": lambda x: cmds.ls(x),
        "xform": lambda x: cmds.xform(x, translation=(1, 0, 0)), "ls -selection": lambda x: cmds.ls(selection=x),
        "select -clear": lambda x: cmds.select(clear=x), "xform -query": lambda x: cmds.xform("live", query=x, t=True),
    }
    for form in ("List", "plain"):
        for label, call in flags.items():
            lst, g = held()
            x = lst if form == "List" else [g, Node("live")]
            sys.stdout.write("CASE %s %s\n" % (form, label)); sys.stdout.flush()
            out["%s %s" % (form, label)] = outcome(lambda: call(x))
            out["%s %s live" % (form, label)] = cmds.objExists("live")
            if not cmds.objExists("live"):
                cmds.createNode("transform", name="live")
"""


_RELOAD = _HEAD + r"""
    from rig._internal import journal
    lst, g = held()
    importlib.reload(journal)
    out["reload journal: rig use"] = outcome(lambda: lst.tx)
    out["reload journal: select"] = outcome(lambda: cmds.select(lst))
    cmds.undo()
    out["reload journal: undo"] = outcome(lambda: lst.tx)
    lst2, g2 = held()
    for name in [m for m in sys.modules if m == "rig" or m.startswith("rig.")]:
        del sys.modules[name]
    import rig as rig2
    rig2.List([rig2.Node("live")]).tx
    out["reimport: rig use"] = outcome(lambda: lst2.tx)
    out["reimport: select"] = outcome(lambda: cmds.select(lst2))
    out["reimport: new List"] = outcome(lambda: rig2.List([rig2.Node("live")]).tx)
"""


class TestRawMayaNeverCrashes(MayaTestCase):
    def test_flags_positional_and_plain_lists(self):
        code, out, err = _in_own_mayapy(_RAW + _TAIL)
        lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
        result = json.loads(lines[0][len("RESULT "):])
        result.pop("file")
        for case, outcome in sorted(result.items()):
            with self.subTest(case=case):
                if case.endswith(" live"):
                    self.assertIs(outcome, True, case)  # nothing was deleted
                elif "commandEcho" in case:
                    # a filter of command names: Maya keeps the marked names as text
                    self.assertRegex(outcome, r"^(returned NoneType|TypeError|RuntimeError|ValueError)")
                elif case.startswith("List") and case.split(" ", 1)[1] in ("ls -selection", "select -clear", "xform -query"):
                    # a boolean flag asks the List's truth: the List error, raised in Maya's reader
                    self.assertRegex(outcome, r"^(TypeError|RuntimeError|ValueError): ")
                else:
                    # Maya's own clean error (a marked name names nothing)
                    self.assertRegex(outcome, r"^(TypeError|RuntimeError|ValueError): ", case)

    def test_reload_and_reimport(self):
        code, out, err = _in_own_mayapy(_RELOAD + _TAIL)
        lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
        result = json.loads(lines[0][len("RESULT "):])
        self.assertRegex(result["reload journal: rig use"], f"^RuntimeError: .*{_USED}$")
        self.assertRegex(result["reimport: rig use"], f"^RuntimeError: .*{_USED}$")
        self.assertEqual(result["reload journal: undo"], "returned List")
        self.assertEqual(result["reimport: new List"], "returned List")
        for key in ("reload journal: select", "reimport: select"):
            self.assertRegex(result[key], r"^TypeError: ")
