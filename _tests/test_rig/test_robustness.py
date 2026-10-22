"""The freed / deleted robustness run (2026-10-17): the live crashes the r3b verifier found,
and the user decision "Deleted nodes in a List + the cmds crash" (2026-10-16).

* `TestDataFnSetRefusesCompoundsAndMultis`: ``get_data_fn_set()`` on a compound or a multi
  (array) plug raises TypeError (Maya read freed memory for its data handle: an access
  violation on ``t``, ``input1D``, ``cv``, ``vtx``).
* `TestMeshPointsRefuseRigObjects`: ``Mesh.get_closest_point`` / ``get_uv_at_point`` /
  ``set_points`` given a Plug or a node (or a list holding one) raise TypeError naming the
  fix (OpenMaya reads a Plug, a str with ``__getitem__``, as a sequence: an access violation).
* `TestDeletedNodeName`: a deleted or freed node's str is its last known name, marked
  ("obj1 (deleted)"), its repr ``Transform("obj1", deleted)``; a plug of one has a marked str and
  repr (user decision 6C, 2026-10-17); every operation on the node (and rig's private name reader,
  `_name_of`) still raises "already deleted!".
* `TestListSkipsDeletedElements` (re-pinned, user decisions 1A-4A and 6C 2026-10-17, the List
  rule): every rig operation on a List holding a deleted or freed element (a node, or a plug of
  one) raises the List error; Python's len(), indexing and iteration answer; an undo makes the
  List usable again (it skipped the element with one warning per statement).
* `TestRawCmdsGivenADeletedList`: maya.cmds given a List whose first element is a deleted or
  freed node raises a clean error instead of crashing Maya (select / delete / setAttr / xform /
  parent; deleted to the undo queue, after a flushUndo, a new scene, a file open), in its own
  mayapy so a crash fails the test instead of the run.

Fix round (2026-10-17, the reviews of the robustness run; runs/features/robust/fix):

* `TestAnimCurveOfAFreedPlug`: ``AnimCurve.of`` on a plug of a freed node raises before its
  MPlug is read (an access violation), and a List holding such elements raises the List error.
* `TestDataFnSetOfAMatrixAttribute`: ``get_data_fn_set()`` on a matrix attribute reads the
  plug's matrix data (the data handle access-violated).
* `TestMeshPointsRefuseComponents`: the mesh point methods refuse a Components selection.
* `TestRawCmdsGivenOtherDeletedElements`, `TestMarkedReprOfOtherKinds`: a Components on a
  deleted shape, a membership token of a deleted layer / material / engine, a List of an element
  method, a bare predicate: marked repr, and maya.cmds given a List of them raises cleanly.
* `TestRigWrappersGivenADeletedList`: rig's wrappers given a user's List raise the List error
  (positional and keyword arguments), before Maya is called.
* `TestNoPythonFrame`: a str() / repr() called by C code with no Python frame below it.
* `TestOneWarningPerStatement`, `TestDeletedElementsFollowUps` (re-pinned, the List rule): no
  warning, every statement raises; get() is a numpy array or raises; a deleted probe raises;
  nested Lists, hash, Components.

Negative control: every test fails on b42933c (runs/features/robust/impl/negctl, and
runs/features/robust/fix/out/nc_base for the fix round's).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from maya import cmds
from maya.api import OpenMaya

import rig
from rig import AnimCurve, Layer, List, Node, Plug, Tag
from rig.nodetypes._base import _name_of
from rig.nodetypes import Mesh
from rig.nodetypes import Blinn
from rig.spec import Float
from rig._tests._base import MayaTestCase


def _in_own_mayapy(code):
    """(exit code, stdout, stderr) of `code` run in a new mayapy on this rig."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(rig.__file__)))
    env["MAYA_DISABLE_CER"] = "1"
    cwd = tempfile.mkdtemp(prefix="rig_robustness_")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, timeout=900,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return (proc.returncode, proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


def _result(test, out, err):
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    test.assertEqual(len(lines), 1, out[-3000:] + err[-3000:])
    result = json.loads(lines[0][len("RESULT "):])
    here = os.path.normcase(os.path.dirname(rig.__file__))
    test.assertEqual(os.path.normcase(os.path.dirname(result.pop("file"))), here)
    return result


_HEAD = r"""
import json, os, sys, tempfile
import maya.standalone
maya.standalone.initialize(name="python")
try:
    from maya import cmds
    from maya.api import OpenMaya
    import rig
    from rig import List, Node, Plug
    out = {"file": rig.__file__}
    def outcome(call):
        try:
            value = call()
            return "returned " + type(value).__name__
        except Exception as error:  # noqa: BLE001
            return "%s: %s" % (type(error).__name__, str(error).splitlines()[0][:160])
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


# --------------------------------------------------------------------- #
#  The live crashes
# --------------------------------------------------------------------- #


_DATA_FN = _HEAD + r"""
    cmds.polyCube(name="pc", ch=False)
    cmds.nurbsPlane(name="np", ch=False)
    cmds.createNode("plusMinusAverage", name="pma")
    cmds.setAttr("pma.input1D[0]", 1)
    plugs = {
        "t": lambda: Plug("persp.t"), "find_attr t": lambda: Node("persp").find_attr("t"),
        "input1D": lambda: Plug("pma.input1D"), "output3D": lambda: Plug("pma.output3D"),
        "cv root": lambda: Node("npShape").cv, "vtx root": lambda: Node("pcShape").vtx,
        "tx": lambda: Plug("persp.tx"), "worldMatrix[0]": lambda: Plug("persp.worldMatrix[0]"),
        "outMesh": lambda: Plug("pcShape.outMesh"),
    }
    for label, make in plugs.items():
        out[label] = outcome(lambda: make().get_data_fn_set())
""" + _TAIL


class TestDataFnSetRefusesCompoundsAndMultis(MayaTestCase):
    def test_a_compound_or_a_multi_raises_type_error(self):
        code, out, err = _in_own_mayapy(_DATA_FN)
        result = _result(self, out, err)
        for label in ("t", "find_attr t", "output3D"):
            with self.subTest(plug=label):
                self.assertRegex(result[label], r"^TypeError: .* is a compound plug, which has no data "
                                                r"handle: get_data_fn_set\(\) reads one plug's data; ask a child$")
        for label in ("input1D", "cv root", "vtx root"):
            with self.subTest(plug=label):
                self.assertRegex(result[label], r"^TypeError: .* is a multi \(array\) plug, which has no "
                                                r"data handle: .*; ask an element \(.*\[0\]\)$")
        # one plug's data, as before
        self.assertEqual(result["tx"], "returned NoneType")
        self.assertEqual(result["worldMatrix[0]"], "returned MFnMatrixData")
        self.assertEqual(result["outMesh"], "returned MFnMeshData")


_POINTS = _HEAD + r"""
    from rig.nodetypes import Mesh
    cmds.polyCube(name="pc", ch=False)
    cmds.createNode("transform", name="loc")
    cmds.setAttr("loc.t", 0, 0, 2)
    mesh = Node("pcShape")
    loc  = Node("loc")
    for label, arg in (
        ("plug", loc.t), ("plug str", Plug("loc.t")), ("node", loc), ("child plug", loc.tx),
        ("list of plugs", [loc.tx, loc.ty, loc.tz]), ("List of plugs", List([loc.tx, loc.ty, loc.tz])),
    ):
        out["closest " + label] = outcome(lambda: mesh.get_closest_point(arg))
        out["uv " + label] = outcome(lambda: mesh.get_uv_at_point(arg, "map1"))
    count = mesh.get_num_vertices()
    out["set_points plugs"] = outcome(lambda: mesh.set_points([loc.t] * count))
    out["set_points List"] = outcome(lambda: mesh.set_points(List([loc.t] * count)))
    out["set_points rows"] = outcome(lambda: mesh.set_points([[loc.tx, 0, 0]] * count))
    out["set_points plug"] = outcome(lambda: mesh.set_points(loc.t))
    out["closest value"] = list(mesh.get_closest_point(loc.t.get())[0])[:3]
    out["closest list"] = list(mesh.get_closest_point([0, 0, 2])[0])[:3]
""" + _TAIL


class TestMeshPointsRefuseRigObjects(MayaTestCase):
    def test_a_plug_or_a_node_raises_type_error_naming_the_fix(self):
        code, out, err = _in_own_mayapy(_POINTS)
        result = _result(self, out, err)
        for method in ("closest", "uv"):
            name = "get_closest_point" if method == "closest" else "get_uv_at_point"
            for label in ("plug", "plug str", "child plug", "list of plugs", "List of plugs"):
                with self.subTest(method=method, arg=label):
                    self.assertRegex(
                        result[f"{method} {label}"],
                        rf"^TypeError: Mesh\.{name} takes a point \(an MPoint, or 3 numbers\), not the "
                        rf"plug 'loc\.(translate[XYZ]?|t)': pass its value, Plug\('loc\..*'\)\.get\(\)$",
                    )
            with self.subTest(method=method, arg="node"):
                self.assertRegex(result[f"{method} node"],
                                 rf"^TypeError: Mesh\.{name} takes a point .* not the node 'loc': pass a "
                                 rf"position: Node\('loc'\)\.get_rotate_pivot\(\)")
        for label in ("plugs", "List", "rows", "plug"):
            with self.subTest(set_points=label):
                self.assertRegex(result[f"set_points {label}"],
                                 r"^TypeError: Mesh\.set_points takes points .* not the plug 'loc\.")
        # a value, or numbers, as before
        self.assertEqual(result["closest value"], [0.0, 0.0, 0.5])
        self.assertEqual(result["closest list"], [0.0, 0.0, 0.5])


# --------------------------------------------------------------------- #
#  Deleted nodes: their name
# --------------------------------------------------------------------- #


class TestDeletedNodeName(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)

    def test_a_node_deleted_to_the_undo_queue(self):
        node = Node(cmds.createNode("transform", name="obj1"))
        plug = node.tx
        cmds.rename("obj1", "renamed")
        cmds.delete("renamed")
        self.assertEqual(str(node), "renamed (deleted)")
        self.assertEqual(f"{node}", "renamed (deleted)")
        self.assertEqual(repr(node), 'Transform("renamed", deleted)')
        self.assertEqual(repr(plug), 'Plug("obj1.translateX", deleted)')
        # (re-pinned, user decision 6C 2026-10-17: a plug's str is marked too, the
        # name it was built with; rig's private name reader raises)
        self.assertEqual(str(plug), "obj1.translateX (deleted)")
        for label, op in (
            ("get_name", lambda: node.get_name()), ("get_node_type", lambda: node.get_node_type()),
            ("ty", lambda: node.ty), ("str(plug)", lambda: _name_of(plug)), ("plug.get", lambda: plug.get()),
            ("_name_of(node)", lambda: _name_of(node)),
            ("hash", lambda: hash(node)), ("==", lambda: node == node), ("lock", lambda: node.lock()),
            ("in Tag", lambda: node in Tag("probe")), ("<< Float", lambda: node << Float("k")),
        ):
            with self.subTest(op=label):
                with self.assertRaisesRegex(RuntimeError, r"^renamed already deleted!$"):
                    op()
        cmds.undo()
        self.assertEqual((str(node), repr(node)), ("renamed", 'Transform("renamed")'))

    def test_a_freed_node_reads_its_last_known_name(self):
        node  = Node(cmds.createNode("transform", name="obj1"))
        shape = Node(cmds.createNode("locator", name="obj1Shape", parent="obj1"))
        cmds.rename("obj1", "obj2")
        str(node)  # read once renamed: the last known name
        cmds.file(new=True, force=True)
        self.assertEqual(str(node), "obj2 (deleted)")
        self.assertEqual(str(shape), "obj1Shape (deleted)")
        self.assertEqual(repr(shape), 'Geometry("obj1Shape", deleted)')
        with self.assertRaisesRegex(RuntimeError, r"^Transform node \(freed by a new scene, .*already deleted!$"):
            node.get_name()

    def test_a_converted_material_shows_its_name(self):
        # re-pinned (astype in place, user decision 2026-10-22, option A): the
        # object converted is the new node; another object of the old one is
        # named by the old node's aside name
        red  = Blinn.define("red")
        held = Blinn("red")
        red.astype(rig.nodetypes.Phong)
        self.assertEqual((str(red), repr(red)), ("red", 'Phong("red")'))
        self.assertEqual((str(held), repr(held)), ("red__rigold (deleted)", 'Blinn("red__rigold", deleted)'))
        with self.assertRaisesRegex(RuntimeError, r"^red__rigold already deleted!$"):
            held.get_name()

    def test_the_marked_name_names_no_maya_node(self):
        node = Node(cmds.createNode("transform", name="obj1"))
        cmds.delete("obj1")
        cmds.createNode("transform", name="obj1")
        self.assertFalse(cmds.objExists(str(node)))
        with self.assertRaises((TypeError, ValueError)):
            cmds.select(str(node))


# --------------------------------------------------------------------- #
#  Deleted nodes in a List
# --------------------------------------------------------------------- #


class TestListSkipsDeletedElements(MayaTestCase):
    """Re-pinned (user decisions 1A-4A and 6C, 2026-10-17, the List rule; the test ids are
    kept): every rig operation on a List holding a deleted element raises one RuntimeError
    naming it, before anything is read or written; Python's len(), indexing and iteration
    answer; an undo that brings the element back makes the List usable again (the robustness
    run skipped it, with one warning per statement)."""

    TEST_START_NEW_SCENE = True
    USED = r"^List element \[1\] Transform\(\"c1\", deleted\) was deleted: this List can no longer be used; build a new List$"

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        self.ctls = List([Node(n) for n in names])
        self.c0, self.c1, self.c2 = (Node(n) for n in names)
        cmds.delete("c1")  # the middle one

    def refused(self, call):
        """``call()`` raises the List error naming [1] c1; nothing was warned."""
        with mock.patch.object(cmds, "warning") as warn:
            with self.assertRaisesRegex(RuntimeError, self.USED):
                call()
        warn.assert_not_called()

    def test_the_element_stays_in_its_slot(self):
        # Python's len(), indexing and iteration answer (user decision 6C): Maya's
        # argument readers go through them
        self.assertEqual(len(self.ctls), 3)
        self.assertEqual(
            repr(self.ctls), 'List([Transform("c0"), Transform("c1", deleted), Transform("c2")])'
        )
        self.assertEqual([str(x) for x in self.ctls], ["c0", "c1 (deleted)", "c2"])
        self.assertEqual(str(self.ctls[1]), "c1 (deleted)")
        self.assertEqual(len(list(self.ctls)), 3)

    def test_an_attribute_broadcast_holds_skip_in_its_slot(self):
        self.refused(lambda: self.ctls.tx)
        self.refused(lambda: self.ctls["ty"])
        self.refused(lambda: self.ctls[:2])
        self.refused(lambda: self.ctls[[0, 2]])

    def test_inject_and_assignment(self):
        self.refused(lambda: self.ctls.ty << 5)
        self.refused(lambda: setattr(self.ctls, "tz", 6))
        self.refused(lambda: self.ctls.sx << [2, 3, 4])
        self.refused(lambda: self.ctls << Float("k"))
        self.refused(lambda: self.ctls.ry << AnimCurve([0, 90], times=[0, 24]))
        self.assertEqual([cmds.getAttr(f"c{i}.ty") for i in (0, 2)], [0, 0])
        self.assertFalse(cmds.attributeQuery("k", node="c0", exists=True))

    def test_reads_queries_and_operators(self):
        for call in (
            lambda: self.ctls.get(), lambda: self.ctls >> None, lambda: -self.ctls,
            lambda: self.ctls > 0, lambda: self.ctls * 2, lambda: 2 * self.ctls,
            lambda: self.ctls == self.ctls, lambda: self.ctls.get_inputs(),
            lambda: List([self.c2.tx, self.c0.tx]) + self.ctls, lambda: self.c0.tx << self.ctls,
        ):
            self.refused(call)

    def test_lock_unlock(self):
        self.refused(lambda: self.ctls.lock())
        self.refused(lambda: self.ctls.unlock())
        self.assertEqual(cmds.lockNode("c0", "c2", query=True, lock=True), [False, False])

    def test_membership(self):
        layer = Layer.define("L")
        self.refused(lambda: self.ctls << layer)
        self.refused(lambda: self.ctls in layer)
        self.refused(lambda: layer.remove(self.ctls))
        self.refused(lambda: self.ctls >> Layer())
        self.assertIsNone(cmds.editDisplayLayerMembers("L", query=True))

    def test_list_verbs_match_no_deleted_element(self):
        # in / index / count / remove raise, for a live probe too, and for the
        # List's own element object (no identity match)
        held = list.__getitem__(self.ctls, 1)
        for probe in (self.c2, "c1", self.c1, held):
            self.refused(lambda: probe in self.ctls)
            self.refused(lambda: self.ctls.index(probe))
            self.refused(lambda: self.ctls.count(probe))
            self.refused(lambda: self.ctls.remove(probe))
        self.assertEqual(len(self.ctls), 3)

    def test_a_plug_of_a_deleted_node(self):
        plugs = List([self.c0.tx, self.c0.ty, Plug("c2.tx")])
        cmds.delete("c0")
        with self.assertRaisesRegex(
            RuntimeError,
            r'^List element \[0\] Plug\("c0.translateX", deleted\) belongs to a node that was deleted: '
            r"this List can no longer be used; build a new List$",
        ):
            plugs << 7
        self.assertEqual(cmds.getAttr("c2.tx"), 0)
        self.assertEqual(
            repr(plugs),
            'List([Plug("c0.translateX", deleted), Plug("c0.translateY", deleted), Plug("c2.translateX")])',
        )

    def test_one_undo_step(self):
        # the undo of the delete makes the same List usable again (state-based, 1A);
        # the redo makes it raise again
        self.refused(lambda: self.ctls.tx << [1, 2, 3])
        cmds.undo()  # the delete of c1
        self.ctls.tx << [1, 2, 3]
        self.assertEqual([cmds.getAttr(f"c{i}.tx") for i in range(3)], [1, 2, 3])
        cmds.delete("c1")  # deleted again
        self.refused(lambda: self.ctls.tx.get())
        cmds.undo()
        self.assertEqual(list(self.ctls.tx.get()), [1, 2, 3])

    def test_a_freed_element(self):
        folder = tempfile.mkdtemp(prefix="rig_robust_list_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "s.ma").replace(os.sep, "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(path, open=True, force=True)  # c0 and c2 come back as new nodes
        with self.assertRaisesRegex(RuntimeError, r"^List element \[0\] Transform\(\"c0\", deleted\) was "
                                                  r"deleted and freed: a List holds what is in the scene$"):
            List([self.c0, Node("c2")])
        freed = r"^List element \[0\] Transform\(\"c0\", deleted\) was deleted and freed: this List"
        with self.assertRaisesRegex(RuntimeError, freed):
            self.ctls.tx << 9
        self.assertEqual(cmds.getAttr("c2.tx"), 0)


# --------------------------------------------------------------------- #
#  maya.cmds given a List of deleted nodes
# --------------------------------------------------------------------- #


_RAW_CMDS = _HEAD + r"""
    cmds.undoInfo(state=True, infinity=True)
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "s.ma").replace(os.sep, "/")

    def held():
        cmds.file(new=True, force=True)
        names = [cmds.createNode("transform", name="ctrl%d" % i) for i in range(3)]
        return names, List([Node(n) for n in names])

    def free(how, names):
        if how == "undo":
            cmds.delete(names)
        elif how == "flush":
            cmds.delete(names)
            cmds.flushUndo()
            for _ in range(50):
                cmds.createNode("multiplyDivide")
        elif how == "new":
            cmds.file(new=True, force=True)
        elif how == "open":
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(path, open=True, force=True)

    calls = {
        "select": lambda lst: cmds.select(lst),
        "delete": lambda lst: cmds.delete(lst),
        "setAttr": lambda lst: cmds.setAttr(lst, 1),
        "xform": lambda lst: cmds.xform(lst, translation=(1, 2, 3)),
        "parent": lambda lst: cmds.parent(lst, world=True),
    }
    for how in ("undo", "flush", "new", "open"):
        for fn, call in calls.items():
            names, lst = held()
            free(how, names)
            sys.stdout.write("CASE %s %s\n" % (how, fn)); sys.stdout.flush()
            out["%s %s" % (how, fn)] = outcome(lambda: call(lst))
""" + _TAIL


class TestRawCmdsGivenADeletedList(MayaTestCase):
    def test_a_clean_error_instead_of_a_crash(self):
        code, out, err = _in_own_mayapy(_RAW_CMDS)
        result = _result(self, out, err)
        self.assertEqual(len(result), 20)
        for case, outcome in sorted(result.items()):
            with self.subTest(case=case):
                # Maya's own error for an argument it cannot read (a List of
                # nodes is a list subclass: "Object ... is invalid"), or for a
                # name that names no node; never "already deleted!" raised
                # inside Maya's argument parser, which crashed it
                self.assertRegex(outcome, r"^(TypeError|ValueError|RuntimeError): ")
                self.assertNotIn("already deleted", outcome)


# --------------------------------------------------------------------- #
#  Fix round (2026-10-17): the reviews of the robustness run
# --------------------------------------------------------------------- #


_CURVES_OF = _HEAD + r"""
    from rig import AnimCurve
    cmds.undoInfo(state=False)
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "s.ma").replace(os.sep, "/")
    for how in ("open", "new"):
        cmds.file(new=True, force=True)
        held = {"input": [], "owned": [], "multi": [], "List": []}
        for i in range(700):
            src = cmds.createNode("transform", name="src%d" % i)
            dst = cmds.createNode("transform", name="dst%d" % i)
            cmds.connectAttr(src + ".tx", dst + ".tx")
            held["input"].append(Plug(dst + ".tx").get_inputs()[0])
            held["owned"].append(Node(dst).t)
            pma = cmds.createNode("plusMinusAverage", name="pma%d" % i)
            cmds.setAttr(pma + ".input1D[0]", 1)
            held["multi"].append(Plug(pma + ".input1D"))
            if i < 50:
                held["List"].append(List([Plug(dst + ".tx").get_inputs()[0], Node(dst).ty]))
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        if how == "open":
            cmds.file(path, open=True, force=True)
        else:
            cmds.file(new=True, force=True)
        sys.stdout.write("CASE %s\n" % how); sys.stdout.flush()
        for kind, plugs in held.items():
            seen = {}
            for x in plugs:
                o = outcome(lambda: AnimCurve.of(x))
                seen[o] = seen.get(o, 0) + 1
            out["%s %s" % (how, kind)] = seen
""" + _TAIL


class TestAnimCurveOfAFreedPlug(MayaTestCase):
    """``AnimCurve.of(x)`` read the MPlug of a plug whose node a file open or a new scene
    freed (an access violation, or values read from freed memory): it raises "... already
    deleted!" before the read, and a List holding such elements raises the List error
    (re-pinned, user decisions 1A-4A and 6C 2026-10-17: it skipped them)."""

    def test_raises_already_deleted_and_a_list_skips(self):
        code, out, err = _in_own_mayapy(_CURVES_OF)
        result = _result(self, out, err)
        for how in ("open", "new"):
            for kind in ("input", "owned", "multi"):
                with self.subTest(how=how, kind=kind):
                    # every call raised the freed node's "already deleted!"
                    seen = result[f"{how} {kind}"]
                    self.assertEqual(sum(seen.values()), 700, seen)
                    for message in seen:
                        self.assertRegex(message, r"^RuntimeError: .*already deleted!$")
            with self.subTest(how=how, kind="List"):
                seen = result[f"{how} List"]
                self.assertEqual(sum(seen.values()), 50, seen)
                for message in seen:
                    self.assertRegex(message, r"^RuntimeError: List element \[0\] .* was deleted and freed: "
                                              r"this List can no longer be used; build a new List$")


_MATRIX_DATA = _HEAD + r"""
    cmds.polyCube(name="pc", ch=False)
    cmds.createNode("multMatrix", name="mm")
    cmds.setAttr("mm.matrixIn[0]", [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], type="matrix")
    cmds.addAttr("persp", longName="mtxA", attributeType="matrix")
    cmds.addAttr("persp", longName="fmtxA", attributeType="fltMatrix")
    names = ("pc.offsetParentMatrix", "pc.dagLocalMatrix", "pc.dagLocalInverseMatrix",
             "mm.matrixIn[0]", "mm.matrixSum", "persp.mtxA", "persp.fmtxA", "pc.matrix")
    for rep in range(20):
        for name in names:
            out[name] = outcome(lambda: Plug(name).get_data_fn_set())
    out["matrixIn[0] value"] = [round(v, 3) for v in Plug("mm.matrixIn[0]").get_data_fn_set().matrix()][:2]
""" + _TAIL


class TestDataFnSetOfAMatrixAttribute(MayaTestCase):
    """``get_data_fn_set()`` on a matrix (or float matrix) attribute read Maya's data handle,
    which holds no data there (an access violation): the plug's MObject is its matrix data,
    as a typed matrix attribute's is."""

    def test_matrix_data_as_a_typed_matrix_gives(self):
        code, out, err = _in_own_mayapy(_MATRIX_DATA)
        result = _result(self, out, err)
        for name in ("pc.offsetParentMatrix", "pc.dagLocalMatrix", "pc.dagLocalInverseMatrix",
                     "mm.matrixIn[0]", "mm.matrixSum", "persp.mtxA", "persp.fmtxA", "pc.matrix"):
            with self.subTest(plug=name):
                self.assertEqual(result[name], "returned MFnMatrixData")
        self.assertEqual(result["matrixIn[0] value"], [2.0, 0.0])


_POINT_COMPONENTS = _HEAD + r"""
    cmds.polyCube(name="pc", ch=False)
    mesh = Node("pcShape")
    for label, make in (
        ("face", lambda: mesh.f[0]), ("faces", lambda: mesh.f[:3]), ("edge", lambda: mesh.e[0]),
        ("every face", lambda: mesh.f), ("list of a face", lambda: [mesh.f[0]]),
    ):
        out["closest " + label] = outcome(lambda: mesh.get_closest_point(make()))
        out["uv " + label] = outcome(lambda: mesh.get_uv_at_point(make(), "map1"))
    count = mesh.get_num_vertices()
    out["set_points row"] = outcome(lambda: mesh.set_points([[0, 0, 0]] * (count - 1) + [mesh.f[0]]))
    out["set_points List"] = outcome(lambda: mesh.set_points(List([mesh.f[0], mesh.f[1], mesh.f[2]])))
    out["points after"] = cmds.getAttr("pcShape.vt[0]")[0][0]
""" + _TAIL


class TestMeshPointsRefuseComponents(MayaTestCase):
    """``get_closest_point`` / ``get_uv_at_point`` / ``set_points`` given a Components
    selection (``cube.f[0]``: OpenMaya reads it through its ``__getitem__``, an access
    violation) raise TypeError naming the components and a position to pass."""

    def test_a_face_an_edge_or_a_kind_raises_type_error(self):
        code, out, err = _in_own_mayapy(_POINT_COMPONENTS)
        result = _result(self, out, err)
        for label in ("face", "faces", "edge", "every face", "list of a face"):
            for method, name in (("closest", "get_closest_point"), ("uv", "get_uv_at_point")):
                with self.subTest(method=method, arg=label):
                    self.assertRegex(
                        result[f"{method} {label}"],
                        rf"^TypeError: Mesh\.{name} takes a point \(an MPoint, or 3 numbers\), not the "
                        r"components Components\(\"\|pc\|pcShape\.(f|e)\[.*\]\"\): pass a position",
                    )
        for label in ("row", "List"):
            with self.subTest(set_points=label):
                self.assertRegex(result[f"set_points {label}"],
                                 r"^TypeError: Mesh\.set_points takes points .* not the components ")
        self.assertEqual(result["points after"], -0.5)  # nothing was set


_CMDS_OTHER_KINDS = _HEAD + r"""
    from rig import Layer
    from rig.bridges import commands as rc
    from rig.nodetypes import Blinn
    cmds.undoInfo(state=True, infinity=True)
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "s.ma").replace(os.sep, "/")

    def free(how, names):
        if how == "undo":
            cmds.delete(names)
        elif how == "flush":
            cmds.delete(names)
            cmds.flushUndo()
            for _ in range(50):
                cmds.createNode("multiplyDivide")
        elif how == "new":
            cmds.file(new=True, force=True)
        elif how == "open":
            cmds.file(rename=path)
            cmds.file(save=True, type="mayaAscii", force=True)
            cmds.file(path, open=True, force=True)

    def faces():
        cmds.polyCube(name="c1", ch=False); cmds.polyCube(name="c2", ch=False)
        return ["c1", "c2"], List([Node("c1Shape").f[:3], Node("c2Shape").f[0]])

    def layer_tokens():
        Layer.define("lay")
        token = -Layer("lay")
        return ["lay"], List([token, token])

    def material_tokens():
        red = Blinn.create(name="red")
        engine = Node(cmds.listConnections(str(red) + ".outColor", d=True, s=False)[0])
        token = -red
        return [str(red), str(engine)], List([token, token])

    def engine_tokens():
        red = Blinn.create(name="red")
        engine = Node(cmds.listConnections(str(red) + ".outColor", d=True, s=False)[0])
        return [str(red), str(engine)], List([-engine, -engine])

    def methods():
        t = [Node(cmds.createNode("transform", name="t%d" % i)) for i in range(2)]
        named = List(t).get_name
        return ["t0", "t1"], List([named, named])

    def predicates():
        # (a plain list: a List holds no bare predicate)
        t = [Node(cmds.createNode("transform", name="t%d" % i)) for i in range(2)]
        return ["t0", "t1"], [t[0].is_locked, t[1].is_locked]

    calls = {
        "select": lambda a: cmds.select(a), "delete": lambda a: cmds.delete(a),
        "getAttr": lambda a: cmds.getAttr(a), "sets": lambda a: cmds.sets(a, add="st"),
        "rc.select": lambda a: rc.select(a),
    }
    for how in ("undo", "flush", "new", "open"):
        for kind, make in (("faces", faces), ("layer tokens", layer_tokens),
                           ("material tokens", material_tokens), ("engine tokens", engine_tokens),
                           ("methods", methods), ("predicates", predicates)):
            for fn, call in calls.items():
                cmds.file(new=True, force=True)
                cmds.sets(name="st", empty=True)
                names, held = make()
                free(how, names)
                if how in ("new", "open") and not cmds.objExists("st"):
                    cmds.sets(name="st", empty=True)
                sys.stdout.write("CASE %s %s %s\n" % (how, kind, fn)); sys.stdout.flush()
                out["%s %s %s" % (how, kind, fn)] = outcome(lambda: call(held))
                out["%s %s repr" % (how, kind)] = outcome(lambda: repr(held))
""" + _TAIL


class TestRawCmdsGivenOtherDeletedElements(MayaTestCase):
    """maya.cmds (and rig.bridges.commands) given a List whose first element is not a node
    but holds a deleted or freed one -- a Components selection, a membership removal token
    (a layer's, a material's, a shading engine's), a List of an element method, a bare
    predicate -- raise a clean error instead of crashing Maya (their str() raised inside Maya's
    argument parser); their repr is marked, as a node's is."""

    def test_a_clean_error_instead_of_a_crash(self):
        code, out, err = _in_own_mayapy(_CMDS_OTHER_KINDS)
        result = _result(self, out, err)
        self.assertEqual(len(result), 4 * 6 * 6)
        for case, outcome in sorted(result.items()):
            with self.subTest(case=case):
                if case.endswith(" repr"):
                    self.assertTrue(outcome.startswith("returned str"), outcome)
                    continue
                self.assertRegex(outcome, r"^(TypeError|ValueError|RuntimeError): ")
                self.assertNotIn("already deleted", outcome)


_REPRS = _HEAD + r"""
    from rig import Layer
    cmds.undoInfo(state=True, infinity=True)
    cmds.polyCube(name="c1", ch=False)
    faces = Node("c1Shape").f[:3]
    Layer.define("lay")
    token = -Layer("lay")
    t = Node(cmds.createNode("transform", name="t0"))
    named = List([t]).get_name
    held = List([faces, token, named])  # (a List holds what is in the scene: built first)
    cmds.delete("c1", "lay", "t0")
    out["faces"] = repr(faces)
    out["faces str"] = str(faces)
    out["token"] = repr(token)
    out["token str"] = str(token)
    out["List"] = repr(held)
    out["pred"] = repr(t.is_locked)
""" + _TAIL


class TestMarkedReprOfOtherKinds(MayaTestCase):
    def test_components_tokens_methods_and_predicates(self):
        code, out, err = _in_own_mayapy(_REPRS)
        result = _result(self, out, err)
        self.assertEqual(result["faces"], 'Components("c1Shape.f[0:2]", deleted)')
        self.assertEqual(result["faces str"], 'Components("c1Shape.f[0:2]", deleted)')
        self.assertEqual(result["token"], '-DisplayLayer("lay", deleted)')
        self.assertEqual(result["token str"], "lay (deleted)")
        self.assertEqual(
            result["List"],
            'List([Components("c1Shape.f[0:2]", deleted), -DisplayLayer("lay", deleted), '
            'List([<bound method DGNode.get_name of Transform("t0", deleted)>])])',
        )
        self.assertEqual(
            result["pred"], '<predicate <bound method DGNode.is_locked of Transform("t0", deleted)>: call it>'
        )


_WRAPPERS = _HEAD + r"""
    from rig.bridges import commands as rc
    from rig.nodetypes.blendshape import BlendShape
    from rig.nodetypes.display_layer import DisplayLayer
    from rig.nodetypes.transform import Transform
    from rig.nodetypes import Material
    cmds.undoInfo(state=True, infinity=True)

    def scene():
        cmds.file(new=True, force=True)
        for n in ("t1", "t2", "keep"):
            cmds.createNode("transform", name=n)
        for n in ("base", "tgt1", "tgt2"):
            cmds.polyCube(name=n, ch=False)
        cmds.sets(name="st", empty=True)
        return Node("t1"), Node("t2"), Node("keep")

    def free(how, names):
        cmds.delete(names)
        if how == "flush":
            cmds.flushUndo()
            for _ in range(50):
                cmds.createNode("multiplyDivide")

    calls = {
        "delete(List)": (["t1", "t2"], lambda t1, t2, keep, lst: keep.delete(lst)),
        "delete(list)": (["t1", "t2"], lambda t1, t2, keep, lst: keep.delete(list(lst))),
        "duplicate(List)": (["t1", "t2"], lambda t1, t2, keep, lst: keep.duplicate(lst)),
        "set_parent(List)": (["t1", "t2"], lambda t1, t2, keep, lst: keep.set_parent(lst)),
        "add_members(List)": (["t1", "t2"], lambda t1, t2, keep, lst: Node("st").add_members(lst)),
        "find_all(List)": (["t1", "t2"], lambda t1, t2, keep, lst: Transform.find_all(lst)),
        "Material.find_all(List)": (["t1", "t2"], lambda t1, t2, keep, lst: Material.find_all(lst)),
        "DisplayLayer.create(List)": (["t1", "t2"], lambda t1, t2, keep, lst: DisplayLayer.create(lst)),
        "BlendShape.create(List)": (["tgt1", "tgt2"], lambda t1, t2, keep, lst: BlendShape.create(lst, Node("base"))),
        "rc.select(List)": (["t1", "t2"], lambda t1, t2, keep, lst: rc.select(lst)),
        "rc.select(add=List)": (["t1", "t2"], lambda t1, t2, keep, lst: rc.select("keep", add=lst) or
                                rc.select(lst, add=True)),
        "rc.sets(add=List)": (["t1", "t2"], lambda t1, t2, keep, lst: rc.sets(lst, add="st")),
        "rc.container(addNode=List)": (["t1", "t2"], lambda t1, t2, keep, lst: rc.container(addNode=lst)),
    }
    for how in ("undo", "flush"):
        for label, (names, call) in calls.items():
            t1, t2, keep = scene()
            lst = List([Node(names[0]), Node(names[1])])
            free(how, names)
            sys.stdout.write("CASE %s %s\n" % (how, label)); sys.stdout.flush()
            out["%s %s" % (how, label)] = outcome(lambda: call(t1, t2, keep, lst))
            out["%s %s keep" % (how, label)] = cmds.objExists("keep")
    t1, t2, keep = scene()
    out["live delete(List)"] = outcome(lambda: keep.delete(List([t1, t2])))
    out["live deleted"] = [cmds.objExists("t1"), cmds.objExists("t2")]
""" + _TAIL


class TestRigWrappersGivenADeletedList(MayaTestCase):
    """rig's wrappers given a user's List holding a deleted element (``node.delete(nodes)``,
    ``duplicate``, ``set_parent``, ``ObjectSet.add_members``, ``find_all``,
    ``DisplayLayer.create``, ``BlendShape.create``, ``rc.select``, a List given as a keyword
    argument): re-pinned (user decisions 1A-4A and 6C, 2026-10-17, the List rule): each raises
    the List error before Maya is called, and nothing is changed. A plain list of deleted nodes
    reaches Maya, which raises its own clean error (their str() is the marked name)."""

    def test_a_clean_error_instead_of_a_crash(self):
        code, out, err = _in_own_mayapy(_WRAPPERS)
        result = _result(self, out, err)
        self.assertEqual(result.pop("live delete(List)"), "returned NoneType")
        self.assertEqual(result.pop("live deleted"), [False, False])
        for case, outcome in sorted(result.items()):
            with self.subTest(case=case):
                if case.endswith(" keep"):
                    self.assertIs(outcome, True)  # nothing else was deleted
                    continue
                if "(list)" in case:
                    # a plain list: Maya reads the marked names, its own clean error
                    self.assertRegex(outcome, r"^(TypeError|ValueError|RuntimeError): ")
                    self.assertNotIn("List element", outcome)
                    continue
                freed = " and freed" if case.startswith("flush") else ""
                self.assertRegex(
                    outcome,
                    rf"^RuntimeError: List element \[0\] \w+\(\"\w+\", deleted\) was deleted{freed}: this "
                    r"List can no longer be used; build a new List$",
                )


_NO_FRAME = _HEAD + r"""
    import atexit, functools
    cmds.undoInfo(state=True, infinity=True)
    nodes = List([Node(cmds.createNode("transform", name="ctrl%d" % i)) for i in range(3)])
    cmds.delete("ctrl0")

    def report():
        sys.stdout.write("SELECTION %s\n" % cmds.ls(selection=True)); sys.stdout.flush()

    # atexit runs these last first: the C call with no Python frame below it,
    # then the report
    atexit.register(report)
    atexit.register(functools.partial(cmds.select, nodes))
    print("RESULT " + json.dumps(out))
    sys.stdout.flush()
except BaseException:
    raise
"""


class TestNoPythonFrame(MayaTestCase):
    """C code that calls a deleted node's str() or repr() with no Python frame below it (a Qt
    slot, an evalDeferred or atexit callable, a thread: ``partial(cmds.select, ctls)``) read the
    caller's frame and raised ValueError inside Maya's parser (a crash): no frame is no rig
    code, so the marked name."""

    TEST_START_NEW_SCENE = True

    def test_str_and_repr_from_a_thread(self):
        import _thread
        import time

        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)
        node = Node(cmds.createNode("transform", name="obj1"))
        plug = node.tx
        cmds.delete("obj1")
        got, raised = [], []
        hook = sys.unraisablehook
        sys.unraisablehook = lambda u: raised.append(u.exc_value)
        try:
            # list.extend consumes the map in C: str() and repr() are called with no
            # Python frame below them; one thread at a time
            for count, call, item in ((1, str, node), (2, repr, node), (3, repr, plug)):
                _thread.start_new_thread(got.extend, (map(call, [item]),))
                for _ in range(500):
                    if len(got) == count or raised:
                        break
                    time.sleep(0.01)
        finally:
            sys.unraisablehook = hook
        self.assertEqual(raised, [])
        self.assertEqual(sorted(got), sorted(
            ["obj1 (deleted)", 'Transform("obj1", deleted)', 'Plug("obj1.translateX", deleted)']))

    def test_an_atexit_partial_of_cmds_select(self):
        code, out, err = _in_own_mayapy(_NO_FRAME)
        _result(self, out, err)
        # Maya's clean error, printed by atexit, then the report: the selection
        # is the last node made, as it was
        self.assertIn("SELECTION ['ctrl2']", out, out[-2000:] + err[-2000:])
        self.assertIn("is invalid", err, err[-2000:])
        self.assertNotIn("access violation", err)


class TestOneWarningPerStatement(MayaTestCase):
    """Re-pinned (user decisions 1A-4A and 6C, 2026-10-17, the List rule; the test ids are
    kept): no warning anywhere; every statement on a List holding a deleted element raises the
    List error, a chained one at its first use, each pass of a loop, another List of the same
    names too (it warned once per statement and skipped)."""

    TEST_START_NEW_SCENE = True
    USED = r"^List element \[1\] Transform\(\"c1\", deleted\) was deleted: this List can no longer be used; build a new List$"

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        self.ctls = List([Node(n) for n in names])
        cmds.delete("c1")

    def test_a_chained_statement_warns_once(self):
        ctls = self.ctls
        for call in (lambda: ctls.tx << ctls.ty * 2 + ctls.tz, lambda: setattr(ctls, "tx", ctls.ty),
                     lambda: ctls.ty << ctls.tx.get()):
            with mock.patch.object(cmds, "warning") as warn:
                with self.assertRaisesRegex(RuntimeError, self.USED):
                    call()
            warn.assert_not_called()
        self.assertEqual(cmds.ls(type="multiplyDivide"), [])

    def test_each_statement_and_each_pass_warns(self):
        ctls = self.ctls
        raised = 0
        for value in range(3):
            try:
                ctls.tz << value
            except RuntimeError:
                raised += 1
        self.assertEqual(raised, 3)

    def test_another_list_of_the_same_names_warns(self):
        raised = []
        for _ in range(2):
            cmds.file(new=True, force=True)
            names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
            ctls = List([Node(n) for n in names])
            cmds.delete("c1")
            try:
                ctls.tx << 1
            except RuntimeError as error:
                raised.append(str(error))
        self.assertEqual(len(raised), 2)
        for text in raised:
            self.assertRegex(text, self.USED)


class TestDeletedElementsFollowUps(MayaTestCase):
    """Re-pinned (user decisions 1A-4A and 6C, 2026-10-17, the List rule; the test ids are
    kept): get() is a numpy array, and raises once an element is deleted; a deleted probe
    raises its "already deleted!" (no identity match); a nested List's deleted element raises
    with its path; hash raises; a Components on a deleted shape raises (each was skipped)."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        self.undo_on()  # puts back the undo state it found (S2_finish: no test leaves undo off)

    def test_get_keeps_a_list_with_skip(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        ctls = List([Node(n) for n in names])
        self.assertEqual(type(ctls.tx.get()).__name__, "ndarray")
        tx = ctls.tx
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, r"^List element \[1\] Plug\(\"c1.translateX\", deleted\) "
                                                  r"belongs to a node that was deleted: this List"):
            tx.get()
        cmds.undo()
        self.assertEqual(type(tx.get()).__name__, "ndarray")

    def test_a_deleted_probe_is_found_by_identity(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        ctls = List([Node(n) for n in names])
        gone = list.__getitem__(ctls, 1)
        live = List([Node("c0"), Node("c2")])
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, r"^c1 already deleted!$"):
            gone in live
        with self.assertRaisesRegex(RuntimeError, r"^c1 already deleted!$"):
            live.remove(gone)
        with self.assertRaisesRegex(RuntimeError, r"this List can no longer be used; build a new List$"):
            ctls.remove(gone)
        self.assertEqual(len(ctls), 3)

    def test_nested_lists_and_hash(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        a, b, c = (Node(n) for n in names)
        nested = List([List([a, b]), List([c])])
        flat = List([a, b])
        layer = Layer.define("L")
        cmds.delete("c0")
        with self.assertRaisesRegex(RuntimeError, r"^List element \[0\]\[0\] Transform\(\"c0\", deleted\) "
                                                  r"was deleted: this List can no longer be used"):
            nested << layer
        self.assertIsNone(cmds.editDisplayLayerMembers("L", query=True))
        with self.assertRaisesRegex(RuntimeError, r"this List can no longer be used"):
            hash(flat)
        with self.assertRaisesRegex(RuntimeError, r"a List holds what is in the scene$"):
            List([a, b])
        self.assertIsInstance(hash(List([b, c])), int)

    def test_membership_skips_a_plug_of_a_freed_node(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(2)]
        layer = Layer.define("L")
        Node("c1") << layer
        plugs = List([Node("c0").tx, Node("c1").tx])
        cmds.delete("c0")
        cmds.flushUndo()  # c0 is freed
        with self.assertRaisesRegex(RuntimeError, r"^List element \[0\] Plug\(\"c0.translateX\", deleted\) "
                                                  r"belongs to a node that was deleted and freed: this List"):
            plugs in layer

    def test_a_components_on_a_deleted_shape_is_skipped(self):
        cmds.polyCube(name="c1", ch=False)
        cmds.polyCube(name="c2", ch=False)
        faces = List([Node("c1Shape").f[:3], Node("c2Shape").f[0]])
        cmds.delete("c1")
        with self.assertRaisesRegex(RuntimeError, r'^List element \[0\] Components\("c1Shape.f\[0:2\]", deleted\) '
                                                  r"belongs to a node that was deleted: this List"):
            faces << Tag("cap")
        self.assertEqual(list(Node("c2Shape").f >> Tag("cap")), [])
        self.assertEqual(repr(faces), 'List([Components("c1Shape.f[0:2]", deleted), Components("|c2|c2Shape.f[0]")])')
        self.assertEqual(str(list.__getitem__(faces, 0)), 'Components("c1Shape.f[0:2]", deleted)')
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            list.__getitem__(faces, 0).get_names()  # an operation on it raises
