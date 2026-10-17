"""The freed / deleted robustness run (2026-10-17): the live crashes the r3b verifier found,
and the user decision "Deleted nodes in a List + the cmds crash" (2026-10-16).

* `TestDataFnSetRefusesCompoundsAndMultis`: ``get_data_fn_set()`` on a compound or a multi
  (array) plug raises TypeError (Maya read freed memory for its data handle: an access
  violation on ``t``, ``input1D``, ``cv``, ``vtx``).
* `TestMeshPointsRefuseRigObjects`: ``Mesh.get_closest_point`` / ``get_uv_at_point`` /
  ``set_points`` given a Plug or a node (or a list holding one) raise TypeError naming the
  fix (OpenMaya reads a Plug, a str with ``__getitem__``, as a sequence: an access violation).
* `TestDeletedNodeName`: a deleted or freed node's str is its last known name, marked
  ("obj1 (deleted)"), its repr ``Transform("obj1", deleted)``; a plug of one has a marked repr;
  every operation on the node (and a plug's str) still raises "already deleted!".
* `TestListSkipsDeletedElements`: rig's List operations skip a deleted or freed element (a node,
  or a plug of one), with ONE warning per statement naming it; a derived List holds rig.skip in
  its slot, which every List operation leaves alone (chained broadcasts stay aligned).
* `TestRawCmdsGivenADeletedList`: maya.cmds given a List whose first element is a deleted or
  freed node raises a clean error instead of crashing Maya (select / delete / setAttr / xform /
  parent; deleted to the undo queue, after a flushUndo, a new scene, a file open), in its own
  mayapy so a crash fails the test instead of the run.

Fix round (2026-10-17, the reviews of the robustness run; runs/features/robust/fix):

* `TestAnimCurveOfAFreedPlug`: ``AnimCurve.of`` on a plug of a freed node raises before its
  MPlug is read (an access violation), and a List skips such elements.
* `TestDataFnSetOfAMatrixAttribute`: ``get_data_fn_set()`` on a matrix attribute reads the
  plug's matrix data (the data handle access-violated).
* `TestMeshPointsRefuseComponents`: the mesh point methods refuse a Components selection.
* `TestRawCmdsGivenOtherDeletedElements`, `TestMarkedReprOfOtherKinds`: a Components on a
  deleted shape, a membership token of a deleted layer / material / engine, a List of an element
  method, a bare predicate: marked repr, and maya.cmds given a List of them raises cleanly.
* `TestRigWrappersGivenADeletedList`: rig's wrappers that forward a user's List to cmds.
* `TestNoPythonFrame`: a str() / repr() called by C code with no Python frame below it.
* `TestOneWarningPerStatement`, `TestDeletedElementsFollowUps`: the warning per statement,
  get() keeping a List, a deleted probe found by identity, nested Lists, hash, Components.

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
from rig import AnimCurve, Layer, List, Node, Plug, Tag, skip
from rig.nodetypes import Mesh
from rig.shade import Blinn
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
        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(cmds.undoInfo, state=False)

    def test_a_node_deleted_to_the_undo_queue(self):
        node = Node(cmds.createNode("transform", name="obj1"))
        plug = node.tx
        cmds.rename("obj1", "renamed")
        cmds.delete("renamed")
        self.assertEqual(str(node), "renamed (deleted)")
        self.assertEqual(f"{node}", "renamed (deleted)")
        self.assertEqual(repr(node), 'Transform("renamed", deleted)')
        self.assertEqual(repr(plug), 'Plug("obj1.translateX", deleted)')
        for label, op in (
            ("get_name", lambda: node.get_name()), ("get_node_type", lambda: node.get_node_type()),
            ("ty", lambda: node.ty), ("str(plug)", lambda: str(plug)), ("plug.get", lambda: plug.get()),
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
        red = Blinn.define("red")
        red.astype(rig.shade.Phong)
        self.assertEqual((str(red), repr(red)), ("red (deleted)", 'Blinn("red", deleted)'))
        with self.assertRaisesRegex(RuntimeError, r"^'red' was converted to a phong"):
            red.get_name()

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
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(cmds.undoInfo, state=False)
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        self.ctls = List([Node(n) for n in names])
        self.c0, self.c1, self.c2 = (Node(n) for n in names)
        cmds.delete("c1")  # the middle one: alignment shows

    def warned(self, call, count=1):
        """``call()``'s result; `count` warnings were printed, each naming [1] c1."""
        with mock.patch.object(cmds, "warning") as warn:
            result = call()
        self.assertEqual(warn.call_count, count)
        for args in warn.call_args_list:
            self.assertEqual(args[0][0], "List element [1] c1 was deleted; skipped")
        return result

    def test_the_element_stays_in_its_slot(self):
        self.assertEqual(len(self.ctls), 3)
        self.assertEqual(
            repr(self.ctls), 'List([Transform("c0"), Transform("c1", deleted), Transform("c2")])'
        )
        self.assertEqual([str(x) for x in self.ctls], ["c0", "c1 (deleted)", "c2"])

    def test_an_attribute_broadcast_holds_skip_in_its_slot(self):
        tx = self.warned(lambda: self.ctls.tx)
        self.assertEqual(list(tx), [self.c0.tx, skip, self.c2.tx])
        self.assertEqual(self.warned(lambda: self.ctls["ty"])[1], skip)
        # a chained broadcast keeps skipping, with no new warning
        self.warned(lambda: tx << [1.0, 2.0, 3.0], count=0)
        self.assertEqual((cmds.getAttr("c0.tx"), cmds.getAttr("c2.tx")), (1.0, 3.0))
        self.assertEqual(self.warned(lambda: tx.get(), count=0), [1.0, skip, 3.0])
        summed = self.warned(lambda: tx + 1, count=0)
        self.assertIs(summed[1], skip)
        self.assertEqual(summed[2].get(), 4.0)

    def test_inject_and_assignment(self):
        self.warned(lambda: self.ctls.ty << 5)
        self.warned(lambda: setattr(self.ctls, "tz", 6))
        self.warned(lambda: self.ctls.sx << [2, 3, 4])  # pairs stay aligned
        self.assertEqual([cmds.getAttr(f"c{i}.ty") for i in (0, 2)], [5, 5])
        self.assertEqual([cmds.getAttr(f"c{i}.tz") for i in (0, 2)], [6, 6])
        self.assertEqual([cmds.getAttr(f"c{i}.sx") for i in (0, 2)], [2, 4])
        added = self.warned(lambda: self.ctls << Float("k"))
        self.assertEqual([str(x) for x in added if x is not skip], ["c0.k", "c2.k"])
        self.assertIs(added[1], skip)
        self.warned(lambda: self.ctls.ry << AnimCurve([0, 90], times=[0, 24]))
        self.assertEqual(sorted(cmds.listConnections("c0.ry", "c2.ry", source=True) or []),
                         ["c0_rotateY", "c2_rotateY"])

    def test_reads_queries_and_operators(self):
        self.assertEqual(self.warned(lambda: self.ctls.get()), [self.c0, skip, self.c2])
        self.assertEqual(self.warned(lambda: self.ctls >> None), [self.c0, skip, self.c2])
        tx = self.warned(lambda: self.ctls.tx)
        inputs = self.warned(lambda: tx.get_inputs(), count=0)
        self.assertEqual((len(inputs), inputs[1]), (3, skip))
        self.assertEqual(self.warned(lambda: list(-tx), count=0)[1], skip)
        self.assertEqual(self.warned(lambda: list(tx > 0), count=0)[1], skip)
        self.assertEqual(self.warned(lambda: list(self.ctls.tx * 2))[1], skip)
        # a List operand with a deleted element: its row is skipped too
        with mock.patch.object(cmds, "warning") as warn:
            rows = List([self.c2.tx, self.c0.tx, self.c0.ty]) + self.ctls.ty
        self.assertEqual(warn.call_count, 1)  # the operand's broadcast warned
        self.assertEqual([x is skip for x in rows], [False, True, False])

    def test_lock_unlock(self):
        self.warned(lambda: self.ctls.lock())
        self.assertEqual(cmds.lockNode("c0", "c2", query=True, lock=True), [True, True])
        self.warned(lambda: self.ctls.unlock())
        self.assertEqual(cmds.lockNode("c0", "c2", query=True, lock=True), [False, False])

    def test_membership(self):
        layer = Layer.define("L")
        self.warned(lambda: self.ctls << layer)
        self.assertEqual(sorted(cmds.editDisplayLayerMembers("L", query=True)), ["c0", "c2"])
        self.assertIs(self.warned(lambda: self.ctls in layer), True)
        self.warned(lambda: layer.remove(self.ctls))
        self.assertIsNone(cmds.editDisplayLayerMembers("L", query=True))
        self.assertIs(self.warned(lambda: self.ctls in layer), False)

    def test_list_verbs_match_no_deleted_element(self):
        self.assertIs(self.warned(lambda: self.c2 in self.ctls), True)
        self.assertEqual(self.warned(lambda: self.ctls.index(self.c2)), 2)
        self.assertEqual(self.warned(lambda: self.ctls.count("c1")), 0)
        # re-pinned (fix round 2026-10-17, the same decision): a deleted probe
        # is found by identity only, never read nor warned for (it raised "c1
        # already deleted!"): the List's own element object matches, so a List
        # can drop it; another object of the deleted node matches nothing
        held = list.__getitem__(self.ctls, 1)
        self.assertIs(self.warned(lambda: self.c1 in self.ctls, count=0), False)
        self.assertIs(self.warned(lambda: held in self.ctls, count=0), True)
        self.assertEqual(self.warned(lambda: self.ctls.index(held), count=0), 1)
        self.warned(lambda: self.ctls.remove(self.c2))
        self.assertEqual(len(self.ctls), 2)
        self.warned(lambda: self.ctls.remove(held), count=0)
        self.assertEqual(list.__len__(self.ctls), 1)

    def test_a_plug_of_a_deleted_node(self):
        plugs = List([self.c0.tx, self.c0.ty, Plug("c2.tx")])
        cmds.delete("c0")
        with mock.patch.object(cmds, "warning") as warn:
            plugs << 7
        warn.assert_called_once_with(
            "List elements [0] c0.translateX, [1] c0.translateY were deleted; skipped"
        )
        self.assertEqual(cmds.getAttr("c2.tx"), 7)
        self.assertEqual(
            repr(plugs),
            'List([Plug("c0.translateX", deleted), Plug("c0.translateY", deleted), Plug("c2.translateX")])',
        )

    def test_one_undo_step(self):
        self.warned(lambda: self.ctls.tx << [1, 2, 3])
        self.assertEqual((cmds.getAttr("c0.tx"), cmds.getAttr("c2.tx")), (1, 3))
        cmds.undo()
        self.assertEqual((cmds.getAttr("c0.tx"), cmds.getAttr("c2.tx")), (0, 0))
        cmds.undo()  # the delete of c1
        self.assertEqual([str(x) for x in self.ctls], ["c0", "c1", "c2"])

    def test_a_freed_element(self):
        folder = tempfile.mkdtemp(prefix="rig_robust_list_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "s.ma").replace(os.sep, "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(path, open=True, force=True)  # c0 and c2 come back as new nodes
        live = List([self.c0, Node("c2")])
        with mock.patch.object(cmds, "warning") as warn:
            self.assertEqual(list(live.tx), [skip, Node("c2").tx])
            live.tx << 9
        self.assertEqual(warn.call_args_list[0][0][0], "List element [0] c0 was deleted; skipped")
        self.assertEqual((cmds.getAttr("c0.tx"), cmds.getAttr("c2.tx")), (0, 9))


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
    deleted!" before the read, and a List skips such elements (the List rule)."""

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
                self.assertEqual(result[f"{how} List"], {"returned List": 50})


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
    from rig.shade import Blinn
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
    cmds.delete("c1", "lay", "t0")
    out["faces"] = repr(faces)
    out["faces str"] = str(faces)
    out["token"] = repr(token)
    out["token str"] = str(token)
    out["List"] = repr(List([faces, token, named]))
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
    from rig.shade import Material
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
    """rig's thin wrappers that forward a user's List to maya.cmds (``node.delete(nodes)``,
    ``duplicate``, ``set_parent``, ``ObjectSet.add_members``, ``find_all``,
    ``DisplayLayer.create``, ``BlendShape.create``, ``rc.select``): Maya's parser reads it in
    rig's own frame, where a deleted node's str() raised (a crash); it reads the marked name
    there now (`journal.forward`), so Maya raises its own clean error, or, as the user's own
    cmds call would, acts on the elements that still name a node."""

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
                if "rc.select" in case:
                    # the bridge names each node in Python first: the clean error
                    self.assertRegex(outcome, r"^RuntimeError: .*already deleted!$")
                    continue
                self.assertNotIn("already deleted", outcome)
                if "duplicate" in case:
                    # Maya duplicates the node and ignores the names that match none
                    self.assertEqual(outcome, "returned list")
                elif "find_all" in case:
                    self.assertRegex(outcome, r"^RuntimeError: Syntax error")
                else:
                    self.assertRegex(outcome, r"^(TypeError|ValueError|RuntimeError): ")


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

        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(cmds.undoInfo, state=False)
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
    """The skip warning is printed once per statement (user decision 2026-10-16): a statement
    whose operations each skip the same element names it once; the next statement, or a
    loop's next pass, warns again."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(cmds.undoInfo, state=False)
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        self.ctls = List([Node(n) for n in names])
        cmds.delete("c1")

    def test_a_chained_statement_warns_once(self):
        ctls = self.ctls
        with mock.patch.object(cmds, "warning") as warn:
            ctls.tx << ctls.ty * 2 + ctls.tz
        self.assertEqual(warn.call_count, 1)
        with mock.patch.object(cmds, "warning") as warn:
            ctls.tx = ctls.ty
        self.assertEqual(warn.call_count, 1)
        with mock.patch.object(cmds, "warning") as warn:
            ctls.ty << ctls.tx.get()
        self.assertEqual(warn.call_count, 1)
        warn.assert_called_once_with("List element [1] c1 was deleted; skipped")

    def test_each_statement_and_each_pass_warns(self):
        ctls = self.ctls
        with mock.patch.object(cmds, "warning") as warn:
            ctls.tx << 1
            ctls.ty << 2
        self.assertEqual(warn.call_count, 2)
        with mock.patch.object(cmds, "warning") as warn:
            for value in range(3):
                ctls.tz << value
        self.assertEqual(warn.call_count, 3)

    def test_another_list_of_the_same_names_warns(self):
        def broadcast(ctls):
            ctls.tx << 1

        with mock.patch.object(cmds, "warning") as warn:
            for _ in range(2):
                cmds.file(new=True, force=True)
                names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
                ctls = List([Node(n) for n in names])
                cmds.delete("c1")
                broadcast(ctls)
        self.assertEqual(warn.call_count, 2)


class TestDeletedElementsFollowUps(MayaTestCase):
    """The review's follow-ups of the List rule: get() keeps a List when a slot is skipped, a
    deleted probe is found by identity, a nested List's deleted element is skipped by
    membership, a List with a deleted element hashes, a Components on a deleted shape is
    skipped."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.undoInfo(state=True, infinity=True)
        self.addCleanup(cmds.undoInfo, state=False)

    def test_get_keeps_a_list_with_skip(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        ctls = List([Node(n) for n in names])
        self.assertEqual(type(ctls.tx.get()).__name__, "ndarray")
        cmds.delete("c1")
        with mock.patch.object(cmds, "warning"):
            got = ctls.tx.get()
            self.assertIs(type(got), List)
            self.assertIs(got[1], skip)
            plus = got + 1
            self.assertEqual([plus[0], plus[2]], [1.0, 1.0])
            self.assertIs(plus[1], skip)
            ctls.ty << got
            self.assertIs(type(ctls.t.get()), List)

    def test_a_deleted_probe_is_found_by_identity(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        ctls = List([Node(n) for n in names])
        gone = list.__getitem__(ctls, 1)
        cmds.delete("c1")
        with mock.patch.object(cmds, "warning") as warn:
            self.assertIn(gone, ctls)
            self.assertEqual((ctls.index(gone), ctls.count(gone)), (1, 1))
            ctls.remove(gone)
        warn.assert_not_called()
        self.assertEqual([str(x) for x in ctls], ["c0", "c2"])

    def test_nested_lists_and_hash(self):
        names = [cmds.createNode("transform", name=f"c{i}") for i in range(3)]
        a, b, c = (Node(n) for n in names)
        nested = List([List([a, b]), List([c])])
        layer = Layer.define("L")
        cmds.delete("c0")
        with mock.patch.object(cmds, "warning") as warn:
            nested << layer
        warn.assert_called_once_with("List element [0][0] c0 was deleted; skipped")
        self.assertEqual(sorted(cmds.editDisplayLayerMembers("L", query=True)), ["c1", "c2"])
        self.assertIsInstance(hash(List([a, b])), int)
        self.assertEqual(len({List([a, b]): 1}), 1)

    def test_a_components_on_a_deleted_shape_is_skipped(self):
        cmds.polyCube(name="c1", ch=False)
        cmds.polyCube(name="c2", ch=False)
        faces = List([Node("c1Shape").f[:3], Node("c2Shape").f[0]])
        cmds.delete("c1")
        with mock.patch.object(cmds, "warning") as warn:
            faces << Tag("cap")
        warn.assert_called_once_with("List element [0] c1Shape.f[0:2] was deleted; skipped")
        self.assertEqual(list(Node("c2Shape").f >> Tag("cap")), [0])
        self.assertEqual(repr(faces), 'List([Components("c1Shape.f[0:2]", deleted), Components("|c2|c2Shape.f[0]")])')
        with self.assertRaisesRegex(RuntimeError, "already deleted!"):
            list.__getitem__(faces, 0).get_names()  # an operation on it raises
