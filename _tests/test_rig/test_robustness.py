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
  or a plug of one), with ONE warning per operation naming it; a derived List holds rig.skip in
  its slot, which every List operation leaves alone (chained broadcasts stay aligned).
* `TestRawCmdsGivenADeletedList`: maya.cmds given a List whose first element is a deleted or
  freed node raises a clean error instead of crashing Maya (select / delete / setAttr / xform /
  parent; deleted to the undo queue, after a flushUndo, a new scene, a file open), in its own
  mayapy so a crash fails the test instead of the run.

Negative control: every test fails on b42933c (runs/features/robust/impl/negctl).
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
        with self.assertRaisesRegex(RuntimeError, r"^c1 already deleted!$"):
            self.c1 in self.ctls  # the probe itself is the deleted node
        self.warned(lambda: self.ctls.remove(self.c2))
        self.assertEqual(len(self.ctls), 2)

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
