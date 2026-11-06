"""A literal operand survives a save and a reopen on Maya 2024+ (2026-11-05).

Maya does not save an element of a numeric array that holds its default, 0.
The 2024+ ``multiply`` / ``average`` / ``max`` / ``min`` nodes hold their
operands in ``input[]``, so ``x * 0`` read ``x`` once the scene was saved and
opened again, and so did ``avg([x, 0])``, ``max([x, 0])`` and ``min([x, 0])``.
A literal operand now goes to a node with plain attributes: ``x * lit`` is a
``lerp`` (input1 = 0, input2 = x, weight = lit: exactly ``x * lit``),
``avg`` a ``lerp`` and ``max`` / ``min`` a ``clampRange``. ``x + 0`` keeps its
``sum``: a lost 0 adds nothing.
"""

import math
import os
import shutil
import tempfile

from maya import cmds
from rig import force_nodes, functions as f, Node
from rig._internal.maya_version import get_maya_version
from rig._tests._base import MayaTestCase

R      = math.radians(1.0)
XS     = (-7.25, 0.0, 1234.56789)
Y      = 3.5
ARRAYS = ("multiply", "average", "max", "min")

# label: (build from the plugs x, y; the same in Python; exact; the result's node type)
CASES = {
    "x_times_0":    (lambda x, y: x * 0.0,              lambda x, y: x * 0.0,           True,  "lerp"),
    "0_times_x":    (lambda x, y: 0.0 * x,              lambda x, y: 0.0 * x,           True,  "lerp"),
    "x_times_half": (lambda x, y: x * 0.5,              lambda x, y: x * 0.5,           True,  "lerp"),
    "x_times_r":    (lambda x, y: x * R,                lambda x, y: x * R,             True,  "lerp"),
    "r_times_x":    (lambda x, y: R * x,                lambda x, y: R * x,             True,  "lerp"),
    "x_times_y":    (lambda x, y: x * y,                lambda x, y: x * y,             True,  "multiply"),
    "avg_x_0":      (lambda x, y: f.avg([x, 0]),        lambda x, y: x / 2,             False, "lerp"),
    "avg_x_y_0":    (lambda x, y: f.avg([x, y, 0]),     lambda x, y: (x + y) / 3,       False, "lerp"),
    "avg_x_0_3":    (lambda x, y: f.avg([x, 0, 3]),     lambda x, y: (x + 3) / 3,       False, "lerp"),
    "max_x_0":      (lambda x, y: f.max([x, 0]),        lambda x, y: max(x, 0),         True,  "clampRange"),
    "min_x_0":      (lambda x, y: f.min([x, 0]),        lambda x, y: min(x, 0),         True,  "clampRange"),
    "max_x_y_0":    (lambda x, y: f.max([x, y, 0]),     lambda x, y: max(x, y, 0),      True,  "clampRange"),
    "min_x_0_y_m2": (lambda x, y: f.min([x, 0, y, -2]), lambda x, y: min(x, 0, y, -2),  True,  "clampRange"),
    "x_plus_0":     (lambda x, y: x + 0,                lambda x, y: x + 0,             True,  "sum"),
}


class _LiteralCase(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        if get_maya_version() < 2024:
            self.skipTest("the array-input math nodes ship with Maya 2024+")
        super().setUp()

    def _round_trip(self, file_type, ext):
        """Save the scene as ``file_type`` in a temporary folder and open it again."""
        folder = tempfile.mkdtemp(prefix="rig_literal_save_")
        self.addCleanup(shutil.rmtree, folder, True)
        self.addCleanup(cmds.file, new=True, force=True)
        path = os.path.join(folder, "literal." + ext).replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type=file_type, force=True)
        cmds.file(path, open=True, force=True)

    def _drive(self, name, src):
        """A transform ``name`` whose tx reads ``src``."""
        Node(cmds.createNode("transform", name=name)).tx << src


class TestALiteralSurvivesTheSave(_LiteralCase):
    def _build(self):
        x = Node(cmds.createNode("transform", name="src")).ty
        y = Node(cmds.createNode("transform", name="src2")).ty
        cmds.setAttr("src2.ty", Y)
        results = {}
        for label, (build, _python, _exact, _node_type) in CASES.items():
            results[label] = build(x, y)
            self._drive("dst_" + label, results[label])
        return results

    def _check_values(self):
        for x in XS:
            cmds.setAttr("src.ty", x)
            for label, (_build, python, exact, _node_type) in CASES.items():
                with self.subTest(case=label, x=x):
                    got  = cmds.getAttr(f"dst_{label}.tx")
                    want = python(x, Y)
                    if exact:
                        self.assertEqual(got, want)
                    else:
                        self.assertAlmostEqual(got, want, places=9)

    def test_the_values_before_a_save(self):
        self._build()
        self._check_values()

    def test_the_values_after_a_reopen_of_a_maya_ascii_file(self):
        self._build()
        self._round_trip("mayaAscii", "ma")
        self._check_values()

    def test_the_values_after_a_reopen_of_a_maya_binary_file(self):
        self._build()
        self._round_trip("mayaBinary", "mb")
        self._check_values()

    def test_the_node_types(self):
        for label, result in self._build().items():
            with self.subTest(case=label):
                self.assertEqual(cmds.nodeType(str(result).split(".")[0]), CASES[label][3])

    def test_a_product_with_a_literal_is_a_lerp_named_for_the_multiply(self):
        x    = Node(cmds.createNode("transform", name="src")).ty
        node = str(x * 2).split(".")[0]
        self.assertEqual((cmds.nodeType(node), node), ("lerp", "mul1"))
        self.assertEqual(cmds.getAttr(node + ".input1"), 0.0)
        self.assertEqual(cmds.listConnections(node + ".input2", source=True, plugs=True), ["src.translateY"])
        self.assertEqual(cmds.getAttr(node + ".weight"), 2.0)

    def test_no_literal_is_left_in_an_array_input(self):
        self._build()
        nodes = cmds.ls(type=list(ARRAYS))
        self.assertTrue(nodes)  # x * y, and the plugs' own average / max / min
        for node in nodes:
            for index in cmds.getAttr(node + ".input", multiIndices=True) or []:
                with self.subTest(element=f"{node}.input[{index}]"):
                    self.assertTrue(cmds.connectionInfo(f"{node}.input[{index}]", isDestination=True))


class TestAllLiteralsUnderForceNodes(_LiteralCase):
    """With no plug at all (``force_nodes()``), the first literal is the input."""

    def test_the_values_after_a_reopen(self):
        with force_nodes():
            self._drive("dst_max", f.max([1.0, 5.0, 3.0]))
            self._drive("dst_min", f.min([0.0, 5.0, 3.0]))
            self._drive("dst_avg", f.avg([0.0, 4.0]))
        for name, node_type in (("dst_max", "clampRange"), ("dst_min", "clampRange"), ("dst_avg", "lerp")):
            src = cmds.listConnections(name + ".tx", source=True, destination=False)[0]
            self.assertEqual(cmds.nodeType(src), node_type)
        self._round_trip("mayaAscii", "ma")
        self.assertEqual(cmds.getAttr("dst_max.tx"), 5.0)
        self.assertEqual(cmds.getAttr("dst_min.tx"), 0.0)
        self.assertEqual(cmds.getAttr("dst_avg.tx"), 2.0)


class TestAWorkingUnitOtherThanCentimetres(_LiteralCase):
    """The ``lerp`` weight of an average is a ratio: written so the attribute
    holds it as it is, where a distance (an operand, a bound) is written in the
    scene's working unit."""

    def test_a_metre_scene(self):
        unit = cmds.currentUnit(query=True, linear=True)
        self.addCleanup(cmds.currentUnit, linear=unit)
        cmds.currentUnit(linear="m")
        x = Node(cmds.createNode("transform", name="src")).ty
        y = Node(cmds.createNode("transform", name="src2")).ty
        cmds.setAttr("src.ty", 0.5)
        cmds.setAttr("src2.ty", 3.5)
        self._drive("dst_avg",  f.avg([x, 1]))
        self._drive("dst_avg3", f.avg([x, y, 1]))
        self._drive("dst_max",  f.max([x, 1]))
        self._drive("dst_min",  f.min([x, 1]))
        for label in ("before the save", "after the reopen"):
            if label == "after the reopen":
                self._round_trip("mayaAscii", "ma")
            with self.subTest(label):
                self.assertEqual(cmds.currentUnit(query=True, linear=True), "m")
                self.assertAlmostEqual(cmds.getAttr("dst_avg.tx"),  0.75,      places=9)
                self.assertAlmostEqual(cmds.getAttr("dst_avg3.tx"), 5.0 / 3.0, places=9)
                self.assertAlmostEqual(cmds.getAttr("dst_max.tx"),  1.0,       places=9)
                self.assertAlmostEqual(cmds.getAttr("dst_min.tx"),  0.5,       places=9)
