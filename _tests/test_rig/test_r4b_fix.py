"""Round 4b, the FIX step: regression tests for the review findings.

* ``TestFailedValueLeavesNothing``: a create or define whose attribute value the
  new node refuses (``t=(1, 2)``) deletes every node the call made before its
  error propagates, inside and outside a container scope, with the undo queue on
  and off; the corrected re-run makes the node with its values; a create with
  attributes is one undo step.
* ``TestDefineGuards``: the define refusals that read nothing come before a
  plug-in load; a name a namespace has is refused before any write; the
  elsewhere hint names a define only when it would find the node; a re-run of a
  renamed scope names both containers.

Every refusal writes nothing (a zero ``cmds.ls()`` delta).
"""

from maya import cmds

from rig import container, Node
from rig.nodetypes import Blinn, DisplayLayer, Material, Transform
from rig._tests._base import MayaTestCase


def _scene():
    return set(cmds.ls())


class _Case(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def tearDown(self):
        cmds.undoInfo(state=True)
        cmds.namespace(relativeNames=False)
        cmds.namespace(setNamespace=":")
        super().tearDown()

    def assertLeavesNothing(self, error, call, pattern=""):
        """``call`` raises ``error`` (matching ``pattern``) and the scene is as
        it was."""
        before = _scene()
        with self.assertRaisesRegex(error, pattern):
            call()
        self.assertEqual(_scene() - before, set())
        self.assertEqual(before - _scene(), set())


class TestFailedValueLeavesNothing(_Case):
    """Review blockers (safety D1, attrs_undo AC/Y): no half-built node."""

    CALLS = {
        "Transform.define":   lambda: Transform.define("y", t=(1, 2)),
        "Transform.create":   lambda: Transform.create(name="c2", tx=1, visibility=[1, 2]),
        "Node.create typed":  lambda: Node.create("transform", name="c3", t=(1, 2)),
        "Node.create DG":     lambda: Node.create("multiplyDivide", name="md", input1=(1, 2)),
        "Node.define DG":     lambda: Node.define("multiplyDivide", "md", input1=(1, 2)),
        "Blinn.define":       lambda: Blinn.define("red", color=(1, 0)),
        "Blinn.create":       lambda: Blinn.create(name="red", color=(1, 0)),
        "Material.create":    lambda: Material.create(type="phong", name="ph", color=(1, 0)),
        "DisplayLayer.define": lambda: DisplayLayer.define("L", color=[1, 2]),
        "DisplayLayer.create": lambda: DisplayLayer.create(name="L", color=[1, 2]),
    }

    def _each(self):
        for label, call in self.CALLS.items():
            with self.subTest(label):
                self.assertLeavesNothing(Exception, call)

    def test_outside_a_scope(self):
        self._each()

    def test_with_the_undo_queue_off(self):
        cmds.undoInfo(state=False)
        self._each()

    def test_in_a_real_and_a_flattened_scope(self):
        with container("arm") as arm:
            Transform.define("first")
            self._each()
            with container("inner"):
                self._each()
        self.assertEqual(cmds.container(str(arm), query=True, nodeList=True), ["first"])

    def test_the_corrected_rerun_makes_it_with_its_values(self):
        with container("arm"):
            self.assertLeavesNothing(Exception, lambda: Transform.define("arm_root", t=(1, 2)))
        cmds.delete("arm")
        with container("arm"):
            root = Transform.define("arm_root", t=(1, 2, 3))
        self.assertEqual(cmds.getAttr(f"{root}.t"), [(1.0, 2.0, 3.0)])
        self.assertEqual(cmds.container("arm", query=True, nodeList=True), ["arm_root"])
        self.assertLeavesNothing(Exception, lambda: Blinn.define("red", color=(1, 0)))
        red = Blinn.define("red", color=(1, 0, 0))
        self.assertEqual(cmds.getAttr(f"{red}.color"), [(1.0, 0.0, 0.0)])

    def test_the_undo_of_the_failed_call_changes_nothing(self):
        cmds.createNode("transform", name="user_node")
        before = _scene()
        self.assertLeavesNothing(Exception, lambda: Transform.define("y", t=(1, 2)))
        cmds.undo()   # the failed define's own step: made and deleted
        self.assertEqual(_scene(), before)
        cmds.redo()
        self.assertEqual(_scene(), before)

    def test_a_create_with_attributes_is_one_undo_step(self):
        cmds.flushUndo()
        for label, call in {
            "Transform.create": lambda: Transform.create(name="t", tx=1, ty=2, rotateOrder="xzy"),
            "Node.create typed": lambda: Node.create("transform", name="t", tx=1, ty=2),
            "Node.create DG":    lambda: Node.create("multiplyDivide", name="md", input1X=3, operation="divide"),
            "DisplayLayer.create": lambda: DisplayLayer.create(name="L", displayType=2, visibility=False),
        }.items():
            with self.subTest(label):
                before = _scene()
                call()
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.create")
                cmds.undo()
                self.assertEqual(_scene(), before)


class TestDefineGuards(_Case):
    """Review minors (safety D7, D8, R4, R10)."""

    def test_the_name_checks_come_before_a_plugin_load(self):
        if cmds.pluginInfo("matrixNodes", query=True, loaded=True):
            try:
                cmds.unloadPlugin("matrixNodes")
            except RuntimeError:
                self.skipTest("matrixNodes is in use and cannot be unloaded")
        try:
            self.assertLeavesNothing(ValueError, lambda: Node.define("decomposeMatrix", "1bad"),
                                     "is not a name Maya keeps as written")
            self.assertLeavesNothing(TypeError, lambda: Node.define("decomposeMatrix", "d", n="x"),
                                     "takes its name after the type")
            self.assertFalse(cmds.pluginInfo("matrixNodes", query=True, loaded=True))
            self.assertEqual(Node.define("decomposeMatrix", "d").node_type, "decomposeMatrix")
        finally:
            cmds.loadPlugin("matrixNodes", quiet=True)

    def test_a_name_a_namespace_has_is_refused_before_any_write(self):
        cmds.namespace(add="rig")
        self.assertLeavesNothing(
            ValueError, lambda: Transform.define("rig"),
            r"^Transform\.define\('rig'\): 'rig' is the name of a namespace; Maya would rename "
            r"a new transform \(rig1\): pick another name$",
        )
        self.assertLeavesNothing(ValueError, lambda: Node.define("multiplyDivide", "rig"), "name of a namespace")
        cmds.namespace(add="sub", parent="rig")
        self.assertLeavesNothing(ValueError, lambda: Transform.define("rig:sub"), "'rig:sub' is the name of a namespace")
        self.assertEqual(Transform.define("rig:x").name, "rig:x")

    def test_the_elsewhere_hint_names_a_define_only_when_it_finds_the_node(self):
        cmds.polyCube(name="box", constructionHistory=False)
        self.assertLeavesNothing(
            ValueError, lambda: Transform.define("boxShape"),
            r"^Transform\.define\('boxShape'\): 'boxShape' exists at \|box\|boxShape; "
            r"Node\('boxShape'\) refers to it$",
        )
        cmds.createNode("transform", name="t", parent="box")
        self.assertLeavesNothing(
            ValueError, lambda: Transform.define("t"),
            r"Transform\('t'\) refers to it, and Transform\.define\('t', parent='box'\) keys it there$",
        )

    def test_a_rerun_of_a_renamed_scope_names_both_containers(self):
        cmds.createNode("transform", name="arm")   # run 1's container becomes arm1

        def build():
            with container("arm"):
                return Transform.define("arm_root")

        build()
        with self.assertRaisesRegex(
            ValueError,
            r"^'arm_root' belongs to container 'arm1' from an earlier run; this scope is 'arm2'\. "
            r"Delete 'arm1' and 'arm2' to rebuild it, or build in a new scene\.$",
        ):
            build()
