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


class TestOneLookupRule(_Case):
    """Review majors (safety): a name reads one way whatever the verb, the
    current namespace and ``namespace -relativeNames``."""

    def _char(self):
        cmds.namespace(add="char")

    def test_a_clone_onto_the_plugs_own_node_never_looks_it_up(self):
        from rig import AmbiguousNodeError

        cmds.createNode("transform", name="cube")
        self._char()
        cmds.createNode("transform", name="char:cube")
        cmds.namespace(setNamespace="char")
        for relative in (False, True):
            with self.subTest(relativeNames=relative):
                cmds.namespace(relativeNames=relative)
                cube = Node(":cube")
                plug = cube.tx >> f"k{int(relative)}"
                self.assertIs(plug.node, cube)
                self.assertTrue(cmds.attributeQuery(f"k{int(relative)}", node=":cube", exists=True))
                self.assertFalse(cmds.attributeQuery(f"k{int(relative)}", node=":char:cube", exists=True))
        cmds.namespace(relativeNames=False)
        # a user-written 'node.name' is the user's name: the lookup rule reads it
        before = _scene()
        with self.assertRaises(AmbiguousNodeError):
            Node(":cube").tx >> "cube.k2"
        self.assertEqual(_scene(), before)

    def test_a_qualified_name_is_read_from_the_root(self):
        from rig import NodeNotFoundError

        cmds.namespace(add="sub")
        cmds.createNode("transform", name="sub:x")
        self._char()
        cmds.namespace(add="sub", parent="char")
        cmds.createNode("transform", name="char:sub:x")
        root, inner = cmds.ls(":sub:x", uuid=True)[0], cmds.ls(":char:sub:x", uuid=True)[0]
        cmds.namespace(setNamespace="char")
        for relative in (False, True):
            with self.subTest(relativeNames=relative):
                cmds.namespace(relativeNames=relative)
                before = _scene()
                self.assertEqual(Transform("sub:x").uuid, root)
                self.assertEqual(Node("sub:x").uuid, root)
                self.assertEqual(Node("sub:x.tx").uuid, root)
                self.assertEqual(Transform.define("sub:x").uuid, root)
                self.assertTrue(Transform.exists("sub:x"))
                self.assertEqual(Transform(":char:sub:x").uuid, inner)
                self.assertEqual(_scene(), before)
        cmds.namespace(relativeNames=False)
        cmds.delete(":sub:x")
        cmds.namespace(relativeNames=True)
        # only :char:sub:x: 'sub:x' is :sub:x, which does not exist
        with self.assertRaises(NodeNotFoundError):
            Transform("sub:x")
        self.assertFalse(Transform.exists("sub:x"))
        self.assertEqual(Transform("char:sub:x").uuid, inner)

    def test_create_reads_parent_by_the_reference_rule(self):
        from rig import AmbiguousNodeError, NodeNotFoundError
        from rig.bridges import nodes as rn

        self._char()
        cmds.createNode("transform", name="char:grp")
        cmds.namespace(setNamespace="char")
        # only :char:grp: every create parents under it, as define does
        for label, call in {
            "Transform.create":  lambda: Transform.create(name="a", parent="grp"),
            "Node.create":       lambda: Node.create("transform", name="b", parent="grp"),
            "rn.transform":      lambda: rn.transform(name="c", parent="grp"),
            "container.createNode": lambda: container.createNode("transform", name="d", parent="grp"),
            "Transform.define":  lambda: Transform.define("e", parent="grp"),
        }.items():
            with self.subTest(label):
                self.assertEqual(call().get_parent().long_name, "|char:grp")
        cmds.createNode("transform", name=":grp")
        # :grp and :char:grp: every create refuses, as define does
        for label, call in {
            "Transform.create":  lambda: Transform.create(name="a", parent="grp"),
            "Node.create":       lambda: Node.create("transform", name="b", parent="grp"),
            "rn.transform":      lambda: rn.transform(name="c", parent="grp"),
            "container.createNode": lambda: container.createNode("transform", name="d", parent="grp"),
            "Transform.define":  lambda: Transform.define("e2", parent="grp"),
        }.items():
            with self.subTest(label):
                self.assertLeavesNothing(AmbiguousNodeError, call)
        cmds.namespace(setNamespace=":")
        # a parent= typo raises, where Maya made the node at the world with a warning
        for label, call in {
            "rn.transform":      lambda: rn.transform(name="c", parent="gpr"),
            "rn.joint":          lambda: rn.joint(name="j", parent="gpr"),
            "container.createNode": lambda: container.createNode("transform", name="k", p="gpr"),
            "Transform.create":  lambda: Transform.create(name="t", parent="gpr"),
        }.items():
            with self.subTest(label):
                self.assertLeavesNothing(NodeNotFoundError, call, "no DAG node named 'gpr'")
        self.assertEqual(rn.transform(name="ok", parent=Node(":grp")).get_parent().long_name, "|grp")

    def test_container_create_node_refuses_shared(self):
        cmds.createNode("multiplyDivide", name="md")
        for _ in range(2):
            self.assertLeavesNothing(
                TypeError, lambda: container.createNode("multiplyDivide", name="md", shared=True),
                "create always makes a new node",
            )


class TestRerunInANamespace(_Case):
    """FIX probe (a build with a namespace current): the re-run of a scope
    made in the current namespace ('char:arm') gets the re-run text."""

    def test_the_rerun_text(self):
        cmds.namespace(add="char")
        cmds.namespace(setNamespace="char")

        def build():
            with container("arm"):
                return Transform.define("arm_root")

        build()
        with self.assertRaisesRegex(
            ValueError,
            r"^'char:arm_root' belongs to container 'char:arm' from an earlier run; this scope is "
            r"'char:arm1'\. Delete 'char:arm' and 'char:arm1' to rebuild it",
        ):
            build()



def _q(plug, flag):
    return cmds.addAttr(plug, query=True, **{flag: True})


def _inputs(plug):
    return cmds.listConnections(plug, source=True, destination=False, plugs=True) or []


class TestRedeclareFixes(_Case):
    """Review blockers and majors (attrs_undo, completeness): re-declaring an
    attribute checks everything before any edit and is one undo step that
    restores everything."""

    def setUp(self):
        super().setUp()
        self.node = Node.create("transform", name="rd")
        self.drv  = Node.create("transform", name="drv")
        cmds.setAttr("drv.tx", 7)

    def _state(self, plug):
        node, attr = plug.split(".", 1)
        return (
            cmds.getAttr(plug), _inputs(plug), _q(plug, "minValue"), _q(plug, "maxValue"),
            _q(plug, "defaultValue"), cmds.attributeQuery(attr, node=node, niceName=True),
            cmds.getAttr(plug, keyable=True), _q(plug, "hidden"), set(cmds.ls()),
        )

    def test_an_unknown_keyword_is_refused_when_the_spec_is_made(self):
        from rig.spec import Float, Vector

        for kwargs, text in (
            ({"niceNmae": "W"},    r"^Float\('qd'\): niceNmae= is not an addAttr flag \(did you mean niceName=\?\); nothing was changed$"),
            ({"update": True},     r"update= is not an addAttr flag; re-declaring an attribute applies the settings you pass"),
            ({"channelBox": True}, r"channelBox= is not an addAttr flag"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(TypeError, text):
                    Float("qd", **kwargs)
        # overwrite=True never deletes an attribute it cannot add again
        self.node << Float("qd") << 1
        self.node.qd << self.drv.tx
        self.node << Vector("qv", dv=[1, 2, 3])
        self.node.qvY << self.drv.tx
        before = (self._state("rd.qd"), cmds.getAttr("rd.qv"), _inputs("rd.qvY"))
        for spec in (
            lambda: Vector("qv", dv=["a", "b", "c"], overwrite=True),
            lambda: Float("qd", min="0", overwrite=True),
            lambda: Float("qd", dv="x", overwrite=True),
        ):
            with self.assertRaisesRegex(TypeError, "is not a number"):
                self.node << spec()
        self.assertEqual((self._state("rd.qd"), cmds.getAttr("rd.qv"), _inputs("rd.qvY")), before)

    def test_a_default_edit_undoes_and_redoes(self):
        from rig.spec import Angle, Enum, Float

        self.node << Float("a", dv=5)
        self.node << Float("rg", dv=1, min=0, max=10) << 4
        self.node << Enum("e", en="a:b:c", dv=1)
        self.node << Angle("ang", dv=0.5)
        cmds.flushUndo()
        for spec, before, after in (
            (Float("a", dv=3),                5.0, 3.0),
            (Float("rg", dv=6, min=5, max=20), 1.0, 6.0),
            (Enum("e", dv="c"),               1.0, 2.0),
            (Angle("ang", dv=1.5),            0.5, 1.5),
        ):
            plug = f"rd.{spec.kargs['longName']}"
            with self.subTest(plug):
                value = cmds.getAttr(plug)
                self.node << spec
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.attr")
                self.assertAlmostEqual(_q(plug, "defaultValue"), after)
                cmds.undo()
                self.assertAlmostEqual(_q(plug, "defaultValue"), before)
                cmds.redo()
                self.assertAlmostEqual(_q(plug, "defaultValue"), after)
                self.assertEqual(cmds.getAttr(plug), value)
        # the range moved with the default, and comes back with it
        for _ in range(3):
            cmds.undo()
        self.assertEqual((_q("rd.rg", "minValue"), _q("rd.rg", "maxValue"), _q("rd.rg", "defaultValue")), (0, 10, 1))

    def test_one_redeclaration_is_one_undo_step(self):
        from rig.spec import Float, Vector

        self.node << Float("qw", max=3) << 2
        self.node.qw << self.drv.tx
        self.node << Vector("qv")
        cmds.flushUndo()
        for spec, plug in (
            (Float("qw", max=20, nn="After", hidden=True, k=False), "rd.qw"),
            (Vector("qv", max=20, k=False, nn="V", hidden=True), "rd.qvX"),
        ):
            with self.subTest(plug):
                before = self._state(plug)
                self.node << spec
                self.assertNotEqual(self._state(plug), before)
                self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.attr")
                cmds.undo()
                self.assertEqual(self._state(plug), before)

    def test_a_compound_redeclared_is_the_compound_created(self):
        from rig.spec import Vector

        def look(name):
            parts = [name] + [f"{name}{axis}" for axis in "XYZ"]
            return [
                (_q(f"rd.{p}", "hidden"), cmds.attributeQuery(p, node="rd", niceName=True),
                 cmds.getAttr(f"rd.{p}", keyable=True))
                for p in parts
            ]

        self.node << Vector("qv", hidden=True, nn="Vee", k=False)
        self.node << Vector("qv", hidden=False, nn="Other", k=True)
        self.node << Vector("fresh", hidden=False, nn="Other", k=True)
        self.assertEqual(look("qv"), [(h, n.replace("Other", "Other"), k) for h, n, k in look("fresh")])
        self.node << Vector("qv", hidden=True)
        self.node << Vector("qv", hidden=False)
        self.assertEqual(
            sorted(cmds.listAttr("rd", keyable=True, visible=True, userDefined=True) or []),
            ["fresh", "freshX", "freshY", "freshZ", "qv", "qvX", "qvY", "qvZ"],
        )

    def test_an_alias_or_a_published_name_is_refused(self):
        from rig import set_options
        from rig.spec import Float

        net = Node.create("network", name="net")
        net << Float("qw", max=3) << 2
        cmds.aliasAttr("smile", "net.qw")
        self.assertLeavesNothing(TypeError, lambda: net << Float("smile", max=9),
                                 r"^'net\.smile' is an alias of net\.qw; re-declare qw, or pick another name$")
        self.assertEqual(_q("net.qw", "maxValue"), 3)
        base = cmds.polyCube(name="base")[0]
        tgt  = cmds.polyCube(name="tgt")[0]
        bs   = Node(cmds.blendShape(tgt, base, name="bs")[0])
        self.assertLeavesNothing(TypeError, lambda: bs << Float("tgt"), r"'bs\.tgt' is an alias of bs\.weight\[0\]")
        set_options(flatten_containers=False)
        try:
            with container("outer"):
                inner = Node.create("transform", name="cube1")
                inner << Float("weight", dv=0.5, max=1) << 0.7
                container.publish_input(inner.weight, "weight")
            outer = Node("outer")
            attrs = cmds.listAttr("outer", userDefined=True)
            self.assertLeavesNothing(TypeError, lambda: outer << Float("weight", max=5),
                                     r"^'outer\.weight' is a name the container publishes")
            self.assertEqual(cmds.listAttr("outer", userDefined=True), attrs)
            self.assertEqual(_q("cube1.weight", "maxValue"), 1)
        finally:
            set_options(flatten_containers=True)

    def test_enum_forms_refuse_a_repeated_value_or_name(self):
        from rig.spec import Enum

        for en, text in (
            ({"a": 1, "b": 1},      r"fields 'a' and 'b' both have the value 1; Maya keeps only the first"),
            (["red", ("green", 0)], r"fields 'red' and 'green' both have the value 0"),
            (["x", ("y", 0), "z"],  r"fields 'x' and 'y' both have the value 0"),
            ([("a", 1), ("a", 2)],  r"names the field 'a' twice"),
        ):
            with self.subTest(en=en):
                with self.assertRaisesRegex(TypeError, text):
                    Enum("e", en=en)
        self.assertEqual(Enum("e", en={"a": 1, "b": 5}).kargs["en"], "a=1:b=5")
        self.assertEqual(Enum("e", en=["red", ("green", 5), "blue"]).kargs["en"], "red:green=5:blue")

    def test_an_enum_default_reads_the_attributes_fields(self):
        from rig.spec import Enum

        self.node << Enum("mode", en="a:b:c") << 2
        self.node << Enum("mode", dv="b")
        self.assertEqual((_q("rd.mode", "defaultValue"), cmds.getAttr("rd.mode")), (1, 2))
        before = self._state("rd.mode")
        for spec, text in (
            (lambda: Enum("mode", dv=9),   r"'rd\.mode' dv: 9 is not the value of one of its enum fields: a=0, b=1, c=2"),
            (lambda: Enum("mode", dv="z"), r"'rd\.mode' dv: 'z' is not one of its enum fields"),
        ):
            with self.assertRaisesRegex(TypeError, text):
                self.node << spec()
        self.assertEqual(self._state("rd.mode"), before)
        # on creation: against en= when the spec is made, else the default fields
        with self.assertRaisesRegex(TypeError, r"7 is not the value of one of its enum fields: a=0, b=1, c=2"):
            Enum("ex", en="a:b:c", dv=7)
        self.assertLeavesNothing(TypeError, lambda: self.node << Enum("ey", dv="b"), r"'b' is not one of its enum fields")
        self.assertFalse(cmds.attributeQuery("ey", node="rd", exists=True))
        self.node << Enum("ez", dv="True")
        self.assertEqual(_q("rd.ez", "defaultValue"), 1)

    def test_a_list_checks_every_element_first(self):
        from rig import List
        from rig.spec import Enum, Float

        na, nb = Node.create("network", name="na"), Node.create("network", name="nb")
        na << Float("qw", max=3)
        nb << Enum("qw", en="a:b")
        before = (self._state("na.qw"), cmds.addAttr("nb.qw", query=True, enumName=True))
        with self.assertRaisesRegex(TypeError, r"'nb\.qw' exists as an enum, not a double"):
            List([na, nb]) << Float("qw", max=9, nn="Nine")
        self.assertEqual((self._state("na.qw"), cmds.addAttr("nb.qw", query=True, enumName=True)), before)

    def test_plug_clone_onto_a_taken_name_is_refused(self):
        from rig.spec import Enum, Float

        src, dst = Node.create("network", name="src"), Node.create("network", name="dst")
        src << Enum("qn", en="x:y:z")
        dst << Enum("qn", en="p:q") << 1
        src << Float("qm")
        dst << Enum("qm", en="a:b")
        for plug in ("qn", "qm"):
            with self.subTest(plug):
                self.assertLeavesNothing(
                    TypeError, lambda: getattr(src, plug) >> dst,
                    rf"^'dst' already has an attribute '{plug}': '>>' clones, it never overwrites",
                )
        self.assertEqual(cmds.addAttr("dst.qn", query=True, enumName=True), "p:q")
        self.assertEqual(cmds.getAttr("dst.qn"), 1)
        self.assertEqual(str(src.qm >> Node.create("network", name="free")), "free.qm")

    def test_the_soft_range_is_checked(self):
        from rig.spec import Float

        self.node << Float("qw", min=0, max=10)
        before = self._state("rd.qw")
        for kwargs, text in (
            ({"smn": 6, "smx": 2}, r"softMinValue=6 is above softMaxValue=2"),
            ({"smn": -5},          r"softMinValue=-5 is below min=0"),
            ({"smx": 50},          r"softMaxValue=50 is above max=10"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(TypeError, text):
                    self.node << Float("qw", **kwargs)
                self.assertEqual(self._state("rd.qw"), before)
        self.node << Float("qw", smn=2, smx=8)
        self.assertEqual((_q("rd.qw", "softMinValue"), _q("rd.qw", "softMaxValue")), (2, 8))

    def test_time_is_a_time_attribute(self):
        from rig.spec import Time

        plug = self.node << Time("qt", dv=3)
        self.assertEqual(cmds.getAttr("rd.qt", type=True), "time")
        self.assertEqual(str(plug), "rd.qt")
        self.node << Time("qt", max=48)
        self.assertEqual(_q("rd.qt", "maxValue"), 48)
        clone = plug >> Node.create("network", name="other")
        self.assertEqual(cmds.getAttr(str(clone), type=True), "time")


    def test_a_referenced_attribute_keeps_its_keyable_and_hidden_state(self):
        import os
        import shutil
        import tempfile

        from rig.spec import Float

        folder = tempfile.mkdtemp(prefix="r4b_fix_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(cmds.file, new=True, force=True)
        net = Node.create("network", name="net")
        net << Float("qk", max=3) << 2
        src = os.path.join(folder, "src.ma").replace("\\", "/")
        cmds.file(rename=src)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(src, reference=True, namespace="ref")
        ref   = Node("ref:net")
        edits = cmds.referenceQuery("refRN", editStrings=True)
        for kwargs, text in (
            ({"max": 9, "nn": "Nine", "k": False}, r"its keyable state is set in that file"),
            ({"max": 8, "hidden": True},          r"its hidden flag is set in that file"),
        ):
            with self.subTest(kwargs=kwargs):
                before = self._state("ref:net.qk")
                with self.assertRaisesRegex(TypeError, rf"^'ref:net\.qk' comes from a referenced file: {text}"):
                    ref << Float("qk", **kwargs)
                self.assertEqual(self._state("ref:net.qk"), before)
                self.assertEqual(cmds.referenceQuery("refRN", editStrings=True), edits)
        # the settings Maya keeps as reference edits still apply
        ref << Float("qk", max=9)
        self.assertEqual(_q("ref:net.qk", "maxValue"), 9)
        # an attribute added in this scene is the scene's: its keyable state changes
        ref << Float("mine")
        ref << Float("mine", k=False)
        self.assertFalse(cmds.getAttr("ref:net.mine", keyable=True))

    def test_a_new_max_undoes_and_redoes(self):
        from rig.spec import Float

        self.node << Float("qw") << 4
        cmds.flushUndo()
        self.node << Float("qw", max=8)
        self.assertEqual((_q("rd.qw", "hasMaxValue"), _q("rd.qw", "maxValue")), (True, 8))
        cmds.undo()
        self.assertFalse(_q("rd.qw", "hasMaxValue"))
        cmds.redo()
        self.assertEqual((_q("rd.qw", "hasMaxValue"), _q("rd.qw", "maxValue")), (True, 8))
        self.assertEqual(cmds.getAttr("rd.qw"), 4)



def _sets(member):
    return sorted(cmds.listSets(object=member, type=1) or [])


class TestMaterialEngines(_Case):
    """Review majors (membership M1, M2) and a minor (M9): every Material.of
    answer goes back through ``in`` and ``<<``; a read never raises where a
    shader feeds several engines; an engine node is exactly that engine."""

    def setUp(self):
        super().setUp()
        from rig.bridges import nodes as rn
        from rig.nodetypes import ShadingEngine

        self.cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        self.red  = Blinn.define("red")
        self.cube << self.red
        alt = ShadingEngine.create(name="altSG")
        alt.set_material(self.red)
        self.cube.f[1] << alt
        ShadingEngine.create(name="rampSG")
        rn.ramp(name="ramp1")
        cmds.connectAttr("ramp1.outColor", "rampSG.surfaceShader")
        cmds.sets("cube.f[2]", edit=True, forceElement="rampSG")
        ShadingEngine.create(name="emptySG")
        cmds.sets("cube.f[3]", edit=True, forceElement="emptySG")

    def test_every_answer_round_trips(self):
        from rig.nodetypes import ShadingEngine
        from rig.shade import Default  # noqa: F401

        expected = {0: [self.red], 1: [ShadingEngine("altSG")], 2: [ShadingEngine("rampSG")],
                    3: [ShadingEngine("emptySG")]}
        for i in range(6):
            face    = self.cube.f[i]
            answers = Material.of(face)
            with self.subTest(face=i):
                self.assertEqual(answers, expected.get(i, [self.red]))
                engines = _sets(f"cube.f[{i}]")
                for answer in answers:
                    self.assertIn(face, answer)
                    other = Node(cmds.polyCube(name=f"o{i}", constructionHistory=False)[0])
                    other << answer
                    self.assertEqual(_sets(f"o{i}Shape"), engines)
                    face << -answer
                    self.assertEqual(_sets(f"cube.f[{i}]"), [])
        # the purge takes the faces out of every engine, the shaderless one too
        cmds.sets("cube.f[1]", edit=True, forceElement="altSG")
        cmds.sets("cube.f[2]", edit=True, forceElement="rampSG")
        cmds.sets("cube.f[3]", edit=True, forceElement="emptySG")
        self.cube << Material()
        for i in range(6):
            self.assertEqual(_sets(f"cube.f[{i}]"), [])

    def test_a_read_looks_at_every_engine_the_shader_feeds(self):
        self.assertIn(self.cube.f[1], self.red)
        self.assertEqual(list(self.cube.f[:2] >> self.red), [0, 1])
        self.cube.f[1] << -self.red
        self.assertEqual(_sets("cube.f[1]"), [])
        # a shader over two engines, neither named <shader>SG: reads answer,
        # '<<' still needs the engine named
        from rig.nodetypes import ShadingEngine

        blue = Blinn.create(name="blue")
        cmds.delete("blueSG")
        for name in ("b1SG", "b2SG"):
            ShadingEngine.create(name=name).set_material(blue)
        cmds.sets("cube.f[4]", edit=True, forceElement="b1SG")
        cmds.sets("cube.f[5]", edit=True, forceElement="b2SG")
        before = _scene()
        self.assertIn(self.cube.f[4:6], blue)
        self.assertEqual(list(self.cube >> blue), [4, 5])
        self.assertEqual(Material.of(self.cube.f[4]), [ShadingEngine("b1SG")])
        self.assertEqual(Blinn.of(self.cube.f[4]), [blue])
        self.assertLeavesNothing(ValueError, lambda: self.cube << blue, "feeds 2 shading engines")
        self.cube.f[4:6] << -blue
        self.assertEqual((_sets("cube.f[4]"), _sets("cube.f[5]")), ([], []))

    def test_the_default_engine_with_relative_names(self):
        from rig.nodetypes import ShadingEngine, StandardSurface
        from rig.shade import Default

        other = Node(cmds.polyCube(name="other", constructionHistory=False)[0])
        cmds.namespace(add="lib")
        cmds.namespace(setNamespace="lib")
        cmds.namespace(relativeNames=True)
        other << Default()
        self.assertEqual(Material.of(other), [ShadingEngine("initialShadingGroup")])
        std = StandardSurface(":standardSurface1")
        self.assertIn(other, std)
        other << self.red
        other << std
        self.assertEqual(_sets(":otherShape"), [":initialShadingGroup"])


class TestQueriesAnswerByContents(_Case):
    """Review major (membership M3): a yes/no query never raises for what a
    kind can never hold; ``<<`` still refuses it before any write."""

    def test_every_kind(self):
        from rig import List, Tag

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        grp  = Node(cmds.group(empty=True, name="grp"))
        loc  = Node(cmds.spaceLocator(name="loc")[0])
        crv  = Node(cmds.circle(name="crv", constructionHistory=False)[0])
        red  = Blinn.define("red")
        lay  = DisplayLayer.define("L")
        cube << red
        cube.vtx[:3] << Tag("cap")
        before = _scene()
        for lhs, kind in (
            (loc, red), (grp, red), (crv, red), (List([cube, loc]), red), (cube.vtx[0], red),
            (cube.f[:3], lay), (Node("lambert1"), lay),
            (grp, Tag("cap")), (loc, Tag("cap")), (cube.f[:2], Tag("cap")),
        ):
            with self.subTest(lhs=repr(lhs), kind=repr(kind)):
                self.assertNotIn(lhs, kind)
        self.assertEqual([n for n in (cube, grp, loc) if n in red], [cube])
        for lhs in (grp, loc, crv, cube.vtx[0]):
            with self.subTest(of=repr(lhs)):
                self.assertEqual(Material.of(lhs), [])
                self.assertEqual(lhs >> Material(), [])
        self.assertIsNone(cube.f[0] >> DisplayLayer())
        self.assertEqual(DisplayLayer.of(cube.vtx[0]), [])
        self.assertEqual(Tag.of(grp), [])
        self.assertEqual(_scene(), before)
        # '<<' refuses them, before any write
        for call in (lambda: grp << red, lambda: cube.vtx[0] << red, lambda: cube.f[0] << lay, lambda: grp << Tag("cap")):
            self.assertLeavesNothing(TypeError, call)

    def test_an_empty_left_hand_side_names_the_query(self):
        from rig import List

        red = Blinn.define("red")
        self.assertLeavesNothing(ValueError, lambda: List([]) in red,
                                 r"^nothing to ask about: the left-hand side of 'in' names no member$")
        self.assertLeavesNothing(ValueError, lambda: List([]) >> red,
                                 r"^nothing to ask about: the left-hand side of '>>' names no member$")
        self.assertLeavesNothing(ValueError, lambda: List([]) << red, r"^nothing to inject$")


class TestLayerCreate(_Case):
    """Review major (membership M5): Layer.create(grp) holds grp, not its
    subtree, as grp << layer does."""

    def test_the_objects_given_join_themselves(self):
        grp = cmds.group(empty=True, name="grp")
        cmds.polyCube(name="child", constructionHistory=False)
        cmds.parent("child", grp)
        la = DisplayLayer.create(Node("grp"), name="La")
        self.assertEqual([str(m) for m in la.get_members()], ["grp"])
        self.assertNotIn(Node("child"), la)
        lb = DisplayLayer.create(Node("grp"), name="Lb", noRecurse=False)
        self.assertIn(Node("child"), lb)


class TestPairBroadcast(_Case):
    """Review major (membership M6): a pair broadcast with memberships checks
    every pair before its first write, and is one undo step."""

    def test_a_refused_pair_writes_nothing(self):
        from rig import List

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        sph  = Node(cmds.polySphere(name="sph", constructionHistory=False)[0])
        grp  = Node(cmds.group(empty=True, name="grp"))
        red, blue = Blinn.define("red"), Blinn.define("blue")
        lay = DisplayLayer.define("L")
        engines = _sets("cubeShape")
        for call in (lambda: List([cube, grp]) << [red, blue], lambda: List([cube, sph.tx]) << [red, lay]):
            self.assertLeavesNothing(TypeError, call)
            self.assertEqual(_sets("cubeShape"), engines)
        self.assertLeavesNothing(TypeError, lambda: List([cube, sph]) << [red, 5], "a membership on every right-hand side")
        cmds.flushUndo()
        List([cube, sph]) << [red, blue]
        self.assertEqual((_sets("cubeShape"), _sets("sphShape")), (["redSG"], ["blueSG"]))
        self.assertEqual(cmds.undoInfo(query=True, undoName=True), "rig.membership")
        cmds.undo()
        self.assertEqual(_sets("cubeShape"), engines)


class TestConvertInANamespace(_Case):
    """Review major (membership M4, safety C1): a conversion with another
    namespace current keeps the material's own name."""

    def test_root_and_namespaced_materials(self):
        from rig.nodetypes import Lambert, Phong

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        cube << Blinn.define("red")
        cmds.namespace(add="look")
        cmds.shadingNode("blinn", asShader=True, name="look:red")
        cmds.namespace(add="lib")
        cmds.namespace(setNamespace="lib")
        for relative, kind in ((False, Lambert), (True, Phong)):
            with self.subTest(relativeNames=relative):
                cmds.namespace(relativeNames=relative)
                new = Material(":red").astype(kind)
                self.assertEqual(new.uuid, cmds.ls(":red", uuid=True)[0])
                self.assertEqual(cmds.nodeType(":red"), kind.NATIVE_NODE_TYPE)
                self.assertIn(cube, new)
                look = Material(":look:red").astype(kind)
                self.assertEqual(look.uuid, cmds.ls(":look:red", uuid=True)[0])
                self.assertEqual(cmds.ls("lib:*"), [])


class TestConvertedRecord(_Case):
    """Review minor (membership M7): each held node names its own conversion."""

    def test_a_chain_and_a_namespace(self):
        from rig.nodetypes import Lambert, Phong

        x   = Blinn.define("x")
        p   = x.astype(Phong)
        l_  = p.astype(Lambert)
        l_.astype(Blinn)
        for held, kind in ((x, "phong"), (p, "lambert"), (l_, "blinn")):
            with self.subTest(kind):
                with self.assertRaisesRegex(RuntimeError, rf"^'x' was converted to a {kind}; use the node astype"):
                    str(held)
        cmds.namespace(add="ns")
        root = Blinn.define("y")
        root.astype(Phong)
        ns_y = Blinn.define("ns:y")
        ns_y.astype(Lambert)
        with self.assertRaisesRegex(RuntimeError, r"^'y' was converted to a phong"):
            str(root)


class TestReferencedShaderEngine(_Case):
    """Review minor (membership M8): the engine of a referenced shader is
    built at the root, not in the reference's namespace."""

    def test_the_engine_is_built_at_the_root(self):
        import os
        import shutil
        import tempfile

        folder = tempfile.mkdtemp(prefix="r4b_fix_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.addCleanup(cmds.file, new=True, force=True)
        cmds.shadingNode("blinn", asShader=True, name="refbare")
        path = os.path.join(folder, "mat.ma").replace("\\", "/")
        cmds.file(rename=path)
        cmds.file(save=True, type="mayaAscii", force=True)
        cmds.file(new=True, force=True)
        cmds.file(path, reference=True, namespace="lib")
        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        cube << Blinn("lib:refbare")
        self.assertEqual(_sets("cubeShape"), ["refbareSG"])
        self.assertIn(cube, Blinn("lib:refbare"))
        self.assertEqual(cmds.ls("lib:*SG"), [])
        other = Node(cmds.polyCube(name="other", constructionHistory=False)[0])
        other << Blinn("lib:refbare")
        self.assertEqual(_sets("otherShape"), ["refbareSG"])


class TestMembershipMessages(_Case):
    """Review minors (membership M10, M11, M12, M16)."""

    def test_a_class_without_parentheses(self):
        from rig import Tag

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        for call, text in (
            (lambda: cube in Blinn,         r"^Blinn is the class: Blinn\(\) is the kind token"),
            (lambda: cube in DisplayLayer,  r"^DisplayLayer is the class: DisplayLayer\(\) is the kind token"),
            (lambda: cube in Tag,           r"^Tag is the class: Tag\(\) is the kind token"),
            (lambda: cube << Blinn,         r"^Blinn is the class: .*write x << Blinn\('x'\)$"),
            (lambda: cube >> Material,      r"^Material is the class: .*write x >> Material\('x'\)$"),
            (lambda: cube in Transform,     r"^Transform is a node class: 'in' asks a membership node or Tag"),
        ):
            with self.subTest(text):
                self.assertLeavesNothing(TypeError, call, text)

    def test_the_plug_refusal_names_the_clone_spelling(self):
        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        from rig.spec import Float

        cube << Float("w")
        red, lay = Blinn.define("red"), DisplayLayer.define("L")
        for token, node in ((red, "red"), (lay, "L"), (red.engine, "redSG")):
            with self.subTest(node):
                self.assertLeavesNothing(TypeError, lambda: cube.w >> token,
                                         rf"; to clone the attribute onto it: cube\.w >> '{node}\.w'$")
        self.assertEqual(str(cube.w >> "red.w"), "red.w")

    def test_a_hint_spells_a_call_that_works(self):
        nsph = Node(cmds.sphere(name="nsph", constructionHistory=False)[0])
        red  = Blinn.define("red")
        nsph << red
        with self.assertRaises(TypeError) as ctx:
            nsph >> red
        self.assertIn('Material.of(NurbsSurface("nsphShape"))', str(ctx.exception))
        self.assertEqual(Material.of(Node("nsphShape")), [red])

    def test_tags_compare_by_value(self):
        from rig import Tag

        cube = Node(cmds.polyCube(name="cube", constructionHistory=False)[0])
        cube.vtx[:3] << Tag("cap")
        self.assertEqual(Tag("cap"), Tag("cap"))
        self.assertNotEqual(Tag("cap"), -Tag("cap"))
        self.assertNotEqual(Tag("cap"), Tag("lid"))
        self.assertIn(Tag("cap"), Tag.of(cube.vtx[1]))
        self.assertEqual(len({Tag("cap"), Tag("cap")}), 1)
