"""Tests for ``rig._internal.types`` predicates."""

from unittest import mock

from maya import cmds
from maya.api import OpenMaya
from rig import Node, Plug, PlugList
from rig._internal.types import (
    _arity_of,
    _get_compound,
    _is_array,
    _is_attribute,
    _is_attribute_spec,
    _is_compound,
    _is_control_point,
    _is_euler,
    _is_list,
    _is_matrix,
    _is_node,
    _is_plug,
    _is_quaternion,
    _is_real,
    _is_sequence,
    _is_transform,
    _is_vector,
    _selected_choice_source,
    math_type,
)
from rig.spec import Float, lock, Quat, String
from rig._tests._base import MayaTestCase


class TestSequencePredicates(MayaTestCase):
    def test_is_sequence_for_list(self):
        self.assertTrue(_is_sequence([1, 2, 3]))
        self.assertTrue(_is_sequence((1, 2)))

    def test_is_sequence_excludes_strings(self):
        self.assertFalse(_is_sequence("abc"))

    def test_is_sequence_excludes_dicts(self):
        self.assertFalse(_is_sequence({"a": 1}))

    def test_is_sequence_for_scalars(self):
        self.assertFalse(_is_sequence(5))
        self.assertFalse(_is_sequence(None))

    def test_is_real(self):
        self.assertTrue(_is_real(5))
        self.assertTrue(_is_real(3.14))
        self.assertFalse(_is_real("5"))
        self.assertFalse(_is_real([1, 2]))


class TestClassPredicates(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_is_node(self):
        node = Node.create("transform", name="cube1")
        self.assertTrue(_is_node(node))
        self.assertFalse(_is_node("cube1"))
        self.assertFalse(_is_node(5))

    def test_is_plug(self):
        node = Node.create("transform", name="cube1")
        plug = node.translateX
        self.assertTrue(_is_plug(plug))
        self.assertFalse(_is_plug(node))
        self.assertFalse(_is_plug("cube1.tx"))

    def test_is_attribute(self):
        node = Node.create("transform", name="cube1")
        plug = node.translateX
        self.assertTrue(_is_attribute(plug))  # Plug subclasses Attribute

    def test_is_list(self):
        plist = PlugList([Node.create("transform", name="a")])
        self.assertTrue(_is_list(plist))
        self.assertFalse(_is_list([1, 2, 3]))  # plain list is not PlugList

    def test_is_attribute_spec(self):
        spec = Float("blend")
        self.assertTrue(_is_attribute_spec(spec))
        self.assertTrue(_is_attribute_spec(lock))
        self.assertFalse(_is_attribute_spec(5))


class TestAttributeKindPredicates(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_is_compound_for_translate(self):
        node = Node.create("transform", name="cube1")
        self.assertTrue(_is_compound(node.translate))

    def test_is_compound_false_for_scalar(self):
        node = Node.create("transform", name="cube1")
        self.assertFalse(_is_compound(node.translateX))

    def test_is_vector_for_translate(self):
        node = Node.create("transform", name="cube1")
        self.assertTrue(_is_vector(node.translate))

    def test_is_vector_false_for_quaternion(self):
        node = Node.create("transform", name="q1")
        node << Float("rotQ", multi=False)
        # Add a 4-channel attr to test quaternion detection.
        from rig.spec import Quat

        node << Quat("myQuat")
        self.assertTrue(_is_quaternion(node.myQuat))
        self.assertFalse(_is_vector(node.myQuat))

    def test_is_matrix(self):
        node = Node.create("transform", name="cube1")
        self.assertTrue(_is_matrix(node.matrix))
        self.assertFalse(_is_matrix(node.translateX))

    def test_is_transform(self):
        node = Node.create("transform", name="cube1")
        self.assertTrue(_is_transform(node.translate))

    def test_is_transform_false_for_dg_node_attr(self):
        node = Node.create("network", name="net1")
        node << Float("foo")
        self.assertFalse(_is_transform(node.foo))

    def test_is_array(self):
        node = Node.create("transform", name="cube1")
        node << Float("blend", multi=True)
        # node.blend is the multi root.
        self.assertTrue(_is_array(node.blend))

    def test_is_array_false_for_scalar(self):
        node = Node.create("transform", name="cube1")
        self.assertFalse(_is_array(node.translateX))

    def test_is_control_point_for_mesh_vtx(self):
        cube = cmds.polyCube(name="poly1")[0]
        # Idiomatic component access -- Node().vtx[N] returns a Plug for
        # the underlying controlPoints[N] (vtx is the alias).
        shape    = Node(cmds.listRelatives(cube, shapes=True)[0])
        vtx_plug = shape.vtx[0]
        self.assertTrue(_is_control_point(vtx_plug))


def _legacy_is_compound(obj):
    """The runtime ``data_type`` rule ``_is_compound`` answered with before the
    attribute-kind fast path."""
    try:
        if obj.plug.isCompound:
            return True
        try:
            if obj.num_children > 0:
                return True
        except Exception:
            pass
        return obj.data_type in ("double3", "float3", "long3", "short3", "double4", "float4")
    except Exception:
        return False


def _legacy_is_matrix(obj):
    """The runtime ``data_type`` rule ``_is_matrix`` answered with before the
    attribute-kind fast path."""
    try:
        return obj.data_type == "matrix"
    except Exception:
        return False


def _all_plugs(node_name):
    """Every top-level plug of a node, with compound children and element [0]
    of multis, as raw MPlugs."""
    sel = OpenMaya.MSelectionList()
    sel.add(node_name)
    mobj  = sel.getDependNode(0)
    fn    = OpenMaya.MFnDependencyNode(mobj)
    plugs = []
    for i in range(fn.attributeCount()):
        attr = fn.attribute(i)
        if not OpenMaya.MFnAttribute(attr).parent.isNull():
            continue
        stack = [OpenMaya.MPlug(mobj, attr)]
        while stack:
            plug = stack.pop()
            plugs.append(plug)
            if plug.isArray:
                stack.append(plug.elementByLogicalIndex(0))
            elif plug.isCompound:
                stack.extend(plug.child(c) for c in range(plug.numChildren()))
    return plugs


class TestAttributeKindFastPath(MayaTestCase):
    """``_is_compound`` / ``_is_matrix`` answer fixed attribute kinds from the
    API and keep the runtime ``data_type`` rule for generic and typed ones."""

    TEST_START_NEW_SCENE = True

    # (long name, addAttr flags, _is_compound, _is_matrix)
    KINDS = (
        ("dbl", {"at": "double"}, False, False),
        ("lin", {"at": "doubleLinear"}, False, False),
        ("ang", {"at": "doubleAngle"}, False, False),
        ("tim", {"at": "time"}, False, False),
        ("enm", {"at": "enum", "en": "a:b"}, False, False),
        ("mss", {"at": "message"}, False, False),
        ("boo", {"at": "bool"}, False, False),
        ("lng", {"at": "long"}, False, False),
        ("dt3", {"dt": "double3"}, True, False),
        ("dtm", {"dt": "matrix"}, False, True),
        ("atm", {"at": "matrix"}, False, True),
        ("flm", {"at": "fltMatrix"}, False, True),
        ("mdb", {"at": "double", "multi": True}, False, False),
    )

    def _kind_node(self):
        net = cmds.createNode("network", name="kinds")
        for long_name, flags, _, _ in self.KINDS:
            cmds.addAttr(net, ln=long_name, **flags)
        cmds.addAttr(net, ln="cmp", at="double3")
        for axis in "XYZ":
            cmds.addAttr(net, ln=f"cmp{axis}", at="double", p="cmp")
        cmds.setAttr(f"{net}.mdb[0]", 1.0)
        return Node(net)

    def test_is_compound_is_matrix_by_attr_kind(self):
        node     = self._kind_node()
        expected = [(getattr(node, n), c, m) for n, _, c, m in self.KINDS]
        expected.append((node.cmp, True, False))
        expected.append((node.mdb[0], False, False))
        for plug, compound, matrix in expected:
            with self.subTest(plug=str(plug)):
                self.assertIs(_is_compound(plug), compound)
                self.assertIs(_is_matrix(plug), matrix)

    def test_fixed_kinds_skip_the_type_query(self):
        node  = self._kind_node()
        plugs = [node.dbl, node.lin, node.ang, node.tim, node.enm, node.mss]
        plugs += [node.boo, node.lng, node.mdb, node.mdb[0], node.cmp, node.cmpX]
        with mock.patch.object(cmds, "getAttr", wraps=cmds.getAttr) as probe:
            for plug in plugs:
                _is_compound(plug)
                _is_matrix(plug)
        self.assertEqual(probe.call_count, 0)

    def test_generic_choice_output_still_uses_runtime_type(self):
        src = Node.create("transform", name="src")
        ch  = cmds.createNode("choice")
        self.assertFalse(_is_compound(Node(ch).output))
        cmds.connectAttr(str(src.translate), f"{ch}.input[0]")
        self.assertTrue(_is_compound(Node(ch).output))
        self.assertFalse(_is_matrix(Node(ch).output))

        ch_matrix = cmds.createNode("choice")
        cmds.connectAttr(f"{src}.worldMatrix[0]", f"{ch_matrix}.input[0]")
        self.assertTrue(_is_matrix(Node(ch_matrix).output))
        self.assertFalse(_is_compound(Node(ch_matrix).output))

        cmds.loadPlugin("matrixNodes", quiet=True)
        decompose = Node(cmds.createNode("decomposeMatrix"))
        self.assertTrue(_is_matrix(decompose.inputMatrix))
        self.assertFalse(_is_matrix(decompose.outputTranslate))
        self.assertTrue(_is_compound(decompose.outputTranslate))

    def test_fast_predicates_match_legacy_on_node_zoo(self):
        cmds.loadPlugin("matrixNodes", quiet=True)
        cmds.loadPlugin("quatNodes", quiet=True)
        names = [str(self._kind_node())]
        for node_type in (
            "transform",
            "joint",
            "multiplyDivide",
            "plusMinusAverage",
            "condition",
            "blendColors",
            "remapValue",
            "decomposeMatrix",
            "composeMatrix",
            "multMatrix",
            "quatProd",
        ):
            names.append(cmds.createNode(node_type))
        src = names[1]
        for source in ("translate", "worldMatrix[0]", "rotateX"):
            choice = cmds.createNode("choice")
            cmds.connectAttr(f"{src}.{source}", f"{choice}.input[0]")
            names.append(choice)
        driver = cmds.createNode("multiplyDivide")
        cmds.connectAttr(f"{driver}.outputX", f"{src}.rotateY")
        names.extend(cmds.ls(type="unitConversion"))
        checked = 0
        for name in names:
            for mplug in _all_plugs(name):
                fast, legacy = Plug(mplug), Plug(mplug)
                with self.subTest(plug=mplug.name()):
                    self.assertIs(_is_compound(fast), _legacy_is_compound(legacy))
                    self.assertIs(_is_matrix(fast), _legacy_is_matrix(legacy))
                checked += 1
        self.assertGreater(checked, 500)

    def test_missing_element_is_created_by_the_type_check(self):
        net = cmds.createNode("network", name="net")
        cmds.addAttr(net, ln="md", at="double", multi=True)
        cmds.addAttr(net, ln="mm", dt="matrix", multi=True)
        cmds.addAttr(net, ln="mc", at="compound", multi=True, numberOfChildren=1)
        cmds.addAttr(net, ln="mcx", at="double", p="mc")
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        cmds.setAttr(f"{net}.mm[3]", identity, type="matrix")
        node = Node(net)
        with self.assertRaises(TypeError):
            node.mm[3] * node.md[7]
        self.assertEqual(cmds.getAttr(f"{net}.md", multiIndices=True), [7])
        self.assertEqual(list(node.md.get_logical_indices()), [7])
        self.assertFalse(_is_compound(node.md[4]))
        self.assertFalse(_is_matrix(node.md[5]))
        self.assertFalse(_is_compound(node.mc[2].mcx))
        self.assertEqual(cmds.getAttr(f"{net}.md", multiIndices=True), [4, 5, 7])
        self.assertEqual(cmds.getAttr(f"{net}.mc", multiIndices=True), [2])

    def test_readded_attr_answers_with_its_new_kind(self):
        net = cmds.createNode("network", name="st")
        for flags, compound, matrix in (
            ({"dt": "matrix"}, False, True),
            ({"dt": "double3"}, True, False),
        ):
            cmds.addAttr(net, ln="foo", at="double")
            plug = Node(net).foo
            cmds.deleteAttr(f"{net}.foo")
            cmds.addAttr(net, ln="foo", **flags)
            self.assertIs(_is_compound(plug), compound)
            self.assertIs(_is_matrix(plug), matrix)
            self.assertIs(_is_compound(plug), _legacy_is_compound(plug))
            self.assertIs(_is_matrix(plug), _legacy_is_matrix(plug))
            cmds.deleteAttr(f"{net}.foo")

        cmds.addAttr(net, ln="foo", at="double")
        plug = Node(net).foo
        cmds.deleteAttr(f"{net}.foo")
        cmds.addAttr(net, ln="foo", dt="matrix")
        with self.assertRaises(TypeError) as ctx:
            plug * 2.0
        self.assertTrue(str(ctx.exception).startswith("matrix * scalar is undefined"))
        self.assertEqual(cmds.ls(type="multiply") + cmds.ls(type="multiplyDivide"), [])


class TestGetCompound(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_compound_returns_three_children(self):
        node     = Node.create("transform", name="cube1")
        children = _get_compound(node.translate)
        self.assertEqual(len(children), 3)

    def test_scalar_returns_single_element_list(self):
        children = _get_compound(5)
        self.assertEqual(children, [5])

    def test_sequence_passes_through(self):
        children = _get_compound([1, 2, 3])
        self.assertEqual(children, [1, 2, 3])


class TestMathTypeResolver(MayaTestCase):
    """Tests for the operation-first dispatch keystone: ``math_type()``.

    ``math_type`` is the single resolver every top-level dispatching verb
    routes through. It must classify scalars, vectors, eulers, quaternions,
    matrices and literals, and it must recover the arity of a quaternion that
    hides behind a ``TdataCompound`` output (``quatProd``/``eulerToQuat``) or a
    ``choice`` node by walking the structure -- the ``data_type`` string is
    useless there ('compound').
    """

    TEST_START_NEW_SCENE = True

    @staticmethod
    def _quat_output():
        """A ``quatProd.outputQuat`` plug: a TdataCompound whose ``data_type``
        is the useless string 'compound' but whose ``num_children`` is 4."""
        cmds.loadPlugin("quatNodes", quiet=True)
        return Node(cmds.createNode("quatProd")).outputQuat

    @staticmethod
    def _choice_over(plug_a, plug_b):
        """A ``choice.output`` selecting between two source plugs."""
        ch = cmds.createNode("choice")
        cmds.connectAttr(str(plug_a), f"{ch}.input[0]")
        cmds.connectAttr(str(plug_b), f"{ch}.input[1]")
        cmds.setAttr(f"{ch}.selector", 0)
        return Node(ch).output

    # ---- scalars / matrix ----
    def test_python_number_is_scalar(self):
        self.assertEqual(math_type(5), "scalar")
        self.assertEqual(math_type(3.14), "scalar")

    def test_scalar_plug_is_scalar(self):
        node = Node.create("transform", name="t1")
        self.assertEqual(math_type(node.translateX), "scalar")

    def test_matrix_plug_is_matrix(self):
        node = Node.create("transform", name="t1")
        self.assertEqual(math_type(node.matrix), "matrix")

    # ---- native compounds ----
    def test_vector_plug_is_vector(self):
        node = Node.create("transform", name="t1")
        self.assertEqual(math_type(node.translate), "vector")

    def test_euler_plug_is_euler(self):
        node = Node.create("transform", name="t1")
        self.assertEqual(math_type(node.rotate), "euler")

    def test_native_quaternion_is_quaternion(self):
        node = Node.create("transform", name="t1")
        node << Quat("myQuat")
        self.assertEqual(math_type(node.myQuat), "quaternion")

    # ---- the TdataCompound / structural cases ----
    def test_quatprod_output_is_quaternion_despite_compound_data_type(self):
        quat = self._quat_output()
        # The data_type string is the useless 'compound' -- only structure
        # (num_children == 4) reveals it is a quaternion.
        self.assertEqual(quat.data_type, "compound")
        self.assertEqual(math_type(quat), "quaternion")

    def test_decompose_output_quat_is_quaternion(self):
        cmds.loadPlugin("matrixNodes", quiet=True)
        decompose = Node(cmds.createNode("decomposeMatrix"))
        self.assertEqual(math_type(decompose.outputQuat), "quaternion")

    # ---- the KEYSTONE: choice walks ----
    def test_choice_over_quaternions_is_quaternion(self):
        out = self._choice_over(self._quat_output(), self._quat_output())
        self.assertEqual(math_type(out), "quaternion")

    def test_choice_over_vectors_is_vector(self):
        node_a = Node.create("transform", name="a")
        node_b = Node.create("transform", name="b")
        out    = self._choice_over(node_a.translate, node_b.translate)
        self.assertEqual(math_type(out), "vector")

    def test_choice_over_eulers_is_euler(self):
        node_a = Node.create("transform", name="a")
        node_b = Node.create("transform", name="b")
        out    = self._choice_over(node_a.rotate, node_b.rotate)
        self.assertEqual(math_type(out), "euler")

    # ---- literals ----
    def test_literal_lengths(self):
        self.assertEqual(math_type([1.0]), "scalar")
        self.assertEqual(math_type([1.0, 2.0, 3.0]), "vector")
        self.assertEqual(math_type([1.0, 2.0, 3.0, 4.0]), "quaternion")
        self.assertEqual(math_type([0.0] * 9), "matrix")
        self.assertEqual(math_type([0.0] * 16), "matrix")

    def test_literal_nested_rows_is_matrix(self):
        self.assertEqual(math_type([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "matrix")
        self.assertEqual(math_type([[1, 0, 0, 0]] * 4), "matrix")

    def test_literal_bad_length_raises(self):
        with self.assertRaises(ValueError):
            math_type([1.0, 2.0])
        with self.assertRaises(ValueError):
            math_type([1.0, 2.0, 3.0, 4.0, 5.0])

    # ---- unknown ----
    def test_unknown_for_none_and_string_attr(self):
        self.assertEqual(math_type(None), "unknown")
        node = Node.create("network", name="net1")
        node << String("label")
        self.assertEqual(math_type(node.label), "unknown")


class TestArityHelpers(MayaTestCase):
    """Direct coverage of the structural arity helpers behind ``math_type``."""

    TEST_START_NEW_SCENE = True

    def test_arity_of_native_quaternion(self):
        node = Node.create("transform", name="t1")
        node << Quat("q")
        self.assertEqual(_arity_of(node.q), 4)

    def test_arity_of_vector_and_scalar(self):
        node = Node.create("transform", name="t1")
        self.assertEqual(_arity_of(node.translate), 3)
        self.assertIsNone(_arity_of(node.translateX))

    def test_arity_of_non_attribute_is_none(self):
        self.assertIsNone(_arity_of(5))
        self.assertIsNone(_arity_of([1, 2, 3]))

    def test_selected_choice_source(self):
        Node.create("transform", name="a")
        Node.create("transform", name="b")
        ch = cmds.createNode("choice")
        cmds.connectAttr("a.translate", f"{ch}.input[0]")
        cmds.connectAttr("b.translate", f"{ch}.input[1]")
        cmds.setAttr(f"{ch}.selector", 1)
        src = _selected_choice_source(Node(ch).output)
        self.assertIsNotNone(src)
        self.assertEqual(src.num_children, 3)

    def test_selected_choice_source_none_for_non_choice(self):
        node = Node.create("transform", name="t1")
        self.assertIsNone(_selected_choice_source(node.translate))
        self.assertIsNone(_selected_choice_source(5))

    def test_arity_of_choice_over_scalars_is_none(self):
        # The walked source is a scalar, so num_children raises on both the
        # generic choice.output AND the source -- arity stays None.
        Node.create("transform", name="a")
        Node.create("transform", name="b")
        ch = cmds.createNode("choice")
        cmds.connectAttr("a.translateX", f"{ch}.input[0]")
        cmds.connectAttr("b.translateX", f"{ch}.input[1]")
        cmds.setAttr(f"{ch}.selector", 0)
        self.assertIsNone(_arity_of(Node(ch).output))

    def test_is_euler(self):
        node = Node.create("transform", name="t1")
        self.assertTrue(_is_euler(node.rotate))
        self.assertFalse(_is_euler(node.translate))
        node << Quat("q")
        self.assertFalse(_is_euler(node.q))  # arity 4, not euler