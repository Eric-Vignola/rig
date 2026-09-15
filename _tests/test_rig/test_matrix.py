"""Smoke tests for ``rig.matrix``."""

from maya import cmds
from rig import (
    dist,
    inverse,
    lerp,
    matrix as m,
    Node,
    normalize,
    Plug,
    set_options,
    slerp,
    to_euler,
    to_quaternion,
)
from rig._internal.maya_version import get_maya_version
from rig._tests._base import MayaTestCase


class TestDecompose(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_decompose_creates_node(self):
        a      = Node.create("transform", name="a")
        result = m.decompose(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "decomposeMatrix")


class TestInverse(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_inverse_creates_node(self):
        a      = Node.create("transform", name="a")
        result = inverse(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "inverseMatrix")


class TestTranspose(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_transpose_creates_node(self):
        a      = Node.create("transform", name="a")
        result = m.transpose(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "transposeMatrix")


class TestToQuaternion(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_to_quaternion_returns_quat_plug(self):
        a      = Node.create("transform", name="a")
        result = to_quaternion(a.matrix)
        self.assertTrue(str(result).endswith(".outputQuat"))


class TestToEuler(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_to_euler_returns_rotate_plug(self):
        a      = Node.create("transform", name="a")
        result = to_euler(a.matrix)
        self.assertTrue(str(result).endswith(".outputRotate"))


class TestCompose(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Disable publishing so this test reads the internal node directly.
        # The published-interface contract for ``m.compose`` is covered by
        # ``TestMatrixPublishedInterface``.
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_compose_creates_node(self):
        a      = Node.create("transform", name="a")
        result = m.compose(translate=a.t)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "composeMatrix")


class TestFourByFour(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_fourbyfour_creates_4x4(self):
        a      = Node.create("transform", name="a")
        result = m.fourbyfour(x=a.t)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "fourByFourMatrix")


class TestFourByFourValue(MayaTestCase):
    """Lock the fourByFourMatrix row wiring (which input vec3 lands in
    which matrix row) so the C901 restructure can't silently scramble it.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_fourbyfour_x_axis_lands_in_row_0(self):
        a = Node.create("transform", name="a")
        a.t << (1, 2, 3)
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.fourbyfour(x=a.t)
        mtx = _eval(out.offsetParentMatrix)
        # x -> row 0 -> M[0], M[1], M[2].
        self.assertAlmostEqual(mtx[0], 1.0, places=4)
        self.assertAlmostEqual(mtx[1], 2.0, places=4)
        self.assertAlmostEqual(mtx[2], 3.0, places=4)

    def test_fourbyfour_position_lands_in_row_3(self):
        a = Node.create("transform", name="a")
        a.t << (7, 8, 9)
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.fourbyfour(position=a.t)
        mtx = _eval(out.offsetParentMatrix)
        # position -> row 3 (translation) -> M[12], M[13], M[14].
        self.assertAlmostEqual(mtx[12], 7.0, places=4)
        self.assertAlmostEqual(mtx[13], 8.0, places=4)
        self.assertAlmostEqual(mtx[14], 9.0, places=4)


class TestAim(MayaTestCase):
    """``matrix.aim`` -- build an orthonormal orientation matrix from an aim +
    up vector via an ``aimMatrix`` node. This is the lone vector->orientation
    constructor (migrated from the old ``vector.to_matrix`` in the Model-A
    rescope); euler / quaternion forms are ``to_euler(aim(...))`` /
    ``to_quaternion(aim(...))``."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        # Internal-node test -- disable publishing so the result is the raw
        # ``aimMatrix`` output, not the published container alias.
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_aim_creates_aimmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = m.aim(a.t, b.t)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "aimMatrix")


class TestAimValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_aim_x_up_y_is_identity(self):
        aim_src = Node.create("transform", name="aim")
        aim_src.t << (1, 0, 0)
        up = Node.create("transform", name="up")
        up.t << (0, 1, 0)
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.aim(aim_src.t, up.t)
        mtx = _eval(out.offsetParentMatrix)
        # X axis = (1,0,0).
        for actual, expected in zip(mtx[0:3], [1, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=3)
        # Y axis = (0,1,0).
        for actual, expected in zip(mtx[4:7], [0, 1, 0]):
            self.assertAlmostEqual(actual, expected, places=3)
        # Z axis = (0,0,1).
        for actual, expected in zip(mtx[8:11], [0, 0, 1]):
            self.assertAlmostEqual(actual, expected, places=3)


class TestMultiply(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_multiply_creates_multmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = m.multiply(a.matrix, b.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "multMatrix")


class TestAdd(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_add_creates_addmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = m.add(a.matrix, b.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "addMatrix")

    def test_add_with_weights_creates_wtaddmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = m.add(a.matrix, b.matrix, weights=[0.5, 0.5])
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "wtAddMatrix")


class TestLerp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_lerp_creates_wtaddmatrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = lerp(a.matrix, b.matrix, weight=0.25)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "wtAddMatrix")


class TestSlerp(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_slerp_returns_compose_matrix(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = slerp(a.matrix, b.matrix, weight=0.5)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "composeMatrix")


class TestNormalize(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(publish_attributes=False)

    def tearDown(self):
        super().tearDown()
        set_options(publish_attributes=True)

    def test_normalize_returns_compose_matrix(self):
        a      = Node.create("transform", name="a")
        result = normalize(a.matrix)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "composeMatrix")


class TestDeterminant(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_determinant_returns_plug(self):
        a      = Node.create("transform", name="a")
        result = m.determinant(a.matrix)
        self.assertIsInstance(result, Plug)


class TestMatrixDist(MayaTestCase):
    """Topology checks for the ``matrix.dist`` re-export (mirrored from
    :mod:`rig.functions`).

    Confirms the wrapper delegates to the canonical :mod:`functions`
    impl -- same node type, same memoised instance -- and that matrix
    inputs land on the ``inMatrix1`` / ``inMatrix2`` slots (not
    ``point1`` / ``point2``), so Maya's C++ pulls the translation
    column directly without spawning extra ``decomposeMatrix`` /
    ``translationFromMatrix`` nodes.
    """

    TEST_START_NEW_SCENE = True

    def test_dist_creates_distancebetween(self):
        a      = Node.create("transform", name="a")
        b      = Node.create("transform", name="b")
        result = dist(a.wm, b.wm)
        self.assertEqual(cmds.nodeType(str(result).split(".")[0]), "distanceBetween")

    def test_dist_uses_inmatrix_slots(self):
        a         = Node.create("transform", name="a")
        b         = Node.create("transform", name="b")
        result    = dist(a.wm, b.wm)
        node_name = str(result).split(".")[0]
        self.assertTrue(
            cmds.listConnections(f"{node_name}.inMatrix1", source=True),
            "inMatrix1 should have a connection from a.wm",
        )
        self.assertTrue(
            cmds.listConnections(f"{node_name}.inMatrix2", source=True),
            "inMatrix2 should have a connection from b.wm",
        )

    def test_dist_is_memoized(self):
        """The top-level ``dist`` verb memoises its network, so two calls with
        the same matrix inputs return the same underlying node -- one distance
        op regardless of how many call-sites reference it."""
        a        = Node.create("transform", name="a")
        b        = Node.create("transform", name="b")
        result_1 = dist(a.wm, b.wm)
        result_2 = dist(a.wm, b.wm)
        self.assertEqual(str(result_1), str(result_2))


# --------------------------------------------------------------------- #
#  VALUE-COMPARING TESTS -- build network, set inputs, getAttr, compare
#  against ground-truth math. Loads matrixNodes plugin (needed for
#  inverseMatrix / transposeMatrix in headless Maya).
# --------------------------------------------------------------------- #


def _eval(plug):
    val = cmds.getAttr(str(plug))
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], tuple):
        return list(val[0])
    return val


class TestInverseValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_inverse_identity_is_identity(self):
        a   = Node.create("transform", name="a")
        out = Node.create("transform", name="out")
        out.matrix << inverse(a.matrix)
        mtx = _eval(out.matrix)
        # Identity check.
        self.assertAlmostEqual(mtx[0],  1.0, places=4)
        self.assertAlmostEqual(mtx[5],  1.0, places=4)
        self.assertAlmostEqual(mtx[10], 1.0, places=4)
        self.assertAlmostEqual(mtx[15], 1.0, places=4)

    def test_inverse_translation(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        a.ty << 3.0
        a.tz << 2.0
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << inverse(a.matrix)
        mtx = _eval(out.offsetParentMatrix)
        self.assertAlmostEqual(mtx[12], -5.0, places=4)
        self.assertAlmostEqual(mtx[13], -3.0, places=4)
        self.assertAlmostEqual(mtx[14], -2.0, places=4)


class TestTransposeValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_transpose_translation_lands_in_row_0(self):
        a = Node.create("transform", name="a")
        a.tx << 7.0
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.transpose(a.matrix)
        mtx = _eval(out.offsetParentMatrix)
        # Original M[12]=7. After transpose, M[3]=7.
        self.assertAlmostEqual(mtx[3], 7.0, places=4)


class TestDeterminantValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_determinant_identity_is_one(self):
        a   = Node.create("transform", name="a")
        out = Node.create("transform", name="out")
        out.ty << m.determinant(a.matrix)
        self.assertAlmostEqual(_eval(out.ty), 1.0, places=4)

    def test_determinant_uniform_scale(self):
        a = Node.create("transform", name="a")
        a.scaleX << 2.0
        a.scaleY << 2.0
        a.scaleZ << 2.0
        out = Node.create("transform", name="out")
        out.ty << m.determinant(a.matrix)
        self.assertAlmostEqual(_eval(out.ty), 8.0, places=4)


class TestComposeValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_compose_matches_src_matrix(self):
        # compose(t, r, s) on a source's channels should equal the
        # source's local matrix.
        src = Node.create("transform", name="src")
        src.tx      << 5.0
        src.ty      << 3.0
        src.tz      << 2.0
        src.rotateX << 30.0
        src.rotateY << 45.0
        src.rotateZ << 60.0
        src.scaleX  << 1.5
        src.scaleY  << 0.5
        src.scaleZ  << 2.0

        composed = m.compose(translate=src.t, rotate=src.r, scale=src.s)
        probe    = Node.create("transform", name="probe")
        probe.offsetParentMatrix << composed
        wm = _eval(probe.worldMatrix[0])
        sm = _eval(src.matrix)
        for actual, expected in zip(wm, sm):
            self.assertAlmostEqual(actual, expected, places=3)


class TestMultiplyValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_multiply_two_translations(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        b = Node.create("transform", name="b")
        b.ty << 3.0
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.multiply(a.matrix, b.matrix)
        mtx = _eval(out.offsetParentMatrix)
        # A @ B = translate(5, 3, 0) since both are pure translations.
        self.assertAlmostEqual(mtx[12], 5.0, places=4)
        self.assertAlmostEqual(mtx[13], 3.0, places=4)


class TestBlendValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_blend_midpoint_of_translations(self):
        a = Node.create("transform", name="a")
        a.tx << 0.0
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = Node.create("transform", name="out")
        out.offsetParentMatrix << m.blend(a.matrix, b.matrix, weight=0.5)
        mtx = _eval(out.offsetParentMatrix)
        # Halfway between identity@tx=0 and translation@tx=10 -> tx=5.
        self.assertAlmostEqual(mtx[12], 5.0, places=4)

    def test_blend_trs_components(self):
        """``matrix.blend`` interpolates T/R/S/shear via ``blendMatrix``:
        linear translate & scale, quaternion-slerp rotation."""
        a = Node.create("transform", name="a")
        a.tx     << 2.0
        a.scaleX << 1.0
        b = Node.create("transform", name="b")
        b.tx      << 8.0
        b.rotateY << 90.0
        b.scaleX  << 3.0
        out = m.blend(a.matrix, b.matrix, weight=0.3)

        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        trans = cmds.getAttr(f"{dm}.outputTranslate")[0]
        rot   = cmds.getAttr(f"{dm}.outputRotate")[0]
        scale = cmds.getAttr(f"{dm}.outputScale")[0]
        # translate: lerp(2, 8, 0.3) = 3.8
        self.assertAlmostEqual(trans[0], 3.8, places=4)
        # rotateY: quaternion-slerp(0deg, 90deg, 0.3) = 27deg (constant angular velocity)
        self.assertAlmostEqual(rot[1], 27.0, places=2)
        # scaleX: ``blendMatrix`` uses LINEAR scale => lerp(1, 3, 0.3) = 1.6
        # (the old ``slerp`` used exponential lerp = ``3 ** 0.3``).
        self.assertAlmostEqual(scale[0], 1.6, places=3)
        self.assertAlmostEqual(scale[1], 1.0, places=3)


class TestNormalizeValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_normalize_drops_scale_keeps_rotation_and_translation(self):
        # A matrix with non-unit scale (2,3,4), rotation, and translation.
        # After orthonormalize: scale -> (1,1,1); rotation + translation
        # preserved.
        src = Node.create("transform", name="src")
        src.t << (5, 6, 7)
        src.r << (10, 20, 30)
        src.s << (2, 3, 4)
        out = normalize(src.matrix)

        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        scale = cmds.getAttr(f"{dm}.outputScale")[0]
        trans = cmds.getAttr(f"{dm}.outputTranslate")[0]
        rot   = cmds.getAttr(f"{dm}.outputRotate")[0]
        for s in scale:
            self.assertAlmostEqual(s, 1.0, places=4)
        for actual, expected in zip(trans, (5, 6, 7)):
            self.assertAlmostEqual(actual, expected, places=4)
        for actual, expected in zip(rot, (10, 20, 30)):
            self.assertAlmostEqual(actual, expected, places=3)


# --------------------------------------------------------------------- #
#  v3.A -- matrix decomposition / extraction helpers
# --------------------------------------------------------------------- #


class TestTranslationValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_translation_extracts_t(self):
        a = Node.create("transform", name="a")
        a.tx << 5.0
        a.ty << 3.0
        a.tz << 2.0
        out = Node.create("transform", name="out")
        out.t << m.translation(a.matrix)
        for actual, expected in zip(_eval(out.t), [5, 3, 2]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestRotationValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_rotation_extracts_r(self):
        a = Node.create("transform", name="a")
        a.rotateX << 30.0
        a.rotateY << 45.0
        a.rotateZ << 60.0
        out = Node.create("transform", name="out")
        out.r << m.rotation(a.matrix)
        # Compare via decompose since rotationFromMatrix returns Maya's
        # canonical rotation extraction (XYZ rotate-order).
        for actual, expected in zip(_eval(out.r), [30, 45, 60]):
            self.assertAlmostEqual(actual, expected, places=2)


class TestScaleOfValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_scale_of_extracts_s(self):
        a = Node.create("transform", name="a")
        a.scaleX << 2.0
        a.scaleY << 0.5
        a.scaleZ << 3.0
        out = Node.create("transform", name="out")
        out.s << m.scale_of(a.matrix)
        for actual, expected in zip(_eval(out.s), [2, 0.5, 3]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestAxisValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_axis_x_of_identity(self):
        a   = Node.create("transform", name="a")
        out = Node.create("transform", name="out")
        out.t << m.axis(a.matrix, axis=0)
        # Identity matrix's X axis = (1, 0, 0).
        for actual, expected in zip(_eval(out.t), [1, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_axis_y_of_identity(self):
        a   = Node.create("transform", name="a")
        out = Node.create("transform", name="out")
        out.t << m.axis(a.matrix, axis="y")
        for actual, expected in zip(_eval(out.t), [0, 1, 0]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestColumnRowValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_column_3_of_translation(self):
        # Maya matrices are row-major: translate lives in ROW 3 (cols 0-2).
        # column(m, 3) returns the LAST column = (0, 0, 0, 1) for any matrix.
        # Use row(m, 3) instead to fetch the translation row.
        a = Node.create("transform", name="a")
        a.tx << 7.0
        a.ty << 9.0
        a.tz << 11.0
        out  = Node.create("transform", name="out")
        row3 = m.row(a.matrix, index=3)
        # Index the compound rather than naming its children: pre-2024 row(3)
        # falls back to ``decomposeMatrix.outputTranslate``, whose children are
        # ``outputTranslateX/Y/Z`` -- not the native ``rowFromMatrix`` node's
        # ``outputX/Y/Z``.
        out.tx << row3[0]
        out.ty << row3[1]
        out.tz << row3[2]
        self.assertAlmostEqual(_eval(out.tx), 7.0,  places=4)
        self.assertAlmostEqual(_eval(out.ty), 9.0,  places=4)
        self.assertAlmostEqual(_eval(out.tz), 11.0, places=4)

    def test_row_0_of_identity(self):
        # Row 0 of identity = (1, 0, 0, 0).
        a        = Node.create("transform", name="a")
        out      = Node.create("transform", name="out")
        row_plug = m.row(a.matrix, index=0)
        out.tx << row_plug.outputX
        out.ty << row_plug.outputY
        out.tz << row_plug.outputZ
        self.assertAlmostEqual(_eval(out.tx), 1.0, places=4)
        self.assertAlmostEqual(_eval(out.ty), 0.0, places=4)
        self.assertAlmostEqual(_eval(out.tz), 0.0, places=4)


class TestTransformPointValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_transform_point_with_translation(self):
        # Point (1,0,0) transformed by translate(5,0,0) -> (6,0,0).
        p = Node.create("transform", name="p")
        p.t << (1, 0, 0)
        a = Node.create("transform", name="a")
        a.tx << 5.0
        out = Node.create("transform", name="out")
        out.t << m.transform_point(p.t, a.matrix)
        for actual, expected in zip(_eval(out.t), [6, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestTransformVectorValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_transform_vector_ignores_translation(self):
        # Vector (1,0,0) transformed by translate(5,0,0) -> (1,0,0)
        # because vector transforms ignore translation.
        v = Node.create("transform", name="v")
        v.t << (1, 0, 0)
        a = Node.create("transform", name="a")
        a.tx << 5.0
        out = Node.create("transform", name="out")
        out.t << m.transform_vector(v.t, a.matrix)
        for actual, expected in zip(_eval(out.t), [1, 0, 0]):
            self.assertAlmostEqual(actual, expected, places=4)


class TestMatrixDistValue(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def test_dist_3_4_5_triangle_is_5(self):
        # Drive two transforms 3 units apart in X and 4 in Y -- the
        # 3-4-5 right triangle. distanceBetween reads translation
        # straight off the worldMatrix slots; result == 5.
        a = Node.create("transform", name="a")
        a.t << (0, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (3, 4, 0)
        out = Node.create("transform", name="out")
        out.ty << dist(a.wm, b.wm)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)

    def test_dist_cross_type_matrix_and_vector(self):
        # Matrix x vec3 input still works -- matrix to inMatrix1,
        # vec3 to point2 in the same distanceBetween node.
        a = Node.create("transform", name="a")
        a.t << (0, 0, 0)
        b = Node.create("transform", name="b")
        b.t << (3, 4, 0)
        out = Node.create("transform", name="out")
        out.ty << dist(a.wm, b.t)
        self.assertAlmostEqual(_eval(out.ty), 5.0, places=4)


# --------------------------------------------------------------------- #
#  v4.J -- container + published-attribute wrap (third_party.rig style)
# --------------------------------------------------------------------- #


class TestMatrixPublishedInterface(MayaTestCase):
    """Verify each wrapped composite matrix function creates a container
    with the expected published attributes when ``flatten_containers=False``.
    Spot-checks the 8 functions wrapped in v4.J."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        set_options(flatten_containers=False)

    def tearDown(self):
        super().tearDown()
        set_options(flatten_containers=True)

    def _assert_attrs(self, container_name, attrs):
        ctn = cmds.ls(f"{container_name}*", type="container") or []
        self.assertTrue(
            any(c.startswith(container_name) for c in ctn),
            f"expected {container_name} container, got {ctn}",
        )
        target = next(c for c in ctn if c.startswith(container_name))
        for attr in attrs:
            self.assertTrue(
                cmds.attributeQuery(attr, node=target, exists=True),
                f"{target}.{attr} should be published",
            )

    def test_compose_publishes_inputs(self):
        a = Node.create("transform", name="a")
        a.t << [1, 2, 3]
        m.compose(translate=a.t, rotate=a.r, scale=a.s)
        self._assert_attrs("compose1", ["translate", "rotate", "scale", "output"])

    def test_fourbyfour_publishes_inputs(self):
        a = Node.create("transform", name="a")
        a.t << [4, 5, 6]
        m.fourbyfour(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1], position=a.t)
        self._assert_attrs("fourbyfour1", ["x", "y", "z", "position", "output"])

    def test_aim_publishes(self):
        a = Node.create("transform", name="a")
        a.t << [1, 0, 0]
        m.aim(a.t, [0, 1, 0])
        self._assert_attrs(
            "matrix_aim1",
            ["aim_vector", "up_vector", "aim_axis", "up_axis", "output"],
        )

    def test_lerp_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        lerp(a.matrix, b.matrix, weight=0.5)
        self._assert_attrs("matrix_lerp1", ["input1", "input2", "weight", "output"])

    def test_slerp_publishes(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        slerp(a.matrix, b.matrix, weight=0.5)
        self._assert_attrs("matrix_slerp1", ["input1", "input2", "weight", "output"])

    def test_normalize_publishes(self):
        a = Node.create("transform", name="a")
        normalize(a.matrix)
        self._assert_attrs("matrix_normalize1", ["input", "output"])

    def test_transform_point_publishes(self):
        p      = Node.create("transform", name="p")
        m_node = Node.create("transform", name="mtx")
        p.t << [1, 0, 0]
        m.transform_point(p.t, m_node.matrix)
        self._assert_attrs("transform_point1", ["point", "matrix", "output"])

    def test_transform_vector_publishes(self):
        v      = Node.create("transform", name="v")
        m_node = Node.create("transform", name="mtx")
        v.t << [1, 0, 0]
        m.transform_vector(v.t, m_node.matrix)
        self._assert_attrs("transform_vector1", ["vector", "matrix", "output"])

    def test_axis_publishes(self):
        a = Node.create("transform", name="a")
        m.axis(a.matrix, axis=0)
        self._assert_attrs("axis1", ["matrix", "output"])

    def test_column_publishes(self):
        # ``columnFromMatrix`` has no pre-2024 equivalent, so matrix.column()
        # raises rather than falling back.
        if get_maya_version() < 2024:
            self.skipTest("matrix.column() requires Maya 2024+")
        a = Node.create("transform", name="a")
        m.column(a.matrix, index=0)
        self._assert_attrs("column1", ["matrix", "index", "output"])

    def test_row_publishes(self):
        a = Node.create("transform", name="a")
        m.row(a.matrix, index=0)
        if get_maya_version() < 2024:
            # Pre-2024 row() delegates rows 0-2 to axis(), so the container
            # and its published attrs carry that name instead.
            self._assert_attrs("axis1", ["matrix", "output"])
        else:
            self._assert_attrs("row1", ["matrix", "index", "output"])


# --------------------------------------------------------------------- #
#  Public API surface (__all__)
# --------------------------------------------------------------------- #


class TestMatrixAll(MayaTestCase):
    """``matrix``'s public surface is the matrix-specific construction /
    decomposition / arithmetic / extraction ops PLUS the cross-type ops, now
    exposed as public per-type functions (also reachable through the
    top-level dispatch verbs)."""

    def test_public_surface(self):
        self.assertEqual(
            set(m.__all__),
            {
                "decompose",
                "inverse",
                "transpose",
                "to_quaternion",
                "to_euler",
                "compose",
                "fourbyfour",
                "aim",
                "multiply",
                "add",
                "lerp",
                "blend",
                "slerp",
                "pow",
                "normalize",
                "determinant",
                "translation",
                "rotation",
                "scale_of",
                "axis",
                "column",
                "row",
                "transform_point",
                "transform_vector",
                "dist",
            },
        )
        for name in m.__all__:
            self.assertTrue(hasattr(m, name), f"matrix.{name} missing")

    def test_cross_type_ops_public(self):
        # The cross-type ops are now PUBLIC per-type funcs on matrix
        # (callable directly), in addition to the top-level dispatch verbs.
        # The old ``_``-prefixed impl names are gone.
        for name in (
            "inverse",
            "lerp",
            "slerp",
            "normalize",
            "to_euler",
            "to_quaternion",
            "dist",
        ):
            self.assertIn(name, m.__all__)
            self.assertTrue(hasattr(m, name), f"matrix.{name} should be public")
            self.assertFalse(
                hasattr(m, f"_{name}"), f"matrix._{name} private alias should be gone"
            )


# --------------------------------------------------------------------- #
#  Interpolation: orientation slerp / full blend (blendMatrix) / power
# --------------------------------------------------------------------- #


class TestMatrixSlerpOrientation(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_slerp_is_orientation_only(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx      << 10.0
        b.scaleX  << 4.0
        b.rotateY << 90.0
        # weight=1 -> output equals b's rotation only (T=0, S=1, no shear).
        out = slerp(a.matrix, b.matrix, 1.0)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 0.0, places=3
        )
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputScale")[0][0], 1.0, places=3)
        # Rotation is preserved (b.rotateY = 90deg).
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputRotate")[0][1], 90.0, places=2)


class TestMatrixBlend(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_blend_preserves_shear_and_endpoints(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx      << 10.0
        b.scaleX  << 3.0
        b.shearXY << 0.5
        # weight=1 -> output equals b (full T/R/S/shear preserved).
        out = m.blend(a.matrix, b.matrix, 1.0)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        # shearXY survives (lerp does not preserve shear because it's a
        # raw element blend; blendMatrix has native shear handling).
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputShear")[0][0], 0.5, places=3)
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputScale")[0][0], 3.0, places=3)

    def test_blend_weight_zero_equals_input1(self):
        # weight=0 -> output equals input1 (a = identity). Pins the
        # input1/target wiring: a swapped inputMatrix/target would still
        # pass the weight=1 endpoint test above, but not this one.
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx      << 10.0
        b.scaleX  << 3.0
        b.shearXY << 0.5
        out = m.blend(a.matrix, b.matrix, 0.0)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 0.0, places=3
        )
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputScale")[0][0], 1.0, places=3)
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputShear")[0][0], 0.0, places=3)


class TestMatrixPow(MayaTestCase):
    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_pow_half_moves_halfway_to_identity(self):
        b = Node.create("transform", name="b")
        b.tx     << 10.0
        b.scaleX << 3.0
        out = m.pow(b.matrix, 0.5)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        # ``pow`` is ``blend(identity, m, t)`` -- linear translate => 5,
        # linear scale => 2.
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 5.0, places=3
        )
        self.assertAlmostEqual(cmds.getAttr(f"{dm}.outputScale")[0][0], 2.0, places=3)


class TestMatrixInterpExtrapolation(MayaTestCase):
    """``matrix.blend`` / ``matrix.pow`` extrapolate past the [0, 1]
    weight range -- no clamp on ``blendMatrix.envelope``. Pins the
    coverage gap flagged in review: linear translate ``lerp(0, 10, 1.5) = 15``.
    """

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_blend_extrapolates_past_one(self):
        a = Node.create("transform", name="a")
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = m.blend(a.matrix, b.matrix, 1.5)  # past the [0, 1] endpoint
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        # linear translate extrapolates: lerp(0, 10, 1.5) = 15
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 15.0, places=3
        )

    def test_pow_extrapolates_past_one(self):
        b = Node.create("transform", name="b")
        b.tx << 10.0
        out = m.pow(b.matrix, 1.5)  # blend(identity, b, 1.5)
        dm  = cmds.createNode("decomposeMatrix")
        cmds.connectAttr(str(out), f"{dm}.inputMatrix")
        self.assertAlmostEqual(
            cmds.getAttr(f"{dm}.outputTranslate")[0][0], 15.0, places=3
        )


class TestSpineExampleRegression(MayaTestCase):
    """End-to-end regression for the slerp(orientation-only) / blend(TRS)
    refactor: the ``rail_spine_simple`` example must still build cleanly.
    It uses vector ``slerp`` for the up-vector blend (vector code path,
    unaffected by the matrix changes), so this proves the public surface
    is intact for downstream rigs."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_rail_spine_simple_builds(self):
        from rig import PlugList
        from rig.examples.rail_spine_simple import create_simple_rail

        controls = PlugList()
        for i in range(4):
            loc = cmds.spaceLocator()[0]
            controls.append(Node(loc))
            controls[i].ty << i * 5.0
        rail = create_simple_rail(controls, riders=8)
        self.assertIsNotNone(rail)