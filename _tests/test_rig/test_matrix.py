"""Smoke tests for ``rig.matrix``."""

import math

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
    It turns each rider's up vector by its share of one quaternion
    (``quaternion.pow``, a ``quatSlerp``), so this proves the public surface
    is intact for downstream rigs. The other tests pin the two rails' end
    tangents and the riders' shared twist."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        cmds.loadPlugin("matrixNodes", quiet=True)

    def test_rail_spine_simple_builds(self):
        from rig import List
        from rig.examples.rail_spine_simple import create_simple_rail

        controls = List()
        for i in range(4):
            loc = cmds.spaceLocator()[0]
            controls.append(Node(loc))
            controls[i].ty << i * 5.0
        rail = create_simple_rail(controls, riders=8)
        self.assertIsNotNone(rail)

    def controls(self, count):
        """``count`` locators stacked five units apart along Y"""
        from rig import List

        controls = List()
        for i in range(count):
            controls.append(Node(cmds.spaceLocator()[0]))
            controls[i].ty << i * 5.0
        return controls

    @staticmethod
    def unit(vector):
        length = math.sqrt(sum(c * c for c in vector))
        return [c / length for c in vector]

    def assert_extrapolates_from_the_ends(self, rail):
        """riders pushed to u = -0.3 and 1.3 sit 0.3 of the curve's length past
        its ends, along its end tangents; crossing an end does not jump"""
        import maya.api.OpenMaya as om

        selection = om.MSelectionList()
        selection.add(cmds.listRelatives(str(rail), type="nurbsCurve")[0])
        curve  = om.MFnNurbsCurve(selection.getDagPath(0))
        length = curve.length()
        riders = cmds.listRelatives(str(rail), type="joint")
        for shift, rider, param in ((-0.3, riders[0], curve.knotDomain[0]), (0.3, riders[-1], curve.knotDomain[1])):
            rail.shift << shift
            end     = curve.getPointAtParam(param, om.MSpace.kWorld)
            tangent = curve.tangent(param, om.MSpace.kWorld).normal()
            want    = end + tangent * (shift * length)
            got     = cmds.xform(rider, query=True, worldSpace=True, translation=True)
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=3)
        rail.shift << 0
        on_the_end = cmds.xform(riders[0], query=True, worldSpace=True, translation=True)
        rail.shift << -1e-9
        just_past  = cmds.xform(riders[0], query=True, worldSpace=True, translation=True)
        self.assertLess(math.dist(on_the_end, just_past), 1e-6)

    def test_rail_spine_simple_extrapolates_from_the_curve_ends(self):
        from rig.examples.rail_spine_simple import create_simple_rail

        controls = self.controls(4)
        rail     = create_simple_rail(controls, riders=3, debug=False)
        rail.stretch   << 1   # locked: each rider's u is its default u plus shift
        controls[1].tx << 10  # bend the spine
        self.assert_extrapolates_from_the_ends(rail)

    def test_rail_spine_extrapolates_from_the_curve_ends_whatever_invert_aim(self):
        from rig.examples.rail_spine import create_rail

        controls = self.controls(5)
        rail     = create_rail(controls, 3, debug=False)  # stretch is 1 by default
        controls[1].tx << 10
        for invert in (0, 1):
            with self.subTest(invertAim=invert):
                rail.invertAim << invert  # flips the riders' aim axis, never the curve's tangent
                self.assert_extrapolates_from_the_ends(rail)

    def test_rail_spine_simple_twist_is_the_slerp_of_the_end_ups(self):
        """on a straight rail along Y, each rider's up axis (X) is the slerp of
        the end controls' X axes by its default u, flattened across the rail"""
        from rig.examples.rail_spine_simple import create_simple_rail

        controls = self.controls(4)
        rail     = create_simple_rail(controls, riders=5, debug=False)
        controls[0].r  << (0, 40, 0)
        controls[-1].r << (0, -70, 30)  # tips the last up out of the plane across the rail
        first = self.unit(cmds.getAttr(f"{controls[0]}.worldMatrix[0]")[0:3])
        last  = self.unit(cmds.getAttr(f"{controls[-1]}.worldMatrix[0]")[0:3])
        theta = math.acos(sum(a * b for a, b in zip(first, last)))
        for rider in cmds.listRelatives(str(rail), type="joint"):
            u    = cmds.getAttr(f"{rider}.uDefault")
            up   = [(math.sin((1 - u) * theta) * a + math.sin(u * theta) * b) / math.sin(theta) for a, b in zip(first, last)]
            want = self.unit([up[0], 0, up[2]])
            got  = self.unit(cmds.getAttr(f"{rider}.worldMatrix[0]")[0:3])
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=5)


class TestLimbTwistExample(MayaTestCase):
    """The ``limb_twist`` example: five twist nodes between an upper arm at
    the origin and a lower arm ten units down X, with the upper arm's Z and
    the lower arm's -Y as the poles (the lower arm turned -90 degrees in X,
    so the two agree at rest). A node's twist is measured about the limb,
    X, from the world Z its up axis starts on."""

    TEST_START_NEW_SCENE = True

    def setUp(self):
        super().setUp()
        from rig.examples.limb_twist import create_limb_twist

        self.create   = create_limb_twist
        self.upperarm = Node.create("joint", name="upperarm")
        self.lowerarm = Node.create("joint", name="lowerarm")
        self.lowerarm.t << (10, 0, 0)
        self.lowerarm.r << (-90, 0, 0)
        self.twist = self.build()

    def build(self, **kwargs):
        """the example between the two arms, with the upper arm's Z and the lower arm's -Y as the poles"""
        return self.create(self.upperarm.wm, self.lowerarm.wm, start_axis=2, end_axis=1, end_invert=True, **kwargs)

    def twists(self, twist=None):
        """each twist node's twist about the limb, in degrees"""
        angles = []
        for node in (twist or self.twist).twists:
            y_axis = cmds.getAttr(f"{node}.worldMatrix[0]")[4:7]
            angles.append(math.degrees(math.atan2(-y_axis[1], y_axis[2])))
        return angles

    def assert_close(self, got, want):
        self.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=6)

    def test_at_rest_the_nodes_are_even_and_untwisted(self):
        self.assert_close(self.twist.twists.u.get(), [0, 0.25, 0.5, 0.75, 1])
        self.assert_close(self.twist.twists.tx.get(), [0, 2.5, 5, 7.5, 10])
        self.assert_close(self.twists(), [0] * 5)

    def test_a_twist_node_is_a_transform_by_default(self):
        self.assertEqual([cmds.nodeType(str(node)) for node in self.twist.twists], ["transform"] * 5)
        self.assertEqual([cmds.listRelatives(str(node), shapes=True) for node in self.twist.twists], [None] * 5)

    def test_the_twist_spreads_by_u(self):
        self.lowerarm.rx << 0
        self.assert_close(self.twists(), [0, 22.5, 45, 67.5, 90])

    def test_u_moves_and_retwists_a_node(self):
        self.lowerarm.rx       << 0
        self.twist.twists[1].u << 0.5
        self.assertAlmostEqual(self.twist.twists[1].tx.get(), 5)
        self.assertAlmostEqual(self.twists()[1], 45)

    def test_the_pole_toggles_are_live(self):
        root = self.twist.root
        self.assertEqual(cmds.listAttr(str(root), userDefined=True), ["startAxis", "startInvert", "endAxis", "endInvert"])
        self.lowerarm.rx << 0
        root.endInvert   << 0                      # the end's +Y: the twist runs the other way
        self.assert_close(self.twists(), [0, -22.5, -45, -67.5, -90])
        root.endInvert   << 1
        self.lowerarm.rx << -90
        root.endAxis     << 2                        # the end's -Z, world -Y at rest: a quarter turn
        self.assert_close(self.twists(), [0, 22.5, 45, 67.5, 90])

    def test_every_node_type_twists_the_same_outside_the_container(self):
        kinds = {"transform": ("transform", None), "locator": ("transform", "locator"), "joint": ("joint", None),
                 "cube": ("transform", "mesh")}
        built = {node_type: self.build(node_type=node_type, name=f"limbTwist{index}")
                 for index, node_type in enumerate(kinds, start=2)}
        self.lowerarm.rx << 0
        for index, (node_type, (kind, shape_kind)) in enumerate(kinds.items(), start=2):
            twist = built[node_type]
            with self.subTest(node_type=node_type):
                # full paths: every build names its first node twist1, under its own root
                nodes  = cmds.ls([str(node) for node in twist.twists], long=True)
                shapes = cmds.listRelatives(nodes, shapes=True, fullPath=True) or []
                members = set(cmds.ls(cmds.container(f"limbTwist{index}Setup", query=True, nodeList=True) or [],
                                      long=True))
                self.assertEqual(cmds.nodeType(nodes[0]), kind)
                self.assertEqual(shapes and cmds.nodeType(shapes[0]) or None, shape_kind)
                self.assertEqual(members & set(nodes + shapes), set())
                self.assert_close(self.twists(twist), [0, 22.5, 45, 67.5, 90])

    def test_a_cube_is_a_flat_fin_with_no_history(self):
        cube  = cmds.ls(str(self.build(node_type="cube", name="limbTwist2").twists[0]), long=True)[0]
        shape = cmds.listRelatives(cube, shapes=True, fullPath=True)[0]
        box   = cmds.exactWorldBoundingBox(cube)
        self.assertEqual(cmds.ls(cmds.listHistory(shape), long=True), [shape])
        # along the limb, across it, and along its up axis (world Z at rest)
        self.assert_close([box[i + 3] - box[i] for i in range(3)], [1.0, 0.2, 2.0])

    def test_weights_give_each_node_its_u(self):
        twist = self.create(self.upperarm.wm, self.lowerarm.wm, count=3, name="limbTwist2", weights=[0, 0.1, 1])
        self.assert_close(twist.twists.u.get(), [0, 0.1, 1])
        self.assert_close(twist.twists.tx.get(), [0, 1, 10])

    def test_bad_arguments_raise_before_building(self):
        before = len(cmds.ls())
        for kwargs in (dict(count=0), dict(node_type="sphere"), dict(weights=[0, 1]), dict(count=2, weights=[0, 1.5]),
                       dict(start_axis=3), dict(aim_axis=1)):
            with self.subTest(**{key: str(value) for key, value in kwargs.items()}):
                with self.assertRaises(ValueError):
                    self.create(self.upperarm.wm, self.lowerarm.wm, **kwargs)
                self.assertEqual(len(cmds.ls()), before)