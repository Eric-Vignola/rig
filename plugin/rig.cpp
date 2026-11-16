// rig.mll: the node types rig builds, one plug-in for all of them.
//
// rigExpression: one node that evaluates a fused chain of rig operators
// (rig.fuse()). rig compiles the expressions of a block (rig/_internal/fuse.py)
// into a small program: a list of instructions over three register files, each
// register written once:
//   R3   scalars and vectors (3 doubles; a scalar uses the first);
//   R4   quaternions (x, y, z, w) and matrix rows / columns (4 doubles);
//   R16  matrices (16 doubles, row-major, Maya's row vectors).
// The node runs the program at every evaluation.
//
// Attributes
//   program     intArray  the compiled program (layout below)
//   constants   doubleArray
//   scalarIn[]  double    inputs (connected plugs)
//   vectorIn[]  double3
//   angleIn[]   doubleAngle: an angle plug connects without a unitConversion;
//               the load turns radians into degrees, as Maya's unitConversion
//               does for rig's own nodes
//   eulerIn[]   double3 of doubleAngle (a rotate): read in degrees, as angleIn
//   quatIn[]    4 doubles (a quaternion: decomposeMatrix.outputQuat ...)
//   matrixIn[]  matrix
//   scalarOut[] double    outputs
//   vectorOut[] double3
//   quatOut[]   4 doubles
//   matrixOut[] matrix
//   eulerOut[]  double3 of doubleAngle: a rotation computed in degrees, written
//               as an angle, so it drives a rotate without a unitConversion
//   angleOut[]  doubleAngle: the same for one angle
//   timeIn      time: the scene's time (time1.outTime), in frames of the current
//               unit for TIME (rig.functions.frame())
//
// Program layout (ints):
//   version 1: [0] MAGIC  [1] 1  [2] R3 registers  [3] instructions  [4] scalar outs
//              [5] vector outs
//   version 2: the same six, then [6] R4 registers  [7] R16 registers  [8] quaternion
//              outs  [9] matrix outs  [10] euler outs  [11] angle outs
//   then 5 ints per instruction: op, dst, a, b, c (an op with more operands is
//   followed by an EXT instruction holding them), then one register per output:
//   scalars, vectors, quaternions, matrices, eulers, angles.
//
// Semantics: numpy's for scalars and vectors, in double precision (the fused
// path agrees with numpy, not with the quirks of rig's network):
//   / is IEEE (x / 0 is +-inf, 0 / 0 is nan); % is np.remainder (floor modulo,
//   the sign of the divisor) and // np.floor_divide, both numpy's npy_divmod;
//   round is np.round (half to even); sign is np.sign (0 at 0, nan at nan);
//   sqrt / exp / log are np.sqrt / np.exp / np.log (log(0) = -inf, nan below);
//   asin / acos are nan outside [-1, 1]; min / max / clamp are np.minimum /
//   np.maximum / np.clip (a NaN propagates; clamp with lo > hi gives hi);
//   and / or / xor / not are np.logical_* and condition is np.where: non-zero
//   is true; normalize is v / np.linalg.norm(v) (nan for a zero vector).
// Matrices and quaternions: Maya's nodes' (decomposeMatrix, composeMatrix,
// quatSlerp, aimMatrix, blendMatrix ...), see rig_math.h. Q_ARC is Maya's own
// MQuaternion(a, b), angleBetween's turn: its rule for parallel, opposite and short
// vectors mixes thresholds a hand-written copy matched on all but 2 of 4,500 pairs.
// Q_EULER and R_EULER are Maya's decompositions too (MEulerRotation::decompose of the
// quaternion's matrix, MTransformationMatrix's rotation): their choice of euler at an
// exact half turn ((0, 180, 0)) is not the rule a hand-written one follows elsewhere.
//
// A program that does not validate (bad magic, an operand out of range, an
// unknown op) is not run: compute fails and Maya keeps the outputs.

#include <maya/MAngle.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MDoubleArray.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnPlugin.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MIntArray.h>
#include <maya/MMatrix.h>
#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include <maya/MQuaternion.h>
#include <maya/MTime.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MTypeId.h>
#include <maya/MVector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "rig_math.h"

namespace {

constexpr int MAGIC = 0x52455850;  // "REXP"
constexpr int HEADER1 = 6, HEADER2 = 12;
constexpr double DEG = 57.29577951308232;     // Maya's radian -> degree factor
constexpr double RAD = 0.017453292519943295;  // degree -> radian


enum Op : int {
    LOAD_S = 1,     // dst = scalarIn[a]
    LOAD_V = 2,     // dst = vectorIn[a]
    LOAD_A = 3,     // dst = degrees(angleIn[a])
    CONST_S = 4,    // dst = constants[a]
    CONST_V = 5,    // dst = constants[a .. a + 2]
    SPLAT = 6,      // dst(v) = (a, a, a)
    COMP = 7,       // dst(s) = a(v)[b]
    MAKE = 8,       // dst(v) = (a, b, c)
    BIN_S = 10,     // dst = binary c (a, b)
    BIN_V = 11,     //   per channel
    UN_S = 12,      // dst = unary b (a)
    UN_V = 13,      //   per channel
    SELECT_S = 14,  // dst = test(a) ? b : c
    SELECT_V = 15,
    CLAMP_S = 16,   // dst = clamp(a, b, c)
    CLAMP_V = 17,   //   per channel
    LERP_S = 18,    // dst = a + (b - a) * c
    LERP_V = 19,    //   per channel, c a scalar
    DOT = 20,       // dst(s) = a . b
    CROSS = 21,     // dst(v) = a x b
    LENGTH = 22,    // dst(s) = |a|
    NORMALIZE = 23, // dst(v) = a / |a|, zero for a zero vector
    DIST = 24,      // dst(s) = |a - b|
    V1_END = 25,    // (version 1 programs use the ops above only)
    // version 2: q a quaternion register (R4), m a matrix (R16), s / v in R3; an
    // order is a scalar register (a rotate order, rounded into 0..5); angles degrees
    LOAD_Q = 25,    // q = quatIn[a]
    LOAD_M = 26,    // m = matrixIn[a]
    LOAD_E = 27,    // v = degrees(eulerIn[a])
    CONST_Q = 28,   // q = constants[a .. a + 3]
    CONST_M = 29,   // m = constants[a .. a + 15]
    EXT = 30,       // the operands after c of the instruction before it
    Q_COMP = 31,    // s = (q a)[b]
    Q_VEC = 32,     // v = (q a).xyz
    Q_MAKE = 33,    // q = (v a, s b)
    Q_BIN = 34,     // q = a (c: 0 add, 1 sub, 2 Maya's product) b
    Q_UN = 35,      // q = (b: 0 negate, 1 conjugate, 2 inverse, 3 normal)(q a)
    Q_SLERP = 36,   // q = slerp(q a, q b, s c), the shortest way (quatSlerp)
    Q_EULER = 37,   // v = the euler of q a in order s b (quatToEuler)
    E_QUAT = 38,    // q = the quaternion of the euler v a in order s b (eulerToQuat)
    AXIS_Q = 39,    // q = axis v a, angle s b (axisAngleToQuat)
    Q_AXIS = 40,    // v = the axis of q a (quatToAxisAngle)
    Q_ANGLE = 41,   // s = the angle of q a
    M_BIN = 42,     // m = a (c: 0 product, 1 sum) b (multMatrix, addMatrix)
    M_SCALE = 43,   // m = (m a) * (s b), each element (a wtAddMatrix term)
    M_INV = 44,     // m = inverse(m a), the identity when singular (inverseMatrix)
    M_T = 45,       // m = transpose(m a)
    M_DET = 46,     // s = det(m a), 4x4 (determinant)
    PT_MUL = 47,    // v = point v a * m b (multiplyPointByMatrix)
    VEC_MUL = 48,   // v = vector v a * m b (multiplyVectorByMatrix)
    M_ROW = 49,     // q = row (s b, 0..3) of m a (rowFromMatrix)
    M_COL = 50,     // q = column (s b, 0..3) of m a (columnFromMatrix)
    M_AXIS = 51,    // v = axis b (0..5: x y z -x -y -z) of m a (axisFromMatrix)
    M_TRANS = 52,   // v = the translation of m a (translationFromMatrix)
    M_SCALE_OF = 53,// v = the scale of m a (scaleFromMatrix, decomposeMatrix)
    M_SHEAR_OF = 54,// v = the shear of m a
    M_ROT = 55,     // m = the rotation of m a (no scale, shear or translation)
    R_EULER = 56,   // v = the euler of the rotation m a in order s b
    R_QUAT = 57,    // q = the quaternion of the rotation m a (decomposeMatrix.outputQuat)
    E_MATRIX = 58,  // m = the rotation of the euler v a in order s b
    Q_MATRIX = 59,  // m = the rotation of q a, normalized
    COMPOSE = 60,   // m = S(v b) * SH(v c) * (m a), translated by EXT's v a (composeMatrix)
    M_SETROW = 61,  // m = m a with row c's x y z = v b (fourByFourMatrix)
    BLEND = 62,     // m = blendMatrix(m a, m b, weight s c)
    AIM = 63,       // m = aimMatrix(aim v a, up v b, EXT: aim axis v a, up axis v b)
    NOP = 64,       // nothing (an instruction rig.fuse() patched out)
    MOVE = 65,      // dst = a, in the register file c (0 R3, 1 R4, 2 R16): a value of its own
    // rig.mll 1.2
    Q_ARC = 66,     // q = the turn of v a onto v b: MQuaternion(a, b) (angleBetween)
    TIME = 67,      // s = the scene's time in frames of the current unit (timeIn)
    RAND = 68,      // s = rig.random's value for seed s a, trigger s b: in [0, 1) (rig_math.h)
    OP_END = 69,
};

enum Bin : int {
    ADD, SUB, MUL, DIV, POW, MOD, FLOORDIV, MIN, MAX,
    LT, LE, GT, GE, EQ, NE, AND, OR, XOR, ATAN2, ATAN2D, BIN_END,
};

enum Un : int {
    NEG, ABS, FLOOR, CEIL, ROUND, TRUNC, SIGN, REV, SQRT, EXP, LOG, NOT,
    SIN, COS, TAN, ASIN, ACOS, ATAN, SIND, COSD, TAND, ASIND, ACOSD, ATAND, UN_END,
};

using Reg = std::array<double, 3>;
using Reg4 = std::array<double, 4>;
struct alignas(16) Reg16 {
    double v[16];
};

bool truthy(double x) { return x != 0.0; }  // non-zero is true (np.logical_*, np.where); nan is true

// np.maximum / np.minimum: a NaN propagates
double np_max(double a, double b) { return std::isnan(a) ? a : std::isnan(b) ? b : (a >= b ? a : b); }
double np_min(double a, double b) { return std::isnan(a) ? a : std::isnan(b) ? b : (a <= b ? a : b); }

// np.clip(x, lo, hi) = np.minimum(np.maximum(x, lo), hi): with lo > hi, hi
double clamp(double x, double lo, double hi) { return np_min(np_max(x, lo), hi); }

// numpy's npy_divmod: the floor division and the modulus of np.floor_divide / np.remainder
double np_divmod(double a, double b, double& mod)
{
    mod = std::fmod(a, b);
    if (b == 0.0)
        return a / b;                       // inf or nan; fmod gave nan
    double div = (a - mod) / b;
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) {
            mod += b;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, b);
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5)
            floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    return floordiv;
}

double np_sign(double x) { return std::isnan(x) ? x : x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }

// one channel of a binary op, as numpy computes it
bool binary(int f, double a, double b, double& out)
{
    switch (f) {
    case ADD: out = a + b; return true;
    case SUB: out = a - b; return true;
    case MUL: out = a * b; return true;
    case DIV: out = a / b; return true;
    case POW: out = std::pow(a, b); return true;
    case MOD: { double m; np_divmod(a, b, m); out = m; return true; }
    case FLOORDIV: { double m; out = np_divmod(a, b, m); return true; }
    case MIN: out = np_min(a, b); return true;
    case MAX: out = np_max(a, b); return true;
    case LT: out = a < b; return true;
    case LE: out = a <= b; return true;
    case GT: out = a > b; return true;
    case GE: out = a >= b; return true;
    case EQ: out = a == b; return true;
    case NE: out = a != b; return true;
    case AND: out = truthy(a) && truthy(b); return true;
    case OR: out = truthy(a) || truthy(b); return true;
    case XOR: out = (a != 0.0) != (b != 0.0); return true;
    case ATAN2: out = std::atan2(a, b); return true;
    case ATAN2D: out = std::atan2(a, b) * DEG; return true;
    }
    return true;
}

double unary(int f, double a)
{
    switch (f) {
    case NEG: return -a;
    case ABS: return std::fabs(a);
    case FLOOR: return std::floor(a);
    case CEIL: return std::ceil(a);
    case ROUND: return std::nearbyint(a);      // half to even (np.round)
    case TRUNC: return std::trunc(a);
    case SIGN: return np_sign(a);
    case REV: return 1.0 - a;
    case SQRT: return std::sqrt(a);
    case EXP: return std::exp(a);
    case LOG: return std::log(a);
    case NOT: return truthy(a) ? 0.0 : 1.0;
    case SIN: return std::sin(a);
    case COS: return std::cos(a);
    case TAN: return std::tan(a);
    case ASIN: return std::asin(a);
    case ACOS: return std::acos(a);
    case ATAN: return std::atan(a);
    case SIND: return std::sin(a * RAD);
    case COSD: return std::cos(a * RAD);
    case TAND: return std::tan(a * RAD);
    case ASIND: return std::asin(a) * DEG;
    case ACOSD: return std::acos(a) * DEG;
    case ATAND: return std::atan(a) * DEG;
    }
    return 0.0;
}

// an index (row, column) read from a register: rounded half away from zero, into 0..hi
int index_of(double v, int hi)
{
    const int k = int(std::lround(v));
    return k < 0 ? 0 : k > hi ? hi : k;
}

// the values of a multi by logical index; a missing element reads `empty`
template <typename T, typename Read>
void by_index(MArrayDataHandle h, std::vector<T>& out, Read read, const T& empty)
{
    out.clear();
    const unsigned count = h.elementCount();
    for (unsigned k = 0; k < count; ++k, h.next()) {
        const unsigned index = h.elementIndex();
        if (index >= 100000)
            continue;  // a sparse index this far out is not one rig makes
        if (index >= out.size())
            out.resize(index + 1, empty);
        MDataHandle element = h.inputValue();
        out[index] = read(element);
    }
}

// writes n outputs: in place once the elements exist; a builder only when the count changes
template <typename Write>
void write_outputs(MDataBlock& block, const MObject& attr, int n, Write write)
{
    MArrayDataHandle h = block.outputArrayValue(attr);
    if (h.elementCount() == unsigned(n)) {
        for (int k = 0; k < n; ++k, h.next()) {
            MDataHandle element = h.outputValue();
            write(k, element);
        }
    } else {
        MArrayDataBuilder b(&block, attr, unsigned(n));
        for (int k = 0; k < n; ++k) {
            MDataHandle element = b.addElement(k);
            write(k, element);
        }
        h.set(b);
    }
    h.setAllClean();
}

constexpr double IDENTITY[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

}  // namespace

// the program, its validation and the instruction loop: the base of rig's
// expression nodes, which differ only in how the inputs are read and the outputs
// written
class ExprBase : public MPxNode
{
public:
    SchedulingType schedulingType() const override { return kParallel; }

protected:
    // the program and constants read in place, validated when they change; false: do not run
    bool load(MDataHandle hProgram, MDataHandle hConstants);
    // the program over the inputs (filled by the node), into the registers
    void run();
    bool validate(const std::vector<int>& p, size_t constants) const;
    void count(MDataBlock& block, const MObject& attr);
    int header() const { return program_[1] == 2 ? HEADER2 : HEADER1; }
    int field(int k) const { return program_[1] == 2 ? program_[k] : 0; }   // a version 2 header field

    std::vector<int> program_;
    unsigned checkedConstants_ = ~0u;
    bool valid_ = false;
    std::vector<Reg> regs_;
    std::vector<Reg4> regs4_;
    std::vector<Reg16> regs16_;
    std::vector<double> sIn_, aIn_;
    std::vector<Reg> vIn_, eIn_;
    std::vector<Reg4> qIn_;
    std::vector<Reg16> mIn_;
    std::vector<double> consts_;
    int evaluations_ = 0;
    double time_ = 0.0;      // timeIn, in frames (TIME)
};

class RigExpression : public ExprBase
{
public:
    static const MTypeId id;
    static MObject aProgram, aConstants, aScalarIn, aVectorIn, aAngleIn, aEulerIn, aQuatIn, aMatrixIn, aScalarOut,
        aVectorOut, aQuatOut, aMatrixOut, aEulerOut, aAngleOut, aEvaluations, aTimeIn;

    static void* creator() { return new RigExpression; }
    static MStatus initialize();
    MStatus compute(const MPlug& plug, MDataBlock& block) override;
};

const MTypeId RigExpression::id(0x0007F010);
MObject RigExpression::aProgram, RigExpression::aConstants, RigExpression::aScalarIn, RigExpression::aVectorIn,
    RigExpression::aAngleIn, RigExpression::aEulerIn, RigExpression::aQuatIn, RigExpression::aMatrixIn,
    RigExpression::aScalarOut, RigExpression::aVectorOut, RigExpression::aQuatOut, RigExpression::aMatrixOut,
    RigExpression::aEulerOut, RigExpression::aAngleOut, RigExpression::aEvaluations, RigExpression::aTimeIn;

bool ExprBase::validate(const std::vector<int>& p, size_t constants) const
{
    if (p.size() < HEADER1 || p[0] != MAGIC || (p[1] != 1 && p[1] != 2))
        return false;
    const bool v2 = p[1] == 2;
    if (v2 && p.size() < HEADER2)
        return false;
    const int H = v2 ? HEADER2 : HEADER1;
    const int regs = p[2], n = p[3], ns = p[4], nv = p[5];
    const int regs4 = v2 ? p[6] : 0, regs16 = v2 ? p[7] : 0;
    const int nq = v2 ? p[8] : 0, nm = v2 ? p[9] : 0, ne = v2 ? p[10] : 0, na = v2 ? p[11] : 0;
    for (int x : {regs, n, ns, nv, regs4, regs16, nq, nm, ne, na})
        if (x < 0 || x > 1000000)
            return false;
    if (p.size() != size_t(H) + size_t(n) * 5 + size_t(ns + nv + nq + nm + ne + na))
        return false;
    auto r3 = [&](int r) { return r >= 0 && r < regs; };
    auto r4 = [&](int r) { return r >= 0 && r < regs4; };
    auto r16 = [&](int r) { return r >= 0 && r < regs16; };
    auto cst = [&](int a, int width) { return a >= 0 && size_t(a) + size_t(width) <= constants; };
    for (int i = 0; i < n; ++i) {
        const int* ins = &p[H + i * 5];
        const int op = ins[0], dst = ins[1], a = ins[2], b = ins[3], c = ins[4];
        if (!v2 && op >= V1_END)
            return false;
        // an op with an EXT: the next instruction must be it
        const bool extended = op == COMPOSE || op == AIM;
        const int* ext = extended && i + 1 < n ? &p[H + (i + 1) * 5] : nullptr;
        if (extended && (ext == nullptr || ext[0] != EXT))
            return false;
        bool ok;
        switch (op) {
        case LOAD_S: case LOAD_V: case LOAD_A: case LOAD_E: ok = r3(dst) && a >= 0; break;
        case LOAD_Q: ok = r4(dst) && a >= 0; break;
        case LOAD_M: ok = r16(dst) && a >= 0; break;
        case CONST_S: ok = r3(dst) && cst(a, 1); break;
        case CONST_V: ok = r3(dst) && cst(a, 3); break;
        case CONST_Q: ok = r4(dst) && cst(a, 4); break;
        case CONST_M: ok = r16(dst) && cst(a, 16); break;
        case SPLAT: case LENGTH: case NORMALIZE: ok = r3(dst) && r3(a); break;
        case COMP: ok = r3(dst) && r3(a) && b >= 0 && b <= 2; break;
        case UN_S: case UN_V: ok = r3(dst) && r3(a) && b >= 0 && b < UN_END; break;
        case BIN_S: case BIN_V: ok = r3(dst) && r3(a) && r3(b) && c >= 0 && c < BIN_END; break;
        case DOT: case CROSS: case DIST: ok = r3(dst) && r3(a) && r3(b); break;
        case MAKE: case SELECT_S: case SELECT_V: case CLAMP_S: case CLAMP_V: case LERP_S: case LERP_V:
            ok = r3(dst) && r3(a) && r3(b) && r3(c);
            break;
        case EXT: ok = false; break;  // only after the op it extends (skipped there)
        case NOP: ok = true; break;
        case MOVE: ok = c == 0 ? r3(dst) && r3(a) : c == 1 ? r4(dst) && r4(a) : c == 2 && r16(dst) && r16(a); break;
        case Q_ARC: ok = r4(dst) && r3(a) && r3(b); break;
        case TIME: ok = r3(dst); break;
        case RAND: ok = r3(dst) && r3(a) && r3(b); break;
        case Q_COMP: ok = r3(dst) && r4(a) && b >= 0 && b <= 3; break;
        case Q_VEC: ok = r3(dst) && r4(a); break;
        case Q_MAKE: ok = r4(dst) && r3(a) && r3(b); break;
        case Q_BIN: ok = r4(dst) && r4(a) && r4(b) && c >= 0 && c <= 2; break;
        case Q_UN: ok = r4(dst) && r4(a) && b >= 0 && b <= 3; break;
        case Q_SLERP: ok = r4(dst) && r4(a) && r4(b) && r3(c); break;
        case Q_EULER: ok = r3(dst) && r4(a) && r3(b); break;
        case E_QUAT: ok = r4(dst) && r3(a) && r3(b); break;
        case AXIS_Q: ok = r4(dst) && r3(a) && r3(b); break;
        case Q_AXIS: case Q_ANGLE: ok = r3(dst) && r4(a); break;
        case M_BIN: ok = r16(dst) && r16(a) && r16(b) && c >= 0 && c <= 1; break;
        case M_SCALE: ok = r16(dst) && r16(a) && r3(b); break;
        case M_INV: case M_T: case M_ROT: ok = r16(dst) && r16(a); break;
        case M_DET: case M_TRANS: case M_SCALE_OF: case M_SHEAR_OF: ok = r3(dst) && r16(a); break;
        case PT_MUL: case VEC_MUL: ok = r3(dst) && r3(a) && r16(b); break;
        case M_ROW: case M_COL: ok = r4(dst) && r16(a) && r3(b); break;
        case M_AXIS: ok = r3(dst) && r16(a) && b >= 0 && b <= 5; break;
        case R_EULER: ok = r3(dst) && r16(a) && r3(b); break;
        case R_QUAT: ok = r4(dst) && r16(a); break;
        case E_MATRIX: ok = r16(dst) && r3(a) && r3(b); break;
        case Q_MATRIX: ok = r16(dst) && r4(a); break;
        case COMPOSE: ok = r16(dst) && r16(a) && r3(b) && r3(c) && r3(ext[2]); break;
        case M_SETROW: ok = r16(dst) && r16(a) && r3(b) && c >= 0 && c <= 3; break;
        case BLEND: ok = r16(dst) && r16(a) && r16(b) && r3(c); break;
        case AIM: ok = r16(dst) && r3(a) && r3(b) && r3(ext[2]) && r3(ext[3]); break;
        default: ok = false;
        }
        if (!ok)
            return false;
        if (extended)
            ++i;  // its EXT, checked with it
    }
    const int* outs = &p[H + n * 5];
    for (int k = 0; k < ns + nv; ++k)
        if (!r3(outs[k]))
            return false;
    for (int k = 0; k < nq; ++k)
        if (!r4(outs[ns + nv + k]))
            return false;
    for (int k = 0; k < nm; ++k)
        if (!r16(outs[ns + nv + nq + k]))
            return false;
    for (int k = 0; k < ne + na; ++k)
        if (!r3(outs[ns + nv + nq + nm + k]))
            return false;
    return true;
}

bool ExprBase::load(MDataHandle hProgram, MDataHandle hConstants)
{
    // the program and constants read in place (no array copies): a new program
    // starts with fresh registers
    MObject programData = hProgram.data(), constantsData = hConstants.data();
    MFnIntArrayData pfn;
    MFnDoubleArrayData cfn;
    const unsigned plen = !programData.isNull() && pfn.setObject(programData) ? pfn.length() : 0;
    const unsigned clen = !constantsData.isNull() && cfn.setObject(constantsData) ? cfn.length() : 0;
    bool same = plen == program_.size();
    for (unsigned k = 0; same && k < plen; ++k)
        same = pfn[k] == program_[k];
    if (!same) {
        program_.resize(plen);
        for (unsigned k = 0; k < plen; ++k)
            program_[k] = pfn[k];
        regs_.clear();
    }
    consts_.resize(clen);
    for (unsigned k = 0; k < clen; ++k)
        consts_[k] = cfn[k];
    if (!same || clen != checkedConstants_) {  // validated once per program and constants size
        checkedConstants_ = clen;
        valid_ = validate(program_, checkedConstants_);
        if (valid_) {
            if (regs_.size() != size_t(program_[2]))
                regs_.assign(size_t(program_[2]), Reg{0.0, 0.0, 0.0});
            regs4_.assign(size_t(field(6)), Reg4{0.0, 0.0, 0.0, 1.0});
            regs16_.assign(size_t(field(7)), Reg16{});
        }
    }
    return valid_;
}

void ExprBase::run()
{
    const std::vector<double>& constants = consts_;
    const int n = program_[3];
    const int H = header();
    std::vector<Reg>& r = regs_;
    Reg4* q = regs4_.data();
    Reg16* m = regs16_.data();
    for (int i = 0; i < n; ++i) {
        const int* ins = &program_[H + i * 5];
        const int a = ins[2], b = ins[3], c = ins[4];
        const int dst = ins[1];
        switch (ins[0]) {
        case LOAD_S: r[dst] = {size_t(a) < sIn_.size() ? sIn_[a] : 0.0, 0.0, 0.0}; break;
        case LOAD_V: r[dst] = size_t(a) < vIn_.size() ? vIn_[a] : Reg{0.0, 0.0, 0.0}; break;
        case LOAD_A: r[dst] = {size_t(a) < aIn_.size() ? aIn_[a] * DEG : 0.0, 0.0, 0.0}; break;
        case CONST_S: r[dst] = {constants[a], 0.0, 0.0}; break;
        case CONST_V: r[dst] = {constants[a], constants[a + 1], constants[a + 2]}; break;
        case SPLAT: { const double v = r[a][0]; r[dst] = {v, v, v}; break; }
        case COMP: r[dst] = {r[a][b], 0.0, 0.0}; break;
        case MAKE: r[dst] = {r[a][0], r[b][0], r[c][0]}; break;
        case BIN_S: binary(c, r[a][0], r[b][0], r[dst][0]); break;
        case BIN_V: {
            Reg& d = r[dst];
            for (int k = 0; k < 3; ++k)
                binary(c, r[a][k], r[b][k], d[k]);
            break;
        }
        case UN_S: r[dst] = {unary(b, r[a][0]), 0.0, 0.0}; break;
        case UN_V: { const Reg x = r[a]; r[dst] = {unary(b, x[0]), unary(b, x[1]), unary(b, x[2])}; break; }
        case SELECT_S: r[dst][0] = truthy(r[a][0]) ? r[b][0] : r[c][0]; break;
        case SELECT_V: r[dst] = truthy(r[a][0]) ? r[b] : r[c]; break;
        case CLAMP_S: r[dst][0] = clamp(r[a][0], r[b][0], r[c][0]); break;
        case CLAMP_V: { const Reg x = r[a], lo = r[b], hi = r[c];
                        for (int k = 0; k < 3; ++k) r[dst][k] = clamp(x[k], lo[k], hi[k]);
                        break; }
        case LERP_S: r[dst][0] = r[a][0] + (r[b][0] - r[a][0]) * r[c][0]; break;
        case LERP_V: { const Reg x = r[a], y = r[b]; const double w = r[c][0];
                       for (int k = 0; k < 3; ++k) r[dst][k] = x[k] + (y[k] - x[k]) * w;
                       break; }
        case DOT: r[dst] = {r[a][0] * r[b][0] + r[a][1] * r[b][1] + r[a][2] * r[b][2], 0.0, 0.0}; break;
        case CROSS: { const Reg x = r[a], y = r[b];
                      r[dst] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
                      break; }
        case LENGTH: { const Reg x = r[a]; r[dst] = {std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), 0.0, 0.0}; break; }
        case NORMALIZE: { const Reg x = r[a];     // v / np.linalg.norm(v): nan for a zero vector
                          const double l = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
                          r[dst] = Reg{x[0] / l, x[1] / l, x[2] / l};
                          break; }
        case DIST: { const Reg x = r[a], y = r[b];
                     const double dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];
                     r[dst] = {std::sqrt(dx * dx + dy * dy + dz * dz), 0.0, 0.0};
                     break; }
        // -- version 2
        case LOAD_Q: q[dst] = size_t(a) < qIn_.size() ? qIn_[a] : Reg4{0.0, 0.0, 0.0, 1.0}; break;
        case LOAD_M:
            if (size_t(a) < mIn_.size())
                m[dst] = mIn_[a];
            else
                rigm::identity(m[dst].v);
            break;
        case LOAD_E: {
            const Reg e = size_t(a) < eIn_.size() ? eIn_[a] : Reg{0.0, 0.0, 0.0};
            r[dst] = {e[0] * DEG, e[1] * DEG, e[2] * DEG};
            break;
        }
        case CONST_Q: q[dst] = {constants[a], constants[a + 1], constants[a + 2], constants[a + 3]}; break;
        case CONST_M: std::copy(&constants[a], &constants[a] + 16, m[dst].v); break;
        case Q_COMP: r[dst] = {q[a][b], 0.0, 0.0}; break;
        case Q_VEC: r[dst] = {q[a][0], q[a][1], q[a][2]}; break;
        case Q_MAKE: q[dst] = {r[a][0], r[a][1], r[a][2], r[b][0]}; break;
        case Q_BIN:
            if (c == 2) {
                rigm::qmul(q[a].data(), q[b].data(), q[dst].data());
            } else {
                const double sgn = c == 0 ? 1.0 : -1.0;
                for (int k = 0; k < 4; ++k)
                    q[dst][k] = q[a][k] + sgn * q[b][k];
            }
            break;
        case Q_UN: {
            const Reg4 x = q[a];
            Reg4& d = q[dst];
            switch (b) {
            case 0: d = {-x[0], -x[1], -x[2], -x[3]}; break;
            case 1: d = {-x[0], -x[1], -x[2], x[3]}; break;
            case 2: rigm::qinverse(x.data(), d.data()); break;
            default: rigm::qnormal(x.data(), d.data()); break;
            }
            break;
        }
        case Q_SLERP: rigm::slerp(q[a].data(), q[b].data(), r[c][0], q[dst].data()); break;
        case Q_EULER: {
            // quatToEuler: Maya's own decomposition of the quaternion's rotation, its choice of
            // euler at a half turn too ((0, 180, 0), where Shoemake's gives (180, 0, 180))
            double n[4];
            rigm::qnormal(q[a].data(), n);
            const MEulerRotation e = MEulerRotation::decompose(
                MQuaternion(n[0], n[1], n[2], n[3]).asMatrix(),
                static_cast<MEulerRotation::RotationOrder>(rigm::order_of(r[b][0])));
            r[dst] = {e.x * DEG, e.y * DEG, e.z * DEG};
            break;
        }
        case E_QUAT: {
            const double e[3] = {r[a][0] * RAD, r[a][1] * RAD, r[a][2] * RAD};
            rigm::euler_to_quat(e, rigm::order_of(r[b][0]), q[dst].data());
            break;
        }
        case AXIS_Q: rigm::axis_angle_to_quat(r[a].data(), r[b][0] * RAD, q[dst].data()); break;
        case Q_AXIS: {
            double axis[3], angle;
            rigm::quat_to_axis_angle(q[a].data(), axis, angle);
            r[dst] = {axis[0], axis[1], axis[2]};
            break;
        }
        case Q_ANGLE: {
            double axis[3], angle;
            rigm::quat_to_axis_angle(q[a].data(), axis, angle);
            r[dst] = {angle * DEG, 0.0, 0.0};
            break;
        }
        case M_BIN:
            if (c == 0)
                rigm::mmul(m[a].v, m[b].v, m[dst].v);
            else
                rigm::madd(m[a].v, m[b].v, m[dst].v);
            break;
        case M_SCALE: rigm::mscale(m[a].v, r[b][0], m[dst].v); break;
        case M_INV: rigm::inverse(m[a].v, m[dst].v); break;
        case M_T: rigm::transpose(m[a].v, m[dst].v); break;
        case M_DET: r[dst] = {rigm::det4(m[a].v), 0.0, 0.0}; break;
        case PT_MUL: { double o[3]; rigm::point(r[a].data(), m[b].v, o); r[dst] = {o[0], o[1], o[2]}; break; }
        case VEC_MUL: { double o[3]; rigm::vector(r[a].data(), m[b].v, o); r[dst] = {o[0], o[1], o[2]}; break; }
        case M_ROW: { const double* x = m[a].v + 4 * index_of(r[b][0], 3); q[dst] = {x[0], x[1], x[2], x[3]}; break; }
        case M_COL: { const double* x = m[a].v + index_of(r[b][0], 3);
                      q[dst] = {x[0], x[4], x[8], x[12]};
                      break; }
        case M_AXIS: { const double* x = m[a].v + 4 * (b % 3); const double k = b < 3 ? 1.0 : -1.0;
                       r[dst] = {k * x[0], k * x[1], k * x[2]};
                       break; }
        case M_TRANS: r[dst] = {m[a].v[12], m[a].v[13], m[a].v[14]}; break;
        case M_SCALE_OF: { rigm::Parts p; rigm::parts(m[a].v, p); r[dst] = {p.s[0], p.s[1], p.s[2]}; break; }
        case M_SHEAR_OF: { rigm::Parts p; rigm::parts(m[a].v, p); r[dst] = {p.sh[0], p.sh[1], p.sh[2]}; break; }
        case M_ROT: { rigm::Parts p; rigm::parts(m[a].v, p); std::copy(p.R, p.R + 16, m[dst].v); break; }
        case R_EULER: {
            // decomposeMatrix.outputRotate: MTransformationMatrix's rotation, reordered (Maya's
            // own choice of euler, a half turn's and a zero row's too)
            const double* x = m[a].v;
            const double rows[4][4] = {{x[0], x[1], x[2], x[3]}, {x[4], x[5], x[6], x[7]},
                                       {x[8], x[9], x[10], x[11]}, {x[12], x[13], x[14], x[15]}};
            MEulerRotation e = MTransformationMatrix(MMatrix(rows)).eulerRotation();
            e.reorderIt(static_cast<MEulerRotation::RotationOrder>(rigm::order_of(r[b][0])));
            r[dst] = {e.x * DEG, e.y * DEG, e.z * DEG};
            break;
        }
        case R_QUAT: rigm::quat_of(m[a].v, q[dst].data()); break;
        case E_MATRIX: {
            const double e[3] = {r[a][0] * RAD, r[a][1] * RAD, r[a][2] * RAD};
            rigm::euler_to_matrix(e, rigm::order_of(r[b][0]), m[dst].v);
            break;
        }
        case Q_MATRIX: rigm::quat_to_matrix(q[a].data(), m[dst].v); break;
        case COMPOSE: {
            const int* ext = ins + 5;
            rigm::compose(r[ext[2]].data(), m[a].v, r[b].data(), r[c].data(), m[dst].v);
            ++i;
            break;
        }
        case M_SETROW: {
            m[dst] = m[a];
            double* row = m[dst].v + 4 * c;
            row[0] = r[b][0];
            row[1] = r[b][1];
            row[2] = r[b][2];
            break;
        }
        case BLEND: rigm::blend(m[a].v, m[b].v, r[c][0], m[dst].v); break;
        case AIM: {
            const int* ext = ins + 5;
            rigm::aim(r[a].data(), r[b].data(), r[ext[2]].data(), r[ext[3]].data(), m[dst].v);
            ++i;
            break;
        }
        case Q_ARC: {
            const MQuaternion t(MVector(r[a][0], r[a][1], r[a][2]), MVector(r[b][0], r[b][1], r[b][2]));
            q[dst] = {t.x, t.y, t.z, t.w};
            break;
        }
        case TIME: r[dst] = {time_, 0.0, 0.0}; break;
        case RAND: r[dst] = {rigm::random01(r[a][0], r[b][0]), 0.0, 0.0}; break;
        case MOVE:
            if (c == 0)
                r[dst] = r[a];
            else if (c == 1)
                q[dst] = q[a];
            else
                m[dst] = m[a];
            break;
        }
    }
}

void ExprBase::count(MDataBlock& block, const MObject& attr)
{
    MDataHandle he = block.outputValue(attr);
    he.setInt(++evaluations_);
    he.setClean();
}

// the plug being computed, as its top-level attribute
MObject top_attribute(const MPlug& plug)
{
    MPlug root = plug;
    if (root.isChild())
        root = root.parent();
    if (root.isElement())
        root = root.array();
    return root.attribute();
}

// ----------------------------------------------------------------- rigExpression

MStatus RigExpression::initialize()
{
    MFnTypedAttribute typed;
    MFnNumericAttribute num;
    MFnUnitAttribute unit;
    MFnMatrixAttribute mat;

    aProgram = typed.create("program", "program", MFnData::kIntArray);
    aConstants = typed.create("constants", "constants", MFnData::kDoubleArray);

    auto input_array = [](MFnAttribute& fn) {
        fn.setArray(true);
        fn.setUsesArrayDataBuilder(true);
        fn.setKeyable(true);
    };
    auto output_array = [](MFnAttribute& fn) {
        fn.setArray(true);
        fn.setUsesArrayDataBuilder(true);
        fn.setWritable(false);
        fn.setStorable(false);
    };

    aScalarIn = num.create("scalarIn", "scalarIn", MFnNumericData::kDouble, 0.0);
    input_array(num);

    MObject vx = num.create("vectorInX", "vectorInX", MFnNumericData::kDouble, 0.0);
    MObject vy = num.create("vectorInY", "vectorInY", MFnNumericData::kDouble, 0.0);
    MObject vz = num.create("vectorInZ", "vectorInZ", MFnNumericData::kDouble, 0.0);
    aVectorIn = num.create("vectorIn", "vectorIn", vx, vy, vz);
    input_array(num);

    aAngleIn = unit.create("angleIn", "angleIn", MFnUnitAttribute::kAngle, 0.0);
    input_array(unit);

    MObject ex = unit.create("eulerInX", "eulerInX", MFnUnitAttribute::kAngle, 0.0);
    MObject ey = unit.create("eulerInY", "eulerInY", MFnUnitAttribute::kAngle, 0.0);
    MObject ez = unit.create("eulerInZ", "eulerInZ", MFnUnitAttribute::kAngle, 0.0);
    aEulerIn = num.create("eulerIn", "eulerIn", ex, ey, ez);
    input_array(num);

    MObject qx = num.create("quatInX", "quatInX", MFnNumericData::kDouble, 0.0);
    MObject qy = num.create("quatInY", "quatInY", MFnNumericData::kDouble, 0.0);
    MObject qz = num.create("quatInZ", "quatInZ", MFnNumericData::kDouble, 0.0);
    MObject qw = num.create("quatInW", "quatInW", MFnNumericData::kDouble, 1.0);
    aQuatIn = num.create("quatIn", "quatIn", qx, qy, qz, qw);
    input_array(num);

    aMatrixIn = mat.create("matrixIn", "matrixIn", MFnMatrixAttribute::kDouble);
    input_array(mat);

    aScalarOut = num.create("scalarOut", "scalarOut", MFnNumericData::kDouble, 0.0);
    output_array(num);

    MObject ox = num.create("vectorOutX", "vectorOutX", MFnNumericData::kDouble, 0.0);
    MObject oy = num.create("vectorOutY", "vectorOutY", MFnNumericData::kDouble, 0.0);
    MObject oz = num.create("vectorOutZ", "vectorOutZ", MFnNumericData::kDouble, 0.0);
    aVectorOut = num.create("vectorOut", "vectorOut", ox, oy, oz);
    output_array(num);

    MObject oqx = num.create("quatOutX", "quatOutX", MFnNumericData::kDouble, 0.0);
    MObject oqy = num.create("quatOutY", "quatOutY", MFnNumericData::kDouble, 0.0);
    MObject oqz = num.create("quatOutZ", "quatOutZ", MFnNumericData::kDouble, 0.0);
    MObject oqw = num.create("quatOutW", "quatOutW", MFnNumericData::kDouble, 1.0);
    aQuatOut = num.create("quatOut", "quatOut", oqx, oqy, oqz, oqw);
    output_array(num);

    aMatrixOut = mat.create("matrixOut", "matrixOut", MFnMatrixAttribute::kDouble);
    output_array(mat);

    MObject oex = unit.create("eulerOutX", "eulerOutX", MFnUnitAttribute::kAngle, 0.0);
    MObject oey = unit.create("eulerOutY", "eulerOutY", MFnUnitAttribute::kAngle, 0.0);
    MObject oez = unit.create("eulerOutZ", "eulerOutZ", MFnUnitAttribute::kAngle, 0.0);
    aEulerOut = num.create("eulerOut", "eulerOut", oex, oey, oez);
    output_array(num);

    aAngleOut = unit.create("angleOut", "angleOut", MFnUnitAttribute::kAngle, 0.0);
    output_array(unit);

    // how many times this node computed (diagnostics: written by compute, never dirtied)
    aEvaluations = num.create("evaluations", "evaluations", MFnNumericData::kInt, 0);
    num.setWritable(false);
    num.setStorable(false);

    // the scene's time, for TIME (rig.functions.frame()): connected only when a program reads it
    aTimeIn = unit.create("timeIn", "timeIn", MFnUnitAttribute::kTime, 0.0);

    const std::vector<MObject*> inputs = {&aProgram, &aConstants, &aScalarIn, &aVectorIn, &aAngleIn,
                                          &aEulerIn, &aQuatIn, &aMatrixIn, &aTimeIn};
    const std::vector<MObject*> outputs = {&aScalarOut, &aVectorOut, &aQuatOut, &aMatrixOut, &aEulerOut, &aAngleOut};
    for (MObject* a : inputs)
        addAttribute(*a);
    for (MObject* a : outputs)
        addAttribute(*a);
    addAttribute(aEvaluations);
    for (MObject* in : inputs)
        for (MObject* out : outputs)
            attributeAffects(*in, *out);
    return MS::kSuccess;
}

MStatus RigExpression::compute(const MPlug& plug, MDataBlock& block)
{
    const MObject attr = top_attribute(plug);
    if (attr != aScalarOut && attr != aVectorOut && attr != aQuatOut && attr != aMatrixOut && attr != aEulerOut &&
        attr != aAngleOut)
        return MS::kUnknownParameter;

    // every input pulled before the registers are touched
    MDataHandle hProgram = block.inputValue(aProgram);
    MDataHandle hConstants = block.inputValue(aConstants);
    MArrayDataHandle hs = block.inputArrayValue(aScalarIn);
    MArrayDataHandle hv = block.inputArrayValue(aVectorIn);
    MArrayDataHandle ha = block.inputArrayValue(aAngleIn);
    time_ = block.inputValue(aTimeIn).asTime().as(MTime::uiUnit());
    if (!load(hProgram, hConstants))
        return MS::kFailure;
    const bool v2 = program_[1] == 2;

    by_index(hs, sIn_, [](MDataHandle& h) { return h.asDouble(); }, 0.0);
    by_index(ha, aIn_, [](MDataHandle& h) { return h.asAngle().asRadians(); }, 0.0);
    by_index(hv, vIn_, [](MDataHandle& h) {
        const double3& d = h.asDouble3();
        return Reg{d[0], d[1], d[2]};
    }, Reg{0.0, 0.0, 0.0});
    if (v2) {
        MArrayDataHandle he = block.inputArrayValue(aEulerIn);
        MArrayDataHandle hq = block.inputArrayValue(aQuatIn);
        MArrayDataHandle hm = block.inputArrayValue(aMatrixIn);
        by_index(he, eIn_, [](MDataHandle& h) {
            const double3& d = h.asDouble3();
            return Reg{d[0], d[1], d[2]};
        }, Reg{0.0, 0.0, 0.0});
        by_index(hq, qIn_, [](MDataHandle& h) {
            const double4& d = h.asDouble4();
            return Reg4{d[0], d[1], d[2], d[3]};
        }, Reg4{0.0, 0.0, 0.0, 1.0});
        Reg16 id;
        std::copy(IDENTITY, IDENTITY + 16, id.v);
        by_index(hm, mIn_, [](MDataHandle& h) {
            const MMatrix& x = h.asMatrix();
            Reg16 o;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    o.v[4 * i + j] = x.matrix[i][j];
            return o;
        }, id);
    }
    run();

    const int n = program_[3], ns = program_[4], nv = program_[5];
    const int nq = field(8), nm = field(9), ne = field(10), na = field(11);
    const std::vector<Reg>& r = regs_;
    const int* outs = &program_[header() + n * 5];
    write_outputs(block, aScalarOut, ns, [&](int k, MDataHandle& h) { h.setDouble(r[outs[k]][0]); });
    write_outputs(block, aVectorOut, nv, [&](int k, MDataHandle& h) {
        const Reg& v = r[outs[ns + k]];
        h.set3Double(v[0], v[1], v[2]);
    });
    if (v2) {
        const int* oq = outs + ns + nv;
        const int* om = oq + nq;
        const int* oe = om + nm;
        const int* oa = oe + ne;
        write_outputs(block, aQuatOut, nq, [&](int k, MDataHandle& h) {
            const Reg4& x = regs4_[oq[k]];
            h.set4Double(x[0], x[1], x[2], x[3]);
        });
        write_outputs(block, aMatrixOut, nm, [&](int k, MDataHandle& h) {
            const double* x = regs16_[om[k]].v;
            const double rows[4][4] = {{x[0], x[1], x[2], x[3]}, {x[4], x[5], x[6], x[7]},
                                       {x[8], x[9], x[10], x[11]}, {x[12], x[13], x[14], x[15]}};
            h.setMMatrix(MMatrix(rows));
        });
        write_outputs(block, aEulerOut, ne, [&](int k, MDataHandle& h) {
            const Reg& v = r[oe[k]];
            h.set3Double(v[0] * RAD, v[1] * RAD, v[2] * RAD);
        });
        write_outputs(block, aAngleOut, na, [&](int k, MDataHandle& h) {
            h.setMAngle(MAngle(r[oa[k]][0] * RAD, MAngle::kRadians));
        });
    } else {
        for (const MObject* a : {&aQuatOut, &aMatrixOut, &aEulerOut, &aAngleOut})
            block.outputArrayValue(*a).setAllClean();
    }
    count(block, aEvaluations);
    return MS::kSuccess;
}

// ---------------------------------------------------------------------- the plug-in
// every node type of rig, registered in order (deregistered in reverse)
struct NodeType {
    const char* name;
    const MTypeId* id;
    MCreatorFunction creator;
    MInitializeFunction initialize;
};

const NodeType NODE_TYPES[] = {
    {"rigExpression", &RigExpression::id, RigExpression::creator, RigExpression::initialize},
};

MStatus initializePlugin(MObject obj)
{
    MFnPlugin plugin(obj, "rig", "1.2", "Any");
    const int count = int(sizeof(NODE_TYPES) / sizeof(NODE_TYPES[0]));
    for (int i = 0; i < count; ++i) {
        const NodeType& t = NODE_TYPES[i];
        MStatus status = plugin.registerNode(t.name, *t.id, t.creator, t.initialize);
        if (!status) {
            for (int j = i - 1; j >= 0; --j)
                plugin.deregisterNode(*NODE_TYPES[j].id);
            return status;
        }
    }
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj)
{
    MFnPlugin plugin(obj);
    MStatus result = MS::kSuccess;
    const int count = int(sizeof(NODE_TYPES) / sizeof(NODE_TYPES[0]));
    for (int i = count - 1; i >= 0; --i) {
        MStatus status = plugin.deregisterNode(*NODE_TYPES[i].id);
        if (!status)
            result = status;
    }
    return result;
}
