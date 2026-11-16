// rig_math.h: the matrix and quaternion math of rig's nodes (rig.mll), written to give
// what Maya's own nodes give: decomposeMatrix, composeMatrix, inverseMatrix,
// multMatrix, quatSlerp, quatToEuler, aimMatrix, blendMatrix ... (each function names
// the node it stands for). Checked against those nodes by check_math_nodes.py, and
// timed against Maya's classes (MMatrix, MTransformationMatrix, MEulerRotation,
// MQuaternion) by rigMathBench.cpp: 1.4x to 11x faster than the classes, equal to
// them within float rounding (the matrix product bit for bit).
//
// Conventions (Maya's):
//   a matrix is 16 doubles, row-major, for row vectors: p' = p * M, the translation
//   in row 3, and a transform is S * SH * R * T (scale, shear xy xz yz, rotation,
//   translation);
//   a quaternion is x, y, z, w (MQuaternion's); p * q is p, then q;
//   rotate orders 0..5 are xyz, yzx, zxy, xzy, yxz, zyx, and xyz means x first
//   (R = Rx * Ry * Rz); angles are radians here (the node converts).
#pragma once

#include <emmintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rigm {

constexpr double PI = 3.14159265358979323846;

// --------------------------------------------------------------- matrices
inline void identity(double* m)
{
    for (int k = 0; k < 16; ++k)
        m[k] = (k % 5 == 0) ? 1.0 : 0.0;
}

// c = a * b (multMatrix; MMatrix::operator*: the same sums in the same order, bit for
// bit). SSE2: a row of c is a[i][0] * b0 + a[i][1] * b1 + a[i][2] * b2 + a[i][3] * b3.
// c is neither a nor b (the program's registers are written once).
inline void mmul(const double* a, const double* b, double* c)
{
    const __m128d b0l = _mm_loadu_pd(b), b0h = _mm_loadu_pd(b + 2), b1l = _mm_loadu_pd(b + 4),
                  b1h = _mm_loadu_pd(b + 6), b2l = _mm_loadu_pd(b + 8), b2h = _mm_loadu_pd(b + 10),
                  b3l = _mm_loadu_pd(b + 12), b3h = _mm_loadu_pd(b + 14);
    for (int i = 0; i < 4; ++i) {
        const __m128d a0 = _mm_set1_pd(a[4 * i]), a1 = _mm_set1_pd(a[4 * i + 1]), a2 = _mm_set1_pd(a[4 * i + 2]),
                      a3 = _mm_set1_pd(a[4 * i + 3]);
        const __m128d lo = _mm_add_pd(
            _mm_add_pd(_mm_add_pd(_mm_mul_pd(a0, b0l), _mm_mul_pd(a1, b1l)), _mm_mul_pd(a2, b2l)),
            _mm_mul_pd(a3, b3l));
        const __m128d hi = _mm_add_pd(
            _mm_add_pd(_mm_add_pd(_mm_mul_pd(a0, b0h), _mm_mul_pd(a1, b1h)), _mm_mul_pd(a2, b2h)),
            _mm_mul_pd(a3, b3h));
        _mm_storeu_pd(c + 4 * i, lo);
        _mm_storeu_pd(c + 4 * i + 2, hi);
    }
}

inline void madd(const double* a, const double* b, double* c)   // addMatrix
{
    for (int k = 0; k < 16; ++k)
        c[k] = a[k] + b[k];
}

inline void mscale(const double* a, double w, double* c)         // a wtAddMatrix term
{
    for (int k = 0; k < 16; ++k)
        c[k] = a[k] * w;
}

inline void transpose(const double* a, double* c)                // transposeMatrix
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            c[4 * i + j] = a[4 * j + i];
}

// the 4x4 determinant (the determinant node; MMatrix::det4x4)
inline double det4(const double* m)
{
    const double s0 = m[0] * m[5] - m[4] * m[1], s1 = m[0] * m[6] - m[4] * m[2], s2 = m[0] * m[7] - m[4] * m[3];
    const double s3 = m[1] * m[6] - m[5] * m[2], s4 = m[1] * m[7] - m[5] * m[3], s5 = m[2] * m[7] - m[6] * m[3];
    const double c5 = m[10] * m[15] - m[14] * m[11], c4 = m[9] * m[15] - m[13] * m[11];
    const double c3 = m[9] * m[14] - m[13] * m[10], c2 = m[8] * m[15] - m[12] * m[11];
    const double c1 = m[8] * m[14] - m[12] * m[10], c0 = m[8] * m[13] - m[12] * m[9];
    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}

// the inverse (inverseMatrix; MMatrix::inverse): identity for a singular matrix. An
// affine matrix (last column 0 0 0 1, every matrix a transform makes) inverts its 3x3
// and moves the translation; any other by cofactors
inline void inverse(const double* m, double* inv)
{
    if (m[3] == 0.0 && m[7] == 0.0 && m[11] == 0.0 && m[15] == 1.0) {
        const double c00 = m[5] * m[10] - m[6] * m[9], c01 = m[6] * m[8] - m[4] * m[10],
                     c02 = m[4] * m[9] - m[5] * m[8];
        const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if (det == 0.0) {
            identity(inv);
            return;
        }
        const double k = 1.0 / det;
        inv[0] = c00 * k;
        inv[1] = (m[2] * m[9] - m[1] * m[10]) * k;
        inv[2] = (m[1] * m[6] - m[2] * m[5]) * k;
        inv[4] = c01 * k;
        inv[5] = (m[0] * m[10] - m[2] * m[8]) * k;
        inv[6] = (m[2] * m[4] - m[0] * m[6]) * k;
        inv[8] = c02 * k;
        inv[9] = (m[1] * m[8] - m[0] * m[9]) * k;
        inv[10] = (m[0] * m[5] - m[1] * m[4]) * k;
        inv[3] = inv[7] = inv[11] = 0.0;
        inv[15] = 1.0;
        const double tx = m[12], ty = m[13], tz = m[14];
        inv[12] = -(tx * inv[0] + ty * inv[4] + tz * inv[8]);
        inv[13] = -(tx * inv[1] + ty * inv[5] + tz * inv[9]);
        inv[14] = -(tx * inv[2] + ty * inv[6] + tz * inv[10]);
        return;
    }
    double t[16];
    t[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
           m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    t[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
           m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    t[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
           m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    t[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
            m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    t[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
           m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    t[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
           m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    t[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
           m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    t[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
            m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    t[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
           m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    t[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
           m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    t[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
            m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    t[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
            m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    t[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    t[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    t[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    t[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = m[0] * t[0] + m[1] * t[4] + m[2] * t[8] + m[3] * t[12];
    if (det == 0.0) {
        identity(inv);
        return;
    }
    const double k = 1.0 / det;
    for (int i = 0; i < 16; ++i)
        inv[i] = t[i] * k;
}

// p * M with w = 1, no division by the result's w (multiplyPointByMatrix)
inline void point(const double* p, const double* m, double* out)
{
    const double x = p[0], y = p[1], z = p[2];
    out[0] = x * m[0] + y * m[4] + z * m[8] + m[12];
    out[1] = x * m[1] + y * m[5] + z * m[9] + m[13];
    out[2] = x * m[2] + y * m[6] + z * m[10] + m[14];
}

// v * M without the translation (multiplyVectorByMatrix)
inline void vector(const double* v, const double* m, double* out)
{
    const double x = v[0], y = v[1], z = v[2];
    out[0] = x * m[0] + y * m[4] + z * m[8];
    out[1] = x * m[1] + y * m[5] + z * m[9];
    out[2] = x * m[2] + y * m[6] + z * m[10];
}

// ---------------------------------------------------------- rotate orders
// Shoemake's Euler angle conversion (Graphics Gems IV), static frame: Maya's order xyz
// (x first, row vectors) is Shoemake's xyz on the column-vector matrix, Maya's
// transpose. The first axis i, then j, k, and the parity n
struct Order {
    int i, j, k, n;
};
constexpr Order ORDERS[6] = {{0, 1, 2, 0}, {1, 2, 0, 0}, {2, 0, 1, 0}, {0, 2, 1, 1}, {1, 0, 2, 1}, {2, 1, 0, 1}};

inline int order_of(double v)   // a rotate order read from a register: rounded, 0..5
{
    const int o = int(std::lround(v));
    return o < 0 ? 0 : o > 5 ? 5 : o;
}

// the rotation matrix of the euler r (radians) in order o (MEulerRotation::asMatrix)
inline void euler_to_matrix(const double* r, int o, double* m)
{
    const Order& ord = ORDERS[o];
    double ti = r[ord.i], tj = r[ord.j], th = r[ord.k];
    if (ord.n) {
        ti = -ti;
        tj = -tj;
        th = -th;
    }
    const double ci = std::cos(ti), cj = std::cos(tj), ch = std::cos(th);
    const double si = std::sin(ti), sj = std::sin(tj), sh = std::sin(th);
    const double cc = ci * ch, cs = ci * sh, sc = si * ch, ss = si * sh;
    double c[3][3];  // the column-vector matrix
    const int i = ord.i, j = ord.j, k = ord.k;
    c[i][i] = cj * ch;
    c[i][j] = sj * sc - cs;
    c[i][k] = sj * cc + ss;
    c[j][i] = cj * sh;
    c[j][j] = sj * ss + cc;
    c[j][k] = sj * cs - sc;
    c[k][i] = -sj;
    c[k][j] = cj * si;
    c[k][k] = cj * ci;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            m[4 * a + b] = c[b][a];
    m[3] = m[7] = m[11] = m[12] = m[13] = m[14] = 0.0;
    m[15] = 1.0;
}

// the quaternion of the euler r (radians) in order o (eulerToQuat; asQuaternion)
inline void euler_to_quat(const double* r, int o, double* q)
{
    const Order& ord = ORDERS[o];
    double ti = r[ord.i] * 0.5, tj = r[ord.j] * 0.5, th = r[ord.k] * 0.5;
    if (ord.n)
        tj = -tj;
    const double ci = std::cos(ti), cj = std::cos(tj), ch = std::cos(th);
    const double si = std::sin(ti), sj = std::sin(tj), sh = std::sin(th);
    const double cc = ci * ch, cs = ci * sh, sc = si * ch, ss = si * sh;
    double a[3];
    a[ord.i] = cj * sc - sj * cs;
    a[ord.j] = cj * ss + sj * cc;
    a[ord.k] = cj * cs - sj * sc;
    if (ord.n)
        a[ord.j] = -a[ord.j];
    q[0] = a[0];
    q[1] = a[1];
    q[2] = a[2];
    q[3] = cj * cc + sj * ss;
}

// the euler (radians) of a rotation matrix (rows orthonormal) in order o: Maya's
// MEulerRotation::decompose (decomposeMatrix.outputRotate). Shoemake's solution has
// its middle angle in [-90, 90]; Maya gives the other one, (first + pi, pi - middle,
// last + pi), when the sum of its angles' magnitudes is smaller
inline void euler_of(const double* m, int o, double* r)
{
    const Order& ord = ORDERS[o];
    const int i = ord.i, j = ord.j, k = ord.k;
    auto C = [m](int a, int b) { return m[4 * b + a]; };  // the column-vector matrix
    const double cy = std::sqrt(C(i, i) * C(i, i) + C(j, i) * C(j, i));
    double x, y, z;
    if (cy > 16 * 1.1920928955078125e-07) {
        x = std::atan2(C(k, j), C(k, k));
        y = std::atan2(-C(k, i), cy);
        z = std::atan2(C(j, i), C(i, i));
    } else {
        x = std::atan2(-C(j, k), C(j, j));
        y = std::atan2(-C(k, i), cy);
        z = 0.0;
    }
    if (ord.n) {
        x = -x;
        y = -y;
        z = -z;
    }
    const double ax = x + PI, ay = PI - y, az = z + PI;
    if (std::fabs(ax) + std::fabs(ay) + std::fabs(az) < std::fabs(x) + std::fabs(y) + std::fabs(z)) {
        x = ax;
        y = ay;
        z = az;
    }
    r[i] = x;
    r[j] = y;
    r[k] = z;
}

// decomposeMatrix.outputRotate: the euler of R in order o. A zero scale leaves R with a
// zero row (no rotation): Maya then reorders R's xyz euler, the rotation of which is R's
// only where R is one
inline void decompose_euler(const double* R, int o, double* r)
{
    if (o != 0 && (R[0] * R[0] + R[1] * R[1] + R[2] * R[2] < 0.25 || R[4] * R[4] + R[5] * R[5] + R[6] * R[6] < 0.25 ||
                   R[8] * R[8] + R[9] * R[9] + R[10] * R[10] < 0.25)) {
        double e[3], M[16];
        euler_of(R, 0, e);
        euler_to_matrix(e, 0, M);
        euler_of(M, o, r);
        return;
    }
    euler_of(R, o, r);
}

// --------------------------------------------------------------- quaternions
// p * q, Maya's: p, then q (the Hamilton product q p); quatProd
inline void qmul(const double* p, const double* q, double* o)
{
    const double px = p[0], py = p[1], pz = p[2], pw = p[3], qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    o[0] = qw * px + qx * pw + qy * pz - qz * py;
    o[1] = qw * py - qx * pz + qy * pw + qz * px;
    o[2] = qw * pz + qx * py - qy * px + qz * pw;
    o[3] = qw * pw - qx * px - qy * py - qz * pz;
}

// quatNormalize (MQuaternion::normal): the identity for a zero quaternion
inline void qnormal(const double* q, double* o)
{
    const double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (l == 0.0) {
        o[0] = o[1] = o[2] = 0.0;
        o[3] = 1.0;
        return;
    }
    const double k = 1.0 / l;
    o[0] = q[0] * k;
    o[1] = q[1] * k;
    o[2] = q[2] * k;
    o[3] = q[3] * k;
}

// quatInvert: the conjugate over the norm squared (nan for a zero quaternion, as Maya's)
inline void qinverse(const double* q, double* o)
{
    const double n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    o[0] = -q[0] / n;
    o[1] = -q[1] / n;
    o[2] = -q[2] / n;
    o[3] = q[3] / n;
}

// the rotation matrix of q (MQuaternion::asMatrix, composeMatrix's inputQuat): a q not of
// length 1 is scaled by 2 / |q|^2 (nan for a zero quaternion, as Maya's)
inline void quat_to_matrix(const double* q, double* m)
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    const double s = 2.0 / (x * x + y * y + z * z + w * w);
    const double xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y,
                 wz = w * z;
    m[0] = 1.0 - s * (yy + zz);
    m[1] = s * (xy + wz);
    m[2] = s * (xz - wy);
    m[4] = s * (xy - wz);
    m[5] = 1.0 - s * (xx + zz);
    m[6] = s * (yz + wx);
    m[8] = s * (xz + wy);
    m[9] = s * (yz - wx);
    m[10] = 1.0 - s * (xx + yy);
    m[3] = m[7] = m[11] = m[12] = m[13] = m[14] = 0.0;
    m[15] = 1.0;
}

// a rotation matrix's quaternion as decomposeMatrix.outputQuat gives it: from its xyz
// euler (MTransformationMatrix::rotation), which decides its sign
inline void quat_of(const double* m, double* q)
{
    double r[3];
    euler_of(m, 0, r);
    euler_to_quat(r, 0, q);
}

// a rotation matrix's quaternion, any sign (Shepperd): where only the rotation counts
inline void quat_any_sign(const double* m, double* q)
{
    const double trace = m[0] + m[5] + m[10];
    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        q[3] = 0.25 / s;
        q[0] = (m[6] - m[9]) * s;
        q[1] = (m[8] - m[2]) * s;
        q[2] = (m[1] - m[4]) * s;
    } else if (m[0] > m[5] && m[0] > m[10]) {
        const double s = 2.0 * std::sqrt(1.0 + m[0] - m[5] - m[10]);
        q[3] = (m[6] - m[9]) / s;
        q[0] = 0.25 * s;
        q[1] = (m[4] + m[1]) / s;
        q[2] = (m[8] + m[2]) / s;
    } else if (m[5] > m[10]) {
        const double s = 2.0 * std::sqrt(1.0 + m[5] - m[0] - m[10]);
        q[3] = (m[8] - m[2]) / s;
        q[0] = (m[4] + m[1]) / s;
        q[1] = 0.25 * s;
        q[2] = (m[9] + m[6]) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + m[10] - m[0] - m[5]);
        q[3] = (m[1] - m[4]) / s;
        q[0] = (m[8] + m[2]) / s;
        q[1] = (m[9] + m[6]) / s;
        q[2] = 0.25 * s;
    }
}

// the euler (radians) of q in order o: quatToEuler (q normalized, a zero one the
// identity, then decompose)
inline void quat_to_euler(const double* q, int o, double* r)
{
    double n[4], m[16];
    qnormal(q, n);
    quat_to_matrix(n, m);
    euler_of(m, o, r);
}

// Maya's slerp (quatSlerp, its Shortest mode; blendMatrix): the shortest way, q flipped
// to p's side; nearly equal quaternions are blended linearly
inline void slerp(const double* p, const double* qin, double t, double* o)
{
    double q[4] = {qin[0], qin[1], qin[2], qin[3]};
    double c = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3];
    if (c < 0.0) {
        c = -c;
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    double k0, k1;
    if (1.0 - c > 1e-10) {
        const double angle = std::acos(std::min(1.0, c));
        const double s = std::sin(angle);
        k0 = std::sin((1.0 - t) * angle) / s;
        k1 = std::sin(t * angle) / s;
    } else {
        k0 = 1.0 - t;
        k1 = t;
    }
    for (int k = 0; k < 4; ++k)
        o[k] = k0 * p[k] + k1 * q[k];
}

// axisAngleToQuat: the axis normalized (MQuaternion(angle, axis)); a zero axis, the identity
inline void axis_angle_to_quat(const double* axis, double angle, double* q)
{
    const double l = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (l == 0.0) {
        q[0] = q[1] = q[2] = 0.0;
        q[3] = 1.0;
        return;
    }
    const double k = std::sin(angle * 0.5) / l;
    q[0] = axis[0] * k;
    q[1] = axis[1] * k;
    q[2] = axis[2] * k;
    q[3] = std::cos(angle * 0.5);
}

// quatToAxisAngle: of q normalized (a zero one the identity; MQuaternion::getAxisAngle);
// no rotation: the angle 0 about z
inline void quat_to_axis_angle(const double* qin, double* axis, double& angle)
{
    double q[4];
    qnormal(qin, q);
    const double w = std::max(-1.0, std::min(1.0, q[3]));
    angle = 2.0 * std::acos(w);
    const double s = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    if (s > 0.0) {
        axis[0] = q[0] / s;
        axis[1] = q[1] / s;
        axis[2] = q[2] / s;
    } else {
        axis[0] = 0.0;
        axis[1] = 0.0;
        axis[2] = 1.0;
    }
}

// ------------------------------------------------- decomposition, composition
inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// MTransformationMatrix's decomposition of m = S * SH * R * T (decomposeMatrix,
// scaleFromMatrix ...): Gram-Schmidt on the rows; a negative determinant puts the
// negative scale on z. R is written as a matrix (rows orthonormal). A row whose
// remaining length is 1e-10 or less (a zero scale) is not normalized and its shear not
// divided: it stays (nearly) zero in R, as Maya's
struct Parts {
    double s[3], sh[3], R[16];
};

constexpr double SCALE_EPSILON = 1e-10;

inline void parts(const double* m, Parts& p)
{
    double r0[3] = {m[0], m[1], m[2]}, r1[3] = {m[4], m[5], m[6]}, r2[3] = {m[8], m[9], m[10]};
    double sx = std::sqrt(dot3(r0, r0));
    if (sx > SCALE_EPSILON)
        for (double& c : r0) c /= sx;
    double xy = dot3(r0, r1);
    for (int k = 0; k < 3; ++k) r1[k] -= xy * r0[k];
    double sy = std::sqrt(dot3(r1, r1));
    if (sy > SCALE_EPSILON) {
        for (double& c : r1) c /= sy;
        xy /= sy;
    }
    double xz = dot3(r0, r2);
    for (int k = 0; k < 3; ++k) r2[k] -= xz * r0[k];
    double yz = dot3(r1, r2);
    for (int k = 0; k < 3; ++k) r2[k] -= yz * r1[k];
    double sz = std::sqrt(dot3(r2, r2));
    if (sz > SCALE_EPSILON) {
        for (double& c : r2) c /= sz;
        xz /= sz;
        yz /= sz;
    }
    const double det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) +
                       r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
    if (det < 0.0) {
        sz = -sz;
        xz = -xz;
        yz = -yz;
        for (double& c : r2) c = -c;
    }
    p.s[0] = sx;
    p.s[1] = sy;
    p.s[2] = sz;
    p.sh[0] = xy;
    p.sh[1] = xz;
    p.sh[2] = yz;
    double* R = p.R;
    R[0] = r0[0], R[1] = r0[1], R[2] = r0[2], R[3] = 0.0;
    R[4] = r1[0], R[5] = r1[1], R[6] = r1[2], R[7] = 0.0;
    R[8] = r2[0], R[9] = r2[1], R[10] = r2[2], R[11] = 0.0;
    R[12] = R[13] = R[14] = 0.0;
    R[15] = 1.0;
}

// S * SH * R, then the translation t (composeMatrix; MTransformationMatrix::asMatrix)
inline void compose(const double* t, const double* R, const double* s, const double* sh, double* m)
{
    const double xy = sh[0], xz = sh[1], yz = sh[2];
    for (int k = 0; k < 3; ++k) {
        const double R0 = R[k], R1 = R[4 + k], R2 = R[8 + k];
        m[k] = s[0] * R0;
        m[4 + k] = s[1] * (xy * R0 + R1);
        m[8 + k] = s[2] * (xz * R0 + yz * R1 + R2);
    }
    m[3] = m[7] = m[11] = 0.0;
    // (a nan rotation, a zero quaternion's, spreads to the translation, as Maya's product does)
    const double nan = 0.0 * (R[0] + R[5] + R[10]);
    m[12] = t[0] + nan;
    m[13] = t[1] + nan;
    m[14] = t[2] + nan;
    m[15] = 1.0;
}

// blendMatrix of one target with weight w (rig's matrix.blend, and matrix.pow from the
// identity): translate, scale and shear blended linearly, the rotation by slerp; no clamp
inline void blend(const double* m0, const double* m1, double w, double* out)
{
    Parts p0, p1;
    parts(m0, p0);
    parts(m1, p1);
    double q0[4], q1[4], q[4], R[16], t[3], s[3], sh[3];
    // a quaternion's sign does not change the slerp's rotation: Shepperd's, unless a zero
    // scale left R no rotation (then Maya's own, from R's euler)
    auto quat = [](const Parts& p, double* out) {
        const bool zero = std::fabs(p.s[0]) <= SCALE_EPSILON || std::fabs(p.s[1]) <= SCALE_EPSILON ||
                          std::fabs(p.s[2]) <= SCALE_EPSILON;
        if (zero)
            quat_of(p.R, out);
        else
            quat_any_sign(p.R, out);
    };
    quat(p0, q0);
    quat(p1, q1);
    slerp(q0, q1, w, q);
    quat_to_matrix(q, R);
    const double u = 1.0 - w;
    for (int k = 0; k < 3; ++k) {
        t[k] = u * m0[12 + k] + w * m1[12 + k];
        s[k] = u * p0.s[k] + w * p1.s[k];
        sh[k] = u * p0.sh[k] + w * p1.sh[k];
    }
    compose(t, R, s, sh, out);
}

// the rotation taking direction u to direction v the shortest way (MQuaternion(u, v)).
// Opposite directions take half a turn: about `about` when given (a unit axis across
// both), else about any axis perpendicular to u
inline bool arc(const double* u0, const double* v0, double* m, const double* about = nullptr)
{
    const double lu = std::sqrt(dot3(u0, u0)), lv = std::sqrt(dot3(v0, v0));
    if (lu == 0.0 || lv == 0.0)
        return false;
    const double u[3] = {u0[0] / lu, u0[1] / lu, u0[2] / lu}, v[3] = {v0[0] / lv, v0[1] / lv, v0[2] / lv};
    const double c = dot3(u, v);
    double q[4];
    if (c < -1.0 + 1e-12) {
        double a[3] = {0.0, -u[2], u[1]};
        if (about) {
            a[0] = about[0];
            a[1] = about[1];
            a[2] = about[2];
        } else if (dot3(a, a) < 1e-12) {
            a[0] = u[2];
            a[1] = 0.0;
            a[2] = -u[0];
        }
        const double la = std::sqrt(dot3(a, a));
        q[0] = a[0] / la, q[1] = a[1] / la, q[2] = a[2] / la, q[3] = 0.0;
    } else {
        q[0] = u[1] * v[2] - u[2] * v[1];
        q[1] = u[2] * v[0] - u[0] * v[2];
        q[2] = u[0] * v[1] - u[1] * v[0];
        q[3] = 1.0 + c;
    }
    quat_to_matrix(q, m);
    return true;
}

// aimMatrix as rig.matrix.aim makes it (identity input, primary aim, secondary align):
// aim_axis turned onto aim, then about aim until up_axis is toward up. Identity when
// aim or up is zero, or up is along aim
inline void aim(const double* aimv, const double* up, const double* aim_axis, const double* up_axis, double* out)
{
    double R1[16], R2[16];
    if (!arc(aim_axis, aimv, R1)) {
        identity(out);
        return;
    }
    const double la = std::sqrt(dot3(aimv, aimv));
    const double a[3] = {aimv[0] / la, aimv[1] / la, aimv[2] / la};
    double s1[3];
    vector(up_axis, R1, s1);
    const double ds = dot3(s1, a), du = dot3(up, a);
    const double sp[3] = {s1[0] - ds * a[0], s1[1] - ds * a[1], s1[2] - ds * a[2]};
    const double upp[3] = {up[0] - du * a[0], up[1] - du * a[1], up[2] - du * a[2]};
    // an up (or up axis) along the aim leaves only rounding across it: no second turn
    const bool along = dot3(upp, upp) <= 1e-20 * dot3(up, up) || dot3(sp, sp) <= 1e-20 * dot3(s1, s1);
    // sp and upp both lie across the aim, so the second turn is about it: half a turn
    // when they are opposite (an arc's own pick there would turn the aim away)
    if (along || !arc(sp, upp, R2, a)) {
        for (int k = 0; k < 16; ++k)
            out[k] = R1[k];
        return;
    }
    mmul(R1, R2, out);
}

// rig.random, fused (RAND): a counter-based hash of the seed and the trigger (the scene's
// time by default), splitmix64's finalizer over their bits. The same seed and trigger give
// the same value, in [0, 1), 53 random bits; -0 counts as +0.
inline std::uint64_t mix64(std::uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline double random01(double seed, double trigger)
{
    const double s = seed + 0.0, t = trigger + 0.0;
    std::uint64_t a, b;
    std::memcpy(&a, &s, sizeof a);
    std::memcpy(&b, &t, sizeof b);
    return static_cast<double>(mix64(mix64(a) ^ b) >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace rigm
