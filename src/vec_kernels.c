/* vec_kernels.c — SIMD/OpenMP vector reductions (dot, sum). */
#include "vec_kernels.h"
#include "a.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SHAKTI_USE_ACCELERATE
#define SHAKTI_USE_ACCELERATE 0
#endif
#if SHAKTI_USE_ACCELERATE
#include "shakti_accelerate.h"
#endif

/* OpenMP over query needles — far cheaper than binary search itself below this. */
#ifndef SHAKTI_BIN_OMP_MIN
#define SHAKTI_BIN_OMP_MIN 4096
#endif
#ifndef SHAKTI_BIN_OMP_MAX_THREADS
#define SHAKTI_BIN_OMP_MAX_THREADS 16
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
static inline int isl_vec_omp_threads(int64_t ne) {
    int max = omp_get_max_threads();
    int want;
    if (max <= 1 || ne < ISL_OMP_VEC_MIN) return 1;
    want = (int)(ne / ISL_OMP_VEC_CHUNK);
    if (want < 1) want = 1;
    if (want > max) want = max;
    if (want > ISL_OMP_VEC_MAX_THREADS) want = ISL_OMP_VEC_MAX_THREADS;
    return want;
}
#else
static inline int isl_vec_omp_threads(int64_t ne) { (void)ne; return 1; }
#endif

#if defined(__aarch64__)
static double dot_f64_neon(const double *a, const double *b, int64_t n) {
    float64x2_t acc = vdupq_n_f64(0);
    int64_t i = 0;
    for (; i + 2 <= n; i += 2)
        acc = vfmaq_f64(acc, vld1q_f64(a + i), vld1q_f64(b + i));
    double r = vaddvq_f64(acc);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
#endif

/* A float sum adds in one order on every platform and at any thread count, so its bits depend only
 * on the items: blocks of SUM_BLOCK items, each summed in SUM_LANES interleaved lanes (item i into
 * lane i % SUM_LANES), the lanes then folded in halves (lane k adds lane k + w for each k < w, for
 * w = SUM_LANES/2 down to 1), then the last n % SUM_LANES items added to lane 0 in order; the block
 * sums are added left to right. Fewer than SUM_LANES items add left to right. Threads only share
 * out whole blocks.
 * The lanes are held in pairs (lanes 2j and 2j + 1 in s[j]) of the compilers' generic vector type,
 * not any target's intrinsics: a + on pairs is two separate IEEE adds, so the order is as stated,
 * and clang and gcc keep the pairs in SIMD registers (gcc once the pragmas unroll the loops).
 * Folding in halves only ever adds whole pairs (s[j] += s[j + w/2]) until the last step, so no
 * lanes are shuffled. 32 lanes keep enough adds in flight to hide their latency on NEON and
 * AVX2. */
#define SUM_LANES 32
#define SUM_BLOCK 65536
typedef double sum_f64x2 __attribute__((vector_size(16)));
static double sum_f64_block(const double *d, int64_t n) {
    sum_f64x2 s[SUM_LANES / 2];
    for (int j = 0; j < SUM_LANES / 2; j++) s[j] = (sum_f64x2){0.0, 0.0};
    int64_t i = 0;
    for (; i + SUM_LANES <= n; i += SUM_LANES) {
#pragma GCC unroll 16
        for (int j = 0; j < SUM_LANES / 2; j++) {
            sum_f64x2 x;
            memcpy(&x, d + i + 2 * j, sizeof x);
            s[j] += x;
        }
    }
#pragma GCC unroll 8
    for (int w = SUM_LANES / 4; w > 0; w /= 2) {
#pragma GCC unroll 16
        for (int j = 0; j < w; j++) s[j] += s[j + w];
    }
    double r = s[0][0] + s[0][1];
    for (; i < n; i++) r += d[i];
    return r;
}

double shakti_sum_f64(const double *d, int64_t n) {
    if (n <= 0) return 0.0;
    if (n <= SUM_BLOCK) return sum_f64_block(d, n);
    int64_t nb = (n + SUM_BLOCK - 1) / SUM_BLOCK;
    double r = 0;
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    double *part = nt > 1 ? malloc((size_t)nb * sizeof *part) : NULL;
    if (part) {
        #pragma omp parallel for num_threads(nt)
        for (int64_t b = 0; b < nb; b++)
            part[b] = sum_f64_block(d + b * SUM_BLOCK, b < nb - 1 ? SUM_BLOCK : n - b * SUM_BLOCK);
        for (int64_t b = 0; b < nb; b++) r += part[b];
        free(part);
        return r;
    }
#endif
    for (int64_t b = 0; b < nb; b++)
        r += sum_f64_block(d + b * SUM_BLOCK, b < nb - 1 ? SUM_BLOCK : n - b * SUM_BLOCK);
    return r;
}

double shakti_min_f64(const double *d, int64_t n) {
    if (n <= 0) return 0.0;
    if (n == 1) return d[0];
    double r = d[0];
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(min:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 1; i < n; i++) if (d[i] < r) r = d[i];
    return r;
}

double shakti_max_f64(const double *d, int64_t n) {
    if (n <= 0) return 0.0;
    if (n == 1) return d[0];
    double r = d[0];
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(max:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 1; i < n; i++) if (d[i] > r) r = d[i];
    return r;
}

int64_t shakti_min_i64(const int64_t *d, int64_t n) {
    if (n <= 0) return 0;
    int64_t r = d[0];
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(min:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 1; i < n; i++) if (d[i] < r) r = d[i];
    return r;
}

int64_t shakti_max_i64(const int64_t *d, int64_t n) {
    if (n <= 0) return 0;
    int64_t r = d[0];
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(max:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 1; i < n; i++) if (d[i] > r) r = d[i];
    return r;
}

double shakti_dot_f64(const double *a, const double *b, int64_t n) {
    if (n <= 0) return 0.0;
#if SHAKTI_USE_ACCELERATE
    if (n >= SHAKTI_ACCEL_VEC_MIN) {
        double r = 0.0;
        vDSP_dotprD(a, 1, b, 1, &r, (vDSP_Length)n);
        return r;
    }
#endif
#if defined(__aarch64__)
    if (n >= ISL_OMP_VEC_MIN) {
        double r = 0;
#ifdef _OPENMP
        int nt = isl_vec_omp_threads(n);
        #pragma omp parallel for reduction(+:r) num_threads(nt)
#endif
        for (int64_t i = 0; i < n; i++) r += a[i] * b[i];
        return r;
    }
    return dot_f64_neon(a, b, n);
#else
    double r = 0;
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(+:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 0; i < n; i++) r += a[i] * b[i];
    return r;
#endif
}

double shakti_dot_numeric(const int64_t *aj, const double *af, int a_fvec,
                          const int64_t *bj, const double *bf, int b_fvec,
                          int64_t n) {
    if (n <= 0) return 0.0;
    if (a_fvec && b_fvec) return shakti_dot_f64(af, bf, n);
    double r = 0;
#ifdef _OPENMP
    int nt = isl_vec_omp_threads(n);
    #pragma omp parallel for reduction(+:r) if (n >= ISL_OMP_VEC_MIN) num_threads(nt)
#endif
    for (int64_t i = 0; i < n; i++) {
        double x = a_fvec ? af[i] : (double)aj[i];
        double y = b_fvec ? bf[i] : (double)bj[i];
        r += x * y;
    }
    return r;
}

/* ── q-compatible predecessor search (bin) ───────────────────────────── */

int64_t shakti_bin_i64(const int64_t *keys, int64_t n, int64_t q) {
    if (!keys || n <= 0) return -1;
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (keys[mid] <= q) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

static int shakti_i64_ascending(const int64_t *a, int64_t n) {
    for (int64_t i = 1; i < n; i++)
        if (a[i] < a[i - 1]) return 0;
    return 1;
}

void shakti_bin_i64_batch(const int64_t *keys, int64_t n,
                          const int64_t *qs, int64_t m, int64_t *out) {
    if (!out) return;
    if (!keys || n <= 0 || !qs || m <= 0) {
        for (int64_t j = 0; j < m; j++) out[j] = -1;
        return;
    }
    /* Sorted needles → O(n+m) two-pointer merge. */
    if (m >= 2 && shakti_i64_ascending(qs, m)) {
        int64_t i = 0;
        for (int64_t j = 0; j < m; j++) {
            while (i < n && keys[i] <= qs[j]) i++;
            out[j] = i - 1;
        }
        return;
    }
#ifdef _OPENMP
    int max = omp_get_max_threads();
    int nt = 1;
    if (max > 1 && m >= SHAKTI_BIN_OMP_MIN) {
        nt = (int)(m / (SHAKTI_BIN_OMP_MIN / 4));
        if (nt < 1) nt = 1;
        if (nt > max) nt = max;
        if (nt > SHAKTI_BIN_OMP_MAX_THREADS) nt = SHAKTI_BIN_OMP_MAX_THREADS;
    }
    #pragma omp parallel for schedule(static) if (m >= SHAKTI_BIN_OMP_MIN) num_threads(nt)
#endif
    for (int64_t j = 0; j < m; j++)
        out[j] = shakti_bin_i64(keys, n, qs[j]);
}

int64_t shakti_bin_f64(const double *keys, int64_t n, double q) {
    /* NaN query is incomparable; treat as below range. */
    if (!keys || n <= 0 || isnan(q)) return -1;
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (keys[mid] <= q) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

static int shakti_f64_ascending(const double *a, int64_t n) {
    for (int64_t i = 1; i < n; i++)
        if (a[i] < a[i - 1]) return 0;
    return 1;
}

void shakti_bin_f64_batch(const double *keys, int64_t n,
                          const double *qs, int64_t m, int64_t *out) {
    if (!out) return;
    if (!keys || n <= 0 || !qs || m <= 0) {
        for (int64_t j = 0; j < m; j++) out[j] = -1;
        return;
    }
    if (m >= 2 && shakti_f64_ascending(qs, m)) {
        int64_t i = 0;
        for (int64_t j = 0; j < m; j++) {
            if (isnan(qs[j])) { out[j] = -1; continue; }
            while (i < n && keys[i] <= qs[j]) i++;
            out[j] = i - 1;
        }
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (m >= SHAKTI_BIN_OMP_MIN)
#endif
    for (int64_t j = 0; j < m; j++)
        out[j] = shakti_bin_f64(keys, n, qs[j]);
}

typedef struct { int64_t eq, tm; } ShaktiEt;

static int shakti_cmp_et(const void *a, const void *b) {
    const ShaktiEt *x = (const ShaktiEt *)a, *y = (const ShaktiEt *)b;
    if (x->eq < y->eq) return -1;
    if (x->eq > y->eq) return 1;
    if (x->tm < y->tm) return -1;
    if (x->tm > y->tm) return 1;
    return 0;
}

void shakti_asof_sort_i64(const int64_t *eq, const int64_t *tm, int64_t n,
                          int64_t *eq_out, int64_t *tm_out) {
    if (!eq_out || !tm_out) return;
    if (!eq || !tm || n <= 0) return;
    ShaktiEt *buf = (ShaktiEt *)malloc((size_t)n * sizeof(ShaktiEt));
    if (!buf) return;
    for (int64_t i = 0; i < n; i++) {
        buf[i].eq = eq[i];
        buf[i].tm = tm[i];
    }
    qsort(buf, (size_t)n, sizeof(ShaktiEt), shakti_cmp_et);
    for (int64_t i = 0; i < n; i++) {
        eq_out[i] = buf[i].eq;
        tm_out[i] = buf[i].tm;
    }
    free(buf);
}

static int64_t shakti_group_start(const int64_t *eq, int64_t n, int64_t key) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (eq[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= n || eq[lo] != key) return -1;
    return lo;
}

static int64_t shakti_group_end(const int64_t *eq, int64_t n, int64_t key, int64_t start) {
    int64_t lo = start, hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (eq[mid] <= key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

void shakti_asof_bin_i64(const int64_t *eq, const int64_t *tm, int64_t n,
                         const int64_t *query_eq, const int64_t *query_tm,
                         int64_t m, int64_t scalar_tm, int64_t *out) {
    if (!out) return;
    if (!eq || !tm || n <= 0 || !query_eq || m <= 0) {
        for (int64_t j = 0; j < m; j++) out[j] = -1;
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (m >= SHAKTI_BIN_OMP_MIN)
#endif
    for (int64_t j = 0; j < m; j++) {
        int64_t start = shakti_group_start(eq, n, query_eq[j]);
        if (start < 0) { out[j] = -1; continue; }
        int64_t end = shakti_group_end(eq, n, query_eq[j], start);
        int64_t qtm = query_tm ? query_tm[j] : scalar_tm;
        int64_t rel = shakti_bin_i64(tm + start, end - start, qtm);
        out[j] = (rel < 0) ? -1 : (start + rel);
    }
}

double shakti_dot_f32(const float *a, const float *b, int64_t n) {
    int64_t i;
    double r = 0.0;
    if (n <= 0 || !a || !b) return 0.0;
    for (i = 0; i < n; i++) r += (double)a[i] * (double)b[i];
    return r;
}

double shakti_sumsq_f32(const float *d, int64_t n) {
    if (n <= 0 || !d) return 0.0;
    return shakti_dot_f32(d, d, n);
}
