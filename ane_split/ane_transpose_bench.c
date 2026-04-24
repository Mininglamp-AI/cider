// Quick NEON transpose benchmark
// clang -O2 -o ane_transpose_bench ane_transpose_bench.c -framework Accelerate
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>

// Naive tile transpose (current)
static void tile_transpose_naive(const float * __restrict__ src, float * __restrict__ dst, int rows, int cols) {
    const int T = 16;
    for (int r = 0; r < rows; r += T) {
        int rend = r + T < rows ? r + T : rows;
        for (int c = 0; c < cols; c += T) {
            int cend = c + T < cols ? c + T : cols;
            for (int ri = r; ri < rend; ri++)
                for (int ci = c; ci < cend; ci++)
                    dst[ci * rows + ri] = src[ri * cols + ci];
        }
    }
}

// NEON 4x4 transpose
static void transpose_neon(const float * __restrict__ src, float * __restrict__ dst, int rows, int cols) {
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t r0 = vld1q_f32(&src[(r+0)*cols + c]);
            float32x4_t r1 = vld1q_f32(&src[(r+1)*cols + c]);
            float32x4_t r2 = vld1q_f32(&src[(r+2)*cols + c]);
            float32x4_t r3 = vld1q_f32(&src[(r+3)*cols + c]);

            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);

            float32x4_t o0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t o1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t o2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t o3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

            vst1q_f32(&dst[(c+0)*rows + r], o0);
            vst1q_f32(&dst[(c+1)*rows + r], o1);
            vst1q_f32(&dst[(c+2)*rows + r], o2);
            vst1q_f32(&dst[(c+3)*rows + r], o3);
        }
        // Remainder columns
        for (; c < cols; c++)
            for (int ri = r; ri < r+4 && ri < rows; ri++)
                dst[c * rows + ri] = src[ri * cols + c];
    }
    // Remainder rows
    for (; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// vDSP transpose
static void transpose_vdsp(const float * __restrict__ src, float * __restrict__ dst, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, cols, rows);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main() {
    int rows = 512, cols = 2048;  // [SEQ, IC] -> [IC, SEQ]
    size_t n = (size_t)rows * cols;
    float *src = malloc(n * sizeof(float));
    float *dst1 = malloc(n * sizeof(float));
    float *dst2 = malloc(n * sizeof(float));
    float *dst3 = malloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) src[i] = (float)i / n;

    int N = 500;

    // Warmup
    for (int i = 0; i < 20; i++) {
        tile_transpose_naive(src, dst1, rows, cols);
        transpose_neon(src, dst2, rows, cols);
        transpose_vdsp(src, dst3, rows, cols);
    }

    // Bench naive
    double t0 = now_ms();
    for (int i = 0; i < N; i++) tile_transpose_naive(src, dst1, rows, cols);
    double t1 = now_ms();

    // Bench NEON
    double t2 = now_ms();
    for (int i = 0; i < N; i++) transpose_neon(src, dst2, rows, cols);
    double t3 = now_ms();

    // Bench vDSP
    double t4 = now_ms();
    for (int i = 0; i < N; i++) transpose_vdsp(src, dst3, rows, cols);
    double t5 = now_ms();

    printf("[%d×%d] = %.1f MB\n", rows, cols, n*4/1e6);
    printf("Naive tile: %.3f ms\n", (t1-t0)/N);
    printf("NEON 4x4:   %.3f ms\n", (t3-t2)/N);
    printf("vDSP:       %.3f ms\n", (t5-t4)/N);

    // Verify correctness
    int ok1 = 1, ok2 = 1;
    for (size_t i = 0; i < n && ok1; i++) ok1 = (dst1[i] == dst2[i]);
    for (size_t i = 0; i < n && ok2; i++) ok2 = (dst1[i] == dst3[i]);
    printf("NEON correct: %d, vDSP correct: %d\n", ok1, ok2);

    // Also bench smaller sizes
    int sizes[][2] = {{512,1280},{512,640},{512,3968},{512,2048}};
    for (int s = 0; s < 4; s++) {
        int r = sizes[s][0], c = sizes[s][1];
        size_t nn = (size_t)r*c;
        float *ss = calloc(nn, sizeof(float));
        float *dd = calloc(nn, sizeof(float));
        for (int i = 0; i < 20; i++) transpose_vdsp(ss, dd, r, c);
        double ta = now_ms();
        for (int i = 0; i < N; i++) transpose_vdsp(ss, dd, r, c);
        double tb = now_ms();
        printf("vDSP [%d×%d] (%.1fMB): %.3f ms\n", r, c, nn*4/1e6, (tb-ta)/N);
        free(ss); free(dd);
    }

    free(src); free(dst1); free(dst2); free(dst3);
    return 0;
}
