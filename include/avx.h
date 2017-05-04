#ifndef AVX_COMMON_H
#define AVX_COMMON_H

#ifdef __AVX2__
#define AVX_ENABLED true
#include <immintrin.h>

namespace tnn {

    inline float mm256_sum(__m256 x) {
        __m128 sumQuad = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
        __m128 sumDual = _mm_add_ps(sumQuad, _mm_movehl_ps(sumQuad, sumQuad));
        __m128 sum = _mm_add_ss(sumDual, _mm_shuffle_ps(sumDual, sumDual, 0x1));
        return _mm_cvtss_f32(sum);
    }

}
#else
#define AVX_ENABLED false
#endif

#endif
