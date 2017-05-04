#ifndef RELU_H
#define RELU_H

#include <type_traits>

#include "layer.h"
#include "avx.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class relu: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            std::vector<std::future<void> > sync;
            sync.reserve(threads.get_thread_num());
            std::size_t start = 0;
            double step = (double) x.size() / threads.get_thread_num();
            for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                std::size_t end = (int) (step * (i + 1) + 0.5);
                if (start != end)
                    sync.emplace_back(threads.enqueue([this, &x](std::size_t s, std::size_t e) {
                        single_relu(x, s, e);
                    }, start, end));
                start = end;
            }
            for (std::size_t i = 0; i < sync.size(); ++i)
                sync[i].get();
            return x;
        }
    protected:
#if AVX_ENABLED
        template<bool ForceDisableAVX = false>
        typename std::enable_if<std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX>::type
            single_relu(tensor_type &x, std::size_t s, std::size_t e) const {
            __m256 zeros = _mm256_setzero_ps();
            for (; s + 7 < e; s += 8)
                _mm256_storeu_ps(x.get_raw(s), _mm256_max_ps(zeros, _mm256_loadu_ps(x.get_raw(s))));
            single_relu<true>(x, s, e);
        }
#endif
        template<bool ForceDisableAVX = false>
        typename std::enable_if<!(std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX)>::type
            single_relu(tensor_type &x, std::size_t s, std::size_t e) const {
            for (; s < e; ++s)
                if (x.at(s) < 0)
                    x.at(s) = 0;
        }
    };
}

#endif
