#ifndef LINEAR_H
#define LINEAR_H

#include <type_traits>

#include "layer.h"
#include "avx.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class linear: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        linear(std::size_t in_features, std::size_t out_features, bool bias = true)
                : m_in_features(in_features), m_out_features(out_features), m_has_bias(bias),
                  m_weight({out_features, in_features}) {
            if (bias)
                m_bias.resize({out_features});
        }

        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            assert(x.ndim() == 2 && x.shape(1) == m_in_features);
            tensor_type y{x.shape(0), m_out_features};
            std::size_t start = 0;
            std::vector<std::future<void> > sync;
            sync.reserve(threads.get_thread_num());
            double step = (double) y.size() / threads.get_thread_num();
            for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                std::size_t end = (int) (step * (i + 1) + 0.5);
                if (start != end)
                    sync.emplace_back(threads.enqueue([this, &x, &y](std::size_t s, std::size_t e) {
                        for (; s < e; ++s) {
                            std::size_t i = s / m_out_features, j = s % m_out_features;
                            single_linear(x, y, i, j);
                        }
                    }, start, end));
                start = end;
            }
            for (std::size_t i = 0; i < sync.size(); ++i)
                sync[i].get();
            return y;
        }
        void load(std::istream &in) {
            m_weight.load(in);
            if (m_has_bias)
                m_bias.load(in);
        }
    protected:
#if AVX_ENABLED
        template<bool ForceDisableAVX = false>
        typename std::enable_if<std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX>::type
            single_linear(const tensor_type &x, tensor_type &y, std::size_t i, std::size_t j) const {
            __m256 acc = _mm256_setzero_ps();
            std::size_t k;
            for (k = 0; k + 7 < x.shape(1); k += 8) {
                __m256 a = _mm256_loadu_ps(x.get_raw(i, k));
                __m256 b = _mm256_loadu_ps(m_weight.get_raw(j, k));
                __m256 prod = _mm256_mul_ps(a, b);
                acc = _mm256_add_ps(acc, prod);
            }
            float sum = mm256_sum(acc);
            for (; k < x.shape(1); ++k)
                sum += x.at(i, k) * m_weight.at(j, k);
            if (m_has_bias)
                sum += m_bias.at(j);
            y.at(i, j) = sum;
        }
#endif
        template<bool ForceDisableAVX = false>
        typename std::enable_if<!(std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX)>::type
            single_linear(const tensor_type &x, tensor_type &y, std::size_t i, std::size_t j) const {
            typename tensor_type::data_type sum = 0;
            for (std::size_t k = 0; k < x.shape(1); ++k)
                sum += x.at(i, k) * m_weight.at(j, k);
            if (m_has_bias)
                sum += m_bias.at(j);
            y.at(i, j) = sum;
        }
    private:
        std::size_t m_in_features, m_out_features;
        bool m_has_bias;
        tensor_type m_weight, m_bias;
    };
}

#endif
