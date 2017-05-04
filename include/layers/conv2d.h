#ifndef CONV2D_H
#define CONV2D_H

#include <cstring>
#include <type_traits>

#include "layer.h"
#include "avx.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class conv2d: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        conv2d(std::size_t in_channels, std::size_t out_channels, std::size_t kernel_size, std::size_t stride = 1, std::size_t padding = 0, bool bias = true)
                : m_in_channels(in_channels), m_out_channels(out_channels), m_kernel_size(kernel_size), m_stride(stride),
                  m_padding(padding), m_has_bias(bias), m_weight({out_channels, in_channels, kernel_size, kernel_size}) {
            if (bias)
                m_bias.resize({out_channels});

        }
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            assert(x.ndim() == 4 && x.shape(1) == m_in_channels);
            std::size_t n = x.shape(0), start = 0;
            std::vector<std::future<void> > sync;
            double step;
            if (m_padding) {
                sync.reserve(threads.get_thread_num());
                tensor_type temp{n, x.shape(1), x.shape(2) + 2 * m_padding, x.shape(3) + 2 * m_padding};
                step = (double) n * x.shape(1) * x.shape(2) / threads.get_thread_num();
                for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                    std::size_t end = (int) (step * (i + 1) + 0.5);
                    if (start != end)
                        sync.emplace_back(threads.enqueue([this, &temp, &x](std::size_t s, std::size_t e) {
                            for (std::size_t j = s; j < e; ++j) {
                                std::size_t c = j / x.shape(2), h = j % x.shape(2);
                                memcpy(temp.get_raw(c, h + m_padding, m_padding), x.get_raw(c, h, 0), x.shape(3) * sizeof(float));
                            }
                        }, start, end));
                    start = end;
                }
                for (std::size_t i = 0; i < sync.size(); ++i)
                    sync[i].get();
                sync.clear();
                start = 0;
                x = std::move(temp);
            }
            tensor_type y{n, m_out_channels, (x.shape(2) - m_kernel_size) / m_stride + 1,
                          (x.shape(3) - m_kernel_size) / m_stride + 1};
            sync.reserve(threads.get_thread_num());
            step = (double) n * m_out_channels / threads.get_thread_num();
            for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                std::size_t end = (int) (step * (i + 1) + 0.5);
                sync.emplace_back(threads.enqueue([this, &x, &y](std::size_t s, std::size_t e) {
                    for (std::size_t j = s; j < e; ++j)
                        single_conv(x, y, j / m_out_channels, j % m_out_channels);
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
    private:
#if AVX_ENABLED
        template<bool ForceDisableAVX = false>
        typename std::enable_if<std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX>::type
            single_conv(const tensor_type &x, tensor_type &y, std::size_t i, std::size_t out) const {
            std::size_t height = y.shape(2), width = y.shape(3);
            for (std::size_t h = 0; h < height; ++h) {
                std::size_t hs = m_stride * h, w;
                for (w = 0; w + 7 < width; w += 8) {
                    __m256 sum = _mm256_setzero_ps();
                    std::size_t ws = m_stride * w;
                    for (std::size_t in = 0; in < m_in_channels; ++in)
                        for (std::size_t kh = 0; kh < m_kernel_size; ++kh)
                            for (std::size_t kw = 0; kw < m_kernel_size; ++kw) {
                                sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(m_weight.at(out, in, kh, kw)), _mm256_set_ps(
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 7),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 6),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 5),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 4),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 3),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 2),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 1),
                                        x.at(i, in, hs + kh, ws + kw + m_stride * 0)
                                )));
                            }
                    if (m_has_bias)
                        sum = _mm256_add_ps(sum, _mm256_set1_ps(m_bias.at(out)));
                    _mm256_storeu_ps(y.get_raw(i, out, h, w), sum);
                }
                for (; w < width; ++w) {
                    float sum = 0;
                    std::size_t ws = m_stride * w;
                    for (std::size_t in = 0; in < m_in_channels; ++in)
                        for (std::size_t kh = 0; kh < m_kernel_size; ++kh)
                            for (std::size_t kw = 0; kw < m_kernel_size; ++kw)
                                sum += x.at(i, in, hs + kh, ws + kw) * m_weight.at(out, in, kh, kw);
                    if (m_has_bias)
                        sum += m_bias.at(out);
                    y.at(i, out, h, w) = sum;
                }
            }
        }
#endif
        template<bool ForceDisableAVX = false>
        typename std::enable_if<!(std::is_same<U, float>::value && AVX_ENABLED && !ForceDisableAVX)>::type
            single_conv(const tensor_type &x, tensor_type &y, std::size_t i, std::size_t out) const {
            std::size_t height = y.shape(2), width = y.shape(3);
            for (std::size_t h = 0; h < height; ++h) {
                std::size_t hs = m_stride * h;
                for (std::size_t w = 0; w < width; ++w) {
                    typename tensor_type::data_type sum = 0;
                    std::size_t ws = m_stride * w;
                    for (std::size_t in = 0; in < m_in_channels; ++in)
                        for (std::size_t kh = 0; kh < m_kernel_size; ++kh)
                            for (std::size_t kw = 0; kw < m_kernel_size; ++kw)
                                sum += x.at(i, in, hs + kh, ws + kw) * m_weight.at(out, in, kh, kw);
                    y.at(i, out, h, w) = m_has_bias ? m_bias.at(out) + sum : sum;
                }
            }
        }


        std::size_t m_in_channels, m_out_channels, m_kernel_size, m_stride, m_padding;
        bool m_has_bias;
        tensor_type m_weight, m_bias;
    };
}

#endif
