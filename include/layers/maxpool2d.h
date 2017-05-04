#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include <limits>
#include "layer.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class maxpool2d: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        maxpool2d(std::size_t kernel_size, std::size_t stride = 0, std::size_t padding = 0)
                : m_kernel_size(kernel_size), m_stride(stride), m_padding(padding) {
            if (stride == 0)
                stride = kernel_size;
        }
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            assert(x.ndim() == 4);
            std::size_t n = x.shape(0), channels = x.shape(1), start = 0;
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
            tensor_type y{n, channels, (x.shape(2) - m_kernel_size) / m_stride + 1,
                          (x.shape(3) - m_kernel_size) / m_stride + 1};
            sync.reserve(threads.get_thread_num());
            step = (double) n * channels / threads.get_thread_num();
            for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                std::size_t end = (int) (step * (i + 1) + 0.5);
                sync.emplace_back(threads.enqueue([this, &x, &y, channels](std::size_t s, std::size_t e) {
                    for (std::size_t j = s; j < e; ++j)
                        single_maxpool2d(x, y, j / channels, j % channels);
                }, start, end));
                start = end;
            }
            for (std::size_t i = 0; i < sync.size(); ++i)
                sync[i].get();
            return y;
        }
    private:
        void single_maxpool2d(const tensor_type &x, tensor_type &y, std::size_t i, std::size_t c) const {
            std::size_t height = y.shape(2), width = y.shape(3);
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    typename tensor_type::data_type max = -std::numeric_limits<typename tensor_type::data_type>::max(), value;
                    std::size_t hs = m_stride * h, ws = m_stride * w;
                    for (std::size_t kh = 0; kh < m_kernel_size; ++kh)
                        for (std::size_t kw = 0; kw < m_kernel_size; ++kw) {
                            value = x.at(i, c, hs + kh, ws + kw);
                            if (value > max)
                                max = value;
                        }
                    y.at(i, c, h, w) = max;
                }
            }
        }
        std::size_t m_kernel_size, m_stride, m_padding;
    };

}

#endif