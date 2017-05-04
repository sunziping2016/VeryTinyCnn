#ifndef BIAS_H
#define BIAS_H

#include <type_traits>

#include "layer.h"
#include "avx.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class bias: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        bias(std::size_t features) : m_features(features), m_bias{features} {}
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            assert(x.ndim() == 2 && x.shape(1) == m_features);
            std::size_t start = 0;
            std::vector<std::future<void> > sync;
            sync.reserve(threads.get_thread_num());
            double step = (double) x.size() / threads.get_thread_num();
            for (std::size_t i = 0; i < threads.get_thread_num(); ++i) {
                std::size_t end = (int) (step * (i + 1) + 0.5);
                if (start != end)
                    sync.emplace_back(threads.enqueue([this, &x](std::size_t s, std::size_t e) {
                        for (; s < e; ++s)
                            x.at(s) += m_bias.at(s % m_features);
                    }, start, end));
                start = end;
            }
            for (std::size_t i = 0; i < sync.size(); ++i)
                sync[i].get();
            return x;
        }
        void load(std::istream &in) {
            m_bias.load(in);
        }
    private:
        std::size_t m_features;
        tensor_type m_bias;
    };
}

#endif
