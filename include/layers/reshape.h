#ifndef RESHAPE_H
#define RESHAPE_H

#include "layer.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class reshape: public layer<U, Allocator> {
    public:
        typedef typename layer<U, Allocator>::tensor_type tensor_type;
        reshape(std::initializer_list<std::size_t> shape): m_shape(shape), m_size(1) {
            for (std::size_t i = 0; i < m_shape.size(); ++i)
                m_size *= m_shape[i];
        }
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            std::vector<std::size_t> shape;
            shape.reserve(m_shape.size() + 1);
            shape.emplace_back(x.size() / m_size);
            shape.insert(shape.end(), m_shape.begin(), m_shape.end());
            x.reshape(shape.begin(), shape.end());
            return x;
        }

    private:
        std::vector<std::size_t> m_shape;
        std::size_t m_size;
    };
}

#endif