#ifndef LAYER_H
#define LAYER_H


#include "threadpool.h"
#include "tensor/tensor.h"

namespace tnn {
    template <typename U = float, typename Allocator = std::allocator<U> >
    class layer {
    public:
        typedef tensor<U, Allocator> tensor_type;
        virtual tensor_type forward(tensor_type &&tensor, thread_pool &threads) const = 0;
        virtual void load(std::istream &in) {}
        virtual ~layer() {};
    };

    template <typename U = float, typename Allocator = std::allocator<U> >
    class layers: public layer<U, Allocator> {
    public:
        typedef layer<U, Allocator> layer_type;
        typedef typename layer_type::tensor_type tensor_type;
        layers(std::initializer_list<std::shared_ptr<layer_type> > layers)
                : m_layers(layers) {}
        tensor_type forward(tensor_type &&x, thread_pool &threads) const {
            for (std::size_t i = 0; i < m_layers.size(); ++i)
                x = m_layers[i]->forward(std::move(x), threads);
            return x;
        }
        void load(std::istream &in) {
            for (std::size_t i = 0; i < m_layers.size(); ++i)
                m_layers[i]->load(in);
        }

    private:
        std::vector<std::shared_ptr<layer_type> > m_layers;
    };
}

#endif