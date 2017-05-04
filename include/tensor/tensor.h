//
// Created by sun on 5/2/17.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <initializer_list>
#include <cassert>
#include <iostream>

namespace tnn {

    template<typename U = float, typename Allocator = std::allocator<U> >
    class tensor_storage {
    public:
        typedef U data_type;
        typedef std::vector<U, Allocator> container;
        tensor_storage() {}
        tensor_storage(const tensor_storage &other)
                : m_data(other.m_data) {}
        tensor_storage(tensor_storage &&other)
                : m_data(std::move(other.m_data)) {}
        tensor_storage &operator=(const tensor_storage &other) {
            m_data = other.m_data;
            return *this;
        }
        tensor_storage &operator=(tensor_storage &&other) {
            m_data = std::move(other.m_data);
            return *this;
        }
        explicit tensor_storage(std::initializer_list<data_type> data)
                : m_data(data) {}
        data_type &at(std::size_t i) {
            return m_data[i];
        }
        const data_type &at(std::size_t i) const {
            return m_data[i];
        }
        std::size_t size() const {
            return m_data.size();
        }
        void resize(std::size_t i) {
            m_data.resize(i);
        }
        data_type *get_raw() {
            return &m_data[0];
        }
        const data_type *get_raw() const {
            return &m_data[0];
        }
    private:
        container m_data;
    };

    template<typename U = float, typename Allocator = std::allocator<U> >
    class tensor {
        template<typename V, typename A>
        friend std::ostream &operator<<(std::ostream &out, const tensor<V, A> &t);
    public:
        typedef U data_type;
        typedef tensor_storage<U, Allocator> storage;
        typedef typename std::shared_ptr<storage> storage_pointer;
        tensor() : m_data(std::make_shared<storage>()) {
            update_base();
        }
        tensor(const tensor &other)
                : m_data(std::make_shared<storage>(*other.m_data)), m_shape(other.m_shape), m_base(other.m_base) {}
        tensor(tensor &&other)
                : m_data(std::move(other.m_data)), m_shape(std::move(other.m_shape)), m_base(std::move(other.m_base)) {}
        tensor &operator=(const tensor &other) {
            m_data = std::make_shared<storage>(*other.m_data);
            m_shape = other.m_shape;
            m_base = other.m_base;
            return *this;
        }
        tensor &operator=(tensor &&other) {
            m_data = std::move(other.m_data);
            m_shape = std::move(other.m_shape);
            m_base = std::move(other.m_base);
            return *this;
        }
        explicit tensor(std::initializer_list<std::size_t> shape)
                : m_data(std::make_shared<storage>()), m_shape(shape) {
            update_base();
            m_data->resize(size());
        }
        tensor(std::initializer_list<std::size_t> shape, std::initializer_list<data_type> data)
                : m_data(std::make_shared<storage>(data)), m_shape(shape) {
            update_base();
            assert(size() == m_data->size());
        }
        void load(std::istream &in) {
            in.read(reinterpret_cast<char *>(get_raw()), sizeof(data_type) * size());
        }
        void save(std::ostream &out) const {
            out.write(reinterpret_cast<const char *>(get_raw()), sizeof(data_type) * size());
        }
        template<class InputIt>
        void reshape(InputIt first, InputIt last) {
            m_shape.assign(first, last);
            update_base();
            assert(size() == m_data->size());
        }
        void reshape(std::initializer_list<std::size_t> shape) {
            m_shape = shape;
            update_base();
            assert(size() == m_data->size());
        }
        template<class InputIt>
        void resize(InputIt first, InputIt last) {
            m_shape.assign(first, last);
            update_base();
            m_data->resize(size());
        }
        void resize(std::initializer_list<std::size_t> shape) {
            m_shape = shape;
            update_base();
            m_data->resize(size());
        }
        std::size_t size() const {
            return m_base.front();
        }
        template<typename ...Args>
        data_type &at(Args ...args) {
            return m_data->at(get_pos(std::forward<Args>(args)...));
        }
        template<typename ...Args>
        const data_type &at(Args ...args) const {
            return m_data->at(get_pos(std::forward<Args>(args)...));
        }
        const std::vector<std::size_t> &shape() const {
            return m_shape;
        }
        std::size_t shape(std::size_t i) const {
            return m_shape[i];
        }
        std::size_t ndim() const {
            return m_shape.size();
        }
        template<typename ...Args>
        data_type *get_raw(Args ...args) {
            return m_data->get_raw() + get_pos(std::forward<Args>(args)...);
        }
        template<typename ...Args>
        const data_type *get_raw(Args ...args) const {
            return m_data->get_raw() + get_pos(std::forward<Args>(args)...);
        }

    private:
        std::size_t get_pos() const {
            return 0;
        }
        template<typename ...Args>
        std::size_t get_pos(std::size_t i, Args ...args) const {
            return i * m_base[m_shape.size() - sizeof...(args)] + get_pos(std::forward<Args>(args)...);
        }
        void update_base() {
            m_base.resize(m_shape.size() + 1);
            m_base[m_shape.size()] = m_shape.empty() ? 0 : 1;
            for (std::size_t i = m_shape.size(); i != 0; --i)
                m_base[i - 1] = m_base[i] * m_shape[i - 1];
            assert(m_base.size() == 1 || m_base.front() != 0);
        }
        storage_pointer m_data;
        std::vector<std::size_t> m_shape, m_base;
    };

    template<typename U, typename Allocator>
    std::ostream &operator<<(std::ostream &out, const tensor<U, Allocator> &t) {
        if (t.m_data->size())
            out << t.m_data->at(0);
        for (std::size_t i = 1; i < t.m_data->size(); ++i)
            out << " " << t.m_data->at(i);
        out << "\n[";
        if (!t.m_shape.empty())
            out << t.m_shape.front();
        for (std::size_t i = 1; i < t.m_shape.size(); ++i)
            out << "x" << t.m_shape[i];
        out << "]";
        return out;
    };
}

#endif
