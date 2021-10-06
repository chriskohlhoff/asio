//
// experimental/compute/cuda/device_iterator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_ITERATOR_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_ITERATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/experimental/compute/cuda/error.hpp"
#include <cuda.h>
#include <iterator>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

template <typename T>
class device_iterator
{
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  __host__ __device__
  device_iterator() noexcept
    : ptr_(nullptr)
  {
  }

  __host__ __device__
  explicit device_iterator(T* ptr)
    : ptr_(ptr)
  {
  }

  __host__ __device__
  device_iterator(const device_iterator& other) noexcept
    : ptr_(other.ptr_)
  {
  }

  __host__ __device__
  device_iterator& operator=(const device_iterator& other) noexcept
  {
    ptr_ = other.ptr_;
    return *this;
  }

  __device__
  reference operator*() const noexcept
  {
    return *ptr_;
  }

  __device__
  pointer operator->() const noexcept
  {
    return ptr_;
  }

  pointer data() const noexcept
  {
    return ptr_;
  }

  __host__ __device__
  device_iterator& operator++()
  {
    ptr_++;
    return *this;
  }

  __host__ __device__
  device_iterator operator++(int)
  {
    device_iterator tmp = *this;
    ptr_++;
    return tmp;
  }

  __host__ __device__
  device_iterator& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }

  __host__ __device__
  device_iterator& operator--()
  {
    ptr_--;
    return *this;
  }

  __host__ __device__
  device_iterator operator--(int)
  {
    device_iterator tmp = *this;
    ptr_--;
    return tmp;
  }

  __host__ __device__
  device_iterator& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  __host__ __device__
  friend bool operator==(const device_iterator& a,
      const device_iterator& b) noexcept
  {
    return a.ptr_ == b.ptr_;
  }

  __host__ __device__
  friend bool operator!=(const device_iterator& a,
      const device_iterator& b) noexcept
  {
    return a.ptr_ != b.ptr_;
  }

  __host__ __device__
  friend device_iterator operator+(const device_iterator& a,
      difference_type b) noexcept
  {
    device_iterator tmp = a;
    tmp += b;
    return tmp;
  }

  __host__ __device__
  friend device_iterator operator+(difference_type a,
      const device_iterator& b) noexcept
  {
    device_iterator tmp = b;
    tmp += a;
    return tmp;
  }

  __host__ __device__
  friend difference_type operator-(const device_iterator& a,
      const device_iterator& b) noexcept
  {
    return a.ptr_ - b.ptr_;
  }

private:
  T* ptr_;
};

template <typename T>
struct is_device_iterator : false_type {};

template <typename T>
struct is_device_iterator<device_iterator<T>> : true_type {};

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_ITERATOR_HPP
