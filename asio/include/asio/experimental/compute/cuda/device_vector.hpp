//
// experimental/compute/cuda/device_vector.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_VECTOR_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_VECTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/experimental/compute/cuda/device_iterator.hpp"
#include "asio/experimental/compute/cuda/error.hpp"
#include <cuda.h>
#include <iterator>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

template <typename T>
class device_vector
{
public:
  explicit device_vector(std::size_t n)
  {
    void* mem = nullptr;
    std::error_code ec = cudaMalloc(&mem, n * sizeof(T));
    if (ec)
      throw std::system_error(ec, "cudaMalloc");
    data_ = static_cast<T*>(mem);
    size_ = n;
  }

  device_vector(const device_vector& other) = delete;

  __host__ __device__
  device_vector(device_vector&& other) noexcept
  {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
  }

  device_vector& operator=(const device_vector& other) = delete;

  __host__ __device__
  device_vector& operator=(device_vector&& other) noexcept
  {
    if (this != &other)
    {
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
    }
    return *this;
  }

  __host__ __device__
  ~device_vector()
  {
    if (data_)
      cudaFree(data_);
  }

  __host__ __device__
  std::size_t size() const noexcept
  {
    return size_;
  }

  __device__
  T& operator[](std::size_t i) noexcept
  {
    return *data_ + i;
  }

  __device__
  const T& operator[](std::size_t i) const noexcept
  {
    return *data_ + i;
  }

  __host__ __device__
  T* data() noexcept
  {
    return data_;
  }

  __host__ __device__
  const T* data() const noexcept
  {
    return data_;
  }

  __host__ __device__
  device_iterator<T> begin() noexcept
  {
    return device_iterator{data_};
  }

  __host__ __device__
  device_iterator<const T> begin() const noexcept
  {
    return device_iterator{data_};
  }

  __host__ __device__
  device_iterator<T> end()
  {
    return device_iterator{data_ + size_};
  }

  __host__ __device__
  device_iterator<const T> end() const noexcept
  {
    return device_iterator{data_ + size_};
  }

private:
  T* data_;
  std::size_t size_;
};

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_DEVICE_VECTOR_HPP
