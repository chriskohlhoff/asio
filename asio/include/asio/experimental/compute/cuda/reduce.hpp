//
// experimental/compute/cuda/reduce.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_FOR_EACH_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_FOR_EACH_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/experimental/compute/cuda/basic_command_queue.hpp"
#include "asio/experimental/compute/cuda/device_iterator.hpp"
#include "asio/experimental/compute/cuda/device_vector.hpp"
#include "asio/experimental/compute/cuda/run_kernel.hpp"
#include "asio/experimental/linked_group.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {
namespace detail {

template <typename InputPointer, typename OutputPointer,
    typename T, typename BinaryFunction>
struct reduce_kernel_1
{
  InputPointer input_ptr;
  OutputPointer output_ptr;
  std::size_t n;
  T init;
  BinaryFunction f;

  void __device__ operator()()
  {
    std::size_t batches = std::min<std::size_t>(n, 128);
    std::size_t batch_size = n / batches + ((n % batches) ? 1 : 0);
    std::size_t start = threadIdx.x * batch_size;
    std::size_t end = std::min(n, start + batch_size);
    T value = init;
    for (std::size_t i = start; i < end; ++i)
      value = f(input_ptr[i], value);
    output_ptr[threadIdx.x] = value;
  }
};

template <typename OutputPointer, typename T, typename BinaryFunction>
struct reduce_kernel_2
{
  OutputPointer output_ptr;
  std::size_t n;
  T init;
  BinaryFunction f;

  void __device__ operator()()
  {
    std::size_t batches = std::min<std::size_t>(n, 128);
    T value = init;
    for (std::size_t i = 0; i < batches; ++i)
      value = f(input_ptr[i], value);
    output_ptr[0] = value;
  }
};

} // namespace detail

ASIO_EXEC_CHECK_DISABLE
template <typename Executor, typename Error, typename InputIterator,
    typename OutputIterator, typename T, typename BinaryFunction,
    typename CompletionToken>
ASIO_HOST_DEVICE
auto reduce(basic_command_queue<Executor, Error>& cq,
    InputInputIterator first, InputIterator last, OutputIterator out,
    T init, BinaryFunction f, CompletionToken&& token)
{
  static_assert(is_device_iterator<InputIterator>::value);
  static_assert(is_device_iterator<OutputIterator>::value);
  return make_linked_group(
      [=, &cq](auto&& token)
      {
      }

  return asio::async_initiate<CompletionToken, void(Error)>(
      [](auto&& handler, auto first, auto last, auto out, auto f)
      {
        device_vector intermediate
        return run_kernel(cq, std::size_t(1),
            std::min<std::size_t>(last - first, 128),
            detail::reduce_kernel_1<decltype(first.data()), decltype(second.data()), T, BinaryFunction>{
              first.data(), std::size_t(last - first), std::move(f)},
      },
      token, first, last, out, std::move(f));
  return run_kernel(cq, std::size_t(1),
      std::min<std::size_t>(last - first, 128),
      detail::reduce_kernel<decltype(first.data()), BinaryFunction>{
        first.data(), std::size_t(last - first), std::move(f)},
      std::forward<CompletionToken>(token));
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_FOR_EACH_HPP
