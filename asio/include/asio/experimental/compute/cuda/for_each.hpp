//
// experimental/compute/cuda/for_each.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/experimental/compute/cuda/run_kernel.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {
namespace detail {

template <typename Pointer, typename Function>
struct for_each_kernel
{
  Pointer ptr;
  std::size_t n;
  Function f;

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  void operator()()
  {
    std::size_t batches = std::min<std::size_t>(n, 128);
    std::size_t batch_size = n / batches + ((n % batches) ? 1 : 0);
    std::size_t start = threadIdx.x * batch_size;
    std::size_t end = std::min(n, start + batch_size);
    for (std::size_t i = start; i < end; ++i)
      f(ptr[i]);
  }
};

} // namespace detail

ASIO_EXEC_CHECK_DISABLE
template <typename Executor, typename Error,
    typename Iterator, typename Function, typename CompletionToken>
ASIO_HOST_DEVICE
auto for_each(basic_command_queue<Executor, Error>& cq,
    Iterator first, Iterator last, Function f, CompletionToken&& token)
  -> decltype(run_kernel(cq, declval<std::size_t>(), declval<std::size_t>(),
      detail::for_each_kernel<decltype(first.data()), Function>{
        first.data(), declval<std::size_t>(), std::move(f)},
      std::forward<CompletionToken>(token)))
{
  static_assert(is_device_iterator<Iterator>::value);
  return run_kernel(cq, std::size_t(1),
      std::min<std::size_t>(last - first, 128),
      detail::for_each_kernel<decltype(first.data()), Function>{
        first.data(), std::size_t(last - first), std::move(f)},
      std::forward<CompletionToken>(token));
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_FOR_EACH_HPP
