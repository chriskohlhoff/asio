//
// experimental/compute/cuda/bulk.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_BULK_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_BULK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/experimental/compute/cuda/basic_command_queue.hpp"
#include "asio/experimental/compute/cuda/run_kernel.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {
namespace detail {

template <typename Function>
struct bulk_kernel
{
  Function f;

  void __device__ operator()()
  {
    f(threadIdx.x);
  }
};

} // namespace detail

ASIO_EXEC_CHECK_DISABLE
template <typename Executor, typename Error,
    typename Function, typename CompletionToken>
ASIO_HOST_DEVICE
auto bulk(basic_command_queue<Executor, Error>& cq,
    std::size_t n, Function f, CompletionToken&& token)
  -> decltype(run_kernel(cq, std::size_t(1), n,
      detail::bulk_kernel<Function>{std::move(f)},
      std::forward<CompletionToken>(token)))
{
  return run_kernel(cq, std::size_t(1), n,
      detail::bulk_kernel<Function>{std::move(f)},
      std::forward<CompletionToken>(token));
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_BULK_HPP
