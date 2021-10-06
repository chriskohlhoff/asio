//
// experimental/compute/cuda/run_kernel.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_RUN_KERNEL_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_RUN_KERNEL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/experimental/compute/cuda/basic_command_queue.hpp"
#include "asio/experimental/compute/cuda/detail/kernel_command.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

ASIO_EXEC_CHECK_DISABLE
template <typename Executor, typename Error, typename Shape,
    typename Function, typename CompletionToken>
ASIO_HOST_DEVICE
auto run_kernel(basic_command_queue<Executor, Error>& cq,
    Shape grid, Shape block, Function f, CompletionToken&& token)
{
  return cq.async_submit(
      detail::kernel_command<Shape, Function>{grid, block, std::move(f)},
      std::forward<CompletionToken>(token));
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_RUN_KERNEL_HPP
