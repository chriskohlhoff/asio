//
// experimental/compute/cuda/detail/kernel_command.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_KERNEL_COMMAND_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_KERNEL_COMMAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

template <typename, typename>
class basic_command_queue;

namespace detail {

template <typename Function>
void __global__ kernel_proxy(Function f)
{
  std::move(f)();
}

template <typename Function1, typename Function2>
struct fused_kernel
{
  Function1 f1;
  Function2 f2;

  void __device__ operator()()
  {
    std::move(f1)();
    __syncthreads();
    std::move(f2)();
  }
};

template <typename Shape, typename Function>
struct kernel_command
{
  Shape grid;
  Shape block;
  Function f;

  ASIO_EXEC_CHECK_DISABLE
  template <typename Executor, typename Error>
  ASIO_HOST_DEVICE
  cudaError_t operator()(basic_command_queue<Executor, Error>& cq)
  {
    kernel_proxy<<<grid, block, 0, cq.native_handle()>>>(std::move(f));
    return cudaPeekAtLastError();
  }
};

} // namespace detail
} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_KERNEL_COMMAND_HPP
