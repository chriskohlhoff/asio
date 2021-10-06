//
// experimental/compute/cuda/memcpy.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_MEMCPY_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_MEMCPY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/experimental/compute/cuda/basic_command_queue.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

template <typename Executor, typename Error, typename CompletionToken>
auto memcpy_host_to_host(basic_command_queue<Executor, Error>& cq,
    void* dest, const void* src, std::size_t n, CompletionToken&& token)
{
  return cq.async_submit(
      [=](basic_command_queue<Executor, Error>& cq) -> Error
      {
        return cudaMemcpyAsync(dest, src, n,
            cudaMemcpyHostToHost, cq.native_handle());
      }, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename CompletionToken>
auto memcpy_host_to_device(basic_command_queue<Executor, Error>& cq,
    void* dest, const void* src, std::size_t n, CompletionToken&& token)
{
  return cq.async_submit(
      [=](basic_command_queue<Executor, Error>& cq) -> Error
      {
        return cudaMemcpyAsync(dest, src, n,
            cudaMemcpyHostToDevice, cq.native_handle());
      }, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename CompletionToken>
auto memcpy_device_to_host(basic_command_queue<Executor, Error>& cq,
    void* dest, const void* src, std::size_t n, CompletionToken&& token)
{
  return cq.async_submit(
      [=](basic_command_queue<Executor, Error>& cq) -> Error
      {
        return cudaMemcpyAsync(dest, src, n,
            cudaMemcpyDeviceToHost, cq.native_handle());
      }, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename CompletionToken>
auto memcpy_device_to_device(basic_command_queue<Executor, Error>& cq,
    void* dest, const void* src, std::size_t n, CompletionToken&& token)
{
  return cq.async_submit(
      [=](basic_command_queue<Executor, Error>& cq) -> Error
      {
        return cudaMemcpyAsync(dest, src, n,
            cudaMemcpyDeviceToDevice, cq.native_handle());
      }, std::forward<CompletionToken>(token));
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_MEMCPY_HPP
