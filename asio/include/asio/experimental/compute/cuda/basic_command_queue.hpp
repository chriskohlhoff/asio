//
// experimental/compute/cuda/basic_command_queue.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_BASIC_COMMAND_QUEUE_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_BASIC_COMMAND_QUEUE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/error_code.hpp"
#include "asio/experimental/compute/cuda/detail/callback.hpp"
#include "asio/experimental/compute/cuda/detail/kernel_command.hpp"
#include "asio/experimental/compute/cuda/error.hpp"
#include "asio/experimental/linked_continuation.hpp"
#include <cuda.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {

template <typename Executor = asio::any_io_executor,
    typename Error = asio::error_code>
class basic_command_queue
{
public:
  using executor_type = Executor;
  using error_type = Error;
  using native_handle_type = cudaStream_t;

  explicit basic_command_queue(const executor_type& executor)
    : executor_(executor)
  {
    if (asio::error_code e = cudaStreamCreateWithFlags(
          &stream_, cudaStreamNonBlocking))
      throw std::system_error(e, "cudaStreamCreateWithFlags");
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  basic_command_queue(const executor_type& executor, error_type& e)
    : executor_(executor)
  {
    e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  basic_command_queue(const executor_type& executor, native_handle_type stream)
    : executor_(executor),
      stream_(stream)
  {
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  basic_command_queue(basic_command_queue&& other) noexcept
    : executor_(other.executor_),
      stream_(std::exchange(other.stream_, nullptr))
  {
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  ~basic_command_queue()
  {
    if (stream_)
      cudaStreamDestroy(stream_);
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  basic_command_queue& operator=(basic_command_queue&& other) noexcept
  {
    executor_ = other.executor_;
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  executor_type get_executor() noexcept
  {
    return executor_;
  }

  ASIO_HOST_DEVICE
  native_handle_type native_handle() noexcept
  {
    return stream_;
  }

  ASIO_EXEC_CHECK_DISABLE
  ASIO_HOST_DEVICE
  native_handle_type release() noexcept
  {
    return std::exchange(stream_, nullptr);
  }

  template <typename Command, typename CompletionToken>
  ASIO_HOST_DEVICE
  auto async_submit(Command command, CompletionToken&& token)
  {
    return asio::async_initiate<CompletionToken, void(error_type)>(
        initiate_async_compute{}, token, this, std::move(command));
  }

private:
  ASIO_EXEC_CHECK_DISABLE
  template <typename Command>
  ASIO_HOST_DEVICE
  static error_type run_commands(basic_command_queue& cq, Command command)
  {
    return command(cq);
  }

  ASIO_EXEC_CHECK_DISABLE
  template <typename Head, typename... Tail>
  ASIO_HOST_DEVICE
  static error_type run_commands(
      basic_command_queue& cq, Head head, Tail... tail)
  {
    if (error_type e = head(cq))
      return e;
    return run_commands(cq, std::move(tail)...);
  }

  ASIO_EXEC_CHECK_DISABLE
  template <typename Shape, typename Function1,
      typename Function2, typename... Tail>
  ASIO_HOST_DEVICE
  static error_type run_commands(basic_command_queue& cq,
      detail::kernel_command<Shape, Function1> kernel1,
      detail::kernel_command<Shape, Function2> kernel2, Tail... tail)
  {
    if (kernel1.grid == kernel2.grid && kernel1.block == kernel2.block)
    {
      return run_commands(cq,
          detail::kernel_command<Shape,
            detail::fused_kernel<Function1, Function2>>{
              kernel1.grid, kernel1.block,
              { std::move(kernel1.f), std::move(kernel2.f) } },
          std::move(tail)...);
    }
    else
    {
      if (error_type e = kernel1(cq))
        return e;
      return run_commands(cq, std::move(kernel2), std::move(tail)...);
    }
  }

  ASIO_EXEC_CHECK_DISABLE
  template <typename CompletionHandler, typename... Commands>
  ASIO_HOST_DEVICE
  static void launch(CompletionHandler completion_handler,
      basic_command_queue& cq, Commands... commands)
  {
#if defined(__CUDA_ARCH__)
    error_type e = run_commands(cq, std::move(commands)...);
    completion_handler(e);
#else // defined(__CUDA_ARCH__)
    using callback_type = detail::callback<
        decltype(completion_handler), executor_type>;
    callback_type* callback = callback_type::create(
        std::move(completion_handler), cq.executor_);
    if (error_type e = run_commands(cq, std::move(commands)...))
      return callback_type::call(e, callback);
    if (error_type e = cudaStreamAddCallback(
          cq.stream_, &callback_type::call, callback, 0))
      return callback_type::call(e, callback);
#endif // defined(__CUDA_ARCH__)
  }

  struct initiate_async_compute
  {
    ASIO_EXEC_CHECK_DISABLE
    template <typename CompletionHandler, typename... Commands>
    ASIO_HOST_DEVICE
    void operator()(CompletionHandler completion_handler,
        basic_command_queue* cq, Commands... commands) const
    {
      launch(std::move(completion_handler), *cq, std::move(commands)...);
    }

    ASIO_EXEC_CHECK_DISABLE
    template <typename CompletionHandler,
        typename Command, typename... Commands>
    ASIO_HOST_DEVICE
    void operator()(
        linked_continuation<
          CompletionHandler, initiate_async_compute,
          basic_command_queue*, Command> link,
        basic_command_queue* cq, Commands... commands) const
    {
      if (std::get<0>(link.init_args) == cq)
        (*this)(std::move(link.completion_handler), cq,
            commands..., std::move(std::get<1>(link.init_args)));
      else
        launch(std::move(link), *cq, std::move(commands)...);
    }
  };

  executor_type executor_;
  native_handle_type stream_;
};

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_COMMAND_QUEUE_HPP
