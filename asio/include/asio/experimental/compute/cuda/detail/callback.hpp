//
// experimental/compute/cuda/detail/callback.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_CALLBACK_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_CALLBACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/associated_executor.hpp"
#include "asio/detail/recycling_allocator.hpp"
#include "asio/experimental/compute/cuda/error.hpp"
#include <cuda.h>
#include <new>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {
namespace detail {

template <typename Handler, typename Executor>
struct callback
{
  using allocator_type = associated_allocator_t<Handler,
        asio::detail::recycling_allocator<void>>;

  using executor_type = std::decay_t<
      decltype(
        asio::prefer(
          asio::require(declval<Executor>(),
            asio::execution::blocking.never),
          asio::execution::outstanding_work.tracked,
          asio::execution::allocator(std::declval<allocator_type>())))>;

  callback(Handler handler, executor_type executor, allocator_type allocator)
    : handler_(std::move(handler)),
      executor_(std::move(executor)),
      allocator_(std::move(allocator))
  {
  }

  static callback* create(Handler handler, const Executor& candidate_executor)
  {
    allocator_type allocator = asio::get_associated_allocator(
        handler, asio::detail::recycling_allocator<void>());

    executor_type executor = asio::prefer(
        asio::require(candidate_executor,
          asio::execution::blocking.never),
        asio::execution::outstanding_work.tracked,
        asio::execution::allocator(allocator));

    using rebound_alloc_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<callback>;
    rebound_alloc_type rebound_allocator(allocator);

    callback* callback_mem = rebound_allocator.allocate(1);
    try
    {
      return new (callback_mem) callback(std::move(handler),
          std::move(executor), std::move(allocator));
    }
    catch (...)
    {
      rebound_allocator.deallocate(callback_mem, 1);
      throw;
    }
  }

  static void call(cudaStream_t, cudaError_t status, void* data)
  {
    call(make_error_code(status), static_cast<callback*>(data));
  }

  static void call(const std::error_code& e, callback* self)
  {
    executor_type executor(std::move(self->executor_));
    using rebound_alloc_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<callback>;
    rebound_alloc_type rebound_allocator(self->allocator_);

    try
    {
      Handler handler(std::move(self->handler_));
      self->~callback();
      rebound_allocator.deallocate(self, 1);
      self = nullptr;
      asio::execution::execute(std::move(executor),
          [handler=std::move(handler), e]() mutable
          {
            std::move(handler)(e);
          });
    }
    catch (...)
    {
      if (self)
      {
        self->~callback();
        rebound_allocator.deallocate(self, 1);
      }
    }
  }

  Handler handler_;
  executor_type executor_;
  allocator_type allocator_;
};

} // namespace detail
} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_DETAIL_CALLBACK_HPP
