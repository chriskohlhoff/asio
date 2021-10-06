//
// experimental/compute/cuda/copy.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_COMPUTE_CUDA_COPY_HPP
#define ASIO_EXPERIMENTAL_COMPUTE_CUDA_COPY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/experimental/compute/cuda/basic_command_queue.hpp"
#include "asio/experimental/compute/cuda/device_iterator.hpp"
#include "asio/experimental/compute/cuda/memcpy.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace compute {
namespace cuda {
namespace detail {

template <typename Executor, typename Error, typename InIterator,
    typename OutIterator, typename CompletionToken>
auto copy_impl(basic_command_queue<Executor, Error>& ctx,
    InIterator first, InIterator last,
    OutIterator out, CompletionToken&& token,
    std::false_type is_device_input, std::false_type is_device_output)
{
  using value_type = typename InIterator::value_type;
  static_assert(is_same<value_type, typename OutIterator::value_type>::value);
  std::size_t size = (last - first) * sizeof(value_type);
  return memcpy_host_to_host(ctx, &*out,
      &*first, size, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename InIterator,
    typename OutIterator, typename CompletionToken>
auto copy_impl(basic_command_queue<Executor, Error>& ctx,
    InIterator first, InIterator last,
    OutIterator out, CompletionToken&& token,
    std::false_type is_device_input, std::true_type is_device_output)
{
  using value_type = typename InIterator::value_type;
  static_assert(is_same<value_type, typename OutIterator::value_type>::value);
  std::size_t size = (last - first) * sizeof(value_type);
  return memcpy_host_to_device(ctx, out.data(),
      &*first, size, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename InIterator,
    typename OutIterator, typename CompletionToken>
auto copy_impl(basic_command_queue<Executor, Error>& ctx,
    InIterator first, InIterator last,
    OutIterator out, CompletionToken&& token,
    std::true_type is_device_input, std::false_type is_device_output)
{
  using value_type = typename InIterator::value_type;
  static_assert(is_same<value_type, typename OutIterator::value_type>::value);
  std::size_t size = (last - first) * sizeof(value_type);
  return memcpy_device_to_host(ctx, &*out,
      first.data(), size, std::forward<CompletionToken>(token));
}

template <typename Executor, typename Error, typename InIterator,
    typename OutIterator, typename CompletionToken>
auto copy_impl(basic_command_queue<Executor, Error>& ctx,
    InIterator first, InIterator last,
    OutIterator out, CompletionToken&& token,
    std::true_type is_device_input, std::true_type is_device_output)
{
  using value_type = typename InIterator::value_type;
  static_assert(is_same<value_type, typename OutIterator::value_type>::value);
  std::size_t size = (last - first) * sizeof(value_type);
  return memcpy_device_to_device(ctx, out.data(),
      first.data(), size, std::forward<CompletionToken>(token));
}

} // namespace detail

template <typename Executor, typename Error, typename InIterator,
    typename OutIterator, typename CompletionToken>
auto copy(basic_command_queue<Executor, Error>& ctx,
    InIterator first, InIterator last,
    OutIterator out, CompletionToken&& token)
{
  return detail::copy_impl(ctx, first, last, out,
      std::forward<CompletionToken>(token),
      is_device_iterator<InIterator>{},
      is_device_iterator<OutIterator>{});
}

} // namespace cuda
} // namespace compute
} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_COMPUTE_CUDA_COPY_HPP
