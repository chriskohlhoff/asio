//
// detail/apple_nw_sync_result.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_SYNC_RESULT_HPP
#define ASIO_DETAIL_APPLE_NW_SYNC_RESULT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/error.hpp"
#include <condition_variable>
#include <mutex>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class apple_nw_sync_result
{
public:
  apple_nw_sync_result()
    : ec_(asio::error::would_block),
      result_()
  {
  }

  void set(const asio::error_code& ec, T result)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ec_ = ec;
    result_ = ASIO_MOVE_CAST(T)(result);
    condition_.notify_all();
  }

  void set(nw_error_t error, T result)
  {
    set(asio::error_code(error ? nw_error_get_error_code(error) : 0,
          asio::system_category()), ASIO_MOVE_CAST(T)(result));
  }

  T get(asio::error_code& ec)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (ec_ == asio::error::would_block)
      condition_.wait(lock);
    ec = ec_;
    return ASIO_MOVE_CAST(T)(result_);
  }

private:
  apple_nw_sync_result(
      const apple_nw_sync_result& other) ASIO_DELETED;
  apple_nw_sync_result& operator=(
      const apple_nw_sync_result& other) ASIO_DELETED;

  std::mutex mutex_;
  std::condition_variable condition_;
  asio::error_code ec_;
  T result_;
};

template <>
class apple_nw_sync_result<void>
{
public:
  apple_nw_sync_result()
    : ec_(asio::error::would_block)
  {
  }

  void set(const asio::error_code& ec)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ec_ = ec;
    condition_.notify_all();
  }

  void set(nw_error_t error)
  {
    set(asio::error_code(
          error ? nw_error_get_error_code(error) : 0,
          asio::system_category()));
  }

  asio::error_code get(asio::error_code& ec)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (ec_ == asio::error::would_block)
      condition_.wait(lock);
    ec = ec_;
    return ec;
  }

private:
  apple_nw_sync_result(
      const apple_nw_sync_result& other) ASIO_DELETED;
  apple_nw_sync_result& operator=(
      const apple_nw_sync_result& other) ASIO_DELETED;

  std::mutex mutex_;
  std::condition_variable condition_;
  asio::error_code ec_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_SYNC_RESULT_HPP
