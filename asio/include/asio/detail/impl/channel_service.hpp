//
// detail/impl/channel_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_CHANNEL_SERVICE_HPP
#define ASIO_DETAIL_IMPL_CHANNEL_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T, typename T0>
void channel_service::put(implementation_type<T>& impl,
    ASIO_MOVE_ARG(T0) value, asio::error_code& ec)
{
  if (!impl.open_)
  {
    ec = asio::error::broken_pipe;
  }
  else if (channel_op<T>* getter =
      static_cast<channel_op<T>*>(impl.getters_.front()))
  {
    getter->set_value(ASIO_MOVE_CAST(T0)(value));
    impl.getters_.pop();
    io_service_.post_deferred_completion(getter);
    ec = asio::error_code();
  }
  else if (impl.buffer_.size() < impl.max_buffer_size_)
  {
    impl.buffer_.resize(impl.buffer_.size() + 1);
    impl.buffer_.back() = ASIO_MOVE_CAST(T0)(value);
    ec = asio::error_code();
  }
  else
  {
    ec = asio::error::would_block;
  }
}

template <typename T>
T channel_service::get(implementation_type<T>& impl,
    asio::error_code& ec)
{
  if (!impl.buffer_.empty())
  {
    T tmp(ASIO_MOVE_CAST(T)(impl.buffer_.front()));
    impl.buffer_.pop_front();
    if (channel_op<T>* putter =
        static_cast<channel_op<T>*>(impl.putters_.front()))
    {
      impl.buffer_.resize(impl.buffer_.size() + 1);
      impl.buffer_.back() = ASIO_MOVE_CAST(T)(putter->get_value());
      impl.putters_.pop();
      io_service_.post_deferred_completion(putter);
    }
    ec = asio::error_code();
    return ASIO_MOVE_CAST(T)(tmp);
  }
  else if (channel_op<T>* putter =
      static_cast<channel_op<T>*>(impl.putters_.front()))
  {
    T tmp(ASIO_MOVE_CAST(T)(putter->get_value()));
    impl.putters_.pop();
    io_service_.post_deferred_completion(putter);
    ec = asio::error_code();
    return ASIO_MOVE_CAST(T)(tmp);
  }
  else if (impl.open_)
  {
    ec = asio::error::would_block;
    return T();
  }
  else
  {
    ec = asio::error::broken_pipe;
    return T();
  }
}

template <typename T>
void channel_service::start_put_op(implementation_type<T>& impl,
    channel_op<T>* putter, bool is_continuation)
{
  if (!impl.open_)
  {
    putter->on_close();
    io_service_.post_immediate_completion(putter, is_continuation);
  }
  else if (channel_op<T>* getter =
      static_cast<channel_op<T>*>(impl.getters_.front()))
  {
    getter->set_value(ASIO_MOVE_CAST(T)(putter->get_value()));
    impl.getters_.pop();
    io_service_.post_deferred_completion(getter);
    io_service_.post_immediate_completion(putter, is_continuation);
  }
  else
  {
    if (impl.buffer_.size() < impl.max_buffer_size_)
    {
      impl.buffer_.resize(impl.buffer_.size() + 1);
      impl.buffer_.back() = ASIO_MOVE_CAST(T)(putter->get_value());
      io_service_.post_immediate_completion(putter, is_continuation);
    }
    else
    {
      impl.putters_.push(putter);
      io_service_.work_started();
    }
  }
}

template <typename T>
void channel_service::start_get_op(implementation_type<T>& impl,
    channel_op<T>* getter, bool is_continuation)
{
  if (!impl.buffer_.empty())
  {
    getter->set_value(ASIO_MOVE_CAST(T)(impl.buffer_.front()));
    impl.buffer_.pop_front();
    if (channel_op<T>* putter =
        static_cast<channel_op<T>*>(impl.putters_.front()))
    {
      impl.buffer_.resize(impl.buffer_.size() + 1);
      impl.buffer_.back() = ASIO_MOVE_CAST(T)(putter->get_value());
      impl.putters_.pop();
      io_service_.post_deferred_completion(putter);
    }
    io_service_.post_immediate_completion(getter, is_continuation);
  }
  else if (channel_op<T>* putter =
      static_cast<channel_op<T>*>(impl.putters_.front()))
  {
    getter->set_value(ASIO_MOVE_CAST(T)(putter->get_value()));
    impl.putters_.pop();
    io_service_.post_deferred_completion(putter);
    io_service_.post_immediate_completion(getter, is_continuation);
  }
  else if (impl.open_)
  {
    impl.getters_.push(getter);
    io_service_.work_started();
  }
  else
  {
    getter->on_close();
    io_service_.post_immediate_completion(getter, is_continuation);
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_CHANNEL_SERVICE_HPP
