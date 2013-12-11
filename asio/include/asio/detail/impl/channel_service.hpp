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

inline bool channel_service::is_open(const base_implementation_type& impl) const
{
  return impl.put_state_ != closed;
}

template <typename T>
void channel_service::reset(implementation_type<T>& impl)
{
  cancel(impl);
  if (impl.get_state_ == closed)
    impl.get_state_ = block;
  if (impl.put_state_ == closed)
    impl.put_state_ = impl.max_buffer_size_ ? buffer : block;
  impl.buffer_clear();
}

template <typename T>
inline bool channel_service::ready(const implementation_type<T>& impl) const
{
  return impl.get_state_ != block;
}

template <typename T, typename T0>
void channel_service::put(implementation_type<T>& impl,
    ASIO_MOVE_ARG(T0) value, asio::error_code& ec)
{
  switch (impl.put_state_)
  {
  case block:
    {
      ec = asio::error::would_block;
      break;
    }
  case buffer:
    {
      impl.buffer_push(ASIO_MOVE_CAST(T0)(value));
      impl.get_state_ = buffer;
      if (impl.buffer_size() == impl.max_buffer_size_)
        impl.put_state_ = block;
      ec = asio::error_code();
      break;
    }
  case waiter:
    {
      channel_op<T>* getter =
        static_cast<channel_op<T>*>(impl.waiters_.front());
      getter->set_value(ASIO_MOVE_CAST(T0)(value));
      impl.waiters_.pop();
      if (impl.waiters_.empty())
        impl.put_state_ = impl.max_buffer_size_ ? buffer : block;
      io_service_.post_deferred_completion(getter);
      ec = asio::error_code();
      break;
    }
  case closed:
  default:
    {
      ec = asio::error::broken_pipe;
      break;
    }
  }
}

template <typename T>
typename channel_service::implementation_type<T>::value_type
channel_service::get(
    channel_service::implementation_type<T>& impl,
    asio::error_code& ec)
{
  switch (impl.get_state_)
  {
  case block:
    {
      ec = asio::error::would_block;
      return typename implementation_type<T>::value_type();
    }
  case buffer:
    {
      typename implementation_type<T>::value_type tmp(impl.buffer_front());
      if (channel_op<T>* putter =
          static_cast<channel_op<T>*>(impl.waiters_.front()))
      {
        impl.buffer_push(putter->get_value());
        impl.buffer_pop();
        impl.waiters_.pop();
        io_service_.post_deferred_completion(putter);
      }
      else
      {
        impl.buffer_pop();
        if (impl.buffer_size() == 0)
          impl.get_state_ = (impl.put_state_ == closed) ? closed : block;
        impl.put_state_ = (impl.put_state_ == closed) ? closed : buffer;
      }
      ec = asio::error_code();
      return tmp;
    }
  case waiter:
    {
      channel_op<T>* putter =
        static_cast<channel_op<T>*>(impl.waiters_.front());
      typename implementation_type<T>::value_type tmp(putter->get_value());
      impl.waiters_.pop();
      if (impl.waiters_.front() == 0)
        impl.get_state_ = (impl.put_state_ == closed) ? closed : block;
      io_service_.post_deferred_completion(putter);
      ec = asio::error_code();
      return tmp;
    }
  case closed:
  default:
    {
      ec = asio::error::broken_pipe;
      return typename implementation_type<T>::value_type();
    }
  }
}

template <typename T>
void channel_service::start_put_op(implementation_type<T>& impl,
    channel_op<T>* putter, bool is_continuation)
{
  switch (impl.put_state_)
  {
  case block:
    {
      impl.waiters_.push(putter);
      io_service_.work_started();
      if (impl.get_state_ == block)
        impl.get_state_ = waiter;
      return;
    }
  case buffer:
    {
      impl.buffer_push(putter->get_value());
      impl.get_state_ = buffer;
      if (impl.buffer_size() == impl.max_buffer_size_)
        impl.put_state_ = block;
      break;
    }
  case waiter:
    {
      channel_op<T>* getter =
        static_cast<channel_op<T>*>(impl.waiters_.front());
      getter->set_value(putter->get_value());
      impl.waiters_.pop();
      if (impl.waiters_.empty())
        impl.put_state_ = impl.max_buffer_size_ ? buffer : block;
      io_service_.post_deferred_completion(getter);
      break;
    }
  case closed:
  default:
    {
      putter->on_close();
      break;
    }
  }

  io_service_.post_immediate_completion(putter, is_continuation);
}

template <typename T>
void channel_service::start_get_op(implementation_type<T>& impl,
    channel_op<T>* getter, bool is_continuation)
{
  switch (impl.get_state_)
  {
  case block:
    {
      impl.waiters_.push(getter);
      io_service_.work_started();
      if (impl.put_state_ != closed)
        impl.put_state_ = waiter;
      return;
    }
  case buffer:
    {
      getter->set_value(impl.buffer_front());
      if (channel_op<T>* putter =
          static_cast<channel_op<T>*>(impl.waiters_.front()))
      {
        impl.buffer_push(putter->get_value());
        impl.buffer_pop();
        impl.waiters_.pop();
        io_service_.post_deferred_completion(putter);
      }
      else
      {
        impl.buffer_pop();
        if (impl.buffer_size() == 0)
          impl.get_state_ = (impl.put_state_ == closed) ? closed : block;
        impl.put_state_ = (impl.put_state_ == closed) ? closed : buffer;
      }
      break;
    }
  case waiter:
    {
      channel_op<T>* putter =
        static_cast<channel_op<T>*>(impl.waiters_.front());
      getter->set_value(putter->get_value());
      impl.waiters_.pop();
      if (impl.waiters_.front() == 0)
        impl.get_state_ = (impl.put_state_ == closed) ? closed : block;
      io_service_.post_deferred_completion(putter);
      break;
    }
  case closed:
  default:
    {
      getter->on_close();
      break;
    }
  }

  io_service_.post_immediate_completion(getter, is_continuation);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_CHANNEL_SERVICE_HPP
