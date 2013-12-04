//
// channel.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_CHANNEL_HPP
#define ASIO_CHANNEL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/channel_service.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/throw_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename T>
class channel : noncopyable
{
public:
  explicit channel(asio::io_service& io_service,
      std::size_t max_buffer_size = 0)
    : service_(asio::use_service<
        asio::detail::channel_service>(io_service))
  {
    service_.construct(impl_, max_buffer_size);
  }

  ~channel()
  {
    service_.destroy(impl_);
  }

  bool is_open() const
  {
    return service_.is_open(impl_);
  }

  void open()
  {
    service_.open(impl_);
  }

  void close()
  {
    service_.close(impl_);
  }

  void cancel()
  {
    service_.cancel(impl_);
  }

  bool ready() const
  {
    return service_.ready(impl_);
  }

  template <typename T0>
  void put(ASIO_MOVE_ARG(T0) value)
  {
    asio::error_code ec;
    service_.put(impl_, ASIO_MOVE_CAST(T0)(value), ec);
    asio::detail::throw_error(ec, "put");
  }

  template <typename T0>
  void put(ASIO_MOVE_ARG(T0) value,
      asio::error_code& ec)
  {
    service_.put(impl_, ASIO_MOVE_CAST(T0)(value), ec);
  }

  template <typename T0, typename PutHandler>
  ASIO_INITFN_RESULT_TYPE(PutHandler,
      void (asio::error_code))
  async_put(ASIO_MOVE_ARG(T0) value,
      ASIO_MOVE_ARG(PutHandler) handler)
  {
    detail::async_result_init<
      PutHandler, void (asio::error_code)> init(
        ASIO_MOVE_CAST(PutHandler)(handler));

    service_.async_put(impl_, ASIO_MOVE_CAST(T0)(value), init.handler);

    return init.result.get();
  }

  T get()
  {
    asio::error_code ec;
    T tmp(service_.get(impl_, ec));
    asio::detail::throw_error(ec, "get");
    return tmp;
  }

  T get(asio::error_code& ec)
  {
    return service_.get(impl_, ec);
  }

  template <typename GetHandler>
  ASIO_INITFN_RESULT_TYPE(GetHandler,
      void (asio::error_code, T))
  async_get(ASIO_MOVE_ARG(GetHandler) handler)
  {
    detail::async_result_init<
      GetHandler, void (asio::error_code, T)> init(
        ASIO_MOVE_CAST(GetHandler)(handler));

    service_.async_get(impl_, init.handler);

    return init.result.get();
  }

private:
  asio::detail::channel_service& service_;
  asio::detail::channel_service::implementation_type<T> impl_;
};

template <>
class channel<void> : noncopyable
{
public:
  explicit channel(asio::io_service& io_service,
      std::size_t max_buffer_size = 0)
    : service_(asio::use_service<
        asio::detail::channel_service>(io_service))
  {
    service_.construct(impl_, max_buffer_size);
  }

  ~channel()
  {
    service_.destroy(impl_);
  }

  bool is_open() const
  {
    return service_.is_open(impl_);
  }

  void open()
  {
    service_.open(impl_);
  }

  void close()
  {
    service_.close(impl_);
  }

  void cancel()
  {
    service_.cancel(impl_);
  }

  bool ready() const
  {
    return service_.ready(impl_);
  }

  void put()
  {
    asio::error_code ec;
    service_.put(impl_, ec);
    asio::detail::throw_error(ec, "put");
  }

  void put(asio::error_code& ec)
  {
    service_.put(impl_, ec);
  }

  template <typename PutHandler>
  ASIO_INITFN_RESULT_TYPE(PutHandler,
      void (asio::error_code))
  async_put(ASIO_MOVE_ARG(PutHandler) handler)
  {
    detail::async_result_init<
      PutHandler, void (asio::error_code)> init(
        ASIO_MOVE_CAST(PutHandler)(handler));

    service_.async_put(impl_, init.handler);

    return init.result.get();
  }

  void get()
  {
    asio::error_code ec;
    service_.get(impl_, ec);
    asio::detail::throw_error(ec, "get");
  }

  void get(asio::error_code& ec)
  {
    service_.get(impl_, ec);
  }

  template <typename GetHandler>
  ASIO_INITFN_RESULT_TYPE(GetHandler,
      void (asio::error_code))
  async_get(ASIO_MOVE_ARG(GetHandler) handler)
  {
    detail::async_result_init<
      GetHandler, void (asio::error_code)> init(
        ASIO_MOVE_CAST(GetHandler)(handler));

    service_.async_get(impl_, init.handler);

    return init.result.get();
  }

private:
  asio::detail::channel_service& service_;
  asio::detail::channel_service::implementation_type<void> impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_CHANNEL_HPP
