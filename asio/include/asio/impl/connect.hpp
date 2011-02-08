//
// impl/connect.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_CONNECT_HPP
#define ASIO_IMPL_CONNECT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/consuming_buffers.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail
{
  struct default_connect_condition
  {
    template <typename Iterator>
    Iterator operator()(const asio::error_code&, Iterator next)
    {
      return next;
    }
  };
}

template <typename Socket, typename Iterator>
Iterator connect(Socket& s, Iterator begin)
{
  asio::error_code ec;
  Iterator result = connect(s, begin, ec);
  asio::detail::throw_error(ec);
  return result;
}

template <typename Socket, typename Iterator>
inline Iterator connect(Socket& s, Iterator begin,
    asio::error_code& ec)
{
  return connect(s, begin, Iterator(), detail::default_connect_condition(), ec);
}

template <typename Socket, typename Iterator>
Iterator connect(Socket& s, Iterator begin, Iterator end)
{
  asio::error_code ec;
  Iterator result = connect(s, begin, end, ec);
  asio::detail::throw_error(ec);
  return result;
}

template <typename Socket, typename Iterator>
inline Iterator connect(Socket& s, Iterator begin, Iterator end,
    asio::error_code& ec)
{
  return connect(s, begin, end, detail::default_connect_condition(), ec);
}

template <typename Socket, typename Iterator, typename ConnectCondition>
Iterator connect(Socket& s, Iterator begin,
    ConnectCondition connect_condition)
{
  asio::error_code ec;
  Iterator result = connect(s, begin, connect_condition, ec);
  asio::detail::throw_error(ec);
  return result;
}

template <typename Socket, typename Iterator, typename ConnectCondition>
inline Iterator connect(Socket& s, Iterator begin,
    ConnectCondition connect_condition, asio::error_code& ec)
{
  return connect(s, begin, Iterator(), connect_condition, ec);
}

template <typename Socket, typename Iterator, typename ConnectCondition>
Iterator connect(Socket& s, Iterator begin, Iterator end,
    ConnectCondition connect_condition)
{
  asio::error_code ec;
  Iterator result = connect(s, begin, end, connect_condition, ec);
  asio::detail::throw_error(ec);
  return result;
}

template <typename Socket, typename Iterator, typename ConnectCondition>
Iterator connect(Socket& s, Iterator begin, Iterator end,
    ConnectCondition connect_condition, asio::error_code& ec)
{
  ec = asio::error_code();

  for (Iterator iter = begin; iter != end; ++iter)
  {
    iter = connect_condition(ec, iter);
    if (iter != end)
    {
      s.close(ec);
      s.connect(*iter, ec);
      if (!ec)
        return iter;
    }
  }

  if (!ec)
    ec = asio::error::not_found;

  return end;
}

namespace detail
{
  // Enable the empty base class optimisation for the connect condition.
  template <typename ConnectCondition>
  class base_from_connect_condition
  {
  protected:
    explicit base_from_connect_condition(
        const ConnectCondition& connect_condition)
      : connect_condition_(connect_condition)
    {
    }

    template <typename Iterator>
    void check_condition(const asio::error_code& ec,
        Iterator& iter, Iterator& end)
    {
      if (iter != end)
        iter = connect_condition_(ec, static_cast<const Iterator&>(iter));
    }

  private:
    ConnectCondition connect_condition_;
  };

  // The default_connect_condition implementation is essentially a no-op. This
  // template specialisation lets us eliminate all costs associated with it.
  template <>
  class base_from_connect_condition<default_connect_condition>
  {
  protected:
    explicit base_from_connect_condition(const default_connect_condition&)
    {
    }

    template <typename Iterator>
    void check_condition(const asio::error_code&, Iterator&, Iterator&)
    {
    }
  };

  template <typename Socket, typename Iterator,
      typename ConnectCondition, typename ComposedConnectHandler>
  class connect_op : base_from_connect_condition<ConnectCondition>
  {
  public:
    connect_op(Socket& sock,
        const Iterator& begin, const Iterator& end,
        const ConnectCondition& connect_condition,
        ComposedConnectHandler& handler)
      : base_from_connect_condition<ConnectCondition>(connect_condition),
        socket_(sock),
        iter_(begin),
        end_(end),
        handler_(ASIO_MOVE_CAST(ComposedConnectHandler)(handler))
    {
    }

    void operator()(asio::error_code ec, int start = 0)
    {
      switch (start)
      {
        case 1:
        for (;;)
        {
          this->check_condition(ec, iter_, end_);

          if (iter_ != end_)
          {
            socket_.close(ec);
            socket_.async_connect(*iter_, *this);
            return;
          }

          if (start)
          {
            ec = asio::error::not_found;
            socket_.get_io_service().post(detail::bind_handler(*this, ec));
            return;
          }

          default:

          if (iter_ == end_)
            break;

          if (!socket_.is_open())
          {
            ec = asio::error::operation_aborted;
            break;
          }

          if (!ec)
            break;

          ++iter_;
        }

        handler_(static_cast<const asio::error_code&>(ec),
            static_cast<const Iterator&>(iter_));
      }
    }

  //private:
    Socket& socket_;
    Iterator iter_;
    Iterator end_;
    ComposedConnectHandler handler_;
  };

  template <typename Socket, typename Iterator,
      typename ConnectCondition, typename ComposedConnectHandler>
  inline void* asio_handler_allocate(std::size_t size,
      connect_op<Socket, Iterator, ConnectCondition,
        ComposedConnectHandler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename Socket, typename Iterator,
      typename ConnectCondition, typename ComposedConnectHandler>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      connect_op<Socket, Iterator, ConnectCondition,
        ComposedConnectHandler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Function, typename Socket, typename Iterator,
      typename ConnectCondition, typename ComposedConnectHandler>
  inline void asio_handler_invoke(const Function& function,
      connect_op<Socket, Iterator, ConnectCondition,
        ComposedConnectHandler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

template <typename Socket, typename Iterator, typename ComposedConnectHandler>
inline void async_connect(Socket& s, Iterator begin,
    ComposedConnectHandler handler)
{
  detail::connect_op<Socket, Iterator,
    detail::default_connect_condition, ComposedConnectHandler>(
      s, begin, Iterator(), detail::default_connect_condition(), handler)(
        asio::error_code(), 1);
}

template <typename Socket, typename Iterator, typename ComposedConnectHandler>
inline void async_connect(Socket& s, Iterator begin, Iterator end,
    ComposedConnectHandler handler)
{
  detail::connect_op<Socket, Iterator,
    detail::default_connect_condition, ComposedConnectHandler>(
      s, begin, end, detail::default_connect_condition(), handler)(
        asio::error_code(), 1);
}

template <typename Socket, typename Iterator,
    typename ConnectCondition, typename ComposedConnectHandler>
inline void async_connect(Socket& s, Iterator begin,
    ConnectCondition connect_condition, ComposedConnectHandler handler)
{
  detail::connect_op<Socket, Iterator,
    ConnectCondition, ComposedConnectHandler>(
      s, begin, Iterator(), connect_condition, handler)(
        asio::error_code(), 1);
}

template <typename Socket, typename Iterator,
    typename ConnectCondition, typename ComposedConnectHandler>
void async_connect(Socket& s, Iterator begin, Iterator end,
    ConnectCondition connect_condition, ComposedConnectHandler handler)
{
  detail::connect_op<Socket, Iterator,
    ConnectCondition, ComposedConnectHandler>(
      s, begin, end, connect_condition, handler)(
        asio::error_code(), 1);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_CONNECT_HPP
