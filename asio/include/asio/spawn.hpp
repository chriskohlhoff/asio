//
// spawn.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SPAWN_HPP
#define ASIO_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/coroutine/coroutine.hpp>
#include "asio/async_result.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/shared_ptr.hpp"
#include "asio/detail/weak_ptr.hpp"
#include "asio/detail/wrapped_handler.hpp"
#include "asio/handler_type.hpp"
#include "asio/io_service.hpp"
#include "asio/strand.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Handler>
class basic_yield_context
{
public:
  basic_yield_context(
      const detail::weak_ptr<boost::coroutines::coroutine<void()> >& coro,
      boost::coroutines::coroutine<void()>::caller_type& ca, Handler& handler)
    : coro_(coro),
      ca_(ca),
      handler_(handler),
      ec_(0)
  {
  }

  basic_yield_context operator()(asio::error_code& ec)
  {
    basic_yield_context tmp(*this);
    tmp.ec_ = &ec;
    return tmp;
  }

//private:
  detail::weak_ptr<boost::coroutines::coroutine<void()> > coro_;
  boost::coroutines::coroutine<void()>::caller_type& ca_;
  Handler& handler_;
  asio::error_code* ec_;
};

typedef basic_yield_context<
  detail::wrapped_handler<
    io_service::strand, void(*)()> > yield_context;

template <typename Handler, typename T>
class coro_handler
{
public:
  coro_handler(basic_yield_context<Handler> ctx)
    : coro_(ctx.coro_.lock()),
      ca_(ctx.ca_),
      handler_(ctx.handler_),
      ec_(ctx.ec_),
      value_(0)
  {
  }

  void operator()(T value)
  {
    *ec_ = asio::error_code();
    *value_ = value;
    (*coro_)();
  }

  void operator()(asio::error_code ec, T value)
  {
    *ec_ = ec;
    *value_ = value;
    (*coro_)();
  }

//private:
  detail::shared_ptr<boost::coroutines::coroutine<void()> > coro_;
  boost::coroutines::coroutine<void()>::caller_type& ca_;
  Handler& handler_;
  asio::error_code* ec_;
  T* value_;
};

template <typename Handler>
class coro_handler<Handler, void>
{
public:
  coro_handler(basic_yield_context<Handler> ctx)
    : coro_(ctx.coro_.lock()),
      ca_(ctx.ca_),
      handler_(ctx.handler_),
      ec_(ctx.ec_)
  {
  }

  void operator()()
  {
    *ec_ = asio::error_code();
    (*coro_)();
  }

  void operator()(asio::error_code ec)
  {
    *ec_ = ec;
    (*coro_)();
  }

//private:
  detail::shared_ptr<boost::coroutines::coroutine<void()> > coro_;
  boost::coroutines::coroutine<void()>::caller_type& ca_;
  Handler& handler_;
  asio::error_code* ec_;
};

template <typename Function, typename Handler, typename T>
inline void asio_handler_invoke(Function& function,
    coro_handler<Handler, T>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename T>
inline void asio_handler_invoke(const Function& function,
    coro_handler<Handler, T>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename ReturnType>
struct handler_type<basic_yield_context<Handler>, ReturnType()>
{
  typedef coro_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg1>
struct handler_type<basic_yield_context<Handler>, ReturnType(Arg1)>
{
  typedef coro_handler<Handler, Arg1> type;
};

template <typename Handler, typename ReturnType>
struct handler_type<basic_yield_context<Handler>,
    ReturnType(asio::error_code)>
{
  typedef coro_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg2>
struct handler_type<basic_yield_context<Handler>,
    ReturnType(asio::error_code, Arg2)>
{
  typedef coro_handler<Handler, Arg2> type;
};

template <typename Handler, typename T>
class async_result<coro_handler<Handler, T> >
{
public:
  typedef T type;

  explicit async_result(coro_handler<Handler, T>& h)
    : ca_(h.ca_)
  {
    out_ec_ = h.ec_;
    if (!out_ec_) h.ec_ = &ec_;
    h.value_ = &value_;
  }

  type get()
  {
    ca_();
    if (!out_ec_ && ec_) throw asio::system_error(ec_);
    return value_;
  }

private:
  boost::coroutines::coroutine<void()>::caller_type& ca_;
  asio::error_code* out_ec_;
  asio::error_code ec_;
  type value_;
};

template <typename Handler>
class async_result<coro_handler<Handler, void> >
{
public:
  typedef void type;

  explicit async_result(coro_handler<Handler, void>& h)
    : ca_(h.ca_)
  {
    out_ec_ = h.ec_;
    if (!out_ec_) h.ec_ = &ec_;
  }

  void get()
  {
    ca_();
    if (!out_ec_ && ec_) throw asio::system_error(ec_);
  }

private:
  boost::coroutines::coroutine<void()>::caller_type& ca_;
  asio::error_code* out_ec_;
  asio::error_code ec_;
};

namespace detail {

  template <typename Handler, typename Function>
  struct spawn_data : private noncopyable
  {
    spawn_data(ASIO_MOVE_ARG(Handler) handler,
        bool call_handler, ASIO_MOVE_ARG(Function) function)
      : handler_(ASIO_MOVE_CAST(Handler)(handler)),
        call_handler_(call_handler),
        function_(ASIO_MOVE_CAST(Function)(function))
    {
    }

    weak_ptr<boost::coroutines::coroutine<void()> > coro_;
    Handler handler_;
    bool call_handler_;
    Function function_;
  };

  template <typename Handler, typename Function>
  struct coro_entry_point
  {
    void operator()(boost::coroutines::coroutine<void()>::caller_type& ca)
    {
      shared_ptr<spawn_data<Handler, Function> > data(data_);
      ca(); // Yield until coroutine pointer has been initialised.
      const basic_yield_context<Handler> yield(
          data->coro_, ca, data->handler_);
      (data->function_)(yield);
      if (data->call_handler_)
        (data->handler_)();
    }

    shared_ptr<spawn_data<Handler, Function> > data_;
  };

  template <typename Handler, typename Function>
  struct spawn_helper
  {
    void operator()()
    {
      coro_entry_point<Handler, Function> entry_point = { data_ };
      shared_ptr<boost::coroutines::coroutine<void()> > coro(
          new boost::coroutines::coroutine<void()>(entry_point, attributes_));
      data_->coro_ = coro;
      (*coro)();
    }

    shared_ptr<spawn_data<Handler, Function> > data_;
    boost::coroutines::attributes attributes_;
  };

  inline void default_spawn_handler() {}

} // namespace detail

template <typename Handler, typename Function>
void spawn(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes())
{
  detail::spawn_helper<Handler, Function> helper;
  helper.data_.reset(
      new detail::spawn_data<Handler, Function>(
        ASIO_MOVE_CAST(Handler)(handler), true,
        ASIO_MOVE_CAST(Function)(function)));
  helper.attributes_ = attributes;
  asio_handler_invoke_helpers::invoke(helper, helper.data_->handler_);
}

template <typename Handler, typename Function>
void spawn(basic_yield_context<Handler> ctx,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes())
{
  Handler handler(ctx.handler_); // Explicit copy that might be moved from.
  detail::spawn_helper<Handler, Function> helper;
  helper.data_.reset(
      new detail::spawn_data<Handler, Function>(
        ASIO_MOVE_CAST(Handler)(handler), false,
        ASIO_MOVE_CAST(Function)(function)));
  helper.attributes_ = attributes;
  asio_handler_invoke_helpers::invoke(helper, helper.data_->handler_);
}

template <typename Function>
void spawn(asio::io_service::strand strand,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes())
{
  asio::spawn(strand.wrap(&detail::default_spawn_handler),
      ASIO_MOVE_CAST(Function)(function), attributes);
}

template <typename Function>
void spawn(asio::io_service& io_service,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes())
{
  asio::spawn(asio::io_service::strand(io_service),
      ASIO_MOVE_CAST(Function)(function), attributes);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SPAWN_HPP
