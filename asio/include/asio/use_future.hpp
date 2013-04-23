//
// use_future.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_USE_FUTURE_HPP
#define ASIO_USE_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <future>
#include "asio/error_code.hpp"
#include "asio/handler_token.hpp"
#include "asio/handler_type.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

// A special value, similar to std::nothrow.
struct use_future_t { constexpr use_future_t() {} };
constexpr use_future_t use_future;

// Completion handler to adapt a promise as a completion handler.
template <typename T>
class promise_handler
{
public:
  // Construct from use_future special value.
  promise_handler(use_future_t)
  {
  }

  void operator()(T t)
  {
    assert(!!promise_);
    promise_->set_value(t);
  }

  void operator()(const asio::error_code& ec, T t)
  {
    assert(!!promise_);
    if (ec)
      promise_->set_exception(std::make_exception_ptr(asio::system_error(ec)));
    else
      promise_->set_value(t);
  }

//private:
  std::shared_ptr<std::promise<T> > promise_;
};

// Completion handler to adapt a void promise as a completion handler.
template <>
class promise_handler<void>
{
public:
  // Construct from use_future special value. Used during rebinding.
  promise_handler(use_future_t)
  {
  }

  void operator()()
  {
    assert(!!promise_);
    promise_->set_value();
  }

  void operator()(const asio::error_code& ec)
  {
    assert(!!promise_);
    if (ec)
      promise_->set_exception(std::make_exception_ptr(asio::system_error(ec)));
    else
      promise_->set_value();
  }

//private:
  std::shared_ptr<std::promise<void> > promise_;
};

// Ensure any exceptions thrown from the handler are propagated back to the
// caller via the future.
template <typename Function, typename T>
void asio_handler_invoke(Function f, promise_handler<T>* h)
{
  std::shared_ptr<std::promise<T> > p(h->promise_);
  try
  {
    f();
  }
  catch (...)
  {
    p->set_exception(std::current_exception());
  }
}

// Handler traits specialisation for promise_handler.
template <typename T>
class handler_token<promise_handler<T> >
{
public:
  // The initiating function will return a future.
  typedef std::future<T> type;

  // Token constructor creates a new promise for the async operation, and
  // obtains the corresponding future.
  explicit handler_token(promise_handler<T>& h)
  {
    h.promise_ = std::make_shared<std::promise<T> >();
    value_ = h.promise_->get_future();
  }

  // Obtain the future to be returned from the initiating function.
  type get() { return std::move(value_); }

private:
  type value_;
};

// Handler type specialisation for use_future.
template <typename ReturnType>
struct handler_type<use_future_t, ReturnType()>
{
  typedef promise_handler<void> type;
};

// Handler type specialisation for use_future.
template <typename ReturnType, typename Arg1>
struct handler_type<use_future_t, ReturnType(Arg1)>
{
  typedef promise_handler<Arg1> type;
};

// Handler type specialisation for use_future.
template <typename ReturnType>
struct handler_type<use_future_t, ReturnType(asio::error_code)>
{
  typedef promise_handler<void> type;
};

// Handler type specialisation for use_future.
template <typename ReturnType, typename Arg2>
struct handler_type<use_future_t, ReturnType(asio::error_code, Arg2)>
{
  typedef promise_handler<Arg2> type;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_USE_FUTURE_HPP
