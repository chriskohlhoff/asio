//
// impl/go.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_GO_HPP
#define ASIO_IMPL_GO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/handler_type.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

  template <typename Handler>
  class stackless_impl_base
  {
  public:
    template <typename Handler1>
    explicit stackless_impl_base(ASIO_MOVE_ARG(Handler1) handler,
        void (*resume_fn)(const basic_stackless_context<Handler>&))
      : handler_(ASIO_MOVE_CAST(Handler1)(handler)),
        async_result_(0),
        resume_(resume_fn)
    {
    }

    static void resume(const shared_ptr<stackless_impl_base>& impl)
    {
      impl->async_result_ = 0;
      const basic_stackless_context<Handler> ctx(impl, impl->handler_,
          &impl->coroutine_, &impl->throw_ec_, &impl->async_result_);
      impl->resume_(ctx);
    }

  //private:
    Handler handler_;
    asio::coroutine coroutine_;
    asio::error_code throw_ec_;
    void* async_result_;
    void (*resume_)(const basic_stackless_context<Handler>&);
  };

  template <typename Handler, typename Function>
  class stackless_impl :
    public stackless_impl_base<Handler>
  {
  public:
    template <typename Handler1, typename Function1>
    stackless_impl(ASIO_MOVE_ARG(Handler1) handler,
        ASIO_MOVE_ARG(Function1) function)
      : stackless_impl_base<Handler>(
          ASIO_MOVE_CAST(Handler1)(handler),
          &stackless_impl::do_resume),
        function_(ASIO_MOVE_CAST(Function1)(function))
    {
    }

  private:
    static void do_resume(const basic_stackless_context<Handler>& ctx)
    {
      stackless_impl_base<Handler>* base = ctx.stackless_impl_.get();
      static_cast<stackless_impl*>(base)->function_(ctx);
    }

    Function function_;
  };

  template <typename Handler, typename T>
  class stackless_handler
  {
  public:
    stackless_handler(basic_stackless_context<Handler> ctx)
      : stackless_impl_(ctx.stackless_impl_),
        handler_(ctx.handler_),
        ec_(ctx.ec_),
        value_(*ctx.async_result_)
    {
    }

    void operator()(T value)
    {
      *ec_ = asio::error_code();
      if (value_) *static_cast<T*>(value_) = value;
      stackless_impl_base<Handler>::resume(stackless_impl_);
    }

    void operator()(asio::error_code ec, T value)
    {
      *ec_ = ec;
      if (value_) *static_cast<T*>(value_) = value;
      stackless_impl_base<Handler>::resume(stackless_impl_);
    }

  //private:
    shared_ptr<stackless_impl_base<Handler> > stackless_impl_;
    Handler& handler_;
    asio::error_code* ec_;
    void* value_;
  };

  template <typename Handler>
  class stackless_handler<Handler, void>
  {
  public:
    stackless_handler(basic_stackless_context<Handler> ctx)
      : stackless_impl_(ctx.stackless_impl_),
        handler_(ctx.handler_),
        ec_(ctx.ec_)
    {
    }

    void operator()()
    {
      *ec_ = asio::error_code();
      stackless_impl_base<Handler>::resume(stackless_impl_);
    }

    void operator()(asio::error_code ec)
    {
      *ec_ = ec;
      stackless_impl_base<Handler>::resume(stackless_impl_);
    }

  //private:
    shared_ptr<stackless_impl_base<Handler> > stackless_impl_;
    Handler& handler_;
    asio::error_code* ec_;
  };

  template <typename Handler, typename T>
  inline void* asio_handler_allocate(std::size_t size,
      stackless_handler<Handler, T>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename Handler, typename T>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      stackless_handler<Handler, T>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Handler, typename T>
  inline bool asio_handler_is_continuation(
      stackless_handler<Handler, T>*)
  {
    return true;
  }

  template <typename Function, typename Handler, typename T>
  inline void asio_handler_invoke(Function& function,
      stackless_handler<Handler, T>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename Handler, typename T>
  inline void asio_handler_invoke(const Function& function,
      stackless_handler<Handler, T>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Handler>
  struct resume_helper
  {
    explicit resume_helper(
        shared_ptr<stackless_impl_base<Handler> > stackless_impl)
      : stackless_impl_(stackless_impl)
    {
    }

    void operator()()
    {
      stackless_impl_base<Handler>::resume(stackless_impl_);
    }

    shared_ptr<stackless_impl_base<Handler> > stackless_impl_;
  };

  inline void default_go_handler() {}

} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename Handler, typename ReturnType>
struct handler_type<basic_stackless_context<Handler>, ReturnType()>
{
  typedef detail::stackless_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg1>
struct handler_type<basic_stackless_context<Handler>, ReturnType(Arg1)>
{
  typedef detail::stackless_handler<Handler, Arg1> type;
};

template <typename Handler, typename ReturnType>
struct handler_type<basic_stackless_context<Handler>,
    ReturnType(asio::error_code)>
{
  typedef detail::stackless_handler<Handler, void> type;
};

template <typename Handler, typename ReturnType, typename Arg2>
struct handler_type<basic_stackless_context<Handler>,
    ReturnType(asio::error_code, Arg2)>
{
  typedef detail::stackless_handler<Handler, Arg2> type;
};

template <typename Handler, typename T>
class async_result<detail::stackless_handler<Handler, T> >
{
public:
  typedef detail::coroutine_async_result<T> type;

  explicit async_result(detail::stackless_handler<Handler, T>&)
  {
  }

  type get()
  {
    return type();
  }
};

template <typename Handler, typename Function>
void go(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function)
{
  typedef typename remove_const<
    typename remove_reference<Handler>::type>::type
      handler_type;

  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::resume_helper<handler_type> helper(
      detail::shared_ptr<detail::stackless_impl_base<handler_type> >(
        new detail::stackless_impl<handler_type, function_type>(
          ASIO_MOVE_CAST(Handler)(handler),
          ASIO_MOVE_CAST(Function)(function))));

  asio_handler_invoke_helpers::invoke(
      helper, helper.stackless_impl_->handler_);
}

template <typename Handler, typename Function>
void go(basic_stackless_context<Handler> ctx,
    ASIO_MOVE_ARG(Function) function)
{
  Handler handler(ctx.handler_);
  go(ASIO_MOVE_CAST(Handler)(handler),
      ASIO_MOVE_CAST(Function)(function));
}

template <typename Function>
void go(asio::io_service::strand strand,
    ASIO_MOVE_ARG(Function) function)
{
  asio::go(strand.wrap(&detail::default_go_handler),
      ASIO_MOVE_CAST(Function)(function));
}

template <typename Function>
void go(asio::io_service& io_service,
    ASIO_MOVE_ARG(Function) function)
{
  asio::go(asio::io_service::strand(io_service),
      ASIO_MOVE_CAST(Function)(function));
}

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_GO_HPP
