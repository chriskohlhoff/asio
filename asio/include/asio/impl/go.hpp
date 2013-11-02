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
#include "asio/detail/atomic_count.hpp"
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
  protected:
    template <typename Handler1>
    explicit stackless_impl_base(ASIO_MOVE_ARG(Handler1) handler,
        void (*resume_fn)(const basic_stackless_context<Handler>&),
        void (*destroy_fn)(stackless_impl_base*))
      : handler_(ASIO_MOVE_CAST(Handler1)(handler)),
        ref_count_(1),
        result_ec_(&throw_ec_),
        result_value_(0),
        resume_(resume_fn),
        destroy_(destroy_fn)
    {
    }

  public:
    static stackless_impl_base* add_ref(stackless_impl_base* impl);
    static stackless_impl_base* move_ref(stackless_impl_base*& impl);
    static void release_ref(stackless_impl_base* impl);
    static void resume(stackless_impl_base*& impl);

  //private:
    Handler handler_;
    asio::detail::atomic_count ref_count_;
    asio::coroutine coroutine_;
    asio::error_code throw_ec_;
    asio::error_code* result_ec_;
    void* result_value_;
    void (*resume_)(const basic_stackless_context<Handler>&);
    void (*destroy_)(stackless_impl_base*);
  };

  template <typename Handler>
  stackless_impl_base<Handler>* stackless_impl_base<Handler>::add_ref(
      stackless_impl_base<Handler>* impl)
  {
    if (impl)
      ++impl->ref_count_;
    return impl;
  }

  template <typename Handler>
  inline stackless_impl_base<Handler>* stackless_impl_base<Handler>::move_ref(
      stackless_impl_base<Handler>*& impl)
  {
    stackless_impl_base* tmp = impl;
    impl = 0;
    return tmp;
  }

  template <typename Handler>
  void stackless_impl_base<Handler>::release_ref(
      stackless_impl_base<Handler>* impl)
  {
    if (impl)
    {
      if (--impl->ref_count_ == 0)
        impl->destroy_(impl);
    }
  }

  template <typename Handler>
  void stackless_impl_base<Handler>::resume(
      stackless_impl_base<Handler>*& impl)
  {
    impl->result_value_ = 0;
    const basic_stackless_context<Handler> ctx = { &impl, &impl->throw_ec_ };
    impl->resume_(ctx);
    if (impl && impl->coroutine_.is_complete())
      impl->handler_();
  }

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
          &stackless_impl::do_resume,
          &stackless_impl::do_destroy),
        function_(ASIO_MOVE_CAST(Function1)(function))
    {
    }

  private:
    static void do_resume(const basic_stackless_context<Handler>& ctx)
    {
      stackless_impl_base<Handler>* base = *ctx.impl_;
      static_cast<stackless_impl*>(base)->function_(ctx);
    }

    static void do_destroy(stackless_impl_base<Handler>* impl)
    {
      delete static_cast<stackless_impl*>(impl);
    }

    Function function_;
  };

  template <typename Handler, typename T>
  class stackless_handler
  {
  public:
    stackless_handler(const basic_stackless_context<Handler>& ctx)
      : impl_(stackless_impl_base<Handler>::move_ref(*ctx.impl_))
    {
      impl_->result_ec_ = ctx.ec_;
    }

    stackless_handler(const stackless_handler& other)
      : impl_(stackless_impl_base<Handler>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    stackless_handler(stackless_handler&& other)
      : impl_(stackless_impl_base<Handler>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~stackless_handler()
    {
      stackless_impl_base<Handler>::release_ref(impl_);
    }

    void operator()(T value)
    {
      *impl_->result_ec_ = asio::error_code();
      if (impl_->result_value_)
        *static_cast<T*>(impl_->result_value_) = value;
      stackless_impl_base<Handler>::resume(impl_);
    }

    void operator()(asio::error_code ec, T value)
    {
      *impl_->result_ec_ = ec;
      if (impl_->result_value_)
        *static_cast<T*>(impl_->result_value_) = value;
      stackless_impl_base<Handler>::resume(impl_);
    }

  //private:
    stackless_impl_base<Handler>* impl_;
  };

  template <typename Handler>
  class stackless_handler<Handler, void>
  {
  public:
    stackless_handler(const basic_stackless_context<Handler>& ctx)
      : impl_(stackless_impl_base<Handler>::move_ref(*ctx.impl_))
    {
      impl_->result_ec_ = ctx.ec_;
    }

    stackless_handler(const stackless_handler& other)
      : impl_(stackless_impl_base<Handler>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    stackless_handler(stackless_handler&& other)
      : impl_(stackless_impl_base<Handler>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~stackless_handler()
    {
      stackless_impl_base<Handler>::release_ref(impl_);
    }

    void operator()()
    {
      *impl_->result_ec_ = asio::error_code();
      stackless_impl_base<Handler>::resume(impl_);
    }

    void operator()(asio::error_code ec)
    {
      *impl_->result_ec_ = ec;
      stackless_impl_base<Handler>::resume(impl_);
    }

  //private:
    stackless_impl_base<Handler>* impl_;
  };

  template <typename Handler, typename T>
  inline void* asio_handler_allocate(std::size_t size,
      stackless_handler<Handler, T>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->impl_->handler_);
  }

  template <typename Handler, typename T>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      stackless_handler<Handler, T>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->impl_->handler_);
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
        function, this_handler->impl_->handler_);
  }

  template <typename Function, typename Handler, typename T>
  inline void asio_handler_invoke(const Function& function,
      stackless_handler<Handler, T>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->impl_->handler_);
  }

  template <typename Handler, typename Function>
  struct go_helper
  {
    template <typename Handler1, typename Function1>
    go_helper(ASIO_MOVE_ARG(Handler1) handler,
        ASIO_MOVE_ARG(Function1) function)
      : impl_(new stackless_impl<Handler, Function>(
            ASIO_MOVE_CAST(Handler1)(handler),
            ASIO_MOVE_CAST(Function1)(function)))
    {
    }

    go_helper(const go_helper& other)
      : impl_(stackless_impl_base<Handler>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    go_helper(go_helper&& other)
      : impl_(stackless_impl_base<Handler>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~go_helper()
    {
      stackless_impl_base<Handler>::release_ref(impl_);
    }

    void operator()()
    {
      stackless_impl_base<Handler>::resume(impl_);
    }

    stackless_impl_base<Handler>* impl_;
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
  typedef awaitable<T> type;

  explicit async_result(detail::stackless_handler<Handler, T>&)
  {
  }

  type get()
  {
    return type();
  }
};

template <typename Handler>
inline coroutine& get_coroutine(
    basic_stackless_context<Handler>& c)
{
  return (*c.impl_)->coroutine_;
}

template <typename Handler>
inline coroutine& get_coroutine(
    basic_stackless_context<Handler>* c)
{
  return (*c->impl_)->coroutine_;
}

template <typename Handler>
inline const asio::error_code* get_coroutine_error(
    basic_stackless_context<Handler>& c)
{
  return &(*c.impl_)->throw_ec_;
}

template <typename Handler>
inline const asio::error_code* get_coroutine_error(
    basic_stackless_context<Handler>* c)
{
  return &(*c->impl_)->throw_ec_;
}

template <typename Handler>
inline void** get_coroutine_async_result(
    basic_stackless_context<Handler>& c)
{
  return &(*c.impl_)->result_value_;
}

template <typename Handler>
inline void** get_coroutine_async_result(
    basic_stackless_context<Handler>* c)
{
  return &(*c->impl_)->result_value_;
}

template <typename Handler, typename Function>
void go(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function)
{
  typedef typename handler_type<Handler, void()>::type
      real_handler_type;

  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<real_handler_type, function_type> helper(
      ASIO_MOVE_CAST(Handler)(handler),
      ASIO_MOVE_CAST(Function)(function));

  asio_handler_invoke_helpers::invoke(
      helper, helper.impl_->handler_);
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
