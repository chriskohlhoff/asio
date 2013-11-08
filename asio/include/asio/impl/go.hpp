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
#include "asio/detail/wrapped_handler.hpp"
#include "asio/handler_type.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

  template <typename Handler, typename Signature>
  struct stackless_handler_type
  {
    typedef typename handler_type<Handler, Signature>::type type;
  };

  template <>
  struct stackless_handler_type<void, void()>
  {
    typedef void (*type)();
  };

  template <>
  struct stackless_handler_type<asio::io_service, void()>
  {
    typedef wrapped_handler<asio::io_service&, void(*)()> type;
  };

  template <>
  struct stackless_handler_type<asio::io_service::strand, void()>
  {
    typedef wrapped_handler<asio::io_service::strand,
        void(*)(), is_continuation_if_running> type;
  };

  template <typename Handler, typename Signature>
  class stackless_impl_base
  {
  protected:
    template <typename Handler1>
    explicit stackless_impl_base(ASIO_MOVE_ARG(Handler1) handler,
        void (*resume_fn)(const stackless_context<Handler, Signature>&),
        void (*destroy_fn)(stackless_impl_base*))
      : handler_(ASIO_MOVE_CAST(Handler1)(handler)),
        ref_count_(1),
        result_ec_(&throw_ec_),
        result_value_(0),
        resume_(resume_fn),
        destroy_(destroy_fn),
        completed_(false)
    {
    }

  public:
    static stackless_impl_base* add_ref(stackless_impl_base* impl);
    static stackless_impl_base* move_ref(stackless_impl_base*& impl);
    static void release_ref(stackless_impl_base* impl);
    static void resume(stackless_impl_base*& impl);

  //private:
    typename stackless_handler_type<Handler, Signature>::type handler_;
    asio::detail::atomic_count ref_count_;
    asio::coroutine coroutine_;
    asio::error_code throw_ec_;
    asio::error_code* result_ec_;
    void* result_value_;
    void (*resume_)(const stackless_context<Handler, Signature>&);
    void (*destroy_)(stackless_impl_base*);
    bool completed_;
  };

  template <typename Handler, typename Signature>
  stackless_impl_base<Handler, Signature>*
  stackless_impl_base<Handler, Signature>::add_ref(
      stackless_impl_base<Handler, Signature>* impl)
  {
    if (impl)
      ++impl->ref_count_;
    return impl;
  }

  template <typename Handler, typename Signature>
  inline stackless_impl_base<Handler, Signature>*
  stackless_impl_base<Handler, Signature>::move_ref(
      stackless_impl_base<Handler, Signature>*& impl)
  {
    stackless_impl_base* tmp = impl;
    impl = 0;
    return tmp;
  }

  template <typename Handler, typename Signature>
  void stackless_impl_base<Handler, Signature>::release_ref(
      stackless_impl_base<Handler, Signature>* impl)
  {
    if (impl)
    {
      if (--impl->ref_count_ == 0)
        impl->destroy_(impl);
    }
  }

  template <typename Handler, typename Signature>
  void stackless_impl_base<Handler, Signature>::resume(
      stackless_impl_base<Handler, Signature>*& impl)
  {
    impl->result_value_ = 0;
    stackless_context<Handler, Signature> ctx = { &impl, &impl->throw_ec_ };
    impl->resume_(ctx);
  }

  template <typename Handler, typename Signature, typename Function>
  class stackless_impl :
    public stackless_impl_base<Handler, Signature>
  {
  public:
    template <typename Handler1, typename Function1>
    stackless_impl(ASIO_MOVE_ARG(Handler1) handler,
        ASIO_MOVE_ARG(Function1) function)
      : stackless_impl_base<Handler, Signature>(
          ASIO_MOVE_CAST(Handler1)(handler),
          &stackless_impl::do_resume,
          &stackless_impl::do_destroy),
        function_(ASIO_MOVE_CAST(Function1)(function))
    {
    }

  private:
    static void do_resume(const stackless_context<Handler, Signature>& ctx)
    {
      stackless_impl_base<Handler, Signature>* base = *ctx.impl_;
      static_cast<stackless_impl*>(base)->function_(ctx);
    }

    static void do_destroy(stackless_impl_base<Handler, Signature>* impl)
    {
      delete static_cast<stackless_impl*>(impl);
    }

    Function function_;
  };

  template <typename Handler, typename Signature, typename T>
  class stackless_handler
  {
  public:
    stackless_handler(const stackless_context<Handler, Signature>& ctx)
      : impl_(stackless_impl_base<Handler, Signature>::move_ref(*ctx.impl_))
    {
      impl_->result_ec_ = ctx.ec_;
    }

    stackless_handler(const stackless_handler& other)
      : impl_(stackless_impl_base<Handler, Signature>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    stackless_handler(stackless_handler&& other)
      : impl_(stackless_impl_base<Handler, Signature>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~stackless_handler()
    {
      stackless_impl_base<Handler, Signature>::release_ref(impl_);
    }

    void operator()(T value)
    {
      *impl_->result_ec_ = asio::error_code();
      if (impl_->result_value_)
        *static_cast<T*>(impl_->result_value_) = value;
      stackless_impl_base<Handler, Signature>::resume(impl_);
    }

    void operator()(asio::error_code ec, T value)
    {
      *impl_->result_ec_ = ec;
      if (impl_->result_value_)
        *static_cast<T*>(impl_->result_value_) = value;
      stackless_impl_base<Handler, Signature>::resume(impl_);
    }

  //private:
    stackless_impl_base<Handler, Signature>* impl_;

  private:
    stackless_handler& operator=(const stackless_handler&);
  };

  template <typename Handler, typename Signature>
  class stackless_handler<Handler, Signature, void>
  {
  public:
    stackless_handler(const stackless_context<Handler, Signature>& ctx)
      : impl_(stackless_impl_base<Handler, Signature>::move_ref(*ctx.impl_))
    {
      impl_->result_ec_ = ctx.ec_;
    }

    stackless_handler(const stackless_handler& other)
      : impl_(stackless_impl_base<Handler, Signature>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    stackless_handler(stackless_handler&& other)
      : impl_(stackless_impl_base<Handler, Signature>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~stackless_handler()
    {
      stackless_impl_base<Handler, Signature>::release_ref(impl_);
    }

    void operator()()
    {
      *impl_->result_ec_ = asio::error_code();
      stackless_impl_base<Handler, Signature>::resume(impl_);
    }

    void operator()(asio::error_code ec)
    {
      *impl_->result_ec_ = ec;
      stackless_impl_base<Handler, Signature>::resume(impl_);
    }

  //private:
    stackless_impl_base<Handler, Signature>* impl_;

  private:
    stackless_handler& operator=(const stackless_handler&);
  };

  template <typename Handler, typename Signature, typename T>
  inline void* asio_handler_allocate(std::size_t size,
      stackless_handler<Handler, Signature, T>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->impl_->handler_);
  }

  template <typename Handler, typename Signature, typename T>
  inline void asio_handler_deallocate(void* pointer, std::size_t size,
      stackless_handler<Handler, Signature, T>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->impl_->handler_);
  }

  template <typename Handler, typename Signature, typename T>
  inline bool asio_handler_is_continuation(
      stackless_handler<Handler, Signature, T>*)
  {
    return true;
  }

  template <typename Function, typename Handler, typename Signature, typename T>
  inline void asio_handler_invoke(Function& function,
      stackless_handler<Handler, Signature, T>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->impl_->handler_);
  }

  template <typename Function, typename Handler, typename Signature, typename T>
  inline void asio_handler_invoke(const Function& function,
      stackless_handler<Handler, Signature, T>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->impl_->handler_);
  }

  template <typename Handler, typename Signature, typename Function>
  struct go_helper
  {
    template <typename Handler1, typename Function1>
    go_helper(ASIO_MOVE_ARG(Handler1) handler,
        ASIO_MOVE_ARG(Function1) function)
      : impl_(new stackless_impl<Handler, Signature, Function>(
            ASIO_MOVE_CAST(Handler1)(handler),
            ASIO_MOVE_CAST(Function1)(function)))
    {
    }

    go_helper(const go_helper& other)
      : impl_(stackless_impl_base<Handler, Signature>::add_ref(other.impl_))
    {
    }

#if defined(ASIO_HAS_MOVE)
    go_helper(go_helper&& other)
      : impl_(stackless_impl_base<Handler, Signature>::move_ref(other.impl_))
    {
    }
#endif // defined(ASIO_HAS_MOVE)

    ~go_helper()
    {
      stackless_impl_base<Handler, Signature>::release_ref(impl_);
    }

    void operator()()
    {
      stackless_impl_base<Handler, Signature>::resume(impl_);
    }

    stackless_impl_base<Handler, Signature>* impl_;

  private:
    go_helper& operator=(const go_helper&);
  };

  inline void default_go_handler() {}

} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename Handler, typename Signature, typename ReturnType>
struct handler_type<stackless_context<Handler, Signature>, ReturnType()>
{
  typedef detail::stackless_handler<Handler, Signature, void> type;
};

template <typename Handler, typename Signature,
    typename ReturnType, typename Arg1>
struct handler_type<stackless_context<Handler, Signature>, ReturnType(Arg1)>
{
  typedef detail::stackless_handler<Handler, Signature, Arg1> type;
};

template <typename Handler, typename Signature, typename ReturnType>
struct handler_type<stackless_context<Handler, Signature>,
    ReturnType(asio::error_code)>
{
  typedef detail::stackless_handler<Handler, Signature, void> type;
};

template <typename Handler, typename Signature,
    typename ReturnType, typename Arg2>
struct handler_type<stackless_context<Handler, Signature>,
    ReturnType(asio::error_code, Arg2)>
{
  typedef detail::stackless_handler<Handler, Signature, Arg2> type;
};

template <typename Handler, typename Signature, typename T>
class async_result<detail::stackless_handler<Handler, Signature, T> >
{
public:
  typedef awaitable<T> type;

  explicit async_result(detail::stackless_handler<Handler, Signature, T>&)
  {
  }

  type get()
  {
    return type();
  }
};

template <typename Handler, typename Signature>
inline coroutine& get_coroutine(stackless_context<Handler, Signature>& c)
{
  return (*c.impl_)->coroutine_;
}

template <typename Handler, typename Signature>
inline coroutine& get_coroutine(stackless_context<Handler, Signature>* c)
{
  return (*c->impl_)->coroutine_;
}

template <typename Handler, typename Signature>
inline const asio::error_code* get_coroutine_error(
    stackless_context<Handler, Signature>& c)
{
  return &(*c.impl_)->throw_ec_;
}

template <typename Handler, typename Signature>
inline const asio::error_code* get_coroutine_error(
    stackless_context<Handler, Signature>* c)
{
  return &(*c->impl_)->throw_ec_;
}

template <typename Handler, typename Signature>
inline void** get_coroutine_async_result(
    stackless_context<Handler, Signature>& c)
{
  return &(*c.impl_)->result_value_;
}

template <typename Handler, typename Signature>
inline void** get_coroutine_async_result(
    stackless_context<Handler, Signature>* c)
{
  return &(*c->impl_)->result_value_;
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Handler, typename Signature>
template <typename... T>
void stackless_context<Handler, Signature>::complete(T&&... args)
{
  if (!(*impl_)->completed_)
  {
    (*impl_)->completed_ = true;
    (*impl_)->handler_(static_cast<T&&>(args)...);
  }
}

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Handler, typename Signature>
void stackless_context<Handler, Signature>::complete()
{
  if (!(*impl_)->completed_)
  {
    (*impl_)->completed_ = true;
    (*impl_)->handler_();
  }
}

// A macro that should expand to:
//   template <typename Handler, typename Signature>
//   template <typename T1, ..., typename Tn>
//   void stackless_context<Handler, Signature>::complete(
//       const T1& x1, ..., const Tn& xn)
//   {
//     if (!(*impl_)->completed_)
//     {
//       (*impl_)->completed_ = true;
//       (*impl_)->handler_(x1, ..., xn);
//     }
//   }
// This macro should only persist within this file.

# define ASIO_PRIVATE_COMPLETE_DEF(n) \
  template <typename Handler, typename Signature> \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void stackless_context<Handler, Signature>::complete( \
      ASIO_VARIADIC_CONSTREF_PARAMS(n)) \
  { \
    if (!(*impl_)->completed_) \
    { \
      (*impl_)->completed_ = true; \
      (*impl_)->handler_(ASIO_VARIADIC_ARGS(n)); \
    } \
  } \
  /**/

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_COMPLETE_DEF)

# undef ASIO_PRIVATE_COMPLETE_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Function>
void go(ASIO_MOVE_ARG(Function) function)
{
  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<void, void(), function_type>(
      &detail::default_go_handler,
      ASIO_MOVE_CAST(Function)(function))();
}

template <typename Handler, typename Function>
ASIO_INITFN_RESULT_TYPE(Handler, void ())
go(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function)
{
  typedef typename remove_const<
    typename remove_reference<Handler>::type>::type
      passed_handler_type;

  typedef typename handler_type<
    Handler, void ()>::type real_handler_type;

  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<passed_handler_type, void (), function_type> helper(
      ASIO_MOVE_CAST(Handler)(handler),
      ASIO_MOVE_CAST(Function)(function));

  async_result<real_handler_type> result(helper.impl_->handler_);

  asio_handler_invoke_helpers::invoke(
      helper, helper.impl_->handler_);

  return result.get();
}

template <typename Signature, typename Handler, typename Function>
ASIO_INITFN_RESULT_TYPE(Handler, Signature)
go(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function)
{
  typedef typename remove_const<
    typename remove_reference<Handler>::type>::type
      passed_handler_type;

  typedef typename handler_type<
    Handler, Signature>::type real_handler_type;

  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<passed_handler_type, Signature, function_type> helper(
      ASIO_MOVE_CAST(Handler)(handler),
      ASIO_MOVE_CAST(Function)(function));

  async_result<real_handler_type> result(helper.impl_->handler_);

  asio_handler_invoke_helpers::invoke(
      helper, helper.impl_->handler_);

  return result.get();
}

template <typename Handler, typename Function>
void go(stackless_context<Handler> ctx,
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
  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<asio::io_service::strand, void(), function_type>(
      strand.wrap(&detail::default_go_handler),
      ASIO_MOVE_CAST(Function)(function))();
}

template <typename Function>
void go(asio::io_service& io_service,
    ASIO_MOVE_ARG(Function) function)
{
  typedef typename remove_const<
    typename remove_reference<Function>::type>::type
      function_type;

  detail::go_helper<asio::io_service, void(), function_type>(
      io_service.wrap(&detail::default_go_handler),
      ASIO_MOVE_CAST(Function)(function))();
}

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_GO_HPP
