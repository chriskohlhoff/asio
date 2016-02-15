//
// impl/redirect_error.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_REDIRECT_ERROR_HPP
#define ASIO_IMPL_REDIRECT_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_executor.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/handler_type.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

  // Class to adapt a redirect_error_t as a completion handler.
  template <typename Handler>
  class redirect_error_handler
  {
  public:
    template <typename CompletionToken>
    redirect_error_handler(redirect_error_t<CompletionToken> e)
      : ec_(e.ec_),
        handler_(ASIO_MOVE_CAST(CompletionToken)(e.token_))
    {
    }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

    template <typename... Args>
    void operator()(ASIO_MOVE_ARG(Args)... args)
    {
      handler_(ASIO_MOVE_CAST(Args)(args)...);
    }

    template <typename... Args>
    void operator()(const asio::error_code& ec,
        ASIO_MOVE_ARG(Args)... args)
    {
      ec_ = ec;
      handler_(ASIO_MOVE_CAST(Args)(args)...);
    }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

    void operator()()
    {
      handler_();
    }

    void operator()(const asio::error_code& ec)
    {
      ec_ = ec;
      handler_();
    }

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
    template <ASIO_VARIADIC_TPARAMS(n)> \
    void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
    { \
      handler_(ASIO_VARIADIC_MOVE_ARGS(n)); \
    } \
    \
    template <ASIO_VARIADIC_TPARAMS(n)> \
    void operator()(const asio::error_code& ec, \
        ASIO_VARIADIC_MOVE_PARAMS(n)) \
    { \
      ec_ = ec; \
      handler_(ASIO_VARIADIC_MOVE_ARGS(n)); \
    } \
    /**/
    ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  //private:
    asio::error_code& ec_;
    Handler handler_;
  };

  template <typename Handler>
  inline void* asio_handler_allocate(std::size_t size,
      redirect_error_handler<Handler>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, this_handler->handler_);
  }

  template <typename Handler>
  inline void* asio_handler_deallocate(std::size_t size,
      redirect_error_handler<Handler>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, this_handler->handler_);
  }

  template <typename Handler>
  inline bool asio_handler_is_continuation(
      redirect_error_handler<Handler>* this_handler)
  {
    return asio_handler_cont_helpers::is_continuation(
          this_handler->handler_);
  }

  template <typename Function, typename Handler>
  inline void asio_handler_invoke(Function& function,
      redirect_error_handler<Handler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }

  template <typename Function, typename Handler>
  inline void asio_handler_invoke(const Function& function,
      redirect_error_handler<Handler>* this_handler)
  {
    asio_handler_invoke_helpers::invoke(
        function, this_handler->handler_);
  }
} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename CompletionToken, typename Signature>
struct handler_type<redirect_error_t<CompletionToken>, Signature>
{
  typedef detail::redirect_error_handler<
    typename handler_type<CompletionToken,
      Signature>::type> type;
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename CompletionToken, typename R, typename... Args>
struct handler_type<redirect_error_t<CompletionToken>,
    R(asio::error_code, Args...)>
{
  typedef detail::redirect_error_handler<
    typename handler_type<CompletionToken,
      R(Args...)>::type> type;
};

template <typename CompletionToken, typename R, typename... Args>
struct handler_type<redirect_error_t<CompletionToken>,
    R(const asio::error_code&, Args...)>
{
  typedef detail::redirect_error_handler<
    typename handler_type<CompletionToken,
      R(Args...)>::type> type;
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename CompletionToken, typename R>
struct handler_type<redirect_error_t<CompletionToken>,
    R(asio::error_code)>
{
  typedef detail::redirect_error_handler<
    typename handler_type<CompletionToken,
      R()>::type> type;
};

template <typename CompletionToken, typename R>
struct handler_type<redirect_error_t<CompletionToken>,
    R(const asio::error_code&)>
{
  typedef detail::redirect_error_handler<
    typename handler_type<CompletionToken,
      R()>::type> type;
};

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
  template <typename CompletionToken, \
      typename R, ASIO_VARIADIC_TPARAMS(n)> \
  struct handler_type<redirect_error_t<CompletionToken>, \
      R(asio::error_code, ASIO_VARIADIC_TARGS(n))> \
  { \
    typedef detail::redirect_error_handler< \
      typename handler_type<CompletionToken, \
        R(ASIO_VARIADIC_TARGS(n))>::type> type; \
  }; \
  \
  template <typename CompletionToken, \
      typename R, ASIO_VARIADIC_TPARAMS(n)> \
  struct handler_type<redirect_error_t<CompletionToken>, \
      R(const asio::error_code&, ASIO_VARIADIC_TARGS(n))> \
  { \
    typedef detail::redirect_error_handler< \
      typename handler_type<CompletionToken, \
        R(ASIO_VARIADIC_TARGS(n))>::type> type; \
  }; \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Handler>
struct async_result<detail::redirect_error_handler<Handler> >
  : async_result<Handler>
{
  explicit async_result(detail::redirect_error_handler<Handler>& h)
    : async_result<Handler>(h.handler_)
  {
  }
};

template <typename Handler, typename Executor>
struct associated_executor<detail::redirect_error_handler<Handler>, Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const detail::redirect_error_handler<Handler>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

template <typename Handler, typename Allocator>
struct associated_allocator<detail::redirect_error_handler<Handler>, Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const detail::redirect_error_handler<Handler>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_REDIRECT_ERROR_HPP
