//
// detail/chain.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CHAIN_HPP
#define ASIO_DETAIL_CHAIN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/continuation_of.hpp"
#include "asio/detail/arg_pack.hpp"
#include "asio/detail/signature.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/executor_work.hpp"
#include "asio/get_allocator.hpp"
#include "asio/get_executor.hpp"
#include "asio/handler_type.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
struct is_unspecified_allocator
  : false_type {};

template <typename T>
struct is_unspecified_allocator<unspecified_allocator<T> >
  : true_type {};

template <typename T>
inline typename get_allocator_type<T>::type get_allocator_helper(const T& t)
{
  return get_allocator(t);
}

template <typename T>
inline typename get_executor_type<T>::type get_executor_helper(const T& t)
{
  return get_executor(t);
}

template <typename Signature, typename CompletionTokens>
class passive_chain;

template <typename Signature, typename CompletionTokens>
class chain_invoker
{
public:
  typedef passive_chain<Signature, CompletionTokens> passive;

  chain_invoker(const chain_invoker& other)
    : passive_(other.passive_),
      args_(other.args_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  chain_invoker(chain_invoker&& other)
    : passive_(ASIO_MOVE_CAST(passive)(other.passive_)),
      args_(ASIO_MOVE_CAST(arg_pack<Signature>)(other.args_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename C, typename... Tn>
  explicit chain_invoker(ASIO_MOVE_ARG(C) c,
      ASIO_MOVE_ARG(Tn)... tn)
    : passive_(ASIO_MOVE_CAST(C)(c)),
      args_(ASIO_MOVE_CAST(Tn)(tn)...)
  {
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename C>
  explicit chain_invoker(ASIO_MOVE_ARG(C) c)
    : passive_(ASIO_MOVE_CAST(C)(c))
  {
  }

#define ASIO_PRIVATE_CHAIN_INVOKER_CTOR_DEF(n) \
  template <typename C, ASIO_VARIADIC_TPARAMS(n)> \
  explicit chain_invoker(ASIO_MOVE_ARG(C) c, \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
    : passive_(ASIO_MOVE_CAST(C)(c)), \
      args_(ASIO_VARIADIC_MOVE_ARGS(n)) \
  { \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CHAIN_INVOKER_CTOR_DEF)
#undef ASIO_PRIVATE_CHAIN_INVOKER_CTOR_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    args_.invoke(passive_);
  }

private:
  passive_chain<Signature, CompletionTokens> passive_;
  arg_pack<Signature> args_;
};

template <typename Signature, typename CompletionTokens>
class active_chain
{
public:
  typedef passive_chain<Signature, CompletionTokens> passive;
  typedef typename passive::handler handler;
  typedef typename passive::tail_signature tail_signature;
  typedef typename passive::terminal_handler terminal_handler;
  typedef typename passive::handler_executor handler_executor;
  typedef typename passive::executor_type executor_type;
  typedef typename passive::allocator_type allocator_type;

  active_chain(const active_chain& other)
    : passive_(other.passive_),
      work_(other.work_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  active_chain(active_chain&& other)
    : passive_(ASIO_MOVE_CAST(passive)(other.passive_)),
      work_(ASIO_MOVE_CAST(executor_work<handler_executor>)(other.work_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  active_chain(ASIO_MOVE_ARG(passive) p,
      ASIO_MOVE_ARG(executor_work<handler_executor>) w)
    : passive_(ASIO_MOVE_CAST(passive)(p)),
      work_(ASIO_MOVE_CAST(executor_work<handler_executor>)(w))
  {
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  explicit active_chain(ASIO_MOVE_ARG(Tn)... tn)
    : passive_(ASIO_MOVE_CAST(Tn)(tn)...),
      work_(passive_.get_handler_executor())
  {
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_ACTIVE_CHAIN_CTOR_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  explicit active_chain(ASIO_VARIADIC_MOVE_PARAMS(n)) \
    : passive_(ASIO_VARIADIC_MOVE_ARGS(n)), \
      work_(passive_.get_handler_executor()) \
  { \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ACTIVE_CHAIN_CTOR_DEF)
#undef ASIO_PRIVATE_ACTIVE_CHAIN_CTOR_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  terminal_handler& get_terminal_handler()
  {
    return passive_.get_terminal_handler();
  }

  handler_executor get_handler_executor() const
  {
    return passive_.get_handler_executor();
  }

  executor_type get_executor() const
  {
    return passive_.get_executor();
  }

  allocator_type get_allocator() const
  {
    return passive_.get_allocator();
  }

  template <typename T>
  active_chain<Signature,
    typename signature_cat<CompletionTokens, void(T)>::type>
  chain(ASIO_MOVE_ARG(T) t)
  {
    return active_chain<Signature,
      typename signature_cat<CompletionTokens, void(T)>::type>(
        passive_.chain(ASIO_MOVE_CAST(T)(t)),
        ASIO_MOVE_CAST(executor_work<handler_executor>)(work_));
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  void operator()(ASIO_MOVE_ARG(Tn)... tn)
  {
    allocator_type a(passive_.get_allocator());
    handler_executor ex(passive_.get_handler_executor());
    ex.dispatch(
        chain_invoker<Signature, CompletionTokens>(
          ASIO_MOVE_CAST(passive)(passive_),
          ASIO_MOVE_CAST(Tn)(tn)...), a);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    allocator_type a(passive_.get_allocator());
    handler_executor ex(passive_.get_handler_executor());
    ex.dispatch(chain_invoker<Signature, CompletionTokens>(
          ASIO_MOVE_CAST(passive)(passive_)), a);
  }

#define ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    allocator_type a(passive_.get_allocator()); \
    handler_executor ex(passive_.get_handler_executor()); \
    ex.dispatch( \
        chain_invoker<Signature, CompletionTokens>( \
          ASIO_MOVE_CAST(passive)(passive_), \
          ASIO_VARIADIC_MOVE_ARGS(n)), a); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF)
#undef ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

private:
  passive passive_;
  executor_work<handler_executor> work_;
};

template <typename Signature, typename CompletionToken>
class passive_chain<Signature, void(CompletionToken)>
{
public:
  typedef typename handler_type<CompletionToken, Signature>::type handler;
  typedef typename signature_cat<handler(), Signature>::type result_of_arg;
  typedef typename continuation_of<result_of_arg>::signature tail_signature;
  typedef handler terminal_handler;
  typedef typename get_executor_type<handler>::type handler_executor;
  typedef handler_executor executor_type;
  typedef typename get_allocator_type<handler>::type handler_allocator;
  typedef handler_allocator allocator_type;

  passive_chain(const passive_chain& other)
    : handler_(other.handler_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  passive_chain(passive_chain&& other)
    : handler_(ASIO_MOVE_CAST(handler)(other.handler_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  template <typename T>
  explicit passive_chain(ASIO_MOVE_ARG(T) token)
    : handler_(ASIO_MOVE_CAST(T)(token))
  {
  }

  terminal_handler& get_terminal_handler()
  {
    return handler_;
  }

  handler_executor get_handler_executor() const
  {
    return get_executor_helper(handler_);
  }

  executor_type get_executor() const
  {
    return get_executor_helper(handler_);
  }

  allocator_type get_allocator() const
  {
    return get_allocator_helper(handler_);
  }

  template <typename T>
  passive_chain<Signature, void(CompletionToken, T)> chain(ASIO_MOVE_ARG(T) t)
  {
    return passive_chain<Signature, void(CompletionToken, T)>(
        ASIO_MOVE_CAST(handler)(handler_),
        ASIO_MOVE_CAST(T)(t));
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  void operator()(ASIO_MOVE_ARG(Tn)... tn)
  {
    handler_(ASIO_MOVE_CAST(Tn)(tn)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    handler_();
  }

#define ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    handler_(ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF)
#undef ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

protected:
  handler handler_;
};

template <typename Signature, typename CompletionTokens>
class passive_chain
{
public:
  typedef typename signature<CompletionTokens>::head head_token;
  typedef typename handler_type<head_token, Signature>::type handler;
  typedef typename signature_cat<handler(), Signature>::type result_of_arg;
  typedef typename continuation_of<result_of_arg>::signature tail_signature;
  typedef typename signature<CompletionTokens>::tail tail_tokens;
  typedef active_chain<tail_signature, tail_tokens> tail;
  typedef typename tail::terminal_handler terminal_handler;
  typedef typename get_executor_type<handler>::type handler_executor;
  typedef typename conditional<
    is_same<handler_executor, unspecified_executor>::value,
    typename tail::executor_type, handler_executor>::type executor_type;
  typedef typename get_allocator_type<handler>::type handler_allocator;
  typedef typename conditional<
    is_unspecified_allocator<handler_allocator>::value,
    typename tail::allocator_type, handler_allocator>::type allocator_type;

  passive_chain(const passive_chain& other)
    : handler_(other.handler_),
      tail_(other.tail_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  passive_chain(passive_chain&& other)
    : handler_(ASIO_MOVE_CAST(handler)(other.handler_)),
      tail_(ASIO_MOVE_CAST(tail)(other.tail_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename T0, typename... Tn>
  explicit passive_chain(ASIO_MOVE_ARG(T0) t0,
      ASIO_MOVE_ARG(Tn)... tn)
    : handler_(ASIO_MOVE_CAST(T0)(t0)),
      tail_(ASIO_MOVE_CAST(Tn)(tn)...)
  {
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_CHAIN_CTOR_DEF(n) \
  template <typename T0, ASIO_VARIADIC_TPARAMS(n)> \
  explicit passive_chain(ASIO_MOVE_ARG(T0) t0, \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
    : handler_(ASIO_MOVE_CAST(T0)(t0)), \
      tail_(ASIO_VARIADIC_MOVE_ARGS(n)) \
  { \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CHAIN_CTOR_DEF)
#undef ASIO_PRIVATE_CHAIN_CTOR_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  terminal_handler& get_terminal_handler()
  {
    return tail_.get_terminal_handler();
  }

  handler_executor get_handler_executor() const
  {
    return get_executor_helper(handler_);
  }

  executor_type get_executor() const
  {
    return this->get_executor(is_same<handler_executor, unspecified_executor>());
  }

  typename tail::executor_type get_executor(true_type) const
  {
    return tail_.get_executor();
  }

  handler_executor get_executor(false_type) const
  {
    return get_executor_helper(handler_);
  }

  allocator_type get_allocator() const
  {
    return this->get_allocator(
        is_unspecified_allocator<handler_allocator>());
  }

  typename tail::allocator_type get_allocator(true_type) const
  {
    return tail_.get_allocator();
  }

  handler_allocator get_allocator(false_type) const
  {
    return get_allocator_helper(handler_);
  }

  template <typename T>
  passive_chain<Signature,
    typename signature_cat<CompletionTokens, void(T)>::type>
  chain(ASIO_MOVE_ARG(T) t)
  {
    return passive_chain<Signature,
      typename signature_cat<CompletionTokens, void(T)>::type>(
        ASIO_MOVE_CAST(handler)(handler_),
        tail_.chain(ASIO_MOVE_CAST(T)(t)));
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  void operator()(ASIO_MOVE_ARG(Tn)... tn)
  {
    continuation_of<result_of_arg>::chain(
        ASIO_MOVE_CAST(handler)(handler_),
        ASIO_MOVE_CAST(tail)(tail_))(
          ASIO_MOVE_CAST(Tn)(tn)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    continuation_of<result_of_arg>::chain(
        ASIO_MOVE_CAST(handler)(handler_),
        ASIO_MOVE_CAST(tail)(tail_))();
  }

#define ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    continuation_of<result_of_arg>::chain( \
        ASIO_MOVE_CAST(handler)(handler_), \
        ASIO_MOVE_CAST(tail)(tail_))( \
          ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF)
#undef ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

protected:
  handler handler_;
  tail tail_;
};

template <typename CompletionTokens>
struct chain_result
{
  typedef typename async_result<passive_chain<
    void(), CompletionTokens> >::type type;
};

struct chain_no_result {};

template <typename CompletionTokens>
struct chain_result_without_executor
  : conditional<
      is_executor<typename decay<
        typename signature<CompletionTokens>::head>::type>::value,
      chain_no_result, chain_result<CompletionTokens> >::type
{
};

template <typename Executor, typename CompletionTokens>
struct chain_result_with_executor
  : conditional<
      is_executor<typename decay<Executor>::type>::value,
      chain_result<CompletionTokens>, chain_no_result>::type
{
};

} // namespace detail

template <typename Signature, typename CompletionTokens>
class async_result<detail::passive_chain<Signature, CompletionTokens> >
  : public async_result<
      typename detail::passive_chain<Signature,
        CompletionTokens>::terminal_handler>
{
public:
  explicit async_result(
      typename detail::passive_chain<Signature, CompletionTokens>& h)
    : async_result<
        typename detail::passive_chain<Signature,
          CompletionTokens>::terminal_handler>(
            h.get_terminal_handler())
  {
  }
};

template <typename Signature, typename CompletionTokens>
class async_result<detail::active_chain<Signature, CompletionTokens> >
  : public async_result<
      typename detail::active_chain<Signature,
        CompletionTokens>::terminal_handler>
{
public:
  explicit async_result(
      typename detail::active_chain<Signature, CompletionTokens>& h)
    : async_result<
        typename detail::active_chain<Signature,
          CompletionTokens>::terminal_handler>(
            h.get_terminal_handler())
  {
  }
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Signature, typename CompletionTokens, typename... Args>
struct continuation_of<
  detail::passive_chain<Signature, CompletionTokens>(Args...)>
{
  typedef typename detail::passive_chain<
    Signature, CompletionTokens>::tail_signature signature;

  template <typename T>
  struct chain_type
  {
    typedef detail::passive_chain<Signature,
      typename detail::signature_cat<CompletionTokens, void(T)>::type> type;
  };

  template <typename T>
  static typename chain_type<T>::type
  chain(detail::passive_chain<Signature, CompletionTokens> c,
      ASIO_MOVE_ARG(T) t)
  {
    return c.chain(t);
  }
};

template <typename Signature, typename CompletionTokens, typename... Args>
struct continuation_of<
  detail::active_chain<Signature, CompletionTokens>(Args...)>
{
  typedef typename detail::active_chain<
    Signature, CompletionTokens>::tail_signature signature;

  template <typename T>
  struct chain_type
  {
    typedef detail::active_chain<Signature,
      typename detail::signature_cat<CompletionTokens, void(T)>::type> type;
  };

  template <typename T>
  static detail::active_chain<Signature,
    typename detail::signature_cat<CompletionTokens, void(T)>::type>
  chain(detail::active_chain<Signature, CompletionTokens> c,
      ASIO_MOVE_ARG(T) t)
  {
    return c.chain(ASIO_MOVE_CAST(T)(t));
  }
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Signature, typename CompletionTokens>
struct continuation_of<
  detail::passive_chain<Signature, CompletionTokens>()>
{
  typedef typename detail::passive_chain<
    Signature, CompletionTokens>::tail_signature signature;

  template <typename T>
  struct chain_type
  {
    typedef detail::passive_chain<Signature,
      typename detail::signature_cat<CompletionTokens, void(T)>::type> type;
  };

  template <typename T>
  static typename chain_type<T>::type
  chain(detail::passive_chain<Signature, CompletionTokens> c,
      ASIO_MOVE_ARG(T) t)
  {
    return c.chain(t);
  }
};

template <typename Signature, typename CompletionTokens>
struct continuation_of<
  detail::active_chain<Signature, CompletionTokens>()>
{
  typedef typename detail::active_chain<
    Signature, CompletionTokens>::tail_signature signature;

  template <typename T>
  struct chain_type
  {
    typedef detail::active_chain<Signature,
      typename detail::signature_cat<CompletionTokens, void(T)>::type> type;
  };

  template <typename T>
  static detail::active_chain<Signature,
    typename detail::signature_cat<CompletionTokens, void(T)>::type>
  chain(detail::active_chain<Signature, CompletionTokens> c,
      ASIO_MOVE_ARG(T) t)
  {
    return c.chain(ASIO_MOVE_CAST(T)(t));
  }
};

# define ASIO_PRIVATE_CONTINUATION_OF_DEF(n) \
  template <typename Signature, typename CompletionTokens, \
    ASIO_VARIADIC_TPARAMS(n)> \
  struct continuation_of< \
    detail::passive_chain<Signature, CompletionTokens>( \
        ASIO_VARIADIC_TARGS(n))> \
  { \
    typedef typename detail::passive_chain< \
      Signature, CompletionTokens>::tail_signature signature; \
  \
    template <typename T> \
    struct chain_type \
    { \
      typedef detail::passive_chain<Signature, \
        typename detail::signature_cat<CompletionTokens, void(T)>::type> type; \
    }; \
  \
    template <typename T> \
    static typename chain_type<T>::type \
    chain(detail::passive_chain<Signature, CompletionTokens> c, \
        ASIO_MOVE_ARG(T) t) \
    { \
      return c.chain(t); \
    } \
  }; \
  \
  template <typename Signature, typename CompletionTokens, \
    ASIO_VARIADIC_TPARAMS(n)> \
  struct continuation_of< \
    detail::active_chain<Signature, CompletionTokens>( \
        ASIO_VARIADIC_TARGS(n))> \
  { \
    typedef typename detail::active_chain< \
      Signature, CompletionTokens>::tail_signature signature; \
  \
    template <typename T> \
    struct chain_type \
    { \
      typedef detail::active_chain<Signature, \
        typename detail::signature_cat<CompletionTokens, void(T)>::type> type; \
    }; \
  \
    template <typename T> \
    static detail::active_chain<Signature, \
      typename detail::signature_cat<CompletionTokens, void(T)>::type> \
    chain(detail::active_chain<Signature, CompletionTokens> c, \
        ASIO_MOVE_ARG(T) t) \
    { \
      return c.chain(ASIO_MOVE_CAST(T)(t)); \
    } \
  }; \
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CONTINUATION_OF_DEF)
#undef ASIO_PRIVATE_CONTINUATION_OF_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CHAIN_HPP
