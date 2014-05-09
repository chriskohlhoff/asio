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
#include "asio/detail/type_list.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/handler_type.hpp"
#include "asio/is_executor.hpp"
#include "asio/make_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

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
  typedef typename passive::terminal_handler terminal_handler;
  typedef typename passive::executor executor;
  typedef typename passive::initial_executor initial_executor;

  active_chain(const active_chain& other)
    : passive_(other.passive_),
      work_(other.work_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  active_chain(active_chain&& other)
    : passive_(ASIO_MOVE_CAST(passive)(other.passive_)),
      work_(ASIO_MOVE_CAST(typename executor::work)(other.work_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  explicit active_chain(ASIO_MOVE_ARG(Tn)... tn)
    : passive_(ASIO_MOVE_CAST(Tn)(tn)...),
      work_(passive_.make_current_executor())
  {
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_ACTIVE_CHAIN_CTOR_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  explicit active_chain(ASIO_VARIADIC_MOVE_PARAMS(n)) \
    : passive_(ASIO_VARIADIC_MOVE_ARGS(n)), \
      work_(passive_.make_current_executor()) \
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

  executor make_current_executor() const
  {
    return passive_.make_current_executor();
  }

  initial_executor make_initial_executor() const
  {
    return passive_.make_initial_executor();
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  void operator()(ASIO_MOVE_ARG(Tn)... tn)
  {
    executor ex(passive_.make_current_executor());
    ex.dispatch(
        chain_invoker<Signature, CompletionTokens>(
          ASIO_MOVE_CAST(passive)(passive_),
          ASIO_MOVE_CAST(Tn)(tn)...));
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    executor ex(passive_.make_current_executor());
    ex.dispatch(chain_invoker<Signature, CompletionTokens>(
          ASIO_MOVE_CAST(passive)(passive_)));
  }

#define ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    executor ex(passive_.make_current_executor()); \
    ex.dispatch( \
        chain_invoker<Signature, CompletionTokens>( \
          ASIO_MOVE_CAST(passive)(passive_), \
          ASIO_VARIADIC_MOVE_ARGS(n))); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF)
#undef ASIO_PRIVATE_ACTIVE_CHAIN_CALL_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

private:
  passive passive_;
  typename executor::work work_;
};

template <typename Signature, typename CompletionToken>
class passive_chain<Signature, void(CompletionToken)>
{
public:
  typedef typename handler_type<CompletionToken, Signature>::type handler;
  typedef handler terminal_handler;
  typedef typename make_executor_result<handler>::type executor;
  typedef executor initial_executor;

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

  executor make_current_executor() const
  {
    return make_executor(handler_);
  }

  initial_executor make_initial_executor() const
  {
    return make_executor(handler_);
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
  typedef typename type_list<CompletionTokens>::head head_token;
  typedef typename handler_type<head_token, Signature>::type handler;
  typedef typename continuation_of<handler>::signature tail_signature;
  typedef typename type_list<CompletionTokens>::tail tail_tokens;
  typedef active_chain<tail_signature, tail_tokens> tail;
  typedef typename tail::terminal_handler terminal_handler;
  typedef typename make_executor_result<handler>::type executor;
  typedef typename conditional<is_same<executor, unspecified_executor>::value,
    typename tail::initial_executor, executor>::type initial_executor;

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

  executor make_current_executor() const
  {
    return make_executor(handler_);
  }

  initial_executor make_initial_executor() const
  {
    return make_initial_executor(is_same<executor, unspecified_executor>());
  }

  typename tail::initial_executor make_initial_executor(true_type) const
  {
    return tail_.make_initial_executor();
  }

  executor make_initial_executor(false_type) const
  {
    return make_executor(handler_);
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Tn>
  void operator()(ASIO_MOVE_ARG(Tn)... tn)
  {
    continuation_of<handler>::chain(
        ASIO_MOVE_CAST(handler)(handler_),
        ASIO_MOVE_CAST(tail)(tail_))(
          ASIO_MOVE_CAST(Tn)(tn)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    continuation_of<handler>::chain(
        ASIO_MOVE_CAST(handler)(handler_),
        ASIO_MOVE_CAST(tail)(tail_))();
  }

#define ASIO_PRIVATE_PASSIVE_CHAIN_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    continuation_of<handler>::chain( \
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
        typename type_list<CompletionTokens>::head>::type>::value,
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

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CHAIN_HPP
