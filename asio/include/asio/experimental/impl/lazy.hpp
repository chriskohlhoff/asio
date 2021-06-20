//
// experimental/impl/lazy.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_IMPL_LAZY_HPP
#define ASIO_EXPERIMENTAL_IMPL_LAZY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <tuple>
#include <utility>
#include "asio/async_result.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

template <typename Signature, typename Initiation, typename... InitArgs>
class lazy_init
{
private:
  typename decay<Initiation>::type initiation_;
  typedef std::tuple<typename decay<InitArgs>::type...> init_args_t;
  init_args_t init_args_;

  template <typename CompletionToken, std::size_t... I>
  decltype(auto) invoke(ASIO_MOVE_ARG(CompletionToken) token,
      std::index_sequence<I...>)
  {
    return asio::async_initiate<CompletionToken, Signature>(
        ASIO_MOVE_CAST(typename decay<Initiation>::type)(initiation_),
        token, std::get<I>(ASIO_MOVE_CAST(init_args_t)(init_args_))...);
  }

public:
  template <typename I, typename... A>
  explicit lazy_init(ASIO_MOVE_ARG(I) initiation,
      ASIO_MOVE_ARG(A)... init_args)
    : initiation_(ASIO_MOVE_CAST(I)(initiation)),
      init_args_(ASIO_MOVE_CAST(A)(init_args)...)
  {
  }

  template <ASIO_COMPLETION_TOKEN_FOR(Signature) CompletionToken>
  decltype(auto) operator()(ASIO_MOVE_ARG(CompletionToken) token)
  {
    return this->invoke(
        ASIO_MOVE_CAST(CompletionToken)(token),
        std::index_sequence_for<InitArgs...>());
  }
};

struct lazy_signature_probe {};

template <typename T>
struct lazy_signature_probe_result
{
  typedef T type;
};

template <typename T>
struct lazy_signature
{
  typedef typename decltype(
      declval<T>()(declval<lazy_signature_probe>()))::type type;
};

template <typename HeadSignature, typename Tail>
struct lazy_link_signature;

template <typename R, typename... Args, typename Tail>
struct lazy_link_signature<R(Args...), Tail>
{
  typedef typename decltype(
      declval<Tail>()(declval<Args>()...)(
        declval<lazy_signature_probe>()))::type type;
};

template <typename Handler, typename Tail>
class lazy_link_handler
{
public:
  template <typename H, typename T>
  explicit lazy_link_handler(
      ASIO_MOVE_ARG(H) handler, ASIO_MOVE_ARG(T) tail)
    : handler_(ASIO_MOVE_CAST(H)(handler)),
      tail_(ASIO_MOVE_CAST(T)(tail))
  {
  }

  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)... args)
  {
    ASIO_MOVE_OR_LVALUE(Tail)(tail_)(
        ASIO_MOVE_CAST(Args)(args)...)(
          ASIO_MOVE_OR_LVALUE(Handler)(handler_));
  }

//private:
  Handler handler_;
  Tail tail_;
};

struct lazy_link_initiate
{
  template <typename Handler, typename Head, typename Tail>
  void operator()(ASIO_MOVE_ARG(Handler) handler,
      Head head, ASIO_MOVE_ARG(Tail) tail)
  {
    ASIO_MOVE_OR_LVALUE(Head)(head)(
        lazy_link_handler<typename decay<Handler>::type,
          typename decay<Tail>::type>(
            ASIO_MOVE_CAST(Handler)(handler),
            ASIO_MOVE_CAST(Tail)(tail)));
  }
};

template <typename Head, typename Tail>
class lazy_link
{
public:
  typedef typename lazy_link_signature<
    typename lazy_signature<Head>::type, Tail>::type
      signature;

  template <typename H, typename T>
  explicit lazy_link(ASIO_MOVE_ARG(H) head, ASIO_MOVE_ARG(T) tail)
    : head_(ASIO_MOVE_CAST(H)(head)),
      tail_(ASIO_MOVE_CAST(T)(tail))
  {
  }

  template <ASIO_COMPLETION_TOKEN_FOR(signature) CompletionToken>
  decltype(auto) operator()(ASIO_MOVE_ARG(CompletionToken) token)
  {
    return asio::async_initiate<CompletionToken, signature>(
        lazy_link_initiate(), token,
        ASIO_MOVE_OR_LVALUE(Head)(head_),
        ASIO_MOVE_OR_LVALUE(Tail)(tail_));
  }

private:
  Head head_;
  Tail tail_;
};

} // namespace detail
} // namespace experimental

#if !defined(GENERATING_DOCUMENTATION)

template <typename Signature>
class async_result<experimental::lazy_t, Signature>
{
public:
  template <typename Initiation, typename... InitArgs>
  static experimental::lazy_operation<
    experimental::detail::lazy_init<Signature, Initiation, InitArgs...> >
  initiate(ASIO_MOVE_ARG(Initiation) initiation,
      experimental::lazy_t, ASIO_MOVE_ARG(InitArgs)... args)
  {
    return experimental::lazy_operation<
      experimental::detail::lazy_init<Signature, Initiation, InitArgs...> >(
        experimental::detail::lazy_init<Signature, Initiation, InitArgs...>(
          ASIO_MOVE_CAST(Initiation)(initiation),
          ASIO_MOVE_CAST(InitArgs)(args)...));
    }
};

template <typename R, typename... Args>
class async_result<experimental::detail::lazy_signature_probe, R(Args...)>
{
public:
  typedef experimental::detail::lazy_signature_probe_result<void(Args...)>
    return_type;

  template <typename Initiation, typename... InitArgs>
  static return_type initiate(
      ASIO_MOVE_ARG(Initiation),
      experimental::detail::lazy_signature_probe,
      ASIO_MOVE_ARG(InitArgs)...)
  {
    return return_type{};
  }
};

template <typename Tail, typename R, typename... Args>
  requires (experimental::is_lazy_operation<std::invoke_result_t<Tail, Args...>>::value)
class asio::async_result<Tail, R(Args...)>
{
public:
  template <typename Initiation, typename... InitArgs>
  static auto initiate(ASIO_MOVE_ARG(Initiation) initiation,
      Tail tail, ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    return experimental::lazy_operation<
      experimental::detail::lazy_link<
        experimental::detail::lazy_init<R(Args...), Initiation, InitArgs...>,
        Tail> >(
          experimental::detail::lazy_link<
            experimental::detail::lazy_init<R(Args...), Initiation, InitArgs...>,
            Tail>(
              experimental::detail::lazy_init<R(Args...), Initiation, InitArgs...>(
                ASIO_MOVE_CAST(Initiation)(initiation),
                ASIO_MOVE_CAST(InitArgs)(init_args)...),
              ASIO_MOVE_CAST(Tail)(tail)));
  }
};

template <template <typename, typename> class Associator,
    typename Handler, typename Tail, typename DefaultCandidate>
struct associator<Associator,
    experimental::detail::lazy_link_handler<Handler, Tail>, DefaultCandidate>
  : Associator<Handler, DefaultCandidate>
{
  static typename Associator<Handler, DefaultCandidate>::type get(
      const experimental::detail::lazy_link_handler<Handler, Tail>& h,
      const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
  {
    return Associator<Handler, DefaultCandidate>::get(h.handler_, c);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_IMPL_LAZY_HPP
