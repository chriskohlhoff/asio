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

} // namespace detail
} // namespace experimental

#if !defined(GENERATING_DOCUMENTATION)

template <typename Signature>
class async_result<experimental::lazy_t, Signature>
{
public:
  template <typename Initiation, typename... InitArgs>
  static experimental::detail::lazy_init<Signature, Initiation, InitArgs...>
  initiate(ASIO_MOVE_ARG(Initiation) initiation,
      experimental::lazy_t, ASIO_MOVE_ARG(InitArgs)... args)
  {
    return experimental::detail::lazy_init<Signature, Initiation, InitArgs...>(
        ASIO_MOVE_CAST(Initiation)(initiation),
        ASIO_MOVE_CAST(InitArgs)(args)...);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_IMPL_LAZY_HPP
