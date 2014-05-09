//
// impl/chain.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_CHAIN_HPP
#define ASIO_IMPL_CHAIN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/chain.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... CompletionTokens>
detail::active_chain<void(), void(CompletionTokens...)>
chain(ASIO_MOVE_ARG(CompletionTokens)... tokens)
{
  static_assert(sizeof...(CompletionTokens) > 0,
      "chain() must be called with one or more completion tokens");

  return detail::active_chain<void(), void(CompletionTokens...)>(
      ASIO_MOVE_CAST(CompletionTokens)(tokens)...);
}

template <typename Signature, typename... CompletionTokens>
detail::active_chain<Signature, void(CompletionTokens...)>
chain(ASIO_MOVE_ARG(CompletionTokens)... tokens)
{
  static_assert(sizeof...(CompletionTokens) > 0,
      "chain() must be called with one or more completion tokens");

  return detail::active_chain<Signature, void(CompletionTokens...)>(
      ASIO_MOVE_CAST(CompletionTokens)(tokens)...);
}

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_CHAIN_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  detail::active_chain<void(), void(ASIO_VARIADIC_TARGS(n))> \
  chain(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    return detail::active_chain< \
      void(), void(ASIO_VARIADIC_TARGS(n))>( \
        ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  \
  template <typename Signature, ASIO_VARIADIC_TPARAMS(n)> \
  detail::active_chain<Signature, void(ASIO_VARIADIC_TARGS(n))> \
  chain(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    return detail::active_chain< \
      Signature, void(ASIO_VARIADIC_TARGS(n))>( \
        ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CHAIN_DEF)
#undef ASIO_PRIVATE_CHAIN_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_CHAIN_HPP
