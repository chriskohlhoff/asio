//
// impl/defer.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_DEFER_HPP
#define ASIO_IMPL_DEFER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/chain.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... CompletionTokens>
typename detail::chain_result_without_executor<void(CompletionTokens...)>::type
defer(ASIO_MOVE_ARG(CompletionTokens)... tokens)
{
  static_assert(sizeof...(CompletionTokens) > 0,
      "defer() must be called with one or more completion tokens");

  typedef detail::passive_chain<void(), void(CompletionTokens...)> chain_type;

  chain_type chain(ASIO_MOVE_CAST(CompletionTokens)(tokens)...);
  async_result<chain_type> result(chain);

  typename chain_type::executor_type ex(chain.get_executor());
  typename chain_type::allocator_type allocator(chain.get_allocator());
  ex.defer(ASIO_MOVE_CAST(chain_type)(chain), allocator);

  return result.get();
}

template <typename Executor, typename... CompletionTokens>
typename detail::chain_result_with_executor<
  Executor, void(CompletionTokens...)>::type
defer(ASIO_MOVE_ARG(Executor) executor,
    ASIO_MOVE_ARG(CompletionTokens)... tokens)
{
  static_assert(sizeof...(CompletionTokens) > 0,
      "defer() must be called with one or more completion tokens");

  typedef detail::passive_chain<void(), void(CompletionTokens...)> chain_type;

  chain_type chain(ASIO_MOVE_CAST(CompletionTokens)(tokens)...);
  async_result<chain_type> result(chain);

  Executor ex(ASIO_MOVE_CAST(Executor)(executor));
  typename chain_type::allocator_type allocator(chain.get_allocator());
  ex.defer(ASIO_MOVE_CAST(chain_type)(chain), allocator);

  return result.get();
}

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_DEFER_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  typename detail::chain_result_without_executor< \
    void(ASIO_VARIADIC_TARGS(n))>::type \
  defer(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    typedef detail::passive_chain<void(), \
      void(ASIO_VARIADIC_TARGS(n))> chain_type; \
    \
    chain_type chain(ASIO_VARIADIC_MOVE_ARGS(n)); \
    async_result<chain_type> result(chain); \
    \
    typename chain_type::executor_type ex(chain.get_executor()); \
    typename chain_type::allocator_type allocator(chain.get_allocator()); \
    ex.defer(ASIO_MOVE_CAST(chain_type)(chain), allocator); \
    \
    return result.get(); \
  } \
  \
  template <typename Executor, ASIO_VARIADIC_TPARAMS(n)> \
  typename detail::chain_result_with_executor< \
    Executor, void(ASIO_VARIADIC_TARGS(n))>::type \
  defer(ASIO_MOVE_ARG(Executor) executor, \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    typedef detail::passive_chain<void(), \
      void(ASIO_VARIADIC_TARGS(n))> chain_type; \
    \
    chain_type chain(ASIO_VARIADIC_MOVE_ARGS(n)); \
    async_result<chain_type> result(chain); \
    \
    Executor ex(ASIO_MOVE_CAST(Executor)(executor)); \
    typename chain_type::allocator_type allocator(chain.get_allocator()); \
    ex.defer(ASIO_MOVE_CAST(chain_type)(chain), allocator); \
    \
    return result.get(); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_DEFER_DEF)
#undef ASIO_PRIVATE_DEFER_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_DEFER_HPP
