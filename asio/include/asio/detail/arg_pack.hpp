//
// detail/arg_pack.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ARG_PACK_HPP
#define ASIO_DETAIL_ARG_PACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_list.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Signature>
class arg_pack
{
public:
  arg_pack(const arg_pack& other)
    : value_(other.value_),
      next_(other.next_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  arg_pack(arg_pack&& other)
    : value_(ASIO_MOVE_CAST(value_type)(other.value_)),
      next_(ASIO_MOVE_CAST(next_type)(other.next_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename T0, typename... Tn>
  explicit arg_pack(ASIO_MOVE_ARG(T0) t0,
      ASIO_MOVE_ARG(Tn)... tn)
    : value_(ASIO_MOVE_CAST(T0)(t0)),
      next_(ASIO_MOVE_CAST(Tn)(tn)...)
  {
  }

  template <typename Function, typename... Tn>
  void invoke(Function& f, ASIO_MOVE_ARG(Tn)... tn)
  {
    next_.invoke(f, ASIO_MOVE_CAST(Tn)(tn)...,
        ASIO_MOVE_CAST(value_type)(value_));
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename T0>
  arg_pack(ASIO_MOVE_ARG(T0) t0)
    : value_(ASIO_MOVE_CAST(T0)(t0))
  {
  }

#define ASIO_PRIVATE_ARG_PACK_CTOR_DEF(n) \
  template <typename T0, ASIO_VARIADIC_TPARAMS(n)> \
  explicit arg_pack(ASIO_MOVE_ARG(T0) t0, \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
    : value_(ASIO_MOVE_CAST(T0)(t0)), \
      next_(ASIO_VARIADIC_MOVE_ARGS(n)) \
  { \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ARG_PACK_CTOR_DEF)
#undef ASIO_PRIVATE_ARG_PACK_CTOR_DEF

  template <typename Function>
  void invoke(Function& f)
  {
    next_.invoke(f, ASIO_MOVE_CAST(value_type)(value_));
  }

#define ASIO_PRIVATE_ARG_PACK_INVOKE_DEF(n) \
  template <typename Function, ASIO_VARIADIC_TPARAMS(n)> \
  void invoke(Function& f, ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    next_.invoke(f, ASIO_VARIADIC_MOVE_ARGS(n), \
        ASIO_MOVE_CAST(value_type)(value_)); \
  }
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ARG_PACK_INVOKE_DEF)
#undef ASIO_PRIVATE_ARG_PACK_INVOKE_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

private:
  typedef typename decay<typename type_list<Signature>::head>::type value_type;
  value_type value_;
  typedef arg_pack<typename type_list<Signature>::tail> next_type;
  next_type next_;
};

template <typename Result>
class arg_pack<Result()>
{
public:
#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename Function, typename... Tn>
  void invoke(Function& f, ASIO_MOVE_ARG(Tn)... tn)
  {
    f(ASIO_MOVE_CAST(Tn)(tn)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename Function>
  void invoke(Function& f)
  {
    f();
  }

#define ASIO_PRIVATE_ARG_PACK_INVOKE_DEF(n) \
  template <typename Function, ASIO_VARIADIC_TPARAMS(n)> \
  void invoke(Function& f, ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    f(ASIO_VARIADIC_MOVE_ARGS(n)); \
  }
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ARG_PACK_INVOKE_DEF)
#undef ASIO_PRIVATE_ARG_PACK_INVOKE_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_ARG_PACK_HPP
