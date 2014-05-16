//
// detail/signature.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SIGNATURE_HPP
#define ASIO_DETAIL_SIGNATURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Signature>
struct signature;

template <typename R>
struct signature<R()>
{
  typedef R type();
  typedef R result;
};

template <typename R, typename T0>
struct signature<R(T0)>
{
  typedef R type(T0);
  typedef R result;
  typedef T0 head;
  typedef R tail();
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R, typename T0, typename... Tn>
struct signature<R(T0, Tn...)>
{
  typedef R type(T0, Tn...);
  typedef R result;
  typedef T0 head;
  typedef R tail(Tn...);
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_SIGNATURE_DEF(n) \
template <typename R, typename T0, ASIO_VARIADIC_TPARAMS(n)> \
struct signature<R(T0, ASIO_VARIADIC_TARGS(n))> \
{ \
  typedef R type(T0, ASIO_VARIADIC_TARGS(n)); \
  typedef R result; \
  typedef T0 head; \
  typedef R tail(ASIO_VARIADIC_TARGS(n)); \
}; \
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_SIGNATURE_DEF)
#undef ASIO_PRIVATE_SIGNATURE_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Signature1, typename Signature2>
struct signature_cat;

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R1, typename... Tn, typename R2, typename... Un>
struct signature_cat<R1(Tn...), R2(Un...)>
  : signature<R1(Tn..., Un...)> {};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R1, typename R2>
struct signature_cat<R1(), R2()>
  : signature<R1()> {};

template <typename R1, typename Signature2>
struct signature_cat<R1(), Signature2>
  : signature_cat<
      R1(typename signature<Signature2>::head),
      typename signature<Signature2>::tail> {};

template <typename R1, typename T0, typename R2>
struct signature_cat<R1(T0), R2()>
  : signature<R1(T0)> {};

template <typename R1, typename T0, typename Signature2>
struct signature_cat<R1(T0), Signature2>
  : signature_cat<
      R1(T0, typename signature<Signature2>::head),
      typename signature<Signature2>::tail> {};

#define ASIO_PRIVATE_SIGNATURE_CAT_DEF(n) \
template <typename R1, typename T0, \
  ASIO_VARIADIC_TPARAMS(n), typename R2> \
struct signature_cat<R1(T0, ASIO_VARIADIC_TARGS(n)), R2()> \
  : signature<R1(T0, ASIO_VARIADIC_TARGS(n))> {}; \
\
template <typename R1, typename T0, \
  ASIO_VARIADIC_TPARAMS(n), typename Signature2> \
struct signature_cat<R1(T0, ASIO_VARIADIC_TARGS(n)), Signature2> \
  : signature_cat< \
      R1(T0, ASIO_VARIADIC_TARGS(n), \
          typename signature<Signature2>::head), \
      typename signature<Signature2>::tail> {};
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_SIGNATURE_CAT_DEF)
#undef ASIO_PRIVATE_SIGNATURE_CAT_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SIGNATURE_HPP
