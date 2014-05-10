//
// detail/type_list.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TYPE_LIST_HPP
#define ASIO_DETAIL_TYPE_LIST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Types>
struct type_list;

template <typename R>
struct type_list<R()>
{
};

template <typename R, typename T0>
struct type_list<R(T0)>
{
  typedef T0 head;
  typedef void tail();
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R, typename T0, typename... Tn>
struct type_list<R(T0, Tn...)>
{
  typedef T0 head;
  typedef void tail(Tn...);
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_TYPE_LIST_DEF(n) \
template <typename R, typename T0, ASIO_VARIADIC_TPARAMS(n)> \
struct type_list<R(T0, ASIO_VARIADIC_TARGS(n))> \
{ \
  typedef T0 head; \
  typedef void tail(ASIO_VARIADIC_TARGS(n)); \
}; \
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_TYPE_LIST_DEF)
#undef ASIO_PRIVATE_TYPE_LIST_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Types1, typename Types2>
struct type_list_cat;

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R1, typename... Tn, typename R2, typename... Un>
struct type_list_cat<R1(Tn...), R2(Un...)>
{
  typedef void type(Tn..., Un...);
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R1, typename R2>
struct type_list_cat<R1(), R2()>
{
  typedef void type();
};

template <typename R1, typename Types2>
struct type_list_cat<R1(), Types2>
{
  typedef Types2 type;
};

template <typename R1, typename T0, typename R2>
struct type_list_cat<R1(T0), R2()>
{
  typedef void type(T0);
};

template <typename R1, typename T0, typename Types2>
struct type_list_cat<R1(T0), Types2>
{
  typedef typename type_list_cat<
    void(T0, typename type_list<Types2>::head),
    typename type_list<Types2>::tail>::type type;
};

#define ASIO_PRIVATE_TYPE_LIST_CAT_DEF(n) \
template <typename R1, typename T0, \
  ASIO_VARIADIC_TPARAMS(n), typename R2> \
struct type_list_cat<R1(T0, ASIO_VARIADIC_TARGS(n)), R2()> \
{ \
  typedef void type(T0, ASIO_VARIADIC_TARGS(n)); \
}; \
\
template <typename R1, typename T0, \
  ASIO_VARIADIC_TPARAMS(n), typename Types2> \
struct type_list_cat<R1(T0, ASIO_VARIADIC_TARGS(n)), Types2> \
{ \
  typedef typename type_list_cat< \
    void(T0, ASIO_VARIADIC_TARGS(n), typename type_list<Types2>::head), \
    typename type_list<Types2>::tail>::type type; \
}; \
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_TYPE_LIST_CAT_DEF)
#undef ASIO_PRIVATE_TYPE_LIST_CAT_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TYPE_LIST_HPP
