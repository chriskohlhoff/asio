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

template <>
struct type_list<void()>
{
};

template <typename T0>
struct type_list<void(T0)>
{
  typedef T0 head;
  typedef void tail();
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T0, typename... Tn>
struct type_list<void(T0, Tn...)>
{
  typedef T0 head;
  typedef void tail(Tn...);
};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

#define ASIO_PRIVATE_TYPE_LIST_DEF(n) \
template <typename T0, ASIO_VARIADIC_TPARAMS(n)> \
struct type_list<void(T0, ASIO_VARIADIC_TARGS(n))> \
{ \
  typedef T0 head; \
  typedef void tail(ASIO_VARIADIC_TARGS(n)); \
}; \
/**/
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_TYPE_LIST_DEF)
#undef ASIO_PRIVATE_TYPE_LIST_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TYPE_LIST_HPP
