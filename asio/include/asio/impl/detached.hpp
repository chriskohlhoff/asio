//
// impl/detached.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_DETACHED_HPP
#define ASIO_IMPL_DETACHED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/handler_type.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

  // Class to adapt a detached_t as a completion handler.
  class detached_handler
  {
  public:
    detached_handler(detached_t)
    {
    }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

    template <typename... Args>
    void operator()(Args...)
    {
    }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

    void operator()()
    {
    }

#define ASIO_PRIVATE_DETACHED_DEF(n) \
    template <ASIO_VARIADIC_TPARAMS(n)> \
    void operator()(ASIO_VARIADIC_BYVAL_PARAMS(n)) \
    { \
    } \
    /**/
    ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_DETACHED_DEF)
#undef ASIO_PRIVATE_DETACHED_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
  };

} // namespace detail

#if !defined(GENERATING_DOCUMENTATION)

template <typename Signature>
struct handler_type<detached_t, Signature>
{
  typedef detail::detached_handler type;
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_DETACHED_HPP
