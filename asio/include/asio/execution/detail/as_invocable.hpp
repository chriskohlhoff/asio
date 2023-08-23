//
// execution/detail/as_invocable.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_AS_INVOCABLE_HPP
#define ASIO_EXECUTION_DETAIL_AS_INVOCABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/atomic_count.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/receiver_invocation_error.hpp"
#include "asio/execution/set_done.hpp"
#include "asio/execution/set_error.hpp"
#include "asio/execution/set_value.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Receiver, typename>
struct as_invocable
{
  Receiver* receiver_;

  explicit as_invocable(Receiver& r) noexcept
    : receiver_(asio::detail::addressof(r))
  {
  }

  as_invocable(as_invocable&& other) noexcept
    : receiver_(other.receiver_)
  {
    other.receiver_ = 0;
  }

  ~as_invocable()
  {
    if (receiver_)
      execution::set_done(static_cast<Receiver&&>(*receiver_));
  }

  void operator()() & noexcept
  {
#if !defined(ASIO_NO_EXCEPTIONS)
    try
    {
#endif // !defined(ASIO_NO_EXCEPTIONS)
      execution::set_value(static_cast<Receiver&&>(*receiver_));
      receiver_ = 0;
#if !defined(ASIO_NO_EXCEPTIONS)
    }
    catch (...)
    {
      execution::set_error(static_cast<Receiver&&>(*receiver_),
          std::make_exception_ptr(receiver_invocation_error()));
      receiver_ = 0;
    }
#endif // !defined(ASIO_NO_EXCEPTIONS)
  }
};

template <typename T>
struct is_as_invocable : false_type
{
};

template <typename Function, typename T>
struct is_as_invocable<as_invocable<Function, T>> : true_type
{
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_AS_INVOCABLE_HPP
