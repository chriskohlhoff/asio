//
// experimental/channel_traits.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_CHANNEL_TRAITS_HPP
#define ASIO_EXPERIMENTAL_CHANNEL_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <deque>
#include "asio/error.hpp"
#include "asio/experimental/channel_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

#if defined(GENERATING_DOCUMENTATION)

template <typename... Signatures>
struct channel_traits
{
  /// Rebind the traits to a new set of signatures.
  /**
   * This nested structure must have a single nested type @c other that
   * aliases a traits type with the specified set of signatures.
   */
  template <typename... NewSignatures>
  struct rebind
  {
    typedef user_defined other;
  };

  /// Determine the container for the specified elements.
  /**
   * This nested structure must have a single nested type @c other that
   * aliases a container type for the specified element type.
   */
  template <typename Element>
  struct container
  {
    typedef user_defined type;
  };

  /// The signature of a channel cancellation notification.
  typedef void receive_cancelled_signature(...);

  /// Invoke the specified handler with a cancellation notification.
  template <typename Handler>
  static void invoke_receive_cancelled(Handler handler);

  /// The signature of a channel closed notification.
  typedef void receive_closed_signature(...);

  /// Invoke the specified handler with a closed notification.
  template <typename Handler>
  static void invoke_receive_closed(Handler handler);
};

#else // defined(GENERATING_DOCUMENTATION)

/// Traits used for customising channel behaviour.
template <typename... Signatures>
struct channel_traits
{
  template <typename... NewSignatures>
  struct rebind
  {
    typedef channel_traits<NewSignatures...> other;
  };
};

template <typename R, typename... Args, typename... Signatures>
struct channel_traits<R(asio::error_code, Args...), Signatures...>
{
  template <typename... NewSignatures>
  struct rebind
  {
    typedef channel_traits<NewSignatures...> other;
  };

  template <typename Element>
  struct container
  {
    typedef std::deque<Element> type;
  };

  typedef R receive_cancelled_signature(asio::error_code, Args...);

  template <typename Handler>
  static void invoke_receive_cancelled(Handler handler)
  {
    const asio::error_code e = error::channel_cancelled;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(
        e, typename decay<Args>::type()...);
  }

  typedef R receive_closed_signature(asio::error_code, Args...);

  template <typename Handler>
  static void invoke_receive_closed(Handler handler)
  {
    const asio::error_code e = error::channel_closed;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(
        e, typename decay<Args>::type()...);
  }
};

template <typename R, typename... Args, typename... Signatures>
struct channel_traits<R(std::exception_ptr, Args...), Signatures...>
{
  template <typename... NewSignatures>
  struct rebind
  {
    typedef channel_traits<NewSignatures...> other;
  };

  template <typename Element>
  struct container
  {
    typedef std::deque<Element> type;
  };

  typedef R receive_cancelled_signature(std::exception_ptr, Args...);

  template <typename Handler>
  static void invoke_receive_cancelled(Handler handler)
  {
    const asio::error_code e = error::channel_cancelled;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(
        std::make_exception_ptr(asio::system_error(e)),
        typename decay<Args>::type()...);
  }

  typedef R receive_closed_signature(std::exception_ptr, Args...);

  template <typename Handler>
  static void invoke_receive_closed(Handler handler)
  {
    const asio::error_code e = error::channel_closed;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(
        std::make_exception_ptr(asio::system_error(e)),
        typename decay<Args>::type()...);
  }
};

template <typename R>
struct channel_traits<R()>
{
  template <typename... NewSignatures>
  struct rebind
  {
    typedef channel_traits<NewSignatures...> other;
  };

  template <typename Element>
  struct container
  {
    typedef std::deque<Element> type;
  };

  typedef R receive_cancelled_signature(asio::error_code);

  template <typename Handler>
  static void invoke_receive_cancelled(Handler handler)
  {
    const asio::error_code e = error::channel_cancelled;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(e);
  }

  typedef R receive_closed_signature(asio::error_code);

  template <typename Handler>
  static void invoke_receive_closed(Handler handler)
  {
    const asio::error_code e = error::channel_closed;
    ASIO_MOVE_OR_LVALUE(Handler)(handler)(e);
  }
};

#endif // defined(GENERATING_DOCUMENTATION)

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_CHANNEL_TRAITS_HPP
