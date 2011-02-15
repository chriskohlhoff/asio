//
// basic_signal_set.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SIGNAL_SET_HPP
#define ASIO_BASIC_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/signal_set_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Provides signal functionality.
template <typename SignalSetService = signal_set_service>
class basic_signal_set
  : public basic_io_object<SignalSetService>
{
public:
  /// Construct a signal set without adding any signals.
  explicit basic_signal_set(asio::io_service& io_service)
    : basic_io_object<SignalSetService>(io_service)
  {
  }

  /// Construct a signal set and add one signal.
  basic_signal_set(asio::io_service& io_service, int signal_number_1)
    : basic_io_object<SignalSetService>(io_service)
  {
    asio::error_code ec;
    this->service.add(this->implementation, signal_number_1, ec);
    asio::detail::throw_error(ec);
  }

  /// Construct a signal set and add two signals.
  basic_signal_set(asio::io_service& io_service, int signal_number_1,
      int signal_number_2)
    : basic_io_object<SignalSetService>(io_service)
  {
    asio::error_code ec;
    this->service.add(this->implementation, signal_number_1, ec);
    asio::detail::throw_error(ec);
    this->service.add(this->implementation, signal_number_2, ec);
    asio::detail::throw_error(ec);
  }

  /// Construct a signal set and add three signals.
  basic_signal_set(asio::io_service& io_service, int signal_number_1,
      int signal_number_2, int signal_number_3)
    : basic_io_object<SignalSetService>(io_service)
  {
    asio::error_code ec;
    this->service.add(this->implementation, signal_number_1, ec);
    asio::detail::throw_error(ec);
    this->service.add(this->implementation, signal_number_2, ec);
    asio::detail::throw_error(ec);
    this->service.add(this->implementation, signal_number_3, ec);
    asio::detail::throw_error(ec);
  }

  /// Add a signal to a signal_set.
  void add(int signal_number)
  {
    asio::error_code ec;
    this->service.add(this->implementation, signal_number, ec);
    asio::detail::throw_error(ec);
  }

  /// Add a signal to a signal_set.
  asio::error_code add(int signal_number,
      asio::error_code& ec)
  {
    return this->service.add(this->implementation, signal_number, ec);
  }

  /// Remove a signal to a signal_set.
  void remove(int signal_number)
  {
    asio::error_code ec;
    this->service.remove(this->implementation, signal_number, ec);
    asio::detail::throw_error(ec);
  }

  /// Remove a signal to a signal_set.
  asio::error_code remove(int signal_number,
      asio::error_code& ec)
  {
    return this->service.remove(this->implementation, signal_number, ec);
  }

  /// Remove all signals from a signal_set.
  void clear()
  {
    asio::error_code ec;
    this->service.clear(this->implementation, ec);
    asio::detail::throw_error(ec);
  }

  /// Remove all signals from a signal_set.
  asio::error_code clear(asio::error_code& ec)
  {
    return this->service.clear(this->implementation, ec);
  }

  /// Cancel all operations associated with the signal set.
  void cancel()
  {
    asio::error_code ec;
    this->service.cancel(this->implementation, ec);
    asio::detail::throw_error(ec);
  }

  /// Cancel all operations associated with the signal set.
  asio::error_code cancel(asio::error_code& ec)
  {
    return this->service.cancel(this->implementation, ec);
  }

  // Start an asynchronous operation to wait for a signal to be delivered.
  template <typename SignalHandler>
  void async_wait(SignalHandler handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a SignalHandler.
    ASIO_SIGNAL_HANDLER_CHECK(SignalHandler, handler) type_check;

    this->service.async_wait(this->implementation,
        ASIO_MOVE_CAST(SignalHandler)(handler));
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_SIGNAL_SET_HPP
