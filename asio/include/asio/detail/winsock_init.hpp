//
// detail/winsock_init.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINSOCK_INIT_HPP
#define ASIO_DETAIL_WINSOCK_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <int Major = 2, int Minor = 0>
class winsock_init
  : private noncopyable
{
public:
  winsock_init(bool allow_throw = true)
  {
    init();
    if (allow_throw)
      throw_on_error();
  }

  winsock_init(const winsock_init&)
  {
    init();
    throw_on_error();
  }

  ~winsock_init()
  {
    cleanup();
  }

private:
  void init()
  {
    if (::InterlockedIncrement(&data_.init_count_) == 1)
    {
      WSADATA wsa_data;
      long result = ::WSAStartup(MAKEWORD(Major, Minor), &wsa_data);
      ::InterlockedExchange(&data_.result_, result);
    }
  }

  void cleanup()
  {
    if (::InterlockedDecrement(&data_.init_count_) == 0)
    {
      ::WSACleanup();
    }
  }

  void throw_on_error()
  {
    long result = ::InterlockedExchangeAdd(&data_.result_, 0);
    if (result != 0)
    {
      asio::error_code ec(result,
          asio::error::get_system_category());
      asio::detail::throw_error(ec, "winsock");
    }
  }

  // Structure to track result of initialisation and number of uses. POD is used
  // to ensure that the values are zero-initialised prior to any code being run.
  static struct data
  {
    long init_count_;
    long result_;
  } data_;
};

template <int Major, int Minor>
typename winsock_init<Major, Minor>::data winsock_init<Major, Minor>::data_;

// Static variable to ensure that winsock is initialised before main, and
// therefore before any other threads can get started.
static const winsock_init<>& winsock_init_instance = winsock_init<>(false);

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)

#endif // ASIO_DETAIL_WINSOCK_INIT_HPP
