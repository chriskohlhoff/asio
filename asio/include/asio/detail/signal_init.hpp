//
// signal_init.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SIGNAL_INIT_HPP
#define ASIO_DETAIL_SIGNAL_INIT_HPP

#include "asio/detail/push_options.hpp"

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <csignal>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

template <int Signal = SIGPIPE>
class signal_init
{
public:
  // Constructor.
  signal_init()
  {
    std::signal(Signal, SIG_IGN);
  }

  // Used to ensure that the signal stuff is initialised.
  static void use()
  {
    while (&instance_ == 0);
  }

private:
  // Instance to force initialisation of signal at global scope.
  static signal_init<Signal> instance_;
};

template <int Signal>
signal_init<Signal> signal_init<Signal>::instance_;

} // namespace detail
} // namespace asio

#endif // !_WIN32

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SIGNAL_INIT_HPP
