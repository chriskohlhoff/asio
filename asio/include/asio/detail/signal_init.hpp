//
// signal_init.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
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
