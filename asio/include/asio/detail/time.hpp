//
// time.hpp
// ~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_TIME_HPP
#define ASIO_DETAIL_TIME_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_time.hpp"
#include "asio/detail/win_time.hpp"

namespace asio {
namespace detail {

#if defined(_WIN32)
typedef win_time time;
#else
typedef posix_time time;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIME_HPP
