//
// arg.hpp
// ~~~~~~~
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

#ifndef ASIO_ARG_HPP
#define ASIO_ARG_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind/arg.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

namespace arg {

namespace {

#if defined(__BORLANDC__)

static inline boost::arg<1> error() { return boost::arg<1>(); }
static inline boost::arg<2> bytes_sent() { return boost::arg<2>(); }
static inline boost::arg<2> last_bytes_sent() { return boost::arg<2>(); }
static inline boost::arg<2> bytes_recvd() { return boost::arg<2>(); }
static inline boost::arg<2> last_bytes_recvd() { return boost::arg<2>(); }
static inline boost::arg<3> total_bytes_sent() { return boost::arg<3>(); }
static inline boost::arg<3> total_bytes_recvd() { return boost::arg<3>(); }

#elif defined(_MSC_VER) && (_MSC_VER < 1400)

static boost::arg<1> error;
static boost::arg<2> bytes_sent;
static boost::arg<2> last_bytes_sent;
static boost::arg<2> bytes_recvd;
static boost::arg<2> last_bytes_recvd;
static boost::arg<3> total_bytes_sent;
static boost::arg<3> total_bytes_recvd;

#else

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the error argument of a handler for any of the asynchronous socket-related
/// functions.
boost::arg<1> error;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the bytes_sent argument of a handler for asynchronous functions such as
/// asio::async_send or asio::stream_socket::async_send.
boost::arg<2> bytes_sent;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the last_bytes_sent argument of a handler for the asio::async_send_n or
/// asio::async_send_at_least_n functions.
boost::arg<2> last_bytes_sent;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the bytes_recvd argument of a handler for asynchronous functions such as
/// asio::async_recv or asio::stream_socket::async_recv.
boost::arg<2> bytes_recvd;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the last_bytes_recvd argument of a handler for the asio::async_recv_n or
/// asio::async_recv_at_least_n functions.
boost::arg<2> last_bytes_recvd;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the total_bytes_sent argument of a handler for the asio::async_send_n or
/// asio::async_send_at_least_n functions.
boost::arg<3> total_bytes_sent;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the total_bytes_recvd argument of a handler for the asio::async_recv_n or
/// asio::async_recv_at_least_n functions.
boost::arg<3> total_bytes_recvd;

#endif

} // namespace

} // namespace arg

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ARG_HPP
