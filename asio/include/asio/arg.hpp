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

#elif defined(_MSC_VER) && (_MSC_VER <= 1300)

static boost::arg<1> error;
static boost::arg<2> bytes_sent;
static boost::arg<2> last_bytes_sent;
static boost::arg<2> bytes_recvd;
static boost::arg<2> last_bytes_recvd;
static boost::arg<3> total_bytes_sent;
static boost::arg<3> total_bytes_recvd;

#else

boost::arg<1> error;
boost::arg<2> bytes_sent;
boost::arg<2> last_bytes_sent;
boost::arg<2> bytes_recvd;
boost::arg<2> last_bytes_recvd;
boost::arg<3> total_bytes_sent;
boost::arg<3> total_bytes_recvd;

#endif

} // namespace

} // namespace arg

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ARG_HPP
