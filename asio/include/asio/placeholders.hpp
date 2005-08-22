//
// placeholders.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ARG_HPP
#define ASIO_ARG_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind/arg.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

namespace placeholders {

namespace {

#if defined(__BORLANDC__)

static inline boost::arg<1> error()
{
  return boost::arg<1>();
}

static inline boost::arg<2> bytes_transferred()
{
  return boost::arg<2>();
}

static inline boost::arg<2> last_bytes_transferred()
{
  return boost::arg<2>();
}

static inline boost::arg<3> total_bytes_transferred()
{
  return boost::arg<3>();
}

#elif defined(_MSC_VER) && (_MSC_VER < 1400)

static boost::arg<1> error;
static boost::arg<2> bytes_transferred;
static boost::arg<2> last_bytes_transferred;
static boost::arg<3> total_bytes_transferred;

#else

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the error argument of a handler for any of the asynchronous functions.
boost::arg<1> error;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the bytes_transferred argument of a handler for asynchronous functions such
/// as asio::async_write or asio::stream_socket::async_write.
boost::arg<2> bytes_transferred;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the last_bytes_transferred argument of a handler for asynchronous functions
/// such as asio::async_write_n or asio::async_read_at_least_n.
boost::arg<2> last_bytes_transferred;

/// An argument placeholder, for use with \ref boost_bind, that corresponds to
/// the total_bytes_transferred argument of a handler for asynchronous functions
/// such as asio::async_write_n or asio::async_read_at_least_n.
boost::arg<3> total_bytes_transferred;

#endif

} // namespace

} // namespace placeholders

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ARG_HPP
