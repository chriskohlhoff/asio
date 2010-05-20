//
// detail/descriptor_ops.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_DESCRIPTOR_OPS_HPP
#define ASIO_DETAIL_DESCRIPTOR_OPS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include <cstddef>
#include "asio/error_code.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace descriptor_ops {

ASIO_DECL int open(const char* path, int flags, asio::error_code& ec);

ASIO_DECL int close(int d, asio::error_code& ec);

typedef iovec buf;

ASIO_DECL std::size_t sync_read(int d, buf* bufs, std::size_t count,
    bool all_empty, bool non_blocking, asio::error_code& ec);

ASIO_DECL bool non_blocking_read(int d, buf* bufs, size_t count,
    asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL std::size_t sync_write(int d, const buf* bufs, std::size_t count,
    bool all_empty, bool non_blocking, asio::error_code& ec);

ASIO_DECL bool non_blocking_write(int d, const buf* bufs, size_t count,
    asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL int ioctl(int d, long cmd, ioctl_arg_type* arg,
    asio::error_code& ec);

ASIO_DECL int fcntl(int d, long cmd, asio::error_code& ec);

ASIO_DECL int fcntl(int d, long cmd, long arg, asio::error_code& ec);

ASIO_DECL int poll_read(int d, asio::error_code& ec);

ASIO_DECL int poll_write(int d, asio::error_code& ec);

} // namespace descriptor_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/descriptor_ops.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_DESCRIPTOR_OPS_HPP
