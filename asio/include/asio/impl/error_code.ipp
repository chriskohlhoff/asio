//
// error_code.ipp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_CODE_IPP
#define ASIO_ERROR_CODE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <cerrno>
#include <cstring>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/detail/local_free_on_block_exit.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {

inline std::string error_code::message() const
{
  if (*this == error::already_open)
    return "Already open.";
  if (*this == error::not_found)
    return "Not found.";
  if (*this == error::fd_set_failure)
    return "The descriptor does not fit into the select call's fd_set.";
  if (category_ == error::get_ssl_category())
    return "SSL error.";
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  value_type value = value_;
  if (category() != error::get_system_category() && *this != error::eof)
    return "asio error";
  if (*this == error::eof)
    value = ERROR_HANDLE_EOF;
  char* msg = 0;
  DWORD length = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
      | FORMAT_MESSAGE_FROM_SYSTEM
      | FORMAT_MESSAGE_IGNORE_INSERTS, 0, value,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, 0);
  detail::local_free_on_block_exit local_free_obj(msg);
  if (length && msg[length - 1] == '\n')
    msg[--length] = '\0';
  if (length && msg[length - 1] == '\r')
    msg[--length] = '\0';
  if (length)
    return msg;
  else
    return "asio error";
#else // defined(BOOST_WINDOWS)
  if (*this == error::eof)
    return "End of file.";
  if (*this == error::host_not_found)
    return "Host not found (authoritative).";
  if (*this == error::host_not_found_try_again)
    return "Host not found (non-authoritative), try again later.";
  if (*this == error::no_recovery)
    return "A non-recoverable error occurred during database lookup.";
  if (*this == error::no_data)
    return "The query is valid, but it does not have associated data.";
  if (*this == error::not_found)
    return "Element not found.";
#if !defined(__sun)
  if (*this == error::operation_aborted)
    return "Operation aborted.";
#endif // !defined(__sun)
  if (*this == error::service_not_found)
    return "Service not found.";
  if (*this == error::socket_type_not_supported)
    return "Socket type not supported.";
  if (category() != error::get_system_category())
    return "asio error";
#if defined(__sun) || defined(__QNX__)
  return strerror(value_);
#elif defined(__MACH__) && defined(__APPLE__) \
|| defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__) \
|| defined(_AIX) || defined(__hpux) || defined(__osf__)
  char buf[256] = "";
  strerror_r(value_, buf, sizeof(buf));
  return buf;
#else
  char buf[256] = "";
  return strerror_r(value_, buf, sizeof(buf));
#endif
#endif // defined(BOOST_WINDOWS)
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_CODE_IPP
