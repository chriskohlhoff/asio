//
// impl/error_code.ipp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ERROR_CODE_IPP
#define ASIO_IMPL_ERROR_CODE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/local_free_on_block_exit.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/error.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

std::string error_code::message() const
{
  if (category_ == error::get_misc_category())
  {
    if (value_ == error::already_open)
      return "Already open.";
    if (value_ == error::not_found)
      return "Not found.";
    if (value_ == error::fd_set_failure)
      return "The descriptor does not fit into the select call's fd_set.";
    if (value_ == error::not_found)
      return "Element not found.";
#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
    if (value_ == error::eof)
      return "End of file.";
#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
  }
  if (category_ == error::get_ssl_category())
    return "SSL error.";
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  value_type value = value_;
  if (category_ == error::get_misc_category() && value_ == error::eof)
    value = ERROR_HANDLE_EOF;
  else if (category_ != error::get_system_category())
    return "asio error";
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
  if (category_ == error::get_netdb_category())
  {
    if (value_ == error::host_not_found)
      return "Host not found (authoritative).";
    if (value_ == error::host_not_found_try_again)
      return "Host not found (non-authoritative), try again later.";
    if (value_ == error::no_recovery)
      return "A non-recoverable error occurred during database lookup.";
    if (value_ == error::no_data)
      return "The query is valid, but it does not have associated data.";
  }
  if (category_ == error::get_addrinfo_category())
  {
    if (value_ == error::service_not_found)
      return "Service not found.";
    if (value_ == error::socket_type_not_supported)
      return "Socket type not supported.";
  }
  if (category_ != error::get_system_category())
    return "asio error";
#if !defined(__sun)
  if (value_ == error::operation_aborted)
    return "Operation aborted.";
#endif // !defined(__sun)
#if defined(__sun) || defined(__QNX__) || defined(__SYMBIAN32__)
  using namespace std;
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

#endif // ASIO_IMPL_ERROR_CODE_IPP
