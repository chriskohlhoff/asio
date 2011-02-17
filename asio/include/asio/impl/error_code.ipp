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
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
# include <winerror.h>
#else
# include <cerrno>
#endif
#include "asio/detail/local_free_on_block_exit.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class system_category : public error_category
{
public:
  const char* name() const
  {
    return "asio.system";
  }

  std::string message(int value) const
  {
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
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
      return "asio.system error";
#else // defined(BOOST_WINDOWS)
#if !defined(__sun)
    if (value == ECANCELED)
      return "Operation aborted.";
#endif // !defined(__sun)
#if defined(__sun) || defined(__QNX__) || defined(__SYMBIAN32__)
    using namespace std;
    return strerror(value);
#elif defined(__MACH__) && defined(__APPLE__) \
  || defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__) \
  || defined(_AIX) || defined(__hpux) || defined(__osf__) \
  || defined(__ANDROID__)
    char buf[256] = "";
    strerror_r(value, buf, sizeof(buf));
    return buf;
#else
    char buf[256] = "";
    return strerror_r(value, buf, sizeof(buf));
#endif
#endif // defined(BOOST_WINDOWS)
  }
};

} // namespace detail

const error_category& system_category()
{
  static detail::system_category instance;
  return instance;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_ERROR_CODE_IPP
