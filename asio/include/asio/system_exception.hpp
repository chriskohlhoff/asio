//
// error.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SYSTEM_EXCEPTION_HPP
#define ASIO_SYSTEM_EXCEPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <cerrno>
#include <cstring>
#include <exception>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/win_local_free_on_block_exit.hpp"

namespace asio {

/// The system_exception class is used to represent system conditions that
/// prevent the library from operating correctly.
class system_exception
  : public std::exception
{
public:
  /// The execution context of exception. The values are intended to be
  /// implementation-defined.
  enum context_type
  {
    thread,
    mutex,
    event,
    tss,
    winsock,
    epoll,
    kqueue,
    iocp
  };

  /// Construct with a specific context and error code.
  system_exception(context_type context, int code)
    : context_(context),
      code_(code)
  {
  }

  /// Destructor.
  virtual ~system_exception() throw ()
  {
  }

  /// Get the string for the type of exception.
  virtual const char* what() const throw ()
  {
    return "asio system_exception";
  }

  /// Get the implementation-defined context associated with the exception.
  context_type context() const
  {
    return context_;
  }

  /// Get the implementation-defined code associated with the exception.
  int code() const
  {
    return code_;
  }

private:
  // The context associated with the error.
  context_type context_;

  // The code associated with the error.
  int code_;
};

/// Output the string associated with a system exception.
/**
 * Used to output a human-readable string that is associated with a system
 * exception.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param e The exception to be written.
 *
 * @return The output stream.
 *
 * @relates asio::system_exception
 */
template <typename Ostream>
Ostream& operator<<(Ostream& os, const system_exception& e)
{
  os << e.what();
  switch (e.context())
  {
  case system_exception::thread:
    os << " (thread): ";
    break;
  case system_exception::mutex:
    os << " (mutex): ";
    break;
  case system_exception::event:
    os << " (event): ";
    break;
  case system_exception::tss:
    os << " (tss): ";
    break;
  case system_exception::winsock:
    os << " (winsock): ";
    break;
  case system_exception::epoll:
    os << " (epoll): ";
    break;
  case system_exception::kqueue:
    os << " (kqueue): ";
    break;
  case system_exception::iocp:
    os << " (iocp): ";
    break;
  default:
    os << " (unknown): ";
    break;
  }
#if defined(BOOST_WINDOWS)
  char* msg = 0;
  DWORD length = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
      | FORMAT_MESSAGE_FROM_SYSTEM
      | FORMAT_MESSAGE_IGNORE_INSERTS, 0, e.code(),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, 0);
  detail::win_local_free_on_block_exit local_free_obj(msg);
  if (length && msg[length - 1] == '\n')
    msg[--length] = '\0';
  if (length && msg[length - 1] == '\r')
    msg[--length] = '\0';
  if (length)
    os << msg;
  else
    os << e.code();
#elif defined(__sun)
  os << strerror(e.code());
#elif defined(__MACH__) && defined(__APPLE__)
  char buf[256] = "";
  strerror_r(e.code(), buf, sizeof(buf));
  os << buf;
#else
  char buf[256] = "";
  os << strerror_r(e.code(), buf, sizeof(buf));
#endif
  return os;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SYSTEM_EXCEPTION_HPP
