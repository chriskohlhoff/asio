//
// socket_error.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_ERROR_HPP
#define ASIO_SOCKET_ERROR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cerrno>
#include <exception>
#include <string>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {

#if defined(_WIN32)
# define ASIO_SOCKET_ERROR(e) WSA ## e
#else
# define ASIO_SOCKET_ERROR(e) e
#endif

/// The socket_error class is used to encapsulate socket error codes.
class socket_error
  : public std::exception
{
public:
  /// Error codes.
  enum
  {
    access_denied = ASIO_SOCKET_ERROR(EACCES),
    address_family_not_supported = ASIO_SOCKET_ERROR(EAFNOSUPPORT),
    address_in_use = ASIO_SOCKET_ERROR(EADDRINUSE),
    already_connected = ASIO_SOCKET_ERROR(EISCONN),
    already_started = ASIO_SOCKET_ERROR(EALREADY),
    connection_refused = ASIO_SOCKET_ERROR(ECONNREFUSED),
    connection_reset = ASIO_SOCKET_ERROR(ECONNRESET),
    bad_descriptor = ASIO_SOCKET_ERROR(EBADF),
    fault = ASIO_SOCKET_ERROR(EFAULT),
    host_unreachable = ASIO_SOCKET_ERROR(EHOSTUNREACH),
    in_progress = ASIO_SOCKET_ERROR(EINPROGRESS),
    interrupted = ASIO_SOCKET_ERROR(EINTR),
    invalid_argument = ASIO_SOCKET_ERROR(EINVAL),
    message_size = ASIO_SOCKET_ERROR(EMSGSIZE),
    network_down = ASIO_SOCKET_ERROR(ENETDOWN),
    network_reset = ASIO_SOCKET_ERROR(ENETRESET),
    network_unreachable = ASIO_SOCKET_ERROR(ENETUNREACH),
    no_descriptors = ASIO_SOCKET_ERROR(EMFILE),
    no_buffer_space = ASIO_SOCKET_ERROR(ENOBUFS),
    no_memory = ENOMEM,
    no_permission = EPERM,
    no_protocol_option = ASIO_SOCKET_ERROR(ENOPROTOOPT),
    not_connected = ASIO_SOCKET_ERROR(ENOTCONN),
    not_socket = ASIO_SOCKET_ERROR(ENOTSOCK),
    not_supported = ASIO_SOCKET_ERROR(EOPNOTSUPP),
#if defined(_WIN32)
    operation_aborted = ERROR_OPERATION_ABORTED,
#else // defined(_WIN32)
    operation_aborted = ECANCELED,
#endif // defined(_WIN32)
    shut_down = ASIO_SOCKET_ERROR(ESHUTDOWN),
    success = 0,
    timed_out = ASIO_SOCKET_ERROR(ETIMEDOUT),
    try_again = EAGAIN,
    would_block = ASIO_SOCKET_ERROR(EWOULDBLOCK)
  };

  /// Constructor.
  socket_error(int code)
    : code_(code),
      message_()
  {
  }

  // Destructor.
  virtual ~socket_error() throw ()
  {
  }

  // Get the string for the type of exception.
  virtual const char* what() const throw ()
  {
    return "Socket error";
  }

  /// Get the code associated with the error.
  int code() const
  {
    return code_;
  }

  /// Get the message associated with the error.
  std::string message() const
  {
    if (message_.length() == 0)
    {
#if defined(_WIN32)
      if (code_ == ENOMEM || code_ == EPERM || code_ == EAGAIN)
      {
        message_ = strerror(code_);
      }
      else
      {
        void* msg_buf;
        ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS, 0, code_,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&msg_buf, 0, 0);
        message_ = (LPCTSTR)msg_buf;
        ::LocalFree(msg_buf);
      }
#else
      char buf[256] = "";
      message_ = strerror_r(code_, buf, sizeof(buf));
#endif
    }

    return message_;
  }

  /// Operator returns non-null if there is a non-success error code.
  operator void*() const
  {
    if (code_ == success)
      return 0;
    else
      return const_cast<void*>(static_cast<const void*>(this));
  }

  /// Operator to test if the error represents success.
  bool operator!() const
  {
    return code_ == success;
  }

private:
  // The code associated with the error.
  int code_;

  // The message associated with the error.
  mutable std::string message_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ERROR_HPP
