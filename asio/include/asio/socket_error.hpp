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
    /// Permission denied.
    access_denied = ASIO_SOCKET_ERROR(EACCES),

    /// Address family not supported by protocol.
    address_family_not_supported = ASIO_SOCKET_ERROR(EAFNOSUPPORT),

    /// Address already in use.
    address_in_use = ASIO_SOCKET_ERROR(EADDRINUSE),

    /// Transport endpoint is already connected.
    already_connected = ASIO_SOCKET_ERROR(EISCONN),

    /// Operation already in progress.
    already_started = ASIO_SOCKET_ERROR(EALREADY),

    /// Connection refused.
    connection_refused = ASIO_SOCKET_ERROR(ECONNREFUSED),

    /// Connection reset by peer.
    connection_reset = ASIO_SOCKET_ERROR(ECONNRESET),

    /// Bad file descriptor.
    bad_descriptor = ASIO_SOCKET_ERROR(EBADF),

    /// Bad address.
    fault = ASIO_SOCKET_ERROR(EFAULT),

    /// Host not found (authoritative).
    host_not_found = ASIO_SOCKET_ERROR(HOST_NOT_FOUND),

    /// Host not found (non-authoritative).
    host_not_found_try_again = ASIO_SOCKET_ERROR(TRY_AGAIN),

    /// No route to host.
    host_unreachable = ASIO_SOCKET_ERROR(EHOSTUNREACH),

    /// Operation now in progress.
    in_progress = ASIO_SOCKET_ERROR(EINPROGRESS),

    /// Interrupted system call.
    interrupted = ASIO_SOCKET_ERROR(EINTR),

    /// Invalid argument.
    invalid_argument = ASIO_SOCKET_ERROR(EINVAL),

    /// Message too long.
    message_size = ASIO_SOCKET_ERROR(EMSGSIZE),

    /// Network is down.
    network_down = ASIO_SOCKET_ERROR(ENETDOWN),

    /// Network dropped connection on reset.
    network_reset = ASIO_SOCKET_ERROR(ENETRESET),

    /// Network is unreachable.
    network_unreachable = ASIO_SOCKET_ERROR(ENETUNREACH),

    /// Too many open files.
    no_descriptors = ASIO_SOCKET_ERROR(EMFILE),

    /// No buffer space available.
    no_buffer_space = ASIO_SOCKET_ERROR(ENOBUFS),

    /// The host is valid but does not have address data.
    no_host_data = ASIO_SOCKET_ERROR(NO_DATA),

    /// Cannot allocate memory.
    no_memory = ENOMEM,

    /// Operation not permitted.
    no_permission = EPERM,

    /// Protocol not available.
    no_protocol_option = ASIO_SOCKET_ERROR(ENOPROTOOPT),

    /// A non-recoverable error occurred.
    no_recovery = ASIO_SOCKET_ERROR(NO_RECOVERY),

    /// Transport endpoint is not connected.
    not_connected = ASIO_SOCKET_ERROR(ENOTCONN),

    /// Socket operation on non-socket.
    not_socket = ASIO_SOCKET_ERROR(ENOTSOCK),

    /// Operation not supported.
    not_supported = ASIO_SOCKET_ERROR(EOPNOTSUPP),

    /// Operation cancelled.
#if defined(_WIN32)
    operation_aborted = ERROR_OPERATION_ABORTED,
#else // defined(_WIN32)
    operation_aborted = ECANCELED,
#endif // defined(_WIN32)

    /// Cannot send after transport endpoint shutdown.
    shut_down = ASIO_SOCKET_ERROR(ESHUTDOWN),

    /// Success.
    success = 0,

    /// Connection timed out.
    timed_out = ASIO_SOCKET_ERROR(ETIMEDOUT),

    /// Resource temporarily unavailable.
    try_again = EAGAIN,

    /// The socket is marked non-blocking and the requested operation would
    /// block.
    would_block = ASIO_SOCKET_ERROR(EWOULDBLOCK)
  };

  /// Constructor.
  socket_error(int code)
    : code_(code)
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
#if defined(_WIN32)
    if (code_ == ENOMEM || code_ == EPERM || code_ == EAGAIN)
    {
      return std::string(strerror(code_));
    }
    else
    {
      void* msg_buf = 0;
      ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
          | FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS, 0, code_,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&msg_buf, 0, 0);
      std::string msg((LPCTSTR)msg_buf);
      ::LocalFree(msg_buf);
      if (msg.size() && msg[msg.size() - 1] == '\n')
        msg.resize(msg.size() - 1);
      if (msg.size() && msg[msg.size() - 1] == '\r')
        msg.resize(msg.size() - 1);
      return msg;
    }
#elif defined(__sun)
    return std::string(strerror(code_));
#else
    char buf[256] = "";
    return std::string(strerror_r(code_, buf, sizeof(buf)));
#endif
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
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ERROR_HPP
