//
// error.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_HPP
#define ASIO_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cerrno>
#include <exception>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {

#if defined(_WIN32)
# define ASIO_SOCKET_ERROR(e) WSA ## e
# define ASIO_NETDB_ERROR(e) WSA ## e
# define ASIO_WIN_OR_POSIX_ERROR(e_win, e_posix) e_win
#else
# define ASIO_SOCKET_ERROR(e) e
# define ASIO_NETDB_ERROR(e) 16384 + e
# define ASIO_WIN_OR_POSIX_ERROR(e_win, e_posix) e_posix
#endif

/// The error class is used to encapsulate system error codes.
class error
  : public std::exception
{
public:
  /// Error codes.
  enum code_type
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

    /// A connection has been aborted.
    connection_aborted = ASIO_SOCKET_ERROR(ECONNABORTED),

    /// Connection refused.
    connection_refused = ASIO_SOCKET_ERROR(ECONNREFUSED),

    /// Connection reset by peer.
    connection_reset = ASIO_SOCKET_ERROR(ECONNRESET),

    /// Bad file descriptor.
    bad_descriptor = ASIO_SOCKET_ERROR(EBADF),

    /// End of file or stream.
    eof = ASIO_WIN_OR_POSIX_ERROR(ERROR_HANDLE_EOF, -1),

    /// Bad address.
    fault = ASIO_SOCKET_ERROR(EFAULT),

    /// Host not found (authoritative).
    host_not_found = ASIO_NETDB_ERROR(HOST_NOT_FOUND),

    /// Host not found (non-authoritative).
    host_not_found_try_again = ASIO_NETDB_ERROR(TRY_AGAIN),

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
    no_host_data = ASIO_NETDB_ERROR(NO_DATA),

    /// Cannot allocate memory.
    no_memory = ASIO_WIN_OR_POSIX_ERROR(ERROR_OUTOFMEMORY, ENOMEM),

    /// Operation not permitted.
    no_permission = ASIO_WIN_OR_POSIX_ERROR(ERROR_ACCESS_DENIED, EPERM),

    /// Protocol not available.
    no_protocol_option = ASIO_SOCKET_ERROR(ENOPROTOOPT),

    /// A non-recoverable error occurred.
    no_recovery = ASIO_NETDB_ERROR(NO_RECOVERY),

    /// Transport endpoint is not connected.
    not_connected = ASIO_SOCKET_ERROR(ENOTCONN),

    /// Socket operation on non-socket.
    not_socket = ASIO_SOCKET_ERROR(ENOTSOCK),

    /// Operation not supported.
    not_supported = ASIO_SOCKET_ERROR(EOPNOTSUPP),

    /// Operation cancelled.
    operation_aborted =
      ASIO_WIN_OR_POSIX_ERROR(ERROR_OPERATION_ABORTED, ECANCELED),

    /// Cannot send after transport endpoint shutdown.
    shut_down = ASIO_SOCKET_ERROR(ESHUTDOWN),

    /// Success.
    success = 0,

    /// Connection timed out.
    timed_out = ASIO_SOCKET_ERROR(ETIMEDOUT),

    /// Resource temporarily unavailable.
    try_again = ASIO_WIN_OR_POSIX_ERROR(ERROR_RETRY, EAGAIN),

    /// The socket is marked non-blocking and the requested operation would
    /// block.
    would_block = ASIO_SOCKET_ERROR(EWOULDBLOCK)
  };

  /// Default constructor.
  error()
    : code_(success)
  {
  }

  /// Construct with a specific error code.
  error(int code)
    : code_(code)
  {
  }

  // Destructor.
  virtual ~error() throw ()
  {
  }

  // Get the string for the type of exception.
  virtual const char* what() const throw ()
  {
    return "asio error";
  }

  /// Get the code associated with the error.
  int code() const
  {
    return code_;
  }

  struct unspecified_bool_type_t;
  typedef unspecified_bool_type_t* unspecified_bool_type;

  /// Operator returns non-null if there is a non-success error code.
  operator unspecified_bool_type() const
  {
    if (code_ == success)
      return 0;
    else
      return reinterpret_cast<unspecified_bool_type>(1);
  }

  /// Operator to test if the error represents success.
  bool operator!() const
  {
    return code_ == success;
  }

  /// Equality operator to compare two error objects.
  friend bool operator==(const error& e1, const error& e2)
  {
    return e1.code_ == e2.code_;
  }

  /// Inequality operator to compare two error objects.
  friend bool operator!=(const error& e1, const error& e2)
  {
    return e1.code_ != e2.code_;
  }

private:
  // The code associated with the error.
  int code_;
};

/// Output the string associated with an error.
/**
 * Used to output a human-readable string that is associated with an error.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param e The error to be written.
 *
 * @return The output stream.
 *
 * @relates asio::error
 */
template <typename Ostream>
Ostream& operator<<(Ostream& os, const error& e)
{
#if defined(_WIN32)
  LPTSTR msg = 0;
  DWORD length = ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
      | FORMAT_MESSAGE_FROM_SYSTEM
      | FORMAT_MESSAGE_IGNORE_INSERTS, 0, e.code(),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&msg, 0, 0);
  if (length && msg[length - 1] == '\n')
    msg[--length] = '\0';
  if (length && msg[length - 1] == '\r')
    msg[--length] = '\0';
  if (length)
    os << msg;
  else
    os << e.what() << ' ' << e.code();
  ::LocalFree(msg);
#else // _WIN32
  switch (e.code())
  {
  case error::eof:
    os << "End of file.";
    break;
  case error::host_not_found:
    os << "Host not found (authoritative).";
    break;
  case error::host_not_found_try_again:
    os << "Host not found (non-authoritative), try again later.";
    break;
  case error::no_recovery:
    os << "A non-recoverable error occurred during database lookup.";
    break;
  case error::no_host_data:
    os << "The name is valid, but it does not have associated data.";
    break;
#if !defined(__sun)
  case error::operation_aborted:
    os << "Operation aborted.";
    break;
#endif // !__sun
  default:
#if defined(__sun)
    os << strerror(e.code());
#elif defined(__MACH__) && defined(__APPLE__)
    {
      char buf[256] = "";
      strerror_r(e.code(), buf, sizeof(buf));
      os << buf;
    }
#else
    {
      char buf[256] = "";
      os << strerror_r(e.code(), buf, sizeof(buf));
    }
#endif
    break;
  }
#endif // _WIN32
  return os;
}

} // namespace asio

#undef ASIO_SOCKET_ERROR
#undef ASIO_NETDB_ERROR
#undef ASIO_WIN_OR_POSIX_ERROR

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_HPP
