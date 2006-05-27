//
// error.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <boost/config.hpp>
#include <boost/scoped_ptr.hpp>
#include <cerrno>
#include <cstring>
#include <exception>
#include <string>
#include <boost/detail/workaround.hpp>
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
# include <iostream>
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_local_free_on_block_exit.hpp"

namespace asio {

#if defined(GENERATING_DOCUMENTATION)
/// INTERNAL ONLY.
# define ASIO_SOCKET_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_NETDB_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_GETADDRINFO_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_OS_ERROR(e_win, e_posix) implementation_defined
#elif defined(BOOST_WINDOWS) || defined(__CYGWIN__)
# define ASIO_SOCKET_ERROR(e) WSA ## e
# define ASIO_NETDB_ERROR(e) WSA ## e
# define ASIO_GETADDRINFO_ERROR(e) e
# define ASIO_OS_ERROR(e_win, e_posix) e_win
#else
# define ASIO_SOCKET_ERROR(e) e
# define ASIO_NETDB_ERROR(e) 16384 + e
# define ASIO_GETADDRINFO_ERROR(e) 32768 + e
# define ASIO_OS_ERROR(e_win, e_posix) e_posix
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
    eof = ASIO_OS_ERROR(ERROR_HANDLE_EOF, -1),

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

    /// The query is valid but does not have associated address data.
    no_data = ASIO_NETDB_ERROR(NO_DATA),

    /// Cannot allocate memory.
    no_memory = ASIO_OS_ERROR(ERROR_OUTOFMEMORY, ENOMEM),

    /// Operation not permitted.
    no_permission = ASIO_OS_ERROR(ERROR_ACCESS_DENIED, EPERM),

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
    operation_aborted = ASIO_OS_ERROR(ERROR_OPERATION_ABORTED, ECANCELED),

    /// The service is not supported for the given socket type.
    service_not_found = ASIO_OS_ERROR(
        WSATYPE_NOT_FOUND,
        ASIO_GETADDRINFO_ERROR(EAI_SERVICE)),

    /// The socket type is not supported.
    socket_type_not_supported = ASIO_OS_ERROR(
        WSAESOCKTNOSUPPORT,
        ASIO_GETADDRINFO_ERROR(EAI_SOCKTYPE)),

    /// Cannot send after transport endpoint shutdown.
    shut_down = ASIO_SOCKET_ERROR(ESHUTDOWN),

    /// Success.
    success = 0,

    /// Connection timed out.
    timed_out = ASIO_SOCKET_ERROR(ETIMEDOUT),

    /// Resource temporarily unavailable.
    try_again = ASIO_OS_ERROR(ERROR_RETRY, EAGAIN),

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

  /// Copy constructor.
  error(const error& e)
    : std::exception(e),
      code_(e.code_)
  {
  }

  /// Destructor.
  virtual ~error() throw ()
  {
  }

  /// Assignment operator.
  error& operator=(const error& e)
  {
    code_ = e.code_;
    what_.reset();
    return *this;
  }

  /// Get a string representation of the exception.
  virtual const char* what() const throw ()
  {
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
    try
    {
      if (!what_)
      {
        char* msg = 0;
        DWORD length = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
            | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS, 0, code_,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, 0);
        detail::win_local_free_on_block_exit local_free_obj(msg);
        if (length && msg[length - 1] == '\n')
          msg[--length] = '\0';
        if (length && msg[length - 1] == '\r')
          msg[--length] = '\0';
        if (length)
          what_.reset(new std::string(msg));
        else
          return "asio error";
      }
      return what_->c_str();
    }
    catch (std::exception&)
    {
      return "asio error";
    }
#else // defined(BOOST_WINDOWS)
    switch (code_)
    {
    case error::eof:
      return "End of file.";
    case error::host_not_found:
      return "Host not found (authoritative).";
    case error::host_not_found_try_again:
      return "Host not found (non-authoritative), try again later.";
    case error::no_recovery:
      return "A non-recoverable error occurred during database lookup.";
    case error::no_data:
      return "The query is valid, but it does not have associated data.";
#if !defined(__sun)
    case error::operation_aborted:
      return "Operation aborted.";
#endif // !defined(__sun)
    case error::service_not_found:
      return "Service not found.";
    case error::socket_type_not_supported:
      return "Socket type not supported.";
    default:
#if defined(__sun) || defined(__QNX__)
      return strerror(code_);
#elif defined(__MACH__) && defined(__APPLE__)
      try
      {
        char buf[256] = "";
        strerror_r(code_, buf, sizeof(buf));
        what_.reset(new std::string(buf));
        return what_->c_str();
      }
      catch (std::exception&)
      {
        return "asio error";
      }
#else
      try
      {
        char buf[256] = "";
        what_.reset(new std::string(strerror_r(code_, buf, sizeof(buf))));
        return what_->c_str();
      }
      catch (std::exception&)
      {
        return "asio error";
      }
#endif
    }
#endif // defined(BOOST_WINDOWS)
  }

  /// Get the code associated with the error.
  int code() const
  {
    return code_;
  }

  struct unspecified_bool_type_t
  {
  };

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

  // The string representation of the error.
  mutable boost::scoped_ptr<std::string> what_;
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
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
std::ostream& operator<<(std::ostream& os, const error& e)
{
  os << e.what();
  return os;
}
#else // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
template <typename Ostream>
Ostream& operator<<(Ostream& os, const error& e)
{
  os << e.what();
  return os;
}
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))

} // namespace asio

#undef ASIO_SOCKET_ERROR
#undef ASIO_NETDB_ERROR
#undef ASIO_GETADDRINFO_ERROR
#undef ASIO_OS_ERROR

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_HPP
