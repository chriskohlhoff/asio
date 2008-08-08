//
// basic_endpoint.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Derived from a public domain implementation written by Daniel Casimiro.
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCAL_BASIC_ENDPOINT_HPP
#define ASIO_LOCAL_BASIC_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <cstddef>
#include <cstring>
#include <ostream>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_error.hpp"

#if !defined(ASIO_DISABLE_LOCAL_SOCKETS)
# if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#  define ASIO_HAS_LOCAL_SOCKETS 1
# endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#endif // !defined(ASIO_DISABLE_LOCAL_SOCKETS)

#if defined(ASIO_HAS_LOCAL_SOCKETS) \
  || defined(GENERATING_DOCUMENTATION)


namespace asio {
namespace local {

/// Describes an endpoint for a UNIX socket.
/**
 * The asio::local::basic_endpoint class template describes an endpoint
 * that may be associated with a particular UNIX socket.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Endpoint.
 */
template <typename Protocol>
class basic_endpoint
{
public:
  /// The protocol type associated with the endpoint.
  typedef Protocol protocol_type;

  /// The type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined data_type;
#else
  typedef asio::detail::socket_addr_type data_type;
#endif

  /// Default constructor.
  basic_endpoint()
  {
    init("", 0);
  }

  /// Construct an endpoint using the specified path name.
  basic_endpoint(const char* path)
  {
    using namespace std; // For strlen.
    init(path, strlen(path));
  }

  /// Construct an endpoint using the specified path name.
  basic_endpoint(const std::string& path)
  {
    init(path.data(), path.length());
  }

  /// Copy constructor.
  basic_endpoint(const basic_endpoint& other)
    : data_(other.data_),
      path_length_(other.path_length_)
  {
  }

  /// Assign from another endpoint.
  basic_endpoint& operator=(const basic_endpoint& other)
  {
    data_ = other.data_;
    path_length_ = other.path_length_;
    return *this;
  }

  /// The protocol associated with the endpoint.
  protocol_type protocol() const
  {
    return protocol_type();
  }

  /// Get the underlying endpoint in the native type.
  data_type* data()
  {
    return &data_.base;
  }

  /// Get the underlying endpoint in the native type.
  const data_type* data() const
  {
    return &data_.base;
  }

  /// Get the underlying size of the endpoint in the native type.
  std::size_t size() const
  {
    return path_length_
      + offsetof(asio::detail::sockaddr_un_type, sun_path);
  }

  /// Set the underlying size of the endpoint in the native type.
  void resize(std::size_t size)
  {
    if (size > sizeof(asio::detail::sockaddr_un_type))
    {
      asio::system_error e(asio::error::invalid_argument);
      boost::throw_exception(e);
    }
    else if (size == 0)
    {
      path_length_ = 0;
    }
    else
    {
      path_length_ = size
        - offsetof(asio::detail::sockaddr_un_type, sun_path);

      // The path returned by the operating system may be NUL-terminated.
      if (path_length_ > 0 && data_.local.sun_path[path_length_ - 1] == 0)
        --path_length_;
    }
  }

  /// Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return sizeof(asio::detail::sockaddr_un_type);
  }

  /// Get the path associated with the endpoint.
  std::string path() const
  {
    return std::string(data_.local.sun_path, path_length_);
  }

  /// Set the path associated with the endpoint.
  void path(const char* p)
  {
    using namespace std; // For strlen.
    init(p, strlen(p));
  }

  /// Set the path associated with the endpoint.
  void path(const std::string& p)
  {
    init(p.data(), p.length());
  }

  /// Compare two endpoints for equality.
  friend bool operator==(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.path() == e2.path();
  }

  /// Compare two endpoints for inequality.
  friend bool operator!=(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.path() != e2.path();
  }

  /// Compare endpoints for ordering.
  friend bool operator<(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.path() < e2.path();
  }

private:
  // The underlying UNIX socket address.
  union data_union
  {
    asio::detail::socket_addr_type base;
    asio::detail::sockaddr_un_type local;
  } data_;

  // The length of the path associated with the endpoint.
  std::size_t path_length_;

  // Initialise with a specified path.
  void init(const char* path, std::size_t path_length)
  {
    if (path_length > sizeof(data_.local.sun_path) - 1)
    {
      // The buffer is not large enough to store this address.
      asio::error_code ec(asio::error::name_too_long);
      asio::detail::throw_error(ec);
    }

    using namespace std; // For memcpy.
    data_.local = asio::detail::sockaddr_un_type();
    data_.local.sun_family = AF_UNIX;
    memcpy(data_.local.sun_path, path, path_length);
    path_length_ = path_length;

    // NUL-terminate normal path names. Names that start with a NUL are in the
    // UNIX domain protocol's "abstract namespace" and are not NUL-terminated.
    if (path_length > 0 && data_.local.sun_path[0] == 0)
      data_.local.sun_path[path_length] = 0;
  }
};

/// Output an endpoint as a string.
/**
 * Used to output a human-readable string for a specified endpoint.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param endpoint The endpoint to be written.
 *
 * @return The output stream.
 *
 * @relates asio::local::basic_endpoint
 */
template <typename Elem, typename Traits, typename Protocol>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os,
    const basic_endpoint<Protocol>& endpoint)
{
  os << endpoint.path();
  return os;
}

} // namespace local
} // namespace asio

#endif // defined(ASIO_HAS_LOCAL_SOCKETS)
       //   || defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_LOCAL_BASIC_ENDPOINT_HPP
