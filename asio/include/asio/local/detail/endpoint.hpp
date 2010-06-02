//
// local/detail/endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Derived from a public domain implementation written by Daniel Casimiro.
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCAL_DETAIL_ENDPOINT_HPP
#define ASIO_LOCAL_DETAIL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_LOCAL_SOCKETS)

#include <cstddef>
#include <cstring>
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace local {
namespace detail {

// Helper class for implementing a UNIX domain endpoint.
class endpoint
{
public:
  // Default constructor.
  endpoint()
  {
    init("", 0);
  }

  // Construct an endpoint using the specified path name.
  endpoint(const char* path)
  {
    using namespace std; // For strlen.
    init(path, strlen(path));
  }

  // Construct an endpoint using the specified path name.
  endpoint(const std::string& path)
  {
    init(path.data(), path.length());
  }

  // Copy constructor.
  endpoint(const endpoint& other)
    : data_(other.data_),
      path_length_(other.path_length_)
  {
  }

  // Assign from another endpoint.
  endpoint& operator=(const endpoint& other)
  {
    data_ = other.data_;
    path_length_ = other.path_length_;
    return *this;
  }

  // Get the underlying endpoint in the native type.
  asio::detail::socket_addr_type* data()
  {
    return &data_.base;
  }

  // Get the underlying endpoint in the native type.
  const asio::detail::socket_addr_type* data() const
  {
    return &data_.base;
  }

  // Get the underlying size of the endpoint in the native type.
  std::size_t size() const
  {
    return path_length_
      + offsetof(asio::detail::sockaddr_un_type, sun_path);
  }

  // Set the underlying size of the endpoint in the native type.
  void resize(std::size_t size)
  {
    if (size > sizeof(asio::detail::sockaddr_un_type))
    {
      asio::error_code ec(asio::error::invalid_argument);
      asio::detail::throw_error(ec);
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

  // Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return sizeof(asio::detail::sockaddr_un_type);
  }

  // Get the path associated with the endpoint.
  std::string path() const
  {
    return std::string(data_.local.sun_path, path_length_);
  }

  // Set the path associated with the endpoint.
  void path(const char* p)
  {
    using namespace std; // For strlen.
    init(p, strlen(p));
  }

  // Set the path associated with the endpoint.
  void path(const std::string& p)
  {
    init(p.data(), p.length());
  }

  // Compare two endpoints for equality.
  friend bool operator==(const endpoint& e1, const endpoint& e2)
  {
    return e1.path() == e2.path();
  }

  // Compare two endpoints for inequality.
  friend bool operator!=(const endpoint& e1, const endpoint& e2)
  {
    return e1.path() != e2.path();
  }

  // Compare endpoints for ordering.
  friend bool operator<(const endpoint& e1, const endpoint& e2)
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

} // namespace detail
} // namespace local
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_LOCAL_SOCKETS)

#endif // ASIO_LOCAL_DETAIL_ENDPOINT_HPP
