//
// Endpoint.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Endpoint concept.
/**
 * @par Implemented By:
 * asio::ipv4::tcp::endpoint @n
 * asio::ipv4::udp::endpoint
 */
class Endpoint
{
public:
  /// The protocol type associated with the endpoint.
  typedef implementation_defined protocol_type;

  /// The type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef implementation_defined data_type;

  /// The type for the size of the endpoint structure. This type is dependent on
  /// the underlying implementation of the socket layer.
  typedef implementation_defined size_type;

  /// The protocol object associated with the endpoint. The returned object
  /// must implement the Protocol concept.
  implementation_defined protocol() const;

  /// Get the underlying endpoint in the implementation-defined type. The
  /// returned object may be modified by the caller.
  data_type* data();

  /// Get the underlying endpoint in the implementation-defined type.
  const data_type* data() const;

  /// Get the underlying size of the endpoint in the implementation-defined
  /// type.
  size_type size() const;

  /// Set the underlying size of the endpoint in the implementation-defined
  /// type.
  void size(size_type s);
};
