//
// Endpoint.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
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

  /// The native type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef implementation_defined native_data_type;

  /// The native type for the size of the endpoint structure. This type is
  /// dependent on the underlying implementation of the socket layer.
  typedef implementation_defined native_size_type;

  /// The protocol object associated with the endpoint. The returned object
  /// must implement the Protocol concept.
  implementation_defined protocol() const;

  /// Get the underlying endpoint in the native type. The returned object may
  /// be modified by the caller.
  native_data_type* native_data();

  /// Get the underlying endpoint in the native type.
  const native_data_type* native_data() const;

  /// Get the underlying size of the endpoint in the native type.
  native_size_type native_size() const;

  /// Set the underlying size of the endpoint in the native type.
  void native_size(native_size_type size);
};
