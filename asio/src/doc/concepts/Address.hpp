//
// address.hpp
// ~~~~~~~~~~~
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

/// Address concept.
/**
 * @par Implemented By:
 * asio::ipv4::address
 */
class Address
{
public:
  /// The default stream-based protocol associated with the address type. This
  /// typedef is not required if the address family does not support any stream
  /// protocols.
  typedef implementation_defined default_stream_protocol;

  /// The default datagram-based protocol associated with the address type.
  /// This typedef is not required if the address family does not support any
  /// datagram protocols.
  typedef implementation_defined default_dgram_protocol;

  /// The native type of the address structure. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef implementation_defined native_address_type;

  /// The native type for the size of the address structure. This type is
  /// dependent on the underlying implementation of the socket layer.
  typedef implementation_defined native_size_type;

  /// The address family.
  int family() const;

  /// Get the underlying address in the native type. The returned object may be
  /// modified by the caller.
  native_address_type* native_address();

  /// Get the underlying address in the native type.
  const native_address_type* native_address() const;

  /// Get the underlying size of the address in the native type.
  native_size_type native_size() const;

  /// Set the underlying size of the address in the native type.
  void native_size(native_size_type size);
};
