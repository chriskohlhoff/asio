//
// dgram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_DGRAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_DGRAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio { class completion_context; }
namespace asio { class socket_address; }
namespace asio { class socket_error; }

namespace asio {
namespace detail {

class dgram_socket_service
  : public virtual service
{
public:
  // The service type id.
  static const service_type_id id;

  // The native type of the dgram socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // The value to use for uninitialised implementations.
  static const impl_type invalid_impl;

  // Create a new dgram socket implementation.
  void create(impl_type& impl, const socket_address& address);

  // Destroy a dgram socket implementation.
  void destroy(impl_type& impl);

  // Send a datagram to the specified address. Returns the number of bytes
  // sent. Throws a socket_error exception on failure.
  size_t sendto(impl_type& impl, const void* data, size_t length,
      const socket_address& destination);

  // The handler when a sendto operation is completed. The first argument is
  // the error code, the second is the number of bytes sent.
  typedef boost::function2<void, const socket_error&, size_t> sendto_handler;

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  void async_sendto(impl_type& impl, const void* data, size_t length,
      const socket_address& destination, const sendto_handler& handler,
      completion_context& context);

  // Receive a datagram with the address of the sender. Returns the number of
  // bytes received. Throws a socket_error exception on failure.
  size_t recvfrom(impl_type& impl, void* data, size_t max_length,
      socket_address& sender_address);
  
  // The handler when a recvfrom operation is completed. The first argument is
  // the error code, the second is the number of bytes received.
  typedef boost::function2<void, const socket_error&, size_t> recvfrom_handler;

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_address obejct must both be valid for the lifetime of the
  // asynchronous operation.
  void async_recvfrom(impl_type& impl, void* data, size_t max_length,
      socket_address& sender_address, const recvfrom_handler& handler,
      completion_context& context);

private:
  // Create a dgram socket implementation.
  virtual void do_dgram_socket_create(impl_type& impl,
		  const socket_address& address) = 0;

  // Destroy a dgram socket implementation.
  virtual void do_dgram_socket_destroy(impl_type& impl) = 0;

  // Start an asynchronous sendto.
  virtual void do_dgram_socket_async_sendto(impl_type& impl, const void* data,
      size_t length, const socket_address& destination,
      const sendto_handler& handler, completion_context& context) = 0;

  // Start an asynchronous recvfrom.
  virtual void do_dgram_socket_async_recvfrom(impl_type& impl, void* data,
      size_t max_length, socket_address& sender_address,
      const recvfrom_handler& handler, completion_context& context) = 0;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DGRAM_SOCKET_SERVICE_HPP
