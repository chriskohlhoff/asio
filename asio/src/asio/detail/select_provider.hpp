//
// select_provider.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SELECT_PROVIDER_HPP
#define ASIO_DETAIL_SELECT_PROVIDER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service_provider.hpp"
#include "asio/detail/dgram_socket_service.hpp"
#include "asio/detail/selector.hpp"
#include "asio/detail/socket_acceptor_service.hpp"
#include "asio/detail/socket_connector_service.hpp"
#include "asio/detail/stream_socket_service.hpp"

namespace asio {
namespace detail {

class select_provider
  : public service_provider,
    public dgram_socket_service,
    public socket_acceptor_service,
    public socket_connector_service,
    public stream_socket_service
{
public:
  // Constructor.
  select_provider(demuxer& d);

  // Destructor.
  virtual ~select_provider();

  // Return the service interface corresponding to the given type.
  virtual service* do_get_service(const service_type_id& service_type);

  // Create a dgram socket implementation.
  virtual void do_dgram_socket_create(dgram_socket_service::impl_type& impl,
		  const socket_address& address);

  // Destroy a dgram socket implementation.
  virtual void do_dgram_socket_destroy(dgram_socket_service::impl_type& impl);

  // Start an asynchronous sendto.
  virtual void do_dgram_socket_async_sendto(
      dgram_socket_service::impl_type& impl, const void* data, size_t length,
      const socket_address& destination, const sendto_handler& handler,
      completion_context& context);

  // Start an asynchronous recvfrom.
  virtual void do_dgram_socket_async_recvfrom(
      dgram_socket_service::impl_type& impl, void* data, size_t max_length,
      socket_address& sender_address, const recvfrom_handler& handler,
      completion_context& context);

  // Destroy a socket connector implementation.
  virtual void do_socket_acceptor_destroy(
      socket_acceptor_service::impl_type& impl);

  // Start an asynchronous accept on the given socket. The peer_socket object
  // must be valid until the accept's completion handler is invoked.
  virtual void do_socket_acceptor_async_accept(
      socket_acceptor_service::impl_type& impl,
      socket_acceptor_service::peer_type& peer_socket,
      const accept_handler& handler, completion_context& context);

  // Start an asynchronous accept on the given socket. The peer_socket and
  // peer_address objects must be valid until the accept's completion handler
  // is invoked.
  virtual void do_socket_acceptor_async_accept(
      socket_acceptor_service::impl_type& impl,
      socket_acceptor_service::peer_type& peer_socket,
      socket_address& peer_address, const accept_handler& handler,
      completion_context& context);

  // Destroy a socket connector implementation.
  virtual void do_socket_connector_destroy(
      socket_connector_service::impl_type& impl);

  // Start an asynchronous connect.
  virtual void do_socket_connector_async_connect(
      socket_connector_service::impl_type& impl,
      socket_connector_service::peer_type& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context);

  // Create a new socket connector implementation.
  virtual void do_stream_socket_create(stream_socket_service::impl_type& impl,
      stream_socket_service::impl_type new_impl);

  // Destroy a socket connector implementation.
  virtual void do_stream_socket_destroy(
      stream_socket_service::impl_type& impl);

  // Start an asynchronous send.
  virtual void do_stream_socket_async_send(
      stream_socket_service::impl_type& impl, const void* data, size_t length,
      const send_handler& handler, completion_context& context);

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs.
  virtual void do_stream_socket_async_send_n(
      stream_socket_service::impl_type& impl, const void* data, size_t length,
      const send_n_handler& handler, completion_context& context);

  // Start an asynchronous receive.
  virtual void do_stream_socket_async_recv(
      stream_socket_service::impl_type& impl, void* data, size_t max_length,
      const recv_handler& handler, completion_context& context);

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs.
  virtual void do_stream_socket_async_recv_n(
      stream_socket_service::impl_type& impl, void* data, size_t length,
      const recv_n_handler& handler, completion_context& context);

private:
  // The demuxer used for delivering completion notifications.
  demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  selector selector_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_PROVIDER_HPP
