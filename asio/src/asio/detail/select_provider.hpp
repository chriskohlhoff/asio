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

#include "asio/dgram_socket_service.hpp"
#include "asio/service_provider.hpp"
#include "asio/socket_acceptor_service.hpp"
#include "asio/socket_connector_service.hpp"
#include "asio/stream_socket_service.hpp"
#include "asio/detail/selector.hpp"

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

  // Register a new dgram socket with the service. This should be called only
  // after the socket has been opened.
  virtual void register_dgram_socket(dgram_socket& socket);

  // Remove a dgram socket registration from the service. This should be
  // called immediately before the socket is closed.
  virtual void deregister_dgram_socket(dgram_socket& socket);

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_dgram_socket_sendto(dgram_socket& socket,
      const void* data, size_t length, const socket_address& destination,
      const sendto_handler& handler, completion_context& context);

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_address obejct must both be valid for the lifetime of the
  // asynchronous operation.
  virtual void async_dgram_socket_recvfrom(dgram_socket& socket, void* data,
      size_t max_length, socket_address& sender_address,
      const recvfrom_handler& handler, completion_context& context);

  // Register a new socket_acceptor with the service. This should be called
  // only after the socket acceptor has been opened.
  virtual void register_socket_acceptor(socket_acceptor& acceptor);

  // Remove a socket acceptor registration from the service. This should be
  // called immediately before the socket acceptor is closed.
  virtual void deregister_socket_acceptor(socket_acceptor& acceptor);

  // Start an asynchronous accept on the given socket. The peer_socket object
  // must be valid until the accept's completion handler is invoked.
  virtual void async_socket_accept(socket_acceptor& acceptor,
      stream_socket& peer_socket, const accept_handler& handler,
      completion_context& context);

  // Start an asynchronous accept on the given socket. The peer_socket and
  // peer_address objects must be valid until the accept's completion handler
  // is invoked.
  virtual void async_socket_accept(socket_acceptor& acceptor,
      stream_socket& peer_socket, socket_address& peer_address,
      const accept_handler& handler, completion_context& context);

  // Register a new socket_connector with the service. This should be called
  // only after the socket connector has been opened.
  virtual void register_socket_connector(socket_connector& connector);

  // Remove a socket connector registration from the service. This should be
  // called immediately before the socket connector is closed.
  virtual void deregister_socket_connector(socket_connector& connector);

  // Start an asynchronous connect on the given socket. The peer_socket object
  // be valid until the connect's completion handler is invoked.
  virtual void async_socket_connect(socket_connector& connector,
      stream_socket& peer_socket, const socket_address& peer_address,
      const connect_handler& handler, completion_context& context);

  // Register a new stream socket with the service. This should be called only
  // after the socket has been opened, i.e. after an accept or just before a
  // connect.
  virtual void register_stream_socket(stream_socket& socket);

  // Remove a stream socket registration from the service. This should be
  // called immediately before the socket is closed.
  virtual void deregister_stream_socket(stream_socket& socket);

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_stream_socket_send(stream_socket& socket,
      const void* data, size_t length, const send_handler& handler,
      completion_context& context);

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_stream_socket_send_n(stream_socket& socket,
      const void* data, size_t length, const send_n_handler& handler,
      completion_context& context);

  // Start an asynchronous receive. The buffer for the data being received must
  // be valid for the lifetime of the asynchronous operation.
  virtual void async_stream_socket_recv(stream_socket& socket, void* data,
      size_t max_length, const recv_handler& handler,
      completion_context& context);

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs. The buffer for the
  // data being received must be valid for the lifetime of the asynchronous
  // operation.
  virtual void async_stream_socket_recv_n(stream_socket& socket, void* data,
      size_t length, const recv_n_handler& handler,
      completion_context& context);

  // Provide access to these functions for the operation implementations.
  static void do_associate_accepted_stream_socket(socket_acceptor& acceptor,
      stream_socket& peer_socket, stream_socket::native_type handle);
  static void do_associate_connected_stream_socket(socket_connector& connector,
      stream_socket& peer_socket, stream_socket::native_type handle);

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
