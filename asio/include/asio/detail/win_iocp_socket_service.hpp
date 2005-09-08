//
// win_iocp_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#if defined(_WIN32) // This service is only supported on Win32

#include "asio/detail/push_options.hpp"
#include <boost/shared_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {
namespace detail {

template <typename Allocator>
class win_iocp_socket_service
{
public:
  // The native type of the socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // The demuxer type for this service.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

  // The type of the reactor used for connect operations.
  typedef detail::select_reactor<true> reactor_type;

  // The maximum number of buffers to support in a single operation.
  enum { max_buffers = 16 };

  // Constructor. This socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_socket_service(
      demuxer_type& demuxer)
    : demuxer_(demuxer),
      demuxer_service_(demuxer.get_service(
          service_factory<win_iocp_demuxer_service>())),
      reactor_(demuxer.get_service(service_factory<reactor_type>()))
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Return a null socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // Open a new socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol()));
    if (sock.get() == invalid_socket)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    demuxer_service_.register_socket(sock.get());

    impl = sock.release();
  }

  // Assign a new socket implementation.
  void assign(impl_type& impl, impl_type new_impl)
  {
    demuxer_service_.register_socket(new_impl);
    impl = new_impl;
  }

  // Destroy a socket implementation.
  template <typename Error_Handler>
  void close(impl_type& impl, Error_Handler error_handler)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl);
      if (socket_ops::close(impl) == socket_error_retval)
        error_handler(asio::error(socket_ops::get_error()));
      else
        impl = null();
    }
  }

  // Bind the socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.data(),
          endpoint.size()) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Place the socket into the state where it will listen for new connections.
  template <typename Error_Handler>
  void listen(impl_type& impl, int backlog, Error_Handler error_handler)
  {
    if (backlog == 0)
      backlog = SOMAXCONN;

    if (socket_ops::listen(impl, backlog) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(impl_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    if (socket_ops::setsockopt(impl, option.level(), option.name(),
          option.data(), option.size()))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const impl_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(impl_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    if (socket_ops::ioctl(impl, command.name(),
          static_cast<ioctl_arg_type*>(command.data())))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getsockname(impl, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getpeername(impl, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(impl_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    if (socket_ops::shutdown(impl, what) != 0)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly.
  template <typename Const_Buffers, typename Error_Handler>
  size_t send(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(const_cast<void*>(iter->data()));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASend(impl, bufs, i, &bytes_transferred, flags, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class send_operation
    : public win_iocp_operation
  {
  public:
    send_operation(Handler handler)
      : win_iocp_operation(&send_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      send_operation<Handler>* h = static_cast<send_operation<Handler>*>(op);
      asio::error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_send(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    send_operation<Handler>* op = new send_operation<Handler>(handler);

    demuxer_service_.work_started();

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(const_cast<void*>(iter->data()));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASend(impl, bufs, i, &bytes_transferred, flags, op, 0);
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename Const_Buffers, typename Endpoint, typename Error_Handler>
  size_t send_to(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(const_cast<void*>(iter->data()));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASendTo(impl, bufs, i, &bytes_transferred, flags,
        destination.data(), destination.size(), 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class send_to_operation
    : public win_iocp_operation
  {
  public:
    send_to_operation(Handler handler)
      : win_iocp_operation(&send_to_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      send_to_operation<Handler>* h =
        static_cast<send_to_operation<Handler>*>(op);
      asio::error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Endpoint, typename Handler>
  void async_send_to(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Handler handler)
  {
    send_to_operation<Handler>* op = new send_to_operation<Handler>(handler);

    demuxer_service_.work_started();

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(const_cast<void*>(iter->data()));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASendTo(impl, bufs, i, &bytes_transferred, flags,
        destination.data(), destination.size(), op, 0);
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Mutable_Buffers, typename Error_Handler>
  size_t receive(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(iter->data());
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecv(impl, bufs, i,
        &bytes_transferred, &recv_flags, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class receive_operation
    : public win_iocp_operation
  {
  public:
    receive_operation(Handler handler)
      : win_iocp_operation(&receive_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      receive_operation<Handler>* h
        = static_cast<receive_operation<Handler>*>(op);
      asio::error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    receive_operation<Handler>* op = new receive_operation<Handler>(handler);

    demuxer_service_.work_started();

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(iter->data());
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecv(impl, bufs, i,
        &bytes_transferred, &recv_flags, op, 0);
    DWORD last_error = ::WSAGetLastError();
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename Mutable_Buffers, typename Endpoint, typename Error_Handler>
  size_t receive_from(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(iter->data());
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int endpoint_size = sender_endpoint.size();
    int result = ::WSARecvFrom(impl, bufs, i, &bytes_transferred, &recv_flags,
        sender_endpoint.data(), &endpoint_size, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }

    sender_endpoint.size(endpoint_size);

    return bytes_transferred;
  }

  template <typename Endpoint, typename Handler>
  class receive_from_operation
    : public win_iocp_operation
  {
  public:
    receive_from_operation(Endpoint& endpoint, Handler handler)
      : win_iocp_operation(
          &receive_from_operation<Endpoint, Handler>::do_completion_impl),
        endpoint_(endpoint),
        endpoint_size_(endpoint.size()),
        handler_(handler)
    {
    }

    int& endpoint_size()
    {
      return endpoint_size_;
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      receive_from_operation<Endpoint, Handler>* h =
        static_cast<receive_from_operation<Endpoint, Handler>*>(op);
      h->endpoint_.size(h->endpoint_size_);
      asio::error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Endpoint& endpoint_;
    int endpoint_size_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Mutable_Buffers, typename Endpoint, typename Handler>
  void async_receive_from(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Handler handler)
  {
    receive_from_operation<Endpoint, Handler>* receive_from_op =
      new receive_from_operation<Endpoint, Handler>(sender_endpoint, handler);

    demuxer_service_.work_started();

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(iter->size());
      bufs[i].buf = static_cast<char*>(iter->data());
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecvFrom(impl, bufs, i, &bytes_transferred, &recv_flags,
        sender_endpoint.data(), &receive_from_op->endpoint_size(),
        receive_from_op, 0);
    DWORD last_error = ::WSAGetLastError();
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete receive_from_op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Accept a new connection.
  template <typename Socket, typename Error_Handler>
  void accept(impl_type& impl, Socket& peer, Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    socket_type new_socket = socket_ops::accept(impl, 0, 0);
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer.set_impl(new_socket);
  }

  // Accept a new connection.
  template <typename Socket, typename Endpoint, typename Error_Handler>
  void accept_endpoint(impl_type& impl, Socket& peer, Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    socket_addr_len_type addr_len = peer_endpoint.size();
    socket_type new_socket = socket_ops::accept(impl,
        peer_endpoint.data(), &addr_len);
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer_endpoint.size(addr_len);

    peer.set_impl(new_socket);
  }

  template <typename Socket, typename Handler>
  class accept_handler
  {
  public:
    accept_handler(impl_type impl, demuxer_type& demuxer, Socket& peer,
        Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_type new_socket = socket_ops::accept(impl_, 0, 0);
      asio::error error(new_socket == invalid_socket
          ? socket_ops::get_error() : asio::error::success);
      peer_.set_impl(new_socket);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    demuxer_type& demuxer_;
    Socket& peer_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer object must be valid until the
  // accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(impl_type& impl, Socket& peer, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl,
          accept_handler<Socket, Handler>(impl, demuxer_, peer, handler));
    }
  }

  template <typename Socket, typename Endpoint, typename Handler>
  class accept_endp_handler
  {
  public:
    accept_endp_handler(impl_type impl, demuxer_type& demuxer, Socket& peer,
        Endpoint& peer_endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        peer_endpoint_(peer_endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = peer_endpoint_.size();
      socket_type new_socket = socket_ops::accept(impl_,
          peer_endpoint_.data(), &addr_len);
      asio::error error(new_socket == invalid_socket
          ? socket_ops::get_error() : asio::error::success);
      peer_endpoint_.size(addr_len);
      peer_.set_impl(new_socket);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    demuxer_type& demuxer_;
    Socket& peer_;
    Endpoint& peer_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Endpoint, typename Handler>
  void async_accept_endpoint(impl_type& impl, Socket& peer,
      Endpoint& peer_endpoint, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl,
          accept_endp_handler<Socket, Endpoint, Handler>(
            impl, demuxer_, peer, peer_endpoint, handler));
    }
  }

  // Connect the socket to the specified endpoint.
  template <typename Endpoint, typename Error_Handler>
  void connect(impl_type& impl, const Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // Open the socket if it is not already open.
    if (impl == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl = socket_ops::socket(family, type, proto);
      if (impl == invalid_socket)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }
      demuxer_service_.register_socket(impl);
    }

    // Perform the connect operation.
    int result = socket_ops::connect(impl, peer_endpoint.data(),
        peer_endpoint.size());
    if (result == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  template <typename Handler>
  class connect_handler
  {
  public:
    connect_handler(impl_type& impl, boost::shared_ptr<bool> completed,
        demuxer_type& demuxer, reactor_type& reactor, Handler handler)
      : impl_(impl),
        completed_(completed),
        demuxer_(demuxer),
        reactor_(reactor),
        handler_(handler)
    {
    }

    void do_operation()
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      if (*completed_)
      {
        demuxer_.work_finished();
        return;
      }

      // Cancel the other reactor operation for the connection.
      *completed_ = true;
      reactor_.enqueue_cancel_ops_unlocked(impl_);

      // Get the error code from the connect operation.
      int connect_error = 0;
      size_t connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(impl_, SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // If connection failed then post the handler with the error code.
      if (connect_error)
      {
        asio::error error(connect_error);
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(impl_, FIONBIO, &non_blocking))
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Post the result of the successful connection operation.
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler_, error));
    }

    void do_cancel()
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      if (*completed_)
      {
        demuxer_.work_finished();
        return;
      }

      // Cancel the other reactor operation for the connection.
      *completed_ = true;
      reactor_.enqueue_cancel_ops_unlocked(impl_);

      // The socket is closed when the reactor_.close_descriptor is called,
      // so no need to close it here.
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
    }

  private:
    impl_type& impl_;
    boost::shared_ptr<bool> completed_;
    demuxer_type& demuxer_;
    reactor_type& reactor_;
    Handler handler_;
  };

  // Start an asynchronous connect.
  template <typename Endpoint, typename Handler>
  void async_connect(impl_type& impl, const Endpoint& peer_endpoint,
      Handler handler)
  {
    // Open the socket if it is not already open.
    if (impl == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl = socket_ops::socket(family, type, proto);
      if (impl == invalid_socket)
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler, error));
        return;
      }
      demuxer_service_.register_socket(impl);
    }

    // Mark the socket as non-blocking so that the connection will take place
    // asynchronously.
    ioctl_arg_type non_blocking = 1;
    if (socket_ops::ioctl(impl, FIONBIO, &non_blocking))
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Start the connect operation.
    if (socket_ops::connect(impl, peer_endpoint.data(),
          peer_endpoint.size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // handler immediately.
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (socket_ops::get_error() == asio::error::in_progress
        || socket_ops::get_error() == asio::error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      boost::shared_ptr<bool> completed(new bool(false));
      demuxer_.work_started();
      reactor_.start_write_and_except_ops(impl, connect_handler<Handler>(
            impl, completed, demuxer_, reactor_, handler));
    }
    else
    {
      // The connect operation has failed, so post the handler immediately.
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
    }
  }


private:
  // The demuxer associated with the service.
  demuxer_type& demuxer_;

  // The demuxer service used for running asynchronous operations and
  // dispatching handlers.
  win_iocp_demuxer_service& demuxer_service_;

  // The reactor used for performing accept and connect operations.
  reactor_type& reactor_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
