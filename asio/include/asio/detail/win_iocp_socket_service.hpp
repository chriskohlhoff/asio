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

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
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

  // Constructor. This socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_socket_service(
      demuxer_type& demuxer)
    : demuxer_(demuxer),
      demuxer_service_(demuxer.get_service(
          service_factory<win_iocp_demuxer_service>()))
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
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      socket_ops::close(impl);
      impl = null();
    }
  }

  // Bind the socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.native_data(),
          endpoint.native_size()) == socket_error_retval)
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
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.native_size(addr_len);
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
  template <typename Error_Handler>
  size_t send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::send(impl, data, length, flags);
    if (bytes_sent < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_sent;
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
  template <typename Handler>
  void async_send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Handler handler)
  {
    send_operation<Handler>* op = new send_operation<Handler>(handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(length);
    buf.buf = static_cast<char*>(const_cast<void*>(data));
    DWORD bytes_transferred = 0;

    int result = ::WSASend(impl, &buf, 1, &bytes_transferred, flags, op, 0);
    DWORD last_error = ::WSAGetLastError();

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
  template <typename Endpoint, typename Error_Handler>
  size_t send_to(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, const Endpoint& destination,
      Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::sendto(impl, data, length, flags,
        destination.native_data(), destination.native_size());
    if (bytes_sent < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }

    return bytes_sent;
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
  template <typename Endpoint, typename Handler>
  void async_send_to(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, const Endpoint& destination,
      Handler handler)
  {
    send_to_operation<Handler>* send_to_op =
      new send_to_operation<Handler>(handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(length);
    buf.buf = static_cast<char*>(const_cast<void*>(data));
    DWORD bytes_transferred = 0;

    int result = ::WSASendTo(impl, &buf, 1, &bytes_transferred, flags,
        destination.native_data(), destination.native_size(), send_to_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete send_to_op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    int bytes_recvd = socket_ops::recv(impl, data, max_length, flags);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_recvd;
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
  template <typename Handler>
  void async_receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Handler handler)
  {
    receive_operation<Handler>* op = new receive_operation<Handler>(handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(max_length);
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;

    int result = ::WSARecv(impl, &buf, 1,
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
  template <typename Endpoint, typename Error_Handler>
  size_t receive_from(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = sender_endpoint.native_size();
    int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, flags,
        sender_endpoint.native_data(), &addr_len);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }

    sender_endpoint.native_size(addr_len);

    return bytes_recvd;
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
        endpoint_size_(endpoint.native_size()),
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
      h->endpoint_.native_size(h->endpoint_size_);
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
  template <typename Endpoint, typename Handler>
  void async_receive_from(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Handler handler)
  {
    receive_from_operation<Endpoint, Handler>* receive_from_op =
      new receive_from_operation<Endpoint, Handler>(sender_endpoint, handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(max_length);
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;

    int result = ::WSARecvFrom(impl, &buf, 1, &bytes_transferred, &recv_flags,
        sender_endpoint.native_data(), &receive_from_op->endpoint_size(),
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
    }

    // Perform the connect operation.
    int result = socket_ops::connect(impl, peer_endpoint.native_data(),
        peer_endpoint.native_size());
    if (result == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Start an asynchronous connect.
  template <typename Endpoint, typename Handler>
  void async_connect(impl_type& impl, const Endpoint& peer_endpoint,
      Handler handler)
  {
    typedef detail::reactive_socket_service<
      demuxer_type, detail::select_reactor<true> > socket_service_type;
    socket_service_type& service = demuxer_.get_service(
        service_factory<socket_service_type>());
    service.async_connect(impl, peer_endpoint, handler);
  }

private:
  // The demuxer associated with the service.
  demuxer_type& demuxer_;

  // The demuxer service used for running asynchronous operations and
  // dispatching handlers.
  win_iocp_demuxer_service& demuxer_service_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
