//
// win_iocp_stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32) // This service is only supported on Win32

#include "asio/basic_demuxer.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/stream_socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {
namespace detail {

class win_iocp_stream_socket_service
{
public:
  // The native type of the stream socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // Return a null stream socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // The demuxer type for this service.
  typedef basic_demuxer<win_iocp_demuxer_service> demuxer_type;

  // Constructor. This stream_socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_stream_socket_service(
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

  // Open a new stream socket implementation.
  void open(impl_type& impl, impl_type new_impl)
  {
    demuxer_service_.register_socket(new_impl);
    impl = new_impl;
  }

  // Close a stream socket implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      socket_ops::close(impl);
      impl = null();
    }
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
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getpeername(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(impl_type& impl, stream_socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    int shutdown_flag;
    switch (what)
    {
    case stream_socket_base::shutdown_recv:
      shutdown_flag = shutdown_recv;
      break;
    case stream_socket_base::shutdown_send:
      shutdown_flag = shutdown_send;
      break;
    case stream_socket_base::shutdown_both:
    default:
      shutdown_flag = shutdown_both;
      break;
    }
    if (socket_ops::shutdown(impl, shutdown_flag) != 0)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t send(impl_type& impl, const void* data, size_t length,
      Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::send(impl, data, length, 0);
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
      Handler handler)
  {
    send_operation<Handler>* send_op = new send_operation<Handler>(handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(length);
    buf.buf = static_cast<char*>(const_cast<void*>(data));
    DWORD bytes_transferred = 0;

    int result = ::WSASend(impl, &buf, 1, &bytes_transferred, 0, send_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete send_op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t recv(impl_type& impl, void* data, size_t max_length,
      Error_Handler error_handler)
  {
    int bytes_recvd = socket_ops::recv(impl, data, max_length, 0);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_recvd;
  }

  template <typename Handler>
  class recv_operation
    : public win_iocp_operation
  {
  public:
    recv_operation(Handler handler)
      : win_iocp_operation(&recv_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      recv_operation<Handler>* h = static_cast<recv_operation<Handler>*>(op);
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
  void async_recv(impl_type& impl, void* data, size_t max_length,
      Handler handler)
  {
    recv_operation<Handler>* recv_op = new recv_operation<Handler>(handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = static_cast<u_long>(max_length);
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD flags = 0;

    int result = ::WSARecv(impl, &buf, 1, &bytes_transferred, &flags, recv_op,
        0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete recv_op;
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  /// Peek at the incoming data on the stream socket. Returns the number of
  /// bytes received or 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t peek(impl_type& impl, void* data, size_t max_length,
      Error_Handler error_handler)
  {
    int bytes_recvd = socket_ops::recv(impl, data, max_length, MSG_PEEK);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_recvd;
  }

  /// Determine the amount of data that may be received without blocking.
  template <typename Error_Handler>
  size_t in_avail(impl_type& impl, Error_Handler error_handler)
  {
    ioctl_arg_type bytes_avail = 0;
    if (socket_ops::ioctl(impl, FIONREAD, &bytes_avail))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return static_cast<size_t>(bytes_avail);
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

#endif // ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
