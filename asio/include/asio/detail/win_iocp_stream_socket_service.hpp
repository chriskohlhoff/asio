//
// win_iocp_stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
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

  // Create a new stream socket implementation.
  void create(impl_type& impl, impl_type new_impl)
  {
    demuxer_service_.register_socket(new_impl);
    impl = new_impl;
  }

  // Destroy a stream socket implementation.
  void destroy(impl_type& impl)
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
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    socket_len_type size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Get the local socket address.
  template <typename Address, typename Error_Handler>
  void get_local_address(impl_type& impl, Address& address,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = address.native_size();
    if (socket_ops::getsockname(impl, address.native_address(), &addr_len))
      error_handler(socket_error(socket_ops::get_error()));
    address.native_size(addr_len);
  }

  // Get the remote socket address.
  template <typename Address, typename Error_Handler>
  void get_remote_address(impl_type& impl, Address& address,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = address.native_size();
    if (socket_ops::getpeername(impl, address.native_address(), &addr_len))
      error_handler(socket_error(socket_ops::get_error()));
    address.native_size(addr_len);
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
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }
    return bytes_sent;
  }

  template <typename Handler, typename Completion_Context>
  class send_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    send_operation(Handler handler, Completion_Context context)
      : win_iocp_demuxer_service::operation(false),
        handler_(handler),
        context_(context)
    {
    }

    virtual bool do_completion(HANDLE iocp, DWORD last_error,
        size_t bytes_transferred)
    {
      if (!acquire_context(iocp, context_))
        return false;

      socket_error error(last_error);
      do_upcall(handler_, error, bytes_transferred);
      release_context(context_);
      delete this;
      return true;
    }

    static void do_upcall(Handler handler, const socket_error& error,
        size_t bytes_sent)
    {
      try
      {
        handler(error, bytes_sent);
      }
      catch (...)
      {
      }
    }

  private:
    Handler handler_;
    Completion_Context context_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_send(impl_type& impl, const void* data, size_t length,
      Handler handler, Completion_Context context)
  {
    send_operation<Handler, Completion_Context>* send_op =
      new send_operation<Handler, Completion_Context>(handler, context);

    demuxer_service_.operation_started();

    WSABUF buf;
    buf.len = length;
    buf.buf = static_cast<char*>(const_cast<void*>(data));
    DWORD bytes_transferred = 0;

    int result = ::WSASend(impl, &buf, 1, &bytes_transferred, 0, send_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete send_op;
      socket_error error(last_error);
      demuxer_service_.operation_completed(
          bind_handler(handler, error, bytes_transferred), context, false);
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
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }
    return bytes_recvd;
  }

  template <typename Handler, typename Completion_Context>
  class recv_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    recv_operation(Handler handler, Completion_Context context)
      : win_iocp_demuxer_service::operation(false),
        handler_(handler),
        context_(context)
    {
    }

    virtual bool do_completion(HANDLE iocp, DWORD last_error,
        size_t bytes_transferred)
    {
      if (!acquire_context(iocp, context_))
        return false;

      socket_error error(last_error);
      do_upcall(handler_, error, bytes_transferred);
      release_context(context_);
      delete this;
      return true;
    }

    static void do_upcall(Handler handler, const socket_error& error,
        size_t bytes_recvd)
    {
      try
      {
        handler(error, bytes_recvd);
      }
      catch (...)
      {
      }
    }

  private:
    Handler handler_;
    Completion_Context context_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_recv(impl_type& impl, void* data, size_t max_length,
      Handler handler, Completion_Context context)
  {
    recv_operation<Handler, Completion_Context>* recv_op
      = new recv_operation<Handler, Completion_Context>(handler, context);

    demuxer_service_.operation_started();

    WSABUF buf;
    buf.len = max_length;
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD flags = 0;

    int result = ::WSARecv(impl, &buf, 1, &bytes_transferred, &flags, recv_op,
        0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete recv_op;
      socket_error error(last_error);
      demuxer_service_.operation_completed(
          bind_handler(handler, error, bytes_transferred), context, false);
    }
  }

private:
  // The demuxer used for delivering completion notifications.
  demuxer_type& demuxer_;

  // The demuxer service used for running asynchronous operations.
  win_iocp_demuxer_service& demuxer_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
