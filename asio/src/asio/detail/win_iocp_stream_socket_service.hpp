//
// win_iocp_stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

  // Constructor. This stream_socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_stream_socket_service(
      basic_demuxer<win_iocp_demuxer_service>& demuxer)
    : demuxer_service_(demuxer.get_service(
          service_factory<win_iocp_demuxer_service>()))
  {
  }

  // Return a null stream socket implementation.
  static impl_type null()
  {
    return invalid_socket;
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

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly. Throws a socket_error exception
  // on failure.
  size_t send(impl_type& impl, const void* data, size_t length)
  {
    int bytes_sent = socket_ops::send(impl, data, length, 0);
    if (bytes_sent < 0)
      throw socket_error(socket_ops::get_error());
    return bytes_sent;
  }

  template <typename Handler, typename Completion_Context>
  class send_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    send_operation(Handler handler, Completion_Context& context)
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
    Completion_Context& context_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_send(impl_type& impl, const void* data, size_t length,
      Handler handler, Completion_Context& context)
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

  // Send all of the given data to the peer before returning. Returns the
  // number of bytes sent on the last send or 0 if the connection was closed
  // cleanly. Throws a socket_error exception on failure.
  size_t send_n(impl_type& impl, const void* data, size_t length,
      size_t* total_bytes_sent)
  {
    // TODO handle non-blocking sockets using select to wait until ready.

    int bytes_sent = 0;
    size_t total_sent = 0;
    while (total_sent < length)
    {
      bytes_sent = socket_ops::send(impl,
          static_cast<const char*>(data) + total_sent, length - total_sent, 0);
      if (bytes_sent < 0)
      {
        throw socket_error(socket_ops::get_error());
      }
      else if (bytes_sent == 0)
      {
        if (total_bytes_sent)
          *total_bytes_sent = total_sent;
        return bytes_sent;
      }
      total_sent += bytes_sent;
    }
    if (total_bytes_sent)
      *total_bytes_sent = total_sent;
    return bytes_sent;
  }

  template <typename Handler, typename Completion_Context>
  class send_n_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    send_n_operation(impl_type impl, const void* data, DWORD length,
        Handler handler, Completion_Context& context)
      : win_iocp_demuxer_service::operation(false),
        impl_(impl),
        data_(data),
        length_(length),
        total_bytes_sent_(0),
        handler_(handler),
        context_(context)
    {
    }

    virtual bool do_completion(HANDLE iocp, DWORD last_error,
        size_t bytes_transferred)
    {
      if (!acquire_context(iocp, context_))
        return false;

      total_bytes_sent_ += bytes_transferred;
      if (last_error || bytes_transferred == 0 || total_bytes_sent_ == length_)
      {
        socket_error error(last_error);
        do_upcall(handler_, error, total_bytes_sent_, bytes_transferred);
        delete this;
        return true;
      }
      else
      {
        WSABUF buf;
        buf.len = length_ - total_bytes_sent_;
        buf.buf = static_cast<char*>(const_cast<void*>(data_));
        buf.buf += total_bytes_sent_;
        DWORD bytes_transferred = 0;

        int result = ::WSASend(impl_, &buf, 1, &bytes_transferred, 0, this, 0);
        DWORD last_error = ::WSAGetLastError();

        if (result != 0 && last_error != WSA_IO_PENDING)
        {
          socket_error error(last_error);
          do_upcall(handler_, error, total_bytes_sent_, bytes_transferred);
          delete this;
          return true;
        }
        else
        {
          // Not finished sending, so keep the operation alive.
          release_context(context_);
          return false;
        }
      }
    }

    static void do_upcall(Handler handler, const socket_error& error,
        size_t total_bytes_sent, size_t last_bytes_sent)
    {
      try
      {
        handler(error, total_bytes_sent, last_bytes_sent);
      }
      catch (...)
      {
      }
    }

  private:
    impl_type impl_;
    const void* data_;
    DWORD length_;
    DWORD total_bytes_sent_;
    Handler handler_;
    Completion_Context& context_;
  };

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_send_n(impl_type& impl, const void* data, size_t length,
      Handler handler, Completion_Context& context)
  {
    send_n_operation<Handler, Completion_Context>* send_n_op =
      new send_n_operation<Handler, Completion_Context>(impl, data, length,
          handler, context);

    demuxer_service_.operation_started();

    WSABUF buf;
    buf.len = length;
    buf.buf = static_cast<char*>(const_cast<void*>(data));
    DWORD bytes_transferred = 0;

    int result = ::WSASend(impl, &buf, 1, &bytes_transferred, 0, send_n_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete send_n_op;
      socket_error error(last_error);
      demuxer_service_.operation_completed(
          bind_handler(handler, error, bytes_transferred, bytes_transferred),
          context, false);
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly. Throws a socket_error exception
  // on failure.
  size_t recv(impl_type& impl, void* data, size_t max_length)
  {
    int bytes_recvd = socket_ops::recv(impl, data, max_length, 0);
    if (bytes_recvd < 0)
      throw socket_error(socket_ops::get_error());
    return bytes_recvd;
  }

  template <typename Handler, typename Completion_Context>
  class recv_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    recv_operation(Handler handler, Completion_Context& context)
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
    Completion_Context& context_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_recv(impl_type& impl, void* data, size_t max_length,
      Handler handler, Completion_Context& context)
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

  // Receive the specified amount of data from the peer. Returns the number of
  // bytes received on the last recv call or 0 if the connection
  // was closed cleanly. Throws a socket_error exception on failure.
  size_t recv_n(impl_type& impl, void* data, size_t length,
      size_t* total_bytes_recvd)
  {
    // TODO handle non-blocking sockets using select to wait until ready.

    int bytes_recvd = 0;
    size_t total_recvd = 0;
    while (total_recvd < length)
    {
      bytes_recvd = socket_ops::recv(impl,
          static_cast<char*>(data) + total_recvd, length - total_recvd, 0);
      if (bytes_recvd < 0)
      {
        throw socket_error(socket_ops::get_error());
      }
      else if (bytes_recvd == 0)
      {
        if (total_bytes_recvd)
          *total_bytes_recvd = total_recvd;
        return bytes_recvd;
      }
      total_recvd += bytes_recvd;
    }
    if (total_bytes_recvd)
      *total_bytes_recvd = total_recvd;
    return bytes_recvd;
  }

  template <typename Handler, typename Completion_Context>
  class recv_n_operation
    : public win_iocp_demuxer_service::operation
  {
  public:
    recv_n_operation(impl_type impl, void* data, DWORD length, Handler handler,
        Completion_Context& context)
      : win_iocp_demuxer_service::operation(false),
        impl_(impl),
        data_(data),
        length_(length),
        total_bytes_recvd_(0),
        handler_(handler),
        context_(context)
    {
    }

    virtual bool do_completion(HANDLE iocp, DWORD last_error,
        size_t bytes_transferred)
    {
      if (!acquire_context(iocp, context_))
        return false;

      total_bytes_recvd_ += bytes_transferred;
      if (last_error || bytes_transferred == 0
          || total_bytes_recvd_ == length_)
      {
        socket_error error(last_error);
        do_upcall(handler_, error, total_bytes_recvd_, bytes_transferred);
        delete this;
        return true;
      }
      else
      {
        WSABUF buf;
        buf.len = length_ - total_bytes_recvd_;
        buf.buf = static_cast<char*>(data_);
        buf.buf += total_bytes_recvd_;
        DWORD bytes_transferred = 0;
        DWORD flags = 0;

        int result = ::WSARecv(impl_, &buf, 1, &bytes_transferred, &flags,
            this, 0);
        DWORD last_error = ::WSAGetLastError();

        if (result != 0 && last_error != WSA_IO_PENDING)
        {
          socket_error error(last_error);
          do_upcall(handler_, error, total_bytes_recvd_, bytes_transferred);
          delete this;
          return true;
        }
        else
        {
          // Not finished sending, so keep the operation alive.
          release_context(context_);
          return false;
        }
      }
    }

    static void do_upcall(Handler handler, const socket_error& error,
        size_t total_bytes_recvd, size_t last_bytes_recvd)
    {
      try
      {
        handler(error, total_bytes_recvd, last_bytes_recvd);
      }
      catch (...)
      {
      }
    }

  private:
    impl_type impl_;
    void* data_;
    DWORD length_;
    DWORD total_bytes_recvd_;
    Handler handler_;
    Completion_Context& context_;
  };

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs. The buffer for the
  // data being received must be valid for the lifetime of the asynchronous
  // operation.
  template <typename Handler, typename Completion_Context>
  void async_recv_n(impl_type& impl, void* data, size_t length,
      Handler handler, Completion_Context& context)
  {
    recv_n_operation<Handler, Completion_Context>* recv_n_op =
      new recv_n_operation<Handler, Completion_Context>(impl, data, length,
          handler, context);

    demuxer_service_.operation_started();

    WSABUF buf;
    buf.len = length;
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD flags = 0;

    int result = ::WSARecv(impl, &buf, 1, &bytes_transferred, &flags,
        recv_n_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete recv_n_op;
      socket_error error(last_error);
      demuxer_service_.operation_completed(
          bind_handler(handler, error, bytes_transferred, bytes_transferred),
          context, false);
    }
  }

private:
  // The demuxer service used for running asynchronous operations.
  win_iocp_demuxer_service& demuxer_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
