//
// reactive_stream_socket_service.hpp
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

#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_stream_socket_service
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

  // Constructor.
  reactive_stream_socket_service(Demuxer& d)
    : demuxer_(d),
      reactor_(d.get_service(service_factory<Reactor>()))
  {
  }

  // Create a new stream socket implementation.
  void create(impl_type& impl, impl_type new_impl)
  {
    impl = new_impl;
  }

  // Destroy a stream socket implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl, socket_ops::close);
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
  class send_handler
  {
  public:
    send_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, Handler handler, Completion_Context& context)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::send(impl, data_, length_, 0);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_.operation_completed(bind_handler(handler_, error, bytes),
          context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error, 0),
          context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    Handler handler_;
    Completion_Context& context_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_send(impl_type& impl, const void* data, size_t length,
      Handler handler, Completion_Context& context)
  {
    demuxer_.operation_started();
    reactor_.start_write_op(impl, send_handler<Handler, Completion_Context>(
          impl, demuxer_, data, length, handler, context));
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
  class send_n_handler
  {
  public:
    send_n_handler(impl_type impl, Demuxer& demuxer, Reactor& reactor,
        const void* data, size_t length, size_t already_sent, Handler handler,
        Completion_Context& context)
      : impl_(impl),
        demuxer_(demuxer),
        reactor_(reactor),
        data_(data),
        length_(length),
        already_sent_(already_sent),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::send(impl_,
          static_cast<const char*>(data_) + already_sent_,
          length_ - already_sent_, 0);
      size_t last_bytes = (bytes > 0 ? bytes : 0);
      size_t total_bytes = already_sent_ + last_bytes;
      if (last_bytes == 0 || total_bytes == length_)
      {
        socket_error error(bytes < 0
            ? socket_ops::get_error() : socket_error::success);
        demuxer_.operation_completed(bind_handler(handler_, error, total_bytes,
              last_bytes), context_);
      }
      else
      {
        already_sent_ = total_bytes;
        reactor_.restart_write_op(impl_, *this);
      }
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error, already_sent_,
            0), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    Reactor& reactor_;
    const void* data_;
    size_t length_;
    size_t already_sent_;
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
    demuxer_.operation_started();
    reactor_.start_write_op(impl, send_n_handler<Handler, Completion_Context>(
          impl, demuxer_, reactor_, data, length, 0, handler, context));
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
  class recv_handler
  {
  public:
    recv_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, Handler handler, Completion_Context& context)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::recv(impl_, data_, max_length_, 0);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_.operation_completed(bind_handler(handler_, error, bytes),
          context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error, 0), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    Handler handler_;
    Completion_Context& context_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_recv(impl_type& impl, void* data, size_t max_length,
      Handler handler, Completion_Context& context)
  {
    demuxer_.operation_started();
    reactor_.start_read_op(impl, recv_handler<Handler, Completion_Context>(
          impl, demuxer_, data, max_length, handler, context));
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
  class recv_n_handler
  {
  public:
    recv_n_handler(impl_type impl, Demuxer& demuxer, Reactor& reactor,
        void* data, size_t length, size_t already_recvd, Handler handler,
        Completion_Context& context)
      : impl_(impl),
        demuxer_(demuxer),
        reactor_(reactor),
        data_(data),
        length_(length),
        already_recvd_(already_recvd),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::recv(impl_,
          static_cast<char*>(data_) + already_recvd_,
          length_ - already_recvd_, 0);
      size_t last_bytes = (bytes > 0 ? bytes : 0);
      size_t total_bytes = already_recvd_ + last_bytes;
      if (last_bytes == 0 || total_bytes == length_)
      {
        socket_error error(bytes < 0
            ? socket_ops::get_error() : socket_error::success);
        demuxer_.operation_completed(bind_handler(handler_, error, total_bytes,
              last_bytes), context_);
      }
      else
      {
        already_recvd_ = total_bytes;
        reactor_.restart_read_op(impl_, *this);
      }
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error,
            already_recvd_, 0), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    Reactor& reactor_;
    void* data_;
    size_t length_;
    size_t already_recvd_;
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
    demuxer_.operation_started();
    reactor_.start_read_op(impl, recv_n_handler<Handler, Completion_Context>(
          impl, demuxer_, reactor_, data, length, 0, handler, context));
  }

private:
  // The demuxer used for delivering completion notifications.
  Demuxer& demuxer_;

  // The reactor that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
